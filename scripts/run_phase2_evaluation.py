"""Phase 2 evaluation harness.

For each trained policy at `runs/phase2/{config}/seed_{seed}/policy_final.pt`,
run 100 fresh evaluation episodes against the same stress regime that Phase 1
baselines were measured on. Compute identical metrics (aggregate throughput,
Jain fairness, link-utilization variance, median agent RTT) and write
per-policy JSON for the report script to consume.

The actor is run in deterministic mode (argmax) at eval time — gives clean,
reproducible numbers and matches the "what does the policy actually do at
deployment" semantics. Per-episode metric arrays are persisted so the
selection script can compute mean ± std without re-running.

Usage:
    uv run python scripts/run_phase2_evaluation.py
    uv run python scripts/run_phase2_evaluation.py --only sweep_01_balanced
    uv run python scripts/run_phase2_evaluation.py --workers 8
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = REPO_ROOT / "runs" / "phase2"

# Eval seeds disjoint from training (cfg.seed * 1_000_000 + ep) and Phase 1
# baselines (BASELINE_SEED_OFFSET = 1_000_000).
EVAL_SEED_OFFSET = 2_000_000

logger = logging.getLogger("phase2_eval")


def _eval_one_episode(env, actor, episode_seed: int, device, top_k: int) -> dict:
    """Run one full episode under the trained policy; return per-episode metrics."""
    import torch

    obs, infos = env.reset(seed=episode_seed)
    masks = {a: infos[a]["action_mask"] for a in env.possible_agents}
    agent_total_tp: dict[str, float] = {}
    link_util_series: list[np.ndarray] = []
    per_tick_median_rtt: list[float] = []

    while env.agents:
        obs_t = torch.from_numpy(
            np.stack([obs[a] for a in env.possible_agents], axis=0)
        ).to(device)
        mask_t = torch.from_numpy(
            np.stack([masks[a] for a in env.possible_agents], axis=0)
        ).to(device)
        with torch.no_grad():
            action, _ = actor.act(obs_t, mask_t, deterministic=True)
        actions = {a: int(action[i].item()) for i, a in enumerate(env.possible_agents)}
        obs, _, _, _, infos = env.step(actions)
        if not env.agents:
            break
        masks = {a: infos[a]["action_mask"] for a in env.agents}

        # Metric capture from env state.
        link_util_series.append(env.sim.link_util.copy())
        rtts: list[float] = []
        step = env._last_step
        for o in step.per_agent:
            if o.selected_path_fp is None:
                continue
            for ap, m in zip(o.available_paths, o.path_metrics, strict=True):
                if ap == o.selected_path_fp:
                    agent_total_tp[o.agent_id] = agent_total_tp.get(o.agent_id, 0.0) + float(m["bandwidth_mbps"])
                    rtts.append(float(m["rtt_ms"]))
                    break
        if rtts:
            per_tick_median_rtt.append(float(np.median(rtts)))

    from scion_sim.baselines.runner import episode_metrics
    return episode_metrics(
        episode_id=f"eval_{episode_seed}",
        agent_total_tp=agent_total_tp,
        link_util_series=link_util_series,
        per_tick_median_rtt=per_tick_median_rtt,
    )


def _eval_one_policy(payload: tuple[str, str, int]) -> dict:
    """Worker: load one policy.pt, run n_episodes, return summary."""
    import torch
    from scion_marl.agent import ActorNetwork
    from scion_marl.config import Phase2Config
    from scion_marl.device import select_device
    from scion_sim.env import make_env
    from scion_sim.env.observation_builder import PER_PATH_FEATURES

    ckpt_path_str, out_json_str, n_episodes = payload
    ckpt_path = Path(ckpt_path_str)
    out_json = Path(out_json_str)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = Phase2Config.model_validate(ckpt["config"])
    device_info = select_device()
    device = device_info.device

    obs_dim = cfg.phase1.observation.top_k * PER_PATH_FEATURES + 7
    actor = ActorNetwork(
        obs_dim=obs_dim, action_dim=cfg.phase1.observation.top_k,
        hidden_sizes=cfg.mappo.hidden_sizes, activation=cfg.mappo.activation,
    ).to(device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()

    env = make_env(cfg.phase1)
    t0 = time.perf_counter()
    per_episode: list[dict] = []
    for ep in range(n_episodes):
        seed = EVAL_SEED_OFFSET + cfg.seed * 1000 + ep
        m = _eval_one_episode(env, actor, seed, device, cfg.phase1.observation.top_k)
        per_episode.append(m)
    elapsed = time.perf_counter() - t0

    summary = _summarize(per_episode)
    payload_out = {
        "config_name": cfg.name,
        "seed": cfg.seed,
        "checkpoint": str(ckpt_path),
        "n_episodes": n_episodes,
        "elapsed_sec": elapsed,
        "device_backend": device_info.backend,
        "summary": summary,
        "episodes": per_episode,
    }
    out_json.write_text(json.dumps(payload_out, indent=2))
    return {"name": cfg.name, "seed": cfg.seed, "elapsed_sec": elapsed, "summary": summary, "out_json": str(out_json)}


def _summarize(per_episode: list[dict]) -> dict:
    keys = ("aggregate_throughput", "jain_fairness", "link_utilization_variance", "median_agent_rtt_ms")
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        arr = np.asarray([e[k] for e in per_episode], dtype=np.float64)
        out[k] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100, help="Episodes per policy.")
    p.add_argument("--only", type=str, default=None, help="Comma-separated config names.")
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    p.add_argument("--runs-root", type=str, default=str(RUNS_ROOT))
    args = p.parse_args()

    runs_root = Path(args.runs_root)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
    else:
        wanted = None

    payloads: list[tuple[str, str, int]] = []
    for ckpt_path in sorted(runs_root.glob("sweep_*/seed_*/policy_final.pt")):
        config_name = ckpt_path.parent.parent.name
        if wanted is not None and config_name not in wanted:
            continue
        out_json = ckpt_path.parent / "eval.json"
        payloads.append((str(ckpt_path), str(out_json), args.n))

    logger.info("found %d trained policies; running %d eval episodes each", len(payloads), args.n)
    if not payloads:
        logger.warning("no checkpoints found under %s", runs_root)
        return

    t0 = time.perf_counter()
    results: list[dict] = []
    if args.workers == 1:
        for payload in payloads:
            r = _eval_one_policy(payload)
            logger.info(
                "[%s/seed_%d] eval %.1fs throughput=%.0f rtt=%.0f var=%.4f",
                r["name"], r["seed"], r["elapsed_sec"],
                r["summary"]["aggregate_throughput"]["mean"],
                r["summary"]["median_agent_rtt_ms"]["mean"],
                r["summary"]["link_utilization_variance"]["mean"],
            )
            results.append(r)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            for r in pool.imap_unordered(_eval_one_policy, payloads, chunksize=1):
                logger.info(
                    "[%s/seed_%d] eval %.1fs throughput=%.0f rtt=%.0f var=%.4f",
                    r["name"], r["seed"], r["elapsed_sec"],
                    r["summary"]["aggregate_throughput"]["mean"],
                    r["summary"]["median_agent_rtt_ms"]["mean"],
                    r["summary"]["link_utilization_variance"]["mean"],
                )
                results.append(r)

    elapsed = time.perf_counter() - t0
    summary_path = runs_root / "evaluation_summary.json"
    summary_path.write_text(json.dumps({
        "elapsed_sec": elapsed, "n_policies": len(results), "results": results,
    }, indent=2))
    logger.info("evaluation complete in %.1fs (%.1fh); summary -> %s",
                elapsed, elapsed / 3600.0, summary_path)


if __name__ == "__main__":
    mp.freeze_support()
    main()
