"""Generate Phase 1 baseline rollouts and the comparison report.

Per PHASE_1_SPEC §2.6: 100 fresh evaluation episodes per baseline (3 ×
100 = 300 episodes total) with metrics on aggregate throughput, Jain
fairness, per-link utilization variance (oscillation), and median agent
RTT. Output: per-baseline parquet shards + manifest + a markdown report
at `data/baselines/baseline_report.md`.

Usage:
    uv run python scripts/run_baselines.py           # 100 episodes per baseline
    uv run python scripts/run_baselines.py --n 10    # smoke run
    uv run python scripts/run_baselines.py --workers 1
"""
from __future__ import annotations

import argparse
import logging
import math
import multiprocessing as mp
import time
from pathlib import Path
from typing import Callable

import numpy as np

from scion_sim.baselines import OracleGreedyPolicy, RandomPolicy, ShortestPathPolicy
from scion_sim.baselines.runner import write_baseline_shard
from scion_sim.config import SimulatorConfig, TrajectoryConfig
from scion_sim.env import Phase1Config

REPO_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("baselines")

# Dedicated seed offset; never collide with Phase 0's seed lineage so baselines
# are demonstrably "fresh rollouts" per J-014.
BASELINE_SEED_OFFSET = 1_000_000


def _make_policy_factory(name: str, cfg: Phase1Config) -> Callable[[str], object]:
    """Each agent gets its own policy instance (state isolation)."""
    if name == "random":
        # Distinct seed per agent so they don't draw the same actions in lockstep.
        return lambda agent_id: RandomPolicy(seed=hash(("random", agent_id)) & 0xFFFFFFFF)
    if name == "shortest_path":
        return lambda agent_id: ShortestPathPolicy()
    if name == "oracle_greedy":
        b = cfg.baseline
        return lambda agent_id: OracleGreedyPolicy(
            top_k=cfg.observation.top_k,
            ema_window=b.oracle_ema_window,
            loss_penalty=b.oracle_loss_penalty,
        )
    raise ValueError(f"unknown baseline: {name}")


def _shard_specs(n_episodes: int, shard_size: int) -> list[tuple[int, list[tuple[str, int]]]]:
    n_shards = math.ceil(n_episodes / shard_size)
    out = []
    for s in range(n_shards):
        lo = s * shard_size
        hi = min((s + 1) * shard_size, n_episodes)
        episodes = [
            (f"ep-{i:06d}", BASELINE_SEED_OFFSET + i) for i in range(lo, hi)
        ]
        out.append((s, episodes))
    return out


def _run_one_shard(args: tuple[str, int, list[tuple[str, int]], dict, str, str]) -> dict:
    name, shard_index, episode_specs, cfg_dict, repo_root_str, out_dir_str = args
    cfg = Phase1Config.model_validate(cfg_dict)
    factory = _make_policy_factory(name, cfg)
    _manifest, metrics = write_baseline_shard(
        cfg=cfg,
        shard_index=shard_index,
        episode_specs=episode_specs,
        policy_factory=factory,
        out_dir=Path(out_dir_str),
        repo_root=Path(repo_root_str),
    )
    return {"name": name, "shard_index": shard_index, **metrics}


def _summary_stats(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def _format_report(per_baseline: dict[str, list[dict]], cfg: Phase1Config) -> str:
    """Markdown report — one row per baseline per metric (mean ± std)."""
    lines = [
        "# Phase 1 Baseline Evaluation Report",
        "",
        f"- Episodes per baseline: {sum(len(v) for v in per_baseline.values()) // len(per_baseline)}",
        f"- Phase 1 config hash (sim portion): `{cfg.sim.hash()}`",
        f"- Oracle-greedy EMA window: {cfg.baseline.oracle_ema_window}",
        f"- Reward weights (v0): w_t={cfg.reward.w_throughput}, w_c={cfg.reward.w_congestion}, w_d={cfg.reward.w_diversity}",
        f"- Congestion knee: {cfg.reward.congestion_util_threshold}",
        "",
        "## Per-baseline metrics (mean ± std, across episodes)",
        "",
        "| Baseline | Aggregate Throughput | Jain Fairness | Link Util Variance | Median Agent RTT (ms) |",
        "|---|---|---|---|---|",
    ]
    for name in ("random", "shortest_path", "oracle_greedy"):
        episodes = per_baseline.get(name, [])
        if not episodes:
            lines.append(f"| {name} | (no episodes) | | | |")
            continue
        tp = [e["aggregate_throughput"] for e in episodes]
        jf = [e["jain_fairness"] for e in episodes]
        lu = [e["link_utilization_variance"] for e in episodes]
        rt = [e["median_agent_rtt_ms"] for e in episodes]
        tp_m, tp_s = _summary_stats(tp)
        jf_m, jf_s = _summary_stats(jf)
        lu_m, lu_s = _summary_stats(lu)
        rt_m, rt_s = _summary_stats(rt)
        lines.append(
            f"| {name} | {tp_m:.1f} ± {tp_s:.1f} "
            f"| {jf_m:.3f} ± {jf_s:.3f} "
            f"| {lu_m:.4f} ± {lu_s:.4f} "
            f"| {rt_m:.0f} ± {rt_s:.0f} |"
        )

    # Diagnostic checks expected by spec §3 / §5.
    lines.extend(["", "## Validation against spec expectations", ""])
    rand = per_baseline.get("random", [])
    sp = per_baseline.get("shortest_path", [])
    og = per_baseline.get("oracle_greedy", [])
    if rand and sp and og:
        rand_tp = float(np.mean([e["aggregate_throughput"] for e in rand]))
        sp_tp = float(np.mean([e["aggregate_throughput"] for e in sp]))
        og_tp = float(np.mean([e["aggregate_throughput"] for e in og]))
        sp_lu = float(np.mean([e["link_utilization_variance"] for e in sp]))
        og_lu = float(np.mean([e["link_utilization_variance"] for e in og]))
        ordering_ok = og_tp > sp_tp > rand_tp
        oscillation_ok = og_lu > sp_lu
        lines.append(f"- Throughput ordering oracle_greedy > shortest_path > random: **{'OK' if ordering_ok else 'FAIL'}**")
        lines.append(f"  - random: {rand_tp:.1f}, shortest_path: {sp_tp:.1f}, oracle_greedy: {og_tp:.1f}")
        lines.append(f"- Oracle-greedy oscillation > shortest-path: **{'OK' if oscillation_ok else 'FAIL'}**")
        lines.append(f"  - shortest_path link util variance: {sp_lu:.4f}, oracle_greedy: {og_lu:.4f}")
        if not (ordering_ok and oscillation_ok):
            lines.append("")
            lines.append("> **Surface to user.** One or more spec expectations were not met.")
            lines.append("> Per PHASE_1_SPEC §5: this is a design-validity question, not just tuning.")
    return "\n".join(lines) + "\n"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100, help="episodes per baseline")
    p.add_argument("--episode-seconds", type=float, default=3600.0)
    p.add_argument("--shard-size", type=int, default=25)
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    p.add_argument("--output-root", type=str, default="data/baselines")
    p.add_argument("--baselines", type=str, default="random,shortest_path,oracle_greedy")
    args = p.parse_args()

    base_sim_cfg = SimulatorConfig(
        trajectory=TrajectoryConfig(
            num_episodes=args.n,
            episode_seconds=args.episode_seconds,
            shard_size_episodes=args.shard_size,
            output_dir=args.output_root,  # overridden per-baseline below
        )
    )
    cfg = Phase1Config(sim=base_sim_cfg)
    cfg_dict = cfg.model_dump(mode="json")

    output_root = REPO_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    shard_specs = _shard_specs(args.n, args.shard_size)
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]

    payloads: list[tuple[str, int, list[tuple[str, int]], dict, str, str]] = []
    for name in baselines:
        out_dir = output_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for shard_index, episodes in shard_specs:
            payloads.append(
                (name, shard_index, episodes, cfg_dict, str(REPO_ROOT), str(out_dir))
            )

    logger.info(
        "running %d baselines × %d episodes = %d total (workers=%d)",
        len(baselines), args.n, len(baselines) * args.n, args.workers,
    )
    t0 = time.perf_counter()

    per_baseline: dict[str, list[dict]] = {b: [] for b in baselines}
    if args.workers == 1:
        for payload in payloads:
            r = _run_one_shard(payload)
            per_baseline[r["name"]].extend(r["episodes"])
            logger.info("[%s] shard %d done (%d eps)", r["name"], r["shard_index"], len(r["episodes"]))
    else:
        with mp.Pool(processes=args.workers) as pool:
            for r in pool.imap_unordered(_run_one_shard, payloads, chunksize=1):
                per_baseline[r["name"]].extend(r["episodes"])
                logger.info("[%s] shard %d done (%d eps)", r["name"], r["shard_index"], len(r["episodes"]))

    dt = time.perf_counter() - t0
    logger.info("all baselines complete in %.1fs (%.2fs/ep)", dt, dt / (len(baselines) * args.n))

    report = _format_report(per_baseline, cfg)
    report_path = output_root / "baseline_report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("report written to %s", report_path)

    # Stdout summary so the user sees the validation immediately.
    print()
    print(report)


if __name__ == "__main__":
    main()
