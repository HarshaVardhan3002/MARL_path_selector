"""Phase 2 selection + report generator.

Reads all `runs/phase2/{config}/seed_{seed}/eval.json` produced by
`run_phase2_evaluation.py`, aggregates per-config (mean ± std across
seeds), identifies the Pareto frontier across (throughput, latency,
oscillation), picks the most balanced Pareto-optimal config as the
winner, and produces:

* `runs/phase2/PHASE_2_REPORT.md` — summary table + Pareto + winner
* `runs/phase2/figures/throughput_vs_latency.png`
* `runs/phase2/figures/throughput_vs_oscillation.png`
* `runs/phase2/figures/training_curves_{winner}.png`
* `runs/phase2/winner/policy.pt` — copied from winning seed's checkpoint
* `runs/phase2/winner/manifest.json` — winner provenance
* `runs/phase2/winner/rollouts.parquet` — 100 fresh rollouts under winner (optional)

Usage:
    uv run python scripts/run_phase2_select_and_report.py
    uv run python scripts/run_phase2_select_and_report.py --skip-rollouts
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = REPO_ROOT / "runs" / "phase2"
DEFAULT_BASELINE_REPORT = REPO_ROOT / "data" / "baselines_stress_50a35m" / "baseline_report_stress.md"

logger = logging.getLogger("phase2_report")

METRIC_KEYS = (
    "aggregate_throughput",
    "jain_fairness",
    "link_utilization_variance",
    "median_agent_rtt_ms",
)
# Direction of "better" per metric, used by the Pareto check.
HIGHER_IS_BETTER = {
    "aggregate_throughput": True,
    "jain_fairness": True,
    "link_utilization_variance": False,
    "median_agent_rtt_ms": False,
}
PARETO_AXES = ("aggregate_throughput", "median_agent_rtt_ms", "link_utilization_variance")


# ---------- Baseline parsing ----------------------------------------------

_ROW_RE = re.compile(
    r"^\|\s*(?P<name>random|shortest_path|oracle_greedy)\s*\|"
    r"\s*(?P<tp>[\d.eE+\-]+)\s*[±\+/\-]+\s*(?P<tp_s>[\d.eE+\-]+)\s*\|"
    r"\s*(?P<jf>[\d.eE+\-]+)\s*[±\+/\-]+\s*(?P<jf_s>[\d.eE+\-]+)\s*\|"
    r"\s*(?P<lv>[\d.eE+\-]+)\s*[±\+/\-]+\s*(?P<lv_s>[\d.eE+\-]+)\s*\|"
    r"\s*(?P<rt>[\d.eE+\-]+)\s*[±\+/\-]+\s*(?P<rt_s>[\d.eE+\-]+)\s*\|",
)


def parse_baseline_report(report_path: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Parse the Phase 1 stress baseline_report_stress.md headline numbers.

    Returns: {baseline_name: {metric: {"mean": float, "std": float}}}.
    """
    if not report_path.exists():
        logger.warning("no baseline report at %s; comparison section will be empty", report_path)
        return {}
    out: dict[str, dict[str, dict[str, float]]] = {}
    for line in report_path.read_text(encoding="utf-8").splitlines():
        m = _ROW_RE.match(line)
        if not m:
            continue
        gd = m.groupdict()
        out[gd["name"]] = {
            "aggregate_throughput": {"mean": float(gd["tp"]), "std": float(gd["tp_s"])},
            "jain_fairness": {"mean": float(gd["jf"]), "std": float(gd["jf_s"])},
            "link_utilization_variance": {"mean": float(gd["lv"]), "std": float(gd["lv_s"])},
            "median_agent_rtt_ms": {"mean": float(gd["rt"]), "std": float(gd["rt_s"])},
        }
    return out


# ---------- Eval aggregation ----------------------------------------------

def load_evaluations(runs_root: Path) -> dict[str, list[dict]]:
    """Group per-policy eval.json files by config name."""
    out: dict[str, list[dict]] = {}
    for eval_path in sorted(runs_root.glob("sweep_*/seed_*/eval.json")):
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        out.setdefault(payload["config_name"], []).append({
            "seed": payload["seed"],
            "checkpoint": payload["checkpoint"],
            "summary": payload["summary"],
            "episodes": payload.get("episodes", []),
            "eval_path": str(eval_path),
        })
    return out


def aggregate_per_config(evals: dict[str, list[dict]]) -> dict[str, dict]:
    """Per config: mean ± std across seeds for each metric."""
    out: dict[str, dict] = {}
    for name, seed_evals in evals.items():
        per_metric: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
        for r in seed_evals:
            for k in METRIC_KEYS:
                per_metric[k].append(r["summary"][k]["mean"])
        agg = {}
        for k, vs in per_metric.items():
            arr = np.asarray(vs, dtype=np.float64)
            agg[k] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "per_seed": vs}
        out[name] = {"per_seed_summaries": seed_evals, "aggregate": agg}
    return out


# ---------- Pareto + selection --------------------------------------------

def _dominates(a: dict, b: dict, axes: tuple[str, ...]) -> bool:
    """True iff `a` Pareto-dominates `b`: better-or-equal on all axes, strict on one."""
    strict = False
    for k in axes:
        a_val = a[k]["mean"]
        b_val = b[k]["mean"]
        if HIGHER_IS_BETTER[k]:
            if a_val < b_val:
                return False
            if a_val > b_val:
                strict = True
        else:
            if a_val > b_val:
                return False
            if a_val < b_val:
                strict = True
    return strict


def pareto_frontier(per_config: dict[str, dict], axes: tuple[str, ...] = PARETO_AXES) -> list[str]:
    """Return config names on the Pareto frontier (non-dominated)."""
    frontier: list[str] = []
    aggs = {n: c["aggregate"] for n, c in per_config.items()}
    for name, agg in aggs.items():
        dominated = False
        for other_name, other_agg in aggs.items():
            if other_name == name:
                continue
            if _dominates(other_agg, agg, axes):
                dominated = True
                break
        if not dominated:
            frontier.append(name)
    return sorted(frontier)


def pick_balanced_winner(
    per_config: dict[str, dict], frontier: list[str], axes: tuple[str, ...] = PARETO_AXES,
) -> tuple[str, dict]:
    """Pick the Pareto config closest to the normalized 'best corner'.

    Each axis is min-max normalized to [0, 1] over the frontier (with
    'better' = 1.0). The winner minimizes Euclidean distance to (1,1,1).
    Tiebreaker (per spec auto-resolve): lowest oscillation.
    """
    if not frontier:
        raise ValueError("Pareto frontier is empty (no eval data?)")
    if len(frontier) == 1:
        return frontier[0], {"distance": 0.0, "tiebreaker_applied": False}

    aggs = {n: per_config[n]["aggregate"] for n in frontier}
    normed: dict[str, dict[str, float]] = {n: {} for n in frontier}
    for k in axes:
        vals = np.asarray([aggs[n][k]["mean"] for n in frontier], dtype=np.float64)
        lo, hi = float(vals.min()), float(vals.max())
        for n in frontier:
            v = aggs[n][k]["mean"]
            if hi - lo < 1e-12:
                normed[n][k] = 0.5
            else:
                rel = (v - lo) / (hi - lo)
                normed[n][k] = rel if HIGHER_IS_BETTER[k] else (1.0 - rel)

    distances = {
        n: float(np.sqrt(sum((1.0 - normed[n][k]) ** 2 for k in axes)))
        for n in frontier
    }
    min_d = min(distances.values())
    near = [n for n, d in distances.items() if abs(d - min_d) < 1e-6]
    tiebreaker_applied = False
    if len(near) > 1:
        tiebreaker_applied = True
        # Lowest oscillation wins.
        near.sort(key=lambda n: aggs[n]["link_utilization_variance"]["mean"])
    winner = near[0]
    return winner, {
        "distance": distances[winner],
        "all_distances": distances,
        "tiebreaker_applied": tiebreaker_applied,
    }


def pick_winner_seed(per_config_winner: dict) -> int:
    """Choose the seed whose eval is closest to the seed-averaged metrics."""
    seeds = per_config_winner["per_seed_summaries"]
    agg = per_config_winner["aggregate"]
    best_seed = None
    best_score = float("inf")
    for r in seeds:
        score = 0.0
        for k in METRIC_KEYS:
            mu = agg[k]["mean"]
            sd = max(agg[k]["std"], 1e-9)
            score += abs(r["summary"][k]["mean"] - mu) / sd
        if score < best_score:
            best_score = score
            best_seed = r["seed"]
    assert best_seed is not None
    return best_seed


# ---------- Figures --------------------------------------------------------

def _scatter_with_baselines(
    per_config: dict[str, dict], baselines: dict[str, dict], frontier: list[str],
    x_key: str, y_key: str, x_label: str, y_label: str, out_path: Path, title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, c in sorted(per_config.items()):
        agg = c["aggregate"]
        is_pareto = name in frontier
        ax.errorbar(
            agg[x_key]["mean"], agg[y_key]["mean"],
            xerr=agg[x_key]["std"], yerr=agg[y_key]["std"],
            fmt="o" if is_pareto else "x",
            ms=10 if is_pareto else 7,
            mfc="tab:orange" if is_pareto else "tab:blue",
            mec="black", capsize=3, label=name + (" (Pareto)" if is_pareto else ""),
        )
    for bname, bm in baselines.items():
        ax.errorbar(
            bm[x_key]["mean"], bm[y_key]["mean"],
            xerr=bm[x_key]["std"], yerr=bm[y_key]["std"],
            fmt="s", ms=12, mfc="lightgray", mec="black",
            capsize=3, label=f"baseline:{bname}",
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_training_curves(winner_name: str, runs_root: Path, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    n_seeds_found = 0
    for seed_dir in sorted((runs_root / winner_name).glob("seed_*")):
        jsonl = seed_dir / "metrics.jsonl"
        if not jsonl.exists():
            continue
        episodes: list[int] = []
        rewards: list[float] = []
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            episodes.append(row["episodes_done"])
            rewards.append(row["episode_reward"])
        if episodes:
            ax.plot(episodes, rewards, label=seed_dir.name, alpha=0.8, lw=1.5)
            n_seeds_found += 1
    if n_seeds_found == 0:
        return
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Episode total reward (per rollout)")
    ax.set_title(f"Training curves — {winner_name}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------- Report rendering ----------------------------------------------

def render_report(
    per_config: dict[str, dict],
    baselines: dict[str, dict],
    frontier: list[str],
    winner: str,
    winner_diag: dict,
    winner_seed: int,
    figs_dir: Path,
) -> str:
    lines: list[str] = []
    lines.append("# Phase 2 — MAPPO Sweep Report")
    lines.append("")
    lines.append(f"- Configs evaluated: {len(per_config)}")
    lines.append(f"- Pareto frontier: {', '.join(frontier) if frontier else '(empty)'}")
    lines.append(f"- Recommended winner: **{winner}** (seed {winner_seed})")
    lines.append(f"  - distance to ideal corner (normalized): {winner_diag['distance']:.4f}")
    if winner_diag.get("tiebreaker_applied"):
        lines.append("  - tiebreaker (lowest oscillation) applied")
    lines.append("")
    lines.append("## Per-config metrics (mean ± std across 3 seeds)")
    lines.append("")
    lines.append("| Config | Aggregate Throughput | Jain Fairness | Link Util Variance | Median RTT (ms) | Pareto |")
    lines.append("|---|---|---|---|---|---|")
    for name in sorted(per_config):
        agg = per_config[name]["aggregate"]
        marker = "✓" if name in frontier else ""
        lines.append(
            f"| {name} | {agg['aggregate_throughput']['mean']:.0f} ± {agg['aggregate_throughput']['std']:.0f} "
            f"| {agg['jain_fairness']['mean']:.3f} ± {agg['jain_fairness']['std']:.3f} "
            f"| {agg['link_utilization_variance']['mean']:.4f} ± {agg['link_utilization_variance']['std']:.4f} "
            f"| {agg['median_agent_rtt_ms']['mean']:.0f} ± {agg['median_agent_rtt_ms']['std']:.0f} "
            f"| {marker} |"
        )
    lines.append("")
    if baselines:
        lines.append("## Phase 1 baseline reference (stress regime, n=100)")
        lines.append("")
        lines.append("| Baseline | Aggregate Throughput | Jain Fairness | Link Util Variance | Median RTT (ms) |")
        lines.append("|---|---|---|---|---|")
        for name in ("random", "shortest_path", "oracle_greedy"):
            if name not in baselines:
                continue
            b = baselines[name]
            lines.append(
                f"| {name} | {b['aggregate_throughput']['mean']:.0f} ± {b['aggregate_throughput']['std']:.0f} "
                f"| {b['jain_fairness']['mean']:.3f} ± {b['jain_fairness']['std']:.3f} "
                f"| {b['link_utilization_variance']['mean']:.4f} ± {b['link_utilization_variance']['std']:.4f} "
                f"| {b['median_agent_rtt_ms']['mean']:.0f} ± {b['median_agent_rtt_ms']['std']:.0f} |"
            )
        lines.append("")

    if winner in per_config and baselines:
        lines.append(f"## Winner ({winner}) vs baselines")
        lines.append("")
        wagg = per_config[winner]["aggregate"]
        lines.append("| Metric | Winner | random | shortest_path | oracle_greedy |")
        lines.append("|---|---|---|---|---|")
        for k, label in (
            ("aggregate_throughput", "Aggregate Throughput"),
            ("jain_fairness", "Jain Fairness"),
            ("link_utilization_variance", "Link Util Variance"),
            ("median_agent_rtt_ms", "Median RTT (ms)"),
        ):
            row = [f"| {label} | {wagg[k]['mean']:.4f}"]
            for bname in ("random", "shortest_path", "oracle_greedy"):
                if bname in baselines:
                    row.append(f"{baselines[bname][k]['mean']:.4f}")
                else:
                    row.append("—")
            lines.append(" | ".join(row + [""]))
        lines.append("")

    lines.append("## Figures")
    lines.append("")
    lines.append(f"- ![throughput vs latency]({figs_dir.name}/throughput_vs_latency.png)")
    lines.append(f"- ![throughput vs oscillation]({figs_dir.name}/throughput_vs_oscillation.png)")
    lines.append(f"- ![training curves]({figs_dir.name}/training_curves_{winner}.png)")
    lines.append("")
    lines.append("## Selection rule")
    lines.append("")
    lines.append("Per PHASE_2_SPEC §1.6: identify Pareto frontier across "
                 "(throughput, latency, oscillation), then pick the config closest "
                 "to the normalized 'best corner' (1,1,1 in min-max normalized "
                 "space). Tiebreaker: lowest oscillation. Per-seed selection: "
                 "lowest evaluation variance vs the seed-averaged means.")
    return "\n".join(lines) + "\n"


# ---------- Winner export -------------------------------------------------

def export_winner(
    runs_root: Path, winner_name: str, winner_seed: int, per_config: dict, n_rollout_episodes: int,
    skip_rollouts: bool,
) -> dict:
    winner_dir = runs_root / "winner"
    winner_dir.mkdir(parents=True, exist_ok=True)

    # Source checkpoint comes from the eval.json record (handles renamed dirs).
    seed_summaries = per_config[winner_name]["per_seed_summaries"]
    matching = [r for r in seed_summaries if r["seed"] == winner_seed]
    if not matching:
        raise RuntimeError(f"no per-seed summary found for {winner_name} seed {winner_seed}")
    src_ckpt = Path(matching[0]["checkpoint"])
    dst_ckpt = winner_dir / "policy.pt"
    shutil.copy2(src_ckpt, dst_ckpt)

    manifest = {
        "winner_config": winner_name,
        "winner_seed": winner_seed,
        "checkpoint_source": str(src_ckpt),
        "aggregate_metrics": per_config[winner_name]["aggregate"],
        "per_seed": [
            {"seed": r["seed"], "summary": r["summary"]}
            for r in per_config[winner_name]["per_seed_summaries"]
        ],
        "rollouts_path": None if skip_rollouts else str(winner_dir / "rollouts.parquet"),
        "rollouts_episodes": 0 if skip_rollouts else n_rollout_episodes,
    }
    (winner_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if not skip_rollouts:
        # Lazy import — heavy.
        from scion_marl.config import Phase2Config
        from scion_sim.baselines.runner import write_baseline_shard

        ckpt = _torch_load(src_ckpt)
        cfg = Phase2Config.model_validate(ckpt["config"])
        policy_factory = _make_actor_policy_factory(src_ckpt, cfg)
        write_baseline_shard(
            cfg=cfg.phase1,
            shard_index=0,
            episode_specs=[(f"winner_eval_{i:04d}", 3_000_000 + i) for i in range(n_rollout_episodes)],
            policy_factory=policy_factory,
            out_dir=winner_dir,
            repo_root=REPO_ROOT,
        )

    return manifest


def _torch_load(path: Path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def _make_actor_policy_factory(ckpt_path: Path, cfg) -> object:
    """Wrap the trained actor as a per-agent callable matching the baseline harness API."""
    import numpy as np
    import torch

    from scion_marl.agent import ActorNetwork
    from scion_sim.env.observation_builder import PER_PATH_FEATURES

    obs_dim = cfg.phase1.observation.top_k * PER_PATH_FEATURES + 7
    actor = ActorNetwork(
        obs_dim=obs_dim, action_dim=cfg.phase1.observation.top_k,
        hidden_sizes=cfg.mappo.hidden_sizes, activation=cfg.mappo.activation,
    )
    ckpt = _torch_load(ckpt_path)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()

    class _ActorAsPolicy:
        name = "phase2_winner"
        def reset(self) -> None: ...
        def __call__(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
            with torch.no_grad():
                obs_t = torch.from_numpy(observation).unsqueeze(0).float()
                mask_t = torch.from_numpy(action_mask).unsqueeze(0)
                action, _ = actor.act(obs_t, mask_t, deterministic=True)
                return int(action.item())

    def factory(_agent_id: str) -> object:
        return _ActorAsPolicy()

    return factory


# ---------- Main ----------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", type=str, default=str(RUNS_ROOT))
    p.add_argument("--baseline-report", type=str, default=str(DEFAULT_BASELINE_REPORT))
    p.add_argument("--rollout-episodes", type=int, default=100)
    p.add_argument("--skip-rollouts", action="store_true",
                   help="Skip running 100 fresh rollouts under the winner (saves compute).")
    args = p.parse_args()

    runs_root = Path(args.runs_root)
    figs_dir = runs_root / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    evals = load_evaluations(runs_root)
    if not evals:
        logger.error("no eval.json files found under %s; run run_phase2_evaluation.py first", runs_root)
        return
    per_config = aggregate_per_config(evals)
    frontier = pareto_frontier(per_config)
    winner, diag = pick_balanced_winner(per_config, frontier)
    winner_seed = pick_winner_seed(per_config[winner])

    baselines = parse_baseline_report(Path(args.baseline_report))

    _scatter_with_baselines(
        per_config, baselines, frontier,
        x_key="median_agent_rtt_ms", y_key="aggregate_throughput",
        x_label="Median agent RTT (ms)  [lower is better]",
        y_label="Aggregate throughput  [higher is better]",
        out_path=figs_dir / "throughput_vs_latency.png",
        title="Phase 2 sweep — throughput vs latency",
    )
    _scatter_with_baselines(
        per_config, baselines, frontier,
        x_key="link_utilization_variance", y_key="aggregate_throughput",
        x_label="Link utilization variance  [lower is better]",
        y_label="Aggregate throughput  [higher is better]",
        out_path=figs_dir / "throughput_vs_oscillation.png",
        title="Phase 2 sweep — throughput vs oscillation",
    )
    plot_training_curves(winner, runs_root, figs_dir / f"training_curves_{winner}.png")

    report_text = render_report(per_config, baselines, frontier, winner, diag, winner_seed, figs_dir)
    report_path = runs_root / "PHASE_2_REPORT.md"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("report written to %s", report_path)

    t0 = time.perf_counter()
    manifest = export_winner(
        runs_root=runs_root, winner_name=winner, winner_seed=winner_seed,
        per_config=per_config, n_rollout_episodes=args.rollout_episodes,
        skip_rollouts=args.skip_rollouts,
    )
    if not args.skip_rollouts:
        logger.info("winner rollouts written in %.1fs", time.perf_counter() - t0)
    logger.info("winner manifest: %s", json.dumps({k: manifest[k] for k in ("winner_config", "winner_seed", "rollouts_episodes")}))


if __name__ == "__main__":
    main()
