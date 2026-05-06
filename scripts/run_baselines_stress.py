"""Phase 1 baseline harness — stress regime (48 agents, 50 Mbps link caps).

One-off load test (ordered 2026-05-02): does the simulator produce the
spec-predicted oscillation ordering when pushed into a higher-contention
regime than the calibrated Phase 0 default? Same 100 eps × 3 baselines,
EMA=20, only `topology.num_agents` and `topology.link_capacity_mbps_mean`
differ from the canonical run.
"""
from __future__ import annotations

import argparse
import logging
import math
import multiprocessing as mp
import time
from pathlib import Path

from scion_sim.config import SimulatorConfig, TopologyConfig, TrajectoryConfig
from scion_sim.env import Phase1Config

from scripts.run_baselines import (  # noqa: E402  re-use orchestration
    _format_report,
    _run_one_shard,
    _shard_specs,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger("baselines_stress")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--episode-seconds", type=float, default=3600.0)
    p.add_argument("--shard-size", type=int, default=25)
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    p.add_argument("--output-root", type=str, default="data/baselines_stress")
    p.add_argument("--num-agents", type=int, default=48)
    p.add_argument("--link-capacity-mbps", type=float, default=50.0)
    p.add_argument("--baselines", type=str, default="random,shortest_path,oracle_greedy")
    args = p.parse_args()

    sim_cfg = SimulatorConfig(
        topology=TopologyConfig(
            num_agents=args.num_agents,
            paths_per_agent=12,
            num_links=64,
            link_capacity_mbps_mean=args.link_capacity_mbps,
        ),
        trajectory=TrajectoryConfig(
            num_episodes=args.n,
            episode_seconds=args.episode_seconds,
            shard_size_episodes=args.shard_size,
            output_dir=args.output_root,
        ),
    )
    cfg = Phase1Config(sim=sim_cfg)
    cfg_dict = cfg.model_dump(mode="json")

    output_root = REPO_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    shard_specs = _shard_specs(args.n, args.shard_size)
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]

    payloads = []
    for name in baselines:
        out_dir = output_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for shard_index, episodes in shard_specs:
            payloads.append(
                (name, shard_index, episodes, cfg_dict, str(REPO_ROOT), str(out_dir))
            )
    logger.info(
        "STRESS: %d agents, %.1f Mbps caps; %d baselines × %d eps = %d total (workers=%d)",
        args.num_agents, args.link_capacity_mbps,
        len(baselines), args.n, len(baselines) * args.n, args.workers,
    )
    t0 = time.perf_counter()
    per_baseline: dict[str, list[dict]] = {b: [] for b in baselines}
    with mp.Pool(processes=args.workers) as pool:
        for r in pool.imap_unordered(_run_one_shard, payloads, chunksize=1):
            per_baseline[r["name"]].extend(r["episodes"])
            logger.info("[%s] shard %d done (%d eps)", r["name"], r["shard_index"], len(r["episodes"]))
    dt = time.perf_counter() - t0
    logger.info("stress run complete: %.1fs (%.2fs/ep)", dt, dt / (len(baselines) * args.n))
    report = _format_report(per_baseline, cfg)
    report_path = output_root / "baseline_report_stress.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
