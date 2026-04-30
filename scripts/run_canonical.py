"""Generate the canonical Phase 0 trajectory dataset.

Per Task #11 in PHASE_0_SPEC and JUSTIFICATIONS.md J-013, this writes a
sequence of shards to `data/phase0_trajectories/` with per-shard manifests.

Usage:
    uv run python scripts/run_canonical.py             # 1000 episodes, multi-process
    uv run python scripts/run_canonical.py --n 100     # smaller pilot
    uv run python scripts/run_canonical.py --workers 1 # serial (debug)
"""
from __future__ import annotations

import argparse
import logging
import math
import multiprocessing as mp
import time
from pathlib import Path

from scion_sim.config import SimulatorConfig, TrajectoryConfig
from scion_sim.wrapper import write_shard

REPO_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("canonical")


def _shard_specs(cfg: SimulatorConfig, n_episodes: int) -> list[tuple[int, list[tuple[str, int]]]]:
    shard_size = cfg.trajectory.shard_size_episodes
    n_shards = math.ceil(n_episodes / shard_size)
    out: list[tuple[int, list[tuple[str, int]]]] = []
    for s in range(n_shards):
        lo = s * shard_size
        hi = min((s + 1) * shard_size, n_episodes)
        episodes = [(f"ep-{i:06d}", cfg.seed + i) for i in range(lo, hi)]
        out.append((s, episodes))
    return out


def _run_one_shard(args: tuple[int, list[tuple[str, int]], dict, str]) -> dict:
    shard_index, episode_specs, cfg_dict, repo_root_str = args
    cfg = SimulatorConfig.model_validate(cfg_dict)
    return write_shard(cfg, shard_index, episode_specs, Path(repo_root_str))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000, help="number of episodes")
    p.add_argument("--episode-seconds", type=float, default=3600.0)
    p.add_argument("--shard-size", type=int, default=50)
    p.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() // 2)))
    p.add_argument("--output-dir", type=str, default="data/phase0_trajectories")
    args = p.parse_args()

    cfg = SimulatorConfig(
        trajectory=TrajectoryConfig(
            num_episodes=args.n,
            episode_seconds=args.episode_seconds,
            shard_size_episodes=args.shard_size,
            output_dir=args.output_dir,
        )
    )
    logger.info("config hash %s", cfg.hash())
    shard_specs = _shard_specs(cfg, args.n)
    logger.info("running %d episodes in %d shards (workers=%d)", args.n, len(shard_specs), args.workers)

    t0 = time.perf_counter()
    cfg_dict = cfg.model_dump(mode="json")
    payloads = [(idx, eps, cfg_dict, str(REPO_ROOT)) for idx, eps in shard_specs]

    if args.workers == 1:
        for payload in payloads:
            m = _run_one_shard(payload)
            logger.info("shard %d done -> %s", m["shard_index"], m["dataset_hash"][:16])
    else:
        with mp.Pool(processes=args.workers) as pool:
            for m in pool.imap_unordered(_run_one_shard, payloads, chunksize=1):
                logger.info("shard %d done -> %s", m["shard_index"], m["dataset_hash"][:16])

    dt = time.perf_counter() - t0
    logger.info("canonical dataset complete: %d episodes in %.1fs (%.2fs/ep)", args.n, dt, dt / args.n)


if __name__ == "__main__":
    main()
