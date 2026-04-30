"""Smoke-test: run a tiny Phase 0 config end-to-end and write one shard.

    uv run python scripts/run_smoke.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from scion_sim.config import SimulatorConfig, TrajectoryConfig, TopologyConfig
from scion_sim.wrapper import write_shard

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    cfg = SimulatorConfig(
        trajectory=TrajectoryConfig(
            num_episodes=2,
            episode_seconds=60.0,
            shard_size_episodes=2,
            output_dir="data/_smoke",
        ),
        topology=TopologyConfig(num_agents=4, paths_per_agent=6, num_links=16),
    )
    print("config hash:", cfg.hash())
    t0 = time.perf_counter()
    manifest = write_shard(
        cfg,
        shard_index=0,
        episode_specs=[(f"ep-{i:06d}", cfg.seed + i) for i in range(cfg.trajectory.num_episodes)],
        repo_root=REPO_ROOT,
    )
    dt = time.perf_counter() - t0
    print(f"wrote shard in {dt:.2f}s")
    print(json.dumps({k: v for k, v in manifest.items() if k != "config_inline"}, indent=2))


if __name__ == "__main__":
    main()
