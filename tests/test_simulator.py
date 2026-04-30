"""Phase 0 minimum test set per PHASE_0_SPEC.md task 5."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from scion_sim.config import (
    DiscrepancyConfig,
    SimulatorConfig,
    TopologyConfig,
    TrajectoryConfig,
)
from scion_sim.control_plane.topology import sample_lifetime
from scion_sim.wrapper import Phase0Simulator, write_shard
from scion_sim.wrapper.trajectories import EVENT_SCHEMA, TIMESTEP_SCHEMA


def _tiny_cfg(**overrides) -> SimulatorConfig:
    base = SimulatorConfig(
        trajectory=TrajectoryConfig(
            num_episodes=1,
            episode_seconds=10.0,
            shard_size_episodes=1,
            output_dir="data/_test",
        ),
        topology=TopologyConfig(num_agents=3, paths_per_agent=4, num_links=8),
    )
    return base.model_copy(update=overrides)


def test_simulator_runs_without_crash() -> None:
    cfg = _tiny_cfg()
    sim = Phase0Simulator(cfg)
    n_steps = 0
    for step in sim.run_episode():
        n_steps += 1
        assert step.t == n_steps - 1
    assert n_steps == 10  # 10s episode at observation_dt=1.0


def test_lifetimes_finite_and_positive() -> None:
    rng = np.random.default_rng(0)
    cfg = _tiny_cfg()
    samples = np.asarray([sample_lifetime(cfg.lifetime, rng) for _ in range(2000)])
    assert np.all(samples > 0.0), "all lifetimes must be strictly positive"
    assert np.all(np.isfinite(samples)), "all lifetimes must be finite"


def test_discrepancy_rate_positive_when_configured() -> None:
    """A non-zero asymmetry_rate must produce paths that are visible-to-owner only."""
    cfg = _tiny_cfg(discrepancy=DiscrepancyConfig(asymmetry_rate=0.5))
    sim = Phase0Simulator(cfg)
    for _ in sim.run_episode():
        pass
    asymmetric = [r for r in sim.control.all_records() if not r.visible_to_reverse]
    total = list(sim.control.all_records())
    assert total, "expected at least some live paths"
    rate = len(asymmetric) / len(total)
    assert rate > 0.05, f"asymmetry_rate=0.5 should yield non-trivial asymmetric births, got {rate:.3f}"


def test_zero_discrepancy_when_rate_zero() -> None:
    cfg = _tiny_cfg(discrepancy=DiscrepancyConfig(asymmetry_rate=0.0))
    sim = Phase0Simulator(cfg)
    for _ in sim.run_episode():
        pass
    asymmetric = [r for r in sim.control.all_records() if not r.visible_to_reverse]
    assert not asymmetric, "rate=0 should yield zero asymmetric births"


def test_trajectory_conforms_to_schema(tmp_path: Path) -> None:
    cfg = _tiny_cfg(
        trajectory=TrajectoryConfig(
            num_episodes=2,
            episode_seconds=5.0,
            shard_size_episodes=2,
            output_dir=str(tmp_path / "traj"),
        ),
    )
    write_shard(
        cfg,
        shard_index=0,
        episode_specs=[(f"ep-{i:06d}", cfg.seed + i) for i in range(2)],
        repo_root=tmp_path,
    )
    ts_path = tmp_path / "traj" / "shard_0000.timesteps.parquet"
    ev_path = tmp_path / "traj" / "shard_0000.events.parquet"
    manifest_path = tmp_path / "traj" / "shard_0000.manifest.json"
    for p in (ts_path, ev_path, manifest_path):
        assert p.exists(), f"missing {p}"
    # Schema-level conformance: column names and types match TIMESTEP_SCHEMA.
    ts_table = pq.read_table(ts_path)
    assert ts_table.schema.names == TIMESTEP_SCHEMA.names
    for f_obs, f_exp in zip(ts_table.schema, TIMESTEP_SCHEMA):
        assert f_obs.type.equals(f_exp.type), f"{f_obs.name}: got {f_obs.type}, want {f_exp.type}"
    ev_table = pq.read_table(ev_path)
    assert ev_table.schema.names == EVENT_SCHEMA.names
    # Row-count expectation: 2 ep * 5 obs * (3 agents + 1 global) = 40 rows
    assert ts_table.num_rows == 2 * 5 * (3 + 1)
    # Path-appeared events at t=0 are emitted for the initial visible pool.
    df = ev_table.to_pandas()
    assert (df["event_type"] == "path_appeared").any()


def test_fingerprint_format() -> None:
    cfg = _tiny_cfg()
    sim = Phase0Simulator(cfg)
    for rec in sim.control.all_records():
        assert len(rec.fingerprint) == 16, "fingerprints must be 16 hex chars"
        int(rec.fingerprint, 16)  # raises if not valid hex


def test_lifetime_distribution_median_matches_config() -> None:
    """Sanity: sampling the configured lognormal recovers the configured median within ~5%."""
    cfg = _tiny_cfg()
    rng = np.random.default_rng(42)
    samples = np.asarray([sample_lifetime(cfg.lifetime, rng) for _ in range(50_000)])
    rel_err = abs(float(np.median(samples)) - cfg.lifetime.median_seconds) / cfg.lifetime.median_seconds
    assert rel_err < 0.05, f"sample median {np.median(samples):.0f} vs configured {cfg.lifetime.median_seconds}, rel_err={rel_err:.3f}"


def test_config_hash_is_stable_under_no_op_roundtrip() -> None:
    cfg = _tiny_cfg()
    h1 = cfg.hash()
    cfg2 = SimulatorConfig.model_validate(cfg.model_dump(mode="json"))
    h2 = cfg2.hash()
    assert h1 == h2


def test_global_row_present_per_timestep() -> None:
    cfg = _tiny_cfg()
    sim = Phase0Simulator(cfg)
    for step in sim.run_episode():
        assert step.global_state is not None
        assert math.isfinite(step.global_state.total_throughput)
