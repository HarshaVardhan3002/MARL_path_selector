"""Schema-conforming Parquet writer for Phase 0 trajectories.

Schema is locked in `docs/SCHEMA.md` (J-013). Two tables per shard:

* `<shard>.timesteps.parquet`
* `<shard>.events.parquet`

Plus a `<shard>.manifest.json`.
"""
from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from scion_sim.config import SimulatorConfig
from scion_sim.wrapper.simulator import Phase0Simulator, StepResult

SCHEMA_VERSION = 1
SIM_VERSION = "scion_sim 0.0.1"
PAMI_PINNED_SHA = "294c07979c1d58128ac3ec80e18288388d5bf60f"
SCIONPATHML_PINNED_SHA = "362ad60885dabeb2a067d3dae058c506ce5f092a"

GLOBAL_AGENT = "__global__"

PATH_METRIC_STRUCT = pa.struct([
    pa.field("fp", pa.string()),
    pa.field("rtt_ms", pa.float64()),
    pa.field("bandwidth_mbps", pa.float64()),
    pa.field("loss_pct", pa.float64()),
    pa.field("jitter_ms", pa.float64()),
    pa.field("age_sec", pa.float64()),
])
LINK_UTIL_STRUCT = pa.struct([
    pa.field("link_id", pa.int32()),
    pa.field("utilization", pa.float64()),
])
ASSIGNMENT_STRUCT = pa.struct([
    pa.field("agent_id", pa.string()),
    pa.field("fp", pa.string()),
])
REWARD_COMPONENTS_STRUCT = pa.struct([
    pa.field("throughput", pa.float64()),
    pa.field("congestion", pa.float64()),
    pa.field("diversity", pa.float64()),
])

TIMESTEP_SCHEMA = pa.schema([
    pa.field("episode_id", pa.string()),
    pa.field("t", pa.int32()),
    pa.field("wall_time_sec", pa.float64()),
    pa.field("agent_id", pa.string()),
    pa.field("selected_path_fp", pa.string()),
    pa.field("available_paths", pa.list_(pa.string())),
    pa.field("path_metrics", pa.list_(PATH_METRIC_STRUCT)),
    pa.field("reward", pa.float64()),
    pa.field("reward_components", REWARD_COMPONENTS_STRUCT),
    pa.field("link_utilizations", pa.list_(LINK_UTIL_STRUCT)),
    pa.field("total_throughput", pa.float64()),
    pa.field("agent_assignments", pa.list_(ASSIGNMENT_STRUCT)),
])
EVENT_SCHEMA = pa.schema([
    pa.field("episode_id", pa.string()),
    pa.field("t", pa.int32()),
    pa.field("wall_time_sec", pa.float64()),
    pa.field("agent_id", pa.string()),
    pa.field("fingerprint", pa.string()),
    pa.field("event_type", pa.string()),
])


def _git_sha(repo_root: Path) -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(repo_root), stderr=subprocess.DEVNULL
        ).decode().strip())
        return sha, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "uninitialized", True


def _agent_row(episode_id: str, step: StepResult, obs) -> dict:
    return {
        "episode_id": episode_id,
        "t": step.t,
        "wall_time_sec": step.wall_time_sec,
        "agent_id": obs.agent_id,
        "selected_path_fp": obs.selected_path_fp,
        "available_paths": obs.available_paths,
        "path_metrics": obs.path_metrics,
        "reward": None,
        "reward_components": None,
        "link_utilizations": None,
        "total_throughput": None,
        "agent_assignments": None,
    }


def _global_row(episode_id: str, step: StepResult) -> dict:
    return {
        "episode_id": episode_id,
        "t": step.t,
        "wall_time_sec": step.wall_time_sec,
        "agent_id": GLOBAL_AGENT,
        "selected_path_fp": None,
        "available_paths": None,
        "path_metrics": None,
        "reward": None,
        "reward_components": None,
        "link_utilizations": step.global_state.link_utilizations,
        "total_throughput": step.global_state.total_throughput,
        "agent_assignments": step.global_state.agent_assignments,
    }


def _event_rows(episode_id: str, step: StepResult) -> list[dict]:
    rows = []
    for t_sec, agent_id, fp, etype in step.events:
        rows.append({
            "episode_id": episode_id,
            "t": step.t,
            "wall_time_sec": t_sec,
            "agent_id": agent_id,
            "fingerprint": fp,
            "event_type": etype,
        })
    return rows


def stream_episode(
    cfg: SimulatorConfig, episode_id: str, episode_seed: int
) -> Iterator[tuple[list[dict], list[dict]]]:
    """Yield per-observation row batches for one episode.

    Memory-bounded: each yielded chunk covers one observation tick (one
    set of per-agent rows + one global row + that tick's events), so peak
    in-memory rows ~ num_agents+1, not the full episode.
    """
    sim = Phase0Simulator(cfg, episode_seed=episode_seed)
    for step in sim.run_episode():
        ts = [_agent_row(episode_id, step, obs) for obs in step.per_agent]
        ts.append(_global_row(episode_id, step))
        ev = _event_rows(episode_id, step)
        yield ts, ev


def write_shard(
    cfg: SimulatorConfig,
    shard_index: int,
    episode_specs: Iterable[tuple[str, int]],
    repo_root: Path,
    write_batch_obs: int = 600,
) -> dict:
    """Stream-write a shard of episodes to per-shard Parquet files.

    Per-episode rows are accumulated up to `write_batch_obs` observation
    ticks (default 600 ticks ~ 600 sec ~ 1/6 of an episode) before being
    flushed to disk via a persistent `ParquetWriter`. This caps per-worker
    peak memory at roughly `write_batch_obs * (num_agents + 1)` rows.
    """
    out_dir = repo_root / cfg.trajectory.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ts_path = out_dir / f"shard_{shard_index:04d}.timesteps.parquet"
    ev_path = out_dir / f"shard_{shard_index:04d}.events.parquet"
    manifest_path = out_dir / f"shard_{shard_index:04d}.manifest.json"

    episode_ids: list[str] = []
    ts_writer = pq.ParquetWriter(ts_path, TIMESTEP_SCHEMA, compression="zstd")
    ev_writer = pq.ParquetWriter(ev_path, EVENT_SCHEMA, compression="zstd")
    try:
        ts_buf: list[dict] = []
        ev_buf: list[dict] = []
        ticks_in_buf = 0
        for episode_id, episode_seed in episode_specs:
            episode_ids.append(episode_id)
            for tick_ts, tick_ev in stream_episode(cfg, episode_id, episode_seed):
                ts_buf.extend(tick_ts)
                ev_buf.extend(tick_ev)
                ticks_in_buf += 1
                if ticks_in_buf >= write_batch_obs:
                    ts_writer.write_table(pa.Table.from_pylist(ts_buf, schema=TIMESTEP_SCHEMA))
                    if ev_buf:
                        ev_writer.write_table(pa.Table.from_pylist(ev_buf, schema=EVENT_SCHEMA))
                    ts_buf.clear()
                    ev_buf.clear()
                    ticks_in_buf = 0
        if ts_buf:
            ts_writer.write_table(pa.Table.from_pylist(ts_buf, schema=TIMESTEP_SCHEMA))
        if ev_buf:
            ev_writer.write_table(pa.Table.from_pylist(ev_buf, schema=EVENT_SCHEMA))
    finally:
        ts_writer.close()
        ev_writer.close()

    sha = hashlib.sha256()
    for p in (ts_path, ev_path):
        sha.update(p.read_bytes())
    git_sha, git_dirty = _git_sha(repo_root)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "shard_index": shard_index,
        "shard_size_episodes": cfg.trajectory.shard_size_episodes,
        "episode_ids": episode_ids,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "config_hash": cfg.hash(),
        "config_inline": cfg.model_dump(mode="json"),
        "dataset_hash": "sha256:" + sha.hexdigest(),
        "sim_version": SIM_VERSION,
        "pami_pinned_sha": PAMI_PINNED_SHA,
        "scionpathml_pinned_sha": SCIONPATHML_PINNED_SHA,
        "created_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest
