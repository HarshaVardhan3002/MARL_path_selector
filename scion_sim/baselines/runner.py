"""Run a baseline policy through the Phase 1 env and stream rows to Parquet.

Mirrors `scion_sim.wrapper.trajectories.write_shard` but drives episodes
through `ScionMARLEnv` so that `selected_path_fp` reflects policy decisions
rather than the Phase 0 placeholder selector. Reward columns are still left
null (Phase 1 reward components are logged in info, not the schema; the
schema is shared with Phase 0).
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scion_sim.env import Phase1Config, make_env
from scion_sim.wrapper.trajectories import (
    EVENT_SCHEMA,
    PAMI_PINNED_SHA,
    SCHEMA_VERSION,
    SCIONPATHML_PINNED_SHA,
    SIM_VERSION,
    TIMESTEP_SCHEMA,
    _agent_row,
    _event_rows,
    _git_sha,
    _global_row,
)

PolicyFactory = Callable[[str], object]
"""Returns a fresh policy callable for one agent. Receives the agent_id."""


def stream_baseline_episode(
    cfg: Phase1Config,
    episode_id: str,
    episode_seed: int,
    policy_factory: PolicyFactory,
) -> Iterator[tuple[list[dict], list[dict], dict]]:
    """Yield (timestep_rows, event_rows, tick_metrics) per observation tick.

    `tick_metrics` is a small dict the harness uses to compute per-episode
    aggregates:
        {
            "agent_throughput": {agent_id: bandwidth_mbps_observed},
            "agent_rtt": {agent_id: rtt_ms_observed_on_selected},
            "link_util": np.ndarray of per-link utilization,
        }
    """
    env = make_env(cfg)
    policies = {a: policy_factory(a) for a in env.possible_agents}
    for p in policies.values():
        if hasattr(p, "reset"):
            p.reset()

    obs, infos = env.reset(seed=episode_seed)
    masks = {a: infos[a]["action_mask"] for a in env.possible_agents}

    def emit_current() -> tuple[list[dict], list[dict], dict]:
        step = env._last_step
        ts = [_agent_row(episode_id, step, o) for o in step.per_agent]
        ts.append(_global_row(episode_id, step))
        ev = _event_rows(episode_id, step)
        agent_tp: dict[str, float] = {}
        agent_rtt: dict[str, float] = {}
        for o in step.per_agent:
            if o.selected_path_fp is None:
                continue
            for ap, m in zip(o.available_paths, o.path_metrics, strict=True):
                if ap == o.selected_path_fp:
                    agent_tp[o.agent_id] = float(m["bandwidth_mbps"])
                    agent_rtt[o.agent_id] = float(m["rtt_ms"])
                    break
        link_util = env.sim.link_util.copy() if env.sim is not None else np.zeros(0)
        return ts, ev, {"agent_throughput": agent_tp, "agent_rtt": agent_rtt, "link_util": link_util}

    yield emit_current()

    while env.agents:
        actions = {a: int(policies[a](obs[a], masks[a])) for a in env.agents}
        obs, _rewards, terms, _truncs, infos = env.step(actions)
        if not env.agents:
            break
        masks = {a: infos[a]["action_mask"] for a in env.agents}
        yield emit_current()


def write_baseline_shard(
    cfg: Phase1Config,
    shard_index: int,
    episode_specs: Iterable[tuple[str, int]],
    policy_factory: PolicyFactory,
    out_dir: Path,
    repo_root: Path,
    write_batch_obs: int = 600,
) -> tuple[dict, dict]:
    """Stream-write one shard's worth of episodes under `policy_factory`.

    Returns (manifest, episode_metrics) where `episode_metrics` is a
    per-episode dict of aggregate metrics (throughput, fairness, oscillation,
    median_rtt).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_path = out_dir / f"shard_{shard_index:04d}.timesteps.parquet"
    ev_path = out_dir / f"shard_{shard_index:04d}.events.parquet"
    manifest_path = out_dir / f"shard_{shard_index:04d}.manifest.json"

    episode_ids: list[str] = []
    per_episode: list[dict] = []
    ts_writer = pq.ParquetWriter(ts_path, TIMESTEP_SCHEMA, compression="zstd")
    ev_writer = pq.ParquetWriter(ev_path, EVENT_SCHEMA, compression="zstd")
    try:
        ts_buf: list[dict] = []
        ev_buf: list[dict] = []
        ticks_in_buf = 0
        for episode_id, episode_seed in episode_specs:
            episode_ids.append(episode_id)
            agent_total_tp: dict[str, float] = {}
            link_util_series: list[np.ndarray] = []
            per_tick_median_rtt: list[float] = []
            for tick_ts, tick_ev, m in stream_baseline_episode(
                cfg, episode_id, episode_seed, policy_factory
            ):
                ts_buf.extend(tick_ts)
                ev_buf.extend(tick_ev)
                ticks_in_buf += 1
                for a, tp in m["agent_throughput"].items():
                    agent_total_tp[a] = agent_total_tp.get(a, 0.0) + tp
                if m["link_util"].size:
                    link_util_series.append(m["link_util"])
                if m["agent_rtt"]:
                    per_tick_median_rtt.append(float(np.median(list(m["agent_rtt"].values()))))
                if ticks_in_buf >= write_batch_obs:
                    ts_writer.write_table(pa.Table.from_pylist(ts_buf, schema=TIMESTEP_SCHEMA))
                    if ev_buf:
                        ev_writer.write_table(pa.Table.from_pylist(ev_buf, schema=EVENT_SCHEMA))
                    ts_buf.clear()
                    ev_buf.clear()
                    ticks_in_buf = 0
            per_episode.append(
                episode_metrics(
                    episode_id=episode_id,
                    agent_total_tp=agent_total_tp,
                    link_util_series=link_util_series,
                    per_tick_median_rtt=per_tick_median_rtt,
                )
            )
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
        "shard_size_episodes": len(episode_ids),
        "episode_ids": episode_ids,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "config_hash": cfg.sim.hash(),
        "phase1_config_inline": cfg.model_dump(mode="json"),
        "dataset_hash": "sha256:" + sha.hexdigest(),
        "sim_version": SIM_VERSION,
        "pami_pinned_sha": PAMI_PINNED_SHA,
        "scionpathml_pinned_sha": SCIONPATHML_PINNED_SHA,
        "created_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest, {"episodes": per_episode}


def episode_metrics(
    *,
    episode_id: str,
    agent_total_tp: dict[str, float],
    link_util_series: list[np.ndarray],
    per_tick_median_rtt: list[float],
) -> dict:
    """Compute per-episode aggregates from in-flight tick observations."""
    tps = np.asarray(list(agent_total_tp.values()), dtype=np.float64)
    if tps.size == 0:
        jain = 1.0
    else:
        denom = float(len(tps) * float((tps ** 2).sum()))
        jain = float((tps.sum() ** 2) / denom) if denom > 0 else 1.0
    if link_util_series:
        link_util_arr = np.stack(link_util_series, axis=0)  # (T, L)
        link_var = float(link_util_arr.var(axis=0).mean()) if link_util_arr.shape[0] > 1 else 0.0
    else:
        link_var = 0.0
    median_rtt = float(np.mean(per_tick_median_rtt)) if per_tick_median_rtt else 0.0
    return {
        "episode_id": episode_id,
        "aggregate_throughput": float(tps.sum()),
        "jain_fairness": jain,
        "link_utilization_variance": link_var,
        "median_agent_rtt_ms": median_rtt,
    }
