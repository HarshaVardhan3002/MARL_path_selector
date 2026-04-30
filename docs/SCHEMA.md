# Trajectory Schema (Phase 0 Lock)

> Locked 2026-04-30 per JUSTIFICATIONS.md J-013. Phase 1's PettingZoo
> wrapper consumes this schema; changing it after lock invalidates Phase 1
> code. Any post-lock change requires re-running Phase 0 calibration.

## Files on disk

```
data/phase0_trajectories/
  shard_0000.timesteps.parquet
  shard_0000.events.parquet
  shard_0000.manifest.json
  shard_0001.timesteps.parquet
  ...
data/scionpathml_summary.json
```

Each shard contains 50 episodes by default (configurable via
`SimulatorConfig.shard_size_episodes`). The manifest records the git SHA,
configuration hash, dataset hash (SHA-256 of the parquet bytes), simulator
version, and episode IDs included in the shard.

## Conventions

* `episode_id`: ASCII slug, e.g. `ep-000123`.
* `agent_id`: `agent_00` … `agent_23`. The reserved value `__global__`
  marks the per-timestep row carrying centralized critic state (Phase 2
  CTDE).
* `fingerprint`: lower-case 16-character hex string (8 bytes), matching
  the format used in `selected_paths.json` of the public ScionPathML
  release. Fingerprints are stable for a path's lifetime; new path = new
  fingerprint.
* Timestamps:
  * `t` — integer observation index, starting at 0 within each episode.
  * `wall_time_sec` — float seconds since episode start.
  * The fluid simulator integrates at `dt = 0.1 s`; observations are
    emitted every `observation_dt_sec` (default 1.0 s).

## `timesteps` table

| column                  | type                                       | nullable | notes |
|-------------------------|--------------------------------------------|----------|-------|
| `episode_id`            | string                                     | no       | foreign key to manifest |
| `t`                     | int32                                      | no       | observation index |
| `wall_time_sec`         | float64                                    | no       | seconds since episode reset |
| `agent_id`              | string                                     | no       | `agent_NN` or `__global__` |
| `selected_path_fp`      | string                                     | yes      | per-agent rows; null if no path active |
| `available_paths`       | list&lt;string&gt;                         | yes      | per-agent rows; visible-to-agent path fingerprints (post-discrepancy) |
| `path_metrics`          | list&lt;struct{fp, rtt_ms, bandwidth_mbps, loss_pct, jitter_ms, age_sec}&gt; | yes | per-agent rows; one entry per visible path; metrics already include observation noise |
| `reward`                | float64                                    | yes      | Phase 1 will populate; Phase 0 = null |
| `reward_components`     | struct{throughput, congestion, diversity}  | yes      | Phase 1 will populate; Phase 0 = null |
| `link_utilizations`     | list&lt;struct{link_id, utilization}&gt;   | yes      | global rows only |
| `total_throughput`      | float64                                    | yes      | global rows only |
| `agent_assignments`     | list&lt;struct{agent_id, fp}&gt;           | yes      | global rows only |

Per-agent rows have null `link_utilizations`, `total_throughput`,
`agent_assignments`. Global rows have null `selected_path_fp`,
`available_paths`, `path_metrics`, `reward`, `reward_components`.

## `events` table

| column         | type   | notes |
|----------------|--------|-------|
| `episode_id`   | string | |
| `t`            | int32  | observation-index nearest the event |
| `wall_time_sec`| float64 | exact wall time of the event |
| `agent_id`     | string | which agent's view changed (or `__global__` for topology-wide events) |
| `fingerprint`  | string | the path involved |
| `event_type`   | string | one of `path_appeared`, `path_expired`, `path_metrics_updated` |

`path_metrics_updated` events are emitted when control-plane noise
re-samples a path's RTT/BW/loss observation between observation timesteps;
they are advisory and primarily useful for debugging reward hacking later.

## Manifest schema (`*.manifest.json`)

```json
{
  "schema_version": 1,
  "shard_index": 0,
  "shard_size_episodes": 50,
  "episode_ids": ["ep-000000", "ep-000001", ...],
  "git_sha": "abcd1234...",
  "git_dirty": false,
  "config_hash": "sha256:...",
  "config_inline": { /* full SimulatorConfig dict */ },
  "dataset_hash": "sha256:...",
  "sim_version": "scion_sim 0.0.1",
  "pami_pinned_sha": "294c0797...",
  "scionpathml_pinned_sha": "362ad608...",
  "created_utc": "2026-04-30T01:23:45Z"
}
```

## What is *not* in the schema

* The K = 16 rolling-history window. Phase 1 computes it on the fly from
  per-timestep rows; storing it would denormalize 16× per row.
* Underlying ODE state (per-path flow rates `x_r(t)`). Useful for
  debugging but not part of the contract; Phase 1 never reads it.
* Reward function definition. Phase 1 owns the reward; this schema only
  reserves the columns to fill in.

## Versioning

`schema_version` in the manifest starts at 1. Any change increments it.
Phase 1 must check the version on load and refuse mismatches.
