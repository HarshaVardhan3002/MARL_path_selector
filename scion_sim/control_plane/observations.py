"""Per-observation measurement noise applied on top of static path properties.

The fluid simulator yields a per-path flow rate `x_r(t)` and per-link load.
The agent observes RTT, bandwidth, loss, jitter — all noisy versions of
the underlying path properties. Reasonable noise shapes:

* RTT: relative log-normal (queuing-induced spikes; heavy right tail)
* Bandwidth observed: clamped at a per-path achievable based on
  current link utilization, then perturbed
* Loss: half-normal centred on a base loss with utilization-driven scaling
* Jitter: lognormal driven by RTT

Public function: `observe_path` returns the dict the schema expects.
"""
from __future__ import annotations

import numpy as np

from scion_sim.config import ObservationNoiseConfig
from scion_sim.control_plane.topology import PathRecord


def observe_path(
    rec: PathRecord,
    flow_rate: float,
    link_utilizations: np.ndarray,
    cfg: ObservationNoiseConfig,
    rng: np.random.Generator,
    t_sec: float,
) -> dict:
    """Return one path_metrics struct (see docs/SCHEMA.md)."""
    rtt_ms = max(1.0, rec.base_rtt_ms * float(rng.lognormal(mean=0.0, sigma=cfg.rtt_relative_std)))
    avg_util = float(np.mean(link_utilizations[list(rec.link_indices)])) if rec.link_indices else 0.0
    capacity_factor = max(0.0, 1.0 - 0.6 * min(1.0, avg_util))  # heavy haircut as utilization → 1
    bw_mbps = max(
        0.1,
        rec.base_bandwidth_mbps * capacity_factor
        * float(rng.normal(loc=1.0, scale=cfg.bandwidth_relative_std)),
    )
    base_loss = cfg.loss_base_pct + 30.0 * max(0.0, avg_util - 0.6)  # saturation knee at ~0.6 util
    loss_pct = float(np.clip(base_loss + abs(rng.normal(0.0, cfg.loss_relative_std)), 0.0, 100.0))
    jitter_ms = max(0.0, rtt_ms * 0.15 * float(rng.lognormal(0.0, cfg.jitter_relative_std)))
    return {
        "fp": rec.fingerprint,
        "rtt_ms": rtt_ms,
        "bandwidth_mbps": bw_mbps,
        "loss_pct": loss_pct,
        "jitter_ms": jitter_ms,
        "age_sec": rec.age_sec(t_sec),
    }
