"""Slot ↔ fingerprint translation and action coercion.

The Phase 1 action space is `Discrete(4)` — a top-K slot index. The env
must translate an agent's chosen slot back to the corresponding path
fingerprint at the moment the action is committed.

Top-K ordering (locked at observation design): lowest RTT first, ties
broken by highest bandwidth. Mirrors `observation_builder._top_k_order`
intentionally — both modules read the same path-metrics list and must
agree on slot positions.
"""
from __future__ import annotations

import numpy as np

from scion_sim.env.config import ObservationConfig
from scion_sim.env.observation_builder import _top_k_order


def slot_to_fingerprint(
    slot: int,
    available_paths: list[str],
    path_metrics: list[dict],
    cfg: ObservationConfig,
) -> str | None:
    """Resolve a top-K slot index to a path fingerprint.

    Returns None if the slot is out of range or the agent has no paths.
    The caller is responsible for checking the action mask before
    calling this — passing a masked-out slot returns None deterministically.
    """
    if not available_paths:
        return None
    order = _top_k_order(path_metrics)[: cfg.top_k]
    if slot < 0 or slot >= len(order):
        return None
    return available_paths[order[slot]]


def coerce_action(action: int, mask: np.ndarray) -> int:
    """Snap an action into the valid mask.

    If the policy emits a slot that is masked out (or out of range),
    fall back to the first valid slot. Returns -1 only when the mask
    is fully zero — which the simulator guarantees does not happen
    (at least one path is always visible per agent).
    """
    n = mask.shape[0]
    if 0 <= action < n and bool(mask[action]):
        return int(action)
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        return -1
    return int(valid[0])
