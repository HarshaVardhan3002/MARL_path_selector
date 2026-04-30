"""Multi-source PAN primal ODE adapter.

The vendored `pan_primal_equations` from PAMI is single-source. We retain
its scheduler/active-paths semantics but compute per-source `y_s` so that
24 agents share link resources realistically. This is our code, not a
modification of vendored upstream.

State convention: a 1-D numpy array indexed by *active path slot*. The set
of active paths can change between calls (e.g., when a path expires);
the caller is responsible for resizing the state vector accordingly.
"""
from __future__ import annotations

import numpy as np

from scion_sim.config import FluidConfig
from scion_sim.control_plane.topology import PathRecord


def step_flow_rates(
    state: np.ndarray,
    active_records: list[PathRecord],
    link_capacities: np.ndarray,
    cfg: FluidConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """One Euler step. Returns (next_state, link_utilizations).

    Args:
        state: shape (len(active_records),). Per-active-path flow rates x_r.
        active_records: paths currently flowing (one per state entry).
        link_capacities: shape (num_links,). Resource capacities C_j.
        cfg: PAMI fluid hyperparameters.
    """
    if state.size == 0:
        return state, np.zeros_like(link_capacities)

    a_bar, b_r, beta_j = cfg.a_bar, cfg.b_r, cfg.beta_j
    num_links = link_capacities.size

    # 1. Per-link load z_j (from active paths only).
    z = np.zeros(num_links)
    for i, p in enumerate(active_records):
        x = state[i]
        if x > 0.0:
            for li in p.link_indices:
                z[li] += x

    # Utilization (used by observation noise). Capacity guarded against zero.
    cap_safe = np.maximum(link_capacities, 1e-9)
    util = np.clip(z / cap_safe, 0.0, np.inf)

    # 2. Penalty per link (PAMI Eq. 4): penalty_j = (z/C)^beta.
    penalty = np.power(util, beta_j)
    penalty = np.clip(penalty, 0.0, 1.0)

    # 3. Per-source y_s (sum of that source's active flow).
    by_source: dict[str, float] = {}
    for i, p in enumerate(active_records):
        by_source[p.agent_id] = by_source.get(p.agent_id, 0.0) + float(state[i])

    # 4. Derivatives.
    derivatives = np.empty_like(state)
    for i, p in enumerate(active_records):
        x_r = state[i]
        prod_term = 1.0
        for li in p.link_indices:
            prod_term *= 1.0 - penalty[li]
        lambda_r = 1.0 - prod_term
        y_s = by_source[p.agent_id]
        # PAMI Eq. 1 inner: a̅(1-λ) - b·y·λ
        inner = a_bar * (1.0 - lambda_r) - b_r * y_s * lambda_r
        rtt_sec = max(1e-3, p.base_rtt_ms / 1000.0)
        dx = (x_r / rtt_sec) * inner
        if x_r <= 0.0 and dx < 0.0:
            dx = 0.0
        derivatives[i] = dx

    next_state = np.maximum(0.0, state + derivatives * cfg.dt_sec)
    return next_state, util
