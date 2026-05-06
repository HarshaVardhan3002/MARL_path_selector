"""Phase 1 reward function.

Three independent terms with a composite that returns a *dict*, not a
scalar — Phase 2 ablation needs each component logged. The env emits the
total as the per-agent reward (PettingZoo contract) and the full dict
into `info[agent]["reward_components"]`.

* `throughput_term`           — bandwidth normalized to mean link capacity.
* `congestion_contribution`   — per-agent share of over-utilization on the
                                selected path's links, with a knee at
                                `congestion_util_threshold` (v0=0.7).
* `diversity_term`            — system-level: unique selected paths
                                divided by number of agents (in [1/N, 1.0]).

Inputs are bundled in `RewardInputs`. The env builds one of these per
agent per step from the simulator state.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from scion_sim.env.config import RewardConfig


@dataclasses.dataclass
class RewardInputs:
    """Everything the three reward terms need at one env step.

    Bundled rather than passed as a long argument list; the env constructs
    one of these per agent.
    """

    agent_id: str
    selected_fp: str | None
    selected_metric: dict | None
    selected_link_indices: tuple[int, ...]
    link_utilizations: np.ndarray
    link_capacities: np.ndarray
    link_agent_demand: dict[int, dict[str, float]]
    agent_assignments: dict[str, str | None]


def throughput_term(inp: RewardInputs, cfg: RewardConfig) -> float:
    """Bandwidth on the selected path, normalized to mean link capacity.

    Returns 0.0 if the agent has no selected path or no metric is observable.
    """
    if inp.selected_metric is None:
        return 0.0
    bw = float(inp.selected_metric.get("bandwidth_mbps", 0.0))
    cap_mean = float(inp.link_capacities.mean()) if inp.link_capacities.size else 1.0
    if cap_mean <= 0.0:
        return 0.0
    return bw / cap_mean


def congestion_contribution_term(inp: RewardInputs, cfg: RewardConfig) -> float:
    """Per-agent share of over-utilization summed across the selected path's links.

    For each link L on the selected path:
        over_L  = max(0, util_L - threshold)
        share_L = agent_demand_on_L / total_demand_on_L  (0 if total = 0)
        penalty_L = over_L * share_L
    Return the sum across L.
    """
    if not inp.selected_link_indices:
        return 0.0
    threshold = cfg.congestion_util_threshold
    util = inp.link_utilizations
    total_penalty = 0.0
    for li in inp.selected_link_indices:
        if li >= util.shape[0]:
            continue
        over = float(util[li]) - threshold
        if over <= 0.0:
            continue
        per_agent = inp.link_agent_demand.get(li, {})
        total = sum(per_agent.values())
        if total <= 0.0:
            continue
        share = per_agent.get(inp.agent_id, 0.0) / total
        total_penalty += over * share
    return float(total_penalty)


def diversity_term(inp: RewardInputs, cfg: RewardConfig) -> float:
    """System-level: unique selected paths / number of agents.

    Equal across agents within one step (CTDE-friendly). Pushes against
    the trivial "everyone on the same path" equilibrium.
    """
    fps = [fp for fp in inp.agent_assignments.values() if fp is not None]
    if not fps:
        return 0.0
    return float(len(set(fps))) / float(len(fps))


def composite_reward(inp: RewardInputs, cfg: RewardConfig) -> dict:
    """Combine the three terms with v0 weights. Returns the full breakdown.

    Caller (env) emits `result["total"]` as the per-agent reward and forwards
    the dict to `info[agent]["reward_components"]`.
    """
    t = throughput_term(inp, cfg)
    c = congestion_contribution_term(inp, cfg)
    d = diversity_term(inp, cfg)
    total = cfg.w_throughput * t - cfg.w_congestion * c + cfg.w_diversity * d
    return {
        "total": float(total),
        "throughput": float(t),
        "congestion_contribution": float(c),
        "diversity": float(d),
        "weights": {
            "throughput": float(cfg.w_throughput),
            "congestion": float(cfg.w_congestion),
            "diversity": float(cfg.w_diversity),
        },
    }
