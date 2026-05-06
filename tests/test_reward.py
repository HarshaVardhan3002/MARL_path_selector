"""Phase 1 reward unit tests (PHASE_1_SPEC §2.4).

Each scenario constructs a hand-built `RewardInputs` to anchor the term
math independent of simulator dynamics. Test 4 is the lossless-operating-
point *finding* test (J-018, J-019): it measures the gap between the v0
reward's optimum and the formal Keshvadi 2510.26060 lossless point — the
spec explicitly authorizes this as a finding rather than a passing
assertion of equivalence.
"""
from __future__ import annotations

import math

import numpy as np

from scion_sim.env.config import RewardConfig
from scion_sim.env.reward import (
    RewardInputs,
    composite_reward,
    congestion_contribution_term,
    diversity_term,
    throughput_term,
)


def _inp(
    *,
    agent_id: str = "agent_00",
    selected_fp: str | None = "fp_a",
    bw: float = 95.0,
    selected_link_indices: tuple[int, ...] = (0,),
    link_util: list[float] | None = None,
    link_caps: list[float] | None = None,
    link_demand: dict[int, dict[str, float]] | None = None,
    assignments: dict[str, str | None] | None = None,
) -> RewardInputs:
    return RewardInputs(
        agent_id=agent_id,
        selected_fp=selected_fp,
        selected_metric=None if selected_fp is None else {
            "fp": selected_fp, "rtt_ms": 200.0, "bandwidth_mbps": bw,
            "loss_pct": 0.0, "jitter_ms": 1.0, "age_sec": 10.0,
        },
        selected_link_indices=selected_link_indices,
        link_utilizations=np.asarray(link_util if link_util is not None else [0.3]),
        link_capacities=np.asarray(link_caps if link_caps is not None else [100.0]),
        link_agent_demand=link_demand if link_demand is not None else {0: {agent_id: 10.0}},
        agent_assignments=assignments if assignments is not None else {agent_id: selected_fp},
    )


# Test 1 — single-agent isolated path -----------------------------------------
def test_single_agent_isolated_path() -> None:
    cfg = RewardConfig()
    inp = _inp(
        bw=95.0,
        link_util=[0.3],
        link_caps=[100.0],
        link_demand={0: {"agent_00": 30.0}},
        assignments={"agent_00": "fp_a"},
    )
    assert math.isclose(throughput_term(inp, cfg), 0.95, rel_tol=1e-6)
    assert congestion_contribution_term(inp, cfg) == 0.0
    assert diversity_term(inp, cfg) == 1.0
    comp = composite_reward(inp, cfg)
    assert math.isclose(comp["total"], 1.0 * 0.95 + 0.5 * 1.0, rel_tol=1e-6)


# Test 2 — congested shared path ----------------------------------------------
def test_congested_shared_path_four_agents() -> None:
    cfg = RewardConfig()
    # 4 agents all on the same path which traverses link 0 at util=0.85.
    # Each contributes equal demand → share = 0.25.
    demand = {0: {f"agent_{i:02d}": 25.0 for i in range(4)}}
    assignments = {f"agent_{i:02d}": "fp_shared" for i in range(4)}
    inp = _inp(
        agent_id="agent_00",
        selected_fp="fp_shared",
        bw=95.0,
        selected_link_indices=(0,),
        link_util=[0.85],
        link_caps=[100.0],
        link_demand=demand,
        assignments=assignments,
    )
    expected_congestion = (0.85 - 0.7) * 0.25
    assert math.isclose(congestion_contribution_term(inp, cfg), expected_congestion, rel_tol=1e-6)
    assert diversity_term(inp, cfg) == 0.25
    assert math.isclose(throughput_term(inp, cfg), 0.95, rel_tol=1e-6)
    comp = composite_reward(inp, cfg)
    expected_total = 1.0 * 0.95 - 2.0 * expected_congestion + 0.5 * 0.25
    assert math.isclose(comp["total"], expected_total, rel_tol=1e-6)


# Test 3 — perfect diversity, low congestion ----------------------------------
def test_perfect_diversity_low_congestion() -> None:
    cfg = RewardConfig()
    # 4 agents each on own path; util=0.4 on every link (below knee).
    # Each agent's link is distinct.
    link_util = [0.4, 0.4, 0.4, 0.4]
    link_caps = [100.0, 100.0, 100.0, 100.0]
    demand = {i: {f"agent_{i:02d}": 40.0} for i in range(4)}
    assignments = {f"agent_{i:02d}": f"fp_{i}" for i in range(4)}
    for i in range(4):
        inp = _inp(
            agent_id=f"agent_{i:02d}",
            selected_fp=f"fp_{i}",
            bw=95.0,
            selected_link_indices=(i,),
            link_util=link_util,
            link_caps=link_caps,
            link_demand=demand,
            assignments=assignments,
        )
        assert math.isclose(throughput_term(inp, cfg), 0.95, rel_tol=1e-6)
        assert congestion_contribution_term(inp, cfg) == 0.0
        assert diversity_term(inp, cfg) == 1.0
        comp = composite_reward(inp, cfg)
        assert math.isclose(comp["total"], 0.95 + 0.5, rel_tol=1e-6)


# Test 4 — lossless operating point (FINDING test, J-018) ---------------------
def test_lossless_operating_point_is_a_finding_not_a_max() -> None:
    """Per user-approved Test 4 (Option A-lite): construct a simplified
    lossless point — N agents on distinct paths, every link at util=1.0,
    no losses — and compare its v0 composite to a less-utilized perturbation.

    The Keshvadi 2510.26060 lossless point asserts L̂(0)=C_π (load equals
    capacity). Our v0 reward's congestion knee at 0.7 means full utilization
    pays a non-trivial penalty, so v0 reward does NOT attain its maximum at
    the formal lossless point. This test documents that gap. Phase 2 reward
    tuning addresses it; precise (ρ, σ_z, α̂) → config mapping is deferred
    to supervisor dialogue (J-018).
    """
    cfg = RewardConfig()
    n = 4
    bw_at_lossless = 100.0  # bandwidth tracks capacity at the lossless point
    bw_undersaturated = 95.0  # at util=0.6 we'd see slightly lower BW (synthetic)
    link_caps = [100.0] * n
    assignments = {f"agent_{i:02d}": f"fp_{i}" for i in range(n)}

    # Lossless: every link at util=1.0
    lossless_util = [1.0] * n
    lossless_demand = {i: {f"agent_{i:02d}": 100.0} for i in range(n)}
    lossless_total = 0.0
    for i in range(n):
        inp = _inp(
            agent_id=f"agent_{i:02d}",
            selected_fp=f"fp_{i}",
            bw=bw_at_lossless,
            selected_link_indices=(i,),
            link_util=lossless_util,
            link_caps=link_caps,
            link_demand=lossless_demand,
            assignments=assignments,
        )
        lossless_total += composite_reward(inp, cfg)["total"]

    # Under-saturated perturbation: every link at util=0.6 (below knee)
    under_util = [0.6] * n
    under_demand = {i: {f"agent_{i:02d}": 60.0} for i in range(n)}
    under_total = 0.0
    for i in range(n):
        inp = _inp(
            agent_id=f"agent_{i:02d}",
            selected_fp=f"fp_{i}",
            bw=bw_undersaturated,
            selected_link_indices=(i,),
            link_util=under_util,
            link_caps=link_caps,
            link_demand=under_demand,
            assignments=assignments,
        )
        under_total += composite_reward(inp, cfg)["total"]

    # The finding: v0 reward at the formal lossless point is LOWER than at the
    # under-saturated perturbation, because the v0 congestion term penalizes
    # util > 0.7. Phase 2 reward tuning must address this.
    assert under_total > lossless_total, (
        "v0 finding invalidated: composite reward is *higher* at the formal "
        "lossless point than at the under-saturated perturbation. Re-examine the "
        "v0 reward shape — the gap to the formal target is no longer present."
    )


# Test 5 — components sum correctly -------------------------------------------
def test_components_sum_to_weighted_total() -> None:
    cfg = RewardConfig(w_throughput=1.5, w_congestion=3.0, w_diversity=0.7)
    inp = _inp(
        bw=80.0,
        link_util=[0.9],
        link_caps=[100.0],
        link_demand={0: {"agent_00": 50.0, "agent_01": 50.0}},
        assignments={"agent_00": "fp_a", "agent_01": "fp_a"},
    )
    comp = composite_reward(inp, cfg)
    expected = (
        1.5 * comp["throughput"]
        - 3.0 * comp["congestion_contribution"]
        + 0.7 * comp["diversity"]
    )
    assert math.isclose(comp["total"], expected, rel_tol=1e-9)
    assert comp["weights"] == {"throughput": 1.5, "congestion": 3.0, "diversity": 0.7}


# Test 6 — congestion threshold inactive below 0.7 ----------------------------
def test_congestion_zero_below_threshold() -> None:
    cfg = RewardConfig()
    inp = _inp(
        bw=80.0,
        link_util=[0.65],  # exactly below threshold
        link_caps=[100.0],
        link_demand={0: {"agent_00": 100.0}},  # 100% share
        assignments={"agent_00": "fp_a"},
    )
    assert congestion_contribution_term(inp, cfg) == 0.0


# Test 7 — masked-out top-K slots don't break reward --------------------------
def test_reward_does_not_crash_when_top_k_underfilled() -> None:
    """Agent's selected path is the only one visible — slots 1, 2, 3 masked.
    Reward must compute cleanly without reading the empty slots.
    """
    cfg = RewardConfig()
    inp = _inp(
        bw=70.0,
        link_util=[0.5],
        link_caps=[100.0],
        link_demand={0: {"agent_00": 50.0}},
        assignments={"agent_00": "fp_only"},
    )
    comp = composite_reward(inp, cfg)
    assert math.isfinite(comp["total"])
    assert math.isclose(comp["throughput"], 0.7, rel_tol=1e-6)
    assert comp["congestion_contribution"] == 0.0
    assert comp["diversity"] == 1.0
