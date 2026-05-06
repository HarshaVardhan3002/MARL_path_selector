"""Unit tests for the Phase 1 observation builder.

These tests construct path-metrics dicts directly (not via the simulator)
to anchor the builder's contract independent of physics dynamics. The
contract is `docs/PHASE_1_OBSERVATION_DESIGN.md` (file at repo root).
"""
from __future__ import annotations

import numpy as np

from scion_sim.env.config import ObservationConfig
from scion_sim.env.observation_builder import (
    PER_PATH_FEATURES,
    SCALAR_FEATURES,
    ObservationBuilder,
)


def _metric(fp: str, rtt: float, bw: float, loss: float = 0.0, jitter: float = 1.0, age: float = 10.0) -> dict:
    return {
        "fp": fp,
        "rtt_ms": rtt,
        "bandwidth_mbps": bw,
        "loss_pct": loss,
        "jitter_ms": jitter,
        "age_sec": age,
    }


def _builder(top_k: int = 4, paths_per_agent_norm: int = 12) -> ObservationBuilder:
    cfg = ObservationConfig(top_k=top_k, paths_per_agent_norm=paths_per_agent_norm)
    return ObservationBuilder(agent_ids=["a"], cfg=cfg)


def test_obs_shape_dtype_and_range_with_full_paths() -> None:
    """4 paths visible: shape (31,), float32, mask all True, F1 bits at 0/6/12/18 == 1."""
    b = _builder()
    paths = ["p0", "p1", "p2", "p3"]
    metrics = [_metric(fp, rtt=200 + 50 * i, bw=80 - 10 * i) for i, fp in enumerate(paths)]
    obs, mask = b.build("a", paths, metrics, selected_path_fp="p0")
    assert obs.shape == (4 * PER_PATH_FEATURES + SCALAR_FEATURES,) == (31,)
    assert obs.dtype == np.float32
    assert mask.tolist() == [True, True, True, True]
    for slot in range(4):
        assert obs[slot * PER_PATH_FEATURES] == 1.0  # F1 mask bit
    assert float(obs.min()) >= -1.0 and float(obs.max()) <= 1.0


def test_obs_zero_fills_when_fewer_than_4_paths() -> None:
    """Fewer than top-K visible paths zero-fill higher slots; mask reflects validity."""
    b = _builder()
    paths = ["p0", "p1"]
    metrics = [_metric("p0", 200, 90), _metric("p1", 250, 85)]
    obs, mask = b.build("a", paths, metrics, selected_path_fp="p0")
    assert mask.tolist() == [True, True, False, False]
    # Slots 2 and 3 are entirely zero.
    for empty_slot in (2, 3):
        block = obs[empty_slot * PER_PATH_FEATURES:(empty_slot + 1) * PER_PATH_FEATURES]
        assert np.all(block == 0.0)


def test_top_k_ordering_lowest_rtt_first() -> None:
    """5 visible paths, top-K=4: slots 0..3 must be ascending RTT."""
    b = _builder()
    rtts = [400, 100, 300, 200, 500]  # path indices 0..4
    paths = [f"p{i}" for i in range(5)]
    metrics = [_metric(p, rtt=r, bw=50.0) for p, r in zip(paths, rtts, strict=True)]
    obs, mask = b.build("a", paths, metrics, selected_path_fp="p1")
    # We can recover slot RTT ordering by reading F2 (norm RTT), which is monotonic in raw RTT.
    f2_vals = [obs[s * PER_PATH_FEATURES + 1] for s in range(4)]
    # All four mask bits set (we have ≥4 paths)
    assert mask.tolist() == [True, True, True, True]
    # Ascending RTT order means ascending F2 (after normalization is monotonic).
    assert f2_vals == sorted(f2_vals)
    # Slot 0 must be the path with the lowest RTT (p1, rtt=100). Confirmed via S2: selected=p1
    # ranks first → S2 = 0/4 = 0.0.
    assert obs[4 * PER_PATH_FEATURES + 1] == 0.0


def test_top_k_tiebreak_by_highest_bandwidth() -> None:
    """Same RTT → higher BW gets the lower slot."""
    b = _builder()
    paths = ["p_low_bw", "p_high_bw"]
    metrics = [_metric("p_low_bw", rtt=100, bw=50.0), _metric("p_high_bw", rtt=100, bw=90.0)]
    obs, _ = b.build("a", paths, metrics, selected_path_fp="p_high_bw")
    # If tiebreak works, p_high_bw is slot 0 → S2 = 0.0.
    s2 = obs[4 * PER_PATH_FEATURES + 1]
    assert s2 == 0.0


def test_first_tick_normalization_fallback_when_window_too_small() -> None:
    """One path, one tick → window has 1 distinct value per feature → fall back to 0.5."""
    b = _builder()
    metrics = [_metric("p0", rtt=300, bw=70, jitter=5.0)]
    obs, _ = b.build("a", ["p0"], metrics, selected_path_fp="p0")
    # F2 (RTT), F3 (BW), F6 (jitter) all fall back to 0.5.
    assert obs[1] == 0.5
    assert obs[2] == 0.5
    assert obs[5] == 0.5


def test_s2_sentinel_when_selected_not_in_top_k() -> None:
    """6 paths, selected has the highest RTT → not in top-K=4 → S2 == -0.25."""
    b = _builder()
    paths = [f"p{i}" for i in range(6)]
    metrics = [_metric(p, rtt=100 + 50 * i, bw=80) for i, p in enumerate(paths)]
    # The selected is the worst (highest RTT) → not in top-K=4
    obs, _ = b.build("a", paths, metrics, selected_path_fp="p5")
    assert obs[4 * PER_PATH_FEATURES + 1] == -0.25


def test_s3_increments_when_no_switch() -> None:
    """Selected stable across ticks → S3 monotonically increases (saturating at 1.0)."""
    b = _builder()
    paths = ["p0", "p1"]
    metrics = [_metric("p0", 200, 90), _metric("p1", 250, 85)]
    s3_vals = []
    for _ in range(5):
        obs, _ = b.build("a", paths, metrics, selected_path_fp="p0")
        s3_vals.append(obs[4 * PER_PATH_FEATURES + 2])
    assert s3_vals[0] == 0.0  # first tick after fp set
    assert all(s3_vals[i] < s3_vals[i + 1] for i in range(len(s3_vals) - 1))


def test_s3_resets_on_switch() -> None:
    b = _builder()
    paths = ["p0", "p1"]
    metrics = [_metric("p0", 200, 90), _metric("p1", 250, 85)]
    for _ in range(3):
        b.build("a", paths, metrics, selected_path_fp="p0")
    obs, _ = b.build("a", paths, metrics, selected_path_fp="p1")  # switch
    assert obs[4 * PER_PATH_FEATURES + 2] == 0.0


def test_s7_churn_fires_when_path_set_changes() -> None:
    """Adding a path triggers churn; quiescing for the full window clears it."""
    cfg = ObservationConfig(top_k=4, churn_history_steps=3)
    b = ObservationBuilder(agent_ids=["a"], cfg=cfg)
    # Tick 1: initial path set. last_path_set was empty → counts as a change.
    obs, _ = b.build("a", ["p0"], [_metric("p0", 200, 90)], selected_path_fp="p0")
    assert obs[4 * PER_PATH_FEATURES + 6] == 1.0
    # Tick 2: stable (still {p0}).
    b.build("a", ["p0"], [_metric("p0", 200, 90)], selected_path_fp="p0")
    # Tick 3: stable.
    b.build("a", ["p0"], [_metric("p0", 200, 90)], selected_path_fp="p0")
    # Tick 4: stable again — initial change has now scrolled out of the 3-step window.
    obs, _ = b.build("a", ["p0"], [_metric("p0", 200, 90)], selected_path_fp="p0")
    assert obs[4 * PER_PATH_FEATURES + 6] == 0.0


def test_obs_values_within_box_range() -> None:
    """Whole vector must fit in the env's [-1, 1] observation Box."""
    b = _builder()
    # Pathological metrics: huge loss, huge age, huge RTT/BW spread.
    paths = ["p0", "p1", "p2", "p3"]
    metrics = [
        _metric("p0", 50, 1000, loss=99.9, jitter=200, age=1e9),
        _metric("p1", 5000, 1, loss=80.0, jitter=0.1, age=0.0),
        _metric("p2", 100, 50, loss=2.0, jitter=10, age=86400),
        _metric("p3", 800, 30, loss=10.0, jitter=20, age=3600),
    ]
    obs, _ = b.build("a", paths, metrics, selected_path_fp="p2")
    assert float(obs.min()) >= -1.0 and float(obs.max()) <= 1.0
