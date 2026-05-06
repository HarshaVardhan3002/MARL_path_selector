"""Per-agent observation builder.

Implements `docs/PHASE_1_OBSERVATION_DESIGN.md` (file lives at repo root).
Maps the raw `Phase0Simulator` per-agent observation into the 31-dim
float32 observation vector and a 4-bit action mask.

Layout (slot-major, feature-minor):

    indices 0-23  : 4 slots × [F1 mask, F2 RTT, F3 BW, F4 loss, F5 age, F6 jitter]
    index    24   : S1 num-paths / paths_per_agent
    index    25   : S2 selected-slot/K  (-0.25 sentinel if not in top-K)
    index    26   : S3 steps-since-switch / 100, saturating
    index    27   : S4 tanh(RTT slope on selected, last 5 steps / scale)
    index    28   : S5 tanh(BW slope on selected, last 5 steps / scale)
    index    29   : S6 (count of last 10 ticks where selected loss > threshold) / 10
    index    30   : S7 churn indicator over last 10 ticks (binary)

Top-K=4 ordering: lowest RTT first, ties broken by highest bandwidth (J-017).

State that persists across ticks (per agent) is in `AgentBuffers`; the
caller (env) owns these and resets at episode start.
"""
from __future__ import annotations

import collections
import dataclasses

import numpy as np

from scion_sim.env.config import ObservationConfig

PER_PATH_FEATURES = 6
SCALAR_FEATURES = 7


@dataclasses.dataclass
class AgentBuffers:
    """Per-agent rolling state. Reset at episode start."""

    rtt_window: collections.deque  # deque of np.ndarray (per-tick RTTs across visible paths)
    bw_window: collections.deque
    jitter_window: collections.deque
    selected_rtt_trail: collections.deque  # of float, last `trend_window_steps` selected-path RTTs
    selected_bw_trail: collections.deque
    selected_loss_trail: collections.deque  # of bool: selected loss > threshold
    churn_trail: collections.deque  # of bool: path-set changed this tick
    steps_since_switch: int
    last_selected_fp: str | None
    last_path_set: frozenset

    @classmethod
    def empty(cls, cfg: ObservationConfig) -> "AgentBuffers":
        return cls(
            rtt_window=collections.deque(maxlen=cfg.norm_window_steps),
            bw_window=collections.deque(maxlen=cfg.norm_window_steps),
            jitter_window=collections.deque(maxlen=cfg.norm_window_steps),
            selected_rtt_trail=collections.deque(maxlen=cfg.trend_window_steps),
            selected_bw_trail=collections.deque(maxlen=cfg.trend_window_steps),
            selected_loss_trail=collections.deque(maxlen=cfg.loss_history_steps),
            churn_trail=collections.deque(maxlen=cfg.churn_history_steps),
            steps_since_switch=0,
            last_selected_fp=None,
            last_path_set=frozenset(),
        )


def _top_k_order(metrics: list[dict]) -> list[int]:
    """Indices into `metrics` sorted by lowest RTT first, ties broken by highest BW.

    `metrics` is the simulator's per-path list in `available_paths` order.
    """
    keyed = [(m["rtt_ms"], -m["bandwidth_mbps"], i) for i, m in enumerate(metrics)]
    keyed.sort()
    return [i for _, _, i in keyed]


def _normalize_min_max(value: float, window: collections.deque, fallback: float = 0.5) -> float:
    """Min-max scale `value` to [0, 1] using values pooled across the window.

    Per the design's edge case: if the pool has fewer than 2 distinct values
    for any feature, fall back to 0.5 (mid-range).
    """
    if not window:
        return fallback
    pool = np.concatenate([np.asarray(v, dtype=np.float64) for v in window])
    if pool.size == 0:
        return fallback
    lo, hi = float(pool.min()), float(pool.max())
    if hi - lo < 1e-9:
        return fallback
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def _slope(values: collections.deque) -> float:
    """OLS slope of `values` against integer step indices. Returns 0.0 if <2 points."""
    n = len(values)
    if n < 2:
        return 0.0
    y = np.asarray(values, dtype=np.float64)
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = float(((x - x_mean) ** 2).sum())
    if denom < 1e-12:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


class ObservationBuilder:
    """Pure (state-explicit) builder of per-agent observation vectors.

    Holds one `AgentBuffers` per agent. The env constructs one builder
    per episode and calls `build()` per agent per observation tick.
    """

    def __init__(self, agent_ids: list[str], cfg: ObservationConfig) -> None:
        self.agent_ids = list(agent_ids)
        self.cfg = cfg
        self.buffers: dict[str, AgentBuffers] = {a: AgentBuffers.empty(cfg) for a in agent_ids}

    def reset(self) -> None:
        for a in self.agent_ids:
            self.buffers[a] = AgentBuffers.empty(self.cfg)

    def build(
        self,
        agent_id: str,
        available_paths: list[str],
        path_metrics: list[dict],
        selected_path_fp: str | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update the agent's buffers and emit (obs[31] float32, mask[4] bool).

        `path_metrics` mirrors the simulator's per-path dict list:
        [{"fp", "rtt_ms", "bandwidth_mbps", "loss_pct", "jitter_ms", "age_sec"}, ...]
        in the same order as `available_paths`.
        """
        if agent_id not in self.buffers:
            raise KeyError(f"unknown agent_id: {agent_id}")
        cfg = self.cfg
        buf = self.buffers[agent_id]

        # 1. Update rolling normalization windows (current tick included).
        rtts = np.asarray([m["rtt_ms"] for m in path_metrics], dtype=np.float64)
        bws = np.asarray([m["bandwidth_mbps"] for m in path_metrics], dtype=np.float64)
        jts = np.asarray([m["jitter_ms"] for m in path_metrics], dtype=np.float64)
        if rtts.size:
            buf.rtt_window.append(rtts)
            buf.bw_window.append(bws)
            buf.jitter_window.append(jts)

        # 2. Top-K ordering and slot assembly.
        order = _top_k_order(path_metrics)[: cfg.top_k]
        K = cfg.top_k
        obs = np.zeros(K * PER_PATH_FEATURES + SCALAR_FEATURES, dtype=np.float32)
        mask = np.zeros(K, dtype=bool)
        selected_slot: int | None = None
        for slot, idx in enumerate(order):
            m = path_metrics[idx]
            base = slot * PER_PATH_FEATURES
            obs[base + 0] = 1.0  # F1 mask bit
            obs[base + 1] = _normalize_min_max(float(m["rtt_ms"]), buf.rtt_window)
            obs[base + 2] = _normalize_min_max(float(m["bandwidth_mbps"]), buf.bw_window)
            obs[base + 3] = float(np.clip(m["loss_pct"] / cfg.loss_observation_scale_pct, 0.0, 1.0))
            obs[base + 4] = float(np.clip(m["age_sec"] / cfg.max_age_seconds, 0.0, 1.0))
            obs[base + 5] = _normalize_min_max(float(m["jitter_ms"]), buf.jitter_window)
            mask[slot] = True
            if available_paths[idx] == selected_path_fp:
                selected_slot = slot

        # 3. Track switch counter and selected-path trails.
        if selected_path_fp != buf.last_selected_fp:
            buf.steps_since_switch = 0
        else:
            buf.steps_since_switch += 1
        buf.last_selected_fp = selected_path_fp

        # Selected-path metric (if visible at all in available_paths, even outside top-K).
        sel_metric: dict | None = None
        if selected_path_fp is not None:
            for ap, m in zip(available_paths, path_metrics, strict=True):
                if ap == selected_path_fp:
                    sel_metric = m
                    break
        if sel_metric is not None:
            buf.selected_rtt_trail.append(float(sel_metric["rtt_ms"]))
            buf.selected_bw_trail.append(float(sel_metric["bandwidth_mbps"]))
            buf.selected_loss_trail.append(bool(sel_metric["loss_pct"] > cfg.loss_event_threshold_pct))

        # Churn = path set membership changed since previous tick.
        current_set = frozenset(available_paths)
        path_set_changed = current_set != buf.last_path_set
        buf.churn_trail.append(bool(path_set_changed))
        buf.last_path_set = current_set

        # 4. Scalar block.
        scalars_base = K * PER_PATH_FEATURES
        # S1: num_paths / paths_per_agent_norm
        obs[scalars_base + 0] = float(np.clip(len(available_paths) / cfg.paths_per_agent_norm, 0.0, 1.0))
        # S2: selected slot / K, sentinel if not in top-K
        if selected_slot is None:
            obs[scalars_base + 1] = cfg.s2_sentinel
        else:
            obs[scalars_base + 1] = float(selected_slot) / float(cfg.top_k)
        # S3: steps_since_switch / saturation_steps
        obs[scalars_base + 2] = float(min(buf.steps_since_switch / cfg.s3_saturation_steps, 1.0))
        # S4, S5: tanh(slope / scale)
        obs[scalars_base + 3] = float(np.tanh(_slope(buf.selected_rtt_trail) / cfg.trend_slope_scale))
        obs[scalars_base + 4] = float(np.tanh(_slope(buf.selected_bw_trail) / cfg.trend_slope_scale))
        # S6: loss-event fraction over the loss-history window
        if buf.selected_loss_trail:
            obs[scalars_base + 5] = float(sum(buf.selected_loss_trail)) / float(cfg.loss_history_steps)
        else:
            obs[scalars_base + 5] = 0.0
        # S7: any churn in the recent window
        obs[scalars_base + 6] = 1.0 if any(buf.churn_trail) else 0.0

        return obs, mask
