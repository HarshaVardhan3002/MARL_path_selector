"""Phase 1 environment configuration.

Wraps a `SimulatorConfig` (the underlying Phase 0 physics) without modifying
it — preserving Phase 0's `config_hash` reproducibility (J-013, J-016).
Phase 1-specific parameters (observation window sizes, reward weights,
oracle-EMA window) live here.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from scion_sim.config import SimulatorConfig


class ObservationConfig(BaseModel):
    """Observation-builder parameters mirroring docs/PHASE_1_OBSERVATION_DESIGN.md."""
    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=4, ge=1, le=16, description="Top-K path slots exposed per agent.")
    norm_window_steps: int = Field(
        default=16, ge=2,
        description="K — rolling window for F2/F3/F6 min-max normalization.",
    )
    trend_window_steps: int = Field(
        default=5, ge=2,
        description="Steps used for S4/S5 linear regression slope on the selected path.",
    )
    loss_history_steps: int = Field(
        default=10, ge=2, description="Window for S6 loss-event count.",
    )
    churn_history_steps: int = Field(
        default=10, ge=2, description="Window for S7 path-set-changed indicator.",
    )
    max_age_seconds: float = Field(
        default=86400.0, gt=0.0,
        description="F5 normalization scale — paths older than this saturate at 1.0.",
    )
    loss_observation_scale_pct: float = Field(
        default=100.0, gt=0.0,
        description=(
            "F4 divisor to fit loss into the env's [-1,1] observation Box. "
            "100.0 maps the raw 0-100% range into [0,1]. See J-016."
        ),
    )
    loss_event_threshold_pct: float = Field(
        default=1.0, ge=0.0,
        description="Threshold above which a tick counts toward S6.",
    )
    s2_sentinel: float = Field(
        default=-0.25,
        description="Returned in S2 when current selected path is not in top-K. -1/K with K=4.",
    )
    s3_saturation_steps: int = Field(
        default=100, ge=1, description="Steps-since-switch saturates here (S3=1.0).",
    )
    trend_slope_scale: float = Field(
        default=10.0, gt=0.0,
        description=(
            "Divisor applied to S4/S5 slope before tanh, in (units / step). v0 placeholder "
            "(e.g., 10 ms/step or 10 Mbps/step is a 'large' move). See J-018."
        ),
    )
    paths_per_agent_norm: int = Field(
        default=12, ge=1,
        description="S1 divisor; should match SimulatorConfig.topology.paths_per_agent.",
    )


class RewardConfig(BaseModel):
    """Reward-shaping parameters. v0 placeholders, tuned in Phase 2."""
    model_config = ConfigDict(extra="forbid")

    w_throughput: float = Field(default=1.0, description="Multiplier on throughput term.")
    w_congestion: float = Field(default=2.0, description="Multiplier on congestion penalty (applied negative).")
    w_diversity: float = Field(default=0.5, description="Multiplier on diversity bonus.")
    congestion_util_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Utilization 'knee' below which no congestion penalty is paid.",
    )


class BaselineConfig(BaseModel):
    """Baseline-policy hyperparameters. v0 placeholders."""
    model_config = ConfigDict(extra="forbid")

    oracle_ema_window: int = Field(
        default=20, ge=1,
        description=(
            "Steps for oracle-greedy's per-slot reward EMA. v0=20 (J-021); "
            "if the policy fails to oscillate / underperforms shortest-path, drop to 5."
        ),
    )
    oracle_loss_penalty: float = Field(
        default=1.0, ge=0.0,
        description="Per-loss-percent penalty added to oracle-greedy's slot reward proxy.",
    )


class Phase1Config(BaseModel):
    """Top-level Phase 1 configuration.

    Holds a `SimulatorConfig` for Phase 0 physics (passed unchanged into
    `Phase0Simulator`) plus the observation, reward, and baseline knobs
    that Phase 1 introduces.
    """
    model_config = ConfigDict(extra="forbid")

    sim: SimulatorConfig = Field(default_factory=SimulatorConfig)
    observation: ObservationConfig = Field(default_factory=ObservationConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
