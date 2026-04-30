"""Pydantic configuration models for the Phase 0 simulator.

The configuration captures the calibratable parameters explicitly. Default
values are intentionally chosen to match the broad shape of the empirical
ScionPathML summary (`data/scionpathml_summary.json`) — they are not yet
the calibrated values; Task #9 tunes them.

YAML round-trip:

    cfg = SimulatorConfig.model_validate(yaml.safe_load(open("foo.yaml")))
    yaml.safe_dump(cfg.model_dump(mode="json"), open("foo.yaml", "w"))
"""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TopologyConfig(BaseModel):
    """Underlying network topology fed to the PAMI fluid simulator."""
    model_config = ConfigDict(extra="forbid")

    num_agents: int = Field(default=24, ge=2, le=200, description="Number of MARL agents (PAMI sources).")
    paths_per_agent: int = Field(
        default=12, ge=2, le=64,
        description="Candidate paths per agent (only a subset visible at any time per discrepancy).",
    )
    num_links: int = Field(default=64, ge=4, le=1024, description="Distinct link resources shared across paths.")
    avg_hops_per_path: float = Field(
        default=8.7, ge=1.0, le=20.0,
        description="Empirical avg from Lab1 comparer_analysis.txt (path length of changes).",
    )
    link_capacity_mbps_mean: float = Field(default=100.0, gt=0.0)
    link_capacity_mbps_std: float = Field(default=20.0, ge=0.0)
    base_rtt_ms_mean: float = Field(
        default=400.0, gt=0.0,
        description="Empirical median across pairs is 328-446 ms; we centre near 400 ms.",
    )
    base_rtt_ms_std: float = Field(default=80.0, ge=0.0)


class FluidConfig(BaseModel):
    """Inner-loop fluid model integration parameters (passed through to PAMI)."""
    model_config = ConfigDict(extra="forbid")

    dt_sec: float = Field(default=0.1, gt=0.0, le=1.0)
    a_bar: float = Field(default=0.05, gt=0.0, description="PAMI primal Eq. 1 a̅; AIMD-style increase rate.")
    b_r: float = Field(default=0.875, gt=0.0, le=1.0, description="PAMI Eq. 1 b_r; multiplicative-decrease coefficient.")
    beta_j: float = Field(default=10.0, gt=0.0, description="PAMI Eq. 4 β_j; congestion-curve sharpness.")
    init_flow_rate: float = Field(default=1.0, gt=0.0, description="Initial x_r per active path.")


class LifetimeConfig(BaseModel):
    """Path-lifetime distribution. Empirical: heavy-right-tailed; lognormal is the standard fit."""
    model_config = ConfigDict(extra="forbid")

    distribution: Literal["lognormal", "exponential", "weibull"] = "lognormal"
    median_seconds: float = Field(
        default=27000.0, gt=0.0,
        description="Empirical pooled median (matches scionpathml_summary).",
    )
    sigma: float = Field(
        default=1.6, gt=0.0,
        description="Lognormal log-space std; controls tail heaviness.",
    )
    weibull_shape: float = Field(default=0.7, gt=0.0, description="Used only when distribution=='weibull'.")


class ChurnConfig(BaseModel):
    """Path-appearance arrival process."""
    model_config = ConfigDict(extra="forbid")

    appear_rate_per_agent_per_hour: float = Field(
        default=0.5, ge=0.0,
        description="Poisson rate at which new paths appear in an agent's candidate pool.",
    )
    initial_visible_fraction: float = Field(default=0.5, gt=0.0, le=1.0)


class DiscrepancyConfig(BaseModel):
    """Asymmetric path-set views. See JUSTIFICATIONS J-002, J-010."""
    model_config = ConfigDict(extra="forbid")

    asymmetry_rate: float = Field(
        default=0.30, ge=0.0, le=1.0,
        description="Probability a path appears in only one direction's view at birth (proxy).",
    )


class ObservationNoiseConfig(BaseModel):
    """Per-observation measurement noise applied on top of static path properties."""
    model_config = ConfigDict(extra="forbid")

    rtt_relative_std: float = Field(default=0.20, ge=0.0)
    bandwidth_relative_std: float = Field(default=0.05, ge=0.0)
    loss_base_pct: float = Field(default=0.5, ge=0.0)
    loss_relative_std: float = Field(default=2.0, ge=0.0)
    jitter_relative_std: float = Field(default=0.40, ge=0.0)


class TrajectoryConfig(BaseModel):
    """How trajectories are sampled and stored."""
    model_config = ConfigDict(extra="forbid")

    observation_dt_sec: float = Field(default=1.0, gt=0.0)
    episode_seconds: float = Field(default=3600.0, gt=0.0)
    num_episodes: int = Field(default=1000, ge=1)
    shard_size_episodes: int = Field(default=50, ge=1)
    output_dir: str = Field(default="data/phase0_trajectories")


class SimulatorConfig(BaseModel):
    """Top-level Phase 0 simulator configuration."""
    model_config = ConfigDict(extra="forbid")

    seed: int = Field(default=2026)
    topology: TopologyConfig = Field(default_factory=TopologyConfig)
    fluid: FluidConfig = Field(default_factory=FluidConfig)
    lifetime: LifetimeConfig = Field(default_factory=LifetimeConfig)
    churn: ChurnConfig = Field(default_factory=ChurnConfig)
    discrepancy: DiscrepancyConfig = Field(default_factory=DiscrepancyConfig)
    observation_noise: ObservationNoiseConfig = Field(default_factory=ObservationNoiseConfig)
    trajectory: TrajectoryConfig = Field(default_factory=TrajectoryConfig)

    @field_validator("seed")
    @classmethod
    def seed_in_range(cls, v: int) -> int:
        if v < 0 or v >= 2**31:
            raise ValueError("seed must be in [0, 2**31).")
        return v

    def hash(self) -> str:
        """Deterministic SHA-256 over the canonical JSON dump (used in artifact manifests)."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
