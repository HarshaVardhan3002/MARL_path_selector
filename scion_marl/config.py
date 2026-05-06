"""Phase 2 configuration — wraps Phase1Config with MAPPO hyperparameters.

Phase 2 reward-weight sweep varies `phase1.reward.{w_throughput, w_congestion,
w_diversity}` across 9 configs while holding mappo hyperparameters constant.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from scion_sim.config import SimulatorConfig, TopologyConfig, TrajectoryConfig
from scion_sim.env.config import (
    BaselineConfig,
    ObservationConfig,
    Phase1Config,
    RewardConfig,
)


class MappoConfig(BaseModel):
    """MAPPO training hyperparameters. Locked across the 9-config sweep."""
    model_config = ConfigDict(extra="forbid")

    episodes_per_run: int = Field(default=1500, ge=1)
    episode_length: int = Field(default=3600, ge=1, description="Observation ticks per episode (matches sim.trajectory).")
    rollout_episodes: int = Field(default=4, ge=1, description="Episodes collected per PPO update.")
    ppo_epochs: int = Field(default=4, ge=1, description="Passes through the rollout buffer per update.")
    minibatch_size: int = Field(default=256, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    gamma: float = Field(default=0.99, gt=0.0, le=1.0)
    gae_lambda: float = Field(default=0.95, gt=0.0, le=1.0)
    clip_ratio: float = Field(default=0.2, gt=0.0, le=1.0)
    entropy_coef: float = Field(default=0.01, ge=0.0)
    value_loss_coef: float = Field(default=0.5, ge=0.0)
    max_grad_norm: float = Field(default=0.5, gt=0.0)
    hidden_sizes: list[int] = Field(default_factory=lambda: [256, 256])
    activation: Literal["tanh", "relu"] = "tanh"
    eval_episodes: int = Field(default=100, ge=1, description="Eval-time fresh rollouts per trained policy.")
    auto_restart_entropy_coef: float = Field(
        default=0.05, ge=0.0,
        description="Entropy bumped to this on auto-restart (one-shot, before episode 200).",
    )
    auto_restart_threshold_episode: int = Field(
        default=200, ge=1,
        description="If policy entropy crashes to ~0 before this episode, auto-restart once.",
    )
    auto_restart_entropy_floor: float = Field(
        default=0.05, ge=0.0,
        description="Policy-entropy threshold below which we say 'collapsed'.",
    )
    log_interval_episodes: int = Field(default=10, ge=1)
    checkpoint_every_episodes: int = Field(default=200, ge=1)


class Phase2Config(BaseModel):
    """Top-level Phase 2 config. One YAML per sweep cell."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Sweep cell name, e.g. 'sweep_01_balanced'.")
    seed: int = Field(ge=0)
    phase1: Phase1Config = Field(default_factory=Phase1Config)
    mappo: MappoConfig = Field(default_factory=MappoConfig)
    output_dir: str = Field(
        default="runs/phase2",
        description="Per-config directory; per-seed subdir created automatically.",
    )

    @classmethod
    def stress_default(
        cls,
        *,
        name: str,
        seed: int,
        w_throughput: float,
        w_congestion: float,
        w_diversity: float,
    ) -> "Phase2Config":
        """Construct a sweep cell with locked stress-regime sim params (J-023)."""
        phase1 = Phase1Config(
            sim=SimulatorConfig(
                topology=TopologyConfig(
                    num_agents=50, paths_per_agent=12, num_links=64,
                    link_capacity_mbps_mean=35.0,
                ),
                trajectory=TrajectoryConfig(
                    num_episodes=1,  # ignored in env-driven mode
                    episode_seconds=3600.0,
                    shard_size_episodes=1,
                    output_dir=f"runs/phase2/{name}/seed_{seed}",
                ),
            ),
            observation=ObservationConfig(),
            reward=RewardConfig(
                w_throughput=w_throughput,
                w_congestion=w_congestion,
                w_diversity=w_diversity,
            ),
            baseline=BaselineConfig(),
        )
        return cls(name=name, seed=seed, phase1=phase1)
