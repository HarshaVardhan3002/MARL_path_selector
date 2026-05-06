"""PettingZoo Parallel API environment wrapping `Phase0Simulator`.

Each agent emits a 31-dim float32 observation in [-1, 1] and chooses a
top-K slot via `Discrete(4)`. The env translates slot → fingerprint via
`action_handler`, injects pending picks into the simulator, and advances
one observation tick. Rewards are computed per-agent from the new global
state; the per-component breakdown is logged to `info[agent]["reward_components"]`.

The env neither bypasses nor duplicates simulator logic — it drives the
existing `Phase0Simulator.run_episode()` generator step-by-step.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from scion_sim.env.action_handler import coerce_action, slot_to_fingerprint
from scion_sim.env.config import Phase1Config
from scion_sim.env.observation_builder import (
    PER_PATH_FEATURES,
    SCALAR_FEATURES,
    ObservationBuilder,
)
from scion_sim.env.reward import RewardInputs, composite_reward
from scion_sim.wrapper.simulator import Phase0Simulator, StepResult


def _link_agent_demand(sim: Phase0Simulator) -> dict[int, dict[str, float]]:
    """Build the (link_id -> {agent_id: demand_mbps}) map from sim state.

    Demand on link L from agent A = the flow rate of A's currently-selected
    path on L (single-path-per-agent is the Phase 0/1 convention).
    """
    out: dict[int, dict[str, float]] = {}
    for (agent_id, fp), rate in sim.flow_rates.items():
        if rate <= 0.0:
            continue
        rec = sim.control.live_paths.get(agent_id, {}).get(fp)
        if rec is None:
            continue
        for li in rec.link_indices:
            out.setdefault(li, {})
            out[li][agent_id] = out[li].get(agent_id, 0.0) + float(rate)
    return out


def _selected_link_indices(sim: Phase0Simulator, agent_id: str) -> tuple[int, ...]:
    fp = sim.selected.get(agent_id)
    if fp is None:
        return ()
    rec = sim.control.live_paths.get(agent_id, {}).get(fp)
    return tuple(rec.link_indices) if rec is not None else ()


def _selected_metric(step: StepResult, agent_id: str) -> tuple[str | None, dict | None]:
    """Return (selected_fp, observation dict for that fp) from a per-agent step."""
    for obs in step.per_agent:
        if obs.agent_id == agent_id:
            sel = obs.selected_path_fp
            if sel is None:
                return None, None
            for ap, m in zip(obs.available_paths, obs.path_metrics, strict=True):
                if ap == sel:
                    return sel, m
            return sel, None
    return None, None


@dataclasses.dataclass
class _StepInfo:
    """Bookkeeping ferried between observation builds and reward computation."""

    step: StepResult
    obs: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]


class ScionMARLEnv(ParallelEnv):
    """PettingZoo Parallel env for SCION multi-path selection.

    `agents` is the list of agent ids alive on the *current* step. After
    the episode terminates, `agents` becomes empty (PettingZoo convention).
    """

    metadata = {"name": "scion_marl_v0", "is_parallelizable": True}

    def __init__(self, cfg: Phase1Config) -> None:
        self.cfg = cfg
        self.possible_agents: list[str] = [
            f"agent_{i:02d}" for i in range(cfg.sim.topology.num_agents)
        ]
        self.agents: list[str] = list(self.possible_agents)

        obs_dim = cfg.observation.top_k * PER_PATH_FEATURES + SCALAR_FEATURES
        self._obs_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32,
        )
        self._action_space = spaces.Discrete(cfg.observation.top_k)

        self.sim: Phase0Simulator | None = None
        self.builder: ObservationBuilder | None = None
        self._episode_iter = None
        self._n_obs_per_episode = int(round(
            cfg.sim.trajectory.episode_seconds / cfg.sim.trajectory.observation_dt_sec
        ))
        self._t_step = 0  # 0-indexed observation tick within the current episode

    # PettingZoo Parallel API
    def observation_space(self, agent: str) -> spaces.Space:
        return self._obs_space

    def action_space(self, agent: str) -> spaces.Space:
        return self._action_space

    def reset(
        self, seed: int | None = None, options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        self.agents = list(self.possible_agents)
        self.sim = Phase0Simulator(self.cfg.sim, episode_seed=seed)
        self.builder = ObservationBuilder(self.agents, self.cfg.observation)
        self._episode_iter = self.sim.run_episode()
        self._t_step = 0

        step = next(self._episode_iter)
        obs, masks = self._observations_from_step(step)
        infos = {a: {"action_mask": masks[a]} for a in self.agents}
        return obs, infos

    def step(
        self, actions: dict[str, int],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        if self.sim is None or self.builder is None or self._episode_iter is None:
            raise RuntimeError("call reset() before step()")

        # Translate slot actions to path fingerprints, then inject.
        last_step = self._last_step
        picks: dict[str, str] = {}
        for agent_id in self.agents:
            slot = int(actions.get(agent_id, 0))
            mask = self._last_masks[agent_id]
            slot = coerce_action(slot, mask)
            if slot < 0:
                continue
            obs_for_agent = next(
                o for o in last_step.per_agent if o.agent_id == agent_id
            )
            fp = slot_to_fingerprint(
                slot, obs_for_agent.available_paths, obs_for_agent.path_metrics,
                self.cfg.observation,
            )
            if fp is not None:
                picks[agent_id] = fp
        self.sim.set_pending_selections(picks)

        # Advance one observation tick.
        try:
            step = next(self._episode_iter)
        except StopIteration:
            terminations = {a: True for a in self.agents}
            truncations = {a: False for a in self.agents}
            obs = {a: np.zeros(self._obs_space.shape, dtype=np.float32) for a in self.agents}
            rewards = {a: 0.0 for a in self.agents}
            infos = {a: {"action_mask": np.zeros(self.cfg.observation.top_k, dtype=bool)} for a in self.agents}
            self.agents = []
            return obs, rewards, terminations, truncations, infos

        # Build observations for the new tick.
        obs, masks = self._observations_from_step(step)

        # Reward per agent from the new global state.
        link_demand = _link_agent_demand(self.sim)
        link_caps = self.sim.control.link_capacities
        link_utils = self.sim.link_util
        agent_assignments = dict(self.sim.selected)

        rewards: dict[str, float] = {}
        infos: dict[str, dict] = {}
        for agent_id in self.agents:
            sel_fp, sel_metric = _selected_metric(step, agent_id)
            link_idx = _selected_link_indices(self.sim, agent_id)
            inp = RewardInputs(
                agent_id=agent_id,
                selected_fp=sel_fp,
                selected_metric=sel_metric,
                selected_link_indices=link_idx,
                link_utilizations=link_utils,
                link_capacities=link_caps,
                link_agent_demand=link_demand,
                agent_assignments=agent_assignments,
            )
            components = composite_reward(inp, self.cfg.reward)
            rewards[agent_id] = float(components["total"])
            infos[agent_id] = {
                "action_mask": masks[agent_id],
                "reward_components": components,
            }

        self._t_step += 1
        episode_done = self._t_step >= self._n_obs_per_episode - 1
        terminations = {a: episode_done for a in self.agents}
        truncations = {a: False for a in self.agents}
        if episode_done:
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    def _observations_from_step(
        self, step: StepResult,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Run the builder on a StepResult; cache the step + masks for next step()."""
        assert self.builder is not None
        obs: dict[str, np.ndarray] = {}
        masks: dict[str, np.ndarray] = {}
        per_agent_lookup = {o.agent_id: o for o in step.per_agent}
        for agent_id in self.agents:
            o = per_agent_lookup[agent_id]
            vec, mask = self.builder.build(
                agent_id, o.available_paths, o.path_metrics, o.selected_path_fp,
            )
            obs[agent_id] = vec
            masks[agent_id] = mask
        self._last_step = step
        self._last_masks = masks
        return obs, masks

    def render(self) -> Any:
        """No-op renderer; PettingZoo allows None."""
        return None

    def close(self) -> None:
        self.sim = None
        self.builder = None
        self._episode_iter = None


def make_env(cfg: Phase1Config | None = None) -> ScionMARLEnv:
    """Factory that returns a configured `ScionMARLEnv`.

    Pass `None` for default `Phase1Config` (matching default Phase 0 sim params).
    """
    return ScionMARLEnv(cfg if cfg is not None else Phase1Config())
