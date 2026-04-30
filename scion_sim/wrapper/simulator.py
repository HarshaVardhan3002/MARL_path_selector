"""Top-level Phase 0 simulator that drives the fluid ODE under control-plane
events and emits trajectories conforming to docs/SCHEMA.md.

The action interface is intentionally minimal: at each observation tick, each
agent picks one currently-visible path (random in Phase 0). The fluid model
runs at `cfg.fluid.dt_sec` regardless of observation cadence; observations
are sampled every `cfg.trajectory.observation_dt_sec`.
"""
from __future__ import annotations

import dataclasses
from typing import Iterator

import numpy as np

from scion_sim.config import SimulatorConfig
from scion_sim.control_plane.observations import observe_path
from scion_sim.control_plane.topology import PathRecord, TopologyControlPlane
from scion_sim.wrapper.network import step_flow_rates


@dataclasses.dataclass
class StepObservation:
    """Per-agent view at one observation tick."""
    agent_id: str
    selected_path_fp: str | None
    available_paths: list[str]
    path_metrics: list[dict]


@dataclasses.dataclass
class GlobalObservation:
    link_utilizations: list[dict]
    total_throughput: float
    agent_assignments: list[dict]


@dataclasses.dataclass
class StepResult:
    t: int
    wall_time_sec: float
    per_agent: list[StepObservation]
    global_state: GlobalObservation
    events: list[tuple[float, str, str, str]]  # (t_sec, agent_id, fp, type)


class Phase0Simulator:
    def __init__(self, cfg: SimulatorConfig, *, episode_seed: int | None = None) -> None:
        self.cfg = cfg
        seed = cfg.seed if episode_seed is None else int(episode_seed)
        self.rng = np.random.default_rng(seed)
        self.agent_ids = [f"agent_{i:02d}" for i in range(cfg.topology.num_agents)]

        self.control = TopologyControlPlane(
            agent_ids=self.agent_ids,
            paths_per_agent=cfg.topology.paths_per_agent,
            num_links=cfg.topology.num_links,
            avg_hops_per_path=cfg.topology.avg_hops_per_path,
            link_capacity_mbps_mean=cfg.topology.link_capacity_mbps_mean,
            link_capacity_mbps_std=cfg.topology.link_capacity_mbps_std,
            base_rtt_ms_mean=cfg.topology.base_rtt_ms_mean,
            base_rtt_ms_std=cfg.topology.base_rtt_ms_std,
            churn=cfg.churn,
            discrepancy=cfg.discrepancy,
            lifetime=cfg.lifetime,
            episode_seconds=cfg.trajectory.episode_seconds,
            rng=self.rng,
        )

        self.t_sec: float = 0.0
        self.t_index: int = 0
        # Per-agent currently-selected path fingerprint.
        self.selected: dict[str, str | None] = {a: None for a in self.agent_ids}
        # Active flow rate per (agent, fingerprint). Deleted when path expires.
        self.flow_rates: dict[tuple[str, str], float] = {}
        # Current link utilization (filled after first step).
        self.link_util: np.ndarray = np.zeros(cfg.topology.num_links)

        self._initial_select()

    # ------------------------------------------------------------------
    def _initial_select(self) -> None:
        """Pick an arbitrary visible path per agent at episode start."""
        for agent_id in self.agent_ids:
            paths = list(self.control.visible_paths(agent_id))
            if not paths:
                continue
            choice = paths[int(self.rng.integers(0, len(paths)))]
            self.selected[agent_id] = choice.fingerprint
            self.flow_rates[(agent_id, choice.fingerprint)] = self.cfg.fluid.init_flow_rate

    def _ensure_selection(self) -> None:
        """Re-pick if an agent's selection expired or it has no selection."""
        for agent_id in self.agent_ids:
            current_fp = self.selected[agent_id]
            visible = list(self.control.visible_paths(agent_id))
            visible_fps = {p.fingerprint for p in visible}
            if current_fp is None or current_fp not in visible_fps:
                # drop the stale flow if any
                if current_fp is not None:
                    self.flow_rates.pop((agent_id, current_fp), None)
                if visible:
                    choice = visible[int(self.rng.integers(0, len(visible)))]
                    self.selected[agent_id] = choice.fingerprint
                    self.flow_rates.setdefault((agent_id, choice.fingerprint), self.cfg.fluid.init_flow_rate)
                else:
                    self.selected[agent_id] = None

    def _active_records(self) -> tuple[list[PathRecord], list[float]]:
        """Records and rates for paths currently carrying flow (one per agent's selection)."""
        recs: list[PathRecord] = []
        rates: list[float] = []
        for agent_id in self.agent_ids:
            fp = self.selected[agent_id]
            if fp is None:
                continue
            rec = self.control.live_paths[agent_id].get(fp)
            if rec is None:
                continue
            recs.append(rec)
            rates.append(self.flow_rates.get((agent_id, fp), self.cfg.fluid.init_flow_rate))
        return recs, rates

    # ------------------------------------------------------------------
    def _advance_one_observation(self) -> tuple[float, np.ndarray]:
        """Run multiple ODE micro-steps until we accumulate observation_dt_sec."""
        steps_per_obs = max(1, int(round(self.cfg.trajectory.observation_dt_sec / self.cfg.fluid.dt_sec)))
        last_util = self.link_util
        for _ in range(steps_per_obs):
            # 1. Control plane: birth/death events at this micro-step.
            self.control.step(self.t_sec, self.cfg.fluid.dt_sec)
            # 2. Re-select if the agent's chosen path just expired.
            self._ensure_selection()
            # 3. Build active state vector and step the ODE.
            recs, rates = self._active_records()
            if recs:
                state = np.asarray(rates, dtype=np.float64)
                next_state, util = step_flow_rates(
                    state, recs, self.control.link_capacities, self.cfg.fluid,
                )
                last_util = util
                for rec, x in zip(recs, next_state, strict=True):
                    self.flow_rates[(rec.agent_id, rec.fingerprint)] = float(x)
            else:
                last_util = np.zeros_like(self.control.link_capacities)
            self.t_sec += self.cfg.fluid.dt_sec
        self.link_util = last_util
        total_flow = float(sum(self.flow_rates.values()))
        return total_flow, last_util

    def _build_observation(self, total_flow: float) -> StepResult:
        per_agent: list[StepObservation] = []
        for agent_id in self.agent_ids:
            visible = list(self.control.visible_paths(agent_id))
            metrics = [
                observe_path(
                    rec,
                    self.flow_rates.get((agent_id, rec.fingerprint), 0.0),
                    self.link_util,
                    self.cfg.observation_noise,
                    self.rng,
                    self.t_sec,
                )
                for rec in visible
            ]
            per_agent.append(
                StepObservation(
                    agent_id=agent_id,
                    selected_path_fp=self.selected[agent_id],
                    available_paths=[rec.fingerprint for rec in visible],
                    path_metrics=metrics,
                )
            )
        global_obs = GlobalObservation(
            link_utilizations=[
                {"link_id": int(i), "utilization": float(u)} for i, u in enumerate(self.link_util)
            ],
            total_throughput=total_flow,
            agent_assignments=[
                {"agent_id": a, "fp": self.selected[a]} for a in self.agent_ids if self.selected[a]
            ],
        )
        events = self.control.event_log.drain()
        return StepResult(
            t=self.t_index,
            wall_time_sec=self.t_sec,
            per_agent=per_agent,
            global_state=global_obs,
            events=events,
        )

    def run_episode(self) -> Iterator[StepResult]:
        n_obs = int(round(self.cfg.trajectory.episode_seconds / self.cfg.trajectory.observation_dt_sec))
        for _ in range(n_obs):
            total_flow, _ = self._advance_one_observation()
            yield self._build_observation(total_flow)
            self.t_index += 1
