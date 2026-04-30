"""Path birth/death + asymmetric-view control plane.

Per JUSTIFICATIONS J-002, lifetimes / discrepancy / churn live outside the
fluid simulator. This module owns:

* fingerprint generation (8-byte hex, matching ScionPathML)
* per-agent candidate-path pools that grow and shrink over time
* asymmetric view: at path creation, decide whether the reverse direction's
  vantage point also sees this path (J-010 proxy)
"""
from __future__ import annotations

import dataclasses
import secrets
from typing import Iterable

import numpy as np

from scion_sim.config import ChurnConfig, DiscrepancyConfig, LifetimeConfig


def make_fingerprint(rng: np.random.Generator) -> str:
    """8-byte (16 hex char) lowercase fingerprint, matching ScionPathML."""
    # numpy Generators yield bit-exact reproducibility; secrets is for cross-platform fallback only.
    raw = int(rng.integers(0, 1 << 63, dtype=np.uint64))
    return f"{raw:016x}"


@dataclasses.dataclass
class PathRecord:
    fingerprint: str
    agent_id: str            # owning agent (the path is in this agent's candidate pool)
    visible_to_owner: bool   # always True at birth in our model
    visible_to_reverse: bool # asymmetric view: may be False (discrepancy proxy)
    born_t_sec: float
    expires_t_sec: float
    base_rtt_ms: float
    base_bandwidth_mbps: float
    link_indices: tuple[int, ...]  # which network link resources this path uses

    def is_alive(self, t_sec: float) -> bool:
        return self.born_t_sec <= t_sec < self.expires_t_sec

    def age_sec(self, t_sec: float) -> float:
        return max(0.0, t_sec - self.born_t_sec)


def sample_lifetime(cfg: LifetimeConfig, rng: np.random.Generator) -> float:
    """Sample one path lifetime in seconds from the configured distribution."""
    if cfg.distribution == "lognormal":
        # median = exp(mu)  ->  mu = ln(median)
        mu = float(np.log(cfg.median_seconds))
        return float(rng.lognormal(mean=mu, sigma=cfg.sigma))
    if cfg.distribution == "exponential":
        # exponential's median = ln(2) * scale  ->  scale = median / ln(2)
        return float(rng.exponential(scale=cfg.median_seconds / np.log(2.0)))
    if cfg.distribution == "weibull":
        # Weibull's median = scale * (ln 2)^(1/k); we solve for scale.
        k = cfg.weibull_shape
        scale = cfg.median_seconds / (np.log(2.0) ** (1.0 / k))
        return float(scale * rng.weibull(k))
    raise ValueError(f"Unknown distribution: {cfg.distribution}")


class TopologyEventLog:
    """Append-only structure of (t_sec, agent_id, fingerprint, type) events."""
    __slots__ = ("_events",)

    def __init__(self) -> None:
        self._events: list[tuple[float, str, str, str]] = []

    def log(self, t_sec: float, agent_id: str, fingerprint: str, event_type: str) -> None:
        self._events.append((t_sec, agent_id, fingerprint, event_type))

    def drain(self) -> list[tuple[float, str, str, str]]:
        out, self._events = self._events, []
        return out


class TopologyControlPlane:
    """Owns the per-agent candidate-path pools and fires birth/death events."""

    def __init__(
        self,
        agent_ids: list[str],
        paths_per_agent: int,
        num_links: int,
        avg_hops_per_path: float,
        link_capacity_mbps_mean: float,
        link_capacity_mbps_std: float,
        base_rtt_ms_mean: float,
        base_rtt_ms_std: float,
        churn: ChurnConfig,
        discrepancy: DiscrepancyConfig,
        lifetime: LifetimeConfig,
        episode_seconds: float,
        rng: np.random.Generator,
    ) -> None:
        self.agent_ids = agent_ids
        self.paths_per_agent = paths_per_agent
        self.num_links = num_links
        self.avg_hops = max(1.0, avg_hops_per_path)
        self.cap_mean = link_capacity_mbps_mean
        self.cap_std = link_capacity_mbps_std
        self.rtt_mean = base_rtt_ms_mean
        self.rtt_std = base_rtt_ms_std
        self.churn = churn
        self.discrepancy = discrepancy
        self.lifetime = lifetime
        self.episode_seconds = episode_seconds
        self.rng = rng
        self.event_log = TopologyEventLog()

        # Stable per-link capacity (used both as fluid-model resource capacity and observed-BW base).
        self.link_capacities = np.maximum(
            5.0,
            self.rng.normal(loc=self.cap_mean, scale=self.cap_std, size=num_links),
        )
        # Per-agent candidate pool — the spec calls these "available paths".
        self.live_paths: dict[str, dict[str, PathRecord]] = {a: {} for a in agent_ids}
        self._populate_initial(t_sec=0.0)

    def _sample_link_indices(self) -> tuple[int, ...]:
        n = max(1, int(round(self.rng.normal(self.avg_hops, max(1.0, self.avg_hops * 0.2)))))
        n = min(n, self.num_links)
        return tuple(int(i) for i in self.rng.choice(self.num_links, size=n, replace=False))

    def _new_record(self, agent_id: str, t_sec: float) -> PathRecord:
        fp = make_fingerprint(self.rng)
        lifetime_s = sample_lifetime(self.lifetime, self.rng)
        link_idx = self._sample_link_indices()
        cap = float(np.mean(self.link_capacities[list(link_idx)]))
        rtt = max(1.0, float(self.rng.normal(self.rtt_mean, self.rtt_std)))
        visible_to_reverse = bool(self.rng.random() >= self.discrepancy.asymmetry_rate)
        return PathRecord(
            fingerprint=fp,
            agent_id=agent_id,
            visible_to_owner=True,
            visible_to_reverse=visible_to_reverse,
            born_t_sec=t_sec,
            expires_t_sec=t_sec + lifetime_s,
            base_rtt_ms=rtt,
            base_bandwidth_mbps=cap,
            link_indices=link_idx,
        )

    def _populate_initial(self, t_sec: float) -> None:
        target_visible = max(1, int(round(self.paths_per_agent * self.churn.initial_visible_fraction)))
        for agent_id in self.agent_ids:
            for _ in range(target_visible):
                rec = self._new_record(agent_id, t_sec)
                self.live_paths[agent_id][rec.fingerprint] = rec
                self.event_log.log(t_sec, agent_id, rec.fingerprint, "path_appeared")

    def step(self, t_sec: float, dt_sec: float) -> None:
        """Advance the control plane by dt; expire old paths and birth new ones (Poisson)."""
        # --- Expirations (deterministic given lifetime sampled at birth)
        for agent_id, paths in self.live_paths.items():
            expired = [fp for fp, p in paths.items() if not p.is_alive(t_sec)]
            for fp in expired:
                del paths[fp]
                self.event_log.log(t_sec, agent_id, fp, "path_expired")

        # --- New arrivals (Poisson with rate per agent per hour)
        rate_per_sec = self.churn.appear_rate_per_agent_per_hour / 3600.0
        for agent_id in self.agent_ids:
            # Cap candidate pool at paths_per_agent
            slots = self.paths_per_agent - len(self.live_paths[agent_id])
            if slots <= 0:
                continue
            # Number of arrivals in (t, t+dt) is Poisson(rate * dt).
            n_new = int(self.rng.poisson(rate_per_sec * dt_sec))
            for _ in range(min(n_new, slots)):
                rec = self._new_record(agent_id, t_sec)
                self.live_paths[agent_id][rec.fingerprint] = rec
                self.event_log.log(t_sec, agent_id, rec.fingerprint, "path_appeared")

    def visible_paths(self, agent_id: str) -> Iterable[PathRecord]:
        return self.live_paths[agent_id].values()

    def all_records(self) -> Iterable[PathRecord]:
        for paths in self.live_paths.values():
            yield from paths.values()
