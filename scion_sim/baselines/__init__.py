"""Phase 1 baseline policies: random, shortest-path, oracle-greedy.

Each policy is a callable `policy(observation, action_mask) -> int` that
returns a top-K slot index. Policy classes carry a `name` attribute and a
`reset()` method called at episode start.
"""
from __future__ import annotations

from scion_sim.baselines.oracle_greedy_policy import OracleGreedyPolicy
from scion_sim.baselines.random_policy import RandomPolicy
from scion_sim.baselines.shortest_path_policy import ShortestPathPolicy

__all__ = ["OracleGreedyPolicy", "RandomPolicy", "ShortestPathPolicy"]
