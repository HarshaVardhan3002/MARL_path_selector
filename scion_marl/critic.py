"""Centralized critic for MAPPO.

Input is the *global* state at a timestep:
    [stacked per-agent observations | per-agent selected-slot one-hot | link utilizations]
of dimension `N*obs_dim + N*top_k + num_links`. With Phase 2 stress-regime
defaults (N=48, obs_dim=31, top_k=4, num_links=64) the critic input is
`48*31 + 48*4 + 64 = 1744`.

Output is a scalar V(s) shared across agents at training time. The
critic is discarded at deployment (CTDE).

The exact concatenation format is locked in J-026 — changing it
invalidates trained checkpoints.
"""
from __future__ import annotations

import torch
from torch import nn

from scion_marl.agent import _mlp


class CriticNetwork(nn.Module):
    def __init__(
        self,
        num_agents: int = 48,
        obs_dim: int = 31,
        top_k: int = 4,
        num_links: int = 64,
        hidden_sizes: list[int] | None = None,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.top_k = top_k
        self.num_links = num_links
        self.input_dim = num_agents * obs_dim + num_agents * top_k + num_links
        hs = hidden_sizes or [256, 256]
        self.trunk = _mlp(self.input_dim, hs[:-1], hs[-1], activation)
        self.head = nn.Linear(hs[-1], 1)
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=2.0**0.5)
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.head.weight, gain=1.0)
        nn.init.zeros_(self.head.bias)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """`global_state`: (B, input_dim) — returns (B,) value estimates."""
        h = self.trunk(global_state)
        return self.head(h).squeeze(-1)


def build_global_state(
    per_agent_obs: torch.Tensor,
    selected_slot: torch.Tensor,
    link_utilizations: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Pack one timestep into the critic input vector.

    Args:
        per_agent_obs: (N, obs_dim) — one row per agent.
        selected_slot: (N,) int — current top-K slot per agent (-1 if not in top-K).
        link_utilizations: (num_links,) float.
        top_k: K (number of slots).

    Returns:
        (input_dim,) float tensor.
    """
    n, obs_dim = per_agent_obs.shape
    one_hot = torch.zeros(n, top_k, dtype=per_agent_obs.dtype, device=per_agent_obs.device)
    valid_mask = (selected_slot >= 0) & (selected_slot < top_k)
    if bool(valid_mask.any()):
        valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        one_hot[valid_idx, selected_slot[valid_idx].long()] = 1.0
    return torch.cat(
        [per_agent_obs.reshape(n * obs_dim),
         one_hot.reshape(n * top_k),
         link_utilizations.to(per_agent_obs.dtype)],
        dim=0,
    )
