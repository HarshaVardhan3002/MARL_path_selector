"""Shared MLP actor for MAPPO.

Takes a 31-dim observation, outputs 4-dim action logits. Action masking
is applied via a large-negative additive bias on masked slots before
the softmax — standard PPO-with-mask trick (`-INF_MASK` survives
log-softmax cleanly).

Single shared network across all agents (parameter sharing per
CONTEXT.md / J-021 deployment story).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

INF_MASK = 1e9


def _mlp(in_dim: int, hidden_sizes: list[int], out_dim: int, activation: str) -> nn.Sequential:
    act_cls = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(act_cls())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def _orthogonal_init(module: nn.Module, gain: float) -> None:
    """PPO standard: orthogonal init with policy-output gain 0.01."""
    for layer in module:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)


class ActorNetwork(nn.Module):
    """Shared MLP. Forward returns (logits, dist) ready for PPO."""

    def __init__(
        self,
        obs_dim: int = 31,
        action_dim: int = 4,
        hidden_sizes: list[int] | None = None,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        hs = hidden_sizes or [256, 256]
        self.trunk = _mlp(obs_dim, hs[:-1], hs[-1], activation)
        self.head = nn.Linear(hs[-1], action_dim)
        # PPO-standard small policy-head gain so initial actions are near-uniform.
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=2.0**0.5)
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(
        self, obs: torch.Tensor, action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Categorical]:
        """Compute action logits and a masked Categorical distribution.

        `obs`: (B, obs_dim). `action_mask`: (B, action_dim) of 0/1 (or bool).
        Masked slots get a large negative bias so they vanish under softmax.
        """
        h = self.trunk(obs)
        logits = self.head(h)
        if action_mask is not None:
            mask = action_mask.to(dtype=torch.bool)
            logits = logits.masked_fill(~mask, -INF_MASK)
        return logits, Categorical(logits=logits)

    def act(
        self, obs: torch.Tensor, action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample (or argmax) an action; return (action, log_prob)."""
        logits, dist = self.forward(obs, action_mask)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate(
        self, obs: torch.Tensor, action: torch.Tensor, action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """For PPO update: log-prob of `action` under current policy + entropy."""
        _, dist = self.forward(obs, action_mask)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy
