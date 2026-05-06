"""On-policy rollout buffer for MAPPO.

Stores `T` timesteps × `N` agents of (obs, action, mask, log_prob, reward,
done) plus per-timestep V(s) from the centralized critic. After the
rollout is collected, `compute_gae` populates per-agent advantages and
returns. `iter_minibatches` yields PPO minibatches over the flattened
(T*N) transition set.

Reward stored is the *total* composite reward (per-agent); per-component
breakdowns are forwarded to TensorBoard separately by the trainer.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch


@dataclasses.dataclass
class RolloutBuffer:
    """Pre-allocated on-policy buffer.

    Indexing convention: `t` is the timestep within the rollout
    (0..T-1). Per-agent dims are 0..N-1. The critic value `values[t]`
    is shared across all agents at timestep t (centralized critic).
    """

    num_timesteps: int       # T = rollout_episodes * episode_length
    num_agents: int          # N
    obs_dim: int
    action_dim: int          # K
    global_state_dim: int
    device: torch.device

    obs: torch.Tensor = dataclasses.field(init=False)
    masks: torch.Tensor = dataclasses.field(init=False)
    actions: torch.Tensor = dataclasses.field(init=False)
    log_probs: torch.Tensor = dataclasses.field(init=False)
    rewards: torch.Tensor = dataclasses.field(init=False)
    dones: torch.Tensor = dataclasses.field(init=False)
    global_states: torch.Tensor = dataclasses.field(init=False)
    values: torch.Tensor = dataclasses.field(init=False)
    advantages: torch.Tensor = dataclasses.field(init=False)
    returns: torch.Tensor = dataclasses.field(init=False)

    _ptr: int = 0
    full: bool = False

    def __post_init__(self) -> None:
        T, N, D, K, G = (
            self.num_timesteps, self.num_agents, self.obs_dim,
            self.action_dim, self.global_state_dim,
        )
        d = self.device
        self.obs = torch.zeros(T, N, D, device=d)
        self.masks = torch.zeros(T, N, K, device=d, dtype=torch.bool)
        self.actions = torch.zeros(T, N, device=d, dtype=torch.long)
        self.log_probs = torch.zeros(T, N, device=d)
        self.rewards = torch.zeros(T, N, device=d)
        self.dones = torch.zeros(T, N, device=d)
        self.global_states = torch.zeros(T, G, device=d)
        self.values = torch.zeros(T, device=d)
        self.advantages = torch.zeros(T, N, device=d)
        self.returns = torch.zeros(T, N, device=d)

    def reset(self) -> None:
        self._ptr = 0
        self.full = False

    def store(
        self,
        *,
        obs: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        global_state: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        t = self._ptr
        self.obs[t] = obs
        self.masks[t] = mask
        self.actions[t] = action
        self.log_probs[t] = log_prob
        self.rewards[t] = reward
        self.dones[t] = done
        self.global_states[t] = global_state
        self.values[t] = value
        self._ptr += 1
        if self._ptr >= self.num_timesteps:
            self.full = True

    def compute_gae(
        self,
        last_value: torch.Tensor,
        last_done: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Standard GAE-λ over the rollout.

        `last_value`: V(s_T) — the bootstrap value at the timestep AFTER
        the buffer's last entry. Shape (). `last_done`: per-agent done
        flag at that boundary, shape (N,). Per-agent advantages and
        returns are written to `self.advantages` / `self.returns`.
        """
        T = self._ptr
        N = self.num_agents
        adv = torch.zeros(T, N, device=self.device)
        gae = torch.zeros(N, device=self.device)
        next_value = last_value.expand(N).clone()
        next_nonterminal = (1.0 - last_done.float())
        for t in reversed(range(T)):
            value_t = self.values[t].expand(N)
            delta = self.rewards[t] + gamma * next_value * next_nonterminal - value_t
            gae = delta + gamma * gae_lambda * next_nonterminal * gae
            adv[t] = gae
            next_value = value_t
            next_nonterminal = (1.0 - self.dones[t])
        self.advantages = adv
        self.returns = adv + self.values[:T].unsqueeze(1).expand(T, N)

    def iter_minibatches(
        self, minibatch_size: int, normalize_adv: bool = True,
    ):
        """Yield PPO minibatches as dicts of flattened tensors.

        Flattens (T, N, ...) → (T*N, ...). Advantages are optionally
        z-score normalized over the full rollout (PPO-standard).
        """
        T = self._ptr
        N = self.num_agents
        n_total = T * N

        obs_flat = self.obs[:T].reshape(n_total, self.obs_dim)
        mask_flat = self.masks[:T].reshape(n_total, self.action_dim)
        act_flat = self.actions[:T].reshape(n_total)
        logp_flat = self.log_probs[:T].reshape(n_total)
        adv_flat = self.advantages[:T].reshape(n_total)
        ret_flat = self.returns[:T].reshape(n_total)
        gs_flat = self.global_states[:T].unsqueeze(1).expand(T, N, self.global_state_dim).reshape(n_total, self.global_state_dim)

        if normalize_adv and n_total > 1:
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        idx = torch.randperm(n_total, device=self.device)
        for start in range(0, n_total, minibatch_size):
            mb = idx[start:start + minibatch_size]
            yield {
                "obs": obs_flat[mb],
                "mask": mask_flat[mb],
                "action": act_flat[mb],
                "old_log_prob": logp_flat[mb],
                "advantage": adv_flat[mb],
                "return_": ret_flat[mb],
                "global_state": gs_flat[mb],
            }
