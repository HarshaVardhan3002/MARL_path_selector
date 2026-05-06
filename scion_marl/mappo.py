"""MAPPO trainer (CleanRL-style single-file).

Wraps `ScionMARLEnv` (Phase 1) with a centralized critic and a shared
actor. Standard PPO clipped objective, GAE-λ advantages, value-loss
clipping omitted in favour of the simpler unclipped value loss (PPO
implementations diverge here; the simpler form is fine for our
small-scale problem). TensorBoard logs reward, all loss components,
policy entropy, approximate KL, clip fraction, and per-component
reward breakdown.

Auto-restart rule (PHASE_2_SPEC §1.3): if mean policy entropy drops
below `auto_restart_entropy_floor` before
`auto_restart_threshold_episode`, re-initialize the actor + critic
once with `entropy_coef = auto_restart_entropy_coef`. Logged in the
per-run manifest. Beyond that single restart, training proceeds even
if entropy collapses.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from scion_marl.agent import ActorNetwork
from scion_marl.buffer import RolloutBuffer
from scion_marl.config import Phase2Config
from scion_marl.critic import CriticNetwork, build_global_state
from scion_marl.device import DeviceInfo, select_device
from scion_sim.env import make_env
from scion_sim.env.observation_builder import PER_PATH_FEATURES

logger = logging.getLogger("scion_marl.mappo")

# Index of the S2 (selected-slot/K) feature in the 31-dim observation.
# scalar block starts at top_k * PER_PATH_FEATURES; S2 is offset 1 within.
def _s2_index(top_k: int) -> int:
    return top_k * PER_PATH_FEATURES + 1


@dataclasses.dataclass
class TrainResult:
    final_episode: int
    auto_restarted: bool
    final_metrics: dict[str, float]
    elapsed_sec: float
    checkpoint_path: Path


class MappoTrainer:
    def __init__(
        self,
        cfg: Phase2Config,
        run_dir: Path,
        device_info: DeviceInfo | None = None,
    ) -> None:
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device_info = device_info or select_device()
        self.device = self.device_info.device

        self._set_seed(cfg.seed)

        self.env = make_env(cfg.phase1)
        self.num_agents = cfg.phase1.sim.topology.num_agents
        self.num_links = cfg.phase1.sim.topology.num_links
        self.top_k = cfg.phase1.observation.top_k
        self.obs_dim = self.top_k * PER_PATH_FEATURES + 7
        self.s2_idx = _s2_index(self.top_k)

        self.actor: ActorNetwork
        self.critic: CriticNetwork
        self.actor_optim: torch.optim.Optimizer
        self.critic_optim: torch.optim.Optimizer
        self._build_networks()

        self.buffer = RolloutBuffer(
            num_timesteps=cfg.mappo.rollout_episodes * cfg.mappo.episode_length,
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
            action_dim=self.top_k,
            global_state_dim=self.critic.input_dim,
            device=self.device,
        )

        self.writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))
        self._jsonl_path = self.run_dir / "metrics.jsonl"
        # Truncate any prior JSONL on fresh init.
        self._jsonl_path.write_text("", encoding="utf-8")
        self.episodes_done = 0
        self.entropy_coef = cfg.mappo.entropy_coef
        self.auto_restarted = False
        self.last_metrics: dict[str, float] = {}

    # ------------------------------------------------------------------
    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _build_networks(self) -> None:
        self.actor = ActorNetwork(
            obs_dim=self.obs_dim, action_dim=self.top_k,
            hidden_sizes=self.cfg.mappo.hidden_sizes,
            activation=self.cfg.mappo.activation,
        ).to(self.device)
        self.critic = CriticNetwork(
            num_agents=self.num_agents, obs_dim=self.obs_dim, top_k=self.top_k,
            num_links=self.num_links,
            hidden_sizes=self.cfg.mappo.hidden_sizes,
            activation=self.cfg.mappo.activation,
        ).to(self.device)
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=self.cfg.mappo.learning_rate, eps=1e-5,
        )
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=self.cfg.mappo.learning_rate, eps=1e-5,
        )

    # ------------------------------------------------------------------
    # Env <-> tensor plumbing
    def _agents_obs_to_tensor(self, obs_dict: dict[str, np.ndarray]) -> torch.Tensor:
        return torch.from_numpy(
            np.stack([obs_dict[a] for a in self.env.possible_agents], axis=0)
        ).to(self.device)

    def _agents_mask_to_tensor(self, infos: dict[str, dict]) -> torch.Tensor:
        return torch.from_numpy(
            np.stack([infos[a]["action_mask"] for a in self.env.possible_agents], axis=0)
        ).to(self.device)

    def _selected_slot_per_agent(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        s2 = obs_tensor[:, self.s2_idx]
        slots = (s2 * self.top_k).round().long()
        return torch.where(
            s2 < 0,
            torch.full_like(slots, -1),
            slots.clamp(0, self.top_k - 1),
        )

    def _global_state(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        link_util = torch.from_numpy(self.env.sim.link_util).float().to(self.device)
        slots = self._selected_slot_per_agent(obs_tensor)
        return build_global_state(obs_tensor, slots, link_util, self.top_k)

    # ------------------------------------------------------------------
    def collect_rollout(self) -> tuple[float, dict[str, float]]:
        """Fill the buffer with `rollout_episodes` worth of transitions.

        Returns (mean episode total reward across rollout, mean reward
        components dict).
        """
        self.buffer.reset()
        rollout_rewards: list[float] = []
        rollout_components: dict[str, list[float]] = {
            "throughput": [], "congestion_contribution": [], "diversity": [], "total": [],
        }
        last_done_per_agent = torch.zeros(self.num_agents, device=self.device)
        last_obs_tensor: torch.Tensor | None = None

        for ep_in_rollout in range(self.cfg.mappo.rollout_episodes):
            seed = self.cfg.seed * 1_000_000 + self.episodes_done
            obs_dict, infos = self.env.reset(seed=seed)
            episode_reward_sum = 0.0
            ep_components: dict[str, float] = {k: 0.0 for k in rollout_components}

            done_flag = False
            while self.env.agents and not done_flag:
                obs_tensor = self._agents_obs_to_tensor(obs_dict)
                mask_tensor = self._agents_mask_to_tensor(infos)
                with torch.no_grad():
                    action, log_prob = self.actor.act(obs_tensor, mask_tensor)
                    g = self._global_state(obs_tensor)
                    value = self.critic(g.unsqueeze(0)).squeeze(0)

                action_dict = {
                    a: int(action[i].item()) for i, a in enumerate(self.env.possible_agents)
                }
                next_obs_dict, rewards_dict, terms, truncs, infos = self.env.step(action_dict)
                rewards_t = torch.tensor(
                    [rewards_dict[a] for a in self.env.possible_agents],
                    device=self.device, dtype=torch.float32,
                )
                done_t = torch.tensor(
                    [float(terms[a] or truncs[a]) for a in self.env.possible_agents],
                    device=self.device, dtype=torch.float32,
                )

                self.buffer.store(
                    obs=obs_tensor, mask=mask_tensor, action=action,
                    log_prob=log_prob, reward=rewards_t, done=done_t,
                    global_state=g, value=value,
                )

                episode_reward_sum += float(rewards_t.sum())
                for k in rollout_components:
                    if k == "total":
                        ep_components[k] += float(rewards_t.sum())
                    else:
                        ep_components[k] += sum(
                            float(infos[a]["reward_components"][k])
                            for a in self.env.possible_agents
                            if a in infos and "reward_components" in infos[a]
                        )

                done_flag = bool(done_t.any().item())
                last_done_per_agent = done_t
                last_obs_tensor = self._agents_obs_to_tensor(next_obs_dict)
                obs_dict = next_obs_dict
                if not self.env.agents:
                    break

            rollout_rewards.append(episode_reward_sum)
            for k, v in ep_components.items():
                rollout_components[k].append(v)
            self.episodes_done += 1

        # Bootstrap value for GAE.
        if last_obs_tensor is not None:
            with torch.no_grad():
                last_g = self._global_state(last_obs_tensor)
                last_value = self.critic(last_g.unsqueeze(0)).squeeze(0)
        else:
            last_value = torch.zeros((), device=self.device)
        self.buffer.compute_gae(
            last_value=last_value, last_done=last_done_per_agent,
            gamma=self.cfg.mappo.gamma, gae_lambda=self.cfg.mappo.gae_lambda,
        )

        mean_components = {k: float(np.mean(v)) for k, v in rollout_components.items()}
        return float(np.mean(rollout_rewards)), mean_components

    # ------------------------------------------------------------------
    def update(self) -> dict[str, float]:
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropies: list[float] = []
        clip_fracs: list[float] = []
        kls: list[float] = []
        clip = self.cfg.mappo.clip_ratio

        for _ in range(self.cfg.mappo.ppo_epochs):
            for mb in self.buffer.iter_minibatches(self.cfg.mappo.minibatch_size):
                new_log_prob, entropy = self.actor.evaluate(
                    mb["obs"], mb["action"], mb["mask"],
                )
                ratio = torch.exp(new_log_prob - mb["old_log_prob"])
                surr1 = ratio * mb["advantage"]
                surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * mb["advantage"]
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_bonus = -self.entropy_coef * entropy.mean()
                actor_loss = policy_loss + entropy_bonus

                new_value = self.critic(mb["global_state"])
                critic_loss = self.cfg.mappo.value_loss_coef * (
                    (new_value - mb["return_"]) ** 2
                ).mean()

                self.actor_optim.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.mappo.max_grad_norm)
                self.actor_optim.step()

                self.critic_optim.zero_grad(set_to_none=True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.mappo.max_grad_norm)
                self.critic_optim.step()

                actor_losses.append(float(actor_loss.detach()))
                critic_losses.append(float(critic_loss.detach()))
                entropies.append(float(entropy.mean().detach()))
                clip_fracs.append(float(((ratio - 1.0).abs() > clip).float().mean().detach()))
                kls.append(float((mb["old_log_prob"] - new_log_prob).mean().detach()))

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "policy_entropy": float(np.mean(entropies)),
            "clip_fraction": float(np.mean(clip_fracs)),
            "approx_kl": float(np.mean(kls)),
        }

    # ------------------------------------------------------------------
    def _maybe_auto_restart(self, mean_entropy: float) -> bool:
        if self.auto_restarted:
            return False
        if self.episodes_done >= self.cfg.mappo.auto_restart_threshold_episode:
            return False
        if mean_entropy >= self.cfg.mappo.auto_restart_entropy_floor:
            return False
        logger.warning(
            "[%s/seed_%d] auto-restart: entropy %.4f < %.4f at episode %d",
            self.cfg.name, self.cfg.seed, mean_entropy,
            self.cfg.mappo.auto_restart_entropy_floor, self.episodes_done,
        )
        self.entropy_coef = self.cfg.mappo.auto_restart_entropy_coef
        self.episodes_done = 0
        self.auto_restarted = True
        self._build_networks()
        return True

    # ------------------------------------------------------------------
    def save_checkpoint(self, tag: str = "current") -> Path:
        ckpt_path = self.run_dir / f"policy_{tag}.pt"
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "episodes_done": self.episodes_done,
            "auto_restarted": self.auto_restarted,
            "entropy_coef": self.entropy_coef,
            "config": self.cfg.model_dump(mode="json"),
        }, ckpt_path)
        return ckpt_path

    def write_manifest(self, extra: dict[str, Any] | None = None) -> Path:
        manifest = {
            "name": self.cfg.name,
            "seed": self.cfg.seed,
            "config": self.cfg.model_dump(mode="json"),
            "device": {
                "backend": self.device_info.backend,
                "reason": self.device_info.reason,
                "torch_version": self.device_info.torch_version,
                "platform": self.device_info.platform,
            },
            "auto_restarted": self.auto_restarted,
            "episodes_done": self.episodes_done,
            "entropy_coef_used": self.entropy_coef,
            "last_metrics": self.last_metrics,
        }
        if extra:
            manifest.update(extra)
        path = self.run_dir / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2))
        return path

    # ------------------------------------------------------------------
    def train(self) -> TrainResult:
        t0 = time.perf_counter()
        target_episodes = self.cfg.mappo.episodes_per_run
        log_every = self.cfg.mappo.log_interval_episodes
        ckpt_every = self.cfg.mappo.checkpoint_every_episodes

        last_log_episode = 0
        last_ckpt_episode = 0
        while self.episodes_done < target_episodes:
            episode_reward, components = self.collect_rollout()
            metrics = self.update()
            self.last_metrics = {
                "episode_reward": episode_reward,
                **{f"reward_component/{k}": v for k, v in components.items()},
                **metrics,
            }

            # Restart check (one-shot before episode 200).
            if self._maybe_auto_restart(metrics["policy_entropy"]):
                last_log_episode = 0
                last_ckpt_episode = 0
                continue

            # TensorBoard.
            self.writer.add_scalar("reward/episode_total", episode_reward, self.episodes_done)
            for k, v in components.items():
                self.writer.add_scalar(f"reward_component/{k}", v, self.episodes_done)
            for k, v in metrics.items():
                self.writer.add_scalar(f"loss/{k}", v, self.episodes_done)
            self.writer.add_scalar("misc/entropy_coef", self.entropy_coef, self.episodes_done)

            # JSONL: one line per PPO update — easier to consume than TB event files.
            row = {
                "episodes_done": self.episodes_done,
                "episode_reward": episode_reward,
                "entropy_coef": self.entropy_coef,
                **{f"reward_component_{k}": v for k, v in components.items()},
                **metrics,
            }
            with self._jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            if self.episodes_done - last_log_episode >= log_every:
                logger.info(
                    "[%s/seed_%d] ep=%d reward=%.2f entropy=%.3f kl=%.4f actor=%.4f critic=%.4f",
                    self.cfg.name, self.cfg.seed, self.episodes_done, episode_reward,
                    metrics["policy_entropy"], metrics["approx_kl"],
                    metrics["actor_loss"], metrics["critic_loss"],
                )
                last_log_episode = self.episodes_done

            if self.episodes_done - last_ckpt_episode >= ckpt_every:
                self.save_checkpoint(f"ep{self.episodes_done:06d}")
                last_ckpt_episode = self.episodes_done

        ckpt = self.save_checkpoint("final")
        elapsed = time.perf_counter() - t0
        self.write_manifest({"elapsed_sec": elapsed})
        self.writer.flush()
        self.writer.close()
        return TrainResult(
            final_episode=self.episodes_done,
            auto_restarted=self.auto_restarted,
            final_metrics=self.last_metrics,
            elapsed_sec=elapsed,
            checkpoint_path=ckpt,
        )
