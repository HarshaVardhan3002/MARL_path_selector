"""Per-slot EMA of (throughput proxy − loss penalty) — Path-Oracle-style greedy.

The policy maintains an exponential moving average per top-K slot and picks
the highest-EMA valid slot at each step. The reward signal is read from the
observation vector itself (F3 normalized BW, F4 loss/100), since the policy
interface is `(observation, action_mask) -> int` per spec §2.5.

Tracking per *slot* rather than per *fingerprint* trades correctness for
interface simplicity: when a path expires and a new one moves into its
slot, the EMA carries over to the new occupant. The bias cancels in
aggregate because slot ordering is deterministic in RTT (J-021).
"""
from __future__ import annotations

import numpy as np

from scion_sim.env.observation_builder import PER_PATH_FEATURES


class OracleGreedyPolicy:
    name = "oracle_greedy"

    def __init__(self, top_k: int = 4, ema_window: int = 20, loss_penalty: float = 1.0) -> None:
        self.top_k = top_k
        self.alpha = 1.0 / float(max(1, ema_window))
        self.loss_penalty = float(loss_penalty)
        self._ema = np.zeros(top_k, dtype=np.float64)
        self._initialized = np.zeros(top_k, dtype=bool)

    def reset(self) -> None:
        self._ema[:] = 0.0
        self._initialized[:] = False

    def __call__(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        # Update EMA for every valid slot from this observation.
        for slot in range(self.top_k):
            if not bool(action_mask[slot]):
                continue
            base = slot * PER_PATH_FEATURES
            bw = float(observation[base + 2])      # F3: normalized BW
            loss = float(observation[base + 3])    # F4: loss/100
            proxy = bw - self.loss_penalty * loss
            if not self._initialized[slot]:
                self._ema[slot] = proxy
                self._initialized[slot] = True
            else:
                self._ema[slot] = self.alpha * proxy + (1.0 - self.alpha) * self._ema[slot]

        # Greedy pick over valid slots.
        valid = np.flatnonzero(action_mask)
        if valid.size == 0:
            return 0
        ema_valid = self._ema[valid]
        return int(valid[int(np.argmax(ema_valid))])
