"""Uniform random policy over valid (mask=True) action slots."""
from __future__ import annotations

import numpy as np


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        # No internal state — RNG persists across episodes for variety.
        return None

    def __call__(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        valid = np.flatnonzero(action_mask)
        if valid.size == 0:
            return 0
        return int(self._rng.choice(valid))
