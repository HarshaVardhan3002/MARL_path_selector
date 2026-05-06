"""Always pick slot 0 — the lowest-RTT path under the locked top-K ordering."""
from __future__ import annotations

import numpy as np


class ShortestPathPolicy:
    name = "shortest_path"

    def reset(self) -> None:
        return None

    def __call__(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        if bool(action_mask[0]):
            return 0
        # Defensive: if slot 0 is somehow masked, fall back to first valid.
        valid = np.flatnonzero(action_mask)
        return int(valid[0]) if valid.size > 0 else 0
