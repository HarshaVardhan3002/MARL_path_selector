"""Phase 2 — MAPPO training, sweep, evaluation, distillation-ready exports.

Public API:

    from scion_marl import select_device, ActorNetwork, CriticNetwork
    from scion_marl import RolloutBuffer, MappoTrainer, Phase2Config
"""
from __future__ import annotations

from scion_marl.config import Phase2Config
from scion_marl.device import DeviceInfo, select_device

__all__ = [
    "DeviceInfo",
    "Phase2Config",
    "select_device",
    "ActorNetwork",
    "CriticNetwork",
    "RolloutBuffer",
    "MappoTrainer",
]


def __getattr__(name: str):
    """Lazy re-exports for heavy torch-dependent modules."""
    if name == "ActorNetwork":
        from scion_marl.agent import ActorNetwork
        return ActorNetwork
    if name == "CriticNetwork":
        from scion_marl.critic import CriticNetwork
        return CriticNetwork
    if name == "RolloutBuffer":
        from scion_marl.buffer import RolloutBuffer
        return RolloutBuffer
    if name == "MappoTrainer":
        from scion_marl.mappo import MappoTrainer
        return MappoTrainer
    raise AttributeError(name)
