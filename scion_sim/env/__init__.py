"""Phase 1 — PettingZoo MARL environment, observation builder, reward, action handler.

Public API:

    from scion_sim.env import make_env, ScionMARLEnv, Phase1Config
"""
from __future__ import annotations

from scion_sim.env.config import Phase1Config

__all__ = ["Phase1Config", "ScionMARLEnv", "make_env"]


def __getattr__(name: str):
    """Lazy re-export — env class imports pettingzoo + gymnasium, kept off
    the package import path until first use to keep test discovery cheap."""
    if name in ("ScionMARLEnv", "make_env"):
        from scion_sim.env.scion_marl_env import ScionMARLEnv, make_env
        return {"ScionMARLEnv": ScionMARLEnv, "make_env": make_env}[name]
    raise AttributeError(name)
