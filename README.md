# SCION MARL Path-Selection PoC

Master's-thesis proof-of-concept: distributed multi-agent reinforcement learning for SCION
path selection. Trained agents implicitly coordinate so that no central runtime component
(such as the Krüger et al. 2023 Path Oracle) is required.

See `CONTEXT.md` for project context, `BUILDER_BRIEF.md` for the high-level frame,
`METHODOLOGY.md` for the formal phase-by-phase methodology, and `GLOSSARY.md` for
ML/RL term definitions calibrated to a networking audience. Non-obvious design
decisions are recorded in `JUSTIFICATIONS.md`.

## Status

Phase 0 — calibrated synthetic data pipeline. See `PHASE_0_SPEC.md`.

## Layout

```
scion_sim/          Simulator wrapper (fluid model + control plane on top)
calibration/        Calibration scripts, summary-statistics extraction, reports
data/               Empirical summaries (committed) and generated trajectories (gitignored)
docs/               Schema, deployment-target memo, generated figures
tests/              Unit tests
external/           Cloned upstream repos (gitignored, regenerable)
```

## Setup

Requires Python 3.11 and `uv`.

```bash
uv sync
uv run pytest
```
