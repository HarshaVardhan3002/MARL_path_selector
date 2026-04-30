# SCION MARL Project — Builder Brief

> Read this first. Everything else is reference.

## What you're building, in one paragraph

A proof-of-concept where N simulated SCION end-hosts each run a small policy that picks paths. The policies were trained together using multi-agent reinforcement learning so they implicitly coordinate without talking to each other at runtime. You'll show this beats both random path selection and a "follow-the-best" Path-Oracle-style baseline. Then you'll distill one trained policy into a tiny tree-based model to demonstrate the "ships in firmware" deployment story, without actually deploying anything. That's the PoC. Real SCION integration, real hardware, real-world rollout — all explicitly out of scope.

## Why this is interesting

The Path Oracle (your own university's prior work) is a centralized donation-aggregator that recommends paths. It works in isolation but breaks under load because everyone follows the same recommendation, like Waze causing traffic on the "fast" route. Your system replaces it with distributed agents who each learned, during training, to expect that other agents exist and not all chase the same path. The intelligence is baked into each model's weights, not into a central server.

## Why your supervisor cares about "dynamic"

SCION paths aren't stable. They expire on the order of hours, the path set itself changes, and A→B doesn't see the same paths as B→A. If you train on a static snapshot, the model fails the moment it meets reality. Phase 0's whole job is producing training data that has these dynamics baked in.

## The three rings (commit / aim / stretch)

```
COMMIT (must finish):    Phase 0  →  Phase 1  →  Phase 2-lite (one MARL config)
AIM (do if Phase 2 lands):                       Phase 3 (distillation)
STRETCH (future work):                                    Phase 4 (compile to C)
```

If you walk in with the commit ring done cleanly and a good story, that's a passing thesis. If you push for stretch and only half-deliver commit, you're in trouble. Set the expectation with your supervisor *now* that this is the shape.

## Phase-by-phase, in plain words

**Phase 0 — Reality.** Build a calibrated simulator. This means: take an existing SCION simulator (Path-Aware-Multipath-Instability fluid model), tune its parameters until the statistics it produces — path lifetimes, churn rates, RTT distributions — roughly match real SCIONLab measurements (Herschbach et al.). Output: a simulator that produces ML-friendly trajectories. *This is what the next chat starts on.*

**Phase 1 — Environment.** Wrap the simulator in a PettingZoo interface so RL algorithms can talk to it. Define the state vector each agent observes, the action space (which path to pick), and the reward function. Output: `env.reset()` and `env.step()` work, with unit tests on the reward.

**Phase 2 — Train.** Run MAPPO (a multi-agent RL algorithm) using CTDE (centralized critic during training, decentralized policies at deploy). Get policies that beat the baselines. Output: trained policy weights + benchmark results.

**Phase 3 — Distill.** Use the trained policy to generate a labeled dataset (states → actions). Train an XGBoost ensemble on that dataset. Show the student model matches the teacher's behavior closely. Output: tiny model + accuracy comparison.

**Phase 4 — Compile (stretch).** Treelite converts the XGBoost model into a standalone C header. Don't bother unless Phases 0-3 are solid.

## What Claude (the AI) does vs what you do

**Claude Code does:** all the code, all the calibration glue, all the experimental harness, the methodology writeup, the results section, figures, debugging when given logs. You do not need to read the simulator's source line-by-line.

**You do:** read the supervisor-facing papers (Herschbach measurement, Keshvadi dynamic-equilibria, Path Oracle), confirm design decisions when Claude Code surfaces them, run the actual training on your hardware, watch the dashboards, defend the work to your supervisor.

**Shared:** the reward function story. This is the one piece where networking and ML meet, and your supervisor will probe it. You should be able to explain in networking terms (load balancing, equilibrium, stability) why each reward term is shaped the way it is.

## The one habit that protects you

Every non-obvious design decision goes into a `JUSTIFICATIONS.md` file at the repo root. "We chose lognormal path lifetimes because Herschbach Figure 5 showed empirical distribution had heavy right tail." "We weighted congestion 3x throughput because lower weights caused agents to converge on a single path during pilot runs." When your supervisor asks why, you read the entry, internalize it, answer in your own words. This is your safety net against the "Claude did it" failure mode.

## The two failure modes to watch for in yourself

1. **Snap-to-fiction.** Your brain fills in plausible context when real context is thin. It feels identical to snap-to-reality from the inside. The defense: when you have a strong intuition, check it against the source paper before committing. The Path Oracle "smarter centralized version" was an example of this; you didn't notice from the inside.

2. **Delegating the design.** The more confident you feel about *what* you're building, the more tempting it is to hand the *how* to Claude Code. The how is partly your thesis. Stay in the loop on reward shape, state design, and baseline choice. Those are the questions your supervisor will probe.

## What's already settled

- Deployment target: constrained end-host / first-hop gateway (not ISP).
- Simulator: Path-Aware-Multipath-Instability fluid model. mpcc-dynamics is future work.
- Algorithm: MAPPO with CTDE.
- Distillation: XGBoost.
- Stack: Python 3.11, uv, PettingZoo, CleanRL-style MAPPO implementation, XGBoost, TensorBoard.

## What's not yet settled

- Number of agents N for the headline experiment. Probably 20-40.
- Exact reward weights. Will be tuned during Phase 2.
- Whether to use one shared policy across agents or N separate policies. Default: shared (parameter sharing). Defensible because it matches the "deploy the same compiled artifact to every node" deployment story.
- Calibration tolerance — how close does synthetic data need to match real before we call it good. Will set in Phase 0.

## The next thing you do

Open Claude Code. Feed it `CONTEXT.md` as the project context. Then feed it `PHASE_0_SPEC.md` as the first task. Read `GLOSSARY.md` while it's working so the next time it says "PPO clip ratio" you know what it means.
