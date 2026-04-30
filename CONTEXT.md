# Project Context — SCION MARL Path Selection PoC

You are working on a Master's thesis project (12 academic credits, ~200-240 productive hours total) that builds a proof-of-concept multi-agent reinforcement learning system for SCION path selection. This document is your source of truth for project context, decisions already made, and constraints. Treat anything here as authoritative; if the user contradicts something, ask before changing course.

## The user

- Master's student, second/third day deep on SCION but solid AI/DS bachelor's background.
- ADHD; prefers high-level architectural reasoning, delegates implementation.
- Has a "Claude 20x" plan with excess usage enabled.
- Supervisor is a SCION/networking expert, not an ML expert. Defenses must be in networking terms.
- Will run training on a Strix Halo box (128GB unified memory, no GPU cluster). CPU-only training is fine for PoC.

## The problem

SCION is a path-aware Internet architecture where end-hosts pick routes from a set the network exposes. Naive path selection causes the "Waze effect" — every host picks the same best-looking path, congesting it. The university's prior work (Path Oracle, Krüger et al. 2023) is a centralized donation-aggregator that recommends paths but suffers oscillation under load.

This project replaces the centralized recommender with a distributed MARL system where each end-host runs a small policy that was trained, alongside other agents, to implicitly account for what others are likely doing. No central component at runtime. Coordination emerges from training.

## Decisions already made (do not relitigate without asking)

| Decision | Choice | Rationale |
|---|---|---|
| Deployment target | Constrained end-host / first-hop gateway | SCION host-authority model; ISP-side violates threat model |
| Simulator base | Path-Aware-Multipath-Instability (fluid model) | Cleaner for MARL than mpcc-dynamics; MPQUIC coupling is future work |
| Algorithm | MAPPO with CTDE | CTDE-native, well-documented, less finicky than MADDPG |
| Distillation target | XGBoost ensemble | Compiles cleanly via Treelite; reduces to integer comparisons |
| Parameter sharing | Single shared policy across agents | Matches deployment story (same artifact on every node) |
| Stack | Python 3.11, uv, PettingZoo, CleanRL-style MAPPO, XGBoost, TensorBoard | All defensible, all reasonably standard |
| Reward target | "Lossless operating point" from Keshvadi 2510.26060 | Formal target, citable, not hand-waved |

## Scope rings

- **Commit ring (must deliver):** Phase 0 (calibrated simulator), Phase 1 (PettingZoo env), Phase 2 lite (one MARL config trained, beats baselines).
- **Aim ring (deliver if commit lands cleanly):** Phase 3 (XGBoost distillation, accuracy comparison vs MARL teacher).
- **Stretch ring (frame as future work):** Phase 4 (Treelite compile to C), real SCION integration, hardware deployment, multi-ISA support.

Real SCIONLab deployment, fresh measurement campaigns, online learning, adversarial robustness, and SmartNIC offload are explicitly out of scope.

## Reference papers and code (read before deep work)

- **Herschbach et al., arXiv 2509.04695** — empirical SCIONLab measurement study. Source of truth for what "realistic dynamics" means. Path lifetimes ~8.6h, path discrepancy (asymmetric A→B vs B→A path sets), multipath performance trade-off.
- **Keshvadi, arXiv 2510.26060** — formal analysis of dynamic equilibria in joint path selection + congestion control. Defines the "lossless operating point" we target with reward shaping.
- **Krüger et al. 2023 (Path Oracle, IFIP CNSM)** — the centralized baseline. Donation-based aggregator with EMA-scored links. Their Figure 3 shows the oscillation our system fixes.
- **Baumeister & Keshvadi, arXiv 2509.05938** — axiomatic analysis of multipath path-selection strategies. Quantifies >18,000% packet-loss increase from greedy selection at scale.
- **Rossi/Herschbach/Keshvadi, arXiv 2509.07154 (ScionPathML)** — the data schema we're matching against. RTT, bandwidth, jitter, loss, per-hop RTT, path fingerprints, ISD-AS tags, JSON/CSV format.
- **Path-Aware-Multipath-Instability** repo (github.com/keshvadi/Path-Aware-Multipath-Instability) — fluid-model simulator. The base we calibrate.
- **mpcc-dynamics** repo (github.com/keshvadi/mpcc-dynamics) — future-work, MPQUIC-coupled discrete-event simulator.

## Repo conventions

- All non-obvious design decisions get an entry in `JUSTIFICATIONS.md` at repo root. Format: decision, alternatives considered, networking-literature backing, ML-literature backing, who decided. The user reads this before supervisor meetings.
- Reward function terms live in separate functions, never one summed scalar in code. Always log the un-summed components so we can ablate.
- Unit tests on reward function and env transitions are non-optional. Hand-constructed scenarios from the dynamic-equilibria paper become test cases.
- Per-agent trajectory logging: state, action, reward components separately, global state, path-churn events. Granular enough to debug reward hacking later.
- Manifests on every output artifact (model checkpoint, dataset, figure): git SHA, config hash, dataset hash, env version.

## Failure modes to flag, not silently fix

If during a build session you notice:
- Calibration drift (synthetic data statistics diverging from real ScionPathML)
- Reward hacking (agents exploiting a shaping term in unintended ways)
- Action space degenerating (e.g., the agent only ever picks one path)
- Critic loss not converging
- Performance gap between teacher and distilled student exceeding ~10%

Stop, write up what you observed in plain language, propose a fix, and surface it to the user before changing the design. The user owns design decisions.

## Communication style

- Networking justifications first, ML justifications second. The supervisor is a networking expert.
- Plain-language summaries before code blocks.
- Surface design questions explicitly ("should congestion weight be 2x or 5x throughput?") rather than picking silently.
- When the user asks for code, write the code. When the user asks for analysis, write analysis. Don't proactively over-explain.

## What's currently next

Phase 0 — synthetic data generation calibrated against ScionPathML real measurements. See `PHASE_0_SPEC.md` for the detailed task description.
