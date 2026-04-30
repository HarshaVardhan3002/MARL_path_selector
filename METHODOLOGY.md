# SCION MARL — Methodology

## 0. Overview

This project develops a proof-of-concept for distributed reinforcement-learning-based path selection in SCION, the path-aware Internet architecture. The system replaces centralized recommendation services (e.g., the Path Oracle of Krüger et al., 2023) with end-host policies trained jointly under multi-agent reinforcement learning (MARL), such that path-level coordination emerges implicitly from training rather than from runtime communication. The work is structured as five phases: a foundational data-and-environment phase (Phase 0, recently added in response to supervisor feedback regarding the absence of training data and an unsettled deployment target), an environment-construction phase (Phase 1), a training phase (Phase 2), a distillation phase (Phase 3), and a compilation phase (Phase 4) framed as future work. The first three phases form the deliverable scope. The remainder is positioned as the natural follow-on for thesis or later doctoral work.

## 1. Scope and out-of-scope

This project produces a calibrated synthetic SCION-like environment, a multi-agent training pipeline, and a benchmark comparing learned policies against random selection, shortest-path selection, and a Path-Oracle-style greedy follower baseline. It additionally produces a distilled tree-based student model demonstrating that the learned policy can be reduced to a deployable form factor. Real SCIONLab integration, fresh empirical measurement campaigns beyond what is published in Rossi et al. (2025), online learning, adversarial robustness, multi-ISA C compilation, and SmartNIC offload are explicitly out of scope; each is positioned as a logical extension in the conclusions chapter.

## 2. Phase 0 — Calibrated Synthetic Data

The objective of Phase 0 is to produce training trajectories whose statistical properties match real SCIONLab measurements closely enough that policies trained on them behave reasonably under realistic dynamics. The calibration target is the empirical measurement study of Herschbach, Rossi, and Keshvadi (2025), which characterizes path lifetime distributions, control-plane churn, path discrepancy (asymmetric path availability between endpoints), and the multipath performance trade-off in deployed SCIONLab.

Phase 0 selects the Path-Aware-Multipath-Instability fluid-model simulator (Keshvadi, 2025) as the substrate, calibrates its parameters against summary statistics extracted from the public ScionPathML dataset (Rossi, Herschbach, and Keshvadi, 2025), and emits trajectories conforming to a fixed schema documented in the project repository. The phase produces three signed-off artifacts: a deployment-target commitment memo (resolved to constrained end-host / first-hop gateway), a calibrated simulator pipeline with reproducible configuration, and a data schema fixed for downstream phases. The calibration metric is a multi-criterion check: the synthetic distribution of path lifetimes must match the empirical Herschbach distribution within agreed tolerance on Kolmogorov-Smirnov distance; per-path RTT and bandwidth marginals must lie within the empirical interquartile ranges; path-discrepancy rate must approximate the rate reported in Herschbach Section 4.

Verification gate for Phase 0: the synthetic and empirical distributions are visually aligned, all three statistics are within agreed tolerance, and the schema documentation is complete.

## 3. Phase 1 — Environment

The objective of Phase 1 is to wrap the Phase 0 simulator in a PettingZoo-compatible multi-agent reinforcement learning interface. The phase defines the per-agent observation space, the discrete path-selection action space (with action masking for paths that do not currently exist due to churn), and the reward function. The reward function decomposes into three terms: a throughput term, a congestion-contribution term penalizing the agent for choices that overload links proportionally to the agent's contribution, and a path-diversity bonus discouraging trivial single-path equilibria. Each term is implemented as a separately callable function and logged as a separate scalar during training so that ablation is straightforward.

Phase 1 also establishes the unit test suite. Hand-constructed scenarios, including the lossless-operating-point scenario from Keshvadi (2510.26060, 2025), serve as test cases: in scenario X with action Y, the reward must produce result Z. Tests fail loudly on regression, and any change to the reward function requires that the tests pass before integration.

Verification gate for Phase 1: `env.reset()` and `env.step()` produce schema-conforming trajectories, all reward unit tests pass, and the random-policy baseline yields expected (low) reward over a sample episode.

## 4. Phase 2 — Multi-Agent Training

The objective of Phase 2 is to train a policy under Multi-Agent Proximal Policy Optimization (MAPPO) using the Centralized Training, Decentralized Execution (CTDE) paradigm. During training, a centralized critic observes the global state — including all agents' observations and actions — and provides advantage estimates to a single shared policy. Parameter sharing is justified by the deployment story: in a real deployment, every end-host receives the same compiled artifact. At test time, the critic is discarded; only the policy ships.

Training employs CleanRL-style MAPPO for transparency and ease of inspection. Hyperparameters are fixed conservatively at MAPPO defaults (γ = 0.99, λ = 0.95, clip ratio = 0.2, entropy coefficient at standard initial value with optional schedule). Tuning, if required, is constrained to entropy coefficient and reward-term weights; learning-rate and clip-ratio adjustments are documented as exceptions.

Phase 2 evaluates the trained policy against three baselines: uniform-random path selection, shortest-path selection (using static SCION metadata), and a Path-Oracle-style greedy follower that selects the highest-EMA-scored path each interval. The headline metric is total throughput at the system level under a representative load, with secondary metrics including per-agent fairness (Jain index), oscillation amplitude (variance of per-path utilization over time), and median per-agent latency.

Verification gate for Phase 2: the trained policy achieves higher system throughput than all three baselines on at least three independent training seeds, and the variance-of-utilization metric for the trained policy is lower than that of the Path-Oracle baseline (i.e., less oscillation).

## 5. Phase 3 — Distillation

Phase 3 distills the trained MARL policy into an XGBoost ensemble using behavioral cloning over rollouts. The teacher policy generates a dataset of (observation, action distribution) pairs across a wide sweep of states, including states drawn from the churn-and-asymmetry regimes calibrated in Phase 0. The student is trained to match the teacher's action distribution via cross-entropy loss on softened targets. The student is evaluated under the same benchmark suite as Phase 2, and the performance gap relative to the teacher is reported.

A model-size sweep selects a single Pareto-frontier point trading off student size against benchmark performance loss. The selected student model and its size, training data, and accuracy gap are documented.

Verification gate for Phase 3: the student matches the teacher's chosen action on a held-out evaluation set within an agreed tolerance (target: ≥90% top-1 agreement) and the student's benchmark performance lies within the agreed degradation budget relative to the teacher (target: ≤10% relative throughput loss).

## 6. Phase 4 — Compilation (Future Work)

Phase 4, framed as future work, would compile the selected XGBoost student via Treelite into a standalone C header, target a single ISA (ARMv7 or similar, fixed at the time of execution), and verify bit-exact agreement between the C output and the original XGBoost predictions on a regression test set. Phase 4 is not part of the deliverable scope; the project's conclusions chapter discusses Phase 4 as the natural extension and the constraints it would impose on training (quantization-aware distillation, fixed-point arithmetic).

## 7. Reproducibility

Each output artifact — model checkpoints, datasets, figures, evaluation results — carries a manifest recording the git commit SHA, the configuration hash, the dataset hash, and the simulator version. Random seeds are fixed in configuration and recorded in manifests. The repository contains a reproducibility script that, given a manifest, regenerates the corresponding artifact. The `JUSTIFICATIONS.md` document at the repository root records the rationale, alternatives considered, and supporting literature for each non-obvious design decision; this document is the primary supervisor-facing artifact bridging code-level decisions to thesis-level argument.

## 8. Risks and mitigations

The primary risks are calibration drift between synthetic and empirical distributions, reward hacking during MARL training, and a teacher-student gap large enough to invalidate the deployment story. Each is mitigated by an explicit verification gate and ablation logs, and any breach of the gate requires explicit replanning rather than silent adjustment. A secondary risk is scope creep into Phase 4 or real SCIONLab deployment; the project guards against this by maintaining a written commitment to scope rings (commit, aim, stretch) and reviewing progress against those rings at each phase boundary.

## 9. Timeline

Calendar duration is approximately 12-16 weeks, with active session hours estimated at 200-240 productive hours allocated across phases roughly as: Phase 0 at 30-40h, Phase 1 at 25-35h, Phase 2 at 60-80h (training waits absorb wall-clock time but not session time), Phase 3 at 30-40h. Phase 4 is not budgeted within scope.
