# Phase 0 — Calibrated Synthetic Data Pipeline

> Feed this to Claude Code after `CONTEXT.md`. This is the first task.

## Objective

Produce training trajectories whose statistical properties match real SCIONLab measurements (Herschbach et al., 2509.04695) closely enough that downstream MARL training generalizes to realistic dynamics. Calibrate against the published ScionPathML dataset.

## Deliverables

1. `scion_sim/` Python package containing:
   - A wrapper around the Path-Aware-Multipath-Instability fluid-model simulator (clone repo, depend on it, do not fork).
   - A configuration system (YAML or pydantic) exposing the calibratable parameters: number of ASes, paths-per-AS-pair, churn rate, path-lifetime distribution, RTT base + noise, bandwidth base + noise, asymmetry rate.
   - A trajectory generator that runs the simulator and emits trajectories in a fixed schema (defined below).

2. `data/scionpathml_summary.json` — extracted summary statistics from the public ScionPathML dataset:
   - Path-lifetime empirical CDF
   - RTT marginal distribution per AS pair
   - Bandwidth marginal distribution per AS pair
   - Path-discrepancy rate
   - Path-churn events per hour

3. `calibration/` directory containing:
   - `calibrate.py` — script that runs the simulator under a given config, computes the same statistics, and reports a multi-criterion gap vs the empirical summary.
   - `calibration_report.md` — auto-generated report showing visual overlays of synthetic vs empirical distributions and the numerical gaps.

4. `docs/SCHEMA.md` — fixed trajectory schema documentation. Once locked, downstream phases depend on this not changing.

5. `tests/test_simulator.py` — minimum test set:
   - Simulator runs without crashing for a small config.
   - Output trajectories conform to schema.
   - Path lifetimes are non-zero and finite.
   - Path discrepancy occurs at non-zero rate when configured to.

6. Entry in `JUSTIFICATIONS.md` for: simulator choice, calibration metric choice, schema design choices.

## Trajectory schema (proposed — finalize in implementation)

Each trajectory is a sequence of timesteps. Each timestep contains:

- `t`: timestep index (integer)
- `wall_time`: simulated wall-clock time (seconds since episode start)
- `agents`: dict mapping agent_id to per-agent observation:
  - `available_paths`: list of currently-valid path fingerprints
  - `path_metrics`: dict mapping fingerprint to {rtt, bandwidth, loss, jitter, age_seconds}
  - `recent_history`: rolling window of last K observations
- `global_state`: (used by critic only, not policy):
  - `link_utilizations`: dict mapping link_id to current utilization
  - `agent_path_assignments`: dict mapping agent_id to current path fingerprint
  - `total_throughput`: aggregate
- `events`: list of path-level events at this timestep:
  - `{type: "path_appeared", agent_id, fingerprint, t}`
  - `{type: "path_expired", agent_id, fingerprint, t}`
  - `{type: "path_metrics_updated", agent_id, fingerprint, t}`

## Calibration metric

A composite score across three dimensions, each with a tolerance:

1. **Path-lifetime distribution gap** — Kolmogorov-Smirnov distance between synthetic and empirical CDFs. Tolerance: KS ≤ 0.15 (loose, because empirical sample is small and noisy).
2. **RTT/bandwidth marginal gap** — synthetic median per AS pair must lie within empirical IQR per AS pair, and synthetic IQR must be within ±50% of empirical IQR.
3. **Path-discrepancy rate gap** — synthetic discrepancy rate must lie within ±25% of empirical discrepancy rate.

A run is "calibrated" when all three are within tolerance. If we cannot meet tolerance with any setting of the exposed parameters, that finding is itself a deliverable — flag it, document what we learned about the simulator's expressiveness, and we may need to extend the simulator (this should surface as a design question, not a silent code change).

## Implementation order

1. Clone the two simulator repos. Read their READMEs and one example notebook each. Identify the function or entry point that produces a stream of path observations.
2. Pull the ScionPathML dataset from its GitHub. Inspect the JSON/CSV schema. Write the summary-statistics extractor and produce `scionpathml_summary.json`.
3. Write the simulator wrapper. Initial config can be arbitrary defaults. Confirm it produces *some* trajectory in the proposed schema.
4. Write the calibration script. Run it once with default config, report the gap. Almost certainly will be large.
5. Iterate on simulator parameters (manually first, no fancy optimization) until gap is within tolerance. Document what each parameter actually controls in `JUSTIFICATIONS.md`.
6. Write the unit tests. Add a CI-like check that runs them.
7. Lock the schema. Generate one full-scale trajectory dataset (e.g., 1000 episodes) and save it as the canonical Phase 0 output for Phase 1 to consume.

## What to surface to the user (don't decide silently)

- If the empirical summary statistics extracted from ScionPathML look implausible (e.g., empirical sample is too small to compute an IQR), surface it. We may need to widen the tolerance or accept calibration only on a subset of metrics.
- If the simulator's exposed parameters cannot produce path discrepancy at all, surface it. Discrepancy is critical; we may need to add it as a post-hoc transformation rather than a simulator parameter.
- If calibration converges trivially on one parameter setting, surface it — it might mean the simulator is under-specified for our purposes and we should sanity check with a second simulator (mpcc-dynamics) before relying on it.
- If the simulator's runtime to generate 1000 episodes is too slow to fit Phase 0's time budget, surface it — we may need to parallelize, reduce episode length, or accept fewer episodes.

## Definition of done for Phase 0

- All six deliverables exist in the repo.
- Calibration report shows all three metrics within tolerance.
- Tests pass.
- One full-scale trajectory dataset (>= 1000 episodes) exists at `data/phase0_trajectories.parquet` (or similar).
- `JUSTIFICATIONS.md` has entries for every non-default decision.
- The user has read the calibration report and signed off.

## What Phase 1 will need from Phase 0

- The fixed schema (`docs/SCHEMA.md`).
- A function `make_env(config)` that returns a calibrated simulator instance ready to be wrapped in PettingZoo.
- The canonical trajectory dataset (for offline analysis, baseline computations, and reward-function unit tests).

This is the contract. Phase 0 ends when Phase 1 can start without going back.
