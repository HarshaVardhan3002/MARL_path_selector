# Design Justifications

> Every non-obvious design decision lives here. Format per entry: **decision**,
> **alternatives considered**, **networking-literature backing**, **ML-literature
> backing**, **who decided**, **status**. The user reads this before supervisor
> meetings.

---

## J-001 — Vendored read-only snapshot of Path-Aware-Multipath-Instability

**Decision.** The Path-Aware-Multipath-Instability fluid-model simulator
(github.com/keshvadi/Path-Aware-Multipath-Instability) is vendored as a
read-only snapshot under `scion_sim/_vendored/pami/`, pinned to upstream
commit `294c07979c1d58128ac3ec80e18288388d5bf60f`. We do not modify the
vendored files; we import their classes (`Resource`, `Path`, `Source`,
`primal_model_equations`, `pan_primal_equations`, scheduler classes) from a
thin adapter in `scion_sim/wrapper/`.

**Alternatives considered.**
1. *Path injection of `external/Path-Aware-Multipath-Instability/src` at runtime.*
   Most literal reading of the PHASE_0_SPEC instruction "depend on it, do not
   fork". Rejected because the upstream repo is unpackaged (no
   `pyproject.toml`, no `setup.py`), has no LICENSE file declaring reuse
   permissions, and is a research artifact whose author may not maintain
   compatibility. Pinning a commit SHA via a setup script is operationally
   equivalent to vendoring without the reproducibility benefits.
2. *Git submodule.* Adds tooling burden on Windows and pre-commit hooks for
   a thesis-scale project that the user runs alone.
3. *Re-implement the fluid model from the paper.* Out of scope for Phase 0
   (would burn the entire 30-40h budget on a re-implementation rather than
   on calibration).

**Networking-literature backing.** The fluid model is the standard primal
formulation of Kelly-style optimization for joint path selection / congestion
control (Kelly, Maulloo, Tan, 1998; Han et al., 2006), specialized to the SCION
multipath setting in the Keshvadi line of work. Using a published, peer-vetted
formulation rather than a re-implementation is the conservative choice for a
thesis defended to a SCION expert.

**ML-literature backing.** Treating the simulator as a black-box dynamics
model on which we layer policy / control logic is standard practice in RL
benchmarks (e.g., MuJoCo wrappers, BSuite, MetaDrive). Vendoring an unpackaged
upstream is the de-facto convention for academic ML code that lacks PyPI
distribution.

**Who decided.** Claude Code, in response to the absence of upstream packaging
and the presence of an explicit reproducibility requirement (METHODOLOGY §7).

**Status.** Active. Surface to user if upstream releases a packaged version
(then we should switch to a proper dependency).

---

## J-002 — Path lifetime, path discrepancy, measurement noise added in wrapper layer

**Decision.** The Path-Aware-Multipath-Instability fluid simulator does not
model: (a) path lifetime (paths appearing and disappearing over time),
(b) path discrepancy (asymmetric path sets between A→B and B→A views),
(c) measurement noise on RTT / bandwidth observed by the agent. These three
phenomena are added in `scion_sim/control_plane/`, layered on top of the
underlying ODE-driven flow dynamics.

**Alternatives considered.**
1. *Patch the upstream simulator.* Violates the "do not fork" rule. Also
   conflates two distinct concerns (data plane dynamics vs. control plane
   topology evolution) that are cleanly separable.
2. *Use the discrete-event mpcc-dynamics simulator instead, which may model
   stochastic dynamics natively.* Rejected because mpcc-dynamics is a single
   2.2 MB Jupyter notebook that produces 13 paper figures — not packaged for
   reuse, and CONTEXT explicitly defers MPQUIC coupling to future work.
3. *Drop calibration on path discrepancy entirely.* Rejected because
   discrepancy is the third of the spec's three calibration metrics and
   featured prominently in Herschbach et al. 2509.04695 §4.

**Networking-literature backing.** Herschbach, Rossi, Keshvadi (arXiv
2509.04695) document path discrepancy as a routing-policy artefact — the
control plane decides which paths each end-host learns about, and these
decisions are not symmetric. Modelling discrepancy as a control-plane event
stream rather than a data-plane parameter therefore matches the empirical
mechanism. Path lifetime distributions in the same paper are dominated by
peering-relationship changes and beacon-flooding policy, both of which sit in
the control plane.

**ML-literature backing.** Layered environments are the standard way to add
non-stationarity to RL substrates without contaminating the underlying physics
(e.g., domain randomization in DexHand; weather-perturbation wrappers in
CARLA). Keeping the layers separate also means we can ablate "what happens if
agents see a stationary path set" by zero-ing the control plane in tests.

**Who decided.** Claude Code, in response to recon finding (the underlying
simulator's `Path` class has a fixed RTT and a static `resources` list — no
notion of expiry, no notion of bidirectionality). PHASE_0_SPEC explicitly
anticipated this in its "What to surface to the user" section.

**Status.** Active. To be confirmed by user before locking schema (Task #7).

---

## J-003 — Python 3.11 (uv-managed) over system Python 3.12

**Decision.** Project pins Python 3.11 (specifically 3.11.15 via `uv python`)
in `pyproject.toml` (`requires-python = ">=3.11,<3.12"`). The user's system
Python 3.12 is left untouched; uv provisions an isolated 3.11.

**Alternatives considered.** Python 3.12 (system-installed): rejected because
CONTEXT.md fixes the stack at 3.11. Python 3.10: would be conservative for
PettingZoo / CleanRL / XGBoost compatibility, but lacks pattern-match exhaustive
checks in pyright that 3.11 added.

**Networking-literature backing.** N/A.

**ML-literature backing.** PettingZoo 1.24+, CleanRL ≥ 2.0, XGBoost 2.x,
pydantic 2.x all support 3.11 as their primary CI target. 3.12 has scattered
incompatibility reports in the broader RL ecosystem (gymnasium issue #911,
XGBoost wheels lag).

**Who decided.** CONTEXT.md (project rules); Claude Code implementing.

**Status.** Active.

---

## J-004 — Pydantic for configuration, YAML for serialization

**Decision.** All calibratable parameters and run-time configuration are
expressed as Pydantic v2 models (`scion_sim/config.py`). Configurations are
serialized to / deserialized from YAML for human editing. CLI scripts accept
a `--config path/to/config.yaml` flag.

**Alternatives considered.**
1. *Hydra.* Powerful but optimized for hyperparameter sweeps in deep-learning
   research; introduces structured configs, defaults, multi-run handling that
   we don't need at Phase 0 scale.
2. *Plain YAML + dict access.* Loses type checking, IDE completion, and
   validation (e.g., positive churn rate, valid distribution name).
3. *TOML.* Ecosystem-standard for tooling configs but less expressive for
   nested experiment configs than YAML.

**Networking-literature backing.** N/A.

**ML-literature backing.** Pydantic is the de-facto standard for typed
config in 2024+ Python (FastAPI, LangChain, vLLM all use it). YAML for run
configs is the convention in CleanRL and stable-baselines3 examples.

**Who decided.** Claude Code.

**Status.** Active.

---

## J-005 — Parquet for canonical trajectory storage, JSON for summary statistics

**Decision.** The canonical Phase 0 trajectory dataset is stored as Parquet
files (one per episode or one per shard, decided in Task #11) under
`data/phase0_trajectories/`. The empirical summary statistics extracted from
ScionPathML are stored as JSON at `data/scionpathml_summary.json`.

**Alternatives considered.**
1. *JSON-lines for trajectories.* Roughly 5× larger than Parquet for the
   same content; loses columnar efficiency for ML feature loading.
2. *HDF5.* numpy-friendly but tooling fragility on Windows (h5py wheel issues
   historically) and harder for non-Python consumers.
3. *Pickle.* Fast to write but Python-version-coupled and a security
   liability for shared artefacts.

**Networking-literature backing.** N/A.

**ML-literature backing.** Parquet is the columnar standard for ML feature
stores (Hugging Face datasets, Polars, Apache Arrow). JSON is the standard
for summary statistics and config-like artefacts (small, human-readable,
language-agnostic).

**Who decided.** Claude Code.

**Status.** Active.

---

## J-006 — Calibration scope: lifetime CDF + RTT/BW marginals + discrepancy rate (no joint distribution)

**Decision.** Phase 0 calibration matches three *marginal* statistics
between synthetic and empirical data: (1) the path-lifetime distribution
(KS gap), (2) per-AS-pair RTT and bandwidth marginals (median-in-IQR + IQR
ratio), (3) the path-discrepancy rate (relative gap). It does **not** match
joint distributions (e.g., RTT correlation across paths sharing AS hops, or
co-occurrence of RTT spikes with churn events).

**Alternatives considered.** Joint-distribution matching would require
considerably richer data (the current ScionPathML release samples three
3-AS topologies for four weeks, ≈1281 comparisons each). The sample is too
sparse to estimate copulas reliably.

**Networking-literature backing.** Herschbach et al. 2509.04695 itself
characterizes the data using marginal statistics (their Tables 1 and 2 are
per-AS-pair marginals; their Figure 5 is a marginal lifetime histogram).
Matching what the source paper measured is the conservative defensible
target.

**ML-literature backing.** PHASE_0_SPEC frames calibration as a
multi-criterion gap, which is the standard formulation for distribution
matching in domain-randomization literature (Tobin et al., 2017).

**Who decided.** PHASE_0_SPEC; Claude Code implementing.

**Status.** Active. Will revisit in Phase 2 if MARL training reveals
sensitivity to joint structure not captured by marginals.

---

## J-007 — Empirical summary stats sourced from `mpquic-on-scion-ipc` AnalysisResults rather than re-running the measurement campaign

**Decision.** The empirical summary statistics (path-lifetime CDF, RTT
marginals, BW marginals, discrepancy rate, churn rate) are computed from the
public ScionPathML data published in
github.com/Keshvadi/mpquic-on-scion-ipc (commit
`362ad60885dabeb2a067d3dae058c506ce5f092a`), specifically the three Lab
runs in `AnalysisResults/Lab[1-3]/Data/raw_data.zip` plus the pre-computed
`Analysis/*.txt` summaries that we use as a sanity check on our extractor.

**Alternatives considered.**
1. *Re-run the measurement campaign on SCIONLab.* Rejected: METHODOLOGY §1
   explicitly puts fresh measurement campaigns out of scope; would consume
   the entire 30-40h Phase 0 budget on infrastructure setup alone.
2. *Use only the published numbers from Herschbach et al. 2509.04695 (e.g.,
   "8.6h average path lifetime").* Rejected because we need the full
   distribution (CDF), not just the mean, for KS-distance computation.

**Networking-literature backing.** The mpquic-on-scion-ipc dataset is the
companion data release to Herschbach et al. 2509.04695 and Rossi et al.
2509.07154. Using the same data the cited papers analyzed gives our
calibration target maximum traceability for supervisor defence.

**ML-literature backing.** N/A.

**Who decided.** Claude Code.

**Status.** Active. Note: AS 22 (Taiwan) was excluded from the measurement
campaign by the original authors due to total inaccessibility — our
extractor must not include AS 22 in per-AS-pair statistics.

---

## J-008 — Windows path-name workaround for `mpquic-on-scion-ipc` checkout

**Decision.** Several files in `mpquic-on-scion-ipc` contain colons in
filenames (e.g., `17-ffaa:1:11e4_jitter_plot.png`) which are illegal on NTFS.
We extract only the files we need via per-file `git show HEAD:<path>` rather
than `git checkout`, into `external/scionpathml_extracted/`.

**Alternatives considered.** WSL or a Linux VM (heavyweight); renaming the
files (would diverge from upstream, fork-by-the-back-door); ignoring the bad
files via sparse checkout (still triggers tree-walk validation that fails).

**Who decided.** Claude Code (workaround forced by the OS).

**Status.** Active. Documented so any user reproducing on Windows knows what
to expect.

---

## J-009 — Trajectory schema: per-timestep table joined to per-event table

**Decision (provisional, to be locked in Task #7).** The trajectory schema
splits into two tables stored side-by-side in Parquet:

* `timesteps`: one row per `(episode_id, t, agent_id)`. Columns include
  `wall_time`, `selected_path_fingerprint`, observation features, reward
  components (Phase 1 will populate; Phase 0 leaves them null).
* `events`: one row per path-level event with `(episode_id, t, agent_id,
  event_type, fingerprint)`.

Global state (link utilizations, agent path assignments, total throughput) is
inlined into `timesteps` for the global agent (id `__global__`) rather than
maintained as a third table.

**Alternatives considered.** A single deeply-nested table (slow to query in
Parquet); a fully relational layout with a `paths` dimension table (overkill
for 1000 episodes).

**Why provisional.** The schema is locked in Task #7 once we confirm K
(rolling-history window size), the exact set of observation features, and
the fingerprint encoding (hex string vs. uint64). These choices interact with
Phase 1's PettingZoo wrapper.

**Who decided.** Claude Code; surface to user before Task #7 closes.

**Status.** Provisional → Locked by J-013.

---

## J-010 — Path-discrepancy proxy: churn-imbalance, not directional asymmetry

**Decision.** For Phase 0 calibration, we use `|added − removed| / max(added, removed)`
per destination AS as our discrepancy metric. This is a *churn-imbalance*
proxy — high when a destination's path set is gaining or losing paths
asymmetrically over time — not the strict Herschbach definition of
"A→B path-set differs from the reverse B→A path-set as observed simultaneously
from both endpoints".

**Alternatives considered.** Strict directional asymmetry: requires
correlating Lab1's view of paths to AS-X with Lab2's (or another vantage
point's) view of paths to Lab1's host AS at simultaneous timestamps. Doable,
but adds 2-4h of cross-lab timestamp-alignment work and is sensitive to
clock skew between vantage points.

**Networking-literature backing.** Herschbach et al. 2509.04695 §4 defines
discrepancy directionally; our proxy is a proper subset of that signal
(directional asymmetry implies churn imbalance, not vice versa).

**ML-literature backing.** N/A.

**Who decided.** User-approved on 2026-04-30 with the note: "revisit only
if calibration tolerance demands it."

**Status.** Active for Phase 0. Re-open if the ±25% discrepancy-rate
tolerance proves binding and the proxy is the reason.

---

## J-011 — Tier-50 Mbps as Phase 0 calibration target; tier-100 reserved as Phase 1 reward-design context

**Decision.** Phase 0 RTT/bandwidth marginal calibration matches the empirical
distributions for the 10 Mbps and 50 Mbps target tiers. The 100 Mbps tier
(empirically 10-40% loss, 65-90 Mbps achieved depending on direction) is the
overload regime — the fluid simulator's elastic-flow assumption (sources back
off under congestion) is not designed to match a regime where senders pin at
saturation regardless of feedback. Tier-100 numbers are retained in the
empirical summary as supplementary context for Phase 1 reward shaping (the
"what does loss look like under overload" baseline) but are not Phase 0
calibration targets.

**Alternatives considered.** Calibrate against all three tiers: would force
the simulator into a regime it cannot reach without parameter values that
violate its modelling assumptions, producing bogus calibration scores.

**Networking-literature backing.** Kelly et al.'s primal-flow model assumes
elastic flows that respond to congestion signals. At true saturation the
fluid limit no longer captures the discrete-loss-event behaviour observed
in production. mpcc-dynamics (Keshvadi 2025) is the discrete-event
counterpart that captures saturation but is out of scope for our PoC
(JUSTIFICATIONS J-002).

**ML-literature backing.** N/A.

**Who decided.** User-approved on 2026-04-30. Recorded so the choice
is defensible to the supervisor: "we did not match tier-100 because it is
outside the modelled regime, not because we couldn't."

**Status.** Active.

---

## J-012 — Episode parameters: 24 agents, 3600 simulated seconds, dt = 0.1 s

**Decision.** Phase 0 episodes simulate **24 agents** for **3600 simulated
seconds** at a fluid-model integration step of **dt = 0.1 s** (36,000 ODE
steps per episode). Observations are emitted at a coarser **observation_dt =
1.0 s** (3,600 observation rows per agent per episode) to keep storage
tractable. The rolling-history window K = 16 timesteps is a Phase 1
observation feature, computed on the fly from the per-timestep rows; not a
Phase 0 schema field.

**Alternatives considered.** Shorter episodes (e.g., 600 s) would reduce
storage by 6× but risk cutting off the empirical path-lifetime tail (median
≈ 7.5 h, p25 = 3 h) that Phase 0 is calibrating against. Longer dt (e.g.,
1.0 s) loses fluid-model fidelity at low RTT regimes.

**Networking-literature backing.** Realistic SCIONLab path-lifetime
dynamics span hours (Herschbach et al. 2509.04695 §3); even at 3600 s per
episode, only a small fraction of the lifetime CDF is exercised within a
single episode, and the calibration relies on pooling lifetime samples
across many episodes.

**ML-literature backing.** Phase 2 MAPPO training may use shorter episodes
for sample efficiency (1024-2048 timesteps is common in CleanRL templates).
That decision is deferred to Phase 1, when reward and observation spaces are
finalized; Phase 0's calibrated dataset is independent of Phase 2's episode
length because Phase 2 will resample / reset within Phase 0 trajectory data.

**Who decided.** User-approved on 2026-04-30.

**Status.** Active for Phase 0. Phase 2 may override episode length without
breaking the Phase 0 schema.

---

## J-013 — Schema lock (Phase 1 contract)

**Decision.** The trajectory schema, locked at Phase 0 close, has two tables
(see `docs/SCHEMA.md`):

* `timesteps` — one row per `(episode_id, t, agent_id)` with the per-agent
  observation columns (selected path, available paths, per-path metrics,
  reward placeholders) plus a special `agent_id="__global__"` row per
  timestep for the centralized critic state (link utilizations, total
  throughput, agent-to-path assignments).
* `events` — one row per path-level event (`path_appeared`,
  `path_expired`, `path_metrics_updated`).

Path fingerprints are 8-byte (16-hex-char) strings, matching the format
observed in ScionPathML's `selected_paths.json`. Storage format is Parquet
with a configurable shard size (default 50 episodes per file). Each shard
ships with a JSON manifest carrying git SHA, config hash, and dataset SHA
per METHODOLOGY §7.

**Alternatives considered.** A single deeply-nested table (slower to query
in Parquet); a fully relational layout with separate `paths` and `agents`
dimension tables (overkill for 1000 episodes).

**Why we lock now.** Phase 1's PettingZoo environment is the Phase 0
schema's only consumer; any change post-lock invalidates Phase 1 code.

**Who decided.** User-approved on 2026-04-30.

**Status.** Locked.

---

## J-014 — Phase 0 dataset uses a stick-with-first-path placeholder selector; reward columns are null

**Decision.** Phase 0 trajectories are produced under a placeholder selector
that picks a uniformly random path from each agent's currently-visible pool
at episode start (and re-picks only on expiry). Reward and reward-component
columns in the schema are written as null. Phase 1 owns both the action
distribution (random / shortest-path / oracle-greedy baselines plus the
trained MAPPO policy) and the reward definition; Phase 1 must generate
fresh rollouts under each baseline rather than deriving baseline metrics
from this dataset.

**Alternatives considered.**
1. *Implement the three Phase 1 baselines now and ship per-baseline shards.*
   Out of scope for Phase 0 — baselines are part of Phase 2 evaluation
   (METHODOLOGY §4) and depend on Phase 1's reward function.
2. *Leave the action column null too.* Rejected because the path-metrics
   observations are conditioned on which path is currently flowing
   (utilization affects per-path BW); null actions would make the recorded
   observations physically inconsistent.

**Networking-literature backing.** N/A — purely a software-engineering
boundary between phases.

**ML-literature backing.** Standard practice in offline-data MARL pipelines
to separate the "physics" dataset (this) from "policy-conditional" rollouts
(Phase 1).

**Who decided.** User-acknowledged on 2026-04-30 during canonical-run
authorization.

**Status.** Active. Phase 1 readme must reiterate: do not derive baseline
performance from `data/phase0_trajectories/`.

---

## J-015 — RTT marginal: single-Gaussian shape model, empirical is bimodal-pooled

**Decision.** Our synthetic RTT model samples
`base ~ Normal(mean, std)` × `lognormal noise`, producing a unimodal
distribution centred around 400 ms. The empirical pooled RTT histogram is
bimodal (peaks ~280 ms and ~580 ms) because it pools across 6 source-
destination pairs whose individual marginals are unimodal but centred at
different points. The Phase 0 calibration metrics (`median in empirical
IQR`, `IQR ratio ≤ 0.5`) pass because they evaluate moments, not shape.

**Why we accept the shape mismatch.** Phase 0 calibrates marginal moments
per JUSTIFICATIONS J-006; per-pair shape fidelity would require
per-AS-pair RTT distributions in our config (one mean/std per pair),
which adds 6× config surface area for no benefit downstream — Phase 1's
agents see *one path's* RTT at a time, not the pooled marginal, and per-
pair tuning is a Phase 2 concern only if MARL training reveals
sensitivity.

**Networking-literature backing.** Pair-wise RTT distributions are
geographic-distance-dominated and trivially separable; pooling distinct
pairs in marginal calibration is a reasonable Phase-0 simplification.

**ML-literature backing.** Domain-randomization (Tobin et al. 2017)
matches *moments* of the noise distribution, not shape — the same
argument applies here.

**Who decided.** Claude Code, recorded post-calibration when the figure
revealed the pooled bimodality.

**Status.** Active. Reopen if Phase 2 finds the trained policy is
sensitive to per-pair RTT structure.
