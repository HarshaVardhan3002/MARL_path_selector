# Deployment-Target Commitment Memo

> METHODOLOGY §2 names this memo as a Phase 0 artefact. The decision is
> already settled in CONTEXT.md; this document writes it up in
> supervisor-facing terms so the choice and its consequences are explicit.

## Decision

The trained MARL path-selection policy targets execution on a **constrained
end-host or first-hop gateway** under SCION's host-authority model. ISP-side
deployment, smartNIC offload, and in-network execution are explicitly
out of scope.

## Why this target

SCION is, by design, an architecture in which the *end-host* selects the path
(Perrig et al., "The SCION Internet Architecture", 2017). The end-host
authoritatively constructs path segments from beacons it receives; the network
exposes choices but does not impose them. A path-selection mechanism that
required cooperation or trust from the network would invert this architectural
property and reintroduce exactly the failure modes — single point of trust,
oscillation under shared recommendation — that motivate this work.

The first-hop gateway is permitted as an alternative target because, from the
SCION threat model's perspective, a host's first-hop SCION endhost-router
shares the host's trust domain (Krähenbühl et al. PISA, 2021). The two
targets share the same constraint envelope: limited compute, no GPU, latency
sensitivity, and the requirement that the path-selection artefact ships
identically to every node.

## What this implies for the system

1. **Inference budget.** Path selection runs once per path-selection epoch
   (order of seconds, per Herschbach et al. 2509.04695 §3 path-discovery
   cadence). The teacher MARL policy is a small MLP (hundreds of KB);
   the distilled XGBoost student (Phase 3) is the first-class shipping
   target and reduces inference to integer comparisons (Treelite,
   Phase 4 stretch).

2. **No global state at inference.** The policy sees only this host's local
   observation. CTDE supplies global context during *training* via a critic
   that is discarded after training. This commitment must hold through every
   downstream design choice: any feature requiring runtime communication
   between hosts violates the deployment story.

3. **Same artefact on every host.** Parameter-sharing across agents during
   MAPPO training (CONTEXT.md decision row 5) is justified specifically by
   this deployment property: every host receives the same compiled model.
   N-policy training (one policy per agent) is rejected on these grounds,
   not on ML grounds.

4. **Adversarial and online-learning boundaries.** Because the artefact ships
   compiled and frozen, online learning at deployment is out of scope
   (METHODOLOGY §1) and adversarial-robustness research is positioned as
   future work. The PoC defends the offline-training, frozen-deployment
   variant only.

## What this rules out (and is reframed as future work in conclusions)

* **ISP-side aggregator.** The Path Oracle of Krüger et al. (2023) operates
  in this regime. We position our work as the *complement* — distributed
  end-host policies that no longer require an aggregator at runtime.
* **SmartNIC offload.** Reasonable extension of the Treelite C-compilation
  story (Phase 4 stretch), but introduces ISA portability, driver, and
  certification questions outside thesis scope.
* **Hardware deployment to real SCIONLab nodes.** Reframed as the natural
  continuation of Phase 4 once the C artefact is verified bit-exact against
  the XGBoost teacher.

## Sign-off

This memo records a decision made in CONTEXT.md row 1 and is submitted as a
Phase 0 artefact per METHODOLOGY §2.
