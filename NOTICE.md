# Third-Party Code

This repository vendors source code from upstream research repositories under
`scion_sim/_vendored/`. Vendored files are read-only — we do not modify them
and we do not contribute changes back. Pinned upstream commit SHAs are
recorded below for reproducibility.

## scion_sim/_vendored/pami/

Source: https://github.com/keshvadi/Path-Aware-Multipath-Instability
Pinned commit: `294c07979c1d58128ac3ec80e18288388d5bf60f`
Files: `algorithms.py`, `algorithms_dual.py`, `algorithms_pan.py`,
`network_components.py`, `scheduler.py`.
Purpose: Fluid-model multipath congestion-control simulator (primal/dual
ODE formulations). Used as the dynamics substrate for our MARL environment.
License: not explicitly declared by the upstream author. We reuse under
academic fair-use for thesis purposes. If the upstream author publishes a
license, this notice will be updated to reflect it. Contact: skeshvadi@tru.ca

## external/ (gitignored, regenerable)

The `external/` directory is gitignored and contains shallow clones of the
following upstream repositories. To regenerate, run
`scripts/refresh_external.sh`.

* `Path-Aware-Multipath-Instability` @ `294c07979c1d58128ac3ec80e18288388d5bf60f`
* `mpquic-on-scion-ipc` @ `362ad60885dabeb2a067d3dae058c506ce5f092a`
* `mpcc-dynamics` @ `5d0595cee65480a213cf347c40e03bfbcc38858b`
