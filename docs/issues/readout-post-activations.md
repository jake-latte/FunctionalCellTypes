---
type: infrastructure
complexity: complex
status: in review
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# The readout is always a linear map of the post-activations (remove the "h" case)

The PI: "the readout of the network is always a linear mapping of the post-activations,
y_hat = (1/N) V phi(h^L)". Today `fct.model.MLP.forward` reads the pre-activation
(`Z = V^T h^L / N`, as the original playground did), and `MeanFieldConfig.readout` selects
"h" (default) or "phi" for the solver and the kernel plots. This item makes phi(h^L) the only
readout: in the model, the solver and every notebook.

## Proposed plan
1. `MLP.forward`: `Z = V^T phi(h^L) / N`. `MeanFieldConfig.readout` is removed (a config.json
   that still carries it fails to load, naming the field); `readout_activities` loses its
   argument and returns phi(h^l) for every layer; `empirical_kernels`, `net_kernels`, the
   solver's learning term and `fct.exemplar` follow.
2. Every cached training run and mean-field result was made with the h readout. Add
   `MODEL_VERSION = 2` folded into the net section hash (as `LOSS_VERSION` is into train), so
   old artefacts are rejected as stale instead of being read as if they came from this model.
   The pinned fixtures of `tests/test_meanfield_pins.py` / `pinned_runs.py` must be regenerated
   and the HMC/solver regression values re-derived from the notebook convention.
3. The notebooks (`spatial`, `exemplar`, `phase_transition`, `modularity`) lose the readout
   cases in their equations and titles.

## Why this needs your confirmation
- A SLURM array (job 3847945, root `pt_sweep_big`) is running from this tree's `.venv`: array
  tasks that start after the merge would train the new model under the old hashes, and the
  grid would silently mix two models. This item must not be merged while that array runs,
  or the running sweep must be done in a separate checkout. Say which.
- It invalidates `tmp/spatial_cache` and every cached sweep (they become stale, loudly).
- Alternative, if you meant only the figures: keep the model, and drop the "h" case from the
  spatial notebooks only (that part is in `spatial-grid-notebook`). Say if this is what you want.

## Acceptance criteria
- `MLP.forward` output equals `phi(H[:, -1]) @ V / N`; the solver's learning term and the
  net's empirical kernels use phi(h^L); no `readout` field remains.
- Old artefacts raise StaleCacheError; new ones load.
- Tests CPU only; ruff clean; design doc updated.

## PI confirmation (2026-10-28)
Confirmed as planned (the model, the solver and every notebook read out phi(h^L)). The PI cancelled
the `pt_sweep_big` arrays because this change invalidates them, so the hazard above is gone: merge
as soon as reviewed. The notebook part of the "figures only" alternative is moot.

## Implementation (branch `readout-post-activations`)
- `MLP.forward` returns `Z = V^T phi(h^L) / N`. `config.MODEL_VERSION = 2` is folded into every
  section hash that holds `net` (train, meanfield and analysis caches, sweep settings), so a run
  or solve of the h-readout model raises `StaleCacheError` / shows as `stale`.
- `MeanFieldConfig.readout` is gone; a config.json carrying it fails with a ValidationError
  naming `readout`. `readout_activities(H, phi)` and `empirical_kernels(net, X)` use phi(h) on
  every layer; the solver always samples with psi = phi. The samplers keep their `psi_is_phi`
  argument: psi = h is a Gaussian tilt with closed-form moments that the sampler tests use.
- Fixtures: `hmc_solve_lazy_net_f32.pt` (it had a net read out from h) and
  `meanfield_hashes.json` were regenerated; `readout` was deleted by hand from the flat
  `legacy_config.json`. `hmc_solve_gaussian_phi_f64.pt` (already the phi readout, no net) is
  untouched and still reproduces bit for bit, which checks that the phi path of the solver did
  not move.
- Notebooks: `spatial` (network and output equations, the psi cases of the kernel section, the
  panel titles and `data_kernels` / `plot_data_kernels` signatures) and `exemplar` (psi of the
  kernel figure) regenerated from their builders. `phase_transition` and `modularity` contain no
  readout equation or case, so they are unchanged.
- `MODEL_VERSION` 1 is left out of the hash, so a version-1 hash is the one the earlier code wrote
  (the modularity-sweep and spatial hash pins hold at version 1, which the tests set).

## Open: the tilt projection removes the learning term of the read-out layer (needs a PI decision)
With `solver.project_tilt = True` (the default) every inner step projects Phi_hat_l onto
normalisable tilts. For psi = phi that projection is `M + lambda_r I >= tilt_floor * lambda_r I`,
independent of the prior Sigma (the ReLU orthant condition). The learning tilt of the read-out
layer, Phi_hat_L = -2c A Y Y^T A, is negative semidefinite, so it is cut to at most
`(1 - tilt_floor) lambda_r` in size: with `rate_tilt = 0` it is removed entirely, and on a
random P = 20 relu example its norm falls to 2e-6 of itself at lambda_r = 1e-3 (1.7e-4 at 0.1);
the psi = h projection used before (`Sigma^{1/2} M Sigma^{1/2} >= -(1 - tilt_floor) I`) kept
3%. The same held before for `readout = "phi"`; this item makes it the only case. So, until the
projection changes, a default solve cannot represent learning in the last layer. The analytic
L = 1 tests and the ESS-floor test now set `project_tilt = False` (with it off the analytic
saddle is recovered to 1.3e-2). Candidate fixes, for the PI: the Sigma-aware sufficient
condition `lambda_min(M) >= -(1 - delta) lambda_min(Sigma^-1)` that `normalisability_margin`
already uses for psi = phi; no projection for bounded phi (tanh, sigmoid); or `project_tilt =
False` by default.
