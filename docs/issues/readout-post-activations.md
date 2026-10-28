---
type: infrastructure
complexity: complex
status: done
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

## Consequences
- It invalidates `tmp/spatial_cache` and every cached sweep (they become stale, loudly). The
  `pt_sweep_big` SLURM arrays that ran from the shared tree were cancelled by the PI, so nothing
  running can mix the two models.
- Alternative, if you meant only the figures: keep the model, and drop the "h" case from the
  spatial notebooks only (that part is in `spatial-grid-notebook`). Say if this is what you want.

## Acceptance criteria
- `MLP.forward` output equals `phi(H[:, -1]) @ V / N`; the solver's learning term and the
  net's empirical kernels use phi(h^L); no `readout` field remains.
- Old artefacts raise StaleCacheError; new ones load.
- Tests CPU only; ruff clean; design doc updated.

## PI confirmation (2026-10-28)
Confirmed as planned (the model, the solver and every notebook read out phi(h^L)). The PI cancelled
the `pt_sweep_big` arrays because this change invalidates them: merge as soon as reviewed. The notebook part of the "figures only" alternative is moot.

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
- `MODEL_VERSION` is hashed unconditionally, like `LOSS_VERSION`; the spatial-prior and
  modularity-sweep hash pins are re-pinned to the current code.

## Tilt projection (decided in review)
With psi = phi on every layer, the old prior-blind projection for psi = phi
(`M + lambda_r I >= tilt_floor * lambda_r I`) removed the negative semidefinite learning tilt of
the read-out layer, Phi_hat_L = -2c A Y Y^T A, entirely at `rate_tilt = 0`: the analytic L = 1
saddle solve returned Phi_hat = 0 and returned normally. The projection is now activation-aware
(`fct.hmc.tilt_rule`, from `fct.activations.tilt_class`), `project_tilt = True` stays the default:
- bounded phi (tanh, sigmoid): no projection; the tilt is bounded, the density always normalisable;
- linear phi (`fct.activations.identity`) or psi = h: the Gaussian condition Sigma^-1 + Q > 0
  (necessary and sufficient);
- relu: `lambda_min(M + lambda_r I) >= -(1 - tilt_floor) lambda_min(Sigma^-1)`, the certificate
  `normalisability_margin` reports (one implementation, `relu_tilt_spectrum`);
- any other activation: ValueError when the solve starts (with `project_tilt`).
The analytic saddle is recovered under the default again (the overrides are gone), and a test pins
that the default projection leaves the analytic Phi_hat* whole. Every outer step records
`tilt_projected_rel` (per layer, the largest relative change the projection made). A tilt
tripwire, not a convergence test: a solve whose largest Stein residual of Phi_hat over the last
`solver.tilt_tripwire_steps` (3) outer steps is >= `solver.tilt_tripwire_resid` (0.9; neither
hashed) raises `UnconvergedSolveError` instead of returning, so no such final.pt is written, and
the CLI reports it in one line.

Reviewer's caveat, for the PI: the relu floor is sufficient, not necessary (the exact condition
is a copositivity check per orthant, 2^P of them). With the library defaults
(prior_scale = 1/lambda_w, lambda_min(Sigma^-1) ~ 1e-4 on a relu P = 20 example) it still cuts a
relu learning tilt to about 0.7 % at rate_tilt 0 (5 % at 1e-3, 100 % at 0.1). The PI decides
whether that is acceptable for relu solves.

## The exemplar notebook's default solve no longer converges
Executed at its defaults (L = 1, tanh, beta = 10, 80 HMC outer steps, `INIT = "net"`), the
exemplar notebook now ends with `UnconvergedSolveError`: the Stein residual of Phi_hat is 0.33 at
step 0 and 0.49 at step 1, jumps to 32 at step 2 and stays at 20 to 60 (40 at step 80), with the
samples' scale ||h||^2/P up to 10 to 20 times the prior's. For L = 1 the read-out layer is the only
layer, so it is now a tanh tilt (unprojected, being bounded) instead of the Gaussian one. Smaller
inner steps help but do not settle it (20 steps: `eta_max = 0.1` plateaus at a residual of
about 2; `eta_max = 0.01` drifts from 0.21 to 0.54). Under the prior-blind projection of the first
review round the tilt would have been cut to near 0 (rate_tilt 0.02), a residual near 1, so this
is a consequence of the readout change, not of the new projection; before the final-residual
check it would have returned silently. The notebook defaults are unchanged.

## Follow-ups raised
The PM opens two items, out of scope here:
- `solver-inner-step-control`: the exemplar divergence above. The inner step size grows without
  a check that the inner residual makes progress, and the outer Phi step scaled by c^2 is unfit
  for a bounded-psi last layer; the old 0.76 baseline was itself an unconverged solve of the
  Gaussian problem.
- `relu-tilt-rule`: the sufficient relu floor cuts the learning tilt to about 33 % on the
  exemplar's relu problem and about 0.2 % at the library default prior scale, so every relu
  setting of the modularity sweep's solve stage ends in `UnconvergedSolveError` until the PI
  decides the rule.
