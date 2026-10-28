---
type: infrastructure
complexity: complex
status: awaiting your confirmation
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Mean-field solver: control the inner step on progress and rescale the outer step for a bounded-psi last layer

Raised by the re-review of `readout-post-activations` (2026-10-29). With the readout phi(h^L) the
exemplar notebook's default solve (L 1, tanh, beta 10, net init, P 20, k 2) diverges: the
Phi_hat fixed-point residual goes 0.33, 0.49, then 32 at outer step 2 and stays at 20-140.
Diagnosis (reviewer's experiments, logs in the worker's worktree `tmp/review/`):
- Not the sampler (2048 chains or 5x transitions diverge earlier), not a sign/scale error (tilt
  density and factors verified; projection size 0), not the new projection (the tanh tilt is
  unprojected).
- The inner loop grows `eta_hat` 1.1x per accepted step up to `eta_max = 1`
  (`fct/meanfield.py:565`) and `_healthy` (`:480-503`) gates on sampler health only, never on
  progress: a step that raises the inner residual from 0.04 to 0.42 is accepted; the chains
  lag the moving tilt (Stein mismatch 0.02 -> 1.29) and the ascent keeps stepping.
- With the inner step held at 0.05 the iteration is stable (inner residual 0.02-0.07, Stein
  <= 0.04) but the outer Phi descent moves Phi by < 5e-4 relative per step because its step is
  scaled by c^2 = (tr Phi / P)^2 ~ 0.1 for a saturated tanh kernel; the self-consistent tilt is
  about twice the learning tilt and would need thousands of outer steps.
- The design doc's old baseline (residual 0.76, lazy init, readout h) was itself an unconverged
  solve of the Gaussian last-layer problem (Phi_hat_L a quarter of its fixed point); it is not
  evidence the solver converged before.

## Proposed plan
1. Inner step: accept a step only if the inner residual does not increase (Armijo-style
   backtracking in `_inner_layer_step`, halving `eta_hat` on failure, growth only after
   successes), with `eta_max` per layer; record accepted/rejected counts in the history.
2. Outer step: scale the Phi step by ||Phi|| ||Phi_hat|| (or an estimate of the local curvature)
   instead of c^2, so bounded-psi layers move at a sensible rate; keep the Gaussian-layer
   behaviour identical where c^2 and the new scale coincide (pin test).
3. Re-run the exemplar at its defaults; it must finish under the tripwire with the inner loop
   converged at every step; update the design doc's reported residual with the new number and
   remove the 0.76 baseline claim.
4. Tests: a synthetic tilt whose response is weak (saturated tanh) converges within a bounded
   number of outer steps; a Gaussian layer's trajectory is unchanged to tolerance.

## Acceptance criteria
- `notebooks/build_exemplar.py` executes at its defaults; the solve's last-k residual is
  below the tripwire and the inner loop converged at every outer step.
- No pinned Gaussian-layer result moves beyond the stated tolerance; tests CPU only; ruff clean.
