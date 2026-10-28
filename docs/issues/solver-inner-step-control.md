---
type: infrastructure
complexity: complex
status: in review
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

## PI confirmation (2026-10-29)
Confirmed as planned. Starts once `prior-scaling-convention` is merged (it moves the solver's
default constants); runs in parallel with `relu-tilt-rule` (different functions of the solver).

## Result (worker, 2026-10-28)
Branch `solver-inner-step-control`, from main e60f086. Three solver changes, each its own
commit with its own pins, every one switchable back (`SolverConfig`; all four fields are
hashed, `config.STEP_CONTROL_FIELDS`, so every meanfield result solved before is stale):
1. Inner step (`inner_step_control="progress"`), as planned, with one deviation: a step is
   rejected when the inner residual rises by more than the convergence tolerance of the point
   before it, not when it rises at all. A strict test ratchets: the residual before the step
   is a Monte Carlo estimate, the lucky low one among retries, and with 20 transitions per
   run eta_phihat fell to its floor and the inner loop stopped converging. Halving stops at
   eta_min_rel x eta_phihat. Own inner cap `eta_max_phihat` = 0.25 per layer. History:
   `n_accepted_inner`, `n_rejected_inner` (per layer); solver-health panel added.
2. Outer step (`phi_step_scale="kernel"`): Phi_l dS/dPhi_l Phi_l, the local-curvature choice
   (a Gaussian layer's Newton direction), not ||Phi|| ||Phi_hat||, which is not even
   dimensionally a step scale. It equals the c^2 step exactly when Phi_l = c_l I (unit test).
   `eta_max` (the outer cap) 1 -> 0.25: the measured tanh response at the exemplar's kernel
   is 0.3 of the Gaussian one, so eta = 1 overshoots; at eta = 1 the kernel wandered by up
   to 13 % and the residual reached 2.
3. Not in the plan, and needed: `inner_start="target"` starts each inner ascent of the last
   layer at its exact Stein target -2c A Y Y^T A. With 1 and 2 alone the exemplar still
   failed the tripwire (Frobenius residual 1 to 3, ||Phi_hat|| 0.44 -> 0.8-1.4 against a
   fixed point of 0.41), although Phi was 2 to 4 % from an independent Picard reference
   (8192 chains) and the Phi-weighted residual was 0.03 to 0.2: the inner residual cannot
   see directions of Phi_hat the kernel barely responds to, and inner steps fill them with
   noise. An earlier variant that also froze a layer below its tolerance ("no step at
   rest") was dropped: it stopped the Gaussian layer at the inner tolerance.

Exemplar at its defaults (`python notebooks/build_exemplar.py tmp/...`, seed 0, HMC, net
init, 80 steps): finishes under the tripwire; inner loop converged at all 80 outer steps
(1 to 16 inner steps, 5 rejected for progress, no health backtracks). Stein fixed-point
residual of Phi_hat per step:
0.326 1.114 0.175 0.181 0.386 0.277 0.258 0.232 0.185 0.216 0.254 0.398 0.583 0.148 0.147
0.169 0.21 0.207 0.113 0.166 0.152 0.131 0.137 0.15 0.16 0.161 0.194 0.166 0.175 0.162
0.128 0.13 0.129 0.091 0.193 0.36 0.615 0.111 0.108 0.117 0.108 0.378 0.188 0.186 0.113
0.111 0.077 0.13 0.082 0.059 0.219 0.124 0.071 0.088 0.098 0.078 0.11 0.105 0.109 0.099
0.12 0.089 0.119 0.148 0.143 0.161 0.112 0.087 0.079 0.118 0.151 0.102 0.142 0.074 0.129
0.086 0.068 0.123 0.119 0.15; largest over the last three 0.150. The solved kernel is 3.1 %
from the Picard reference (the net's own kernel 6.4 %). `INIT = "lazy"`: last three at most
0.163, 3.9 % from the reference. Importance sampling now completes 80 steps (also at
P = 24), so `N_OUTER_IMPORTANCE` is 80; last three at most 0.062, 2.0 % from the reference.

Pins: the old HMC fixtures are kept as `hmc_solve_*_legacy_steps.pt`, reproduced bit for bit
by `tests/pinned_runs.LEGACY_STEP_CONTROL`; the default solves, `meanfield_hashes.json` and
test_sgld's pinned default hash are regenerated (step control enters the hash); the legacy
config.json records eta_max = 1, so `LEGACY_OVERRIDES` sets it. The Gaussian-layer trajectory
is not unchanged: on the L = 1 linear problem it lags the legacy one by up to 0.075 (pinned
< 0.1; 0.054 against 0.019 after 10 steps, outer cap 0.25 against 1) and ends as close to the
analytic saddle (0.015 against 0.013 after 40 steps; pinned within 0.01; the existing
analytic-saddle test, < 0.04, still passes).

Unresolved, for the PI:
- L >= 2 with a bounded phi is not fixed, before or after: the exemplar at `L = 2` (HMC, net
  init, 80 steps) ends in `UnconvergedSolveError` under both the legacy and the new step
  control. The hidden layer's Stein target sym(<u h^T>_{l+1} Phi_l^-1) inverts a nearly
  rank-two kernel (eigenvalues floored only at inv_floor_rel = 1e-4 x lambda_max), so its
  fixed-point residual is about 1200 already at step 0 from the net's own kernels, and that
  layer's inner residual climbs to 1 in both (legacy also stops converging its inner loop; the
  new control converges it only because the tolerance follows the inflated MC error). A
  separate item: a better-conditioned hidden-layer target (the direct form, a larger floor, or
  a Phi-weighted residual).
- The Frobenius fixed-point residual of Phi_hat weights directions the kernel does not respond
  to as much as those it does; a Phi-weighted residual ||Phi^1/2 (Phi_hat - target) Phi^1/2||
  would measure only what changes the density (0.03 to 0.2 on the exemplar where the
  Frobenius one was 1 to 3). Not changed here (the tripwire reads the Frobenius one).
- A Gaussian layer is slower to converge under the 0.25 outer cap; `eta_max = 1` restores the
  legacy speed for Gaussian problems if that matters.

## Where it stopped (worker, 2026-10-28, after the review)
Branch `solver-inner-step-control`, main 4ba7cb8 merged in (7df645a; docs only). Review found
no numerical fault; every point it raised is now done:
- B1 merge with main (6696884), S4 conditional hashing (7c9c98c), S2 noise-tolerance cap and
  S6 progress against the best residual (8f13c92), S1 last layer's residual at its Stein
  target, Picard wording (b3c4b99).
- S5: `eta_max` caps only the outer step, stated in the `SolverConfig` comment and design.md
  (already in 7c9c98c/b3c4b99; the design.md config block reworded in present tense).
- Nits: the floor on health backtracks is in design.md (b39d037). The Picard reference
  `notebooks/exemplar_reference.py` builds its prior, constants and Stein target with the
  solver's own helpers and is tested against the quadrature fixed point (see below). The
  weak-test numbers in design.md (0.042-0.046 / 0.110-0.119) were re-measured over seeds 0-4
  and are exact, so unchanged. History wording in the relu replay and design.md config
  comments reworded. `tests/pinned_runs.py:13` is 83 columns on this tip. No line this branch
  adds to a .py file is over 100 columns (the docstring in `fct/exemplar.py` was wrapped only
  in d671fcb; b39d037 had missed it).
- Exemplar numbers in design.md re-measured (7f09d8c); they supersede those in "Result"
  above. At the defaults (HMC, net init, 80 steps): last three at most 0.196 (tripwire 0.9),
  inner loop converged at all 80 outer steps (1-13 inner steps, 3 rejected, 0 health
  backtracks); importance 0.092; lazy 0.392 (12 rejected), all converged inner loops. Solves
  2.1 / 2.7 / 1.6 % from the Picard reference (net 6.4 %).
- Confirmation review (at b0397bc): mergeable, numbers reproduce. Its points are applied:
  the quadrature test's bound is 0.01, not 0.02 (seeds 0-9 land 0.002-0.007; a target 20 %
  low fails it), `exemplar_reference` reuses the solver's problem (new
  `meanfield._last_stein_target`, called by the solver's three copies too) and is tested on
  a config with the weak problem's constants, returns its last change instead of printing
  it, and rejects activations needing a tilt projection (6c8fff0); the shared weak problem
  lives in `tests/weak_tanh.py` (e3e4dda); the lazy and importance ranges in design.md and
  the exemplar intro are accurate (lazy 0.09-0.80 after step 20, importance up to 0.42 in its
  last 30 steps), design.md's broken wrap fixed (d671fcb).
- Full suite: 1574 passed; `ruff check fct tests notebooks` and `ruff format --check fct
  tests` clean.
Next: merge, then start `outer-run-health-guard`.
