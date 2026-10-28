---
type: infrastructure
complexity: simple
status: in review
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Control the inner steps on a relative Monte Carlo error, not an absolute one

Raised by the review of `outer-run-health-guard` (2026-10-28). The step validation `_healthy`
rejects an inner step whose run has MC error (`psi_psi_rel_mc_err`) above `max_mc_err` (1).
That is an absolute check on a property of the sample size, not of the step, with two effects:

- In the pinned 32-chain default-step solves (`tests/fixtures/hmc_solve_lazy_net_f32.pt`,
  `hmc_solve_gaussian_phi_f64.pt`) the MC error is 1.75 to 4.31 at every run, so every inner
  step after the first is reverted for "sampler health": `n_accepted_inner` is 0 at every outer
  step and eta_phihat halves 0.05 -> 0.0125 -> 0.0031 -> 0.00078 over three outer steps. The
  inner ascent stalls on noise; that is a pre-existing behaviour, not a guarded failure.
- The outer-gradient run, which validates the ascent's trailing step, is deliberately not held
  to the MC error (no revert cures it; `meanfield._outer_run`), so the trailing step is the only
  inner step not checked on it at all.

A related finding: the inner path's give-up, `res.n_backtrack[l] > max_backtrack * (inner + 1)`
in `_inner_layer_step`, cannot fire from `_inner_ascent`, which calls each layer once per
inner index, so a layer has at most `inner + 1` backtracks by then (the counter is incremented
before the check). A step that keeps breaking the sampler ends the ascent at n_inner_max
instead of raising `UnhealthySolveError`.

## Plan
1. Replace the absolute MC-error check with a relative one, like the ESS-drop rule: a step is
   reverted when its run's MC error exceeds `max_mc_err_growth` (new, hashed off its default)
   times that of the accepted point before it. Apply the same relative check to the
   outer-gradient run (the trailing step), so every inner step is held to it.
2. Decide what to do with `max_mc_err` (keep as an absolute ceiling far above 1, or drop it):
   the hash rule for either choice must not move existing hashes where results do not change.
3. Make the inner give-up reachable (count backtracks per inner step, or cap the total), and
   test that it raises `UnhealthySolveError`.

## Acceptance criteria
- The pinned 32-chain default-step solves accept inner steps (n_accepted_inner > 0) or the
  reason they do not is a sampler failure, not the sample size; their fixtures are regenerated
  only with the change stated in the commit.
- A unit test with a stubbed sampler: an MC error that jumps by more than the relative bound
  rejects a step (inner and trailing); a steadily high one does not.
- A unit test drives `_inner_ascent` into the inner give-up and gets `UnhealthySolveError`.
- The tanh exemplar at its defaults and the analytic saddle tests stay within their pinned
  tolerances; design.md documents the rule. Full suite green; ruff clean.

## Also (from the re-review of outer-run-health-guard)
- Wording: at the give-up check a layer has at most `inner + 1` backtracks (the counter is
  incremented first), not `inner`; the conclusion (unreachable) is unchanged.
- `_unhealthy_outer_run_message` ends with the "Chains far above the prior's scale..." hint
  even when only ESS or divergence checks failed; make the hint conditional on the h2 checks.
- `fct/meanfield.py:1323` (solve docstring) is 102 columns.

## Implementation (worker, 2026-10-28; branch `meanfield-mc-error-control` from main a7ba292)
- `SolverConfig.max_mc_err_growth` (default 10, in `HEALTH_BOUND_DEFAULTS`: hashed only off
  10). `meanfield._health_failures` holds a run to it: the run's MC error per transition (MC
  error x sqrt(transitions), so the longer outer-gradient run compares with the inner ones)
  must not exceed max_mc_err_growth times that of the accepted point before the step
  (`_SolverState.mc_ref`, set by `_accept_point`, checkpointed). No reference (warm-up, which
  collects no moments) or a run without moments (an outer step's probe) is not checked.
- One step validation for every run: `_sampler_sound` / `_soundness_failures` are gone, the
  outer-gradient run is held to `_healthy` like the inner runs, so the trailing step is
  checked on the MC error too.
- `max_mc_err` dropped (plan 2, "drop it"): it is in `config.RETIRED_SOLVER_FIELDS`, which
  keeps it in the hash payload at 1 so no hash moves; a config that sets it loads at 1 (every
  saved config.json has the key, `legacy_config.json` included) and is rejected naming
  max_mc_err_growth at any other value. Rejected alternative: keeping it as an absolute ceiling
  far above 1 means changing a hashed default (every hash moves although no default solve
  changes) or hashing a value the config does not hold.
- Inner give-up counted per step: `_InnerResult.n_unhealthy`, health reverts since the layer's
  last accepted point; the failure after max_backtrack of them raises `UnhealthySolveError`
  (message: inner step, reverts, max_backtrack, eta_phihat, the failed checks).
- `_unhealthy_outer_run_message`: the escaped-chains hint only when an h2 check failed. The
  solve docstring rewrapped (no line over 100). Wording fixed here and in
  `outer-run-health-guard.md` (`inner + 1`).
- Tests: `tests/test_mc_error_control.py` (stubbed exact-Gaussian sampler: jump above the bound
  rejects an inner step and the trailing step, steadily high MC error (0.5 to 50, and 4 in an
  inner ascent and an outer run) does not; per-transition comparison; NaN fails; no reference
  / no moments not checked; `_inner_ascent` driven into the give-up raises); config tests for
  the new bound's hash rule and the retired field (flat and nested, at 1 and at 2); the
  pinned small solves accept inner steps; the hint test; existing tests moved to the new
  signatures.

## Results
- Default value, a judgement call (the issue gives none): 10, not the 3.2 a tenfold ESS drop
  (min_ess_drop) would suggest. Per-transition ratio of a run's MC error to the accepted point
  before, over runs sound in every other way: pinned 32-chain solves up to 1.66; tanh exemplar 1.31
  (HMC), 1.15 (importance); relu exemplar problem at default step control, 40 outer steps, up to 6.3
  (99th percentile 3.2, 9 of 579 above 3). At 3, the eta_max 0.1 relu replay reverts nine steps at
  outer step 2, five on the MC error alone (retaken steps of ever smaller size, still 3 to 6x a
  lucky-low reference), its trajectory changes and `test_relu_tilt`'s incident test fails; at 10 the
  replay is that of main (step-5 tilt rejected by the h2 guard as before). 3 would be the stricter
  choice if the PI prefers it; the relu numbers say it rejects sound steps.
- Pinned fixtures: `hmc_solve_lazy_net_f32.pt` and `hmc_solve_gaussian_phi_f64.pt` regenerated
  (only those two): n_accepted_inner 0 -> 2 per layer and outer step, n_backtrack_inner 2 -> 0,
  eta_phihat 0.05 -> 0.0605 -> 0.0732 -> 0.0886 (was halving to 0.00078); Phi moves 9.5 % /
  10.1 % relative; still no outer-run retries. The `_legacy_steps` fixtures, the default and
  legacy hashes (`meanfield_hashes.json`) and `legacy_config.json` are unchanged.
- Tanh exemplar at its notebook defaults (HMC, net init, 80 steps, seed 0): Phi, Phi_hat, the
  Phi_hat residual and MC-error columns bit for bit those of main a7ba292, before and after:
  last three residuals 0.086 / 0.196 / 0.105 (largest 0.196 < 0.9), inner loop converged at all
  80 steps (1 to 13 inner steps, 3 rejected for progress, 0 health backtracks, 0 outer
  retries). Importance: bit for bit too, last three at most 0.092, all converged.
- Analytic saddle, weak-tanh and relu tests pass unchanged.
- Full suite 1639 passed; `ruff check fct tests notebooks` and
  `ruff format --check fct tests` clean; no added line over 100 columns.

## Open points for the review
- The default 10 (above): a PI call if 3 is wanted; the relu data argue against it.
- `hmc_solve_gaussian_phi_f64.pt` (relu, 3 outer steps, 32 chains): with inner steps now
  accepted, layer 2's Stein fixed-point residual is 1.1e4 to 1.6e4 (was 800 falling to 50 while
  eta_phihat collapsed). A regression pin far from convergence, not a quality measure; noted
  because it moved.
- Not changed: the first run of each inner ascent is accepted without a health check (it has
  no pending step), including the restart of the last layer at its Stein target under
  inner_start="target"; it also sets the MC-error reference for that ascent.
