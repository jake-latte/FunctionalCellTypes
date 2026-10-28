---
type: infrastructure
complexity: simple
status: in review
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Health-check the outer-gradient run and bound the sampled scale absolutely

Found by the review of `solver-inner-step-control` (2026-10-30), replaying the relu incident
recorded in `relu-tilt-rule`: at outer step 5 the inner loop had converged; the
non-normalisable tilt was the inner loop's unchecked trailing step; the outer-gradient run
sampled it (h2/prior 2.98e3) and was accepted without any health check, because
`_outer_gradients` calls `_accept_point` but never `_healthy`. The health gate also bounds only
the GROWTH of h2, not its size. Progress-based inner acceptance does not catch this: the
"1.3 -> 4.5" was the outer fixed-point residual, which the inner rule never sees.

Legitimate h2/prior values seen: 1.36 (tanh exemplar), up to ~3 (relu near-misses), 5.6
(L = 2 tanh). The incident: 3e3.

## Plan
1. A hashed solver field `max_h2_prior_ratio` (default 100) checked in `_healthy`: a run whose
   sampled second moment exceeds that multiple of the prior's fails the health check.
2. Apply `_healthy` to the outer-gradient run too. On failure: revert the trailing inner step,
   halve eta_hat and rerun; if it persists after the retry budget, raise (loudly, with the
   ratio in the message).
3. The relu incident test (`tests/test_relu_tilt.py`, legacy step control, eta_max 0.1)
   becomes the regression test: under the guard, step 5's non-normalisable tilt is rejected
   (or the solve raises loudly), never accepted.

## Acceptance criteria
- The incident replay no longer accepts the step-5 tilt; a unit test with a stubbed sampler
  shows an outer run with h2/prior above the bound is rejected and retried.
- The tanh exemplar at its defaults and the analytic saddle tests are unchanged (the bound
  never binds there; assert it).
- Full suite green; ruff clean; design.md documents the bound and why the outer run is checked.

Starts after `solver-inner-step-control` merges (same functions).

## Implementation (worker, 2026-10-28; branch `outer-run-health-guard` from main ae3dd45)
- `SolverConfig.max_h2_prior_ratio` (default 100), in a new hash rule `config.HEALTH_BOUND_FIELDS`:
  hashed only when it differs from `HEALTH_BOUND_DEFAULTS`, so the default and legacy meanfield
  hashes (`meanfield_hashes.json`) do not move. Judgement call: a default-config result cached
  before the guard keeps its hash; it differs from what the guard would give only if one of its
  outer runs failed the check, i.e. its chains escaped.
- `meanfield._sampler_sound` = the step validation without the MC error (h2 growth, the absolute
  bound, HMC divergences / resets / step-size collapse, importance ESS drop); `_healthy` = it plus
  the MC-error bound. Deviation from the plan's "apply `_healthy`": the outer run is held to
  `_sampler_sound`, because the MC error is a property of the sample size that no revert cures
  (the pinned 32-chain solves sit at 2 to 2.5 in every outer run and would all raise).
- `meanfield._outer_run`: an unsound outer-gradient run reverts the trailing inner step (kept in
  `_InnerResult.pending`) and retakes it at half size (counted in `n_backtrack_inner`); at the
  progress-control floor, where halving stops, the step is dropped back to the validated point;
  after max_backtrack retries, or a failure with nothing left to revert, `UnhealthySolveError`
  (a RuntimeError; one line in the CLI) with h2/prior, h2 growth and eta_phihat in the message.
  A sound first run draws nothing extra.
- Tests: `tests/test_outer_run_health.py` (stubbed exact-Gaussian sampler: the bound under HMC
  and importance, relative to the prior; a healthy run kept; an above-bound run reverted and
  retried at half size; growth also checked; MC error alone not; drop at the floor; raise after
  the budget with the ratio; raise with no step); config hash-rule tests; a CLI one-line test;
  the relu replay in `tests/test_relu_tilt.py` now shows that the step-5 tilt (margin +0.19, plane
  -0.007) is the trailing step, rejected and retaken, and that all six accepted tilts are
  normalisable (h2/prior < 10); the analytic saddle tests (HMC, importance) and the weak-tanh
  test assert the bound never binds (largest outer-run h2/prior 1.1 to 1.3, no health
  backtracks).

## Results
- Pins: `hmc_solve_*.pt` (default and legacy) bit for bit unchanged; no fixture or hash moved.
- Tanh exemplar at its notebook defaults, HMC and importance, 80 steps: Phi, Phi_hat and every
  history column bit for bit those of main ae3dd45; largest h2/prior over every judged run 1.33
  (HMC) / 1.32 (importance); no outer-run retries.
- Relu replay, eta_max 0.1, 80 steps: step 5's tilt rejected (one retry; accepted tilt plane
  +0.03, h2/prior 2.3); 0 of the 80 accepted tilts non-normalisable by the plane scan (18 before).
  The growth check alone (max_h2_growth 10) catches step 5 too (growth about 3e3): the absolute
  bound is not what fires in this replay. The solve still runs away (residual about 1, excursions
  to 6, to step 39, then 7 to 80), so `relu-tilt-rule` is not settled by the guard.
- Full suite 1600 passed; `ruff check fct tests notebooks`, `ruff format --check fct tests` clean.
