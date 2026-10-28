---
type: infrastructure
complexity: simple
status: in progress
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
