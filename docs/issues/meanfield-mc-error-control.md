---
type: infrastructure
complexity: simple
status: todo
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
inner index, so a layer has at most `inner` backtracks by then. A step that keeps breaking the
sampler ends the ascent at n_inner_max instead of raising `UnhealthySolveError`.

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
