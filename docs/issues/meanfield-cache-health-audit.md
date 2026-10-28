---
type: infrastructure
complexity: simple
status: todo
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Audit a cached mean-field result against the solver's health checks before using it

Raised by the review of `outer-run-health-guard` (2026-10-28). `max_h2_prior_ratio` is hashed
only off its default, and no hash covers the health checks themselves, so a meanfield cache
written before the outer-gradient run was checked keeps its hash and `load_meanfield` /
`run_meanfield` return it, even where the current solver would have retaken or rejected a step
(escaped chains, divergences, resets, an ESS drop) and produced a different result, or raised
`UnhealthySolveError`. Moving the hashes would invalidate every cache, including the large
majority the checks never touch; auditing the cached history finds the affected ones.

## Plan
1. A pure function `audit_history(history, solver_cfg) -> list[str]` in `fct.meanfield`: the
   outer steps and layers whose recorded outer-gradient run fails what `_sampler_sound` checks
   and the history records: `hmc_h2_ratio` > max_h2_prior_ratio, `hmc_reset` > 0,
   `hmc_div` / (n_chains x n_trans_outer) > max_div_frac (HMC), and under importance sampling
   an `hmc_ess_frac` drop below min_ess_drop x that of the step before (the step-to-step h2
   growth needs the inner point before, which the history does not hold: say so, do not guess).
2. `load_meanfield` and `run_meanfield` (on a cache hit) run it: a non-empty audit refuses the
   cache loudly (a `StaleCacheError` subclass naming the steps, layers and values), with an
   explicit opt-out for inspection (e.g. `allow_unhealthy=True`) that logs the findings.
3. The CLI and the sweep status report such a cache as one line / a distinct status
   ("unhealthy"), not as fresh.

## Acceptance criteria
- Unit tests on synthetic histories: a history with one step at h2/prior 3e3 (or one reset, or
  a divergence fraction above the bound) is refused with the step and value in the message; a
  clean history loads unchanged; the opt-out returns it and logs the findings.
- Every existing cache fixture and test cache passes the audit (the pinned solves, the
  exemplar's defaults).
- design.md (cache section) states what is audited, what cannot be (h2 growth), and why the
  hash does not cover it. Full suite green; ruff clean.
