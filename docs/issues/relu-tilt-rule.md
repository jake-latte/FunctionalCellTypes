---
type: infrastructure
complexity: complex
status: todo
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Decide the tilt projection for relu layers (the sufficient floor cuts the learning tilt)

Raised by the review of `readout-post-activations` (2026-10-29). The projection is now
activation-aware: none for tanh/sigmoid, the exact Gaussian condition for linear phi, and for
relu the sufficient orthant floor `lambda(M + lambda_r I) >= -(1 - delta) lambda_min(Sigma^-1)`
(the condition `normalisability_margin` certifies). That floor is sufficient, not necessary:
on the exemplar's relu problem (prior_scale 5, lambda_min(Sigma^-1) 0.0997, rate_tilt 0.02) the
learning tilt's leading eigenvalue -0.349 is cut to -0.115 (33 % kept; fixed-point residual
~1), and at the library default prior_scale = 1/lambda_w = 1000 the cut is ~99.8 %, so every
relu setting of the modularity sweep's solve stage ends in `UnconvergedSolveError` (loudly;
before this branch the same solves returned a wrong answer quietly). The exact relu condition
is per orthant (2^P checks); the all-active orthant condition Sigma^-1 + M > 0 is necessary
but not sufficient.

## Options for the PI
1. Accept the sufficient floor and run relu solves only where the prior scale makes it loose
   enough (document the regime; the mean-field constants of `prior-scaling-convention` change
   Sigma and therefore the floor).
2. Use the necessary condition Sigma^-1 + M > 0 for relu, backed by the existing Stein-mismatch
   and ESS backstops (risk: a non-normalisable density caught late).
3. Exact per-orthant check for small P (the exemplar's P = 20 is 10^6 orthants: feasible once
   per outer step with batching, not for the sweep's P), sufficient floor otherwise.
4. Drop the projection for relu and rely on the backstops (what `project_tilt=False` does).

## Acceptance criteria (once decided)
- The analytic relu checks in `tests/test_hmc.py` / `test_meanfield.py` pass under the chosen
  rule with the default config; the exemplar's relu variant and one relu setting of the
  modularity grid solve under the tripwire; the rule and its regime are stated in
  `docs/design.md`.

## PI confirmation (2026-10-29)
Confirmed. The PI did not name an option; the PM's choice, to be corrected if wrong: **option 2
as the rule, option 3 as the oracle.** For relu, `tilt_rule` uses the Gaussian (all-active
orthant) condition `Sigma^-1 + M + lambda_r I > 0` (necessary; keeps the learning tilt), relying
on the existing Stein-mismatch / ESS backstops and the tilt tripwire to catch a non-normalisable
density; and an exact per-orthant check (Schur complement per sign pattern, batched, feasible
for P <= ~20) is implemented as a diagnostic / test oracle only, used to validate the rule on
the exemplar's relu problem (P 20) and in tests. Acceptance: the analytic relu checks pass under
the default config; the exemplar's relu variant and one relu setting of the modularity grid
solve under the tripwire; the oracle agrees with the rule on the exemplar problem (or the
disagreement is reported, with the fraction of orthants violated); the rule, its regime and the
oracle's cost are stated in `docs/design.md`. Starts once `prior-scaling-convention` is merged.
