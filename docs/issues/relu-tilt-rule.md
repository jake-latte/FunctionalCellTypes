---
type: infrastructure
complexity: complex
status: awaiting your confirmation
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
