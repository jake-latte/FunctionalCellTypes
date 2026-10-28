---
type: infrastructure
complexity: complex
status: in review
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

## Implementation (branch `relu-tilt-rule`)
- `fct.hmc.project_tilt` / `normalisability_margin`: for relu the all-active orthant's condition Sigma^-1 + M + lambda_r I > 0,
  by the Gaussian rule's delta-scaled clamp; the margin is lambda_min(Sigma^-1 + M + lambda_r I), documented as a necessary,
  not sufficient, condition. `relu_tilt_spectrum` (the floor) is gone. `tilt_rule` still returns "relu"; `_make_problem`'s
  up-front check is unchanged.
- `fct.hmc.relu_orthant_check(Sigma_inv, M, lambda_r, max_P=20)` -> `OrthantCheck(n_orthants, n_violated, worst_min_eig,
  fraction_violated)`: K_S = (Sigma_SS)^-1 + T_SS > 0 for every active set, batched by padding (15 s at P = 20). Tested against
  a sub-matrix loop, on a hand-built tilt that the rule admits and is not normalisable (the oracle catches it), and on a
  copositive one it flags although normalisable (a pass certifies; a flag does not prove).
- `tests/test_relu_tilt.py`, the exemplar's relu problem (no training): an exact normalisability test by scanning the plane
  of the rank-2 prior; at the analytic tilt (learning tilt at the arc-cosine kernel) the rule cuts a non-normalisable tilt to
  a normalisable one (keeps 25 %, the floor 12 %); oracle 20 / 2^20 orthants flagged (worst -0.003), false alarms; at the
  solver's Phi_hat after 12 outer steps every admitted tilt is normalisable and the oracle passes all orthants.
- Fixtures `hmc_solve_*.pt` regenerated: only their `norm_margin` column changed (the projection never fired in them; Phi,
  Phi_hat and the other columns are bit-identical).

## Acceptance: what holds and what does not
- Holds: the analytic relu checks in `test_hmc.py` / `test_meanfield.py` pass under the default config; the oracle agrees
  that the densities the rule admits on the exemplar are normalisable, or its disagreement (20 / 2^20, and 20389 / 2^20 at
  the end of an 80-step solve from the net start) is shown to be false alarms by the exact plane scan.
- Does not hold, as stated: "the rule keeps the exemplar's relu learning tilt in full". The all-active condition is
  positive definiteness, but normalisability there needs only copositivity; a learning tilt -2c A Y Y^T A of mixed sign
  (centred targets) that is normalisable is still cut. At the reviewer's point (leading eigenvalue about -0.35, floor 31 %)
  the rule keeps 59 %; along solves the share of the learning tilt at the solver's kernel goes 0.24 to 0.58 (floor 0.12 to
  0.37) and is 1.0 only late in some solves. Pinned in `test_the_rule_cuts_a_normalisable_learning_tilt_that_is_not_positive_definite`.
- Does not hold: the exemplar's relu variant (HMC, notebook sizes, lazy or net start, 80 steps; also with eta_max 0.1,
  eta_phi 0.01 or max_rel_step 0.02) and the tiny grid's linear_teacher L1 relu r1e-3 setting (library solver, 16 points, 40
  steps) all end in the tilt tripwire. The projection is idle in the runaway phase (tilt_projected_rel 0) while the Phi_hat
  residual climbs to 5 to 200, and the plane scan says each admitted density is normalisable: this is the solver's iteration
  (`solver-inner-step-control`), not the rule. Re-check both once that item lands.

## For the PI
An exact relu rule would project onto the copositive cone of each orthant, which is NP-hard in general. Two cheaper options if
the remaining cut matters: (a) for a prior of low rank r (the exemplar: r = 2), scan the r-dimensional range of Sigma, exact and
cheap for r <= 3; (b) accept the all-active rule (current) and rely on the backstops. Not decided here.
