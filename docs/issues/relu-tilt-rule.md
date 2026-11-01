---
type: infrastructure
complexity: complex
status: blocked
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
relu setting of the modularity experiment's solve stage ends in `UnconvergedSolveError` (loudly;
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
   per outer step with batching, not for the experiment's P), sufficient floor otherwise.
4. Drop the projection for relu and rely on the backstops (what `project_tilt=False` does).

## Acceptance criteria (once decided)
- The analytic relu checks in `tests/test_hmc.py` / `test_meanfield.py` pass under the chosen
  rule with the default config; the exemplar's relu variant and one relu setting of the
  modularity experiment solve under the tripwire; the rule and its regime are stated in
  `docs/design.md`.

## PI confirmation (2026-10-29)
Confirmed. The PI did not name an option; the PM's choice, to be corrected if wrong: **option 2
as the rule, option 3 as the oracle.** For relu, `tilt_rule` uses the Gaussian (all-active
orthant) condition `Sigma^-1 + M + lambda_r I > 0` (necessary; keeps the learning tilt), relying
on the existing Stein-mismatch / ESS backstops and the tilt tripwire to catch a non-normalisable
density; and an exact per-orthant check (Schur complement per sign pattern, batched, feasible
for P <= ~20) is implemented as a diagnostic / test oracle only, used to validate the rule on
the exemplar's relu problem (P 20) and in tests. Acceptance: the analytic relu checks pass under
the default config; the exemplar's relu variant and one relu setting of the modularity experiment
solve under the tripwire; the oracle agrees with the rule on the exemplar problem (or the
disagreement is reported, with the fraction of orthants violated); the rule, its regime and the
oracle's cost are stated in `docs/design.md`. Starts once `prior-scaling-convention` is merged.

## Implementation (branch `relu-tilt-rule`)
- `fct.hmc.project_tilt` / `normalisability_margin`: for relu the all-active orthant's condition Sigma^-1 + M + lambda_r I > 0,
  by the Gaussian rule's delta-scaled clamp; the margin is lambda_min(Sigma^-1 + M + lambda_r I), documented as neither
  necessary nor sufficient. `relu_tilt_spectrum` (the floor) is gone. `tilt_rule` still returns "relu"; `_make_problem`'s
  up-front check is unchanged.
- `fct.hmc.relu_orthant_check(Sigma_inv, M, lambda_r, max_P=20)` -> `OrthantCheck(n_orthants, n_violated, worst_min_eig,
  fraction_violated)`: K_S = (Sigma_SS)^-1 + T_SS > 0 for every active set, batched by padding (15 s at P = 20). Tested against
  a sub-matrix loop, on a hand-built tilt that the rule admits and is not normalisable (the oracle catches it), and on a
  copositive one it flags although normalisable (a pass certifies; a flag does not prove).
- `tests/test_relu_tilt.py`, the exemplar's relu problem (no training): a normalisability test by scanning the plane of the
  rank-2 prior (with the off-plane bound asserted); at the analytic tilt (learning tilt at the arc-cosine kernel) the rule
  cuts a non-normalisable tilt to a normalisable one (keeps 25 %, the floor 12 %); oracle 20 / 2^20 orthants flagged (worst
  -0.003), false alarms; a solve with eta_max 0.1 admits a non-normalisable tilt at step 5 (margin +0.19, plane -0.007,
  -log p falling as -r^2 along a ray, Stein mismatch 0.06); the default-step solve's first 12 tilts are normalisable.
- Fixtures `hmc_solve_*.pt` regenerated: only their `norm_margin` column changed (the projection never fired in them; Phi,
  Phi_hat and the other columns are bit-identical).

## Acceptance: what holds and what does not
- Holds: the analytic relu checks in `test_hmc.py` / `test_meanfield.py` pass under the default config.
- Partly: the oracle and the rule were compared on the exemplar. The oracle's flags on the projected analytic tilt (20 /
  2^20) and at the end of an 80-step net-start solve (20389 / 2^20) are false alarms by the plane scan; but the rule admitted
  non-normalisable tilts in the eta_max 0.1 solve (step 5; 17 of steps 55 to 79), where the oracle flags 619367 / 2^20.
- Does not hold, as stated: "the rule keeps the exemplar's relu learning tilt in full". The all-active condition is
  positive definiteness, but normalisability there needs only copositivity; a learning tilt -2c A Y Y^T A of mixed sign
  (centred targets) that is normalisable is still cut. At the reviewer's point (leading eigenvalue about -0.35, floor 31 %)
  the rule keeps 59 %; along solves the share of the learning tilt at the solver's kernel goes 0.24 to 0.58 (floor 0.12 to
  0.37) and is 1.0 only late in some solves. Pinned in `test_the_rule_cuts_a_normalisable_learning_tilt_that_is_not_positive_definite`.
- Does not hold: the exemplar's relu variant (HMC, notebook sizes, lazy or net start, 80 steps; also with eta_max 0.1,
  eta_phi 0.01 or max_rel_step 0.02) and the tiny experiment's linear_teacher L1 relu r1e-3 setting (library solver, 16 points, 40
  steps) all end in the tilt tripwire, for two different reasons:
  - rule-driven (eta_max 0.1, lazy start): the all-active rule admitted non-normalisable tilts and the Stein / ESS / h^2
    backstops did not catch them. At step 5 (margin +0.19, plane -0.007) the chains escape (h^2/prior 3e3), the Stein
    mismatch jumps to 1.14 and the residual goes 1.3, 4.5, 16, 63, 92; from step 55 to 79 the tilt rides the delta floor,
    the projection fires every step (also every step from 20 to 70), and 17 of the 25 admitted densities are not
    normalisable while the backstops flag nothing. A tilt on the floor leaves the all-active direction a precision of about
    0.005 (std about 14 against the prior's about 3), on which HMC diverges. `solver-inner-step-control` will not fix this
    part;
  - step control (eta_phi 0.01 lazy and eta_max 0.1 net start; the default-step solves, whose first 40 steps admitted only
    normalisable tilts): Stein mismatch near 1 with a normalisable density, residual climbing to 30 to 200, with the
    projection idle in the first two but firing on about half of steps 18 to 60 of the default net-start solve. Re-check
    these once `solver-inner-step-control` lands.

## For the PI
An exact relu rule would project onto the copositive cone of each orthant, which is NP-hard in general. Options: (a) for a
prior of low rank r (the exemplar: r = 2), scan the r-dimensional range of Sigma, exact and cheap for r <= 3; (b) keep the
all-active rule and its backstops, knowing they missed non-normalisable tilts on the exemplar; (c) a stricter relu margin
than delta (keep tilts off the floor), which cuts more. Not decided here.

## Merged with status `blocked` (PM, 2026-10-29)
The rule (all-active orthant condition), the per-orthant oracle, the plane-scan diagnostic and
the pinned record are merged. The item is blocked on the PI's decision about the rule, because
the review refuted the bet behind option 2: the rule admitted non-normalisable tilts in the
eta_max 0.1 solve and the backstops did not catch them. PM recommendation: keep the rule and add
a runtime guard in `solver-inner-step-control`'s backtracking that rejects an inner step whose
chains escape the prior scale (absolute h2/prior bound) and shrinks the tilt's negative part;
fallback: restore the sufficient floor as the default. "Relu solves under the tripwire" is
re-tested after both land.
