---
type: infrastructure
complexity: complex
status: done
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Scale the rate penalty by 1/(2 beta) so that rate_lambda is the notes' r

PI request (2026-11-06). Today `compute_loss` (`fct/train.py:57-95`) adds
`rate_lambda * rate_l2` to $\mathcal L$, so $\beta\mathcal L$ carries
$\beta\,$`rate_lambda`$\sum\phi^2$ and the notes' coefficient is
$g^2 = 2\beta\,$`rate_lambda` (`fct/meanfield.py:272`, `rate_tilt`). Make the
$1/(2\beta)$ explicit, exactly as `prior_coefficient` does for the weight
prior (`fct/train.py:45-54`, `LOSS_VERSION = 4`), so that $\beta\mathcal L$
carries $\frac{\text{rate\_lambda}}{2}\sum\phi^2$ and `rate_lambda` is the
notes' $r$ (after `rate-penalty-latex-r`).

## What to change
- `rate_coefficient(tc) = rate_lambda / (2 beta)` with $\beta = 1$ under Adam
  (the `prior_coefficient` rule); `compute_loss` uses it; the history key
  `rate_l2` stays the raw $\sum\phi^2$.
- `LOSS_VERSION = 5` (`fct/config.py:29`) with the history comment "5: rate
  penalty scaled by 1/(2 beta)".
- `meanfield.py:272`: `rate_tilt = rate_lambda`; `lrtune.py:428` and any other
  caller of the raw coefficient.
- `docs/design.md` ("Prior scaling convention", the cache-layout paragraph on
  `LOSS_VERSION`, `train.py`, `meanfield.py` rows); the README if it states
  the loss.
- Tests: `tests/test_train.py`, `test_prior_convention.py` (a test that
  $\beta\mathcal L$'s rate term is `rate_lambda/2` times $\sum\phi^2$ for
  $\beta \in \{0.25, 4\}$ and Adam); `pinned_runs.py` (`LAMBDA_R` is passed to
  `solve` directly: keep the pinned solves bit for bit by passing the value
  that gives the same `rate_tilt`, or re-pin and say why in the commit).

## Hash rule (PI confirmed the recommendation 2026-11-07)
`LOSS_VERSION` is folded into every hash that includes `train`, so version 5
would mark every existing train/meanfield cache stale, including the PI's
running `phase_transition` sweeps (`pt_sweep*`, `sweep_cache`; Adam,
`train.rate_lambda` axis `[0, 0.001]`). The numbers of a `rate_lambda = 0`
run do not change. **Recommended:** fold version 5 into the hash only when
`rate_lambda > 0` (the pattern of fields dropped at their default), so
`rate_lambda = 0` caches stay valid and only runs whose loss really changed
(by a factor $1/(2\beta)$, i.e. $1/2$ under Adam) become stale. The
alternative is the plain bump, which invalidates the whole sweep.

Also: `fct/linear_theory.py` (`theory_rate`) carries today's relation
$r = 2\beta\,$`rate_lambda`; it becomes `rate_lambda`, and the drift-guard test
`test_theory_rate_is_the_rate_part_of_the_training_loss` must pass unchanged
(it compares the theory against the trainer's own loss). Starts after
`whitened-spatial-prior` merges (both edit `fct/train.py` and `fct/config.py`'s
hash rule); fold version 5 into the shared omission/hash table that branch
introduces.

Also update the LaTeX statements of today's relation, which become wrong when
this lands: `fct_experiment_plan.tex` (the dictionary row and every
`rate_lambda = r/(2 beta)` statement, and the `--set rate_lambda=` values of P2
and P4: `0.2` for r = 0.1 becomes `0.1`, `2.0` for r = 1 becomes `1.0`) and
`linear_solution_limits.tex` (the "what fct does not yet match" item). The
acceptance criterion "G = r I and the code's loss show the same coefficient"
of `rate-penalty-latex-r` is met only once this item lands.

## Acceptance criteria
- For every optimiser and $\beta$: `beta * loss` at fixed weights equals
  `0.5 * beta * N * mse + 0.5 * weight_lambda * form + 0.5 * rate_lambda * sum(phi^2)`.
- `rate_tilt == rate_lambda` in the solver's problem; the pinned solves are
  unchanged (or re-pinned with the stated reason).
- A `rate_lambda = 0` config keeps its train hash across the change (test
  with a fixture hash); a `rate_lambda > 0` config's hash changes and
  `exp status` reports such a setting `stale`.
- Full suite and `ruff` pass.
