---
type: infrastructure
complexity: simple
status: in review
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# The prior term is always `lambda_w / (2 beta)` times the prior quadratic form (`beta = 1` under Adam)

PI decision (2026-10-28, under `spatial-sweep`): the weight prior enters the loss as
`lambda_w / (2 beta) * tr(M_post W M_pre W^T)` per layer (the iid prior K = M = I included, i.e. the
plain L2 becomes `lambda_w / (2 beta) * sum W^2`), where `beta` is the SGLD inverse temperature
(`train.beta`) and is taken as 1 whenever the optimiser is Adam. Today the library uses
`lambda_w * tr(M_post W M_pre W^T)` (see `fct.train.make_weight_penalty`), and the mean-field
constants were derived for that. Under the new convention the SGLD stationary law `exp(-beta L)`
gives `vec(W) ~ N(0, K / lambda_w)` for every beta, so `prior_scale = 1 / lambda_w` (the solver's
current default) becomes exact, while the data and rate terms are multiplied by beta in the law
(`fct.exemplar.sgld_theory_constants` and the "SGLD theory constants" decision in `docs/status.md`
must be re-derived under the new convention and the result written into `docs/design.md`).

## Inputs / outputs / interfaces
- `fct/train.py`: `make_weight_penalty` (and the SGLD stability check, which bounds `lr * weight_lambda`),
  `LOSS_VERSION` bumped so every cached run is rejected as stale.
- `fct/meanfield.py` constants and docstrings; `fct/exemplar.py::sgld_theory_constants`.
- Pinned fixtures (`tests/pinned_runs.py`, `tests/test_meanfield_pins.py`) regenerated.
- The notebooks' equation cells (`spatial`, `exemplar`, `phase_transition`, `modularity`) and the
  Latex equations that state the loss; `docs/design.md`.

## Acceptance criteria
- A test computes the penalty by hand for a tiny net under Adam (`lambda_w / 2 * sum W^2`, and the
  spatial form with a hand-built M) and under SGLD with `beta != 1`, and it equals `make_weight_penalty`.
- An old `config.json` / cache made under `LOSS_VERSION` 3 raises `StaleCacheError`.
- `sgld_theory_constants` and the solver's defaults agree for every beta (a test on the linear L = 1
  net's closed-form weight variance, as the exemplar notebook does), or the remaining discrepancy is
  documented with its reason.
- Tests CPU only; ruff clean; `docs/design.md` states the convention with the equation.

## Resolution

All `weight_lambda` values are stated under the new convention. An old value maps to the same
effective loss by x 2 beta (SGLD) or x 2 (Adam, beta = 1): the library default is 2e-4 (was 1e-4), the
notebooks take the old-convention loss coefficient as `PRIOR_COEFF` (spatial 3e-3, exemplar 1e-2) and derive `weight_lambda = 2 beta PRIOR_COEFF` (beta = 1 under Adam).
The `spatial_tiny` / `spatial_n_scan` presets (3e-3 -> 60) are updated on the `spatial-sweep` branch;
the `phase_transition` preset's `weight_lambda` axis (log-spaced 1e-6..1) is NOT rescaled.

The mean-field solver's defaults follow the network (`fct.meanfield.network_constants`, `resolve_beta`):
prior_scale = 1/weight_lambda, alpha = weight_lambda P d_out / beta, rate_tilt = 2 beta rate_lambda,
learning_prefactor = weight_lambda / 2, beta = train.beta (SGLD) or 1 (Adam); explicit `meanfield`
fields still win, so `exemplar_config` no longer overrides them.
