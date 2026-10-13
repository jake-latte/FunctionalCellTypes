---
type: infrastructure
complexity: simple
status: done
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Constrain config fields at the boundary and add ruff to the dev group

Add pydantic field constraints to the config models so that impossible
hyperparameters fail at `load_config` time with a clear message, and add
`ruff` to the `dev` dependency group so `uv run ruff` works.

Why: reviewers found that `log_int=0` or `save_int=0` reach the training loop
as a bare `ZeroDivisionError`, `L=0` only fails inside `MLP`, and several
other fields (`P`, `D_in`, `d`, `N`, `n_chains`, `n_grid`, ...) accept zero
or negative values that would surface as obscure tensor errors deep in a run.
The config is the boundary (principle 9); constraints belong there. Every
reviewer also had to fall back to `uvx ruff` because the tool is not a dev
dependency (principle 1).

## Inputs / outputs / interfaces

`fct/config.py` only: use `pydantic.Field(ge=1)` / `gt=0` on the fields
listed below. No signature or default changes. `pyproject.toml`: `ruff` in
`[dependency-groups].dev`; `uv.lock` regenerated with `uv lock`.

- `DataConfig`: `P, D_in, d >= 1`; `var > 0`; `D_in` even when
  `embedding == "fourier"` (model validator with a message saying why).
- `NetConfig`: `N, L >= 1`; `g > 0`.
- `TrainConfig`: `n_epochs >= 0`; `lr > 0`; `weight_lambda, rate_lambda >= 0`;
  `log_int, save_int >= 1`.
- `MeanFieldConfig`: `n_points, n_outer, n_inner_max, n_chains, n_trans,
  n_trans_outer, n_leap, warmup_trans, n_init_samples, log_int, save_int,
  max_backtrack >= 1`; `beta, eps0, target_accept, adapt_rate, eta_phi,
  eta_phihat, eta_max, eta_grow, inner_tol, inner_tol_mc, max_rel_step,
  pd_floor_rel, inv_floor_rel, jitter_rel, learning_prefactor,
  max_h2_growth, max_mc_err > 0`;
  `0 < target_accept < 1`; `0 <= tilt_floor < 1`; `max_div_frac,
  eps_collapse, eta_min_rel >= 0`. `prior_scale`, `alpha` are `> 0` when set;
  `rate_tilt` is `>= 0` when set.
- `AnalysisConfig`: `n_grid, n_components, n_cols >= 1`; `plot_layer >= 0`.

## Acceptance criteria

- `default_config()` still validates and its JSON round-trip is unchanged
  (existing tests pass).
- Each constraint above has a parametrised test showing the offending value
  raises `pydantic.ValidationError` whose message names the field.
- `DataConfig(D_in=63, embedding="fourier")` raises; `D_in=63,
  embedding="linear"` is accepted.
- `uv run ruff check fct tests` and `uv run ruff format --check fct tests`
  run.
- `section_hash` values of `default_config()` are unchanged by this work
  (constraints do not alter the dumped JSON).
