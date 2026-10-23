---
type: data-science
complexity: complex
status: done
---

<!-- status: todo | in progress | in review | done | blocked | awaiting your confirmation -->

# Exemplar notebook: task, SGLD training, mean-field solve with live diagnostics

Confirmed by the PI 2026-10-22. Linear teacher task.

## Proposed plan
A source notebook (built by script, executed copy not committed) with sections, each a few cells:
1. **Task / network hyperparameters.** One cell at the top with every task and net setting (P, m, k, N, L, phi, g, seeds) and the loss weights.
2. **Training.** One cell for the training hyperparameters (SGLD: beta, lr, n_epochs, ...) and one that trains through the normal `fct` cache path (resumable) and plots the loss curves, the outputs, and the empirical kernels at the end of training.
3. **Mean-field solve.** One cell for solver hyperparameters (sampler hmc / importance, n_outer, n_chains, ...), one that runs the solver verbosely with *live* diagnostic plots refreshed every few outer steps: layer kernels Phi_l and Phi_hat_l, solver health (residuals, step sizes, backtracking, theory-vs-net train MSE), sampler health (HMC acceptance / eps / divergences / ESS, or importance ESS and max weight). A final cell plots the solved kernels next to the empirical ones.
Live plotting is built on `solve`'s existing `log` hook plus an optional per-outer-step callback added to `solve` (it must not change results). Plot functions are library functions (tested), not notebook code.

## Acceptance criteria
- The notebook builds and executes end to end on a tiny configuration on CPU, once with each sampler.
- The callback is off by default and changes no solver output (regression test).
- Plot helpers have tests that render from synthetic histories.
- Tests CPU only, ruff clean.

## Decisions made in review
Review of the first version found it mergeable after these fixes, all made:
- **Theory vs net scale (medium).** At the original defaults the solved kernels were 18 times too large (top eigenvalue 712 against 39). Cause: the library's mean-field constants (`prior_scale = 1/weight_lambda`, `alpha = weight_lambda P d_out / beta`, `rate_tilt = rate_lambda`) describe the SGLD stationary law `exp(-beta L)` only at `beta = 1/2`; under it each weight has variance `1 / (2 beta weight_lambda)`. The training was equilibrated (loss and parameter step flat, step msd = 2 lr / beta). `fct.exemplar.sgld_theory_constants` gives the solver the matching constants for SGLD (prior_scale = 1/(2 beta weight_lambda), alpha = 2 weight_lambda P d_out, rate_tilt = 2 beta rate_lambda); the library defaults are untouched (they feed existing hashes and results), so **the PI should decide whether `run_meanfield` should apply this mapping for SGLD nets**. With it, scales agree (top eigenvalue 11 against 13 at P = 20) and the theory's train MSE (0.030) equals the net's kernel-GP mean predictor (0.029).
- **Defaults** are now P = 20 (not 24: importance sampling raises at P = 24) and HMC 5 / 10 transitions per run (smallest ESS fraction 0.14 instead of 0.0016). The fixed-point residual falls (1.0 to 0.74 HMC, 0.72 importance) but does not reach its Monte Carlo floor in 40 steps; the notebook text and a printed summary (residual trend with its direction, ESS and where it was smallest, eigenvalue and diagonal scales of theory and net) say so. Process history and the wrong wall-time claims are gone.
- The kernel comparison uses a colour scale per panel. All three live figures count layers from 1 (the full dashboard `fig_solver_history` keeps 0-based layers).
- **The shared dashboard changed:** `fig_solver_history` gained two panels, "backtracking: inner (lines), outer (black)" and "eigenvalues raised to the PD floor" (`n_pd_clamped`), and a single-point line now has a marker. They appear in the CLI's solver figure too.
- `LiveView` holds the last log lines and prints them under a frame; `solve_exemplar(on_failure=...)` (the notebook passes `LiveView.flush`) prints them when the solve raises, e.g. an `InadequateSamplerError` before the first frame.
- Deduplicated against the spatial notebook: the builder reuses `build_spatial.build_main`, `set_parameters`, `execute_notebook`; `cache_root_from_env` moved to `fct.cache`; `exemplar_config` / `train_exemplar` wrap `spatial_view.make_config` / `train_in`. `exemplar_config` lost `log_int` / `save_int` (derived there).
- Tests: the reload test asserts the "loaded finished solve" text, the dashboard test lists literal panel titles, `on_outer` is checked not to fire for restored steps, a failing solve is tested, `example_outputs_of` takes P from the dataset, the `on_outer` docstring notes that plotting time enters `time`.
