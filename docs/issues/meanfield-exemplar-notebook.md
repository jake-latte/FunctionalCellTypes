---
type: data-science
complexity: complex
status: in review
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
