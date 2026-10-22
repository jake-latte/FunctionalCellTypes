---
type: data-science
complexity: simple
status: in progress
---

<!-- status: todo | in progress | in review | done | blocked | awaiting your confirmation -->

# Library helpers for the phase-transition notebook: tuning curves, MIR ordering, grid plots

Split from `grid-results-notebook` (confirmed by the PI 2026-10-22) so that the parts that do not depend on
`generic-grid-sweep` can be built first. These are pure functions on plain inputs; the notebook wires them
to the sweep later.

## Interfaces
- `fct/tuning.py`
  - `tuning_curves(rates, factors, n_bins) -> TuningCurves`: `rates` (P, N) firing rates of one layer,
    `factors` (P, K) latent values; for each neuron and latent, the mean rate in `n_bins` equal-count bins
    of that latent (other latents marginalised), plus the bin centres. Output arrays `(N, K, n_bins)` and
    `(K, n_bins)`.
  - `select_neurons(mir_per_neuron, n, order, seed) -> indices` with `order in {"random", "mir_ascending",
    "mir_descending"}`.
- `fct/draw.py`
  - `fig_tuning(curves, neuron_indices, mir_per_neuron, latent_names)`: one panel per neuron, one curve per
    latent, common axes, title = neuron index and MIR.
  - `fig_mir_matrix(lambda_r_values, lambda_w_values, mir_by_layer)`: one heat-map per hidden layer,
    lambda_r on one axis, lambda_w on the other (log tick labels), colour = layer-average MIR, shared colour
    scale, NaN cells blank.
  - `fig_mir_vs_ratio(records)`: `records` is a list of `(lambda_r, lambda_w, layer, mir)`; x =
    lambda_r / lambda_w on a log axis, y = MIR, one dot per network, colour by hidden layer.
- `fct/report.py` or a new `fct/grid_report.py`: `mir_grid(records, lambda_r_values, lambda_w_values)
  -> (L, n_r, n_w)` array assembling the matrix from records, NaN where a setting is missing.
- Per-neuron MIR: use what `fct/modularity.py` already exposes (check `modularity.py` for the per-neuron
  quantity the layer-average MIR is a mean of; expose it if it is only internal).

## Acceptance criteria
- A synthetic layer in which neuron 0's rate is a known monotone function of latent 1 plus noise gives a
  tuning curve that is monotone in latent 1 and flat (within noise) in the others; bins are equal-count.
- `select_neurons` orders correctly and is reproducible under a seed; asks for more neurons than exist ->
  clear error.
- `mir_grid` on records with a hole gives NaN there and the right values elsewhere; duplicate records for
  one cell -> error.
- The three figures render from synthetic inputs in tests (CPU, Agg backend).
- Tests CPU only, ruff clean, `docs/design.md` updated.
