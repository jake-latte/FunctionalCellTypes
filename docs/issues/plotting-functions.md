---
type: infrastructure
complexity: simple
status: in review
---

# Port all figures to `fct/draw.py` as axis-accepting plot functions

Move the plotting of notebook cells 13, 15, 16, 21 (`plot_history`) and 26
into `fct/draw.py`, split into single-panel `plot_*` functions that draw
into a given `ax` and `fig_*` functions that assemble them, plus the
result-independent example figure of the generated task.

Why: the CLI must save every figure from cached results without a display,
and single panels must be reusable inside larger paper figures
(principle 8).

## Inputs / outputs / interfaces

Exactly the *draw.py* section of `docs/design.md`. Every `plot_*` has
signature `(..., ax: Axes | None = None) -> Axes`, creates a figure only if
`ax is None`, never calls `plt.show()`. `fig_*` return a `Figure`. Keep the
notebook's visual choices (colormaps, `LogNorm(vmin=1e-6)`, extents,
`_LAYER_COLORS`, panel titles and y-scales of `plot_history`). Inputs are
the dataclasses from `fct.analyse` and the history dicts from `fct.train` /
`fct.meanfield` (history values may be lists or numpy arrays: convert with
`np.asarray`). `fig_example(dataset, cfg)` shows the target positions `y`
coloured by index and the embedding `x` of a few example points (one row per
point): the "example trial" figure that needs no training result.
`fig_kernels(Phi_theory, Phi_net)` draws `L` rows x 2 columns of kernel
matrices with a shared colour scale per row.

Depends on `fct.analyse`, `fct.data`. Use the `Agg` backend in tests.

## Acceptance criteria

- Each `plot_*` called with `ax=None` returns an `Axes` on a new figure;
  called with a provided `ax` draws into it (the returned object *is* that
  `ax`) and creates no new figure (`plt.get_fignums()` unchanged).
- `fig_learning`, `fig_representation`, `fig_neuron_maps`, `fig_example`,
  `fig_kernels`, `fig_solver_history`, and `plot_training_history` each
  produce a figure from small synthetic inputs and `savefig` to a temporary
  PNG without error; the PNG is non-empty.
- `fig_representation` has `L + L^2` axes; `fig_neuron_maps` has
  `ceil(N / n_cols) * n_cols` axes with the surplus turned off;
  `fig_solver_history` includes exactly the panels whose keys are present in
  the history (the `align_net` panel only when a net was used).
- `fig_solver_history` handles a history whose per-layer entries are lists
  (as loaded from the cache) and whose `train_mse_net` is `None`.
- `plt.show` is never called (grep the module).
