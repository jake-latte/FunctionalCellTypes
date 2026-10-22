---
type: data-science
complexity: simple
status: done
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Port the analysis computations to `fct/analyse.py` with cached results

Move the computations of notebook cells 13, 15, 16 and 26 into pure
functions in `fct/analyse.py` (no plotting, no file access), plus thin
cache-or-compute wrappers that store the intermediate results under
`CACHE_DIR/analysis/`.

Why: the plots in `draw.py` must be re-drawable in seconds; the expensive
parts (network evaluation on the test grid, per-layer PCA) are computed once
and cached, keyed on the config.

## Inputs / outputs / interfaces

Exactly the *analyse.py* section of `docs/design.md`: `GridPredictions`,
`grid_predictions`, `LayerPCA`, `layer_pca`, `neuron_maps`, `kernel_scales`,
`get_grid_predictions`, `get_layer_pca`. `grid_predictions` uses
`fct.data.grid_targets` and `fct.data.embed`; `half_width = data_cfg.var / 2`
as in the notebook. `test_error` is the per-point MSE over output
dimensions, reshaped `(n_grid, n_grid)` in `'ij'` order; `train_error` is
per training point. `layer_pca` fits `sklearn.decomposition.PCA` on
`phi(h_grid[:, l]).T` (shape `(N, p²)`) per layer, as the notebook does, and
`similarity = einsum('lnp,LNp->lnLN', components, components) / p²`.
Cached payloads hold tensors only (convert sklearn outputs). Hash is
`cfg.section_hash("data","net","train","analysis")`. Depends on
`fct.data`, `fct.model`, `fct.cache`.

## Acceptance criteria

Synthetic ground truth, CPU, tiny sizes:

- A net whose readout is set by hand so that `Z == y` gives `test_error` and
  `train_error` all below 1e-20; shapes are `(p, p)` and `(P,)`.
- `layer_pca` on rank-2 activations (two orthogonal spatial modes with
  variances 4 and 1 plus 1e-6 noise) recovers explained-variance ratios
  `~ [0.8, 0.2, ~0, ...]` within 1e-3 and components aligned with the planted
  modes; the similarity diagonal equals `1/p²` (unit-norm components).
- `neuron_maps` returns `(N, p, p)` and `neuron_maps(...)[k]` equals
  `phi(h_grid[:, layer, k]).reshape(p, p)`.
- `kernel_scales` of a stack of identity matrices is all ones.
- `get_grid_predictions` / `get_layer_pca` compute once and load on the
  second call; `StaleCacheError` when `analysis.n_grid` changes.
