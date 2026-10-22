---
type: infrastructure
complexity: simple
status: done
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Port task data generation to `fct/data.py` with a cached dataset

Move `fourier_embedding` and `position_task` from notebook cells 7 and 13 into
`fct/data.py`, seeded and device/dtype-explicit, plus the test-grid helpers
the analysis uses and a cache-or-compute `get_dataset`.

Why: the dataset is the first artefact of the workflow; training, the
mean-field solver and the example plot all consume it, and it must be
reproducible from `config.json` alone.

## Inputs / outputs / interfaces

Implement exactly the *data.py* section of `docs/design.md`:
`Dataset`, `fourier_embedding`, `make_dataset`, `embed`, `grid_targets`,
`get_dataset`. Randomness comes from a `torch.Generator` seeded with
`cfg.seed` (both the latent `z` and the random `U` for the linear embedding).
`Dataset.period` is `(y.max() - y.min())` as a Python float for the Fourier
embedding, `None` otherwise. `get_dataset` stores `x, y, Q, period` in
`CACHE_DIR/data/dataset.pt` via `fct.cache.save_artefact` keyed on
`cfg.section_hash("data")`.

## Acceptance criteria

- `make_dataset` is deterministic for a given seed and differs for another.
- Shapes: `x (P, D_in)`, `y (P, d)`; linear: `Q` has orthonormal columns
  (`Q.T @ Q ≈ I_d`) and `x == y @ Q.T`; fourier: `Q is None`, `period` set.
- `fourier_embedding`: raises on odd `D`; each `(cos, sin)` pair has unit
  norm per row; the wavevectors returned are the `D//2` lowest-|k|² lattice
  vectors with the first non-zero component positive, none equal to zero.
- `embed(dataset.y, dataset, cfg)` reproduces `dataset.x` (both embeddings).
- `grid_targets(h, n)` has shape `(n², 2)`, covers `[-h, h]²`, and its
  reshape to `(n, n)` matches `torch.meshgrid(..., indexing="ij")`.
- `get_dataset` computes on the first call, loads on the second (no
  regeneration: patch/spy `make_dataset` or compare file mtimes), and raises
  `StaleCacheError` if `config.json`'s data section changed.
- Sample covariance of `y` (P=20_000) is within 10 % of `var * cov`.

## Implementation notes
- `make_dataset(cfg: DataConfig, device, dtype)`; `get_dataset(cache_dir, cfg: Config)` resolves device/dtype with `fct.config.resolve_device/resolve_dtype`. Saving `Q=None` / `period=None` is fine (None is weights_only-loadable).
- `cov` in the config is `list[list[float]] | None`; validate that it is `d x d` and raise ValueError otherwise (principle 9).
- The Fourier period in `position_task` is `(y.max() - y.min())` computed from the generated `y` — keep that, store it as a float on the Dataset so `embed` reuses it for test points.
- Keep functions small; tests use `tmp_path`, CPU, float64, tiny P except the covariance test.
