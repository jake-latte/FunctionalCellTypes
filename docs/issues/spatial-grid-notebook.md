---
type: data-science
complexity: complex
status: in review
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Notebook for the spatial grid: one setting's figures, Phi kernels on the test set

`notebooks/build_spatial_grid.py` builds `notebooks/spatial_grid.ipynb` (executed copies go
to `tmp/`, as for the other notebooks): given a sweep ROOT and the axis values of one setting
(`SETTING = {"net.N": 256}`, snapped through `fct.sweep.find_setting_dir`), it loads that
setting's trained net and dataset and shows every figure of `notebooks/spatial.ipynb` (prior
kernels with the PI's eigenvalue-spectrum row, eigenvalue-floor tables, loss curves, example
outputs, first-layer kernel comparison, weights by position, the data kernels A^(l) next to
the latent kernel B = Z Z^T / k, the chosen trials' latents, the Phi^(l) pair grids, tuning by
position, information and rate overlaps), with one change: the Phi kernels (A^(l), the trial
picks and the pair grids) are computed from the activities on the structured TEST dataset of
`linear-teacher-test-dataset` (3^k points, one per sign pattern), not on the training samples.
The readout is always phi(h^L): psi^(l) = phi(h^(l)) on every hidden layer (the "h" case is
removed from the equations and titles). Example outputs and R^2 are shown on the test set too
(it is the held-out set); the loss curves are the training history.

## Proposed plan
- Plotting functions written out in the cells, equation markdown before each figure (the
  convention of `build_spatial.py`); the library computes. Reuse `build_spatial.py`'s cell
  sources where they are unchanged (import the constants rather than copying them, so the two
  notebooks cannot drift), and give `build_spatial.py`'s Phi cells the same test-set change.
- A `settings` cell lists the grid (`sweep list`) so the reader sees what can be selected.
- The notebook is built and executed on CPU and on the T1000 by the PM before merging (device
  tensors must be `.detach().cpu().numpy()` in the cells).

## Acceptance criteria
- `python notebooks/build_spatial_grid.py ROOT --set 'SETTING={"net.N": 8}'` executes on a
  `spatial_tiny` sweep run on a CPU; every figure cell draws (count asserted in the test, as
  `tests/test_build_spatial.py` does).
- The Phi cells use `get_test_dataset`; the A^(l) are (3^k, 3^k); the trial picks spread over
  the sign patterns sorted by the chosen latent.
- The spatial notebook test suite still passes; ruff clean.

## PI confirmation (2026-10-28)
Confirmed. Starts once `spatial-sweep`, `readout-post-activations` and `prior-scaling-convention`
are merged (it needs the spatial presets, the phi(h^L) readout and the final loss).

## Implementation (branch `spatial-grid-notebook`)
- `fct/spatial_grid.py`: `select_setting` (through `find_setting_dir`; a grid setting that was never run
  is status "missing"; a value outside the grid, or an ambiguous request, is a `LookupError`), `settings_table`, `load_grid_run`
  (net, history, training set and the 3^k test set); the loader uses `fct.train.prior_coefficient`. `fct.spatial.latent_kernel` is B = Z Z^T / k.
- `notebooks/build_spatial_grid.py` builds `notebooks/spatial_grid.ipynb`; its cells are `build_spatial.py`'s
  module constants. Cells that need the trained net keep their plotting function at the top level and run
  the rest under `if TRAINED:`, else they repeat the "not trained" note; the prior kernels and floor
  tables need only the config and are always drawn.
- `build_spatial.py` (and so `spatial.ipynb`): `TEST_DATASET` in the train cell; A^(l), the trial picks, the pair
  grids and the example outputs read it (the R^2 and scatter are over all 3^k test points, the panels show a few); `plot_data_kernels` draws B first. `execute_notebook` and
  `build_main` take the name of the root parameter (`ROOT` here).
- Decisions to confirm: the example outputs of `spatial.ipynb` also moved to the test set (the cell is shared;
  the issue only asked for it in the grid notebook); the tuning, overlaps, weights and kernel comparison stay
  on the training samples; a partly trained setting counts as not trained (only `done` is drawn).
- Tests: `tests/test_spatial_grid.py`, `tests/test_build_spatial_grid.py` (executes on a `spatial_tiny` sweep with one
  trained and one never-run setting), additions to `tests/test_build_spatial.py`.
