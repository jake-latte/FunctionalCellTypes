---
type: infrastructure
complexity: simple
status: in progress
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Structured test dataset for the linear teacher: the 3^k latent sign patterns

The linear teacher trains on P random latents z ~ U[-a, a]^k. For looking at a trained
network (kernels, example outputs, tuning) we want a fixed, structured test set instead: one
sample per point of the lattice {-a, 0, a}^k, so a teacher with k latents gives 3^k test points,
each latent taking -a, 0 or a (a = `latent_range`). Inputs and targets are the same linear maps
as the training set: x = z W_X^T, y = z W_Y^T with the W_X, W_Y of the dataset being tested
(`reference.extras`), so the test points are held-out data of the same teacher.

## Inputs / outputs / interfaces
- `fct/tasks.py`: `test_dataset(cfg: DataConfig, reference: Dataset) -> Dataset` (pure; next to
  `resample`, which it resembles). `cfg.task` must be `"linear_teacher"` (ValueError otherwise).
  The returned `Dataset` has `x` (3^k, m), `y` (3^k, m), `factors` {"z0".."z{k-1}"} each (3^k,),
  `groups` singleton, `extras` = the reference's; same device and dtype as `reference.x`.
  Row order: the Cartesian product with z0 slowest and the last latent fastest, values in the
  order (-a, 0, a), so row i is reproducible from i alone (document this in the docstring).
- `fct/data.py`: `get_test_dataset(cache_dir, cfg: Config) -> Dataset` loads the training
  dataset through `get_dataset` and returns `test_dataset(cfg.data, ...)`. It does not need its
  own cache file: 3^k points are cheap to rebuild. Keep I/O (this function) apart from the pure
  generator.
- No config field changes: nothing enters any hash. (A running SLURM sweep imports `fct` from
  this tree, so this item must not change any existing behaviour or hash.)

## Acceptance criteria
- For k = 2, a = 1: 9 rows, latents exactly the 9 sign patterns in the documented order; for
  k = 3: 27 rows; every latent column takes each of the three values exactly 3^(k-1) times.
- `x` and `y` equal `z @ W_X.T` and `z @ W_Y.T` with the reference's maps (exact to float
  round-off), on the reference's device and dtype; `extras` is the reference's.
- A non-linear-teacher config raises ValueError naming the task; a reference without
  `extras["W_X"]` raises.
- `get_test_dataset` on a cache directory with a dataset returns the same thing as the pure
  function; nothing is written to disk by it beyond what `get_dataset` writes.
- Tests CPU only, synthetic fixtures, `uv run pytest -q` and `ruff check`/`format --check` clean.
