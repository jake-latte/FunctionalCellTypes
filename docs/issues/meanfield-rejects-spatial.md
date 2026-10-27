---
type: infrastructure
complexity: simple
status: in review
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# The mean-field solver refuses a config with a spatial prior

The saddle-point solver (`fct.meanfield`) knows only the iid weight prior. Until it takes the
spatial kernels, running it on a config whose `spatial` section is set must be an error, not a
silent solve of the wrong problem. The PI asks that the solver "raises an error when running on
a config with SPATIAL=True".

## Inputs / outputs / interfaces
- `fct.meanfield.run_meanfield(cfg, ...)` raises `ValueError` (message: the solver has no
  spatial prior; unset `spatial` or wait for the solver item) when `cfg.spatial is not None`,
  before touching the cache or the sampler.
- `fct.meanfield.net_kernels` / `load_meanfield` are unaffected (they read, they do not solve).
- `fct.sweep.run_setting(..., stage="solve")` therefore fails loudly for a spatial setting;
  `setting_status(stage="solve")` of such a setting stays "missing" (no artefact), which is the
  truthful state. `python -m fct solve CACHE_DIR` and `sweep run --stage solve` surface the
  ValueError as a CliError (one line, no traceback), like a SweepError. Check how `cmd_solve`
  and `cmd_sweep_run` map exceptions and extend them minimally.
- Nothing else changes: a running SLURM sweep imports `fct` from this tree, so no behaviour or
  hash of a non-spatial config may move.

## Acceptance criteria
- `run_meanfield` with a tiny spatial config raises ValueError before any file is written
  (the `meanfield/` directory is absent afterwards); with `spatial=None` it behaves as before
  (an existing test covers this; do not weaken it).
- The CLI `solve` and `sweep run --stage solve` on a spatial setting exit non-zero with the
  one-line message (test through `fct.cli.main`, tiny CPU config).
- Tests CPU only, `uv run pytest -q`, ruff check and format clean.
