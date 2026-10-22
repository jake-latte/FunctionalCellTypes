---
type: infrastructure
complexity: complex
status: done
---

<!-- status: todo | in progress | in review | done | blocked | awaiting your confirmation -->

# Sweep runner: expand the 36-setting grid and run one setting per index

The brief's grid (3 tasks x hidden layers {1,3,5} x phi {tanh,relu} x rate_lambda {0,1e-3} = 36 settings) needs one place that defines the settings and runs any one of them, so SLURM array jobs and local checks share the same code path.

## Interfaces
- `fct/sweep.py`: pydantic `SweepConfig` (stored as `ROOT/sweep.json`): presets `tiny` (local pipeline checks) and `full` (cluster); per-task data parameters; N, P, n_epochs, lr, dtype (default float32), monitor_int, monitor_bins, monitor_shuffles, seeds. `expand(sweep) -> list[Setting]` in a fixed order (index i is stable).
- A setting lives in `ROOT/<task>/L{L}_{phi}_r{rate}/`, a normal fct cache dir. `run_setting(sweep, i)` builds its `Config` (sets `data.D_in` from `data.task_dims`), `get_dataset`, `make_net`, and `train` with `make_mir_monitor`; idempotent and resumable.
- CLI: `python -m fct sweep init|list|count|status|run ROOT [--index i]`. `status` reports done / partial / missing per setting (done = final.pt present; partial = checkpoint only).
- `weight_lambda` stays at the library default 1e-3 ; `rate_lambda` is the swept rate regulariser.

## Acceptance criteria
- `expand` gives exactly 36 distinct settings with stable order and distinct cache dirs; `count`/`list` agree.
- `run --index i` on the tiny preset trains setting i and records the monitor history; re-running is a no-op; interrupting and resuming gives the same history as an uninterrupted run.
- `status` is correct for done / partial / missing on constructed directories.
- Invalid index or missing `sweep.json` fails with a clear error.
- Tests are CPU only, including one end-to-end tiny setting per task; ruff clean; `docs/design.md` documents the new interfaces.

## Why complex
New config surface and CLI that the SLURM scripts, notebook and tiny run all depend on; the PI should confirm the grid, presets and directory layout before building.
