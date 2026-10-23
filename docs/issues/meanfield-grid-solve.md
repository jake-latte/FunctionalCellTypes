---
type: infrastructure
complexity: complex
status: in review
---

<!-- status: todo | in progress | in review | done | blocked | awaiting your confirmation -->

# Train or solve any initialised grid, locally or on the cluster

PI brief (Mean Field 2): given an initialised grid search (trained or not), one can either train the networks or solve the corresponding saddle points; cluster compatible. Confirmed by the PI 2026-10-22.

## Proposed plan
- `fct/sweep.py`: `run_setting(..., stage="train" | "solve")`. Solve for index i loads (or builds) the dataset, loads the trained net if a final one exists (otherwise runs with `net=None`: `solve` already accepts that; the net-comparison history keys stay absent), and calls `run_meanfield` into `ROOT/<setting>/meanfield/`. It resumes from a solver checkpoint and skips a finished one.
- Solver hyperparameters of a sweep: `SweepConfig.meanfield` (solver fields of `MeanFieldConfig`, defaults from the library), so the same `sweep.json` defines training and solving; `rate_tilt` follows the library default (`rate_lambda`).
- `setting_status(..., stage)` reports done/partial/missing/stale per stage; `sweep status ROOT --stage solve`; `sweep run ROOT --index i --stage solve`; `sweep list`/`count` unchanged.
- SLURM: `jobs/fct_solve.sh` (array job; one GPU per task; `--stage solve`), written but never submitted; dry-run prints the sbatch lines. `jobs/fct_sweep.sh` gets `--stage train` explicitly.
- The notebook helpers (`report.py`) are not changed here; they only need the solve status to exist.

## Acceptance criteria
- On a tiny grid on CPU: train-only, solve-only (no trained net), and train-then-solve all work; interrupted solves resume; reruns are no-ops.
- Status per stage is correct on constructed directories, including stale (hash mismatch) and partial.
- Existing legacy sweeps keep working; the new script passes `bash -n` and a dry run.
- Tests CPU only, ruff clean, `docs/design.md` and README updated. **No job is submitted.**
