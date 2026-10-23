---
type: infrastructure
complexity: complex
status: done
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

## Review findings (fresh reviewer, 2026-10-23; addressed in e815f25 and the merge of main; n_points > P is checked at solve time, not at load, so train-only sweeps with a small P still load)
Correctness of hashing, stale detection, exit codes, resume and the SLURM plumbing verified. Before merge:
1. A solve finished WITHOUT the net is reported `done` and skipped after the net is trained, so the grid
   never gets the net comparison; README says "in either order ... using the trained network if there is one".
   At least document (delete `meanfield/` to redo); better, flag `net_Phi is None` in final.pt when
   `train/final.pt` now exists.
2. The documented ValueError on resuming a no-net partial solve after training has no test; add one.
3. `_write_config_once` now rewrites config.json non-atomically (`save_config` is a plain write_text); use a
   tmp file + `os.replace`. Also say in design.md that the rewrite happens before the stale check.
Follow-ups: the solve hash is spelled out three times (sweep, meanfield, test: import one); `fct_solve.sh`
duplicates the whole sbatch header (document or test that the headers agree); `meanfield.n_points > P` is not
caught at sweep load; minor ordering/annotation nits. Merging needs a mechanical README.md module-table
conflict resolution against main (keep main's notebooks row and this branch's two jobs rows).
