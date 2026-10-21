# FunctionalCellTypes modularity study: status and how to resume

Written 2026-10-20 by the PM agent. Read this, then `todo.md` (the PI's brief; untracked in the PI's
tree), `docs/design.md` (the interface contract) and `docs/issues/`.

## The task (todo.md, summarised)
Train feed-forward nets (the `fct` MLP) on three tasks and measure whether hidden neurons become
modular for the task factors. Tasks: (1) linear teacher, (2) object-action (G x G grid, agent cell +
object map -> optimal action toward the nearest object), (3) what-where (9x9 grid of 3x3 motifs).
Grid: hidden layers {1,3,5} x phi {tanh, relu} x rate regulariser {0, 1e-3} per task (36 settings);
N, P, steps tunable. Track the modularity index MIR per hidden layer through training. Deliver a
notebook plotting per task and setting: loss curves, example outputs of the final net, MIR per layer
over training. Local runs are tiny pipeline checks only; WRITE SLURM scripts (template
`/ceph/scratch/jlaherty/DMFT/jobs/dmft_sweep.sh`) but NEVER SUBMIT; the PI reviews code and the small
results, then submits the full runs. Work only inside `Projects/`; scratch in `FunctionalCellTypes/tmp/`
(git-ignored); uv cache is the project's `.uv-cache`.

## Rules for running things
- `unset VIRTUAL_ENV; export UV_CACHE_DIR=/ceph/scratch/jlaherty/Projects/FunctionalCellTypes/.uv-cache`
  before any `uv`; unit tests are CPU only (`CUDA_VISIBLE_DEVICES=""`), `uv run pytest -q`,
  `uv run ruff check fct tests && uv run ruff format --check fct tests`.
- GPUs for small local runs (PI, 2026-10-20): either, but pin by UUID and check `nvidia-smi` first.
  T1000 `GPU-71887a86-0384-411b-29cf-b76af75d0ab8`; RTX 4500 `GPU-e058dce4-5227-fc65-8fd3-fc5604ac22e8`
  (torch `cuda:0` = RTX 4500). An unrelated 5.5 GB idle process of the PI sits on the RTX 4500: keep
  our use small. Verify `torch.cuda.get_device_name(0)` before a run.
- The PI's shared tree has uncommitted unrelated changes (LaTeX moves; untracked `Latex/`,
  `playground.ipynb`, `todo.md`): never touch them; do work in git worktrees (`.claude/worktrees/`)
  and merge into local `main` with `--no-ff`; create worktrees from inside this repo's directory.
- Old `fct` caches are stale (config hashes changed with the new `DataConfig`/`TrainConfig` fields):
  use fresh cache directories.

## Done and merged into main (reviewed by fresh correctness + style reviewers; fixes applied)
| item | what |
|---|---|
| add-task-generators | `fct/tasks.py`: TaskSpec registry, linear_teacher / object_action / what_where; `Dataset` gains `factors`, `groups`, `extras`; `data.task_dims`; position task bit-identical to before |
| add-modularity-metrics | `fct/modularity.py`: rank binning, plug-in MI (nats), preference r, MIR, dead neurons, shuffled-label baseline, shuffle-based bias correction |
| add-result-plots | `analyse.py` example outputs / decoders / `prediction_report` (both R^2 conventions); `draw.py` loss curves, MIR history (raw, corrected, chance, dead counts), per-task example figures |

## State (2026-10-21): all work items merged; STOPPED for the PI's review
Merged into main, each reviewed by fresh correctness + style reviewers (fixes applied): add-task-generators,
add-modularity-metrics, add-result-plots, add-mir-training-monitor, add-sweep-runner (`fct/sweep.py`,
`python -m fct sweep ...`), add-slurm-scripts (`jobs/fct_sweep.sh`, never submitted), add-results-notebook
(`fct/report.py`, `notebooks/build_modularity.py`, source notebook `notebooks/modularity.ipynb`).
`run-tiny-pipeline` is `in review`: tiny preset ran 36/36 on CPU, 3 settings on the T1000 (pinned by UUID),
notebook executed into git-ignored `tmp/modularity.ipynb`. Nothing was submitted.
Next (PI): review code and the small results, set the `full` preset sizes in `fct/sweep.py` (N=256, P=4096,
15000 epochs are placeholders), tune `--cpus-per-task/--mem/-t` in `jobs/fct_sweep.sh`, run `--dry-run`,
submit. Then rebuild the notebook on the real ROOT (executed notebooks are NOT committed: PI decision).
New TrainConfig monitor fields changed config hashes: pre-existing caches are invalid (one-off).

## Decisions to flag to the PI at review (assumptions I made; easy to change)
- Factor groups for MIR: object-action = {position, objects (G*G occupancy bits), actions (5 bits)};
  a group's MI is the SUM of its members' MIs; what-where = {where, what}; linear teacher = one group
  per latent. Reason: the plug-in estimator cannot resolve MI with the joint object configuration.
- What-where motif library: all 511 non-empty 3x3 patterns (the brief says 510 but counts the all-on
  pattern, which gives 511).
- Dead neurons (no information about any factor) get chance preference 1/K and are counted; MIR is
  reported raw, shuffle-corrected, and with the shuffled-label chance baseline, because the raw MIR has
  a cardinality-dependent floor (about 0.1 for three 5-way factors, about 0.9 with a 25-way factor).
- SLURM scripts go in this repo's `jobs/`, not into `DMFT/jobs`.
- Linear-teacher weights W_X, W_Y are iid N(0, 1/k); inputs/targets m-dimensional with m = D_in > k.
- Monitoring cost on the real runs: set `monitor_int` to keep it a small fraction of training time
  (see the cost lines in docs/design.md); GPU timing was not measured on the full-size runs.
