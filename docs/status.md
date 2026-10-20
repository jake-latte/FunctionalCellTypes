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

## In flight: branch `add-mir-training-monitor` (NOT merged; worktree
`.claude/worktrees/agent-ab2a0046294391610`, head 69b4e07, work item status `in review`)
Adds `TrainConfig.monitor_int/monitor_bins/monitor_shuffles`, `train(..., monitor=...)` hook (called at
epoch starts after the checkpoint and once after the last step; resume-exact), `fct/monitor.py`
`make_mir_monitor` (history keys `monitor_epoch`, `monitor_mir`, `monitor_mir_corrected`,
`monitor_mir_chance`, `monitor_n_dead`, `monitor_n_dead_corrected`), CLI wiring. First review: no
correctness bugs. Review-fix round applied afterwards (one MI pass per layer; schedule tidy; lazy CLI
monitor; design.md cost: about 1.5 s/call object_action, 0.22 s/call what_where at N=256, L=5, P=4096,
CPU); suite there: 589 passed, 2 skipped. TO DO: (a) read the fix commits `15eefca`, `69b4e07` and run a
fresh re-review (correctness: numbers unchanged vs the separate `network_mir`/`shuffled_mir` calls;
resume exactness; style); (b) run its CUDA test once on a GPU pinned by UUID if not yet done; (c) set
the work item `status: done`, merge into main with a review summary in the merge message, remove the
worktree (slow; do in background) and delete the branch.

## Remaining work items (not started). Same workflow: worker per item in a worktree, TDD or
synthetic-data-first, then a fresh correctness reviewer and a fresh style reviewer, fixes, merge.
1. `add-sweep-runner` (needs the monitor merged). New `fct/sweep.py` + CLI subcommands
   `python -m fct sweep init|list|count|status|run ROOT [--index i]`. A `SweepConfig` (pydantic,
   `ROOT/sweep.json`; presets `tiny` for local checks and `full` for the cluster; per-task data
   parameters; N, P, n_epochs, lr, dtype (default float32), monitor_int, seeds) expands to
   3 tasks x L{1,3,5} x phi{tanh,relu} x rate_lambda{0,1e-3} = 36 settings; setting i lives in
   `ROOT/<task>/L{L}_{phi}_r{rate}/` (a normal fct cache dir). `run --index i` builds that setting's
   Config (set `data.D_in` from `data.task_dims`), `get_dataset`, `make_net`, `train` with
   `make_mir_monitor`; idempotent and resumable; `status` reports done / partial / missing.
   `weight_lambda` stays at the library default 1e-3 (the brief says "regularised MSE").
2. `add-slurm-scripts`. `jobs/fct_sweep.sh` modelled on `DMFT/jobs/dmft_sweep.sh`: actions
   `--submit [--index N]`, `--dry-run`, `--list`, `--status`; array over settings with a concurrency
   cap; `#SBATCH -p gpu --gres=gpu:1`, exclude the P5000 nodes `gpu-380-10,gpu-380-12,gpu-380-14`,
   logs under the project `tmp/logs/`; runs `.venv/bin/python -m fct sweep run ROOT --index $SLURM_ARRAY_TASK_ID`.
   Tests: `bash -n`, `--dry-run`, and a PATH-poisoned fake `sbatch` that fails if called. NEVER run
   `--submit`. Write into this repo's `jobs/` (DMFT/jobs is read-only template).
3. `add-results-notebook`. `modularity.ipynb` (source kept in the repo, e.g. `notebooks/` with a
   builder; executed with outputs): reads a sweep ROOT; per task and setting: loss curves
   (`draw.plot_loss_curves`), example outputs (`draw.fig_example_outputs`), MIR per layer over
   training (`draw.fig_mir_history`: raw with the shuffled baseline, corrected, dead counts); summary
   table/heatmap of final MIR per layer across the 36 settings; prediction metrics
   (`analyse.prediction_report`). Prose generated from numbers; no logic in cells.
4. `run-tiny-pipeline`. Run the `tiny` preset for all 36 settings locally (CPU and one tiny GPU smoke
   run pinned by UUID), execute the notebook on it, commit the executed notebook. Then STOP: the PI
   reviews the code and the small results and submits the full jobs themselves.

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
