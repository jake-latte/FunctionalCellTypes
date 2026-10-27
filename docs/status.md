# FunctionalCellTypes modularity study: status and how to resume

Written 2026-10-20 by the PM agent. Read this, then `todo.md` (the PI's brief; untracked in the PI's
tree), `docs/design.md` (the interface contract) and `docs/issues/` (current items; the first program's items are in `docs/issues/archive/`).

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
  before any `uv`; never write outside the project (pytest's temp dir is pinned to `tmp/pytest` in pyproject.toml;
  scratch in `tmp/`); unit tests are CPU only (`CUDA_VISIBLE_DEVICES=""`), `uv run pytest -q`,
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

## State (2026-10-24): both programs are complete and merged; everything now awaits the PI
The first program (task generators, MIR, sweep runner, SLURM script never submitted, notebook, lr tuning,
loss change to LOSS_VERSION 3 = rate penalty summed over P, `weight_lambda` 1e-4, mean-field tilt defaulting
to `rate_lambda`) is fully merged; its items are in `docs/issues/archive/`. Nothing was submitted. Still for
the PI: review the tiny results (`tmp/modularity.ipynb`), decide the `rate_lambda` values, set the `full`
preset sizes in `fct/sweep.py`, tune the SBATCH lines in `jobs/fct_sweep.sh`, `--dry-run`, submit.

Phase 2 (ten items, plans confirmed by the PI 2026-10-22; current state in `docs/issues/*.md`):
- merged 2026-10-23, each reviewed by a fresh reviewer with the record in its merge commit:
  `remove-rate-penalty-p-scaling`, `generic-grid-sweep` (arbitrary grids, `phase_transition` preset,
  `s{index}_label` directories), `spatial-kernels` (`fct/spatial.py`), `tuning-and-grid-plot-helpers`
  (`fct/tuning.py`, `fct/grid_report.py`, grid figures), `sgld-training` (`train.optimizer`, `train.beta`),
  `spatial-weight-prior-training` (`Config.spatial`, factorised prior in the loss, `fct/schema.py`),
  `grid-results-notebook` (`fct/grid_view.py`, `notebooks/build_phase_transition.py`),
  `importance-weighted-sampler` (`fct/importance.py`, `MeanFieldConfig` split into problem fields +
  `solver: SolverConfig`; reviewed twice: the first review found the ESS gate freezing the solver silently).
- merged 2026-10-23/24, likewise reviewed: `spatial-notebook` (`fct/spatial_view.py`, `fct/spatial_report.py`,
  `notebooks/build_spatial.py`), `meanfield-exemplar-notebook` (`solve(..., on_outer=...)` hook, live solver
  figures in `draw.py`, `fct/exemplar.py`, `notebooks/build_exemplar.py`; reviewed twice), `meanfield-grid-solve`
  (`SweepConfig.meanfield`, `run_setting(stage="solve")`, `sweep status/run --stage solve`, `jobs/fct_solve.sh`
  never submitted; reviewed twice). Main tip after the last merge: 871ccf7; 1321 tests pass, ruff clean.
No worker branches or worktrees remain. Nothing has been submitted to SLURM.
Still for the PI: the first-program checklist above, plus the phase-2 decisions below (the SGLD mapping one
changes numbers). Executed notebooks are not committed: build them with `notebooks/build_*.py` into `tmp/`.
Tooling: pytest's temp dir is pinned to `tmp/pytest` (tests/conftest.py); never write outside the project.

## Decisions to flag to the PI at review (assumptions I made; easy to change)
- Phase 2: `penalty_matrix` raises on a clearly negative kernel eigenvalue instead of flooring it; the
  minimum-image Gaussian is indefinite from length_scale ~ box / 10, hence the wrapped Gaussian. Grid sweeps:
  adding a value to any axis but the first renumbers later settings (documented; use a fresh ROOT). SGLD:
  `beta` is silently ignored with Adam (documented); default `beta = 1.0`. `mir_grid` requires `n_layers`.
- Importance sampler: the ESS gate is step-relative (`solver.min_ess_drop`) and `solver.min_ess_frac` is an
  absolute floor on accepted points that RAISES `InadequateSamplerError`; with the prior as proposal (PI's
  choice) the sampler only works for small P / mild tilts (at the default n_points = 100 it raises). Both
  fields enter the importance result hash.
- Spatial prior: with the default `SpatialConfig` (min_rel_eig 1e-6, N = 100) the SGLD stability check
  rejects every default lr * weight_lambda (needs < 6e-10) and Adam starts from a prior term of order 1e9;
  the notebook uses min_rel_eig 1e-2, N = 16. Open for the PI: eigenvalue floor vs nugget, and drawing the
  initial weights from the prior. The history key `weight_l2` holds the prior's value in spatial runs.
- **SGLD theory constants (exemplar review, PI decision pending).** The library's mean-field constants
  (`prior_scale = 1/weight_lambda`, `alpha = weight_lambda P d_out / beta`, `rate_tilt = rate_lambda`, data-term
  prefactor 1/2) describe the SGLD stationary law exp(-beta L) only at beta = 1/2 and prior_scale = 1. The
  matching constants are prior_scale = 1/(2 beta weight_lambda), alpha = 2 weight_lambda P d_out,
  rate_tilt = 2 beta rate_lambda, learning_prefactor = 1/(2 prior_scale); `fct.exemplar.sgld_theory_constants`
  returns them and the exemplar notebook uses them (weight variance 4.81 vs 5.00 predicted; closed-form check on
  the linear L=1 net). `run_meanfield`'s defaults are UNCHANGED, so every other mean-field result (grid solves,
  pinned fixtures) still uses the old constants. Decide whether the library should adopt the mapping.
- Exemplar notebook: the solves are not converged at the default 40/80 outer steps (residual still falling);
  the importance sampler only works for small P (default P = 20; raises at 24).
- Grid solve: `jobs/fct_solve.sh` duplicates the sbatch header of `jobs/fct_sweep.sh` (keep them in step by hand).
- Mean-field rate tilt defaults to `rate_lambda` (exact for the summed penalty); `meanfield.rate_tilt` overrides.
This was decided in `remove-rate-penalty-p-scaling`, not by the PI.
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

## 2026-10-27: PI feedback on the spatial notebook (all merged)
Four items, each reviewed by a fresh reviewer with the record in its merge commit:
`fix-tuning-view-cuda` (tuning_view moves rates and latents to the CPU once; fake-device regression test),
`spatial-notebook-inline-plots-and-equations` (every figure's plotting function written out in the
cell that draws it, a markdown cell with the equation before each; `fct/spatial_report.py` and the
spatial-only figures of `fct/draw.py` deleted; the kernel cells say K and the weight second moments are
over units, not the mean-field Phi over samples; the floor enters squared: eps = 1e-6 is 1e8 times
stiffer than 1e-2, not 1e4), `phi-kernels-in-spatial-notebook` (empirical Phi^(l) of the trained net
on all P samples, sorted by a latent, readout cases in the equation and titles), `mi-overlap-matrix`
(`fct.modularity.information_overlap`, `TuningView.member_mi`, C = I I^T / k per layer). The notebook
builds on CPU and on the T1000 with `DEVICE='cuda'` (9 figures). The PI's own parameter set (N 256,
P 64, k 4, m 16, no rate penalty, 200k epochs, CUDA) is saved uncommitted in
`tmp/pm/spatial_pi_edits.ipynb`; rerun it with
`uv run python notebooks/build_spatial.py tmp/spatial_cache --set N=256 --set P=64 --set K_LATENTS=4 --set M_IN=16 --set RATE_LAMBDA=0 --set N_EPOCHS=200_000 --set "DEVICE='cuda'"`.
Known and untouched: `ruff format --check notebooks` flags the generated `notebooks/modularity.ipynb`,
which the committed-notebook test requires verbatim (fix the builder's output or exclude the file).
Open for the PI: a theory Phi for a net under the spatial prior would need the mean-field solver to
take the prior (new item); whether the PI's parameter set should become the notebook defaults.
