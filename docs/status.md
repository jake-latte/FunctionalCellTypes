# FunctionalCellTypes modularity study: status and how to resume

Written 2026-10-20 by the PM agent. Read this, then `todo.md` (the PI's brief; untracked in the PI's
tree), `docs/design.md` (the interface contract) and `docs/issues/` (open items; finished ones are in `docs/issues/archive/`). Dated history of the earlier
programs (2026-10-20 to 2026-10-27) and the first program's learning-rate tuning are in `docs/archive/`.

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

### Still open from the 2026-10-27 programs (history in `docs/archive/status_history.md`)
- A theory Phi for a net under the spatial prior needs the mean-field solver to take the prior (new item).
- Whether the PI's spatial parameter set (N 256, P 64, k 4, m 16, rate 0, 200k epochs, CUDA; saved
  uncommitted in `tmp/pm/spatial_pi_edits.ipynb`) becomes the notebook defaults.
- N_BINS should respect P (B = 8 at P = 64 is one sample per joint cell; `N_BINS = min(8, sqrt(P)/2)`
  would be the mechanical default); small-P object-action runs have no usable corrected MIR under the
  group noise rule; cached sweeps predate that rule (caches key on the config only).
- `ruff format --check notebooks` flags the generated `notebooks/modularity.ipynb`, which the
  committed-notebook test requires verbatim (fix the builder's output or exclude the file).

## 2026-10-28: spatial grid search program (todo.md of 2026-10-27), first two items merged
The PI's brief asks for a structured test set, a grid search over spatially embedded networks
(a SPATIAL_PRIOR switch; the solver refusing spatial configs), a notebook over that grid with the Phi
kernels on the test set, the readout always phi(h^L), and presets (`--tiny`, an N scan). Work items are
in `docs/issues/` with their statuses. Merged (each reviewed by a fresh reviewer, record in the merge
commit): `linear-teacher-test-dataset` (`fct.tasks.latent_lattice_dataset`, 3^k sign patterns of the
latents; `fct.data.get_latent_lattice_dataset`) and `meanfield-rejects-spatial`
(`SpatialPriorUnsupportedError`; `fct solve` and `sweep run` report it in one line;
`fct.cli.USER_FIXABLE_ERRORS`). Awaiting the PI's confirmation of the plans: `spatial-sweep`,
`spatial-grid-notebook`, `readout-post-activations`.

**Hazard while the `pt_sweep_big` SLURM arrays run (3847945 on the morning of 2026-10-28; by the evening
3850616 tasks 1548-1999 and 3851591 tasks 2000-2499, same `jobs/fct_sweep.sh`; check `squeue -u jlaherty`):** their tasks import `fct` from this
tree's `.venv` as they start, so every merge into main must leave non-spatial code paths and hashes
identical (the two merges above are additive). `readout-post-activations` changes the model and must
not be merged while that array runs. Known: a dict-valued grid axis (`grid={"spatial": [...]}`) makes
a directory name longer than NAME_MAX; the spatial-sweep item gives the sweep a `spatial` field and
short labels. The three committed-notebook tests fail in the shared tree only because the PI's
uncommitted notebook edits differ from the builders' output (green in a clean checkout: 1426 passed).
