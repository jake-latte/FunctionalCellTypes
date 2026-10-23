# Functional Cell Types

`fct` is a Python package for studying how deep networks come to represent the factors of a task.
It has two parts that share one configuration, cache and command-line interface:

1. **Mean-field theory.** Train a deep network on a regression task, then solve the matching
   saddle-point equations (single-site HMC sampler, checkpointed) and compare kernels.
2. **Modularity study.** Train feed-forward networks on three tasks with known factors and measure
   whether hidden neurons become modular for those factors, using a mutual-information
   modularity index (MIR) per hidden layer, tracked through training.

The package is `fct/`; `docs/design.md` is the interface contract between its modules, `docs/status.md`
records the state of the modularity study, and `docs/issues/` holds the current work items (finished ones are archived in `docs/issues/archive/`).

## Install

The environment is managed with [`uv`](https://docs.astral.sh/uv/); `uv.lock` pins every dependency
and the first `uv sync` downloads the CUDA-enabled torch wheel (several GB). Python >= 3.12.

```sh
uv sync                                  # create the environment (dev tools included)
unset VIRTUAL_ENV                        # if your shell carries another project's venv
```

## Package layout

| Module | Purpose |
|---|---|
| `config.py` | pydantic `Config` (data / net / train / meanfield / analysis), JSON I/O, per-section hashes |
| `tasks.py`, `data.py` | the task generators (below) and the `Dataset` (inputs, targets, ground-truth factors and groups) |
| `model.py`, `train.py` | MLP and the training loop with checkpoint / resume and an optional monitor hook |
| `modularity.py` | rank binning, plug-in mutual information, preference ratio, MIR, dead neurons, shuffle baseline and bias correction (pure functions) |
| `monitor.py` | `make_mir_monitor`: the training hook that records per-layer MIR during training |
| `sweep.py` | a grid of runs over any `Config` fields: `SweepConfig`, `expand`, `run_setting`, `setting_status`, presets |
| `report.py` | read-only loading of a sweep and everything the results notebook shows (final-MIR tables, prose generated from the numbers) |
| `grid_view.py` | from a sweep ROOT to the phase-transition figures: snapping a setting to the grid, MIR records of one depth and activation, tuning curves of one layer |
| `analyse.py`, `draw.py` | pure analysis functions and plotting (`plot_*` take an axis, `fig_*` build figures) |
| `hmc.py`, `meanfield.py` | HMC sampler and the mean-field saddle-point solver |
| `cache.py`, `cli.py` | cache layout with cache-or-compute helpers; the `python -m fct` command line |
| `notebooks/` | `build_modularity.py` and `build_phase_transition.py` build the results notebooks; the `.ipynb` files are their sources (no outputs) |
| `jobs/fct_sweep.sh` | SLURM array script for the sweep |
| `tests/` | pytest suite: synthetic fixtures, CPU only, tiny sizes |

## The modularity study

**Tasks** (`fct/tasks.py`, selected with `data.task`; each returns the ground-truth factors used for scoring):

- `linear_teacher`: inputs and targets are generated from latent variables through random linear maps
  (`W_X`, `W_Y`, i.i.d. N(0, 1/k)); one factor group per latent.
- `object_action`: a G x G grid with an agent cell and an object map; the target is the optimal action
  (stay, up, down, left, right) toward the nearest object. Factor groups: agent position, object
  occupancy bits, action bits.
- `what_where`: a 9 x 9 grid of 3 x 3 macro-cells, one binary 3 x 3 motif placed in a macro-cell
  (a library of all 511 non-empty patterns); the target is `where` (which cell) and `what` (which motif).

**Modularity index.** For each hidden layer the code bins neuron rates by rank, estimates the mutual
information between each neuron and each factor group (plug-in, in nats), turns it into a preference
ratio per neuron, and averages to the layer's MIR. Neurons carrying no information about any factor are
counted as dead. Three numbers are reported, because the raw plug-in estimate has a
cardinality-dependent floor: the **raw** MIR, the **chance baseline** (the raw MIR recomputed with the
factor labels shuffled) and the **corrected** MIR (the shuffled null removed). Their exact definitions,
with equations, are in `notebooks/modularity.ipynb` and `fct/modularity.py`.

**Sweep.** A sweep is a grid over any `Config` fields: `grid` in `ROOT/sweep.json` maps a dotted path
(`net.L`, `net.phi`, `train.rate_lambda`, `train.weight_lambda`, `data.task`, ...) to a list of values, and the
settings are the Cartesian product in the order written (first axis slowest), so index `i` is stable. Sizes,
seeds and the task parameters are shared by all settings (`--set N=64 --set n_epochs=2000 ...`); an axis
overrides the shared value, except fields the sweep owns (data size, seeds, log/save/monitor schedules).
Presets: `tiny` (seconds per setting on a CPU, a pipeline check that learns nothing meaningful), `full` and
`modularity` (cluster sizes, placeholders to tune; the 36-setting grid of 3 tasks x layers {1, 3, 5} x phi
{tanh, relu} x rate regulariser {0, 1e-3}, directories `ROOT/<task>/L{L}_{phi}_r{rate}/`, weight regulariser at
its library default), and `phase_transition --n-lambda N_LAMBDA` (linear teacher, L in {1, 2}, phi in {tanh, relu},
`rate_lambda` and `weight_lambda` each `N_LAMBDA` log-spaced values in [1e-6, 1]). Other grids use directories
`ROOT/<task>/s{index:04d}_<label>/`, each with a `setting.json` recording its axis values, so a notebook can
find a setting from its values (`fct.sweep.find_setting_dir`). `--axis PATH=[values]` replaces or adds an axis.
Indices follow the product order, so adding a value to any axis but the first renumbers the settings after it
and orphans their directories (`status` then reports them missing): grow a grid in a fresh ROOT, or append to
the first axis only.

```sh
uv run python -m fct sweep init ROOT --preset tiny        # write ROOT/sweep.json (--set n_epochs=100 ...)
uv run python -m fct sweep init ROOT --preset phase_transition --n-lambda 8 --axis 'net.L=[1,2,3]'
uv run python -m fct sweep list ROOT                      # "<index> <task>/<setting directory>"
uv run python -m fct sweep count ROOT                     # number of settings (36 for modularity)
uv run python -m fct sweep run ROOT --index 7             # train one setting (resumable); no --index: all
uv run python -m fct sweep status ROOT                    # done / partial / missing / stale per setting
```

The results notebook and `fct.report` support the modularity grid (they need the axes `net.L`, `net.phi` and
`train.rate_lambda`).

**Cluster.** `jobs/fct_sweep.sh` submits the sweep as a SLURM array on the GPU partition (one setting
per task, a concurrency cap, logs under `tmp/logs/`). Check what it would run before submitting:

```sh
bash jobs/fct_sweep.sh ROOT --dry-run                     # print the sbatch command, run nothing
bash jobs/fct_sweep.sh ROOT --submit [--index N]          # FCT_MAX_CONCURRENT caps running tasks (default 4)
bash jobs/fct_sweep.sh ROOT --list | --status
```

**Results notebook.** The notebook shows, per task and setting, loss curves, example outputs of the
trained network, and MIR per layer over training, plus a summary table and heatmap of the final MIR
across the sweep. All logic is in `fct/report.py`; the cells only call it. Only the source notebook is
committed. Build and execute it on a sweep (the output goes to the git-ignored `tmp/` by default):

```sh
uv run python notebooks/build_modularity.py ROOT                 # execute on ROOT -> tmp/modularity.ipynb
uv run python notebooks/build_modularity.py --no-execute         # rebuild the committed source notebook
```

**Phase-transition notebook.** For a sweep made with `--preset phase_transition`: pick a setting by
`L`, `PHI`, `LAMBDA_R`, `LAMBDA_W` (snapped to the nearest grid value, which is printed) to see its loss
curves, example outputs, MIR over training and the tuning curves of its neurons; then the MIR of every
network of one depth and activation, as a (lambda_r, lambda_w) matrix per hidden layer and against
lambda_r / lambda_w. A setting still running is shown from its latest checkpoint and labelled
provisional; a missing one gets a message. The default MIR is the shuffle-bias-corrected one (`MIR_KIND`
switches to raw). The lambdas must be positive (log axes), and a request outside the grid snaps to its
nearest endpoint, signalled only by the printed "snapped to" line. Logic is in `fct/grid_view.py`; only the source notebook is committed.

```sh
uv run python notebooks/build_phase_transition.py ROOT           # execute on ROOT -> tmp/phase_transition.ipynb
uv run python notebooks/build_phase_transition.py --no-execute   # rebuild the committed source notebook
```

## Python library: single runs and the mean-field solver

A single run is one cache directory. `train` records the per-layer MIR during training when
`train.monitor_int > 0` (set `train.monitor_bins`, `train.monitor_shuffles` for its cost).

```sh
uv run python -m fct init CACHE_DIR --set data.P=512 --set net.L=2  # write CACHE_DIR/config.json
uv run python -m fct train CACHE_DIR          # generate data, train the network
uv run python -m fct solve CACHE_DIR          # mean-field saddle-point solver
uv run python -m fct plot CACHE_DIR           # figures that need results (--which NAME for one)
uv run python -m fct plot-example CACHE_DIR   # figure that needs no results
```

`solve` uses `train.rate_lambda` as its rate tilt, the per-sample coefficient of the trained
(summed) rate penalty; `--set meanfield.rate_tilt=...` overrides it.

`init --set` takes dotted keys with JSON values (`--set net.phi=tanh`,
`--set data.cov=[[1,0],[0,1]]`); unknown keys are rejected, and an existing `config.json` is
only replaced with `--force`. `plot` draws every figure whose inputs exist and lists the
skipped ones; `--which learning|representation|neurons|training|solver|kernels` draws one
(`solver` needs the `solve` result, `kernels` needs both `train` and `solve`).

Everything lives under `CACHE_DIR`:

| Path                                           | Content                                       |
|------------------------------------------------|-----------------------------------------------|
| `CACHE_DIR/config.json`                        | hyperparameters, written by `init`            |
| `CACHE_DIR/data/dataset.pt`                    | generated task data                           |
| `CACHE_DIR/train/ckpt_<epoch:06d>.pt`          | periodic checkpoint (only the latest is kept) |
| `CACHE_DIR/train/final.pt`                     | final trained model and its history           |
| `CACHE_DIR/meanfield/ckpt_<outer:05d>.pt`      | solver checkpoint                             |
| `CACHE_DIR/meanfield/final.pt`                 | mean-field kernels and solver history         |
| `CACHE_DIR/analysis/<name>.pt`                 | intermediate analysis results                 |
| `CACHE_DIR/figures/<name>.png`                 | figures                                       |

Re-running any step reuses what is in the cache, and interrupted `train` and `solve` runs
resume from their latest checkpoint. Each artefact records the config sections it was
computed from: editing `config.json` against an existing cache raises a stale-cache error,
so start from a fresh `CACHE_DIR` (or delete the stale file). `docs/design.md` is the
interface contract between modules.

```sh
uv run pytest         # tests (CPU only, synthetic data)
uv run ruff check fct tests            # lint
uv run ruff format --check fct tests   # formatting
```

## Paper (LaTeX)

The manuscript lives in `Latex/`; run the commands below from that directory. It is a modular LaTeX document. One file per section under `sections/`; each compiles
on its own (via the `subfiles` package) so sections can be worked on
independently.

### Build

```sh
make                                    # full document -> build/main.pdf
make sec SEC=02_renormalisation_group   # one section   -> build/02_renormalisation_group.pdf
make clean                              # removes build/
```

All intermediate files and PDFs go into `build/`, which is gitignored.

In VS Code with LaTeX Workshop, `.vscode/settings.json` makes Ctrl+Alt+B
compile whichever file is open (a section on its own, or everything from
`main.tex`), writes output to `build/`, and hides that folder in the explorer.

### Layout

| File              | Purpose                                                   |
|-------------------|-----------------------------------------------------------|
| `main.tex`        | Title, section order, bibliography. Edit to add/reorder.  |
| `preamble.tex`    | Packages, theorem environments, layout.                   |
| `macros.tex`      | Notation (`\R`, `\E`, `\norm`, `\RG`, ...). Edit globally. |
| `sections/*.tex`  | One section per file.                                     |
| `references.bib`  | Bibliography (biblatex + biber, author-year).             |
| `figures/`        | Graphics; `\includegraphics{name}` resolves here.         |

### Adding a section

1. Create `sections/05_new_section.tex`:
   ```latex
   \documentclass[../main.tex]{subfiles}
   \begin{document}
   \section{New Section}
   \label{sec:new}
   ...
   \end{document}
   ```
2. Add `\subfile{sections/05_new_section}` to `main.tex` where it should appear.

Renaming or reordering is just editing the `\subfile` list in `main.tex`.
Numbering starts at 0 (Motivation is Section 0) because of
`\setcounter{section}{-1}` in `main.tex`; delete that line for 1-based numbering.
