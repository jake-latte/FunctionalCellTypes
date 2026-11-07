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
| `exp.py` | an experiment, a batch of runs over varied `Config` fields (its axes): `ExpConfig`, `expand`, `run_setting`, `setting_status`, preset readers; the presets themselves are the JSON files in `presets/` |
| `report.py` | read-only loading of an experiment and everything the results notebook shows (final-MIR tables, prose generated from the numbers) |
| `exp_view.py` | from an experiment ROOT to the figures of a `lambda_exp` experiment (the phase-transition notebook): snapping a setting to the axis values, MIR records of one depth and activation, tuning curves of one layer |
| `spatial_view.py` | the spatial-prior notebook: its config and resumable run, ring / lattice layout of neurons and weights, empirical kernels (the notebook draws its own figures) |
| `linear_theory.py` | the linear one-hidden-layer theory on a ring, trained by SGLD, as pure functions of a `Config` and a `Dataset`: per-mode spectra and their summaries for an orthonormal teacher, the ansatz-free kernel for any teacher (`latex/notes/linear_solution_limits.tex`, `linear_without_ansatz.tex`) |
| `analyse.py`, `draw.py` | pure analysis functions and plotting (`plot_*` take an axis, `fig_*` build figures) |
| `hmc.py`, `importance.py`, `meanfield.py` | HMC and importance samplers, the mean-field saddle-point solver |
| `exemplar.py` | the exemplar notebook's logic: config from flat settings, cached train and solve, live solver view |
| `cache.py`, `cli.py` | cache layout with cache-or-compute helpers; the `python -m fct` command line |
| `notebooks/` | `build_modularity.py`, `build_phase_transition.py`, `build_spatial.py`, `build_spatial_exp.py`, `build_phase_transition_exp.py` and `build_exemplar.py` build the notebooks; the `.ipynb` files are their sources (no outputs) |
| `jobs/fct_exp.sh` | SLURM array script for the experiment (train stage) |
| `jobs/fct_solve.sh` | the same array job at the solve stage (mean-field saddle points) |
| `tests/` | pytest suite: synthetic fixtures, CPU only, tiny sizes |

## The modularity study

**Tasks** (`fct/tasks.py`, selected with `data.task`; each returns the ground-truth factors used for scoring):

- `linear_teacher`: inputs and targets are generated from latent variables through random linear maps
  (`W_X`, `W_Y`, i.i.d. N(0, 1/k), or with orthonormal columns for `data.teacher = "orthonormal"`); one factor group per latent.
  `fct.data.get_latent_lattice_dataset` gives a structured test set for it: one sample per point of {lo, (lo + hi)/2, hi}^k, the latents being uniform on [lo, hi] = [`latent_min`, `latent_max`].
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

**Experiment.** An experiment is a batch of runs over varied `Config` fields, defined entirely by the hyperparameters it varies: `axes` in `ROOT/exp.json` maps a dotted path
(`net.L`, `net.phi`, `train.rate_lambda`, `train.weight_lambda`, `data.task`, ...) to a list of values, and the
settings are the Cartesian product in the order written (first axis slowest), so index `i` is stable. Sizes,
seeds and the task parameters are shared by all settings (`--set N=64 --set n_epochs=2000 ...`); an axis
overrides the shared value, except fields the experiment owns (data size, seeds, log/save/monitor schedules).
Base experiments are the presets (below). Other experiments use directories
`ROOT/<task>/s{index:04d}_<label>/`, each with a `setting.json` recording its axis values, so a notebook can
find a setting from its values (`fct.exp.find_setting_dir`). `--axis PATH=[values]` replaces or adds an axis.
Indices follow the product order, so adding a value to any axis but the first renumbers the settings after it
and orphans their directories (`status` then reports them missing): grow an experiment in a fresh ROOT, or append to
the first axis only.

**Spatial experiments.** An experiment's optional `spatial` section (`ExpConfig.spatial`, a `fct.spatial.SpatialConfig`)
makes every setting train under the spatial weight prior `tr(M_post W M_pre W^T)` per layer; without it every
setting trains under the plain weight L2. The axes may then vary any field of the section
(`--axis spatial.hidden.kernel.length_scale=[0.05,0.15,0.3]`, `spatial.hidden.embedding.kind`, `spatial.min_rel_eig`,
`spatial.d`, ...); such an axis on an experiment without the section is rejected at load. The shared network and loss
fields the spatial presets set (`L`, `g`, `phi`, `B`, `weight_lambda`, `rate_lambda`, `optimizer`, `beta`) default to the
library values, and an axis of the same name wins (`phase_transition` sets its ensemble size, `B = 8`, through a `net.B` axis). The spatial presets are `spatial_tiny`, `spatial_n_scan`, `spatial_exp` and `phase_transition`; the
description in each JSON file (see Presets) carries the details and caveats. The results notebook of `phase_transition`
is `notebooks/phase_transition_exp.ipynb` (`fct/phase_view.py`, built by `notebooks/build_phase_transition_exp.py`; see
below); it has no solve stage yet, and the modularity readers (`fct.report`) refuse B > 1 and need `net.L`, `net.phi`
and `rate_lambda` axes. A directory name shows a dict-valued axis value as a short
hash. The mean-field solver knows only the iid prior, so `exp run --stage solve` on a spatial setting fails with its
one-line message. `net.phi` is one of `tanh`, `relu`, `sigmoid` or `id` (phi(h) = h: a linear network);
the presets keep their activations, and a variant with `id` is `--axis 'net.phi=["tanh","relu","id"]'`.

**Presets.** A preset is a base experiment stored as data: one JSON file per preset in `fct/presets/`, writing out
every `ExpConfig` field (`tiny`, `full`, `modularity`, `lambda_exp`, `spatial_tiny`, `spatial_n_scan`, `spatial_exp`,
`phase_transition`). The file's `description` says what the experiment is and keeps its caveats; read the JSON file
for it rather than a copy here. `exp init ROOT --preset NAME` copies the file into `ROOT/exp.json`; with any `--set`
or `--axis` it replaces the description with a derived one that lists the overrides (`Preset 'tiny' with overrides:
--set N=16`). The description is documentation only: it never enters a hash, a directory name or any check that two
experiments are the same, so editing it never makes results stale.
To change a preset, edit its JSON file. To add one, put a new `<name>.json` there (a complete experiment, no
inheritance and no comments); the name before `.json` must match `[A-Za-z_][A-Za-z0-9_]*`. Other files (a README,
hidden files, folders) are ignored, and a `*.json` file with a name that breaks the rule is listed as ignored when a
preset is not found. Presets are read each time they are asked for, so with an editable install (`uv sync`, the
default) an edit or a new file is picked up at once; an installed wheel holds the files it was built with.
`fct.exp.preset(name, directory=None)` and `preset_names(directory=None)` read the packaged folder or any other.

```sh
uv run python -m fct exp init ROOT --preset tiny        # write ROOT/exp.json (--set n_epochs=100 ...)
uv run python -m fct exp init ROOT --preset lambda_exp --axis 'net.L=[1,2,3]'
uv run python -m fct exp list ROOT                      # "<index> <task>/<setting directory>"
uv run python -m fct exp count ROOT                     # number of settings (36 for modularity)
uv run python -m fct exp run ROOT --index 7             # train one setting (resumable); no --index: all
uv run python -m fct exp status ROOT                    # done / partial / missing / stale per setting
uv run python -m fct exp run ROOT --index 7 --stage solve    # solve one setting's saddle point (resumable)
uv run python -m fct exp status ROOT --stage solve      # the same states for the solves
uv run python -m fct exp migrate ROOT                   # convert a folder written by older code (once)
```

A folder that holds a legacy `sweep.json` (with a `grid` key and `grid_hash` in
every `setting.json`) is converted once, in place, by `uv run python -m fct exp migrate ROOT`: it renames those names,
checks everything before writing anything, leaves every train / solve artefact and hash alone, and refuses a finished folder.
The rename keeps the folder as it is: a `setting.json` left by an earlier experiment of the same folder (an old `init --force`
never deleted settings) is renamed too and reported as "kept from an earlier experiment"; `exp run` still refuses it, as before.
If the write fails part-way (disk full, say), run the command again to finish. A legacy `sweep.json` that no longer
validates under the current config (for example one that still carries the removed `meanfield.readout`) is refused, naming
the field: the old code could not load it either, so the folder's results are stale under the current code; make it validate
first. Do not migrate a folder while any checkout or SLURM job still runs the old code: it would write `grid_hash` again.
Every other `exp` command on an unmigrated folder stops and points to this one. The notebook environment variable is now
`$FCT_EXP_ROOT` (it was `$FCT_SWEEP_ROOT`).

An initialised experiment can be trained, solved, or both, in either order (a solve finished before the net was
trained is reported `stale` and redone with the net, so the comparison with the net is never missing): `--stage train` (the default) trains the
networks, `--stage solve` solves the mean-field saddle point of each setting into `<setting>/meanfield/`, using
the trained network of that setting for the comparison if there is one and solving on its own otherwise. The solver
settings are the `meanfield` section of `exp.json` (for example
`--set meanfield.n_points=50 --set meanfield.solver.sampler=importance` at `exp init`); changing them never
invalidates a trained network, only the solves. An importance-sampling solve whose proposal cannot represent the
problem fails that setting with `InadequateSamplerError` and writes no result.

The results notebook and `fct.report` support the modularity experiment (they need the axes `net.L`, `net.phi` and
`train.rate_lambda`).

**Cluster.** `jobs/fct_exp.sh` submits the experiment as a SLURM array on the GPU partition (one setting
per task, a concurrency cap, logs under `tmp/logs/`). Check what it would run before submitting:

```sh
bash jobs/fct_exp.sh ROOT --dry-run                     # print the sbatch command, run nothing
bash jobs/fct_exp.sh ROOT --submit [--index N]          # FCT_MAX_CONCURRENT caps running tasks (default 4)
bash jobs/fct_exp.sh ROOT --list | --status
bash jobs/fct_solve.sh ROOT --dry-run                     # the same array job at the solve stage
```

**Results notebook.** The notebook shows, per task and setting, loss curves, example outputs of the
trained network, and MIR per layer over training, plus a summary table and heatmap of the final MIR
across the experiment. All logic is in `fct/report.py`; the cells only call it. Only the source notebook is
committed. Build and execute it on an experiment (the output goes to the git-ignored `tmp/` by default):

```sh
uv run python notebooks/build_modularity.py ROOT                 # execute on ROOT -> tmp/modularity.ipynb
uv run python notebooks/build_modularity.py --no-execute         # rebuild the committed source notebook
```

**Phase-transition notebook** (the viewer for the `lambda_exp` preset). For an experiment made with `--preset lambda_exp`: pick a setting by
`L`, `PHI`, `LAMBDA_R`, `LAMBDA_W` (snapped to the nearest axis value, which is printed) to see its loss
curves, example outputs, MIR over training and the tuning curves of its neurons; then the MIR of every
network of one depth and activation, as a (lambda_r, lambda_w) matrix per hidden layer and against
lambda_r / lambda_w. A setting still running is shown from its latest checkpoint and labelled
provisional; a missing one gets a message. The default MIR is the shuffle-bias-corrected one (`MIR_KIND`
switches to raw). The lambdas must be positive (log axes), and a request outside the axis values snaps to its
nearest endpoint, signalled only by the printed "snapped to" line. Logic is in `fct/exp_view.py`; only the source notebook is committed.

```sh
uv run python notebooks/build_phase_transition.py ROOT           # execute on ROOT -> tmp/phase_transition.ipynb
uv run python notebooks/build_phase_transition.py --no-execute   # rebuild the committed source notebook
```

**Exemplar notebook.** One linear-teacher run end to end: every task and network setting in one cell, SGLD
training (or Adam) through the `fct` cache with loss curves, outputs and the net's empirical kernels, then the
mean-field solve with live figures refreshed every `PLOT_EVERY` outer steps: the kernels Phi_l and Phi_hat_l,
solver health (residuals, step sizes, backtracking, PD clamping, theory-vs-net train MSE) and sampler health (HMC acceptance,
step size, divergences, ESS, or the importance ESS and maximum weight), and a final cell with the solved kernels
next to the empirical ones. `SAMPLER` picks `"hmc"` or `"importance"`. The default sizes are tiny
(20 to 45 s per sampler on a CPU) and chosen so the importance sampler, a small-P / mild-tilt tool, does not raise
`InadequateSamplerError`. The theory is given the mean-field constants of the SGLD stationary law, which
differ from the library defaults unless beta = 1/2 (see `docs/design.md`), and a printed summary reports the
solve's residual trend, ESS and kernel scales. The live view is `solve(..., on_outer=callback)`, an observer that cannot change the
result. Logic is in `fct/exemplar.py` and `fct/draw.py`; only the source notebook is committed.

```sh
uv run python notebooks/build_exemplar.py CACHE_ROOT                     # execute -> tmp/exemplar.ipynb
uv run python notebooks/build_exemplar.py CACHE_ROOT --set "SAMPLER='importance'" --set N_OUTER_IMPORTANCE=10
uv run python notebooks/build_exemplar.py --no-execute                   # rebuild the committed source notebook
```

**Spatial-prior notebook.** One linear-teacher run under the spatial weight prior (`Config.spatial`):
set the task, network, training (`OPTIMIZER = "sgld" | "adam"`, `BETA`, `LR`, `N_EPOCHS`) and the
embedding and kernel of each population in the first cells; the notebook tabulates the eigenvalue
floor and the SGLD step-size condition, plots the prior kernels, trains
through the normal cache (re-running reloads a finished run or resumes an interrupted one; the run
directory under `CACHE_ROOT` is named by a hash of the settings), then shows the loss curves, example
outputs, the first-layer weights sorted by ring position next to the prior's kernels, the paper's
spatially indexed activity kernels Phi^(l)_{mu nu, ij} = K^post_ij A^(l)_{mu nu} of every hidden layer
(the data kernel A^(l) = psi K^pre psi^T / N over the training samples, sorted by a chosen latent, then
for a few samples spread over that latent the S x S grid of n_post x n_post blocks), and the tuning of
every neuron to every latent, one panel per neuron placed by its position (ring order for d = 1, lattice
cell for d = 2). The default sizes are small so SGLD is stable and both optimisers take seconds on a CPU;
the notebook text says why. The library computes (`fct/spatial_view.py`, `fct/spatial.py`,
`fct/exp_view.py`); every figure is drawn by a plotting function written out in the cell that shows
it, after a markdown cell with the equation of what is plotted. The kernels shown are the weight
prior and the weights' second moments over units, not the activity kernels Phi^(l) over samples, which
have their own figures. Only the
source notebook is committed.

```sh
uv run python notebooks/build_spatial.py CACHE_ROOT                       # execute -> tmp/spatial.ipynb
uv run python notebooks/build_spatial.py CACHE_ROOT --set OPTIMIZER=\'adam\' --set N_EPOCHS=500
uv run python notebooks/build_spatial.py --no-execute                     # rebuild the committed source notebook
```

The spatial-prior notebook draws its spatially indexed kernels Phi^(l) (the data kernels A^(l) next to the latent
kernel B = Z Z^T / k, the chosen trials, the pair grids) and its example outputs (a few test points drawn as panels, the R^2 and the scatter over all of them) from the
held-out test set, the 3^k points of the latent lattice (each latent at `latent_min`, the midpoint or `latent_max`), not from the training samples.

**Spatial-experiment notebook.** The same figures for one setting of an already-run spatial experiment (for instance
`python -m fct exp init ROOT --preset spatial_n_scan`, then `exp run`): give the experiment ROOT and the axis values
of the setting, `SETTING = {"net.N": 256}` (a subset of the axes is enough when it picks one). The notebook lists the
settings with each setting's status, loads the setting's trained net, and shows the spatial notebook's figures with
the Phi kernels and example outputs (and their R^2, over all 3^k test points) computed on the test set. A setting of the experiment that is not trained is reported in
the cells and only its prior is drawn; a setting that is not in the experiment is an error. The cells are the spatial
notebook's (`build_spatial_exp.py` imports them), so both build the same figures. Only the source notebook is committed.

```sh
uv run python notebooks/build_spatial_exp.py ROOT --set 'SETTING={"net.N": 256}'   # execute -> tmp/spatial_exp.ipynb
uv run python notebooks/build_spatial_exp.py ROOT --set 'SETTING={"net.N": 8}' --set DEVICE=\'cuda\'
uv run python notebooks/build_spatial_exp.py --no-execute                          # rebuild the committed source notebook
```

**Phase-transition-experiment notebook** (`phase_transition_exp.ipynb`; the viewer for the `phase_transition` preset, an ensemble of B networks per setting; not the lambda_exp viewer `phase_transition.ipynb`). Give the experiment ROOT and the axis values of one setting, `SETTING = {"spatial.d": 2, "spatial.lambda_max": 0.5}`, and the parameters `MEMBER` (the member of the single-network figures), `SORT_DIM` (every unit ordering uses this coordinate of the random embedding; with d > 1 it is one coordinate, not a ring position, and out of range raises), `P_TEST` (size of the fresh random test set of the tuning and information/rate overlaps; the data kernels A keep the 3^k latent lattice), `N_EIG` (spectra truncation) `N_NEURONS` with `NEURON_PICK` (the tuning figure shows the `"first"` N_NEURONS in SORT_DIM order, or `"spread"` evenly along it) and `N_OUTPUT_DIMS` (output dimensions drawn in the example panels). It draws, for the prior, the covariance K / lambda_w, the effective (floored) covariance the nets were trained under and the penalty lambda_w M side by side, and their spectra with the floor; the loss curves as mean +- sem over members; every member's example outputs on the same axes (one palette colour each, at most the palette's ten, a subset of the output dimensions); and, for member `MEMBER` and for the ensemble average, the trained weight kernels, the data kernels A and the overlaps (the average overlap is the mean over members of each member's overlap). The first-layer weight image and the trial and pair grids of the spatial notebook are not drawn. A setting that is not trained gets only its prior; a partly trained one is drawn from its latest checkpoint, and every figure title says so (PROVISIONAL, with the epoch reached). Logic is in `fct/phase_view.py`; only the source notebook is committed.

```sh
uv run python notebooks/build_phase_transition_exp.py ROOT --set 'SETTING={"spatial.d": 2}' --set MEMBER=1   # execute -> tmp/phase_transition_exp.ipynb
uv run python notebooks/build_phase_transition_exp.py --no-execute                                          # rebuild the committed source notebook
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

`solve` uses `train.rate_lambda` as its rate tilt: training puts the rate penalty
`rate_lambda / (2 beta) * sum phi^2` in the loss (beta = `train.beta` under SGLD, 1 under Adam),
so the law exp(-beta L) carries `(rate_lambda / 2) sum phi^2` for every beta; `--set
meanfield.rate_tilt=...` overrides it. The `meanfield` section
holds the problem (points, `beta`, prior, `alpha`, rate tilt); how it is solved lives in
`meanfield.solver`, e.g. `--set meanfield.solver.sampler=importance` for importance sampling
from the prior instead of HMC, or `--set meanfield.solver.n_chains=512`. A `config.json` in
the older flat layout still loads.

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
uv run ruff check fct tests notebooks  # lint (notebooks: the generated cells' imports)
uv run ruff format --check fct tests   # formatting
```

## Paper (LaTeX)

The manuscript lives in `latex/`; run the commands below from that directory. It is a modular LaTeX document. One file per section under `sections/`; each compiles
on its own (via the `subfiles` package) so sections can be worked on
independently. Building needs TeX Live with `latexmk`, `pdflatex`, `biber`,
and the `subfiles` and `biblatex` packages.

### Build

```sh
make                                    # full document -> build/main.pdf
make sec SEC=02_renormalisation_group   # one section   -> sections/build/02_renormalisation_group.pdf
make note NOTE=continuum_limit          # one note      -> notes/build/continuum_limit.pdf
make notes                              # every notes/*.tex -> notes/build/
make check                              # verify every build lands in a build/ dir
make clean                              # removes build/, sections/build/, notes/build/
```

Every build writes to a `build/` directory beside its root `.tex` file
(`latex/`, `latex/sections/` or `latex/notes/`); these are gitignored. The
notes in `notes/` are self-contained articles built one at a time. A bare
`latexmk <file>.tex` run from any of those three directories does the same,
via that directory's `.latexmkrc`.

In VS Code with LaTeX Workshop, `.vscode/settings.json` makes Ctrl+Alt+B
compile whichever file is open (the whole document from `main.tex`, a section,
or a note), writes output to `build/` beside that file, and hides those
folders in the explorer.

### Layout

| File              | Purpose                                                   |
|-------------------|-----------------------------------------------------------|
| `main.tex`        | Title, section order, bibliography. Edit to add/reorder.  |
| `preamble.tex`    | Packages, theorem environments, layout.                   |
| `macros.tex`      | Notation (`\R`, `\E`, `\norm`, `\RG`, ...). Edit globally. |
| `sections/*.tex`  | One section per file.                                     |
| `references.bib`  | Bibliography (biblatex + biber, author-year).             |
| `figures/`        | Graphics; `\includegraphics{name}` resolves here.         |
| `notes/*.tex`     | Self-contained articles (own preamble); built one at a time. |
| `scripts/`        | `check_build_dirs.sh`, run by `make check`.               |
| `.latexmkrc`      | latexmk settings (output to `build/`); `sections/` and `notes/` delegate to it. |

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
