---
type: infrastructure
complexity: complex
status: in progress
---

<!-- status: todo | in progress | in review | done | blocked | awaiting your confirmation -->

# Generalise the sweep to an arbitrary grid of hyperparameters

PI brief: "Generalise your grid search infrastructure to work with any grid of hyperparameters", then define the phase-transition grid as a preset: L in [1,2], phi in [tanh, relu], lambda_r = linspace(1e-6, 1, N_LAMBDA), lambda_w = linspace(1e-6, 1, N_LAMBDA), linear teacher task. The PI tunes N_LAMBDA, N, N_EPOCHS etc. and submits.

## Proposed plan
1. **Grid spec.** `SweepConfig` gains `grid: dict[str, list]`: axis name = dotted `Config` path (`net.L`, `net.phi`, `train.rate_lambda`, `train.weight_lambda`, `data.task`, ...), values = explicit lists. `expand` is the Cartesian product in the order the dict is written (first axis slowest), so an index is stable for a given sweep.json. Shared settings (N, P, n_epochs, lr, seeds, task parameters) stay as they are and become the base `Config`; a grid axis overrides the base value. Axis paths and values are validated at load time by building every setting's `Config`.
2. **Setting.** `Setting` becomes an ordered mapping `{axis: value}` (a NamedTuple cannot have variable fields). Directory name `ROOT/<task>/<axis=value,...>` is replaced by `ROOT/<task>/s{index:04d}_<short human label>`; the label is for people, the index is what makes it unique (linspace floats are not safe as unique names). `setting.json` in each directory records the axis values so the notebook can find a setting from its values without recomputing the grid.
3. **Backward compatibility.** The existing 36-setting grid becomes the preset `modularity` (task axis included), so `sweep.json` files written before this change still load (missing `grid` -> legacy grid) and their directories keep their old names (legacy naming kept when `grid` is absent). Existing `report.py`, `lrtune.py`, CLI and `jobs/fct_sweep.sh` keep working on legacy sweeps.
4. **Presets.** Add `phase_transition(n_lambda)` as a preset *factory* (preset functions take N_LAMBDA). The CLI: `python -m fct sweep init ROOT --preset phase_transition --n-lambda 8` (plus `--set` overrides for N, P, n_epochs, ...). The SLURM script needs no change beyond using `sweep count`.
5. **lambda_w / lambda_r** are log-spaced: `logspace(-6, 0, N_LAMBDA)` (PI decision 2026-10-22; the brief's `linspace` was a slip).

## Acceptance criteria
- `expand` of an arbitrary grid has prod(len) distinct settings in a stable order; `count`/`list`/`status`/`run --index i` agree.
- The phase-transition preset with N_LAMBDA=3 gives 2*2*3*3 = 36 settings, all linear-teacher, with `weight_lambda` and `rate_lambda` taken from the grid and equal to `logspace(-6, 0, 3)` = {1e-6, 1e-3, 1}.
- A legacy sweep.json (written by the current code) loads, expands to the same 36 settings and directories as before.
- A setting is findable from its axis values via `setting.json`; unknown axis names or bad values fail at load time naming the axis.
- Tiny end-to-end run of a 2x2 grid on CPU; resume/skip behaviour unchanged. Tests CPU only, ruff clean, `docs/design.md` updated.

## PI decisions (2026-10-22)
- Plan confirmed. New grids use `s{index}_label` directories (legacy grids keep theirs); L in {1,2} is `net.L`; lambdas log-spaced.
