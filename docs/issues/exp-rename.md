---
type: infrastructure
complexity: complex
status: done
---

# Rename the sweep/grid machinery to "experiment" (exp)

Why: the PI is changing the vocabulary. A batch of runs over varied hyperparameters is an *experiment* ("exp"), defined entirely by the hyperparameters it varies. It is no longer a "sweep" or a "grid" in the hyperparameter-optimisation sense, and "sweep" and "grid" are both folded into "exp". This is a cosmetic change. Behaviour, numbers and cache hashes must not change. The same rename is being done in parallel in the DynamicalMultiplication (`dm`) and FeedbackAlignment (`fa`) repos with the same naming table, so follow the table exactly.

Naming table (apply wherever the word refers to the experiment machinery):
- modules:
  - `fct/sweep.py` -> `fct/exp.py`
  - `fct/grid_view.py` -> `fct/exp_view.py`
  - `fct/grid_report.py` -> `fct/exp_report.py`
  - `fct/spatial_grid.py` -> `fct/spatial_exp.py`
  - Check what each one does first: if a module is about something other than the experiment machinery, leave it and report it.
- `SweepConfig` -> `ExpConfig`; `SweepError` (or similar) -> `ExpError`; `SWEEP_OWNED_AXES` -> `EXP_OWNED_AXES`; `SWEEP_TASKS` -> `EXP_TASKS`; every other `*sweep*` identifier tied to the machinery -> `*exp*`
- `"sweep.json"` -> `"exp.json"`, with constant `EXP_FILE`
- the config field `grid` (dotted path -> list of values, or None for the built-in modularity grid) -> `axes`. None then means the built-in modularity axes. `with_grid` -> `with_axes`.
- `grid_hash` -> `exp_hash`, including the key written to each setting.json. The hash VALUE must be identical for the same experiment. Assert this in a test.
- preset names: `spatial_grid` -> `spatial_exp`; `lambda_grid` -> `lambda_exp`; `FIXED_GRID_PRESETS`/`GRID_PRESETS` -> names without "grid" (for example `FIXED_AXES_PRESETS`/`LAMBDA_PRESETS`; pick something clear). The other preset names stay. Do NOT restructure presets in this item; a later item moves them to JSON files. The `n_lambda` parameter stays for now.
- the CLI command `fct sweep ...` -> `fct exp ...`, with the same actions (init/list/count/status/run including `--stage solve`, ...). There is NO `sweep` alias.
- `jobs/fct_sweep.sh` -> `jobs/fct_exp.sh`; `jobs/fct_solve.sh` keeps its name but its text and commands use the new names.
- tests: `test_sweep*.py` -> `test_exp*.py` (for example `test_sweep_grid.py` -> `test_exp_axes.py`); `test_grid_view.py` -> `test_exp_view.py`; `test_grid_report.py` -> `test_exp_report.py`; `test_spatial_grid.py` -> `test_spatial_exp.py`; `test_build_spatial_grid.py` and `test_build_phase_transition_grid.py` follow their build scripts.
  - Fixtures `tests/fixtures/modularity_sweep.json` and similar: rename them and update their content to the new keys. Check first whether a fixture is deliberately a legacy-format file, such as a pinned old config. If so, keep it as the old format and use it for the migration test.
- "preset" stays "preset". A preset is now described as a *base experiment*.
- "setting" (one point of an experiment) stays "setting".
- Docs: module docstrings, comments, CLI help, README.md, docs/design.md and docs/status.md (current-state prose only) switch to experiment/exp/axes.
- `fct/lrtune.py` and `fct/report.py` use "sweep" often. Rename where the word refers to the experiment machinery (`SweepConfig`, sweep.json, the sweep CLI, settings of a sweep). If lrtune has its own independent loop over learning rates called a "sweep", decide whether it is the same concept (a batch of runs over a varied hyperparameter, so it becomes exp) or an internal algorithmic loop (stays). Report each decision.
- Notebook build scripts:
  - `notebooks/build_spatial_grid.py` -> `notebooks/build_spatial_exp.py`, writing `notebooks/spatial_exp.ipynb`
  - `notebooks/build_phase_transition_grid.py` -> `notebooks/build_phase_transition_exp.py`, writing `notebooks/phase_transition_exp.ipynb`
  - Change the wording and every API call in all build scripts, including build_modularity, build_spatial and build_phase_transition.
  - Regenerate the notebooks under the new names, commit them, and `git rm` the old tracked `spatial_grid.ipynb` and `phase_transition_grid.ipynb`.
  - For notebooks whose name does not change, regenerate only if the build output must change because of the rename. If a notebook regenerates differently for unrelated reasons, report it rather than commit it.
  - The PI keeps their own modified copies of several notebooks in the shared tree; never touch the shared tree.
  - `.vscode/settings.json`: update it if it references renamed files.

Do NOT rename:
- other meanings of the words:
  - `object_action.grid_size` (the task's G x G spatial grid)
  - `n_grid`/`h_grid`/`grid_targets`/`x_grid`/`y_grid` in analyse.py (the latent-position evaluation grid)
  - mean-field quadrature or integration grids
  - `P_GRID`/`N_GRID`-style constants if they are numerical grids; check
  - plotting grids
  - "sweep" meaning an algorithmic loop
  Use judgment, and list every borderline case you left alone.
- historical records:
  - `docs/archive/`
  - `docs/issues/archive/`
  - done issue files in `docs/issues/` (for example `spatial-grid-preset.md`, `phase-transition-notebook.md`)
  - the history sections of docs/status.md
  These record what was true at the time.
- the PI's data directories (`pt_sweep_big/`, `pt_sweep/`, `sweep_cache/`, `tmp/*`) or anything outside the repo.

One-off migration command (new behaviour, TDD):
- `fct exp migrate ROOT` converts an experiment folder written by the old code:
  - `ROOT/sweep.json` becomes `ROOT/exp.json`, with its `grid` key renamed `axes`;
  - every setting.json under ROOT gets its `grid_hash` key renamed `exp_hash`;
  - nothing else is touched: train/solve artefacts and their hashes stay as they are.
- Make it safe:
  - Validate the converted exp.json with `ExpConfig` before writing anything.
  - Write the new files first and remove `sweep.json` last.
  - Refuse, without changing anything, if `exp.json` already exists, if `sweep.json` is missing, or if setting files are inconsistent.
  - Print what it changed.
  - A second run refuses cleanly ("already migrated") and changes nothing.
- Old sweep.json files may contain the old preset names (none are stored, I believe; check) or an explicit `"grid": null`. Both must migrate.
- The new code reads only `exp.json`. When `exp.json` is absent but `sweep.json` is present, every exp command fails with a message telling the user to run `fct exp migrate ROOT`.
- Copy one real old-format sweep.json into the test fixtures as a migration fixture. A small one is `/ceph/scratch/jlaherty/Projects/FunctionalCellTypes/tmp/tiny_sweep_v2/sweep.json`; you may read it (read only).

Acceptance criteria:
1. The full test suite passes.
2. For every preset (`lambda_exp` at a couple of `n_lambda` values), `exp_hash` and every setting's train and solve hashes equal what the old code gives. Pin them in a test computed from the old code before the rename.
3. A migration test: build an old-style folder (sweep.json with `grid`, setting.json with `grid_hash`), migrate it, and check that it loads and reports the same status as a folder written fresh by the new code. A second migrate refuses and changes nothing. The refusal cases are tested. The real-file fixture migrates and validates.
4. Running an exp command on an unmigrated folder fails, and the error names `fct exp migrate`.
5. `git grep -niE 'sweep|grid'` over the non-historical tree shows only: migration code that reads the legacy names, the borderline non-experiment senses you list, and historical records.
6. `fct exp --help` and both SLURM scripts use only the new names; the SLURM script tests pass.

## Leftovers (from the confirmation review; optional, not done)
- migrate's note says "kept from an earlier experiment" also for a record whose hash is null, not a
  string, or the current hash in a non-member directory; "kept as it is (hash X is not this
  experiment's)" would not overclaim. Refusing a non-string hash is an option.
- The stale-legacy-file refusal says the folder's results "are stale"; in general they only may be
  (check with `exp status` after fixing the field).
- Pre-existing, not from this item: after an old `init --force` extended an axis, `exp run` (and
  `find_setting_dir`) still advise deleting a kept setting whose config.json still matches. A later
  item could let `run` rewrite such a stale setting.json instead.
