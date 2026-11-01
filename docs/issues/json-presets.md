---
type: infrastructure
complexity: complex
status: in progress
---

# Presets become base experiments stored as JSON in fct/presets/

Why:
- The PI wants presets to be *base experiments* that can be read and edited as data.
- Each preset is one JSON file in a `presets/` folder inside the library, laying out every parameter that base experiment prescribes.
- Editing the JSON file edits the preset. There is no Python copy to keep in sync.
- The same change has been made and reviewed in the FeedbackAlignment and DynamicalMultiplication repos. Their reviews raised the points listed below, so build them in from the start.

Current presets (`fct/exp.py`):
- `tiny`, `full`, `modularity` (the same as `full`; both have `axes` None, meaning the built-in modularity axes);
- `lambda_exp`, which takes `n_lambda`;
- `spatial_tiny`, `spatial_n_scan`, `spatial_exp` (built from `spatial_n_scan`) and `phase_transition` (built from `spatial_exp`);
- the dicts `FIXED_AXES_PRESETS`, `LAMBDA_PRESETS` and `PRESET_NAMES`.

FCT's `ExpConfig` is flat: the shared fields sit at the top level plus `axes`, and there is no `base`.

What to build:
- **One file per preset.** `fct/presets/<name>.json` for each of the eight presets, each one complete and self-contained: every `ExpConfig` field written out, no inheritance, no comments.
  - Write the files by dumping the current Python presets with `model_dump(mode="json")`, indent 2 and a trailing newline.
  - Keep `full` and `modularity` as two files: they are PI-visible names.
  - Then delete the Python builders, the preset dicts and `n_lambda`.
- **`lambda_exp` gets a fixed list (PI decision).** Its two lambda axes become a fixed 7-value list, `np.logspace(-6, 0, 7)`, which is 1e-6 to 1 in decades. Dump the old `lambda_exp(7)`.
  - The CLI `--n-lambda` option goes. The `--axis` override still works.
  - Existing tests pinned at other `n_lambda` values must move to the new fixed preset. Re-pin them against values computed by the OLD code at `n_lambda=7`, before you delete it.
  - Remove the `n_lambda` plumbing everywhere (CLI, notebooks, docs).
- **`description` field.** Add an optional `description: str | None = None` to `ExpConfig`.
  - Move the prose from the current preset docstrings there, reworded as describing an experiment.
  - Keep the important caveats, for example spatial_exp's length-scale/prior-strength confound and phase_transition's SGLD/stationarity notes.
  - It must NOT enter `exp_hash`, any train or solve hash, setting directory names, setting.json records, or any "same experiment" check: init on an existing ROOT, run's config/setting checks, migrate's resume equality.
  - Implement "same experiment" by comparing every field except description (`model_dump(exclude={"description"})`), never by listing fields.
  - A description-only change of an existing ROOT's exp.json must not make anything stale or refused. Test this.
  - Omit `description` from exp.json when it is None, so pre-existing exp.json files and migrate output stay byte-identical.
  - All writers of exp.json go through one function and use `encoding="utf-8"`.
  - The sha256 pins of each preset's exp.json text in `tests/test_phase_transition_preset.py` will change, because a description is now present. Update them, and check that the text minus the description key is byte-identical to the old dump.
- **Overrides.** When `exp init --preset X` is given any `--set` or `--axis`, write a derived description that stays true: `"Preset 'X' with overrides: --set a.b=1; --axis c.d=[...]"`, listing each override as given. With no overrides, keep the preset's description.
- **`preset(name, directory=None)`** reads `<directory or packaged fct/presets>/<name>.json` through `importlib.resources` and validates it.
  - An unreadable or invalid file raises the CLI's user-error class, naming the full path. The CLI prints `error:`, not a traceback.
  - An unknown name lists the available presets AND any ignored files with the naming rule.
- **`preset_names(directory=None)`** discovers presets at call time.
  - It lists only regular files ending `.json` that don't start with `.` and whose stem matches `[A-Za-z_][A-Za-z0-9_]*`.
  - Subdirectories (even one named `x.json`), dotfiles, a bare `.json` and README files are ignored.
  - Non-identifier `*.json` files are reported in the unknown-preset error, not silently dropped.
- **CLI choices.** `--help` for `exp init` lists the available presets, computed lazily with a fallback, so a broken presets folder fails only `exp init` and never `exp status`/`run`/`count`.
- **Packaging.** Verify the JSON files ship in the wheel.
  - Add a test that builds the wheel into tmp. It skips only when stderr shows an offline or resolution failure ("No solution found", "Failed to fetch", "offline"), or when `uv` is missing; any other build failure fails the test.
  - In a `-I` subprocess with the wheel on `sys.path`, the test asserts the module comes from the wheel and calls `preset_names()` and `preset('tiny')`.
- **Docs.** README and docs/design.md describe presets as base experiments in `fct/presets/`, how to edit or add one, and the naming rule. Say that a new file is picked up at once only with an editable install. Point to the JSON files for each preset's description rather than repeating it. Notebook builders that call `preset(..., n_lambda)` must be updated; regenerate the notebooks they write. The PI's own notebooks are not your concern.

Acceptance criteria:
1. For each preset, `preset(name)` after the change equals the old Python preset exactly, up to description: `exp_hash`, every setting's train and solve hash, and every directory.
   - For `lambda_exp`, compare against the old `lambda_exp(7)`.
   - `tests/test_exp_hashes.py` and `tests/fixtures/preset_hashes.json` must pass, with only the `lambda_exp` entries re-pinned to the old code's n=7 values.
2. Editing a copied presets folder (via the `directory` parameter) changes `preset` and `fct exp init --preset`. Adding a JSON file adds a choice. Ignored entries are ignored, and non-identifier files are named in the error.
3. A description-only change in an existing ROOT is not refused by init/status/run/migrate-resume and changes no hash.
4. Overrides produce the derived description.
5. Invalid and unreadable preset files give `error:` through the CLI.
6. The JSON is in the wheel and loads from the zip, or the test skips with its offline reason.
7. No Python preset definitions remain and `n_lambda` is gone. The full suite passes, `ruff check fct tests notebooks` is clean, and tests assert behaviour rather than reimplementing the code.
