---
type: infrastructure
complexity: simple
status: in review
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Add the `python -m fct` CLI: init / train / solve / plot / plot-example

Wire the library into the init -> train / solve -> plot workflow: initialise a hyperparameter
JSON in a cache directory, then train, solve the mean field, draw the
result-dependent figures, or draw the result-independent example figure,
every step defaulting to whatever already exists in the cache.

Why: this is the user-facing entry point; everything else is a library.

## Inputs / outputs / interfaces

Exactly the *cli.py* section of `docs/design.md`, in `fct/cli.py` with
`main(argv: list[str] | None = None) -> int` and `fct/__main__.py`. argparse
subcommands:

- `init CACHE_DIR [--set key=value ...] [--force]`: writes
  `default_config()` with the overrides applied to `CACHE_DIR/config.json`;
  refuses to overwrite an existing file unless `--force`. Values parse as JSON
  first and fall back to a string (`--set net.phi=tanh`,
  `--set data.cov=[[1,0],[0,1]]`); unknown keys fail with pydantic's message.
- `train CACHE_DIR`: `get_dataset` -> `make_net` -> `train`.
- `solve CACHE_DIR`: `get_dataset` -> `load_trained` -> `run_meanfield`.
- `plot CACHE_DIR [--which ...]`: `learning`, `representation`, `neurons`,
  `training` need `final.pt` from training; `solver`, `kernels` need the
  mean-field result; `all` (default) draws those whose inputs exist and says
  which were skipped and why. Figures are saved as
  `CACHE_DIR/figures/<which>.png` (dpi 150) and the paths printed.
- `plot-example CACHE_DIR`: `get_dataset` -> `fig_example` ->
  `CACHE_DIR/figures/example.png`.

Missing `config.json` or missing prerequisites produce a one-line error and
exit code 1 (no traceback). Document the workflow in `README.md` (replace
the placeholder section from the scaffold with real commands and the cache
layout table from `docs/design.md`).

## Acceptance criteria

- End-to-end test on CPU with a tiny config (`data.P=64, D_in=8`,
  `net.N=8, L=2`, `train.n_epochs=6, save_int=3, log_int=3`,
  `meanfield.n_points=8, n_chains=64, n_outer=2, n_inner_max=3,
  warmup_trans=10, n_init_samples=256, save_int=1`,
  `analysis.n_grid=5, n_components=3`, `device=cpu`): `init` then `train`,
  `solve`, `plot`, `plot-example` each return 0 and the expected files
  exist (`data/dataset.pt`, `train/final.pt`, `meanfield/final.pt`,
  `figures/{learning,representation,neurons,training,solver,kernels,example}.png`).
- Running `train` a second time returns 0 and does not change
  `train/final.pt` (mtime / bytes).
- `init` on an existing config returns 1 unless `--force`; `--set` with an
  unknown key returns 1 with pydantic's message on stderr; `--set net.L=2`
  is reflected in the written JSON.
- `plot` before training returns 1 with a message naming the missing file;
  `plot --which solver` before solving likewise.
- `python -m fct --help` lists the five subcommands.
