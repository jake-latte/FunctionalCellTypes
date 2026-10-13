# Functional Cell Types

Modular LaTeX document. One file per section under `sections/`; each compiles
on its own (via the `subfiles` package) so sections can be worked on
independently.

## Build

```sh
make                                    # full document -> build/main.pdf
make sec SEC=02_renormalisation_group   # one section   -> build/02_renormalisation_group.pdf
make clean                              # removes build/
```

All intermediate files and PDFs go into `build/`, which is gitignored.

In VS Code with LaTeX Workshop, `.vscode/settings.json` makes Ctrl+Alt+B
compile whichever file is open (a section on its own, or everything from
`main.tex`), writes output to `build/`, and hides that folder in the explorer.

## Layout

| File              | Purpose                                                   |
|-------------------|-----------------------------------------------------------|
| `main.tex`        | Title, section order, bibliography. Edit to add/reorder.  |
| `preamble.tex`    | Packages, theorem environments, layout.                   |
| `macros.tex`      | Notation (`\R`, `\E`, `\norm`, `\RG`, ...). Edit globally. |
| `sections/*.tex`  | One section per file.                                     |
| `references.bib`  | Bibliography (biblatex + biber, author-year).             |
| `figures/`        | Graphics; `\includegraphics{name}` resolves here.         |

## Adding a section

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

## Python library

The `fct` package trains deep networks and solves the matching mean-field theory. The
environment is managed with `uv`; `uv.lock` pins every dependency. The first `uv sync`
downloads the CUDA-enabled torch wheel (several GB).

```sh
uv sync                                                             # create the environment
uv run python -m fct init CACHE_DIR --set data.P=512 --set net.L=2  # write CACHE_DIR/config.json
uv run python -m fct train CACHE_DIR          # generate data, train the network
uv run python -m fct solve CACHE_DIR          # mean-field saddle-point solver
uv run python -m fct plot CACHE_DIR           # figures that need results (--which NAME for one)
uv run python -m fct plot-example CACHE_DIR   # figure that needs no results
```

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
uv run ruff check     # lint
```
