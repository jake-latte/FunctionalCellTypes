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

The `fct` package trains deep networks and solves the matching mean-field
theory. The environment is managed with `uv`; `uv.lock` pins every dependency.

**Status:** the CLI (`python -m fct ...`) is not implemented yet on this branch; only
config, activations and cache exist so far. The commands under "Intended workflow" are the
plan, and the CLI work item will replace this sentence.

`uv sync` downloads the CUDA-enabled torch wheel (several GB) on first run.

```sh
uv sync          # create the environment (first run downloads torch)
uv run pytest    # run the tests (CPU only, synthetic data)
```

Intended workflow. Everything lives under one `CACHE_DIR`, and every step
resumes from, or reuses, what is already there:

```sh
uv run python -m fct init CACHE_DIR --set data.P=512 --set net.L=2   # writes CACHE_DIR/config.json
uv run python -m fct train CACHE_DIR          # train the network
uv run python -m fct solve CACHE_DIR          # mean-field saddle-point solver
uv run python -m fct plot CACHE_DIR           # figures that need results
uv run python -m fct plot-example CACHE_DIR   # figure that needs no results
```

`docs/design.md` is the interface contract between modules. Changing a config
section that an artefact depends on makes loading it raise `StaleCacheError`;
use a fresh `CACHE_DIR` or delete the stale file.
