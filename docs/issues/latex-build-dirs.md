---
type: infrastructure
complexity: simple
status: done
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Send every LaTeX build into a build/ directory beside its source, and support notes/

PI request (2026-11-06). Builds under `latex/` currently scatter artefacts
into three places: `latex/build/` (from `latex/Makefile`, which sets
`OUTDIR := $(CURDIR)/build` for both `make` and `make sec`), `latex/sections/build/`
(from standalone section builds), and in-place files next to the sources
(`latex/main.aux`, `latex/main.pdf`, ..., and `latex/notes/*.aux|log|out|pdf|toc`)
from bare `latexmk`/`pdflatex` runs and from VS Code, whose
`latex-workshop.latex.outDir` is `%WORKSPACE_FOLDER%/build` (the repo root,
which no longer matches where the LaTeX lives). `latex/.latexmkrc` was
deleted at some point, so a bare `latexmk main.tex` writes in place.

Make the rule simple and uniform: **every build writes to a `build/`
directory inside the directory of the root `.tex` file**, however it is
invoked (Makefile, bare latexmk, VS Code LaTeX Workshop):

- `latex/main.tex`               -> `latex/build/`
- `latex/sections/<sec>.tex`     -> `latex/sections/build/`   (standalone subfile build)
- `latex/notes/<note>.tex`       -> `latex/notes/build/`      (standalone article; new)

The notes in `latex/notes/` are self-contained `\documentclass{article}`
files with their own preambles and no bibliography; they are built one at a
time. Some read figures from `latex/notes/figures/`.

## What to do
- `latex/Makefile`: `make` builds main into `latex/build/`; `make sec SEC=<name>`
  builds `sections/<name>.tex` into `latex/sections/build/`; add
  `make note NOTE=<name>` building `notes/<name>.tex` into `latex/notes/build/`,
  and `make notes` building every `notes/*.tex`; `make clean` removes all
  three `build/` directories and nothing else. Pass `-synctex=1
  -interaction=nonstopmode -file-line-error` consistently so VS Code and the
  Makefile produce the same outputs. Keep the header comment accurate.
- Add a `.latexmkrc` (`$pdf_mode = 1; $bibtex_use = 2; $out_dir = 'build';`
  plus the pdflatex flags above) in each of `latex/`, `latex/sections/` and
  `latex/notes/`, so a bare `latexmk <file>.tex` run from any of those
  directories also lands in that directory's `build/`. latexmk only reads the
  rc file in its cwd, so three small files are needed; say so in a comment.
- `.vscode/settings.json` (repo root): change `latex-workshop.latex.outDir`
  to `%DIR%/build` so LaTeX Workshop writes beside the root file it is
  building; update the comments to describe the three cases (main, section,
  note). Keep the existing `files.exclude` / `search.exclude` /
  `files.watcherExclude` (they already hide `**/build`). Keep
  `rootFile.doNotPrompt` / `rootFile.useSubFile`.
- `.gitignore` (repo root): already ignores `build/` and the artefact
  extensions; add `*-SAVE-ERROR` (biber leaves `main.bbl-SAVE-ERROR` /
  `main.bcf-SAVE-ERROR` on a failed run). No other change.
- `latex/main.tex` header comment and `README.md` "Paper (LaTeX)" section
  (around line 298; note it currently says `Latex/`, which should be `latex/`):
  document the three build targets and where each writes. Keep edits to
  those two files to the comment/README text only.

## Inputs / outputs / interfaces
- Inputs: `latex/Makefile`, `.vscode/settings.json`, `.gitignore`,
  `latex/main.tex` (header comment only), `README.md` (LaTeX section only).
- Outputs: those files edited; three new `.latexmkrc` files (the ones in
  `sections/` and `notes/` delegate to `latex/.latexmkrc`);
  `latex/scripts/check_build_dirs.sh` and a `make check` target that runs it.
- Interfaces: `make`, `make sec SEC=..`, `make note NOTE=..`, `make notes`,
  `make clean`, all run from `latex/`; VS Code Ctrl+Alt+B on any root file.

## Acceptance criteria
- From `latex/`: `make` succeeds and produces `latex/build/main.pdf`;
  `make sec SEC=01_setup` produces `latex/sections/build/01_setup.pdf`;
  `make note NOTE=continuum_limit` produces `latex/notes/build/continuum_limit.pdf`;
  `make notes` builds every `notes/*.tex` into `latex/notes/build/`
  (report any note that fails to compile, but do not edit note content).
- After those builds, `git status --short --ignored latex` lists only the
  three `build/` directories as ignored entries: no `.aux`, `.log`, `.pdf`,
  `.synctex.gz` etc. anywhere beside a `.tex` source.
- From `latex/`, `latex/sections/` and `latex/notes/` respectively, a bare
  `latexmk main.tex`, `latexmk 01_setup.tex`, `latexmk continuum_limit.tex`
  writes only into that directory's `build/`.
- `make clean` removes exactly the three `build/` directories.
- The biblatex/biber step still works for `main.tex` (its bibliography
  resolves; no "Please (re)run Biber" warning left in `latex/build/main.log`
  after `make`).
- Report the branch and worktree path of your work.
