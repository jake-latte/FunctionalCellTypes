# Functional Cell Types

Modular LaTeX document. One file per section under `sections/`; each compiles
on its own (via the `subfiles` package) so sections can be worked on
independently.

## Build

```sh
make                      # full document -> main.pdf
make sec SEC=02_renormalisation_group   # one section -> sections/02_renormalisation_group.pdf
make clean
```

Or directly: `latexmk -pdf main.tex`, and inside `sections/`
`latexmk -pdf 01_setup.tex`.

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
