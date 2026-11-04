---
type: infrastructure
complexity: simple
status: done
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Add an appendix with A1, A2, ... numbering and move the neuroscience comments into it

PI request (2026-11-04, `latex/edit.md` item 5). `latex/sections/A1_comments.tex`
holds a section "Some Comments on Neuroscience" that is not included anywhere
in `latex/main.tex` and is not a `subfiles` subfile (it has no
`\documentclass[../main.tex]{subfiles}` / `\begin{document}` wrapper). Make
it the first appendix, numbered **A1** (not "A"), so that further appendices
come out as A2, A3, and so on, and so that `make sec SEC=A1_comments` builds
it standalone like the other sections.

## What to do
- In `latex/main.tex`, replace the commented-out appendix stub with a real
  back-matter block *after* `\printbibliography`:
  `\appendix`, then a redefinition so that sections are numbered `A1`, `A2`,
  ... (e.g. `\renewcommand{\thesection}{A\arabic{section}}` with the section
  counter reset by `\appendix`), then `\subfile{sections/A1_comments}`.
  Equations and theorems inside the appendix should number as `(A1.1)`,
  `(A1.2)` etc. automatically because the preamble has
  `\numberwithin{equation}{section}`; check that `\cref`/`\eqref` to appendix
  labels print "A1.1"-style, and that cleveref says "Appendix A1" for a
  `\cref{sec:...}` of the appendix section (add `\label{sec:comments}` to it).
  Note `\setcounter{section}{-1}` at the top of main.tex (Motivation is
  Section 0) must not leak into appendix numbering.
- Wrap `A1_comments.tex` as a subfile: add
  `\documentclass[../main.tex]{subfiles}` and `\begin{document}` at the top
  and `\end{document}` at the end, matching the other section files. Touch
  nothing else in that file: another worker is editing its body in parallel.
- Keep the comment in `main.tex` that explains how to add the next appendix
  (`A2_...tex`).

## Inputs / outputs / interfaces
- Inputs: `latex/main.tex`, `latex/sections/A1_comments.tex`,
  `latex/preamble.tex` (read only; prefer not to edit it).
- Outputs: those files edited in place.
- Build: `cd latex && make` and `cd latex && make sec SEC=A1_comments`.

## Acceptance criteria
- `cd latex && make` succeeds with no undefined references; the PDF ends with
  a section headed "A1 Some Comments on Neuroscience" after the bibliography,
  and the `\eqref`s inside it (to `eq:saddle-phi-hat`, `eq:backward-pass`,
  `eq:ff-network`) resolve to the numbers of Section 1.
- `make sec SEC=A1_comments` builds a standalone PDF.
- Sections 0-4 keep their numbering; the appendix's equations are numbered
  `A1.n`.
- Report the branch and worktree path of your work.
