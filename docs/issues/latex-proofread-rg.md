---
type: data-science
complexity: complex
status: todo
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Proofread and correct Section 2 and Appendix A1 of the notes with colour-marked edits

PI request (2026-11-04, `latex/edit.md` items 2-4). The notes are a draft
statistical-mechanics derivation (a spatially embedded mean-field theory of a
feedforward network trained by SGLD). Read
`latex/sections/02_renormalisation_group.tex` and `A1_comments.tex` line by
line, as a careful referee would, re-deriving every equation (Section 2 is a
chain of algebra from the single-site distribution to the implicit solution
`eq:linear-soln`; check each step, including normalisations, factors of `N`
and `1/N`, signs, inverses, powers, and the Fourier-basis bookkeeping with
`F`, `F^*`, `q` and `-q`), and correct what is wrong. Every correction is
**marked** with the macros in `latex/macros.tex` so the PI can see at a
glance what changed and why:

- **Trivial typos** in prose or equations (spelling, a dropped index, a
  `\sigma_2` for `\sigma^2`, a missing conjugate star, a missing sum sign
  that the next line restores): replace the wrong text by
  `\typo{corrected text}` (green). No comment needed.
- **Non-trivial but non-load-bearing mistakes** (an equation that is wrong but
  whose error does not propagate into `eq:linear-soln`: a dropped
  normalisation absorbed into a constant, an index contraction written
  wrongly but used correctly afterwards, a wrong sign silently corrected two
  lines later): replace by `\fix{corrected}` (orange) and put a
  `\fixnote{...}` (orange) immediately after the equation or sentence,
  stating (i) what the mistake was, (ii) its consequences, (iii) how it was
  fixed. Keep notes to one to three sentences.
- **Load-bearing / conceptual mistakes** (something `eq:linear-soln` or its
  interpretation depends on: a wrong power of a diagonal element, a sum or a
  `1/N` missing in a self-consistency equation so that the large-`N` scaling
  is wrong, an ansatz that is inconsistent, a step that does not follow):
  correct the text with `\major{corrected}` (red), add
  `\majornote{<note file name>}` (red) right after, and write a
  **self-contained** LaTeX document `latex/notes/rg_<slug>.tex` (own
  `\documentclass{article}` preamble, like the existing
  `latex/notes/latent_covariance_variational_principles.tex`; it must build
  standalone with `latexmk -pdf`) that states the problem precisely, shows
  why it is wrong (with the derivation), what it affects downstream
  (including the corrected form of `eq:linear-soln` if it changes), and how
  to address it. One note per distinct problem; one note may cover several
  related red edits.

Use judgement on the boundary between orange and red: ask "if this stayed
wrong, would `eq:linear-soln` or its limiting behaviour be different?" If
yes, it is red. If `eq:linear-soln` changes, write the corrected equation in
the section as a `\major` edit and keep the label `eq:linear-soln` on it: a
later analysis note is built on it.

Appendix A1 is informal commentary plus a perturbation expansion and a
meta-learning recursion; proofread it at the same standard, but expect mostly
green and orange there.

Do not rewrite for style and do not change notation (a separate item has
already made index placement consistent: layer `^{(l)}`, data indices
superscript, spatial indices subscript; follow it in anything you write).
Do not mark edits you did not make; do not leave unmarked edits.

## Inputs / outputs / interfaces
- Inputs: the two section files above; `latex/macros.tex` (read only, the
  `\typo`, `\fix`, `\fixnote`, `\major`, `\majornote` macros);
  `01_setup.tex` for the definitions Section 2 relies on (**do not edit it**:
  another worker owns Sections 0, 1, 3, 4 in parallel; if you find a mistake
  there, report it instead).
- Outputs: the two section files edited in place; zero or more
  `latex/notes/rg_*.tex` files.
- Build: `cd latex && make`; for a note, `cd latex/notes && latexmk -pdf
  -outdir=build <file>.tex`.

## Acceptance criteria
- `cd latex && make` builds with no undefined references and no new
  warnings; every note under `latex/notes/` builds standalone.
- Every change in `git diff` of the section files is inside a `\typo`, `\fix`
  or `\major` (plus their notes), except for the status line of this file.
- Every `\fix` has a `\fixnote` within a few lines; every `\major` has a
  `\majornote` naming an existing file in `latex/notes/`.
- Each red note states: the problem, the derivation showing it, the
  consequences (with the corrected `eq:linear-soln` if affected), and the
  recommended fix; it builds standalone.
- Your report lists every edit by colour with `file:line` and a one-line
  reason, lists mistakes you noticed in `01_setup.tex` for the other worker,
  flags anything you were unsure about, and gives the worktree path and
  branch.
