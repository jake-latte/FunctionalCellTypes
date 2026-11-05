---
type: data-science
complexity: complex
status: done
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Proofread and correct Sections 0, 1, 3, 4 of the notes with colour-marked edits

PI request (2026-11-04, `latex/edit.md` items 2-4). The notes are a draft
statistical-mechanics derivation (a spatially embedded mean-field theory of a
feedforward network trained by SGLD). Read `latex/sections/00_motivation.tex`,
`01_setup.tex`, `03_numerics.tex` and `04_conclusions.tex` line by line, as a
careful referee would, re-deriving every equation in `01_setup.tex`, and
correct what is wrong. Every correction is **marked** with the macros in
`latex/macros.tex` so the PI can see at a glance what changed and why:

- **Trivial typos** in prose or equations (spelling, a dropped index, a `2`
  for a `\sqrt{\cdot}`, a missing factor of `1/2` that nothing downstream
  depends on, `\sigma_2` for `\sigma^2`): replace the wrong text by
  `\typo{corrected text}` (green). No comment needed.
- **Non-trivial but non-load-bearing mistakes** (an equation that is wrong but
  whose error does not propagate into the main results: a wrong sign that is
  silently corrected two lines later, a missing inverse in a side result, a
  dropped normalisation that is absorbed into a constant, an index that is
  summed over the wrong set): replace by `\fix{corrected}` (orange) and put a
  `\fixnote{...}` (orange) immediately after the equation or sentence,
  stating (i) what the mistake was, (ii) its consequences, (iii) how it was
  fixed. Keep notes to one to three sentences.
- **Load-bearing / conceptual mistakes** (something the main results depend
  on: a wrong saddle-point equation, a normalisation that changes the
  large-`N` scaling, an identity that does not hold, an argument that does not
  go through): correct the text with `\major{corrected}` (red), add
  `\majornote{<note file name>}` (red) right after, and write a
  **self-contained** LaTeX document `latex/notes/setup_<slug>.tex` (own
  `\documentclass{article}` preamble, like the existing
  `latex/notes/latent_covariance_variational_principles.tex`; it must build
  standalone with `latexmk -pdf`) that states the problem precisely, shows
  why it is wrong (with the derivation), what it affects downstream, and how
  to address it. One note per distinct problem; one note may cover several
  related red edits.

Use judgement on the boundary between orange and red: ask "if this stayed
wrong, would `eq:linear-soln` in Section 2, or the saddle-point equations, be
different?" If yes, it is red.

Do not rewrite for style and do not change notation (a separate item has
already made index placement consistent: layer `^{(l)}`, data indices
superscript, spatial indices subscript; follow it in anything you write).
Do not mark edits you did not make; do not leave unmarked edits. Prose in
Section 0 is informal by design; fix only genuine typos there.

## Inputs / outputs / interfaces
- Inputs: the four section files above; `latex/macros.tex` (read only, the
  `\typo`, `\fix`, `\fixnote`, `\major`, `\majornote` macros);
  `02_renormalisation_group.tex` and `A1_comments.tex` for context only
  (**do not edit them**: another worker owns them in parallel).
- Outputs: the four section files edited in place; zero or more
  `latex/notes/setup_*.tex` files.
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
  consequences, and the recommended fix; it builds standalone.
- Your report lists every edit by colour with `file:line` and a one-line
  reason, flags anything you were unsure about, and gives the worktree path
  and branch.
