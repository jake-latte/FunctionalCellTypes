---
type: infrastructure
complexity: simple
status: in progress
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Make the notes' index notation consistent and add the edit-marking macros

PI request (2026-11-04, `latex/edit.md` items 1 and the macro preamble to 2-4).
The notes under `latex/` are a work in progress and their notation drifts
between files and even within equations. Before the notes are proofread, bring
every section to one convention, **without marking these changes** (plain
edits, no colour). Then add the colour macros that the proofreading pass will
use to mark its corrections.

## Notation convention (apply everywhere: `latex/sections/*.tex`)

- **Layer index**: superscript in parentheses, `\cdot^{(l)}`. So `h^l_{\mu,i}`
  becomes `h^{(l)\mu}_{i}`, `W^l_{j,b}` becomes `W^{(l)}_{j,b}`, `W^L_{dj}`
  becomes `W^{(L)}_{\alpha j}`-style, `\Phi_{(L)}^{-1}` becomes
  `[\Phi^{(L)}]^{-1}`, `\Phi^{l-1}` becomes `\Phi^{(l-1)}`, `\sigma_l`
  (the layer-dependent normaliser) becomes `\sigma^{(l)}`, `K^l`, `M^l`,
  `\hat h^{(1), \mu}` etc. likewise. Multiple superscripts on one symbol are
  written as one group, layer first, then data: `h^{(l)\mu}_{i}` (or
  `h^{(l)\,\mu}_i` where readability needs a thin space).
- **Data indices** (`\mu, \nu, m, n, \star`): always superscript.
  `\phi^{(l)}_{\mu,i}` becomes `\phi^{(l)\mu}_{i}`; `\xi^l_{\mu,i}` becomes
  `\xi^{(l)\mu}_i`; `x^{\mu}_a` is already right.
- **Spatial / unit indices** (`i, j, a, b, \alpha, \beta, q, q', k`): always
  subscript. Pairs are written comma-separated in one subscript: `K_{i,a,j,b}`
  stays; `M^{(l)}_{(i,a),(j,b)}` (which appears once) becomes
  `M^{(l)}_{i,a,j,b}` so the two spellings agree. `\delta^{\mu,\nu}` (data) vs
  `\delta_{i,j}` (space) is already the right split: keep it.
- **Kernels**: `\Phi^{(l)\mu,\nu}_{i,j}` is the canonical form (layer, then
  data, superscript; space subscript). Already used in most places; make the
  rest match. The compact matrix forms (`\Sigma_q`, `\tilde A_q`, `D_q`,
  `XX^\top`) in `02_renormalisation_group.tex` are fine as they are: `q` is a
  spatial (Fourier) index and belongs in the subscript.
- Where a symbol is used with inconsistent *names* for the same thing, prefer
  the spelling used in the defining equation and make later uses match. Do NOT
  change the mathematics: if an index looks wrong rather than merely
  misplaced (e.g. a sum over `d` that should be over `a`, a `j` that should be
  an `i`), leave it alone and list it in your final report. The proofreading
  pass marks those in colour.
- Prose is out of scope here (typos get marked in the next pass).

## Edit-marking macros (add to `latex/macros.tex`, new block "Editing marks")

`xcolor` is already loaded by the preamble. Define exactly these names, so the
proofreading work items can rely on them; all must work in both text and math
mode (`\textcolor` does):

```latex
\colorlet{typocolor}{green!50!black}
\colorlet{fixcolor}{orange!90!black}
\colorlet{majorcolor}{red!80!black}
\newcommand{\typo}[1]{\textcolor{typocolor}{#1}}        % trivial typo, corrected text
\newcommand{\fix}[1]{\textcolor{fixcolor}{#1}}          % non-trivial, non-load-bearing correction
\newcommand{\fixnote}[1]{\textcolor{fixcolor}{\textit{[#1]}}}   % orange comment placed near a \fix
\newcommand{\major}[1]{\textcolor{majorcolor}{#1}}      % load-bearing / conceptual correction
\newcommand{\majornote}[1]{\textcolor{majorcolor}{\textit{[See note: #1]}}} % pointer to a file in latex/notes
```

Add a short comment above the block explaining the three-colour scheme (green:
typo; orange: mistake with a nearby orange comment; red: load-bearing mistake,
discussed in a separate note under `latex/notes/`). Add a one-line usage test
of each macro to NO section file: instead, verify by compiling a scratch
snippet in your worktree and deleting it.

## Inputs / outputs / interfaces
- Inputs: `latex/sections/00_motivation.tex`, `01_setup.tex`,
  `02_renormalisation_group.tex`, `03_numerics.tex`, `04_conclusions.tex`,
  `A1_comments.tex`, `latex/macros.tex`. Note `A1_comments.tex` currently has
  no `\documentclass`/`\begin{document}` wrapper: another worker is adding
  that in parallel, so edit only its body and do not add a wrapper yourself.
- Outputs: the same files, edited in place. No new files.
- Build: `cd latex && make` (latexmk + biber, output in `latex/build/`).

## Acceptance criteria
- `cd latex && make` succeeds; `build/main.log` has no "Undefined" references
  and no new warnings versus the branch base (two warnings is the baseline).
- `grep -nE '\^[lL]_|\^\{?[lL]\}?\b[^)]' latex/sections/*.tex` style checks
  find no bare-layer superscripts (`^l`, `^L`, `^{l-1}`, `^{l+1}`) remaining;
  no `_{\mu` / `_{\nu` data-index subscripts remain; `\Phi_{(L)}` is gone.
- Every equation label and `\eqref` still resolves.
- The five macros compile in text and math mode.
- Your final report lists (a) every index you believed to be *wrong* rather
  than misplaced, with file:line, for the proofreading pass; (b) the branch
  and worktree path of your work.
