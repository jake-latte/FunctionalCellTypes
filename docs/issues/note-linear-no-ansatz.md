---
type: data-science
complexity: complex
status: in review
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Discuss solving the linear case without the mutual-diagonalisability ansatz

PI request (2026-11-04, `latex/edit.md` item 8). In
`latex/sections/02_renormalisation_group.tex`, the one-hidden-layer linear
problem is closed by the ansatz `\Sigma_q = U D_q U^\top` for all `q` with
`X = U S_X V_X^\top` and `Y = U S_Y V_Y^\top`: one basis `U` diagonalises the
per-frequency data-interaction matrices and both data Gram matrices. Write a
**self-contained LaTeX document** `latex/notes/linear_without_ansatz.tex`
(own preamble, builds standalone with `latexmk -pdf -outdir=build`) that
discusses the possibility of solving the linear case without that ansatz.

## Required content
1. **The problem without the ansatz.** Restate the fixed-point problem as a
   matrix equation: `\Sigma_q = (\tilde A_q^{-1} - \tilde B_q)^{-1}` with
   `\tilde A_q = \lambda_q k_0 X X^\top + \sigma^2 I` and
   `\tilde B_q = (\lambda_q / k_1) \tilde C^{-1} Y Y^\top \tilde C^{-1} - g^2 I`,
   `\tilde C = \frac1N \sum_q \lambda_q \Sigma_q + \frac{1}{k_1 \kappa} I`
   (use the corrected forms from the proofreading pass if they differ; say
   which you use). Identify exactly what the ansatz buys (commuting
   families) and exactly where it fails when `XX^\top` and `YY^\top` do not
   commute.
2. **Structure of the general solution.** Show what can be said exactly:
   e.g. that `\tilde C` is the only coupling between frequencies, so that
   given `\tilde C` every `\Sigma_q` is an explicit matrix function of
   `XX^\top`, `YY^\top` and `\tilde C`; that the fixed point is a single
   `P \times P` matrix equation for `\tilde C` (write it); its symmetries;
   whether it is a Riccati-type / algebraic equation; existence and
   uniqueness considerations (positive-definiteness, the role of `\sigma^2`
   and `g^2`); what happens in the rank-deficient case `K < P` of the
   latent task where `XX^\top` and `YY^\top` share a column space; the
   two-latent case worked explicitly where feasible.
3. **Perturbative treatment.** Expand around the commuting case: write
   `YY^\top = U (S_Y^2 + \epsilon E) U^\top` with `E` not diagonal, and derive
   the first-order correction to `D_q` and to `U` (the rotation of the
   functional basis with `q`). Interpret: this is precisely the mechanism by
   which the functional basis could *depend on spatial scale*, which the
   ansatz forbids by construction, so say what the first-order result
   predicts about `q`-dependent cell types.
4. **Numerical route.** A short description of how to solve the matrix
   fixed point numerically (iteration scheme, damping, a check against the
   ansatz solution when `XX^\top` and `YY^\top` commute), with a `numpy`
   script under `latex/notes/scripts/` that demonstrates it on a small
   instance (`P = 6`, two non-commuting Gram matrices, a ring of `N = 32`
   sites) and reports whether the per-`q` eigenbases rotate. Quote its
   output in the note.
5. **Assessment.** Is the ansatz-free solution worth pursuing for the
   functional-cell-type question, and what would be the cost?

Use the notes' notation. Do not edit section files.

## Inputs / outputs / interfaces
- Inputs: `latex/sections/01_setup.tex`, `02_renormalisation_group.tex`,
  `latex/notes/rg_*.tex` (if any), and, if it exists by the time you start,
  `latex/notes/linear_solution_limits.tex` (another worker's note on the
  same equation; do not edit it).
- Outputs: `latex/notes/linear_without_ansatz.tex`,
  `latex/notes/scripts/linear_without_ansatz.py`.

## Acceptance criteria
- The note builds standalone; the script runs with `uv run python` from the
  git root in under a minute on CPU and prints the quantities quoted in the
  note.
- Sections 1-5 are all present with derivations, not just prose.
- The perturbative result is checked against the numerical solver on the
  small instance (quote the agreement).
- Report: conclusions, and the worktree path and branch.
