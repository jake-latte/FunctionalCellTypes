---
type: data-science
complexity: complex
status: in progress
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Analyse the one-hidden-layer linear solution (eq:linear-soln) in its limiting cases

PI request (2026-11-04, `latex/edit.md` item 6). At the end of
`latex/sections/02_renormalisation_group.tex` the notes arrive at an implicit
self-consistency equation, labelled `eq:linear-soln`, for the scalar `c`
(the regularised hidden-layer data kernel's eigenvalue) in a one-hidden-layer
linear network with a translation-invariant spatial prior whose Fourier
eigenvalues are `\lambda_q`. The derivation makes the mutual-diagonalisability
ansatz `\Sigma_q = U D_q U^\top` with `X = U S_X V_X^\top`,
`Y = U S_Y V_Y^\top`. A proofreading pass has already marked corrections in
that section in colour (`\typo` green, `\fix` orange, `\major` red) and any
red edit has a companion note in `latex/notes/rg_*.tex`: read those first.

Write a **self-contained LaTeX document** `latex/notes/linear_solution_limits.tex`
(own `\documentclass{article}` preamble, builds standalone with
`latexmk -pdf -outdir=build`) that analyses this solution for intuition.

## Required content
1. **Starting point.** Restate the model, the assumptions and `eq:linear-soln`
   as it stands after the proofreading pass (with the corrected form if a red
   edit changed it). Re-derive it yourself from the single-site distribution,
   verifying every step. **If your re-derivation yields substantial changes
   to the conclusion** (a different power, a missing `1/N` or sum, a different
   dependence on `\lambda_q`), put a section at the **top** of the document
   that states how your result differs from the current form of
   `eq:linear-soln`, why, and what the corrected equation is; then perform
   the whole analysis on the corrected equation. Be explicit about which
   version each later result refers to.
2. **Limiting cases.** Analyse, with derivations and closed forms where they
   exist, and a sentence of interpretation for each:
   - `\sigma \to 0` (the input-noise regulariser removed);
   - `g \to 0` (no firing-rate penalty);
   - `k_0 = k_1 \equiv k` (equal pre- and post-synaptic prior scales);
   - a specified spectrum for the `\lambda_q` (choose at least one concrete
     family, e.g. a wrapped-Gaussian / squared-exponential kernel on a ring
     of `N` sites with length scale `\ell`, so `\lambda_q` decays with `|q|`
     on the scale `1/\ell`; state the normalisation of `\lambda_q` and of
     `F` you use);
   - the **latent student-teacher task**: `X = Z W_X^\top`, `Y = Z W_Y^\top`,
     `Z \in \mathbb{R}^{P \times K}` the latents with entries i.i.d. uniform
     on `[0, a]`, `W_X` (`m \times K`) and `W_Y` (`n \times K`) with
     orthonormal columns. Work out `S_X`, `S_Y` and `U` for this task
     (including the effect of the non-zero mean of the uniform distribution
     on the leading singular direction, and the `P \to \infty` behaviour of
     `Z^\top Z / P`), and whether the mutual-diagonalisability assumptions
     hold exactly, approximately, or not at all for it.
3. **Functional cell types.** The question the notes are after: which
   features of this task lead to the development of *functional cell types*,
   meaning cells whose activity correlates with a particular latent `z_k`,
   and at what spatial scale (which `q`) those cells emerge. Use `D_q` (the
   per-frequency data-interaction spectrum, in the basis `U`): a functional
   cell type at scale `1/|q|` means a latent direction whose weight `d_q`
   dominates at that `q`. Derive how `d_q` for each latent direction depends
   on `\lambda_q`, `\sigma`, `g`, `k_0`, `k_1`, `\kappa`, `a`, `K`, `P` and
   the singular values of `X` and `Y`; state the conditions (inequalities or
   crossovers) under which (a) all latents share the same spatial profile,
   (b) different latents separate across spatial scales, (c) nothing is
   spatially structured. Say which of these are driven by the task
   (`Z`, `W_X`, `W_Y`) and which by the prior (`K`, `k_0`, `k_1`, `g`).
4. **Conditions to test numerically.** A concrete, numbered list of
   predictions, each with: the parameter setting (values, not symbols), the
   quantity to measure in a trained network (e.g. the correlation of each
   neuron's activity with each latent, binned by spatial frequency of the
   neuron-activity map; the spectrum of the empirical `\Sigma_q`), the
   predicted outcome, and what observation would falsify the theory. These
   feed a follow-up item that maps them onto the `fct` codebase, so keep
   them precise.

Use the notation of the notes (layer index `^{(l)}`, data indices
superscript, spatial indices subscript). Where you have to make a choice
(normalisation of `F`, definition of `\lambda_q`, whether `P \gg K`), say so
and keep it consistent. Prefer small, checkable lemmas over long prose. Do
not edit the section files.

## Inputs / outputs / interfaces
- Inputs: `latex/sections/01_setup.tex`, `02_renormalisation_group.tex`
  (after the proofreading pass), `latex/notes/rg_*.tex` (red notes, if any),
  `latex/notes/latent_covariance_variational_principles.tex` (an earlier
  note on the same latent task: reuse its conventions where they fit).
- Outputs: `latex/notes/linear_solution_limits.tex` and nothing else
  (figures may go in `latex/notes/figures/` if you generate them with a
  script; the script then lives in `latex/notes/scripts/` and is committed).
- Build: `cd latex/notes && latexmk -pdf -outdir=build linear_solution_limits.tex`.

## Acceptance criteria
- The note builds standalone with no undefined references.
- Section 1 re-derives `eq:linear-soln` and states explicitly whether it
  agrees with the notes; if not, the discrepancy section is at the top.
- Each of the five limiting cases has a derivation and an interpretation.
- The functional-cell-type section gives explicit conditions, each tagged as
  task-driven or prior-driven.
- The numerical-conditions list has at least four numbered predictions in
  the format above.
- Every non-trivial algebraic claim is either derived in the note or
  verified by a short numerical check (a `numpy` script under
  `latex/notes/scripts/`, run with `uv run python`) whose result is quoted.
- Report: a summary of conclusions, any disagreement with the notes, and the
  worktree path and branch.
