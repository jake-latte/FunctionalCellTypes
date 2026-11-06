---
type: data-science
complexity: complex
status: in review
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# The linear theory from a config (`fct/linear_theory.py`)

W5 of `latex/notes/fct_experiment_plan.tex`, confirmed by the PI 2026-11-06,
with two amendments:

1. **sigma -> 0.** The PI treats the input noise $\sigma$ as a formal
   regulariser and takes $\sigma \to 0$ in every analysis (see
   `linear_solution_limits.tex`, Section `sec:sigma0`). The functions keep
   `sig2` as an argument (the theory has it, and the quoted numbers at
   $\sigma^2 = 0.01$ are the regression tests), but `sig2 = 0` must be a
   first-class case: null modes ($s_X = s_Y = 0$) get $d_q = 0$, modes with
   $\lambda_q = 0$ get $d_q = 0$, no division by zero anywhere, and the
   per-mode root is found on the physical branch exactly as for
   $\sigma > 0$. The presets will use $\sigma^2 = 0$.
2. **The rate coefficient is called `r`.** The notes currently write the
   rate penalty as $\frac{g^2}{2}\sum\phi^2$; the PI is renaming $g^2 \to r$
   so that `train.rate_lambda == r` after a pending code change. Name the
   argument `r` in every signature and docstring ("the rate-penalty
   coefficient $r$, written $g^2$ in the notes until the rename").

Read Section D4 (`\label{sec:D4}`) of the plan for the module's API and the
dictionary of Section `sec:dictionary` (theory symbol to `Config` field); the
plan's W5 entry is copied verbatim below. The numbers to reproduce are in the
plan's Section `sec:smoke` and Appendix `app:numbers`, computed by
`latex/notes/scripts/check_linear_soln_limits.py`, which is the reference
implementation to port (its `g2` argument becomes `r`).

  \item \textbf{W5. The linear theory from a config.} Type data-science; complexity complex.\\
  \emph{Description:} the per-mode solution and the ansatz-free kernel as pure functions of a \code{Config} and a dataset, so that every prediction is computed from the same object that configures the run.\\
  \emph{Inputs:} \path{latex/notes/scripts/check_linear_soln_limits.py}, \path{linear_without_ansatz.py}; Section~\ref{sec:D4} and the dictionary of Section~\ref{sec:dictionary}.\\
  \emph{Outputs:} \path{fct/linear_theory.py} with \code{theory\_parameters}, \code{per\_mode\_spectrum}, \code{predicted\_spectra}, \code{spectrum\_summary}, \code{ansatz\_free\_kernel}.\\
  \emph{Acceptance:} reproduces the limits note's P1--P3 numbers from a \code{Config} built as the preset of W8 (Section~\ref{sec:D4}); the ansatz-free kernel equals the per-mode roots to $10^{-10}$ for the orthonormal teacher; \code{theory\_parameters} on that config returns $\lambda_0 = 10.0265$, $\bar\lambda = 1$ and $s_1^2$ within $10\%$ of $213.3$.

## Acceptance criteria
- Those of the plan's W5 (the quoted numbers at $\sigma^2 = 0.01$, the
  ansatz-free kernel against the per-mode roots, `theory_parameters` values).
- At `sig2 = 0`: the S1 shares and ratio of the plan's smoke section recomputed
  with `check_linear_soln_limits.py` at `sig2 = 0` are reproduced; null modes
  return exactly $0$; `predicted_spectra` on the smoke config with
  `input_noise_var` absent (there is no such field yet; $\sigma^2$ comes from
  an explicit argument defaulting to $0$) runs.
- Synthetic-data-first: a test builds a ring config, computes the prediction,
  and checks it against an independent brute-force evaluation of the master
  equation for one small case ($N = 12$, as in the limits note's check).
- Full suite and `ruff` pass.
