---
type: data-science
complexity: simple
status: in progress
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Plots and the results notebook for the linear ring

W9 of `latex/notes/fct_experiment_plan.tex`, confirmed 2026-11-06. Depends on
`linear-ring-presets`. Amendment: the notebook also shows the `"penalty"` and `"whitened"`
settings side by side. The amendment's "every measured summary is per member;
error bars are the spread across members" is superseded by the plan
(protocol (ii) of `latex/notes/fct_experiment_plan.tex`): every statistic is read off the member-averaged
final-state spectrum, one sample per member, with its error from the bootstrap
over members; per-member values are diagnostics only.

  \item \textbf{W9. Plots and the results notebook.} Type data-science; complexity simple.\\
  \emph{Description:} the figures of Section~\ref{sec:measure} and a notebook that shows one setting against its prediction.\\
  \emph{Inputs:} W5, W6, W8; \path{fct/draw.py}; \path{notebooks/build_spatial_exp.py} as the pattern.\\
  \emph{Outputs:} the \code{plot\_*} functions; \path{notebooks/build_linear_ring.py}; a table cell comparing measured and predicted summaries with standard errors.\\
  \emph{Acceptance:} the notebook builds and executes on the smoke ROOT of W8; every \code{plot\_*} draws into a given \code{ax} and makes its own figure only without one; tests on synthetic inputs.
