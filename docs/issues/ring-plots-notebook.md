---
type: data-science
complexity: simple
status: in progress
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Plots and the results notebook for the linear ring

W9 of `latex/notes/fct_experiment_plan.tex`, confirmed 2026-11-06. Depends on
`linear-ring-presets`. Amendment: every measured summary is per member
(final state of each chain); error bars are the spread across members; the
notebook also shows the `"penalty"` and `"whitened"` settings side by side.

  \item \textbf{W9. Plots and the results notebook.} Type data-science; complexity simple.\\
  \emph{Description:} the figures of Section~\ref{sec:measure} and a notebook that shows one setting against its prediction.\\
  \emph{Inputs:} W5, W6, W8; \path{fct/draw.py}; \path{notebooks/build_spatial_exp.py} as the pattern.\\
  \emph{Outputs:} the \code{plot\_*} functions; \path{notebooks/build_linear_ring.py}; a table cell comparing measured and predicted summaries with standard errors.\\
  \emph{Acceptance:} the notebook builds and executes on the smoke ROOT of W8; every \code{plot\_*} draws into a given \code{ax} and makes its own figure only without one; tests on synthetic inputs.
