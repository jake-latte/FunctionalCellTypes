---
type: infrastructure
complexity: simple
status: in progress
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# The spectrum monitor and its wiring (`train.monitor_kind`)

W7 of `latex/notes/fct_experiment_plan.tex`, confirmed 2026-11-06. Depends on
`spectrum-module`. Amendment: there is no `noise=True` forward (no $\eta$ in
the code); the monitor calls `net(dataset.x)`.

  \item \textbf{W7. The spectrum monitor and its wiring.} Type infrastructure; complexity simple.\\
  \emph{Description:} a training monitor that records the spectra and the angle statistics every \code{monitor\_int} steps, selected by a config field, so that an experiment produces the measurement as part of its history.\\
  \emph{Inputs:} W6; \path{fct/monitor.py}, \path{fct/config.py:112--116}, \path{fct/exp.py:875}, \path{fct/train.py:347--355}; Section~\ref{sec:measure}, ``The monitor''.\\
  \emph{Outputs:} \code{make\_spectrum\_monitor}; \code{train.monitor\_kind} deleted from the hashed dump at \code{"mir"}; \code{run\_setting} choosing the monitor by it.\\
  \emph{Acceptance:} a monitored 20-step run writes \code{monitor\_spectrum} of shape (records, layers, $K+1$, $N$), \code{monitor\_cos4} and \code{monitor\_theta\_hist}; an ensemble run adds the member axis; the monitor draws nothing from the global RNG (the training stream is unchanged by monitoring, as for the MIR monitor); the \code{"mir"} default keeps every hash (a pinned-hash test).
