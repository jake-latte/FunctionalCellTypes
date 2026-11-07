---
type: infrastructure
complexity: simple
status: done
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# The spectrum monitor and its wiring (`train.monitor_kind`)

W7 of `latex/notes/fct_experiment_plan.tex`, confirmed 2026-11-06. Depends on
`spectrum-module`. Amendments to the text quoted below:

- There is no `noise=True` forward (no $\eta$ in the code); the monitor calls
  `net(dataset.x)`.
- `monitor_hex` replaces `monitor_cos4` (PI decision 2026-11-07). The monitor
  records the complex hexatic moment $\langle e^{6i\theta}\rangle$ as (Re, Im),
  whose modulus is frame-free: it does not depend on where the SVD puts $u_2$
  in the contrast plane, whereas $\langle\cos 4\theta\rangle$ is measured
  against the SVD axes and averages to zero for neurons each tuned to one of
  the three latents (they sit $120^\circ$ apart). `hex` and `theta_hist` are
  recorded only for $K \ge 3$ and mean something only at $K = 3$.
- The 20-step criterion is met by
  `test_the_spectrum_monitor_leaves_training_unchanged`: a 20-epoch monitored
  run of one network writes `monitor_spectrum` of shape (records, layers,
  $K+1$, $N$), `monitor_hex` and `monitor_theta_hist`, and ends at weights
  identical to the unmonitored run's.

  \item \textbf{W7. The spectrum monitor and its wiring.} Type infrastructure; complexity simple.\\
  \emph{Description:} a training monitor that records the spectra and the angle statistics every \code{monitor\_int} steps, selected by a config field, so that an experiment produces the measurement as part of its history.\\
  \emph{Inputs:} W6; \path{fct/monitor.py}, \path{fct/config.py:112--116}, \path{fct/exp.py:875}, \path{fct/train.py:347--355}; Section~\ref{sec:measure}, ``The monitor''.\\
  \emph{Outputs:} \code{make\_spectrum\_monitor}; \code{train.monitor\_kind} deleted from the hashed dump at \code{"mir"}; \code{run\_setting} choosing the monitor by it.\\
  \emph{Acceptance:} a monitored 20-step run writes \code{monitor\_spectrum} of shape (records, layers, $K+1$, $N$), \code{monitor\_cos4} and \code{monitor\_theta\_hist}; an ensemble run adds the member axis; the monitor draws nothing from the global RNG (the training stream is unchanged by monitoring, as for the MIR monitor); the \code{"mir"} default keeps every hash (a pinned-hash test).
