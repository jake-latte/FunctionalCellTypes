---
type: infrastructure
complexity: simple
status: in review
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Tuning-map spectra: the pure analysis module (`fct/spectrum.py`)

W6 of `latex/notes/fct_experiment_plan.tex`, confirmed by the PI 2026-11-06,
with one change of emphasis: the PI does not trust one SGLD chain to be
ergodic enough to yield several independent networks, so every estimator is
defined **per network** (one trained network, or one member of an ensemble,
gives one sample of every summary), and spread is taken across members.
Pooling of records along one chain is a diagnostic, not an estimator: keep
`record_sem` (it diagnoses burn-in) but do not make any estimator depend on
pooling.

Read Section `sec:measure` of the plan ("Measurement: functions, inputs,
outputs, maths") for the definitions (tuning maps, the latent basis, per-mode
spectra, the null mean, $R(r)$, $q_{1/2}$, the $q = 0$ share, lag-1, the
angle statistic $\cos 4\theta$ in the orthonormal frame). The plan's W6
entry is copied verbatim below.

  \item \textbf{W6. Tuning-map spectra: the pure analysis module.} Type infrastructure; complexity simple.\\
  \emph{Description:} the functions that turn hidden activity into tuning maps, their unitary DFT power, the null-direction power and the summary statistics.\\
  \emph{Inputs:} Section~\ref{sec:measure}, ``Pure analysis''.\\
  \emph{Outputs:} \path{fct/tuning_maps.py} (\code{latent\_basis}, \code{tuning\_maps}, \code{power\_spectrum}, \code{null\_power}, \code{fold\_q}, \code{spectrum\_summary}, \code{gaussian\_width}, \code{contrast\_angle\_statistic}, \code{theta\_histogram}, \code{average\_records}, \code{record\_sem}).\\
  \emph{Acceptance:} on a synthetic $h = \sum_ku_k\otimes t_k$ with $t_k$ a Gaussian field of known spectrum, \code{power\_spectrum} recovers it within the $1/\sqrt n$ sampling error over $n = 10^4$ draws; \code{null\_power} equals the explicit projection on a random orthonormal complement to $10^{-12}$; \code{spectrum\_summary} of the limits note's $d_q$ gives its $q_{1/2}$, share and lag-1 numbers; every function rejects bad shapes loudly.

## Acceptance criteria
- Those of the plan's W6, on synthetic inputs with known answers (a map built
  from a known spectrum recovers it; a null direction recovers the floor; the
  orthonormal-frame angle statistic is $0$ for an isotropic sample and
  negative for the oblique frame, as the limits note warns).
- Every function takes arrays and returns arrays; no file I/O, no `Config`.
- Full suite and `ruff` pass.
