---
type: data-science
complexity: complex
status: todo
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Presets `linear_ring`, `linear_ring_tiny` and the end-to-end smoke test

W8 of `latex/notes/fct_experiment_plan.tex`, confirmed 2026-11-06 with
amendments: (a) the embedding is deterministic (`"grid"`, as the plan has it;
never `"random"`); (b) no `input_noise_var` field exists, $\sigma^2 = 0$
throughout, and the predicted numbers are the $\sigma^2 = 0$ ones;
(c) the latents are drawn on `[latent_min, latent_max] = [0, 1]` (W1 as
revised); (d) `spatial.parametrisation` is an axis with both values; (e) the
smoke test's estimator is per member with spread across members, so the tiny
preset uses `B = 16`, and record pooling is only a burn-in diagnostic; (f)
the rate coefficient is `r = rate_lambda` once `rate-penalty-code` lands
(check its status before setting `rate_lambda` in the presets).
Depends on `latent-bounds`, `orthonormal-teacher`, `whitened-spatial-prior`,
`linear-theory-module`, `spectrum-module`, `spectrum-monitor`.

  \item \textbf{W8. Presets and the end-to-end smoke test.} Type data-science; complexity complex.\\
  \emph{Description:} the two presets and the synthetic-data-first test of the whole pipeline: the smoke ring sampled by SGLD against the theory of W5.\\
  \emph{Inputs:} Sections~\ref{sec:presets}, \ref{sec:smoke}; W1--W7.\\
  \emph{Outputs:} \path{fct/presets/linear_ring.json} and \path{linear_ring_tiny.json}; a test that runs \code{linear\_ring\_tiny} (S1) through the CLI on a CPU and compares the pooled spectrum after burn-in with \code{predicted\_spectra}.\\
  \emph{Acceptance:} with $B = 4$, $10^4$ steps and records every $10$ steps the test pools about $3.8\times10^3$ records, of which roughly $750$ are independent (records every relaxation time, about $50$ steps at $\mathrm{lr} = 0.005$), so the standard error per $q$ is about $3.6\%$ ($2.6\%$ after $\pm q$ pooling), on a share about $0.01$ and on the ratio at one $|q|$ about $0.35$; the tolerances are set at three standard errors: the S1 shares within $0.03$ of $0.314/0.310$, the ratio over $|q|\le3$ within $7\pm1$, $R(1)/R(0)$ within $0.03$ of $0.881/0.870$; the test reports its own standard errors from \code{record\_sem} and fails if they exceed a third of the tolerance; runtime under two minutes on a CPU; \path{tests/test_presets.py:77} updated for the two new files.
