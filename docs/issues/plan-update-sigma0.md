---
type: data-science
complexity: simple
status: todo
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Bring the experiment plan up to date with the 2026-11-06 decisions

Edit `latex/notes/fct_experiment_plan.tex` so that an implementing session
reads one consistent document. Blocked on the confirmation of
`rate-penalty-latex-r`, `rate-penalty-code` and `ensemble-extend` (their
outcomes change the dictionary and the sampling protocol).

## Changes
- $\sigma \to 0$ everywhere: D2 and W4 are replaced by a pointer to
  `input_noise_average.tex`; `input_noise_var` disappears from the presets,
  the dictionary and every invocation; predictions 1-3, 7, 8 are stated with
  their $\sigma^2 = 0$ numbers only; predictions 4 and 5 are marked deferred
  (they need the sampler of the note); W12 shrinks to P6.
- W1 is `latent-bounds` (`latent_min`, `latent_max`; the preset uses
  `[0, 1]`); the lattice set is `{min, mid, max}`.
- W3 adds `spatial.parametrisation` as an axis of every run (W10, W11).
- Sampling protocol: one sample per member (the final state), spread across
  members, `B` as the sample size, `exp extend` to add members; the record
  pooling of `sec:protocol` and the acceptance numbers of W8 and P6 are
  restated in terms of `B` (P6's "$10^3$ neuron-samples" becomes $64 B \ge 10^3$,
  so $B \ge 16$).
- The rate coefficient is $r$ = `rate_lambda` (after the two rate items);
  the dictionary row and every `--set rate_lambda=` value are restated
  ($g^2 = 1$ at $\beta = 0.25$ was `rate_lambda = 2.0`; it becomes `1.0`).
- The analysis module is `fct/spectrum.py` (the plan says `fct/tuning_maps.py`
  at its lines 266 and 429); W7 and W9 import from `fct.spectrum`.
- The measurement API as merged: `spectrum_summary` returns
  `Summary(share0, lag1, d0, d_nyq, R)` without $q_{1/2}$ (plan lines ~274,
  ~340, ~365 read $q_{1/2}$ from it); `half_power_q` applies to averaged
  spectra only; `record_sem` and `integrated_autocorr_time` return a
  `reliable` mask and need at least 200 records after burn-in and
  $n \ge 20\tau$ per element.
- Prediction 6's frame (PI decision pending). The spectrum-module review
  showed that even on $[0, a]$ the contrast singular values are degenerate in
  expectation ($Pa^2/12$ each), so the SVD contrast frame $(u_2, u_3)$ is a
  random rotation within the contrast plane and $\langle\cos 4\theta\rangle$,
  which is not rotation-invariant, is only defined relative to that random
  frame; histograms from different datasets are not comparable. On symmetric
  latents ($[-a, a]$) all $K$ singular values are degenerate and the "mean
  direction" is undefined; `latent_basis` now refuses such data. The plan must
  state the frame prediction 6 uses; the candidate is the orthonormalised
  latent axes themselves (Gram--Schmidt on $z_2$ then $z_3$, which keeps the
  axis of $z_2$ exact and makes the statistic a statement about the latents
  rather than about a random frame), with the limits note's oblique-frame bias
  re-derived for that frame.
- Prediction 6's statistic (PI decision pending). The spectrum re-review
  showed that at $K = 3$ the three latents project onto the contrast plane as
  axes $120^\circ$ apart, so $\langle\cos4\theta\rangle$ averages to zero over
  them: neurons each tuned to exactly one latent give $\langle\cos4\theta\rangle
  = 0.036$ (below the $0.1$ threshold) and $\langle\cos6\theta\rangle = -0.998$.
  As written, P6 passes even when every neuron is an individual-latent cell
  type. Candidate: the frame-independent $|\langle e^{6i\theta}\rangle|$
  (sixfold symmetry of the projected latent axes, $\pm$ sign), which also
  removes the dependence on the noise-fixed $(u_2, u_3)$ frame.
- $q_{1/2}$ is a statistic of a mean spectrum: on one record $d_0$ and
  $d_{N/2}$ are single $\chi^2_1$ draws, so a per-network $q_{1/2}$ needs an
  average over something (PI decision: records within a network are pooling,
  which the PI distrusts; members are the other option).
- The summary of conclusions and the order of work list the new item slugs.

## Acceptance criteria
- `make -C latex note NOTE=fct_experiment_plan` builds; no mention of
  `input_noise_var`, `noise=True`, $\zeta$ or `latent_range` remains except
  in the pointer to the note; every numbered prediction states which items it
  needs by slug.
