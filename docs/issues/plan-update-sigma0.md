---
type: data-science
complexity: simple
status: done
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Bring the experiment plan up to date with the 2026-11-06 decisions

Edit `latex/notes/fct_experiment_plan.tex` so that an implementing session
reads one consistent document. Blocked on the confirmation of
`rate-penalty-latex-r`, `rate-penalty-code` and `ensemble-extend` (their
outcomes change the dictionary and the sampling protocol).

## PI decisions of 2026-11-07
- Notation: keep $r$ for the rate; the ring distance and lag become $\Delta$
  ($R(\Delta)$, $K(\Delta)$, $\Delta\le6$); the mode index in
  `linear_without_ansatz.tex` becomes $s$; the rank in A1 becomes $\rho$ only
  where it sits near a rate use (done on the `rate-penalty-latex-r` branch).
- Prediction 6: the frame-free statistic $|\langle e^{6i\theta}\rangle|$
  (sixfold: three projected latent axes with sign); no frame decision needed.
- $q_{1/2}$: average the final-state spectra across the $B$ members, one
  $q_{1/2}$ per setting, error bar by bootstrap over members.
- Parametrisation equivalence: each setting at the largest lr its own run-start
  check accepts, run to stationarity, compared on final-state statistics across
  members (two experiments, or an lr axis paired with the parametrisation).

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
- The parametrisation axis as an equivalence test (PI decision pending).
  Review of `whitened-spatial-prior` showed: (a) the two paths started from
  different initial laws (penalty W ~ N(0, 1), whitened from the prior); fixed
  by `net.init = "prior"`, which draws both from their own prior; (b) SGLD in
  $\xi$ is Langevin preconditioned by $S^2$ in $W$, so one shared `train.lr`
  gives the two settings different numbers of relaxation times, and under
  `min_rel_eig = 1e-6` the penalty step check forces lr below about $5\times
  10^{-6}$. The stationary laws agree (up to the eigenvalue floor); the
  dynamics do not. The comparison must therefore be of stationary statistics,
  with each setting at its own admissible lr and run to stationarity, and with
  `min_rel_eig` small enough that the floor does not separate the posteriors
  (the `spatial_tiny` preset floors one eigenvalue at N = 8). Options: pair
  `train.lr` with the parametrisation (two experiments, or a paired axis), or
  run both at the penalty path's lr. The plan's P1-P3 axes must say which.
- After the $g^2\to r$ rename the plan's naming-clash entries (D5 at line ~39,
  the "name clash D5" cell at ~137, item (a) at ~205) describe a clash that no
  longer exists between `net.g` and the notes' symbol: rewrite them.
- Notation (PI decision pending): $r$ is now the rate coefficient and also the
  ring distance in $R(r)$, $K(r)$ (limits note, plan), a mode index in
  `linear_without_ansatz.tex`, `r_{i,j}` in `continuum_limit.tex:398` and a rank
  in `A1_comments.tex:224-228`; the limits note's table row (a1) uses both
  meanings in one cell.
- The summary of conclusions and the order of work list the new item slugs.

## Acceptance criteria
- `make -C latex note NOTE=fct_experiment_plan` builds; no mention of
  `input_noise_var`, `noise=True`, $\zeta$ or `latent_range` remains except
  in the pointer to the note; every numbered prediction states which items it
  needs by slug.
