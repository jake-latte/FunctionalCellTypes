---
type: data-science
complexity: complex
status: in progress
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

## Implementation notes (worker, 2026-11-07)

### Presets
- `fct/presets/linear_ring_tiny.json`: N 16, P 32, m = n = 4, K 2, hidden
  length_scale 2 (box 16), B 256 (shared field), 10^4 epochs, monitor every
  250, device cpu, `min_rel_eig` 1e-3.
- `fct/presets/linear_ring.json`: N 64, P 256, m = n = 8, K 3, hidden
  length_scale 4 (box 64), B 64, 2 x 10^5 epochs, monitor every 500, device
  auto, `min_rel_eig` 1e-4.
- Both: orthonormal teacher, latents on [0, 1], phi id, L 1, g 1,
  weight_lambda 1, rate_lambda 0, beta 0.25, SGLD, float64, grid embeddings,
  `init` prior, `monitor_kind` spectrum, no input-noise field.
- lr pairing: experiment axes are a Cartesian product and cannot be zipped, so
  each preset crosses `spatial.parametrisation` [penalty, whitened] with
  `train.lr` [penalty lr, whitened lr] and the diagonal (settings 0 and 3) is
  the experiment. Setting 1 is refused at run start; setting 2 runs (an lr
  check of the whitened path). Tiny: 6e-4 / 0.005; full: 1.2e-4 / 0.005.
- lr rule (not the literal largest accepted lr): the penalty runs at
  lr k_max = 0.48 (its check refuses >= 2), the whitened net at
  lr lambda_max = 0.031 (tiny) / 0.025 (full) as logged (its check refuses 1).
  At the literal limits the stiffest modes' variance is inflated 2x or more;
  for the whitened net at beta 0.25 that is every prior mode.
- `min_rel_eig` departs from the plan's 1e-6: under 1e-6 the penalty check
  refuses any lr above 2.5e-6 (tiny) / 5e-6 (full), and the soft modes would
  need ~10^6-10^7 epochs. 1e-3 / 1e-4 move the predicted R(1)/R(0) by 3e-3 /
  9e-4 (each setting is compared with its own prediction, which includes the
  floor).

### Estimator and tolerances (per member, B = 256)
A member's tuning map is a real Gaussian field, so its power is D(q) chi^2_1 at
q = 0, N/2 and D(q) Exp(1) elsewhere, identical at q and -q: folding +-q adds
no independent samples (the plan's "halved by pooling +-q" does not hold for a
real map). Delta-method SEs of statistics of the member-averaged final-state
spectrum (S1, sigma = 0): q = 0 share 0.021, R(1)/R(0) 0.0042, power over
|q| <= 3 relative to the prediction 0.042, mode-1/mode-2 band-power ratio 0.44.
SE_MAX = 1.35 x these (the 99th percentile of a bootstrap SE over the true SE
is 1.30 on synthetic ensembles); tolerance = 3 SE_MAX: share 0.087, lag-1
0.0174, band 0.171, ratio 1.77; q_1/2 exact. Derivation in
`tests/linear_ring_helpers.py`, checked on maps drawn from the predicted law in
`tests/test_linear_ring_estimator.py` (spreads within 15% of the delta method,
>= 98% of bootstrap SEs below SE_MAX, >= 97% acceptance, flat spectrum and
equal mode amplitudes always rejected). At B = 16 a share's SE would be 0.085.

### Predicted versus plan (sigma^2 = 0, seed-0 data, measured s_k^2)
Tiny (whitened / penalty floor): q_1/2 1, 1; shares 0.314 / 0.313 (penalty
0.314 / 0.313); R(1)/R(0) 0.883 / 0.883 (0.880 / 0.879); ratio 7.39 at q = 0,
7.37 over |q| <= 3 (plan at sigma^2 = 0.01: shares 0.314 / 0.310, R(1)/R(0)
0.881 / 0.870, ratio 7.04 from the formula s_k^2; the measured s^2 ratio is
7.33). Full: q_1/2 2; shares 0.157; R(1)/R(0) 0.969; d_0 ratio 9.36 (plan
10.08 from the formula s_k^2).

### Measured (smoke test, CLI, one CPU thread, 56 s for both settings)
| statistic | predicted (pen / wh) | penalty | whitened |
|---|---|---|---|
| share_1 | 0.314 / 0.314 | 0.331 +- 0.022 | 0.295 +- 0.019 |
| share_2 | 0.313 / 0.313 | 0.295 +- 0.021 | 0.305 +- 0.019 |
| lag1_1 | 0.880 / 0.883 | 0.878 +- 0.005 | 0.876 +- 0.004 |
| lag1_2 | 0.879 / 0.883 | 0.881 +- 0.004 | 0.883 +- 0.004 |
| band_1 | 1 | 0.981 +- 0.039 | 0.973 +- 0.038 |
| band_2 | 1 | 1.007 +- 0.041 | 1.050 +- 0.041 |
| ratio | 7.37 | 7.18 +- 0.41 | 6.82 +- 0.39 |
| q_1/2 | 1, 1 | 1, 1 | 1, 1 |
Burn-in diagnostic (second-half minus first-half time average after 2000
epochs, across members): every statistic within 2 SE of zero.

### Unresolved
- At beta 0.25 (S1) the prediction is within 0.11 SE of the prior alone
  (tau_1 = 0.008), so the smoke test cannot distinguish the theory from
  "the data does nothing"; it checks the sampler and measurement.
- At beta 10 (S3) on the smoke ring, SGLD (whitened, B = 256, lr 0.1 and 0.02)
  is stationary but gives mode 1 about half the predicted q = 0 power
  (normalised power 0.51 at q = 0, matching at q >= 1) and mode 2 below the
  prior at q = 0 (0.8); share_1 0.38 versus 0.540. At N = 64 the q = 0 ratio is
  0.72, at N = 256 1.00 (shares 0.0359 versus 0.0355): a finite-N failure of
  the mean-field theory, so the smoke ring cannot test the data term.
- The PI should confirm the lr rule and the `min_rel_eig` values above.
