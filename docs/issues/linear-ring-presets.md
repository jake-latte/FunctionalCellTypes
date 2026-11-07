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
smoke test's estimator is per member with spread across members (B >= 16), and
record pooling is only a burn-in diagnostic; (f)
the rate coefficient is `r = rate_lambda` once `rate-penalty-code` lands
(check its status before setting `rate_lambda` in the presets).
Depends on `latent-bounds`, `orthonormal-teacher`, `whitened-spatial-prior`,
`linear-theory-module`, `spectrum-module`, `spectrum-monitor`.

  \item \textbf{W8. Presets and the end-to-end smoke test.} Type data-science; complexity complex.\\
  \emph{Description:} the two presets and the synthetic-data-first test of the whole pipeline: the smoke ring sampled by SGLD against the theory of W5.\\
  \emph{Inputs:} Sections~\ref{sec:presets}, \ref{sec:smoke}; W1--W7.\\
  \emph{Outputs:} \path{fct/presets/linear_ring.json} and \path{linear_ring_tiny.json}; a test that runs \code{linear\_ring\_tiny} (S1) through the CLI on a CPU and compares the pooled spectrum after burn-in with \code{predicted\_spectra}.\\
  \emph{Acceptance:} with $B = 4$, $10^4$ steps and records every $10$ steps the test pools about $3.8\times10^3$ records, of which roughly $750$ are independent (records every relaxation time, about $50$ steps at $\mathrm{lr} = 0.005$), so the standard error per $q$ is about $3.6\%$ ($2.6\%$ after $\pm q$ pooling), on a share about $0.01$ and on the ratio at one $|q|$ about $0.35$; the tolerances are set at three standard errors: the S1 shares within $0.03$ of $0.314/0.310$, the ratio over $|q|\le3$ within $7\pm1$, $R(1)/R(0)$ within $0.03$ of $0.881/0.870$; the test reports its own standard errors from \code{record\_sem} and fails if they exceed a third of the tolerance; runtime under two minutes on a CPU; \path{tests/test_presets.py:77} updated for the two new files.

## Acceptance (replaces the plan's record-pooling numbers above)
- `linear_ring` sits in the large-N regime: on every primary statistic (R(1)/R(0),
  R(ell)/R(0), the Gaussian width at ell, the band share, the mode-1/mode-2 band power
  ratio) the exact posterior and `predicted_spectra` differ by less than one third of
  the statistic's 3-SE margin at the preset's B, and the data term moves the
  statistic from the prior by several SEs; both recorded in the description.
- Falsifiers are those ring-averaged statistics of the member-averaged final-state
  spectrum (one sample per member), within 3 x 1.35 delta-method SEs at the run's N
  and B of the prediction, with bootstrap SEs over members below 1.35 SE. The q = 0
  share and q_1/2 are reported only.
- `linear_ring` runs each parametrisation at 0.9 of its own run-start check's bound
  with `min_rel_eig` 1e-6, and records the variance inflation of its stiffest modes.
- `linear_ring_tiny` is a pipeline test, labelled as one: through the CLI on the CPU,
  against its own floored prediction, with a guard that fails a frozen sampler.
- The estimator and thresholds are checked on ensembles drawn from the predicted law
  before the SGLD comparison; full suite and ruff pass.

## Implementation notes

### Presets
- Both: orthonormal teacher, latents on [0, 1], phi id, L 1, g 1, weight_lambda 1,
  rate_lambda 0, SGLD, float64, grid embeddings, `init` prior, `monitor_kind`
  spectrum, no input-noise field, B a shared field. Sizes, lrs and costs are in each
  description.
- lr pairing: experiment axes are a Cartesian product and cannot be zipped, so each
  preset crosses `spatial.parametrisation` with `train.lr` and the diagonal (settings 0
  and 3) is the experiment; setting 1 is refused at run start, setting 2 runs (the lr
  check of the whitened net).
- `linear_ring`: N 512, ell 2 sites, P 256, m = n = 4, K 3, beta 10, B 32,
  min_rel_eig 1e-6, lr 9e-5 (penalty) and 0.96 (whitened), 0.9 of each check's bound;
  2 x 10^6 epochs (the penalty's relaxation time is an estimate, 6 x 10^5 epochs).
- `linear_ring_tiny`: the smoke geometry at beta 0.25, B 256, min_rel_eig 1e-3, lrs
  far inside both bounds (0.9 of the bound would inflate the whitened net's prior modes
  1.4x at N = 16 and beta 0.25).

### Why N = 512, n = 4
`predicted_spectra` is the N -> infinity mean-field limit and omits the readout's
(n/2) log det C~ term, an O(n/N) correction. Reviewer's scan (exact Gibbs sampler, smoke
geometry, beta 10): exact over predicted q = 0 power of mode 1 is 0.53, 0.69, 0.81,
0.90, 0.94 at N = 16, 32, 64, 128, 256, and the old `linear_ring` (N 64, m 8) at beta 10
gives 0.57; at beta 0.25 the theory has the sign of the data effect wrong (about 5%
below the prior on both rings). At ell = 2, P = 256, K = 3, m = n = 4, beta = 10 the
exact-minus-predicted error on the primary statistics, in SEs at B = 64, is 0.1-0.8 at
N = 256 and 0.1-0.55 at N = 512 (relative 0.1-1.3% and 0.06-0.7%). The plan's regime
rule asks for at most one SE at B_max, the largest B run including the fourfold
follow-up extension: with B = 32 (B_max = 128) N = 512 gives at most 0.78 SE, while
N = 256 would give 1.1. The data term moves R(1)/R(0) by 33/5/3.5 SE (modes 1/2/3) and
the band share by 21/3.1/2.2 SE at B = 32, twice that at B_max. N = 1024 would allow
B = 64 (B_max 256) at twice the cost.

### Estimator
A member's tuning map is a real Gaussian field, so its power is D(q) chi^2_1 at q = 0,
N/2 and D(q) Exp(1) elsewhere, identical at q and -q: folding +-q adds no independent
samples. The delta method on the member mean gives each statistic's SE
(`tests/linear_ring_helpers.py`); the estimator test checks the spreads, the bootstrap
SEs and the acceptance rate on ensembles drawn from the predicted law, and that a flat
spectrum and equal mode amplitudes are rejected.

### Unresolved
- The whitened run at 0.9 of its bound inflates its stiffest (data-carrying) mode 1.8x
  and its prior-dominated modes 1.05x; setting 2 (whitened at the penalty lr) measures
  the effect on the falsifiers. Not yet run at full size.
- The penalty's run length is an estimate; the drift check over the records must
  confirm stationarity.
