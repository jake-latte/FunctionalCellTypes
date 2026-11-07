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

## Acceptance
- `linear_ring` (whitened) and `linear_ring_equivalence` (its penalty twin) sit in the
  large-N regime: on every primary statistic (R(1)/R(0), R(ell)/R(0), the Gaussian width
  at ell, the band share, the mode-1/mode-2 band power ratio) the exact posterior and
  `predicted_spectra` differ by at most one SE at B_max (the preset's B with the fourfold
  `exp extend` follow-up), and the data term moves each statistic from the prior by
  several SEs; both recorded in the description.
- Rule of the experiments: an agreement test fails when |observed - predicted| exceeds 3
  bootstrap SEs at the run's B, and counts only if it persists at 4B. The q = 0 share and
  q_1/2 are reported only.
- Each real run takes 0.9 of its own run-start check's bound, `min_rel_eig` 1e-6; the
  descriptions record the resulting inflation and the measured lr bias.
- Stationarity: the relaxation fit of the records leaves a deficit below one SE at B_max.
- `linear_ring_tiny` is a pipeline test, labelled as one, with its own fixed thresholds,
  against its own floored prediction and with a guard that fails a frozen sampler.
- The estimator and the pipeline thresholds are checked on ensembles drawn from the
  predicted law before the SGLD comparison; full suite and ruff pass.

## Implementation notes

### Presets
- All three: orthonormal teacher, latents on [0, 1], phi id, L 1, g 1, weight_lambda 1,
  rate_lambda 0, SGLD, float64, grid embeddings, `init` prior, `monitor_kind`
  spectrum, no input-noise field, B a shared field. Sizes, lrs and costs are in each
  description.
- `linear_ring` (whitened, the base point and every sweep) and `linear_ring_equivalence`
  (its penalty twin) are two experiments with the same one-value beta axis: the paths
  need different lr and n_epochs, and an experiment's axes can only cross them.
  Run plan: both at the base point first, then the beta and k sweeps whitened only.
- `linear_ring`: N 512, ell 2 sites, P 256, m = n = 4, K 3, beta 10, B 32, lr 0.96
  (0.9 of the whitened check at the prior start), 10^4 epochs. Measured by the
  reviewer against lr 0.2: mode 1's R(1)/R(0) and R(2)/R(0) shift by -1.2 to -1.3 SE at
  B_max = 128, its band share by -0.7 SE, the ratio by -1.3 SE (modes 2 and 3 at most
  0.25); at stationarity lambda_max = 1.19, lr lambda_max = 1.14, past the check's
  bound, stiffest-mode inflation about 2.3x. That lr bias alone exceeds the one-SE
  accuracy budget, so at this lr the regime is not accurate by the plan's rule.
- `linear_ring_equivalence`: penalty, min_rel_eig 1e-6, lr 9e-5 (0.9 of its check),
  2.5 x 10^7 epochs (five e-folds of the measured relaxation, about 5 x 10^6 epochs per
  e-fold at this lr), about 56 GPU hours at B = 32.
- `linear_ring_tiny`: the pipeline check, smoke geometry at beta 0.25, B 256,
  min_rel_eig 1e-3, the two paths crossed with their lrs, which stay far inside both
  bounds (0.9 of the bound would inflate the whitened net's prior modes 1.4x there).
- `phase_transition`: B = 8 is now a shared field instead of a one-value `net.B` axis,
  so `exp extend` can grow it; its exp hash and directory names (no `_B8`) change,
  its train and solve hashes do not. Existing phase_transition ROOTs keep their old
  directories, which the new exp.json does not name.

### Why N = 512, n = 4
`predicted_spectra` is the N -> infinity mean-field limit and omits the readout's
(n/2) log det C~ term, an O(n/N) correction. An exact Gibbs sampler (off-repo), smoke
geometry, beta 10: exact over predicted q = 0 power of mode 1 is 0.53, 0.69, 0.81,
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
(`fct/linear_ring.py`); the estimator test checks the spreads, the bootstrap
SEs and the acceptance rate on ensembles drawn from the predicted law, and that a flat
spectrum and equal mode amplitudes are rejected.

### Unresolved
- At the PI's lr (0.9 of the check) the whitened path's lr bias exceeds the regime's
  accuracy budget on mode 1 and the ratio (above).
- The penalty run's length rests on the whitened relaxation time scaled by the lr; the
  relaxation fit of its records must confirm it.
