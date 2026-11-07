---
type: data-science
complexity: complex
status: in review
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

Every size, lr, run length, regime number, measured relaxation and cost is in the presets'
descriptions (`fct/presets/linear_ring*.json`), which are the only copy; the analysis is
`fct/linear_ring.py`, the shared design is in `docs/design.md` ("The linear-ring presets").

### Presets
- All three: orthonormal teacher, latents on [0, 1], phi id, L 1, g 1, weight_lambda 1,
  rate_lambda 0, SGLD, float64, grid embeddings, `init` prior, `monitor_kind`
  spectrum, no input-noise field, B a shared field.
- `linear_ring` (whitened, the base point and every sweep) and `linear_ring_equivalence`
  (its penalty twin) are two experiments with the same one-value beta axis, so the same
  exp hash and setting directory: they need separate ROOTs. Run plan: both at the base
  point first, then the beta and k sweeps on the whitened path only.
- Each real run takes 0.9 of its own run-start check's bound. On the whitened path the
  resulting lr bias exceeds the regime's one-SE budget on mode 1 and the ratio, so at
  that lr the regime is not accurate by the plan's rule (numbers in the description).
- The penalty run length follows from the whitened relaxation in lr-epoch units through
  the mobility mapping written out in its description, with the late-window relaxation
  fit as the guard.
- `linear_ring_tiny` is the pipeline check, the two paths crossed with their lrs, which
  stay far inside both bounds.
- `phase_transition`: its ensemble size is a shared field instead of a one-value `net.B`
  axis, so `exp extend` can grow it; its exp hash and directory names change, its train
  and solve hashes do not. Existing ROOTs read their own exp.json and are unaffected;
  re-creating one from the preset with `--force` would leave its old directories unnamed.

### The large-N regime
`predicted_spectra` is the N -> infinity mean-field limit and omits the readout's (n/2)
log det C~ term, an O(n/N) correction. An exact Gibbs sampler (off-repo) gives, on the
smoke geometry at beta 10, exact over predicted q = 0 power of mode 1 of 0.53, 0.69,
0.81, 0.90, 0.94 at N = 16, 32, 64, 128, 256; at beta 0.25 the theory has the sign of
the data effect wrong. `linear_ring`'s N and B are the smallest that keep the exact
posterior within one SE at B_max of the prediction on every primary statistic.

### Estimator
A member's tuning map is a real Gaussian field, so its power is D(q) chi^2_1 at q = 0,
N/2 and D(q) Exp(1) elsewhere, identical at q and -q: folding +-q adds no independent
samples. The delta method on the member mean gives each statistic's SE
(`fct/linear_ring.py`); the estimator test checks the spreads, the bootstrap SEs and the
acceptance rate on ensembles drawn from the predicted law, and that a flat spectrum and
equal mode amplitudes are rejected.

### Unresolved
- The whitened path's lr bias exceeds the regime's accuracy budget at 0.9 of its check.
- Modes 2 and 3 of the measured base point sit below the prediction by more than the
  regime error alone; `exp extend` to B_max tells the offset from the sampling noise.
