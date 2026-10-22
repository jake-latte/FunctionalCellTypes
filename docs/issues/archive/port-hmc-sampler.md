---
type: data-science
complexity: simple
status: done
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Port the single-site HMC sampler to `fct/hmc.py` with serialisable state

Move `SingleSiteHMC` and the helpers `_sym`, `project_pd`, `relative_fro`,
`kernel_alignment` from notebook cell 19 into `fct/hmc.py`, numerically
unchanged, and turn the notebook's `validate_sampler` (cell 23) into pytest
tests with known ground truth. Add `state_dict` / `load_state_dict` so the
mean-field solver can checkpoint its chains.

Why: the sampler is the inner engine of the mean-field solver; its three
known-answer checks are the only ground truth we have for it, so they become
permanent tests rather than a cell someone has to remember to run.

## Inputs / outputs / interfaces

Implement exactly the *hmc.py* section of `docs/design.md`. `sym` replaces
`_sym` (public name). `phi_prime` comes from `fct.activations` (do not
duplicate it). `state_dict()` returns only tensors and Python scalars:
chain state in h-coordinates (float64, whitening-independent), `log_eps`,
`_prev_summary` (or None), `n_resets_total`, `n_transitions_total`.
`load_state_dict` restores them (it must work after `set_prior`/`set_tilt`
have been called with the same matrices, i.e. after `_refresh`).
Keep the port faithful: do not change formulas, defaults, dtype handling or
the adaptation rules; small functions are welcome only where they do not
alter the arithmetic. Fix only genuine defects, and record each in the
report (not as history comments in the code — principle 10).

## Acceptance criteria

Tests use `P=20`, CPU, seeded generator, `n_chains=2048`, `warmup=300`,
`n_trans=200` (adjust down only if a test exceeds ~60 s; report the timings).

- Untilted Gaussian (`M=0`, `λ_r=0`): `relative_fro(<hhᵀ>, Σ) < 5 ×` the
  expected Monte Carlo error `1/sqrt(n_chains*n_trans*ess_frac)`.
- Linear φ with PD tilt: `<hhᵀ>` matches `(Σ⁻¹+M)⁻¹` to the same tolerance.
- ReLU with tilt (`λ_r=0.1`): `stein_mismatch < 0.05` and
  `relative_fro(<φφᵀ>, importance-sampling estimate) < 0.05`.
- Acceptance rate after warm-up is within ±0.15 of `target_accept`.
- `state_dict` → fresh sampler (same Σ, M) → `load_state_dict` → `h()`
  equals the original's `h()` to float32 precision, and the next
  `transition()` of both samplers with identically seeded generators agree.
- `project_tilt` output satisfies the stated normalisability condition
  (ψ=φ: `M + λ_r I ⪰ δ λ_r I`; ψ=h: `Σ^{1/2} M Σ^{1/2} ⪰ -(1-δ) I`).
- `project_pd` floors eigenvalues and reports the count clamped.

## Implementation notes
- The notebook's `validate_sampler` importance-sampling check uses `logw = -0.5*((ph@M)*ph).sum(-1) - 0.05*(ph*ph).sum(-1)` for λ_r=0.1 (i.e. ½·λ_r); keep that in the test.
- For the state round-trip test, build the second sampler with the same Σ, M, λ_r, φ, and its own generator seeded identically to the first one's *after* the first has finished warm-up (easiest: reseed both generators to the same value right before the compared `transition()` call).
- Keep the existing diagnostics keys (`accept`, `eps`, `n_divergent`, `n_reset`, `max_abs_dH`, `mean_abs_dH`, `rho1`, `ess_frac`, `h2_mean`, `h2_max`, `prior_h2`, `psi_psi_rel_mc_err`, `stein_mismatch`) and the moments keys (`psi_psi`, `h_h`, `h_u`) exactly — the solver and the plots depend on them.
- Tests must be deterministic (seeded generators) and must not print.
