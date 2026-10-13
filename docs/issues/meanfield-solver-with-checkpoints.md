---
type: data-science
complexity: complex
status: in progress
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Port the mean-field saddle-point solver to `fct/meanfield.py` with checkpoint/resume

Move `empirical_kernels` and `solve` (notebook cell 20, Algorithm 1 of
arXiv:2502.07998 with persistent-HMC single-site expectations) into
`fct/meanfield.py`, driven by `MeanFieldConfig`, and make the outer loop
checkpoint every `save_int` outer steps and resume from the latest checkpoint.

Why: a solve takes hours; it must survive interruption, and its result
(`Phi`, `Phi_hat`, `history`) must be a cached artefact the plots read.

## Plan (confirmed by the PI)

1. **Faithful port.** `solve()` keeps the notebook's algorithm, formulas,
   defaults and diagnostics exactly; its ~40 keyword arguments become fields
   of `MeanFieldConfig` (see `docs/design.md`). Only the packaging changes.
2. **Decompose without changing arithmetic.** The 300-line function is split
   into named steps the outer loop calls in the same order:
   `_init_kernels` (lazy / gaussian / net), `_make_samplers`,
   `_inner_ascent` (Φ̂ ascent with retrospective backtracking),
   `_outer_gradients` (Stein + direct forms, Eq. 31b/c), `_outer_descent`
   (trust-region Φ step with sampler probing and backtracking), `_record`
   (diagnostics). Each has a docstring stating the equation it implements.
3. **Checkpoint at outer-step boundaries.** State = `outer` (next step),
   `Phi`, `Phi_hat`, `history`, `eta`, `eta_hat`, `h2_last`, `eps_ref`,
   per-layer `sampler.state_dict()`, the generator state, `mse_net`, and
   `net_Phi` (if a net was given). Written with `fct.cache.write_checkpoint`
   (prefix `ckpt_`, width=5), `final.pt` with `save_artefact`; the hash is
   `cfg.section_hash("data","net","train","meanfield")`. History arrays are
   stored as lists; `load_meanfield` returns them as numpy arrays again.
4. **Resume** rebuilds the samplers from the checkpointed `Phi`/`Phi_hat`
   (prior/tilt), loads their chain states, and continues from `outer`. No
   warm-up is re-run on resume (the chains are already equilibrated).
5. **Synthetic ground truth** (data-science dev method): with `L=1`, linear
   φ (`readout="h"`), no rate tilt, the saddle point is analytic — the
   single-site density is Gaussian, so the inner fixed point is
   `Φ = (Σ⁻¹ + Φ̂)⁻¹` with `Σ = Φ0/λ_w`, and the outer stationarity gives
   `Φ̂ = -2c A Y Yᵀ A`. Write a test that `solve` on a tiny problem
   (`P=8`, `D_in=8`, `n_chains=256`, few outer steps) drives the fixed-point
   residuals `phihat_fp_resid_stein` below 0.1 and the inner residual below
   its tolerance. A second test checks resume: run `n_outer=4` with
   `save_int=2`, stop, resume, and compare `history` length and `Phi` with an
   uninterrupted run (same seed; equality to float32 chain precision, i.e.
   relative Frobenius < 1e-3).
6. **Print → log.** `print` calls become calls to the injected `log`.

Risks: (a) `solve` currently hard-codes `torch.manual_seed` globally; the
port scopes randomness to the `torch.Generator` only (no global seeding).
(b) The linear-φ validation uses `ident = lambda h: h`, which
`activations.PHI` does not contain; the test passes the callable directly to
`solve`, so `solve`'s `phi` argument stays a callable (the CLI resolves the
name). (c) Exact resumption equality is only expected on CPU with float64
chains; the test uses that.

## Inputs / outputs / interfaces

Exactly the *meanfield.py* section of `docs/design.md`: `empirical_kernels`,
`SolveResult`, `solve(X, Y, L, lambda_w, lambda_r, phi, g, cfg, net=None,
log=print, checkpoint_dir=None, config_hash="")`, `run_meanfield(cfg,
dataset, net, cache_dir, log=print)`, `load_meanfield(cache_dir, cfg,
device)`. Depends on `fct.hmc`, `fct.model`, `fct.data`, `fct.cache`.

## Acceptance criteria

- The analytic `L=1` linear test above passes.
- The resume test above passes and leaves exactly one `ckpt_*.pt` plus
  `final.pt`.
- `run_meanfield` called twice returns the cached result without running
  any outer step the second time.
- `load_meanfield` raises `StaleCacheError` when `cfg.meanfield.beta`
  changes.
- All history keys of the notebook's `keys` list are present with one entry
  per outer step; `align_net`/`dist_net` are `None`-free lists when a net is
  given and absent otherwise.
- A smoke run with `L=2`, ReLU, `P=16`, `n_chains=128`, `n_outer=3`,
  `warmup_trans=20` completes on CPU in under 60 s.
