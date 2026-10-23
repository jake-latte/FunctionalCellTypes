---
type: data-science
complexity: complex
status: done
---

<!-- status: todo | in progress | in review | done | blocked | awaiting your confirmation -->

# Importance-weighted sampling in the saddle-point solver, and split solver hyperparameters

PI brief (Mean Field 1): in addition to HMC, offer simple importance-weighted sampling; thread its hyperparameters through the context; stay backwards-compatible with training caches that lack the new solver hyperparameters; ideally separate global problem-definition hyperparameters from training- and solver-specific ones.

## Proposed plan
1. **Sampler.** `fct/importance.py`: `ImportanceSampler` with the same interface the solver uses on `SingleSiteHMC` (draw/advance, the expectations `<psi psi^T>`, `<u h^T>`, accept-like diagnostics). Proposal = the Gaussian prior N(0, Sigma_l) (already whitened by a Cholesky factor); weights = exp(-U(h)) with U the tilt (`1/2 psi^T M psi + 1/2 lambda_r phi^T phi`), self-normalised; fresh independent draws each transition (no persistent chains). Diagnostics fill the existing history keys where meaningful (effective sample size fraction in `hmc_ess_frac`, max weight share) and a separate `is_*` set where not; HMC-only keys are NaN. A hard check that the weights are finite; log-sum-exp for stability.
2. **Solver wiring.** `MeanFieldConfig.sampler: "hmc" | "importance"` (default "hmc", behaviour unchanged) and `n_importance_samples`, `importance_proposal_scale` (inflate the proposal covariance, default 1). `_new_sampler` dispatches on it.
3. **Config / caches.** Split the mean-field config into problem-definition fields (`n_points`, `beta`, `readout`, `prior_scale`, `alpha`, `learning_prefactor`, `rate_tilt`: they define the equations being solved) and solver fields (everything about Algorithm 1, HMC, importance sampling, logging). Old flat `config.json` files still load (a pre-validator moves the flat keys into the solver group). The solver result's cache hash covers data, net, train and the *problem* fields; solver fields are stored in the artefact for provenance but only the explicitly-versioned ones that change the fixed point invalidate it. Training caches are unaffected (the meanfield section never entered their hash); adding solver fields never makes a training cache stale. A hash-pin test proves a config written by the current code loads and hashes identically.

## PI decisions (2026-10-22)
- Plan confirmed. The proposal is the Gaussian prior N(0, Sigma_l): keep the sampler as simple and transparent as possible, no tilt-aware / Laplace proposal.
- Changing `sampler` makes an existing mean-field result stale (it enters the result hash); diagnostics-only fields do not.
- Queued behind `remove-rate-penalty-p-scaling`, which edits the same files (meanfield.py, config.py, cli.py); starts from the merged main.

## Acceptance criteria
- On a 1-D / small-P problem with a known single-site density (Gaussian tilt, closed form), both samplers recover `<psi psi^T>` within Monte-Carlo error; ESS and weight diagnostics behave (collapse flagged when the tilt is extreme).
- `sampler="hmc"` reproduces current solver output bit-for-bit under a fixed seed (regression test on a tiny problem).
- A tiny end-to-end solve with `sampler="importance"` runs, checkpoints and resumes.
- Old `config.json` with only the old keys loads; its meanfield hash is unchanged.
- Tests CPU only, ruff clean, `docs/design.md` updated.

## Notes for the PI (decisions made here, easy to revert)
- `log_int` and `save_int` still enter the mean-field result hash. They change no number, but they were in the hash before
  the split, and leaving them out would change the hash of every existing result (every cached solve would go stale once).
  The rule "diagnostics-only fields do not invalidate" holds for every field added from now on: `config.LEGACY_SOLVER_FIELDS`
  is frozen, and a test makes every new solver field pick a hash rule.
- Added ESS checks to the solver (see "Decisions made in review" for their final form). They are needed because the
  variance-based MC error reads ~0 when one draw carries all the weight, so on its own it cannot see a collapse.
- Under `sampler="importance"` the HMC-only fields (`n_chains`, `n_leap`, ...) still enter the hash. That is harmless:
  changing one makes an importance result stale when it need not.
- `--set meanfield.n_chains=...` is now an error that says to use `meanfield.solver.n_chains`. Flat `config.json` files
  load. Mixing both on the command line would be ambiguous, because `--set` edits the dumped (nested) config.
- New history column `is_max_weight`. The solver dashboard gains ESS / weight-share panels and leaves out all-NaN columns.
- The pinned fixtures (`tests/fixtures/`, written by d1d3091) are bit-for-bit on this host. A different BLAS or thread
  count could reorder float reductions and fail them; regenerate only on purpose (`tests/pinned_runs.py`).
- Merge with the SGLD item: the result-hash pins include the train section. Main (with SGLD) is merged in; adam keeps
  `optimizer` / `beta` out of the train hash, so the pins hold.

## Decisions made in review (PM, 2026-10-23; the PI can overturn them)
- The first version gated on an absolute ESS fraction: a step whose run fell below `min_ess_frac` was reverted. But the
  ESS of the prior proposal belongs to the point, not the step, and it falls with P. At P >= 48 with defaults, every step
  was reverted, the step sizes collapsed to their floors, no error was raised, and `run_meanfield` wrote the frozen
  initial guess as `final.pt`.
- Now: (i) the gate is step-relative, like `max_h2_growth`. A step is reverted when the ESS fraction after it is below
  `min_ess_drop` (new field, default 0.1) times that of the accepted point before it. (ii) `min_ess_frac` is an absolute
  floor on accepted points (the initial point, each validated inner step, each outer-gradient run, so it also applies with
  one inner step). Below it, `solve` raises `InadequateSamplerError` saying the prior proposal does not suit the problem,
  and nothing is written as final.
- Limitation, documented in design.md: prior-proposal importance sampling is for small P and mild tilts. Measured
  layer-1 ESS fraction at the start of relu, L=2, beta=1e3: 0.08 (P=8), 0.03 (P=16), 0.0085 (P=48; seed-dependent: 0.013 at seed 0), 2e-4 to 2e-3 (P=100).
  At the default `n_points=100`, `sampler="importance"` now fails loudly at the initial point; use HMC there.
