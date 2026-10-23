---
type: data-science
complexity: complex
status: in progress
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
