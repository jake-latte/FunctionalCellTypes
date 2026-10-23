---
type: infrastructure
complexity: complex
status: in review
---

<!-- status: todo | in progress | in review | done | blocked | awaiting your confirmation -->

# SGLD training with noise scale beta (alongside Adam)

PI brief: the exemplar notebook and the spatial notebook train with SGLD at inverse temperature beta, as in `Latex/sections/01_setup.tex`: `d(theta) = -grad L dt + sqrt(2/beta) eta`, equilibrium `p(theta) ~ exp(-beta L)`. The spatial notebook must offer SGLD or Adam. Depends on `remove-rate-penalty-p-scaling` (touches the same loss and hash).

## Proposed plan
- `TrainConfig.optimizer: "adam" | "sgld"` (default "adam": existing runs and hashes unchanged because the field enters the hash only through its non-default value) and `beta` (PosFloat, used by SGLD; for Adam ignored and not hashed).
- SGLD step (full batch, Euler-Maruyama): `theta <- theta - lr * grad L + sqrt(2 * lr / beta) * N(0, I)`, with `lr` as the step size dt. Gradient is of the *same* `compute_loss` the repo already uses, so the loss and hash conventions stay in one place. Noise is drawn from the torch global generator so RNG checkpointing already covers it; checkpoint/resume must reproduce an uninterrupted run exactly.
- Convention: the Latex puts `1/beta` on the regularisers and `beta` on the data term (`beta N/(2Pn)(f-y)^2` in the weight), i.e. `exp(-beta L)` with `L = N/(2Pn) sum (f-y)^2 + (1/beta)(P_w + P_r)`. The repo's loss is `0.5 N mse + weight_lambda * weight_l2 + rate_lambda * rate_l2`. Under SGLD at temperature 1/beta this is already `exp(-beta * [0.5 N mse + lam_w ... ])`, so **the Latex regulariser strengths are `lambda = (weight_lambda) * beta`**; we keep `weight_lambda`/`rate_lambda` as the repo's own and document the mapping (the mean-field code already uses `alpha = lambda_w P D_out / beta`). No change to the loss.
- Diagnostics recorded in history for SGLD: gradient norm, and an effective temperature check (mean squared parameter displacement vs. theory on a quadratic test).

## PI decisions (2026-10-22)
- Plan confirmed as written: plain full-batch SGLD, `lr` as the step size, the existing loss unchanged, beta sets only the noise.
- Queued behind `remove-rate-penalty-p-scaling`, which edits `fct/train.py` and `fct/config.py`; starts from the merged main.

## Acceptance criteria
- On a one-parameter quadratic loss, long SGLD samples have variance `1/(beta * curvature)` within Monte-Carlo error (ground-truth test).
- `optimizer="adam"` reproduces current training bit-for-bit; hashes of existing configs unchanged.
- SGLD checkpoint/resume equals an uninterrupted run exactly.
- Tests CPU only, ruff clean, `docs/design.md` documents the mapping to the Latex.
