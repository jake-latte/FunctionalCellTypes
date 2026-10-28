---
type: infrastructure
complexity: simple
status: in progress
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Add the `spatial_grid` preset: phi x latents x hidden length scale

PI request (2026-10-28): a preset for a grid of spatially embedded networks, searching
- `net.phi` over ["id", "tanh", "relu"]
- the linear teacher's `n_latents` over [2, 4]
- the hidden population's kernel length scale over [0.05, 0.1, 0.15, 0.2]. With L = 1 this one kernel is both K^1_post (hidden side of U) and K^2_pre (hidden side of V); the PI chose to tie them (one axis) rather than split the hidden kernel by role. Input and output kernels stay at 0.15.

24 settings. Fixed values: rate_lambda 0, L 1, P 128 samples, g 1, N 512, weight_lambda 20 (the PI gave lambda_w = 1e-3 in the old convention; under the lambda_w / (2 beta) convention at SGLD beta 1e4 that is 2 x 1e4 x 1e-3 = 20). Everything else as in `spatial_n_scan`.

## Inputs / outputs / interfaces
- `fct/sweep.py`: `spatial_grid()` registered in FIXED_GRID_PRESETS (so in PRESET_NAMES and the CLI's --preset), using the existing axis paths `net.phi`, `data.n_latents`, `spatial.hidden.kernel.length_scale`.
- docs/design.md preset list and README list `spatial_grid` with its grid.

## Acceptance criteria
- `preset("spatial_grid")` expands to exactly 24 settings, every combination once, with the fixed values above; tested.
- `sweep init` + `sweep count` prints 24; `jobs/fct_sweep.sh ROOT --dry-run` shows an array of 24 (not submitted).
- A smoke run of the id, tanh and relu settings at the smallest length scale trains on CPU with finite losses.
- Full suite green, ruff clean, docs updated.
