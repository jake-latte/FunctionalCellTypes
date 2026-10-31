---
type: infrastructure
complexity: simple
status: in progress
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Add the phase_transition preset (d x lambda_max, linear, one hidden layer); rename the old one lambda_grid

Depends on `lambda-max-prior-scale` and `ensemble-training`.

PI decisions (2026-10-30): the new preset takes the name `phase_transition`; the old
L x phi x rate_lambda x weight_lambda grid becomes `lambda_grid` (its references renamed:
`fct/sweep.py`, `fct/grid_view.py`, `notebooks/build_phase_transition.py`, `jobs/fct_sweep.sh`,
README, tests; existing sweep.json files do not record the preset name, so `pt_sweep/` is
unaffected). Embeddings are RANDOM (uniform on the d-torus) for every population, so any N works
in any d. B = 8.

The preset: grid `spatial.d` in [1, 2, 3, 4] x `spatial.lambda_max` in [0.1, 0.5, 0.75, 0.9, 0.99]
(20 settings, d slowest), phi "id", L = 1, net.B = 8, weight_lambda = 1, every population's
length scale 0.15 (PI CONFIRMED 2026-10-30: 0.15 for every population; the per-weight prior variance shrinking with population size under lambda_max is intended), otherwise `spatial_grid`'s fields (linear teacher,
64 features, 2 latents, SGLD beta 1e4, lr 1e-2, float64, N 512, P 128, 250k epochs, MIR monitor).
It takes no n_lambda (a FIXED_GRID_PRESET); `lambda_grid` keeps taking n_lambda.

Random embeddings already accept any d, and the wrapped Gaussian is a product over axes (a Schur
product of PSD matrices), so no embedding change is needed; the tests below check it.

## Acceptance criteria
- `python -m fct sweep init ROOT --preset phase_transition` writes a sweep.json with the 20
  settings in a stable order, each with phi id, L 1, B 8, random embeddings.
- `--preset lambda_grid --n-lambda k` reproduces the old phase_transition sweep.json exactly.
- Every setting's population kernels (rebuilt from its saved config.json) are PSD and have top
  eigenvalue lambda_max, for d = 1..4.
- The SGLD stability check passes for all 20 settings at the preset's lr (report the worst
  margin: the prior's stiffest curvature grows like 1 / (min_rel_eig lambda_max)^2).
- One setting trains end to end on the CPU at reduced size (few epochs) in the test suite.
