---
type: infrastructure
complexity: simple
status: in review
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

## Results (worker, 2026-10-31)

SGLD stability at the preset's lr 1e-2 (beta 1e4, weight_lambda 1, min_rel_eig 1e-2, N 512, with
the real output population of 64 units). The check's quantity is `lr * c * stiffest` =
lr * k / 2, with k = 2 c r s the curvature of the stiffest prior mode, so it must be < 1 and
lr * k < 2. The stiffest curvature is 1 / (min_rel_eig lambda_max)^2 where the floor binds
(d <= 3). All 20 settings pass.

At lambda_max 0.1, d <= 3, lr * k = 1.0 on the stiffest modes: Euler-Maruyama then gives exactly
2x the target variance (target / (1 - lr k / 2)). Total prior power of U / V is inflated by about
+13% / +3.6% / +0.8% at d = 1 / 2 / 3 and +0.2% at d = 4; every other lambda_max row is < 0.3%.
The data term adds < 5e-4 to lr * k (reviewer's measurement). lr * k per setting, with the real
output population (64 units, which sets the V factor and the worst case):

| lambda_max | d = 1, 2, 3 | d = 4 |
|---|---|---|
| 0.1 | 1.0000 | 0.302 |
| 0.5 | 0.0400 | 0.0121 |
| 0.75 | 0.0178 | 0.0054 |
| 0.9 | 0.0123 | 0.0037 |
| 0.99 | 0.0102 | 0.0031 |

Run length: the softest prior mode relaxes on tau ~ 2 beta lambda_max^2 / (lr weight_lambda) ~
2e6 lambda_max^2 epochs (1e4 at 0.1, ~1e6 at 0.99), so at 250k epochs settings with
lambda_max >= 0.5 are not fully stationary from the N(0, 1) init (reviewer: prior power still
7-14% above target at >= 0.75; mse still falling at d 1, lambda 0.99). lr and n_epochs are left
as they are pending a PI decision.
