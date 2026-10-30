---
type: infrastructure
complexity: simple
status: awaiting your confirmation
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Add the phase-transition grid preset: d x lambda_max, linear activation, one hidden layer

Depends on `torus-embedding-any-d`, `lambda-max-prior-scale`, `ensemble-training`.

A fixed-grid preset over `spatial.d` in [1, 2, 3, 4] and `spatial.lambda_max` in
[0.1, 0.5, 0.75, 0.9, 0.99] (20 settings), phi "id", L = 1, ensemble of B networks per setting.
Other fields follow `spatial_grid` (linear teacher, 64 features, 2 latents, SGLD beta 1e4,
float64, N 512, P 128, MIR monitor) unless the PI says otherwise. The preset name, B and the
hidden length scale are PI decisions (see the PM's reply).

Interfaces: `fct/sweep.py` (`FIXED_GRID_PRESETS`, `PRESET_NAMES`, the CLI choice list follows),
`README.md`, `jobs/fct_sweep.sh` if it lists presets.

## Acceptance criteria
- `python -m fct sweep init ROOT --preset <name>` writes a sweep.json with the 20 settings in a
  stable order (d slowest), each with phi id, L 1, B as decided.
- Every setting's config validates and its derived weight_lambda puts the hidden kernel's top
  eigenvalue at that setting's lambda_max (checked through the saved config.json).
- One setting trains end to end on the CPU at reduced size (tiny epochs) in the test suite.
- The SGLD stability check passes for all 20 settings at the preset's lr.
