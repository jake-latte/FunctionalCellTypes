---
type: infrastructure
complexity: simple
status: done
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Draw the linear-teacher latents uniformly on [latent_min, latent_max]

PI decision (2026-11-06), replacing W1 of `latex/notes/fct_experiment_plan.tex`.
The linear teacher draws its latents $Z$ (P, k) i.i.d. uniform on
$[-a, a]$ with `a = latent_range` (`fct/tasks.py:88-90`). The PI's
uncommitted edit changed that to $[0, a]$; instead of committing a fixed
interval, make both ends hyperparameters so the non-isotropic regime
($[0, a]$, or any interval) can be tested from a config.

## What to change
- `DataConfig` (`fct/config.py:61`): replace `latent_range: PosFloat = 1.0`
  by `latent_min: float = -1.0` and `latent_max: float = 1.0`, with a
  validator `latent_min < latent_max`. A config that still carries
  `latent_range` must fail loudly with a message naming the two new fields
  (the strict model already rejects unknown fields; make the message say
  what replaced it).
- `fct/tasks.py:88-90`: `z = latent_min + (latent_max - latent_min) * rand`.
  With the defaults this is `-1 + 2 * rand`, bit for bit the old
  `1.0 * (2 * rand - 1)`; keep it that way (same generator, same order of
  draws, so `W_X`, `W_Y` are unchanged).
- The latent lattice test set (`fct/tasks.py:356-373`, `docs/design.md:313`):
  the three values per latent become `{latent_min, (latent_min + latent_max)/2,
  latent_max}`; docstrings accordingly.
- `fct/phase_view.py:138-146` (the random test set uniform on the same
  interval; its mismatch message).
- `LinearTeacherParams` (`fct/exp.py:152-155`, `:259`): the two fields in place
  of `latent_range`, plumbed to `DataConfig`.
- Presets `fct/presets/*.json` (8 files) and the fixtures under
  `tests/fixtures/*.json` (4 files): `"latent_range": 1.0` becomes
  `"latent_min": -1.0, "latent_max": 1.0`. Legacy fixtures that pin a hash
  (`legacy_config.json`, the meanfield hashes): the data section's dump
  changes, so every hash that includes `data` changes; update the pinned
  hashes and say so in the commit. No `DATA_VERSION` is needed: the field
  names themselves change the hash, so no cache is silently reused.
- `docs/design.md:139, 313, 910` and the README if it names the field.
- Tests: `tests/test_tasks.py`, `test_latent_lattice.py`, `test_phase_view.py`,
  `test_draw.py`, `test_build_spatial_exp.py`, `test_resample.py`,
  `test_config_constraints.py`, `test_exp_spatial.py` use `latent_range`.

## Inputs / outputs / interfaces
- Inputs: the files above; the PI's intent (uniform on an arbitrary interval).
- Outputs: `data.latent_min`, `data.latent_max`; `LinearTeacherParams.latent_min/max`.

## Acceptance criteria
- With the defaults, `linear_teacher` returns `z`, `w_x`, `w_y` bit for bit
  equal to the previous formula (test against `2 * rand - 1` on the same
  generator).
- With `latent_min = 0, latent_max = 1` the latents have mean $1/2$ and
  min/max inside $[0, 1]$ at $P = 20000$; the lattice set is `{0, 0.5, 1}^k`.
- `latent_min >= latent_max` and a config with `latent_range` raise
  `ValueError`s whose messages name the fields.
- Every preset loads (`tests/test_presets.py`), full suite and `ruff` pass.
