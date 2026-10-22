---
type: infrastructure
complexity: simple
status: in review
---

<!-- status: todo | in progress | in review | done | blocked | awaiting your confirmation -->

# Remove the 1/P scaling from the firing-rate penalty

PI brief (todo.md, "Housekeeping"): the rate penalty goes back to being summed over the P samples, i.e. it no longer carries the 1/P introduced by `average-rate-penalty` (LOSS_VERSION 2).

## Changes
- `fct/train.py` `compute_loss`: `rate_l2 = phi(h).square().sum()` over samples, neurons and layers. History key `rate_l2` keeps its name and now holds the summed value. Update the docstring.
- `fct/config.py`: `LOSS_VERSION = 3` (comment history: 1 summed, 2 averaged, 3 summed again). Every earlier train/sweep/meanfield cache becomes stale by design.
- Mean field: the solver's rate tilt is a per-sample coefficient over the solved points, so with a summed penalty the matching tilt is exactly `rate_lambda`. Make `rate_tilt=None` default to `train.rate_lambda`, remove `RateTiltError` / `_require_explicit_rate_tilt` and their handling in `cli.py`, keep `MeanFieldConfig.rate_tilt` as an explicit override. Update the `fct/meanfield.py` module docstring.
- Tests: hand-computed loss, the "independent of P (rows replicated)" test must now assert the penalty scales with P, stale-checkpoint tests use the new version, the only literal hash pin (`DATA_HASH`) excludes LOSS_VERSION and needs no change, rate-tilt tests updated.
- Docs: `docs/design.md`, docstrings mentioning averaging (`fct/train.py`, `fct/draw.py`, `fct/meanfield.py`, `fct/lrtune.py` if present), `docs/status.md`.

## Acceptance criteria
- `compute_loss` with P replicated rows gives a rate term P times that of one copy.
- Checkpoints / final.pt written under LOSS_VERSION 2 are rejected as stale; sweep status shows them `stale`.
- `run_meanfield` with `train.rate_lambda != 0` and no `rate_tilt` runs, using `rate_tilt = rate_lambda`; an explicit `rate_tilt` still overrides.
- `uv run pytest -q`, ruff check and format clean (CPU only).

## Notes for the PI (decision made here, easy to revert)
Defaulting the tilt to `rate_lambda` is a consequence of the summed penalty (the open question in `average-rate-penalty.md` disappears), not something the brief states.
## Open
- `docs/lr_tuning.md` was measured under the summed penalty (LOSS_VERSION 1) at `weight_lambda = 1e-3`, so this change restores the penalty form that was tuned. The tuning is historical only because the default `weight_lambda` is now 1e-4; the runs are reproducible with `--set train.weight_lambda=1e-3`. The learning rate is not re-tuned.
