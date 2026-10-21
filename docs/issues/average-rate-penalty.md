---
type: infrastructure
complexity: simple
status: done
---

# Average the rate penalty over the P samples

`rate_l2 = phi(h).square().sum()` summed over all P x N x L entries, so at P=4096 and
rate_lambda=1e-3 the penalty swamped the data term and many tuned sweep settings collapsed to the
all-zero network. PI decision: `rate_l2` is now summed over neurons and layers and averaged over
samples. `weight_l2`, the data term and lr are unchanged.

## Changes

- `fct/train.py`: `compute_loss` divides the rate penalty by P; history key `rate_l2` keeps its name
  and now holds the averaged value.
- `fct/config.py`: `LOSS_VERSION` (2) enters every hash that includes the `train` section, so all
  earlier train, sweep, meanfield and analysis-with-train caches are rejected once as stale.
- Tests: hand-computed loss, independence of P (rows replicated), history, old-definition
  checkpoint and final.pt rejected, hash pins updated (intentionally changed).
- Docs: `docs/design.md`, docstrings in `fct/train.py`, `fct/draw.py`, `fct/meanfield.py`.

- `weight_lambda` default 1e-4 (PI decision; enters the hash by value, no loss version bump).
- Sweep: a finished setting from another loss version or weight_lambda is `stale` (status, run, report).
- Mean-field: no default rate tilt; `run_meanfield` raises unless `meanfield.rate_tilt` is set when
  `train.rate_lambda != 0`.

## Open

- The PI decides the mean-field rate tilt: `rate_lambda / P_train` (exact for `n_points == P_train`)
  or `rate_lambda / n_points` (solved points treated as the whole training set).
- `playground.ipynb` sums the penalty (default rate_lambda=0): the library departs from it by design.
