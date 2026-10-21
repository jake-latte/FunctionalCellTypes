---
type: infrastructure
complexity: simple
status: in review
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

## Open

- The mean-field solver's rate tilt (`rate_tilt = rate_lambda`, multiplying phi^T phi summed over the
  solved points) is the summed convention; matching the averaged network needs a PI decision (see
  the report of this item). Not changed.
- `playground.ipynb` sums the penalty (default rate_lambda=0): the library departs from it by design.
