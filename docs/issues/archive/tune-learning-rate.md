---
type: data-science
complexity: simple
status: done
---

<!-- status: todo | in progress | in review | done | blocked | awaiting your confirmation -->

# Tune the learning rate of the modularity sweep on the local GPUs

Find one learning rate (a single global value preferred) for the 36-setting sweep, using N and epochs
larger than `tiny` and far smaller than the cluster `full` run, on the two local GPUs.

## Acceptance criteria
- Synthetic-first tests for the table and selection logic (known best lr, diverged run never selected and
  counted as failure, ties reported, single-lr verdict True / False) and for the held-out evaluation
  (shapes, determinism, training cache untouched): `tests/test_lrtune.py`, `tests/test_resample.py`, CPU only.
- GPU runs through the real sweep machinery, resumable, logs in `tmp/lr_tune*/`; results table under `docs/`;
  short written summary with setup, timings, per task and depth tables, figures and a recommendation:
  `docs/lr_tuning.md`, `docs/lr_tuning_results.csv`, `docs/figures/`.
- `full` preset not changed; recommended change stated.

## Result (2026-10-21)
- Recommendation, provisional: one global `lr = 1e-3` at 15000 epochs, already the `full` preset's value, so no
  preset change. The PI accepted it as good enough; no further tuning. lr 3e-4, 1e-3 and 3e-3 are tied at 15000
  epochs (loss within 0.2%, held-out MSE within 2%); 1e-3 is chosen by the tie-break and by that acceptance.
  No per-task or per-depth values, no schema change.
- All tuning was done in the collapsed regime: 15 of the 24 tuned settings end at the all-zero network under
  `weight_lambda = rate_lambda = 1e-3` (all what-where settings, linear teacher and most object-action
  settings at `rate_lambda = 1e-3`) and carry no information on the lr needed to learn. L = 3 and the other 12
  grid settings were not run. Details and the numbers behind it: `docs/lr_tuning.md`.
- Held-out MSE, the specified primary criterion, is confounded by the collapse, so the loss and the 9
  settings that learn were used as well.

## Open for the PI / separate items
- Regulariser scales (separate item, not part of this one). What the analysis in `docs/lr_tuning.md` says about
  it: what-where collapses from the weight penalty alone (at L = 1, `rate_lambda = 0`: 4.25 <= 4.61), so
  averaging the rate penalty over P will not rescue it; the summed rate penalty is what collapses linear
  teacher and object-action at `rate_lambda = 1e-3`; the one object-action relu L=1 rate-1e-3 "learner" exploits
  that at L = 1 the readout bypasses phi, so it is no evidence that the rate penalty is benign.
- Decide whether collapsed nets are intended to be reported (the notebook's MIR curves for them describe nets
  that output zero) or the regime should be changed; if the scales change, re-check the learning rate.
- GPU time used: about 3 h on the two local GPUs, nothing submitted to SLURM.
