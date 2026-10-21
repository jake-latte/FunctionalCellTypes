---
type: data-science
complexity: simple
status: in review
---

<!-- status: todo | in progress | in review | done | blocked | awaiting your confirmation -->

# Tune the learning rate of the modularity sweep on the local GPUs

Find one learning rate (a single global value preferred) for the 36-setting sweep, using N and epochs
larger than `tiny` and far smaller than the cluster `full` run, on the two local GPUs.

## Acceptance criteria
- Synthetic-first tests for the table and selection logic (known best lr, diverged run never selected and
  counted as failure, ties reported, single-lr verdict True / False) and for the held-out evaluation helper
  (shapes, determinism, training cache untouched): `tests/test_lrtune.py`, CPU only.
- GPU runs through the real sweep machinery, resumable, logs in `tmp/lr_tune*/`; results table under `docs/`;
  short written summary with setup, timings, per task and depth tables, figures and a recommendation:
  `docs/lr_tuning.md`, `docs/lr_tuning_results.csv`, `docs/figures/`.
- `full` preset not changed; recommended change stated.

## Result (2026-10-21)
- Recommendation: one global `lr = 1e-3` at 15000 epochs (the `full` preset already has it; use 3e-3 if the
  epochs are cut to about 6000). No per-task or per-depth values, no schema change.
- At convergence the learning rate only affects speed: lr 3e-4..3e-3 reach the same loss in all 24 tuned
  settings at 15000 epochs; lr >= 1e-2 sends relu L=5 linear teacher to a worse solution; nothing diverged.
- Held-out MSE, the specified primary criterion, is confounded: 15 of the 24 tuned settings converge to the
  all-zero network under `weight_lambda = rate_lambda = 1e-3` (all what-where settings, all linear-teacher and
  most object-action settings with rate 1e-3), so it rewards under-trained runs there. The selection was
  therefore also done on the regularised loss and on the settings that learn (see `docs/lr_tuning.md`).
- Open for the PI: the regulariser scales (separate issue); L = 3 and the other 12 grid settings not run.
- GPUs: T1000 and RTX 4500, pinned by UUID. Nothing submitted to SLURM.
