---
type: data-science
complexity: simple
status: in review
---

<!-- status: todo | in progress | in review | done | blocked | awaiting your confirmation -->

# Run the tiny preset end to end, then stop for review

Run the `tiny` preset for all 36 settings locally (CPU, plus one tiny GPU smoke run pinned by UUID, T1000 preferred), execute the notebook on it into the git-ignored tmp/ (do not commit the executed notebook; only the source notebook is committed). Depends on the three items above.

## Acceptance criteria
- All 36 settings `done` in `status`; notebook executed without errors.
- The PI reviews the code and small results and submits the full jobs. No SLURM submission by the agent.

## Result (2026-10-21, rerun under the final loss)
The first run (before the loss change) is obsolete: its caches are reported `stale`. Rerun with the loss
as merged (rate penalty averaged over P, `weight_lambda = 1e-4`, `LOSS_VERSION = 2`):
- `python -m fct sweep run tmp/tiny_sweep_v2` (tiny preset, CPU): 36 of 36 settings done in 8 s.
- GPU smoke (T1000, pinned by UUID, `device=cuda`, `tmp/tiny_gpu_v2`): settings 0, 17, 35 done.
- `python notebooks/build_modularity.py tmp/tiny_sweep_v2` executed without errors in 44 s into the
  git-ignored `tmp/modularity.ipynb` (14 MB). The tiny preset learns nothing meaningful; this is a
  pipeline check only.
- STOP: the PI reviews the code and these small results, sets the `full` preset sizes, and submits the
  cluster jobs. Nothing was submitted.
