---
type: data-science
complexity: simple
status: done
---

<!-- status: todo | in progress | in review | done | blocked | awaiting your confirmation -->

# Results notebook `modularity.ipynb`

Reads a sweep ROOT. Per task and setting: loss curves (`draw.plot_loss_curves`), example outputs (`draw.fig_example_outputs`), MIR per layer over training (`draw.fig_mir_history`), prediction metrics (`analyse.prediction_report`). Plus a summary table/heatmap of final MIR per layer across the 36 settings. Source kept in the repo (`notebooks/` with a builder); the source notebook (no outputs) is committed, executed notebooks go to the git-ignored tmp/ (PI decision: executed outputs are large and record the ROOT path). Prose is generated from numbers; no logic in cells. Depends on `add-sweep-runner`.

## Acceptance criteria
- Builds and executes from a tiny-preset ROOT (synthetic ground truth: the hand-set modular net recovers high MIR in the summary).
- Handles missing / partial settings with a visible message, not a crash.
- Ruff clean on the builder and source modules.
