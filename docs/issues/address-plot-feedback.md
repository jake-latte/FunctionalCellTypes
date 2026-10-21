---
type: data-science
complexity: simple
status: in review
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Address the PI's plot feedback

The PI's review of the results notebook:

1. plots are too small (font size and width);
2. the notebook does not explain "corrected" MIR, "raw" MIR or the "chance baseline";
3. linear_teacher example outputs split true and predicted over two axes; they belong on one.

Why: the notebook is read at full width, and a curve is only interpretable if the reader knows
what each metric is.

## Changes

- `fct/draw.py`: one place for sizes (`FIG_WIDTH`, `STYLE_RC`, `LINE_WIDTH`, `ANNOTATION_SIZE`);
  the notebook figures are built under `STYLE_RC` (`_styled`); new `fig_loss_curves`; loss legend
  outside the axis; `fig_example_outputs` for linear_teacher draws true (ring) and predicted
  (diamond) on one axis per example. Existing signatures are unchanged.
- `fct/explanations.py`: `MIR_EXPLANATION`, the markdown (with LaTeX) defining the metrics from
  `fct/modularity.py`, with its numbers taken from the code (`DEAD_TOL`, `NOISE_FACTOR`, config
  defaults) and tested against it. `fct/report.py`: `setting_items` uses `fig_loss_curves`.
- `notebooks/build_modularity.py`: the explanation cell after the setup cell, `figure.dpi = 100`.
- `docs/design.md` updated.

## Acceptance criteria

- Tests: full-width figures with tick labels >= 11 pt, legends outside the axes, linear_teacher
  has one axis per example holding both series, the explanation's numbers match the code (floor
  examples recomputed) and its equations are balanced, its position in the notebook.
- Figures from the executed tiny-sweep notebook were rendered to PNG and inspected: heatmaps,
  loss curves, MIR history (1 and 5 layers) and the example figures of all three tasks. Checked:
  text legible at full width, no clipping or overlap, legends clear of the data, and the
  linear_teacher true/predicted overlay readable. The scatter legend moved below its axis because
  its R² label ran over the points.
