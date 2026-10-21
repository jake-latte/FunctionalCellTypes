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
- `fct/report.py`: `MIR_EXPLANATION`, the markdown (with LaTeX) defining the metrics from
  `fct/modularity.py`; `setting_items` uses `fig_loss_curves`.
- `notebooks/build_modularity.py`: the explanation cell after the setup cell, `figure.dpi = 100`.
- `docs/design.md` updated.

## Acceptance criteria

- Tests: style constants, full-width figures with tick labels >= 11 pt, linear_teacher has one
  axis per example holding both series, the explanation's key terms and balanced equations, its
  position in the notebook.
- Figures from a tiny sweep rendered and inspected (see the report of the worker).
