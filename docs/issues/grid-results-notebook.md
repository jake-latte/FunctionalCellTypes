---
type: data-science
complexity: complex
status: in review
---

<!-- status: todo | in progress | in review | done | blocked | awaiting your confirmation -->

# Phase-transition results notebook: one-setting viewer and global modularity plots

Confirmed by the PI 2026-10-22. The helpers (tuning curves, neuron ordering, matrix/ratio plots) are split out as `tuning-and-grid-plot-helpers`; this item wires them to the sweep and builds the notebook.

## Proposed plan
Source notebook built by a script (as `notebooks/build_modularity.py`; executed notebooks are not committed). Cells:
1. **Setting selector.** One cell with `L`, `PHI`, `LAMBDA_R`, `LAMBDA_W` (snapped to the nearest grid value, the snapped value printed). If the setting is trained, it loads the final result; if only partial, the latest checkpoint, labelled provisional (existing `report.py` machinery); if missing, a clear message and no plots.
2. **Existing plot suite** for that one setting: loss curves, example outputs, MIR per layer over training.
3. **Tuning plot.** `N_PLOT_NEURONS`, `PLOT_LAYER`, `NEURON_ORDER in {"random", "mir_ascending", "mir_descending"}` (and a seed for random). One panel per neuron, one curve per latent k: the neuron's mean activation (phi) as a function of that latent's value (binned on rank/value, other latents marginalised), on common axes. Panel titles show neuron index and its MIR.
4. **Global matrix plot.** Select depth and activation; for each hidden layer one heat-map with lambda_r on one axis, lambda_w on the other, colour = average MIR of that layer (final or latest checkpoint; missing cells blank). Shared colour scale.
5. **Global scatter plot.** Same selection; x = lambda_r / lambda_w (log axis), y = layer-average MIR, one dot per network, colour = hidden layer.

Note (2026-10-23): the helper's signature is `mir_grid(records, lambda_r_values, lambda_w_values, n_layers)`;
`n_layers` is required so a layer with no records is a NaN plane, not silently dropped. Choices to confirm: "average MIR" = the existing per-layer mean MIR with the shuffle-bias-corrected variant as default and a toggle for raw; tuning curves use phi (firing rate), not h; lambda ratio uses grid values as is.

## Acceptance criteria
- Pure data-preparation functions (tuning curves, MIR-order selection, grid-to-matrix assembly, ratio table) live in `fct/` with tests on synthetic data with known tuning (e.g. a neuron that is a known function of latent 2 gives a curve with that shape on latent 2 and flat on the others; ordering by MIR is correct; a grid with holes gives NaN cells).
- The notebook builds and executes on a tiny phase-transition sweep (CPU), including a partially trained setting and a missing one.
- Tests CPU only, ruff clean.
