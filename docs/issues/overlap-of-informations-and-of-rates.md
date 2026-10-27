---
type: infrastructure
complexity: simple
status: in review
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Show the overlap of the neurons' mean rates next to the overlap of their informations

**Why.** The PI wants the information overlap C of each hidden layer compared with the same
construction on the tuning curves, which does not depend on the mutual-information estimate (on
the PI's run that estimate is zeroed for every neuron, see `tuning-panels-with-information-bars`).

**What.** In `notebooks/build_spatial.py`, the `OVERLAP` cell (and `OVERLAP_TEXT`): each hidden
layer's figure gets two panels side by side. Left, the information overlap exactly as now,
C^I_ij = (1/k) sum_f I_if I_jf from `view.member_mi`. Right, the rate overlap built from the bin
means of the tuning curves, `view.curves.mean` (N, k, B) with entries rbar_if(b):
C^r_ij = (1 / n_valid) sum_{f,b} rbar_if(b) rbar_jf(b), the sum and n_valid over the (f, b) with a
non-empty bin (NaN bins excluded; with equal-count rank bins all are valid and n_valid = kB).
Both panels show the neurons in the same order (ring order for d = 1, index order otherwise) with
neuron indices as tick labels, as now. C^I keeps its colour scale (viridis from 0, one vmax over
layers, the "(no information)" rule). C^r is signed for tanh rates, so use a diverging map
(`RdBu_r`) symmetric about 0 with one bound over layers; its colour bar is labelled in rate^2.
Update `OVERLAP_TEXT` with the second equation and what to compare: blocks that appear in C^r but
not in C^I are neurons with similar tuning whose information the estimate did not resolve.

**Library.** Add the rate overlap as a pure function next to `information_overlap`:
`fct.modularity.rate_overlap(mean: Tensor) -> Tensor` taking the (N, k, B) bin means (NaN allowed)
and returning (N, N); ValueError on a non-3-D input. Both overlaps are the same inner product over
a flattened feature axis, so share the implementation if that is cleaner (the information one is
divided by k, the rate one by the number of valid (f, b)).

Only the `OVERLAP_TEXT`/`OVERLAP` cells of the notebook change; the `TUNING` cells are another item
in flight (`tuning-panels-with-information-bars`); do not edit them. One figure per layer still, so
`N_FIGURES` in `tests/test_build_spatial.py` is unchanged; `PLOTTING_FUNCTIONS` changes if you
rename or add a plotting function.

## Inputs / outputs / interfaces
- In: `TuningView.member_mi` (N, k) and `TuningView.curves.mean` (N, k, B).
- Out: the per-layer figure with two panels; `fct.modularity.rate_overlap`.

## Acceptance criteria
- `tests/test_modularity.py`: synthetic (N = 3, k = 2, B = 4) bin means with one NaN bin: the
  result equals a hand-computed matrix; a curve copied between two neurons gives
  C^r_ij = C^r_ii; the ValueError on wrong dimensionality.
- `tests/test_build_spatial.py`: the overlap figure has two axes with images; the committed
  `notebooks/spatial.ipynb` is regenerated (`--no-execute`) and the committed-notebook test passes.
- The notebook builds on CPU at the defaults with no error.
- Lint and the full suite pass.
