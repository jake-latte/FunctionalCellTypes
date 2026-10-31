---
type: infrastructure
complexity: simple
status: done
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# The phase_transition_grid notebook: spatial_grid's figures adapted to d x lambda_max ensembles

Depends on `phase-view-library`. A new builder `notebooks/build_phase_transition_grid.py` writing
`notebooks/phase_transition_grid.ipynb` (PI chose the name 2026-10-31; the existing
`phase_transition.ipynb`, the lambda_grid viewer, is untouched). Same pattern as
`build_spatial_grid.py`: import cells from `build_spatial.py` where unchanged, a parameters cell,
the grid table, a loader, cells guarded for untrained settings, `--no-execute` for the committed
source notebook, executed copies to tmp/.

Parameters at the top: ROOT, SETTING (axis values), MEMBER (the ensemble member for single-network
plots), SORT_DIM (PI point 11: units of every population ordered by their coordinate along this
axis; every view that sorted by ring position uses it), P_TEST (size of the random test set for
tuning/MIR), N_EIG (spectrum truncation), N_NEURONS (tuning truncation), DEVICE.

Changes from spatial_grid.ipynb (PI's list, 2026-10-31):
1. Prior plots show, side by side (PI chose both), the covariance with the overall scale
   absorbed, K / lambda_w, and the penalty lambda_w M - all scaling except 1 / (2 beta).
2. Prior plots also show the spectra of K^pre and K^post (and of the penalty factors), truncated
   to the first N_EIG eigenvalues.
3. No U weight plot (the "First-layer weights U, sorted by position" section is dropped).
4. No "trials whose kernels are drawn" and no "Phi^(l) between pairs of trials" sections.
5. Tuning / MIR use a fresh random test set of P_TEST points from the training teacher (not the
   training set). The A plot keeps the latent-lattice test set.
6. A plots and the information/rate overlap plots each come twice: for member MEMBER as now, and
   the ensemble average.
7. Loss curves: mean over members with a +/- sem band, for loss, mse, weight penalty, rate penalty.
8. Example outputs: every member's output on the same axes, one palette colour per member (as many
   members as the palette has colours; say so in the cell).
9. Trained weight kernels: for member MEMBER and the ensemble average.
10. Tuning / MIR plot truncated to the first N_NEURONS neurons (in SORT_DIM order).
11. Every position-sorted view uses the SORT_DIM order; the text says that with random embeddings
    and d > 1 this is one coordinate, not a ring position, and that general distance measures
    come later.

## Acceptance criteria
- `python notebooks/build_phase_transition_grid.py --no-execute` writes the source notebook; a test
  checks the committed notebook equals the builder's output (as for spatial_grid).
- The notebook executes end to end (test) on a tiny trained `phase_transition` sweep (reduced
  sizes, B 2, d 2) and on a sweep with one untrained setting (only the prior drawn).
- Each of the 11 changes is visible: a test per change on the built cells or executed outputs
  where practical (e.g. no U / trials / pairs sections; SORT_DIM out of range raises in the
  notebook; N_EIG and N_NEURONS truncate).
- README notebook table lists it; the `phase_transition` preset paragraph points at it.
- No existing .ipynb changes.

## Leftovers after review (not blocking)
- Two seeded mutations the tests miss: the tuning shuffle seed (the wiring session uses the raw
  MIR, which has no shuffles), and an overlap colour limit taken from one view only (the test
  checks the limits are equal, not that they cover both maxima).
- Cosmetic: spectra legends overlap a few points near j = 8-10; the example-output legend
  touches a target marker; a missing comma in the README notebook paragraph.
- The member's tuning view is recomputed in the notebook (MemberView keeps only overlaps).
- Open for the PI: default NEURON_PICK ("first", the literal request, or "spread").
