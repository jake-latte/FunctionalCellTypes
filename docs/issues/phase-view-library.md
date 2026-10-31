---
type: infrastructure
complexity: complex
status: in review
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Library support for viewing a phase_transition sweep: ensembles, position order, a random test set

The `phase_transition` preset (d x lambda_max, phi id, L 1, B 8 ensembles, random embeddings)
has no results reader: `fct.spatial_grid`/`report`/`grid_view` refuse B > 1. The PI asked
(2026-10-31) for a notebook like `spatial_grid.ipynb` adapted to it; this item is the pure,
tested computation layer it needs, in a new module `fct/phase_view.py` (reusing
`fct.spatial_view`, `fct.spatial_grid`, `fct.train.select_member/member_history` where they fit).
The notebook itself is `phase-transition-notebook`.

Needed (each a small pure function unless it is the loader):
1. Loading: one setting of a sweep by axis values -> its config, the trained `EnsembleMLP`,
   history, spaces (`build_spaces` for the net's sizes), the training dataset. Works for a
   partly trained setting (latest checkpoint) and reports an untrained one, as `spatial_grid` does.
2. Unit order by position (PI point 11): for a population and an axis `sort_dim` in 0..d-1, the
   permutation sorting units by their coordinate along that axis (stable; ties by index). Raise
   for sort_dim >= d. Every "sorted by ring position" view uses it.
3. Random test set (PI point 5): a fresh random dataset from the SAME linear teacher as the
   training set (same teacher maps, new samples drawn with a different, recorded seed) with a
   chosen number of points P_test, for the tuning/MIR calculation, so the MIR noise floor can be
   lowered by raising P_test. (The latent-lattice test set stays for the A plot.)
4. Prior matrices (PI points 1, 2; PI chose BOTH): for each weight layer, the covariance
   factors K^pre, K^post with the overall scale absorbed, K / lambda_w, and the penalty factors
   lambda_w M^pre, lambda_w M^post (everything in the loss term (lambda_w / 2 beta) tr(M^post W
   M^pre W^T) except 1 / (2 beta)). Note lambda_w multiplies the Kronecker product once, not each
   factor: document how it is split for plotting (e.g. all of it on the post factor, or sqrt on
   each) and say so in the docstring. Their spectra, descending, truncated to the first n_eig.
5. Ensemble statistics (PI point 7): from the per-member history, mean and standard error of
   the mean over members at every logged epoch, for loss, mse, weight penalty and rate penalty
   (sem = std(ddof=1) / sqrt(B); B = 1 gives sem 0 or NaN - pick and document).
6. Per-member and ensemble-averaged matrices (PI points 6, 9): the data kernel A^(l) on the
   lattice test set, the information and rate overlaps (as the spatial notebook's overlap cell
   computes them, on the random test set), and the trained weight kernels (`empirical_kernels`)
   - each for one chosen member and averaged over all members (members share the embedding, so
   unit indices align; average in unit-index space, then order for display).
7. Example outputs (PI point 8): every member's outputs on the chosen samples, shaped
   (B, n_examples, D_out), for plotting one colour per member.

Interfaces: new `fct/phase_view.py`, tests `tests/test_phase_view.py`, `docs/design.md` (a short
section). No change to training, hashing or the sweep.

## Acceptance criteria
- Position order: on a hand-made 2-D embedding the permutation is right for both axes; ties are
  stable; sort_dim >= d raises.
- Random test set: inputs differ from the training set's, targets equal teacher(inputs) with the
  training set's teacher maps (checked against the training set's own x -> y), P_test points, the
  same seed gives the same set, and the training cache is not modified.
- Prior matrices: (K / lambda_w)^post (x) (K / lambda_w)^pre... i.e. the plotted factors' Kronecker
  product equals the layer covariance K^post (x) K^pre / lambda_w, and the penalty factors' product
  equals lambda_w M^post (x) M^pre, checked on a small layer by forming the Kronecker product;
  spectra descending, truncation exact, n_eig > size handled.
- Ensemble statistics: mean and sem match numpy on a synthetic history with known values.
- Averages: the ensemble average of A, overlaps and weight kernels equals the mean of the
  per-member values computed one member at a time through `member(b)` (independent path); the
  single-member result equals the spatial notebook's computation on `member(b)`.
- A tiny `phase_transition` sweep (reduced sizes, B 2, few epochs, CPU) trained in the test
  suite loads through the loader, and every function above runs on it.
- Full suite green, ruff clean.
