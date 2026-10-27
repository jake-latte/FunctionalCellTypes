---
type: infrastructure
complexity: simple
status: in progress
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Show each neuron's mutual information with every latent next to its tuning curves

**Why.** On the PI's run (`tmp/spatial_cache/spatial_0624b6833433`: P = 64 samples, k = 4 latents,
N = 256, `N_BINS = 8`, `MIR_KIND = "corrected"`) every neuron's title reads "MIR 0.00" although the
tuning curves are visibly selective. Reproduced in `tmp/pm/mi_repro.py`: the raw plug-in
informations are fine (one neuron carries 0.85 nats about z2 against a null of 0.05), but at
P = 64 and B = 8 the plug-in bias is about (B-1)^2 / (2P) = 0.38 nats per latent, so the shuffled
null total (1.9 nats) is more than half the raw total (2.4 nats) and the row rule in
`fct/modularity.py: _subtract_null` ("raw total <= NOISE_FACTOR x null total means noise") zeroes
every neuron: 256/256 dead, all I_if = 0, MIR 0. With B = 4 the corrected MIR is 0.64. Changing the
rule is a separate item (`mi-noise-rule-per-latent`, awaiting the PI); this item makes the
informations visible so the reader can see what the MIR is computed from.

**What.** In `notebooks/build_spatial.py`, the `TUNING` cell (and `TUNING_TEXT`): each neuron gets
two panels side by side instead of one: left, the tuning curves exactly as now; right, a column
(bar) chart of the neuron's mutual information with each latent f: one bar per latent at x = f,
coloured `C{f}` like the curve of that latent. The bar is the raw plug-in I_if (nats, light, e.g.
alpha 0.35) with the `MIR_KIND` estimate (the I_if the MIR uses) overlaid as a full-colour bar at
the same x (for `"raw"` they coincide), and a black horizontal marker at the shuffled-null mean of
that latent when the null was computed. All bar panels share one y axis (nats); x ticks are the
latent names. The title keeps the neuron index, position and MIR. The layout stays by position:
`panel_grid` gives the neuron's cell, which now holds a pair of axes (a 1 x 2 sub-grid or two
columns per cell; keep `N_COLS` meaning neurons per row).

Before the figures, print one line per layer that lets the reader judge B against P:
`P`, `B`, the expected plug-in bias (B-1)^2 / (2P) nats per latent, and the mean over neurons and
latents of the shuffled null (when computed). Update `TUNING_TEXT` with the bar description, the
bias estimate, and a sentence saying that when the null is close to the raw bars B is too large
for P.

**Library.** `fct.modularity.FactorInformation` gains `raw_mi` (N, n_members; the plain plug-in
estimate, equal to `member_mi` for `bias_correction="none"`) and `null_mi` (N, n_members; the mean
over the shuffles, `None` for `"none"`). `fct.grid_view.TuningView` gains the same two fields
(CPU tensors), filled by `tuning_view` from `neuron_information`, so the notebook cell reuses
`tuning_view` and the bars are the same numbers as the MIR by construction. Update the signature
line in `docs/design.md` (search `FactorInformation(`). Grep for positional constructions of
`FactorInformation(` and `TuningView(` and fix them.

Only the `TUNING_TEXT`/`TUNING` cells of the notebook change; the `OVERLAP` cells are another
item in flight (`overlap-of-informations-and-of-rates`); do not edit them. The figure count per
layer stays one, so `N_FIGURES` in `tests/test_build_spatial.py` is unchanged; `PLOTTING_FUNCTIONS`
changes if you rename or add a plotting function.

## Inputs / outputs / interfaces
- In: `TuningView` from `tuning_view(RUN.net, RUN.dataset, layer, N_BINS, MIR_KIND, N_SHUFFLES, TRAIN_SEED)`.
- Out: the per-layer figure, now with 2 axes per neuron; the printed bias line.
- Interfaces: `FactorInformation.raw_mi`, `.null_mi`; `TuningView.raw_mi`, `.null_mi`.

## Acceptance criteria
- `tests/test_modularity.py`: `raw_mi` equals `member_mi` under `"none"` and equals the `"none"`
  `member_mi` under `"shuffle"` (same rates, same factors); `null_mi` is `None` under `"none"`,
  has shape (N, n_members) and is non-negative under `"shuffle"`, and for rates independent of the
  factors (seeded noise, P = 4096, B = 10) `raw_mi - null_mi` is within 0.02 nats of 0 on average.
- `tests/test_grid_view.py` (or where `tuning_view` is tested): `TuningView.raw_mi`/`null_mi` are
  CPU tensors with the right shapes, also for the fake-device tensors (`tests/draw_helpers.py`).
- `tests/test_build_spatial.py`: the tuning figure has exactly 2 visible axes per neuron; the bar
  axes share a y axis; the committed `notebooks/spatial.ipynb` is regenerated (`--no-execute`) and
  the committed-notebook test passes.
- The notebook builds on CPU at the defaults with no error (`uv run python notebooks/build_spatial.py tmp/<yours>`).
- Lint and the full suite pass.
