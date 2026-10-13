---
type: infrastructure
complexity: simple
status: done
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Fix the neuron-map title overlap and the kernel figure's flat net panels

Two layout defects seen when the CLI rendered real-size figures:
`fig_neuron_maps` draws its suptitle over the first row of panels at the
default size (N=100, 8 columns), and `fig_kernels` shares one colour scale
per row between theory and net, so when the two differ by orders of
magnitude the net panel renders as a flat square.

Why: both figures are what the PI reads to judge a run; an unreadable
title or a blank panel defeats the purpose.

## Inputs / outputs / interfaces

`fct/draw.py` only (`fig_neuron_maps`, `fig_kernels`, `plot_kernel`); no
signature changes except an optional keyword on `fig_kernels`:
`shared_scale: bool = True`. When `shared_scale` is False each panel gets
its own colour scale and colorbar. The CLI keeps the default. Every
`plot_*`/`fig_*` contract in `docs/design.md` stays as is.

## Acceptance criteria

- `fig_neuron_maps` with N=100, n_cols=8: the suptitle's bounding box does
  not intersect any axes' bounding box (check with
  `fig.canvas.draw(); title.get_window_extent()` vs `ax.get_window_extent()`
  on the Agg backend). Use `fig.tight_layout(rect=(0, 0, 1, 0.97))` or
  `constrained_layout`; keep the per-panel colorbars.
- `fig_kernels(Phi_theory, Phi_net, shared_scale=True)` keeps today's
  behaviour (tested: both panels of a row have the same `clim`);
  `shared_scale=False` gives each panel `clim == (min, max)` of its own
  matrix and its own colorbar; every panel title names the scale used (the
  default figure's panel titles gain the "(shared scale)" suffix).
- Existing `tests/test_draw.py` pass unchanged; new tests for the two
  criteria above.

## Notes
- The overlap depends on the number of rows (figure height), not N: the 24 pt
  title is fixed in points while tight_layout's pad is a fraction of the
  figure. The test uses 13 rows (reproduces the overlap) and 1 row (short path).
- `docs/design.md` draw.py line for `fig_kernels` mentions `shared_scale`.
