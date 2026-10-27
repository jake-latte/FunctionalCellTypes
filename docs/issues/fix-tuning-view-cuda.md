---
type: infrastructure
complexity: simple
status: done
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Fix the spatial notebook's tuning cell on CUDA

PI feedback 2026-10-27: the spatial notebook (`notebooks/build_spatial.py`) raises in its last
cell when `DEVICE = "cuda"`. Reproduced on the T1000 with
`python notebooks/build_spatial.py CACHE_ROOT --set "DEVICE='cuda'"`:

    fct/grid_view.py:272, in neuron_mir_values
        return neuron_mir(layer.r, len(layer.group_names)).numpy()
    TypeError: can't convert cuda:0 device type tensor to numpy.

`tuning_view` (fct/grid_view.py) gets rates on the training device because `spatial_report.
tuning_by_position_items` passes the trained net and dataset straight through, whereas the
phase-transition notebook loads its nets on the CPU and never hit this.

Inputs: `MLP`, `Dataset` on any device. Outputs: `TuningView` whose arrays are host arrays.
Interfaces: `fct.grid_view.neuron_mir_values`, `tuning_view`, `tuning_curves`, and anything
else on the spatial notebook's path that calls `.numpy()` on a tensor that may live on a device
(grep `fct/grid_view.py`, `fct/spatial_report.py`, `fct/draw.py` for `.numpy()` without `.cpu()`).
Prefer moving to the CPU once at the entry point (`tuning_view`) over sprinkling `.cpu()`.

## Acceptance criteria
- `python notebooks/build_spatial.py CACHE_ROOT --set "DEVICE='cuda'"` executes every cell (the PM verifies this on the GPU; the worker has no GPU).
- A CPU-only regression test fails before the fix and passes after (for example: `tuning_view` on a net and dataset whose tensors come from a device the test can fake, or a test that the function converts through `.cpu()`; the worker chooses, the reviewer checks it would catch the bug).
- Full suite passes, ruff clean. No change to any computed value on the CPU.
