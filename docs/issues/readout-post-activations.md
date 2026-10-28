---
type: infrastructure
complexity: complex
status: in progress
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# The readout is always a linear map of the post-activations (remove the "h" case)

The PI: "the readout of the network is always a linear mapping of the post-activations,
y_hat = (1/N) V phi(h^L)". Today `fct.model.MLP.forward` reads the pre-activation
(`Z = V^T h^L / N`, as the original playground did), and `MeanFieldConfig.readout` selects
"h" (default) or "phi" for the solver and the kernel plots. This item makes phi(h^L) the only
readout: in the model, the solver and every notebook.

## Proposed plan
1. `MLP.forward`: `Z = V^T phi(h^L) / N`. `MeanFieldConfig.readout` is removed (a config.json
   that still carries it fails to load, naming the field); `readout_activities` loses its
   argument and returns phi(h^l) for every layer; `empirical_kernels`, `net_kernels`, the
   solver's learning term and `fct.exemplar` follow.
2. Every cached training run and mean-field result was made with the h readout. Add
   `MODEL_VERSION = 2` folded into the net section hash (as `LOSS_VERSION` is into train), so
   old artefacts are rejected as stale instead of being read as if they came from this model.
   The pinned fixtures of `tests/test_meanfield_pins.py` / `pinned_runs.py` must be regenerated
   and the HMC/solver regression values re-derived from the notebook convention.
3. The notebooks (`spatial`, `exemplar`, `phase_transition`, `modularity`) lose the readout
   cases in their equations and titles.

## Why this needs your confirmation
- A SLURM array (job 3847945, root `pt_sweep_big`) is running from this tree's `.venv`: array
  tasks that start after the merge would train the new model under the old hashes, and the
  grid would silently mix two models. This item must not be merged while that array runs,
  or the running sweep must be done in a separate checkout. Say which.
- It invalidates `tmp/spatial_cache` and every cached sweep (they become stale, loudly).
- Alternative, if you meant only the figures: keep the model, and drop the "h" case from the
  spatial notebooks only (that part is in `spatial-grid-notebook`). Say if this is what you want.

## Acceptance criteria
- `MLP.forward` output equals `phi(H[:, -1]) @ V / N`; the solver's learning term and the
  net's empirical kernels use phi(h^L); no `readout` field remains.
- Old artefacts raise StaleCacheError; new ones load.
- Tests CPU only; ruff clean; design doc updated.

## PI confirmation (2026-10-28)
Confirmed as planned (the model, the solver and every notebook read out phi(h^L)). The PI cancelled
the `pt_sweep_big` arrays because this change invalidates them, so the hazard above is gone: merge
as soon as reviewed. The notebook part of the "figures only" alternative is moot.
