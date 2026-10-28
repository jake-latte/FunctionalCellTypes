---
type: infrastructure
complexity: complex
status: in progress
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Grid search over spatially embedded networks (`SweepConfig.spatial`, presets `spatial_tiny` and `spatial_n_scan`)

Generalise the existing sweep so a grid can run spatially embedded networks: a sweep has an
optional `spatial` section (the SPATIAL_PRIOR flag of the brief is "the section is set");
without it every setting trains under the iid prior K = M = I (the plain weight L2, as now),
with it every setting builds its populations' embeddings, Gaussian kernels K and inverses M
from `fct.spatial` and trains under `tr(M_post W M_pre W^T)` per layer (already implemented
in `fct.train.make_weight_penalty`; training does not know which prior it is given). Grid
axes may name any field of the spatial section (`spatial.hidden.kernel.length_scale`,
`spatial.hidden.embedding.kind`, `spatial.min_rel_eig`, `spatial.d`, ...) as well as the
existing `net.*` / `train.*` axes.

## Proposed plan
1. `SweepConfig.spatial: SpatialConfig | None = None`. `_base_config` passes it to `Config`;
   `_set_leaf` already walks the dumped tree, so `spatial.<path>` axes work once the base has
   the section (an axis under `spatial` with `spatial=None` is rejected at load time, naming
   the axis: "set the sweep's spatial section first"). `grid_hash` is unchanged (axes only),
   so existing sweep.json files keep their hashes and directory names. Config hashes of
   non-spatial settings do not move (the running SLURM sweep imports `fct` from this tree).
2. Presets. `spatial_tiny` (seconds on a CPU: N in [4, 8], P 16, m 6, k 2, L 1, SGLD, a few
   epochs, D 1, grid embeddings, length scale 0.15, min_rel_eig 1e-2) and `spatial_n_scan`:
   the PI's current notebook settings as the shared base, scanning N:
   linear teacher P 164, m 64, k 2, L 1, phi tanh, g 1, weight_lambda 3e-3, rate_lambda 0,
   SGLD beta 1e4, lr 1e-2, 250 000 epochs, train/net/data seeds 0, float64? (the notebook
   trains float64 on CUDA; the sweeps use float32 -- I propose float64 for these small nets),
   spatial d 1, box 1, min_rel_eig 1e-2, grid embeddings, length scale 0.15, amplitude 1 for
   input, hidden and output; grid `net.N` = [64, 128, 256, 512, 1024]. Monitor: the sweep
   always runs the MIR monitor; `monitor_int` 5000, bins 8, shuffles 5 (cheap at P 164).
   Both are plain presets (no `--n-lambda`); `--axis` adds or replaces axes as now, e.g.
   `--axis spatial.hidden.kernel.length_scale=[0.05,0.15,0.3]`.
3. The mean-field stage: `run_meanfield` raises on a spatial config (separate item
   `meanfield-rejects-spatial`); `sweep run --stage solve` reports it per setting.
4. `jobs/fct_sweep.sh` needs no change (it reads `sweep count`); `README.md` and
   `docs/design.md` gain the spatial sweep.
5. Not in this item: the loss convention. The brief writes the prior term as
   `lambda_w / (2 beta) sum W M W`; the library has `lambda_w tr(M_post W M_pre W^T)` (stationary
   law under SGLD: vec(W) ~ N(0, (2 beta lambda_w)^-1 K)). Changing it moves every training
   hash (LOSS_VERSION) and the mean-field constants, and would hit the running sweep. Decide
   separately (see the PM's question).

## Inputs / outputs / interfaces
- `fct/sweep.py`: the field, the presets (`PRESET_NAMES` grows), validation of spatial axes.
- `fct/cli.py`: `--preset spatial_tiny|spatial_n_scan` (no new flags).
- Setting directories: `ROOT/linear_teacher/s0000_N64/` etc., as any grid.

## Acceptance criteria
- A sweep without `spatial` builds exactly the configs it built before (hash equality on the
  existing presets, checked by a test against pinned hashes or by equality with the previous
  `SweepConfig` dump).
- `spatial_n_scan` expands to 5 settings whose configs carry the spatial section, N in the
  listed order, directory names `s000i_N<N>`; `spatial_tiny` runs end to end on a CPU through
  `python -m fct sweep init/run/status` (done afterwards), resume included.
- An axis `spatial.hidden.kernel.length_scale=[0.1, 0.2]` on a spatial sweep gives settings
  whose `cfg.spatial.hidden.kernel.length_scale` differ, with distinct training hashes; the
  same axis on a non-spatial sweep fails at load time naming the axis.
- `sweep run --stage solve` on a spatial setting fails with the solver's one-line message.
- Tests CPU only; ruff clean; README and design doc updated.

## PI confirmation (2026-10-28)
Confirmed. The prior-term convention is decided: it is ALWAYS `lambda_w / (2 beta)` times the prior
quadratic form, with `beta = 1` under Adam. That change is its own item, `prior-scaling-convention`,
which lands after `readout-post-activations`; this item does not touch the loss. Because
`LOSS_VERSION` / `MODEL_VERSION` bumps are landing in parallel, the "non-spatial sweeps build the
same configs as before" criterion is tested as equality of the expanded `Config` objects against the
previous `SweepConfig` expansion (or a dump made at the branch base), NOT against pinned hash strings.
The SLURM hazard is gone (the PI cancelled the arrays), so this branch may change anything it needs.
