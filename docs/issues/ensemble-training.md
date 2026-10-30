---
type: infrastructure
complexity: complex
status: in review
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Train an ensemble of B non-interacting networks in one run

Train B independent copies of the network simultaneously (one optimiser loop, batched over a
leading ensemble dimension) so a setting gives B samples of the trained/SGLD weights at the
cost of one run. PI (2026-10-30): B = 8 for the phase-transition preset; members share the
dataset and differ in initial weights and SGLD noise.

Plan:
- New field `net.B: PosInt = 1`, omitted from the hash at 1 so every existing cache keeps its hash.
- The model stores every parameter with a leading B axis (U (B, N, D_in), J (B, ...),
  V (B, N, D_out)); `forward(x)` returns H (B, P, L, N) and Z (B, P, D_out). `net.member(b)`
  returns a plain single-network `MLP` holding member b's weights, which every existing consumer
  (analyse, monitor internals, spatial_grid/grid_view notebooks, lrtune, meanfield init) uses
  unchanged; consumers that only exist for B = 1 refuse B > 1 loudly rather than reading
  member 0 silently. (Equivalent designs are fine if B = 1 stays bitwise; justify in the report.)
- Member 0 has exactly the initial weights of today's B = 1 net at the same `net.seed`; members
  b > 0 come from distinct, documented generator streams.
- The loss is the SUM over members of each member's loss (the spatial prior applied per member),
  so the gradient of member b is exactly its own; Adam's moments are elementwise and SGLD's
  noise is iid per element, so members do not interact. History records per-member values
  (shape (B,) per log point); the log line prints a summary.
- Checkpoints store the batched state; `load_trained` returns the ensemble; resume works.
- The MIR monitor runs per member (history gains a member axis).

Interfaces: `fct/config.py`, `fct/model.py`, `fct/train.py`, `fct/sgld.py`, `fct/monitor.py`,
`fct/sweep.py` (status, `_check_checkpoint_matches_net`), readers (`spatial_grid.py`,
`grid_view.py`, `report.py`, `lrtune.py`, `analyse.py`, `meanfield.py` net init, `cli.py`) at
least to the point of selecting a member or refusing B > 1; `docs/design.md`.

## Acceptance criteria
- B = 1: bitwise-identical training trajectory, checkpoint contents and config hash to `main`
  (pinned test on a tiny run, Adam and SGLD).
- Non-interaction, Adam: an ensemble of B = 3 trained for k steps equals the 3 single nets
  trained separately from the same initial weights (to float64 round-off).
- Non-interaction, SGLD: with the noise injected from a fixed stream, same check; and members'
  noises are independent (empirical cross-member correlation ~ 0 within its standard error).
- Member 0's initial weights equal the B = 1 net's; members are pairwise distinct.
- Spatial prior: the summed per-member penalty equals B separate penalties.
- Resume mid-run gives the same final ensemble as an uninterrupted run.
- The monitor history has a member axis; readers take a member index or raise for B > 1.
- Full suite green (`uv run pytest -q`), ruff clean.

## Measured cost (2026-10-30, `jobs/bench_ensemble.py`)

The motivation's "at the cost of one run" holds only where it is measured below: an ensemble is
one batched run, not free. Batching wins on the GPU and for small tensors on the CPU; on the
CPU with large activations (large P, or deep nets) B singles in a Python loop are FASTER,
because each elementwise pass over (B, P, N) tensors leaves the cache while a single net's stays
in it (profiled: tanh, mul, tanh_backward and grad accumulation cost 5-17x a single net's for
8x the data). One SGLD step (loss, backward, update), N = 512, D_in = D_out = 64, B = 8, median
of 50 (CPU: 16 torch threads on a shared 32-core host, load ~5; GPU: RTX 4500 Ada, 30 reps).
"loop / ensemble" > 1 means the ensemble is faster.

| device | dtype | case | ensemble ms | loop of 8 ms | loop / ensemble |
|---|---|---|---|---|---|
| cpu | float64 | preset sizes: L 1, id, P 128 | 17.1 | 22.2 | 1.30 |
| cpu | float64 | L 3, tanh, P 256 | 134.6 | 136.7 | 1.02 |
| cpu | float64 | L 1, id, P 2048 | 80.1 | 40.3 | 0.50 |
| cpu | float32 | preset sizes | 4.9 | 10.4 | 2.13 |
| cpu | float32 | L 3, tanh, P 256 | 43.2 | 42.4 | 0.98 |
| cpu | float32 | L 1, id, P 2048 | 31.9 | 17.1 | 0.54 |
| cuda | float64 | preset sizes | 1.32 | 5.89 | 4.46 |
| cuda | float64 | L 3, tanh, P 256 | 17.6 | 23.5 | 1.34 |
| cuda | float64 | L 1, id, P 2048 | 13.6 | 14.7 | 1.08 |
| cuda | float32 | preset sizes | 0.94 | 7.84 | 8.33 |
| cuda | float32 | L 3, tanh, P 256 | 1.76 | 13.4 | 7.59 |
| cuda | float32 | L 1, id, P 2048 | 1.69 | 8.00 | 4.74 |

(The RTX 4500's float64 rate is a small fraction of its float32 rate, so the float64 cases
with large matmuls are compute-bound either way.) At the preset sizes on the CPU, a step is
dominated by drawing the SGLD noise (`normal_`, ~75% of the step for both).

The ensemble forward uses batched matmuls with contiguous results and puts each scale factor
on the smaller of weight and result. Against the first review round's einsum forward, loss +
backward alone (CPU float64, ms; loop = 8 singles):

| case | einsum | matmul (now) | loop |
|---|---|---|---|
| P 128, L 1 | 3.0 | 2.4-3.1 | 8.0 |
| P 256, L 3 tanh | 41-43 | 40-42 | 31 |
| P 2048, L 1 | 81-86 | 47-49 | 22-28 |

Batching therefore still loses to a loop on the CPU for large P or deep nets; chunking members
on the CPU was not attempted. The phase-transition preset (P 128, L 1, float64) is faster
batched on both devices.
