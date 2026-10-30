---
type: infrastructure
complexity: complex
status: awaiting your confirmation
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Train an ensemble of B non-interacting networks in one run

Train B independent copies of the network simultaneously (one optimiser loop, batched over a
leading ensemble dimension) so a setting gives B samples of the trained/SGLD weights at the
cost of one run.

PROPOSED plan (PI to confirm):
- New field `net.B: PosInt = 1`, omitted from the hash at 1 so every existing cache keeps its hash.
- The model stores every parameter with a leading B axis (U (B, N, D_in), J (B, ...),
  V (B, N, D_out)); `forward(x)` returns H (B, P, L, N) and Z (B, P, D_out). `net.member(b)`
  returns a plain single-network `MLP` holding member b's weights, which every existing consumer
  (analyse, monitor internals, spatial_grid/grid_view notebooks, lrtune, meanfield init) uses
  unchanged; consumers that only exist for B = 1 refuse B > 1 loudly rather than reading
  member 0 silently.
- Members share the dataset (same x, y) and differ in their initial weights and, under SGLD, in
  their noise. Member 0 has exactly the initial weights of today's B = 1 net at the same
  `net.seed`; members b > 0 come from distinct, documented generator streams.
- The loss is the SUM over members of each member's loss (the spatial prior applied per member),
  so the gradient of member b is exactly its own; Adam's moments are elementwise and SGLD's
  noise is iid per element, so members do not interact. History records per-member values
  (shape (B,) per log point) plus whatever summary the logs print.
- Checkpoints store the batched state; `load_trained` returns the ensemble.
- The MIR monitor runs per member (history gains a member axis).

Interfaces: `fct/config.py`, `fct/model.py`, `fct/train.py`, `fct/monitor.py`, `fct/sweep.py`
(status/`_check_checkpoint_matches_net`), readers (`spatial_grid.py`, `grid_view.py`,
`report.py`, `lrtune.py`, `cli.py`) at least to the point of refusing or selecting a member;
`docs/design.md`.

## Acceptance criteria
- B = 1: bitwise-identical training trajectory, checkpoint contents and config hash to `main`
  (pinned test on a tiny run).
- Non-interaction, Adam: an ensemble of B = 3 trained for k steps equals the 3 single nets
  trained separately from the same initial weights (to float64 round-off).
- Non-interaction, SGLD: with the noise switched to a fixed stream, same check; and members'
  noises are independent (empirical cross-member correlation of the noise ~ 0).
- Member 0's initial weights equal the B = 1 net's; members are pairwise distinct.
- Spatial prior: the per-member penalty summed equals B separate penalties.
- Resume from a checkpoint mid-run gives the same final ensemble as an uninterrupted run.
- The monitor history has a member axis; readers either take a member index or raise for B > 1.
