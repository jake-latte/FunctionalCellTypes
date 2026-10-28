---
type: data-science
complexity: simple
status: done
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Spatial notebook: the mutual-information overlap matrix C of each hidden layer

PI request 2026-10-27. For each hidden layer $l$, with $I^{(l)}_{if}$ the mutual information between
neuron $i$ of layer $l$ and latent factor $f$ ($K$ factors), plot
$C^{(l)}_{ij} = \frac{1}{K} \sum_{f=1}^{K} I^{(l)}_{if} I^{(l)}_{jf}$, one $N \times N$ matrix per layer.

Blocked on `spatial-notebook-inline-plots-and-equations`: the plot is written in the notebook in
the same inline style, after that item lands.

Inputs: `fct.modularity.neuron_factor_information(rates, factors, ...).member_mi` is $I^{(l)}$
(N, K) (for the linear teacher every group is one latent, so `member_mi` = `group_mi`); rates from
`fct.grid_view.layer_rates`. Output: a small pure function in `fct/modularity.py`,
`information_overlap(member_mi) -> (N, N)` = `member_mi @ member_mi.T / K`, and one notebook cell
with its plotting function (one `imshow` per layer, neurons in ring order for d = 1 as in the
weight image, a shared colour scale from 0, colourbar labelled nats^2) and a markdown cell with
the equation above. Use the same `N_BINS`, `MIR_KIND` (raw or shuffle-corrected $I$) and
`N_SHUFFLES` as the tuning cell and say in the prose which $I$ is used.

## Acceptance criteria
- Synthetic-data-first: rates built from known latents (e.g. neurons 0..a copy latent 1, a..b copy latent 2, the rest noise) give a $C$ that is block-diagonal with the blocks on the copying neurons and ~0 elsewhere; the test asserts that, plus symmetry and $C_{ii} = \frac{1}{K}\sum_f I_{if}^2$.
- `information_overlap` rejects a non-2-D input with a ValueError.
- The notebook builds and executes on the tiny configuration on CPU; the new figure appears once per hidden layer.
- Full suite passes, ruff clean.
