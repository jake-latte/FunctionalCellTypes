---
type: data-science
complexity: complex
status: awaiting your confirmation
---

<!-- status: todo | in progress | in review | done | blocked | awaiting your confirmation -->

# Train with the spatial weight prior in place of the weight L2 norm

Depends on `spatial-kernels`, `sgld-training`, `remove-rate-penalty-p-scaling`.

## Proposed plan
- Penalty: `P_weight = 1/2 sum_l tr(M^post_l W_l M^pre_l W_l^T)` for the factorised case (equals `1/2 sum W M W` with `M = M^pre (x) M^post`; no Kronecker matrix is formed). With identity kernels it must equal the current `0.5 * weight_l2` up to the documented constant (current `weight_l2` is the plain sum of squares without 1/2; keep one convention and document it). `compute_loss` takes the penalty from a function chosen by config, so the default path is untouched.
- Hidden-layer weights get `(post, pre)` kernels of their two populations; first-layer weights use the input embedding as pre, last-layer weights use the output embedding as post. Biases (if any) stay L2.
- The rate penalty stays `rate_lambda * sum phi^2` (identity G); the spatial `G` is out of scope for now.
- Works with both optimisers (`adam`, `sgld`); same lambda convention as the L2 case (`weight_lambda` multiplies the prior term).
- The spatial spec enters the train hash only when active, so existing caches stay valid.

## Acceptance criteria
- Identity kernels reproduce the current loss and gradient exactly.
- Matches a brute-force `1/2 w^T (M^pre (x) M^post) w` on tiny layers; gradient check by finite differences.
- Training on a tiny problem lowers the prior term faster along smooth directions: with a wide kernel the trained first-layer weights are smooth over the neuron embedding (measured by a smoothness statistic) compared with identity kernels (ground-truth style test).
- Checkpoint/resume exact; Adam and SGLD both run. Tests CPU only, ruff clean, design doc updated.
