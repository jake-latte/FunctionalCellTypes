---
type: infrastructure
complexity: complex
status: awaiting your confirmation
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Set the weight-prior scale from lambda_max, the top eigenvalue of the hidden kernel

The PI's parametrisation: the hidden-side connectivity kernel is
K_ij = lambda_w^{-1} exp(-||x_i - x_j||^2 / (2 l^2)), and instead of choosing lambda_w we choose
`lambda_max`, the largest eigenvalue of K, so lambda_w is derived:

    lambda_w = lambda_max(G_hidden) / lambda_max,   G_hidden = the unit-amplitude hidden kernel.

In fct the prior of layer l is vec(W^l) ~ N(0, (K^post (x) K^pre) / weight_lambda) (LOSS_VERSION 4,
the lambda_w / (2 beta) convention), so lambda_w is `train.weight_lambda` and G is the population
kernel at amplitude 1.

PROPOSED plan (PI to confirm the reading - see the question in the PM's reply):
- New field `SpatialConfig.lambda_max: float | None = None`, constrained to (0, inf); omitted from
  the hash when None so every existing cache keeps its hash.
- When it is set, `train.weight_lambda` is DERIVED, not chosen: `fct.spatial.derived_weight_lambda`
  (pure; builds the hidden population's G and returns lambda_max(G) / lambda_max). "Applies to all
  K matrices": in a net with several hidden layers every hidden population must give the same
  derived value (true for grid embeddings; random embeddings differ, and the config is rejected
  rather than picking one silently). The single lambda_w scales every layer, input and readout
  factors included, as today.
- Where the derivation happens: in `Config` validation, which writes the derived value into
  `train.weight_lambda` so config.json, the training loss, `prior_coefficient`, the SGLD
  stability check and `fct.meanfield.network_constants` all see one number. A config that sets
  `spatial.lambda_max` AND an explicit `train.weight_lambda` that disagrees is rejected (two
  sources of one number). `amplitude` of the hidden kernel must be 1 when lambda_max is set
  (else G is ambiguous).
- The eigenvalue floor (`min_rel_eig`) is unaffected: it raises small eigenvalues of K only.

Interfaces: `fct/spatial.py`, `fct/config.py` (validator, hash), `fct/sweep.py` (an axis
`spatial.lambda_max` must work: the derived weight_lambda is per setting and must not be
overridden by the sweep's shared `weight_lambda`), `docs/design.md`.

## Acceptance criteria
- For a grid embedding the derived lambda_w gives a hidden kernel K = G / lambda_w whose largest
  eigenvalue (by `torch.linalg.eigvalsh`) is lambda_max to 1e-10 relative, for d = 1..4 and
  several length scales.
- A config with lambda_max set writes the derived weight_lambda into the saved config.json, and
  `prior_coefficient` uses it; training a tiny net under it runs.
- A sweep with `grid = {"spatial.d": [...], "spatial.lambda_max": [...]}` gives every setting
  its own derived weight_lambda (checked per setting, not one shared value).
- lambda_max None: every existing config hash is unchanged (test against a pinned hash).
- Explicit disagreeing weight_lambda, non-unit hidden amplitude, or hidden populations whose
  derived values differ: each raises naming the field.
