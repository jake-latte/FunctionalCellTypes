---
type: infrastructure
complexity: complex
status: in review
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Scale every population kernel to a chosen top eigenvalue lambda_max

The PI's parametrisation: a connectivity kernel K_ij = lambda_w^{-1} exp(-||x_i - x_j||^2 / (2 l^2))
is specified by its largest eigenvalue `lambda_max` rather than by lambda_w, which it absorbs.

In fct the prior of layer l is vec(W^l) ~ N(0, (K^post (x) K^pre) / weight_lambda) (LOSS_VERSION 4,
the lambda_w / (2 beta) convention) and each population (input, hidden 1..L, output) has its own
kernel K_pop = amplitude * G_pop (G at unit amplitude, the wrapped Gaussian of `fct.spatial`).

DECIDED by the PI (2026-10-30): normalise EVERY population. With `lambda_max` set,

    K_pop = lambda_max * G_pop / lambda_max(G_pop)      for every population,
    weight_lambda = 1.

So each population kernel has top eigenvalue lambda_max (layer l's full prior K^post (x) K^pre
then has top eigenvalue lambda_max^2). The scale is computed per population from its own
eigendecomposition (random embeddings give slightly different G per population and seed).

Plan:
- New field `SpatialConfig.lambda_max: float | None = None`, constrained to (0, inf); omitted from
  the config hash when None so every existing cache keeps its hash (pinned test).
- `fct.spatial._make_population` applies the scale when it is set; `penalty_matrix` and the
  floor (`min_rel_eig`, relative to the top eigenvalue) then act on the scaled K unchanged.
  `Population` records the scale applied (for the notebook and for checking).
- Config validation: with lambda_max set, every population's `kernel.amplitude` must be 1 (it
  would be overwritten, so another value is an error, not ignored) and `train.weight_lambda` must
  be exactly 1 (one source for the overall scale). Both errors name the field.
- A sweep axis `spatial.lambda_max` must work (the sweep's shared `weight_lambda` is then 1).
- The SGLD stability check and the prior penalty need no change (they read the scaled M).

Interfaces: `fct/spatial.py`, `fct/config.py` (validator, hash payload), `fct/sweep.py` only if
the axis needs it, `docs/design.md` (spatial section).

## Acceptance criteria
- For d = 1..4, random and grid embeddings, several length scales, every population built under
  lambda_max has `eigvalsh(K).max() == lambda_max` to 1e-10 relative (computed independently in
  the test, not read back from the recorded scale).
- With lambda_max None, kernels, penalties and every config hash are bitwise those of `main`.
- Amplitude != 1 with lambda_max, or weight_lambda != 1 with lambda_max: each raises naming the
  field.
- A sweep with `grid = {"spatial.d": [1, 2], "spatial.lambda_max": [0.1, 0.9]}` (random
  embeddings) validates, and each setting's saved config.json rebuilds kernels with the right
  top eigenvalue.
- A tiny SGLD net trains under lambda_max end to end.
