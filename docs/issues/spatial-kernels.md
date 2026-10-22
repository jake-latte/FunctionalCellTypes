---
type: infrastructure
complexity: complex
status: in review
---

<!-- status: todo | in progress | in review | done | blocked | awaiting your confirmation -->

# Spatial embeddings and factorised periodic Gaussian weight kernels

Spatial Variation, step 1 (`Latex/sections/01_setup.tex`, "Network"): every neuron, input and output index gets a position `x in R^d`; the weight prior of layer l is `K^l = K^post (x) K^pre`. Start with Gaussian kernels with periodic boundaries.

## Proposed plan
- `fct/spatial.py`:
  - `Embedding`: positions for a population of `n` units on a d-dimensional torus `[0, box)^d`, from `"grid"` (regular lattice; n must factor into the lattice shape, shape given explicitly) or `"random"` (uniform, seeded). Returns an `(n, d)` tensor.
  - `periodic_gaussian_kernel(x_post, x_pre-or-same, length_scale, box, amplitude)`: `K_ij = amplitude * exp(-d_torus(x_i, x_j)^2 / (2 ell^2))`, with torus distance the per-axis minimum-image distance.
  - `LayerSpace` / `SpatialSpec`: per layer l (0..L) a pre embedding and kernel and a post embedding and kernel (post of layer l and pre of layer l+1 are the same population and share an embedding). Defaults: inputs and outputs get their own embeddings (specified in the notebook).
- A pydantic `SpatialConfig` (in `fct/spatial.py`, strict like the other config models) describing embeddings and kernel parameters per population. It is NOT wired into `Config` here: that happens in `spatial-weight-prior-training`, where it first affects a result (and `fct/config.py` is being edited by another item right now).
- `draw.py`: `fig_kernel` plots a kernel as a matrix and as a function of distance, plus the embedding positions; used by the notebook's kernel cell.
- A numerically safe way to get the penalty matrices `M = K^{-1}`: Gaussian kernels are ill-conditioned, so `M` is computed through the eigendecomposition with a relative eigenvalue floor (`min_rel_eig`, default 1e-6) and the floor is part of the config. This floor is the only approximation and is reported.

## PI decisions (2026-10-22)
- Plan confirmed. Start with `d = 1`: every population lives on a ring (the 1-torus `[0, 1)`); keep `d` a parameter so `d = 2` is a later config change, not a code change.
- Inputs are spaced evenly on the 1-D ring (the `grid` embedding); outputs likewise by default. Hidden populations default to `grid` too; `random` stays available.

## Acceptance criteria
- Torus distance wraps (points 0.05 and 0.95 are 0.1 apart); kernel is symmetric, unit diagonal times amplitude, positive semi-definite; grid kernel is circulant for a 1-D ring; the evenly spaced input embedding has equal neighbour spacing 1/m.
- `M K ~ I` on the retained spectrum within tolerance; eigenvalue floor reported.
- Kernel figure renders from a synthetic spec in a test.
- Tests CPU only, ruff clean.
