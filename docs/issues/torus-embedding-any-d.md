---
type: infrastructure
complexity: simple
status: awaiting your confirmation
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Place grid embeddings on the d-torus without an explicit shape, so spatial.d can be a grid axis

`fct.spatial` already places units in [0, box)^d and the wrapped Gaussian kernel factorises
over axes for any d. What stops a `spatial.d` grid axis is the embedding: a `grid` embedding in
d > 1 needs an explicit `shape` with d entries, and `SpatialConfig` rejects a shape whose length
differs from d. So one sweep.json cannot hold d = 1..4 for the same populations.

Change: `EmbeddingSpec.shape = None` in any d means "the default lattice for n units in d
dimensions", built by a pure function `default_grid_shape(n, d) -> list[int]` (PROPOSED, PI to
confirm: the most balanced factorisation, i.e. the d integer factors of n whose max/min ratio is
smallest, sorted ascending; exact n**(1/d) when n is a perfect d-th power). Explicit shapes keep
working unchanged. The `_grid_shapes_match_d` validator no longer requires a shape in d > 1.

Interfaces: `fct/spatial.py` (`_grid_shape`, `SpatialConfig` validator, module docstring);
`docs/design.md` spatial section. Hashes: a config that set shape explicitly is unchanged; d = 1
with shape None gives the same positions as today, so no existing hash or cache moves.

## Acceptance criteria
- `default_grid_shape(n, 1) == [n]`; perfect powers give the cube root etc. (512, 3 -> [8, 8, 8];
  4096, 4 -> [8, 8, 8, 8]); a non-power gives the most balanced factorisation (512, 2 -> [16, 32];
  64, 4 -> [2, 4, 2, 4] sorted -> [2, 2, 4, 4]); a prime n in d > 1 gives [1, ..., 1, n] or raises
  (worker picks, documents; a lattice with a length-1 axis is legal but degenerate).
- `SpatialConfig(d=k)` with default specs validates for k = 1..4, and `build_spaces` produces
  positive semi-definite kernels (no negative eigenvalues beyond round-off) for d = 1..4 at
  N = 512, m = n_out = 64, length_scale 0.15.
- In d dimensions on a regular isotropic lattice the kernel's top eigenvalue equals its row sum
  (translation invariance; the constant vector is the top eigenvector) - a test of the d > 1
  geometry, not just shapes.
- A sweep with `grid = {"spatial.d": [1, 2, 3, 4]}` validates and expands.
- d = 1 positions and kernels are bitwise those of `main`.
