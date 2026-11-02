---
type: infrastructure
complexity: simple
status: in review
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Let a spatial kernel's length_scale be 0, giving the identity kernel

PI request (2026-11-02): when the length scale of an embedding kernel is set to 0, the identity is used for that population's kernel. This gives a spatial setting with an unstructured (plain-L2-shaped) prior on a factor, as the endpoint of a length_scale sweep.

Definition: at length_scale = 0, the wrapped Gaussian is replaced by its limit as length_scale → 0: K_ij = amplitude if x_i ≡ x_j (mod box) on every axis, else 0. Population positions are distinct, so a population's K is exactly amplitude · I (or lambda_max · I when `lambda_max` is set, since the top eigenvalue of I is 1). Implement this in the kernel functions themselves (`wrapped_gaussian_profile`, and so `periodic_gaussian_kernel`), not as a separate branch in `_make_population`, so the prior and the plotting helpers share one code path. Do not let length_scale = 0 reach `_wrapped_gaussian_1d` (division by zero).

## Inputs / outputs / interfaces
- `fct/spatial.py`: `KernelSpec.length_scale` changes from `PosFloat` to `NonNegFloat` (`fct/schema.py` already has it); negative values are still rejected. `wrapped_gaussian_profile` handles length_scale == 0 as above (after its existing reduction of delta to [0, box/2], test delta == 0 exactly). Update the module docstring's kernel section and the `KernelSpec` comment to state the length_scale = 0 case.
- No other module computes with length_scale (`fct/spatial_view.py:population_spec` only passes it through; `fct/exp.py` axes take it via KernelSpec). Check with grep that this is still true, and if anything else divides by length_scale, report it rather than silently changing it.
- Existing tests live in `tests/test_spatial.py`, `tests/test_spatial_prior.py` and others; put new tests next to the existing kernel tests.

## Acceptance criteria
- `KernelSpec(length_scale=0.0)` validates; `KernelSpec(length_scale=-0.1)` raises.
- `periodic_gaussian_kernel(x, x, 0.0, box, amplitude)` equals `amplitude * eye(n)` exactly (float64) for a grid and for a random embedding, in d = 1 and d = 2.
- It also handles positions that differ by a whole multiple of box (they count as coinciding) and returns a non-square (n_post, n_pre) result without error.
- `wrapped_gaussian_profile(delta, 0.0, box, a)` is `a` where delta ≡ 0 (mod box) and 0 elsewhere, for signed delta.
- Building the spaces with length_scale = 0 for the hidden population gives `K == I`, `M == I` and a FloorReport with n_floored = 0, without lambda_max (with amplitude a: K = a·I, M = I/a). With `lambda_max = c` it gives `K == c·I` and `scale == c`.
- Mixed configs work: e.g. hidden length_scale 0 with input/output at 0.15.
- Small positive length scales are unchanged: existing tests pass, and the full test suite passes.
