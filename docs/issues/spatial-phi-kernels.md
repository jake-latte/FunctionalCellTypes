---
type: infrastructure
complexity: complex
status: awaiting your confirmation
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Replace the Phi section of the spatial notebook with the paper's spatially indexed kernels

**Why.** The current Phi cells plot Phi_{mu nu} = (1/N) sum_i psi_i(mu) psi_i(nu), which has no
spatial index and ignores the prior. `Latex/sections/01_setup.tex` (eq. kernel-def and
phi-diagonalised) defines, for the weight layer that reads from hidden layer l with factorised
prior K_{iajb} = K^post_ij K^pre_ab,
  Phi^(l) {mu nu}_{ij} = (1/N) sum_{a,b} psi^(l)_{mu a} K^pre_ab psi^(l)_{nu b} K^post_ij
                       = K^post_ij  A^(l)_{mu nu},
  A^(l)_{mu nu} = (1/N) psi_mu^T K^pre psi_nu   (the data-indexed kernel; the pre-synaptic
                                                 spatial index is summed out).
The post-synaptic kernel only multiplies.

**Plan (to confirm).**
1. Library, pure, in `fct/spatial.py` (next to `weight_quadratic_form`):
   `data_kernel(psi (P, n_pre), K_pre (n_pre, n_pre)) -> (P, P)` = psi K_pre psi^T / n_pre, and
   `spatial_kernel_pairs(A (P, P), K_post (n_post, n_post), trials: sequence of sample indices)
   -> (S, S, n_post, n_post)` with entry [s, t] = A[trials[s], trials[t]] * K_post.
   Tests: `data_kernel` with K_pre = I equals psi psi^T / n_pre; against the full 4-index
   einsum on a tiny case; `spatial_kernel_pairs` against the definition; ValueErrors on shape
   mismatches and trial indices out of range.
2. Which K for which layer: hidden layer l (0-based, as `RUN.net.L` counts) is the
   pre-synaptic population of weight layer l + 1, i.e. `SPACES[l + 1]`: K^pre = `SPACES[l + 1].pre.K`
   (hidden kernel), K^post = `SPACES[l + 1].post.K` (hidden for l + 1 < L, the output population
   for the last hidden layer). psi follows the library's readout convention as in the present
   cells (rates phi(h), except h on the last hidden layer under readout "h"), stated in the
   equation. Normalisation 1/N as in the paper (the input kernel Phi^(0) with 1/m is NOT plotted
   unless you want it: say so).
3. Notebook cells replacing `PHI_TEXT`/`PHI`:
   a. One figure, one panel per hidden layer: A^(l) (P x P), samples sorted by `PHI_SORT_LATENT`
      as now, diverging symmetric colours, title with tr A / P.
   b. `PHI_TRIALS = 4`: the subset is the first `PHI_TRIALS` samples in the sorted order spread
      evenly over it (quantiles of the sorting latent), so the pairs span the latent's range; a
      small figure shows the latent values z_{f mu} of the chosen trials (one group of k bars per
      trial, latent names in the legend) with their sample indices as x labels.
   c. One figure per hidden layer: an S x S grid of panels, panel (mu, nu) the (n_post x n_post)
      matrix Phi^{mu nu}_{ij}, neurons in ring order (`position_order`) for d = 1, one colour
      scale across the grid (symmetric, diverging) so that the factor A^{mu nu} is visible from
      panel to panel; row and column headers give the sample indices.
   d. The markdown states the definition, the factorisation, and that under a factorised prior
      every panel is K^post scaled by A^{mu nu} (that is the content of eq. phi-diagonalised).
4. `N_FIGURES` and `PLOTTING_FUNCTIONS` in `tests/test_build_spatial.py` updated; the committed
   notebook regenerated; `docs/design.md` and the README's notebook description updated.

## Acceptance criteria
- The library tests above pass; the notebook builds on CPU at the defaults and on the T1000 with
  DEVICE='cuda' with no error (device tensors converted with `.detach().cpu().numpy()`).
- Lint and the full suite pass.
