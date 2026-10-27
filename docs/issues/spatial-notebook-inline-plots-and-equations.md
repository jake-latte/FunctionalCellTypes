---
type: infrastructure
complexity: complex
status: awaiting your confirmation
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Spatial notebook: plotting code written out in the cells, prose with equations

PI feedback 2026-10-27 on `notebooks/spatial.ipynb` (built by `notebooks/build_spatial.py`):
1. The plotting functions are hidden in `fct.spatial_report` / `fct.draw`; write them out in full
   in the notebook so a reader sees what is plotted.
2. The prose is unintelligible, in particular "First-layer weights over the embedding" and the
   eigenvalue-floor text; give equations that say exactly what each figure shows, and say whether
   the kernels shown are the Phi kernels of the mean-field theory (they are not).

## Proposed plan
Rewrite `build_spatial.py` so that each figure's plotting function is defined in full in the
code cell that uses it (plain matplotlib on arrays the library returns), and each figure is
introduced by a markdown cell that defines the quantity with an equation. The library keeps the
computations (kernels, penalty matrices, training, rates, tuning curves, MI) and loses the
notebook-only figure code, so there is one copy of it: delete `fct/spatial_report.py` and the
spatial-only figure functions in `fct/draw.py` (`fig_kernel`, `fig_empirical_kernels`,
`fig_weights_over_embedding`, `fig_tuning_by_position`, and `plot_kernel_vs_distance` /
`plot_embedding` if nothing else uses them) with their tests, and update `docs/design.md`.
Keep the pure helpers in `fct/spatial_view.py` (`order_weights`, `panel_grid`,
`empirical_kernels`, `ordered_kernel_comparison`) and their tests. `python -m fct plot` must not
lose a figure (check `fct/cli.py` before deleting).

Notation to use throughout (one symbol per object, consistent with `docs/design.md`):
- Task: latents $z_\mu \in \mathbb{R}^k$, inputs $x_\mu = W_X z_\mu$, targets $y_\mu = W_Y z_\mu$,
  $\mu = 1..P$ (take the exact distributions from `fct/tasks.py`).
- Net: $h^{(l)}_\mu = \phi(W_l h^{(l-1)}_\mu)$, $h^{(0)}_\mu = x_\mu$, first layer $W_1 = U$ of shape
  (N, m); rates $r^{(l)}_{i\mu} = h^{(l)}_{i\mu}$.
- Prior: every unit $i$ of population $p$ has a position $x^p_i$ on the torus $[0, \mathrm{box})^d$;
  the population kernel is the wrapped Gaussian of `fct/spatial.py`
  $K^p_{ij} = A \prod_a W(\delta_a)/W(0)$, $W(\delta) = \sum_{|n| \le n_\mathrm{img}} e^{-(\delta + n\,\mathrm{box})^2 / 2\ell^2}$;
  the layer prior covariance is $K^{\mathrm{post}} \otimes K^{\mathrm{pre}}$; the penalty is
  $M^p = \sum_j \max(\lambda_j, \epsilon\,\lambda_{\max})^{-1} v_j v_j^\top$ from
  $K^p = \sum_j \lambda_j v_j v_j^\top$ with $\epsilon$ = `MIN_REL_EIG`; the loss term is
  $\lambda_w \sum_l \mathrm{tr}(M^{\mathrm{post}}_l W_l M^{\mathrm{pre}}_l W_l^\top)$, and under SGLD at
  inverse temperature $\beta$ the stationary law $\propto e^{-\beta L}$ makes the prior alone
  $\mathrm{vec}(W_l) \sim \mathcal{N}(0, (2\beta\lambda_w)^{-1} K^{\mathrm{post}} \otimes K^{\mathrm{pre}})$.
- "First-layer weights": $\hat K^{\mathrm{post}} = U U^\top / m$ and $\hat K^{\mathrm{pre}} = U^\top U / N$,
  shown next to $K^{\mathrm{post}}$, $K^{\mathrm{pre}}$ with units sorted by ring position; state that
  under the prior $\mathbb{E}[\hat K^{\mathrm{post}}] = c A K^{\mathrm{post}}$ with $c = (2\beta\lambda_w)^{-1}$
  (ignoring the floor), so the comparison is up to that scale; and the image of $U$ itself sorted
  the same way. State explicitly that these are second moments of the trained WEIGHTS over
  units, not the mean-field kernels $\Phi_l = $ (P x P) sample-by-sample activity kernels of
  `fct/meanfield.py`, which this notebook does not compute.
- Tuning: for neuron $i$ and latent $f$, rank-bin $z_f$ into $B$ bins and plot the bin means
  $\bar r_{i}(b) = \frac{1}{|b|}\sum_{\mu \in b} r_{i\mu}$ (the other latents averaged out).
- Per-neuron MIR in the titles: $I_{if}$ = plug-in MI of the rank-binned rate with the rank-binned
  latent (nats; shuffle-corrected when `MIR_KIND = "corrected"`), preference
  $\rho_i = \max_f I_{if} / \sum_f I_{if}$, $\mathrm{MIR}_i = (\rho_i - 1/K)/(1 - 1/K)$.
- Eigenvalue floor text: replace the "Why the default sizes are small" paragraph with the
  stability condition in symbols, $\mathrm{lr}\,\lambda_w\,\lambda_{\max}(M) < 1$ with
  $\lambda_{\max}(M) = 1/(\epsilon\,\lambda_{\max}(K))$, and the numbers for the defaults.

Each markdown cell: one short paragraph, the equation, and one sentence on how to read the figure.

Interfaces: `build_spatial.py` (cell sources), `fct/draw.py`, `fct/spatial_report.py` (deleted),
`fct/spatial_view.py` (unchanged), `docs/design.md`, `README.md` (spatial notebook paragraph).
The PI has uncommitted edits to `notebooks/spatial.ipynb` in the shared tree; the rebuilt source
notebook will conflict with them at merge (the PM resolves, the PI's parameter values are not
the defaults).

## Acceptance criteria
- The executed notebook contains no call into `fct.spatial_report` or `fct.draw`; every figure is produced by a function defined in a notebook cell.
- Every figure is preceded by a markdown cell with the equation of what is plotted, in the notation above; the kernel section says the kernels are weight-prior and weight second-moment kernels, not the mean-field Phi.
- The figures are the same ones as before (kernels and positions per layer; loss curves; example outputs; empirical vs prior kernels; U sorted by position; tuning by position), with the same content.
- Builds and executes on the tiny default configuration on CPU for both optimisers (`--set "OPTIMIZER='adam'"`), and on CUDA once `fix-tuning-view-cuda` is merged.
- No dead code left in the library; `docs/design.md` and `README.md` updated; full suite passes; ruff clean.
