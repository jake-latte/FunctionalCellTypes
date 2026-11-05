---
type: data-science
complexity: complex
status: done
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Translate the discrete spatial analysis into the continuous field limit with a UV cutoff

PI request (2026-11-04, `latex/edit.md` item 9). The notes
(`latex/sections/01_setup.tex`, `02_renormalisation_group.tex`) are written for
`N` neurons on a discrete lattice with a translation-invariant prior `K`
diagonalised by the discrete Fourier transform `F` with eigenvalues
`\lambda_q`. Write a **self-contained LaTeX document**
`latex/notes/continuum_limit.tex` (own preamble, builds standalone with
`latexmk -pdf -outdir=build`) that shows the translation of this discrete
analysis into the continuous field limit, under the PI's convention: as
`N \to \infty` the **lattice spacing stays fixed** while the **volume
increases**; consequently a **maximum Fourier scale `\Lambda`** (UV cutoff,
set by the spacing) is introduced, and momentum sums become integrals with
the usual density of states.

## Required content
1. **Dictionary.** A table from discrete to continuum: site index `i` to
   position `x \in [0, L)^d` (`L = N^{1/d} \cdot` spacing), `\phi^{\mu}_i` to a
   field `\phi^{\mu}(x)`, `\sum_i` to `\int d^d x / a^d`, `\delta_{ij}` to
   `a^d \delta(x - x')`, `K_{ij}` to a kernel `K(x - x')`, `F_{iq}` and
   `\lambda_q` to a continuous Fourier transform with
   `\sum_q \to L^d \int_{|q| < \Lambda} d^d q / (2\pi)^d`, and the
   normalisation of `\lambda(q)` such that `K(x-x') = \int \frac{d^d q}{(2\pi)^d}
   \lambda(q) e^{iq(x-x')}`. Make every factor of `a`, `L`, `N`, `(2\pi)`
   explicit, and be careful that the prior's `K` and the penalty `M = K^{-1}`
   transform consistently (an inverse on the lattice is not the inverse
   kernel in the continuum without the right measure).
2. **The single-site functional.** Rewrite the single-site weight
   `\mathcal W[\Phi, \hat\Phi]` and the free-energy density `\mathcal F`
   as a functional integral over `\phi^{\mu}(x)`; show which quantities are
   intensive, which are extensive (`\propto L^d`), and that `\mathcal F` as
   defined in the notes (`\frac1N \log \mathcal W`) becomes a free energy
   *per unit volume*. Comment on the remark in the notes that the spatial
   average "has no reason to be `O(1)`" in the light of the continuum form.
3. **The one-hidden-layer linear solution in the continuum.** Translate the
   chain `\tilde A_q`, `\tilde B_q`, `\Sigma_q`, `C = \frac1N\sum_q
   \lambda_q \Sigma_q`, `D_C` and `eq:linear-soln` (use the corrected forms
   from the proofreading pass, and say which) into `q`-integrals with the
   cutoff `\Lambda`. Identify the regime of UV sensitivity: for a given
   spectrum `\lambda(q)` (take at least the squared-exponential kernel
   `\lambda(q) \propto e^{-\ell^2 q^2/2}` and a power law
   `\lambda(q) \propto (q^2 + m^2)^{-1}` in `d = 1, 2`), does the solution
   depend on `\Lambda` as `\Lambda \to \infty`, and what role does
   `\sigma^2` play as a regulator? Give the `L \to \infty` limit and the
   `L`-dependence of finite-size corrections.
4. **Towards the RG.** State the field theory (quadratic part plus the
   interaction from a non-linear `\phi`, as sketched in Appendix A1) in the
   form in which a momentum-shell RG step (integrate out
   `\Lambda / b < |q| < \Lambda`, rescale) is set up; write the engineering
   dimensions of the fields and couplings in `d` dimensions; identify the
   Gaussian fixed point and which couplings are relevant at tree level.
   This is a set-up section, not the RG calculation.
5. **Checks.** A `numpy` script under `latex/notes/scripts/` that computes
   `\frac1N \sum_q \lambda_q f(\lambda_q)` on a ring of `N` sites for a
   smooth `f` and compares with the continuum integral for increasing `N`
   at fixed spacing, confirming the dictionary's factors. Quote the result.

Use the notes' notation. Do not edit section files.

## Inputs / outputs / interfaces
- Inputs: `01_setup.tex`, `02_renormalisation_group.tex`, `A1_comments.tex`,
  `latex/notes/rg_*.tex` (if any).
- Outputs: `latex/notes/continuum_limit.tex`,
  `latex/notes/scripts/continuum_limit_check.py`.

## Acceptance criteria
- The note builds standalone; the script runs with `uv run python` in
  under a minute on CPU and prints the quoted numbers.
- The dictionary table makes every factor explicit and is used consistently
  in sections 2-4 (spot-check: the `\sigma \to 0`, `g \to 0` continuum
  `eq:linear-soln` reduces to the discrete one when the integral is
  replaced by the Riemann sum).
- The UV-sensitivity statement is derived for both spectra and both
  dimensions.
- Report: conclusions, and the worktree path and branch.
