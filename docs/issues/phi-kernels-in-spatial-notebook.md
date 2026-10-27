---
type: infrastructure
complexity: simple
status: blocked
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Spatial notebook: the activity kernels Phi of the trained network

PI request 2026-10-27: also plot the $\Phi$ kernels, with equations. Blocked on
`spatial-notebook-inline-plots-and-equations` (same inline style, lands after it).

What $\Phi$ is (`fct/meanfield.py: empirical_kernels`): with the pre-activations
$h^{(l)}_\mu \in \mathbb{R}^N$ of hidden layer $l$ at training input $x_\mu$ and
$\psi^{(l)}_\mu = \phi(h^{(l)}_\mu)$ (for the last hidden layer $\psi = h$ when the readout is
linear, `cfg.meanfield.readout != "phi"`), the empirical kernel of layer $l$ is the $P \times P$
matrix over samples
$$\Phi^{(l)}_{\mu\nu} = \frac{1}{N} \sum_{i=1}^{N} \psi^{(l)}_{i\mu}\,\psi^{(l)}_{i\nu}.$$
This is the object the mean-field theory predicts (its saddle point is $\Phi$, $\hat\Phi$); the
spatial notebook does not run the solver, so only the trained network's $\Phi^{(l)}$ is shown, and
the prose says so. (The library's mean-field solver has no spatial prior, so a theory $\Phi$ for
this net does not exist yet; that would be a separate item.)

Inputs: `fct.meanfield.empirical_kernels(net, dataset.x, readout)` on ALL P training samples
(not `net_kernels`, which slices to `meanfield.n_points`), float64 on the CPU. Output: one
figure per hidden layer (or one row of panels), `imshow` of $\Phi^{(l)}$ with its own colour
scale, titled with $\mathrm{tr}\,\Phi^{(l)}/P$ (the kernel scale, as `kernel_scales` in
`fct/exemplar.py`). Samples are shown in the order of a chosen latent: a parameter cell
`PHI_SORT_LATENT = 0` (index of the latent to sort samples by; `None` keeps dataset order), so
the kernel's block structure along that latent is visible; the markdown cell gives the equation
above, says which readout convention is in force, and that samples are sorted.

## Acceptance criteria
- A test that `Phi` from the notebook's helper equals `psi @ psi.T / N` for a tiny net (already covered for `empirical_kernels`; add a test only for any new library function, e.g. a sample-ordering helper, and that `PHI_SORT_LATENT=None` gives dataset order).
- The notebook builds and executes on the tiny configuration on CPU; one $\Phi$ panel per hidden layer, the equation in the markdown cell.
- Full suite passes, ruff clean.
