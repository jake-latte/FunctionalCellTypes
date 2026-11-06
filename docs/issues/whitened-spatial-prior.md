---
type: infrastructure
complexity: complex
status: in review
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Whitened parametrisation of the spatial prior (`spatial.parametrisation`)

W3 of `latex/notes/fct_experiment_plan.tex`, confirmed by the PI 2026-11-06
with one addition: the experiments will run `spatial.parametrisation` as an
axis (`"penalty"` and `"whitened"` side by side) so that the two ways of
training are shown equivalent; nothing but the parametrisation may differ
between the two paths (same data, seeds, loss coefficients, monitor, history
keys, checkpoint layout up to the trained tensor).

Read Section D1 (`\label{sec:D1}`) of the plan for the full specification
(what the eigenvalue floor does, the change, the tests, the cache rule), and
the plan's "Order of work" entry W3 below, copied verbatim.

  \item \textbf{W3. Whitened spatial prior.} Type infrastructure; complexity complex.\\
  \emph{Description:} an exact parametrisation of the spatial prior, $W = K_{\mathrm{post}}^{1/2}\xi K_{\mathrm{pre}}^{1/2}$ with the plain penalty on $\xi$, so that the sampled prior is $K$ itself (no eigenvalue floor), SGLD is no longer stiff, and a loud run-start check covers the data curvature.\\
  \emph{Inputs:} Section~\ref{sec:D1} and \ref{sec:protocol}(i); \path{fct/spatial.py}, \path{fct/model.py}, \path{fct/train.py:113--164}; the readers named in D1.\\
  \emph{Outputs:} \code{spatial.parametrisation} (hashed only as \code{"whitened"}); \code{Population.S}; \code{whitening\_factors}; \code{MLP}/\code{EnsembleMLP} with whitening buffers and the prior initialisation $\xi\sim\mathcal N(0, I/\lambda_w)$; \code{effective\_weights}; \code{make\_weight\_penalty} with the power-iteration check of $\mathrm{lr}\,\lambda_{\max}(\nabla^2L)$; the readers updated.\\
  \emph{Acceptance:} the four tests of D1, with the covariance test on $n = 2000$ independent draws per mode (the relative standard error of a sample variance is $\sqrt{2/n} = 3\%$, so $5\%$ is a $1.6\sigma$ bound; use $10^4$ draws of $\xi$ for a $1.4\%$ error and a $5\%$ tolerance); a whitened SGLD run on the smoke ring driven by the prior term alone (the test steps \code{SGLD} on the gradient of \code{prior\_coefficient * weight\_l2} only) has, after $2000$ records one relaxation time apart, every prior mode's sample variance within $10\%$ of $\lambda_q^{\mathrm{post}}\lambda^{\mathrm{pre}}/\lambda_w$; the check raises on the smoke S8 setting at $k = 100$ with $\mathrm{lr} = 0.2$ and passes at $0.0005$; every existing test green and \path{tests/test_train_pins.py} bit for bit.

## Acceptance criteria
- Those of the plan's W3 and Section D1 (tests (i)-(iii); the pinned penalty
  path is bit for bit unchanged under the default).
- A `--axis 'spatial.parametrisation=["penalty","whitened"]'` experiment
  initialises, lists two settings with distinct directories and hashes, and
  runs on the `spatial_tiny` preset.
- `docs/design.md` ("Spatial weight prior in training") documents the
  parametrisation, the hash rule and the checkpoint contents.
- Full suite and `ruff` pass.
