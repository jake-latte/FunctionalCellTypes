---
type: infrastructure
complexity: simple
status: in review
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Add the identity activation `id` so a setting can train a linear network

PI request (2026-10-28), mirroring the same change already merged in the sibling FeedbackAlignment repo: a fourth `net.phi` value, `"id"`, with phi(h) = h and phi'(h) = 1, so a setting trains a linear network (every hidden layer linear; the readout V^T phi(h^L)/N is then linear in h^L). The PI will, for now, train only spatially embedded networks (`spatial` section set; the mean-field solver refuses those), and wants to run a grid of spatial networks with `id` alongside tanh/relu.

## Inputs / outputs / interfaces
- `fct/activations.py`: register `"id"` in `PHI` mapped to the EXISTING `identity` function (so its analytic derivative and its tilt class "linear" apply; no second identity function). Update `identity`'s docstring.
- `fct/config.py`: the `Phi` Literal gains `"id"`.
- Anything that enumerates or colours activations accepts `id`: `fct/lrplot.py` (`PHI_COLOURS`), the notebook builders' `PHI = ...` comments, design doc, README.
- Presets are NOT changed; an `id` grid is made with `--axis net.phi=[...]`.
- `fct/meanfield.py`, `fct/hmc.py`, `fct/importance.py` are not edited. The solver's behaviour on an iid `id` setting must not be silent garbage: a test that a tiny iid `id` solve stage completes with a finite residual or raises an error naming the activation.

## Acceptance criteria
- `get_phi("id")` is `identity`; `phi_prime` is ones; a 2-hidden-layer `id` net's forward pass equals the explicit matrix product (linear in the input).
- A tiny spatial `id` sweep (`spatial_tiny`, axis `net.phi` over ["tanh", "id"], 4 settings) runs end to end on CPU through the CLI; every loss is finite and decreasing from its first logged value.
- The spatial grid notebook executes against that sweep with an `id` setting selected; lrplot accepts `id`; tests cover it.
- Full suite green, `ruff check` clean, design doc and README list `id`.
