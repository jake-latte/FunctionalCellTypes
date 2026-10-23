---
type: data-science
complexity: complex
status: in review
---

<!-- status: todo | in progress | in review | done | blocked | awaiting your confirmation -->

# Spatial-prior notebook: kernel cell, training, tuning of each cell

Confirmed by the PI 2026-10-22. Linear teacher task; mirrors the exemplar notebook.

## Proposed plan
Cells: (1) task and net hyperparameters; (2) training hyperparameters including `optimizer = "sgld" | "adam"` and beta; (3) **spatial cell**: embedding choice and kernel parameters per population, then plot the kernels and embeddings; (4) train (resumable through the normal cache path); (5) plots at the end of training: loss curves, outputs, empirical kernels, and the tuning of every cell to the latent variables (one panel per neuron, laid out by its position in space where d <= 2, one curve per latent), plus the first-layer weights arranged over the embedding.

## Acceptance criteria
- Builds and executes on a tiny configuration on CPU for both optimisers.
- The tuning-by-position plot reuses the `grid-results-notebook` tuning helper (no duplicate code).
- Tests CPU only, ruff clean.
