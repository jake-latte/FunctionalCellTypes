---
type: data-science
complexity: complex
status: todo
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Map the linear-solution predictions onto the fct codebase: discrepancies and an experiment plan

PI request (2026-11-04, `latex/edit.md` item 7). The note
`latex/notes/linear_solution_limits.tex` ends with a numbered list of
numerical predictions to test. This item turns that list into a plan that a
later session can implement without further design: it is a **document**,
not code. Write `latex/notes/fct_experiment_plan.tex` (self-contained,
builds standalone with `latexmk -pdf -outdir=build`).

The theory is to lead the code conventions: where the code and the notes
disagree, the recommendation is to change the code.

## Required content
1. **Architecture and conventions, side by side.** For each of the
   following, state what the LaTeX notes define (`latex/sections/01_setup.tex`,
   `02_renormalisation_group.tex`, after the proofreading pass) and what the
   `fct` library implements, with file and function names (and line numbers
   at the commit you read), and whether they agree:
   - the forward pass and its normalisations (`1/\sqrt{m}`, `1/\sqrt{N}`,
     `1/N` readout) and the input noise `\eta` with variance `\sigma^2`;
   - the loss prefactor `N/(2Pn)`, the temperature `\beta`, SGLD, and the
     resulting `\kappa = \beta/(Pn)`;
   - the weight prior: `M = K^{-1}`, the factorisation
     `K^{(l)} = K^{post} \otimes K^{pre}`, the scalars `k_0`, `k_1`, the
     Fourier eigenvalues `\lambda_q`, periodic boundaries, the embedding of
     inputs/outputs in space (`\mathbf{x}_a`, `\mathbf{x}_\alpha`);
   - the firing-rate regulariser `G = g^2 I` and its layer range;
   - the activation (linear case: `id`) and the number of hidden layers;
   - the task: `X = Z W_X^\top`, `Y = Z W_Y^\top`, `Z` i.i.d. uniform on
     `[0,a]`, `W_X`, `W_Y` orthonormal columns, versus what `fct/tasks.py`
     and `fct/data.py` generate (look at the linear-teacher task and the
     latent-lattice test set; the PI has uncommitted edits to `fct/tasks.py`
     and `fct/presets/phase_transition.json` in the main checkout, which
     you may read but not modify: `git -C <root> diff -- fct/tasks.py`);
   - what the mean-field solver (`fct/meanfield.py`, `fct/hmc.py`) solves
     and whether its saddle-point equations match the notes' (including the
     sign and factor of `i` conventions on `\hat\Phi`).
   Relevant entry points: `docs/design.md` (the library contract),
   `fct/config.py` (every hyperparameter), `fct/spatial.py` (kernels),
   `fct/model.py`, `fct/train.py`, `fct/sgld.py`, `fct/tasks.py`,
   `fct/data.py`, `fct/presets/*.json`, `fct/spatial_exp.py`,
   `fct/analyse.py`, `fct/modularity.py`, `tests/test_prior_convention.py`,
   `tests/test_spatial_prior.py`.
2. **Discrepancies and how to address them in the code.** A table or list:
   discrepancy, where in the code, what the notes say, the concrete change
   (new config field, renamed convention, changed normalisation, new task
   generator), which tests would change, and whether existing presets or
   cached results would be invalidated (`Config.section_hash`). Order by how
   much they matter for the planned experiments.
3. **The experiments.** For each numbered prediction in
   `linear_solution_limits.tex`: the exact `fct` invocation (preset or
   `fct experiment` with `--axis` values, every relevant config field with
   its value, the seed policy, `N`, `P`, `m`, `n`, `K`, `a`, `\sigma`, `g`,
   `k_0`, `k_1`, `\beta`, the kernel length scale, number of SGLD steps),
   the quantity to compute from the trained network and *how* (which
   existing function, or a specification of a new one: input, output, maths),
   the figure to draw, and the predicted result. Include a small "smoke"
   version that runs on CPU in minutes and the full version. State what
   must be implemented first (from section 2) before each experiment can
   run.
4. **Order of work** for the implementing session: a numbered list of work
   items in the delab format (title, type, inputs/outputs, acceptance
   criteria), small enough to delegate one at a time.

Give absolute detail; the implementing session should not need to reopen
the theory. Do not edit any code or section files.

## Inputs / outputs / interfaces
- Inputs: `latex/notes/linear_solution_limits.tex`, the two section files,
  `docs/design.md`, the `fct/` package and `tests/`.
- Outputs: `latex/notes/fct_experiment_plan.tex` only.

## Acceptance criteria
- The note builds standalone.
- Every claim about the code cites file and function (and line at the
  commit read, named by hash).
- Every discrepancy has a concrete code change and a note on cache/test
  impact.
- Every prediction from the limits note has an experiment with concrete
  values, and an analysis specification.
- The order-of-work list is in the delab work-item format.
- Report: the main discrepancies found, and the worktree path and branch.
