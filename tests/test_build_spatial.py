"""The spatial-prior notebook: structure of the source, the plotting functions its cells define
(run on synthetic inputs with a known answer), parameter injection, and an end-to-end execution
on a tiny configuration for each optimiser (CPU)."""

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pytest
import torch

from fct.spatial import EmbeddingSpec, PopulationSpec, SpatialConfig, build_spaces
from fct.spatial_view import ordered_kernel_comparison, population_spec
from fct.tuning import tuning_curves
from notebooks.build_spatial import build_notebook, execute_notebook, set_parameters
from tests.draw_helpers import assert_png

# Hidden layers L = 1: prior figures of its L + 1 = 2 weight layers, loss curves, example
# outputs, the two first-layer figures (empirical kernels, weights), the activity kernels Phi
# and one tuning figure.
N_FIGURES = 2 + 1 + 1 + 2 + 1 + 1
TINY = {"P": 32, "M_IN": 6, "K_LATENTS": 2, "N": 8, "N_EPOCHS": 40}
# A code cell that calls one of these draws a figure.
FIGURE_CALLS = ("plt.subplots(", "plt.figure(")
# The plotting functions the notebook's cells define, one per figure cell.
PLOTTING_FUNCTIONS = {
    "plot_layer_prior",
    "plot_loss_curves",
    "plot_example_outputs",
    "plot_kernel_comparison",
    "plot_weights_by_position",
    "plot_activity_kernels",
    "plot_tuning_by_position",
}

# Ring positions of 6 units in a scrambled order: unit 0 is 4th around the ring, and so on.
RING = torch.tensor([[0.50], [0.00], [0.75], [0.25], [0.90], [0.10]], dtype=torch.float64)
RING_ORDER = [1, 5, 3, 0, 2, 4]  # units sorted by position


def _code(nb) -> str:
    return "\n".join(c.source for c in nb.cells if c.cell_type == "code")


def _draws_a_figure(source: str) -> bool:
    return any(call in source for call in FIGURE_CALLS)


def _text_outputs(nb) -> str:
    chunks = []
    for cell in nb.cells:
        for out in cell.get("outputs", []):
            if out.output_type == "error":
                chunks.append(f"ERROR {out.ename}: {out.evalue}")
            elif out.output_type == "display_data":
                chunks.append(out.data.get("text/markdown", ""))
            elif out.output_type == "stream":
                chunks.append(out.text)
    return "\n".join(chunks)


def _n_figures(cell) -> int:
    return sum("image/png" in out.get("data", {}) for out in cell.get("outputs", []))


def _previous_markdown(nb, index: int) -> str:
    previous = nb.cells[index - 1]
    return previous.source if previous.cell_type == "markdown" else ""


@pytest.fixture(scope="module")
def notebook_functions() -> dict:
    """The imports and function definitions of every code cell, run without the rest: the
    plotting functions exactly as the notebook defines them, callable on synthetic data."""
    nodes = []
    for cell in build_notebook().cells:
        if cell.cell_type == "code":
            body = ast.parse(cell.source).body
            nodes += [
                n for n in body if isinstance(n, ast.Import | ast.ImportFrom | ast.FunctionDef)
            ]
    namespace: dict = {}
    # exec of the notebook's own generated source, the only way to call what its cells define
    exec(compile(ast.Module(nodes, type_ignores=[]), "spatial.ipynb", "exec"), namespace)  # noqa: S102
    return namespace


# ---------------------------------------------------------------- the source notebook


def test_source_notebook_validates_has_no_outputs_and_no_path():
    nb = build_notebook()
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert all(not c.outputs and c.execution_count is None for c in code)
    assert "/ceph" not in _code(nb) and "/nfs" not in _code(nb)
    parameter = [c for c in nb.cells if "parameters" in c.metadata.get("tags", [])]
    assert len(parameter) == 1 and "CACHE_ROOT" in parameter[0].source


def test_source_notebook_exposes_task_net_training_and_spatial_parameters():
    nb = build_notebook()
    cells = [c.source for c in nb.cells if c.cell_type == "code"]
    # task and net hyperparameters are one cell at the top
    for name in ("P =", "M_IN =", "K_LATENTS =", "N =", "L =", "PHI =", "G =", "WEIGHT_LAMBDA ="):
        assert name in cells[0]
    assert "OPTIMIZER =" in cells[1] and "BETA =" in cells[1] and "N_EPOCHS =" in cells[1]
    spatial = next(c for c in cells if "INPUT =" in c)
    for name in ("HIDDEN =", "OUTPUT =", "MIN_REL_EIG =", "D =", "population_spec"):
        assert name in spatial
    assert '"sgld"' in cells[1] and '"adam"' in cells[1]


def test_no_cell_uses_the_library_figure_code():
    """Every figure is drawn by code written out in the notebook, never by `fct.draw`."""
    for cell in build_notebook().cells:
        if cell.cell_type == "code":
            for name in ("fct.draw", "from fct import draw", "spatial_report"):
                assert name not in cell.source, cell.source


def test_every_figure_cell_is_introduced_by_an_equation():
    nb = build_notebook()
    figure_cells = [
        i for i, c in enumerate(nb.cells) if c.cell_type == "code" and _draws_a_figure(c.source)
    ]
    assert len(figure_cells) == len(PLOTTING_FUNCTIONS)
    for i in figure_cells:
        assert "$$" in _previous_markdown(nb, i), nb.cells[i].source  # a display equation


def test_every_figure_cell_defines_its_plotting_function_before_calling_it():
    """One plotting function per figure cell, defined at the top and called at the bottom."""
    defined = set()
    for cell in build_notebook().cells:
        if cell.cell_type != "code" or not _draws_a_figure(cell.source):
            continue
        body = ast.parse(cell.source).body
        kinds = [isinstance(n, ast.FunctionDef) for n in body]
        first_def, last_def = kinds.index(True), len(kinds) - 1 - kinds[::-1].index(True)
        # only constant settings may come before the definitions
        assert all(isinstance(n, ast.Assign) for n in body[:first_def]), cell.source
        assert all(kinds[first_def : last_def + 1]) and last_def < len(body) - 1, cell.source
        names = {n.name for n in body if isinstance(n, ast.FunctionDef)}
        assert len(names & PLOTTING_FUNCTIONS) == 1, names
        defined |= names
    assert PLOTTING_FUNCTIONS <= defined


def test_the_loss_and_the_depth_have_different_symbols():
    """L is the depth; the loss is mathcal L, never a bare L in an equation."""
    text = "\n".join(c.source for c in build_notebook().cells if c.cell_type == "markdown")
    for clash in ("$L = ", "$$L = ", "\\nabla_\\theta L", "\\beta L", "$L$ curve", "1/K"):
        assert clash not in text, clash
    assert "\\mathcal{L}" in text


def test_the_kernel_text_says_they_are_not_the_mean_field_kernels():
    text = "\n".join(c.source for c in build_notebook().cells if c.cell_type == "markdown")
    assert "U U^\\top / m" in text and "U^\\top U / N" in text
    assert "not the mean-field" in text and "fct/meanfield.py" in text


# ---------------------------------------------------------------- the plotting functions


def _space(d: int, **sizes):
    spec = PopulationSpec(embedding=EmbeddingSpec(kind="random", seed=1))
    cfg = SpatialConfig(d=d, input=spec, hidden=spec, output=spec)
    return build_spaces(cfg, **{"m": 6, "N": 8, "L": 1, "n_out": 4, **sizes})[0]


@pytest.mark.parametrize("d", [1, 2])
def test_layer_prior_draws_both_kernels_against_distance(notebook_functions, tmp_path, d):
    fig = notebook_functions["plot_layer_prior"](_space(d))
    assert_png(fig, tmp_path)
    ax = next(ax for ax in fig.axes if "distance" in ax.get_xlabel())
    assert len(ax.collections) == 2 and len(ax.lines) == 2  # pre and post, each points + curve


def test_layer_prior_rejects_d3_without_opening_a_figure(notebook_functions):
    n_before = len(plt.get_fignums())
    with pytest.raises(ValueError, match="d = 3"):
        notebook_functions["plot_layer_prior"](_space(3))
    assert len(plt.get_fignums()) == n_before


def test_loss_curves_draw_every_logged_term(notebook_functions):
    history = {
        "epoch": [0, 10, 20],
        "loss": [3.0, 2.0, 1.0],
        "mse": [1.0, 0.5, 0.2],
        "weight_l2": [5.0, 4.0, 3.0],
        "rate_l2": [0.0, 0.0, 0.0],
    }
    fig = notebook_functions["plot_loss_curves"](history, "sgld")
    (ax,) = fig.axes
    labels = [line.get_label() for line in ax.lines]
    # a term that is zero throughout is invisible on a log axis, so the legend says so
    assert len(labels) == 4 and labels[-1].endswith("(zero)") and labels[0] == r"loss $\mathcal{L}$"


def test_example_outputs_show_each_example_and_a_scatter(notebook_functions, tmp_path):
    from fct.analyse import ExampleOutputs

    y = torch.tensor([[1.0, -1.0, 0.5], [0.0, 2.0, -0.5]])
    outputs = ExampleOutputs(torch.tensor([3, 7]), torch.zeros(2, 4), y, 0.9 * y)
    fig = notebook_functions["plot_example_outputs"](outputs, "trained network")
    assert_png(fig, tmp_path)
    titles = [ax.get_title(loc="left") for ax in fig.axes]
    assert "example 3" in titles and "example 7" in titles and len(fig.axes) == 3
    # output-index labels only under the last example row
    first, last = fig.axes[0], fig.axes[1]
    assert not first.xaxis.get_tick_params()["labelbottom"]
    assert last.xaxis.get_tick_params()["labelbottom"]


def test_kernel_comparison_and_weights_render_on_a_ring(notebook_functions, tmp_path):
    space = build_spaces(SpatialConfig(), m=6, N=8, L=1, n_out=2)[0]
    W = torch.randn(8, 6, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    comparison = ordered_kernel_comparison(space, W)
    assert_png(notebook_functions["plot_kernel_comparison"](comparison, space), tmp_path)
    assert_png(notebook_functions["plot_weights_by_position"](W, space), tmp_path)


def test_weight_image_rows_follow_the_ring_order(notebook_functions):
    cfg = SpatialConfig(hidden=population_spec("random", 0.2, seed=4))
    space = build_spaces(cfg, m=4, N=6, L=1, n_out=2)[0]
    W = torch.arange(6, dtype=torch.float64)[:, None].expand(6, 4)  # row u holds the value u
    fig = notebook_functions["plot_weights_by_position"](W, space)
    shown = fig.axes[0].images[0].get_array()[:, 0]
    order = np.argsort(space.post.embedding.positions[:, 0].numpy(), kind="stable")
    assert np.array_equal(shown, order)


def test_weight_image_needs_a_ring(notebook_functions):
    with pytest.raises(ValueError, match="d = 1"):
        notebook_functions["plot_weights_by_position"](torch.zeros(8, 6), _space(2))


def _block_phi() -> torch.Tensor:
    """Two layers of 6 samples: layer 0 has two blocks of 3 (value 1 in, 0 across), layer 1 is
    twice the identity, so its kernel scale tr(Phi) / P is 2."""
    block = torch.kron(torch.eye(2), torch.ones(3, 3))
    return torch.stack([block, 2 * torch.eye(6)]).to(torch.float64)


def test_activity_kernels_one_panel_per_layer_titled_with_the_kernel_scale(
    notebook_functions, tmp_path
):
    fig = notebook_functions["plot_activity_kernels"](_block_phi(), np.arange(6), "")
    assert_png(fig, tmp_path)
    panels = [ax for ax in fig.axes if ax.images]
    assert len(panels) == 2
    assert "hidden layer 1" in panels[0].get_title() and "= 1" in panels[0].get_title()
    assert "hidden layer 2" in panels[1].get_title() and "= 2" in panels[1].get_title()
    # each panel has its own colour scale: layer 0 spans 0 to 1, layer 1 spans 0 to 2
    assert panels[0].images[0].get_clim() == (0.0, 1.0)
    assert panels[1].images[0].get_clim() == (0.0, 2.0)


def test_activity_kernels_sorting_permutes_rows_and_columns_together(notebook_functions):
    order = np.array([0, 3, 1, 4, 2, 5])  # interleaves the two blocks
    fig = notebook_functions["plot_activity_kernels"](_block_phi(), order, "z0")
    shown = next(ax for ax in fig.axes if ax.images).images[0].get_array()
    phi = _block_phi()[0].numpy()
    assert np.array_equal(shown, phi[np.ix_(order, order)])
    assert np.array_equal(shown, shown.T)  # rows and columns moved together
    # the blocks, contiguous in dataset order, now alternate: neighbours are never in one block
    assert shown[0, 1] == 0.0 and shown[0, 2] == 1.0


def test_activity_kernels_refuse_an_order_that_is_not_a_permutation(notebook_functions):
    with pytest.raises(ValueError, match="permutation"):
        notebook_functions["plot_activity_kernels"](_block_phi(), np.array([0, 0, 1, 2, 3, 4]), "")


def _tuned_layer(tuned_unit: int):
    """Rates of 6 units to 2 latents: `tuned_unit` rises with z0, all others are flat."""
    gen = torch.Generator().manual_seed(0)
    z = 2 * torch.rand(400, 2, generator=gen, dtype=torch.float64) - 1
    rates = 0.1 * torch.randn(400, 6, generator=gen, dtype=torch.float64)
    rates[:, tuned_unit] += 3 * z[:, 0]
    return tuning_curves(rates, z, n_bins=8)


def _panel_slope(ax) -> float:
    """Rise of the first curve of a panel (the z0 curve) from its first to its last point."""
    y = ax.lines[0].get_ydata()
    return float(y[-1] - y[0])


def test_the_tuned_neuron_has_its_panel_at_its_ring_position(notebook_functions):
    unit = 2  # ring position 0.75: rank 4 of 6, so (row 1, col 0) with 4 columns
    plot = notebook_functions["plot_tuning_by_position"]
    fig = plot(_tuned_layer(unit), np.zeros(6), ["z0", "z1"], RING, 4)
    axes = np.array(fig.axes[:8]).reshape(2, 4)
    rank = RING_ORDER.index(unit)
    panel = axes[rank // 4, rank % 4]
    assert f"neuron {unit}" in panel.get_title() and "0.75" in panel.get_title()
    assert _panel_slope(panel) > 4  # rises by about 3 * 2 across the latent's range
    others = [
        _panel_slope(ax) for ax in axes.ravel() if ax.get_visible() and ax is not panel and ax.lines
    ]
    assert len(others) == 5 and max(abs(s) for s in others) < 0.5
    assert [ax.get_visible() for ax in axes.ravel()] == [True] * 6 + [False] * 2


def test_tuning_by_position_lays_a_plane_lattice_out_in_two_dimensions(notebook_functions):
    spec = population_spec("grid", 0.1, shape=[2, 3])
    cfg = SpatialConfig(
        d=2, input=spec, hidden=spec, output=population_spec("grid", 0.1, shape=[2, 1])
    )
    positions = build_spaces(cfg, m=6, N=6, L=1, n_out=2)[0].post.embedding.positions
    fig = notebook_functions["plot_tuning_by_position"](
        _tuned_layer(0), np.zeros(6), ["z0", "z1"], positions, 99
    )
    assert len(fig.axes) == 6 and all(ax.get_visible() for ax in fig.axes)


def test_tuning_by_position_checks_sizes(notebook_functions):
    with pytest.raises(ValueError, match="positions"):
        notebook_functions["plot_tuning_by_position"](
            _tuned_layer(0), np.zeros(6), ["z0", "z1"], RING[:4], 2
        )


def test_tuning_by_position_refuses_too_few_latent_names(notebook_functions):
    with pytest.raises(ValueError, match="latent names"):
        notebook_functions["plot_tuning_by_position"](_tuned_layer(0), np.zeros(6), ["z0"], RING, 2)


# ---------------------------------------------------------------- parameters and execution


def test_set_parameters_replaces_the_value_and_keeps_the_comment():
    nb = build_notebook()
    before = nbformat.writes(nb)
    out = set_parameters(nb, {"N": 4, "OPTIMIZER": "adam"})
    assert nbformat.writes(nb) == before
    code = _code(out)
    assert "\nN = 4  # " in code and 'OPTIMIZER = "adam"' not in code
    assert "OPTIMIZER = 'adam'  # " in code


def test_set_parameters_fails_on_a_name_the_notebook_does_not_set():
    with pytest.raises(ValueError, match="NOT_A_PARAMETER"):
        set_parameters(build_notebook(), {"NOT_A_PARAMETER": 1})


@pytest.mark.parametrize("optimizer", ["adam", "sgld"])
def test_executes_on_a_tiny_configuration(tmp_path, optimizer):
    nb = execute_notebook(build_notebook(), tmp_path, {**TINY, "OPTIMIZER": optimizer})
    text = _text_outputs(nb)
    assert "ERROR" not in text
    assert sum(_n_figures(c) for c in nb.cells) == N_FIGURES
    for i, cell in enumerate(nb.cells):
        if _n_figures(cell):
            assert "$" in _previous_markdown(nb, i)
    assert list(tmp_path.glob("spatial_*/train/final.pt"))  # trained through the normal cache
    assert "SGLD needs < 1" in text  # the step-size table
    assert f"Trained with {'SGLD' if optimizer == 'sgld' else 'Adam'}" in text


def test_the_committed_notebook_is_what_the_builder_produces():
    """The source notebook is generated; a stale copy would call an outdated API."""
    committed = nbformat.read(Path(__file__).parents[1] / "notebooks" / "spatial.ipynb", 4)
    assert [(c.cell_type, c.source) for c in committed.cells] == [
        (c.cell_type, c.source) for c in build_notebook().cells
    ]
