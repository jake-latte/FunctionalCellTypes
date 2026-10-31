"""The phase_transition_grid notebook: its source, one test per change the PI asked for (on the
built cells, on the plotting functions the cells define, and on executed outputs), and end-to-end
executions on a tiny `phase_transition` sweep (CPU) with one trained and one never-run setting."""

import ast
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pytest
import torch
from nbclient.exceptions import CellExecutionError

from fct.config import DataConfig, NetConfig
from fct.data import make_dataset
from fct.model import make_net
from fct.phase_view import (
    ensemble_example_outputs,
    ensemble_statistics,
    layer_priors,
    unit_order,
)
from fct.spatial import (
    EmbeddingSpec,
    KernelSpec,
    PopulationSpec,
    SpatialConfig,
    build_spaces,
)
from fct.sweep import expand, run_setting, save_sweep, setting_dir
from notebooks.build_phase_transition_grid import build_notebook
from notebooks.build_spatial import execute_notebook
from tests.test_build_spatial import _code, _n_figures_in, _text_outputs
from tests.test_phase_view import _tiny_sweep

CPU, F64 = torch.device("cpu"), torch.float64
# One figure per weight layer for the prior matrices, the spectra and the weight kernels (L = 1:
# two weight layers), then the loss curves, the example outputs, A twice, tuning of one member,
# the overlaps twice.
N_PRIOR_FIGURES = 2 + 2
N_FIGURES = N_PRIOR_FIGURES + 1 + 1 + 2 + 2 + 1 + 2
GUARDED_CELLS = 7  # trained-only code cells: loss, examples, kernels, A, tuning, overlaps, + loader
PARAMETERS = {"MEMBER": 1, "SORT_DIM": 1, "P_TEST": 40, "N_EIG": 3, "N_NEURONS": 4}


def _plotting_namespace() -> dict:
    """The functions the notebook's cells define, executed in a namespace with its imports."""
    ns: dict = {"np": np, "torch": torch, "plt": plt, "mpl": mpl}
    exec("from fct.analyse import r_squared", ns)  # noqa: S102 - the notebook's own import
    for cell in build_notebook().cells:
        if cell.cell_type != "code":
            continue
        for node in ast.parse(cell.source).body:
            if isinstance(node, ast.FunctionDef):
                exec(ast.unparse(node), ns)  # noqa: S102 - the notebook's own function
    return ns


@pytest.fixture(scope="module")
def draw():
    return _plotting_namespace()


def _random_spatial(d: int) -> SpatialConfig:
    pop = PopulationSpec(
        embedding=EmbeddingSpec(kind="random"), kernel=KernelSpec(length_scale=0.2)
    )
    return SpatialConfig(d=d, input=pop, hidden=pop, output=pop, min_rel_eig=1e-2, lambda_max=0.5)


@pytest.fixture(scope="module")
def spaces():
    return build_spaces(_random_spatial(2), m=8, N=6, L=1, n_out=8)


@pytest.fixture(scope="module")
def ensemble():
    data_cfg = DataConfig(task="linear_teacher", P=20, D_in=8, n_latents=3, seed=1)
    train = make_dataset(data_cfg, CPU, F64)
    net = make_net(NetConfig(N=6, L=1, g=1.0, phi="tanh", seed=3, B=3), 8, 8, CPU, F64)
    return net, train


def _visible(fig):
    return [ax for ax in fig.axes if ax.get_visible() and ax.get_label() != "<colorbar>"]


# ---------------------------------------------------------------- the source notebook


def test_source_notebook_validates_has_no_outputs_and_no_path():
    nb = build_notebook()
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert all(not c.outputs and c.execution_count is None for c in code)
    assert "/ceph" not in _code(nb) and "/nfs" not in _code(nb)
    (parameters,) = [c for c in nb.cells if "parameters" in c.metadata.get("tags", [])]
    for name in ("ROOT", "SETTING", "MEMBER", "SORT_DIM", "P_TEST", "N_EIG", "N_NEURONS", "DEVICE"):
        assert f"\n{name} = " in "\n" + parameters.source, name


def test_the_committed_notebook_is_what_the_builder_produces():
    path = Path(__file__).parents[1] / "notebooks" / "phase_transition_grid.ipynb"
    committed = nbformat.read(path, 4)
    assert [(c.cell_type, c.source) for c in committed.cells] == [
        (c.cell_type, c.source) for c in build_notebook().cells
    ]


def test_every_figure_cell_is_introduced_by_an_equation():
    nb = build_notebook()
    drawing = [
        i
        for i, c in enumerate(nb.cells)
        if c.cell_type == "code" and ("plt.subplots(" in c.source or "plt.figure(" in c.source)
    ]
    assert drawing
    for i in drawing:
        assert "$" in nb.cells[i - 1].source, nb.cells[i].source[:200]


# ---------------------------------------------------------------- change 3 and 4: dropped figures


def test_the_u_weight_plot_and_the_trial_and_pair_sections_are_gone():
    nb = build_notebook()
    code = _code(nb)
    text = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    for name in ("plot_weights_by_position", "plot_trial_latents", "plot_spatial_kernel_pairs"):
        assert name not in code
    for phrase in (
        "First-layer weights $U$, sorted by position",
        "trials whose kernels are drawn",
        "between pairs of trials",
    ):
        assert phrase not in text


# ---------------------------------------------------------------- change 1: the prior matrices


def test_prior_figure_shows_covariance_effective_covariance_and_penalty_side_by_side(draw, spaces):
    prior = layer_priors(spaces, 4.0, n_eig=3)[0]
    fig = draw["plot_prior_matrices"](prior, spaces[0], np.arange(6), np.arange(8), 1)
    axes = _visible(fig)
    assert len(axes) == 6  # pre and post of three matrices
    titles = [ax.get_title() for ax in axes]
    assert sum("covariance" in t and "effective" not in t for t in titles) == 2
    assert sum("effective covariance" in t for t in titles) == 2
    assert sum("penalty" in t for t in titles) == 2
    assert all("lambda_w" in t for t in titles)


def test_prior_matrices_are_drawn_in_the_given_unit_order(draw, spaces):
    prior = layer_priors(spaces, 1.0, n_eig=3)[0]
    order_post, order_pre = np.array([5, 4, 3, 2, 1, 0]), np.arange(8)[::-1].copy()
    fig = draw["plot_prior_matrices"](prior, spaces[0], order_post, order_pre, 0)
    first_row = [ax for ax in _visible(fig) if "covariance" in ax.get_title()][0]
    shown = first_row.images[0].get_array()
    expected = prior.covariance.pre.numpy()[np.ix_(order_pre, order_pre)]
    assert np.allclose(np.asarray(shown), expected)
    assert "coordinate 0" in first_row.get_xlabel()


# ---------------------------------------------------------------- change 2: spectra and the floor


@pytest.mark.parametrize("n_eig", [2, 4])
def test_spectra_are_truncated_to_n_eig_and_the_floor_is_drawn(draw, spaces, n_eig):
    prior = layer_priors(spaces, 4.0, n_eig=n_eig)[0]
    fig = draw["plot_prior_spectra"](prior, spaces[0])
    axes = _visible(fig)
    assert len(axes) == 4
    for ax in axes:
        # the spectrum curves have n_eig points; the floor line is a two-point axhline
        spectra = [line for line in ax.lines if len(line.get_xdata()) == n_eig]
        assert len(spectra) >= 1 and all(len(line.get_xdata()) in (2, n_eig) for line in ax.lines)
    # a horizontal line at the floor (covariance) and at the ceiling (penalty), per population
    covariance_axes, penalty_axes = axes[:2], axes[2:]
    floors = [prior.floors.pre.covariance_floor, prior.floors.post.covariance_floor]
    ceilings = [prior.floors.pre.penalty_ceiling, prior.floors.post.penalty_ceiling]
    for ax, floor in zip(covariance_axes, floors):
        assert any(np.allclose(np.unique(line.get_ydata()), floor) for line in ax.lines)
    for ax, ceiling in zip(penalty_axes, ceilings):
        assert any(np.allclose(np.unique(line.get_ydata()), ceiling) for line in ax.lines)


def test_the_penalty_spectrum_axis_says_it_starts_from_the_smallest_end(draw, spaces):
    prior = layer_priors(spaces, 1.0, n_eig=3)[0]
    axes = _visible(draw["plot_prior_spectra"](prior, spaces[0]))
    assert "smallest" in axes[2].get_xlabel() and "largest" in axes[0].get_xlabel()


# ---------------------------------------------------------------- change 7: loss mean and sem


def test_loss_curves_are_the_member_mean_with_a_sem_band(draw):
    epochs = np.array([0, 10, 20])
    members = np.array([[1.0, 3.0], [2.0, 4.0], [4.0, 8.0]])  # (epochs, B = 2)
    history = {"epoch": epochs, **{k: members * (i + 1) for i, k in enumerate(
        ("loss", "mse", "weight_l2", "rate_l2"))}}  # fmt: skip
    stats = ensemble_statistics(history)
    fig = draw["plot_ensemble_loss"](stats, "t")
    (ax,) = _visible(fig)
    assert len(ax.lines) == 4 and len(ax.collections) == 4  # a mean line and a band per term
    mean = members.mean(axis=1)
    assert np.allclose(ax.lines[0].get_ydata(), mean)
    band = ax.collections[0].get_paths()[0].vertices[:, 1]
    sem = members.std(axis=1, ddof=1) / np.sqrt(2)
    assert np.isclose(band.max(), (mean + sem).max()) and np.isclose(band.min(), (mean - sem).min())
    assert "mean" in ax.get_title() and "sem" in ax.get_title()


# ---------------------------------------------------------------- change 8: example outputs


def test_example_outputs_draw_every_member_in_its_own_palette_colour(draw, ensemble):
    net, train = ensemble
    picks = ensemble_example_outputs(net, train, [0, 3])
    everything = ensemble_example_outputs(net, train, list(range(len(train.x))))
    colours = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    fig = draw["plot_ensemble_outputs"](picks, everything, colours, "t")
    drawn = {
        mpl.colors.to_hex(line.get_color())
        for ax in fig.axes
        for line in ax.lines
        if line.get_marker() == "D"
    }
    assert drawn == {mpl.colors.to_hex(colours[b]) for b in range(3)}
    scatter = {
        mpl.colors.to_hex(c.get_facecolor()[0]) for ax in fig.axes for c in ax.collections
    }
    assert {mpl.colors.to_hex(colours[b]) for b in range(3)} <= scatter


def test_more_members_than_palette_colours_raises(draw, ensemble):
    net, train = ensemble
    outputs = ensemble_example_outputs(net, train, [0, 1])
    with pytest.raises(ValueError, match="palette"):
        draw["plot_ensemble_outputs"](outputs, outputs, ["C0", "C1"], "t")


# ---------------------------------------------------------------- change 9: weight kernels


def test_weight_kernels_show_the_prior_the_member_and_the_average(draw, spaces):
    prior = layer_priors(spaces, 1.0, n_eig=3)[0]
    pair = prior.covariance
    panels = [("prior", pair), ("member 1", pair), ("average", pair)]
    fig = draw["plot_weight_kernels"](panels, spaces[0], np.arange(6), np.arange(8), 0)
    axes = _visible(fig)
    assert len(axes) == 6
    titles = " ".join(ax.get_title() for ax in axes)
    assert "member 1" in titles and "average" in titles and "prior" in titles


# ---------------------------------------------------------------- change 10: tuning truncation


def _tuning_view(n_neurons, k=2, n_bins=4):
    rng = np.random.default_rng(0)
    from fct.grid_view import TuningView
    from fct.tuning import TuningCurves

    curves = TuningCurves(
        torch.tensor(rng.normal(size=(n_neurons, k, n_bins))),
        torch.tensor(np.abs(rng.normal(size=(n_neurons, k, n_bins)))),
        torch.tensor(rng.normal(size=(k, n_bins))),
        torch.full((k, n_bins), 10),
    )
    info = torch.tensor(np.abs(rng.normal(size=(n_neurons, k))))
    return TuningView(
        curves, rng.uniform(size=n_neurons), info, info, None, [f"z{f}" for f in range(k)]
    )


@pytest.mark.parametrize("shown", [3, 6])
def test_tuning_draws_only_the_neurons_it_is_given_in_that_order(draw, shown):
    view = _tuning_view(6)
    positions = torch.rand(6, 2, dtype=F64)
    order = unit_order(positions, 1)[:shown]
    fig = draw["plot_tuning_by_order"](view, positions, order, 1, 4, 0.8)
    titles = [ax.get_title() for ax in fig.axes if ax.get_visible() and ax.get_title()]
    assert len(titles) == shown
    assert [int(t.split()[1]) for t in titles] == order.tolist()
    assert all("coordinate 1" in t for t in titles)


# ---------------------------------------------------------------- change 11: unit order


def test_overlaps_are_shown_in_the_given_order_and_labelled_with_it(draw):
    C = torch.arange(16.0).reshape(4, 4)
    C = C + C.T
    order = np.array([2, 0, 3, 1])
    fig = draw["plot_overlaps"](C, C, order, float(C.max()), float(C.max()), "t", "neuron, by x")
    left = _visible(fig)[0]
    assert np.allclose(np.asarray(left.images[0].get_array()), C.numpy()[np.ix_(order, order)])
    assert left.get_xlabel() == "neuron, by x"


def test_the_text_says_a_sorted_coordinate_is_not_a_ring_position():
    nb = build_notebook()
    text = " ".join(c.source for c in nb.cells if c.cell_type == "markdown")
    assert "not a ring position" in text and "general distance measures come later" in text
    assert "position_order(" not in _code(nb)  # no view sorts by ring position
    assert "unit_order(" in _code(nb)


# ---------------------------------------------------------------- change 5 and 6 and 8: the text


def test_text_says_the_overlap_average_is_the_mean_of_the_members_overlaps():
    nb = build_notebook()
    text = " ".join(c.source for c in nb.cells if c.cell_type == "markdown")
    assert "mean over members of each member's own overlap" in text


def test_text_names_the_palette_limit_of_the_example_outputs():
    nb = build_notebook()
    text = " ".join(c.source for c in nb.cells if c.cell_type == "markdown")
    assert "at most as many members as the palette has colours" in text


# ---------------------------------------------------------------- execution


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    """phase_transition tiny (spatial.d in {2, 3}, B = 2) with only d = 2 trained."""
    root = tmp_path_factory.mktemp("phase_nb")
    sweep = _tiny_sweep()
    save_sweep(sweep, root)
    run_setting(sweep, 0, root, log=lambda _: None)
    return root


def _execute(root: Path, setting: dict, **parameters):
    overrides = {"SETTING": setting, **{**PARAMETERS, **parameters}}
    return execute_notebook(build_notebook(), root, overrides, root_name="ROOT")


@pytest.fixture(scope="module")
def trained(root):
    return _execute(root, {"spatial.d": 2})


def test_executes_on_a_trained_setting_and_draws_every_figure(trained):
    text = _text_outputs(trained)
    assert "ERROR" not in text
    assert _n_figures_in(trained) == N_FIGURES
    assert "| 0 | 2 |" in text and "| 1 | 3 |" in text  # the grid
    assert "done" in text and "missing" in text
    for i, cell in enumerate(trained.cells):
        if any("image/png" in o.get("data", {}) for o in cell.get("outputs", [])):
            assert "$" in trained.cells[i - 1].source  # an equation introduces every figure


def test_tuning_and_mir_use_a_fresh_random_test_set_of_p_test_points(trained):
    text = _text_outputs(trained)
    assert "random test set: 40 points" in text
    assert "lattice test set: 9 points" in text  # A keeps the latent lattice, 3^k with k = 2
    assert "P = 40," in text  # the plug-in bias line of the tuning section uses P_TEST


def test_tuning_shows_the_first_n_neurons_in_sort_dim_order(trained):
    text = _text_outputs(trained)
    assert "first 4 of 6 neurons" in text and "coordinate 1 of d = 2" in text


def test_spectra_report_n_eig_of_each_population(trained):
    assert "3 of 6 eigenvalues" in _text_outputs(trained)


def test_a_and_overlap_figures_come_for_the_member_and_for_the_average(trained):
    sources = [c.source for c in trained.cells if c.cell_type == "code"]
    overlap = [s for s in sources if "plot_overlaps(" in s and "MEAN_VIEW" in s]
    kernels = [s for s in sources if "plot_data_kernels(" in s and "MEAN_VIEW" in s]
    assert overlap and kernels
    assert all("MEMBER_VIEW" in s for s in overlap + kernels)
    # member + average: two figures per hidden layer for the overlaps, two for A
    per_cell = {
        s: sum(
            "image/png" in o.get("data", {}) for o in c.get("outputs", [])
        )
        for c in trained.cells
        if c.cell_type == "code"
        for s in [c.source]
    }
    assert per_cell[overlap[0]] == 2 and per_cell[kernels[0]] == 2


def test_a_setting_that_is_not_trained_is_reported_and_only_its_prior_is_drawn(root):
    nb = _execute(root, {"spatial.d": 3}, SORT_DIM=2)
    text = _text_outputs(nb)
    assert "ERROR" not in text
    assert "is not trained** (status: missing)" in text
    assert text.count("is not trained") == GUARDED_CELLS
    assert _n_figures_in(nb) == N_PRIOR_FIGURES  # prior matrices and spectra need no net
    assert "SGLD needs < 1" in text


def test_sort_dim_outside_the_embedding_fails_loudly_in_the_notebook(root):
    with pytest.raises(CellExecutionError, match="sort_dim"):
        _execute(root, {"spatial.d": 2}, SORT_DIM=2)
    with pytest.raises(CellExecutionError, match="sort_dim"):
        _execute(root, {"spatial.d": 3}, SORT_DIM=3)  # also for a setting never run


def test_a_member_outside_the_ensemble_fails_loudly(root):
    with pytest.raises(CellExecutionError, match="MEMBER"):
        _execute(root, {"spatial.d": 2}, MEMBER=2)


def test_a_setting_outside_the_grid_fails_loudly(root):
    with pytest.raises(CellExecutionError, match="spatial.d"):
        _execute(root, {"spatial.d": 5})


def test_a_partly_trained_setting_is_drawn_and_labelled_provisional(root, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    (setting_dir(copy, expand(_tiny_sweep())[0]) / "train" / "final.pt").unlink()
    nb = _execute(copy, {"spatial.d": 2})
    text = _text_outputs(nb)
    assert "ERROR" not in text and "provisional" in text
    assert _n_figures_in(nb) == N_FIGURES
