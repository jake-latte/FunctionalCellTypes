"""Smoke and behaviour tests for fct.draw on tiny synthetic inputs (Agg backend, CPU)."""

import inspect
import math

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from fct import draw
from fct.analyse import ExampleOutputs, GridPredictions, LayerPCA
from fct.config import DataConfig
from fct.data import Dataset
from fct.grid_report import MirRecord
from fct.tuning import tuning_curves

P_GRID, L, N, K, P_TRAIN = 4, 2, 3, 3, 6


def _rand(*shape, seed=0):
    """Seeded uniform tensor so every test run draws the same pictures."""
    return torch.rand(*shape, generator=torch.Generator().manual_seed(seed))


def _randn(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _has_artist(ax):
    return bool(ax.images or ax.collections or ax.lines or ax.patches)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def y_train():
    return torch.randn(P_TRAIN, 2, generator=torch.Generator().manual_seed(0))


@pytest.fixture
def grid():
    g = torch.Generator().manual_seed(1)
    p2 = P_GRID**2
    return GridPredictions(
        half_width=5.0,
        n_grid=P_GRID,
        y_grid=torch.randn(p2, 2, generator=g),
        h_grid=torch.randn(p2, L, N, generator=g),
        z_grid=torch.randn(p2, 2, generator=g),
        test_error=torch.rand(P_GRID, P_GRID, generator=g),
        train_error=torch.rand(P_TRAIN, generator=g),
        net_weight_l2=1.0,
    )


@pytest.fixture
def pca():
    g = torch.Generator().manual_seed(2)
    return LayerPCA(
        explained_variance=torch.rand(L, K, generator=g),
        components=torch.randn(L, K, P_GRID**2, generator=g),
        similarity=torch.randn(L, K, L, K, generator=g),
    )


@pytest.fixture
def dataset(y_train):
    return Dataset(x=_randn(P_TRAIN, 8), y=y_train, Q=None, period=10.0)


def _solver_history(n_steps=3, with_net=True, as_lists=False):
    rng = np.random.default_rng(0)
    per_layer = [
        "inner_residual",
        "outer_grad_rel",
        "phihat_fp_resid_stein",
        "stein_vs_direct",
        "eta_phihat",
        "hmc_accept",
        "hmc_eps",
        "hmc_div",
        "hmc_mc_err",
        "hmc_rho1",
        "hmc_h2_ratio",
        "hmc_stein_mismatch",
        "norm_margin",
        "phi_lambda_min",
    ]
    h = {"outer_step": np.arange(n_steps), "eta_phi": rng.random(n_steps) + 0.1}
    h |= {k: rng.random((n_steps, L)) + 0.1 for k in per_layer}
    h["train_mse_theory"] = rng.random(n_steps) + 0.1
    h["train_mse_net"] = rng.random(n_steps) + 0.1 if with_net else [None] * n_steps
    if with_net:
        h["align_net"] = rng.random((n_steps, L))
    if as_lists:
        h = {k: np.asarray(v).tolist() if not isinstance(v, list) else v for k, v in h.items()}
    return h


def _training_history():
    n = np.arange(1, 6)
    return {
        "epoch": n.tolist(),
        "mse": (1.0 / n).tolist(),
        "weight_l2": n.tolist(),
        "rate_l2": n.tolist(),
        "loss": (2.0 / n).tolist(),
    }


def _assert_png(fig, tmp_path):
    out = tmp_path / "fig.png"
    fig.savefig(out)
    assert out.stat().st_size > 0


def _plot_calls(grid, pca, dataset):
    """name -> callable(ax); the dict keys are the parametrisation of the generic plot tests."""
    return {
        "test_error": lambda ax: draw.plot_test_error(grid.test_error, grid.half_width, ax=ax),
        "train_error": lambda ax: draw.plot_train_error(dataset.y, grid.train_error, ax=ax),
        "pca_variance": lambda ax: draw.plot_pca_variance(pca.explained_variance[0], ax=ax),
        "pca_similarity": lambda ax: draw.plot_pca_similarity(
            pca.similarity[0, :, 1, :], 1.0, ax=ax
        ),
        "neuron_map": lambda ax: draw.plot_neuron_map(
            _rand(P_GRID, P_GRID), grid.half_width, ax=ax
        ),
        "training_history": lambda ax: draw.plot_training_history(_training_history(), ax=ax),
        "example_trial": lambda ax: draw.plot_example_trial(dataset, ax=ax),
        "kernel": lambda ax: draw.plot_kernel(_rand(P_TRAIN, P_TRAIN), ax=ax, title="k"),
    }


PLOT_NAMES = list(_plot_calls(None, None, None).keys())


@pytest.mark.parametrize("name", PLOT_NAMES)
def test_plot_without_ax_makes_new_figure(name, grid, pca, dataset):
    before = set(plt.get_fignums())
    ax = _plot_calls(grid, pca, dataset)[name](None)
    assert isinstance(ax, plt.Axes)
    new = set(plt.get_fignums()) - before
    assert len(new) == 1
    assert ax.figure.number in new


@pytest.mark.parametrize("name", PLOT_NAMES)
def test_plot_with_ax_draws_into_it(name, grid, pca, dataset):
    _, ax = plt.subplots()
    before = plt.get_fignums()
    assert _plot_calls(grid, pca, dataset)[name](ax) is ax
    assert plt.get_fignums() == before
    assert _has_artist(ax)


def test_test_error_survives_perfect_predictor():
    ax = draw.plot_test_error(torch.zeros(P_GRID, P_GRID), 5.0)
    im = ax.images[0]
    assert im.norm.vmax > im.norm.vmin
    # zeros sit at the bottom of the log scale instead of being masked (invisible)
    assert np.ma.count_masked(im.norm(im.get_array())) == 0


def test_train_error_shows_zero_error_points():
    ax = draw.plot_train_error(_randn(P_TRAIN, 2), torch.zeros(P_TRAIN))
    sc = ax.collections[0]
    assert np.ma.count_masked(sc.norm(sc.get_array())) == 0


@pytest.mark.parametrize("name", ["test_error", "neuron_map"])
def test_image_panels_put_coordinate_zero_on_horizontal_axis(name):
    hw = 2.0
    first_coord = torch.arange(1, P_GRID + 1, dtype=torch.float64)[:, None].expand(P_GRID, P_GRID)
    plot = {"test_error": draw.plot_test_error, "neuron_map": draw.plot_neuron_map}[name]
    im = plot(first_coord, hw).images[0]
    arr = np.asarray(im.get_array())
    assert np.all(np.diff(arr[0]) > 0)  # varies along columns (horizontal)
    assert np.all(arr == arr[0])  # constant down the rows
    assert list(im.get_extent()) == [-hw, hw, -hw, hw]
    assert im.origin == "lower"


def test_fig_learning(grid, y_train, tmp_path):
    fig = draw.fig_learning(grid, y_train)
    ax_test, ax_train = fig.axes[0], fig.axes[1]
    assert ax_test.images
    assert ax_train.collections
    _assert_png(fig, tmp_path)


def test_fig_representation_axes_count(pca, tmp_path):
    fig = draw.fig_representation(pca)
    assert len(fig.axes) == L + L**2
    _assert_png(fig, tmp_path)


@pytest.mark.parametrize("n_neurons, n_cols", [(3, 2), (4, 2), (5, 8)])
def test_fig_neuron_maps_axes_and_surplus_off(n_neurons, n_cols, tmp_path):
    maps = _rand(n_neurons, P_GRID, P_GRID)
    fig = draw.fig_neuron_maps(maps, 5.0, n_cols, "Activity in layer 1/2")
    total = math.ceil(n_neurons / n_cols) * n_cols
    grid_axes = [a for a in fig.axes if a.get_label() != "<colorbar>"]
    assert len(grid_axes) == total
    assert sum(not a.axison for a in grid_axes) == total - n_neurons
    _assert_png(fig, tmp_path)


# The overlap depends on the number of rows, not N: the 24 pt title is fixed in points while
# tight_layout's pad is a fraction of the figure height, so tall figures (13 rows, as for
# N=100 in 8 columns) overlap and short ones (1 row) do not.
@pytest.mark.parametrize("n_neurons, n_cols", [(13, 1), (8, 8)])
def test_fig_neuron_maps_title_clear_of_panels(n_neurons, n_cols):
    fig = draw.fig_neuron_maps(_rand(n_neurons, P_GRID, P_GRID), 5.0, n_cols, "Activity in layer 1")
    fig.canvas.draw()
    (title,) = fig.texts
    title_box = title.get_window_extent()
    for ax in fig.axes:
        assert not title_box.overlaps(ax.get_window_extent()), ax.get_label()


def test_fig_example(dataset, tmp_path):
    cfg = DataConfig(D_in=8, P=P_TRAIN)
    fig = draw.fig_example(dataset, cfg)
    ax_y, ax_x = fig.axes[0], fig.axes[1]
    assert ax_y.collections
    assert ax_x.images
    assert cfg.embedding in ax_x.get_title()
    _assert_png(fig, tmp_path)


def test_fig_example_rejects_non_2d_targets_before_creating_a_figure():
    ds = Dataset(x=_randn(P_TRAIN, 8), y=_randn(P_TRAIN, 3), Q=None, period=1.0)
    with pytest.raises(ValueError, match="2"):
        draw.fig_example(ds, DataConfig(D_in=8, P=P_TRAIN, d=3))
    assert plt.get_fignums() == []


def test_fig_kernels(tmp_path):
    g = torch.Generator().manual_seed(3)
    theory = torch.rand(L, P_TRAIN, P_TRAIN, generator=g)
    net = 5 * torch.rand(L, P_TRAIN, P_TRAIN, generator=g)
    fig = draw.fig_kernels(theory, net)
    images = [a.images[0] for a in fig.axes if a.images]
    assert len(images) == 2 * L
    for row in range(L):  # shared colour scale within a row
        assert images[2 * row].get_clim() == images[2 * row + 1].get_clim()
    _assert_png(fig, tmp_path)


def _kernel_stacks():
    g = torch.Generator().manual_seed(3)
    theory = torch.rand(L, P_TRAIN, P_TRAIN, generator=g)
    net = 1000 * torch.rand(L, P_TRAIN, P_TRAIN, generator=g)
    return theory, net


@pytest.mark.parametrize("shared_scale", [True, False])
def test_fig_kernels_scales_and_titles(shared_scale):
    theory, net = _kernel_stacks()
    # The default must be the shared scale, so only the False case passes the keyword.
    fig = (
        draw.fig_kernels(theory, net)
        if shared_scale
        else draw.fig_kernels(theory, net, shared_scale=False)
    )
    axes = [a for a in fig.axes if a.images]
    assert len(axes) == 2 * L
    assert len(fig.axes) == 2 * L * 2  # every panel keeps its own colorbar
    note = "shared scale" if shared_scale else "own scale"
    assert all(note in a.get_title() for a in axes)
    for row in range(L):
        pair = axes[2 * row : 2 * row + 2]
        clims = [a.images[0].get_clim() for a in pair]
        if shared_scale:
            assert clims[0] == clims[1]
        else:
            for clim, stack in zip(clims, (theory, net)):
                m = np.asarray(stack[row])
                assert clim == pytest.approx((m.min(), m.max()), rel=1e-6)


def test_fig_kernels_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        draw.fig_kernels(_rand(L, 3, 3), _rand(L, 4, 4))


def test_training_history_figure(tmp_path):
    ax = draw.plot_training_history(_training_history())
    _assert_png(ax.figure, tmp_path)


@pytest.mark.parametrize("as_lists", [False, True])
@pytest.mark.parametrize("with_net", [False, True])
def test_fig_solver_history_panels(with_net, as_lists, tmp_path):
    h = _solver_history(with_net=with_net, as_lists=as_lists)
    fig = draw.fig_solver_history(h)
    titles = [a.get_title() for a in fig.axes if a.axison]
    expected = [
        title for key, title, *_ in draw.SOLVER_PANELS if key in h and h[key][0] is not None
    ]
    assert titles == expected
    assert any("alignment" in t for t in titles) == with_net
    _assert_png(fig, tmp_path)


def test_fig_solver_history_accepts_literal_none_net_column(tmp_path):
    h = _solver_history(with_net=False)
    h["train_mse_net"] = None
    _assert_png(draw.fig_solver_history(h), tmp_path)


def test_fig_solver_history_rejects_scattered_none():
    h = _solver_history()
    h["train_mse_net"] = [0.1, None, 0.3]
    with pytest.raises(ValueError, match="train_mse_net"):
        draw.fig_solver_history(h)


def test_fig_solver_history_ignores_unrelated_columns(tmp_path):
    h = _solver_history()
    h["time"] = [None, 1.0, 2.0]  # not plotted, so its Nones are not our business
    _assert_png(draw.fig_solver_history(h), tmp_path)


def test_fig_solver_history_skips_absent_keys():
    h = {k: v for k, v in _solver_history().items() if k != "hmc_eps"}
    fig = draw.fig_solver_history(h)
    assert not any("HMC step size" in a.get_title() for a in fig.axes)


def test_show_is_never_called():
    assert ".show(" not in inspect.getsource(draw)


@pytest.mark.parametrize("title", ["", "a banner"])
def test_fig_example_outputs_title_puts_the_legend_below_the_axes(title):
    outputs = ExampleOutputs(
        indices=torch.arange(2),
        x=torch.zeros(2, 4),
        y_true=torch.randn(2, 4),
        y_pred=torch.randn(2, 4),
    )
    cfg = DataConfig(task="linear_teacher", P=8, D_in=4, n_latents=2, latent_range=1.0)
    fig = draw.fig_example_outputs("linear_teacher", outputs, cfg, title=title)
    fig.canvas.draw()
    legend = fig.legends[0].get_window_extent()
    axes_bottom = min(ax.get_window_extent().y0 for ax in fig.axes)
    axes_top = max(ax.get_window_extent().y1 for ax in fig.axes)
    assert fig.get_suptitle() == title
    assert (legend.y1 <= axes_bottom) if title else (legend.y0 >= axes_top)
    plt.close(fig)


# --------------------------------------------------------------------------- tuning and grid plots


@pytest.fixture
def tuning():
    """Tuning curves of 6 neurons to 3 latents and a per-neuron MIR for them."""
    factors = _rand(200, 3, seed=1)
    rates = _randn(200, 6, seed=2) + factors[:, :1]
    return tuning_curves(rates, factors, 5), np.linspace(0.1, 0.9, 6)


def _mir_records():
    return [MirRecord(lr, lw, layer, 0.5) for lr in (0.1, 1) for lw in (0.1, 1) for layer in (0, 1)]


def test_fig_tuning_has_a_panel_and_a_curve_per_latent_with_a_common_rate_axis(tuning, tmp_path):
    curves, mir = tuning
    fig = draw.fig_tuning(curves, [4, 0, 2], mir, ["a", "b", "c"])
    panels = [ax for ax in fig.axes if ax.get_visible()]
    assert len(panels) == 3
    assert "neuron 4" in panels[0].get_title() and "MIR 0.74" in panels[0].get_title()
    assert all(len(ax.containers) == 3 for ax in panels)
    assert panels[0].get_shared_y_axes().joined(panels[0], panels[2])
    _assert_png(fig, tmp_path)


def test_fig_tuning_latent_colours_differ_from_layer_colours(tuning):
    assert not {draw._latent_color(k) for k in range(6)} & set(draw.LAYER_COLORS)


@pytest.mark.parametrize(
    ("indices", "names", "mir_len", "match"),
    [
        ([0], ["a"], 6, "latent_names"),
        ([], ["a", "b", "c"], 6, "empty"),
        ([6], ["a", "b", "c"], 6, "must lie in"),
        ([0], ["a", "b", "c"], 5, "mir_per_neuron"),
    ],
)
def test_fig_tuning_rejects_bad_inputs(tuning, indices, names, mir_len, match):
    curves, mir = tuning
    with pytest.raises(ValueError, match=match):
        draw.fig_tuning(curves, indices, mir[:mir_len], names)


def test_fig_mir_matrix_is_a_phase_diagram_with_blank_nan_and_log_tick_labels(tmp_path):
    mir = np.random.default_rng(0).random((2, 3, 4))
    mir[1, 0, 0] = np.nan
    fig = draw.fig_mir_matrix([0.1, 1, 10], [0.01, 0.1, 1, 10], mir)
    heatmaps = [ax for ax in fig.axes if ax.images]
    assert len(heatmaps) == 2
    assert heatmaps[0].images[0].origin == "lower"  # lambda_r increases upwards
    assert [t.get_text() for t in heatmaps[0].get_xticklabels()] == [
        "$10^{-2}$",
        "$10^{-1}$",
        "$10^{0}$",
        "$10^{1}$",
    ]
    assert heatmaps[0].get_yticklabels()[0].get_text() == "$10^{-1}$"
    assert heatmaps[1].images[0].get_array().mask[0, 0]
    assert len({im.get_clim() for ax in heatmaps for im in ax.images}) == 1
    _assert_png(fig, tmp_path)


def test_fig_mir_matrix_rejects_mismatched_axes_and_non_positive_lambdas():
    with pytest.raises(ValueError, match="mir_by_layer"):
        draw.fig_mir_matrix([1, 2], [1, 2, 3], np.zeros((1, 3, 2)))
    with pytest.raises(ValueError, match="lambda_w_values"):
        draw.fig_mir_matrix([1, 2], [0, 2], np.zeros((1, 2, 2)))


def test_mir_heatmap_and_matrix_share_the_cell_drawing():
    ax = draw.plot_mir_matrix(np.full((2, 2), 0.5), [1, 10], [1, 10])
    assert ax.images[0].get_clim() == (0.0, 1.0)
    assert len(ax.texts) == 4  # each finite cell is annotated


def test_fig_mir_vs_ratio_has_a_log_axis_and_one_dot_per_network(tmp_path):
    records = _mir_records()
    fig = draw.fig_mir_vs_ratio(records)
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    assert sum(len(c.get_offsets()) for c in ax.collections) == len(records)
    assert ax.get_legend() is not None
    _assert_png(fig, tmp_path)


def test_fig_mir_vs_ratio_rejects_non_positive_lambdas_and_no_records():
    with pytest.raises(ValueError, match="lambda_r"):
        draw.fig_mir_vs_ratio([MirRecord(0.0, 1.0, 0, 0.5)])
    with pytest.raises(ValueError, match="no records"):
        draw.fig_mir_vs_ratio([])


def test_plot_mir_matrix_rejects_a_value_outside_the_colour_scale():
    with pytest.raises(ValueError, match="outside the colour range"):
        draw.plot_mir_matrix(np.full((2, 2), 1.5), [1, 10], [1, 10])


@pytest.mark.parametrize("neuron", [-1, 6])
def test_plot_neuron_tuning_rejects_a_neuron_out_of_range(tuning, neuron):
    with pytest.raises(ValueError, match="outside"):
        draw.plot_neuron_tuning(tuning[0], neuron, 0.5, ["a", "b", "c"])


def test_plot_mir_vs_ratio_validates_records_and_lists_only_layers_present():
    with pytest.raises(ValueError, match="non-finite"):
        draw.plot_mir_vs_ratio([MirRecord(1.0, 1.0, 0, float("nan"))])
    with pytest.raises(TypeError, match="integer"):
        draw.plot_mir_vs_ratio([MirRecord(1.0, 1.0, 1.0, 0.5)])
    ax = draw.plot_mir_vs_ratio([MirRecord(1.0, 1.0, 0, 0.5), MirRecord(1.0, 2.0, 2, 0.5)])
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["layer 1", "layer 3"]
