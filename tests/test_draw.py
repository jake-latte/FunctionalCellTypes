"""Smoke and behaviour tests for fct.draw on tiny synthetic inputs (Agg backend, CPU)."""

import inspect
import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from fct import draw
from fct.analyse import GridPredictions, LayerPCA
from fct.config import DataConfig
from fct.data import Dataset

P_GRID, L, N, K, P_TRAIN = 4, 2, 3, 3, 6


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
    return Dataset(x=torch.randn(P_TRAIN, 8), y=y_train, Q=None, period=10.0)


def _solver_history(n_steps=3, with_net=True, as_lists=False):
    rng = np.random.default_rng(0)
    per_layer = [
        "inner_residual", "outer_grad_rel", "phihat_fp_resid_stein", "stein_vs_direct",
        "eta_phihat", "hmc_accept", "hmc_eps", "hmc_div", "hmc_mc_err", "hmc_rho1",
        "hmc_h2_ratio", "hmc_stein_mismatch", "norm_margin", "phi_lambda_min",
    ]  # fmt: skip
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
    return {
        "test_error": lambda ax: draw.plot_test_error(grid.test_error, grid.half_width, ax=ax),
        "train_error": lambda ax: draw.plot_train_error(dataset.y, grid.train_error, ax=ax),
        "pca_variance": lambda ax: draw.plot_pca_variance(pca.explained_variance[0], ax=ax),
        "pca_similarity": lambda ax: draw.plot_pca_similarity(
            pca.similarity[0, :, 1, :], 1.0, ax=ax
        ),
        "neuron_map": lambda ax: draw.plot_neuron_map(
            torch.rand(P_GRID, P_GRID), grid.half_width, ax=ax
        ),
        "training_history": lambda ax: draw.plot_training_history(_training_history(), ax=ax),
        "example_trial": lambda ax: draw.plot_example_trial(dataset, ax=ax),
        "kernel": lambda ax: draw.plot_kernel(torch.rand(P_TRAIN, P_TRAIN), ax=ax, title="k"),
    }


PLOT_NAMES = [
    "test_error", "train_error", "pca_variance", "pca_similarity", "neuron_map",
    "training_history", "example_trial", "kernel",
]  # fmt: skip


@pytest.mark.parametrize("name", PLOT_NAMES)
def test_plot_without_ax_makes_new_figure(name, grid, pca, dataset):
    before = set(plt.get_fignums())
    ax = _plot_calls(grid, pca, dataset)[name](None)
    assert isinstance(ax, plt.Axes)
    assert len(set(plt.get_fignums()) - before) == 1


@pytest.mark.parametrize("name", PLOT_NAMES)
def test_plot_with_ax_draws_into_it(name, grid, pca, dataset):
    _, ax = plt.subplots()
    before = plt.get_fignums()
    assert _plot_calls(grid, pca, dataset)[name](ax) is ax
    assert plt.get_fignums() == before


def test_test_error_survives_perfect_predictor():
    ax = draw.plot_test_error(torch.zeros(P_GRID, P_GRID), 5.0)
    assert ax.images[0].norm.vmax > ax.images[0].norm.vmin


def test_fig_learning(grid, y_train, tmp_path):
    fig = draw.fig_learning(grid, y_train)
    assert len(fig.axes) >= 2
    _assert_png(fig, tmp_path)


def test_fig_representation_axes_count(pca, tmp_path):
    fig = draw.fig_representation(pca)
    assert len(fig.axes) == L + L**2
    _assert_png(fig, tmp_path)


@pytest.mark.parametrize("n_neurons, n_cols", [(3, 2), (4, 2), (5, 8)])
def test_fig_neuron_maps_axes_and_surplus_off(n_neurons, n_cols, tmp_path):
    maps = torch.rand(n_neurons, P_GRID, P_GRID)
    fig = draw.fig_neuron_maps(maps, 5.0, n_cols, "Activity in layer 1/2")
    total = math.ceil(n_neurons / n_cols) * n_cols
    grid_axes = [a for a in fig.axes if a.get_label() != "<colorbar>"]
    assert len(grid_axes) == total
    assert sum(not a.axison for a in grid_axes) == total - n_neurons
    _assert_png(fig, tmp_path)


def test_fig_example(dataset, tmp_path):
    fig = draw.fig_example(dataset, DataConfig(D_in=8, P=P_TRAIN))
    _assert_png(fig, tmp_path)


def test_fig_example_rejects_non_2d_targets():
    ds = Dataset(x=torch.randn(P_TRAIN, 8), y=torch.randn(P_TRAIN, 3), Q=None, period=1.0)
    with pytest.raises(ValueError, match="2"):
        draw.fig_example(ds, DataConfig(D_in=8, P=P_TRAIN, d=3))


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


def test_fig_kernels_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        draw.fig_kernels(torch.rand(L, 3, 3), torch.rand(L, 4, 4))


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


def test_fig_solver_history_skips_absent_keys():
    h = {k: v for k, v in _solver_history().items() if k != "hmc_eps"}
    fig = draw.fig_solver_history(h)
    assert not any("HMC step size" in a.get_title() for a in fig.axes)


def test_show_is_never_called():
    assert ".show(" not in inspect.getsource(draw)
