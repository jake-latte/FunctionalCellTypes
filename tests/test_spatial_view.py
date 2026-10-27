"""Layout and wiring for the spatial-prior notebook, on synthetic inputs with known structure:
a known permutation of ring positions gives a known panel layout and weight ordering, and the
empirical kernel of weights drawn from a known covariance recovers it. The figures built on them
are drawn in the notebook and tested in test_build_spatial.py."""

import pytest
import torch

from fct.cache import CACHE_ROOT_ENV, cache_root_from_env
from fct.config import Config, DataConfig, NetConfig, TrainConfig
from fct.spatial import KernelSpec, SpatialConfig, build_spaces
from fct.spatial_view import (
    cache_dir_for,
    empirical_kernels,
    make_config,
    order_weights,
    ordered_kernel_comparison,
    panel_grid,
    population_spec,
    position_order,
    train_run,
)

# Ring positions of 6 units in a scrambled order: unit 0 is 3rd around the ring, and so on.
RING = torch.tensor([[0.50], [0.00], [0.75], [0.25], [0.90], [0.10]], dtype=torch.float64)
RING_ORDER = [1, 5, 3, 0, 2, 4]  # units sorted by position


def test_position_order_sorts_a_ring_by_angle():
    assert position_order(RING).tolist() == RING_ORDER


def test_position_order_refuses_more_than_one_dimension():
    with pytest.raises(ValueError, match="d = 1"):
        position_order(torch.zeros(4, 2))


def test_order_weights_applies_the_known_permutation_to_rows_and_columns():
    post_pos, pre_pos = RING, RING[[2, 0, 1]]  # 3 pre units at 0.75, 0.50, 0.00
    W = torch.arange(18, dtype=torch.float64).reshape(6, 3)
    ordered = order_weights(W, post_pos, pre_pos)
    assert ordered.post_order.tolist() == RING_ORDER
    assert ordered.pre_order.tolist() == [2, 1, 0]
    assert torch.equal(ordered.W, W[RING_ORDER][:, [2, 1, 0]])
    # entry (post unit 4, pre unit 0) keeps its value wherever it lands
    row, col = RING_ORDER.index(4), [2, 1, 0].index(0)
    assert ordered.W[row, col] == W[4, 0]


def test_order_weights_checks_shapes():
    with pytest.raises(ValueError, match="rows"):
        order_weights(torch.zeros(5, 3), RING, RING[:3])


def test_panel_grid_for_a_ring_fills_wrapped_rows_in_ring_order():
    grid = panel_grid(RING, n_cols=4)
    assert (grid.n_rows, grid.n_cols) == (2, 4)
    # the unit at ring rank r sits in cell (r // 4, r % 4)
    expected = {unit: (rank // 4, rank % 4) for rank, unit in enumerate(RING_ORDER)}
    assert dict(enumerate(grid.cells)) == expected


def test_panel_grid_for_a_lattice_puts_each_unit_at_its_coordinates():
    # 2 x 3 lattice, units listed in a scrambled order; axis 0 is x (columns), axis 1 is y (up)
    xy = torch.tensor(
        [[1 / 3, 0.5], [0.0, 0.0], [2 / 3, 0.0], [0.0, 0.5], [1 / 3, 0.0], [2 / 3, 0.5]],
        dtype=torch.float64,
    )
    grid = panel_grid(xy, n_cols=99)  # n_cols only applies to a ring
    assert (grid.n_rows, grid.n_cols) == (2, 3)
    # y = 0.5 is the top row (row 0), y = 0 the bottom row
    assert grid.cells == [(0, 1), (1, 0), (1, 2), (0, 0), (1, 1), (0, 2)]


def test_panel_grid_refuses_scattered_two_dimensional_positions():
    xy = torch.rand(7, 2, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    with pytest.raises(ValueError, match="lattice"):
        panel_grid(xy, n_cols=4)


def test_panel_grid_refuses_three_dimensions():
    with pytest.raises(ValueError, match="d <= 2"):
        panel_grid(torch.zeros(4, 3), n_cols=2)


def test_empirical_kernels_recover_a_known_covariance():
    # the columns of W are iid N(0, C): W = A Z with Z iid and A A^T = C
    gen = torch.Generator().manual_seed(1)
    C = torch.tensor([[1.0, 0.8], [0.8, 1.0]], dtype=torch.float64)
    A = torch.linalg.cholesky(C)
    W = A @ torch.randn(2, 4_000, generator=gen, dtype=torch.float64)
    post, _ = empirical_kernels(W)
    assert torch.allclose(post, C, atol=0.08)


def test_empirical_kernels_are_mean_products_over_the_other_axis():
    W = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, -1.0]], dtype=torch.float64)
    post, pre = empirical_kernels(W)
    assert torch.allclose(post, W @ W.T / 3)
    assert torch.allclose(pre, W.T @ W / 2)


def test_ordered_kernel_comparison_orders_theory_and_data_alike():
    cfg = SpatialConfig(
        input=population_spec("random", 0.2, seed=3), hidden=population_spec("random", 0.2, seed=4)
    )
    space = build_spaces(cfg, m=5, N=7, L=1, n_out=2)[0]
    W = torch.randn(7, 5, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    cmp = ordered_kernel_comparison(space, W)
    post_order = position_order(space.post.embedding.positions)
    assert torch.equal(cmp.post_theory, space.post.K[post_order][:, post_order])
    ordered = W[post_order]
    assert torch.allclose(cmp.post_empirical, ordered @ ordered.T / 5)
    # a stationary kernel on the ring decays away from the diagonal once sorted by position
    assert cmp.post_theory[0, 0] > cmp.post_theory[0, 3]


# ---------------------------------------------------------------- configuration and caching


def _tiny(**overrides) -> Config:
    kwargs = {
        "P": 16, "m": 6, "k": 2, "N": 8, "L": 1, "phi": "tanh", "g": 1.0,
        "data_seed": 0, "net_seed": 0, "weight_lambda": 1e-2, "rate_lambda": 1e-3,
        "optimizer": "adam", "beta": 1.0, "lr": 1e-2, "n_epochs": 20, "train_seed": 0,
        "spatial": SpatialConfig(), "device": "cpu",
    }  # fmt: skip
    return make_config(**{**kwargs, **overrides})


def test_make_config_maps_the_notebook_names_onto_the_config():
    cfg = _tiny(optimizer="sgld", beta=5.0, n_epochs=40)
    assert cfg.data == DataConfig(task="linear_teacher", P=16, D_in=6, n_latents=2, seed=0)
    assert cfg.net == NetConfig(N=8, L=1, g=1.0, phi="tanh", seed=0)
    assert isinstance(cfg.train, TrainConfig)
    assert (cfg.train.optimizer, cfg.train.beta, cfg.train.n_epochs) == ("sgld", 5.0, 40)
    assert cfg.train.save_int <= cfg.train.n_epochs and cfg.spatial is not None


def test_make_config_rejects_an_invalid_combination():
    with pytest.raises(ValueError, match="D_in"):
        _tiny(m=2, k=2)  # the teacher needs more features than latents


def test_cache_dir_depends_on_the_training_setup_and_the_root(tmp_path):
    a, b = cache_dir_for(_tiny(), tmp_path), cache_dir_for(_tiny(optimizer="sgld"), tmp_path)
    assert a != b and a.parent == tmp_path == b.parent
    assert cache_dir_for(_tiny(), tmp_path) == a
    other = SpatialConfig(hidden=population_spec("grid", 0.3))
    assert cache_dir_for(_tiny(spatial=other), tmp_path) != a


def test_population_spec_builds_a_kernel_and_an_embedding():
    spec = population_spec("random", 0.2, amplitude=2.0, seed=5)
    assert spec.embedding.kind == "random" and spec.embedding.seed == 5
    assert spec.kernel == KernelSpec(length_scale=0.2, amplitude=2.0)
    assert population_spec("grid", 0.1, shape=[2, 3]).embedding.shape == [2, 3]


@pytest.mark.parametrize("optimizer", ["adam", "sgld"])
def test_train_run_trains_then_reloads_from_the_cache(tmp_path, optimizer):
    cfg = _tiny(optimizer=optimizer, spatial=SpatialConfig(min_rel_eig=1e-2), lr=1e-3)
    lines: list[str] = []
    first = train_run(cfg, tmp_path, log=lines.append)
    assert (first.cache_dir / "config.json").exists() and lines
    lines.clear()
    again = train_run(cfg, tmp_path, log=lines.append)
    assert lines == []  # final.pt exists: nothing is trained or logged
    assert torch.equal(again.net.U, first.net.U)
    assert again.history["epoch"] == first.history["epoch"]


def test_cache_root_comes_from_the_environment_or_fails(monkeypatch, tmp_path):
    monkeypatch.delenv(CACHE_ROOT_ENV, raising=False)
    with pytest.raises(RuntimeError, match=CACHE_ROOT_ENV):
        cache_root_from_env()
    monkeypatch.setenv(CACHE_ROOT_ENV, str(tmp_path))
    assert cache_root_from_env() == tmp_path
