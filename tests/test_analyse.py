import math

import pytest
import torch

from fct import analyse
from fct.analyse import (
    GridPredictions,
    LayerPCA,
    get_grid_predictions,
    get_layer_pca,
    grid_predictions,
    kernel_scales,
    layer_pca,
    neuron_maps,
)
from fct.cache import StaleCacheError, cache_paths
from fct.config import AnalysisConfig, Config, DataConfig, NetConfig, TrainConfig
from fct.data import grid_targets, make_dataset
from fct.model import make_net

CPU, F64 = torch.device("cpu"), torch.float64
N_NET, D_IN, P_TRAIN, N_GRID = 6, 8, 20, 5


def _cfg(n_grid: int = N_GRID, embedding: str = "linear", L: int = 1) -> Config:
    return Config(
        data=DataConfig(P=P_TRAIN, D_in=D_IN, d=2, var=2.0, embedding=embedding),
        net=NetConfig(N=N_NET, L=L, phi="relu"),
        analysis=AnalysisConfig(n_grid=n_grid, n_components=3),
        device="cpu",
        dtype="float64",
    )


def _identity_net(cfg: Config, dataset):
    """Net with Z == y exactly: H0[:, :2] = y (U = sqrt(D_in) Q^T) and V[:2] = N * I.

    The readout acts on the pre-activation H, so no nonlinearity gets in the way.
    """
    net = make_net(cfg.net, D_IN, 2, CPU, F64)
    with torch.no_grad():
        net.U.zero_()
        net.U[:2] = math.sqrt(D_IN) * dataset.Q.T
        net.V.zero_()
        net.V[:2] = N_NET * torch.eye(2, dtype=F64)
    return net


def test_identity_net_has_zero_error():
    cfg = _cfg()
    ds = make_dataset(cfg.data, CPU, F64)
    net = _identity_net(cfg, ds)
    g = grid_predictions(net, ds, cfg.data, N_GRID)
    assert g.test_error.shape == (N_GRID, N_GRID)
    assert g.train_error.shape == (P_TRAIN,)
    assert g.h_grid.shape == (N_GRID**2, 1, N_NET)
    assert g.y_grid.shape == g.z_grid.shape == (N_GRID**2, 2)
    assert g.test_error.max() < 1e-20
    assert g.train_error.max() < 1e-20
    assert g.half_width == cfg.data.var / 2
    assert g.y_grid.min() == -g.half_width and g.y_grid.max() == g.half_width
    assert not g.h_grid.requires_grad and not g.z_grid.requires_grad


def test_test_error_is_ij_ordered_mse():
    cfg = _cfg()
    ds = make_dataset(cfg.data, CPU, F64)
    net = make_net(cfg.net, D_IN, 2, CPU, F64)
    g = grid_predictions(net, ds, cfg.data, N_GRID)
    expected = (g.y_grid - g.z_grid).square().mean(dim=1)
    assert torch.equal(g.test_error.reshape(-1), expected)
    # 'ij' order: the first latent coordinate is constant along each row of the map.
    first_coord = g.y_grid[:, 0].reshape(N_GRID, N_GRID)
    assert torch.equal(first_coord, first_coord[:, :1].expand(N_GRID, N_GRID))


def test_train_error_is_mean_squared_error_of_net_on_training_set():
    cfg = _cfg()
    ds = make_dataset(cfg.data, CPU, F64)
    net = make_net(cfg.net, D_IN, 2, CPU, F64)
    g = grid_predictions(net, ds, cfg.data, N_GRID)
    with torch.no_grad():
        _, z = net(ds.x)
    expected = ((ds.y - z) ** 2).sum(dim=1) / 2
    assert torch.allclose(g.train_error, expected)


def test_grid_predictions_with_fourier_embedding():
    cfg = _cfg(embedding="fourier")
    ds = make_dataset(cfg.data, CPU, F64)
    net = make_net(cfg.net, D_IN, 2, CPU, F64)
    g = grid_predictions(net, ds, cfg.data, N_GRID)
    assert g.test_error.shape == (N_GRID, N_GRID)
    assert torch.isfinite(g.h_grid).all() and torch.isfinite(g.test_error).all()
    assert torch.equal(g.y_grid, grid_targets(cfg.data.var / 2, N_GRID, CPU, F64))


def test_grid_predictions_requires_matching_output_dim():
    cfg = _cfg()
    ds = make_dataset(cfg.data, CPU, F64)
    net = make_net(cfg.net, D_IN, 3, CPU, F64)
    with pytest.raises(ValueError, match="D_out"):
        grid_predictions(net, ds, cfg.data, N_GRID)


def test_grid_predictions_requires_two_latent_dims():
    cfg = _cfg()
    ds = make_dataset(cfg.data, CPU, F64)
    net = make_net(cfg.net, D_IN, 2, CPU, F64)
    with pytest.raises(ValueError, match="d == 2"):
        grid_predictions(net, ds, cfg.data.model_copy(update={"d": 3}), N_GRID)


def _planted_layer(modes: torch.Tensor, stds: tuple[float, float], N: int, seed: int):
    """(p^2, N) activity: two orthogonal spatial modes with sample variances stds**2 + 1e-6 noise."""
    gen = torch.Generator().manual_seed(seed)
    amp = torch.randn(N, 2, generator=gen, dtype=F64)
    amp = amp - amp.mean(0)
    amp = torch.linalg.qr(amp)[0] * math.sqrt(N - 1)  # sample covariance exactly I
    x = (amp * torch.tensor(stds, dtype=F64)) @ modes  # (N, p^2)
    return (x + 1e-6 * torch.randn(N, modes.shape[1], generator=gen, dtype=F64)).T


def _planted_h_grid(p: int, N: int = 50, same_layers: bool = False):
    """Two layers with different planted modes and variances (4,1 -> 0.8,0.2 and 9,1 -> 0.9,0.1).

    Returns h_grid (p^2, 2, N), per-layer modes (2, 2, p^2) and expected ratios (2, 2).
    """
    gen = torch.Generator().manual_seed(0)
    q = torch.linalg.qr(torch.randn(p * p, 4, generator=gen, dtype=F64))[0].T  # (4, p^2)
    modes = torch.stack([q[:2], q[:2] if same_layers else q[2:]])
    stds = [(2.0, 1.0), (2.0, 1.0) if same_layers else (3.0, 1.0)]
    h = torch.stack([_planted_layer(modes[l], stds[l], N, l) for l in range(2)], dim=1)
    ratios = torch.tensor([[0.8, 0.2], [0.8, 0.2] if same_layers else [0.9, 0.1]], dtype=F64)
    return h, modes, ratios


def test_layer_pca_recovers_planted_modes_per_layer():
    p, k = 6, 4
    h, modes, ratios = _planted_h_grid(p)
    out = layer_pca(h, lambda t: t, k)
    assert out.explained_variance.shape == (2, k)
    assert out.components.shape == (2, k, p * p)
    assert out.similarity.shape == (2, k, 2, k)
    for l in range(2):
        ev = out.explained_variance[l]
        assert torch.allclose(ev[:2], ratios[l], atol=1e-3)
        assert ev[2:].max() < 1e-6
        cos = (out.components[l, :2] @ modes[l].T).abs()  # (2 found, 2 planted)
        assert torch.allclose(cos, torch.eye(2, dtype=F64), atol=1e-2)


def test_layer_pca_similarity_diagonal_is_inverse_grid_size():
    p, k = 6, 4
    h, _, _ = _planted_h_grid(p)
    sim = layer_pca(h, lambda t: t, k).similarity
    diag = torch.stack([sim[l, i, l, i] for l in range(2) for i in range(k)])
    assert torch.allclose(diag, torch.full_like(diag, 1 / p**2))


def test_layer_pca_similarity_between_different_planted_modes_is_zero():
    p = 6
    h, _, _ = _planted_h_grid(p)
    sim = layer_pca(h, lambda t: t, 2).similarity
    assert sim[0, :, 1, :].abs().max() < 1e-2 / p**2


def test_layer_pca_similarity_of_identical_layers_matches_within_layer_block():
    p = 6
    h, _, _ = _planted_h_grid(p, same_layers=True)
    sim = layer_pca(h, lambda t: t, 2).similarity
    assert torch.allclose(sim[0, :, 1, :], sim[0, :, 0, :], atol=1e-6)


def test_layer_pca_is_deterministic_at_realistic_size():
    gen = torch.Generator().manual_seed(3)
    h = torch.randn(25**2, 1, 100, generator=gen, dtype=F64)
    a, b = (layer_pca(h, lambda t: t, 10) for _ in range(2))
    assert torch.equal(a.components, b.components)


def test_layer_pca_keeps_input_dtype():
    h, _, _ = _planted_h_grid(5)
    out = layer_pca(h.float(), lambda t: t, 2)
    for t in (out.explained_variance, out.components, out.similarity):
        assert t.dtype == torch.float32


def test_layer_pca_rejects_too_many_components():
    h, _, _ = _planted_h_grid(4, N=10)
    with pytest.raises(ValueError, match="n_components"):
        layer_pca(h, lambda t: t, 11)


def test_neuron_maps_match_phi_of_h():
    gen = torch.Generator().manual_seed(1)
    h = torch.randn(N_GRID**2, 2, N_NET, generator=gen, dtype=F64)
    phi = torch.relu
    maps = neuron_maps(h, phi, 1, N_GRID)
    assert maps.shape == (N_NET, N_GRID, N_GRID)
    for k in range(N_NET):
        assert torch.equal(maps[k], phi(h[:, 1, k]).reshape(N_GRID, N_GRID))


def test_neuron_maps_rejects_wrong_grid_size():
    h = torch.zeros(N_GRID**2, 1, N_NET, dtype=F64)
    with pytest.raises(ValueError, match="n_grid"):
        neuron_maps(h, torch.relu, 0, N_GRID + 1)


@pytest.mark.parametrize("layer", [-1, 2])
def test_neuron_maps_rejects_out_of_range_layer(layer):
    h = torch.zeros(N_GRID**2, 2, N_NET, dtype=F64)
    with pytest.raises(ValueError, match="layer"):
        neuron_maps(h, torch.relu, layer, N_GRID)


def test_kernel_scales_of_identities_is_ones():
    Phi = torch.stack([torch.eye(7, dtype=F64)] * 3)
    assert torch.equal(kernel_scales(Phi), torch.ones(3, dtype=F64))


def test_kernel_scales_is_mean_diagonal():
    Phi = torch.stack([2 * torch.eye(4, dtype=F64), torch.ones(4, 4, dtype=F64)])
    assert torch.allclose(kernel_scales(Phi), torch.tensor([2.0, 1.0], dtype=F64))


def test_get_grid_predictions_computes_once(tmp_path, monkeypatch):
    cfg = _cfg()
    ds = make_dataset(cfg.data, CPU, F64)
    net = make_net(cfg.net, D_IN, 2, CPU, F64)
    first = get_grid_predictions(tmp_path, cfg, net, ds)
    assert (cache_paths(tmp_path).analysis / "grid.pt").exists()

    def boom(*a, **k):
        raise AssertionError("net evaluated on a cache hit")

    monkeypatch.setattr(analyse, "grid_predictions", boom)
    second = get_grid_predictions(tmp_path, cfg, net, ds)
    assert isinstance(second, GridPredictions)
    assert second.n_grid == N_GRID and second.half_width == first.half_width
    for name in ("y_grid", "h_grid", "z_grid", "test_error", "train_error"):
        assert torch.equal(getattr(first, name), getattr(second, name))


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c.model_copy(
            update={"analysis": AnalysisConfig(n_grid=N_GRID + 1, n_components=3)}
        ),
        lambda c: c.model_copy(update={"train": TrainConfig(lr=0.5)}),
        lambda c: c.model_copy(update={"net": NetConfig(N=N_NET + 1, L=1, phi="relu")}),
    ],
    ids=["analysis.n_grid", "train.lr", "net.N"],
)
def test_get_grid_predictions_stale_when_config_changes(tmp_path, change):
    cfg = _cfg()
    ds = make_dataset(cfg.data, CPU, F64)
    net = make_net(cfg.net, D_IN, 2, CPU, F64)
    get_grid_predictions(tmp_path, cfg, net, ds)
    with pytest.raises(StaleCacheError):
        get_grid_predictions(tmp_path, change(cfg), net, ds)


def test_get_grid_predictions_rejects_a_different_net(tmp_path):
    cfg = _cfg()
    ds = make_dataset(cfg.data, CPU, F64)
    net = make_net(cfg.net, D_IN, 2, CPU, F64)
    get_grid_predictions(tmp_path, cfg, net, ds)
    with torch.no_grad():
        net.U.mul_(1.5)  # as if training had moved the weights
    with pytest.raises(StaleCacheError, match="grid.pt"):
        get_grid_predictions(tmp_path, cfg, net, ds)


def test_get_layer_pca_computes_once(tmp_path, monkeypatch):
    cfg = _cfg(L=2)
    ds = make_dataset(cfg.data, CPU, F64)
    net = make_net(cfg.net, D_IN, 2, CPU, F64)
    grid = get_grid_predictions(tmp_path, cfg, net, ds)
    first = get_layer_pca(tmp_path, cfg, grid, net.phi)
    assert first.explained_variance.shape == (2, cfg.analysis.n_components)

    def boom(*a, **k):
        raise AssertionError("PCA refitted on a cache hit")

    monkeypatch.setattr(analyse, "layer_pca", boom)
    second = get_layer_pca(tmp_path, cfg, grid, net.phi)
    assert isinstance(second, LayerPCA)
    assert torch.equal(first.components, second.components)
    assert torch.equal(first.similarity, second.similarity)


def test_get_layer_pca_stale_when_n_grid_changes(tmp_path):
    cfg = _cfg(L=2)
    ds = make_dataset(cfg.data, CPU, F64)
    net = make_net(cfg.net, D_IN, 2, CPU, F64)
    grid = get_grid_predictions(tmp_path, cfg, net, ds)
    get_layer_pca(tmp_path, cfg, grid, net.phi)
    with pytest.raises(StaleCacheError):
        get_layer_pca(tmp_path, _cfg(n_grid=N_GRID + 1, L=2), grid, net.phi)


def test_get_layer_pca_rejects_phi_not_matching_config(tmp_path):
    cfg = _cfg(L=2)
    ds = make_dataset(cfg.data, CPU, F64)
    net = make_net(cfg.net, D_IN, 2, CPU, F64)
    grid = get_grid_predictions(tmp_path, cfg, net, ds)
    with pytest.raises(ValueError, match="phi"):
        get_layer_pca(tmp_path, cfg, grid, torch.tanh)
