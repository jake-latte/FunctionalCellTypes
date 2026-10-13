import itertools

import pytest
import torch

import fct.data as data_module
from fct.cache import StaleCacheError, cache_paths
from fct.config import Config, DataConfig
from fct.data import embed, fourier_embedding, get_dataset, grid_targets, make_dataset

CPU = torch.device("cpu")
F64 = torch.float64


def _cfg(**kw) -> DataConfig:
    return DataConfig(**{"P": 50, "D_in": 8, "d": 2, "var": 2.0, **kw})


@pytest.mark.parametrize("embedding", ["linear", "fourier"])
def test_make_dataset_is_deterministic_per_seed(embedding):
    a = make_dataset(_cfg(embedding=embedding, seed=1), CPU, F64)
    b = make_dataset(_cfg(embedding=embedding, seed=1), CPU, F64)
    c = make_dataset(_cfg(embedding=embedding, seed=2), CPU, F64)
    assert torch.equal(a.x, b.x) and torch.equal(a.y, b.y)
    assert not torch.equal(a.y, c.y)


def test_linear_dataset_shapes_and_orthonormal_Q():
    ds = make_dataset(_cfg(embedding="linear"), CPU, F64)
    assert ds.x.shape == (50, 8) and ds.y.shape == (50, 2)
    assert ds.period is None
    assert torch.allclose(ds.Q.T @ ds.Q, torch.eye(2, dtype=F64), atol=1e-12)
    assert torch.allclose(ds.x, ds.y @ ds.Q.T)


def test_fourier_dataset_has_no_Q_and_float_period():
    ds = make_dataset(_cfg(embedding="fourier"), CPU, F64)
    assert ds.x.shape == (50, 8) and ds.Q is None
    assert isinstance(ds.period, float)
    assert ds.period == pytest.approx((ds.y.max() - ds.y.min()).item())


def test_linear_seed_changes_Q():
    a = make_dataset(_cfg(embedding="linear", seed=1), CPU, F64)
    b = make_dataset(_cfg(embedding="linear", seed=2), CPU, F64)
    assert not torch.allclose(a.Q, b.Q)


def test_fourier_embedding_rejects_odd_D():
    with pytest.raises(ValueError, match="even"):
        fourier_embedding(torch.zeros(3, 2, dtype=F64), D=7, period=1.0)


def _first_nonzero_positive(k: torch.Tensor) -> torch.Tensor:
    first = k[torch.arange(len(k)), (k != 0).long().argmax(dim=1)]
    return first > 0


def test_fourier_embedding_pairs_unit_norm_and_wavevectors():
    y = torch.randn(20, 2, dtype=F64)
    D = 12
    feats, k = fourier_embedding(y, D=D, period=5.0)
    pairs = feats.reshape(20, D // 2, 2)
    assert torch.allclose(pairs.norm(dim=-1), torch.ones(20, D // 2, dtype=F64))
    assert k.shape == (D // 2, 2)
    assert (k != 0).any(dim=1).all()
    assert _first_nonzero_positive(k).all()

    # No omitted canonical lattice vector is shorter than the longest one kept.
    lattice = torch.tensor(list(itertools.product(range(-4, 5), repeat=2)))
    lattice = lattice[(lattice != 0).any(dim=1)]
    lattice = lattice[_first_nonzero_positive(lattice)]
    kept = {tuple(r) for r in k.tolist()}
    max_kept = (k**2).sum(dim=1).max().item()
    for row in lattice.tolist():
        if tuple(row) not in kept:
            assert sum(v * v for v in row) >= max_kept


@pytest.mark.parametrize("embedding", ["linear", "fourier"])
def test_embed_reproduces_dataset_x(embedding):
    cfg = _cfg(embedding=embedding)
    ds = make_dataset(cfg, CPU, F64)
    assert torch.allclose(embed(ds.y, ds, cfg), ds.x)


def test_grid_targets_matches_meshgrid():
    h, n = 1.5, 7
    g = grid_targets(h, n, CPU, F64)
    assert g.shape == (n * n, 2)
    dom = torch.linspace(-h, h, n, dtype=F64)
    mesh = torch.meshgrid(dom, dom, indexing="ij")
    assert torch.equal(g[:, 0].reshape(n, n), mesh[0])
    assert torch.equal(g[:, 1].reshape(n, n), mesh[1])
    assert g.min() == -h and g.max() == h


def test_sample_covariance_matches_var_times_cov():
    cov = [[1.0, 0.5], [0.5, 2.0]]
    cfg = _cfg(P=20_000, var=3.0, cov=cov, embedding="linear")
    y = make_dataset(cfg, CPU, F64).y
    target = 3.0 * torch.tensor(cov, dtype=F64)
    err = (torch.cov(y.T) - target).abs() / target.abs()
    assert err.max() < 0.1


def test_default_cov_is_identity_scaled_by_var():
    y = make_dataset(_cfg(P=20_000, var=4.0, embedding="linear"), CPU, F64).y
    expected = 4.0 * torch.eye(2, dtype=F64)
    assert torch.allclose(torch.cov(y.T), expected, rtol=0.1, atol=0.1)


def test_wrong_shape_cov_raises():
    with pytest.raises(ValueError, match="cov"):
        make_dataset(_cfg(cov=[[1.0, 0.0, 0.0]]), CPU, F64)


def _full_cfg(**kw) -> Config:
    return Config(data=_cfg(**kw), device="cpu", dtype="float64")


@pytest.mark.parametrize("embedding", ["linear", "fourier"])
def test_get_dataset_computes_then_loads(tmp_path, monkeypatch, embedding):
    cfg = _full_cfg(embedding=embedding)
    first = get_dataset(tmp_path, cfg)
    assert (cache_paths(tmp_path).data / "dataset.pt").exists()

    def boom(*args, **kwargs):
        raise AssertionError("regenerated despite cache")

    monkeypatch.setattr(data_module, "make_dataset", boom)
    second = get_dataset(tmp_path, cfg)
    assert torch.equal(first.x, second.x) and torch.equal(first.y, second.y)
    assert second.period == first.period
    assert (second.Q is None) == (first.Q is None)


def test_get_dataset_stale_after_config_change(tmp_path):
    get_dataset(tmp_path, _full_cfg(seed=0))
    with pytest.raises(StaleCacheError):
        get_dataset(tmp_path, _full_cfg(seed=1))
