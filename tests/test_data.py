import dataclasses
import itertools
import math

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


def _lowest_wavevectors(M: int, d: int = 2, box: int = 12) -> set[tuple[int, ...]]:
    """Brute force: the M canonical lattice vectors of lowest |k|^2, ties broken
    lexicographically. Independent of the implementation (plain Python, huge box)."""
    vecs = [
        k
        for k in itertools.product(range(-box, box + 1), repeat=d)
        if any(k) and next(c for c in k if c != 0) > 0
    ]
    vecs.sort(key=lambda k: (sum(c * c for c in k), k))
    return set(vecs[:M])


@pytest.mark.parametrize("embedding", ["linear", "fourier"])
def test_make_dataset_is_deterministic_per_seed(embedding):
    a = make_dataset(_cfg(embedding=embedding, seed=1), CPU, F64)
    b = make_dataset(_cfg(embedding=embedding, seed=1), CPU, F64)
    c = make_dataset(_cfg(embedding=embedding, seed=2), CPU, F64)
    assert torch.equal(a.x, b.x) and torch.equal(a.y, b.y)
    assert not torch.equal(a.y, c.y)


def test_make_dataset_draws_from_cpu_generator():
    # Seeding a CPU generator is what makes the data identical on every device.
    cfg = _cfg(embedding="linear", seed=3, var=1.0)
    gen = torch.Generator().manual_seed(3)
    z = torch.randn((cfg.P, cfg.d), dtype=F64, generator=gen)
    assert torch.equal(make_dataset(cfg, CPU, F64).y, z)


def test_dataset_is_frozen():
    ds = make_dataset(_cfg(), CPU, F64)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ds.x = ds.y


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


def test_float32_dataset_and_grid_dtype():
    ds = make_dataset(_cfg(embedding="fourier"), CPU, torch.float32)
    assert ds.x.dtype == ds.y.dtype == torch.float32
    assert grid_targets(1.0, 3, CPU, torch.float32).dtype == torch.float32


def test_linear_seed_changes_Q():
    a = make_dataset(_cfg(embedding="linear", seed=1), CPU, F64)
    b = make_dataset(_cfg(embedding="linear", seed=2), CPU, F64)
    assert not torch.allclose(a.Q, b.Q)


def test_fourier_embedding_rejects_odd_D():
    with pytest.raises(ValueError, match="even"):
        fourier_embedding(torch.zeros(3, 2, dtype=F64), D=7, period=1.0)


@pytest.mark.parametrize("period", [0.0, -1.0, float("inf"), float("nan")])
def test_fourier_embedding_rejects_bad_period(period):
    with pytest.raises(ValueError, match="period"):
        fourier_embedding(torch.zeros(3, 2, dtype=F64), D=4, period=period)


def test_fourier_embedding_pairs_unit_norm():
    gen = torch.Generator().manual_seed(0)
    y = torch.randn(20, 2, dtype=F64, generator=gen)
    feats, _ = fourier_embedding(y, D=12, period=5.0)
    assert torch.allclose(feats.reshape(20, 6, 2).norm(dim=-1), torch.ones(20, 6, dtype=F64))


def test_fourier_embedding_known_wavevectors_D12():
    _, k = fourier_embedding(torch.zeros(1, 2, dtype=F64), D=12, period=1.0)
    assert {tuple(r) for r in k.tolist()} == {(1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (0, 2)}


@pytest.mark.parametrize("D", [12, 16, 48, 64])
def test_fourier_embedding_keeps_lowest_wavevectors(D):
    _, k = fourier_embedding(torch.zeros(1, 2, dtype=F64), D=D, period=1.0)
    assert {tuple(r) for r in k.tolist()} == _lowest_wavevectors(D // 2)
    assert k.shape == (D // 2, 2)


def test_fourier_embedding_hand_computed_phases():
    D, period = 6, 4.0
    at_origin, k = fourier_embedding(torch.zeros(1, 2, dtype=F64), D=D, period=period)
    assert torch.allclose(at_origin, torch.tensor([[1, 0, 1, 0, 1, 0]], dtype=F64))
    # A quarter period along axis 0 puts every mode with k = (1, 0) at phase pi/2.
    y = torch.tensor([[period / 4, 0.0]], dtype=F64)
    feats, k = fourier_embedding(y, D=D, period=period)
    i = k.tolist().index([1, 0])
    assert feats[0, 2 * i].abs() < 1e-12 and feats[0, 2 * i + 1] == pytest.approx(1.0)
    phase = 2 * math.pi * (y @ k.to(F64).T)[0] / period
    assert torch.allclose(feats[0, 0::2], torch.cos(phase))
    assert torch.allclose(feats[0, 1::2], torch.sin(phase))


@pytest.mark.parametrize("embedding", ["linear", "fourier"])
def test_embed_reproduces_dataset_x(embedding):
    cfg = _cfg(embedding=embedding)
    ds = make_dataset(cfg, CPU, F64)
    assert torch.allclose(embed(ds.y, ds, cfg), ds.x)


def test_embed_rejects_mismatched_dataset():
    lin = make_dataset(_cfg(embedding="linear"), CPU, F64)
    fou = make_dataset(_cfg(embedding="fourier"), CPU, F64)
    with pytest.raises(ValueError, match="Q"):
        embed(lin.y, fou, _cfg(embedding="linear"))
    with pytest.raises(ValueError, match="period"):
        embed(fou.y, lin, _cfg(embedding="fourier"))
    with pytest.raises(ValueError, match="D_in"):
        embed(fou.y, fou, _cfg(embedding="fourier", D_in=10))


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


def test_asymmetric_cov_raises():
    with pytest.raises(ValueError, match="symmetric"):
        make_dataset(_cfg(cov=[[1.0, 0.5], [0.0, 1.0]]), CPU, F64)


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
    assert second.x.dtype == second.y.dtype == F64
    assert second.period == first.period
    if embedding == "linear":
        assert torch.equal(first.Q, second.Q) and second.Q.dtype == F64
    else:
        assert second.Q is None


def test_get_dataset_stale_after_config_change(tmp_path):
    get_dataset(tmp_path, _full_cfg(seed=0))
    with pytest.raises(StaleCacheError):
        get_dataset(tmp_path, _full_cfg(seed=1))
