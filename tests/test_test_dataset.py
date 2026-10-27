"""`fct.tasks.test_dataset` and `fct.data.get_test_dataset`: the 3^k lattice of latent sign patterns."""

import itertools

import pytest
import torch

from fct.cache import cache_paths
from fct.config import Config, DataConfig
from fct.data import get_dataset, get_test_dataset, make_dataset
from fct.tasks import Dataset
from fct.tasks import test_dataset as lattice  # renamed so pytest does not collect it as a test

CPU, F64 = torch.device("cpu"), torch.float64


def _cfg(k=2, a=1.0, **kw) -> DataConfig:
    return DataConfig(task="linear_teacher", P=20, D_in=6, n_latents=k, latent_range=a, **kw)


def _latents(ds: Dataset, k: int) -> torch.Tensor:
    return torch.stack([ds.factors[f"z{i}"] for i in range(k)], dim=1)


def test_k2_gives_the_nine_sign_patterns_in_documented_order():
    cfg = _cfg(k=2, a=1.0)
    ds = lattice(cfg, make_dataset(cfg, CPU, F64))
    expected = torch.tensor(list(itertools.product([-1.0, 0.0, 1.0], repeat=2)), dtype=F64)
    assert torch.equal(_latents(ds, 2), expected)
    assert ds.x.shape == ds.y.shape == (9, 6)


@pytest.mark.parametrize("k", [2, 3])
def test_each_latent_takes_each_value_equally_often_and_scales_with_a(k):
    cfg = _cfg(k=k, a=2.5)
    ds = lattice(cfg, make_dataset(cfg, CPU, F64))
    z = _latents(ds, k)
    assert z.shape == (3**k, k)
    for i in range(k):
        for v in (-2.5, 0.0, 2.5):
            assert int((z[:, i] == v).sum()) == 3 ** (k - 1)


def test_inputs_and_targets_are_the_reference_linear_maps():
    cfg = _cfg(k=3)
    ref = make_dataset(cfg, CPU, F64)
    ds = lattice(cfg, ref)
    z = _latents(ds, 3)
    assert torch.allclose(ds.x, z @ ref.extras["W_X"].T)
    assert torch.allclose(ds.y, z @ ref.extras["W_Y"].T)
    assert ds.extras is ref.extras
    assert ds.x.dtype == ref.x.dtype and ds.x.device == ref.x.device
    assert ds.groups == {f"z{i}": [f"z{i}"] for i in range(3)}


def test_dtype_follows_the_reference():
    cfg = _cfg()
    ref = make_dataset(cfg, CPU, torch.float32)
    ds = lattice(cfg, ref)
    assert ds.x.dtype == ds.y.dtype == torch.float32
    assert all(f.dtype == torch.float32 for f in ds.factors.values())


def test_other_task_raises_naming_it():
    cfg = DataConfig(task="what_where", P=10, D_in=81, n_motifs=4)
    ref = make_dataset(cfg, CPU, F64)
    with pytest.raises(ValueError, match="what_where"):
        lattice(cfg, ref)


def test_reference_without_teacher_maps_raises():
    cfg = _cfg()
    ref = make_dataset(cfg, CPU, F64)
    bare = Dataset(**{**ref.__dict__, "extras": {}})
    with pytest.raises(ValueError, match="W_X"):
        lattice(cfg, bare)


def test_get_test_dataset_matches_pure_function_and_writes_nothing_extra(tmp_path):
    cfg = Config(data=_cfg(k=2), device="cpu", dtype="float64")
    ref = get_dataset(tmp_path, cfg)
    before = sorted(tmp_path.rglob("*"))
    ds = get_test_dataset(tmp_path, cfg)
    assert sorted(tmp_path.rglob("*")) == before
    assert (cache_paths(tmp_path).data / "dataset.pt").exists()
    pure = lattice(cfg.data, ref)
    assert torch.equal(ds.x, pure.x) and torch.equal(ds.y, pure.y)
