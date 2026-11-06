"""`fct.tasks.latent_lattice_dataset` and `fct.data.get_latent_lattice_dataset`: the 3^k lattice of latent sign patterns."""

import dataclasses

import pytest
import torch

from fct.config import Config, DataConfig
from fct.data import get_dataset, get_latent_lattice_dataset, make_dataset
from fct.tasks import Dataset, latent_lattice_dataset

CPU, F64 = torch.device("cpu"), torch.float64


def _cfg(k=2, a=1.0, **kw) -> DataConfig:
    return DataConfig(
        task="linear_teacher", P=20, D_in=6, n_latents=k, latent_min=-a, latent_max=a, **kw
    )


def _latents(ds: Dataset, k: int) -> torch.Tensor:
    return torch.stack([ds.factors[f"z{i}"] for i in range(k)], dim=1)


NINE = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]


def test_k2_gives_the_nine_sign_patterns_in_documented_order():
    cfg = _cfg(k=2, a=1.0)
    ds = latent_lattice_dataset(cfg, make_dataset(cfg, CPU, F64))
    assert torch.equal(_latents(ds, 2), torch.tensor(NINE, dtype=F64))
    assert ds.x.shape == ds.y.shape == (9, 6)


def test_k1_gives_three_rows():
    cfg = _cfg(k=1, a=2.0)
    ds = latent_lattice_dataset(cfg, make_dataset(cfg, CPU, F64))
    assert torch.equal(_latents(ds, 1), torch.tensor([[-2.0], [0.0], [2.0]], dtype=F64))
    assert ds.x.shape == (3, 6)


def test_row_index_is_its_base3_digits_at_k3():
    cfg = _cfg(k=3, a=1.0)
    z = _latents(latent_lattice_dataset(cfg, make_dataset(cfg, CPU, F64)), 3)
    # 19 = 2*9 + 0*3 + 1: base-3 digits (2, 0, 1) select (a, -a, 0) from (-a, 0, a)
    assert z[19].tolist() == [1.0, -1.0, 0.0]


@pytest.mark.parametrize("k", [2, 3])
def test_each_latent_takes_each_value_equally_often_and_scales_with_a(k):
    cfg = _cfg(k=k, a=2.5)
    ds = latent_lattice_dataset(cfg, make_dataset(cfg, CPU, F64))
    z = _latents(ds, k)
    assert z.shape == (3**k, k)
    for i in range(k):
        for v in (-2.5, 0.0, 2.5):
            assert int((z[:, i] == v).sum()) == 3 ** (k - 1)


def test_an_asymmetric_interval_gives_its_ends_and_midpoint():
    cfg = DataConfig(
        task="linear_teacher", P=20, D_in=6, n_latents=2, latent_min=0.0, latent_max=1.0
    )
    z = _latents(latent_lattice_dataset(cfg, make_dataset(cfg, CPU, F64)), 2)
    expected = [[u / 2, v / 2] for u in range(3) for v in range(3)]
    assert torch.equal(z, torch.tensor(expected, dtype=F64))


def test_inputs_and_targets_are_the_reference_linear_maps():
    cfg = _cfg(k=3)
    ref = make_dataset(cfg, CPU, F64)
    ds = latent_lattice_dataset(cfg, ref)
    z = _latents(ds, 3)
    assert torch.allclose(ds.x, z @ ref.extras["W_X"].T)
    assert torch.allclose(ds.y, z @ ref.extras["W_Y"].T)
    assert ds.extras is ref.extras
    assert ds.x.dtype == ref.x.dtype and ds.x.device == ref.x.device
    assert ds.groups == {f"z{i}": [f"z{i}"] for i in range(3)}


def test_dtype_follows_the_reference():
    cfg = _cfg()
    ref = make_dataset(cfg, CPU, torch.float32)
    ds = latent_lattice_dataset(cfg, ref)
    assert ds.x.dtype == ds.y.dtype == torch.float32
    assert all(f.dtype == torch.float32 for f in ds.factors.values())


def test_none_extras_raises_value_error():
    cfg = _cfg()
    ref = dataclasses.replace(make_dataset(cfg, CPU, F64), extras=None)
    with pytest.raises(ValueError, match="W_X"):
        latent_lattice_dataset(cfg, ref)


def test_shape_mismatch_names_fields_and_shapes():
    ref = make_dataset(_cfg(k=2), CPU, F64)
    with pytest.raises(ValueError, match=r"D_in, n_latents.*\(6, 3\).*W_X.*\(6, 2\)"):
        latent_lattice_dataset(_cfg(k=3), ref)


def test_other_task_raises_naming_it():
    cfg = DataConfig(task="what_where", P=10, D_in=81, n_motifs=4)
    ref = make_dataset(cfg, CPU, F64)
    with pytest.raises(ValueError, match="what_where"):
        latent_lattice_dataset(cfg, ref)


def test_reference_without_teacher_maps_raises():
    cfg = _cfg()
    ref = make_dataset(cfg, CPU, F64)
    bare = dataclasses.replace(ref, extras={})
    with pytest.raises(ValueError, match="W_X"):
        latent_lattice_dataset(cfg, bare)


def _snapshot(root):
    files = [p for p in root.rglob("*") if p.is_file()]
    return sorted((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files)


def test_get_latent_lattice_dataset_matches_pure_function_and_writes_nothing(tmp_path):
    cfg = Config(data=_cfg(k=2), device="cpu", dtype="float64")
    ref = get_dataset(tmp_path, cfg)
    before = _snapshot(tmp_path)
    ds = get_latent_lattice_dataset(tmp_path, cfg)
    assert _snapshot(tmp_path) == before
    pure = latent_lattice_dataset(cfg.data, ref)
    assert torch.equal(ds.x, pure.x) and torch.equal(ds.y, pure.y)
    assert ds.groups == pure.groups
    assert all(torch.equal(ds.factors[n], pure.factors[n]) for n in pure.factors)
    assert all(torch.equal(ds.extras[n], pure.extras[n]) for n in pure.extras)
