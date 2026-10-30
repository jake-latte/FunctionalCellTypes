"""`SpatialConfig.lambda_max`: every population kernel is scaled to this top eigenvalue, the
config constraints that go with it, the unchanged behaviour without it, and a sweep axis."""

import json
import math

import pytest
import torch

from fct.config import Config, DataConfig, NetConfig, TrainConfig
from fct.data import make_dataset
from fct.model import make_net
from fct.spatial import (
    EmbeddingSpec,
    KernelSpec,
    PopulationSpec,
    SpatialConfig,
    build_spaces,
    periodic_gaussian_kernel,
)
from fct.sweep import expand, preset, setting_config, with_grid
from fct.train import train

CPU, F64 = torch.device("cpu"), torch.float64
# Grid shapes of (input, hidden, output) per dimension d, so each population has a few points.
SHAPES = {
    1: [[6], [8], [5]],
    2: [[3, 3], [3, 4], [2, 3]],
    3: [[2, 2, 2], [2, 2, 3], [2, 2, 2]],
    4: [[2, 2, 2, 2], [2, 2, 2, 3], [2, 2, 2, 2]],
}
# Section hash of the config in `_pinned_spatial`, from the code before `lambda_max` existed.
PINNED_SPATIAL_HASH = "c93ee7958838"
SECTIONS = ("data", "net", "train")


def _spatial(d, kind, length_scale, lambda_max, amplitude=1.0) -> SpatialConfig:
    def spec(shape):
        emb = (
            EmbeddingSpec(kind="random", seed=1) if kind == "random" else EmbeddingSpec(shape=shape)
        )
        kernel = KernelSpec(length_scale=length_scale, amplitude=amplitude)
        return PopulationSpec(embedding=emb, kernel=kernel)

    inp, hid, out = SHAPES[d]
    return SpatialConfig(
        d=d, input=spec(inp), hidden=spec(hid), output=spec(out), lambda_max=lambda_max
    )


def _pinned_spatial() -> SpatialConfig:
    return SpatialConfig(
        d=2,
        input=PopulationSpec(embedding=EmbeddingSpec(shape=[3, 3])),
        hidden=PopulationSpec(
            embedding=EmbeddingSpec(kind="random", seed=2),
            kernel=KernelSpec(length_scale=0.2, amplitude=2.0),
        ),
        output=PopulationSpec(embedding=EmbeddingSpec(shape=[2, 2])),
    )


@pytest.mark.parametrize("d", [1, 2, 3, 4])
@pytest.mark.parametrize("kind", ["random", "grid"])
@pytest.mark.parametrize("length_scale", [0.05, 0.2, 0.6])
def test_every_population_has_top_eigenvalue_lambda_max(d, kind, length_scale):
    lam = 0.37
    m, N, n_out = (math.prod(s) for s in SHAPES[d])
    spaces = build_spaces(_spatial(d, kind, length_scale, lam), m=m, N=N, L=2, n_out=n_out)
    pops = [spaces[0].pre] + [s.post for s in spaces]
    assert len(pops) == 4
    for pop in pops:
        top = torch.linalg.eigvalsh(pop.K)[-1].item()  # independent of the recorded scale
        assert top == pytest.approx(lam, rel=1e-10)


def test_population_records_the_scale_it_applied():
    cfg = _spatial(2, "random", 0.2, 0.5)
    space = build_spaces(cfg, m=5, N=6, L=1, n_out=4)[0]
    pos = space.pre.embedding.positions
    G = periodic_gaussian_kernel(pos, pos, 0.2, cfg.box, 1.0)
    assert space.pre.scale == pytest.approx(0.5 / torch.linalg.eigvalsh(G)[-1].item(), rel=1e-12)


def test_without_lambda_max_kernels_are_the_amplitude_times_the_unit_kernel_bitwise():
    cfg = _spatial(2, "random", 0.2, None, amplitude=2.5)
    space = build_spaces(cfg, m=5, N=6, L=1, n_out=4)[0]
    pos = space.pre.embedding.positions
    assert torch.equal(space.pre.K, periodic_gaussian_kernel(pos, pos, 0.2, cfg.box, 2.5))
    assert space.pre.scale == 1.0


def test_pinned_spatial_hash_is_unchanged_when_lambda_max_is_unset():
    cfg = Config(spatial=_pinned_spatial())
    assert cfg.section_hash(*SECTIONS) == PINNED_SPATIAL_HASH


def test_a_set_lambda_max_changes_the_hash():
    unset = Config(spatial=_spatial(1, "random", 0.2, None))
    set_ = Config(spatial=_spatial(1, "random", 0.2, 0.5), train=TrainConfig(weight_lambda=1.0))
    assert set_.section_hash(*SECTIONS) != unset.section_hash(*SECTIONS)


def test_amplitude_other_than_one_is_an_error_naming_the_field():
    with pytest.raises(ValueError, match=r"input\.kernel\.amplitude"):
        _spatial(1, "random", 0.2, 0.5, amplitude=2.0)


def test_weight_lambda_other_than_one_is_an_error_naming_the_field():
    with pytest.raises(ValueError, match="weight_lambda"):
        Config(spatial=_spatial(1, "random", 0.2, 0.5))  # the default weight_lambda is 2e-4
    Config(spatial=_spatial(1, "random", 0.2, 0.5), train=TrainConfig(weight_lambda=1.0))


def test_lambda_max_must_be_positive():
    with pytest.raises(ValueError, match="lambda_max"):
        _spatial(1, "random", 0.2, 0.0)


def _sweep(weight_lambda=1.0):
    base = preset("spatial_tiny").model_copy(
        update={"spatial": _spatial(1, "random", 0.3, None), "weight_lambda": weight_lambda}
    )
    return with_grid(base, {"spatial.d": [1, 2], "spatial.lambda_max": [0.1, 0.9]})


def test_a_lambda_max_sweep_rebuilds_kernels_with_the_right_top_eigenvalue():
    sweep = _sweep()
    settings = expand(sweep)
    assert len(settings) == 4
    for setting in settings:
        cfg = setting_config(sweep, setting)
        saved = Config.model_validate(json.loads(cfg.model_dump_json()))  # as config.json reloads
        spaces = build_spaces(saved.spatial, m=6, N=4, L=1, n_out=2)
        for pop in [spaces[0].pre] + [s.post for s in spaces]:
            top = torch.linalg.eigvalsh(pop.K)[-1].item()
            assert top == pytest.approx(setting.axes["spatial.lambda_max"], rel=1e-10)


def test_a_lambda_max_sweep_with_another_weight_lambda_fails_naming_it():
    with pytest.raises(ValueError, match="weight_lambda"):
        _sweep(weight_lambda=0.2)


def test_a_tiny_sgld_net_trains_under_lambda_max(tmp_path):
    spatial = _spatial(1, "random", 0.3, 1.0).model_copy(update={"min_rel_eig": 1e-2})
    cfg = Config(
        data=DataConfig(P=16, D_in=4, d=2, embedding="linear", seed=1),
        net=NetConfig(N=6, L=1, g=1.0, phi="tanh", seed=2),
        train=TrainConfig(
            n_epochs=10, lr=1e-2, optimizer="sgld", beta=1e3, weight_lambda=1.0, log_int=5
        ),
        spatial=spatial,
        dtype="float64",
        device="cpu",
    )
    ds = make_dataset(cfg.data, CPU, F64)
    net = make_net(cfg.net, ds.x.shape[1], ds.y.shape[1], CPU, F64)
    _, hist = train(cfg, ds, net, tmp_path, log=lambda s: None)
    assert all(math.isfinite(v) for v in hist["loss"])
