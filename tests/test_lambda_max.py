"""`SpatialConfig.lambda_max`: every population kernel is scaled to this top eigenvalue, the
config constraints that go with it, the unchanged behaviour without it, and an experiment axis."""

import json
import math

import pytest
import torch

from fct.config import Config, DataConfig, NetConfig, TrainConfig
from fct.data import make_dataset
from fct.exp import ExpConfig, expand, preset, setting_config, with_axes
from fct.model import make_net
from fct.spatial import (
    EmbeddingSpec,
    KernelSpec,
    PopulationSpec,
    SpatialConfig,
    build_spaces,
    parameter_factors,
    periodic_gaussian_kernel,
)
from fct.train import make_weight_penalty, spatial_weight_penalty, train

CPU, F64 = torch.device("cpu"), torch.float64
# Grid shapes of (input, hidden, output) per dimension d, so each population has a few points.
SHAPES = {
    1: [[6], [8], [5]],
    2: [[3, 3], [3, 4], [2, 3]],
    3: [[2, 2, 2], [2, 2, 3], [2, 2, 2]],
    4: [[2, 2, 2, 2], [2, 2, 2, 3], [2, 2, 2, 2]],
}
# Section hash of the config in `_pinned_spatial`. It was pinned with the code before
# `lambda_max` existed and recomputed after `latent_range` was renamed to `latent_min` /
# `latent_max` (that code never had the new fields, so it could not have written this value),
# and again for LOSS_VERSION 5, which the config's default rate_lambda > 0 hashes (version 4
# gave 1ef37eb7a218); neither recomputation changed any field.
PINNED_SPATIAL_HASH = "55da84fd2ded"
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


def test_every_population_records_the_scale_it_applied():
    cfg = _spatial(2, "random", 0.2, 0.5)
    spaces = build_spaces(cfg, m=5, N=6, L=2, n_out=4)
    for pop in [spaces[0].pre] + [s.post for s in spaces]:
        pos = pop.embedding.positions
        G = periodic_gaussian_kernel(pos, pos, 0.2, cfg.box, 1.0)
        assert pop.scale == pytest.approx(0.5 / torch.linalg.eigvalsh(G)[-1].item(), rel=1e-12)
        assert torch.diagonal(pop.K) == pytest.approx(
            pop.scale, rel=1e-12
        )  # diag = amplitude*scale


def test_without_lambda_max_kernels_are_the_amplitude_times_the_unit_kernel_bitwise():
    cfg = _spatial(2, "random", 0.2, None, amplitude=2.5)
    space = build_spaces(cfg, m=5, N=6, L=1, n_out=4)[0]
    pos = space.pre.embedding.positions
    assert torch.equal(space.pre.K, periodic_gaussian_kernel(pos, pos, 0.2, cfg.box, 2.5))
    assert space.pre.scale == 1.0


def test_pinned_spatial_hash_is_unchanged_when_lambda_max_is_unset():
    cfg = Config(spatial=_pinned_spatial())
    assert cfg.section_hash(*SECTIONS) == PINNED_SPATIAL_HASH


def _hash_with(lambda_max) -> str:
    cfg = Config(
        spatial=_spatial(1, "random", 0.2, lambda_max), train=TrainConfig(weight_lambda=1.0)
    )
    return cfg.section_hash(*SECTIONS)


def test_the_hash_depends_on_lambda_max_alone():
    assert _hash_with(0.5) != _hash_with(0.9)
    assert _hash_with(None) != _hash_with(0.5)
    assert _hash_with(0.5) == _hash_with(0.5)


@pytest.mark.parametrize("name", ["input", "hidden", "output"])
def test_amplitude_other_than_one_is_an_error_naming_the_field(name):
    good = _spatial(1, "random", 0.2, None)
    bad = PopulationSpec(embedding=good.input.embedding, kernel=KernelSpec(amplitude=2.0))
    fields = {**good.model_dump(), name: bad.model_dump(), "lambda_max": 0.5}
    with pytest.raises(ValueError, match=rf"{name}\.kernel\.amplitude"):
        SpatialConfig.model_validate(fields)
    SpatialConfig.model_validate({**fields, "lambda_max": None})  # fine without lambda_max


def test_weight_lambda_other_than_one_is_an_error_naming_the_field():
    with pytest.raises(ValueError, match="weight_lambda"):
        Config(spatial=_spatial(1, "random", 0.2, 0.5))  # the default weight_lambda is 2e-4
    Config(spatial=_spatial(1, "random", 0.2, 0.5), train=TrainConfig(weight_lambda=1.0))


@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan])
def test_lambda_max_must_be_positive_and_finite(bad):
    with pytest.raises(ValueError, match="lambda_max"):
        _spatial(1, "random", 0.2, bad)


def test_lambda_max_survives_a_json_round_trip():
    cfg = Config(spatial=_spatial(1, "random", 0.2, 0.5), train=TrainConfig(weight_lambda=1.0))
    assert Config.model_validate_json(cfg.model_dump_json()).spatial.lambda_max == 0.5


def _exp(weight_lambda=1.0):
    base = preset("spatial_tiny").model_copy(
        update={"spatial": _spatial(1, "random", 0.3, None), "weight_lambda": weight_lambda}
    )
    return with_axes(base, {"spatial.d": [1, 2], "spatial.lambda_max": [0.1, 0.9]})


def test_a_lambda_max_exp_rebuilds_kernels_with_the_right_top_eigenvalue():
    exp = _exp()
    settings = expand(exp)
    assert len(settings) == 4
    for setting in settings:
        cfg = setting_config(exp, setting)
        saved = Config.model_validate(json.loads(cfg.model_dump_json()))  # as config.json reloads
        spaces = build_spaces(saved.spatial, m=6, N=4, L=1, n_out=2)
        for pop in [spaces[0].pre] + [s.post for s in spaces]:
            top = torch.linalg.eigvalsh(pop.K)[-1].item()
            assert top == pytest.approx(setting.axes["spatial.lambda_max"], rel=1e-10)


def test_a_lambda_max_exp_with_another_weight_lambda_fails_naming_it():
    with pytest.raises(ValueError, match="weight_lambda"):
        _exp(weight_lambda=0.2)


def test_an_exp_without_axes_checks_its_base_settings_at_load():
    base = preset("spatial_tiny").model_copy(update={"axes": None, "weight_lambda": 0.2})
    spatial = _spatial(1, "random", 0.3, 0.5)
    with pytest.raises(ValueError, match="weight_lambda"):
        ExpConfig.model_validate({**base.model_dump(), "spatial": spatial.model_dump()})


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
    spaces = build_spaces(spatial, m=ds.x.shape[1], N=6, L=1, n_out=ds.y.shape[1])
    factors = parameter_factors(spaces)
    expected = spatial_weight_penalty(net, factors).item()  # scaled M at the initial weights
    assert make_weight_penalty(cfg, net)(net).item() == pytest.approx(expected, rel=1e-12)
    _, hist = train(cfg, ds, net, tmp_path, log=lambda s: None)
    assert all(math.isfinite(v) for v in hist["loss"])
    assert hist["weight_l2"][0] == pytest.approx(expected, rel=1e-6)
