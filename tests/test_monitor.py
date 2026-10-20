import dataclasses

import pytest
import torch

from fct.activations import get_phi
from fct.config import Config, DataConfig, NetConfig, TrainConfig
from fct.data import make_dataset
from fct.model import make_net
from fct.modularity import network_mir, shuffled_mir
from fct.monitor import MIR_MONITOR_KEYS, make_mir_monitor
from fct.tasks import Dataset

CPU, F64 = torch.device("cpu"), torch.float64


def _cfg(data: DataConfig, L=2, N=8, **train_kw) -> Config:
    train_kw = {"monitor_bins": 4, "monitor_shuffles": 3, "seed": 5, **train_kw}
    return Config(
        data=data,
        net=NetConfig(N=N, L=L, phi="tanh", seed=1),
        train=TrainConfig(**train_kw),
        dtype="float64",
        device="cpu",
    )


def _what_where_cfg() -> Config:
    return _cfg(DataConfig(task="what_where", P=96, D_in=81, n_motifs=4, seed=3))


def _linear_teacher_cfg() -> Config:
    return _cfg(DataConfig(task="linear_teacher", P=96, D_in=6, n_latents=3, seed=3), L=3)


@pytest.mark.parametrize("make_cfg", [_what_where_cfg, _linear_teacher_cfg])
def test_monitor_equals_direct_modularity_calls(make_cfg):
    cfg = make_cfg()
    ds = make_dataset(cfg.data, CPU, F64)
    net = make_net(cfg.net, ds.x.shape[1], ds.y.shape[1], CPU, F64)
    out = make_mir_monitor(ds, cfg)(net)

    with torch.no_grad():
        rates = get_phi(cfg.net.phi)(net(ds.x)[0])
    tc = cfg.train
    args = (ds.factors, ds.groups, tc.monitor_bins)
    raw = network_mir(rates, *args, "none", tc.monitor_shuffles, tc.seed)
    corrected = network_mir(rates, *args, "shuffle", tc.monitor_shuffles, tc.seed)
    chance = [
        shuffled_mir(rates[:, layer], *args, seed=tc.seed, n_shuffles=tc.monitor_shuffles)
        for layer in range(cfg.net.L)
    ]

    assert set(out) == set(MIR_MONITOR_KEYS)
    assert out["mir"] == [m.mir for m in raw]
    assert out["mir_corrected"] == [m.mir for m in corrected]
    assert out["mir_chance"] == chance
    assert out["n_dead"] == [m.n_dead for m in raw]
    assert out["n_dead_corrected"] == [m.n_dead for m in corrected]
    assert all(len(v) == cfg.net.L for v in out.values())
    assert all(type(v) is int for v in out["n_dead"] + out["n_dead_corrected"])
    assert all(type(v) is float for v in out["mir"] + out["mir_corrected"] + out["mir_chance"])


def test_monitor_is_deterministic_and_leaves_rng_and_grads_alone():
    cfg = _what_where_cfg()
    ds = make_dataset(cfg.data, CPU, F64)
    net = make_net(cfg.net, ds.x.shape[1], ds.y.shape[1], CPU, F64)
    monitor = make_mir_monitor(ds, cfg)
    state = torch.get_rng_state()
    first, second = monitor(net), monitor(net)
    assert first == second
    assert torch.equal(torch.get_rng_state(), state)
    assert all(p.grad is None for p in net.parameters())


def test_monitor_requires_factors():
    cfg = _what_where_cfg()
    ds = dataclasses.replace(make_dataset(cfg.data, CPU, F64), factors=None, groups=None)
    with pytest.raises(ValueError, match="factors"):
        make_mir_monitor(ds, cfg)


def _two_factor_dataset(P: int, shuffle_factors: bool) -> Dataset:
    """x = (a, b) with a, b independent uniform; the factors are a and b themselves (or, to
    destroy the dependence, a permutation of them)."""
    gen = torch.Generator().manual_seed(0)
    x = torch.rand((P, 2), dtype=F64, generator=gen)
    factors = {"a": x[:, 0].clone(), "b": x[:, 1].clone()}
    if shuffle_factors:
        perm = torch.randperm(P, generator=gen)
        factors = {k: v[perm] for k, v in factors.items()}
    return Dataset(
        x=x,
        y=torch.zeros(P, 1, dtype=F64),
        Q=None,
        period=None,
        factors=factors,
        groups={"a": ["a"], "b": ["b"]},
    )


def _two_factor_cfg(N: int) -> Config:
    # Config requires a DataConfig, but the monitor reads the Dataset it is given, not this.
    data = DataConfig(task="position", P=1, D_in=2, embedding="linear")
    return _cfg(data, L=1, N=N, monitor_bins=5, monitor_shuffles=5)


def test_neurons_hand_set_to_read_one_factor_each_give_high_mir():
    # One layer, tanh (monotone, so ranks are preserved): the first half of the neurons has
    # weight row (1, 0) and so fires as a function of factor a alone, the second half has
    # row (0, 1) and reads b alone. Every neuron is then informative about exactly one group.
    cfg, N, P = _two_factor_cfg(N=16), 16, 2000
    ds = _two_factor_dataset(P, shuffle_factors=False)
    net = make_net(cfg.net, 2, 1, CPU, F64)
    with torch.no_grad():
        net.U.zero_()
        net.U[: N // 2, 0] = 1.0
        net.U[N // 2 :, 1] = 1.0
    out = make_mir_monitor(ds, cfg)(net)
    # Thresholds, with P = 2000 and 5 bins. A neuron that is a monotone function of its factor
    # has information log 5 about it and only the plug-in bias (B-1)(C-1)/(2P) = 0.004 nats
    # about the other, so r = 0.998 and the raw MIR = 0.996 (> 0.95 leaves room for the
    # tanh/ties binning); correcting subtracts that bias, so the corrected MIR is ~1 (> 0.9).
    # The chance level is the raw MIR of unrelated neurons, max/sum of two near-equal noise
    # informations: about 0.2-0.3, so the corrected MIR clears it by ~0.7 (> 0.3 is loose).
    assert out["mir"][0] > 0.95
    assert out["mir_corrected"][0] > 0.9
    assert out["mir_corrected"][0] > out["mir_chance"][0] + 0.3
    assert out["n_dead_corrected"] == [0]


def test_fresh_net_on_shuffled_factors_sits_at_chance():
    cfg, P = _two_factor_cfg(N=64), 2000
    ds = _two_factor_dataset(P, shuffle_factors=True)
    net = make_net(cfg.net, 2, 1, CPU, F64)
    out = make_mir_monitor(ds, cfg)(net)
    # Raw MIR and its chance level are both the mean preference of 64 noise neurons (sd of
    # the mean ~0.02 each), so |difference| < 0.1 is ~2 sd; the corrected MIR is ~0 by design.
    assert abs(out["mir"][0] - out["mir_chance"][0]) < 0.1
    assert abs(out["mir_corrected"][0]) < 0.1
