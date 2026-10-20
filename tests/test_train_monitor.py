"""The `monitor` hook of `train`: when it is called, what lands in the history, resume exactness."""

import pytest
import torch

from fct.cache import cache_paths, latest_checkpoint, load_artefact
from fct.config import Config, DataConfig, NetConfig, TrainConfig
from fct.data import make_dataset
from fct.model import make_net
from fct.train import load_trained, train

CPU, F64 = torch.device("cpu"), torch.float64
QUIET = {"log": lambda s: None}


class Interrupt(Exception):
    pass


def _cfg(n_epochs=20, L=2, **train_kw) -> Config:
    train_kw = {"log_int": 5, "save_int": 5, "rate_lambda": 1e-3, **train_kw}
    return Config(
        data=DataConfig(P=16, D_in=4, d=2, embedding="linear", seed=1),
        net=NetConfig(N=8, L=L, g=1.0, phi="tanh", seed=2),
        train=TrainConfig(n_epochs=n_epochs, lr=1e-2, **train_kw),
        dtype="float64",
        device="cpu",
    )


def _setup(cfg: Config):
    ds = make_dataset(cfg.data, CPU, F64)
    net = make_net(cfg.net, ds.x.shape[1], ds.y.shape[1], CPU, F64)
    return ds, net


def _usum_monitor(net):
    """A deterministic function of the net, shaped like the MIR monitor (lists over 'layers')."""
    total = net.U.detach().sum().item()
    return {"usum": [total, -total], "count": [1, 2]}


def _epoch_of(line: str) -> int:
    return int(line[1:].split("/")[0])


def _assert_same_params(a, b):
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k]), k


@pytest.mark.parametrize("n_epochs, monitor_int", [(20, 5), (17, 5), (4, 7), (6, 1), (0, 3)])
def test_monitor_is_called_at_multiples_and_after_the_last_step(tmp_path, n_epochs, monitor_int):
    cfg = _cfg(n_epochs, monitor_int=monitor_int)
    ds, net = _setup(cfg)
    _, hist = train(cfg, ds, net, tmp_path, monitor=_usum_monitor, **QUIET)
    expected = [*range(0, n_epochs, monitor_int), n_epochs]  # final call; no duplicate
    assert hist["monitor_epoch"] == expected
    assert hist["monitor_count"] == [[1, 2]] * len(expected)
    assert len(hist["monitor_usum"]) == len(expected)
    assert {"epoch", "mse", "weight_l2", "rate_l2", "loss"} <= set(hist)


def test_monitor_sees_the_initial_and_the_trained_net(tmp_path):
    cfg = _cfg(6, monitor_int=3)
    ds, net = _setup(cfg)
    initial = _usum_monitor(net)["usum"]
    net, hist = train(cfg, ds, net, tmp_path, monitor=_usum_monitor, **QUIET)
    assert hist["monitor_usum"][0] == initial
    assert hist["monitor_usum"][-1] == _usum_monitor(net)["usum"]
    assert hist["monitor_usum"][0] != hist["monitor_usum"][-1]


def test_monitor_does_not_perturb_training(tmp_path):
    def rng_hungry_monitor(net):
        torch.rand(5)  # a careless monitor must not shift the training RNG stream
        return _usum_monitor(net)

    def run(monitor_int, monitor, root):
        cfg = _cfg(monitor_int=monitor_int)
        ds, net = _setup(cfg)
        torch.manual_seed(123)
        net, hist = train(cfg, ds, net, root, monitor=monitor, **QUIET)
        return net, hist, torch.get_rng_state()

    net_off, hist_off, rng_off = run(0, None, tmp_path / "off")
    net_on, hist_on, rng_on = run(5, rng_hungry_monitor, tmp_path / "on")
    _assert_same_params(net_off, net_on)
    assert torch.equal(rng_off, rng_on)
    assert {k: v for k, v in hist_on.items() if not k.startswith("monitor_")} == hist_off


def test_monitor_off_leaves_the_history_keys_unchanged(tmp_path):
    cfg = _cfg()
    ds, net = _setup(cfg)
    _, hist = train(cfg, ds, net, tmp_path, **QUIET)
    assert list(hist) == ["epoch", "mse", "weight_l2", "rate_l2", "loss"]


def test_monitor_and_monitor_int_must_agree(tmp_path):
    cfg = _cfg(monitor_int=0)
    ds, net = _setup(cfg)
    with pytest.raises(ValueError, match="monitor_int"):
        train(cfg, ds, net, tmp_path, monitor=_usum_monitor, **QUIET)
    cfg = _cfg(monitor_int=5)
    ds, net = _setup(cfg)
    with pytest.raises(ValueError, match="monitor"):
        train(cfg, ds, net, tmp_path, **QUIET)
    assert not (tmp_path / "train").exists()  # nothing was run or written


def test_monitor_with_a_reserved_or_changing_key_raises(tmp_path):
    cfg = _cfg(monitor_int=5)
    ds, net = _setup(cfg)
    with pytest.raises(ValueError, match="epoch"):
        train(cfg, ds, net, tmp_path / "a", monitor=lambda n: {"epoch": [1.0]}, **QUIET)
    calls = iter([{"a": [1.0]}, {"b": [1.0]}])
    with pytest.raises(ValueError, match="keys"):
        train(cfg, ds, net, tmp_path / "b", monitor=lambda n: next(calls), **QUIET)


@pytest.mark.parametrize(
    "n_epochs, L, save_int, monitor_int, log_int, interrupt_at",
    [(20, 2, 5, 5, 5, 10), (12, 3, 3, 2, 1, 7), (12, 2, 4, 3, 5, 5)],
)
def test_resume_reproduces_the_monitor_history(
    tmp_path, n_epochs, L, save_int, monitor_int, log_int, interrupt_at
):
    cfg = _cfg(n_epochs, L=L, save_int=save_int, monitor_int=monitor_int, log_int=log_int)
    ds, net_ref = _setup(cfg)
    _, hist_ref = train(cfg, ds, net_ref, tmp_path / "ref", monitor=_usum_monitor, **QUIET)

    def log_then_raise(line):
        if line.startswith("[") and _epoch_of(line) == interrupt_at:
            raise Interrupt

    _, net_a = _setup(cfg)
    with pytest.raises(Interrupt):
        train(cfg, ds, net_a, tmp_path / "run", log=log_then_raise, monitor=_usum_monitor)
    train_dir = cache_paths(tmp_path / "run").train
    saved = load_artefact(
        latest_checkpoint(train_dir), cfg.section_hash("data", "net", "train"), CPU
    )
    resume_at = interrupt_at // save_int * save_int
    assert saved["epoch"] == resume_at
    # the checkpoint written at epoch e predates e's monitor entry; the resumed run adds it
    assert saved["history"]["monitor_epoch"] == [
        e for e in range(resume_at) if e % monitor_int == 0
    ]

    _, net_b = _setup(cfg)
    net_b, hist_b = train(cfg, ds, net_b, tmp_path / "run", monitor=_usum_monitor, **QUIET)
    assert hist_b == hist_ref
    _assert_same_params(net_ref, net_b)
    assert load_trained(tmp_path / "run", cfg, CPU, F64)[1] == hist_ref
