import pytest
import torch

from fct.cache import StaleCacheError, cache_paths, latest_checkpoint, load_artefact
from fct.config import Config, DataConfig, NetConfig, TrainConfig
from fct.data import make_dataset
from fct.model import make_net
from fct.train import compute_loss, load_trained, train

CPU, F64 = torch.device("cpu"), torch.float64


class Interrupt(Exception):
    pass


def _cfg(n_epochs=20, **train_kw) -> Config:
    train_kw = {"log_int": 5, "save_int": 5, "rate_lambda": 1e-3, **train_kw}
    return Config(
        data=DataConfig(P=16, D_in=4, d=2, embedding="linear", seed=1),
        net=NetConfig(N=8, L=2, g=1.0, phi="tanh", seed=2),
        train=TrainConfig(n_epochs=n_epochs, lr=1e-2, **train_kw),
        dtype="float64",
    )


def _setup(cfg: Config):
    ds = make_dataset(cfg.data, CPU, F64)
    net = make_net(cfg.net, ds.x.shape[1], ds.y.shape[1], CPU, F64)
    return ds, net


def _epoch_lines(lines):
    return [s for s in lines if s.startswith("[")]


def _assert_same_params(a, b):
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k]), k


def test_compute_loss_matches_hand_computation():
    cfg = Config(net=NetConfig(N=3, L=1, g=1.0, phi="relu", seed=0))
    net = make_net(cfg.net, 2, 1, CPU, F64)
    x = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=F64)
    y = torch.tensor([[0.3], [-0.2]], dtype=F64)
    lw, lr_ = 0.1, 0.2

    h = (x @ net.U.T) / 2**0.5  # (P, N)
    z = (h @ net.V) / 3  # (P, 1)
    mse = ((y - z) ** 2).mean()
    w2 = net.U.square().sum() + net.V.square().sum()
    r2 = torch.relu(h).square().sum()
    expected = 0.5 * 3 * mse + lw * w2 + lr_ * r2

    loss, parts = compute_loss(net, x, y, lw, lr_)
    assert loss.item() == pytest.approx(expected.item())
    assert parts == pytest.approx(
        {"mse": mse.item(), "weight_l2": w2.item(), "rate_l2": r2.item(), "loss": expected.item()}
    )
    assert loss.requires_grad


def test_linear_task_is_learned(tmp_path):
    cfg = Config(
        data=DataConfig(P=64, D_in=4, d=1, embedding="linear", seed=0),
        net=NetConfig(N=16, L=1, phi="relu", seed=0),
        train=TrainConfig(
            n_epochs=300, lr=1e-2, rate_lambda=0.0, weight_lambda=1e-3, log_int=50, save_int=100
        ),
        dtype="float64",
    )
    ds, net = _setup(cfg)
    _, hist = train(cfg, ds, net, tmp_path, log=lambda s: None)
    assert hist["mse"][-1] < 0.1 * hist["mse"][0]


def test_resume_matches_uninterrupted_run(tmp_path):
    cfg = _cfg()
    ds, net_ref = _setup(cfg)
    _, hist_ref = train(cfg, ds, net_ref, tmp_path / "ref", log=lambda s: None)

    def log_then_raise(line):
        if line.startswith("[00010/"):
            raise Interrupt

    _, net_a = _setup(cfg)
    with pytest.raises(Interrupt):
        train(cfg, ds, net_a, tmp_path / "run", log=log_then_raise)
    train_dir = cache_paths(tmp_path / "run").train
    saved = load_artefact(
        latest_checkpoint(train_dir), cfg.section_hash("data", "net", "train"), CPU
    )
    assert saved["epoch"] == 10  # next epoch to run: step 10 had not been saved
    assert saved["history"]["epoch"] == [0, 5]

    resumed = []
    _, net_b = _setup(cfg)
    net_b, hist_b = train(cfg, ds, net_b, tmp_path / "run", log=resumed.append)
    assert "ckpt_000010.pt" in resumed[0]
    assert len(_epoch_lines(resumed)) == 2  # epochs 10 and 15 only; none repeated or skipped
    assert hist_b == hist_ref
    assert hist_b["epoch"] == [0, 5, 10, 15]
    _assert_same_params(net_ref, net_b)
    assert sorted(p.name for p in train_dir.glob("*.pt")) == ["ckpt_000015.pt", "final.pt"]


def test_train_with_final_does_no_work(tmp_path):
    cfg = _cfg()
    ds, net = _setup(cfg)
    net, hist = train(cfg, ds, net, tmp_path, log=lambda s: None)

    lines = []
    _, fresh = _setup(cfg)
    fresh, hist2 = train(cfg, ds, fresh, tmp_path, log=lines.append)
    assert _epoch_lines(lines) == []
    assert hist2 == hist
    _assert_same_params(net, fresh)


def test_load_trained_roundtrip_and_errors(tmp_path):
    cfg = _cfg()
    ds, net = _setup(cfg)
    net, hist = train(cfg, ds, net, tmp_path, log=lambda s: None)

    loaded, hist2 = load_trained(tmp_path, cfg, CPU, F64)
    assert hist2 == hist
    assert torch.equal(loaded(ds.x)[1], net(ds.x)[1])

    stale = cfg.model_copy(update={"train": cfg.train.model_copy(update={"lr": 0.5})})
    with pytest.raises(StaleCacheError):
        load_trained(tmp_path, stale, CPU, F64)
    with pytest.raises(FileNotFoundError):
        load_trained(tmp_path / "empty", cfg, CPU, F64)
