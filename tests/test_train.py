import pytest
import torch

from fct.cache import (
    StaleCacheError,
    cache_paths,
    latest_checkpoint,
    load_artefact,
    save_artefact,
)
from fct.config import Config, DataConfig, NetConfig, TrainConfig
from fct.data import make_dataset
from fct.model import make_net
from fct.train import _restore_rng, compute_loss, load_trained, train

CPU, F64 = torch.device("cpu"), torch.float64


class Interrupt(Exception):
    pass


def _cfg(n_epochs=20, L=2, **train_kw) -> Config:
    train_kw = {"log_int": 5, "save_int": 5, "rate_lambda": 1e-3, **train_kw}
    return Config(
        data=DataConfig(P=16, D_in=4, d=2, embedding="linear", seed=1),
        net=NetConfig(N=8, L=L, g=1.0, phi="tanh", seed=2),
        train=TrainConfig(n_epochs=n_epochs, lr=1e-2, **train_kw),
        dtype="float64",
    )


def _setup(cfg: Config):
    ds = make_dataset(cfg.data, CPU, F64)
    net = make_net(cfg.net, ds.x.shape[1], ds.y.shape[1], CPU, F64)
    return ds, net


def _epoch_lines(lines):
    return [s for s in lines if s.startswith("[")]


def _epoch_of(line: str) -> int:
    return int(line[1:].split("/")[0])


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
    assert loss.requires_grad
    assert set(parts) == {"mse", "weight_l2", "rate_l2", "loss"}
    assert all(v.ndim == 0 and not v.requires_grad for v in parts.values())
    want = {"mse": mse, "weight_l2": w2, "rate_l2": r2, "loss": expected}
    for k, v in want.items():
        assert parts[k].item() == pytest.approx(v.item())


def test_compute_loss_two_layers_sums_every_layer_and_every_weight():
    cfg = Config(net=NetConfig(N=3, L=2, g=1.5, phi="tanh", seed=0))
    net = make_net(cfg.net, 2, 1, CPU, F64)
    x = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=F64)
    y = torch.tensor([[0.3], [-0.2]], dtype=F64)
    lw, lr_ = 0.1, 0.2

    h1 = (x @ net.U.T) / 2**0.5
    h2 = (1.5 / 3**0.5) * torch.tanh(h1) @ net.J[0].T
    z = (h2 @ net.V) / 3
    mse = ((y - z) ** 2).mean()
    w2 = net.U.square().sum() + net.J[0].square().sum() + net.V.square().sum()
    r2 = torch.tanh(h1).square().sum() + torch.tanh(h2).square().sum()
    expected = 0.5 * 3 * mse + lw * w2 + lr_ * r2

    loss, parts = compute_loss(net, x, y, lw, lr_)
    assert loss.item() == pytest.approx(expected.item())
    assert parts["weight_l2"].item() == pytest.approx(w2.item())
    assert parts["rate_l2"].item() == pytest.approx(r2.item())


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


@pytest.mark.parametrize(
    "n_epochs, L, save_int, log_int, interrupt_at",
    [(20, 2, 5, 5, 10), (12, 3, 3, 1, 7)],
)
def test_resume_matches_uninterrupted_run(
    tmp_path, n_epochs, L, save_int, log_int, interrupt_at
):
    cfg = _cfg(n_epochs, L=L, save_int=save_int, log_int=log_int)
    ds, net_ref = _setup(cfg)
    _, hist_ref = train(cfg, ds, net_ref, tmp_path / "ref", log=lambda s: None)

    def log_then_raise(line):
        if line.startswith("[") and _epoch_of(line) == interrupt_at:
            raise Interrupt

    _, net_a = _setup(cfg)
    with pytest.raises(Interrupt):
        train(cfg, ds, net_a, tmp_path / "run", log=log_then_raise)
    train_dir = cache_paths(tmp_path / "run").train
    saved = load_artefact(
        latest_checkpoint(train_dir), cfg.section_hash("data", "net", "train"), CPU
    )
    resume_at = interrupt_at // save_int * save_int  # last save at or before the interrupt
    assert saved["epoch"] == resume_at  # next epoch to run
    assert saved["history"]["epoch"] == [e for e in range(resume_at) if e % log_int == 0]

    resumed = []
    _, net_b = _setup(cfg)
    net_b, hist_b = train(cfg, ds, net_b, tmp_path / "run", log=resumed.append)
    assert f"ckpt_{resume_at:06d}.pt" in resumed[0]
    assert [_epoch_of(s) for s in _epoch_lines(resumed)] == [
        e for e in range(resume_at, n_epochs) if e % log_int == 0
    ]  # none repeated or skipped
    assert hist_b == hist_ref
    _assert_same_params(net_ref, net_b)
    last_save = (n_epochs - 1) // save_int * save_int
    assert sorted(p.name for p in train_dir.glob("*.pt")) == [f"ckpt_{last_save:06d}.pt", "final.pt"]


def test_train_with_final_does_no_work(tmp_path):
    cfg = _cfg()
    ds, net = _setup(cfg)
    net, hist = train(cfg, ds, net, tmp_path, log=lambda s: None)

    lines = []
    _, fresh = _setup(cfg)
    rng_before = torch.get_rng_state()
    fresh, hist2 = train(cfg, ds, fresh, tmp_path, log=lines.append)
    assert torch.equal(torch.get_rng_state(), rng_before)  # no side effects
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
    with pytest.raises(ValueError, match="dtype"):
        load_trained(tmp_path, cfg, CPU, torch.float32)


def _interrupted_run(tmp_path, cfg):
    def stop(line):
        if line.startswith("[") and _epoch_of(line) == 10:
            raise Interrupt

    ds, net = _setup(cfg)
    with pytest.raises(Interrupt):
        train(cfg, ds, net, tmp_path, log=stop)
    ckpt = latest_checkpoint(cache_paths(tmp_path).train)
    h = cfg.section_hash("data", "net", "train")
    return ds, ckpt, load_artefact(ckpt, h, CPU), h


def test_checkpoint_missing_key_names_the_file(tmp_path):
    cfg = _cfg()
    ds, ckpt, payload, h = _interrupted_run(tmp_path, cfg)
    del payload["optim_state_dict"], payload["config_hash"]
    save_artefact(ckpt, payload, h)
    _, net = _setup(cfg)
    with pytest.raises(ValueError, match=ckpt.name):
        train(cfg, ds, net, tmp_path)


def test_checkpoint_beyond_n_epochs_raises(tmp_path):
    cfg = _cfg()
    ds, ckpt, payload, h = _interrupted_run(tmp_path, cfg)
    payload["epoch"] = cfg.train.n_epochs + 1
    del payload["config_hash"]
    save_artefact(ckpt, payload, h)
    _, net = _setup(cfg)
    with pytest.raises(ValueError, match=ckpt.name):
        train(cfg, ds, net, tmp_path)


def test_final_with_wrong_epoch_raises(tmp_path):
    cfg = _cfg()
    ds, net = _setup(cfg)
    train(cfg, ds, net, tmp_path, log=lambda s: None)
    final = cache_paths(tmp_path).train / "final.pt"
    h = cfg.section_hash("data", "net", "train")
    payload = load_artefact(final, h, CPU)
    payload["epoch"] = 3
    del payload["config_hash"]
    save_artefact(final, payload, h)
    with pytest.raises(ValueError, match="final.pt"):
        load_trained(tmp_path, cfg, CPU, F64)


def test_cuda_checkpoint_cannot_resume_on_cpu(tmp_path):
    cfg = _cfg()
    ds, ckpt, payload, h = _interrupted_run(tmp_path, cfg)
    payload["rng_state"]["cuda"] = payload["rng_state"]["cpu"].clone()
    del payload["config_hash"]
    save_artefact(ckpt, payload, h)
    _, net = _setup(cfg)
    with pytest.raises(ValueError, match="CUDA"):
        train(cfg, ds, net, tmp_path)


def test_cpu_checkpoint_on_cuda_reports_missing_cuda_state():
    note = _restore_rng({"cpu": torch.get_rng_state()}, torch.device("cuda"))
    assert "no CUDA RNG state" in note
    assert _restore_rng({"cpu": torch.get_rng_state()}, CPU) is None


def test_train_rejects_net_with_other_dtype_than_config(tmp_path):
    cfg = _cfg()
    ds, _ = _setup(cfg)
    net32 = make_net(cfg.net, 4, 2, CPU, torch.float32)
    with pytest.raises(ValueError, match="dtype"):
        train(cfg, ds, net32, tmp_path)
