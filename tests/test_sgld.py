"""SGLD: the optimiser on a quadratic (ground truth), the config/hash contract, and training."""

import hashlib
import json
import math

import pytest
import torch
from pydantic import ValidationError

from fct.cache import cache_paths, latest_checkpoint
from fct.config import LOSS_VERSION, Config, DataConfig, NetConfig, TrainConfig, default_config
from fct.data import make_dataset
from fct.model import make_net
from fct.sgld import SGLD
from fct.train import compute_loss, loss_coefficients, train

CPU, F64 = torch.device("cpu"), torch.float64

# Hashes of the default config (adam), which the SGLD-only fields must not move. They are those
# of LOSS_VERSION 5 (the default rate_lambda is > 0, so version 5 is hashed), MODEL_VERSION 2
# without `meanfield.readout` and with the latent interval as `latent_min` / `latent_max`, the
# meanfield one with the solver's step control (`config.STEP_CONTROL_FIELDS`); at MODEL_VERSION 1
# the train hash is the one written before the optimiser fields existed (test_spatial_prior
# pins it).
DEFAULT_TRAIN_HASH = "629c7b68d953"
DEFAULT_TRAIN_MEANFIELD_HASH = "1d7593e6a04d"


# ---------------------------------------------------------------- the optimiser itself


def _run_quadratic(curvature, beta, lr, n_steps, n_chains, seed=0):
    """n_chains independent one-parameter chains on L = curvature * theta^2 / 2, held in one
    tensor; returns the final states and the mean squared displacement of the last step."""
    torch.manual_seed(seed)
    theta = torch.zeros(n_chains, dtype=F64, requires_grad=True)
    opt = SGLD([theta], lr=lr, beta=beta)
    for _ in range(n_steps):
        opt.zero_grad()
        (0.5 * curvature * theta.square().sum()).backward()
        opt.step()
    return theta.detach(), opt.last_msd


@pytest.mark.parametrize("curvature, beta", [(1.0, 1.0), (4.0, 0.5), (0.25, 8.0)])
def test_quadratic_samples_have_variance_one_over_beta_curvature(curvature, beta):
    lr = 0.01 / curvature
    n_chains = 40_000
    theta, _ = _run_quadratic(curvature, beta, lr, n_steps=2_000, n_chains=n_chains)
    # Euler-Maruyama on an OU process has the exact stationary variance below, which differs
    # from 1 / (beta * curvature) by the discretisation bias lr * curvature / 2 (relative).
    exact = 1.0 / (beta * curvature * (1 - lr * curvature / 2))
    mc_err = exact * math.sqrt(2 / n_chains)
    assert theta.var().item() == pytest.approx(exact, abs=4 * mc_err)


def test_quadratic_mean_is_zero_within_error():
    theta, _ = _run_quadratic(1.0, 2.0, 0.01, n_steps=2_000, n_chains=40_000)
    sem = math.sqrt(0.5 / 40_000)
    assert abs(theta.mean().item()) < 4 * sem


def test_mean_squared_displacement_is_the_noise_variance_in_the_noise_dominated_limit():
    beta, lr = 3.0, 1e-3
    _, msd = _run_quadratic(curvature=1.0, beta=beta, lr=lr, n_steps=3_000, n_chains=40_000)
    # displacement = -lr*c*theta + sqrt(2 lr/beta) xi with theta ~ N(0, 1/(beta c)), so
    # msd = 2 lr / beta + lr^2 c / beta; the second term is a factor lr/2 smaller.
    assert msd.item() == pytest.approx(2 * lr / beta, rel=0.03)


def test_step_is_gradient_descent_plus_scaled_noise():
    lr, beta = 0.1, 4.0
    theta = torch.tensor([1.0, -2.0, 0.5], dtype=F64, requires_grad=True)
    opt = SGLD([theta], lr=lr, beta=beta)
    (theta**3).sum().backward()
    grad = theta.grad.clone()
    before = theta.detach().clone()
    torch.manual_seed(7)
    opt.step()
    torch.manual_seed(7)
    noise = torch.randn(3, dtype=F64)
    expected = before - lr * grad + math.sqrt(2 * lr / beta) * noise
    assert torch.allclose(theta.detach(), expected, rtol=0, atol=1e-15)


def test_step_noise_comes_from_the_global_generator():
    def run(seed):
        theta = torch.zeros(5, dtype=F64, requires_grad=True)
        opt = SGLD([theta], lr=0.1, beta=1.0)
        theta.grad = torch.zeros_like(theta)
        torch.manual_seed(seed)
        opt.step()
        return theta.detach()

    assert torch.equal(run(1), run(1))
    assert not torch.equal(run(1), run(2))


def test_step_returns_loss_of_closure():
    theta = torch.ones(2, dtype=F64, requires_grad=True)
    opt = SGLD([theta], lr=0.1, beta=1.0)

    def closure():
        opt.zero_grad()
        loss = theta.square().sum()
        loss.backward()
        return loss

    assert opt.step(closure).item() == 2.0


@pytest.mark.parametrize("kw", [{"lr": 0.0}, {"lr": -1.0}, {"beta": 0.0}, {"beta": -2.0}])
def test_optimiser_rejects_non_positive_lr_and_beta(kw):
    theta = torch.zeros(1, requires_grad=True)
    with pytest.raises(ValueError):
        SGLD([theta], **{"lr": 0.1, "beta": 1.0, **kw})


def test_state_dict_roundtrip_keeps_lr_and_beta():
    theta = torch.zeros(1, requires_grad=True)
    opt = SGLD([theta], lr=0.1, beta=2.0)
    other = SGLD([theta], lr=0.5, beta=9.0)
    other.load_state_dict(opt.state_dict())
    assert other.param_groups[0]["lr"] == 0.1 and other.param_groups[0]["beta"] == 2.0


# ---------------------------------------------------------------- config and hashes


def test_default_optimiser_is_adam_and_hashes_are_unchanged():
    cfg = default_config()
    assert cfg.train.optimizer == "adam"
    assert cfg.section_hash("data", "net", "train") == DEFAULT_TRAIN_HASH
    assert cfg.section_hash("data", "net", "train", "meanfield") == DEFAULT_TRAIN_MEANFIELD_HASH


def test_beta_is_ignored_and_not_hashed_for_adam():
    a, b = default_config(), default_config()
    b.train.beta = 123.0
    assert a.section_hash("data", "net", "train") == b.section_hash("data", "net", "train")


def test_sgld_hash_depends_on_optimiser_and_beta():
    sections = ("data", "net", "train")
    adam = default_config()
    sgld = Config(train=TrainConfig(optimizer="sgld", beta=2.0))
    other_beta = Config(train=TrainConfig(optimizer="sgld", beta=3.0))
    hashes = {c.section_hash(*sections) for c in (adam, sgld, other_beta)}
    assert len(hashes) == 3


def test_sgld_hash_is_canonical_json_of_the_train_section_with_optimiser_fields():
    cfg = Config(train=TrainConfig(optimizer="sgld", beta=2.0))
    payload = {**cfg.model_dump(include={"train", "dtype"}), "loss_version": LOSS_VERSION}
    assert payload["train"]["optimizer"] == "sgld" and payload["train"]["beta"] == 2.0
    del payload["train"]["monitor_kind"]  # the MIR monitor: the monitor kind is not hashed
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    assert cfg.section_hash("train") == expected


def test_optimiser_field_is_validated():
    with pytest.raises(ValidationError, match="optimizer"):
        TrainConfig(optimizer="sgd")
    for bad in (0.0, -1.0):
        with pytest.raises(ValidationError, match="beta"):
            TrainConfig(beta=bad)


def test_config_without_the_new_fields_still_loads():
    raw = default_config().model_dump()
    del raw["train"]["optimizer"], raw["train"]["beta"]
    cfg = Config.model_validate_json(json.dumps(raw))
    assert cfg.train.optimizer == "adam"


# ---------------------------------------------------------------- training


def _cfg(n_epochs=20, optimizer="sgld", beta=50.0, **train_kw) -> Config:
    train_kw = {"log_int": 5, "save_int": 5, "rate_lambda": 1e-3, **train_kw}
    return Config(
        data=DataConfig(P=16, D_in=4, d=2, embedding="linear", seed=1),
        net=NetConfig(N=8, L=2, g=1.0, phi="tanh", seed=2),
        train=TrainConfig(n_epochs=n_epochs, lr=1e-2, optimizer=optimizer, beta=beta, **train_kw),
        dtype="float64",
        device="cpu",
    )


def _setup(cfg):
    ds = make_dataset(cfg.data, CPU, F64)
    net = make_net(cfg.net, ds.x.shape[1], ds.y.shape[1], CPU, F64)
    return ds, net


def _same_params(a, b):
    return all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


def test_adam_training_is_bit_for_bit_the_plain_adam_loop(tmp_path):
    cfg = _cfg(optimizer="adam")
    ds, net = _setup(cfg)
    trained, hist = train(cfg, ds, net, tmp_path, log=lambda s: None)

    ds, ref = _setup(cfg)
    optim = torch.optim.Adam(ref.parameters(), lr=cfg.train.lr)
    for _ in range(cfg.train.n_epochs):
        optim.zero_grad()
        loss, _ = compute_loss(ref, ds.x, ds.y, loss_coefficients(cfg.train))
        loss.backward()
        optim.step()
    assert _same_params(trained, ref)
    assert set(hist) == {"epoch", "mse", "weight_l2", "rate_l2", "loss"}


def test_sgld_training_records_diagnostics_and_differs_from_adam(tmp_path):
    cfg = _cfg()
    ds, net = _setup(cfg)
    trained, hist = train(cfg, ds, net, tmp_path / "sgld", log=lambda s: None)
    assert {"grad_norm", "param_msd"} <= set(hist)
    n_logged = len(hist["epoch"])
    assert len(hist["grad_norm"]) == len(hist["param_msd"]) == n_logged > 0
    assert all(g > 0 for g in hist["grad_norm"]) and all(m > 0 for m in hist["param_msd"])

    adam_cfg = _cfg(optimizer="adam")
    ds, adam_net = _setup(adam_cfg)
    train(adam_cfg, ds, adam_net, tmp_path / "adam", log=lambda s: None)
    assert not _same_params(trained, adam_net)


def test_sgld_noise_follows_the_training_seed(tmp_path):
    def final(seed, sub):
        cfg = _cfg(seed=seed)
        ds, net = _setup(cfg)
        return train(cfg, ds, net, tmp_path / sub, log=lambda s: None)[0]

    assert _same_params(final(0, "a"), final(0, "b"))
    assert not _same_params(final(0, "c"), final(1, "d"))


def test_sgld_is_gradient_descent_as_beta_grows(tmp_path):
    cfg = _cfg(beta=1e30, n_epochs=10)
    ds, net = _setup(cfg)
    trained, _ = train(cfg, ds, net, tmp_path, log=lambda s: None)

    ds, ref = _setup(cfg)
    for _ in range(10):
        ref.zero_grad()
        loss, _ = compute_loss(ref, ds.x, ds.y, loss_coefficients(cfg.train))
        loss.backward()
        with torch.no_grad():
            for p in ref.parameters():
                p -= cfg.train.lr * p.grad
    for k, v in trained.state_dict().items():
        assert torch.allclose(v, ref.state_dict()[k], atol=1e-12)


class Interrupt(Exception):
    pass


@pytest.mark.parametrize("interrupt_at, save_int", [(10, 5), (7, 3)])
def test_sgld_resume_equals_uninterrupted_run(tmp_path, interrupt_at, save_int):
    cfg = _cfg(12, save_int=save_int, log_int=1)
    ds, net_ref = _setup(cfg)
    _, hist_ref = train(cfg, ds, net_ref, tmp_path / "ref", log=lambda s: None)

    def stop(line):
        if line.startswith("[") and int(line[1:].split("/")[0]) == interrupt_at:
            raise Interrupt

    _, net_a = _setup(cfg)
    with pytest.raises(Interrupt):
        train(cfg, ds, net_a, tmp_path / "run", log=stop)
    assert latest_checkpoint(cache_paths(tmp_path / "run").train) is not None
    _, net_b = _setup(cfg)
    net_b, hist_b = train(cfg, ds, net_b, tmp_path / "run", log=lambda s: None)
    assert hist_b == hist_ref
    assert _same_params(net_ref, net_b)
