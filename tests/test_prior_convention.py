"""The weight prior enters the loss as weight_lambda / (2 beta) times its quadratic form, with
beta = train.beta under SGLD and 1 under Adam, so that SGLD's stationary law exp(-beta L) has the
prior N(0, K / weight_lambda) for every beta. The tests build the penalty by hand."""

import pytest
import torch

import fct.config
from fct.cache import StaleCacheError
from fct.config import LOSS_VERSION, Config, NetConfig, TrainConfig
from fct.spatial import build_spaces
from fct.train import compute_loss, load_trained, make_weight_penalty, prior_coefficient, train
from tests.test_spatial_prior import _randn, _tiny_cfg, _tiny_net, _tiny_spatial, _train_cfg
from tests.test_train import _cfg, _setup

CPU, F64 = torch.device("cpu"), torch.float64
OPTIMISERS = {
    "adam": TrainConfig(optimizer="adam", weight_lambda=0.3, beta=7.0, lr=1e-4),  # beta is ignored
    "sgld_beta4": TrainConfig(optimizer="sgld", weight_lambda=0.3, beta=4.0, lr=1e-4),
    "sgld_beta_quarter": TrainConfig(optimizer="sgld", weight_lambda=0.3, beta=0.25, lr=1e-4),
}
# weight_lambda / (2 beta), computed by hand for the three settings above
HAND_COEFFICIENTS = {"adam": 0.15, "sgld_beta4": 0.0375, "sgld_beta_quarter": 0.6}


@pytest.mark.parametrize("name", OPTIMISERS)
def test_the_prior_coefficient_is_weight_lambda_over_two_beta(name):
    assert prior_coefficient(OPTIMISERS[name]) == pytest.approx(HAND_COEFFICIENTS[name])


@pytest.mark.parametrize("name", OPTIMISERS)
def test_the_l2_prior_term_is_the_coefficient_times_the_sum_of_squares(name):
    tc = OPTIMISERS[name]
    net = _tiny_net()
    cfg = Config(net=NetConfig(N=4, L=2, g=1.0, phi="tanh", seed=5), train=tc)
    hand = HAND_COEFFICIENTS[name] * sum(p.square().sum().item() for p in net.parameters())
    penalty = make_weight_penalty(cfg, net)(net)
    assert prior_coefficient(tc) * penalty.item() == pytest.approx(hand, rel=1e-12)
    x, y = _randn(5, 3, seed=1), _randn(5, 2, seed=2)
    loss, _ = compute_loss(net, x, y, prior_coefficient(tc), 0.0)
    data = 0.5 * net.N * (y - net(x)[1]).square().mean().item()
    assert loss.item() == pytest.approx(data + hand, rel=1e-12)


@pytest.mark.parametrize("name", OPTIMISERS)
def test_the_spatial_prior_term_is_the_coefficient_times_the_kronecker_form(name):
    tc = OPTIMISERS[name]
    net = _tiny_net()
    cfg = _tiny_cfg(_tiny_spatial()).model_copy(update={"train": tc})
    spaces = build_spaces(_tiny_spatial(), m=3, N=4, L=2, n_out=2)
    post_pre = [net.U, net.J[0], net.V.T]  # every weight in (post, pre) orientation
    form = sum(
        w.reshape(-1) @ torch.kron(s.M_post, s.M_pre) @ w.reshape(-1)
        for w, s in zip(post_pre, spaces, strict=True)
    )
    penalty = make_weight_penalty(cfg, net)(net)
    assert prior_coefficient(tc) * penalty.item() == pytest.approx(
        HAND_COEFFICIENTS[name] * form.item(), rel=1e-12
    )


def test_training_minimises_the_loss_with_that_coefficient(tmp_path):
    """The loss logged at the first epoch is the initial loss under the convention."""
    cfg = _cfg(n_epochs=1, log_int=1, save_int=1, optimizer="sgld", beta=4.0, weight_lambda=0.3)
    ds, net = _setup(cfg)
    first, _ = compute_loss(net, ds.x, ds.y, 0.3 / 8, cfg.train.rate_lambda)
    _, hist = train(cfg, ds, net, tmp_path, log=lambda s: None)
    assert hist["loss"][0] == pytest.approx(first.item(), rel=1e-12)


def _sgld_step_is_accepted(cfg: Config) -> bool:
    """True if `make_weight_penalty` accepts the config's SGLD step."""
    _, net = _setup(cfg)
    try:
        make_weight_penalty(cfg, net)
    except ValueError:
        return False
    return True


def test_the_sgld_stability_bound_is_lr_times_the_scaled_coefficient_times_lambda_max():
    """The prior (weight_lambda / 2 beta) M has curvature (weight_lambda / beta) lambda_max, so a
    larger beta softens it: lambda_max ~ 4e4 here (see the spatial prior tests), lr = 1e-2,
    weight_lambda = 1 and beta = 50 give lr * coefficient * lambda_max = 4 >= 1 (rejected), and
    beta = 500 gives 0.4 (accepted)."""
    base = _train_cfg(optimizer="sgld", weight_lambda=1.0)  # beta = 50
    assert not _sgld_step_is_accepted(base)
    softer = base.model_copy(update={"train": base.train.model_copy(update={"beta": 500.0})})
    assert _sgld_step_is_accepted(softer)


def test_a_run_made_under_loss_version_3_is_stale(tmp_path, monkeypatch):
    assert LOSS_VERSION == 4
    cfg = _cfg()
    ds, net = _setup(cfg)
    with monkeypatch.context() as old:
        old.setattr(fct.config, "LOSS_VERSION", 3)
        train(cfg, ds, net, tmp_path, log=lambda s: None)
    with pytest.raises(StaleCacheError):
        load_trained(tmp_path, cfg, CPU, F64)
