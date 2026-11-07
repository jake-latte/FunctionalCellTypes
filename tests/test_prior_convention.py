"""The weight prior enters the loss as weight_lambda / (2 beta) times its quadratic form, and the
rate penalty as rate_lambda / (2 beta) times sum phi^2, with beta = train.beta under SGLD and 1
under Adam, so that SGLD's stationary law exp(-beta L) has the prior N(0, K / weight_lambda) and
the rate tilt (rate_lambda / 2) sum phi^2 for every beta. The tests build the penalty by hand."""

import pytest
import torch

import fct.config
from fct.cache import StaleCacheError
from fct.config import (
    LOSS_VERSION,
    RATE_FREE_LOSS_VERSION,
    Config,
    NetConfig,
    TrainConfig,
    hashed_loss_version,
    law_beta,
)
from fct.spatial import build_spaces
from fct.train import (
    LossCoefficients,
    compute_loss,
    load_trained,
    loss_coefficients,
    make_weight_penalty,
    prior_coefficient,
    rate_coefficient,
    train,
)
from tests.test_spatial_prior import _randn, _tiny_cfg, _tiny_net, _tiny_spatial, _train_cfg
from tests.test_train import _cfg, _setup

CPU, F64 = torch.device("cpu"), torch.float64
_COMMON = {"weight_lambda": 0.3, "rate_lambda": 0.2, "lr": 1e-4}
OPTIMISERS = {
    "adam": TrainConfig(optimizer="adam", beta=7.0, **_COMMON),  # beta is ignored
    "sgld_beta4": TrainConfig(optimizer="sgld", beta=4.0, **_COMMON),
    "sgld_beta_quarter": TrainConfig(optimizer="sgld", beta=0.25, **_COMMON),
}
# The beta of the stationary law exp(-beta L): train.beta under SGLD, 1 under Adam.
LAW_BETA = {"adam": 1.0, "sgld_beta4": 4.0, "sgld_beta_quarter": 0.25}
# weight_lambda / (2 beta), computed by hand for the three settings above
HAND_COEFFICIENTS = {"adam": 0.15, "sgld_beta4": 0.0375, "sgld_beta_quarter": 0.6}
# rate_lambda / (2 beta), computed by hand for the three settings above
HAND_RATE_COEFFICIENTS = {"adam": 0.1, "sgld_beta4": 0.025, "sgld_beta_quarter": 0.4}


@pytest.mark.parametrize("name", OPTIMISERS)
def test_the_prior_coefficient_is_weight_lambda_over_two_beta(name):
    assert prior_coefficient(OPTIMISERS[name]) == pytest.approx(HAND_COEFFICIENTS[name])


@pytest.mark.parametrize("name", OPTIMISERS)
def test_the_rate_coefficient_is_rate_lambda_over_two_beta(name):
    assert rate_coefficient(OPTIMISERS[name]) == pytest.approx(HAND_RATE_COEFFICIENTS[name])


@pytest.mark.parametrize("name", OPTIMISERS)
def test_the_law_beta_is_train_beta_under_sgld_and_one_under_adam(name):
    tc = OPTIMISERS[name]
    assert law_beta(tc.optimizer, tc.beta) == LAW_BETA[name]


def test_the_law_beta_refuses_an_unknown_optimizer():
    with pytest.raises(ValueError, match="optimizer"):
        law_beta("sgd", 4.0)


@pytest.mark.parametrize("name", OPTIMISERS)
def test_the_loss_coefficients_are_the_prior_and_rate_coefficients(name):
    coefficients = loss_coefficients(OPTIMISERS[name])
    assert coefficients.prior == pytest.approx(HAND_COEFFICIENTS[name])
    assert coefficients.rate == pytest.approx(HAND_RATE_COEFFICIENTS[name])


def test_the_loss_refuses_bare_numbers_for_its_coefficients():
    """A bare (prior, rate) pair could be (weight_lambda, rate_lambda) unscaled; the loss takes
    only `LossCoefficients`, which `loss_coefficients` builds from the train config."""
    net = _tiny_net()
    x, y = _randn(5, 3, seed=1), _randn(5, 2, seed=2)
    with pytest.raises(TypeError, match="LossCoefficients"):
        compute_loss(net, x, y, (0.3, 0.2))


@pytest.mark.parametrize("name", OPTIMISERS)
def test_beta_times_the_loss_is_the_action_of_the_notes(name):
    """beta L = (beta N / 2) mse + (weight_lambda / 2) form + (rate_lambda / 2) sum phi^2 at fixed
    weights: rate_lambda is the notes' r and weight_lambda their prior precision, for every beta."""
    tc, beta = OPTIMISERS[name], LAW_BETA[name]
    net = _tiny_net()
    x, y = _randn(5, 3, seed=1), _randn(5, 2, seed=2)
    loss, _ = compute_loss(net, x, y, loss_coefficients(tc))
    with torch.no_grad():
        h, z = net(x)
        mse = (y - z).square().mean().item()
        form = sum(p.square().sum().item() for p in net.parameters())
        sum_phi2 = net.phi(h).square().sum().item()
    action = (
        0.5 * beta * net.N * mse + 0.5 * tc.weight_lambda * form + 0.5 * tc.rate_lambda * sum_phi2
    )
    assert beta * loss.item() == pytest.approx(action, rel=1e-12)


@pytest.mark.parametrize("name", OPTIMISERS)
def test_the_l2_prior_term_is_the_coefficient_times_the_sum_of_squares(name):
    tc = OPTIMISERS[name]
    net = _tiny_net()
    cfg = Config(net=NetConfig(N=4, L=2, g=1.0, phi="tanh", seed=5), train=tc)
    hand = HAND_COEFFICIENTS[name] * sum(p.square().sum().item() for p in net.parameters())
    penalty = make_weight_penalty(cfg, net)(net)
    assert prior_coefficient(tc) * penalty.item() == pytest.approx(hand, rel=1e-12)
    x, y = _randn(5, 3, seed=1), _randn(5, 2, seed=2)
    loss, _ = compute_loss(net, x, y, LossCoefficients(prior=prior_coefficient(tc), rate=0.0))
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
    first, _ = compute_loss(net, ds.x, ds.y, LossCoefficients(0.3 / 8, cfg.train.rate_lambda / 8))
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


def test_a_run_made_under_loss_version_4_with_a_rate_penalty_is_stale(tmp_path, monkeypatch):
    assert LOSS_VERSION == 5
    cfg = _cfg()  # rate_lambda = 1e-3: version 5 halves its rate penalty under Adam
    ds, net = _setup(cfg)
    with monkeypatch.context() as old:
        old.setattr(fct.config, "LOSS_VERSION", 4)
        train(cfg, ds, net, tmp_path, log=lambda s: None)
    with pytest.raises(StaleCacheError):
        load_trained(tmp_path, cfg, CPU, F64)


def test_a_run_made_under_loss_version_4_without_a_rate_penalty_still_loads(tmp_path, monkeypatch):
    assert RATE_FREE_LOSS_VERSION == 4
    cfg = _cfg(rate_lambda=0.0)  # version 5 changes no number of this run
    ds, net = _setup(cfg)
    with monkeypatch.context() as old:
        old.setattr(fct.config, "LOSS_VERSION", 4)
        train(cfg, ds, net, tmp_path, log=lambda s: None)
    load_trained(tmp_path, cfg, CPU, F64)


# ---------------------------------------------------------------- the rate penalty's hash rule

TRAIN = ("data", "net", "train")
RESULT = ("data", "net", "train", "meanfield")
RATE_FREE = {
    "adam": Config(train=TrainConfig(rate_lambda=0.0)),
    "sgld": Config(train=TrainConfig(rate_lambda=0.0, optimizer="sgld", beta=4.0)),
}
# Hashes of the configs above from the LOSS_VERSION 4, MODEL_VERSION 2 code, before the rate
# penalty was scaled by 1 / (2 beta): a config without a rate penalty must keep them.
RATE_FREE_PINS = {
    "adam": {TRAIN: "bfb7ec92fda0", RESULT: "d5b83dea0258"},
    "sgld": {TRAIN: "91f4d1cbc870", RESULT: "f270bb8e6903"},
}
WITH_RATE = {
    "adam": Config(),  # rate_lambda = 1e-3
    "sgld": Config(train=TrainConfig(optimizer="sgld", beta=4.0)),
}
# Hashes of the configs above from that same LOSS_VERSION 4 code: their loss changed, so must they.
WITH_RATE_VERSION_4 = {
    "adam": {TRAIN: "97e1aa717ba3", RESULT: "feeae8d33c73"},
    "sgld": {TRAIN: "94520fc857f8", RESULT: "577f88ef4903"},
}


@pytest.mark.parametrize("name", RATE_FREE)
def test_a_config_without_a_rate_penalty_keeps_its_hashes(name):
    cfg = RATE_FREE[name]
    assert {s: cfg.section_hash(*s) for s in RATE_FREE_PINS[name]} == RATE_FREE_PINS[name]


@pytest.mark.parametrize("name", WITH_RATE)
def test_a_config_with_a_rate_penalty_has_new_hashes(name):
    cfg = WITH_RATE[name]
    for sections, old in WITH_RATE_VERSION_4[name].items():
        assert cfg.section_hash(*sections) != old


@pytest.mark.parametrize("name", WITH_RATE)
def test_a_config_with_a_rate_penalty_hashed_under_version_4_is_the_old_hash(name, monkeypatch):
    """Version 4 is the only difference: the rule moves nothing else in the hash."""
    monkeypatch.setattr(fct.config, "LOSS_VERSION", 4)
    cfg = WITH_RATE[name]
    assert {s: cfg.section_hash(*s) for s in WITH_RATE_VERSION_4[name]} == (
        WITH_RATE_VERSION_4[name]
    )


@pytest.mark.parametrize(
    ("rate_lambda", "version"),
    [(0.0, RATE_FREE_LOSS_VERSION), (1e-3, LOSS_VERSION), (2.0, LOSS_VERSION)],
)
def test_only_a_config_with_a_rate_penalty_hashes_the_current_loss_version(rate_lambda, version):
    assert hashed_loss_version(TrainConfig(optimizer="sgld", rate_lambda=rate_lambda)) == version
