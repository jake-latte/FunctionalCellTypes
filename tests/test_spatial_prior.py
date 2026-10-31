"""The spatial weight prior in training: the factorised penalty against brute force, the
config/hash contract, smoothness of trained weights (ground truth) and training end to end."""

import hashlib
import itertools
import json
import math
from functools import partial

import pytest
import torch

from fct.cache import StaleCacheError, cache_paths, latest_checkpoint
from fct.config import (
    LOSS_VERSION,
    Config,
    DataConfig,
    NetConfig,
    TrainConfig,
    default_config,
    load_config,
    save_config,
)
from fct.data import make_dataset
from fct.model import MLP, make_net
from fct.spatial import (
    EmbeddingSpec,
    KernelSpec,
    PopulationSpec,
    SpatialConfig,
    build_spaces,
    parameter_factors,
    weight_quadratic_form,
)
from fct.train import (
    compute_loss,
    load_trained,
    make_weight_penalty,
    prior_coefficient,
    spatial_weight_penalty,
    train,
)

CPU, F64 = torch.device("cpu"), torch.float64

# ---------------------------------------------------------------- pins of runs without the prior

# Section hashes of three configs without a spatial prior, from the LOSS_VERSION 4, MODEL_VERSION 2 code: the
# spatial section must not enter them when it is unset.
SMALL = Config(
    data=DataConfig(P=16, D_in=4, d=2, embedding="linear", seed=1),
    net=NetConfig(N=8, L=2, g=1.0, phi="tanh", seed=2),
    train=TrainConfig(n_epochs=20, lr=1e-2, log_int=5, save_int=5, rate_lambda=1e-3),
    dtype="float64",
    device="cpu",
)
TRAIN = ("data", "net", "train")
ANALYSIS = ("data", "net", "train", "analysis")
HASH_PINS = [
    (default_config(), {TRAIN: "542a8cb8b9da", ANALYSIS: "bd9eb0126f20"}),
    (
        Config(train=TrainConfig(optimizer="sgld", beta=2.0)),
        {TRAIN: "90fbfc347a8b", ANALYSIS: "d15368ec62ec"},
    ),
    (SMALL, {TRAIN: "5a456f800eb7", ANALYSIS: "4eb572dcf97e"}),
]
# compute_loss on the SMALL net (prior coefficient 0.1, rate_lambda=1e-3) and the squared norm of
# each parameter's gradient, as float.hex, from the MODEL_VERSION 2 net (output V^T phi(h^L) / N).
LOSS_PIN = "0x1.f4f7158364a70p+5"
GRAD_SQ_PINS = {
    "U": "0x1.043d22892c3e3p+2",
    "V": "0x1.4b0c010b89e84p+2",
    "J.0": "0x1.1cf291f264d20p+3",
}


def _small_net_and_data():
    ds = make_dataset(SMALL.data, CPU, F64)
    net = make_net(SMALL.net, ds.x.shape[1], ds.y.shape[1], CPU, F64)
    return net, ds


def test_hashes_without_a_spatial_prior_are_unchanged():
    for cfg, pins in HASH_PINS:
        for sections, want in pins.items():
            assert cfg.section_hash(*sections) == want, sections


def test_default_loss_and_gradient_are_bit_for_bit_unchanged():
    net, ds = _small_net_and_data()
    loss, _ = compute_loss(net, ds.x, ds.y, 0.1, 1e-3)
    loss.backward()
    assert loss.item().hex() == LOSS_PIN
    for name, p in net.named_parameters():
        assert p.grad.square().sum().item().hex() == GRAD_SQ_PINS[name], name


# ---------------------------------------------------------------- the factorised penalty


def _spd(n: int, seed: int) -> torch.Tensor:
    """A random symmetric positive-definite (n, n) matrix, float64."""
    a = torch.randn(n, n, generator=torch.Generator().manual_seed(seed), dtype=F64)
    return a @ a.T + 0.5 * torch.eye(n, dtype=F64)


def _randn(*shape: int, seed: int) -> torch.Tensor:
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=F64)


def test_quadratic_form_matches_the_brute_force_kronecker_product():
    W, M_rows, M_cols = _randn(3, 4, seed=0), _spd(3, 1), _spd(4, 2)
    got = weight_quadratic_form(W, M_rows, M_cols)
    # vec(W) row-major pairs with M_rows (x) M_cols; column-major with M_cols (x) M_rows, the
    # "w^T (M^pre (x) M^post) w" of the Latex for W in (post, pre) orientation.
    row_major = W.reshape(-1) @ torch.kron(M_rows, M_cols) @ W.reshape(-1)
    col_major = W.T.reshape(-1) @ torch.kron(M_cols, M_rows) @ W.T.reshape(-1)
    assert torch.allclose(got, row_major, rtol=1e-13, atol=0)
    assert torch.allclose(got, col_major, rtol=1e-13, atol=0)


def test_quadratic_form_gradient_matches_finite_differences():
    W, M_rows, M_cols = _randn(3, 4, seed=3), _spd(3, 4), _spd(4, 5)
    W.requires_grad_(True)
    assert torch.autograd.gradcheck(lambda w: weight_quadratic_form(w, M_rows, M_cols), (W,))


def test_quadratic_form_with_identity_factors_is_the_sum_of_squares():
    W = _randn(5, 2, seed=6)
    got = weight_quadratic_form(W, torch.eye(5, dtype=F64), torch.eye(2, dtype=F64))
    assert torch.equal(got, W.square().sum())


@pytest.mark.parametrize("L", [1, 2, 3])
def test_parameter_factors_act_on_each_weight_in_its_stored_orientation(L):
    net = make_net(NetConfig(N=5, L=L, seed=0), 3, 2, CPU, F64)
    spaces = build_spaces(SpatialConfig(), m=3, N=5, L=L, n_out=2)
    factors = parameter_factors(spaces)
    assert set(factors) == {name for name, _ in net.named_parameters()}
    for name, p in net.named_parameters():
        M_rows, M_cols = factors[name]
        assert (M_rows.shape[0], M_cols.shape[0]) == tuple(p.shape), name
    # U is (hidden 1, input); V is stored (hidden L, output) = (pre, post) of the readout.
    assert factors["U"][0] is spaces[0].M_post and factors["U"][1] is spaces[0].M_pre
    assert factors["V"][0] is spaces[L].M_pre and factors["V"][1] is spaces[L].M_post


def _tiny_spatial() -> SpatialConfig:
    """Random embeddings and distinct length scales, so every factor is a dense, distinct M."""
    random = EmbeddingSpec(kind="random", seed=3)
    return SpatialConfig(
        input=PopulationSpec(embedding=random, kernel=KernelSpec(length_scale=0.3)),
        hidden=PopulationSpec(embedding=random, kernel=KernelSpec(length_scale=0.2, amplitude=2.0)),
        output=PopulationSpec(embedding=random, kernel=KernelSpec(length_scale=0.4)),
        min_rel_eig=1e-3,
    )


def _tiny_net():
    return make_net(NetConfig(N=4, L=2, g=1.0, phi="tanh", seed=5), 3, 2, CPU, F64)


def _tiny_cfg(spatial: SpatialConfig | None) -> Config:
    return Config(net=NetConfig(N=4, L=2, g=1.0, phi="tanh", seed=5), spatial=spatial)


def test_net_penalty_matches_the_brute_force_sum_over_layers():
    net = _tiny_net()
    spaces = build_spaces(_tiny_spatial(), m=3, N=4, L=2, n_out=2)
    post_pre = [net.U, net.J[0], net.V.T]  # every weight in (post, pre) orientation
    brute = sum(
        w.reshape(-1) @ torch.kron(s.M_post, s.M_pre) @ w.reshape(-1)
        for w, s in zip(post_pre, spaces, strict=True)
    )
    got = make_weight_penalty(_tiny_cfg(_tiny_spatial()), net)(net)
    assert torch.allclose(got, brute, rtol=1e-12, atol=0)


def _central_differences(loss_fn, p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """d loss / d p by central differences, one entry at a time (p is perturbed in place)."""
    fd = torch.empty_like(p)
    with torch.no_grad():
        for idx in itertools.product(*map(range, p.shape)):
            orig = p[idx].item()
            p[idx] = orig + eps
            up = loss_fn().item()
            p[idx] = orig - eps
            down = loss_fn().item()
            p[idx] = orig
            fd[idx] = (up - down) / (2 * eps)
    return fd


def test_spatial_loss_gradient_matches_finite_differences():
    net = _tiny_net()
    penalty = make_weight_penalty(_tiny_cfg(_tiny_spatial()), net)
    x, y = _randn(6, 3, seed=7), _randn(6, 2, seed=8)

    def loss_fn():
        return compute_loss(net, x, y, 0.3, 0.1, weight_penalty=penalty)[0]

    loss_fn().backward()
    for name, p in net.named_parameters():
        fd = _central_differences(loss_fn, p)
        assert torch.allclose(p.grad, fd, rtol=1e-6, atol=1e-8), name


def _loss_and_grads(net, ds, weight_penalty=MLP.weight_l2):
    net.zero_grad()
    loss, parts = compute_loss(net, ds.x, ds.y, 0.1, 1e-3, weight_penalty)
    loss.backward()
    return loss.detach(), parts, {n: p.grad.clone() for n, p in net.named_parameters()}


def test_identity_factors_reproduce_the_l2_loss_and_gradient_bit_for_bit():
    net, ds = _small_net_and_data()
    eye = {
        n: (torch.eye(p.shape[0], dtype=F64), torch.eye(p.shape[1], dtype=F64))
        for n, p in net.named_parameters()
    }
    identity = partial(spatial_weight_penalty, factors=eye)
    loss_l2, parts_l2, grads_l2 = _loss_and_grads(net, ds)
    loss_id, parts_id, grads_id = _loss_and_grads(net, ds, identity)
    assert loss_id.item().hex() == loss_l2.item().hex() == LOSS_PIN
    assert torch.equal(parts_id["weight_l2"], parts_l2["weight_l2"])
    for name in grads_l2:
        assert torch.equal(grads_id[name], grads_l2[name]), name


def test_a_kernel_far_narrower_than_the_unit_spacing_is_the_l2_norm():
    # Units at least 1/8 apart and length_scale 1e-3: the off-diagonal of K underflows to 0,
    # so K = I and M = I up to the round-off of the eigendecomposition.
    narrow = PopulationSpec(kernel=KernelSpec(length_scale=1e-3))
    spatial = SpatialConfig(input=narrow, hidden=narrow, output=narrow)
    net, ds = _small_net_and_data()
    cfg = SMALL.model_copy(update={"spatial": spatial})
    loss_l2, _, grads_l2 = _loss_and_grads(net, ds)
    loss_sp, _, grads_sp = _loss_and_grads(net, ds, make_weight_penalty(cfg, net))
    assert torch.allclose(loss_sp, loss_l2, rtol=1e-14, atol=0)
    for name in grads_l2:
        assert torch.allclose(grads_sp[name], grads_l2[name], rtol=1e-13, atol=1e-14), name


def test_make_weight_penalty_without_spatial_is_the_l2_norm():
    assert make_weight_penalty(SMALL, _small_net_and_data()[0]) is MLP.weight_l2


def test_make_weight_penalty_casts_the_factors_to_the_net_dtype():
    net64 = _tiny_net()
    net32 = _tiny_net().to(torch.float32)  # the same weights, rounded
    value = make_weight_penalty(_tiny_cfg(_tiny_spatial()), net32)(net32)
    assert value.dtype == torch.float32
    reference = make_weight_penalty(_tiny_cfg(_tiny_spatial()), net64)(net64)
    assert value.item() == pytest.approx(reference.item(), rel=1e-4)


def test_make_weight_penalty_rejects_a_parameter_without_a_factor():
    net = _tiny_net()
    net.bias = torch.nn.Parameter(torch.zeros(4, dtype=F64))
    with pytest.raises(ValueError, match="bias"):
        make_weight_penalty(_tiny_cfg(_tiny_spatial()), net)


# ---------------------------------------------------------------- config and hashes


def test_spatial_is_off_by_default_and_old_config_json_loads():
    raw = default_config().model_dump()
    assert raw.pop("spatial") is None
    assert Config.model_validate_json(json.dumps(raw)).spatial is None


def test_spatial_config_roundtrips_through_json(tmp_path):
    cfg = _tiny_cfg(_tiny_spatial())
    save_config(cfg, tmp_path / "config.json")
    assert load_config(tmp_path / "config.json") == cfg


def _with_spatial(spatial: SpatialConfig) -> Config:
    return SMALL.model_copy(update={"spatial": spatial})


def test_spatial_enters_the_train_hash_only_when_set():
    off, on = SMALL, _with_spatial(SpatialConfig())
    wider = _with_spatial(SpatialConfig(hidden=PopulationSpec(kernel=KernelSpec(length_scale=0.2))))
    assert on.section_hash("data") == off.section_hash("data")
    assert on.section_hash("data", "net") == off.section_hash("data", "net")
    assert len({c.section_hash(*TRAIN) for c in (off, on, wider)}) == 3
    assert on.section_hash(*ANALYSIS) != off.section_hash(*ANALYSIS)


def test_spatial_hash_is_canonical_json_of_the_train_section_with_the_spatial_section():
    cfg = _with_spatial(SpatialConfig())
    payload = {
        **cfg.model_dump(include={"train", "dtype"}),
        "loss_version": LOSS_VERSION,
        "spatial": SpatialConfig().model_dump(),
    }
    del payload["train"]["optimizer"], payload["train"]["beta"]  # not hashed for adam
    del payload["spatial"]["lambda_max"]  # not hashed while unset
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    assert cfg.section_hash("train") == expected


# ---------------------------------------------------------------- ground truth: smooth weights


def _roughness(W: torch.Tensor) -> float:
    """Mean squared difference between neighbouring rows of W on the ring (row k - 1 next to
    row 0), over twice the variance of W: about 1 for independent entries, near 0 for rows that
    vary smoothly around the ring."""
    neighbour_sq = (W.roll(-1, dims=0) - W).square().mean()
    return (neighbour_sq / (2 * W.var())).item()


def _smoothness_cfg(spatial: SpatialConfig | None) -> Config:
    return Config(
        data=DataConfig(P=32, D_in=4, d=2, embedding="linear", seed=1),
        net=NetConfig(N=16, L=1, g=1.0, phi="tanh", seed=2),
        train=TrainConfig(
            n_epochs=600, lr=1e-2, weight_lambda=1e-2, rate_lambda=1e-3, log_int=100, save_int=600
        ),
        spatial=spatial,
        dtype="float64",
        device="cpu",
    )


def _trained_first_layer(cfg: Config, cache_dir) -> torch.Tensor:
    ds = make_dataset(cfg.data, CPU, F64)
    net = make_net(cfg.net, ds.x.shape[1], ds.y.shape[1], CPU, F64)
    return train(cfg, ds, net, cache_dir, log=lambda s: None)[0].U.detach()


def test_a_wide_hidden_kernel_makes_the_trained_first_layer_smooth_over_the_ring(tmp_path):
    # The 16 hidden units sit on a grid ring 1/16 apart; a kernel 2.4 spacings wide puts its
    # large penalties on the rough (high-frequency) patterns across neighbouring units, so
    # training removes those first. The initial weights are independent (roughness ~ 1).
    wide = SpatialConfig(
        hidden=PopulationSpec(kernel=KernelSpec(length_scale=0.15)), min_rel_eig=1e-3
    )
    initial = make_net(_smoothness_cfg(None).net, 4, 2, CPU, F64).U.detach()
    l2 = _trained_first_layer(_smoothness_cfg(None), tmp_path / "l2")
    spatial = _trained_first_layer(_smoothness_cfg(wide), tmp_path / "spatial")
    # Absolute thresholds: iid entries give roughness ~ 1 with an SD of about 0.17 for a 16 x 4
    # layer, so 0.7 is ~2 SD below the null; the relative bound is what pins the smoothing.
    assert _roughness(initial) > 0.7 and _roughness(l2) > 0.7
    assert _roughness(spatial) < 0.3
    assert _roughness(spatial) < _roughness(l2) / 3


# ---------------------------------------------------------------- training end to end

PRIOR = SpatialConfig(hidden=PopulationSpec(kernel=KernelSpec(length_scale=0.15)), min_rel_eig=1e-3)


def _train_cfg(n_epochs=12, optimizer="adam", spatial=PRIOR, **train_kw) -> Config:
    train_kw = {"log_int": 1, "save_int": 5, "rate_lambda": 1e-3, "weight_lambda": 1e-3, **train_kw}
    tc = TrainConfig(n_epochs=n_epochs, lr=1e-2, optimizer=optimizer, beta=50.0, **train_kw)
    return SMALL.model_copy(update={"train": tc, "spatial": spatial})


def _setup(cfg):
    ds = make_dataset(cfg.data, CPU, F64)
    return ds, make_net(cfg.net, ds.x.shape[1], ds.y.shape[1], CPU, F64)


def _same_params(a, b):
    return all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


def test_adam_training_with_the_prior_is_the_plain_adam_loop_on_the_spatial_loss(tmp_path):
    cfg = _train_cfg()
    tc = cfg.train
    ds, net = _setup(cfg)
    trained, hist = train(cfg, ds, net, tmp_path, log=lambda s: None)

    ds, ref = _setup(cfg)
    penalty = make_weight_penalty(cfg, ref)
    optim = torch.optim.Adam(ref.parameters(), lr=tc.lr)
    for _ in range(tc.n_epochs):
        optim.zero_grad()
        loss, _ = compute_loss(ref, ds.x, ds.y, prior_coefficient(tc), tc.rate_lambda, penalty)
        loss.backward()
        optim.step()
    assert _same_params(trained, ref)
    assert set(hist) == {"epoch", "mse", "weight_l2", "rate_l2", "loss"}
    # The logged weight term is the prior, not the sum of squares.
    _, first = _setup(cfg)
    assert hist["weight_l2"][0] == pytest.approx(make_weight_penalty(cfg, first)(first).item())
    assert hist["weight_l2"][0] != pytest.approx(first.weight_l2().item())


@pytest.mark.parametrize("optimizer", ["adam", "sgld"])
def test_training_with_the_prior_lowers_it_and_differs_from_l2_training(tmp_path, optimizer):
    cfg = _train_cfg(n_epochs=30, optimizer=optimizer)
    ds, net = _setup(cfg)
    trained, hist = train(cfg, ds, net, tmp_path / "spatial", log=lambda s: None)
    assert all(math.isfinite(v) for v in hist["loss"])
    assert hist["weight_l2"][-1] < hist["weight_l2"][0]

    plain = _train_cfg(n_epochs=30, optimizer=optimizer, spatial=None)
    ds, plain_net = _setup(plain)
    train(plain, ds, plain_net, tmp_path / "l2", log=lambda s: None)
    assert not _same_params(trained, plain_net)


class Interrupt(Exception):
    pass


@pytest.mark.parametrize("optimizer", ["adam", "sgld"])
def test_resume_with_the_prior_equals_the_uninterrupted_run(tmp_path, optimizer):
    cfg = _train_cfg(optimizer=optimizer, save_int=3)
    ds, net_ref = _setup(cfg)
    _, hist_ref = train(cfg, ds, net_ref, tmp_path / "ref", log=lambda s: None)

    def stop(line):
        if line.startswith("[") and int(line[1:].split("/")[0]) == 7:
            raise Interrupt

    _, net_a = _setup(cfg)
    with pytest.raises(Interrupt):
        train(cfg, ds, net_a, tmp_path / "run", log=stop)
    assert latest_checkpoint(cache_paths(tmp_path / "run").train) is not None
    _, net_b = _setup(cfg)
    net_b, hist_b = train(cfg, ds, net_b, tmp_path / "run", log=lambda s: None)
    assert hist_b == hist_ref
    assert _same_params(net_ref, net_b)


def test_sgld_with_a_step_beyond_the_priors_stability_limit_fails_before_training(tmp_path):
    # The hidden-to-hidden layer of PRIOR on SMALL has lambda_max(M_post) lambda_max(M_pre)
    # ~ 4e4, so lr * weight_lambda / (2 beta) * lambda_max = 1e-2 * 1 / 100 * 4e4 = 4 (beta = 50)
    # is ~4 times the limit.
    cfg = _train_cfg(optimizer="sgld", weight_lambda=1.0)
    ds, net = _setup(cfg)
    with pytest.raises(ValueError, match="diverges on the spatial prior"):
        train(cfg, ds, net, tmp_path, log=lambda s: None)
    assert not any(tmp_path.rglob("*.pt"))
    adam = _train_cfg(optimizer="adam", weight_lambda=1.0)  # Adam rescales its steps
    assert make_weight_penalty(adam, net)(net) > 0


def test_a_run_trained_without_the_prior_is_stale_for_a_config_with_it(tmp_path):
    plain = _train_cfg(spatial=None)
    ds, net = _setup(plain)
    train(plain, ds, net, tmp_path, log=lambda s: None)
    with pytest.raises(StaleCacheError):
        load_trained(tmp_path, _train_cfg(), CPU, F64)
