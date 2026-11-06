"""The whitened parametrisation of the spatial prior (`spatial.parametrisation = "whitened"`):
the trained parameters are xi with W = S_post xi S_pre, S = K^{1/2}, and the penalty is the
plain |xi|^2, so the sampled prior is K itself (no eigenvalue floor). Ground truth throughout:
the prior covariance (K_post (x) K_pre) / weight_lambda, which the penalty path only
approximates."""

import pytest
import torch

from fct.config import Config, NetConfig, TrainConfig
from fct.data import make_dataset
from fct.model import MLP, EnsembleMLP, effective_weights, make_net
from fct.sgld import SGLD
from fct.spatial import (
    EmbeddingSpec,
    KernelSpec,
    PopulationSpec,
    SpatialConfig,
    build_spaces,
    floored_square_root,
    floored_whitening_factors,
    parameter_factors,
    penalty_matrix,
    periodic_gaussian_kernel,
    whitening_factors,
    whitening_matrix,
)
from fct.train import (
    _check_sgld_step_resolves_the_prior,
    _step,
    compute_loss,
    largest_hessian_eigenvalue,
    load_latest_checkpoint,
    load_trained,
    make_net_from_config,
    make_weight_penalty,
    prior_coefficient,
    prior_stiffness,
    train,
    train_from_config,
)
from tests.pinned_training import pinned_config
from tests.test_ensemble import _member_noise, _single_noise, _with_B
from tests.test_lambda_max import PINNED_SPATIAL_HASH, _pinned_spatial

CPU, F64 = torch.device("cpu"), torch.float64
TRAIN = ("data", "net", "train")


def _ring_kernel(n: int, length_scale: float) -> torch.Tensor:
    x = torch.arange(n, dtype=F64)[:, None] / n
    return periodic_gaussian_kernel(x, x, length_scale, 1.0, 1.0)


def _whitened(spatial: SpatialConfig | None = None) -> SpatialConfig:
    return (spatial or SpatialConfig()).model_copy(update={"parametrisation": "whitened"})


PRIOR_INIT = NetConfig(init="prior")


# ---------------------------------------------------------------- config and hash


def test_the_parametrisation_defaults_to_the_penalty_and_an_old_config_loads():
    assert SpatialConfig().parametrisation == "penalty"
    raw = SpatialConfig().model_dump()
    del raw["parametrisation"]
    assert SpatialConfig.model_validate(raw).parametrisation == "penalty"
    with pytest.raises(ValueError, match="parametrisation"):
        SpatialConfig(parametrisation="other")


def test_a_config_is_whitened_only_with_a_spatial_section_under_whitened():
    assert not Config().is_whitened
    assert not Config(spatial=SpatialConfig()).is_whitened
    assert Config(spatial=_whitened(), net=PRIOR_INIT).is_whitened


def test_net_init_defaults_to_standard_and_an_old_config_loads():
    assert NetConfig().init == "standard"
    raw = NetConfig().model_dump()
    del raw["init"]
    assert NetConfig.model_validate(raw).init == "standard"
    with pytest.raises(ValueError, match="init"):
        NetConfig(init="other")


def test_a_whitened_config_must_start_from_its_prior():
    with pytest.raises(ValueError, match=r'net\.init="prior"'):
        Config(spatial=_whitened())
    assert Config(spatial=_whitened(), net=PRIOR_INIT).net.init == "prior"
    assert Config(spatial=SpatialConfig(), net=PRIOR_INIT).net.init == "prior"  # either path
    assert Config(net=PRIOR_INIT).net.init == "prior"  # and without a spatial prior


def test_net_init_enters_the_hash_only_as_prior():
    base = Config(spatial=_pinned_spatial())
    explicit = base.model_copy(update={"net": NetConfig(init="standard")})
    assert explicit.section_hash(*TRAIN) == PINNED_SPATIAL_HASH
    prior = base.model_copy(update={"net": PRIOR_INIT})
    assert prior.section_hash(*TRAIN) != PINNED_SPATIAL_HASH
    assert prior.section_hash("data") == base.section_hash("data")


def test_the_penalty_parametrisation_keeps_the_pinned_hash_and_whitened_changes_it():
    base = Config(spatial=_pinned_spatial())
    assert base.section_hash(*TRAIN) == PINNED_SPATIAL_HASH
    explicit = _pinned_spatial().model_copy(update={"parametrisation": "penalty"})
    assert base.model_copy(update={"spatial": explicit}).section_hash(*TRAIN) == PINNED_SPATIAL_HASH
    prior = base.model_copy(update={"net": PRIOR_INIT})
    whitened = prior.model_copy(update={"spatial": _whitened(_pinned_spatial())})
    assert whitened.section_hash(*TRAIN) != prior.section_hash(*TRAIN)
    assert whitened.section_hash("data", "net") == prior.section_hash("data", "net")


@pytest.mark.parametrize("spatial", [None, SpatialConfig(), _whitened()])
def test_a_prior_draw_needs_a_positive_weight_lambda(spatial):
    with pytest.raises(ValueError, match="weight_lambda"):
        Config(spatial=spatial, net=PRIOR_INIT, train=TrainConfig(weight_lambda=0.0))
    cfg = Config(spatial=spatial, net=PRIOR_INIT, train=TrainConfig(weight_lambda=0.5))
    assert cfg.train.weight_lambda == 0.5


def test_the_standard_draw_keeps_accepting_a_zero_weight_lambda():
    assert Config(spatial=SpatialConfig(), train=TrainConfig(weight_lambda=0.0)).spatial


# ---------------------------------------------------------------- the square root


def test_whitening_matrix_is_the_symmetric_psd_square_root():
    K = _ring_kernel(8, 0.2)
    S = whitening_matrix(K)
    assert S.dtype == F64 and S.shape == (8, 8)
    assert torch.allclose(S, S.T, rtol=0, atol=1e-14)
    assert torch.allclose(S @ S, K, rtol=0, atol=1e-12)
    assert torch.linalg.eigvalsh(S).min() >= 0


def test_whitening_matrix_clamps_round_off_negative_eigenvalues_without_a_floor():
    # The eigenvalues of this ring kernel span 1 down to far below round-off, so some come
    # out slightly negative; a plain sqrt would give NaN. No floor: S S recovers K itself.
    K = _ring_kernel(16, 0.25)
    assert torch.linalg.eigvalsh(K).min() < 1e-12
    S = whitening_matrix(K)
    assert torch.isfinite(S).all()
    assert torch.allclose(S @ S, K, rtol=0, atol=1e-12)


def test_whitening_matrix_rejects_a_clearly_negative_kernel():
    K = torch.diag(torch.tensor([1.0, -0.5], dtype=F64))
    with pytest.raises(ValueError, match="negative"):
        whitening_matrix(K)


def test_the_floored_square_root_is_that_of_the_inverse_penalty():
    """S_floored S_floored = M^{-1}: the kernel the penalty encodes, its eigenvalues floored at
    min_rel_eig times the largest."""
    K = _ring_kernel(8, 0.25)
    M, report = penalty_matrix(K, 1e-2)
    assert report.n_floored > 0
    S = floored_square_root(K, 1e-2)
    assert torch.allclose(S, S.T, rtol=0, atol=1e-14)
    assert torch.allclose(S @ S, torch.linalg.inv(M), rtol=0, atol=1e-10)
    assert not torch.allclose(S @ S, K, rtol=0, atol=1e-4)  # the floor shows


def test_the_floored_square_root_is_the_square_root_bit_for_bit_when_nothing_is_floored():
    K = _ring_kernel(8, 0.1)
    assert penalty_matrix(K, 1e-6)[1].n_floored == 0
    assert torch.equal(floored_square_root(K, 1e-6), whitening_matrix(K))


def test_floored_whitening_factors_follow_the_stored_orientation():
    spaces = _spaces(2)
    floored = floored_whitening_factors(spaces)
    rows, cols = floored["V"]
    assert torch.equal(rows, spaces[-1].pre.S_floored)
    assert torch.equal(cols, spaces[-1].post.S_floored)
    assert torch.equal(floored["U"][0], spaces[0].post.S_floored)


def test_populations_carry_the_square_root_of_their_kernel():
    spaces = build_spaces(SpatialConfig(), m=3, N=5, L=2, n_out=2)
    pops = [spaces[0].pre] + [s.post for s in spaces]
    for pop in pops:
        assert torch.allclose(pop.S @ pop.S, pop.K, rtol=0, atol=1e-12)


@pytest.mark.parametrize("L", [1, 2, 3])
def test_whitening_factors_follow_the_stored_orientation_of_parameter_factors(L):
    spaces = build_spaces(SpatialConfig(), m=3, N=5, L=L, n_out=2)
    factors, penalties = whitening_factors(spaces), parameter_factors(spaces)
    assert set(factors) == set(penalties)
    for name in factors:
        for S, M in zip(factors[name], penalties[name], strict=True):
            assert S.shape == M.shape, name
    assert factors["U"][0] is spaces[0].post.S and factors["U"][1] is spaces[0].pre.S
    assert factors["V"][0] is spaces[L].pre.S and factors["V"][1] is spaces[L].post.S


def test_identity_kernels_give_identity_factors_exactly():
    narrow = PopulationSpec(kernel=KernelSpec(length_scale=0.0))
    spaces = build_spaces(
        SpatialConfig(input=narrow, hidden=narrow, output=narrow), m=3, N=5, L=2, n_out=2
    )
    for rows, cols in whitening_factors(spaces).values():
        assert torch.equal(rows, torch.eye(rows.shape[0], dtype=F64))
        assert torch.equal(cols, torch.eye(cols.shape[0], dtype=F64))


def test_random_embeddings_still_give_a_square_root():
    random = EmbeddingSpec(kind="random", seed=3)
    spatial = SpatialConfig(
        hidden=PopulationSpec(embedding=random, kernel=KernelSpec(length_scale=0.3))
    )
    hidden = build_spaces(spatial, m=3, N=12, L=1, n_out=2)[0].post
    assert torch.allclose(hidden.S @ hidden.S, hidden.K, rtol=0, atol=1e-12)


# ---------------------------------------------------------------- the whitened net

D_IN, D_OUT, N = 4, 2, 8


def _net_cfg(B: int = 1, L: int = 2) -> NetConfig:
    return NetConfig(N=N, L=L, g=1.0, phi="tanh", seed=5, B=B)


def _spaces(L: int = 2, spatial: SpatialConfig | None = None):
    spatial = spatial or SpatialConfig(
        input=PopulationSpec(kernel=KernelSpec(length_scale=0.2)),
        hidden=PopulationSpec(kernel=KernelSpec(length_scale=0.15)),
        output=PopulationSpec(kernel=KernelSpec(length_scale=0.3)),
    )
    return build_spaces(spatial, m=D_IN, N=N, L=L, n_out=D_OUT)


def _whitened_net(B: int = 1, L: int = 2, init_std: float = 1.0):
    factors = whitening_factors(_spaces(L))
    return make_net(_net_cfg(B, L), D_IN, D_OUT, CPU, F64, whitening=factors, init_std=init_std)


def _x(P: int = 6) -> torch.Tensor:
    return torch.randn(P, D_IN, dtype=F64, generator=torch.Generator().manual_seed(11))


def test_a_plain_net_has_no_whitening_and_its_effective_weights_are_its_parameters():
    net = make_net(_net_cfg(), D_IN, D_OUT, CPU, F64)
    assert net.whitening is None
    weights = effective_weights(net)
    assert set(weights) == {"U", "J.0", "V"}
    for name, p in net.named_parameters():
        assert weights[name] is p


def test_effective_weights_are_s_rows_xi_s_cols():
    net = _whitened_net()
    factors = whitening_factors(_spaces())
    weights = effective_weights(net)
    for name, xi in net.named_parameters():
        rows, cols = factors[name]
        assert torch.allclose(weights[name], rows @ xi @ cols, rtol=1e-13, atol=0), name
        assert not torch.equal(weights[name], xi)
    assert weights["V"].requires_grad  # differentiable with respect to xi


def test_the_forward_pass_uses_the_effective_weights():
    net, x = _whitened_net(), _x()
    weights = {k: v.detach() for k, v in effective_weights(net).items()}
    plain = MLP.from_weights(weights["U"], [weights["J.0"]], weights["V"], net.g, net.phi)
    for got, want in zip(net(x), plain(x), strict=True):
        assert torch.equal(got, want)


def test_identity_factors_reproduce_the_plain_net_bit_for_bit():
    narrow = PopulationSpec(kernel=KernelSpec(length_scale=0.0))
    spatial = SpatialConfig(input=narrow, hidden=narrow, output=narrow)
    identity = whitening_factors(_spaces(2, spatial))
    plain = make_net(_net_cfg(), D_IN, D_OUT, CPU, F64)
    whitened = make_net(_net_cfg(), D_IN, D_OUT, CPU, F64, whitening=identity)
    x, y = _x(), torch.randn(6, D_OUT, dtype=F64, generator=torch.Generator().manual_seed(2))
    for got, want in zip(whitened(x), plain(x), strict=True):
        assert torch.equal(got, want)
    for net in (plain, whitened):
        compute_loss(net, x, y, 0.3, 0.1)[0].backward()
    pairs = zip(whitened.named_parameters(), plain.named_parameters(), strict=True)
    for (name, p), (_, q) in pairs:
        assert torch.equal(p, q) and torch.equal(p.grad, q.grad), name


def test_init_std_scales_the_standard_normal_draw():
    plain = make_net(_net_cfg(B=3), D_IN, D_OUT, CPU, F64)
    scaled = make_net(_net_cfg(B=3), D_IN, D_OUT, CPU, F64, init_std=0.25)
    for (name, p), (_, q) in zip(scaled.named_parameters(), plain.named_parameters(), strict=True):
        assert torch.equal(p, 0.25 * q), name


@pytest.mark.parametrize("init_std", [0.0, -1.0, float("inf"), float("nan")])
def test_init_std_must_be_positive_and_finite(init_std):
    with pytest.raises(ValueError, match="init_std"):
        make_net(_net_cfg(), D_IN, D_OUT, CPU, F64, init_std=init_std)


def test_the_state_dict_holds_xi_under_the_parameter_names_and_nothing_else():
    for net in (_whitened_net(), _whitened_net(B=3)):
        assert set(net.state_dict()) == {"U", "J.0", "V"}
        for name, p in net.named_parameters():
            assert torch.equal(net.state_dict()[name], p)


def test_the_factors_follow_the_net_to_another_dtype():
    net = _whitened_net().to(torch.float32)
    weights = effective_weights(net)
    assert all(w.dtype == torch.float32 for w in weights.values())
    assert net(_x().to(torch.float32))[1].dtype == torch.float32


def test_factors_that_do_not_fit_the_parameters_are_rejected():
    good = whitening_factors(_spaces())
    wrong_shape = {**good, "U": (torch.eye(N, dtype=F64), torch.eye(D_IN + 1, dtype=F64))}
    with pytest.raises(ValueError, match="U"):
        make_net(_net_cfg(), D_IN, D_OUT, CPU, F64, whitening=wrong_shape)
    with pytest.raises(ValueError, match="J.0"):
        make_net(_net_cfg(), D_IN, D_OUT, CPU, F64, whitening={"U": good["U"], "V": good["V"]})


def test_prior_draws_of_xi_give_w_the_kronecker_covariance():
    """(ii): xi ~ N(0, I / weight_lambda) and W = S_post xi S_pre over 10^4 draws has sample
    covariance (K_post (x) K_pre) / weight_lambda on an N = 8 ring: every entry's variance,
    and the variance along every eigenvector of the target, whose softest modes lie below
    1e-6 of the stiffest (the penalty path's floor would have raised them). The relative
    standard error of a sample variance is sqrt(2 / n) = 1.4%, so 5% is 3.5 sigma."""
    weight_lambda, n_draws = 2.0, 10_000
    wide = SpatialConfig(
        input=PopulationSpec(kernel=KernelSpec(length_scale=0.2)),
        hidden=PopulationSpec(kernel=KernelSpec(length_scale=0.25)),
    )
    spaces = _spaces(L=1, spatial=wide)
    space = spaces[0]  # U: post = hidden ring of N, pre = input ring of D_IN
    factors = whitening_factors(spaces)
    net = make_net(_net_cfg(L=1), D_IN, D_OUT, CPU, F64, whitening=factors)
    gen = torch.Generator().manual_seed(0)
    xi = torch.randn(n_draws, N, D_IN, dtype=F64, generator=gen) / weight_lambda**0.5
    W = net.whitening.whiten("U", xi).reshape(n_draws, -1)  # vec row-major
    target = torch.kron(space.post.K, space.pre.K) / weight_lambda
    sample = W.T @ W / n_draws
    assert torch.allclose(sample.diagonal(), target.diagonal(), rtol=0.05, atol=0)
    mode_variance, modes = torch.linalg.eigh(target)
    assert mode_variance.min() > 0 and mode_variance.min() < 1e-6 * mode_variance.max()
    sample_mode_variance = (W @ modes).square().mean(0)
    assert torch.allclose(sample_mode_variance, mode_variance, rtol=0.05, atol=0)


def test_an_ensemble_carries_the_factors_and_its_members_do_too():
    ens, x = _whitened_net(B=3), _x()
    assert isinstance(ens, EnsembleMLP) and ens.whitening is not None
    H, Z = ens(x)
    for b in range(3):
        member = ens.member(b)
        assert member.whitening is not None
        h, z = member(x)
        torch.testing.assert_close(H[b], h, rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(Z[b], z, rtol=1e-13, atol=1e-13)
        for name, w in effective_weights(member).items():
            torch.testing.assert_close(effective_weights(ens)[name][b], w, rtol=1e-13, atol=0)
    rebuilt = EnsembleMLP([ens.member(b) for b in range(3)])
    assert rebuilt.whitening is not None
    for key, value in ens.state_dict().items():
        assert torch.equal(rebuilt.state_dict()[key], value)


def test_an_ensemble_refuses_members_with_different_or_missing_factors():
    whitened = [_whitened_net() for _ in range(2)]
    plain = make_net(_net_cfg(), D_IN, D_OUT, CPU, F64)
    with pytest.raises(ValueError, match="whitening"):
        EnsembleMLP([whitened[0], plain])
    other = whitening_factors(_spaces(2, SpatialConfig()))
    different = make_net(_net_cfg(), D_IN, D_OUT, CPU, F64, whitening=other)
    with pytest.raises(ValueError, match="whitening"):
        EnsembleMLP([whitened[0], different])
    assert EnsembleMLP(whitened).B == 2


# ---------------------------------------------------------------- training

QUIET = {"log": lambda s: None}


def _cfg(
    optimizer: str = "sgld",
    B: int = 1,
    parametrisation: str = "whitened",
    init: str | None = None,
    **train_kw,
):
    """The pinned SGLD run (dense random-embedding kernels, N = 6, L = 2) under the given
    parametrisation, starting from the prior under "whitened" and from the standard draw
    under the penalty unless `init` says otherwise; "adam" takes lr 1e-2."""
    init = init or ("prior" if parametrisation == "whitened" else "standard")
    cfg = pinned_config("sgld", B=B, init=init)
    train_kw = {"optimizer": optimizer, **train_kw}
    if optimizer == "adam":
        train_kw.setdefault("lr", 1e-2)
    spatial = cfg.spatial.model_copy(update={"parametrisation": parametrisation})
    train = cfg.train.model_copy(update=train_kw)
    return Config.model_validate(
        {**cfg.model_dump(), "spatial": spatial.model_dump(), "train": train.model_dump()}
    )


def _setup(cfg: Config):
    ds = make_dataset(cfg.data, CPU, F64)
    return ds, make_net_from_config(cfg, ds.x.shape[1], ds.y.shape[1], CPU, F64)


def _same_state(a, b) -> bool:
    return set(a.state_dict()) == set(b.state_dict()) and all(
        torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items()
    )


def test_make_net_from_config_builds_the_plain_net_under_the_penalty_bit_for_bit():
    cfg = _cfg(parametrisation="penalty")
    ds, net = _setup(cfg)
    plain = make_net(cfg.net, ds.x.shape[1], ds.y.shape[1], CPU, F64)
    assert net.whitening is None and _same_state(net, plain)


@pytest.mark.parametrize("B", [1, 3])
def test_make_net_from_config_whitens_with_the_configs_factors_and_draws_xi_from_the_prior(B):
    cfg = _cfg(B=B)
    ds, net = _setup(cfg)
    m, n_out = ds.x.shape[1], ds.y.shape[1]
    spaces = build_spaces(cfg.spatial, m=m, N=cfg.net.N, L=cfg.net.L, n_out=n_out)
    assert net.whitening is not None
    for name, (rows, cols) in whitening_factors(spaces).items():
        got_rows, got_cols = net.whitening.factors(name)
        assert torch.equal(got_rows, rows) and torch.equal(got_cols, cols), name
    standard = make_net(cfg.net, m, n_out, CPU, F64)
    assert cfg.train.weight_lambda == 1e-2  # xi ~ N(0, I / weight_lambda): std 10
    for (name, xi), (_, w) in zip(net.named_parameters(), standard.named_parameters(), strict=True):
        assert torch.equal(xi, 10.0 * w), name


def _unfloored(cfg: Config) -> Config:
    """`cfg` on grid embeddings with kernels narrow enough that nothing is floored."""
    pop = PopulationSpec(kernel=KernelSpec(length_scale=0.15))
    spatial = cfg.spatial.model_copy(update={"input": pop, "hidden": pop, "output": pop})
    return cfg.model_copy(update={"spatial": spatial})


@pytest.mark.parametrize("B", [1, 3])
def test_without_a_floor_both_paths_start_from_the_same_weights(B):
    penalty = _unfloored(_cfg(B=B, parametrisation="penalty", init="prior"))
    whitened = _unfloored(_cfg(B=B))
    ds, penalty_net = _setup(penalty)
    _, whitened_net = _setup(whitened)
    spaces = build_spaces(penalty.spatial, m=ds.x.shape[1], N=6, L=2, n_out=ds.y.shape[1])
    assert all(s.pre.report.n_floored == s.post.report.n_floored == 0 for s in spaces)
    want = effective_weights(whitened_net)
    for name, w in effective_weights(penalty_net).items():
        assert torch.equal(w, want[name]), name


def test_the_standard_draw_under_the_penalty_is_the_pinned_one():
    cfg = _cfg(parametrisation="penalty")
    _, net = _setup(cfg)
    plain = make_net(cfg.net, net.D_in, net.D_out, CPU, F64)
    assert _same_state(net, plain)


def test_the_penalty_prior_draw_has_the_floored_covariance():
    """10^4 members of an ensemble under the penalty with init "prior": the sample covariance of
    each weight is (K_fl^post (x) K_fl^pre) / weight_lambda, K_fl = M^{-1} the floored kernel
    the penalty encodes (5%, against a relative standard error of 1.4%), along every
    eigenvector of the target and entry by entry, and the floor moves some mode by more."""
    weight_lambda, n_draws = 2.0, 10_000
    wide = SpatialConfig(
        input=PopulationSpec(kernel=KernelSpec(length_scale=0.2)),
        hidden=PopulationSpec(kernel=KernelSpec(length_scale=0.25)),
        output=PopulationSpec(kernel=KernelSpec(length_scale=0.3)),
        min_rel_eig=1e-2,
    )
    cfg = Config(
        net=NetConfig(N=N, L=1, g=1.0, phi="tanh", seed=5, B=n_draws, init="prior"),
        train=TrainConfig(weight_lambda=weight_lambda),
        spatial=wide,
    )
    net = make_net_from_config(cfg, D_IN, D_OUT, CPU, F64)
    spaces = build_spaces(wide, m=D_IN, N=N, L=1, n_out=D_OUT)
    assert spaces[0].post.report.n_floored > 0
    for name, (post, pre) in {
        "U": (spaces[0].post, spaces[0].pre),
        "V": (spaces[1].pre, spaces[1].post),
    }.items():
        W = dict(net.named_parameters())[name].detach().reshape(n_draws, -1)
        floored = torch.kron(torch.linalg.inv(post.M), torch.linalg.inv(pre.M)) / weight_lambda
        mode_variance, modes = torch.linalg.eigh(floored)
        sample = W.T @ W / n_draws
        assert torch.allclose(sample.diagonal(), floored.diagonal(), rtol=0.05, atol=0), name
        projected = modes.T @ sample @ modes
        assert torch.allclose(projected.diagonal(), mode_variance, rtol=0.05, atol=0), name
        off = projected - torch.diag(projected.diagonal())
        scale = torch.sqrt(mode_variance[:, None] * mode_variance[None, :])
        assert (off.abs() / scale).max() < 0.06, name  # uncorrelated modes: ~N(0, 1/100)
        exact = torch.kron(post.K, pre.K) / weight_lambda
        assert ((modes.T @ exact @ modes).diagonal() / mode_variance).min() < 0.5, name


def test_the_whitened_penalty_is_the_plain_sum_of_squares_of_xi():
    cfg = _cfg()
    ds, net = _setup(cfg)
    penalty = make_weight_penalty(cfg, net)
    assert penalty is type(net).weight_l2
    hand = sum(p.square().sum() for p in net.parameters())
    assert torch.equal(penalty(net), hand) or torch.allclose(penalty(net), hand, rtol=1e-15)
    tc = cfg.train
    loss, parts = compute_loss(net, ds.x, ds.y, prior_coefficient(tc), tc.rate_lambda, penalty)
    data = 0.5 * net.N * (ds.y - net(ds.x)[1]).square().mean()
    rates = net.phi(net(ds.x)[0]).square().sum()
    want = data + tc.weight_lambda / (2 * tc.beta) * hand + tc.rate_lambda * rates
    assert loss.item() == pytest.approx(want.item(), rel=1e-12)
    assert parts["weight_l2"].item() == pytest.approx(hand.item(), rel=1e-12)


def test_the_parametrisation_of_the_net_must_match_the_config(tmp_path):
    """The message names the net's factors and the config's parametrisation (None without a
    spatial section)."""
    whitened, penalty = _cfg(), _cfg(parametrisation="penalty")
    no_prior = penalty.model_copy(update={"spatial": None})
    ds, whitened_net = _setup(whitened)
    _, plain_net = _setup(penalty)
    cases = (
        (whitened, plain_net, "has no whitening factors .* spatial.parametrisation is 'whitened'"),
        (penalty, whitened_net, "has whitening factors .* spatial.parametrisation is 'penalty'"),
        (no_prior, whitened_net, "has whitening factors .* spatial.parametrisation is None"),
    )
    for cfg, net, message in cases:
        with pytest.raises(ValueError, match=message):
            make_weight_penalty(cfg, net)
        with pytest.raises(ValueError, match=message):
            train(cfg, ds, net, tmp_path, **QUIET)
    assert not any(tmp_path.rglob("*.pt"))


def test_the_prior_only_step_bound_under_whitening_is_lr_times_weight_lambda_over_two_beta():
    """The prior |xi|^2 weight_lambda / (2 beta) has curvature weight_lambda / beta in every
    direction, so lr * weight_lambda / (2 beta) >= 1 is refused, whatever the kernel."""
    too_big = _cfg(lr=1.0, weight_lambda=4.0, beta=1.0)  # 1 * 4 / 2 = 2
    _, net = _setup(too_big)
    with pytest.raises(ValueError, match="diverges on the whitened spatial prior"):
        make_weight_penalty(too_big, net)
    fine = _cfg(lr=0.4, weight_lambda=4.0, beta=1.0)  # 0.4 * 4 / 2 = 0.8
    _, net = _setup(fine)
    assert make_weight_penalty(fine, net) is type(net).weight_l2
    adam = _cfg("adam", lr=1.0, weight_lambda=4.0, beta=1.0)  # Adam rescales its steps
    _, net = _setup(adam)
    assert make_weight_penalty(adam, net) is type(net).weight_l2


def test_one_prior_step_bound_serves_both_paths():
    """lr * weight_lambda / (2 beta) * stiffness >= 1 is refused, the message naming the prior,
    lr and the stiffness; the penalty's stiffness is lambda_max(M_rows) lambda_max(M_cols) of
    its stiffest parameter, the whitened prior's is 1."""
    tc = TrainConfig(optimizer="sgld", lr=0.5, weight_lambda=2.0, beta=1.0)  # coefficient 1
    with pytest.raises(ValueError, match=r"lr=0\.5 diverges on the some prior alone.*= 2\b"):
        _check_sgld_step_resolves_the_prior(4.0, "some prior", tc)  # 0.5 * 1 * 4 = 2
    _check_sgld_step_resolves_the_prior(1.9, "some prior", tc)  # 0.95
    eye, twice = torch.eye(3, dtype=F64), 2 * torch.eye(2, dtype=F64)
    factors = {"U": (eye, twice), "V": (3 * eye, 5 * twice)}
    assert prior_stiffness(factors) == pytest.approx(30.0)


def test_largest_hessian_eigenvalue_of_a_quadratic_is_its_largest_curvature():
    curvatures = torch.tensor([1.0, 2.0, 3.0, 10.0], dtype=F64)
    theta = torch.zeros(4, dtype=F64, requires_grad=True)
    other = torch.ones(2, 3, dtype=F64, requires_grad=True)
    lam = largest_hessian_eigenvalue(
        lambda: 0.5 * (curvatures * theta.square()).sum() + 0.5 * 4.0 * other.square().sum(),
        [theta, other],
    )
    assert lam == pytest.approx(10.0, rel=1e-6)
    assert theta.grad is None and other.grad is None  # nothing left on the parameters


def test_largest_hessian_eigenvalue_is_the_top_of_the_spectrum_not_the_largest_magnitude():
    """0.5 (-100 p0^2 + 50 p1^2): plain power iteration finds -100, under which lr = 0.05 would
    pass the check although lr * 50 = 2.5 diverges along p1. The estimate must be 50."""
    p = torch.tensor([0.3, -0.2], dtype=F64, requires_grad=True)
    curvatures = torch.tensor([-100.0, 50.0], dtype=F64)
    lam = largest_hessian_eigenvalue(lambda: 0.5 * (curvatures * p.square()).sum(), [p])
    assert lam == pytest.approx(50.0, rel=1e-6)
    concave = largest_hessian_eigenvalue(
        lambda: -0.5 * (torch.tensor([3.0, 7.0]) * p.square()).sum(), [p]
    )
    assert concave == pytest.approx(-3.0, rel=1e-6)


def test_largest_hessian_eigenvalue_leaves_the_global_rng_alone():
    cfg = _cfg()
    ds, net = _setup(cfg)
    state = torch.get_rng_state()
    largest_hessian_eigenvalue(lambda: compute_loss(net, ds.x, ds.y, 0.1, 0.0)[0], net.parameters())
    assert torch.equal(torch.get_rng_state(), state)


def test_sgld_under_whitening_refuses_a_step_the_loss_curvature_cannot_take(tmp_path):
    """The prior-only bound misses the data term's curvature; the run-start check estimates
    lambda_max of the loss Hessian at the initial net and refuses lr * lambda_max >= 1."""
    cfg = _cfg()
    ds, net = _setup(cfg)
    tc = cfg.train
    penalty = make_weight_penalty(cfg, net)

    def loss():
        return compute_loss(net, ds.x, ds.y, prior_coefficient(tc), tc.rate_lambda, penalty)[0]

    lam = largest_hessian_eigenvalue(loss, net.parameters())
    assert lam > 0
    too_big = _cfg(lr=2.0 / lam)
    assert too_big.train.lr * too_big.train.weight_lambda / (2 * too_big.train.beta) < 1
    _, net = _setup(too_big)
    with pytest.raises(ValueError, match=rf"lr={too_big.train.lr:g}.*lambda_max"):
        train(too_big, ds, net, tmp_path / "big", **QUIET)
    assert not any((tmp_path / "big").rglob("*.pt"))
    fine = _cfg(lr=0.5 / lam, n_epochs=1)
    lines = []
    _, net = _setup(fine)
    train(fine, ds, net, tmp_path / "fine", log=lines.append)
    assert any("lambda_max" in line for line in lines)


def _smoke_s8_k100(lr: float) -> Config:
    """Setting S8 of the experiment plan's smoke ring at k = 1 / weight_lambda = 100: N = 16
    ring of hidden length scale 2 sites (box 16), unstructured input and output kernels,
    P = 32 linear-teacher patterns with m = n = 4 and 2 latents, phi = id, beta = 10."""
    flat = {"kernel": {"length_scale": 0.0, "amplitude": 1.0}}
    return Config.model_validate(
        {
            "data": {"task": "linear_teacher", "P": 32, "D_in": 4, "n_latents": 2},
            "net": {"N": 16, "L": 1, "phi": "id", "g": 1.0, "init": "prior"},
            "train": {
                "optimizer": "sgld",
                "beta": 10.0,
                "weight_lambda": 0.01,
                "rate_lambda": 0.0,
                "lr": lr,
                "n_epochs": 1,
            },
            "spatial": {
                "d": 1,
                "box": 16.0,
                "parametrisation": "whitened",
                "input": flat,
                "output": flat,
                "hidden": {"kernel": {"length_scale": 2.0, "amplitude": 1.0}},
            },
            "device": "cpu",
            "dtype": "float64",
        }
    )


def test_the_curvature_check_refuses_the_prior_only_step_on_the_smoke_ring_at_k_100(tmp_path):
    """At k = 100 the prior alone admits lr = 0.2 (lr * weight_lambda / (2 beta) = 1e-4), but
    the data term's curvature (lambda_max ~ 40-100 at the prior draw) makes lr * lambda_max
    ~ 10: the run-start check must refuse it before any checkpoint is written, and accept the
    plan's step-rule value 0.0005 (lr * lambda_max ~ 0.05)."""
    too_big = _smoke_s8_k100(0.2)
    ds, net = _setup(too_big)
    assert too_big.train.lr * prior_coefficient(too_big.train) < 1  # the prior check passes
    with pytest.raises(ValueError, match=r"lr=0\.2 .*lambda_max"):
        train(too_big, ds, net, tmp_path / "big", **QUIET)
    assert not any((tmp_path / "big").rglob("*.pt"))
    fine = _smoke_s8_k100(0.0005)
    _, net = _setup(fine)
    lines = []
    train(fine, ds, net, tmp_path / "fine", log=lines.append)
    assert any(line.startswith("SGLD step check") for line in lines)


def test_the_penalty_path_logs_no_curvature_check(tmp_path):
    cfg = _cfg(parametrisation="penalty", n_epochs=1)
    ds, net = _setup(cfg)
    lines = []
    train(cfg, ds, net, tmp_path, log=lines.append)
    assert not any("lambda_max" in line for line in lines)


@pytest.mark.parametrize("optimizer", ["adam", "sgld"])
@pytest.mark.parametrize("B", [1, 2])
def test_a_whitened_run_round_trips_through_its_checkpoints_and_final_file(tmp_path, optimizer, B):
    cfg = _cfg(optimizer, B=B)
    ds = make_dataset(cfg.data, CPU, F64)
    net, history = train_from_config(cfg, ds, tmp_path, **QUIET)
    assert net.whitening is not None
    loaded, loaded_history = load_trained(tmp_path, cfg, CPU, F64)
    assert loaded.whitening is not None and _same_state(loaded, net)
    assert loaded_history == history
    for name, w in effective_weights(net).items():
        assert torch.equal(effective_weights(loaded)[name], w), name
    ckpt_net, _, epoch = load_latest_checkpoint(tmp_path, cfg, CPU, F64)
    assert ckpt_net.whitening is not None and epoch == 6
    assert set(
        torch.load(tmp_path / "train" / "final.pt", weights_only=True)["net_state_dict"]
    ) == {
        "U",
        "J.0",
        "V",
    }
    if optimizer == "sgld":
        penalty_hist = train_from_config(
            _cfg(optimizer, B=B, parametrisation="penalty"), ds, tmp_path / "penalty", **QUIET
        )[1]
        assert set(penalty_hist) == set(history)  # the same history keys on both paths


class Interrupt(Exception):
    pass


def test_a_resumed_whitened_sgld_run_equals_the_uninterrupted_one(tmp_path):
    cfg = _cfg("sgld", save_int=2, n_epochs=7)
    ds, ref = _setup(cfg)
    ref, ref_history = train(cfg, ds, ref, tmp_path / "ref", **QUIET)

    def stop_at_five(line):
        if line.startswith("[00004"):
            raise Interrupt

    _, net = _setup(cfg)
    with pytest.raises(Interrupt):
        train(cfg, ds, net, tmp_path / "run", log=stop_at_five)
    _, net = _setup(cfg)
    net, history = train(cfg, ds, net, tmp_path / "run", **QUIET)
    assert _same_state(net, ref) and history == ref_history


def test_a_whitened_adam_ensemble_equals_its_members_trained_separately(tmp_path):
    cfg = _cfg("adam", B=3)
    ds, ens = _setup(cfg)
    starts = [ens.member(b) for b in range(3)]
    ens, history = train(cfg, ds, ens, tmp_path / "ens", **QUIET)
    single_cfg = _with_B(cfg, 1)
    for b, start in enumerate(starts):
        single, single_history = train(single_cfg, ds, start, tmp_path / f"m{b}", **QUIET)
        for name, p in ens.named_parameters():
            want = dict(single.named_parameters())[name]
            torch.testing.assert_close(p[b], want, rtol=1e-12, atol=1e-14)
        for key in ("mse", "loss", "weight_l2", "rate_l2"):
            got = torch.tensor([entry[b] for entry in history[key]])
            torch.testing.assert_close(got, torch.tensor(single_history[key]), rtol=1e-12, atol=0)


def test_whitened_sgld_steps_of_an_ensemble_equal_separate_members():
    cfg = _cfg("sgld", B=3)
    ds, ens = _setup(cfg)
    singles = [ens.member(b) for b in range(3)]
    tc = cfg.train
    gens = [torch.Generator().manual_seed(100 + b) for b in range(3)]
    optim = SGLD(ens.parameters(), tc.lr, tc.beta, member_axis=True, noise=_member_noise(gens))
    parts = [_step(ens, optim, ds, tc, make_weight_penalty(cfg, ens)) for _ in range(5)]
    for b, single in enumerate(singles):
        gen = torch.Generator().manual_seed(100 + b)
        single_optim = SGLD(single.parameters(), tc.lr, tc.beta, noise=_single_noise(gen))
        penalty = make_weight_penalty(_with_B(cfg, 1), single)
        single_parts = [_step(single, single_optim, ds, tc, penalty) for _ in range(5)]
        for name, p in ens.named_parameters():
            want = dict(single.named_parameters())[name]
            torch.testing.assert_close(p[b], want, rtol=1e-12, atol=1e-14)
        for got, want in zip(parts, single_parts, strict=True):
            for key in ("loss", "weight_l2", "grad_norm", "param_msd"):
                torch.testing.assert_close(got[key][b], want[key], rtol=1e-10, atol=0)


def test_prior_only_sgld_samples_every_mode_of_the_whitened_prior():
    """(iv): SGLD on the prior term alone, from the prior initialisation, on an N = 8 ring
    (U: 8 x 4, V: 8 x 2). Each direction of xi is a quadratic of curvature c = weight_lambda /
    beta, so the finite-lr law (tests/test_sgld.py) gives every mode of W = S xi S the variance
    lambda_q^post lambda_q^pre / (weight_lambda (1 - lr c / 2)): here lr c = 0.3, a 17.6%
    inflation that distinguishes the law from the lr -> 0 limit. 2000 records three
    relaxation times apart give a relative standard error of ~3.3% per mode."""
    weight_lambda, beta, lr = 2.0, 0.5, 0.075  # lr * c = 0.075 * 4 = 0.3
    wide = SpatialConfig(
        input=PopulationSpec(kernel=KernelSpec(length_scale=0.2)),
        hidden=PopulationSpec(kernel=KernelSpec(length_scale=0.25)),
        output=PopulationSpec(kernel=KernelSpec(length_scale=0.3)),
    )
    spaces = _spaces(L=1, spatial=wide)
    factors = whitening_factors(spaces)
    net = make_net(
        _net_cfg(L=1), D_IN, D_OUT, CPU, F64, whitening=factors, init_std=weight_lambda**-0.5
    )
    coefficient = weight_lambda / (2 * beta)
    torch.manual_seed(3)
    optim = SGLD(net.parameters(), lr=lr, beta=beta)
    every, n_records, burn_in = 10, 2000, 100
    records = {name: [] for name, _ in net.named_parameters()}
    for step in range(burn_in + every * n_records):
        optim.zero_grad()
        (coefficient * net.weight_l2()).backward()
        optim.step()
        if step >= burn_in and (step - burn_in) % every == 0:
            for name, p in net.named_parameters():
                records[name].append(p.detach().clone())
    inflation = 1 / (1 - lr * weight_lambda / beta / 2)
    ratios = []
    for name in factors:
        xi = torch.stack(records[name])  # (n_records, r, c)
        W = net.whitening.whiten(name, xi).reshape(n_records, -1)
        post_pre = {
            "U": (spaces[0].post.K, spaces[0].pre.K),
            "V": (spaces[1].pre.K, spaces[1].post.K),
        }
        mode_variance, modes = torch.linalg.eigh(torch.kron(*post_pre[name]) / weight_lambda)
        sample = (W @ modes).square().mean(0)
        ratios.append(sample / (inflation * mode_variance))
    ratios = torch.cat(ratios)
    assert ratios.min() > 0.9 and ratios.max() < 1.1
    assert ratios.mean().item() == pytest.approx(1.0, abs=0.03)  # not the lr -> 0 law (0.85)
