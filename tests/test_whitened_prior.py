"""The whitened parametrisation of the spatial prior (`spatial.parametrisation = "whitened"`):
the trained parameters are xi with W = S_post xi S_pre, S = K^{1/2}, and the penalty is the
plain |xi|^2, so the sampled prior is K itself (no eigenvalue floor). Ground truth throughout:
the prior covariance (K_post (x) K_pre) / weight_lambda, which the penalty path only
approximates."""

import pytest
import torch

from fct.config import Config, NetConfig, TrainConfig
from fct.model import MLP, EnsembleMLP, effective_weights, make_net
from fct.spatial import (
    EmbeddingSpec,
    KernelSpec,
    PopulationSpec,
    SpatialConfig,
    build_spaces,
    parameter_factors,
    periodic_gaussian_kernel,
    whitening_factors,
    whitening_matrix,
)
from fct.train import compute_loss
from tests.test_lambda_max import PINNED_SPATIAL_HASH, _pinned_spatial

CPU, F64 = torch.device("cpu"), torch.float64
TRAIN = ("data", "net", "train")


def _ring_kernel(n: int, length_scale: float) -> torch.Tensor:
    x = torch.arange(n, dtype=F64)[:, None] / n
    return periodic_gaussian_kernel(x, x, length_scale, 1.0, 1.0)


def _whitened(spatial: SpatialConfig | None = None) -> SpatialConfig:
    return (spatial or SpatialConfig()).model_copy(update={"parametrisation": "whitened"})


# ---------------------------------------------------------------- config and hash


def test_the_parametrisation_defaults_to_the_penalty_and_an_old_config_loads():
    assert SpatialConfig().parametrisation == "penalty"
    raw = SpatialConfig().model_dump()
    del raw["parametrisation"]
    assert SpatialConfig.model_validate(raw).parametrisation == "penalty"
    with pytest.raises(ValueError, match="parametrisation"):
        SpatialConfig(parametrisation="other")


def test_the_penalty_parametrisation_keeps_the_pinned_hash_and_whitened_changes_it():
    base = Config(spatial=_pinned_spatial())
    assert base.section_hash(*TRAIN) == PINNED_SPATIAL_HASH
    explicit = _pinned_spatial().model_copy(update={"parametrisation": "penalty"})
    assert base.model_copy(update={"spatial": explicit}).section_hash(*TRAIN) == PINNED_SPATIAL_HASH
    whitened = base.model_copy(update={"spatial": _whitened(_pinned_spatial())})
    assert whitened.section_hash(*TRAIN) != PINNED_SPATIAL_HASH
    assert whitened.section_hash("data", "net") == base.section_hash("data", "net")


def test_whitened_needs_a_positive_weight_lambda():
    with pytest.raises(ValueError, match="weight_lambda"):
        Config(spatial=_whitened(), train=TrainConfig(weight_lambda=0.0))
    cfg = Config(spatial=_whitened(), train=TrainConfig(weight_lambda=0.5))
    assert cfg.train.weight_lambda == 0.5
    # The penalty path keeps accepting weight_lambda = 0 (no prior term).
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
