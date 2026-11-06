"""The whitened parametrisation of the spatial prior (`spatial.parametrisation = "whitened"`):
the trained parameters are xi with W = S_post xi S_pre, S = K^{1/2}, and the penalty is the
plain |xi|^2, so the sampled prior is K itself (no eigenvalue floor). Ground truth throughout:
the prior covariance (K_post (x) K_pre) / weight_lambda, which the penalty path only
approximates."""

import pytest
import torch

from fct.config import Config, TrainConfig
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
