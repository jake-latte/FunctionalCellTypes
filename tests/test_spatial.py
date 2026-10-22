"""Tests for fct.spatial: torus geometry, Gaussian kernels, the penalty matrix, build_spaces."""

import itertools

import pytest
import torch
from pydantic import ValidationError

from fct.spatial import (
    EmbeddingSpec,
    KernelSpec,
    PopulationSpec,
    SpatialConfig,
    build_spaces,
    make_embedding,
    penalty_matrix,
    periodic_gaussian_kernel,
    torus_distance,
)

DT = torch.float64


def _ring(n, box=1.0):
    return make_embedding(EmbeddingSpec(kind="grid"), n, d=1, box=box).positions


# ------------------------------------------------------------------ embeddings


def test_grid_1d_evenly_spaced_from_zero():
    x = _ring(8, box=2.0)
    assert x.shape == (8, 1) and x.dtype == DT
    assert x[0, 0] == 0.0
    torch.testing.assert_close(x.diff(dim=0), torch.full((7, 1), 2.0 / 8, dtype=DT))


def test_grid_2d_needs_matching_shape():
    emb = make_embedding(EmbeddingSpec(kind="grid", shape=[3, 2]), 6, d=2, box=1.0)
    assert emb.positions.shape == (6, 2)
    assert emb.positions.max() < 1.0
    with pytest.raises(ValueError, match="multiply"):
        make_embedding(EmbeddingSpec(kind="grid", shape=[3, 3]), 6, d=2, box=1.0)
    with pytest.raises(ValueError, match="shape"):
        make_embedding(EmbeddingSpec(kind="grid"), 6, d=2, box=1.0)


def test_random_embedding_is_seeded_and_in_box():
    spec = EmbeddingSpec(kind="random", seed=3)
    a = make_embedding(spec, 50, d=2, box=2.0).positions
    b = make_embedding(spec, 50, d=2, box=2.0).positions
    c = make_embedding(spec.model_copy(update={"seed": 4}), 50, d=2, box=2.0).positions
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert a.shape == (50, 2) and a.min() >= 0 and a.max() < 2.0


# ------------------------------------------------------------------ distance and kernel


def test_torus_distance_wraps():
    x = torch.tensor([[0.05]], dtype=DT)
    y = torch.tensor([[0.95]], dtype=DT)
    assert torus_distance(x, y, 1.0).item() == pytest.approx(0.1)
    assert torus_distance(x, y, 10.0).item() == pytest.approx(0.9)


def test_torus_distance_2d_is_euclidean_of_wrapped_axes():
    x = torch.tensor([[0.05, 0.5]], dtype=DT)
    y = torch.tensor([[0.95, 0.1]], dtype=DT)
    assert torus_distance(x, y, 1.0).item() == pytest.approx((0.1**2 + 0.4**2) ** 0.5)


def _point_sets():
    rand2 = make_embedding(EmbeddingSpec(kind="random", seed=0), 30, d=2, box=1.0).positions
    return {"ring10": _ring(10), "ring50": _ring(50), "random2d": rand2}


@pytest.mark.parametrize("length_scale", [0.05, 0.1, 0.3, 0.5, 1.0])
@pytest.mark.parametrize("name", ["ring10", "ring50", "random2d"])
def test_kernel_psd_symmetric_amplitude_diagonal(name, length_scale):
    x = _point_sets()[name]
    K = periodic_gaussian_kernel(x, x, length_scale, box=1.0, amplitude=1.5)
    torch.testing.assert_close(K, K.T)
    assert torch.equal(K.diagonal(), torch.full((len(x),), 1.5, dtype=DT))
    eigs = torch.linalg.eigvalsh(K)
    assert eigs.min().item() >= -1e-10 * eigs.max().item()
    _, report = penalty_matrix(K, min_rel_eig=1e-6)
    assert report.n_negative == 0


def test_kernel_reduces_to_plain_gaussian_for_small_length_scale():
    x = _point_sets()["random2d"]
    K = periodic_gaussian_kernel(x, x, 0.02, 1.0, 1.0)
    plain = torch.exp(-torus_distance(x, x, 1.0).square() / (2 * 0.02**2))
    torch.testing.assert_close(K, plain, atol=1e-12, rtol=0)


def test_kernel_wraps_for_wide_length_scale():
    # Far from the minimum-image Gaussian: with ell = 1 the neighbouring images matter.
    x = torch.tensor([[0.0], [0.5]], dtype=DT)
    K = periodic_gaussian_kernel(x, x, 1.0, 1.0, 1.0)
    plain = torch.exp(torch.tensor(-(0.5**2) / 2, dtype=DT))
    assert K[0, 1].item() > plain.item()


def test_kernel_value_and_rectangular_shape():
    xp = torch.tensor([[0.0], [0.5]], dtype=DT)
    xq = torch.tensor([[0.95], [0.25], [0.5]], dtype=DT)
    K = periodic_gaussian_kernel(xp, xq, length_scale=0.1, box=1.0, amplitude=2.0)
    assert K.shape == (2, 3)
    assert K[0, 0].item() == pytest.approx(2.0 * torch.exp(torch.tensor(-(0.05**2) / 0.02)).item())


def test_grid_kernel_on_ring_is_circulant():
    x = _ring(12)
    K = periodic_gaussian_kernel(x, x, 0.15, 1.0, 1.0)
    for shift in range(1, 12):
        torch.testing.assert_close(K, K.roll((shift, shift), (0, 1)))


# ------------------------------------------------------------------ penalty matrix


def test_penalty_is_inverse_on_retained_spectrum():
    x = _ring(10)
    K = periodic_gaussian_kernel(x, x, 0.08, 1.0, 1.0)
    M, report = penalty_matrix(K, min_rel_eig=1e-12)
    assert report.n_floored == 0
    torch.testing.assert_close(M @ K, torch.eye(10, dtype=DT), atol=1e-6, rtol=0)


def test_penalty_report_on_constructed_ill_conditioned_kernel():
    gen = torch.Generator().manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(5, 5, generator=gen, dtype=DT))
    eigs = torch.tensor([1.0, 0.5, 1e-3, 1e-8, 0.0], dtype=DT)
    K = Q @ torch.diag(eigs) @ Q.T
    M, report = penalty_matrix(K, min_rel_eig=1e-6)
    assert report.n_floored == 2  # 1e-8 and 0 lie below 1e-6 * 1
    assert report.cond_after == pytest.approx(1e6, rel=1e-6)
    assert report.cond_before > 1e10  # smallest raw eigenvalue is round-off sized or negative
    # On the retained eigen-directions M K is the identity.
    P = Q[:, :3]
    torch.testing.assert_close(P.T @ M @ K @ P, torch.eye(3, dtype=DT), atol=1e-8, rtol=0)
    assert torch.linalg.eigvalsh(M).max().item() == pytest.approx(1e6, rel=1e-6)


def test_penalty_rejects_bad_input():
    with pytest.raises(ValueError, match="symmetric"):
        penalty_matrix(torch.tensor([[1.0, 0.5], [0.0, 1.0]], dtype=DT), 1e-6)
    with pytest.raises(ValueError, match="min_rel_eig"):
        penalty_matrix(torch.eye(3, dtype=DT), 0.0)
    with pytest.raises(ValueError, match="positive"):
        penalty_matrix(torch.zeros(3, 3, dtype=DT), 1e-6)


def test_report_counts_negative_eigenvalues_of_non_psd_matrix():
    K = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=DT)  # eigenvalues 3 and -1
    M, report = penalty_matrix(K, min_rel_eig=1e-6)
    assert report.n_negative == 1 and report.n_floored == 1
    assert torch.linalg.eigvalsh(M).min().item() > 0  # clamped, so M stays positive definite


# ------------------------------------------------------------------ config


def test_config_defaults_and_strictness():
    cfg = SpatialConfig()
    assert cfg.d == 1 and cfg.box == 1.0 and cfg.min_rel_eig == 1e-6
    assert cfg.input.embedding.kind == "grid"
    with pytest.raises(ValidationError):
        SpatialConfig(typo=1)
    with pytest.raises(ValidationError):
        SpatialConfig(d="1")
    with pytest.raises(ValidationError):
        KernelSpec(length_scale=0.0)
    with pytest.raises(ValidationError, match="shape"):
        SpatialConfig(d=2, input=PopulationSpec(embedding=EmbeddingSpec(shape=[4])))


# ------------------------------------------------------------------ build_spaces


@pytest.mark.parametrize("L", [1, 3])
def test_build_spaces_shapes_and_sharing(L):
    m, N, n_out = 6, 8, 4
    spaces = build_spaces(SpatialConfig(), m, N, L, n_out)
    assert len(spaces) == L + 1
    sizes = [m] + [N] * L + [n_out]
    for layer, sp in enumerate(spaces):
        n_pre, n_post = sizes[layer], sizes[layer + 1]
        assert sp.layer == layer
        assert sp.pre.embedding.positions.shape == (n_pre, 1)
        assert sp.post.embedding.positions.shape == (n_post, 1)
        assert sp.pre.K.shape == (n_pre, n_pre) and sp.post.K.shape == (n_post, n_post)
        assert sp.M_pre.shape == (n_pre, n_pre) and sp.M_post.shape == (n_post, n_post)
    for a, b in itertools.pairwise(spaces):
        assert a.post is b.pre
        assert a.post.embedding is b.pre.embedding


def test_build_spaces_hidden_random_seeds_differ_per_population():
    hidden = PopulationSpec(embedding=EmbeddingSpec(kind="random", seed=5))
    spaces = build_spaces(SpatialConfig(hidden=hidden), 4, 6, 2, 3)
    h1, h2 = spaces[0].post.embedding.positions, spaces[1].post.embedding.positions
    assert not torch.equal(h1, h2)
    again = build_spaces(SpatialConfig(hidden=hidden), 4, 6, 2, 3)
    assert torch.equal(h1, again[0].post.embedding.positions)


def test_build_spaces_grid_shape_must_match_population_size():
    hidden = PopulationSpec(embedding=EmbeddingSpec(shape=[5]))
    with pytest.raises(ValueError, match="multiply"):
        build_spaces(SpatialConfig(hidden=hidden), 4, 6, 2, 3)
