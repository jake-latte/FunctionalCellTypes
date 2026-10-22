"""Tests for fct.spatial: embeddings, wrapped Gaussian kernels, the penalty matrix, build_spaces."""

import itertools

import pytest
import torch
from pydantic import ValidationError

from fct.spatial import (
    DTYPE,
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


def _ring(n, box=1.0):
    return make_embedding(EmbeddingSpec(kind="grid"), n, d=1, box=box).positions


def _random_2d(n=30, seed=0):
    return make_embedding(EmbeddingSpec(kind="random", seed=seed), n, d=2, box=1.0).positions


def _points(name):
    return {"ring10": _ring(10), "ring50": _ring(50), "random2d": _random_2d()}[name]


POINT_SETS = ["ring10", "ring50", "random2d"]
LENGTH_SCALES = [0.05, 0.1, 0.3, 0.5, 1.0]


# ------------------------------------------------------------------ make_embedding


def test_make_embedding_grid_1d_is_evenly_spaced_from_zero():
    x = _ring(8, box=2.0)
    assert x.shape == (8, 1) and x.dtype == DTYPE
    assert x[0, 0] == 0.0
    torch.testing.assert_close(x.diff(dim=0), torch.full((7, 1), 2.0 / 8, dtype=DTYPE))


def test_make_embedding_grid_2d_has_the_given_shape():
    emb = make_embedding(EmbeddingSpec(kind="grid", shape=[3, 2]), 6, d=2, box=1.0)
    assert emb.positions.shape == (6, 2)
    assert emb.positions.max() < 1.0


def test_make_embedding_rejects_shape_not_multiplying_to_n():
    with pytest.raises(ValueError, match="multiply"):
        make_embedding(EmbeddingSpec(kind="grid", shape=[3, 3]), 6, d=2, box=1.0)


def test_make_embedding_rejects_grid_without_shape_in_2d():
    with pytest.raises(ValueError, match="shape"):
        make_embedding(EmbeddingSpec(kind="grid"), 6, d=2, box=1.0)


def test_make_embedding_random_is_seeded_and_in_box():
    spec = EmbeddingSpec(kind="random", seed=3)
    a = make_embedding(spec, 50, d=2, box=2.0).positions
    b = make_embedding(spec, 50, d=2, box=2.0).positions
    other_seed = make_embedding(spec.model_copy(update={"seed": 4}), 50, d=2, box=2.0).positions
    assert torch.equal(a, b) and not torch.equal(a, other_seed)
    assert a.shape == (50, 2) and a.min() >= 0 and a.max() < 2.0


def test_make_embedding_gives_distinct_streams_to_distinct_seed_index_pairs():
    a = make_embedding(EmbeddingSpec(kind="random", seed=0), 10, d=1, box=1.0, index=1)
    b = make_embedding(EmbeddingSpec(kind="random", seed=1), 10, d=1, box=1.0, index=0)
    assert not torch.equal(a.positions, b.positions)


def test_make_embedding_leaves_global_rng_untouched():
    torch.manual_seed(123)
    before = torch.get_rng_state()
    make_embedding(EmbeddingSpec(kind="random", seed=1), 20, d=2, box=1.0)
    assert torch.equal(torch.get_rng_state(), before)


# ------------------------------------------------------------------ torus_distance


def test_torus_distance_wraps():
    x = torch.tensor([[0.05]], dtype=DTYPE)
    y = torch.tensor([[0.95]], dtype=DTYPE)
    assert torus_distance(x, y, 1.0).item() == pytest.approx(0.1)
    assert torus_distance(x, y, 10.0).item() == pytest.approx(0.9)


def test_torus_distance_2d_is_euclidean_of_wrapped_axes():
    x = torch.tensor([[0.05, 0.5]], dtype=DTYPE)
    y = torch.tensor([[0.95, 0.1]], dtype=DTYPE)
    assert torus_distance(x, y, 1.0).item() == pytest.approx((0.1**2 + 0.4**2) ** 0.5)


# ------------------------------------------------------------------ periodic_gaussian_kernel


@pytest.mark.parametrize("length_scale", LENGTH_SCALES)
@pytest.mark.parametrize("name", POINT_SETS)
def test_periodic_gaussian_kernel_is_positive_semidefinite(name, length_scale):
    x = _points(name)
    eigs = torch.linalg.eigvalsh(periodic_gaussian_kernel(x, x, length_scale, 1.0, 1.5))
    assert eigs.min().item() >= -1e-10 * eigs.max().item()


@pytest.mark.parametrize("length_scale", LENGTH_SCALES)
@pytest.mark.parametrize("name", POINT_SETS)
def test_periodic_gaussian_kernel_is_exactly_symmetric(name, length_scale):
    x = _points(name)
    K = periodic_gaussian_kernel(x, x, length_scale, 1.0, 1.5)
    assert torch.equal(K, K.T)


@pytest.mark.parametrize("length_scale", LENGTH_SCALES)
@pytest.mark.parametrize("name", POINT_SETS)
def test_periodic_gaussian_kernel_diagonal_is_amplitude(name, length_scale):
    x = _points(name)
    K = periodic_gaussian_kernel(x, x, length_scale, 1.0, 1.5)
    assert torch.equal(K.diagonal(), torch.full((len(x),), 1.5, dtype=DTYPE))


def test_periodic_gaussian_kernel_reduces_to_plain_gaussian_for_small_length_scale():
    x = _random_2d()
    K = periodic_gaussian_kernel(x, x, 0.02, 1.0, 1.0)
    plain = torch.exp(-torus_distance(x, x, 1.0).square() / (2 * 0.02**2))
    torch.testing.assert_close(K, plain, atol=1e-12, rtol=0)


def test_periodic_gaussian_kernel_pins_a_value_where_the_images_matter():
    x = torch.tensor([[0.0], [0.5]], dtype=DTYPE)
    K = periodic_gaussian_kernel(x, x, 1.0, 1.0, 1.0)
    assert K[0, 1].item() == pytest.approx(0.9999999892988481, rel=1e-12)


def test_periodic_gaussian_kernel_has_the_requested_rectangular_shape():
    xp = torch.tensor([[0.0], [0.5]], dtype=DTYPE)
    xq = torch.tensor([[0.95], [0.25], [0.5]], dtype=DTYPE)
    K = periodic_gaussian_kernel(xp, xq, length_scale=0.1, box=1.0, amplitude=2.0)
    assert K.shape == (2, 3)
    assert K[0, 0].item() == pytest.approx(2.0 * torch.exp(torch.tensor(-(0.05**2) / 0.02)).item())


def test_periodic_gaussian_kernel_ignores_whole_box_shifts_of_positions():
    x = _random_2d(n=10)
    shift = torch.tensor([[3.0, -2.0]], dtype=DTYPE)
    shifted = periodic_gaussian_kernel(x + shift, x, 0.2, 1.0, 1.0)
    torch.testing.assert_close(shifted, periodic_gaussian_kernel(x, x, 0.2, 1.0, 1.0))


def test_periodic_gaussian_kernel_couples_points_one_box_apart():
    x = torch.tensor([[0.0], [3.0]], dtype=DTYPE)
    K = periodic_gaussian_kernel(x, x, 0.1, 1.0, 1.0)
    assert K[0, 1].item() == pytest.approx(1.0)


@pytest.mark.parametrize("shift", range(1, 12))
def test_periodic_gaussian_kernel_is_circulant_on_a_grid_ring(shift):
    x = _ring(12)
    K = periodic_gaussian_kernel(x, x, 0.15, 1.0, 1.0)
    torch.testing.assert_close(K, K.roll((shift, shift), (0, 1)))


# ------------------------------------------------------------------ penalty_matrix


def test_penalty_matrix_inverts_the_kernel_when_nothing_is_floored():
    x = _ring(10)
    K = periodic_gaussian_kernel(x, x, 0.08, 1.0, 1.0)
    M, report = penalty_matrix(K, min_rel_eig=1e-12)
    assert report.n_floored == 0
    torch.testing.assert_close(M @ K, torch.eye(10, dtype=DTYPE), atol=1e-6, rtol=0)


def _ill_conditioned():
    gen = torch.Generator().manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(5, 5, generator=gen, dtype=DTYPE))
    eigs = torch.tensor([1.0, 0.5, 1e-3, 1e-8, 0.0], dtype=DTYPE)
    return Q, Q @ torch.diag(eigs) @ Q.T


def test_penalty_matrix_report_counts_floored_eigenvalues():
    _, K = _ill_conditioned()
    _, report = penalty_matrix(K, min_rel_eig=1e-6)
    assert report.n_floored == 2  # 1e-8 and 0 lie below 1e-6 * 1
    assert report.fraction_floored == pytest.approx(0.4)


def test_penalty_matrix_report_gives_condition_numbers():
    _, K = _ill_conditioned()
    _, report = penalty_matrix(K, min_rel_eig=1e-6)
    assert report.cond_after == pytest.approx(1e6, rel=1e-6)
    assert report.cond_before > 1e10  # smallest raw eigenvalue is round-off sized or negative


def test_penalty_matrix_inverts_on_the_retained_eigendirections():
    Q, K = _ill_conditioned()
    M, _ = penalty_matrix(K, min_rel_eig=1e-6)
    P = Q[:, :3]
    torch.testing.assert_close(P.T @ M @ K @ P, torch.eye(3, dtype=DTYPE), atol=1e-8, rtol=0)
    assert torch.linalg.eigvalsh(M).max().item() == pytest.approx(1e6, rel=1e-6)


def test_penalty_matrix_counts_and_clamps_negative_eigenvalues():
    K = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=DTYPE)  # eigenvalues 3 and -1
    M, report = penalty_matrix(K, min_rel_eig=1e-6)
    assert report.n_negative == 1 and report.n_floored == 1
    assert torch.linalg.eigvalsh(M).min().item() > 0


def test_build_spaces_floors_83_of_100_eigenvalues_of_a_default_ring():
    pop = build_spaces(SpatialConfig(), m=4, N=100, L=1, n_out=3)[0].post
    assert pop.report.n_floored == 83
    assert pop.report.fraction_floored == 0.83


def test_penalty_matrix_rejects_float32_k():
    with pytest.raises(ValueError, match="float64"):
        penalty_matrix(torch.eye(3, dtype=torch.float32), 1e-6)


def test_penalty_matrix_rejects_asymmetric_k():
    with pytest.raises(ValueError, match="symmetric"):
        penalty_matrix(torch.tensor([[1.0, 0.5], [0.0, 1.0]], dtype=DTYPE), 1e-6)


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_penalty_matrix_rejects_non_finite_k(bad):
    K = torch.eye(3, dtype=DTYPE)
    K[0, 1] = K[1, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        penalty_matrix(K, 1e-6)


@pytest.mark.parametrize("min_rel_eig", [0.0, 1.0])
def test_penalty_matrix_rejects_floor_outside_unit_interval(min_rel_eig):
    with pytest.raises(ValueError, match="min_rel_eig"):
        penalty_matrix(torch.eye(3, dtype=DTYPE), min_rel_eig)


def test_penalty_matrix_rejects_kernel_without_positive_eigenvalue():
    with pytest.raises(ValueError, match="positive"):
        penalty_matrix(torch.zeros(3, 3, dtype=DTYPE), 1e-6)


# ------------------------------------------------------------------ SpatialConfig


def test_spatial_config_defaults():
    cfg = SpatialConfig()
    assert cfg.d == 1 and cfg.box == 1.0 and cfg.min_rel_eig == 1e-6
    assert cfg.input.embedding.kind == "grid"


def test_spatial_config_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        SpatialConfig(typo=1)


def test_spatial_config_is_strict_about_types():
    with pytest.raises(ValidationError):
        SpatialConfig(d="1")


def test_kernel_spec_rejects_non_positive_length_scale():
    with pytest.raises(ValidationError):
        KernelSpec(length_scale=0.0)


def test_spatial_config_rejects_grid_shape_of_wrong_length():
    with pytest.raises(ValidationError, match="shape"):
        SpatialConfig(d=2, input=PopulationSpec(embedding=EmbeddingSpec(shape=[4])))


def test_spatial_config_rejects_grid_without_shape_in_2d():
    with pytest.raises(ValidationError, match="needs a shape"):
        SpatialConfig(d=2)


def test_embedding_spec_rejects_shape_for_random_kind():
    with pytest.raises(ValidationError, match="no shape"):
        EmbeddingSpec(kind="random", shape=[4])


# ------------------------------------------------------------------ build_spaces


@pytest.mark.parametrize("L", [1, 3])
def test_build_spaces_gives_the_shapes_of_every_layer(L):
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


@pytest.mark.parametrize("L", [1, 3])
def test_build_spaces_shares_populations_between_consecutive_layers(L):
    spaces = build_spaces(SpatialConfig(), 6, 8, L, 4)
    for a, b in itertools.pairwise(spaces):
        assert a.post is b.pre
        assert a.post.embedding is b.pre.embedding


def test_build_spaces_names_hidden_populations_one_to_L():
    spaces = build_spaces(SpatialConfig(), 6, 8, 3, 4)
    assert [sp.post.name for sp in spaces] == ["hidden 1", "hidden 2", "hidden 3", "output"]


def test_build_spaces_random_hidden_embeddings_differ_and_repeat():
    hidden = PopulationSpec(embedding=EmbeddingSpec(kind="random", seed=5))
    cfg = SpatialConfig(hidden=hidden)
    spaces = build_spaces(cfg, 4, 6, 2, 3)
    h1, h2 = spaces[0].post.embedding.positions, spaces[1].post.embedding.positions
    assert not torch.equal(h1, h2)
    assert torch.equal(h1, build_spaces(cfg, 4, 6, 2, 3)[0].post.embedding.positions)


def test_build_spaces_input_and_first_hidden_do_not_share_a_random_stream():
    spec = PopulationSpec(embedding=EmbeddingSpec(kind="random", seed=0))
    cfg = SpatialConfig(input=spec, hidden=spec)
    first = build_spaces(cfg, 6, 6, 1, 3)[0]
    assert not torch.equal(first.pre.embedding.positions, first.post.embedding.positions)


def test_build_spaces_rejects_grid_shape_not_matching_population_size():
    hidden = PopulationSpec(embedding=EmbeddingSpec(shape=[5]))
    with pytest.raises(ValueError, match="multiply"):
        build_spaces(SpatialConfig(hidden=hidden), 4, 6, 2, 3)
