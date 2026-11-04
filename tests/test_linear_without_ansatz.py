"""Ground-truth checks for latex/notes/scripts/linear_without_ansatz.py.

Synthetic instances with a known answer: in the commuting case the matrix fixed point
must reproduce the scalar roots of eq:linear-soln; the per-frequency solver must agree
with the full NP x NP kernel map; the first-order theory must match a finite difference
with an O(eps) residual; and a rank-deficient latent task must leave the complement of
the latent column space at its prior value.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "latex" / "notes" / "scripts" / "linear_without_ansatz.py"
)
spec = importlib.util.spec_from_file_location("linear_without_ansatz", SCRIPT)
lwa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lwa)

PARAMS = lwa.Params(k0=0.7, k1=1.3, sig2=0.2, g2=0.3, delta=0.5)


@pytest.fixture
def instance():
    rng = np.random.default_rng(0)
    return lwa.make_instance(rng, n_sites=16, n_patterns=4)


def test_commuting_case_matches_scalar_roots(instance):
    lam = instance.lam
    Ct = lwa.solve_fixed_point(lam, instance.A, instance.gram_y(0.0), PARAMS)
    roots = lwa.scalar_roots(lam, instance.sx2hat(PARAMS), instance.sy2hat(PARAMS), PARAMS)
    np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(Ct)), np.sort(roots), rtol=1e-9)


def test_per_q_solver_is_fixed_point_of_full_map(instance):
    YY = instance.gram_y(1.0)
    Ct = lwa.solve_fixed_point(instance.lam, instance.A, YY, PARAMS)
    K = lwa.ring_kernel(instance.n_sites, instance.width)
    Ct_full = lwa.full_kernel_map(Ct, K, instance.A, YY, PARAMS)
    np.testing.assert_allclose(Ct_full, Ct, atol=1e-9)


def test_fixed_point_is_unique_across_starts(instance):
    YY = instance.gram_y(1.0)
    starts = [5.0 * np.eye(4), 50.0 * np.eye(4)]
    fixed = [
        lwa.solve_fixed_point(instance.lam, instance.A, YY, PARAMS, Ct_start=C0) for C0 in starts
    ]
    np.testing.assert_allclose(fixed[0], fixed[1], atol=1e-9)


def test_first_order_matches_finite_difference(instance):
    theory = lwa.first_order(instance, PARAMS)
    Ct0 = lwa.solve_fixed_point(instance.lam, instance.A, instance.gram_y(0.0), PARAMS)

    def residual(eps):
        Ct = lwa.solve_fixed_point(instance.lam, instance.A, instance.gram_y(eps), PARAMS)
        C1_num = instance.U.T @ (Ct - Ct0) @ instance.U / eps
        return np.abs(C1_num - theory.C1).max()

    r1, r2 = residual(0.02), residual(0.01)
    assert r1 < 0.05 * np.abs(theory.C1).max()
    assert 1.6 < r1 / r2 < 2.4  # residual is O(eps): halving eps halves it


def test_first_order_rotation_matches_eigenvectors(instance):
    theory = lwa.first_order(instance, PARAMS)
    eps = 0.01
    Sigma = lwa.per_q_cov_at(instance.lam, instance.A, instance.gram_y(eps), PARAMS)
    Theta_num = lwa.mixing_from_eigenvectors(instance.U, Sigma) / eps
    err = np.abs(Theta_num - theory.Theta).max()
    assert err < 0.05 * np.abs(theory.Theta).max()


def test_rank_deficient_complement_is_prior():
    rng = np.random.default_rng(1)
    n_sites, P, K = 16, 5, 2
    lam = lwa.ring_eigenvalues(n_sites, np.sqrt(2.0))
    Z = np.linalg.qr(rng.normal(size=(P, K)))[0]
    GX, GY = lwa.random_spd(rng, K), lwa.random_spd(rng, K)
    A, YY = PARAMS.k0 * Z @ GX @ Z.T, Z @ GY @ Z.T
    Ct = lwa.solve_fixed_point(lam, A, YY, PARAMS)
    Pi = Z @ Z.T
    Pi_perp = np.eye(P) - Pi
    c0 = lwa.prior_scalar(lam, PARAMS)
    np.testing.assert_allclose(Pi_perp @ Ct @ Pi_perp, c0 * Pi_perp, atol=1e-9)
    np.testing.assert_allclose(Pi @ Ct @ Pi_perp, 0, atol=1e-9)


def test_non_pd_precision_raises(instance):
    # A tiny C makes the tilt B enormous, so M_q - lam_q B is not positive definite.
    with pytest.raises(lwa.NotPositiveDefiniteError):
        lwa.kernel_map(1e-3 * np.eye(4), instance.lam, instance.A, instance.gram_y(1.0), PARAMS)
