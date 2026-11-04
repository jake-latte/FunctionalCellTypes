"""Ground-truth checks for latex/notes/scripts/linear_without_ansatz.py.

Synthetic instances with a known answer: in the commuting case the matrix fixed point
must reproduce the scalar roots of eq:linear-soln; the per-frequency solver must agree
with the full NP x NP kernel map; the first-order theory must match a finite difference
with an O(eps) residual; the per-q eigenbases must rotate for non-commuting data and
not otherwise; the two-latent angle formula must match the numerical eigenvectors; and
a rank-deficient latent task must leave the complement of the latent column space at
its prior value, whether or not Z has orthonormal columns.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "latex" / "notes" / "scripts" / "linear_without_ansatz.py"
)
spec = importlib.util.spec_from_file_location("linear_without_ansatz", SCRIPT)
lwa = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lwa  # dataclasses resolve postponed annotations through sys.modules
spec.loader.exec_module(lwa)

PARAMS = lwa.Params(k0=0.7, k1=1.3, sig2=0.2, g2=0.3, delta=0.5)
N_SITES, P = 16, 4


@pytest.fixture
def instance():
    return lwa.make_instance(np.random.default_rng(0), n_sites=N_SITES, n_patterns=P)


def test_commuting_case_matches_scalar_roots(instance):
    r = lwa.commuting_check(instance, PARAMS)
    assert r.root_error < 1e-9
    assert r.offdiag_fraction < 1e-12


def test_solver_stops_on_the_true_residual(instance):
    fp = lwa.solve_fixed_point(instance.lam, instance.A(PARAMS), instance.gram_y(1.0), PARAMS)
    F = lwa.kernel_map(fp.Ct, instance.lam, instance.A(PARAMS), instance.gram_y(1.0), PARAMS)
    assert fp.residual < lwa.TOL
    assert np.abs(F - fp.Ct).max() == pytest.approx(fp.residual)


def test_per_q_solver_is_fixed_point_of_full_map(instance):
    assert lwa.non_commuting_solution(instance, PARAMS).full_map_residual < 1e-9


def test_two_starts_converge_to_the_same_point(instance):
    assert lwa.non_commuting_solution(instance, PARAMS).start_difference < 1e-9


def test_eigenbases_rotate_iff_data_do_not_commute(instance):
    still = lwa.non_commuting_solution(instance, PARAMS, eps=0.0)
    rotated = lwa.non_commuting_solution(instance, PARAMS, eps=1.0)
    assert still.angles_to_q0.max() < 0.01  # degrees
    assert rotated.angles_to_q0.max() > 3.0


def test_finest_scale_basis_is_eigenbasis_of_fine_limit(instance):
    s = lwa.non_commuting_solution(instance, PARAMS)
    assert s.angles_to_fine_limit[N_SITES // 2].max() < 0.05  # degrees; lam_{N/2} ~ 4e-4


def test_first_order_residuals_are_order_eps(instance):
    theory = lwa.first_order(instance, PARAMS)
    r = lwa.first_order_residuals(instance, PARAMS, eps_values=(0.02, 0.01))
    assert r.C1[0] < 0.05 * np.abs(theory.C1).max()
    assert r.Theta[0] < 0.05 * np.abs(theory.Theta).max()
    assert 1.6 < r.C1[0] / r.C1[1] < 2.4  # halving eps halves the residual
    assert 1.6 < r.Theta[0] / r.Theta[1] < 2.4


def test_two_latent_angle_formula_matches_eigenvectors():
    inst = lwa.make_instance(np.random.default_rng(2), n_sites=N_SITES, n_patterns=2)
    s = lwa.non_commuting_solution(inst, PARAMS)
    B_in_U = inst.U.T @ s.B @ inst.U
    V = lwa.eigenbases(inst.U.T @ s.Sigma @ inst.U)  # (N, 2, 2) in the basis where A is diagonal
    theta_num = np.arctan2(V[:, 1, 0], V[:, 0, 0])  # either eigenvector gives the same tan 2 theta
    predicted = lwa.two_latent_angle(inst.lam, inst.sx2hat(PARAMS), B_in_U, PARAMS)
    np.testing.assert_allclose(np.tan(2 * theta_num), predicted, rtol=1e-8)
    assert np.abs(theta_num).max() > np.radians(3)  # the angle is not trivially zero


@pytest.mark.parametrize("orthonormal_z", [True, False])
def test_rank_deficient_complement_is_prior(orthonormal_z):
    rd = lwa.make_rank_deficient_instance(
        np.random.default_rng(1),
        n_patterns=5,
        n_latent=2,
        n_sites=N_SITES,
        orthonormal_z=orthonormal_z,
    )
    r = lwa.rank_deficient_check(rd, PARAMS)
    assert r.block_offdiag < 1e-9
    assert r.complement_error < 1e-9


def test_commuting_latents_still_rotate_when_z_is_not_orthonormal():
    """[G_X, G_Y] = 0 does not make the ansatz exact unless Z^T Z is proportional to I."""
    rd = lwa.make_commuting_latents_non_orthonormal_z(np.random.default_rng(3), n_sites=N_SITES)
    r = lwa.rank_deficient_check(rd, PARAMS)
    assert r.commutator_G == 0.0
    assert r.commutator_restricted > 0.01
    assert r.block_offdiag < 1e-9  # the block structure survives
    assert r.max_angle_to_q0 > 3.0  # degrees: the bases rotate, so the ansatz is not exact


def test_non_pd_precision_raises(instance):
    # A tiny Ct makes the tilt B enormous, so M_q - lam_q B is not positive definite.
    with pytest.raises(lwa.NotPositiveDefiniteError):
        lwa.kernel_map(
            1e-3 * np.eye(P), instance.lam, instance.A(PARAMS), instance.gram_y(1.0), PARAMS
        )
