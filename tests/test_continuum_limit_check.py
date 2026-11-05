"""Tests for latex/notes/scripts/continuum_limit_check.py (the dictionary of continuum_limit.tex).

The script is a standalone check, not part of the fct package, so it is loaded by path.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "latex" / "notes" / "scripts" / "continuum_limit_check.py"
)


@pytest.fixture(scope="module")
def clc():
    spec = importlib.util.spec_from_file_location("continuum_limit_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Spacing deliberately not 1, so that a missing factor of a is visible.
SPACING, LENGTH = 0.5, 1.0


def ones(lam: np.ndarray) -> np.ndarray:
    return np.ones_like(lam)


def test_ring_eigenvalues_match_continuum_spectrum_over_spacing(clc):
    """lambda_q = lambda(q) / a at the reciprocal-lattice points, up to the zone-edge alias."""
    n_sites = 64
    K = clc.ring_kernel(n_sites, SPACING, LENGTH)
    q, lam = clc.lattice_spectrum(K, SPACING)
    expected = clc.continuum_spectrum(q, LENGTH) / SPACING
    edge = clc.zone_edge_error(SPACING, LENGTH)
    assert 1e-9 < edge < 1e-8  # the alias is visible but far below the signal
    assert np.allclose(lam, expected, rtol=0, atol=1.01 * edge * lam.max())
    assert not np.allclose(lam, expected, rtol=0, atol=0.99 * edge * lam.max())  # it is the max
    assert np.allclose(np.sort(lam), np.linalg.eigvalsh(K), rtol=1e-10, atol=1e-12)


def test_mode_average_of_lambda_is_the_kernel_diagonal(clc):
    """(1/N) sum_q lambda_q = K_00 = 1 on the lattice; the cutoff integral gives erf(Lambda l / sqrt 2)."""
    K = clc.ring_kernel(32, SPACING, LENGTH)
    _, lam = clc.lattice_spectrum(K, SPACING)
    assert clc.mode_average(lam, ones) == pytest.approx(K[0, 0], rel=1e-13)
    cont = clc.continuum_mode_average(SPACING, LENGTH, ones)
    assert cont == pytest.approx(math.erf(np.pi / SPACING * LENGTH / np.sqrt(2)), rel=1e-13)
    assert cont == pytest.approx(1.0, rel=1e-9)
    assert 1 - cont == pytest.approx(clc.spectral_weight_outside_zone(SPACING, LENGTH, 1), rel=1e-3)


def test_ring_error_decays_as_exp_minus_L_over_xi_down_to_the_derived_floor(clc):
    """At fixed spacing the error falls as e^{-L/xi}, xi from the learning pole, then sits on the floor."""
    params = clc.SolutionParams(**clc.PARAMS_1D)
    f = clc.linear_solution_weight(params, clc.lattice_lambda_zero(SPACING, LENGTH, dim=1))
    cont = clc.continuum_mode_average(SPACING, LENGTH, f)
    sizes = (8, 16, 32, 64, 128)
    errors = clc.convergence_errors(
        sizes, lambda n: clc.ring_mode_average(n, SPACING, LENGTH, f), cont
    )
    assert errors[0] > 1e-4  # finite-size effect is visible at L = 4 l
    assert errors[0] > errors[1] > errors[2] > errors[3]  # and shrinks with L
    xi_fit = clc.fitted_correlation_length(sizes[1:3], errors[1:3], SPACING)
    xi_pole = 1 / clc.learning_pole_imag_q(params, SPACING, LENGTH)
    assert xi_fit == pytest.approx(xi_pole, rel=0.02)  # the pole sets the rate
    floor = clc.aliasing_floor(params, cont, SPACING, LENGTH, dim=1)
    assert errors[-1] == pytest.approx(floor, rel=0.05)  # and the aliased tail sets the floor


def test_tail_coefficients_match_the_expansion_of_the_summand(clc):
    """eq:tail: the lambda^2 coefficient includes the sY^2 term of the geometric series."""
    params = clc.SolutionParams(sx2=0.7, sy2=0.25, sig2=0.2, g2=0.3, c=2.5)
    c1, c2 = clc.tail_coefficients(params, rho=1.0)

    def summand(lam: float) -> float:
        h = clc.h_of_lambda(np.array(lam), params)
        return float(h / (params.c**2 - params.sy2 * h))

    eps = 1e-3
    first = (summand(eps) - summand(-eps)) / (2 * eps)
    second = (summand(eps) - 2 * summand(0.0) + summand(-eps)) / eps**2 / 2
    assert first == pytest.approx(c1, rel=1e-5)
    assert second == pytest.approx(c2, rel=1e-4)
    without_sy_term = params.sx2 / ((1 + params.g2 * params.sig2) ** 2 * params.c**2)
    assert abs(second - without_sy_term) > 10 * abs(second - c2)  # the extra term is needed


def test_lattice_inverse_is_a_squared_times_cutoff_continuum_inverse(clc):
    """(K^{-1})_{0j} = a^2 int_{BZ} dq/(2pi) e^{iq x_j} / lambda(q): the penalty carries a^{2d}.

    Checked on K + nugget I, whose spectrum is bounded below; the pure kernel's inverse is
    dominated by the modes at the cutoff and has no continuum limit (second assertion).
    """
    lattice, predicted, max_rel = clc.inverse_kernel_check(clc.NUGGET, SPACING, LENGTH, n_points=4)
    assert np.allclose(lattice, predicted, rtol=1e-6, atol=0)
    assert max_rel < 1e-6
    _, _, max_rel_pure = clc.inverse_kernel_check(0.0, SPACING, LENGTH)
    assert max_rel_pure > 0.1


def test_torus_average_matches_two_dimensional_integral_to_the_derived_floor(clc):
    """d = 2: the factors a^d and (2 pi)^d on the square zone, and the floor with weight 1 - erf^2."""
    params = clc.SolutionParams(**clc.PARAMS_2D)
    f = clc.linear_solution_weight(params, clc.lattice_lambda_zero(SPACING, LENGTH, dim=2))
    cont = clc.continuum_mode_average_2d(SPACING, LENGTH, f)
    # L = 32 l: the finite-size error is below the aliasing floor
    error = clc.relative_error(clc.torus_mode_average(64, SPACING, LENGTH, f), cont)
    assert error == pytest.approx(
        clc.aliasing_floor(params, cont, SPACING, LENGTH, dim=2), rel=0.05
    )


def test_solution_params_reject_c_below_the_pole(clc):
    """The physical branch needs c > s_Y sqrt(h(lambda_0)); a c below it must fail loudly at construction."""
    params = clc.SolutionParams(sx2=0.7, sy2=0.25, sig2=0.2, g2=0.3, c=0.1)
    with pytest.raises(ValueError, match="pole"):
        clc.linear_solution_weight(params, clc.lattice_lambda_zero(SPACING, LENGTH, dim=1))


def test_power_law_root_is_pinned_above_the_pole_at_accessible_cutoffs(clc):
    """d = 2 power law: c sits just above c_* and its log slope is far below the asymptote."""
    params = clc.SolutionParams(**clc.PARAMS_1D)
    sols = [clc.solve_power_law_2d(params, **clc.POWER_LAW, cutoff_over_mass=r) for r in (2e2, 2e6)]
    asymptote = clc.log_asymptote_slope(params, clc.POWER_LAW["gamma"])
    assert 0 < sols[0].c - sols[0].c_star < 0.01
    assert sols[0].c < sols[1].c
    (slope,) = clc.log_slopes(sols)
    assert slope < 0.3 * asymptote
    assert sols[0].learning_term > sols[1].learning_term > 0  # learning shrinks, c stays near c_*
    assert sols[0].xi_star > sols[1].xi_star > 1
