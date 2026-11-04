"""Tests for latex/notes/scripts/continuum_limit_check.py (the dictionary of continuum_limit.tex).

The script is a standalone check, not part of the fct package, so it is loaded by path.
"""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "latex" / "notes" / "scripts" / "continuum_limit_check.py"


@pytest.fixture(scope="module")
def clc():
    spec = importlib.util.spec_from_file_location("continuum_limit_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Spacing deliberately not 1, so that a missing factor of a is visible.
SPACING, LENGTH = 0.5, 1.0


def test_ring_eigenvalues_match_continuum_spectrum_over_spacing(clc):
    """lambda_q = lambda(q) / a at the reciprocal-lattice points, up to the aliased weight beyond the cutoff."""
    n_sites = 64
    K = clc.ring_kernel(n_sites, SPACING, LENGTH)
    q, lam = clc.lattice_spectrum(K, SPACING)
    expected = clc.continuum_spectrum(q, LENGTH) / SPACING
    floor = clc.aliasing_floor(SPACING, LENGTH)
    assert 1e-10 < floor < 1e-8                       # the floor is visible but far below the signal
    assert np.allclose(lam, expected, rtol=0, atol=2 * floor * lam.max())
    assert not np.allclose(lam, expected, rtol=0, atol=0.1 * floor * lam.max())   # and it is really there
    assert np.allclose(np.sort(lam), np.linalg.eigvalsh(K), rtol=1e-10, atol=1e-12)


def test_mode_average_of_lambda_is_the_kernel_diagonal(clc):
    """(1/N) sum_q lambda_q = K_00 = 1 on the lattice; the cutoff integral gives erf(Lambda l / sqrt 2)."""
    K = clc.ring_kernel(32, SPACING, LENGTH)
    _, lam = clc.lattice_spectrum(K, SPACING)
    assert clc.mode_average(lam, lambda lam_: np.ones_like(lam_)) == pytest.approx(K[0, 0], rel=1e-13)
    cont = clc.continuum_mode_average(SPACING, LENGTH, lambda lam_: np.ones_like(lam_))
    assert cont == pytest.approx(math.erf(np.pi / SPACING * LENGTH / np.sqrt(2)), rel=1e-13)
    assert cont == pytest.approx(1.0, rel=1e-9)


def test_ring_average_converges_exponentially_to_continuum_integral(clc):
    """At fixed spacing the error falls with N (volume) and reaches round-off."""
    params = clc.SolutionParams(**clc.PARAMS_1D)
    f = clc.linear_solution_weight(params)
    cont = clc.continuum_mode_average(SPACING, LENGTH, f)
    errors = [abs(clc.ring_mode_average(n, SPACING, LENGTH, f) - cont) / abs(cont) for n in (8, 16, 32, 64, 128)]
    assert errors[0] > 1e-4                      # finite-size effect is visible at L = 4 l
    assert errors[0] > errors[1] > errors[2] > errors[3]   # and shrinks with L (the last pair sits on the floor)
    assert errors[-1] < 10 * clc.aliasing_floor(SPACING, LENGTH)   # down to the aliased tail


def test_lattice_inverse_is_a_squared_times_cutoff_continuum_inverse(clc):
    """(K^{-1})_{0j} = a^2 int_{BZ} dq/(2pi) e^{iq x_j} / lambda(q): the penalty carries a^{2d}.

    Checked on K + nugget I, whose spectrum is bounded below; the pure kernel's inverse is
    dominated by the modes at the cutoff and has no continuum limit (second assertion).
    """
    n_sites, x = 128, np.arange(4) * SPACING
    M = np.linalg.inv(clc.ring_kernel(n_sites, SPACING, LENGTH, clc.NUGGET))
    predicted = clc.continuum_inverse_kernel(x, SPACING, LENGTH, clc.NUGGET)
    assert np.allclose(M[0, :4], predicted, rtol=1e-6, atol=0)
    M_pure = np.linalg.inv(clc.ring_kernel(n_sites, SPACING, LENGTH))
    predicted_pure = clc.continuum_inverse_kernel(x, SPACING, LENGTH)
    assert np.abs(M_pure[0, 0] / predicted_pure[0] - 1) > 0.1


def test_torus_average_matches_two_dimensional_integral(clc):
    """d = 2: the factors a^d and (2 pi)^d, on the square Brillouin zone."""
    params = clc.SolutionParams(**clc.PARAMS_2D)
    f = clc.linear_solution_weight(params)
    cont = clc.continuum_mode_average_2d(SPACING, LENGTH, f)
    lattice = clc.torus_mode_average(64, SPACING, LENGTH, f)     # L = 32 l: finite-size error is below the aliasing floor
    assert lattice == pytest.approx(cont, rel=10 * clc.aliasing_floor(SPACING, LENGTH))


def test_solution_params_reject_c_below_the_pole(clc):
    """The physical branch needs c > s_Y sqrt(max h); a c below it must fail loudly, not give garbage."""
    params = clc.SolutionParams(sx2=0.7, sy2=0.25, sig2=0.2, g2=0.3, c=0.1)
    f = clc.linear_solution_weight(params)
    with pytest.raises(ValueError, match="pole"):
        clc.ring_mode_average(16, SPACING, LENGTH, f)
