"""Check the discrete-to-continuum dictionary of latex/notes/continuum_limit.tex.

Ground truth: a ring of N sites with spacing a and the squared-exponential kernel
    K_ij = exp(-r_ij^2 / (2 l^2)),   r_ij the ring distance,
whose continuum spectrum is known in closed form, lambda(q) = sqrt(2 pi) l exp(-l^2 q^2 / 2).
The dictionary claims, at fixed spacing a and growing volume L = N a,
    lambda_q = lambda(q) / a^d,
    (1/N) sum_q lambda_q f(lambda_q)  ->  a^d int_{BZ} d^dq/(2 pi)^d (lambda(q)/a^d) f(lambda(q)/a^d),
    (K^{-1})_ij                       ->  a^{2d} int_{BZ} d^dq/(2 pi)^d e^{iq(x-x')} / lambda(q),
with the Brillouin zone |q_alpha| < pi/a. Each claim is tested with the spacing a != 1, so a
missing factor of a would show. Two limits of the dictionary are also exhibited: the lattice
eigenvalue is the aliased spectrum (1/a) sum_n lambda(q + 2 pi n/a), so the first line holds up
to the spectral weight beyond the cutoff, e^{-l^2 Lambda^2/2}, which is the floor the finite-size
error settles on; and for the pure squared-exponential kernel the inverse is dominated by modes at
the cutoff (1/lambda(q) grows as e^{+l^2 q^2/2}), so its lattice and cutoff-continuum values differ
at O(1). The a^{2d} rule is therefore checked on K + nugget * I, whose spectrum is bounded below. The smooth f used is the summand of the corrected eq:linear-soln,
    lambda_q f(lambda_q) = h_q / (c^2 - sY^2 h_q),
    h_q = lambda_q (lambda_q sX^2 + sigma^2) / (1 + g^2 (lambda_q sX^2 + sigma^2)),
and the finite-size error is quoted for increasing N (it falls as e^{-L/xi}, xi the longest
correlation length of the integrand). A torus repeats the check in d = 2 on the square zone.

Run from the git root:  uv run python latex/notes/scripts/continuum_limit_check.py   (seconds)
"""
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# --- parameters quoted in latex/notes/continuum_limit.tex -------------------------
SPACING = 0.5                 # a, held fixed as N grows
LENGTH = 1.0                  # l, kernel width (l / a = 2, aliasing error e^{-2 pi^2 l^2 / a^2})
RING_SIZES = (4, 8, 16, 32, 64, 128, 256)
TORUS_SIDES = (4, 8, 16, 32, 64)
N_QUAD = 2000                 # Gauss-Legendre nodes per dimension for the continuum integrals
PARAMS_1D = {"sx2": 0.7, "sy2": 0.25, "sig2": 0.2, "g2": 0.3, "c": 2.0}   # hat s_X^2, hat s_Y^2, sigma^2, g^2, c
PARAMS_2D = {"sx2": 0.7, "sy2": 0.25, "sig2": 0.2, "g2": 0.3, "c": 6.0}   # c above the pole (lambda_0 is larger in d = 2)
NUGGET = 0.1                  # epsilon in K + epsilon I for the inverse-kernel check

Weight = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class SolutionParams:
    """Scalars of one data mode in the corrected eq:linear-soln."""
    sx2: float
    sy2: float
    sig2: float
    g2: float
    c: float


# --- lattice ------------------------------------------------------------------------
def ring_distance(n_sites: int, spacing: float) -> np.ndarray:
    k = np.arange(n_sites)
    return spacing * np.minimum(k, n_sites - k)


def ring_kernel(n_sites: int, spacing: float, length: float, nugget: float = 0.0) -> np.ndarray:
    """Circulant squared-exponential kernel on a periodic ring, K_00 = 1 + nugget."""
    row = np.exp(-ring_distance(n_sites, spacing) ** 2 / (2 * length**2))
    return np.array([np.roll(row, i) for i in range(n_sites)]) + nugget * np.eye(n_sites)


def torus_kernel_row(side: int, spacing: float, length: float) -> np.ndarray:
    """First row of the kernel on a side x side torus, as a side x side array."""
    r = ring_distance(side, spacing)
    r2 = r[:, None] ** 2 + r[None, :] ** 2
    return np.exp(-r2 / (2 * length**2))


def lattice_spectrum(K: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues lambda_q of a circulant K and their wavenumbers q = 2 pi k / L.

    A circulant is diagonalised by the unitary DFT, so lambda_q is the DFT of the first row;
    the imaginary part is round-off because K is symmetric.
    """
    n_sites = K.shape[0]
    q = 2 * np.pi * np.fft.fftfreq(n_sites, d=spacing)
    return q, np.fft.fft(K[0]).real


# --- continuum ----------------------------------------------------------------------
def continuum_spectrum(q: np.ndarray, length: float) -> np.ndarray:
    """lambda(q) = int dx K(x) e^{-iqx} for K(x) = exp(-x^2 / 2 l^2), in d = 1 (per dimension in d = 2)."""
    return np.sqrt(2 * np.pi) * length * np.exp(-(length**2) * q**2 / 2)


def aliasing_floor(spacing: float, length: float) -> float:
    """Relative spectral weight of lambda(q) beyond the cutoff pi/a: the accuracy limit of lambda_q = lambda(q)/a."""
    return np.exp(-(length**2) * (np.pi / spacing) ** 2 / 2)


def brillouin_nodes(spacing: float, n_quad: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on the zone (-pi/a, pi/a)."""
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    cutoff = np.pi / spacing
    return cutoff * nodes, cutoff * weights


def continuum_mode_average(spacing: float, length: float, f: Weight, n_quad: int = N_QUAD) -> float:
    """a int_{-pi/a}^{pi/a} dq/(2 pi) lambda_q f(lambda_q), lambda_q = lambda(q)/a: the d = 1 image of (1/N) sum_q."""
    q, w = brillouin_nodes(spacing, n_quad)
    lam = continuum_spectrum(q, length) / spacing
    return spacing * np.sum(w * lam * f(lam)) / (2 * np.pi)


def continuum_mode_average_2d(spacing: float, length: float, f: Weight, n_quad: int = N_QUAD // 4) -> float:
    """a^2 int_{BZ} d^2q/(2 pi)^2 lambda_q f(lambda_q) on the square zone, lambda_q = lambda(q)/a^2."""
    q, w = brillouin_nodes(spacing, n_quad)
    lam1 = continuum_spectrum(q, length)
    lam = lam1[:, None] * lam1[None, :] / spacing**2      # the kernel factorises over the two axes
    return spacing**2 * np.sum(w[:, None] * w[None, :] * lam * f(lam)) / (2 * np.pi) ** 2


def continuum_inverse_kernel(x: np.ndarray, spacing: float, length: float, nugget: float = 0.0,
                             n_quad: int = N_QUAD) -> np.ndarray:
    """a^2 int_{BZ} dq/(2 pi) e^{iqx} / lambda(q): the dictionary's image of (K^{-1})_{0j}.

    A nugget epsilon delta_ij on the lattice is a flat epsilon a^d in the continuum spectrum.
    """
    q, w = brillouin_nodes(spacing, n_quad)
    phase = np.cos(q[None, :] * x[:, None])                # the sine part vanishes by symmetry
    lam = continuum_spectrum(q, length) + nugget * spacing
    return spacing**2 * np.sum(w * phase / lam, axis=1) / (2 * np.pi)


# --- the smooth weight: the summand of eq:linear-soln --------------------------------
def h_of_lambda(lam: np.ndarray, p: SolutionParams) -> np.ndarray:
    u = lam * p.sx2 + p.sig2
    return lam * u / (1 + p.g2 * u)


def linear_solution_weight(p: SolutionParams) -> Weight:
    """f with lambda f(lambda) = h / (c^2 - sY^2 h); raises if c is below the pole of the physical branch."""

    def f(lam: np.ndarray) -> np.ndarray:
        h = h_of_lambda(lam, p)
        pole = np.sqrt(p.sy2 * h.max())
        if p.c <= pole:
            raise ValueError(f"c = {p.c} is not above the pole c = {pole:.4f}; the branch with d_q > 0 needs c > pole")
        return (h / lam) / (p.c**2 - p.sy2 * h)

    return f


# --- lattice averages ------------------------------------------------------------------
def mode_average(lam: np.ndarray, f: Weight) -> float:
    """(1/N) sum_q lambda_q f(lambda_q)."""
    return np.mean(lam * f(lam))


def ring_mode_average(n_sites: int, spacing: float, length: float, f: Weight) -> float:
    _, lam = lattice_spectrum(ring_kernel(n_sites, spacing, length), spacing)
    return mode_average(lam, f)


def torus_mode_average(side: int, spacing: float, length: float, f: Weight) -> float:
    lam = np.fft.fft2(torus_kernel_row(side, spacing, length)).real.ravel()
    return mode_average(lam, f)


# --- report -------------------------------------------------------------------------------
def relative_error(lattice: float, continuum: float) -> float:
    return abs(lattice - continuum) / abs(continuum)


def report_convergence(label: str, sizes, lattice_fn, continuum: float, spacing: float, length: float) -> list[float]:
    print(f"\n{label}: continuum value {continuum:.12f}")
    print(f"{'N':>7} {'L/l':>7} {'lattice':>18} {'rel. error':>12}")
    errors = []
    for n in sizes:
        value = lattice_fn(n)
        errors.append(relative_error(value, continuum))
        print(f"{n:>7d} {n * spacing / length:>7.1f} {value:>18.12f} {errors[-1]:>12.2e}")
    return errors


def main() -> None:
    a, ell = SPACING, LENGTH
    print(f"spacing a = {a}, kernel width l = {ell}, cutoff Lambda = pi/a = {np.pi / a:.4f}")

    # (i) lambda_q = lambda(q)/a at the reciprocal-lattice points, up to aliasing
    q, lam = lattice_spectrum(ring_kernel(64, a, ell), a)
    print(f"max |lambda_q - lambda(q)/a| / lambda_0 (N = 64) : {np.max(np.abs(lam - continuum_spectrum(q, ell) / a)) / lam.max():.2e}")
    print(f"aliasing floor e^(-l^2 Lambda^2 / 2)             : {aliasing_floor(a, ell):.2e}")

    # (ii) closed-form moments: (1/N) sum lambda_q = K(0) = 1, (1/N) sum lambda_q^2 = sum_j K_0j^2 -> sqrt(pi) l / a
    one = lambda lam_: np.ones_like(lam_)
    ident = lambda lam_: lam_
    print(f"\n{'moment':>22} {'continuum':>16} {'closed form':>16} {'ring N=256':>16}")
    for label, f, exact in (("(1/N) sum lambda_q", one, 1.0), ("(1/N) sum lambda_q^2", ident, np.sqrt(np.pi) * ell / a)):
        print(f"{label:>22} {continuum_mode_average(a, ell, f):>16.12f} {exact:>16.12f} {ring_mode_average(256, a, ell, f):>16.12f}")

    # (iii) the eq:linear-soln summand, d = 1 ring
    p1 = SolutionParams(**PARAMS_1D)
    f1 = linear_solution_weight(p1)
    errors_1d = report_convergence("d = 1, (1/N) sum_q h_q / (c^2 - sY^2 h_q)", RING_SIZES,
                                   lambda n: ring_mode_average(n, a, ell, f1), continuum_mode_average(a, ell, f1), a, ell)

    # (iv) the same on a d = 2 torus, square Brillouin zone
    p2 = SolutionParams(**PARAMS_2D)
    f2 = linear_solution_weight(p2)
    errors_2d = report_convergence("d = 2, (1/N) sum_q h_q / (c^2 - sY^2 h_q)", TORUS_SIDES,
                                   lambda n: torus_mode_average(n, a, ell, f2), continuum_mode_average_2d(a, ell, f2), a, ell)

    # (v) the penalty M = K^{-1} carries a^{2d}; without a nugget it is cutoff-dominated
    x = np.arange(3) * a
    fmt = lambda v: "  ".join(f"{x_:>13.8g}" for x_ in v)
    for nugget in (0.0, NUGGET):
        M = np.linalg.inv(ring_kernel(128, a, ell, nugget))
        pred = continuum_inverse_kernel(x, a, ell, nugget)
        print(f"\nK + {nugget} I, ring N = 128, j = 0, 1, 2")
        print(f"  lattice (K^-1)_0j                      : {fmt(M[0, :3])}")
        print(f"  a^2 int_BZ dq/2pi e^(iqx_j)/lambda(q)  : {fmt(pred)}")
        print(f"  max relative difference                : {np.max(np.abs(M[0, :3] - pred) / np.abs(pred)):.2e}")

    floor = 10 * aliasing_floor(a, ell)
    if errors_1d[-1] > floor or errors_2d[-1] > floor:
        raise AssertionError(f"dictionary check failed: final errors {errors_1d[-1]:.2e} (d=1), {errors_2d[-1]:.2e} (d=2)")
    print(f"\ndictionary factors confirmed: final rel. errors < {floor:.1e} (10 x aliasing floor) in d = 1, 2")


if __name__ == "__main__":
    main()
