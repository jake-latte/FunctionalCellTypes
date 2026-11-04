"""Check the discrete-to-continuum dictionary of latex/notes/continuum_limit.tex.

Ground truth: a ring of N sites with spacing a and the squared-exponential kernel
    K_ij = exp(-r_ij^2 / (2 l^2)),   r_ij the ring distance,
whose continuum spectrum is known in closed form, lambda(q) = sqrt(2 pi) l exp(-l^2 q^2 / 2).
The dictionary claims, at fixed spacing a and growing volume L = N a,
    lambda_q = lambda(q) / a^d,
    (1/N) sum_q lambda_q f(lambda_q)  ->  a^d int_{BZ} d^dq/(2 pi)^d (lambda(q)/a^d) f(lambda(q)/a^d),
    (K^{-1})_ij                       ->  a^{2d} int_{BZ} d^dq/(2 pi)^d e^{iq(x-x')} / lambda(q),
with the Brillouin zone |q_alpha| < pi/a. Each claim is tested with the spacing a != 1, so a
missing factor of a would show. Two limits of the dictionary are also exhibited. The lattice
eigenvalue is the aliased spectrum (1/a) sum_n lambda(q + 2 pi n/a): its relative error is
largest at the zone edge, e^{-l^2 Lambda^2/2}, while the mode average of a smooth weight F
settles on the smaller floor F'(0) x (spectral weight of lambda outside the zone) / value,
which is what the finite-size error converges to. And for the pure squared-exponential kernel
the inverse is dominated by modes at the cutoff (1/lambda(q) grows as e^{+l^2 q^2/2}), so its
lattice and cutoff-continuum values differ at O(1); the a^{2d} rule is therefore checked on
K + nugget * I, whose spectrum is bounded below. The smooth weight is the summand of the
corrected eq:linear-soln,
    lambda_q f(lambda_q) = h_q / (c^2 - sY^2 h_q),
    h_q = lambda_q (lambda_q sX^2 + sigma^2) / (1 + g^2 (lambda_q sX^2 + sigma^2)),
and the finite-size error is quoted for increasing N: it falls as e^{-L/xi}, with 1/xi the
imaginary part of the nearest pole of the summand, which is compared with the exact pole.
A torus repeats the check in d = 2 on the square zone. Finally the d = 2 power-law spectrum
gamma/(q^2 + m^2) cut at Lambda, for which the note's eq:linear-soln is UV-sensitive, is solved
for c at accessible Lambda/m to show how far the solution is from its log(Lambda^2/m^2)
asymptote.

Run from the git root:  uv run python latex/notes/scripts/continuum_limit_check.py   (seconds)
"""

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

# --- parameters quoted in latex/notes/continuum_limit.tex -------------------------
SPACING = 0.5  # a, held fixed as N grows
LENGTH = 1.0  # l, kernel width (l / a = 2, zone-edge aliasing error e^{-2 pi^2 l^2 / a^2})
RING_SIZES = (4, 8, 16, 32, 64, 128, 256)
TORUS_SIDES = (4, 8, 16, 32, 64)
N_QUAD = 2000  # Gauss-Legendre nodes per dimension for the continuum integrals
# hat s_X^2, hat s_Y^2, sigma^2, g^2, c; c is above the pole (lambda_0 is larger in d = 2)
PARAMS_1D = {"sx2": 0.7, "sy2": 0.25, "sig2": 0.2, "g2": 0.3, "c": 2.0}
PARAMS_2D = {"sx2": 0.7, "sy2": 0.25, "sig2": 0.2, "g2": 0.3, "c": 6.0}
NUGGET = 0.1  # epsilon in K + epsilon I for the inverse-kernel check
INVERSE_RING_SIZE = 128
FLOOR_MARGIN = 2.0  # the pass threshold is this multiple of the derived aliasing floor
# d = 2 power law gamma/(q^2 + m^2) in the continuum (rho = 1), with the delta of
# rg_diagonal_reduction.tex; 1/m = 2 lattice units mirrors l = 2a above
POWER_LAW = {"gamma": 1.0, "mass": 0.5, "delta": 0.5}
POWER_LAW_CUTOFFS = (2e1, 2e2, 2e3, 2e4, 2e5, 2e6)  # Lambda / m

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
    """Distance from site 0 to every site of a periodic ring: a * min(k, N - k)."""
    k = np.arange(n_sites)
    return spacing * np.minimum(k, n_sites - k)


def ring_kernel(n_sites: int, spacing: float, length: float, nugget: float = 0.0) -> np.ndarray:
    """Circulant squared-exponential kernel on a periodic ring, K_00 = 1 + nugget."""
    row = np.exp(-(ring_distance(n_sites, spacing) ** 2) / (2 * length**2))
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
    """lambda(q) = int dx K(x) e^{-iqx} for K(x) = exp(-x^2 / 2 l^2), in d = 1 (per axis in d = 2)."""
    return np.sqrt(2 * np.pi) * length * np.exp(-(length**2) * q**2 / 2)


def zone_edge_error(spacing: float, length: float) -> float:
    """Largest relative error of lambda_q = lambda(q)/a, reached at the zone edge q = pi/a.

    There the nearest alias lambda(q - 2 pi/a) equals lambda(q) itself, so the error relative
    to lambda(0) is e^{-l^2 Lambda^2/2}.
    """
    return math.exp(-(length**2) * (math.pi / spacing) ** 2 / 2)


def spectral_weight_outside_zone(spacing: float, length: float, dim: int) -> float:
    """Fraction of int d^dq/(2 pi)^d lambda(q) = K(0) that lies outside the square zone.

    Per axis the weight inside |q| < pi/a is erf(l Lambda / sqrt 2), and the kernel factorises.
    """
    inside_per_axis = math.erf(length * math.pi / spacing / math.sqrt(2))
    return 1 - inside_per_axis**dim


def brillouin_nodes(spacing: float, n_quad: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on the zone (-pi/a, pi/a)."""
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    cutoff = np.pi / spacing
    return cutoff * nodes, cutoff * weights


def continuum_mode_average(spacing: float, length: float, f: Weight, n_quad: int = N_QUAD) -> float:
    """a int_{-pi/a}^{pi/a} dq/(2 pi) lambda_q f(lambda_q), lambda_q = lambda(q)/a.

    This is the d = 1 image of (1/N) sum_q.
    """
    q, w = brillouin_nodes(spacing, n_quad)
    lam = continuum_spectrum(q, length) / spacing
    return spacing * np.sum(w * lam * f(lam)) / (2 * np.pi)


def continuum_mode_average_2d(
    spacing: float, length: float, f: Weight, n_quad: int = N_QUAD // 4
) -> float:
    """a^2 int_{BZ} d^2q/(2 pi)^2 lambda_q f(lambda_q) on the square zone, lambda_q = lambda(q)/a^2."""
    q, w = brillouin_nodes(spacing, n_quad)
    lam1 = continuum_spectrum(q, length)
    lam = lam1[:, None] * lam1[None, :] / spacing**2  # the kernel factorises over the two axes
    return spacing**2 * np.sum(w[:, None] * w[None, :] * lam * f(lam)) / (2 * np.pi) ** 2


def continuum_inverse_kernel(
    x: np.ndarray, spacing: float, length: float, nugget: float = 0.0, n_quad: int = N_QUAD
) -> np.ndarray:
    """a^2 int_{BZ} dq/(2 pi) e^{iqx} / lambda(q): the dictionary's image of (K^{-1})_{0j}.

    A nugget epsilon delta_ij on the lattice is a flat epsilon a^d in the continuum spectrum.
    """
    q, w = brillouin_nodes(spacing, n_quad)
    phase = np.cos(q[None, :] * x[:, None])  # the sine part vanishes by symmetry
    lam = continuum_spectrum(q, length) + nugget * spacing
    return spacing**2 * np.sum(w * phase / lam, axis=1) / (2 * np.pi)


def lattice_lambda_zero(spacing: float, length: float, dim: int) -> float:
    """lambda_0 = lambda(0)/a^d of the squared-exponential prior, the peak of the spectrum."""
    return float(continuum_spectrum(np.zeros(1), length)[0]) ** dim / spacing**dim


# --- the smooth weight: the summand of eq:linear-soln --------------------------------
def h_of_lambda(lam: np.ndarray, p: SolutionParams) -> np.ndarray:
    """h_q of the corrected eq:linear-soln as a function of lambda_q (increasing in lambda)."""
    u = lam * p.sx2 + p.sig2
    return lam * u / (1 + p.g2 * u)


def pole_of_c(p: SolutionParams, lam0: float) -> float:
    """c_* = sY sqrt(h(lambda_0)): the physical branch of eq:linear-soln needs c > c_*."""
    return math.sqrt(p.sy2 * h_of_lambda(np.array(lam0), p))


def linear_solution_weight(p: SolutionParams, lam0: float) -> Weight:
    """f with lambda f(lambda) = h / (c^2 - sY^2 h), for a spectrum that peaks at lambda_0.

    Raises at construction if c is not above the pole c_*: the branch with every d_q > 0 does
    not exist there, and the mode average would silently sum across the singularity.
    """
    pole = pole_of_c(p, lam0)
    if p.c <= pole:
        raise ValueError(f"c = {p.c} is not above the pole c_* = {pole:.4f}; need c > c_*")

    def f(lam: np.ndarray) -> np.ndarray:
        h = h_of_lambda(lam, p)
        return (h / lam) / (p.c**2 - p.sy2 * h)

    return f


def tail_coefficients(p: SolutionParams, rho: float) -> tuple[float, float]:
    """Coefficients (c_1, c_2) of eq:tail: lambda H / (c^2 - sY^2 rho lambda H) = c_1 lambda + c_2 lambda^2 + ...

    c_1 is also F'(0) of the lattice weight F(lambda_q) = h_q / (c^2 - sY^2 h_q), which sets the
    aliasing floor. c_2 collects the linear term of H and the first term of the geometric series
    in sY^2 rho lambda H / c^2.
    """
    h0 = p.sig2 / (1 + p.g2 * p.sig2)  # H at lambda = 0, the pure-noise mode variance
    c1 = h0 / p.c**2
    c2 = rho * p.sx2 / ((1 + p.g2 * p.sig2) ** 2 * p.c**2) + rho * p.sy2 * h0**2 / p.c**4
    return c1, c2


def aliasing_floor(
    p: SolutionParams, continuum: float, spacing: float, length: float, dim: int
) -> float:
    """Relative error the mode average of the summand settles on as N -> infinity.

    The lattice sum converges to the integral of F over the aliased spectrum; to first order
    in the aliased weight the difference is F'(0) times the spectral weight outside the zone,
    because the aliases only matter where lambda(q) is already tiny.
    """
    slope_at_zero, _ = tail_coefficients(p, rho=1.0)
    return slope_at_zero * spectral_weight_outside_zone(spacing, length, dim) / abs(continuum)


def spectrum_at_learning_pole(p: SolutionParams) -> float:
    """The lambda at which sY^2 h(lambda) = c^2, i.e. where the summand's denominator vanishes.

    h = T is a quadratic in lambda; the positive root is returned.
    """
    target = p.c**2 / p.sy2
    b = p.sig2 - target * p.g2 * p.sx2
    return (-b + math.sqrt(b**2 + 4 * p.sx2 * target * (1 + p.g2 * p.sig2))) / (2 * p.sx2)


def learning_pole_imag_q(p: SolutionParams, spacing: float, length: float) -> float:
    """Im q of the pole of the summand nearest the real axis for the squared-exponential prior.

    On the imaginary axis lambda(i kappa)/a = lambda_0 e^{l^2 kappa^2/2} grows, and the pole sits
    where it reaches the value of spectrum_at_learning_pole; 1/kappa is the correlation length
    xi_* that controls the finite-size error e^{-L/xi_*}.
    """
    lam0 = lattice_lambda_zero(spacing, length, dim=1)
    return math.sqrt(2 * math.log(spectrum_at_learning_pole(p) / lam0)) / length


# --- lattice averages ------------------------------------------------------------------
def mode_average(lam: np.ndarray, f: Weight) -> float:
    """(1/N) sum_q lambda_q f(lambda_q)."""
    return float(np.mean(lam * f(lam)))


def ring_mode_average(n_sites: int, spacing: float, length: float, f: Weight) -> float:
    """(1/N) sum_q lambda_q f(lambda_q) on a ring of n_sites, lambda_q from the DFT of K."""
    _, lam = lattice_spectrum(ring_kernel(n_sites, spacing, length), spacing)
    return mode_average(lam, f)


def torus_mode_average(side: int, spacing: float, length: float, f: Weight) -> float:
    """(1/N) sum_q lambda_q f(lambda_q) on a side x side torus, N = side^2 modes."""
    lam = np.fft.fft2(torus_kernel_row(side, spacing, length)).real.ravel()
    return mode_average(lam, f)


# --- convergence and the inverse ---------------------------------------------------------
def relative_error(lattice: float, continuum: float) -> float:
    """|lattice - continuum| / |continuum|."""
    return abs(lattice - continuum) / abs(continuum)


def convergence_errors(
    sizes: Sequence[int], lattice_fn: Callable[[int], float], continuum: float
) -> list[float]:
    """Relative error of lattice_fn(n) against the continuum value, for each n in sizes."""
    return [relative_error(lattice_fn(n), continuum) for n in sizes]


def fitted_correlation_length(
    sizes: Sequence[int], errors: Sequence[float], spacing: float
) -> float:
    """xi from error ~ e^{-L/xi} between two sizes, L = N a (d = 1).

    Both errors must sit on the exponential part of the curve, i.e. above the aliasing floor.
    """
    (n_a, n_b), (e_a, e_b) = sizes, errors
    return (n_b - n_a) * spacing / math.log(e_a / e_b)


def inverse_kernel_check(
    nugget: float,
    spacing: float = SPACING,
    length: float = LENGTH,
    n_sites: int = INVERSE_RING_SIZE,
    n_points: int = 3,
) -> tuple[np.ndarray, np.ndarray, float]:
    """(K + nugget I)^{-1}_{0j} on the ring, its a^2-rule prediction, and their max relative gap."""
    x = np.arange(n_points) * spacing
    lattice = np.linalg.inv(ring_kernel(n_sites, spacing, length, nugget))[0, :n_points]
    predicted = continuum_inverse_kernel(x, spacing, length, nugget)
    return lattice, predicted, float(np.max(np.abs(lattice - predicted) / np.abs(predicted)))


# --- d = 2 power law: the UV-sensitive case ---------------------------------------------
@dataclass(frozen=True)
class PowerLawSolution:
    """Root of eq:linear-soln for lambda(q) = gamma/(q^2 + m^2) in d = 2, cut at Lambda."""

    cutoff: float  # Lambda / m
    c: float
    c_star: float  # the pole sY sqrt(h(0))
    prior_term: float  # int d^2q/(2 pi)^2 lambda H, the UV-divergent noise-driven part
    learning_term: float  # c - delta - prior_term = sY^2 int (lambda H)^2 / (c^2 - sY^2 lambda H)
    xi_star: float  # correlation length from the learning-induced pole, in units of 1/m


def radial_nodes_2d(cutoff: float, mass: float, n_panels: int = 400, n_nodes: int = 32):
    """Nodes and weights for int_0^cutoff q dq/(2 pi), on log-spaced panels from 1e-6 m.

    Log spacing is what resolves both the pole scale 1/xi_* near q = 0 and the slow
    1/q^2 tail out to Lambda/m = 1e6 with one rule.
    """
    edges = np.geomspace(1e-6 * mass, cutoff, n_panels + 1)
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    half = (edges[1:] - edges[:-1]) / 2
    mid = (edges[1:] + edges[:-1]) / 2
    q = (mid[:, None] + half[:, None] * nodes[None, :]).ravel()
    w = (half[:, None] * weights[None, :]).ravel()
    return q, w * q / (2 * np.pi)


def solve_power_law_2d(
    p: SolutionParams, gamma: float, mass: float, delta: float, cutoff_over_mass: float
) -> PowerLawSolution:
    """Solve 1 - delta/c = c int d^2q/(2 pi)^2 lambda H / (c^2 - sY^2 lambda H) for c (rho = 1).

    The residual is strictly increasing in c above the pole (rg_diagonal_reduction.tex), so a
    bisection on (c_*, c_* + 100) is safe; p.c is ignored.
    """
    q, w = radial_nodes_2d(cutoff_over_mass * mass, mass)
    h = h_of_lambda(gamma / (q**2 + mass**2), p)
    c_star = pole_of_c(p, gamma / mass**2)

    def residual(c: float) -> float:
        return 1 - delta / c - c * float(np.sum(w * h / (c**2 - p.sy2 * h)))

    lo, hi = c_star * (1 + 1e-12), c_star + 100.0
    if residual(lo) > 0 or residual(hi) < 0:
        raise ValueError("no root of eq:linear-soln bracketed above the pole")
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if residual(mid) < 0 else (lo, mid)
    c = (lo + hi) / 2
    prior_term = float(np.sum(w * h))
    lam_pole = spectrum_at_learning_pole(SolutionParams(p.sx2, p.sy2, p.sig2, p.g2, c))
    xi_star = 1 / math.sqrt(mass**2 - gamma / lam_pole) * mass
    return PowerLawSolution(
        cutoff_over_mass, c, c_star, prior_term, c - delta - prior_term, xi_star
    )


def log_asymptote_slope(p: SolutionParams, gamma: float) -> float:
    """dc / dlog(Lambda^2/m^2) as Lambda -> infinity: H(0) gamma / (4 pi), eq:c-log."""
    return p.sig2 / (1 + p.g2 * p.sig2) * gamma / (4 * math.pi)


def log_slopes(solutions: Sequence[PowerLawSolution]) -> list[float]:
    """Finite-difference dc / dlog(Lambda^2/m^2) between successive cutoffs."""
    return [
        (b.c - a.c) / math.log(b.cutoff**2 / a.cutoff**2) for a, b in itertools.pairwise(solutions)
    ]


# --- printing ---------------------------------------------------------------------------
def print_convergence(
    label: str, sizes: Sequence[int], errors: Sequence[float], continuum: float, dim: int
) -> None:
    print(f"\n{label}: continuum value {continuum:.12f}")
    print(f"{'N':>7} {'L/l':>7} {'rel. error':>12}")
    for n, err in zip(sizes, errors):
        print(f"{n**dim:>7d} {n * SPACING / LENGTH:>7.1f} {err:>12.2e}")


def print_inverse(
    nugget: float, lattice: np.ndarray, predicted: np.ndarray, max_rel: float
) -> None:
    def row(values: np.ndarray) -> str:
        return "  ".join(f"{v:>13.8g}" for v in values)

    print(f"\nK + {nugget} I, ring N = {INVERSE_RING_SIZE}, j = 0, 1, 2")
    print(f"  lattice (K^-1)_0j                      : {row(lattice)}")
    print(f"  a^2 int_BZ dq/2pi e^(iqx_j)/lambda(q)  : {row(predicted)}")
    print(f"  max relative difference                : {max_rel:.2e}")


def print_power_law(solutions: Sequence[PowerLawSolution], asymptote: float) -> None:
    print(
        f"\nd = 2 power law gamma/(q^2 + m^2), gamma = {POWER_LAW['gamma']}, m = {POWER_LAW['mass']},"
        f" delta = {POWER_LAW['delta']}: c_* = {solutions[0].c_star:.4f}"
    )
    print(f"asymptotic slope dc/dlog(Lambda^2/m^2) = H(0) gamma/4pi = {asymptote:.5f}")
    print(
        f"{'Lambda/m':>9} {'c':>8} {'c - c_*':>9} {'slope':>9} {'int lam H':>10}"
        f" {'learning':>9} {'xi_* m':>8}"
    )
    for s, slope in zip(solutions, [math.nan, *log_slopes(solutions)]):
        print(
            f"{s.cutoff:>9.0e} {s.c:>8.4f} {s.c - s.c_star:>9.4f} {slope:>9.5f}"
            f" {s.prior_term:>10.4f} {s.learning_term:>9.4f} {s.xi_star:>8.2f}"
        )


def main() -> None:
    a, ell = SPACING, LENGTH
    print(f"spacing a = {a}, kernel width l = {ell}, cutoff Lambda = pi/a = {np.pi / a:.4f}")

    # (i) lambda_q = lambda(q)/a at the reciprocal-lattice points, up to aliasing
    q, lam = lattice_spectrum(ring_kernel(64, a, ell), a)
    edge = np.max(np.abs(lam - continuum_spectrum(q, ell) / a)) / lam.max()
    print(f"max |lambda_q - lambda(q)/a| / lambda_0 (N = 64) : {edge:.2e}")
    print(f"zone-edge aliasing error e^(-l^2 Lambda^2 / 2)   : {zone_edge_error(a, ell):.2e}")
    print(
        f"spectral weight outside the zone, d = 1, 2       : "
        f"{spectral_weight_outside_zone(a, ell, 1):.2e}, {spectral_weight_outside_zone(a, ell, 2):.2e}"
    )

    # (ii) closed-form moments: (1/N) sum lambda_q = K(0) = 1, (1/N) sum lambda_q^2 -> sqrt(pi) l / a
    def one(lam_: np.ndarray) -> np.ndarray:
        return np.ones_like(lam_)

    def identity(lam_: np.ndarray) -> np.ndarray:
        return lam_

    moments = (
        ("(1/N) sum lambda_q", one, 1.0),
        ("(1/N) sum lambda_q^2", identity, np.sqrt(np.pi) * ell / a),
    )
    print(f"\n{'moment':>22} {'continuum':>16} {'closed form':>16} {'ring N=256':>16}")
    for label, f, exact in moments:
        print(
            f"{label:>22} {continuum_mode_average(a, ell, f):>16.12f} {exact:>16.12f}"
            f" {ring_mode_average(256, a, ell, f):>16.12f}"
        )

    # (iii) the eq:linear-soln summand, d = 1 ring: exponential convergence down to the floor
    p1 = SolutionParams(**PARAMS_1D)
    f1 = linear_solution_weight(p1, lattice_lambda_zero(a, ell, dim=1))
    cont_1d = continuum_mode_average(a, ell, f1)
    errors_1d = convergence_errors(RING_SIZES, lambda n: ring_mode_average(n, a, ell, f1), cont_1d)
    floor_1d = aliasing_floor(p1, cont_1d, a, ell, dim=1)
    print_convergence(
        "d = 1, (1/N) sum_q h_q / (c^2 - sY^2 h_q)", RING_SIZES, errors_1d, cont_1d, 1
    )
    xi_fit = fitted_correlation_length(RING_SIZES[2:4], errors_1d[2:4], a)
    xi_pole = 1 / learning_pole_imag_q(p1, a, ell)
    print(
        f"xi/l from the L = 8l -> 16l drop: {xi_fit / ell:.3f}; from the learning pole"
        f" at Im q = {1 / xi_pole * ell:.3f}/l: {xi_pole / ell:.3f}"
    )
    print(f"derived aliasing floor F'(0) x erfc(l Lambda/sqrt2) / value = {floor_1d:.3e}")

    # (iv) the same on a d = 2 torus, square Brillouin zone
    p2 = SolutionParams(**PARAMS_2D)
    f2 = linear_solution_weight(p2, lattice_lambda_zero(a, ell, dim=2))
    cont_2d = continuum_mode_average_2d(a, ell, f2)
    errors_2d = convergence_errors(
        TORUS_SIDES, lambda n: torus_mode_average(n, a, ell, f2), cont_2d
    )
    floor_2d = aliasing_floor(p2, cont_2d, a, ell, dim=2)
    print_convergence(
        "d = 2, (1/N) sum_q h_q / (c^2 - sY^2 h_q)", TORUS_SIDES, errors_2d, cont_2d, 2
    )
    print(f"derived aliasing floor F'(0) x (1 - erf^2(l Lambda/sqrt2)) / value = {floor_2d:.3e}")

    # (v) the penalty M = K^{-1} carries a^{2d}; without a nugget it is cutoff-dominated
    for nugget in (0.0, NUGGET):
        print_inverse(nugget, *inverse_kernel_check(nugget))

    # (vi) d = 2 power law: UV-sensitive, but pinned near the pole at accessible cutoffs
    solutions = [solve_power_law_2d(p1, **POWER_LAW, cutoff_over_mass=r) for r in POWER_LAW_CUTOFFS]
    print_power_law(solutions, log_asymptote_slope(p1, POWER_LAW["gamma"]))

    if errors_1d[-1] > FLOOR_MARGIN * floor_1d or errors_2d[-1] > FLOOR_MARGIN * floor_2d:
        raise AssertionError(
            f"dictionary check failed: final errors {errors_1d[-1]:.2e} (d=1), {errors_2d[-1]:.2e}"
            f" (d=2) exceed {FLOOR_MARGIN} x the derived floors {floor_1d:.2e}, {floor_2d:.2e}"
        )
    print(
        f"\ndictionary factors confirmed: final rel. errors within {FLOOR_MARGIN:.0f} x the derived"
        f" aliasing floor in d = 1, 2"
    )


if __name__ == "__main__":
    main()
