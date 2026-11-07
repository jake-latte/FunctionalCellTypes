"""Numerical checks quoted in latex/notes/linear_solution_limits.tex.

Ground truth is the ansatz-free NP x NP fixed point of the linear one-hidden-layer
network on a ring (section A); every other section checks a closed form or an
asymptotic statement of the note against the per-mode scalar equation
    1 - delta/c = (c/N) sum_q h_q / (c^2 - sY2 h_q),
    h_q = lam_q (lam_q sX2 + sig2) / (1 + r (lam_q sX2 + sig2)),
solved on the branch c > sqrt(sY2 max_q h_q), and d_q = c^2 h_q / (lam_q (c^2 - sY2 h_q)),
with r the rate-penalty coefficient (the notes' G = r I).

Sections (each `section_*` computes and returns the numbers, `report_*` prints them):
  A  re-derivation: ansatz-free fixed point vs the scalar equation (exact and approximate ansatz)
  B  limiting cases: sigma -> 0, r -> 0 (flat spectrum closed form), k0 = k1 sweep, Gaussian ring
  C  latent task: Z^T Z / P, singular values, the mean direction, the ansatz condition
  D  functional cell types: monotonicity, rate-penalty whitening, target condensation,
     continuum width
  E  predictions: the regime numbers (tau, gamma, rho) of the parameter sets listed in the note
  F  figure latex/notes/figures/linear_solution_limits_profiles.png (per-mode spectra d_q)

Run from the git root:  uv run python latex/notes/scripts/check_linear_soln_limits.py
"""

import math
from pathlib import Path

import numpy as np

SEED = 0
# Dropped images of the wrapped Gaussian lie beyond this many ell (as in fct/spatial.py).
N_IMAGE_SIGMAS = 8
PSD_ELLS = [4.0, 8.0, 12.0]  # length scales at which the wrapped and truncated spectra are compared
# PNG rather than PDF: the repository ignores *.pdf, and the note must build from a fresh clone.
FIGURE = Path(__file__).resolve().parents[1] / "figures" / "linear_solution_limits_profiles.png"

# --- section A: ansatz-free fixed point (small, so the NP x NP inverse is cheap) --------------
A_N, A_P, A_M, A_NOUT, A_K = 12, 5, 6, 6, 2  # sites, patterns, m, n, latents
A_ELL = np.sqrt(2.0)  # K_ij = exp(-Delta^2 / (2 ell^2)), ring distance Delta
A_K0, A_K1, A_SIG2, A_R, A_DELTA = 0.7, 1.3, 0.2, 0.3, 0.5
A_C_START, A_NEWTON_STEPS, A_TOL = 30.0, 100, 1e-12  # Newton from C = 30 I, well above the pole
A_WY_SKEW = np.diag([1.0, 0.7])  # non-orthonormal W_Y for the approximate case
FD_EPS = 1e-6  # finite-difference step of the Newton Jacobian
MIN_NEWTON_STEP = 1e-14  # a line search that halves the step below this has failed

# --- sections B-D: a larger ring where the continuum formulae are accurate --------------------
N, ELL = 64, 4.0
BIG_N = 4096  # for the continuum (large-N) check
SY_CONTINUUM = [4.0, 8.0, 16.0, 32.0]  # moderate targets: Lorentzian width >> 1/N
M_IN, N_OUT, KAPPA = 6, 6, 1.0  # m, n, kappa for the k0 = k1 sweep
K_SWEEP = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]
SY_SWEEP = [1.0, 10.0, 100.0, 1000.0]  # target strength sY for the condensation check
# a generic finite-everything mode; r is the rate-penalty coefficient (the notes' G = r I)
BASE = {"sx2": 1.0, "sig2": 0.1, "r": 0.2, "delta": 0.5}
LATENT_P, LATENT_K, LATENT_A = 20000, 3, 1.0  # Z ~ U[0, a]^(P x K) for the law of large numbers
LIMIT_EPS = 1e-8  # stands in for sig2 -> 0 or sY2 -> 0 (small enough for the precision quoted)
NO_RIDGE = 1e-12  # delta small enough to stand in for delta -> 0

# --- section E: parameter sets of the numbered predictions (ring, K = 3 latents) --------------
PRED_N, PRED_K, PRED_A, PRED_M, PRED_NOUT = 64, 3, 1.0, 8, 8
PRED_P = 256
PREDICTIONS = {
    # name: (ell, k0, k1, sig2, r, beta)
    "P1 prior-shaped (weak target, no rate penalty)": (4.0, 1.0, 1.0, 0.01, 0.0, 0.25),
    "P2 rate-penalty whitening": (4.0, 1.0, 1.0, 0.01, 0.1, 0.25),
    "P3 target condensation": (4.0, 1.0, 1.0, 0.01, 0.0, 10.0),
    # P4: sig2 = r = 1 puts the floor sigma_0^2 = 1/2 a factor 2 below the bare sig2, so a
    # factor-1.5 tolerance on the measured floor can tell the two apart.
    "P4 white null modes": (4.0, 1.0, 1.0, 1.0, 1.0, 0.25),
    "P5 noise-dominated": (4.0, 0.001, 1.0, 1.0, 0.0, 0.25),
}
P3_BETA_SWEEP = [0.25, 1.0, 2.0, 5.0, 10.0]
P7_ELL_SWEEP = [2.0, 4.0, 8.0]
P8_K_SWEEP = [0.01, 0.1, 1.0, 10.0, 100.0]
RATIO_Q_MAX = 12  # d_q(mode 1) / d_q(mode 2) is printed for |q| = 0, ..., RATIO_Q_MAX
FIGURE_Q_MAX = 20  # the figure shows |q| <= FIGURE_Q_MAX
FIGURE_PANELS = [
    "P1 prior-shaped (weak target, no rate penalty)",
    "P2 rate-penalty whitening",
    "P3 target condensation",
]


# ============================================================ building blocks
def ring_distance(n_sites):
    """Ring distance min(Delta, N - Delta) from site 0 to sites Delta = 0, ..., N-1."""
    dist = np.arange(n_sites)
    return np.minimum(dist, n_sites - dist)


def wrapped_gaussian(n_sites, ell):
    """Row 0 of the wrapped (image-summed) Gaussian on the ring, normalised to K_ii = 1:
    K(Delta) = sum_k exp(-(Delta + kN)^2 / 2 ell^2) / sum_k exp(-(kN)^2 / 2 ell^2), the kernel of
    fct/spatial.py. It is positive definite at every ell (its spectrum is a sampled Gaussian
    plus positive aliases), unlike the Gaussian of the minimum-image distance."""
    n_images = math.ceil(N_IMAGE_SIGMAS * ell / n_sites) + 1
    shifts = n_sites * np.arange(-n_images, n_images + 1)
    dist = np.arange(n_sites)
    row = np.exp(-((dist[:, None] + shifts) ** 2) / (2 * ell**2)).sum(axis=1)
    return row / row[0]


def truncated_gaussian(n_sites, ell):
    """Row 0 of the Gaussian of the minimum-image ring distance, K_ii = 1. It drops the
    images of `wrapped_gaussian`, a tail of size exp(-N^2 / 8 ell^2) at the antipode, and
    is not positive definite once ell is comparable to N / 8 (ring_check quotes this)."""
    return np.exp(-(ring_distance(n_sites) ** 2) / (2 * ell**2))


def ring_kernel(n_sites, ell):
    """The N x N circulant kernel of `wrapped_gaussian`."""
    row = wrapped_gaussian(n_sites, ell)
    return np.array([np.roll(row, i) for i in range(n_sites)])


def ring_spectrum(n_sites, ell):
    """Eigenvalues lam_q of ring_kernel, q = 0, 1, ..., N-1 in DFT order (lam_q = lam_{-q})."""
    return np.fft.fft(wrapped_gaussian(n_sites, ell)).real


def pi_of(lam, sx2, sig2, r):
    """Prior-shrunk variance pi_q = alpha_q / (1 + r alpha_q), alpha_q = lam_q sX2 + sig2,
    with r the rate-penalty coefficient (the notes' G = r I)."""
    alpha = lam * sx2 + sig2
    return alpha / (1 + r * alpha)


def h_of(lam, sx2, sig2, r):
    """h_q = lam_q pi_q = lam_q alpha_q / (1 + r alpha_q) (eq:hq of the note), with r
    the rate-penalty coefficient (the notes' G = r I)."""
    return lam * pi_of(lam, sx2, sig2, r)


def bisect(f, lo, hi, n=200):
    """Root of an increasing f on (lo, hi); fails loudly if not bracketed."""
    if not (f(lo) < 0 < f(hi)):
        raise ValueError(f"root not bracketed: f(lo)={f(lo)}, f(hi)={f(hi)}")
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def residual(c, h, sy2, delta):
    """1 - delta/c - (c/N) sum_q h_q / (c^2 - sY2 h_q): increasing in c above the pole."""
    return 1 - delta / c - np.mean(c * h / (c**2 - sy2 * h))


def solve_c(lam, sx2, sy2, sig2, r, delta):
    """The physical root of the per-mode equation (branch above the pole)."""
    h = h_of(lam, sx2, sig2, r)
    pole = np.sqrt(sy2 * h.max())
    # The lower bracket must stay positive when the pole is at 0 (sY2 = 0).
    lo = max(pole * (1 + 1e-13), 1e-12)
    return bisect(lambda c: residual(c, h, sy2, delta), lo, 1e9)


def spectrum_d(lam, sx2, sy2, sig2, r, delta):
    """(c, d_q): the self-consistent c and the per-frequency variance of one data mode."""
    c = solve_c(lam, sx2, sy2, sig2, r, delta)
    h = h_of(lam, sx2, sig2, r)
    # d_q = pi_q * amplification (eq:factor), not h_q / lam_q: lam_q is 0 to round-off in
    # the tail of the wrapped Gaussian and the quotient would be 0/0 there.
    return c, pi_of(lam, sx2, sig2, r) * c**2 / (c**2 - sy2 * h)


def tau_of(lam, sx2, sy2, sig2, r, delta):
    """Target strength tau = sY2 max_q h_q / c^2 in (0, 1)."""
    c = solve_c(lam, sx2, sy2, sig2, r, delta)
    return sy2 * h_of(lam, sx2, sig2, r).max() / c**2


def autocorrelation(d):
    """R(Delta) = (1/N) sum_q d_q e^{2 pi i q Delta / N}, Delta = 0, ..., N-1, of the power spectrum d_q."""
    return np.fft.ifft(d).real


def autocorr_lag1(d):
    """R(1)/R(0) of the tuning map whose power spectrum is d_q (DFT order)."""
    R = autocorrelation(d)
    return R[1] / R[0]


def half_power_q(d):
    """Largest |q| at which d_q - d_{N/2} is at least half of d_0 - d_{N/2} (d in DFT order):
    the floor is the measurable Nyquist value, as in the note's definition of q_{1/2}."""
    q = np.abs(np.fft.fftfreq(d.size) * d.size)
    floor = d[d.size // 2]
    return q[(d - floor) >= 0.5 * (d[0] - floor)].max()


def half_crossing(x, profile):
    """The x at which `profile` (sampled on increasing x, decreasing from profile[0]) falls
    to half its peak, linearly interpolated between the two grid points that bracket it;
    the grid value alone is quantised to the spacing of x."""
    level = 0.5 * profile[0]
    i = np.flatnonzero(profile < level)[0]  # first grid point below half power
    weight = (profile[i - 1] - level) / (profile[i - 1] - profile[i])
    return x[i - 1] + weight * (x[i] - x[i - 1])


# ============================================================ A: ansatz-free fixed point
def orthonormal_columns(rng, rows, cols):
    """A rows x cols matrix with orthonormal columns (Q of a Gaussian matrix)."""
    return np.linalg.qr(rng.normal(size=(rows, cols)))[0]


def latent_task(rng, n_patterns, m, n_out, n_latent, a=1.0, wy_skew=None):
    """(X, Y) = (Z W_X^T, Z W_Y^T) with Z ~ U[0, a]^(P x K) and orthonormal W_X, W_Y;
    `wy_skew` (K x K) right-multiplies W_Y to break the orthonormality of the teacher."""
    Z = rng.uniform(0, a, (n_patterns, n_latent))
    W_X = orthonormal_columns(rng, m, n_latent)
    W_Y = orthonormal_columns(rng, n_out, n_latent)
    if wy_skew is not None:
        W_Y = W_Y @ wy_skew
    return Z @ W_X.T, Z @ W_Y.T


def full_precision(C, K, A, YY, k1, sig2, r, delta):
    """The NP x NP precision (A (x) K + sig2 I)^{-1} - (B (x) K - r I), index order (mu, i)."""
    n_patterns, n_sites = A.shape[0], K.shape[0]
    Ci = np.linalg.inv(C + delta * np.eye(n_patterns))
    B = Ci @ YY @ Ci / k1
    I = np.eye(n_patterns * n_sites)
    return np.linalg.inv(np.kron(A, K) + sig2 * I) - (np.kron(B, K) - r * I)


def kernel_map(C, K, A, YY, k1, sig2, r, delta):
    """C -> (1/N) sum_ij K_ij Cov[h_i, h_j], the hidden-layer kernel of eq:kernel-def."""
    n_patterns, n_sites = A.shape[0], K.shape[0]
    Cov = np.linalg.inv(full_precision(C, K, A, YY, k1, sig2, r, delta))
    Cov = Cov.reshape(n_patterns, n_sites, n_patterns, n_sites)
    return np.einsum("ij,uivj->uv", K, Cov) / n_sites


def _sym_from_upper(v, n):
    """The symmetric n x n matrix whose upper triangle (row-major) is the vector v."""
    C = np.zeros((n, n))
    C[np.triu_indices(n)] = v
    return C + np.triu(C, 1).T


def ansatz_free_fixed_point(K, A, YY, k1, sig2, r, delta, c_start, n_steps, tol):
    """Newton's method (finite-difference Jacobian) on the upper triangle of C for the
    residual C - kernel_map(C), started from C = c_start I. Steps are halved until the
    NP x NP precision stays positive definite: plain damped iteration leaves the physical
    branch when the target is strong, because the scalar map c -> R(c) has slope below -3
    near the pole."""
    n = A.shape[0]
    iu = np.triu_indices(n)

    def newton_residual(v):
        C = _sym_from_upper(v, n)
        return (C - kernel_map(C, K, A, YY, k1, sig2, r, delta))[iu]

    def is_physical(v):
        precision = full_precision(_sym_from_upper(v, n), K, A, YY, k1, sig2, r, delta)
        return np.linalg.eigvalsh(precision).min() > 0

    v = (c_start * np.eye(n))[iu]
    if not is_physical(v):
        raise ValueError("starting point is below the pole")
    for _ in range(n_steps):
        res = newton_residual(v)
        if np.abs(res).max() < tol:
            return _sym_from_upper(v, n)
        J = np.column_stack(
            [(newton_residual(v + FD_EPS * e) - res) / FD_EPS for e in np.eye(v.size)]
        )
        step = -np.linalg.solve(J, res)
        while not is_physical(v + step):
            step /= 2
            if np.abs(step).max() < MIN_NEWTON_STEP:
                raise ValueError("line search collapsed: no physical step")
        v = v + step
    raise ValueError(
        f"Newton did not converge: max residual {np.abs(newton_residual(v)).max():.1e}"
    )


def scalar_roots(lam, X, Y, k0, k1, m, sig2, r, delta):
    """Roots of the per-mode equation in the eigenbasis U of X X^T (one root per data mode)."""
    w, U = np.linalg.eigh(X @ X.T)
    sx2 = np.clip(k0 * w / m, 0, None)
    sy2 = np.clip(np.diag(U.T @ Y @ Y.T @ U), 0, None) / k1  # exact when [XX^T, YY^T] = 0
    return np.sort([solve_c(lam, a, b, sig2, r, delta) for a, b in zip(sx2, sy2)])


def fixed_point_case(rng, skew):
    """One section-A case: the ansatz-free fixed point of the latent task against the
    per-mode roots, for an orthonormal (skew None) or skewed teacher."""
    K = ring_kernel(A_N, A_ELL)
    lam = np.linalg.eigvalsh(K)
    X, Y = latent_task(rng, A_P, A_M, A_NOUT, A_K, wy_skew=skew)
    A, YY = A_K0 / A_M * X @ X.T, Y @ Y.T
    C = ansatz_free_fixed_point(
        K, A, YY, A_K1, A_SIG2, A_R, A_DELTA, A_C_START, A_NEWTON_STEPS, A_TOL
    )
    Ct = C + A_DELTA * np.eye(A_P)
    U = np.linalg.eigh(X @ X.T)[1]
    rot = U.T @ Ct @ U
    exact = np.sort(np.linalg.eigvalsh(Ct))
    roots = scalar_roots(lam, X, Y, A_K0, A_K1, A_M, A_SIG2, A_R, A_DELTA)
    precision = full_precision(C, K, A, YY, A_K1, A_SIG2, A_R, A_DELTA)
    return {
        "commutator": np.abs(X @ X.T @ YY - YY @ X @ X.T).max(),
        "min_eig_precision": np.linalg.eigvalsh(precision).min(),
        "off_diagonal": np.abs(rot - np.diag(np.diag(rot))).max(),
        "eig_fixed_point": exact,
        "scalar_roots": roots,
        "max_rel_diff": np.abs(exact / roots - 1).max(),
    }


def section_a():
    """Re-derivation: the ansatz-free NP x NP fixed point against the scalar equation.
    Both cases reseed, so the skewed case is the same task (same Z, W_X, W_Y) with the
    columns of W_Y rescaled, not a fresh draw."""
    cases = [
        ("orthonormal W_Y (ansatz exact)", None),
        ("W_Y skewed (ansatz approximate)", A_WY_SKEW),
    ]
    return {label: fixed_point_case(np.random.default_rng(SEED), skew) for label, skew in cases}


def report_a(res):
    print("=== A: re-derivation against the ansatz-free NP x NP fixed point ===")
    for label, case in res.items():
        print(f"  [{label}]")
        print(f"    max |[XX^T, YY^T]|               : {case['commutator']:.2e}")
        print(f"    min eig of precision             : {case['min_eig_precision']:.4f}")
        print(f"    max off-diag |U^T Ct U|          : {case['off_diagonal']:.2e}")
        print(f"    eig(C + delta I), fixed point    : {np.round(case['eig_fixed_point'], 8)}")
        print(f"    roots of the scalar equation     : {np.round(case['scalar_roots'], 8)}")
        print(f"    max relative difference          : {case['max_rel_diff']:.2e}")


# ============================================================ B: limiting cases
def ring_check(lam):
    """B4: the Gaussian ring spectrum against the continuum formula and its moments."""
    q = np.fft.fftfreq(N) * N
    approx = np.sqrt(2 * np.pi) * ELL * np.exp(-2 * np.pi**2 * ELL**2 * (q / N) ** 2)
    min_eig = [
        (ell, ring_spectrum(N, ell).min(), np.fft.fft(truncated_gaussian(N, ell)).real.min())
        for ell in PSD_ELLS
    ]
    return {
        "lam_0": lam[0],
        "sqrt_2pi_ell": np.sqrt(2 * np.pi) * ELL,
        "max_dev_formula": np.abs(lam - approx).max(),
        "mean_lam": lam.mean(),
        "mean_lam2": (lam**2).mean(),
        "sqrt_pi_ell": np.sqrt(np.pi) * ELL,
        "min_eig": min_eig,  # (ell, min lam_q wrapped, min lam_q truncated)
    }


def sigma_to_zero_check(lam):
    """B1: null data modes lose all variance; latent modes converge to the sigma = 0 formula."""
    c0, d0 = spectrum_d(lam, 0.0, 0.0, LIMIT_EPS, BASE["r"], BASE["delta"])
    c_eps, d_eps = spectrum_d(lam, 1.0, 2.0, LIMIT_EPS, BASE["r"], BASE["delta"])
    c_zero, d_zero = spectrum_d(lam, 1.0, 2.0, 0.0, BASE["r"], BASE["delta"])
    return {
        "null_max_d": d0.max(),
        "null_c_minus_delta": c0 - BASE["delta"],
        "latent_dc": abs(c_eps - c_zero),
        "latent_dd": np.abs(d_eps - d_zero).max(),
    }


def r_to_zero_check(lam):
    """B2: weak target, r = 0: d_q = lam_q sX2 + sig2 exactly and c = delta + mean(h)."""
    sx2, sig2 = 1.0, 0.1
    c, d = spectrum_d(lam, sx2, LIMIT_EPS, sig2, 0.0, BASE["delta"])
    return {
        "max_dev_d": np.abs(d - (lam * sx2 + sig2)).max(),
        "dev_c": abs(c - BASE["delta"] - h_of(lam, sx2, sig2, 0.0).mean()),
    }


def flat_spectrum_check():
    """B2': flat spectrum (K = lam I), delta -> 0: the quadratic closed form of the latent note."""
    flat = np.full(N, 1.3)
    h = h_of(flat, 0.7, 0.2, 0.3)[0]
    return {
        "c": solve_c(flat, 0.7, 2.0, 0.2, 0.3, NO_RIDGE),
        "closed_form": (h + np.sqrt(h**2 + 4 * 2.0 * h)) / 2,
    }


def common_scale_sweep(lam):
    """B3: k0 = k1 = k at sigma = r = 0: tau(k) is non-monotone, maximal near k*."""
    sx2, sy2 = 1.0, 1.0
    rows = []
    for k in K_SWEEP:
        c = solve_c(lam, k * sx2 / M_IN, sy2 / k, 0.0, 0.0, 1 / (k * KAPPA))
        tau = (sy2 / k) * h_of(lam, k * sx2 / M_IN, 0.0, 0.0).max() / c**2
        rows.append((k, c, tau))
    return {
        "pole": lam[0] * np.sqrt(sx2 * sy2 / M_IN),
        "k_star": np.sqrt(M_IN / (KAPPA * sx2 * (lam**2).mean())),
        "rows": rows,
    }


def reduced_form_check(lam):
    """B5: at sigma = r = 0 the equation reduces to
    1 - Delta/x = Theta (x/N) sum l_q^2/(x^2 - l_q^2)
    with x = c/c*, l_q = lam_q/lam_0, Theta = lam_0 sX/sY, Delta = delta/c*, c* = lam_0 sX sY."""
    sx2, sy2, delta = 3.0, 0.5, 2.0
    c_star, ell = lam[0] * np.sqrt(sx2 * sy2), lam / lam[0]
    theta, Delta = lam[0] * np.sqrt(sx2 / sy2), delta / c_star

    def reduced_residual(x):
        return 1 - Delta / x - theta * np.mean(x * ell**2 / (x**2 - ell**2))

    x = bisect(reduced_residual, 1 + 1e-13, 1e9)
    c_full = solve_c(lam, sx2, sy2, 0.0, 0.0, delta)
    return {
        "theta": theta,
        "Delta": Delta,
        "c_star_x": c_star * x,
        "c_full": c_full,
        "tau_reduced": 1 / x**2,
        "tau_full": sy2 * h_of(lam, sx2, 0, 0).max() / c_full**2,
    }


def section_b():
    """Limiting cases on the N = 64, ell = 4 ring."""
    lam = ring_spectrum(N, ELL)
    return {
        "ring": ring_check(lam),
        "sigma0": sigma_to_zero_check(lam),
        "r0": r_to_zero_check(lam),
        "flat": flat_spectrum_check(),
        "k_sweep": common_scale_sweep(lam),
        "reduced": reduced_form_check(lam),
    }


def report_b(res):
    print("=== B: limiting cases ===")
    ring = res["ring"]
    print(
        f"  Gaussian ring N={N}, ell={ELL}: lam_0 = {ring['lam_0']:.6f}, "
        f"sqrt(2 pi) ell = {ring['sqrt_2pi_ell']:.6f}"
    )
    print(f"    max |lam_q - continuum formula| = {ring['max_dev_formula']:.1e}")
    print(
        f"    (1/N) sum lam_q = {ring['mean_lam']:.6f} (= K_ii = 1);  "
        f"(1/N) sum lam_q^2 = {ring['mean_lam2']:.6f}, sqrt(pi) ell = {ring['sqrt_pi_ell']:.6f}"
    )
    for ell, wrapped, truncated in ring["min_eig"]:
        print(
            f"    min_q lam_q at ell = {ell:4.1f}: wrapped Gaussian {wrapped:9.2e}, "
            f"minimum-image Gaussian {truncated:9.2e}"
        )
    s = res["sigma0"]
    print(
        f"  sigma -> 0: null mode (s_X = s_Y = 0) at sig2 = {LIMIT_EPS:.0e}: "
        f"max_q d_q = {s['null_max_d']:.1e}, c - delta = {s['null_c_minus_delta']:.1e}"
    )
    print(
        f"    latent mode: |c({LIMIT_EPS:.0e}) - c(0)| = {s['latent_dc']:.1e}, "
        f"max |d_q({LIMIT_EPS:.0e}) - d_q(0)| = {s['latent_dd']:.1e}"
    )
    zero_rate = res["r0"]
    print(
        f"  r -> 0, weak target (sY2 = {LIMIT_EPS:.0e}): max |d_q - (lam_q sX2 + sig2)| = "
        f"{zero_rate['max_dev_d']:.1e}, |c - delta - mean h| = {zero_rate['dev_c']:.1e}"
    )
    f = res["flat"]
    print(
        f"  flat spectrum, delta -> 0: c = {f['c']:.10f}, "
        f"closed form (h + sqrt(h^2 + 4 sY2 h))/2 = {f['closed_form']:.10f}"
    )
    k = res["k_sweep"]
    print(
        f"  k0 = k1 = k sweep (sigma = r = 0, m = {M_IN}, kappa = {KAPPA}); "
        f"pole c* = lam_0 sX sY / sqrt(m) = {k['pole']:.4f} for every k; k* = {k['k_star']:.3f}"
    )
    for kk, c, tau in k["rows"]:
        print(f"    k = {kk:6.2f}: c = {c:9.4f}, tau = {tau:.4f}")
    x = res["reduced"]
    print(
        f"  reduced form (sigma = r = 0): Theta = {x['theta']:.4f}, Delta = {x['Delta']:.4f}: "
        f"c* x = {x['c_star_x']:.10f}, full solve c = {x['c_full']:.10f}, "
        f"tau = 1/x^2 = {x['tau_reduced']:.6f} vs {x['tau_full']:.6f}"
    )


# ============================================================ C: latent task
def section_c():
    """The latent task Z ~ U[0, a] at P >> K: Gram matrix, eigenstructure, mean direction."""
    rng = np.random.default_rng(SEED)
    P, K, a = LATENT_P, LATENT_K, LATENT_A
    Z = rng.uniform(0, a, (P, K))
    G = Z.T @ Z / P
    G_theory = a**2 / 4 * np.ones((K, K)) + a**2 / 12 * np.eye(K)
    w, V = np.linalg.eigh(G)
    u1 = Z @ V[:, -1] / np.sqrt(P * w[-1])
    return {
        "max_dev_gram": np.abs(G - G_theory).max(),
        "eigenvalues": w,
        "top_eigenvector": np.abs(V[:, -1]),
        "u1_vs_constant": abs(u1 @ np.ones(P)) / np.sqrt(P),
    }


def report_c(res):
    print("=== C: latent task Z ~ U[0, a], P >> K ===")
    P, K, a = LATENT_P, LATENT_K, LATENT_A
    print(
        f"  P = {P}, K = {K}, a = {a}: max |Z^T Z / P - (a^2/4) 11^T - (a^2/12) I| = "
        f"{res['max_dev_gram']:.4f} (O(1/sqrt P) = {1 / np.sqrt(P):.4f})"
    )
    print(
        f"    eigenvalues of Z^T Z / P: {np.round(res['eigenvalues'], 4)};  "
        f"theory a^2/12 = {a**2 / 12:.4f} (x{K - 1}), a^2 (3K+1)/12 = {a**2 * (3 * K + 1) / 12:.4f}"
    )
    print(
        f"    top eigenvector: {np.round(res['top_eigenvector'], 4)}, "
        f"theory 1/sqrt(K) = {1 / np.sqrt(K):.4f}"
    )
    print(
        "    top left singular vector u_1 vs the constant vector 1/sqrt(P): "
        f"correlation = {res['u1_vs_constant']:.4f}"
    )


# ============================================================ D: functional cell types
def monotonicity_check(lam):
    """D1: d_q is increasing in lam_q for the self-consistent c, over a parameter grid."""
    grid = [
        (sx2, sy2, sig2, r, delta)
        for sx2 in (0.1, 1, 10)
        for sy2 in (0.1, 1, 100)
        for sig2 in (0.01, 1)
        for r in (0, 1)
        for delta in (0.1, 10)
    ]
    order = np.argsort(lam)
    monotone = all((np.diff(spectrum_d(lam, *p)[1][order]) > -1e-12).all() for p in grid)
    return {"n_sets": len(grid), "monotone": monotone}


def whitening_check(lam):
    """D2: at weak target the latent-specific part d_q - sigma_0^2 follows
    lam_q sX2 / ((1 + r sig2)(1 + r sig2 + r lam_q sX2)); the half-power |q| rises with gamma."""
    sig2, r, delta = 0.01, 1.0, 50.0
    rows = []
    for sx2 in (0.1, 10.0):
        _, d = spectrum_d(lam, sx2, 1e-6, sig2, r, delta)
        floor = sig2 / (1 + r * sig2)
        pred = lam * sx2 / ((1 + r * sig2) * (1 + r * sig2 + r * lam * sx2))
        rows.append(
            {
                "sx2": sx2,
                "gamma": r * lam[0] * sx2,
                "max_dev": np.abs(d - floor - pred).max(),
                "half_power_q": half_power_q(d),
            }
        )
    return rows


def condensation_check(lam):
    """D3: at fixed N, c^2 - sY2 h_max ~ sY h_max^{3/2} / N and d_0 ~ N sY sqrt(h_max) / lam_0."""
    h = h_of(lam, BASE["sx2"], BASE["sig2"], BASE["r"])
    rows = []
    for sy in SY_SWEEP:
        c, d = spectrum_d(lam, BASE["sx2"], sy**2, BASE["sig2"], BASE["r"], BASE["delta"])
        rows.append(
            {
                "sy": sy,
                "tau": sy**2 * h.max() / c**2,
                "gap": c**2 - sy**2 * h.max(),
                "gap_asymptote": sy * h.max() ** 1.5 / N,
                "d0": d[0],
                "d0_asymptote": N * sy * np.sqrt(h.max()) / lam[0],
                "share0": lam[0] * d[0] / N / (c - BASE["delta"]),
            }
        )
    return rows


def continuum_check():
    """D4: for N >> 1/nu_c the gap a = c^2 - sY2 h_max tends to the sY-independent
    a_inf = 2 pi^2 h_max^3 / |h''(0)| and the amplification is a Lorentzian in nu = q/N of
    half-width nu_c = sqrt(a / b), b = sY2 |h''(0)| / 2,
    so sY nu_c -> 2 pi h_max^{3/2} / |h''(0)|."""
    lam_big = ring_spectrum(BIG_N, ELL)
    nu, half = np.fft.fftfreq(BIG_N), BIG_N // 2  # nu[:half] are the non-negative frequencies
    h_big = h_of(lam_big, BASE["sx2"], BASE["sig2"], BASE["r"])
    h_curv = 2 * (h_big[0] - h_big[1]) * BIG_N**2  # |h''(0)| in nu units, from the grid
    alpha0 = lam_big[0] * BASE["sx2"] + BASE["sig2"]
    dh_dlam = alpha0 / (1 + BASE["r"] * alpha0)
    dh_dlam += lam_big[0] * BASE["sx2"] / (1 + BASE["r"] * alpha0) ** 2
    rows = []
    for sy in SY_CONTINUUM:
        c = solve_c(lam_big, BASE["sx2"], sy**2, BASE["sig2"], BASE["r"], BASE["delta"])
        amp = c**2 / (c**2 - sy**2 * h_big)
        gap = c**2 - sy**2 * h_big[0]
        rows.append(
            {
                "sy": sy,
                "tau": sy**2 * h_big[0] / c**2,
                "gap": gap,
                "nu_c": half_crossing(nu[:half], (amp - 1)[:half]),
                "nu_c_lorentz": np.sqrt(gap / (sy**2 * h_curv / 2)),  # sqrt(a / b) from the gap
            }
        )
    return {
        "h_curv": h_curv,
        "h_curv_closed": 4 * np.pi**2 * ELL**2 * lam_big[0] * dh_dlam,
        "a_inf": 2 * np.pi**2 * h_big[0] ** 3 / h_curv,
        "sy_nu_c_asymptote": 2 * np.pi * h_big[0] ** 1.5 / h_curv,
        "rows": rows,
    }


def section_d():
    """Functional cell types: monotonicity, whitening, condensation, continuum width."""
    lam = ring_spectrum(N, ELL)
    return {
        "monotone": monotonicity_check(lam),
        "whitening": whitening_check(lam),
        "condensation": condensation_check(lam),
        "continuum": continuum_check(),
    }


def report_d(res):
    print("=== D: functional cell types ===")
    m = res["monotone"]
    print(f"  d_q increasing in lam_q on {m['n_sets']} parameter sets: {m['monotone']}")
    for w in res["whitening"]:
        print(
            f"    sX2 = {w['sx2']:5.1f}: gamma = r lam_0 sX2 = {w['gamma']:6.1f}, "
            f"max |d_q - sigma_0^2 - pred| = {w['max_dev']:.1e}, "
            f"half-power |q| = {w['half_power_q']:.0f}"
        )
    print(f"  condensation (N = {N}, {BASE}):")
    for row in res["condensation"]:
        print(
            f"    sY = {row['sy']:6.0f}: tau = {row['tau']:.4f}, gap = {row['gap']:9.3f} "
            f"(asymptote {row['gap_asymptote']:9.3f}), d_0 = {row['d0']:10.1f} "
            f"(asymptote {row['d0_asymptote']:10.1f}), q = 0 share of c - delta = {row['share0']:.3f}"
        )
    c = res["continuum"]
    print(
        f"  continuum N = {BIG_N}: |h''(0)| = {c['h_curv']:.2f} "
        f"(closed form {c['h_curv_closed']:.2f}), "
        f"a_inf = {c['a_inf']:.3f}, predicted sY nu_c = {c['sy_nu_c_asymptote']:.4f}"
    )
    for row in c["rows"]:
        print(
            f"    sY = {row['sy']:4.0f}: tau = {row['tau']:.4f}, gap a = {row['gap']:7.3f}, "
            f"nu_c = {row['nu_c']:.5f} (N nu_c = {BIG_N * row['nu_c']:.1f}), "
            f"sY nu_c = {row['sy'] * row['nu_c']:.4f} (interpolated half-power crossing), "
            f"sY sqrt(a/b) = {row['sy'] * row['nu_c_lorentz']:.4f}"
        )


# ============================================================ E: regime numbers of the predictions
def latent_modes(P, K, a, m, k0, k1, n_out, beta):
    """Per-mode hat s_X^2, hat s_Y^2 and delta for the uniform latent task at P >> K:
    s_1^2 = P a^2 (3K+1)/12 (mean direction), s_k^2 = P a^2/12 (contrasts),
    delta = P n / (beta k1)."""
    s2 = np.array([P * a**2 * (3 * K + 1) / 12] + [P * a**2 / 12] * (K - 1))
    return k0 * s2 / m, s2 / k1, P * n_out / (beta * k1)


def prediction_spectra(ell, k0, k1, sig2, r, beta):
    """(lam_q, [d_q of mode 1, d_q of mode 2]) of one prediction's parameter set."""
    lam = ring_spectrum(PRED_N, ell)
    sx2, sy2, delta = latent_modes(PRED_P, PRED_K, PRED_A, PRED_M, k0, k1, PRED_NOUT, beta)
    return lam, [spectrum_d(lam, a, b, sig2, r, delta)[1] for a, b in zip(sx2[:2], sy2[:2])]


def mode_numbers(lam, sx2, sy2, sig2, r, delta):
    """The regime numbers of one data mode: knobs (rho, gamma, tau) and spectrum summaries."""
    c, d = spectrum_d(lam, sx2, sy2, sig2, r, delta)
    floor = sig2 / (1 + r * sig2)
    return {
        "sx2": sx2,
        "sy2": sy2,
        "rho": lam[0] * sx2 / sig2,
        "gamma": r * lam[0] * sx2,
        "tau": sy2 * h_of(lam, sx2, sig2, r).max() / c**2,
        "half_power_q": half_power_q(d),
        "share0": d[0] / d.sum(),
        "d0_over_floor": d[0] / floor,
        "lag1": autocorr_lag1(d),
    }


def prediction_numbers(ell, k0, k1, sig2, r, beta):
    """Regime numbers of modes 1 (mean direction) and 2 (contrast) for one parameter set,
    and the ratio d_q^(1) / d_q^(2) at q = 0, ..., RATIO_Q_MAX."""
    lam = ring_spectrum(PRED_N, ell)
    sx2, sy2, delta = latent_modes(PRED_P, PRED_K, PRED_A, PRED_M, k0, k1, PRED_NOUT, beta)
    modes = [mode_numbers(lam, a, b, sig2, r, delta) for a, b in zip(sx2[:2], sy2[:2])]
    d1, d2 = prediction_spectra(ell, k0, k1, sig2, r, beta)[1]
    return {"delta": delta, "modes": modes, "ratio": (d1 / d2)[: RATIO_Q_MAX + 1]}


def share0_sweep(settings):
    """q = 0 share (mode 1, mode 2) for each (ell, k0, k1, sig2, r, beta) in `settings`."""
    return [[d[0] / d.sum() for d in prediction_spectra(*s)[1]] for s in settings]


def correlation_length_check(ell):
    """P7: setting P1 at length scale `ell`. For each mode, R(0) and the largest deviation
    of R(Delta)/R(0) from the kernel K(Delta) over 1 <= Delta <= 2 ell, plus the Gaussian
    width read off at Delta = ell, sqrt(-ell^2 / (2 ln(R(ell)/R(0))))."""
    _, k0, k1, sig2, rate, beta = PREDICTIONS["P1 prior-shaped (weak target, no rate penalty)"]
    kernel = wrapped_gaussian(PRED_N, ell)
    lags = np.arange(1, int(2 * ell) + 1)
    rows = []
    for d in prediction_spectra(ell, k0, k1, sig2, rate, beta)[1]:
        R = autocorrelation(d)
        rows.append(
            {
                "R0": R[0],
                "max_dev": np.abs(R[lags] / R[0] - kernel[lags]).max(),
                "width": np.sqrt(-(ell**2) / (2 * np.log(R[int(ell)] / R[0]))),
            }
        )
    return rows


def section_e():
    """Regime numbers of the numbered predictions, the P3 beta sweep, the P7 length-scale
    sweep and the P8 k sweep."""
    ell, k0, k1, sig2, rate, beta = PREDICTIONS["P3 target condensation"]
    return {
        "predictions": {name: prediction_numbers(*p) for name, p in PREDICTIONS.items()},
        "p3_beta_sweep": share0_sweep([(ell, k0, k1, sig2, rate, b) for b in P3_BETA_SWEEP]),
        "p7_ell_sweep": {ell7: correlation_length_check(ell7) for ell7 in P7_ELL_SWEEP},
        "p8_k_sweep": share0_sweep([(ell, k, k, sig2, rate, beta) for k in P8_K_SWEEP]),
    }


def report_e(res):
    print("=== E: regime numbers for the numbered predictions ===")
    print(
        f"  ring N = {PRED_N}, K = {PRED_K}, a = {PRED_A}, m = n = {PRED_M}, P = {PRED_P}; "
        "modes: 1 = mean direction, 2 = contrast"
    )
    for name, (ell, k0, k1, sig2, rate, beta) in PREDICTIONS.items():
        pred = res["predictions"][name]
        print(
            f"  {name}: ell = {ell}, k0 = {k0}, k1 = {k1}, sig2 = {sig2}, rate r = {rate}, "
            f"beta = {beta} (delta = {pred['delta']:.1f})"
        )
        for mode, m in enumerate(pred["modes"], start=1):
            print(
                f"    mode {mode}: hat sX2 = {m['sx2']:8.2f}, hat sY2 = {m['sy2']:8.2f}, "
                f"rho = lam_0 sX2/sig2 = {m['rho']:9.1f}, "
                f"gamma = r lam_0 sX2 = {m['gamma']:8.2f}, "
                f"tau = {m['tau']:.3f}, half-power |q| = {m['half_power_q']:2.0f}, "
                f"q=0 share = {m['share0']:.3f}, d_0/sigma_0^2 = {m['d0_over_floor']:9.2f}, "
                f"R(1)/R(0) = {m['lag1']:.3f}"
            )
        ratio = ", ".join(f"{x:.2f}" for x in pred["ratio"])
        print(f"    d_q(mode 1) / d_q(mode 2) at |q| = 0..{RATIO_Q_MAX}: {ratio}")
    ell, k0, k1, sig2, rate, beta = PREDICTIONS["P3 target condensation"]
    print(f"  P3 beta sweep at sig2 = {sig2}, rate r = {rate}: q = 0 share (mode 1, mode 2)")
    for b, shares in zip(P3_BETA_SWEEP, res["p3_beta_sweep"]):
        print(f"    beta = {b:5.2f}: " + ", ".join(f"{s:.3f}" for s in shares))
    print("  P7 setting P1 at ell = " + ", ".join(f"{e:g}" for e in P7_ELL_SWEEP) + ":")
    for ell7, rows in res["p7_ell_sweep"].items():
        for mode, m in enumerate(rows, start=1):
            print(
                f"    ell = {ell7:g}, mode {mode}: R(0) = {m['R0']:7.3f}, "
                f"max |R(Delta)/R(0) - K(Delta)| (1 <= Delta <= 2 ell) = {m['max_dev']:.1e}, "
                f"Gaussian width at Delta = ell: {m['width']:.3f}"
            )
    print(
        f"  P8 common k sweep at sig2 = {sig2}, rate r = {rate}, beta = {beta}: "
        "q = 0 share (mode 1, mode 2)"
    )
    for k, shares in zip(P8_K_SWEEP, res["p8_k_sweep"]):
        print(f"    k = {k:6.2f}: " + ", ".join(f"{s:.3f}" for s in shares))


# ============================================================ F: figure
def plot_profiles(name, ax=None):
    """Normalised spectra d_q/d_0 of modes 1 and 2 of the prediction `name`, with lam_q/lam_0,
    drawn into `ax` (a new figure if None); returns the axis."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    q = np.fft.fftfreq(PRED_N) * PRED_N
    # Beyond |q| ~ 20 every curve sits on its floor and lam_q is at round-off: nothing is lost.
    keep = (q >= 0) & (q <= FIGURE_Q_MAX)
    lam, spectra = prediction_spectra(*PREDICTIONS[name])
    for mode, (d, style) in enumerate(zip(spectra, ("-", "--")), start=1):
        label = f"mode {mode} ({'mean' if mode == 1 else 'contrast'})"
        ax.plot(q[keep], d[keep] / d[0], style, label=label)
    ax.plot(q[keep], lam[keep] / lam[0], ":", color="grey", label=r"$\lambda_q/\lambda_0$")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 2)
    ax.set_xlim(0, FIGURE_Q_MAX)
    ax.set_xlabel("$|q|$")
    ax.set_title(name.split(" ", 1)[1], fontsize=9)
    return ax


def fig_profiles():
    """The three-panel figure of the note: one `plot_profiles` panel per entry of FIGURE_PANELS."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(FIGURE_PANELS), figsize=(11, 3.2), sharey=False)
    for ax, name in zip(axes, FIGURE_PANELS):
        plot_profiles(name, ax=ax)
    axes[0].set_ylabel(r"$d_q / d_0$")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def section_f():
    """Write the figure to FIGURE (the only I/O of the script)."""
    import matplotlib

    matplotlib.use("Agg")
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig_profiles().savefig(FIGURE, dpi=200)
    print(f"=== F: wrote {FIGURE}")


if __name__ == "__main__":
    report_a(section_a())
    report_b(section_b())
    report_c(section_c())
    report_d(section_d())
    report_e(section_e())
    section_f()
