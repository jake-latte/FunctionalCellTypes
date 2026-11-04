"""Numerical checks quoted in latex/notes/linear_solution_limits.tex.

Ground truth is the ansatz-free NP x NP fixed point of the linear one-hidden-layer
network on a ring (section A); every other section checks a closed form or an
asymptotic statement of the note against the per-mode scalar equation
    1 - delta/c = (c/N) sum_q h_q / (c^2 - sY2 h_q),
    h_q = lam_q (lam_q sX2 + sig2) / (1 + g2 (lam_q sX2 + sig2)),
solved on the branch c > sqrt(sY2 max_q h_q), and d_q = c^2 h_q / (lam_q (c^2 - sY2 h_q)).

Sections (each prints the numbers the note quotes):
  A  re-derivation: ansatz-free fixed point vs the scalar equation (exact and approximate ansatz)
  B  limiting cases: sigma -> 0, g -> 0 (flat spectrum closed form), k0 = k1 sweep, Gaussian ring
  C  latent task: Z^T Z / P, singular values, the mean direction, the ansatz condition
  D  functional cell types: monotonicity, rate-penalty whitening, target condensation, continuum width
  E  predictions: the regime numbers (tau, gamma, rho) of the parameter sets listed in the note
  F  figure latex/notes/figures/linear_solution_limits_profiles.png (per-mode spectra d_q)

Run from the git root:  uv run python latex/notes/scripts/check_linear_soln_limits.py
"""
from pathlib import Path

import numpy as np

SEED = 0
# PNG rather than PDF: the repository ignores *.pdf, and the note must build from a fresh clone.
FIGURE = Path(__file__).resolve().parents[1] / "figures" / "linear_solution_limits_profiles.png"

# --- section A: ansatz-free fixed point (small, so the NP x NP inverse is cheap) --------------
A_N, A_P, A_M, A_NOUT, A_K = 12, 5, 6, 6, 2          # sites, patterns, m, n, latents
A_ELL = np.sqrt(2.0)                                  # K_ij = exp(-r^2 / (2 ell^2)), ring distance r
A_K0, A_K1, A_SIG2, A_G2, A_DELTA = 0.7, 1.3, 0.2, 0.3, 0.5
A_C_START, A_NEWTON_STEPS, A_TOL = 30.0, 100, 1e-12  # Newton from C = 30 I, well above the pole
A_WY_SKEW = np.diag([1.0, 0.7])                       # non-orthonormal W_Y for the approximate case

# --- sections B-D: a larger ring where the continuum formulae are accurate --------------------
N, ELL = 64, 4.0
BIG_N = 4096                                          # for the continuum (large-N) check
SY_CONTINUUM = [4.0, 8.0, 16.0, 32.0]                 # moderate targets: Lorentzian width >> 1/N
M_IN, N_OUT, KAPPA = 6, 6, 1.0                        # m, n, kappa for the k0 = k1 sweep
K_SWEEP = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]
SY_SWEEP = [1.0, 10.0, 100.0, 1000.0]                 # target strength sY for the condensation check
BASE = dict(sx2=1.0, sig2=0.1, g2=0.2, delta=0.5)     # a generic finite-everything mode
LATENT_P, LATENT_K, LATENT_A = 20000, 3, 1.0          # Z ~ U[0, a]^(P x K) for the law of large numbers

# --- section E: parameter sets of the numbered predictions (ring, K = 3 latents) --------------
PRED_N, PRED_K, PRED_A, PRED_M, PRED_NOUT = 64, 3, 1.0, 8, 8
PRED_P = 256
PREDICTIONS = {
    # name: (ell, k0, k1, sig2, g2, beta)
    "P1 prior-shaped (weak target, no rate penalty)": (4.0, 1.0, 1.0, 0.01, 0.0, 0.25),
    "P2 rate-penalty whitening":                      (4.0, 1.0, 1.0, 0.01, 0.1, 0.25),
    "P3 target condensation":                         (4.0, 1.0, 1.0, 0.01, 0.0, 10.0),
    "P4 white null modes":                            (4.0, 1.0, 1.0, 0.25, 1.0, 0.25),
    "P5 noise-dominated":                             (4.0, 0.001, 1.0, 1.0, 0.0, 0.25),
}


# ============================================================ building blocks
def ring_distance(n_sites):
    r = np.arange(n_sites)
    return np.minimum(r, n_sites - r)


def ring_kernel(n_sites, ell):
    """Translation-invariant Gaussian kernel on a periodic ring, K_ii = 1 (no image sum:
    for ell << N the images are negligible, which the eigenvalue check below confirms)."""
    row = np.exp(-ring_distance(n_sites) ** 2 / (2 * ell**2))
    return np.array([np.roll(row, i) for i in range(n_sites)])


def ring_spectrum(n_sites, ell):
    """Eigenvalues lam_q of ring_kernel, q = 0, 1, ..., N-1 in DFT order (lam_q = lam_{-q})."""
    return np.fft.fft(ring_kernel(n_sites, ell)[0]).real


def h_of(lam, sx2, sig2, g2):
    alpha = lam * sx2 + sig2
    return lam * alpha / (1 + g2 * alpha)


def bisect(f, lo, hi, n=200):
    """Root of an increasing f on (lo, hi); fails loudly if not bracketed."""
    if not (f(lo) < 0 < f(hi)):
        raise ValueError(f"root not bracketed: f(lo)={f(lo)}, f(hi)={f(hi)}")
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def solve_c(lam, sx2, sy2, sig2, g2, delta):
    """The physical root of the per-mode equation (branch above the pole)."""
    h = h_of(lam, sx2, sig2, g2)
    f = lambda c: 1 - delta / c - np.mean(c * h / (c**2 - sy2 * h))
    pole = np.sqrt(sy2 * h.max())
    return bisect(f, max(pole * (1 + 1e-13), 1e-12), 1e9)  # lower bracket stays positive when the pole is at 0


def spectrum_d(lam, sx2, sy2, sig2, g2, delta):
    """(c, d_q): the self-consistent c and the per-frequency variance of one data mode."""
    c = solve_c(lam, sx2, sy2, sig2, g2, delta)
    h = h_of(lam, sx2, sig2, g2)
    return c, c**2 * h / lam / (c**2 - sy2 * h)


def tau_of(lam, sx2, sy2, sig2, g2, delta):
    """Target strength tau = sY2 max_q h_q / c^2 in (0, 1)."""
    c = solve_c(lam, sx2, sy2, sig2, g2, delta)
    return sy2 * h_of(lam, sx2, sig2, g2).max() / c**2


# ============================================================ A: ansatz-free fixed point
def orthonormal_columns(rng, rows, cols):
    return np.linalg.qr(rng.normal(size=(rows, cols)))[0]


def latent_task(rng, n_patterns, m, n_out, n_latent, a=1.0, wy_skew=None):
    Z = rng.uniform(0, a, (n_patterns, n_latent))
    W_X = orthonormal_columns(rng, m, n_latent)
    W_Y = orthonormal_columns(rng, n_out, n_latent)
    if wy_skew is not None:
        W_Y = W_Y @ wy_skew
    return Z @ W_X.T, Z @ W_Y.T


def full_precision(C, K, A, YY, k1, sig2, g2, delta):
    """The NP x NP precision (A (x) K + sig2 I)^{-1} - (B (x) K - g2 I), index order (mu, i)."""
    n_patterns, n_sites = A.shape[0], K.shape[0]
    Ci = np.linalg.inv(C + delta * np.eye(n_patterns))
    B = Ci @ YY @ Ci / k1
    I = np.eye(n_patterns * n_sites)
    return np.linalg.inv(np.kron(A, K) + sig2 * I) - (np.kron(B, K) - g2 * I)


def kernel_map(C, K, A, YY, k1, sig2, g2, delta):
    """C -> (1/N) sum_ij K_ij Cov[h_i, h_j], the hidden-layer kernel of eq:kernel-def."""
    n_patterns, n_sites = A.shape[0], K.shape[0]
    Cov = np.linalg.inv(full_precision(C, K, A, YY, k1, sig2, g2, delta))
    return np.einsum("ij,uivj->uv", K, Cov.reshape(n_patterns, n_sites, n_patterns, n_sites)) / n_sites


def _sym_from_upper(v, n):
    C = np.zeros((n, n))
    C[np.triu_indices(n)] = v
    return C + np.triu(C, 1).T


def ansatz_free_fixed_point(K, A, YY, k1, sig2, g2, delta):
    """Newton's method (finite-difference Jacobian) on the upper triangle of C for the
    residual C - kernel_map(C). Steps are halved until the NP x NP precision stays positive
    definite: plain damped iteration leaves the physical branch when the target is strong,
    because the scalar map c -> R(c) has slope below -3 near the pole."""
    n = A.shape[0]
    iu = np.triu_indices(n)

    def residual(v):
        C = _sym_from_upper(v, n)
        return (C - kernel_map(C, K, A, YY, k1, sig2, g2, delta))[iu]

    def is_physical(v):
        return np.linalg.eigvalsh(full_precision(_sym_from_upper(v, n), K, A, YY, k1, sig2, g2, delta)).min() > 0

    v = (A_C_START * np.eye(n))[iu]
    if not is_physical(v):
        raise ValueError("starting point is below the pole")
    for _ in range(A_NEWTON_STEPS):
        r = residual(v)
        if np.abs(r).max() < A_TOL:
            return _sym_from_upper(v, n)
        eps = 1e-6
        J = np.column_stack([(residual(v + eps * e) - r) / eps for e in np.eye(v.size)])
        step = -np.linalg.solve(J, r)
        while not is_physical(v + step):
            step /= 2
            if np.abs(step).max() < 1e-14:
                raise ValueError("line search collapsed: no physical step")
        v = v + step
    raise ValueError(f"Newton did not converge: max residual {np.abs(residual(v)).max():.1e}")


def scalar_roots(lam, X, Y, k0, k1, m, sig2, g2, delta):
    """Roots of the per-mode equation in the eigenbasis U of X X^T (one root per data mode)."""
    w, U = np.linalg.eigh(X @ X.T)
    sx2 = np.clip(k0 * w / m, 0, None)
    sy2 = np.clip(np.diag(U.T @ Y @ Y.T @ U), 0, None) / k1   # exact when [XX^T, YY^T] = 0
    return np.sort([solve_c(lam, a, b, sig2, g2, delta) for a, b in zip(sx2, sy2)])


def section_a():
    print("=== A: re-derivation against the ansatz-free NP x NP fixed point ===")
    rng = np.random.default_rng(SEED)
    K = ring_kernel(A_N, A_ELL)
    lam = np.linalg.eigvalsh(K)
    for label, skew in [("orthonormal W_Y (ansatz exact)", None), ("W_Y skewed (ansatz approximate)", A_WY_SKEW)]:
        X, Y = latent_task(rng, A_P, A_M, A_NOUT, A_K, wy_skew=skew)
        A, YY = A_K0 / A_M * X @ X.T, Y @ Y.T
        C = ansatz_free_fixed_point(K, A, YY, A_K1, A_SIG2, A_G2, A_DELTA)
        Ct = C + A_DELTA * np.eye(A_P)
        U = np.linalg.eigh(X @ X.T)[1]
        rot = U.T @ Ct @ U
        exact = np.sort(np.linalg.eigvalsh(Ct))
        roots = scalar_roots(lam, X, Y, A_K0, A_K1, A_M, A_SIG2, A_G2, A_DELTA)
        print(f"  [{label}]")
        print(f"    max |[XX^T, YY^T]|               : {np.abs(X @ X.T @ YY - YY @ X @ X.T).max():.2e}")
        print(f"    min eig of precision             : {np.linalg.eigvalsh(full_precision(C, K, A, YY, A_K1, A_SIG2, A_G2, A_DELTA)).min():.4f}")
        print(f"    max off-diag |U^T Ct U|          : {np.abs(rot - np.diag(np.diag(rot))).max():.2e}")
        print(f"    eig(C + delta I), fixed point    : {np.round(exact, 8)}")
        print(f"    roots of the scalar equation     : {np.round(roots, 8)}")
        print(f"    max relative difference          : {np.abs(exact / roots - 1).max():.2e}")


# ============================================================ B: limiting cases
def section_b():
    print("=== B: limiting cases ===")
    lam = ring_spectrum(N, ELL)
    q = np.fft.fftfreq(N) * N
    # B4 (first, the others use it): Gaussian ring spectrum.
    approx = np.sqrt(2 * np.pi) * ELL * np.exp(-2 * np.pi**2 * ELL**2 * (q / N) ** 2)
    print(f"  Gaussian ring N={N}, ell={ELL}: lam_0 = {lam[0]:.6f}, sqrt(2 pi) ell = {np.sqrt(2 * np.pi) * ELL:.6f}")
    print(f"    max |lam_q - continuum formula| = {np.abs(lam - approx).max():.1e}")
    print(f"    (1/N) sum lam_q = {lam.mean():.6f} (= K_ii = 1);  (1/N) sum lam_q^2 = {(lam**2).mean():.6f}, sqrt(pi) ell = {np.sqrt(np.pi) * ELL:.6f}")
    # B1: sigma -> 0. Null data modes (s = 0) lose all variance; latent modes converge to the sigma = 0 formula.
    c0, d0 = spectrum_d(lam, sx2=0.0, sy2=0.0, sig2=1e-8, g2=BASE["g2"], delta=BASE["delta"])
    print(f"  sigma -> 0: null mode (s_X = s_Y = 0) at sig2 = 1e-8: max_q d_q = {d0.max():.1e}, c - delta = {c0 - BASE['delta']:.1e}")
    c_eps, d_eps = spectrum_d(lam, 1.0, 2.0, 1e-8, BASE["g2"], BASE["delta"])
    c_zero, d_zero = spectrum_d(lam, 1.0, 2.0, 0.0, BASE["g2"], BASE["delta"])
    print(f"    latent mode: |c(1e-8) - c(0)| = {abs(c_eps - c_zero):.1e}, max |d_q(1e-8) - d_q(0)| = {np.abs(d_eps - d_zero).max():.1e}")
    # B2: g -> 0 with a weak target: d_q = alpha_q = lam_q sX2 + sig2 exactly, c = delta + mean(h).
    sx2, sig2 = 1.0, 0.1
    c, d = spectrum_d(lam, sx2, 1e-8, sig2, 0.0, BASE["delta"])
    print(f"  g -> 0, weak target (sY2 = 1e-8): max |d_q - (lam_q sX2 + sig2)| = {np.abs(d - (lam * sx2 + sig2)).max():.1e}, "
          f"|c - delta - mean h| = {abs(c - BASE['delta'] - h_of(lam, sx2, sig2, 0.0).mean()):.1e}")
    # B2': flat spectrum (K = lam I), delta -> 0: the quadratic closed form of the earlier latent note.
    flat = np.full(N, 1.3)
    h = h_of(flat, 0.7, 0.2, 0.3)[0]
    c = solve_c(flat, 0.7, 2.0, 0.2, 0.3, 1e-12)
    print(f"  flat spectrum, delta -> 0: c = {c:.10f}, closed form (h + sqrt(h^2 + 4 sY2 h))/2 = {(h + np.sqrt(h**2 + 4 * 2.0 * h)) / 2:.10f}")
    # B3: k0 = k1 = k, sigma = g = 0: tau(k) is non-monotone, maximal near k* = sqrt(m / (kappa sX2 mean lam^2)).
    sX2, sY2 = 1.0, 1.0
    k_star = np.sqrt(M_IN / (KAPPA * sX2 * (lam**2).mean()))
    print(f"  k0 = k1 = k sweep (sigma = g = 0, m = {M_IN}, kappa = {KAPPA}); pole c* = lam_0 sX sY / sqrt(m) = {lam[0] * np.sqrt(sX2 * sY2 / M_IN):.4f} for every k; k* = {k_star:.3f}")
    for k in K_SWEEP:
        c = solve_c(lam, k * sX2 / M_IN, sY2 / k, 0.0, 0.0, 1 / (k * KAPPA))
        tau = (sY2 / k) * h_of(lam, k * sX2 / M_IN, 0.0, 0.0).max() / c**2
        print(f"    k = {k:6.2f}: c = {c:9.4f}, tau = {tau:.4f}")


# ============================================================ C: latent task
def section_c():
    print("=== C: latent task Z ~ U[0, a], P >> K ===")
    rng = np.random.default_rng(SEED)
    P, K, a = LATENT_P, LATENT_K, LATENT_A
    Z = rng.uniform(0, a, (P, K))
    G = Z.T @ Z / P
    G_theory = a**2 / 4 * np.ones((K, K)) + a**2 / 12 * np.eye(K)
    w, V = np.linalg.eigh(G)
    print(f"  P = {P}, K = {K}, a = {a}: max |Z^T Z / P - (a^2/4) 11^T - (a^2/12) I| = {np.abs(G - G_theory).max():.4f} (O(1/sqrt P) = {1 / np.sqrt(P):.4f})")
    print(f"    eigenvalues of Z^T Z / P: {np.round(w, 4)};  theory a^2/12 = {a**2 / 12:.4f} (x{K - 1}), a^2 (3K+1)/12 = {a**2 * (3 * K + 1) / 12:.4f}")
    print(f"    top eigenvector: {np.round(np.abs(V[:, -1]), 4)}, theory 1/sqrt(K) = {1 / np.sqrt(K):.4f}")
    u1 = Z @ V[:, -1] / np.sqrt(P * w[-1])
    print(f"    top left singular vector u_1 vs the constant vector 1/sqrt(P): correlation = {abs(u1 @ np.ones(P)) / np.sqrt(P):.4f}")


# ============================================================ D: functional cell types
def section_d():
    print("=== D: functional cell types ===")
    lam = ring_spectrum(N, ELL)
    # D1: d_q is increasing in lam_q for the self-consistent c, over a parameter grid.
    grid = [(sx2, sy2, sig2, g2, delta) for sx2 in (0.1, 1, 10) for sy2 in (0.1, 1, 100)
            for sig2 in (0.01, 1) for g2 in (0, 1) for delta in (0.1, 10)]
    order = np.argsort(lam)
    monotone = all((np.diff(spectrum_d(lam, *p)[1][order]) > -1e-12).all() for p in grid)
    print(f"  d_q increasing in lam_q on {len(grid)} parameter sets: {monotone}")
    # D2: rate-penalty whitening: the latent-specific part d_q - sigma_0^2 follows
    # lam_q / (1 + g2 sig2 + g2 lam_q sX2) at weak target; crossover where g2 lam_q sX2 = 1 + g2 sig2.
    sig2, g2, delta = 0.01, 1.0, 50.0
    for sx2 in (0.1, 10.0):
        c, d = spectrum_d(lam, sx2, 1e-6, sig2, g2, delta)
        floor = sig2 / (1 + g2 * sig2)
        pred = lam * sx2 / ((1 + g2 * sig2) * (1 + g2 * sig2 + g2 * lam * sx2))
        half = np.sort(np.abs(np.fft.fftfreq(N) * N)[(d - floor) >= 0.5 * (d[0] - floor)])[-1]
        print(f"    sX2 = {sx2:5.1f}: gamma = g2 lam_0 sX2 = {g2 * lam[0] * sx2:6.1f}, max |d_q - sigma_0^2 - pred| = {np.abs(d - floor - pred).max():.1e}, half-power |q| = {half:.0f}")
    # D3: target condensation at fixed N: c^2 - sY2 h_max ~ sY h_max^{3/2} / N and d_0 ~ N sY sqrt(h_max) / lam_0.
    h = h_of(lam, BASE["sx2"], BASE["sig2"], BASE["g2"])
    print(f"  condensation (N = {N}, {BASE}):")
    for sy in SY_SWEEP:
        c, d = spectrum_d(lam, BASE["sx2"], sy**2, BASE["sig2"], BASE["g2"], BASE["delta"])
        gap, gap_pred = c**2 - sy**2 * h.max(), sy * h.max() ** 1.5 / N
        d0_pred = N * sy * np.sqrt(h.max()) / lam[0]
        frac0 = lam[0] * d[0] / N / (c - BASE["delta"])
        print(f"    sY = {sy:6.0f}: tau = {sy**2 * h.max() / c**2:.4f}, gap = {gap:9.3f} (asymptote {gap_pred:9.3f}), "
              f"d_0 = {d[0]:10.1f} (asymptote {d0_pred:10.1f}), q = 0 share of c - delta = {frac0:.3f}")
    # D4: continuum (N >> 1/nu_c): the gap a = c^2 - sY2 h_max tends to the sY-independent
    # a_inf = 2 pi^2 h_max^3 / |h''(0)| and the amplification is a Lorentzian in nu = q/N of
    # half-width nu_c = sqrt(a / b), b = sY2 |h''(0)| / 2, so sY nu_c -> 2 pi h_max^{3/2} / |h''(0)|.
    lam_big = ring_spectrum(BIG_N, ELL)
    nu = np.fft.fftfreq(BIG_N)
    h_big = h_of(lam_big, BASE["sx2"], BASE["sig2"], BASE["g2"])
    h_curv = 2 * (h_big[0] - h_big[1]) * BIG_N**2            # |h''(0)| in nu units
    a_inf = 2 * np.pi**2 * h_big[0] ** 3 / h_curv
    print(f"  continuum N = {BIG_N}: a_inf = {a_inf:.3f}, predicted sY nu_c = {2 * np.pi * h_big[0] ** 1.5 / h_curv:.4f}")
    for sy in SY_CONTINUUM:
        c = solve_c(lam_big, BASE["sx2"], sy**2, BASE["sig2"], BASE["g2"], BASE["delta"])
        amp = c**2 / (c**2 - sy**2 * h_big)
        nu_c = np.sort(np.abs(nu)[(amp - 1) >= 0.5 * (amp[0] - 1)])[-1]
        print(f"    sY = {sy:4.0f}: tau = {sy**2 * h_big[0] / c**2:.4f}, gap a = {c**2 - sy**2 * h_big[0]:7.3f}, "
              f"nu_c = {nu_c:.5f} (N nu_c = {BIG_N * nu_c:.0f}), sY nu_c = {sy * nu_c:.4f}")


# ============================================================ E: regime numbers of the predictions
def latent_modes(P, K, a, m, k0, k1, n_out, beta):
    """Per-mode hat s_X^2, hat s_Y^2 and delta for the uniform latent task at P >> K:
    s_1^2 = P a^2 (3K+1)/12 (mean direction), s_k^2 = P a^2/12 (contrasts), delta = P n / (beta k1)."""
    s2 = np.array([P * a**2 * (3 * K + 1) / 12] + [P * a**2 / 12] * (K - 1))
    return k0 * s2 / m, s2 / k1, P * n_out / (beta * k1)


def section_e():
    print("=== E: regime numbers for the numbered predictions ===")
    print(f"  ring N = {PRED_N}, K = {PRED_K}, a = {PRED_A}, m = n = {PRED_M}, P = {PRED_P}; modes: 1 = mean direction, 2 = contrast")
    for name, (ell, k0, k1, sig2, g2, beta) in PREDICTIONS.items():
        lam = ring_spectrum(PRED_N, ell)
        sx2, sy2, delta = latent_modes(PRED_P, PRED_K, PRED_A, PRED_M, k0, k1, PRED_NOUT, beta)
        print(f"  {name}: ell = {ell}, k0 = {k0}, k1 = {k1}, sig2 = {sig2}, g2 = {g2}, beta = {beta} (delta = {delta:.1f})")
        for mode, (a, b) in enumerate(zip(sx2[:2], sy2[:2]), start=1):
            c, d = spectrum_d(lam, a, b, sig2, g2, delta)
            tau = b * h_of(lam, a, sig2, g2).max() / c**2
            floor = sig2 / (1 + g2 * sig2)
            q = np.abs(np.fft.fftfreq(PRED_N) * PRED_N)
            half = np.sort(q[(d - floor) >= 0.5 * (d[0] - floor)])[-1]
            share0 = d[0] / d.sum()
            print(f"    mode {mode}: hat sX2 = {a:8.2f}, hat sY2 = {b:8.2f}, rho = lam_0 sX2/sig2 = {lam[0] * a / sig2:9.1f}, "
                  f"gamma = g2 lam_0 sX2 = {g2 * lam[0] * a:8.2f}, tau = {tau:.3f}, half-power |q| = {half:2.0f}, "
                  f"q=0 share = {share0:.3f}, d_0/sigma_0^2 = {d[0] / floor:9.2f}")


# ============================================================ F: figure
def section_f():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q = np.fft.fftfreq(PRED_N) * PRED_N
    keep = q >= 0
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=False)
    panels = ["P1 prior-shaped (weak target, no rate penalty)", "P2 rate-penalty whitening", "P3 target condensation"]
    for ax, name in zip(axes, panels):
        ell, k0, k1, sig2, g2, beta = PREDICTIONS[name]
        lam = ring_spectrum(PRED_N, ell)
        sx2, sy2, delta = latent_modes(PRED_P, PRED_K, PRED_A, PRED_M, k0, k1, PRED_NOUT, beta)
        for mode, (a, b, style) in enumerate(zip(sx2[:2], sy2[:2], ("-", "--")), start=1):
            _, d = spectrum_d(lam, a, b, sig2, g2, delta)
            ax.plot(q[keep], d[keep] / d[0], style, label=f"mode {mode} ({'mean' if mode == 1 else 'contrast'})")
        ax.plot(q[keep], lam[keep] / lam[0], ":", color="grey", label=r"$\lambda_q/\lambda_0$")
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 2)  # lam_q reaches round-off (1e-14) beyond |q| ~ 20; nothing is lost
        ax.set_xlabel("$|q|$")
        ax.set_title(name.split(" ", 1)[1], fontsize=9)
    axes[0].set_ylabel(r"$d_q / d_0$")
    axes[0].legend(fontsize=8)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=200)
    print(f"=== F: wrote {FIGURE}")


if __name__ == "__main__":
    section_a()
    section_b()
    section_c()
    section_d()
    section_e()
    section_f()
