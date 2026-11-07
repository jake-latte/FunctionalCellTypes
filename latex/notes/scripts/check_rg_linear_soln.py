"""Check the corrected eq:linear-soln of Section 2 against an ansatz-free fixed point.

Ground truth: the linear one-hidden-layer network on a ring of N sites. Iterate
    C -> (1/N) sum_ij K_ij Cov[phi_i, phi_j],
with Cov the inverse of the full NP x NP precision
    (A (x) K + sigma^2 I)^{-1} - (B (x) K - r I),   A = (k0/m) X X^T,
    B = (1/k1) Ct^{-1} Y Y^T Ct^{-1},                  Ct = C + delta I.
No shared-basis ansatz and no scalar reduction is used. The eigenvalues of Ct are
compared with the roots of the corrected scalar equation, one per data mode,
    1 - delta/c = (c/N) sum_q h_q / (c^2 - sY^2 h_q),
    h_q = lam_q (lam_q sX^2 + sigma^2) / (1 + r (lam_q sX^2 + sigma^2)),
and with the roots of the equation as originally written in Section 2.

Run from the git root:  uv run python latex/notes/scripts/check_rg_linear_soln.py
"""
import numpy as np

# --- parameters quoted in latex/notes/rg_*.tex ------------------------------
SEED = 0
N, P, M_IN, N_OUT, N_LATENT = 12, 4, 6, 5, 3   # sites, patterns, m, n, latents
KERNEL_WIDTH = np.sqrt(2.0)                    # K_ij = exp(-r_ij^2 / (2 w^2)), ring distance r
Z_MAX = 1.0                                    # Z_{mu,k} ~ U[0, Z_MAX]
WY_SCALE = 0.6                                 # W_Y = 0.6 * (orthonormal columns)
K0, K1, SIG2, R, DELTA = 0.7, 1.3, 0.2, 0.3, 0.5
DAMPING, N_ITER = 0.5, 4000
C_START = 5.0                                  # iterate from C = C_START * I (large C, small B)


def ring_kernel(n_sites, width):
    """Translation-invariant Gaussian kernel on a periodic ring."""
    r = np.minimum(np.arange(n_sites), n_sites - np.arange(n_sites))
    row = np.exp(-r**2 / (2 * width**2))
    return np.array([np.roll(row, i) for i in range(n_sites)])


def orthonormal_columns(rng, rows, cols):
    return np.linalg.qr(rng.normal(size=(rows, cols)))[0]


def make_data(rng):
    """X = Z W_X^T, Y = Z W_Y^T, so X X^T and Y Y^T commute (both are functions of Z Z^T)."""
    Z = rng.uniform(0, Z_MAX, (P, N_LATENT))
    W_X = orthonormal_columns(rng, M_IN, N_LATENT)
    W_Y = WY_SCALE * orthonormal_columns(rng, N_OUT, N_LATENT)
    return Z @ W_X.T, Z @ W_Y.T


def precision(C, K, A, YY):
    Ci = np.linalg.inv(C + DELTA * np.eye(P))
    B = Ci @ YY @ Ci / K1
    I = np.eye(N * P)
    prior = np.linalg.inv(np.kron(A, K) + SIG2 * I)  # index ordering (mu, i)
    return prior - (np.kron(B, K) - R * I)


def kernel_map(C, K, A, YY):
    Cov = np.linalg.inv(precision(C, K, A, YY)).reshape(P, N, P, N)
    return np.einsum('ij,uivj->uv', K, Cov) / N


def iterate_fixed_point(C0, K, A, YY):
    C = C0
    for _ in range(N_ITER):
        C = DAMPING * C + (1 - DAMPING) * kernel_map(C, K, A, YY)
    return C


def bisect(f, lo, hi, n=200):
    """Root of an increasing f on (lo, hi); fails loudly if not bracketed."""
    if not (f(lo) < 0 < f(hi)):
        raise ValueError(f"root not bracketed: f(lo)={f(lo)}, f(hi)={f(hi)}")
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def h_corrected(lam, sx):
    return lam * (lam * sx + SIG2) / (1 + R * (lam * sx + SIG2))


def h_original(lam, sx):
    return (lam * sx + SIG2) / (1 / lam + R * (lam * sx + SIG2 / lam))


def root_corrected(lam, sx, sy):
    h = h_corrected(lam, sx)
    f = lambda c: 1 - DELTA / c - np.mean(c * h / (c**2 - sy * h))
    pole = np.sqrt(sy * h.max())
    return bisect(f, max(pole, 1e-12) * (1 + 1e-12) + 1e-12, 1e3)


def root_original(lam, sx, sy):
    h = h_original(lam, sx)
    f = lambda c: 1 - DELTA / c - np.sum(h / (c - sy * h))
    return bisect(f, sy * h.max() * (1 + 1e-12) + 1e-12, 1e4)


def data_modes(X, Y):
    """Per-mode scalars hat s_X^2 = k0 s_X^2/m, hat s_Y^2 = s_Y^2/k1 in the common basis U."""
    w, U = np.linalg.eigh(X @ X.T)
    sY2 = np.clip(np.diag(U.T @ Y @ Y.T @ U), 0, None)  # clip round-off negatives
    return U, np.clip(K0 * w / M_IN, 0, None), sY2 / K1


def main():
    rng = np.random.default_rng(SEED)
    K = ring_kernel(N, KERNEL_WIDTH)
    lam = np.linalg.eigvalsh(K)
    X, Y = make_data(rng)
    A, YY = K0 / M_IN * X @ X.T, Y @ Y.T
    U, sX2hat, sY2hat = data_modes(X, Y)

    C = iterate_fixed_point(C_START * np.eye(P), K, A, YY)
    Ct = C + DELTA * np.eye(P)
    rot = U.T @ Ct @ U
    print(f"commutator max|[XX^T, YY^T]|          : {np.abs(X @ X.T @ YY - YY @ X @ X.T).max():.2e}")
    print(f"min eig of precision at fixed point    : {np.linalg.eigvalsh(precision(C, K, A, YY)).min():.4f}")
    print(f"max off-diagonal |U^T Ct U|            : {np.abs(rot - np.diag(np.diag(rot))).max():.2e}")
    print("full NP x NP fixed point, eig(C+dI)    :", np.round(np.sort(np.linalg.eigvalsh(Ct)), 10))
    corrected = [root_corrected(lam, sx, sy) for sx, sy in zip(sX2hat, sY2hat)]
    print("corrected eq:linear-soln roots         :", np.round(np.sort(corrected), 10))
    original = [root_original(lam, sx, sy) for sx, sy in zip(sX2hat, sY2hat)]
    print("original eq:linear-soln roots          :", np.round(np.sort(original), 4))
    print("ratio original / corrected             :", np.round(np.sort(original) / np.sort(corrected), 2))

    # Naive start from the prior: C computed with B = 0 (i.e. no target tilt).
    Cov0 = np.linalg.inv(np.linalg.inv(np.kron(A, K) + SIG2 * np.eye(N * P)) + R * np.eye(N * P))
    C_prior = np.einsum('ij,uivj->uv', K, Cov0.reshape(P, N, P, N)) / N
    C_naive = iterate_fixed_point(C_prior, K, A, YY)
    print("naive start (prior) -> eig(C+dI)       :", np.round(np.sort(np.linalg.eigvalsh(C_naive + DELTA * np.eye(P))), 4))
    print(f"naive start -> min eig of precision    : {np.linalg.eigvalsh(precision(C_naive, K, A, YY)).min():.4f}")


if __name__ == "__main__":
    main()
