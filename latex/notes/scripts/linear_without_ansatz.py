"""Solve the linear one-hidden-layer fixed point without the shared-basis ansatz.

Per spatial frequency q the data-interaction matrix is
    Sigma_q = (M_q - lam_q B)^{-1},   M_q = (lam_q A + sigma^2 I)^{-1} + g^2 I,
    A = (k0/m) X X^T,                 B = (1/k1) Ct^{-1} Y Y^T Ct^{-1},
and the only coupling between frequencies is the hidden-layer kernel
    Ct = delta I + (1/N) sum_q lam_q Sigma_q,    delta = 1/(k1 kappa).
The script iterates this P x P map (damped, from a large Ct so that the tilt B is
small and every precision M_q - lam_q B stays positive definite) and then
  (i)   checks the per-q solver against the full NP x NP kernel map,
  (ii)  checks that the commuting case reproduces the scalar roots of eq:linear-soln,
  (iii) reports how much the eigenbasis of Sigma_q rotates with q when XX^T and YY^T
        do not commute,
  (iv)  checks the first-order (in the non-commuting part of YY^T) formulae for the
        correction to Ct and for the per-q rotation against the solver,
  (v)   checks the block structure of a rank-deficient latent task.

Run from the git root:  uv run python latex/notes/scripts/linear_without_ansatz.py
"""

from dataclasses import dataclass

import numpy as np

SEED = 0
N_SITES, N_PATTERNS, M_IN = 32, 6, 6
KERNEL_WIDTH = np.sqrt(2.0)  # K_ij = exp(-r_ij^2 / (2 w^2))
SX2 = np.array([2.0, 1.4, 1.0, 0.7, 0.4, 0.2])  # singular values^2 of X
SY2 = np.array([1.0, 0.8, 0.6, 0.4, 0.25, 0.15])  # singular values^2 of Y (eps = 0)
ROTATION_ANGLE = np.pi / 6  # output basis vs input basis at eps = 1
PARAMS_DEFAULT = {"k0": 0.7, "k1": 1.3, "sig2": 0.2, "g2": 0.3, "delta": 0.5}
EPS_CHECK = (0.02, 0.01)  # perturbative check: residual must halve


@dataclass(frozen=True)
class Params:
    k0: float
    k1: float
    sig2: float
    g2: float
    delta: float


class NotPositiveDefiniteError(ValueError):
    """A per-frequency precision M_q - lam_q B is not positive definite: no Gaussian exists."""


# --- spatial kernel -----------------------------------------------------------------
def ring_kernel(n_sites, width):
    """Translation-invariant Gaussian kernel on a periodic ring."""
    r = np.minimum(np.arange(n_sites), n_sites - np.arange(n_sites))
    row = np.exp(-(r**2) / (2 * width**2))
    return np.array([np.roll(row, i) for i in range(n_sites)])


def ring_eigenvalues(n_sites, width):
    """lam_q, q = 0..N-1, of the ring kernel (real because the kernel row is symmetric)."""
    lam = np.fft.fft(ring_kernel(n_sites, width)[0])
    if np.abs(lam.imag).max() > 1e-12:
        raise ValueError("kernel row is not symmetric: eigenvalues are complex")
    return lam.real


# --- the per-frequency fixed-point map -----------------------------------------------
def prior_precision(lam, A, p):
    """M_q = (lam_q A + sigma^2 I)^{-1} + g^2 I, shape (N, P, P); depends on X only."""
    I = np.eye(A.shape[0])
    return np.linalg.inv(lam[:, None, None] * A + p.sig2 * I) + p.g2 * I


def tilt(Ct, YY, p):
    """B = (1/k1) Ct^{-1} Y Y^T Ct^{-1}: the target tilt, the only place Y enters."""
    Ci = np.linalg.inv(Ct)
    return Ci @ YY @ Ci / p.k1


def per_q_cov(M, lam, B):
    """Sigma_q = (M_q - lam_q B)^{-1}; fails loudly if any precision is not positive definite."""
    precision = M - lam[:, None, None] * B
    min_eig = np.linalg.eigvalsh(precision).min()
    if min_eig <= 0:
        raise NotPositiveDefiniteError(f"min eigenvalue of a per-q precision is {min_eig:.3g}")
    return np.linalg.inv(precision)


def _kernel_map(Ct, lam, M, YY, p):
    Sigma = per_q_cov(M, lam, tilt(Ct, YY, p))
    return p.delta * np.eye(Ct.shape[0]) + np.einsum("q,qij->ij", lam, Sigma) / len(lam)


def kernel_map(Ct, lam, A, YY, p):
    """Ct -> delta I + (1/N) sum_q lam_q Sigma_q(Ct)."""
    return _kernel_map(Ct, lam, prior_precision(lam, A, p), YY, p)


def solve_fixed_point(
    lam,
    A,
    YY,
    p,
    Ct_start=None,
    c_start=5.0,
    damping=0.5,
    tol=1e-12,
    max_iter=50_000,
    return_n_iter=False,
):
    """Damped iteration Ct <- a Ct + (1 - a) F(Ct) from a large Ct (small tilt).

    The kernel map F is decreasing in Ct (a larger Ct weakens the tilt B), so its Jacobian
    at the fixed point has negative eigenvalues and the undamped iteration oscillates when
    a slope exceeds 1 in magnitude. The damping needed is not known in advance, so whenever
    a step grows the damping is tightened (1 - a halved). Raises if not converged.
    """
    M = prior_precision(lam, A, p)
    Ct = c_start * np.eye(A.shape[0]) if Ct_start is None else Ct_start
    last_step = np.inf
    for n_iter in range(1, max_iter + 1):
        Ct_new = damping * Ct + (1 - damping) * _kernel_map(Ct, lam, M, YY, p)
        step = np.abs(Ct_new - Ct).max()
        Ct = Ct_new
        if step < tol:
            return (Ct, n_iter) if return_n_iter else Ct
        if step > last_step:
            damping = 1 - (1 - damping) / 2
        last_step = step
    raise RuntimeError(
        f"fixed-point iteration did not converge: last step {step:.3g}, damping {damping:.3g}"
    )


def per_q_cov_at(lam, A, YY, p):
    """Sigma_q, shape (N, P, P), at the fixed point."""
    Ct = solve_fixed_point(lam, A, YY, p)
    return per_q_cov(prior_precision(lam, A, p), lam, tilt(Ct, YY, p))


def full_kernel_map(Ct, K, A, YY, p):
    """The kernel map evaluated on the full NP x NP Gaussian (no Fourier decomposition).

    Index ordering (mu, i). Ground truth for the per-q decomposition.
    """
    n_sites, P = K.shape[0], A.shape[0]
    I = np.eye(n_sites * P)
    precision = np.linalg.inv(np.kron(A, K) + p.sig2 * I) - (np.kron(tilt(Ct, YY, p), K) - p.g2 * I)
    Cov = np.linalg.inv(precision).reshape(P, n_sites, P, n_sites)
    return p.delta * np.eye(P) + np.einsum("ij,uivj->uv", K, Cov) / n_sites


# --- the commuting case: the scalar reduction ----------------------------------------
def h_over_lam(lam, sx2hat, p):
    """h_q / lam_q = (lam_q sX^2 + sigma^2) / (1 + g^2 (lam_q sX^2 + sigma^2)); finite at lam_q = 0."""
    a = lam * sx2hat + p.sig2
    return a / (1 + p.g2 * a)


def bisect(f, lo, hi, n=200):
    """Root of an increasing f on (lo, hi); fails loudly if not bracketed."""
    if not (f(lo) < 0 < f(hi)):
        raise ValueError(f"root not bracketed: f(lo)={f(lo)}, f(hi)={f(hi)}")
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def scalar_root(lam, sx2hat, sy2hat, p):
    """Physical root c > sY sqrt(max h_q) of 1 - delta/c = (c/N) sum_q h_q / (c^2 - sY^2 h_q)."""
    h = lam * h_over_lam(lam, sx2hat, p)
    f = lambda c: 1 - p.delta / c - np.mean(c * h / (c**2 - sy2hat * h))
    pole = np.sqrt(sy2hat * h.max())
    return bisect(f, pole * (1 + 1e-12) + 1e-12, 1e3)


def scalar_roots(lam, sx2hat, sy2hat, p):
    return np.array([scalar_root(lam, sx, sy, p) for sx, sy in zip(sx2hat, sy2hat)])


def prior_scalar(lam, p):
    """c on a direction orthogonal to both Gram matrices: delta + (sigma^2 / (1 + g^2 sigma^2)) tr K / N."""
    return p.delta + p.sig2 / (1 + p.g2 * p.sig2) * lam.mean()


# --- instance: input basis U, output basis U R(angle) ------------------------------
def givens_chain(n, angle):
    """Product of Givens rotations by `angle` in the planes (0,1), (1,2), ..., (n-2, n-1)."""
    R = np.eye(n)
    c, s = np.cos(angle), np.sin(angle)
    for i in range(n - 1):
        G = np.eye(n)
        G[i, i], G[i, i + 1], G[i + 1, i], G[i + 1, i + 1] = c, -s, s, c
        R = R @ G
    return R


@dataclass(frozen=True)
class Instance:
    n_sites: int
    width: float
    m_in: int
    k0: float
    lam: np.ndarray
    U: np.ndarray
    sx2: np.ndarray
    sy2: np.ndarray
    E: np.ndarray  # YY^T = U (diag(sy2) + eps E) U^T

    @property
    def A(self):
        return self.U @ np.diag(self.sx2) @ self.U.T * self.k0 / self.m_in

    def gram_y(self, eps):
        G = np.diag(self.sy2) + eps * self.E
        if np.linalg.eigvalsh(G).min() < 0:
            raise ValueError(f"Y Y^T is not positive semi-definite at eps={eps}")
        return self.U @ G @ self.U.T

    def sx2hat(self, p):
        if p.k0 != self.k0:
            raise ValueError("Params.k0 differs from the k0 the instance was built with")
        return p.k0 * self.sx2 / self.m_in

    def sy2hat(self, p):
        return self.sy2 / p.k1


def make_instance(
    rng,
    n_sites=N_SITES,
    n_patterns=N_PATTERNS,
    width=KERNEL_WIDTH,
    angle=ROTATION_ANGLE,
    k0=PARAMS_DEFAULT["k0"],
):
    """X X^T = U S_X^2 U^T; at eps = 1, Y Y^T = U R S_Y^2 R^T U^T with R a rotation by `angle`.

    E = R S_Y^2 R^T - S_Y^2, so diag(sy2) + eps E is a convex combination of PSD matrices
    for eps in [0, 1] and the commutator [X X^T, Y Y^T] grows with eps from zero.
    """
    U = np.linalg.qr(rng.normal(size=(n_patterns, n_patterns)))[0]
    sx2, sy2 = SX2[:n_patterns], SY2[:n_patterns]
    R = givens_chain(n_patterns, angle)
    E = R @ np.diag(sy2) @ R.T - np.diag(sy2)
    return Instance(n_sites, width, M_IN, k0, ring_eigenvalues(n_sites, width), U, sx2, sy2, E)


def random_spd(rng, k):
    G = rng.normal(size=(k, k))
    return G @ G.T / k


# --- first-order theory around the commuting point, in the U basis ------------------
@dataclass(frozen=True)
class FirstOrder:
    c: np.ndarray  # zeroth-order diagonal of Ct
    d: np.ndarray  # zeroth-order diagonal of Sigma_q, shape (N, P)
    B1: np.ndarray  # first-order tilt
    C1: np.ndarray  # first-order kernel
    Sigma1: np.ndarray  # first-order Sigma_q, shape (N, P, P)
    Theta: (
        np.ndarray
    )  # first-order mixing: column p of the eigenbasis of Sigma_q is e_p + eps Theta[q, :, p]


def first_order(inst, p):
    lam, P = inst.lam, len(inst.sx2)
    sx2hat, sy2hat, Ehat = inst.sx2hat(p), inst.sy2hat(p), inst.E / p.k1
    c = scalar_roots(lam, sx2hat, sy2hat, p)
    hl = np.array([h_over_lam(lam, sx, p) for sx in sx2hat]).T  # (N, P)
    d = c**2 * hl / (c**2 - sy2hat * lam[:, None] * hl)
    T = np.einsum("q,qp,qr->pr", lam**2, d, d) / len(lam)
    cp, cr = c[:, None], c[None, :]
    B1 = Ehat * cp * cr / (cp**2 * cr**2 + T * (sy2hat[None, :] * cp + sy2hat[:, None] * cr))
    C1 = T * B1
    Sigma1 = lam[:, None, None] * d[:, :, None] * d[:, None, :] * B1
    gap = d[:, None, :] - d[:, :, None]  # [q, r, p] = d_qp - d_qr
    off = ~np.eye(P, dtype=bool)
    Theta = np.zeros_like(Sigma1)
    Theta[:, off] = (Sigma1 / np.where(off, gap, 1.0))[:, off]
    return FirstOrder(c, d, B1, C1, Sigma1, Theta)


def match_modes(V):
    """Reorder and sign-fix eigenvector columns so that column p is the one closest to e_p."""
    order = np.abs(V).argmax(axis=0)
    if len(set(order)) != V.shape[0]:
        raise ValueError("eigenvectors cannot be matched one-to-one to the reference modes")
    W = np.zeros_like(V)
    W[:, order] = V
    return W * np.sign(np.diag(W))


def mixing_from_eigenvectors(U, Sigma):
    """V_q - I for the eigenbasis V_q of U^T Sigma_q U (modes matched to the columns of U)."""
    out = np.empty_like(Sigma)
    for q, S in enumerate(Sigma):
        V = np.linalg.eigh(U.T @ S @ U)[1]
        out[q] = match_modes(V) - np.eye(S.shape[0])
    return out


# --- rotation diagnostics at finite eps ---------------------------------------------
def relative_commutator(X, Y):
    return np.linalg.norm(X @ Y - Y @ X) / (np.linalg.norm(X) * np.linalg.norm(Y))


def tracked_angles(Sigma, q_ref=0):
    """Angle (deg) at every q between each eigenvector of Sigma_q and its q_ref partner.

    Eigenvectors are matched to q_ref by largest overlap, so a level crossing does not
    swap labels.
    """
    V_ref = np.linalg.eigh(Sigma[q_ref])[1]
    angles = np.empty((len(Sigma), Sigma.shape[1]))
    for q, S in enumerate(Sigma):
        overlap = np.abs(V_ref.T @ np.linalg.eigh(S)[1])
        angles[q] = np.degrees(np.arccos(np.clip(overlap.max(axis=1), 0, 1)))
    return angles


def offdiagonal_fraction(S, V):
    """|off-diagonal part of V^T S V| / |S|: zero iff V diagonalises S."""
    R = V.T @ S @ V
    return np.linalg.norm(R - np.diag(np.diag(R))) / np.linalg.norm(S)


def report_instance(inst, p):
    print(
        f"ring N={inst.n_sites}, width={inst.width:.4g}, P={len(inst.sx2)}, m={inst.m_in}; "
        f"lam_q in [{inst.lam.min():.3g}, {inst.lam.max():.3g}]"
    )
    YY = inst.gram_y(1.0)
    V_Y = np.linalg.eigh(YY)[1]
    angles = np.degrees(np.arccos(np.clip(np.abs(inst.U.T @ V_Y).max(axis=1), 0, 1)))
    print(f"relative commutator |[A, YY^T]| at eps=1   : {relative_commutator(inst.A, YY):.3f}")
    print(
        f"angle (deg) of each eigenvector of A to the nearest eigenvector of YY^T at eps=1: {np.round(angles, 1)}"
    )


def report_commuting_case(inst, p):
    Ct = solve_fixed_point(inst.lam, inst.A, inst.gram_y(0.0), p)
    roots = scalar_roots(inst.lam, inst.sx2hat(p), inst.sy2hat(p), p)
    print("\n[commuting case, eps = 0]")
    print(
        f"  eig(Ct) - scalar roots, max abs           : "
        f"{np.abs(np.sort(np.linalg.eigvalsh(Ct)) - np.sort(roots)).max():.1e}"
    )
    print(f"  off-diagonal fraction of Ct in U          : {offdiagonal_fraction(Ct, inst.U):.1e}")


def report_non_commuting_case(inst, p):
    YY = inst.gram_y(1.0)
    Ct, n_iter = solve_fixed_point(inst.lam, inst.A, YY, p, return_n_iter=True)
    Ct_50 = solve_fixed_point(inst.lam, inst.A, YY, p, c_start=50.0)
    K = ring_kernel(inst.n_sites, inst.width)
    M = prior_precision(inst.lam, inst.A, p)
    Sigma = per_q_cov(M, inst.lam, tilt(Ct, YY, p))
    V_C = np.linalg.eigh(Ct)[1]
    angles = tracked_angles(Sigma)
    print(f"\n[non-commuting case, eps = 1]  ({n_iter} damped iterations to a step below 1e-12)")
    print(
        f"  full NP x NP map residual at fixed point  : "
        f"{np.abs(full_kernel_map(Ct, K, inst.A, YY, p) - Ct).max():.1e}"
    )
    print(f"  start 5 I vs start 50 I, max |diff|       : {np.abs(Ct - Ct_50).max():.1e}")
    print(
        f"  min eig of per-q precisions               : "
        f"{np.linalg.eigvalsh(M - inst.lam[:, None, None] * tilt(Ct, YY, p)).min():.4f}"
    )
    print(f"  eig(Ct)                                   : {np.round(np.linalg.eigvalsh(Ct), 4)}")
    print(
        f"  max_q |[Sigma_q, Sigma_0]| / (|Sigma_q| |Sigma_0|)       : "
        f"{max(relative_commutator(S, Sigma[0]) for S in Sigma):.3f}"
    )
    print(
        f"  max_q off-diagonal fraction of Sigma_q in eig(Ct) basis : "
        f"{max(offdiagonal_fraction(S, V_C) for S in Sigma):.3f}"
    )
    print("  q   lam_q    angle (deg) of each eigenvector of Sigma_q to its q=0 partner")
    for q in (0, 2, 4, 6, 8, 12, 16):
        print(f"  {q:2d}  {inst.lam[q]:7.4f}  {np.round(angles[q], 1)}")


def report_perturbative_check(inst, p):
    theory = first_order(inst, p)
    Ct0 = solve_fixed_point(inst.lam, inst.A, inst.gram_y(0.0), p)
    print("\n[first-order theory vs solver]")
    print(
        f"  max |C1| = {np.abs(theory.C1).max():.4f}, max |Theta| = {np.abs(theory.Theta).max():.4f}"
    )
    res_C, res_T = [], []
    for eps in EPS_CHECK:
        Ct = solve_fixed_point(inst.lam, inst.A, inst.gram_y(eps), p)
        Sigma = per_q_cov_at(inst.lam, inst.A, inst.gram_y(eps), p)
        C1_num = inst.U.T @ (Ct - Ct0) @ inst.U / eps
        Theta_num = mixing_from_eigenvectors(inst.U, Sigma) / eps
        res_C.append(np.abs(C1_num - theory.C1).max())
        res_T.append(np.abs(Theta_num - theory.Theta).max())
        print(
            f"  eps={eps}: max |C1_num - C1| = {res_C[-1]:.2e}, max |Theta_num - Theta| = {res_T[-1]:.2e}"
        )
    print(
        f"  residual ratio eps -> eps/2: C1 {res_C[0] / res_C[1]:.2f}, Theta {res_T[0] / res_T[1]:.2f}"
        "  (2 = O(eps))"
    )
    q_lo, q_hi = 0, inst.n_sites // 2
    pair = tuple(
        int(i)
        for i in np.unravel_index(
            np.abs(theory.Theta[q_lo] - theory.Theta[q_hi]).argmax(), theory.Theta[0].shape
        )
    )
    print(
        f"  largest predicted scale dependence: modes {pair}, Theta[q=0]={theory.Theta[q_lo][pair]:.4f}, "
        f"Theta[q=N/2]={theory.Theta[q_hi][pair]:.4f}"
    )


def report_rank_deficient(p, n_sites=N_SITES, P=N_PATTERNS, K=3, seed=SEED + 1):
    rng = np.random.default_rng(seed)
    lam = ring_eigenvalues(n_sites, KERNEL_WIDTH)
    Z = np.linalg.qr(rng.normal(size=(P, K)))[0]
    GX, GY = random_spd(rng, K), random_spd(rng, K)
    A, YY = p.k0 * Z @ GX @ Z.T, Z @ GY @ Z.T
    Ct, n_iter = solve_fixed_point(lam, A, YY, p, return_n_iter=True)
    Pi = Z @ Z.T
    Pi_perp = np.eye(P) - Pi
    c0 = prior_scalar(lam, p)
    print(
        f"\n[rank-deficient latent task, K={K} < P={P}, non-commuting G_X, G_Y]  ({n_iter} iterations)"
    )
    print(f"  relative commutator |[G_X, G_Y]|          : {relative_commutator(GX, GY):.3f}")
    print(f"  max |Pi Ct Pi_perp|                       : {np.abs(Pi @ Ct @ Pi_perp).max():.1e}")
    print(
        f"  max |Pi_perp Ct Pi_perp - c0 Pi_perp|     : {np.abs(Pi_perp @ Ct @ Pi_perp - c0 * Pi_perp).max():.1e}"
        f"   (c0 = {c0:.6f})"
    )
    print(f"  eig(Ct)                                   : {np.round(np.linalg.eigvalsh(Ct), 4)}")


def main():
    p = Params(**PARAMS_DEFAULT)
    inst = make_instance(np.random.default_rng(SEED))
    report_instance(inst, p)
    report_commuting_case(inst, p)
    report_non_commuting_case(inst, p)
    report_perturbative_check(inst, p)
    report_rank_deficient(p)


if __name__ == "__main__":
    main()
