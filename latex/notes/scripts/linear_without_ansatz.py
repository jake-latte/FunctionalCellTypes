"""Solve the linear one-hidden-layer fixed point without the shared-basis ansatz.

Per spatial frequency q = 0, ..., N-1 the data-interaction matrix is
    Sigma_q = (M_q - lam_q B)^{-1},   M_q = (lam_q A + sigma^2 I)^{-1} + r I,
    A = (k0/m) X X^T,                 B = (1/k1) Ct^{-1} Y Y^T Ct^{-1},
and the only coupling between frequencies is the hidden-layer kernel
    Ct = delta I + (1/N) sum_q lam_q Sigma_q,    delta = 1/(k1 kappa).
The script iterates this P x P map (damped, from a large Ct so that the tilt B is
small and every precision M_q - lam_q B stays positive definite) and then
  (i)   checks the per-q solver against the full NP x NP kernel map,
  (ii)  checks that the commuting case reproduces the scalar roots of eq:linear-soln,
  (iii) reports how much the eigenbasis of Sigma_q rotates with q when XX^T and YY^T
        do not commute, and how close the coarsest and finest frequencies come to the
        limiting bases eig(B) and eig(A/sigma^4 + B),
  (iv)  checks the first-order (in the non-commuting part of YY^T) formulae for the
        correction to Ct and for the per-q rotation against the solver,
  (v)   checks the block structure of a rank-deficient latent task, with orthonormal and
        with non-orthonormal latent coordinates Z.

Conventions: the frequency index q is 0-based (q = 0 the coarsest scale, q = N/2 the
finest); data modes p are 1-based in the printed output, as in the note. Computation
lives in functions that return numbers (the tests assert on them); the report_* functions
only print. Run from the git root:
    uv run python latex/notes/scripts/linear_without_ansatz.py
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

SEED, SEED_RANK_DEFICIENT = 0, 1
N_SITES, N_PATTERNS, M_IN, N_LATENT = 32, 6, 6, 3
KERNEL_WIDTH = np.sqrt(2.0)  # K_ij = exp(-r_ij^2 / (2 w^2))
SX2 = np.array([2.0, 1.4, 1.0, 0.7, 0.4, 0.2])  # singular values^2 of X
SY2 = np.array([1.0, 0.8, 0.6, 0.4, 0.25, 0.15])  # singular values^2 of Y (eps = 0)
ROTATION_ANGLE = np.pi / 6  # output basis vs input basis at eps = 1
PARAMS_DEFAULT = {"k0": 0.7, "k1": 1.3, "sig2": 0.2, "r": 0.3, "delta": 0.5}
EPS_CHECK = (0.02, 0.01)  # perturbative check: the residual must halve
C_START, C_START_ALT = 5.0, 50.0  # two starts, both with a small tilt
DAMPING, TOL, MAX_ITER = 0.5, 1e-12, 50_000
MAX_STEP_HALVINGS = 8  # retries when an iterate leaves the admissible set
BISECT_HI, BISECT_STEPS = 1e3, 200
Q_REPORT = (0, 2, 4, 6, 8, 12, 16)
ZZ_OFFDIAG, GY_DIAG = 0.15, (1.0, 0.3)  # commuting-latents, non-orthonormal-Z counter-instance


@dataclass(frozen=True)
class Params:
    """Network constants of Section 2: input/readout prior scales, noise, rate penalty, ridge."""

    k0: float
    k1: float
    sig2: float
    r: float
    delta: float


class NotPositiveDefiniteError(ValueError):
    """A per-frequency precision M_q - lam_q B is not positive definite: no Gaussian exists."""


# --- spatial kernel -----------------------------------------------------------------
def ring_kernel(n_sites: int, width: float) -> np.ndarray:
    """Translation-invariant Gaussian kernel on a periodic ring, shape (N, N)."""
    r = np.minimum(np.arange(n_sites), n_sites - np.arange(n_sites))
    row = np.exp(-(r**2) / (2 * width**2))
    return np.array([np.roll(row, i) for i in range(n_sites)])


def ring_eigenvalues(n_sites: int, width: float) -> np.ndarray:
    """lam_q for q = 0, ..., N-1 (lam_q = lam_{N-q}; q = N/2 is the finest scale)."""
    lam = np.fft.fft(ring_kernel(n_sites, width)[0])
    if np.abs(lam.imag).max() > 1e-12:
        raise ValueError("kernel row is not symmetric: eigenvalues are complex")
    return lam.real


# --- the per-frequency fixed-point map -----------------------------------------------
def prior_precision(lam: np.ndarray, A: np.ndarray, p: Params) -> np.ndarray:
    """M_q = (lam_q A + sigma^2 I)^{-1} + r I, shape (N, P, P); depends on X only."""
    I = np.eye(A.shape[0])
    return np.linalg.inv(lam[:, None, None] * A + p.sig2 * I) + p.r * I


def tilt(Ct: np.ndarray, YY: np.ndarray, p: Params) -> np.ndarray:
    """B = (1/k1) Ct^{-1} Y Y^T Ct^{-1}: the target tilt, the only place Y enters."""
    Ci = np.linalg.inv(Ct)
    return Ci @ YY @ Ci / p.k1


def per_q_precision(M: np.ndarray, lam: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Sigma_q^{-1} = M_q - lam_q B, shape (N, P, P)."""
    return M - lam[:, None, None] * B


def per_q_cov(M: np.ndarray, lam: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Sigma_q = (M_q - lam_q B)^{-1}; fails loudly if any precision is not positive definite."""
    precision = per_q_precision(M, lam, B)
    min_eig = np.linalg.eigvalsh(precision).min()
    if min_eig <= 0:
        raise NotPositiveDefiniteError(f"min eigenvalue of a per-q precision is {min_eig:.3g}")
    return np.linalg.inv(precision)


def _kernel_map(
    Ct: np.ndarray, lam: np.ndarray, M: np.ndarray, YY: np.ndarray, p: Params
) -> np.ndarray:
    Sigma = per_q_cov(M, lam, tilt(Ct, YY, p))
    return p.delta * np.eye(Ct.shape[0]) + np.einsum("q,qij->ij", lam, Sigma) / len(lam)


def kernel_map(
    Ct: np.ndarray, lam: np.ndarray, A: np.ndarray, YY: np.ndarray, p: Params
) -> np.ndarray:
    """F(Ct) = delta I + (1/N) sum_q lam_q Sigma_q(Ct)."""
    return _kernel_map(Ct, lam, prior_precision(lam, A, p), YY, p)


@dataclass(frozen=True)
class FixedPoint:
    Ct: np.ndarray
    residual: float  # max |F(Ct) - Ct| at the returned Ct
    n_iter: int


def solve_fixed_point(
    lam: np.ndarray,
    A: np.ndarray,
    YY: np.ndarray,
    p: Params,
    Ct_start: np.ndarray | None = None,
    c_start: float = C_START,
    damping: float = DAMPING,
    tol: float = TOL,
    max_iter: int = MAX_ITER,
) -> FixedPoint:
    """Damped iteration Ct <- Ct + (1 - a)(F(Ct) - Ct) from a large Ct (small tilt).

    Stops when the true residual max|F(Ct) - Ct| is below `tol`. F is decreasing in Ct (a
    larger Ct weakens the tilt B), so its Jacobian at the fixed point has negative
    eigenvalues and the undamped iteration oscillates when a slope exceeds 1 in magnitude.
    The damping needed is not known in advance, so whenever a step grows the damping is
    tightened (1 - a halved). A step that leaves the admissible set (some M_q - lam_q B not
    positive definite) is halved and retried a few times before raising.
    """
    M = prior_precision(lam, A, p)
    Ct = c_start * np.eye(A.shape[0]) if Ct_start is None else Ct_start
    F = _kernel_map(Ct, lam, M, YY, p)
    last_step = np.inf
    for n_iter in range(1, max_iter + 1):
        residual = np.abs(F - Ct).max()
        if residual < tol:
            return FixedPoint(Ct, residual, n_iter)
        step = (1 - damping) * (F - Ct)
        for _ in range(MAX_STEP_HALVINGS):
            try:
                F_next = _kernel_map(Ct + step, lam, M, YY, p)
                break
            except NotPositiveDefiniteError:
                step /= 2
        else:
            raise NotPositiveDefiniteError(
                f"iteration left the admissible set at step {n_iter}: start from a larger "
                f"Ct (raise c_start, now {c_start}, or pass a larger Ct_start)"
            )
        Ct, F = Ct + step, F_next
        step_size = np.abs(step).max()
        if step_size > last_step:
            damping = 1 - (1 - damping) / 2
        last_step = step_size
    raise RuntimeError(
        f"not converged after {max_iter} iterations: residual {residual:.3g}, damping {damping:.3g}"
    )


def per_q_cov_at(lam: np.ndarray, A: np.ndarray, YY: np.ndarray, p: Params) -> np.ndarray:
    """Sigma_q at the fixed point, shape (N, P, P)."""
    Ct = solve_fixed_point(lam, A, YY, p).Ct
    return per_q_cov(prior_precision(lam, A, p), lam, tilt(Ct, YY, p))


def full_kernel_map(
    Ct: np.ndarray, K: np.ndarray, A: np.ndarray, YY: np.ndarray, p: Params
) -> np.ndarray:
    """The kernel map evaluated on the full NP x NP Gaussian (no Fourier decomposition).

    Index ordering (mu, i). Ground truth for the per-q decomposition.
    """
    n_sites, P = K.shape[0], A.shape[0]
    I = np.eye(n_sites * P)
    precision = np.linalg.inv(np.kron(A, K) + p.sig2 * I) - (np.kron(tilt(Ct, YY, p), K) - p.r * I)
    Cov = np.linalg.inv(precision).reshape(P, n_sites, P, n_sites)
    return p.delta * np.eye(P) + np.einsum("ij,uivj->uv", K, Cov) / n_sites


# --- the commuting case: the scalar reduction ----------------------------------------
def h_over_lam(lam: np.ndarray, sx2hat: float, p: Params) -> np.ndarray:
    """h_q / lam_q = a / (1 + r a) with a = lam_q sX^2 + sigma^2; finite at lam_q = 0."""
    a = lam * sx2hat + p.sig2
    return a / (1 + p.r * a)


def bisect(f: Callable[[float], float], lo: float, hi: float, n: int = BISECT_STEPS) -> float:
    """Root of an increasing f on (lo, hi); fails loudly if not bracketed."""
    if not (f(lo) < 0 < f(hi)):
        raise ValueError(f"root not bracketed: f(lo)={f(lo)}, f(hi)={f(hi)}")
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def scalar_root(lam: np.ndarray, sx2hat: float, sy2hat: float, p: Params) -> float:
    """Physical root c > sY sqrt(max h_q) of 1 - delta/c = (c/N) sum_q h_q / (c^2 - sY^2 h_q)."""
    h = lam * h_over_lam(lam, sx2hat, p)

    def f(c):
        return 1 - p.delta / c - np.mean(c * h / (c**2 - sy2hat * h))

    pole = np.sqrt(sy2hat * h.max())
    return bisect(f, pole * (1 + 1e-12) + 1e-12, BISECT_HI)


def scalar_roots(lam: np.ndarray, sx2hat: np.ndarray, sy2hat: np.ndarray, p: Params) -> np.ndarray:
    """One physical root of eq:linear-soln per data mode."""
    return np.array([scalar_root(lam, sx, sy, p) for sx, sy in zip(sx2hat, sy2hat)])


def prior_scalar(lam: np.ndarray, p: Params) -> float:
    """Prior c on a direction orthogonal to both Gram matrices (eq:c0 of the note)."""
    return p.delta + p.sig2 / (1 + p.r * p.sig2) * lam.mean()


# --- instance: input basis U, output basis U R(angle) ------------------------------
def givens_chain(n: int, angle: float) -> np.ndarray:
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
    """X X^T = U diag(sx2) U^T and Y Y^T = U (diag(sy2) + eps E) U^T on a ring of N sites."""

    n_sites: int
    width: float
    m_in: int
    lam: np.ndarray
    U: np.ndarray
    sx2: np.ndarray
    sy2: np.ndarray
    E: np.ndarray

    def A(self, p: Params) -> np.ndarray:
        """(k0/m) X X^T."""
        return self.U @ np.diag(self.sx2hat(p)) @ self.U.T

    def gram_y(self, eps: float) -> np.ndarray:
        """Y Y^T at the given eps; fails loudly if it is not positive semi-definite."""
        G = np.diag(self.sy2) + eps * self.E
        if np.linalg.eigvalsh(G).min() < 0:
            raise ValueError(f"Y Y^T is not positive semi-definite at eps={eps}")
        return self.U @ G @ self.U.T

    def sx2hat(self, p: Params) -> np.ndarray:
        """Per-mode k0 sX^2 / m."""
        return p.k0 * self.sx2 / self.m_in

    def sy2hat(self, p: Params) -> np.ndarray:
        """Per-mode sY^2 / k1."""
        return self.sy2 / p.k1


def make_instance(
    rng: np.random.Generator,
    n_sites: int = N_SITES,
    n_patterns: int = N_PATTERNS,
    width: float = KERNEL_WIDTH,
    angle: float = ROTATION_ANGLE,
) -> Instance:
    """X X^T = U S_X^2 U^T; at eps = 1, Y Y^T = U R S_Y^2 R^T U^T with R a rotation by `angle`.

    E = R S_Y^2 R^T - S_Y^2, so diag(sy2) + eps E is a convex combination of PSD matrices
    for eps in [0, 1] and the commutator [X X^T, Y Y^T] grows with eps from zero.
    """
    U = np.linalg.qr(rng.normal(size=(n_patterns, n_patterns)))[0]
    sx2, sy2 = SX2[:n_patterns], SY2[:n_patterns]
    R = givens_chain(n_patterns, angle)
    E = R @ np.diag(sy2) @ R.T - np.diag(sy2)
    return Instance(n_sites, width, M_IN, ring_eigenvalues(n_sites, width), U, sx2, sy2, E)


# --- eigenbasis diagnostics ---------------------------------------------------------
def eigenbases(S: np.ndarray) -> np.ndarray:
    """Eigenvector matrices of a batch of symmetric matrices (..., P, P), eigenvalues ascending."""
    return np.linalg.eigh(S)[1]


def angles_to_nearest(V: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Angle (deg) of each column of V (batched, (..., P, P)) to the nearest column of W (P, P)."""
    overlap = np.abs(np.swapaxes(V, -1, -2) @ W)
    return np.degrees(np.arccos(np.clip(overlap.max(axis=-1), 0, 1)))


def relative_commutator(X: np.ndarray, Y: np.ndarray) -> float:
    """|[X, Y]|_F / (|X|_F |Y|_F): zero iff X and Y commute."""
    return np.linalg.norm(X @ Y - Y @ X) / (np.linalg.norm(X) * np.linalg.norm(Y))


def offdiagonal_fraction(S: np.ndarray, V: np.ndarray) -> np.ndarray:
    """|off-diagonal of V^T S_q V|_F / |S_q|_F per batch element: zero iff V diagonalises S_q."""
    R = V.T @ S @ V
    off = R - np.eye(R.shape[-1]) * np.diagonal(R, axis1=-2, axis2=-1)[..., None, :]
    return np.linalg.norm(off, axis=(-2, -1)) / np.linalg.norm(S, axis=(-2, -1))


def match_modes(V: np.ndarray) -> np.ndarray:
    """Reorder and sign-fix eigenvector columns so that column p is the one closest to e_p."""
    order = np.abs(V).argmax(axis=0)
    if len(set(order)) != V.shape[0]:
        raise ValueError("eigenvectors cannot be matched one-to-one to the reference modes")
    W = np.zeros_like(V)
    W[:, order] = V
    return W * np.sign(np.diag(W))


def mixing_from_eigenvectors(U: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """V_q - I for the eigenbasis V_q of U^T Sigma_q U, modes matched to the columns of U."""
    V = eigenbases(U.T @ Sigma @ U)
    return np.array([match_modes(Vq) - np.eye(Vq.shape[0]) for Vq in V])


def two_latent_angle(lam: np.ndarray, xhat: np.ndarray, B: np.ndarray, p: Params) -> np.ndarray:
    """tan 2 theta_q of eq:two-latent-angle for a 2 x 2 problem with A = diag(xhat) and tilt B."""
    input_term = (xhat[0] - xhat[1]) / ((lam * xhat[0] + p.sig2) * (lam * xhat[1] + p.sig2))
    return 2 * B[0, 1] / (B[0, 0] - B[1, 1] + input_term)


# --- computations reported in the note ------------------------------------------------
@dataclass(frozen=True)
class CommutingCheck:
    root_error: float  # max |eig(Ct) - scalar roots|
    offdiag_fraction: float  # of Ct in the basis U


def commuting_check(inst: Instance, p: Params) -> CommutingCheck:
    """At eps = 0 the fixed point is U-diagonal with the roots of eq:linear-soln on the diagonal."""
    Ct = solve_fixed_point(inst.lam, inst.A(p), inst.gram_y(0.0), p).Ct
    roots = scalar_roots(inst.lam, inst.sx2hat(p), inst.sy2hat(p), p)
    root_error = np.abs(np.sort(np.linalg.eigvalsh(Ct)) - np.sort(roots)).max()
    return CommutingCheck(root_error, float(offdiagonal_fraction(Ct[None], inst.U)[0]))


@dataclass(frozen=True)
class NonCommutingSolution:
    Ct: np.ndarray
    B: np.ndarray
    Sigma: np.ndarray  # (N, P, P)
    n_iter: int
    full_map_residual: float  # max |F_full(Ct) - Ct| with the NP x NP Gaussian
    start_difference: float  # max |Ct(5 I start) - Ct(50 I start)|
    min_precision_eig: float
    commutator_to_q0: np.ndarray  # relative |[Sigma_q, Sigma_0]| per q
    offdiag_in_eigCt: np.ndarray  # off-diagonal fraction of Sigma_q in eig(Ct), per q
    angles_to_q0: np.ndarray  # (N, P): each eigenvector of Sigma_q to its nearest at q = 0
    angles_to_U: np.ndarray  # (N, P): to the input basis U
    angles_to_B: np.ndarray  # (N, P): to the eigenbasis of the tilt B (the lam -> infinity limit)
    angles_to_fine_limit: np.ndarray  # (N, P): to the eigenbasis of A/sigma^4 + B (lam -> 0)
    coarse_limit_ratio: float  # |A^{-1}| / |B| (spectral norms): the B limit needs lam^2 >> this


def non_commuting_solution(inst: Instance, p: Params, eps: float = 1.0) -> NonCommutingSolution:
    """Fixed point at the given eps with the checks (i), (iii) of the module docstring."""
    A, YY = inst.A(p), inst.gram_y(eps)
    fp = solve_fixed_point(inst.lam, A, YY, p)
    Ct_alt = solve_fixed_point(inst.lam, A, YY, p, c_start=C_START_ALT).Ct
    M, B = prior_precision(inst.lam, A, p), tilt(fp.Ct, YY, p)
    Sigma = per_q_cov(M, inst.lam, B)
    V = eigenbases(Sigma)
    K = ring_kernel(inst.n_sites, inst.width)
    return NonCommutingSolution(
        Ct=fp.Ct,
        B=B,
        Sigma=Sigma,
        n_iter=fp.n_iter,
        full_map_residual=np.abs(full_kernel_map(fp.Ct, K, A, YY, p) - fp.Ct).max(),
        start_difference=np.abs(fp.Ct - Ct_alt).max(),
        min_precision_eig=np.linalg.eigvalsh(per_q_precision(M, inst.lam, B)).min(),
        commutator_to_q0=np.array([relative_commutator(S, Sigma[0]) for S in Sigma]),
        offdiag_in_eigCt=offdiagonal_fraction(Sigma, eigenbases(fp.Ct)),
        angles_to_q0=angles_to_nearest(V, V[0]),
        angles_to_U=angles_to_nearest(V, inst.U),
        angles_to_B=angles_to_nearest(V, eigenbases(B)),
        angles_to_fine_limit=angles_to_nearest(V, eigenbases(A / p.sig2**2 + B)),
        coarse_limit_ratio=np.linalg.norm(np.linalg.inv(A), 2) / np.linalg.norm(B, 2),
    )


@dataclass(frozen=True)
class FirstOrder:
    """First-order theory in the basis U. Index layout [q, r, p] per frequency, [p, r] otherwise.

    c, d: zeroth-order diagonals of Ct (P,) and Sigma_q (N, P).
    B1, C1: eq:first-order, (P, P).  Sigma1: eq:Sigma1, (N, P, P).
    Theta: eq:theta, (N, P, P); column p of the eigenbasis of Sigma_q is e_p + eps Theta[q, :, p].
    """

    c: np.ndarray
    d: np.ndarray
    B1: np.ndarray
    C1: np.ndarray
    Sigma1: np.ndarray
    Theta: np.ndarray


def first_order(inst: Instance, p: Params) -> FirstOrder:
    """Closed-form first-order corrections around the commuting point (note, Section 3).

    In the basis U: c and d from the scalar reduction (eq:linear-soln), T from eq:T, B1 and C1
    from eq:first-order, Sigma1 from eq:Sigma1, Theta from eq:theta. Index layout: B1, C1, T
    are [p, r]; d is [q, p]; Sigma1 and Theta are [q, r, p], so Theta[q, r, p] is the
    coefficient of e_r in column p of the eigenbasis of Sigma_q (zero on the diagonal).
    """
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


@dataclass(frozen=True)
class FirstOrderResiduals:
    eps: tuple[float, ...]
    C1: tuple[float, ...]  # max |(Ct(eps) - Ct(0))/eps - C1| per eps
    Theta: tuple[float, ...]  # max |(V_q(eps) - I)/eps - Theta_q| per eps


def first_order_residuals(
    inst: Instance, p: Params, eps_values: Sequence[float] = EPS_CHECK
) -> FirstOrderResiduals:
    """Residuals of the first-order theory against the solver; O(eps) if the theory is right."""
    theory = first_order(inst, p)
    A = inst.A(p)
    Ct0 = solve_fixed_point(inst.lam, A, inst.gram_y(0.0), p).Ct
    res_C, res_T = [], []
    for eps in eps_values:
        YY = inst.gram_y(eps)
        Ct = solve_fixed_point(inst.lam, A, YY, p).Ct
        Sigma = per_q_cov(prior_precision(inst.lam, A, p), inst.lam, tilt(Ct, YY, p))
        res_C.append(np.abs(inst.U.T @ (Ct - Ct0) @ inst.U / eps - theory.C1).max())
        res_T.append(np.abs(mixing_from_eigenvectors(inst.U, Sigma) / eps - theory.Theta).max())
    return FirstOrderResiduals(tuple(eps_values), tuple(res_C), tuple(res_T))


def theta_denominator(
    inst: Instance, p: Params, c: np.ndarray, pair: tuple[int, int], lam: float
) -> float:
    """Denominator of eq:theta for the modes (p, r) = pair as a function of lam_q."""
    sx, sy = inst.sx2hat(p), inst.sy2hat(p)
    i, j = pair
    input_term = (sx[i] - sx[j]) / ((lam * sx[i] + p.sig2) * (lam * sx[j] + p.sig2))
    return input_term + sy[i] / c[i] ** 2 - sy[j] / c[j] ** 2


def resonance_lambda(
    inst: Instance, p: Params, c: np.ndarray, pair: tuple[int, int]
) -> float | None:
    """lam^* above lam_max where the denominator of eq:theta vanishes, or None if it keeps its sign.

    None is a legitimate answer (no avoided crossing beyond the kernel's spectrum), not a failure.
    """
    lo, hi = inst.lam.max(), BISECT_HI
    f_lo, f_hi = (theta_denominator(inst, p, c, pair, x) for x in (lo, hi))
    if f_lo * f_hi > 0:
        return None
    sign = np.sign(f_hi)
    return bisect(lambda x: sign * theta_denominator(inst, p, c, pair, x), lo, hi)


@dataclass(frozen=True)
class RankDeficientInstance:
    """X X^T = Z G_X Z^T, Y Y^T = Z G_Y Z^T with Z (P x L), L < P latents; 1/m absorbed in G_X."""

    lam: np.ndarray
    Z: np.ndarray
    GX: np.ndarray
    GY: np.ndarray

    def A(self, p: Params) -> np.ndarray:
        """k0 Z G_X Z^T."""
        return p.k0 * self.Z @ self.GX @ self.Z.T

    @property
    def YY(self) -> np.ndarray:
        """Z G_Y Z^T."""
        return self.Z @ self.GY @ self.Z.T


def random_spd(rng: np.random.Generator, k: int) -> np.ndarray:
    """G G^T / k for a k x k standard normal G: a generic positive-definite matrix."""
    G = rng.normal(size=(k, k))
    return G @ G.T / k


def make_rank_deficient_instance(
    rng: np.random.Generator,
    n_patterns: int = N_PATTERNS,
    n_latent: int = N_LATENT,
    n_sites: int = N_SITES,
    orthonormal_z: bool = True,
    GX: np.ndarray | None = None,
    GY: np.ndarray | None = None,
) -> RankDeficientInstance:
    """Z is the Q factor of a normal matrix (orthonormal columns) or the normal matrix itself."""
    Z = rng.normal(size=(n_patterns, n_latent))
    if orthonormal_z:
        Z = np.linalg.qr(Z)[0]
    GX = random_spd(rng, n_latent) if GX is None else GX
    GY = random_spd(rng, n_latent) if GY is None else GY
    return RankDeficientInstance(ring_eigenvalues(n_sites, KERNEL_WIDTH), Z, GX, GY)


def make_commuting_latents_non_orthonormal_z(
    rng: np.random.Generator,
    n_patterns: int = N_PATTERNS - 1,
    n_sites: int = N_SITES,
    zz_offdiag: float = ZZ_OFFDIAG,
    gy_diag: tuple[float, float] = GY_DIAG,
) -> RankDeficientInstance:
    """Two latents with [G_X, G_Y] = 0 (G_X = I, G_Y diagonal) but Z^T Z not proportional to I.

    Z = Q L^T with Q orthonormal and L L^T = [[1, zz_offdiag], [zz_offdiag, 1]], so the Gram
    matrices restricted to col(Z) are L G_X L^T and L G_Y L^T, which do not commute.
    """
    Q = np.linalg.qr(rng.normal(size=(n_patterns, 2)))[0]
    L = np.linalg.cholesky(np.array([[1.0, zz_offdiag], [zz_offdiag, 1.0]]))
    return RankDeficientInstance(
        ring_eigenvalues(n_sites, KERNEL_WIDTH), Q @ L.T, np.eye(2), np.diag(gy_diag)
    )


@dataclass(frozen=True)
class RankDeficientCheck:
    n_iter: int
    c0: float  # prior value expected on the complement of col(Z)
    block_offdiag: float  # max |Pi Ct Pi_perp|
    complement_error: float  # max |Pi_perp Ct Pi_perp - c0 Pi_perp|
    eig_Ct: np.ndarray
    commutator_G: float  # relative |[G_X, G_Y]|
    commutator_restricted: float  # relative commutator of the Gram matrices restricted to col(Z)
    max_angle_to_q0: float  # largest angle (deg) of an eigvec of Q^T Sigma_q Q to nearest at q=0


def rank_deficient_check(rd: RankDeficientInstance, p: Params) -> RankDeficientCheck:
    """Block structure of the fixed point (note, Section 2.5) and the rotation within col(Z).

    The rotation is measured on Q^T Sigma_q Q (Q an orthonormal basis of col(Z)): the complement
    carries one degenerate eigenvalue at every q, where eigh returns an arbitrary basis.
    """
    A, YY = rd.A(p), rd.YY
    fp = solve_fixed_point(rd.lam, A, YY, p)
    Q = np.linalg.qr(rd.Z)[0]  # orthonormal basis of col(Z)
    Pi = Q @ Q.T
    Pi_perp = np.eye(Pi.shape[0]) - Pi
    c0 = prior_scalar(rd.lam, p)
    Sigma = per_q_cov(prior_precision(rd.lam, A, p), rd.lam, tilt(fp.Ct, YY, p))
    V = eigenbases(Q.T @ Sigma @ Q)
    return RankDeficientCheck(
        n_iter=fp.n_iter,
        c0=c0,
        block_offdiag=np.abs(Pi @ fp.Ct @ Pi_perp).max(),
        complement_error=np.abs(Pi_perp @ fp.Ct @ Pi_perp - c0 * Pi_perp).max(),
        eig_Ct=np.linalg.eigvalsh(fp.Ct),
        commutator_G=relative_commutator(rd.GX, rd.GY),
        commutator_restricted=relative_commutator(Q.T @ A @ Q, Q.T @ YY @ Q),
        max_angle_to_q0=angles_to_nearest(V, V[0]).max(),
    )


# --- printing ------------------------------------------------------------------------
def row(label: str, value: object, width: int = 44) -> None:
    """One aligned 'label : value' line."""
    print(f"  {label:<{width}}: {value}")


def report_instance(inst: Instance, p: Params) -> None:
    """The instance: ring, spectrum of K, and how far the two Gram matrices are from commuting."""
    YY = inst.gram_y(1.0)
    angles = angles_to_nearest(inst.U, eigenbases(YY))
    print(
        f"ring N={inst.n_sites}, width={inst.width:.4g}, P={len(inst.sx2)}, m={inst.m_in}; "
        f"lam_q in [{inst.lam.min():.3g}, {inst.lam.max():.3g}]"
    )
    row("relative commutator |[A, YY^T]| at eps=1", f"{relative_commutator(inst.A(p), YY):.3f}")
    row("angle (deg) of eigvecs of A to nearest of YY^T", np.round(angles, 1))


def report_commuting_case(inst: Instance, p: Params) -> None:
    """Check (ii): the commuting case reproduces the scalar roots."""
    res = commuting_check(inst, p)
    print("\n[commuting case, eps = 0]")
    row("eig(Ct) - scalar roots, max abs", f"{res.root_error:.1e}")
    row("off-diagonal fraction of Ct in U", f"{res.offdiag_fraction:.1e}")


def report_non_commuting_case(inst: Instance, p: Params) -> None:
    """Checks (i) and (iii): solver consistency and the rotation of the per-q eigenbases."""
    s = non_commuting_solution(inst, p)
    q_fine = inst.n_sites // 2
    print(f"\n[non-commuting case, eps = 1]  ({s.n_iter} iterations to a residual below {TOL:g})")
    row("full NP x NP map residual at fixed point", f"{s.full_map_residual:.1e}")
    row(f"start {C_START:g} I vs start {C_START_ALT:g} I, max |diff|", f"{s.start_difference:.1e}")
    row("min eig of per-q precisions", f"{s.min_precision_eig:.4f}")
    row("eig(Ct)", np.round(np.linalg.eigvalsh(s.Ct), 4))
    row("max_q |[Sigma_q, Sigma_0]| / (|Sigma_q| |Sigma_0|)", f"{s.commutator_to_q0.max():.3f}")
    row("max_q off-diag fraction of Sigma_q in eig(Ct)", f"{s.offdiag_in_eigCt.max():.3f}")
    print("  q   lam_q    angle (deg) of each eigenvector of Sigma_q to its q=0 partner")
    for q in Q_REPORT:
        print(f"  {q:2d}  {inst.lam[q]:7.4f}  {np.round(s.angles_to_q0[q], 1)}")
    print("  angle (deg) of each eigenvector of Sigma_q to the nearest eigenvector of a reference")
    coarse, fine = f"q=0 (lam={inst.lam[0]:.4f})", f"q={q_fine} (lam={inst.lam[q_fine]:.4f})"
    row(f"{coarse} to U", np.round(s.angles_to_U[0], 1))
    row(f"{coarse} to eig(B)", np.round(s.angles_to_B[0], 1))
    row(f"{fine} to U", np.round(s.angles_to_U[q_fine], 1))
    row(f"{fine} to eig(A/sig^4 + B)", np.round(s.angles_to_fine_limit[q_fine], 2))
    row(
        "eig(B) limit needs lam^2 >> |A^-1|/|B|",
        f"{s.coarse_limit_ratio:.0f}; lam_max^2 = {inst.lam.max() ** 2:.1f}",
    )


def report_perturbative_check(inst: Instance, p: Params) -> None:
    """Check (iv): first-order theory against the solver; the pair nearest an avoided crossing."""
    theory = first_order(inst, p)
    res = first_order_residuals(inst, p)
    q_lo, q_hi = 0, inst.n_sites // 2
    print("\n[first-order theory vs solver]")
    row("max |C1|, max |Theta|", f"{np.abs(theory.C1).max():.4f}, {np.abs(theory.Theta).max():.4f}")
    for eps, rc, rt in zip(res.eps, res.C1, res.Theta):
        row(f"eps={eps}: max |C1_num - C1|, |Theta_num - Theta|", f"{rc:.2e}, {rt:.2e}")
    ratios = (
        f"C1 {res.C1[0] / res.C1[1]:.2f}, Theta {res.Theta[0] / res.Theta[1]:.2f}  (2 = O(eps))"
    )
    row("residual ratio eps -> eps/2", ratios)
    scale_dependence = np.abs(theory.Theta[q_lo] - theory.Theta[q_hi])
    r_idx, p_idx = (
        int(i) for i in np.unravel_index(scale_dependence.argmax(), scale_dependence.shape)
    )
    thetas = (
        f"Theta[q=0]={theory.Theta[q_lo, r_idx, p_idx]:.4f}, "
        f"Theta[q=N/2]={theory.Theta[q_hi, r_idx, p_idx]:.4f}"
    )
    row(f"largest scale dependence: (Theta_q)_(r,p), (r,p)=({r_idx + 1},{p_idx + 1})", thetas)
    # the denominator of eq:theta for the pair (p, r) is odd under p <-> r, so lam* is the same
    lam_star = resonance_lambda(inst, p, theory.c, (p_idx, r_idx))
    where = "none above lam_max" if lam_star is None else f"{lam_star:.2f}"
    row(
        f"eq:theta denominator for modes ({r_idx + 1},{p_idx + 1}) vanishes at lam*",
        f"{where} (lam_max = {inst.lam.max():.2f})",
    )


def print_rank_deficient_check(res: RankDeficientCheck) -> None:
    """Check (v) lines shared by the two rank-deficient instances."""
    row("relative commutator |[G_X, G_Y]|", f"{res.commutator_G:.3f}")
    row("relative commutator of the Gram matrices on col(Z)", f"{res.commutator_restricted:.3f}")
    row("max |Pi Ct Pi_perp|", f"{res.block_offdiag:.1e}")
    row(
        "max |Pi_perp Ct Pi_perp - c0 Pi_perp|", f"{res.complement_error:.1e}   (c0 = {res.c0:.6f})"
    )
    row("eig(Ct)", np.round(res.eig_Ct, 4))
    row("largest angle (deg) in col(Z) of a Sigma_q eigvec to q=0", f"{res.max_angle_to_q0:.1f}")


def report_rank_deficient(p: Params) -> None:
    """Check (v) with orthonormal Z: the block structure, and rotation within col(Z)."""
    rd = make_rank_deficient_instance(np.random.default_rng(SEED_RANK_DEFICIENT))
    res = rank_deficient_check(rd, p)
    P, L = rd.Z.shape
    print(
        f"\n[rank-deficient task, L={L} < P={P}, orthonormal Z, non-commuting G_X, G_Y]"
        f"  ({res.n_iter} iterations)"
    )
    print_rank_deficient_check(res)


def report_rank_deficient_non_orthonormal(p: Params) -> None:
    """Check (v) with [G_X, G_Y] = 0 but Z^T Z not proportional to I: the bases still rotate."""
    rd = make_commuting_latents_non_orthonormal_z(np.random.default_rng(SEED_RANK_DEFICIENT))
    res = rank_deficient_check(rd, p)
    P, L = rd.Z.shape
    print(
        f"\n[rank-deficient task, L={L} < P={P}, Z^T Z off-diagonal {ZZ_OFFDIAG}, "
        f"G_X = I, G_Y = diag{GY_DIAG}]  ({res.n_iter} iterations)"
    )
    print_rank_deficient_check(res)


def main() -> None:
    p = Params(**PARAMS_DEFAULT)
    inst = make_instance(np.random.default_rng(SEED))
    report_instance(inst, p)
    report_commuting_case(inst, p)
    report_non_commuting_case(inst, p)
    report_perturbative_check(inst, p)
    report_rank_deficient(p)
    report_rank_deficient_non_orthonormal(p)


if __name__ == "__main__":
    main()
