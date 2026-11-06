"""The linear theory of one hidden layer on a ring, as pure functions of a `Config` and a `Dataset`.

The theory (latex/notes/linear_solution_limits.tex, eq:linear-soln; linear_without_ansatz.tex,
eq:master) describes the stationary law of a linear one-hidden-layer network, N units on a ring
with the spatial prior K (hidden kernel), input prior k0 I, readout prior k1 I, input noise sigma^2,
rate penalty (r/2) sum phi^2 and ridge delta = P n / (beta k1). Per data mode with input and target
weights sX2 = k0 s_X^2 / m, sY2 = s_Y^2 / k1 the hidden kernel eigenvalue c and the power
spectrum d_q over spatial frequencies q solve

    alpha_q = lam_q sX2 + sig2,   pi_q = alpha_q / (1 + r alpha_q),   h_q = lam_q pi_q,
    1 - delta / c = (c / N) sum_q h_q / (c^2 - sY2 h_q)   on the branch c > sqrt(sY2 max_q h_q),
    d_q = pi_q c^2 / (c^2 - sY2 h_q),

and without the shared-basis ansatz the P x P kernel C~ solves the master equation

    C~ = delta I + (1/N) sum_q lam_q Sigma_q,
    Sigma_q = [(lam_q A + sig2 I)^{-1} + r I - lam_q B]^{-1},   A = (k0/m) X X^T,
    B = C~^{-1} Y Y^T C~^{-1} / k1.

sigma -> 0 is a first-class case: the input noise is a formal regulariser (the note's Section
sec:sigma0), so `sig2` defaults to 0 everywhere and nothing divides by it. At sig2 = 0 a null mode
(sX2 = sY2 = 0) and every frequency with lam_q = 0 get d_q = 0 exactly, and Sigma_q is evaluated in
the form S (I + S T S)^{-1} S with S = (lam_q A + sig2 I)^{1/2}, T = r I - lam_q B, which is finite
when lam_q A is singular.

The rate coefficient is called `r` in every signature: it is written g^2 in the notes until their
rename, and `rate_coefficient` is the one place that derives it from a config.

Dictionary (the plan's Section sec:dictionary): N = net.N; m = data.D_in; n = D_out of the task;
P = data.P; K latents = data.n_latents, Z from dataset.factors z0..z{K-1}; the hidden kernel is
`build_spaces`' hidden population K and lam_q its Fourier eigenvalues (DFT order, q = 0 the
coarsest scale, q = N/2 the finest); k0, k1 are the input and output kernel amplitudes over
train.weight_lambda (Cov(W) = K / weight_lambda for every layer); beta is train.beta under SGLD
and 1 under Adam. The per-mode weights sx2, sy2 use the singular values of Z, which is exact for a
teacher with orthonormal columns (X X^T = Y Y^T = Z Z^T) and the zeroth-order ansatz otherwise;
`ansatz_free_kernel` uses the actual Gram matrices, so comparing the two measures the ansatz error.

Everything is float64 numpy; torch tensors are accepted anywhere an array is. No file I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

from fct.config import Config
from fct.data import task_dims
from fct.spatial import build_spaces
from fct.tasks import Dataset

ArrayLike = np.ndarray | torch.Tensor

BISECT_STEPS = 200  # halvings of the root bracket: far below double precision at any scale
# Eigenvalues of a kernel below -NEGATIVE_EIG_TOL * lam_max are an error; above, round-off set to 0.
NEGATIVE_EIG_TOL = 1e-12
CIRCULANT_TOL = 1e-10  # relative agreement required between the FFT of row 0 and eigvalsh
SCALAR_KERNEL_TOL = 1e-12  # relative: an input/output kernel must be amplitude * I to this
# The ansatz-free solver: Newton on the upper triangle of C~ with a finite-difference Jacobian.
NEWTON_STEPS = 100
NEWTON_TOL = 1e-13  # on max |F(C~) - C~| relative to max(1, |C~|)
FD_EPS = 1e-7  # relative finite-difference step
MAX_HALVINGS = 60  # of a Newton step before the line search is declared failed
MAX_DOUBLINGS = 200  # of the starting c before the start is declared unreachable
SUBSPACE_TOL = 1e-12  # relative: eigenvalues of X X^T + Y Y^T below this span nothing


def _array(x: ArrayLike) -> np.ndarray:
    """A float64 numpy copy of an array or a (possibly CUDA) tensor."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


# --------------------------------------------------------------------------- config


def inverse_temperature(cfg: Config) -> float:
    """beta of the stationary law exp(-beta L): train.beta under SGLD, 1 under Adam, which has no
    temperature (the rule of `fct.train.prior_coefficient`)."""
    return cfg.train.beta if cfg.train.optimizer == "sgld" else 1.0


def rate_coefficient(cfg: Config) -> float:
    """The rate-penalty coefficient r (written g^2 in the notes until the rename): the loss carries
    rate_lambda * sum phi^2, so beta L carries beta rate_lambda sum phi^2 = (r/2) sum phi^2 and
    r = 2 beta rate_lambda. This becomes `cfg.train.rate_lambda` when the rate-penalty code item
    (docs/issues/rate-penalty-code.md) scales the penalty by 1 / (2 beta)."""
    return 2.0 * inverse_temperature(cfg) * cfg.train.rate_lambda


def hidden_kernel(cfg: Config) -> np.ndarray:
    """The hidden population's kernel K (N, N) of `build_spaces` for the configured network."""
    if cfg.spatial is None:
        raise ValueError("the linear theory needs a spatial prior; cfg.spatial is None")
    m, n_out = task_dims(cfg.data)
    return _array(build_spaces(cfg.spatial, m=m, N=cfg.net.N, L=cfg.net.L, n_out=n_out)[0].post.K)


def ring_spectrum(K: ArrayLike) -> np.ndarray:
    """lam_q, q = 0..N-1 in DFT order, of a circulant symmetric kernel: the FFT of its first row,
    checked against `eigvalsh` (so a kernel that is not translation invariant fails loudly).
    Round-off negatives are set to 0; a clearly negative eigenvalue is an error."""
    K = _array(K)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be square, got shape {K.shape}")
    lam = np.fft.fft(K[0]).real
    scale = max(np.abs(lam).max(), np.finfo(float).tiny)
    eig = np.linalg.eigvalsh(K)
    if np.abs(np.sort(lam) - eig).max() > CIRCULANT_TOL * scale:
        raise ValueError("K is not circulant: the FFT of its first row is not its spectrum")
    if lam.min() < -NEGATIVE_EIG_TOL * scale:
        raise ValueError(f"K has a negative eigenvalue {lam.min():.3g}; it is not a covariance")
    # -1e-16 lam_0 at the Nyquist tail of a wrapped Gaussian is FFT round-off, not structure.
    return np.maximum(lam, 0.0)


def _scalar_kernel_amplitude(name: str, K: np.ndarray) -> float:
    """The a of K = a I; raises naming the population if K is not a multiple of the identity."""
    a = K[0, 0]
    if np.abs(K - a * np.eye(K.shape[0])).max() > SCALAR_KERNEL_TOL * abs(a):
        raise ValueError(
            f"the theory needs the {name} kernel to be a multiple of the identity "
            f"(spatial.{name}.kernel.length_scale = 0); it is not"
        )
    return float(a)


# --------------------------------------------------------------------------- the latent task


def latent_matrix(factors: dict[str, torch.Tensor] | None, n_latents: int) -> np.ndarray:
    """Z (P, K): the factors z0..z{K-1} of the linear teacher as columns."""
    names = [f"z{i}" for i in range(n_latents)]
    missing = [n for n in names if factors is None or n not in factors]
    if missing:
        raise ValueError(f"dataset.factors lacks the latent(s) {missing}")
    return np.stack([_array(factors[n]) for n in names], axis=1)


def latent_basis(Z: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Thin SVD of Z: (U_Z (P, K), s2 (K,)) with s2 descending, so column 0 is the mean direction
    of the uniform latent task; each column is signed so that its sum is non-negative."""
    U, s, _ = np.linalg.svd(_array(Z), full_matrices=False)
    U = U * np.where(U.sum(axis=0) < 0, -1.0, 1.0)
    return U, s**2


def uniform_latent_s2(P: int, K: int, a: float) -> np.ndarray:
    """The P >> K singular values squared of Z ~ U[0, a]^(P x K) (limits note, prop:svd):
    s_1^2 = P a^2 (3K + 1) / 12 on the mean direction, s_k^2 = P a^2 / 12 on the K - 1 contrasts."""
    return np.array([P * a**2 * (3 * K + 1) / 12] + [P * a**2 / 12] * (K - 1))


@dataclass(frozen=True)
class TheoryParameters:
    """The dictionary applied to one config and dataset (see the module docstring)."""

    N: int
    m: int
    n: int
    P: int
    n_latents: int
    lam: np.ndarray  # (N,) DFT order
    lam0: float  # max_q lam_q
    lam_bar: float  # mean_q lam_q = K_ii
    U_Z: np.ndarray  # (P, K) left singular vectors of Z
    s2: np.ndarray  # (K,) squared singular values of Z
    sx2: np.ndarray  # (K,) k0 s2 / m
    sy2: np.ndarray  # (K,) s2 / k1
    k0: float
    k1: float
    beta: float
    delta: float  # P n / (beta k1)
    sig2: float
    r: float


def theory_parameters(cfg: Config, dataset: Dataset, sig2: float = 0.0) -> TheoryParameters:
    """The theory's parameters from a config and its dataset; `sig2` is explicit because no config
    field holds the input noise (sigma -> 0 is the default)."""
    if cfg.net.L != 1:
        raise ValueError(f"the linear theory is for one hidden layer; net.L = {cfg.net.L}")
    if cfg.net.phi != "id":
        raise ValueError(f"the linear theory needs net.phi = 'id', got {cfg.net.phi!r}")
    if cfg.data.task != "linear_teacher":
        raise ValueError(f"the theory is for the linear_teacher task, got {cfg.data.task!r}")
    if sig2 < 0:
        raise ValueError(f"sig2 must be >= 0, got {sig2}")
    if cfg.spatial is None:
        raise ValueError("the linear theory needs a spatial prior; cfg.spatial is None")
    m, n = task_dims(cfg.data)
    spaces = build_spaces(cfg.spatial, m=m, N=cfg.net.N, L=1, n_out=n)
    lam = ring_spectrum(spaces[0].post.K)
    # Cov(vec W) = (K_post (x) K_pre) / weight_lambda, so the scalar factors carry 1 / weight_lambda
    # and the hidden kernel enters as built.
    k0 = _scalar_kernel_amplitude("input", _array(spaces[0].pre.K)) / cfg.train.weight_lambda
    k1 = _scalar_kernel_amplitude("output", _array(spaces[1].post.K)) / cfg.train.weight_lambda
    Z = latent_matrix(dataset.factors, cfg.data.n_latents)
    P = Z.shape[0]
    if dataset.x.shape != (P, m):
        raise ValueError(f"dataset.x has shape {tuple(dataset.x.shape)}, expected {(P, m)}")
    U_Z, s2 = latent_basis(Z)
    beta = inverse_temperature(cfg)
    return TheoryParameters(
        N=cfg.net.N,
        m=m,
        n=n,
        P=P,
        n_latents=cfg.data.n_latents,
        lam=lam,
        lam0=float(lam.max()),
        lam_bar=float(lam.mean()),
        U_Z=U_Z,
        s2=s2,
        sx2=k0 * s2 / m,
        sy2=s2 / k1,
        k0=k0,
        k1=k1,
        beta=beta,
        delta=P * n / (beta * k1),
        sig2=float(sig2),
        r=rate_coefficient(cfg),
    )


# --------------------------------------------------------------------------- the per-mode solution


def noise_floor(sig2: float, r: float) -> float:
    """sigma_0^2 = sig2 / (1 + r sig2): the flat spectrum of a null mode (lem:floor); 0 at sig2 = 0."""
    return sig2 / (1.0 + r * sig2)


def prior_shrunk_variance(lam: ArrayLike, sx2: float, sig2: float, r: float) -> np.ndarray:
    """pi_q = alpha_q / (1 + r alpha_q), alpha_q = lam_q sx2 + sig2 (eq:factor)."""
    alpha = _array(lam) * sx2 + sig2
    return alpha / (1.0 + r * alpha)


def h_of(lam: ArrayLike, sx2: float, sig2: float, r: float) -> np.ndarray:
    """h_q = lam_q pi_q (eq:hq)."""
    return _array(lam) * prior_shrunk_variance(lam, sx2, sig2, r)


def _residual(c: float, h: np.ndarray, sy2: float, delta: float) -> float:
    """1 - delta/c - (c/N) sum_q h_q / (c^2 - sY2 h_q): increasing in c above the pole."""
    return 1.0 - delta / c - np.mean(c * h / (c**2 - sy2 * h))


def physical_root(h: ArrayLike, sy2: float, delta: float) -> float:
    """The root c of eq:linear-soln on the physical branch c > sqrt(sY2 max_q h_q), by bisection.

    The residual rises from -inf at the pole to 1, so the root is unique. Brackets: just above the
    pole the residual is very negative, and so is it at c = delta / 2 (the term -delta / c alone
    gives -1), which serves when the pole lies below that (no target: pole = 0, where c -> 0 would
    underflow). For c^2 >= 2 sY2 max h every denominator is at least c^2 / 2, so the sum is at most
    2 mean(h) / c and the residual is positive once c > delta + 2 mean(h).
    """
    h = _array(h)
    if delta <= 0:
        raise ValueError(f"delta must be > 0, got {delta}")
    pole = np.sqrt(sy2 * h.max())
    lo = pole * (1 + 1e-13) if pole >= delta / 2 else delta / 2
    hi = max(np.sqrt(2 * sy2 * h.max()), delta + 2 * h.mean()) * (1 + 1e-13) + 1.0
    if not (_residual(lo, h, sy2, delta) < 0 < _residual(hi, h, sy2, delta)):
        raise ValueError(f"root not bracketed on [{lo:.3g}, {hi:.3g}]")
    for _ in range(BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _residual(mid, h, sy2, delta) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def per_mode_spectrum(
    lam: ArrayLike, sx2: float, sy2: float, sig2: float, r: float, delta: float
) -> tuple[float, np.ndarray]:
    """(c, d_q) of one data mode: the self-consistent kernel eigenvalue and the power spectrum
    over q (eq:linear-soln), with `r` the rate-penalty coefficient (g^2 in the notes).

    d_q = pi_q * c^2 / (c^2 - sY2 h_q) rather than h_q / lam_q: the quotient is 0/0 where
    lam_q = 0, while pi_q is 0 there at sig2 = 0 and sigma_0^2 otherwise. A null mode
    (sx2 = sy2 = 0) at sig2 = 0 has h = 0, so c = delta and d = 0 exactly.
    """
    lam = _array(lam)
    for name, value in {"sx2": sx2, "sy2": sy2, "sig2": sig2, "r": r}.items():
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    if (lam < 0).any():
        raise ValueError("lam has negative entries; pass it through ring_spectrum")
    h = h_of(lam, sx2, sig2, r)
    c = physical_root(h, sy2, delta)
    return c, prior_shrunk_variance(lam, sx2, sig2, r) * c**2 / (c**2 - sy2 * h)


def target_strength(
    lam: ArrayLike, sx2: float, sy2: float, sig2: float, r: float, c: float
) -> float:
    """tau = sY2 max_q h_q / c^2 in [0, 1): how far the mode is from the pole (eq:knobs)."""
    return sy2 * h_of(lam, sx2, sig2, r).max() / c**2


@dataclass(frozen=True)
class Prediction:
    """The per-mode theory of every latent mode: `d[k]` is the power spectrum of the tuning map
    on `U_Z[:, k]` (DFT order), `c[k]` its kernel eigenvalue, `tau[k]` its target strength; the
    P - K null directions are flat at `null_floor` = sigma_0^2."""

    lam: np.ndarray  # (N,)
    U_Z: np.ndarray  # (P, K)
    s2: np.ndarray  # (K,)
    d: np.ndarray  # (K, N)
    null_floor: float
    c: np.ndarray  # (K,)
    tau: np.ndarray  # (K,)
    params: TheoryParameters


def predicted_spectra(cfg: Config, dataset: Dataset, sig2: float = 0.0) -> Prediction:
    """`per_mode_spectrum` of every latent mode of the config's task (the theory's prediction for
    the measured tuning-map spectra); sigma^2 = 0 unless `sig2` is given."""
    tp = theory_parameters(cfg, dataset, sig2)
    modes = [
        per_mode_spectrum(tp.lam, sx2, sy2, tp.sig2, tp.r, tp.delta)
        for sx2, sy2 in zip(tp.sx2, tp.sy2)
    ]
    c = np.array([m[0] for m in modes])
    tau = np.array(
        [
            target_strength(tp.lam, sx2, sy2, tp.sig2, tp.r, ck)
            for sx2, sy2, ck in zip(tp.sx2, tp.sy2, c)
        ]
    )
    return Prediction(
        lam=tp.lam,
        U_Z=tp.U_Z,
        s2=tp.s2,
        d=np.stack([m[1] for m in modes]),
        null_floor=noise_floor(tp.sig2, tp.r),
        c=c,
        tau=tau,
        params=tp,
    )


# --------------------------------------------------------------------------- spectrum summaries


def autocorrelation(d: ArrayLike) -> np.ndarray:
    """R(r) = (1/N) sum_q d_q e^{2 pi i q r / N}, r = 0..N-1, of a power spectrum in DFT order."""
    return np.fft.ifft(_array(d)).real


def half_power_q(d: ArrayLike) -> float:
    """Largest |q| at which d_q - d_{N/2} is at least half of d_0 - d_{N/2} (d in DFT order): the
    Nyquist value is the measurable floor, as in the limits note's definition of q_{1/2}."""
    d = _array(d)
    q = np.abs(np.fft.fftfreq(d.size) * d.size)
    floor = d[d.size // 2]
    return float(q[(d - floor) >= 0.5 * (d[0] - floor)].max())


class SpectrumSummary(NamedTuple):
    q_half: float
    share0: float  # d_0 / sum_q d_q
    lag1: float  # R(1) / R(0)


def spectrum_summary(d: ArrayLike) -> SpectrumSummary:
    """(q_half, share0, lag1) of one power spectrum in DFT order; an identically zero spectrum
    (a null mode at sigma = 0) has no summary and is an error."""
    d = _array(d)
    if d.ndim != 1 or d.size < 2:
        raise ValueError(f"d must be a 1-d spectrum with at least 2 entries, got shape {d.shape}")
    if (d < 0).any():
        raise ValueError("a power spectrum cannot be negative")
    if not (d > 0).any():
        raise ValueError("the spectrum is identically zero: no shape to summarise")
    R = autocorrelation(d)
    return SpectrumSummary(half_power_q(d), float(d[0] / d.sum()), float(R[1] / R[0]))


# --------------------------------------------------------------------------- the ansatz-free kernel


class NotPhysicalError(ValueError):
    """Some per-frequency precision is not positive definite: no Gaussian exists for this C~."""


def _symmetric_sqrt(A: np.ndarray) -> np.ndarray:
    """The PSD square root of a symmetric PSD matrix; round-off negatives are clamped to 0."""
    w, V = np.linalg.eigh(A)
    if w.min() < -NEGATIVE_EIG_TOL * max(abs(w).max(), np.finfo(float).tiny):
        raise ValueError(f"matrix is not positive semi-definite (min eigenvalue {w.min():.3g})")
    return (V * np.sqrt(np.maximum(w, 0.0))) @ V.T


def _data_subspace(XX: np.ndarray, YY: np.ndarray) -> np.ndarray:
    """Q (P, L): an orthonormal basis of col(X) + col(Y). Off this subspace both Gram matrices
    vanish and the kernel is its prior value c_0 = delta + sigma_0^2 lam_bar (linear_without_ansatz,
    Section 2.5), so the master equation needs solving only on Q (L x L, L = K for the latent task)."""
    w, V = np.linalg.eigh(XX + YY)
    keep = w > SUBSPACE_TOL * max(w.max(), np.finfo(float).tiny)
    return V[:, keep]


def _per_q_covariances(S: np.ndarray, lam: np.ndarray, B: np.ndarray, r: float) -> np.ndarray:
    """Sigma_q = S_q (I + S_q T_q S_q)^{-1} S_q with T_q = r I - lam_q B, shape (N, L, L).

    Equal to [(lam_q A + sig2 I)^{-1} + r I - lam_q B]^{-1} when S_q^2 = lam_q A + sig2 I is
    invertible, and its limit otherwise (sig2 = 0 with lam_q A singular: Sigma_q vanishes on the
    null space). The Gaussian exists iff every I + S T S is positive definite.
    """
    L = B.shape[0]
    T = r * np.eye(L) - lam[:, None, None] * B
    W = np.eye(L) + S @ T @ S
    min_eig = np.linalg.eigvalsh(W).min()
    if min_eig <= 0:
        raise NotPhysicalError(f"a per-frequency precision has min eigenvalue {min_eig:.3g}")
    return S @ np.linalg.solve(W, S)


def _kernel_map(
    Ct: np.ndarray,
    S: np.ndarray,
    lam: np.ndarray,
    YY: np.ndarray,
    k1: float,
    r: float,
    delta: float,
) -> np.ndarray:
    """F(C~) = delta I + (1/N) sum_q lam_q Sigma_q(C~) on the data subspace (eq:master)."""
    Ci = np.linalg.inv(Ct)
    Sigma = _per_q_covariances(S, lam, Ci @ YY @ Ci / k1, r)
    return delta * np.eye(Ct.shape[0]) + np.einsum("q,qij->ij", lam, Sigma) / lam.size


def _sym_from_upper(v: np.ndarray, n: int) -> np.ndarray:
    C = np.zeros((n, n))
    C[np.triu_indices(n)] = v
    return C + np.triu(C, 1).T


def _physical_start(residual, n: int, c: float) -> np.ndarray:
    """c I on the admissible set: B shrinks like c^{-2}, so doubling c reaches it."""
    iu = np.triu_indices(n)
    for _ in range(MAX_DOUBLINGS):
        v = (c * np.eye(n))[iu]
        try:
            residual(v)
            return v
        except NotPhysicalError:
            c *= 2
    raise NotPhysicalError("no admissible starting point found by doubling c")


def _newton(residual, v: np.ndarray) -> np.ndarray:
    """Newton's method with a finite-difference Jacobian on the upper triangle of C~, each step
    halved until it stays admissible and reduces the residual. Plain damped iteration leaves the
    physical branch near the pole (the limits note's remark), Newton does not."""
    for _ in range(NEWTON_STEPS):
        res = residual(v)
        size = np.abs(res).max()
        if size <= NEWTON_TOL * max(1.0, np.abs(v).max()):
            return v
        eps = FD_EPS * np.maximum(1.0, np.abs(v))
        J = np.column_stack([(residual(v + e * h) - res) / h for e, h in zip(np.eye(v.size), eps)])
        step = -np.linalg.solve(J, res)
        for _ in range(MAX_HALVINGS):
            try:
                if np.abs(residual(v + step)).max() < size:
                    break
            except NotPhysicalError:
                pass
            step /= 2
        else:
            raise RuntimeError(
                "the Newton line search found no admissible step that reduces the residual"
            )
        v = v + step
    raise RuntimeError(f"Newton did not converge in {NEWTON_STEPS} steps: residual {size:.3g}")


def ansatz_free_kernel(cfg: Config, dataset: Dataset, sig2: float = 0.0) -> np.ndarray:
    """C~ (P, P), the fixed point of the master equation eq:master for the config's network and
    the dataset's actual Gram matrices (no shared-basis ansatz), at input noise `sig2`.

    Solved on the subspace spanned by the data (K-dimensional for the latent task) and set to the
    prior value c_0 = delta + sigma_0^2 lam_bar on its complement, which is exact.
    """
    tp = theory_parameters(cfg, dataset, sig2)
    X, Y = _array(dataset.x), _array(dataset.y)
    XX, YY = X @ X.T, Y @ Y.T
    Q = _data_subspace(XX, YY)
    c0 = tp.delta + noise_floor(tp.sig2, tp.r) * tp.lam_bar
    complement = c0 * (np.eye(tp.P) - Q @ Q.T)
    if Q.shape[1] == 0:
        return complement
    A = tp.k0 / tp.m * (Q.T @ XX @ Q)
    YY_q = Q.T @ YY @ Q
    S = np.stack([_symmetric_sqrt(lam_q * A + tp.sig2 * np.eye(A.shape[0])) for lam_q in tp.lam])
    n = A.shape[0]
    iu = np.triu_indices(n)

    def residual(v: np.ndarray) -> np.ndarray:
        Ct = _sym_from_upper(v, n)
        return (Ct - _kernel_map(Ct, S, tp.lam, YY_q, tp.k1, tp.r, tp.delta))[iu]

    v = _newton(residual, _physical_start(residual, n, tp.delta + tp.lam_bar * np.trace(A) / n))
    return Q @ _sym_from_upper(v, n) @ Q.T + complement
