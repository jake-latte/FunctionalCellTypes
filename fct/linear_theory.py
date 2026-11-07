"""The linear theory of one hidden layer on a ring, as pure functions of a `Config` and a `Dataset`.

The theory (latex/notes/linear_solution_limits.tex, eq:linear-soln; linear_without_ansatz.tex,
eq:master) describes the stationary law exp(-beta L) of SGLD on a linear one-hidden-layer network:
N units on a ring with the spatial prior of hidden spectrum lam_q, input prior k0 I, readout prior
k1 I, input noise sigma^2, rate penalty (r/2) sum phi^2 in beta L, and ridge delta = P n / (beta k1).
Per data mode with input and target weights sX2 = k0 s_X^2 / m, sY2 = s_Y^2 / k1 the hidden kernel
eigenvalue c and the power spectrum d_q over spatial frequencies q solve

    alpha_q = lam_q sX2 + sig2,   pi_q = alpha_q / (1 + r alpha_q),   h_q = lam_q pi_q,
    1 - delta / c = (c / N) sum_q h_q / (c^2 - sY2 h_q)   on the branch c > sqrt(sY2 max_q h_q),
    d_q = pi_q c^2 / (c^2 - sY2 h_q),

and without the shared-basis ansatz the P x P kernel C~ solves the master equation

    C~ = delta I + (1/N) sum_q lam_q Sigma_q,
    Sigma_q = [(lam_q A + sig2 I)^{-1} + r I - lam_q B]^{-1},   A = (k0/m) X X^T,
    B = C~^{-1} Y Y^T C~^{-1} / k1.

The per-mode solution needs an orthonormal teacher: X = Z W_X^T, Y = Z W_Y^T with W_X^T W_X =
W_Y^T W_Y = I, so that X X^T = Y Y^T = Z Z^T and the data modes are the singular vectors of Z.
For any other teacher the Gram matrices of X and Y are Z (W^T W) Z^T, not Z Z^T, and sX2, sY2 from
the singular values of Z are wrong in scale (by about m / K for a Gaussian teacher), so
`theory_parameters` and `predicted_spectra` refuse a dataset whose teacher maps are not
orthonormal. `ansatz_free_kernel` uses the actual Gram matrices and holds for any teacher.

sigma -> 0 is a first-class case: the input noise is a formal regulariser (the limits note's
Section sec:sigma0), so `sig2` defaults to 0 everywhere and nothing divides by it. At sig2 = 0 a
null mode (sX2 = sY2 = 0) and every frequency with lam_q = 0 get d_q = 0 exactly, and Sigma_q is
evaluated as S (I + S T S)^{-1} S with S = (lam_q A + sig2 I)^{1/2}, T = r I - lam_q B, which is
finite when lam_q A is singular.

lam_q is by default the spectrum of the prior the run trains with. Under the penalty that is
N(0, M^{-1}) with `post.M` the hidden population's penalty matrix: the eigenvalues of M^{-1} are
max(lam_q, floor), lam_q those of the kernel K and floor = min_rel_eig times K's top eigenvalue
(`fct.spatial.penalty_matrix`, after any lambda_max rescaling of K). They are computed from K and
the floor, not by inverting M, which would lose log10(1 / min_rel_eig) digits on the coarse
modes. Under `spatial.parametrisation = "whitened"` the run trains with N(0, K) itself, so the
default is K's own spectrum, which `exact_kernel=True` also gives on the penalty path (the limits
note's numbers). The run must be float64: a float32 run
casts M to float32 and trains with a prior whose coarse lam_q are off by 1-2%.

The latent frame is `fct.spectrum.latent_basis`, the one the measurement uses: U (column k - 1 is
u_k, signed so that 1^T u_k >= 0) and s2 descending. It needs latents on [0, a] (u_1 the mean
direction), so the per-mode theory refuses symmetric latents and rank-deficient Z, and its
summaries use `fct.spectrum`'s `half_power_q` and `autocorrelation` (unitary DFT, FFT order).

Dictionary (the plan's Section sec:dictionary): N = net.N; m = data.D_in; n = D_out of the task;
P = data.P; K latents = data.n_latents, Z from dataset.factors z0..z{K-1}; the hidden prior is
`build_spaces`' hidden population on a d = 1 grid, lam_q its Fourier eigenvalues (DFT order, q = 0
the coarsest scale, q = N/2 the finest); k0, k1 are the input and output kernel amplitudes over
train.weight_lambda (Cov(W) = K / weight_lambda for every layer); beta = train.beta; r is
`theory_rate(cfg)`.

Everything is float64 numpy; torch tensors are accepted anywhere an array is. No file I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import NamedTuple

import numpy as np
import torch

from fct.config import Config
from fct.data import task_dims
from fct.spatial import LayerSpace, build_spaces
from fct.spectrum import autocorrelation, half_power_q, latent_basis
from fct.tasks import Dataset

ArrayOrTensor = np.ndarray | torch.Tensor

BISECT_STEPS = 200  # halvings of the root bracket: far below double precision at any scale
# Eigenvalues of a kernel below -NEGATIVE_EIG_TOL * lam_max are an error; above, round-off set to 0.
NEGATIVE_EIG_TOL = 1e-12
CIRCULANT_TOL = 1e-10  # relative agreement required between the FFT of row 0 and eigvalsh
SYMMETRY_TOL = 1e-12  # relative: max |K - K^T| allowed of a kernel
SCALAR_KERNEL_TOL = 1e-12  # relative: an input/output kernel must be amplitude * I to this
ORTHONORMAL_TOL = 1e-10  # max |W^T W - I| of an orthonormal teacher map
TEACHER_TOL = 1e-10  # max |x - Z W_X^T| (and for y) relative to max(1, max |x|)
# The ansatz-free solver: Newton on the upper triangle of C~ with a finite-difference Jacobian.
NEWTON_STEPS = 100
NEWTON_TOL = 1e-13  # on max |F(C~) - C~| relative to max(1, |C~|)
FD_EPS = 1e-7  # relative finite-difference step
MAX_HALVINGS = 60  # of a Newton step before the line search is declared failed
MAX_DOUBLINGS = 200  # of the starting c before the start is declared unreachable
SUBSPACE_TOL = 1e-12  # relative: eigenvalues of X X^T + Y Y^T below this span nothing


def _array(x: ArrayOrTensor, name: str = "input") -> np.ndarray:
    """A finite float64 numpy copy of an array or a (possibly CUDA) tensor; NaN or inf raise."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite; it has {int((~np.isfinite(a)).sum())} NaN/inf")
    return a


def _check_nonnegative(**values: float) -> None:
    """Raise naming the first value that is not >= 0 (NaN included)."""
    for name, value in values.items():
        if not value >= 0:
            raise ValueError(f"{name} must be >= 0, got {value}")


# --------------------------------------------------------------------------- config


def _check_sgld(cfg: Config) -> None:
    """The theory is the stationary law exp(-beta L) of SGLD; Adam finds a MAP point instead."""
    if cfg.train.optimizer != "sgld":
        raise ValueError(
            f"the linear theory describes SGLD samples of exp(-beta L); train.optimizer is "
            f"{cfg.train.optimizer!r}, which gives a MAP point, not samples"
        )


def theory_rate(cfg: Config) -> float:
    """The rate-penalty coefficient r of the theory, (r/2) sum phi^2 in beta L. The loss carries
    rate_lambda * sum phi^2 and the SGLD law is exp(-beta L), so r = 2 beta rate_lambda under
    today's loss convention."""
    _check_sgld(cfg)
    return 2.0 * cfg.train.beta * cfg.train.rate_lambda


def _ring_spaces(cfg: Config) -> list[LayerSpace]:
    """`build_spaces` of a one-hidden-layer network on a d = 1 grid ring; raises otherwise."""
    if cfg.net.L != 1:
        raise ValueError(f"the linear theory is for one hidden layer; net.L = {cfg.net.L}")
    if cfg.spatial is None:
        raise ValueError("the linear theory needs a spatial prior; cfg.spatial is None")
    if cfg.spatial.d != 1:
        raise ValueError(f"the linear theory is for a ring; spatial.d = {cfg.spatial.d}")
    if cfg.spatial.hidden.embedding.kind != "grid":
        raise ValueError(
            "the linear theory needs evenly spaced hidden units; "
            f"spatial.hidden.embedding.kind = {cfg.spatial.hidden.embedding.kind!r}, not 'grid'"
        )
    m, n_out = task_dims(cfg.data)
    return build_spaces(cfg.spatial, m=m, N=cfg.net.N, L=1, n_out=n_out)


def _circulant_spectrum(name: str, A: ArrayOrTensor) -> np.ndarray:
    """The DFT-order spectrum of a symmetric circulant matrix: the FFT of its first row, checked
    against `eigvalsh` so that a matrix that is not translation invariant fails loudly."""
    A = _array(A, name)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{name} must be square, got shape {A.shape}")
    scale = max(np.abs(A).max(), np.finfo(float).tiny)
    asymmetry = np.abs(A - A.T).max()
    if asymmetry > SYMMETRY_TOL * scale:
        raise ValueError(f"{name} is not symmetric: max |{name} - {name}^T| = {asymmetry:.3g}")
    lam = np.fft.fft(A[0]).real
    discrepancy = np.abs(np.sort(lam) - np.linalg.eigvalsh(A)).max()
    if discrepancy > CIRCULANT_TOL * max(np.abs(lam).max(), np.finfo(float).tiny):
        raise ValueError(
            f"{name} is not circulant: the FFT of its first row misses its spectrum by "
            f"{discrepancy:.3g}"
        )
    # A symmetric circulant has lam_q = lam_{-q}; averaging with the reflection removes the
    # round-off difference, so every prediction is exactly even in q.
    return 0.5 * (lam + np.roll(lam[::-1], 1))


def ring_spectrum(K: ArrayOrTensor) -> np.ndarray:
    """lam_q, q = 0..N-1 in DFT order, of a symmetric circulant kernel. Round-off negatives are
    set to 0; a clearly negative eigenvalue is an error."""
    lam = _circulant_spectrum("K", K)
    scale = max(np.abs(lam).max(), np.finfo(float).tiny)
    if lam.min() < -NEGATIVE_EIG_TOL * scale:
        raise ValueError(f"K has a negative eigenvalue {lam.min():.3g}; it is not a covariance")
    # -1e-16 lam_0 at the Nyquist tail of a wrapped Gaussian is FFT round-off, not structure.
    return np.maximum(lam, 0.0)


def _scalar_kernel_amplitude(name: str, K: np.ndarray) -> float:
    """The a of K = a I; raises naming the population if K is not a multiple of the identity."""
    a = K[0, 0]
    deviation = np.abs(K - a * np.eye(K.shape[0])).max()
    if deviation > SCALAR_KERNEL_TOL * abs(a):
        raise ValueError(
            f"the theory needs the {name} kernel to be a multiple of the identity "
            f"(spatial.{name}.kernel.length_scale = 0); it deviates by {deviation:.3g}"
        )
    return float(a)


# --------------------------------------------------------------------------- the latent task


def latent_matrix(factors: dict[str, torch.Tensor] | None, n_latents: int) -> np.ndarray:
    """Z (P, K): the factors z0..z{K-1} of the linear teacher as columns; any other z{i} factor
    means the dataset was drawn with a different number of latents and is an error."""
    names = [f"z{i}" for i in range(n_latents)]
    present = set(factors or {})
    missing = [n for n in names if n not in present]
    if missing:
        raise ValueError(f"dataset.factors lacks the latent(s) {missing}")
    extra = sorted(n for n in present - set(names) if n[0] == "z" and n[1:].isdigit())
    if extra:
        raise ValueError(
            f"dataset.factors has latent(s) {extra} beyond data.n_latents = {n_latents}"
        )
    return np.stack([_array(factors[n], f"factor {n}") for n in names], axis=1)


def uniform_latent_s2(P: int, K: int, a: float) -> np.ndarray:
    """The P >> K singular values squared of Z ~ U[0, a]^(P x K) (limits note, prop:svd):
    s_1^2 = P a^2 (3K + 1) / 12 on the mean direction, s_k^2 = P a^2 / 12 on the K - 1 contrasts."""
    return np.array([P * a**2 * (3 * K + 1) / 12] + [P * a**2 / 12] * (K - 1))


def _check_orthonormal_teacher(dataset: Dataset) -> None:
    """Raise unless dataset.extras holds W_X and W_Y with orthonormal columns."""
    extras = dataset.extras or {}
    missing = [k for k in ("W_X", "W_Y") if k not in extras]
    if missing:
        raise ValueError(f"dataset.extras lacks the teacher map(s) {missing}")
    for key in ("W_X", "W_Y"):
        W = _array(extras[key], key)
        deviation = np.abs(W.T @ W - np.eye(W.shape[1])).max()
        if deviation > ORTHONORMAL_TOL:
            raise ValueError(
                f'the per-mode theory needs an orthonormal teacher (data.teacher="orthonormal"): '
                f"{key}^T {key} deviates by {deviation:.3g} from the identity; "
                "ansatz_free_kernel holds for any teacher"
            )


def _check_generated_by_teacher(dataset: Dataset, Z: np.ndarray) -> None:
    """Raise unless x = Z W_X^T and y = Z W_Y^T for the W in dataset.extras, to TEACHER_TOL."""
    for name, data, key in (("x", dataset.x, "W_X"), ("y", dataset.y, "W_Y")):
        values = _array(data, f"dataset.{name}")
        error = np.abs(values - Z @ _array(dataset.extras[key], key).T).max()
        if error > TEACHER_TOL * max(1.0, np.abs(values).max()):
            raise ValueError(
                f"dataset.{name} is not Z {key}^T for the latents and the teacher map in "
                f"dataset.extras: max deviation {error:.3g}"
            )


# --------------------------------------------------------------------------- parameters
# Not comparable or hashable by value (eq=False): they hold numpy arrays, whose == is elementwise,
# so a generated __eq__ would raise; they compare and hash by identity.


@dataclass(frozen=True, eq=False)
class NetworkParameters:
    """The network side of the dictionary, which holds for any teacher (see the module docstring).
    Compared and hashed by identity."""

    N: int
    m: int
    n: int
    P: int
    lam: np.ndarray  # (N,) DFT order
    lam0: float  # max_q lam_q
    lam_bar: float  # mean_q lam_q
    k0: float
    k1: float
    beta: float
    delta: float  # P n / (beta k1)
    sig2: float
    r: float


@dataclass(frozen=True, eq=False)
class TheoryParameters(NetworkParameters):
    """`NetworkParameters` plus the per-mode weights of an orthonormal teacher. Compared and
    hashed by identity."""

    n_latents: int
    U: np.ndarray  # (P, K) left singular vectors of Z
    s2: np.ndarray  # (K,) squared singular values of Z
    sx2: np.ndarray  # (K,) k0 s2 / m
    sy2: np.ndarray  # (K,) s2 / k1


def network_parameters(
    cfg: Config, dataset: Dataset, sig2: float = 0.0, exact_kernel: bool = False
) -> NetworkParameters:
    """The network side of the dictionary for a config and its dataset, for any teacher; `sig2`
    is explicit because no config field holds the input noise (sigma -> 0 is the default)."""
    if cfg.net.phi != "id":
        raise ValueError(f"the linear theory needs net.phi = 'id', got {cfg.net.phi!r}")
    if cfg.data.task != "linear_teacher":
        raise ValueError(f"the theory is for the linear_teacher task, got {cfg.data.task!r}")
    _check_sgld(cfg)
    if cfg.dtype != "float64":
        raise ValueError(
            f"the linear theory needs a float64 run; dtype is {cfg.dtype!r}, under which the "
            "trained prior's coarse lam_q are off by 1-2%"
        )
    _check_nonnegative(sig2=sig2)
    spaces = _ring_spaces(cfg)
    hidden = spaces[0].post
    lam = ring_spectrum(hidden.K)
    if not (exact_kernel or cfg.is_whitened):
        # The prior the penalty trains with, N(0, M^{-1}): `penalty_matrix` raises K's eigenvalues
        # to report.floor = min_rel_eig * (top eigenvalue of K as built), and M inverts the result.
        lam = np.maximum(lam, hidden.report.floor)
    # Cov(vec W) = (K_post (x) K_pre) / weight_lambda, so the scalar factors carry 1 / weight_lambda
    # and the hidden kernel enters as built.
    k0 = _scalar_kernel_amplitude("input", _array(spaces[0].pre.K)) / cfg.train.weight_lambda
    k1 = _scalar_kernel_amplitude("output", _array(spaces[1].post.K)) / cfg.train.weight_lambda
    m, n = task_dims(cfg.data)
    P = cfg.data.P
    for name, data, cols in (("x", dataset.x, m), ("y", dataset.y, n)):
        if tuple(data.shape) != (P, cols):
            raise ValueError(
                f"dataset.{name} has shape {tuple(data.shape)}, expected {(P, cols)} (data.P = {P})"
            )
        _array(data, f"dataset.{name}")
    beta = cfg.train.beta
    return NetworkParameters(
        N=cfg.net.N,
        m=m,
        n=n,
        P=P,
        lam=lam,
        lam0=float(lam.max()),
        lam_bar=float(lam.mean()),
        k0=k0,
        k1=k1,
        beta=beta,
        delta=P * n / (beta * k1),
        sig2=float(sig2),
        r=theory_rate(cfg),
    )


def theory_parameters(
    cfg: Config, dataset: Dataset, sig2: float = 0.0, exact_kernel: bool = False
) -> TheoryParameters:
    """`network_parameters` plus the per-mode weights from the SVD of Z; raises unless the
    dataset's teacher is orthonormal and generated x and y from the latents, the case in which
    those weights are the theory, and unless the latents have a mean direction
    (`fct.spectrum.latent_basis`: latents on [0, a], Z of full rank)."""
    net = network_parameters(cfg, dataset, sig2, exact_kernel)
    _check_orthonormal_teacher(dataset)
    Z = latent_matrix(dataset.factors, cfg.data.n_latents)
    if Z.shape[0] != net.P:
        raise ValueError(f"the latents have {Z.shape[0]} rows, expected data.P = {net.P}")
    _check_generated_by_teacher(dataset, Z)
    frame = latent_basis({f"z{i}": torch.from_numpy(Z[:, i].copy()) for i in range(Z.shape[1])})
    U, s2 = frame.U.numpy(), frame.s2.numpy()
    return TheoryParameters(
        **{f.name: getattr(net, f.name) for f in fields(NetworkParameters)},
        n_latents=cfg.data.n_latents,
        U=U,
        s2=s2,
        sx2=net.k0 * s2 / net.m,
        sy2=s2 / net.k1,
    )


# --------------------------------------------------------------------------- the per-mode solution


def noise_floor(sig2: float, r: float) -> float:
    """sigma_0^2 = sig2 / (1 + r sig2): the flat spectrum of a null mode (lem:floor); 0 at sig2 = 0."""
    return sig2 / (1.0 + r * sig2)


def prior_shrunk_variance(lam: ArrayOrTensor, sx2: float, sig2: float, r: float) -> np.ndarray:
    """pi_q = alpha_q / (1 + r alpha_q), alpha_q = lam_q sx2 + sig2 (eq:factor)."""
    alpha = _array(lam, "lam") * sx2 + sig2
    return alpha / (1.0 + r * alpha)


def pi_lam(lam: ArrayOrTensor, sx2: float, sig2: float, r: float) -> np.ndarray:
    """h_q = lam_q pi_q (eq:hq), the per-frequency weight of the self-consistency sum."""
    return _array(lam, "lam") * prior_shrunk_variance(lam, sx2, sig2, r)


def _residual(c: float, h: np.ndarray, sy2: float, delta: float) -> float:
    """1 - delta/c - (c/N) sum_q h_q / (c^2 - sY2 h_q): increasing in c above the pole."""
    with np.errstate(divide="ignore"):  # c^2 = sY2 h_q at the pole itself: -inf, still < 0
        return 1.0 - delta / c - np.mean(c * h / (c**2 - sy2 * h))


def physical_root(h: ArrayOrTensor, sy2: float, delta: float) -> float:
    """The root c of eq:linear-soln on the physical branch c > sqrt(sY2 max_q h_q), by bisection.

    The residual rises from -inf at the pole to 1, so the root is unique. Lower bracket: the next
    float above the pole, or c = delta / 2 when the pole is below that (the term -delta / c alone
    gives -1; no target puts the pole at 0, where c -> 0 would underflow). At the next float
    above the pole the residual is negative unless the root itself lies within one ulp of the
    pole: the pole's term, about sqrt(h_max / sY2) / (2 N pole eps) there, then no longer
    outweighs the 1, which happens once sqrt(h_max / sY2) is below about 2 N eps pole. Such a
    root cannot be resolved in float64 and is an error. Upper bracket: for c^2 >= 2 sY2 max h
    every denominator is at least c^2 / 2, so the sum is at most 2 mean(h) / c and the residual is
    at least 1 - (delta + 2 mean h) / c, which is 1/2 at twice the larger of the two bounds; a
    bracket whose c^2 overflows float64 is an error.
    """
    h = _array(h, "h")
    _check_nonnegative(sy2=sy2)
    if not delta > 0:
        raise ValueError(f"delta must be > 0, got {delta}")
    pole = np.sqrt(sy2 * h.max())
    at_pole = pole >= delta / 2
    lo = np.nextafter(pole, np.inf) if at_pole else delta / 2
    hi = 2.0 * max(np.sqrt(2 * sy2 * h.max()), delta + 2 * h.mean())
    if not hi < np.sqrt(np.finfo(float).max):  # compared without squaring, which would overflow
        raise ValueError(
            f"the root bracket overflows float64: c up to {hi:.3g}, c^2 > "
            f"{np.finfo(float).max:.3g} (delta = {delta:.3g}, max h = {h.max():.3g})"
        )
    r_lo, r_hi = _residual(lo, h, sy2, delta), _residual(hi, h, sy2, delta)
    if at_pole and r_lo >= 0:
        raise ValueError(
            f"the root lies within one ulp of the pole c = {pole:.17g} (residual {r_lo:.3g} at "
            "the next float): it cannot be resolved in float64"
        )
    if not (r_lo < 0 < r_hi):
        raise ValueError(
            f"root not bracketed: residual {r_lo:.3g} at c = {lo:.6g}, {r_hi:.3g} at c = {hi:.6g}"
        )
    for _ in range(BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _residual(mid, h, sy2, delta) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


class Mode(NamedTuple):
    c: float  # the kernel eigenvalue
    d: np.ndarray  # (N,) the power spectrum
    tau: float  # the target strength


def solve_mode(
    lam: ArrayOrTensor, sx2: float, sy2: float, sig2: float, r: float, delta: float
) -> Mode:
    """(c, d_q, tau) of one data mode (eq:linear-soln, eq:knobs), with `r` the rate coefficient.

    d_q = pi_q * c^2 / (c^2 - sY2 h_q) rather than h_q / lam_q: the quotient is 0/0 where
    lam_q = 0, while pi_q is 0 there at sig2 = 0 and sigma_0^2 otherwise. A null mode
    (sx2 = sy2 = 0) at sig2 = 0 has h = 0, so c = delta and d = 0 exactly. tau = sY2 max h / c^2
    in [0, 1) says how far the mode is from the pole.
    """
    lam = _array(lam, "lam")
    _check_nonnegative(sx2=sx2, sy2=sy2, sig2=sig2, r=r)
    if lam.min() < 0:
        raise ValueError(f"lam has a negative entry {lam.min():.3g}; pass it through ring_spectrum")
    pi = prior_shrunk_variance(lam, sx2, sig2, r)
    h = lam * pi
    c = physical_root(h, sy2, delta)
    return Mode(c, pi * c**2 / (c**2 - sy2 * h), _tau(h, sy2, c))


def _tau(h: np.ndarray, sy2: float, c: float) -> float:
    """tau = sY2 max_q h_q / c^2 (eq:knobs)."""
    return sy2 * h.max() / c**2


def per_mode_spectrum(
    lam: ArrayOrTensor, sx2: float, sy2: float, sig2: float, r: float, delta: float
) -> tuple[float, np.ndarray]:
    """(c, d_q) of one data mode: `solve_mode` without the target strength."""
    c, d, _ = solve_mode(lam, sx2, sy2, sig2, r, delta)
    return c, d


def target_strength(
    lam: ArrayOrTensor, sx2: float, sy2: float, sig2: float, r: float, c: float
) -> float:
    """tau = sY2 max_q h_q / c^2 in [0, 1): how far the mode is from the pole (eq:knobs)."""
    return _tau(pi_lam(lam, sx2, sig2, r), sy2, c)


@dataclass(frozen=True)
class Prediction:
    """The per-mode theory of every latent mode: `d[k]` is the power spectrum of the tuning map
    on `U[:, k]` (DFT order), `c[k]` its kernel eigenvalue, `tau[k]` its target strength; the
    P - K null directions are flat at `null_floor` = sigma_0^2. `lam`, `U` and `s2` are read
    from `params`."""

    d: np.ndarray  # (K, N)
    null_floor: float
    c: np.ndarray  # (K,)
    tau: np.ndarray  # (K,)
    params: TheoryParameters

    @property
    def lam(self) -> np.ndarray:
        return self.params.lam

    @property
    def U(self) -> np.ndarray:
        return self.params.U

    @property
    def s2(self) -> np.ndarray:
        return self.params.s2


def predicted_spectra(
    cfg: Config, dataset: Dataset, sig2: float = 0.0, exact_kernel: bool = False
) -> Prediction:
    """`solve_mode` of every latent mode of the config's task (the theory's prediction for the
    measured tuning-map spectra); sigma^2 = 0 unless `sig2` is given. Needs an orthonormal
    teacher (`theory_parameters`)."""
    tp = theory_parameters(cfg, dataset, sig2, exact_kernel)
    modes = [solve_mode(tp.lam, a, b, tp.sig2, tp.r, tp.delta) for a, b in zip(tp.sx2, tp.sy2)]
    return Prediction(
        d=np.stack([mode.d for mode in modes]),
        null_floor=noise_floor(tp.sig2, tp.r),
        c=np.array([mode.c for mode in modes]),
        tau=np.array([mode.tau for mode in modes]),
        params=tp,
    )


# --------------------------------------------------------------------------- spectrum summaries


class PredictedSummary(NamedTuple):
    q_half: int  # fct.spectrum.half_power_q: defined because a prediction is a mean spectrum
    share0: float  # d_0 / sum_q d_q
    lag1: float  # R(1) / R(0), R = fct.spectrum.autocorrelation(d)


def predicted_summary(d: ArrayOrTensor) -> PredictedSummary:
    """(q_half, share0, lag1) of one predicted power spectrum in DFT order (even N), with the
    measurement's `half_power_q` and `autocorrelation`; an identically zero spectrum (a null mode
    at sigma = 0) has no summary and is an error."""
    d = _array(d, "d")
    if d.ndim != 1 or d.size < 2:
        raise ValueError(f"d must be a 1-d spectrum with at least 2 entries, got shape {d.shape}")
    if d.min() < 0:
        raise ValueError(f"a power spectrum cannot be negative; min d = {d.min():.3g}")
    if not (d > 0).any():
        raise ValueError("the spectrum is identically zero: no shape to summarise")
    d_t = torch.from_numpy(d)
    R = autocorrelation(d_t)
    return PredictedSummary(int(half_power_q(d_t)), float(d[0] / d.sum()), float(R[1] / R[0]))


# --------------------------------------------------------------------------- the ansatz-free kernel


class NotPhysicalError(ValueError):
    """Some per-frequency precision is not positive definite: no Gaussian exists for this C~."""


def _psd_eigh(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (round-off negatives clamped to 0) and eigenvectors of a symmetric PSD matrix."""
    w, V = np.linalg.eigh(A)
    if w.min() < -NEGATIVE_EIG_TOL * max(abs(w).max(), np.finfo(float).tiny):
        raise ValueError(f"matrix is not positive semi-definite (min eigenvalue {w.min():.3g})")
    return np.maximum(w, 0.0), V


def _per_q_roots(A: np.ndarray, lam: np.ndarray, sig2: float) -> np.ndarray:
    """S_q = (lam_q A + sig2 I)^{1/2}, shape (N, L, L), from one eigendecomposition of A:
    every lam_q A + sig2 I shares A's eigenvectors, with eigenvalues lam_q w + sig2."""
    w, V = _psd_eigh(A)
    roots = np.sqrt(lam[:, None] * w[None, :] + sig2)  # (N, L)
    return np.einsum("ik,qk,jk->qij", V, roots, V)


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
    """The symmetric n x n matrix whose upper triangle (row-major, `np.triu_indices`) is v."""
    C = np.zeros((n, n))
    C[np.triu_indices(n)] = v
    return C + np.triu(C, 1).T


def _physical_start(residual, n: int, c: float) -> np.ndarray:
    """c I on the admissible set: B shrinks like c^{-2}, so doubling c reaches it."""
    iu = np.triu_indices(n)
    c_start = c
    for _ in range(MAX_DOUBLINGS):
        v = (c * np.eye(n))[iu]
        try:
            residual(v)
            return v
        except NotPhysicalError:
            c *= 2
    raise NotPhysicalError(
        f"no admissible starting point c I found by doubling c from {c_start:.3g} to {c:.3g}"
    )


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
                f"the Newton line search found no admissible step that reduces the residual "
                f"{size:.3g} at max |C~| = {np.abs(v).max():.3g}"
            )
        v = v + step
    raise RuntimeError(f"Newton did not converge in {NEWTON_STEPS} steps: residual {size:.3g}")


def ansatz_free_kernel(
    cfg: Config, dataset: Dataset, sig2: float = 0.0, exact_kernel: bool = False
) -> np.ndarray:
    """C~ (P, P), the fixed point of the master equation eq:master for the config's network and
    the dataset's actual Gram matrices (no shared-basis ansatz), at input noise `sig2`. Holds for
    any teacher, orthonormal or not.

    Solved on the subspace spanned by the data (K-dimensional for the latent task) and set to the
    prior value c_0 = delta + sigma_0^2 lam_bar on its complement, which is exact.
    """
    net = network_parameters(cfg, dataset, sig2, exact_kernel)
    X, Y = _array(dataset.x, "dataset.x"), _array(dataset.y, "dataset.y")
    XX, YY = X @ X.T, Y @ Y.T
    Q = _data_subspace(XX, YY)
    c0 = net.delta + noise_floor(net.sig2, net.r) * net.lam_bar
    complement = c0 * (np.eye(net.P) - Q @ Q.T)
    if Q.shape[1] == 0:
        return complement
    A = net.k0 / net.m * (Q.T @ XX @ Q)
    YY_q = Q.T @ YY @ Q
    S = _per_q_roots(A, net.lam, net.sig2)
    n = A.shape[0]
    iu = np.triu_indices(n)

    def residual(v: np.ndarray) -> np.ndarray:
        Ct = _sym_from_upper(v, n)
        return (Ct - _kernel_map(Ct, S, net.lam, YY_q, net.k1, net.r, net.delta))[iu]

    v = _newton(residual, _physical_start(residual, n, net.delta + net.lam_bar * np.trace(A) / n))
    return Q @ _sym_from_upper(v, n) @ Q.T + complement
