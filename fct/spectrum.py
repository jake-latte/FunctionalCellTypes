"""Tuning-map spectra on the ring: hidden activity -> tuning maps -> power spectra -> summaries.

Pure functions on torch tensors (any floating dtype, any device): no file I/O, no `Config`.

The maths, with h (P, N) the activity of N neurons over P patterns (neuron i at ring site i):
  latent basis   Z = [z0 .. z{K-1}] (P, K) = U S V^T (thin SVD, s descending, 1^T u_k >= 0);
                 u_1 is the mean direction for latents on [0, a], u_2 .. u_K the contrasts
  tuning map     t_{k,i} = u_k^T h_i, t = U^T h (K, N)
  spectrum       d_q^(k) = |t~_k(q)|^2, t~ = fft(t) / sqrt(N) (unitary; q in FFT order
                 0, 1, .., N/2, .., -1); no mean is subtracted
  null mean      (sum_p |h~_p(q)|^2 - sum_k d_q^(k)) / (P - K): the mean spectrum of the P - K
                 directions orthogonal to col(Z), by Parseval over the basis [U, U_perp]
  R(r)           (1/N) sum_q d_q e^{2 pi i q r / N} = ifft(d).real; lag-1 = R(1)/R(0)
  q_1/2          the largest |q| with d_q - d_{N/2} >= (d_0 - d_{N/2}) / 2 (Nyquist as floor)
  angle          theta_i = atan2(t_{3,i}, t_{2,i}) in the orthonormal frame (u_2, u_3)

Every estimator is per network: one trained network (or one ensemble member) gives one sample
of each spectrum and summary, and the spread is taken across members (`member_mean_sem`).
`average_records` and `record_sem` pool records along one chain: a diagnostic of burn-in and
mixing, not an estimator, because one chain is not trusted to visit independent networks.
"""

import math
import re
from collections.abc import Mapping
from typing import NamedTuple

import torch
from torch import Tensor

N_THETA_BINS = 36
# Sokal's window constant: sum the autocorrelation up to the first lag W >= C * tau(W).
SOKAL_C = 5.0
# A window must reach W >= SOKAL_C * tau >= 5 lags (tau ~ 1 for independent records) and stay
# below a quarter of the chain, where the autocorrelation estimate is still reliable.
MAX_WINDOW_FRACTION = 0.25
MIN_SEM_RECORDS = 20
_LATENT_NAME = re.compile(r"z(\d+)")


class LatentBasis(NamedTuple):
    U: Tensor  # (P, K) orthonormal columns, column 0 the top singular direction
    s2: Tensor  # (K,) squared singular values of Z, descending
    Z: Tensor  # (P, K) the latents z0..z{K-1} stacked as columns


class Summary(NamedTuple):
    q_half: int  # half-power |q|
    share0: float  # d_0 / sum_q d_q
    lag1: float  # R(1) / R(0)
    d0: float  # d_0
    d_nyq: float  # d_{N/2}, the floor of q_half
    R: Tensor  # (N,) autocorrelation R(r), r = 0..N-1


# ------------------------------------------------------------------ validation
def _check_float(name: str, x: Tensor, min_ndim: int) -> None:
    if not isinstance(x, Tensor):
        raise TypeError(f"{name} must be a torch tensor, got {type(x).__name__}")
    if not x.is_floating_point():
        raise ValueError(f"{name} must be floating point, got {x.dtype}")
    if x.ndim < min_ndim:
        raise ValueError(
            f"{name} must be at least {min_ndim}-D, got shape {tuple(x.shape)}"
            if min_ndim > 1
            else f"{name} must not be a scalar"
        )
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} must be finite")


def _check_even_last_dim(name: str, d: Tensor) -> int:
    N = d.shape[-1]
    if N < 2 or N % 2:
        raise ValueError(f"{name} needs an even number of sites N >= 2 (q = N/2 exists), got {N}")
    return N


def _check_activity_and_basis(h: Tensor, U: Tensor) -> None:
    """h (..., P, N) and U (P, K), 1 <= K < P, orthonormal columns, same dtype."""
    _check_float("h (..., P, N)", h, 2)
    _check_float("U", U, 2)
    if U.ndim != 2:
        raise ValueError(f"U must be (P, K), got shape {tuple(U.shape)}")
    P, K = U.shape
    if h.shape[-2] != P:
        raise ValueError(f"h has {h.shape[-2]} rows (patterns) but U has {P}")
    if not 1 <= K < P:
        raise ValueError(f"need 1 <= K < P latent directions, got K = {K}, P = {P}")
    if h.dtype != U.dtype:
        raise ValueError(f"h and U must share a dtype, got {h.dtype} and {U.dtype}")
    # sqrt(eps) admits the round-off of an SVD in either precision and rejects any real
    # non-orthonormality, which would break the Parseval identity of null_power.
    tol = math.sqrt(torch.finfo(U.dtype).eps)
    gram_error = (U.T @ U - torch.eye(K, dtype=U.dtype, device=U.device)).abs().max()
    if gram_error > tol:
        raise ValueError(f"U must have orthonormal columns, |U^T U - I|_max = {gram_error:.2e}")


def _latent_index(name: str) -> int:
    match = _LATENT_NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"factor {name!r} is not a latent z<k>; pass only the z0..z{{K-1}}")
    return int(match.group(1))


def _check_angle_inputs(t2: Tensor, t3: Tensor) -> None:
    _check_float("t2", t2, 1)
    _check_float("t3", t3, 1)
    if t2.shape != t3.shape:
        raise ValueError(f"t2 and t3 must have the same shape, got {t2.shape} and {t3.shape}")
    if t2.ndim != 1 or t2.numel() == 0:
        raise ValueError(f"t2 and t3 must be 1-D and non-empty (N,), got {tuple(t2.shape)}")


def _check_records(records: Tensor, burn_in: int) -> Tensor:
    """Validate (R, ...) records and return those after burn-in."""
    _check_float("records", records, 1)
    if not 0 <= burn_in < records.shape[0]:
        raise ValueError(f"burn_in must be in [0, {records.shape[0]}), got {burn_in}")
    return records[burn_in:]


# ------------------------------------------------------------------ basis and maps
def latent_basis(factors: Mapping[str, Tensor]) -> LatentBasis:
    """Orthonormal frame of the latents: Z (P, K) = U S V^T with 1^T u_k >= 0.

    `factors` must be exactly z0..z{K-1} (1-D, length P); they are stacked by index, so z10
    follows z9. Raises for K >= P and for rank-deficient Z (its frame would be arbitrary).
    """
    if not factors:
        raise ValueError("factors holds no latents z0..z{K-1}")
    by_index = {_latent_index(name): z for name, z in factors.items()}
    K = len(by_index)
    if sorted(by_index) != list(range(K)):
        raise ValueError(f"latents must be z0..z{K - 1}, got {sorted(factors)}")
    columns = [by_index[k] for k in range(K)]
    for k, z in enumerate(columns):
        _check_float(f"z{k}", z, 1)
        if z.ndim != 1:
            raise ValueError(f"z{k} must be 1-D (P,), got shape {tuple(z.shape)}")
    if len({z.shape[0] for z in columns}) != 1:
        raise ValueError(f"latents differ in length: {[z.shape[0] for z in columns]}")
    Z = torch.stack(columns, dim=1)
    P = Z.shape[0]
    if K >= P:
        raise ValueError(f"need K < P (null directions exist), got K = {K}, P = {P}")
    U, S, _ = torch.linalg.svd(Z, full_matrices=False)
    if S[-1] <= P * torch.finfo(Z.dtype).eps * S[0]:
        raise ValueError(f"Z is rank-deficient: singular values {S.tolist()}")
    sign = torch.where(U.sum(dim=0) >= 0, 1.0, -1.0).to(U.dtype)
    return LatentBasis(U=U * sign, s2=S.square(), Z=Z)


def tuning_maps(h: Tensor, U: Tensor) -> Tensor:
    """t = U^T h: (..., P, N) activity and (P, K) basis -> (..., K, N) tuning maps."""
    _check_activity_and_basis(h, U)
    return U.T @ h


def power_spectrum(t: Tensor) -> Tensor:
    """|fft(t) / sqrt(N)|^2 over the last axis: (..., K, N) -> (..., K, N), FFT order in q."""
    _check_float("t (..., K, N)", t, 2)
    return torch.fft.fft(t, dim=-1, norm="ortho").abs().square()


def null_power(h: Tensor, U: Tensor) -> Tensor:
    """Mean spectrum of the P - K directions orthogonal to U: (..., P, N) -> (..., N).

    sum_p |h~_p(q)|^2 is the power summed over any orthonormal basis of R^P; removing the K
    latent modes leaves the null directions' total without forming U_perp.
    """
    _check_activity_and_basis(h, U)
    P, K = U.shape
    total = power_spectrum(h).sum(dim=-2)
    latent = power_spectrum(U.T @ h).sum(dim=-2)
    return (total - latent) / (P - K)


def mode_spectra(h: Tensor, U: Tensor) -> Tensor:
    """(..., P, N) -> (..., K + 1, N): rows 0..K-1 the latent modes, row K the null mean."""
    modes = power_spectrum(tuning_maps(h, U))
    return torch.cat([modes, null_power(h, U).unsqueeze(-2)], dim=-2)


# ------------------------------------------------------------------ spectrum summaries
def fold_q(d: Tensor) -> Tensor:
    """Average d_q and d_{-q}: (..., N) in FFT order -> (..., N/2 + 1) at |q| = 0..N/2."""
    _check_float("d", d, 1)
    N = _check_even_last_dim("fold_q", d)
    half = N // 2
    folded = d[..., : half + 1].clone()
    folded[..., 1:half] = 0.5 * (d[..., 1:half] + d[..., half + 1 :].flip(-1))
    return folded


def autocorrelation(d: Tensor) -> Tensor:
    """R(r) = ifft(d).real over the last axis: (..., N) spectrum -> (..., N) at r = 0..N-1."""
    _check_float("d", d, 1)
    return torch.fft.ifft(d, dim=-1).real


def _check_spectrum(d: Tensor) -> int:
    _check_float("d", d, 1)
    if d.ndim != 1:
        raise ValueError(f"d must be 1-D (N,), got shape {tuple(d.shape)}")
    return _check_even_last_dim("the spectrum", d)


def half_power_q(d: Tensor) -> int:
    """Largest |q| with d_q - d_{N/2} >= (d_0 - d_{N/2}) / 2, for d (N,) in FFT order."""
    N = _check_spectrum(d)
    q = torch.fft.fftfreq(N, d=1.0 / N, dtype=d.dtype, device=d.device).abs()
    floor = d[N // 2]
    return int(q[(d - floor) >= 0.5 * (d[0] - floor)].max())


def spectrum_summary(d: Tensor) -> Summary:
    """q_1/2, the q = 0 share, lag-1, d_0, d_{N/2} and R of one spectrum d (N,), FFT order."""
    N = _check_spectrum(d)
    total = d.sum()
    if total <= 0:
        raise ValueError(f"d must have positive total power, got sum {float(total)}")
    R = autocorrelation(d)
    return Summary(
        q_half=half_power_q(d),
        share0=float(d[0] / total),
        lag1=float(R[1] / R[0]),
        d0=float(d[0]),
        d_nyq=float(d[N // 2]),
        R=R,
    )


def gaussian_width(R: Tensor, ell: float) -> float:
    """sqrt(-ell^2 / (2 ln(R(ell)/R(0)))): the width of a Gaussian through R(0) and R(ell).

    ell is a lag in sites, so it must be a whole number in 1..N-1 (a float is accepted).
    """
    _check_float("R", R, 1)
    if R.ndim != 1:
        raise ValueError(f"R must be 1-D (N,), got shape {tuple(R.shape)}")
    lag = int(ell)
    if lag != ell or not 1 <= lag < R.shape[0]:
        raise ValueError(f"ell must be an integer lag in 1..{R.shape[0] - 1}, got {ell}")
    ratio = float(R[lag] / R[0])
    if not (R[0] > 0 and 0 < ratio < 1):
        raise ValueError(
            f"need R(0) > 0 and 0 < R(ell)/R(0) < 1, got R(0) = {float(R[0])}, "
            f"R(ell)/R(0) = {ratio}"
        )
    return math.sqrt(-(lag**2) / (2 * math.log(ratio)))


# ------------------------------------------------------------------ angle statistic
def _theta(t2: Tensor, t3: Tensor) -> Tensor:
    """atan2(t3, t2) on (-pi, pi]: atan2 can return -pi, the same angle as pi."""
    theta = torch.atan2(t3, t2)
    return torch.where(theta == -math.pi, torch.full_like(theta, math.pi), theta)


def contrast_angle_statistic(t2: Tensor, t3: Tensor) -> float:
    """<cos 4 theta_i> over neurons, theta_i = atan2(t3_i, t2_i); t2, t3 (N,) of one network.

    0 for a rotation-invariant (t2, t3), positive for tuning aligned with the axes, negative
    along the diagonals. Meaningful only in an orthonormal frame: an oblique pair squeezes an
    isotropic cloud along its bisector and biases the value negative.
    """
    _check_angle_inputs(t2, t3)
    return float(torch.cos(4 * _theta(t2, t3)).mean())


def theta_histogram(t2: Tensor, t3: Tensor) -> Tensor:
    """(36,) int64 counts of theta_i on the fixed bins (-pi + j w, -pi + (j+1) w], w = 2pi/36.

    Fixed bins make histograms of different records and networks summable.
    """
    _check_angle_inputs(t2, t3)
    upper = torch.linspace(-math.pi, math.pi, N_THETA_BINS + 1, dtype=t2.dtype, device=t2.device)
    upper = upper[1:]
    upper[-1] = math.pi  # theta = pi must land in the last bin, whatever linspace rounds to
    index = torch.bucketize(_theta(t2, t3), upper, right=False)
    return torch.bincount(index, minlength=N_THETA_BINS)


# ------------------------------------------------------------------ records and members
def average_records(records: Tensor, burn_in: int) -> Tensor:
    """Mean over the records of one chain after `burn_in`: (R, ...) -> (...). A diagnostic."""
    return _check_records(records, burn_in).mean(dim=0)


def _autocorrelation_over_records(x: Tensor) -> tuple[Tensor, Tensor]:
    """(C(0) (...), rho (n, ...)): biased autocovariance at lag 0 and autocorrelation by FFT.

    Zero-padding to 2n turns the FFT's circular correlation into the linear one. An element
    with C(0) = 0 (constant records) gets rho = 0 at every lag > 0.
    """
    n = x.shape[0]
    centred = x - x.mean(dim=0)
    spectrum = torch.fft.rfft(centred, n=2 * n, dim=0).abs().square()
    acov = torch.fft.irfft(spectrum, n=2 * n, dim=0)[:n] / n
    c0 = acov[0]
    constant = c0 <= 0
    rho = torch.where(constant, torch.zeros_like(acov), acov / torch.where(constant, 1.0, c0))
    rho[0] = 1.0
    return c0.clamp(min=0), rho


def integrated_autocorr_time(records: Tensor, burn_in: int) -> Tensor:
    """tau = 1 + 2 sum_{k=1}^{W} rho_k per element, (R, ...) -> (...), Sokal's window.

    W is the first lag with W >= 5 tau(W) (Sokal 1997), searched up to a quarter of the
    chain; raises if some element's window is not reached there (the chain is too short for
    its correlation time). Constant elements get tau = 1.
    """
    x = _check_records(records, burn_in)
    n = x.shape[0]
    if n < MIN_SEM_RECORDS:
        raise ValueError(f"need at least {MIN_SEM_RECORDS} records after burn-in, got {n}")
    _, rho = _autocorrelation_over_records(x)
    max_window = int(MAX_WINDOW_FRACTION * n)
    tau_of_window = 1 + 2 * rho[1 : max_window + 1].cumsum(dim=0)  # (max_window, ...)
    windows = torch.arange(1, max_window + 1, dtype=x.dtype, device=x.device)
    windows = windows.reshape((-1,) + (1,) * (x.ndim - 1))
    reached = windows >= SOKAL_C * tau_of_window
    if not reached.any(dim=0).all():
        raise ValueError(
            f"chain too short: the autocorrelation window is not reached within {max_window} "
            f"of {n} records for {int((~reached.any(dim=0)).sum())} element(s)"
        )
    first = reached.to(torch.int8).argmax(dim=0, keepdim=True)
    tau = tau_of_window.gather(0, first).squeeze(0)
    if (tau <= 0).any():
        raise ValueError(f"non-positive integrated autocorrelation time: {tau.min():.3g}")
    return tau


def record_sem(records: Tensor, burn_in: int) -> Tensor:
    """Standard error of the chain mean after burn-in, (R, ...) -> (...): sqrt(C(0) tau / n).

    n / tau is the effective number of independent records, C(0) the variance of the records
    and tau their integrated autocorrelation time (`integrated_autocorr_time`). A diagnostic:
    comparing it across burn-in values, or with the spread across members, shows whether the
    chain has equilibrated; it is not the error of a per-network estimator.
    """
    tau = integrated_autocorr_time(records, burn_in)
    x = records[burn_in:]
    c0, _ = _autocorrelation_over_records(x)
    return (c0 * tau / x.shape[0]).sqrt()


def member_mean_sem(values: Tensor) -> tuple[Tensor, Tensor]:
    """Mean and standard error across B independent networks: (B, ...) -> ((...), (...)).

    The per-network estimators above give one value per member; members are independent, so
    the standard error is std (ddof = 1) / sqrt(B).
    """
    _check_float("values", values, 1)
    B = values.shape[0]
    if B < 2:
        raise ValueError(f"need at least 2 members for a spread, got {B}")
    return values.mean(dim=0), values.std(dim=0) / math.sqrt(B)
