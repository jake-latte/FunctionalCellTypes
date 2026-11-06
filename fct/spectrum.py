"""Tuning-map spectra on the ring: hidden activity -> tuning maps -> power spectra -> summaries.

Pure functions on torch tensors (any floating dtype, any device; the dtype is preserved): no
file I/O, no `Config`. Leading batch dims `...` (records, members) pass through every function
except `latent_basis`; a summary of a single map is returned as a Python number.

The maths, with h (P, N) the activity of N neurons over P patterns (neuron i at ring site i):
  latent basis   Z = [z0 .. z{K-1}] (P, K) = U S V^T (thin SVD, s descending, 1^T u_k >= 0);
                 u_1 is the mean direction (latents on [0, a]), u_2 .. u_K the contrasts.
                 u_k is column k-1 of U and row k-1 of t, so theta uses rows 1 and 2 of t
  tuning map     t_{k,i} = u_k^T h_i, t = U^T h (K, N)
  spectrum       d_q^(k) = |t~_k(q)|^2, t~ = fft(t) / sqrt(N) (unitary; q in FFT order
                 0, 1, .., N/2, .., -1); no mean is subtracted
  null mean      sum_p |r~_p(q)|^2 / (P - K) with r = h - U U^T h: the mean spectrum of the
                 P - K directions orthogonal to col(Z), by Parseval over any basis of col(Z)^perp
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
# u_1 must be the mean direction: its cosine with the constant pattern vector must reach this
# fraction of sqrt(3K / (3K + 1)), the value for iid latents on [0, a]. Symmetric latents give
# O(P^-1/2) instead (0.07 at P = 256, K = 3), because their K singular values are degenerate.
MEAN_DIRECTION_FRACTION = 0.5
# Sokal's window: sum the autocorrelation up to the first lag W >= SOKAL_C * tau(W). With
# tau = 1 + 2 sum_k rho_k this C = 5 is Sokal's c = 10 in his tau_int = 1/2 + sum_k rho_k
# convention (emcee pairs them the same way).
SOKAL_C = 5.0
# The window is searched up to a quarter of the chain, where the autocorrelation estimate is
# still reliable.
MAX_WINDOW_FRACTION = 0.25
# Independent records need a window of W >= SOKAL_C lags within n/4, which a short chain can
# miss by sampling noise alone: of 10^4 simulated chains of independent records, 19% fail at
# 20 records, 3% at 40, 0.6% at 60 and 0.03% at 100. The minimum keeps such spurious
# "chain too short" errors rare; a correlated chain needs proportionally more records.
MIN_SEM_RECORDS = 100
_LATENT_NAME = re.compile(r"z(\d+)")


class LatentBasis(NamedTuple):
    U: Tensor  # (P, K) orthonormal columns, column 0 the mean direction u_1
    s2: Tensor  # (K,) squared singular values of Z, descending
    Z: Tensor  # (P, K) the latents z0..z{K-1} stacked as columns


class Summary(NamedTuple):
    # each field is a Python number for one spectrum (N,) and a tensor (...) for (..., N)
    q_half: int | Tensor  # half-power |q|
    share0: float | Tensor  # d_0 / sum_q d_q
    lag1: float | Tensor  # R(1) / R(0)
    d0: float | Tensor  # d_0
    d_nyq: float | Tensor  # d_{N/2}, the floor of q_half
    R: Tensor  # (..., N) autocorrelation R(r), r = 0..N-1


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
    # non-orthonormality, which would mix latent power into null_power.
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
    if t2.shape[-1] == 0:
        raise ValueError(f"t2 and t3 must be (..., N) with N non-empty, got {tuple(t2.shape)}")


def _check_records(records: Tensor, burn_in: int) -> Tensor:
    """Validate (R, ...) records and return those after burn-in."""
    _check_float("records", records, 1)
    if not 0 <= burn_in < records.shape[0]:
        raise ValueError(f"burn_in must be in [0, {records.shape[0]}), got {burn_in}")
    return records[burn_in:]


def _number_if_0d(x: Tensor) -> float | int | Tensor:
    """A Python number for a 0-d tensor (one map's summary); the tensor itself otherwise."""
    return x.item() if x.ndim == 0 else x


# ------------------------------------------------------------------ basis and maps
def _check_mean_direction(U: Tensor) -> None:
    """Raise unless u_1 is the mean direction (see MEAN_DIRECTION_FRACTION)."""
    P, K = U.shape
    cosine = float(U[:, 0].sum()) / math.sqrt(P)
    threshold = MEAN_DIRECTION_FRACTION * math.sqrt(3 * K / (3 * K + 1))
    if cosine < threshold:
        raise ValueError(
            f"the latents look symmetric about 0: cos(u_1, 1) = {cosine:.3f} < {threshold:.3f}, "
            "so the top singular values are near-degenerate and the mean direction is undefined; "
            "latent_basis needs latents on [0, a]"
        )


def latent_basis(factors: Mapping[str, Tensor]) -> LatentBasis:
    """Orthonormal frame of the latents: Z (P, K) = U S V^T with 1^T u_k >= 0.

    `factors` must be exactly z0..z{K-1} (1-D, length P); they are stacked by index, so z10
    follows z9, and names that collide (z1, z01) raise. Raises for K >= P, for rank-deficient
    Z (its frame would be arbitrary), and unless u_1 is the mean direction: the cosine of u_1
    with the constant pattern vector must be at least MEAN_DIRECTION_FRACTION (1/2) of
    sqrt(3K / (3K + 1)), its value for iid latents on [0, a]. Symmetric latents (on [-a, a])
    fail this: their singular values are degenerate up to O(P^-1/2) noise and the SVD frame is
    then a random rotation of col(Z).
    """
    if not factors:
        raise ValueError("factors holds no latents z0..z{K-1}")
    by_index = {_latent_index(name): z for name, z in factors.items()}
    if len(by_index) != len(factors):
        raise ValueError(f"latent names collide on one index: {sorted(factors)}")
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
    U = U * torch.where(U.sum(dim=0) >= 0, 1.0, -1.0).to(U.dtype)
    _check_mean_direction(U)
    return LatentBasis(U=U, s2=S.square(), Z=Z)


def tuning_maps(h: Tensor, U: Tensor) -> Tensor:
    """t = U^T h: (..., P, N) activity and (P, K) basis -> (..., K, N) tuning maps."""
    _check_activity_and_basis(h, U)
    return U.T @ h


def _power(x: Tensor) -> Tensor:
    return torch.fft.fft(x, dim=-1, norm="ortho").abs().square()


def _null_power_of(h: Tensor, U: Tensor, t: Tensor) -> Tensor:
    """Null mean from the residual r = h - U t (t = U^T h), never forming U_perp.

    Projecting out col(Z) before the FFT, rather than subtracting the latent power from the
    total, keeps a null floor far below the latent power exact: at sigma = 0 the residual is
    zero to round-off of h itself, not of the latent power.
    """
    P, K = U.shape
    return _power(h - U @ t).sum(dim=-2) / (P - K)


def power_spectrum(t: Tensor) -> Tensor:
    """|fft(t) / sqrt(N)|^2 over the last axis: (..., N) -> (..., N), FFT order in q."""
    _check_float("t (..., N)", t, 1)
    return _power(t)


def null_power(h: Tensor, U: Tensor) -> Tensor:
    """Mean spectrum of the P - K directions orthogonal to U: (..., P, N) -> (..., N)."""
    _check_activity_and_basis(h, U)
    return _null_power_of(h, U, U.T @ h)


def mode_spectra(h: Tensor, U: Tensor) -> Tensor:
    """(..., P, N) -> (..., K + 1, N): rows 0..K-1 the latent modes, row K the null mean."""
    _check_activity_and_basis(h, U)
    t = U.T @ h
    return torch.cat([_power(t), _null_power_of(h, U, t).unsqueeze(-2)], dim=-2)


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


def _half_power_q(d: Tensor) -> Tensor:
    """(..., N) -> (...) int64; d already validated (finite, even N)."""
    N = d.shape[-1]
    k = torch.arange(N, device=d.device)
    q = torch.minimum(k, N - k)  # |q| of each FFT-order index
    d0, floor = d[..., :1], d[..., N // 2 : N // 2 + 1]
    if (d0 < floor).any():
        bad = (d0 < floor).nonzero()[0].tolist()[:-1]
        where = f" at batch index {bad}" if bad else ""
        raise ValueError(
            f"half-power frequency undefined: d_0 < d_N/2{where} (d_0 = "
            f"{float(d0[..., 0][tuple(bad)])}, d_N/2 = {float(floor[..., 0][tuple(bad)])}); "
            "the spectrum rises "
            "with |q|"
        )
    above = (d - floor) >= 0.5 * (d0 - floor)  # always true at q = 0
    return torch.where(above, q, -1).amax(dim=-1)


def _check_spectrum(d: Tensor) -> int:
    _check_float("d", d, 1)
    return _check_even_last_dim("the spectrum", d)


def half_power_q(d: Tensor) -> int | Tensor:
    """Largest |q| with d_q - d_{N/2} >= (d_0 - d_{N/2}) / 2: (..., N) FFT order -> (...).

    Raises if d_0 < d_{N/2} (the spectrum rises to Nyquist and has no half-power point).
    """
    _check_spectrum(d)
    return _number_if_0d(_half_power_q(d))


def spectrum_summary(d: Tensor) -> Summary:
    """q_1/2, the q = 0 share, lag-1, d_0, d_{N/2} and R of spectra d (..., N), FFT order."""
    N = _check_spectrum(d)
    total = d.sum(dim=-1)
    if (total <= 0).any():
        raise ValueError(f"d must have positive total power, got sums down to {float(total.min())}")
    R = autocorrelation(d)
    return Summary(
        q_half=_number_if_0d(_half_power_q(d)),
        share0=_number_if_0d(d[..., 0] / total),
        lag1=_number_if_0d(R[..., 1] / R[..., 0]),
        d0=_number_if_0d(d[..., 0]),
        d_nyq=_number_if_0d(d[..., N // 2]),
        R=R,
    )


def gaussian_width(R: Tensor, ell: float) -> float | Tensor:
    """sqrt(-ell^2 / (2 ln(R(ell)/R(0)))): (..., N) -> (...), the Gaussian through R(0), R(ell).

    ell is a lag in sites, so it must be a whole number in 1..N-1 (a float is accepted).
    """
    _check_float("R", R, 1)
    lag = int(ell)
    if lag != ell or not 1 <= lag < R.shape[-1]:
        raise ValueError(f"ell must be an integer lag in 1..{R.shape[-1] - 1}, got {ell}")
    R0 = R[..., 0]
    ratio = R[..., lag] / R0
    if not ((R0 > 0) & (ratio > 0) & (ratio < 1)).all():
        raise ValueError(
            f"need R(0) > 0 and 0 < R(ell)/R(0) < 1, got R(0) down to {float(R0.min())}, "
            f"R(ell)/R(0) in [{float(ratio.min())}, {float(ratio.max())}]"
        )
    return _number_if_0d(torch.sqrt(-(lag**2) / (2 * torch.log(ratio))))


# ------------------------------------------------------------------ angle statistic
def _theta(t2: Tensor, t3: Tensor) -> Tensor:
    """atan2(t3, t2) on (-pi, pi]: atan2 can return -pi, the same angle as pi."""
    theta = torch.atan2(t3, t2)
    return torch.where(theta == -math.pi, torch.full_like(theta, math.pi), theta)


def contrast_angle_statistic(t2: Tensor, t3: Tensor) -> float | Tensor:
    """<cos 4 theta_i> over neurons, theta_i = atan2(t3_i, t2_i): (..., N) -> (...).

    t2, t3 are rows 1 and 2 of `tuning_maps` (the contrasts u_2, u_3) of one network each.
    0 for a rotation-invariant (t2, t3), positive for tuning aligned with the axes, negative
    along the diagonals. Meaningful only in an orthonormal frame: an oblique pair squeezes an
    isotropic cloud along its bisector and biases the value negative. A silent neuron
    (t2 = t3 = 0) has atan2(0, 0) = 0 and counts as axis-aligned.
    """
    _check_angle_inputs(t2, t3)
    return _number_if_0d(torch.cos(4 * _theta(t2, t3)).mean(dim=-1))


def theta_histogram(t2: Tensor, t3: Tensor) -> Tensor:
    """(..., 36) int64 counts of theta_i on fixed bins (-pi + j w, -pi + (j+1) w], w = 2pi/36.

    Fixed bins make histograms of different records and networks summable. A silent neuron
    (t2 = t3 = 0) has theta = 0, the right edge of bin 17.
    """
    _check_angle_inputs(t2, t3)
    upper = torch.linspace(-math.pi, math.pi, N_THETA_BINS + 1, dtype=t2.dtype, device=t2.device)
    upper = upper[1:]
    upper[-1] = math.pi  # theta = pi must land in the last bin, whatever linspace rounds to
    index = torch.bucketize(_theta(t2, t3), upper, right=False)  # (..., N)
    batch = index.shape[:-1]
    flat = index.reshape(-1, index.shape[-1])
    # one bincount over all maps: offset each map's bins into its own block of 36
    offset = N_THETA_BINS * torch.arange(flat.shape[0], device=flat.device).unsqueeze(-1)
    counts = torch.bincount((flat + offset).flatten(), minlength=N_THETA_BINS * flat.shape[0])
    return counts.reshape(*batch, N_THETA_BINS)


# ------------------------------------------------------------------ records and members
def average_records(records: Tensor, burn_in: int) -> Tensor:
    """Mean over the records of one chain after `burn_in`: (R, ...) -> (...). A diagnostic."""
    return _check_records(records, burn_in).mean(dim=0)


def _c0_and_tau(x: Tensor) -> tuple[Tensor, Tensor]:
    """(C(0) (...), tau (...)) of records x (n, ...), already checked by `_check_sem_records`.

    C(0) is the biased variance; tau = 1 + 2 sum_{k=1}^{W} rho_k with Sokal's window, from the
    biased autocovariance by FFT (zero-padded to 2n, so the correlation is linear, not
    circular). Constant elements have rho = 0 and tau = 1.
    """
    n = x.shape[0]
    centred = x - x.mean(dim=0)
    spectrum = torch.fft.rfft(centred, n=2 * n, dim=0).abs().square()
    acov = torch.fft.irfft(spectrum, n=2 * n, dim=0)[:n] / n
    c0 = acov[0].clamp(min=0)
    constant = c0 <= 0
    rho = torch.where(constant, torch.zeros_like(acov), acov / torch.where(constant, 1.0, c0))

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
    # Sokal's window assumes positive correlations: anticorrelated records (tau < 1) meet it
    # trivially at small W, where tau can even be negative. tau = 1 treats them as independent,
    # which overstates their standard error: the safe side for a diagnostic.
    return c0, tau.clamp(min=1.0)


def _check_sem_records(records: Tensor, burn_in: int) -> Tensor:
    x = _check_records(records, burn_in)
    if x.shape[0] < MIN_SEM_RECORDS:
        raise ValueError(
            f"need at least {MIN_SEM_RECORDS} records after burn-in, got {x.shape[0]}: shorter "
            "chains miss the autocorrelation window by sampling noise"
        )
    return x


def integrated_autocorr_time(records: Tensor, burn_in: int) -> Tensor:
    """tau = 1 + 2 sum_{k=1}^{W} rho_k per element, (R, ...) -> (...), Sokal's window.

    W is the first lag with W >= SOKAL_C * tau(W) (Sokal 1997; SOKAL_C = 5 here is his c = 10,
    see the constant), searched up to a quarter of the chain; raises if some element's window
    is not reached there (the chain is too short for its correlation time) and for fewer than
    MIN_SEM_RECORDS (100) records after burn-in, below which independent records fail by
    sampling noise. tau is clamped at 1 from below (anticorrelated records; constant ones).
    """
    _, tau = _c0_and_tau(_check_sem_records(records, burn_in))
    return tau


def record_sem(records: Tensor, burn_in: int) -> Tensor:
    """Standard error of the chain mean after burn-in, (R, ...) -> (...): sqrt(C(0) tau / n).

    n / tau is the effective number of independent records, C(0) the variance of the records
    and tau their integrated autocorrelation time (`integrated_autocorr_time`, with its
    window, minimum of 100 records and clamp). A diagnostic: comparing it across burn-in
    values, or with the spread across members, shows whether the chain has equilibrated; it is
    not the error of a per-network estimator.
    """
    x = _check_sem_records(records, burn_in)
    c0, tau = _c0_and_tau(x)
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
