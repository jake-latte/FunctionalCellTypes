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
  R(Delta)       (1/N) sum_q d_q e^{2 pi i q Delta / N} = ifft(d).real; lag-1 = R(1)/R(0)
  q_1/2          the largest |q| with d_q - d_{N/2} >= (d_0 - d_{N/2}) / 2 (Nyquist as floor)
  angle          theta_i = atan2(t_{3,i}, t_{2,i}) in the orthonormal frame (u_2, u_3)
  hexatic        <e^{6 i theta_i}> over neurons; prediction 6 is its modulus, frame-free

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
# u_1 is the mean direction only if the top singular value stands clear of the others. For iid
# latents on [0, a], s_1^2 / s_2^2 -> 3K + 1 (7 at K = 2); for symmetric latents it is
# 1 + O(P^-1/2). The gap is required to reach MIN_GAP, and the cosine of u_1 with the constant
# pattern vector to reach MEAN_DIRECTION_FRACTION of its [0, a] value sqrt(3K / (3K + 1)).
# K = 1 has no s_2 and only the cosine check, the weakest case. Simulated rates, 2 x 10^5
# draws per cell (false rejection of [0, a] latents / false acceptance of symmetric ones,
# both checks together):
#   P = 16: K = 1  0 / 8.3%,  K = 2  5e-6 / 1.73%, K = 3  0 / 0.73%,  K = 4  0 / 0.40%
#   P = 32: K = 1  0 / 1.31%, K = 2  0 / 0.064%,   K = 3  0 / 0.011%, K = 4  0 / 0.005%
#   P = 64: K = 1  0 / 3.4e-4, K >= 2  0 / 0
# At P = 16, K <= 3 the two distributions overlap, so no threshold has both near-zero false
# rejection and false acceptance below 0.1%. The presets run at P = 256, where the [0, a]
# and symmetric distributions are far apart and none of this matters.
MIN_GAP = 2.0
MEAN_DIRECTION_FRACTION = 0.5
# Geyer's initial monotone sequence: pairs Gamma_m = rho_{2m} + rho_{2m+1} are summed while
# positive (made non-increasing), over lags up to half the chain.
MAX_LAG_FRACTION = 0.5
# An element of a chain shorter than MIN_EFFECTIVE_RECORDS correlation times (n < 20 tau) is
# too short for its tau to be trusted, and is marked unreliable (NaN) rather than returned
# as an underestimate.
MIN_EFFECTIVE_RECORDS = 20
# Even independent records can look too short by sampling noise. In simulation (1.3 x 10^6
# elements of independent records) 2.6e-5 of elements are marked unreliable at 100 records,
# one element in 1.3 x 10^6 at 150 and none at 200. Fewer than MIN_SEM_RECORDS records is a
# caller error and raises.
MIN_SEM_RECORDS = 200
# Records are constant when C(0) <= (CONSTANT_EPS * eps * max(|mean|, tiny))^2: rounding
# leaves |x - mean| of a constant x below 3 eps |mean| (measured up to 10^5 records in float32
# and float64), so 8 eps leaves a margin while staying far below any real fluctuation.
CONSTANT_EPS = 8.0
_LATENT_NAME = re.compile(r"z(\d+)")


class LatentBasis(NamedTuple):
    U: Tensor  # (P, K) orthonormal columns, column 0 the mean direction u_1
    s2: Tensor  # (K,) squared singular values of Z, descending
    Z: Tensor  # (P, K) the latents z0..z{K-1} stacked as columns


class TauEstimate(NamedTuple):
    tau: Tensor  # (...) integrated autocorrelation time; NaN where not reliable
    reliable: Tensor  # (...) bool: n >= 20 tau, Geyer's pairs ended before n/2, tau > 0


class SemEstimate(NamedTuple):
    sem: Tensor  # (...) standard error of the chain mean; NaN where not reliable
    tau: Tensor  # (...) as in TauEstimate
    reliable: Tensor  # (...) bool, as in TauEstimate


class Summary(NamedTuple):
    # each field is a Python number for one spectrum (N,) and a tensor (...) for (..., N)
    share0: float | Tensor  # d_0 / sum_q d_q
    lag1: float | Tensor  # R(1) / R(0)
    d0: float | Tensor  # d_0
    d_nyq: float | Tensor  # d_{N/2}, the floor of half_power_q
    R: Tensor  # (..., N) autocorrelation R(Delta), Delta = 0..N-1


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
def _check_mean_direction(U: Tensor, S: Tensor) -> None:
    """Raise unless u_1 is the mean direction (see MIN_GAP and MEAN_DIRECTION_FRACTION)."""
    P, K = U.shape
    if K >= 2:
        gap = float((S[0] / S[1]) ** 2)
        if gap < MIN_GAP:
            raise ValueError(
                f"the latents look symmetric about 0: s_1^2 / s_2^2 = {gap:.2f} < {MIN_GAP}, so "
                "the top singular values are near-degenerate and the mean direction is "
                "undefined; latent_basis needs latents on [0, a]"
            )
    cosine = float(U[:, 0].sum()) / math.sqrt(P)
    threshold = MEAN_DIRECTION_FRACTION * math.sqrt(3 * K / (3 * K + 1))
    if cosine < threshold:
        raise ValueError(
            f"the latents look symmetric about 0: cos(u_1, 1) = {cosine:.3f} < {threshold:.3f}, "
            "so u_1 is not along the mean and the mean direction is undefined; latent_basis "
            "needs latents on [0, a]"
        )


def latent_basis(factors: Mapping[str, Tensor]) -> LatentBasis:
    """Orthonormal frame of the latents: Z (P, K) = U S V^T with 1^T u_k >= 0.

    `factors` must be exactly z0..z{K-1} (1-D, length P); they are stacked by index, so z10
    follows z9, and names that collide (z1, z01) raise. Raises for K >= P, for rank-deficient
    Z (its frame would be arbitrary), and unless u_1 is the mean direction. For K >= 2 that
    needs a gap s_1^2 / s_2^2 >= MIN_GAP (2; its [0, a] value is 3K + 1), and for every K
    the cosine of u_1 with the constant pattern vector must be at least
    MEAN_DIRECTION_FRACTION (1/2) of sqrt(3K / (3K + 1)), its value for iid latents on
    [0, a]. Symmetric latents (on [-a, a]) fail: their singular values are degenerate up to
    O(P^-1/2) noise and the SVD frame is then a random rotation of col(Z). The simulated
    false rejection / false acceptance rates are tabulated at MIN_GAP: false acceptance of
    symmetric latents is 1.73% at P = 16, K = 2 and 8.3% at P = 16, K = 1 (only the cosine
    check), and 0.064% at P = 32, K = 2. The presets run at P = 256, where none of this
    matters.
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
    _check_mean_direction(U, S)
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
    """R(Delta) = ifft(d).real over the last axis: (..., N) spectrum -> (..., N) at lags
    Delta = 0..N-1."""
    _check_float("d", d, 1)
    return torch.fft.ifft(d, dim=-1).real


def _check_spectrum(d: Tensor) -> int:
    _check_float("d", d, 1)
    if d.numel() == 0:
        raise ValueError(f"d is empty (shape {tuple(d.shape)}): no spectra to summarise")
    return _check_even_last_dim("the spectrum", d)


def half_power_q(d: Tensor) -> int | Tensor:
    """Largest |q| with d_q - d_{N/2} >= (d_0 - d_{N/2}) / 2: (..., N) FFT order -> (...).

    A statistic of a mean spectrum (averaged over records or members), not of one
    realisation: in one record d_0 and d_{N/2} are single chi^2 draws, so their order is
    noise (a map with mean D_0 = 100 D_{N/2} still has d_0 < d_{N/2} with probability
    (2/pi) atan(sqrt(D_{N/2}/D_0)) = 6%). Raises if d_0 < d_{N/2} anywhere: the spectrum then
    rises to Nyquist and has no half-power point.
    """
    N = _check_spectrum(d)
    k = torch.arange(N, device=d.device)
    q = torch.minimum(k, N - k)  # |q| of each FFT-order index
    d0, floor = d[..., :1], d[..., N // 2 : N // 2 + 1]
    rising = (d0 < floor)[..., 0]
    if rising.any():
        first = rising.nonzero()[0].tolist()
        where = f"{int(rising.sum())} of {rising.numel()} spectra, first at batch index {first}"
        raise ValueError(
            f"half-power frequency undefined: d_0 < d_N/2 in {where} (d_0 = "
            f"{float(d0[..., 0][tuple(first)])}, d_N/2 = {float(floor[..., 0][tuple(first)])}); "
            "the spectrum rises with |q|. half_power_q needs a mean spectrum"
        )
    above = (d - floor) >= 0.5 * (d0 - floor)  # always true at q = 0
    return _number_if_0d(torch.where(above, q, -1).amax(dim=-1))


def spectrum_summary(d: Tensor) -> Summary:
    """The q = 0 share, lag-1, d_0, d_{N/2} and R of spectra d (..., N), FFT order.

    Well defined for one record of one map; the half-power frequency is not, and is a
    separate call on a mean spectrum (`half_power_q`).
    """
    N = _check_spectrum(d)
    total = d.sum(dim=-1)
    if (total <= 0).any():
        raise ValueError(f"d must have positive total power, got sums down to {float(total.min())}")
    R = autocorrelation(d)
    return Summary(
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
    0 for a rotation-invariant (t2, t3), positive for tuning aligned with the SVD contrast axes
    (u_2, u_3), negative along their diagonals. These are not the latent axes: at K = 3 the
    three latents project onto the contrast plane 120 degrees apart, so cos 4 theta does not
    detect tuning to individual latents. Meaningful only in an orthonormal frame: an oblique
    pair squeezes an isotropic cloud along its bisector and biases the value negative. A
    silent neuron (t2 = t3 = 0) has atan2(0, 0) = 0 and counts as axis-aligned.
    """
    _check_angle_inputs(t2, t3)
    return _number_if_0d(torch.cos(4 * _theta(t2, t3)).mean(dim=-1))


def hexatic_moment(t2: Tensor, t3: Tensor) -> Tensor:
    """<e^{6 i theta_i}> over neurons, theta_i = atan2(t3_i, t2_i): (..., N) -> complex (...).

    t2, t3 are rows 1 and 2 of `tuning_maps` of one network each. Prediction 6's statistic is
    the modulus |<e^{6 i theta}>|, which is frame-free: rotating the tunings by phi (the frame
    (u_2, u_3) by -phi) multiplies the moment by e^{6 i phi}. It is near 1 when every neuron
    is tuned to a single latent (at K = 3 the latents project onto the contrast plane 120
    degrees apart, the same angle under 6 theta), and O(N^-1/2) for a rotation-invariant
    (t2, t3). The complex mean, not its modulus, is what averages across members and records:
    the modulus of a mean is not the mean of moduli. Always a tensor (complex64 for float32
    inputs, complex128 for float64), 0-d for one map. A silent neuron has theta = 0.
    """
    _check_angle_inputs(t2, t3)
    theta = _theta(t2, t3)
    return torch.polar(torch.ones_like(theta), 6 * theta).mean(dim=-1)


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


def _c0_tau_reliable(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(C(0), tau, reliable), each (...), of records x (n, ...) checked by `_check_sem_records`.

    C(0) is the biased variance and rho_k = C(k) / C(0) the autocorrelation, by FFT (zero-padded
    to 2n, so the correlation is linear, not circular). tau is Geyer's initial monotone
    sequence estimator: Gamma_m = rho_{2m} + rho_{2m+1} is summed over the initial run of
    positive pairs, made non-increasing, tau = -1 + 2 sum_m Gamma_m. Pairing cancels the
    alternation of anticorrelated components, so a slow component under a fast alternating
    one is still seen. Constant elements (CONSTANT_EPS) get C(0) = 0, tau = 1 and are reliable.
    An element is unreliable, with tau = NaN, when its pairs stay positive to lag n/2, when
    tau <= 0, or when n < MIN_EFFECTIVE_RECORDS * tau.
    """
    n = x.shape[0]
    mean = x.mean(dim=0)
    centred = x - mean
    spectrum = torch.fft.rfft(centred, n=2 * n, dim=0).abs().square()
    acov = torch.fft.irfft(spectrum, n=2 * n, dim=0)[:n] / n
    finfo = torch.finfo(x.dtype)
    constant = acov[0] <= (CONSTANT_EPS * finfo.eps * mean.abs().clamp(min=finfo.tiny)) ** 2
    c0 = torch.where(constant, torch.zeros_like(acov[0]), acov[0])
    rho = torch.where(constant, torch.zeros_like(acov), acov / torch.where(constant, 1.0, c0))
    rho[0] = 1.0

    n_pairs = int(MAX_LAG_FRACTION * n) // 2
    gamma = rho[0 : 2 * n_pairs : 2] + rho[1 : 2 * n_pairs : 2]  # (n_pairs, ...)
    ended = gamma <= 0
    run = ended.to(torch.int8).argmax(dim=0, keepdim=True)  # first non-positive pair
    m = torch.arange(n_pairs, device=x.device).reshape((-1,) + (1,) * (x.ndim - 1))
    monotone = gamma.cummin(dim=0).values
    tau = -1 + 2 * torch.where(m < run, monotone, 0.0).sum(dim=0)
    reliable = ended.any(dim=0) & (tau > 0) & (n >= MIN_EFFECTIVE_RECORDS * tau)
    return c0, torch.where(reliable, tau, torch.nan), reliable


def _check_sem_records(records: Tensor, burn_in: int) -> Tensor:
    x = _check_records(records, burn_in)
    if x.shape[0] < MIN_SEM_RECORDS:
        raise ValueError(
            f"need at least {MIN_SEM_RECORDS} records after burn-in, got {x.shape[0]}: shorter "
            "chains of even independent records look too short by sampling noise"
        )
    return x


def integrated_autocorr_time(records: Tensor, burn_in: int) -> TauEstimate:
    """tau = 1 + 2 sum_k rho_k per element, (R, ...) -> TauEstimate(tau (...), reliable (...)).

    Geyer's initial monotone sequence (Geyer 1992): the sum runs over the initial positive
    run of pairs rho_{2m} + rho_{2m+1}, up to lag n/2; tau < 1 for anticorrelated records.
    Per element, never per call: an element is unreliable, and its tau NaN, when its pairs
    stay positive to n/2, when tau <= 0, or when n < MIN_EFFECTIVE_RECORDS (20) tau, a chain
    too short to trust. Callers must read `reliable`; a NaN never appears without it. Geyer's
    estimator is biased low for strongly alternating records (AR(1) with phi <= -0.8) and at
    phi about -0.9 often gives tau <= 0; such elements are unreliable. Raises only for bad
    input and for fewer than MIN_SEM_RECORDS (200) records after burn-in.
    """
    _, tau, reliable = _c0_tau_reliable(_check_sem_records(records, burn_in))
    return TauEstimate(tau=tau, reliable=reliable)


def record_sem(records: Tensor, burn_in: int) -> SemEstimate:
    """Standard error of the chain mean after burn-in: (R, ...) -> SemEstimate, each (...).

    sem = sqrt(C(0) tau / n): n / tau is the effective number of independent records, C(0)
    the variance of the records and tau their integrated autocorrelation time, with its
    `reliable` mask (`integrated_autocorr_time`); sem is NaN exactly where tau is. Constant
    records give sem 0. A diagnostic: comparing it across burn-in values, or with the spread
    across members, shows whether the chain has equilibrated; it is not the error of a
    per-network estimator, so one unreliable element does not fail the call.
    """
    x = _check_sem_records(records, burn_in)
    c0, tau, reliable = _c0_tau_reliable(x)
    return SemEstimate(sem=(c0 * tau / x.shape[0]).sqrt(), tau=tau, reliable=reliable)


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
