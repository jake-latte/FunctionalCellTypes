"""The estimator of the linear-ring smoke test and its tolerances, shared by the synthetic
check (tests/test_linear_ring_estimator.py) and the SGLD run (tests/test_linear_ring_smoke.py).

One sample per member: member b contributes its final-state spectrum d_b (K, N), rows 0..K-1
of the spectrum monitor's last record. Every statistic is of the member-averaged spectrum
D = mean_b d_b, and its standard error is the bootstrap over members
(`fct.spectrum.bootstrap_members`). The statistics, per latent mode k (FFT order in q):

  share_k   D_k(0) / sum_q D_k(q)                 the q = 0 share
  lag1_k    R_k(1) / R_k(0), R = ifft(D)          the lag-1 autocorrelation
  band_k    sum_{|q| <= 3} D_k / the same sum of the prediction   (1 if the scale is right)
  ratio     sum_{|q| <= 3} D_1 / sum_{|q| <= 3} D_2              (mode 1 over mode 2)

and q_1/2 of each mode, compared exactly. The band |q| <= 3 holds 99.6% of the predicted power
on the smoke ring and is the ratio's range in the experiment plan.

Standard errors. In the theory a member's tuning map t_k is a real stationary Gaussian field
with spectrum D_k(q), so its power d_k(q) is D_k(q) chi^2_1 at q = 0 and N/2 (relative variance
2) and D_k(q) Exp(1) elsewhere (relative variance 1), independent across q >= 0 and across
modes. A real map has d(-q) = d(q) exactly, so folding +-q averages a value with itself and
gains nothing: the independent samples are the B members, nothing else. By the delta method
on the member mean, with T = sum_q D(q) and V_T = 2 D(0)^2 + 2 D(N/2)^2 + 4 sum_{0<q<N/2} D(q)^2,

  SE(share) = share sqrt((2 + V_T / T^2 - 4 D(0) / T) / B)            (0.0214 for S1 at B = 256)
  SE(band)  = sqrt(2 D(0)^2 + 4 sum_{q=1..3} D(q)^2) / sum_{|q|<=3} D / sqrt(B)        (0.042)
  SE(ratio) = ratio sqrt(2) x the relative SE(band) of either mode                     (0.44)

and 0.0042 for lag1; `delta_method_se` evaluates all of them from the Jacobian of the
statistics and the covariance of one member's spectrum. On the smoke ring the q = 0 share of one member is a single
chi^2_1 draw over about three effective frequencies, so B must be large: B = 256 gives the
numbers above (16 members would give 0.085 on a share). The test draws ensembles from exactly
this law (tests/test_linear_ring_estimator.py) and finds the simulated spreads within 7% of
these, and the bootstrap SE of one ensemble at most 1.30 times that spread in 99% of
ensembles. SE_MAX is therefore 1.35 times the true SE, the TOLERANCE is three SE_MAX, and the
test fails when an estimate is further than TOLERANCE from the prediction or when its own
bootstrap SE exceeds SE_MAX = TOLERANCE / 3. The tolerances separate the prediction from a flat
spectrum (shares 12, lag1 200, band powers 13 SEs away; q_1/2 = 8) and from equal mode
amplitudes (ratio 15 SEs away), but not from the prior alone: at beta = 0.25 the data moves no
statistic by more than 0.11 SE.
"""

from collections.abc import Callable

import torch
from torch import Tensor

from fct.spectrum import autocorrelation, bootstrap_members, half_power_q, member_mean_sem

BAND = 3
N_BOOT = 1000
# The largest bootstrap SE accepted, per kind of statistic, at the smoke ring's B = 256: 1.35
# times the SE of the predicted law (see the module docstring), rounded up.
SE_MAX = {"share": 0.029, "lag1": 0.0058, "band": 0.057, "ratio": 0.59}
TOLERANCE = {kind: 3 * se for kind, se in SE_MAX.items()}


def band_mask(N: int) -> Tensor:
    """(N,) bool, FFT order: |q| <= BAND."""
    k = torch.arange(N)
    return torch.minimum(k, N - k) <= BAND


def statistic_names(K: int) -> list[str]:
    modes = range(1, K + 1)
    return (
        [f"share_{k}" for k in modes]
        + [f"lag1_{k}" for k in modes]
        + [f"band_{k}" for k in modes]
        + ["ratio"]
    )


def kind_of(name: str) -> str:
    """The SE_MAX / TOLERANCE key of a statistic name: `share_1` -> `share`."""
    return name.split("_")[0]


def ring_statistics(predicted: Tensor) -> Callable[[Tensor], Tensor]:
    """The statistics of a mean spectrum D (..., K, N) as a vector (..., 3K + 1) in the order
    of `statistic_names`; `predicted` (K, N) normalises the band powers."""
    in_band = band_mask(predicted.shape[-1])
    predicted_band = predicted[:, in_band].sum(-1)

    def statistics(D: Tensor) -> Tensor:
        R = autocorrelation(D)
        band = D[..., in_band].sum(-1)
        return torch.cat(
            [
                D[..., 0] / D.sum(-1),
                R[..., 1] / R[..., 0],
                band / predicted_band.to(D.dtype),
                (band[..., 0] / band[..., 1]).unsqueeze(-1),
            ],
            dim=-1,
        )

    return statistics


def member_covariance(D: Tensor) -> Tensor:
    """(K, N, N): the covariance of one member's power d_k(q) under the Gaussian law of mean
    spectrum D (K, N): D(q) D(q') ([q' = q] + [q' = -q]), so 2 D(q)^2 at q = 0 and N/2 (chi^2_1)
    and D(q)^2 between q and -q elsewhere (one Exp(1) variate shared by both)."""
    N = D.shape[-1]
    q = torch.arange(N)
    pairs = (q[:, None] == q[None, :]).to(D.dtype) + (q[:, None] == (-q[None, :]) % N).to(D.dtype)
    return D[:, :, None] * D[:, None, :] * pairs


def delta_method_se(D: Tensor, B: int) -> Tensor:
    """(3K + 1,): the standard error of each statistic of the mean of B members drawn from the
    law of mean spectrum D (K, N), sqrt(J Sigma J^T / B) with J the Jacobian of
    `ring_statistics` at D and Sigma the block-diagonal `member_covariance`."""
    J = torch.autograd.functional.jacobian(ring_statistics(D), D)  # (3K + 1, K, N)
    variance = torch.einsum("skq,kqr,skr->s", J, member_covariance(D), J)
    return (variance / B).sqrt()


def estimate(member_spectra: Tensor, predicted: Tensor, seed: int = 0) -> tuple[Tensor, Tensor]:
    """Statistics of the member mean of (B, K, N) spectra and their bootstrap standard errors."""
    return bootstrap_members(member_spectra, ring_statistics(predicted), N_BOOT, seed)


def half_power(member_spectra: Tensor) -> Tensor:
    """q_1/2 of each mode of the member-averaged spectrum: (B, K, N) -> (K,) int64."""
    return torch.as_tensor(half_power_q(member_spectra.mean(0)))


def time_drift(records: Tensor, burn_in: int, predicted: Tensor) -> tuple[Tensor, Tensor]:
    """The burn-in diagnostic: per member, the time average of each statistic over the second
    half of the records after `burn_in` minus that over the first half, then the mean of that
    difference across members and its standard error (`member_mean_sem`: the members are
    independent chains, so no autocorrelation time enters): (R, B, K, N) -> two (3K + 1,).
    A chain still relaxing after the burn-in drifts, and the mean difference is then many
    standard errors from 0."""
    kept = records[burn_in:]
    half = kept.shape[0] // 2
    if half < 1:
        raise ValueError(f"need at least 2 records after burn-in, got {kept.shape[0]}")
    statistics = ring_statistics(predicted)
    per_record = statistics(kept)  # (R', B, 3K + 1), each record of each member on its own
    diff = per_record[-half:].mean(0) - per_record[:half].mean(0)  # (B, 3K + 1)
    return member_mean_sem(diff)
