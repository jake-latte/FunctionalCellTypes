"""The estimator of the linear-ring smoke test and its tolerances, shared by the synthetic
check (tests/test_linear_ring_estimator.py) and the SGLD run (tests/test_linear_ring_smoke.py).

One sample per member: member b contributes its final-state spectrum d_b (K, N), rows 0..K-1
of the spectrum monitor's last record. Every statistic is of the member-averaged spectrum
D = mean_b d_b, and its standard error is the bootstrap over members
(`fct.spectrum.bootstrap_members`). The asserted statistics are averages over the ring, per
latent mode k (FFT order in q, R = ifft(D) the autocorrelation, ell the hidden kernel's length
scale in sites):

  lag1_k    R_k(1) / R_k(0)
  lagl_k    R_k(ell) / R_k(0)       (the Gaussian width read at ell, sqrt(-ell^2 / (2 ln lagl_k)),
                                     is a monotone function of it; unlike the width it is
                                     defined for every spectrum, a single record's included)
  band_k    sum_{|q| <= q_b} D_k / sum_q D_k, q_b = round(3 N / 64)    (`band_share`)
  ratio     sum_{|q| <= q_b} D_1 / sum_{|q| <= q_b} D_2            (mode 1 over mode 2)

The q = 0 share and q_1/2 rest on one or two frequencies and are reported, not asserted.

Standard errors. In the theory a member's tuning map t_k is a real stationary Gaussian field
with spectrum D_k(q), so its power d_k(q) is D_k(q) chi^2_1 at q = 0 and N/2 (relative variance
2) and D_k(q) Exp(1) elsewhere (relative variance 1), independent across q >= 0 and across
modes. A real map has d(-q) = d(q) exactly, so folding +-q averages a value with itself and
gains nothing: the independent samples are the B members, nothing else. The delta method on
the member mean (`delta_method_se`: the Jacobian of the statistics against `member_covariance`)
gives each statistic's SE; for the band power P_b = sum_{|q|<=q_b} D, for instance,

  SE(P_b) / P_b = sqrt(2 D(0)^2 + 4 sum_{q=1..q_b} D(q)^2) / P_b / sqrt(B).

The thresholds are relative to the prediction at the run's N and to its SE at the run's B
(`bounds`): SE_MAX is SE_MARGIN = 1.35 times the delta-method SE of each statistic and the
tolerance is N_SE = 3 times SE_MAX, so a measured statistic must lie within 3 SE_MAX of the
predicted one and its bootstrap SE below SE_MAX. The margin covers the spread of a bootstrap SE
around the true one; tests/test_linear_ring_estimator.py pins the 99th percentile of that
ratio below SE_MARGIN on ensembles drawn from the predicted law.
"""

from collections.abc import Callable
from typing import NamedTuple

import torch
from torch import Tensor

from fct.config import Config
from fct.data import Dataset, make_dataset
from fct.exp import expand, preset, setting_config
from fct.linear_theory import predicted_spectra
from fct.spectrum import (
    autocorrelation,
    band_half_width,
    band_share,
    bootstrap_members,
    latent_basis,
    member_mean_sem,
)

# Bootstrap resamples per estimate: the resampling error of an SE is 1 / sqrt(2 N_BOOT) = 2%,
# small beside SE_MARGIN.
N_BOOT = 1000
SE_MARGIN = 1.35
N_SE = 3  # a statistic must lie within N_SE times SE_MAX of the prediction
CPU, F64 = torch.device("cpu"), torch.float64


class Law(NamedTuple):
    """The predicted law of one setting of a preset."""

    cfg: Config
    dataset: Dataset
    D: Tensor  # (K, N) predicted spectra, FFT order
    U: Tensor  # (P, K) the latent frame the monitor uses
    ell: int  # hidden kernel length scale in sites
    B: int


def predicted_law(name: str, index: int) -> Law:
    """Setting `index` of preset `name`: its config, data, predicted spectra and frame."""
    exp = preset(name)
    cfg = setting_config(exp, expand(exp)[index])
    dataset = make_dataset(cfg.data, CPU, F64)
    sites = cfg.spatial.hidden.kernel.length_scale * cfg.net.N / cfg.spatial.box
    if sites != int(sites) or sites < 1:
        raise ValueError(f"the hidden length scale is {sites} sites; widths need a whole lag")
    return Law(
        cfg=cfg,
        dataset=dataset,
        D=torch.from_numpy(predicted_spectra(cfg, dataset).d),
        U=latent_basis(dataset.factors).U,
        ell=int(sites),
        B=cfg.net.B,
    )


def statistic_names(K: int) -> list[str]:
    modes = range(1, K + 1)
    return (
        [f"lag1_{k}" for k in modes]
        + [f"lagl_{k}" for k in modes]
        + [f"band_{k}" for k in modes]
        + ["ratio"]
    )


def kind_of(name: str) -> str:
    """The kind of a statistic name: `lag1_1` -> `lag1`."""
    return name.split("_")[0]


def bounds(law: Law) -> tuple[Tensor, Tensor]:
    """(SE_MAX, tolerance), each (3K + 1,): SE_MARGIN times the delta-method SE of each
    statistic at the law's B, and N_SE times that."""
    se_max = SE_MARGIN * delta_method_se(law.D, law.ell, law.B)
    return se_max, N_SE * se_max


def ring_statistics(predicted: Tensor, ell: int) -> Callable[[Tensor], Tensor]:
    """The statistics of a mean spectrum D (..., K, N) as a vector (..., 3K + 1) in the order
    of `statistic_names`, for spectra of the ring of `predicted` (K, N)."""
    N = predicted.shape[-1]
    k = torch.arange(N)
    in_band = torch.minimum(k, N - k) <= band_half_width(N)

    def statistics(D: Tensor) -> Tensor:
        R = autocorrelation(D)
        power = D[..., in_band].sum(-1)
        return torch.cat(
            [
                R[..., 1] / R[..., 0],
                R[..., ell] / R[..., 0],
                torch.as_tensor(band_share(D)),
                (power[..., 0] / power[..., 1]).unsqueeze(-1),
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


def delta_method_se(D: Tensor, ell: int, B: int) -> Tensor:
    """(3K + 1,): the standard error of each statistic of the mean of B members drawn from the
    law of mean spectrum D (K, N), sqrt(J Sigma J^T / B) with J the Jacobian of
    `ring_statistics` at D and Sigma the block-diagonal `member_covariance`."""
    J = torch.autograd.functional.jacobian(ring_statistics(D, ell), D)  # (3K + 1, K, N)
    variance = torch.einsum("skq,kqr,skr->s", J, member_covariance(D), J)
    return (variance / B).sqrt()


def estimate(member_spectra: Tensor, law: Law, seed: int = 0) -> tuple[Tensor, Tensor]:
    """Statistics of the member mean of (B, K, N) spectra and their bootstrap standard errors."""
    return bootstrap_members(member_spectra, ring_statistics(law.D, law.ell), N_BOOT, seed)


def map_correlation(first: Tensor, last: Tensor) -> Tensor:
    """(B,): per member, the correlation of its tuning maps (B, K, N) at two times, each mode
    normalised to unit power and the K modes averaged. Independent maps give 0 on average (the
    spread across members is about 1 / sqrt(K x the number of frequencies carrying power));
    a frozen sampler gives 1."""
    per_mode = (first * last).sum(-1) / (first.norm(dim=-1) * last.norm(dim=-1))
    return per_mode.mean(-1)


def time_drift(records: Tensor, burn_in: int, law: Law) -> tuple[Tensor, Tensor]:
    """The burn-in diagnostic: per member, the time average of each statistic over the second
    half of the records after `burn_in` minus that over the first half, then the mean of that
    difference across members and its standard error (`member_mean_sem`: the members are
    independent chains, so no autocorrelation time enters): (R, B, K, N) -> two (3K + 1,).
    A chain still relaxing after the burn-in drifts, and the mean difference is then many
    standard errors from 0. Chains started in their stationary law show no drift whether they
    mix or not, so this is no evidence of mixing (`map_correlation` is)."""
    kept = records[burn_in:]
    half = kept.shape[0] // 2
    if half < 1:
        raise ValueError(f"need at least 2 records after burn-in, got {kept.shape[0]}")
    per_record = ring_statistics(law.D, law.ell)(kept)  # (R', B, 3K + 1), one record each
    diff = per_record[-half:].mean(0) - per_record[:half].mean(0)  # (B, 3K + 1)
    return member_mean_sem(diff)
