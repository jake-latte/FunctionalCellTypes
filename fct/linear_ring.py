"""Analysis of the linear-ring experiments: a setting's predicted law, the ring-averaged statistics
of a member-averaged spectrum, their standard errors, the per-member estimator and the run
diagnostics.

One sample per member: member b contributes its final-state spectrum d_b (K, N), rows 0..K-1 of
the spectrum monitor's last record. Every statistic is of the member-averaged spectrum
D = mean_b d_b, with its standard error from the bootstrap over members (`estimate`). The
statistics, per latent mode k (FFT order in q, R = ifft(D), ell the hidden kernel's length scale
in sites, q_b = `fct.spectrum.band_half_width(N)`):

  lag1   R_k(1) / R_k(0)
  lagl   R_k(ell) / R_k(0)    (the Gaussian width at ell, sqrt(-ell^2 / (2 ln lagl)), is a
                               monotone function of it; unlike the width it is defined for
                               every spectrum, a single record's included)
  band   sum_{|q| <= q_b} D_k / sum_q D_k, the band share
  ratio  sum_{|q| <= q_b} D_1 / sum_{|q| <= q_b} D_2, mode 1 over mode 2 (no mode index)

Before any statistic is formed, the frequencies the penalty path's eigenvalue floor raises are
removed from both paths' spectra (`unfloored_frequencies`): their variance on the penalty path
is inflated by the Euler factor of its step, which `predicted_spectra` does not model.

Standard errors. In the theory a member's tuning map t_k is a real stationary Gaussian field
with spectrum D_k(q), so its power d_k(q) is D_k(q) chi^2_1 at q = 0 and N/2 (relative variance
2) and D_k(q) Exp(1) elsewhere (relative variance 1), independent across q >= 0 and across
modes. A real map has d(-q) = d(q) exactly, so folding +-q averages a value with itself and
gains nothing: the independent samples are the B members, nothing else. The delta method on
the member mean (`delta_method_se`) gives each statistic's SE at a given B.

Pure functions on tensors, except `predicted_law`, which builds the setting's dataset in memory.
"""

import math
from collections.abc import Callable
from typing import NamedTuple

import torch
from torch import Tensor

from fct.config import Config
from fct.data import Dataset, make_dataset
from fct.exp import ExpConfig, expand, setting_config
from fct.linear_theory import network_parameters, predicted_spectra
from fct.spectrum import (
    autocorrelation,
    band_mask,
    bootstrap_members,
    latent_basis,
)

CPU, F64 = torch.device("cpu"), torch.float64
# Bootstrap resamples per estimate: the resampling error of an SE is 1 / sqrt(2 N_BOOT) = 2%.
N_BOOT = 1000
KINDS = ("lag1", "lagl", "band")  # per-mode statistics, in the order of the vector


class Statistic(NamedTuple):
    kind: str  # one of KINDS, or "ratio"
    mode: int | None  # 1-based latent mode; None for the ratio of modes 1 and 2


class Law(NamedTuple):
    """The predicted law of one setting."""

    cfg: Config
    dataset: Dataset
    D: Tensor  # (K, N) predicted spectra, FFT order, every frequency
    U: Tensor  # (P, K) the latent frame the monitor uses
    ell: int  # hidden kernel length scale in sites
    B: int
    keep: Tensor  # (N,) bool, the frequencies the statistics use (`unfloored_frequencies`)


def unfloored_frequencies(cfg: Config, dataset: Dataset) -> Tensor:
    """(N,) bool in FFT order: the q whose exact hidden eigenvalue lambda_q is at least the
    penalty path's floor min_rel_eig * max_q lambda_q, the same mask on both paths (the
    whitened path has no floor, but its spectra are compared with the penalty path's)."""
    lam = torch.from_numpy(network_parameters(cfg, dataset, exact_kernel=True).lam)
    return lam >= cfg.spatial.min_rel_eig * lam.max()


def predicted_law(exp: ExpConfig, index: int) -> Law:
    """Setting `index` of `exp`: its config, data, predicted spectra, frame and kept band."""
    cfg = setting_config(exp, expand(exp)[index])
    dataset = make_dataset(cfg.data, CPU, F64)
    sites = cfg.spatial.hidden.kernel.length_scale * cfg.net.N / cfg.spatial.box
    if sites != int(sites) or sites < 1:
        raise ValueError(f"the hidden length scale is {sites} sites; R(ell) needs a whole lag")
    return Law(
        cfg=cfg,
        dataset=dataset,
        D=torch.from_numpy(predicted_spectra(cfg, dataset).d),
        U=latent_basis(dataset.factors).U,
        ell=int(sites),
        B=cfg.net.B,
        keep=unfloored_frequencies(cfg, dataset),
    )


def statistic_list(K: int) -> list[Statistic]:
    """The statistics in the order of `ring_statistics`' vector: each kind of KINDS for modes
    1..K, then the ratio."""
    return [Statistic(kind, k) for kind in KINDS for k in range(1, K + 1)] + [
        Statistic("ratio", None)
    ]


def ring_statistics(N: int, ell: int, keep: Tensor | None = None) -> Callable[[Tensor], Tensor]:
    """The statistics of mean spectra D (..., K, N) on a ring of N sites as a vector
    (..., 3K + 1) in the order of `statistic_list`, after zeroing the frequencies outside
    `keep` (default: all kept)."""
    in_band = band_mask(N)
    kept = torch.ones(N, dtype=torch.bool) if keep is None else keep

    def statistics(D: Tensor) -> Tensor:
        if D.shape[-1] != N:
            raise ValueError(f"spectra have {D.shape[-1]} frequencies, the ring has N = {N}")
        D = D * kept.to(D.dtype)
        R = autocorrelation(D)
        band = D[..., in_band].sum(-1)
        return torch.cat(
            [
                R[..., 1] / R[..., 0],
                R[..., ell] / R[..., 0],
                band / D.sum(-1),
                (band[..., 0] / band[..., 1]).unsqueeze(-1),
            ],
            dim=-1,
        )

    return statistics


def law_statistics(law: Law) -> Callable[[Tensor], Tensor]:
    """`ring_statistics` of the law's ring, length scale and kept frequencies."""
    return ring_statistics(law.D.shape[-1], law.ell, law.keep)


def member_covariance(D: Tensor) -> Tensor:
    """(K, N, N): the covariance of one member's power d_k(q) under the Gaussian law of mean
    spectrum D (K, N): D(q) D(q') ([q' = q] + [q' = -q]), so 2 D(q)^2 at q = 0 and N/2 (chi^2_1)
    and D(q)^2 between q and -q elsewhere (one Exp(1) variate shared by both)."""
    N = D.shape[-1]
    q = torch.arange(N)
    pairs = (q[:, None] == q[None, :]).to(D.dtype) + (q[:, None] == (-q[None, :]) % N).to(D.dtype)
    return D[:, :, None] * D[:, None, :] * pairs


def delta_method_se(law: Law, B: int | None = None) -> Tensor:
    """(3K + 1,): the standard error of each statistic of the mean of B members (the law's B
    unless given) drawn from the predicted law, sqrt(J Sigma J^T / B) with J the Jacobian of
    the statistics at the prediction and Sigma the block-diagonal `member_covariance`."""
    B = law.B if B is None else B
    J = torch.autograd.functional.jacobian(law_statistics(law), law.D)  # (3K + 1, K, N)
    variance = torch.einsum("skq,kqr,skr->s", J, member_covariance(law.D), J)
    return (variance / B).sqrt()


def estimate(member_spectra: Tensor, law: Law, seed: int = 0) -> tuple[Tensor, Tensor]:
    """Statistics of the member mean of (B, K, N) spectra and their bootstrap standard errors."""
    return bootstrap_members(member_spectra, law_statistics(law), N_BOOT, seed)


def map_correlation(first: Tensor, last: Tensor) -> Tensor:
    """(B,): per member, the correlation of its tuning maps (B, K, N) at two times, each mode
    normalised to unit power and the K modes averaged. Independent maps give 0 on average and a
    frozen sampler 1."""
    per_mode = (first * last).sum(-1) / (first.norm(dim=-1) * last.norm(dim=-1))
    return per_mode.mean(-1)


# ------------------------------------------------------------------ relaxation diagnostic
class RelaxationFit(NamedTuple):
    asymptote: Tensor  # (S,) A of y(t) = A - C exp(-t / tau)
    amplitude: Tensor  # (S,) C
    tau: Tensor  # (S,) in the units of the epochs
    deficit: Tensor  # (S,) C exp(-t_end / tau), what is still to relax at the last record
    deficit_se: Tensor  # (S,) bootstrap over members (half the 16-84% range)


# tau is searched on a log grid from TAU_MIN_FRACTION of the record spacing to TAU_MAX_FACTOR
# times the span of the records. A relaxation slower than the span cannot be told from a drift
# over it, and letting tau run past the span makes the extrapolated deficit of a stationary,
# noisy series arbitrarily large; at the cap a slow relaxation shows as a drift, fitted with
# tau = span and the deficit that leaves (an underestimate of what remains, but not zero).
TAU_GRID = 200
TAU_MIN_FRACTION = 0.25
TAU_MAX_FACTOR = 1.0


def _fit_relaxation(t: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Least-squares A, C, tau of y (R, S) = A - C exp(-t / tau), t (R,): for each tau of the
    grid the model is linear in (A, C), so it is solved exactly and the best tau kept."""
    span = float(t[-1] - t[0])
    spacing = float((t[1:] - t[:-1]).min())
    taus = torch.logspace(
        math.log10(TAU_MIN_FRACTION * spacing),
        math.log10(TAU_MAX_FACTOR * span),
        TAU_GRID,
        dtype=F64,
    )
    basis = torch.stack(
        [torch.ones_like(t).expand(TAU_GRID, -1), -torch.exp(-(t - t[0]) / taus[:, None])], dim=-1
    )  # (G, R, 2)
    coef = torch.linalg.lstsq(basis, y.expand(TAU_GRID, *y.shape)).solution  # (G, 2, S)
    residual = (basis @ coef - y).square().sum(-2)  # (G, S)
    best = residual.argmin(0)  # (S,)
    columns = torch.arange(y.shape[-1])
    A, C = coef[best, 0, columns], coef[best, 1, columns]
    return A, C, taus[best]


def relaxation_fit(
    epochs: Tensor,
    records: Tensor,
    statistics: Callable[[Tensor], Tensor],
    start: float | None = None,
    n_boot: int = 200,
) -> RelaxationFit:
    """Fit y(t) = A - C exp(-(t - t_0) / tau) to the member-averaged statistics of the records
    from epoch `start` on (all records if None).

    `records` (R, B, K, N) are the members' spectra at `epochs` (R,); `statistics` maps a mean
    spectrum (..., K, N) to (..., S). The deficit C exp(-(t_end - t_0) / tau) is what the fitted
    relaxation leaves at the last record. Fitting all records gives the time constant of the
    main relaxation; to test the end of a run, fit a late window (`start`), where a relaxation
    still under way shows as a drift that extrapolates to a deficit. The deficit's SE is the
    larger of half the 16-84% range of the refitted deficit over `n_boot` resamples of the
    members (the members are independent chains, so no autocorrelation time enters) and the
    bootstrap SE of the window's mean statistic: a fit whose tau falls inside the first record
    spacing puts the deficit at exactly 0 in every resample, and the floor keeps it from
    claiming more precision than the window's data has.
    """
    window = torch.ones_like(epochs, dtype=torch.bool) if start is None else epochs >= start
    if int(window.sum()) < 4:
        raise ValueError(f"need at least 4 records to fit a relaxation, got {int(window.sum())}")
    t = epochs[window].to(F64)
    records = records[window]
    y = statistics(records.mean(1))  # (R, S)
    A, C, tau = _fit_relaxation(t, y)
    end = float(t[-1] - t[0])
    deficit = C * torch.exp(-end / tau)
    B = records.shape[1]
    gen = torch.Generator().manual_seed(0)
    deficits, means = [], []
    for _ in range(n_boot):
        resampled = statistics(records[:, torch.randint(B, (B,), generator=gen)].mean(1))
        _, Cb, taub = _fit_relaxation(t, resampled)
        deficits.append(Cb * torch.exp(-end / taub))
        means.append(resampled.mean(0))
    # Half the 16-84% range, the SD of a normal: a resample whose tau runs to the end of the
    # grid makes the plain SD meaningless while leaving the quantiles alone.
    q16, q84 = torch.quantile(torch.stack(deficits), torch.tensor([0.16, 0.84], dtype=F64), dim=0)
    floor = torch.stack(means).std(0)
    return RelaxationFit(A, C, tau, deficit, torch.maximum((q84 - q16) / 2, floor))


def is_stationary(fit: RelaxationFit, se_max_b: Tensor) -> Tensor:
    """(S,) bool: the fitted deficit at the last record is below one standard error of the
    statistic at B_max, the largest B that will be run (the accuracy budget of a regime), and
    the fit can tell: its own SE is below that too (otherwise a small deficit is no evidence)."""
    return (fit.deficit.abs() < se_max_b) & (fit.deficit_se < se_max_b)


# ------------------------------------------------------------------ comparison rules
# The experiments' rule: an agreement test fails when |observed - predicted| exceeds FALSIFY_N_SE
# bootstrap SEs at the run's B, and a failure counts only if it persists after the setting is
# extended to 4B members. A regime is accurate when |exact - predicted| is at most one SE at
# B_max, the largest B run, the fourfold extension included.
FALSIFY_N_SE = 3.0
FOLLOW_UP_FACTOR = 4
# The pipeline test's rule, deliberately different: its tolerance is fixed by the predicted law
# rather than by the measured SE, PIPELINE_N_SE times PIPELINE_SE_MARGIN delta-method SEs, and its
# bootstrap SE must stay below PIPELINE_SE_MARGIN delta-method SEs (the margin covers the spread
# of a bootstrap SE around the true one).
PIPELINE_SE_MARGIN = 1.35
PIPELINE_N_SE = 3.0


def disagrees(values: Tensor, se: Tensor, predicted: Tensor) -> Tensor:
    """(S,) bool: the experiments' agreement test fails (|observed - predicted| > 3 SE)."""
    return (values - predicted).abs() > FALSIFY_N_SE * se


def pipeline_bounds(law: Law) -> tuple[Tensor, Tensor]:
    """(largest accepted bootstrap SE, tolerance), each (3K + 1,), of the pipeline test."""
    se_max = PIPELINE_SE_MARGIN * delta_method_se(law)
    return se_max, PIPELINE_N_SE * se_max
