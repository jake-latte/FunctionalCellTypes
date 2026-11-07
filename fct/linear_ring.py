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

The figures of a setting (`measure_final`, `measure_relaxation`) are built from the same
member-averaged final-state spectrum: the folded spectra, the mode ratio and the lag profile
R(Delta)/R(0) per |q| or lag, each with its bootstrap SE, and the secondary statistics (q_1/2
and the q = 0 share, which rest on one or two frequencies and are reported, not falsifiers).

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
from fct.linear_theory import network_parameters, noise_floor, predicted_spectra, theory_rate
from fct.spectrum import (
    autocorrelation,
    band_mask,
    bootstrap_members,
    bootstrap_resamples,
    fold_q,
    half_power_q,
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
    asymptote: Tensor  # (S,) A of y(t) = A - C exp(-(t - t_0) / tau)
    amplitude: Tensor  # (S,) C
    tau: Tensor  # (S,) in the units of the epochs
    deficit: Tensor  # (S,) C exp(-(t_end - t_0) / tau), what is still to relax at the last record
    deficit_se: Tensor  # (S,) bootstrap half 16-84% range, floored at the window mean's SE


# tau is searched on a log grid from TAU_MIN_FRACTION of the record spacing to TAU_MAX_FACTOR
# times the span of the records. A relaxation slower than the span cannot be told from a drift
# over it, and letting tau run past the span makes the extrapolated deficit of a stationary,
# noisy series arbitrarily large; at the cap a slow relaxation shows as a drift, fitted with
# tau = span and the deficit that leaves (an underestimate of what remains, but not zero).
TAU_GRID = 200
TAU_MIN_FRACTION = 0.25
TAU_MAX_FACTOR = 1.0
# Resamples of the relaxation fit's bootstrap. Fewer than N_BOOT because each resample is a full
# grid fit and the resampled means take n_boot x R x K x N floats (about 250 MB at R = 200 records
# of linear_ring); the resampling error of a 16-84% half-range from 100 resamples is about 10%,
# that of the SD floor 1 / sqrt(2 x 100) = 7%.
RELAXATION_N_BOOT = 100


def _fit_relaxation(t: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Least-squares A, C, tau of y (R, S) = A - C exp(-(t - t_0) / tau), t (R,): for each tau
    of the grid the model is linear in (A, C), so it is solved exactly and the best tau kept."""
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


def _deficit_and_mean(t: Tensor, statistics: Callable[[Tensor], Tensor]) -> Callable:
    """The statistic `bootstrap_resamples` resamples: member-mean records (..., R, K, N) -> the
    fitted deficit and the window mean of every statistic, (..., 2S)."""
    end = float(t[-1] - t[0])

    def deficit_and_mean(mean_records: Tensor) -> Tensor:
        batch = mean_records.shape[:-3]
        flat = mean_records.reshape(-1, *mean_records.shape[-3:])
        rows = []
        for one in flat:  # one grid fit per resample
            y = statistics(one)  # (R, S)
            _, C, tau = _fit_relaxation(t, y)
            rows.append(torch.cat([C * torch.exp(-end / tau), y.mean(0)]))
        return torch.stack(rows).reshape(*batch, -1)

    return deficit_and_mean


def _bootstrap_deficit(
    t: Tensor, records: Tensor, statistics: Callable[[Tensor], Tensor], n_boot: int, seed: int
) -> Tensor:
    """(S,): the deficit's spread over resamples of the members, half the 16-84% range of the
    refitted deficits (the SD of a normal), floored at the SD of the window's mean statistic.
    The quantiles, not the SD: a few resamples whose tau runs to the edge of the grid set the
    SD (0.072 against a half-range of 0.0027 at tau = 400). The floor: a fit whose tau falls
    inside the first record spacing puts the deficit at exactly 0 in every resample, and the
    floor keeps it from claiming more precision than the window's data has."""
    members_first = records.transpose(0, 1)  # (B, R, K, N), resampled along the members
    resampled = bootstrap_resamples(members_first, _deficit_and_mean(t, statistics), n_boot, seed)
    S = resampled.shape[-1] // 2
    q16, q84 = torch.quantile(
        resampled[:, :S], torch.tensor([0.16, 0.84], dtype=resampled.dtype), dim=0
    )
    return torch.maximum((q84 - q16) / 2, resampled[:, S:].std(0))


def relaxation_fit(
    epochs: Tensor,
    records: Tensor,
    statistics: Callable[[Tensor], Tensor],
    start: float | None = None,
    n_boot: int = RELAXATION_N_BOOT,
    seed: int = 0,
) -> RelaxationFit:
    """Fit y(t) = A - C exp(-(t - t_0) / tau) to the member-averaged statistics of the records
    from epoch `start` on (all records if None).

    `records` (R, B, K, N) are the members' spectra at `epochs` (R,); `statistics` maps a mean
    spectrum (..., K, N) to (..., S). The deficit C exp(-(t_end - t_0) / tau) is what the fitted
    relaxation leaves at the last record. Fitting all records gives the time constant of the
    main relaxation (of its fast phase, if records are sparse beside a slower tail); to test the
    end of a run, fit a late window (`start`), where a relaxation still under way shows as a
    drift that extrapolates to a deficit. The deficit's SE (`_bootstrap_deficit`) comes from
    `n_boot` resamples of the members, seeded with `seed` (the members are independent chains,
    so no autocorrelation time enters).
    """
    window = torch.ones_like(epochs, dtype=torch.bool) if start is None else epochs >= start
    if int(window.sum()) < 4:
        raise ValueError(f"need at least 4 records to fit a relaxation, got {int(window.sum())}")
    t = epochs[window].to(F64)
    records = records[window]
    A, C, tau = _fit_relaxation(t, statistics(records.mean(1)))
    deficit = C * torch.exp(-float(t[-1] - t[0]) / tau)
    deficit_se = _bootstrap_deficit(t, records, statistics, n_boot, seed)
    return RelaxationFit(A, C, tau, deficit, deficit_se)


def budget_se(law: Law) -> Tensor:
    """(3K + 1,): one delta-method SE of each statistic at B_max = FOLLOW_UP_FACTOR x B, the
    largest B a setting is run at (its B and the follow-up extension): the accuracy budget of a
    regime and the bound a stationary run's deficit must stay under."""
    return delta_method_se(law, FOLLOW_UP_FACTOR * law.B)


def is_stationary(fit: RelaxationFit, budget: Tensor) -> Tensor:
    """(S,) bool: the fitted deficit at the last record is below `budget` (`budget_se` for a
    real run) and the fit can tell, its own SE below `budget` too (otherwise a small deficit is
    no evidence)."""
    return (fit.deficit.abs() < budget) & (fit.deficit_se < budget)


def regime_accurate(exact: Tensor, law: Law) -> Tensor:
    """(3K + 1,) bool: the exact posterior's statistics `exact` are within `budget_se` of the
    law's prediction, the plan's accuracy rule for a regime."""
    return (exact - law_statistics(law)(law.D)).abs() <= budget_se(law)


# ------------------------------------------------------------------ comparison rules
# The experiments' rule: an agreement test fails when |observed - predicted| exceeds FALSIFY_N_SE
# bootstrap SEs at the run's B, and a failure counts only if it persists after the setting is
# extended to FOLLOW_UP_FACTOR times B members. Two parametrisations are equivalent on a
# statistic unless their deviations from their own predictions differ by more than
# EQUIVALENCE_N_SE combined bootstrap SEs (the plan's level, with the same follow-up). Paths that
# share the initial state have correlated members: that is conservative against a false
# disagreement but lowers the test's power, so passing is weaker evidence of equivalence. Here
# it is negligible, because the final state has forgotten the initial one (member-mean map
# correlations +0.007 and -0.034 on the pipeline ring).
FALSIFY_N_SE = 3.0
FOLLOW_UP_FACTOR = 4
EQUIVALENCE_N_SE = 3.0
# The pipeline test's rule, deliberately different: its tolerance is fixed by the predicted law
# rather than by the measured SE, PIPELINE_N_SE times PIPELINE_SE_MARGIN delta-method SEs, and its
# bootstrap SE must stay below PIPELINE_SE_MARGIN delta-method SEs (the margin covers the spread
# of a bootstrap SE around the true one).
PIPELINE_SE_MARGIN = 1.35
PIPELINE_N_SE = 3.0


def disagrees(values: Tensor, se: Tensor, predicted: Tensor) -> Tensor:
    """(S,) bool: the experiments' agreement test fails, |observed - predicted| beyond
    FALSIFY_N_SE bootstrap SEs."""
    return (values - predicted).abs() > FALSIFY_N_SE * se


def paths_disagree(
    first: tuple[Tensor, Tensor, Tensor], second: tuple[Tensor, Tensor, Tensor]
) -> Tensor:
    """(S,) bool: two paths' (values, bootstrap SE, own prediction) disagree, their deviations
    from their own predictions differing by more than EQUIVALENCE_N_SE combined SEs."""
    (v1, se1, p1), (v2, se2, p2) = first, second
    return ((v1 - p1) - (v2 - p2)).abs() > EQUIVALENCE_N_SE * (se1.square() + se2.square()).sqrt()


def pipeline_bounds(law: Law) -> tuple[Tensor, Tensor]:
    """(largest accepted bootstrap SE, tolerance), each (3K + 1,), of the pipeline test:
    PIPELINE_SE_MARGIN and PIPELINE_N_SE x PIPELINE_SE_MARGIN delta-method SEs."""
    se_max = PIPELINE_SE_MARGIN * delta_method_se(law)
    return se_max, PIPELINE_N_SE * se_max


# ------------------------------------------------------------------ the figures' measurements
# Secondary criteria (reported beside the falsifiers, never falsifiers themselves): q_1/2 is
# flagged when it differs from the prediction by more than Q_HALF_TOLERANCE, a q = 0 share by
# `disagrees`, and the hexatic modulus |mean_b m_b| when it exceeds HEXATIC_MAX (prediction 6:
# rotation-invariant tuning has |m| near 0).
Q_HALF_TOLERANCE = 1
HEXATIC_MAX = 0.1
SECONDARY_KINDS = ("q_half", "share0")


def statistic_label(stat: Statistic) -> str:
    """A short name: the kind and the 1-based mode (`lag1_1`), or `ratio`."""
    return stat.kind if stat.mode is None else f"{stat.kind}_{stat.mode}"


def _kept(D: Tensor, keep: Tensor | None) -> Tensor:
    return D if keep is None else D * keep.to(D.dtype)


def lag_profile(D: Tensor, keep: Tensor | None = None) -> Tensor:
    """(..., N/2 + 1): R(Delta)/R(0) at Delta = 0..N/2 of spectra D (..., N), after zeroing the
    frequencies outside `keep`. R is even in Delta on the ring, so half of it is all of it."""
    R = autocorrelation(_kept(D, keep))
    return (R / R[..., :1])[..., : D.shape[-1] // 2 + 1]


def folded_mode_ratio(D: Tensor) -> Tensor:
    """(..., N/2 + 1): the folded power of mode 1 over that of mode 2, D (..., K, N), K >= 2."""
    if D.ndim < 2 or D.shape[-2] < 2:
        raise ValueError(f"the mode ratio needs spectra (..., K >= 2, N), got {tuple(D.shape)}")
    folded = fold_q(D)
    return folded[..., 0, :] / folded[..., 1, :]


def prior_profile(law: Law) -> Tensor:
    """(N/2 + 1,): `lag_profile` of the hidden prior's exact eigenvalues lambda_q over the kept
    band: the shape every mode would have if the data did not act on it."""
    lam = torch.from_numpy(network_parameters(law.cfg, law.dataset, exact_kernel=True).lam)
    return lag_profile(lam, law.keep)


def secondary_names(K: int) -> list[str]:
    """The names of `secondary_statistics`' vector: q_1/2 of modes 1..K, then their q = 0 share."""
    return [f"{kind}_{k}" for kind in SECONDARY_KINDS for k in range(1, K + 1)]


def secondary_statistics(N: int, keep: Tensor | None = None) -> Callable[[Tensor], Tensor]:
    """Mean spectra D (..., K, N) -> (..., 2K): `half_power_q` of each mode, then its q = 0
    share, after zeroing the frequencies outside `keep` (so the half-power floor d_{N/2} is 0
    where the floor excludes N/2, in the prediction and the measurement alike)."""

    def statistics(D: Tensor) -> Tensor:
        if D.shape[-1] != N:
            raise ValueError(f"spectra have {D.shape[-1]} frequencies, the ring has N = {N}")
        D = _kept(D, keep)
        q_half = torch.as_tensor(half_power_q(D), device=D.device).to(D.dtype)
        return torch.cat([q_half, D[..., 0] / D.sum(-1)], dim=-1)

    return statistics


def hexatic_estimate(moments: Tensor, seed: int = 0) -> tuple[Tensor, Tensor]:
    """|mean_b m_b| of the members' complex hexatic moments (B, 2) (real, imaginary parts, as
    the spectrum monitor records them) and its bootstrap SE over members: the complex moment is
    what averages, not its modulus."""
    if moments.ndim != 2 or moments.shape[-1] != 2:
        raise ValueError(
            f"hexatic moments must be (B, 2), real and imaginary part per member, got "
            f"{tuple(moments.shape)}"
        )
    return bootstrap_members(moments, lambda m: torch.linalg.vector_norm(m, dim=-1), N_BOOT, seed)


def latent_angles(Z: Tensor, U: Tensor) -> Tensor:
    """(6,): the contrast-plane angles atan2(u_3 . z_j, u_2 . z_j) of the three latents and of
    their opposites, where a neuron whose activity is +-z_j lies (the marks of prediction 6)."""
    if Z.shape[-1] != 3 or U.shape[-1] != 3:
        raise ValueError(
            f"the contrast plane is the whole contrast space only at K = 3, got Z "
            f"{tuple(Z.shape)} and U {tuple(U.shape)}"
        )
    c = U[:, 1:3].T @ Z  # (2, 3): each latent's coordinates in (u_2, u_3)
    return torch.cat([torch.atan2(c[1], c[0]), torch.atan2(-c[1], -c[0])])


def relaxation_curve(fit: RelaxationFit, t: Tensor, t0: float) -> Tensor:
    """(T, S): the fitted A - C exp(-(t - t0) / tau) at epochs t (T,), t0 the window's first."""
    return fit.asymptote - fit.amplitude * torch.exp(-(t[:, None] - t0) / fit.tau)


class FinalMeasurement(NamedTuple):
    """One setting read off its members' final states: the statistics against the prediction,
    and the per-|q| or per-lag curves the figures draw, each with a bootstrap SE over members.
    Q = N/2 + 1 folded frequencies (or lags)."""

    stats: list[Statistic]
    predicted: Tensor  # (3K + 1,) of the predicted spectrum
    values: Tensor  # (3K + 1,) of the member-averaged final-state spectrum
    se: Tensor  # (3K + 1,)
    disagree: Tensor  # (3K + 1,) bool, `disagrees`
    spectrum: Tensor  # (K + 1, Q) folded member-mean spectra, the null row last
    spectrum_se: Tensor  # (K + 1, Q)
    spectrum_theory: Tensor  # (K, Q)
    null_theory: float  # the predicted null floor (0 at sigma = 0)
    excluded: Tensor  # (Q,) bool, the folded frequencies the statistics leave out
    ratio: Tensor  # (Q,) folded mode 1 / mode 2
    ratio_se: Tensor
    ratio_theory: Tensor
    profile: Tensor  # (K, Q) R(Delta)/R(0) over the kept band
    profile_se: Tensor
    profile_theory: Tensor
    kernel_profile: Tensor  # (Q,) `prior_profile`
    secondary_names: list[str]
    secondary: Tensor  # (2K,)
    secondary_se: Tensor
    secondary_predicted: Tensor
    secondary_disagree: Tensor  # (2K,) bool
    hexatic: tuple[Tensor, Tensor] | None  # |mean m| and its SE, when the moments were given
    B: int


def measure_final(
    law: Law, final: Tensor, hex_moments: Tensor | None = None, seed: int = 0
) -> FinalMeasurement:
    """Everything the notebook shows of one setting's final states: `final` (B, K + 1, N) the
    members' last records (`mode_spectra`, the null row last), and optionally their hexatic
    moments (B, 2). Every curve and statistic is of the member mean, with the bootstrap over
    members seeded by `seed`."""
    K, N = law.D.shape
    if final.ndim != 3 or tuple(final.shape[1:]) != (K + 1, N):
        raise ValueError(
            f"final spectra must be (B, K + 1, N) = (B, {K + 1}, {N}), got {tuple(final.shape)}"
        )
    final = final.to(F64)
    latent = final[:, :K]
    values, se = estimate(latent, law, seed)
    predicted = law_statistics(law)(law.D)
    spectrum, spectrum_se = bootstrap_members(final, fold_q, N_BOOT, seed)
    ratio, ratio_se = bootstrap_members(latent, folded_mode_ratio, N_BOOT, seed)
    profile, profile_se = bootstrap_members(
        latent, lambda D: lag_profile(D, law.keep), N_BOOT, seed
    )
    secondary = secondary_statistics(N, law.keep)
    secondary_values, secondary_se = bootstrap_members(latent, secondary, N_BOOT, seed)
    secondary_predicted = secondary(law.D)
    secondary_disagree = torch.cat(
        [
            (secondary_values[:K] - secondary_predicted[:K]).abs() > Q_HALF_TOLERANCE,
            disagrees(secondary_values[K:], secondary_se[K:], secondary_predicted[K:]),
        ]
    )
    return FinalMeasurement(
        stats=statistic_list(K),
        predicted=predicted,
        values=values,
        se=se,
        disagree=disagrees(values, se, predicted),
        spectrum=spectrum,
        spectrum_se=spectrum_se,
        spectrum_theory=fold_q(law.D),
        null_theory=noise_floor(0.0, theory_rate(law.cfg)),
        excluded=~law.keep[: N // 2 + 1],
        ratio=ratio,
        ratio_se=ratio_se,
        ratio_theory=folded_mode_ratio(law.D),
        profile=profile,
        profile_se=profile_se,
        profile_theory=lag_profile(law.D, law.keep),
        kernel_profile=prior_profile(law),
        secondary_names=secondary_names(K),
        secondary=secondary_values,
        secondary_se=secondary_se,
        secondary_predicted=secondary_predicted,
        secondary_disagree=secondary_disagree,
        hexatic=None if hex_moments is None else hexatic_estimate(hex_moments.to(F64), seed),
        B=final.shape[0],
    )


class RelaxationView(NamedTuple):
    """The burn-in diagnostic of one setting, every statistic relative to its fitted asymptote
    in units of its accuracy budget (`budget_se`), so all of them share one axis."""

    fit: RelaxationFit
    epochs: Tensor  # (R,) the records'
    values: Tensor  # (R, S) (statistic of the member-mean record - asymptote) / budget
    fit_epochs: Tensor  # (T,) the fit window's
    fitted: Tensor  # (T, S) the fitted curve, normalised the same way
    labels: list[str]
    start: float  # the window's start epoch
    budget: Tensor  # (S,)
    stationary: Tensor  # (S,) bool, `is_stationary`


def measure_relaxation(
    law: Law, epochs: Tensor, records: Tensor, start: float | None = None, seed: int = 0
) -> RelaxationView:
    """`relaxation_fit` of the records (R, B, K, N) at `epochs` (R,) from epoch `start` on
    (default: the second half of the run, which tests its end), and the series to draw."""
    start = law.cfg.train.n_epochs / 2 if start is None else float(start)
    statistics = law_statistics(law)
    epochs, records = epochs.to(F64), records.to(F64)
    fit = relaxation_fit(epochs, records, statistics, start, seed=seed)
    budget = budget_se(law)
    fit_epochs = epochs[epochs >= start]
    curve = relaxation_curve(fit, fit_epochs, float(fit_epochs[0]))
    return RelaxationView(
        fit=fit,
        epochs=epochs,
        values=(statistics(records.mean(1)) - fit.asymptote) / budget,
        fit_epochs=fit_epochs,
        fitted=(curve - fit.asymptote) / budget,
        labels=[statistic_label(s) for s in statistic_list(law.D.shape[0])],
        start=start,
        budget=budget,
        stationary=is_stationary(fit, budget),
    )
