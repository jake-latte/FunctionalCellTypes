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
from fct.linear_theory import network_parameters, predicted_spectra
from fct.spectrum import (
    N_THETA_BINS,
    HalfPowerUndefinedError,
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
    B: int  # the experiment's B now (grown by `exp extend`), not the base B of `budget_se`
    keep: Tensor  # (N,) bool, the frequencies the statistics use (`unfloored_frequencies`)
    null_floor: float  # the predicted power of each null direction (0 at sigma = 0)


def exact_eigenvalues(cfg: Config, dataset: Dataset) -> Tensor:
    """(N,) in FFT order: the hidden kernel's exact eigenvalues lambda_q, without the penalty
    path's floor."""
    return torch.from_numpy(network_parameters(cfg, dataset, exact_kernel=True).lam)


def unfloored_frequencies(cfg: Config, dataset: Dataset) -> Tensor:
    """(N,) bool in FFT order: the q whose exact hidden eigenvalue lambda_q is at least the
    penalty path's floor min_rel_eig * max_q lambda_q, the same mask on both paths (the
    whitened path has no floor, but its spectra are compared with the penalty path's)."""
    lam = exact_eigenvalues(cfg, dataset)
    return lam >= cfg.spatial.min_rel_eig * lam.max()


def predicted_law(exp: ExpConfig, index: int) -> Law:
    """Setting `index` of `exp`: its config, data, predicted spectra, frame and kept band."""
    cfg = setting_config(exp, expand(exp)[index])
    dataset = make_dataset(cfg.data, CPU, F64)
    sites = cfg.spatial.hidden.kernel.length_scale * cfg.net.N / cfg.spatial.box
    if sites != int(sites) or sites < 1:
        raise ValueError(f"the hidden length scale is {sites} sites; R(ell) needs a whole lag")
    prediction = predicted_spectra(cfg, dataset)
    return Law(
        cfg=cfg,
        dataset=dataset,
        D=torch.from_numpy(prediction.d),
        U=latent_basis(dataset.factors).U,
        ell=int(sites),
        B=cfg.net.B,
        keep=unfloored_frequencies(cfg, dataset),
        null_floor=prediction.null_floor,
    )


def statistic_list(K: int) -> list[Statistic]:
    """The statistics in the order of `ring_statistics`' vector: each kind of KINDS for modes
    1..K, then the ratio."""
    return [Statistic(kind, k) for kind in KINDS for k in range(1, K + 1)] + [
        Statistic("ratio", None)
    ]


def kept(D: Tensor, keep: Tensor | None = None) -> Tensor:
    """Spectra D (..., N) with the frequencies outside `keep` (N,) zeroed (all kept if None)."""
    return D if keep is None else D * keep.to(D.dtype)


def normalised_autocorrelation(D: Tensor) -> Tensor:
    """(..., N): R(Delta)/R(0) at every lag Delta = 0..N-1 of spectra D (..., N)."""
    R = autocorrelation(D)
    return R / R[..., :1]


def ring_statistics(N: int, ell: int, keep: Tensor | None = None) -> Callable[[Tensor], Tensor]:
    """The statistics of mean spectra D (..., K, N) on a ring of N sites as a vector
    (..., 3K + 1) in the order of `statistic_list`, after zeroing the frequencies outside
    `keep` (default: all kept)."""
    in_band = band_mask(N)

    def statistics(D: Tensor) -> Tensor:
        if D.shape[-1] != N:
            raise ValueError(f"spectra have {D.shape[-1]} frequencies, the ring has N = {N}")
        D = kept(D, keep)
        rho = normalised_autocorrelation(D)
        band = D[..., in_band].sum(-1)
        return torch.cat(
            [
                rho[..., 1],
                rho[..., ell],
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


def budget_se(law: Law, base_B: int) -> Tensor:
    """(3K + 1,): one delta-method SE of each statistic at B_max = FOLLOW_UP_FACTOR x base_B,
    the largest B a setting is run at (its base B and the follow-up extension): the accuracy
    budget of a regime and the bound a stationary run's deficit must stay under. `base_B` is
    the B the experiment was designed with (its preset's), given explicitly: the B in exp.json
    grows with `exp extend`, and the follow-up must not move its own budget."""
    if base_B < 1:
        raise ValueError(f"base_B must be a positive number of members, got {base_B}")
    return delta_method_se(law, FOLLOW_UP_FACTOR * base_B)


def is_stationary(fit: RelaxationFit, budget: Tensor) -> Tensor:
    """(S,) bool: the fitted deficit at the last record is below `budget` (`budget_se` for a
    real run) and the fit can tell, its own SE below `budget` too (otherwise a small deficit is
    no evidence)."""
    return (fit.deficit.abs() < budget) & (fit.deficit_se < budget)


def regime_accurate(exact: Tensor, law: Law, base_B: int) -> Tensor:
    """(3K + 1,) bool: the exact posterior's statistics `exact` are within `budget_se` (at
    `base_B`) of the law's prediction, the plan's accuracy rule for a regime."""
    return (exact - law_statistics(law)(law.D)).abs() <= budget_se(law, base_B)


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
# Secondary criteria, reported beside the falsifiers and never falsifiers themselves.
# Q_HALF_TOLERANCE: q_1/2 is flagged when it differs from the prediction by more than one step
# of the frequency grid (the plan's secondary rule for q_1/2; a single frequency moves it).
Q_HALF_TOLERANCE = 1
# HEXATIC_MAX: prediction 6 is falsified when |mean_b m_b| exceeds it with N B >= 10^3
# neuron-samples, the plan's threshold (decided by the PI). Uniform angles give |m| of order
# 1 / sqrt(N B) (0.03 at N B = 10^3, less with B, but more when neighbours are correlated),
# well below it.
HEXATIC_MAX = 0.1
# HEXATIC_MAX is applied only with at least this many neuron-samples N B: uniform angles give
# |m| of order 1 / sqrt(N B), 0.03 at N B = 10^3 and 0.06 at 256, so below it the bias of a
# rotation-invariant map is too close to HEXATIC_MAX to call a modulus above it tuning.
HEXATIC_MIN_SAMPLES = 1000
# The angle statistics exist only at K = 3: then (u_2, u_3) is the whole contrast space and the
# three latents project onto it 120 degrees apart; at K > 3 the plane is an arbitrary one of a
# near-degenerate contrast space (`fct.spectrum.hexatic_moment`).
ANGLE_K = 3
SECONDARY_KINDS = ("q_half", "share0")
# The relaxation figure draws every statistic in units of its budget, so the band is +-1.
RELAXATION_BAND = 1.0

_NAMES = {
    "lag1": "R(1)/R(0)",
    "lagl": "R({ell})/R(0)",
    "band": "band share",
    "ratio": "band power ratio",
    "q_half": "q_1/2",
    "share0": "q = 0 share",
    "hexatic": "|mean m| (hexatic)",
}
_MODES = {"ratio": "1/2", "hexatic": "2, 3"}


def statistic_name(stat: Statistic, ell: int) -> str:
    """The one name of a statistic every table and legend uses, `R(2)/R(0)` say."""
    return _NAMES[stat.kind].format(ell=ell)


def statistic_mode(stat: Statistic) -> str:
    """The mode(s) a statistic is of: `1`, or `1/2` for the ratio, `2, 3` for the hexatic."""
    return _MODES.get(stat.kind, str(stat.mode))


def statistic_label(stat: Statistic, ell: int) -> str:
    """Name and mode in one string, for a legend: `R(1)/R(0), mode 2`."""
    return f"{statistic_name(stat, ell)}, mode {statistic_mode(stat)}"


def lag_profile(D: Tensor, keep: Tensor | None = None) -> Tensor:
    """(..., N/2 + 1): R(Delta)/R(0) at Delta = 0..N/2 of spectra D (..., N), after zeroing the
    frequencies outside `keep`. R is even in Delta on the ring, so half of it is all of it."""
    return normalised_autocorrelation(kept(D, keep))[..., : D.shape[-1] // 2 + 1]


def folded_mode_ratio(D: Tensor) -> Tensor:
    """(..., N/2 + 1): the folded power of mode 1 over that of mode 2, D (..., K, N), K >= 2."""
    if D.ndim < 2 or D.shape[-2] < 2:
        raise ValueError(f"the mode ratio needs spectra (..., K >= 2, N), got {tuple(D.shape)}")
    folded = fold_q(D)
    return folded[..., 0, :] / folded[..., 1, :]


def prior_profile(law: Law) -> Tensor:
    """(N/2 + 1,): `lag_profile` of the hidden prior's exact eigenvalues over the kept band:
    the shape every mode would have if the data did not act on it."""
    return lag_profile(exact_eigenvalues(law.cfg, law.dataset), law.keep)


def secondary_list(K: int) -> list[Statistic]:
    """The secondary statistics: q_1/2 of modes 1..K, then their q = 0 shares."""
    return [Statistic(kind, k) for kind in SECONDARY_KINDS for k in range(1, K + 1)]


def q_half_statistic(keep: Tensor | None = None) -> Callable[[Tensor], Tensor]:
    """Mean spectra D (..., K, N) -> (..., K): `half_power_q` of each mode over the kept band
    (so the floor d_{N/2} is 0 where the penalty floor excludes N/2, in the prediction and the
    measurement alike). Raises, as `half_power_q` does, on a spectrum rising to Nyquist."""

    def statistic(D: Tensor) -> Tensor:
        D = kept(D, keep)
        return torch.as_tensor(half_power_q(D), device=D.device).to(D.dtype)

    return statistic


def share0_statistic(keep: Tensor | None = None) -> Callable[[Tensor], Tensor]:
    """Mean spectra D (..., K, N) -> (..., K): the q = 0 share of each mode over the kept band."""

    def statistic(D: Tensor) -> Tensor:
        D = kept(D, keep)
        return D[..., 0] / D.sum(-1)

    return statistic


def band_indices(K: int) -> list[int]:
    """The positions of the K band shares in `statistic_list(K)`."""
    return [i for i, stat in enumerate(statistic_list(K)) if stat.kind == "band"]


def band_share_estimate(law: Law, final: Tensor, seed: int = 0) -> tuple[Tensor, Tensor, Tensor]:
    """(values, bootstrap SE, predicted), each (K,): the band shares of the member-averaged
    final-state spectrum of latent rows `final` (B, K, N)."""
    band = band_indices(law.D.shape[0])
    values, se = estimate(final, law, seed)
    return values[band], se[band], law_statistics(law)(law.D)[band]


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
    if Z.shape[-1] != ANGLE_K or U.shape[-1] != ANGLE_K:
        raise ValueError(
            f"the contrast plane is the whole contrast space only at K = {ANGLE_K}, got Z "
            f"{tuple(Z.shape)} and U {tuple(U.shape)}"
        )
    c = U[:, 1:3].T @ Z  # (2, 3): each latent's coordinates in (u_2, u_3)
    return torch.cat([torch.atan2(c[1], c[0]), torch.atan2(-c[1], -c[0])])


def angle_marks(law: Law) -> Tensor:
    """(6,): `latent_angles` of the setting's own latents in its monitor's frame."""
    basis = latent_basis(law.dataset.factors)
    return latent_angles(basis.Z, basis.U)


def relaxation_curve(fit: RelaxationFit, t: Tensor, t0: float) -> Tensor:
    """(T, S): the fitted A - C exp(-(t - t0) / tau) at epochs t (T,), t0 the window's first."""
    return fit.asymptote - fit.amplitude * torch.exp(-(t[:, None] - t0) / fit.tau)


# The parts of a measurement. The curve parts' fields are the keyword arguments of the matching
# `fct.draw.plot_*`, so a figure is `plot_spectrum(**part._asdict(), ax=ax)`.
class Primary(NamedTuple):
    """The falsifiers: the ring-averaged statistics against the prediction."""

    stats: list[Statistic]
    predicted: Tensor  # (3K + 1,) of the predicted spectrum
    values: Tensor  # (3K + 1,) of the member-averaged final-state spectrum
    se: Tensor  # (3K + 1,) bootstrap over members
    disagree: Tensor  # (3K + 1,) bool, `disagrees`


class SpectrumCurves(NamedTuple):
    """`fct.draw.plot_spectrum`'s arguments: Q = N/2 + 1 folded frequencies."""

    measured: Tensor  # (K + 1, Q) folded member-mean spectra, the null row last
    se: Tensor  # (K + 1, Q)
    theory: Tensor  # (K, Q)
    null_theory: float  # the predicted null floor (`Law.null_floor`)
    excluded: Tensor  # (Q,) bool, the folded frequencies the statistics leave out


class RatioCurve(NamedTuple):
    """`fct.draw.plot_spectrum_ratio`'s arguments."""

    ratio: Tensor  # (Q,) folded mode 1 / mode 2
    se: Tensor
    theory: Tensor
    excluded: Tensor


class ProfileCurves(NamedTuple):
    """`fct.draw.plot_autocorrelation`'s arguments: R(Delta)/R(0) over the kept band."""

    measured: Tensor  # (K, Q)
    se: Tensor
    theory: Tensor
    kernel: Tensor  # (Q,) `prior_profile`


class Secondary(NamedTuple):
    """q_1/2 and the q = 0 shares; a statistic that could not be formed is left out of the
    tensors and named in `unavailable` with the reason."""

    stats: list[Statistic]  # the available ones, in `secondary_list` order
    predicted: Tensor
    values: Tensor
    se: Tensor
    disagree: Tensor  # bool: q_1/2 beyond Q_HALF_TOLERANCE, a share by `disagrees`
    unavailable: dict[str, str]  # statistic kind -> why it is not formed


class AngleHistogram(NamedTuple):
    """`fct.draw.plot_angle_histogram`'s arguments."""

    counts: Tensor  # (N_THETA_BINS,) summed over neurons and members
    marks: Tensor  # (6,) `angle_marks`


class Angles(NamedTuple):
    """Prediction 6 at K = 3: the hexatic modulus and the angle histogram."""

    hexatic: Tensor  # |mean_b m_b|
    hexatic_se: Tensor
    flagged: bool | None  # above HEXATIC_MAX; None below HEXATIC_MIN_SAMPLES neuron-samples
    n_samples: int  # N B
    histogram: AngleHistogram


class FinalMeasurement(NamedTuple):
    """One setting read off its members' final states, every part of the member mean with a
    bootstrap SE over members."""

    primary: Primary
    spectrum: SpectrumCurves
    ratio: RatioCurve
    profile: ProfileCurves
    secondary: Secondary
    angles: Angles | None  # None unless K = 3
    B: int


def _check_final(law: Law, final: Tensor) -> None:
    K, N = law.D.shape
    if final.ndim != 3 or tuple(final.shape[1:]) != (K + 1, N):
        raise ValueError(
            f"final spectra must be (B, K + 1, N) = (B, {K + 1}, {N}), got {tuple(final.shape)}"
        )
    if final.shape[0] < 2:
        raise ValueError(
            f"need at least 2 members for a bootstrap over members, got B = {final.shape[0]}"
        )


def measure_primary(law: Law, latent: Tensor, seed: int = 0) -> Primary:
    """The falsifiers of latent rows (B, K, N), `estimate` against the prediction."""
    values, se = estimate(latent, law, seed)
    predicted = law_statistics(law)(law.D)
    return Primary(statistic_list(law.D.shape[0]), predicted, values, se,
                   disagrees(values, se, predicted))  # fmt: skip


def measure_spectrum(law: Law, final: Tensor, seed: int = 0) -> SpectrumCurves:
    """The folded member-mean spectra of every row of `final` (B, K + 1, N), null row last."""
    measured, se = bootstrap_members(final, fold_q, N_BOOT, seed)
    N = law.D.shape[-1]
    return SpectrumCurves(measured, se, fold_q(law.D), law.null_floor, ~law.keep[: N // 2 + 1])


def measure_ratio(law: Law, latent: Tensor, seed: int = 0) -> RatioCurve:
    ratio, se = bootstrap_members(latent, folded_mode_ratio, N_BOOT, seed)
    N = law.D.shape[-1]
    return RatioCurve(ratio, se, folded_mode_ratio(law.D), ~law.keep[: N // 2 + 1])


def measure_profile(law: Law, latent: Tensor, seed: int = 0) -> ProfileCurves:
    def profile(D: Tensor) -> Tensor:
        return lag_profile(D, law.keep)

    measured, se = bootstrap_members(latent, profile, N_BOOT, seed)
    return ProfileCurves(measured, se, profile(law.D), prior_profile(law))


def measure_secondary(law: Law, latent: Tensor, seed: int = 0) -> Secondary:
    """q_1/2 (flagged beyond Q_HALF_TOLERANCE) and the q = 0 shares (by `disagrees`). q_1/2 is
    undefined where a mean spectrum rises to Nyquist, in the data or a bootstrap resample;
    it is then reported unavailable, with `half_power_q`'s reason, and the rest still formed."""
    K = law.D.shape[0]
    parts, unavailable = [], {}
    try:
        statistic = q_half_statistic(law.keep)
        values, se = bootstrap_members(latent, statistic, N_BOOT, seed)
        predicted = statistic(law.D)
        parts.append(
            ("q_half", values, se, predicted, (values - predicted).abs() > Q_HALF_TOLERANCE)
        )
    except HalfPowerUndefinedError as err:
        unavailable["q_half"] = str(err)
    statistic = share0_statistic(law.keep)
    values, se = bootstrap_members(latent, statistic, N_BOOT, seed)
    predicted = statistic(law.D)
    parts.append(("share0", values, se, predicted, disagrees(values, se, predicted)))
    return Secondary(
        stats=[Statistic(kind, k) for kind, *_ in parts for k in range(1, K + 1)],
        predicted=torch.cat([p[3] for p in parts]),
        values=torch.cat([p[1] for p in parts]),
        se=torch.cat([p[2] for p in parts]),
        disagree=torch.cat([p[4] for p in parts]),
        unavailable=unavailable,
    )


def measure_angles(
    law: Law, B: int, hex_moments: Tensor | None, theta_counts: Tensor | None, seed: int = 0
) -> Angles | None:
    """At K = 3, the hexatic modulus of the B members' moments (B, 2) and their summed angle
    histograms (B, N_THETA_BINS), both required; None at any other K, where the angle is not
    defined (`ANGLE_K`). The modulus is flagged against HEXATIC_MAX only with N B >=
    HEXATIC_MIN_SAMPLES neuron-samples."""
    if law.D.shape[0] != ANGLE_K:
        return None
    if hex_moments is None or theta_counts is None:
        raise ValueError(
            f"at K = {ANGLE_K} the measurement needs the members' final hexatic moments and "
            "angle histograms (the spectrum monitor's hex and theta_hist)"
        )
    if tuple(hex_moments.shape) != (B, 2):
        raise ValueError(
            f"hexatic moments must be (B, 2) = ({B}, 2), got {tuple(hex_moments.shape)}"
        )
    if tuple(theta_counts.shape) != (B, N_THETA_BINS):
        raise ValueError(
            f"angle histograms must be (B, {N_THETA_BINS}) = ({B}, {N_THETA_BINS}), got "
            f"{tuple(theta_counts.shape)}"
        )
    value, se = hexatic_estimate(hex_moments.to(F64), seed)
    histogram = AngleHistogram(theta_counts.sum(0).to(F64), angle_marks(law))
    n_samples = law.cfg.net.N * B
    flagged = bool(value > HEXATIC_MAX) if n_samples >= HEXATIC_MIN_SAMPLES else None
    return Angles(value, se, flagged, n_samples, histogram)


def measure_final(
    law: Law,
    final: Tensor,
    hex_moments: Tensor | None = None,
    theta_counts: Tensor | None = None,
    seed: int = 0,
) -> FinalMeasurement:
    """Everything the notebook shows of one setting's final states: `final` (B, K + 1, N), the
    members' last records (`mode_spectra`, the null row last), B >= 2; at K = 3 also their
    hexatic moments (B, 2) and angle histograms (B, N_THETA_BINS). Every part is of the member
    mean, with the bootstrap over members seeded by `seed`."""
    _check_final(law, final)
    final = final.to(F64)
    latent = final[:, : law.D.shape[0]]
    return FinalMeasurement(
        primary=measure_primary(law, latent, seed),
        spectrum=measure_spectrum(law, final, seed),
        ratio=measure_ratio(law, latent, seed),
        profile=measure_profile(law, latent, seed),
        secondary=measure_secondary(law, latent, seed),
        angles=measure_angles(law, final.shape[0], hex_moments, theta_counts, seed),
        B=final.shape[0],
    )


class RelaxationCurves(NamedTuple):
    """`fct.draw.plot_relaxation`'s arguments: every statistic relative to its fitted
    asymptote in units of its accuracy budget, so all of them share one axis."""

    epochs: Tensor  # (R,) the records'
    values: Tensor  # (R, S) (statistic of the member-mean record - asymptote) / budget
    fit_epochs: Tensor  # (T,) the fit window's
    fitted: Tensor  # (T, S) the fitted curve, normalised the same way
    labels: list[str]  # `statistic_label`
    start: float  # the window's start epoch
    band: float  # RELAXATION_BAND


class RelaxationView(NamedTuple):
    """The burn-in diagnostic of one setting."""

    curves: RelaxationCurves
    fit: RelaxationFit
    stats: list[Statistic]
    budget: Tensor  # (S,) `budget_se` at the base B
    stationary: Tensor  # (S,) bool, `is_stationary`


def measure_relaxation(
    law: Law,
    epochs: Tensor,
    records: Tensor,
    base_B: int,
    start: float | None = None,
    seed: int = 0,
) -> RelaxationView:
    """`relaxation_fit` of the latent records (R, B, K, N) at `epochs` (R,) from epoch `start`
    on (default: the second half of the run, which tests its end), judged against the budget
    at `base_B` (`budget_se`), and the series to draw."""
    start = law.cfg.train.n_epochs / 2 if start is None else float(start)
    statistics = law_statistics(law)
    epochs, records = epochs.to(F64), records.to(F64)
    fit = relaxation_fit(epochs, records, statistics, start, seed=seed)
    budget = budget_se(law, base_B)
    fit_epochs = epochs[epochs >= start]
    curve = relaxation_curve(fit, fit_epochs, float(fit_epochs[0]))
    stats = statistic_list(law.D.shape[0])
    curves = RelaxationCurves(
        epochs=epochs,
        values=(statistics(records.mean(1)) - fit.asymptote) / budget,
        fit_epochs=fit_epochs,
        fitted=(curve - fit.asymptote) / budget,
        labels=[statistic_label(s, law.ell) for s in stats],
        start=start,
        band=RELAXATION_BAND,
    )
    return RelaxationView(curves, fit, stats, budget, is_stationary(fit, budget))
