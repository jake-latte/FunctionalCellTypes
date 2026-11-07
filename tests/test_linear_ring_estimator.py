"""The smoke test's estimator and tolerances on maps drawn from the predicted law: the known
ground truth, checked before the estimator is trusted on SGLD samples
(tests/test_linear_ring_smoke.py; the derivation is in tests/linear_ring_helpers.py).

Each synthetic member's activity is h = U t with t (K, N) a real stationary Gaussian field of
the predicted spectrum of a `linear_ring_tiny` diagonal setting and U the data's latent frame,
so the spectra go through the same `mode_spectra` call the monitor makes.

The acceptance thresholds below are sized so that a correct estimator passes with certainty in
practice: with every statistic inside three SE_MAX (4.05 true SEs) the chance that one of the
7 statistics of an ensemble from the predicted law falls outside is below 1e-3, so the count
(98 of 100) allows for the bootstrap SE check, whose 99th percentile is pinned below SE_MARGIN,
and for resampling noise, not for a biased estimator.
"""

import pytest
import torch

from fct.linear_theory import theory_parameters
from fct.spectrum import half_power_q, mode_spectra
from tests.linear_ring_helpers import (
    SE_MARGIN,
    Law,
    bounds,
    delta_method_se,
    estimate,
    kind_of,
    map_correlation,
    predicted_law,
    ring_statistics,
    statistic_names,
    time_drift,
)
from tests.spectrum_helpers import gaussian_field

PRESET = "linear_ring_tiny"
DIAGONAL = (0, 3)  # penalty at its lr, whitened at its lr
N_ENSEMBLES = 200  # replicate ensembles of B members: 5% relative error on a spread
N_CHECKED = 100  # ensembles put through the full bootstrap (the costly step)
F64 = torch.float64


def _member_spectra(D: torch.Tensor, U: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """(n, K, N): rows 0..K-1 of `mode_spectra` of n members h = U t, t from the law of D."""
    t = gaussian_field(D, n, torch.Generator().manual_seed(seed))  # (n, K, N)
    return mode_spectra(U @ t, U)[:, :-1]


@pytest.fixture(scope="module", params=DIAGONAL, ids=["penalty", "whitened"])
def ensembles(request) -> tuple[Law, torch.Tensor]:
    """The law of one diagonal setting and N_ENSEMBLES synthetic ensembles (n, B, K, N)."""
    law = predicted_law(PRESET, request.param)
    spectra = _member_spectra(law.D, law.U, N_ENSEMBLES * law.B, seed=request.param)
    return law, spectra.reshape(N_ENSEMBLES, law.B, *law.D.shape)


def _accepts(spectra: torch.Tensor, law: Law, seed: int) -> bool:
    """The smoke test's verdict on one ensemble (B, K, N): every statistic within its
    tolerance and every bootstrap SE within SE_MAX."""
    values, se = estimate(spectra, law, seed)
    se_max, tolerance = bounds(law)
    truth = ring_statistics(law.D, law.ell)(law.D)
    return bool(((values - truth).abs() <= tolerance).all() and (se <= se_max).all())


def test_the_synthetic_spectra_have_the_predicted_mean(ensembles):
    law, spectra = ensembles
    n = spectra.shape[0] * spectra.shape[1]
    # the relative SE of a mean of n chi^2_1 draws is sqrt(2 / n) = 0.6% at n = 51200
    torch.testing.assert_close(spectra.mean((0, 1)), law.D, rtol=5 * (2 / n) ** 0.5, atol=1e-12)


def test_the_spread_across_ensembles_is_the_delta_method_standard_error(ensembles):
    law, spectra = ensembles
    values = ring_statistics(law.D, law.ell)(spectra.mean(1))  # (N_ENSEMBLES, 3K + 1)
    predicted = delta_method_se(law.D, law.ell, law.B)
    # a spread over 200 ensembles has a relative SE of 5%: allow three
    torch.testing.assert_close(values.std(0), predicted, rtol=0.15, atol=0)
    # A nonlinear statistic of the member mean is biased at O(1/B); none is visible above the
    # resolution of 200 ensembles, 4 SE / sqrt(200) = 0.28 SE.
    bias = (values.mean(0) - ring_statistics(law.D, law.ell)(law.D)).abs()
    assert (bias <= 4 * predicted / N_ENSEMBLES**0.5).all()


def test_the_bootstrap_se_stays_below_the_margin_times_the_true_se(ensembles):
    """The 99th percentile over ensembles of (bootstrap SE / spread across ensembles) is below
    SE_MARGIN for every statistic, which is what lets SE_MAX bound a correct measurement."""
    law, spectra = ensembles
    spread = ring_statistics(law.D, law.ell)(spectra.mean(1)).std(0)
    ses = torch.stack([estimate(spectra[r], law, seed=r)[1] for r in range(N_CHECKED)])
    assert (torch.quantile(ses / spread, 0.99, dim=0) < SE_MARGIN).all()


def test_the_smoke_test_accepts_ensembles_of_the_predicted_law(ensembles):
    law, spectra = ensembles
    accepted = [_accepts(spectra[r], law, seed=r) for r in range(N_CHECKED)]
    assert sum(accepted) >= 98


@pytest.mark.parametrize("alternative", ["flat", "equal_amplitudes"])
def test_the_smoke_test_rejects_the_obvious_alternatives(ensembles, alternative):
    """A flat spectrum of each mode's predicted total power, and both modes at mode 1's
    spectrum (the hidden prior's shape without the input weights s_X,k^2), drawn as B members
    each, fail every time."""
    law, _ = ensembles
    D = law.D
    if alternative == "flat":
        wrong = D.sum(-1, keepdim=True).expand_as(D) / D.shape[-1]
    else:
        wrong = D[:1].expand_as(D).clone()
    spectra = _member_spectra(wrong, law.U, 20 * law.B, seed=11).reshape(20, law.B, *D.shape)
    assert not any(_accepts(spectra[r], law, seed=r) for r in range(20))


def test_the_smoke_statistics_do_not_see_the_data_term(ensembles):
    """At beta = 0.25 the prior alone, d_k(q) = lambda_q s_X,k^2, moves no asserted statistic
    by more than 0.15 of its SE (tau_1 = 0.008): the smoke test checks the pipeline, not the
    theory's data term. q_1/2 is the same too."""
    law, _ = ensembles
    params = theory_parameters(law.cfg, law.dataset)
    prior = torch.from_numpy(params.sx2[:, None] * params.lam[None, :])
    statistics = ring_statistics(law.D, law.ell)
    shift = (statistics(prior) - statistics(law.D)).abs() / delta_method_se(law.D, law.ell, law.B)
    assert (shift <= 0.15).all()
    assert torch.equal(torch.as_tensor(half_power_q(prior)), torch.as_tensor(half_power_q(law.D)))


def test_the_drift_of_stationary_records_is_zero_within_its_standard_error():
    """Independent records of the predicted law, as a stationary chain gives, have no drift
    between the halves; a chain whose q = 0 power is still growing (condensing, by 10x over the
    run) drifts in the band share and the lag-1 autocorrelation of every mode."""
    law = predicted_law(PRESET, DIAGONAL[1])
    records = _member_spectra(law.D, law.U, 20 * law.B, seed=5).reshape(20, law.B, *law.D.shape)
    mean, sem = time_drift(records, burn_in=4, law=law)
    assert (mean.abs() <= 4 * sem).all()
    growing = records.clone()
    growing[..., 0] *= torch.linspace(1.0, 10.0, 20, dtype=F64)[:, None, None]
    mean, sem = time_drift(growing, burn_in=4, law=law)
    names = statistic_names(law.D.shape[0])
    moved = [i for i, n in enumerate(names) if kind_of(n) in ("band", "lag1")]
    assert (mean[moved].abs() > 4 * sem[moved]).all()


def test_map_correlation_is_one_for_a_frozen_sampler_and_zero_for_independent_maps():
    law = predicted_law(PRESET, DIAGONAL[1])
    gen = torch.Generator().manual_seed(9)
    first, independent = (gaussian_field(law.D, law.B, gen) for _ in range(2))
    torch.testing.assert_close(map_correlation(first, first), torch.ones(law.B, dtype=F64))
    correlation = map_correlation(first, independent)
    assert abs(float(correlation.mean())) < 4 * float(correlation.std()) / law.B**0.5
