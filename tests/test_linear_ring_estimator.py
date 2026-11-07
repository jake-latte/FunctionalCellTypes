"""The smoke test's estimator and tolerances on maps drawn from the predicted law: the known
ground truth, checked before the estimator is trusted on SGLD samples
(tests/test_linear_ring_smoke.py; the derivation is in tests/linear_ring_helpers.py).

Each synthetic member's activity is h = U t with t (K, N) a real stationary Gaussian field of
the predicted spectrum of the `linear_ring_tiny` setting and U the data's latent frame, so the
spectra go through the same `mode_spectra` call the monitor makes.
"""

import pytest
import torch

from fct.data import make_dataset
from fct.exp import expand, preset, setting_config
from fct.linear_theory import predicted_spectra
from fct.spectrum import half_power_q, latent_basis, mode_spectra
from tests.linear_ring_helpers import (
    SE_MAX,
    TOLERANCE,
    delta_method_se,
    estimate,
    half_power,
    kind_of,
    ring_statistics,
    statistic_names,
    time_drift,
)
from tests.spectrum_helpers import gaussian_field

PRESET = "linear_ring_tiny"
DIAGONAL = (0, 3)  # penalty at its lr, whitened at its lr
N_ENSEMBLES = 200  # replicate ensembles of B members: 5% relative error on a spread
SE_MARGIN = 1.35  # SE_MAX over the law's SE (linear_ring_helpers)
CPU, F64 = torch.device("cpu"), torch.float64


def _law(index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """(predicted spectrum (K, N), latent frame U (P, K), B) of one setting of the preset."""
    exp = preset(PRESET)
    cfg = setting_config(exp, expand(exp)[index])
    dataset = make_dataset(cfg.data, CPU, F64)
    D = torch.from_numpy(predicted_spectra(cfg, dataset).d)
    return D, latent_basis(dataset.factors).U, cfg.net.B


def _member_spectra(D: torch.Tensor, U: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """(n, K, N): rows 0..K-1 of `mode_spectra` of n members h = U t, t from the law of D."""
    t = gaussian_field(D, n, torch.Generator().manual_seed(seed))  # (n, K, N)
    return mode_spectra(U @ t, U)[:, :-1]


@pytest.fixture(scope="module", params=DIAGONAL, ids=["penalty", "whitened"])
def ensembles(request):
    """(D, B, spectra (N_ENSEMBLES, B, K, N), setting index) for one diagonal setting."""
    D, U, B = _law(request.param)
    spectra = _member_spectra(D, U, N_ENSEMBLES * B, seed=request.param)
    return D, B, spectra.reshape(N_ENSEMBLES, B, *D.shape), request.param


def _accepts(spectra: torch.Tensor, D: torch.Tensor, seed: int) -> bool:
    """The smoke test's verdict on one ensemble (B, K, N): every statistic within its
    tolerance, every bootstrap SE within SE_MAX, and q_1/2 equal to the prediction."""
    values, se = estimate(spectra, D, seed)
    truth = ring_statistics(D)(D)
    names = statistic_names(D.shape[0])
    tolerance = torch.tensor([TOLERANCE[kind_of(n)] for n in names], dtype=F64)
    se_max = torch.tensor([SE_MAX[kind_of(n)] for n in names], dtype=F64)
    q_half = torch.as_tensor(half_power_q(D))
    return bool(
        ((values - truth).abs() <= tolerance).all()
        and (se <= se_max).all()
        and torch.equal(half_power(spectra), q_half)
    )


def test_the_synthetic_spectra_have_the_predicted_mean_and_no_null_power(ensembles):
    D, _, spectra, _ = ensembles
    mean = spectra.mean((0, 1))
    n = spectra.shape[0] * spectra.shape[1]
    # relative SE of a mean of n chi^2_1 draws is sqrt(2 / n) = 0.6% at n = 51200
    torch.testing.assert_close(mean, D, rtol=5 * (2 / n) ** 0.5, atol=1e-12)


def test_the_spread_across_ensembles_is_the_delta_method_standard_error(ensembles):
    D, B, spectra, _ = ensembles
    values = ring_statistics(D)(spectra.mean(1))  # (N_ENSEMBLES, 3K + 1)
    spread = values.std(0)
    predicted = delta_method_se(D, B)
    # a spread over 200 ensembles has a relative SE of 5%: allow 3 of them
    torch.testing.assert_close(spread, predicted, rtol=0.15, atol=0)
    # A nonlinear statistic of the member mean is biased at O(1/B); none is visible above the
    # resolution of 200 ensembles, 4 SE / sqrt(200) = 0.28 SE.
    bias = (values.mean(0) - ring_statistics(D)(D)).abs()
    assert (bias <= 4 * predicted / N_ENSEMBLES**0.5).all()


def test_the_tolerances_are_three_bounded_standard_errors_of_the_law(ensembles):
    D, B, _, _ = ensembles
    names = statistic_names(D.shape[0])
    se = delta_method_se(D, B)
    se_max = torch.tensor([SE_MAX[kind_of(n)] for n in names], dtype=F64)
    assert (se_max >= SE_MARGIN * se).all()
    assert (se_max <= 1.1 * SE_MARGIN * se).all()  # rounded up, not loosened
    for name in names:
        assert TOLERANCE[kind_of(name)] == pytest.approx(3 * SE_MAX[kind_of(name)])


def test_the_bootstrap_se_of_one_ensemble_stays_below_se_max(ensembles):
    """In at least 98% of ensembles from the predicted law (the 99th percentile of bootstrap SE
    over the law's SE is at most 1.30 for every statistic)."""
    D, _, spectra, _ = ensembles
    names = statistic_names(D.shape[0])
    se_max = torch.tensor([SE_MAX[kind_of(n)] for n in names], dtype=F64)
    below = [bool((estimate(spectra[r], D, seed=r)[1] <= se_max).all()) for r in range(100)]
    assert sum(below) >= 98


def test_the_smoke_test_accepts_ensembles_of_the_predicted_law(ensembles):
    D, _, spectra, _ = ensembles
    accepted = [_accepts(spectra[r], D, seed=r) for r in range(100)]
    assert sum(accepted) >= 97


@pytest.mark.parametrize("alternative", ["flat", "equal_amplitudes"])
def test_the_smoke_test_rejects_the_obvious_alternatives(ensembles, alternative):
    """A flat spectrum of each mode's predicted total power, and both modes at mode 1's
    spectrum (the hidden prior's shape without the input weights s_X,k^2), drawn as B members
    each, fail every time."""
    D, B, _, index = ensembles
    _, U, _ = _law(index)
    if alternative == "flat":
        law = D.sum(-1, keepdim=True).expand_as(D) / D.shape[-1]
    else:
        law = D[:1].expand_as(D).clone()
    spectra = _member_spectra(law, U, 20 * B, seed=11).reshape(20, B, *D.shape)
    assert not any(_accepts(spectra[r], D, seed=r) for r in range(20))


def test_at_beta_0_25_the_smoke_statistics_cannot_tell_the_prediction_from_the_prior(ensembles):
    """tau_1 = 0.008: the prior alone, d_k(q) = lambda_q s_X,k^2, moves no statistic by more than
    0.15 of its SE, so this smoke test checks the sampler and the measurement, not the data
    term (see the preset's description)."""
    D, B, _, index = ensembles
    exp = preset(PRESET)
    cfg = setting_config(exp, expand(exp)[index])
    params = predicted_spectra(cfg, make_dataset(cfg.data, CPU, F64)).params
    prior = torch.from_numpy(params.sx2[:, None] * params.lam[None, :])
    statistics = ring_statistics(D)
    shift = (statistics(prior) - statistics(D)).abs() / delta_method_se(D, B)
    assert (shift <= 0.15).all()


def test_the_drift_of_stationary_records_is_zero_within_its_standard_error():
    """Independent records of the predicted law, as a stationary chain would give, have no
    drift between the halves; a chain whose power grows by 30% does."""
    D, U, B = _law(DIAGONAL[1])
    records = _member_spectra(D, U, 20 * B, seed=5).reshape(20, B, *D.shape)
    mean, sem = time_drift(records, burn_in=4, predicted=D)
    assert (mean.abs() <= 4 * sem).all()
    growing = records * torch.linspace(1.0, 1.3, 20, dtype=F64)[:, None, None, None]
    mean, sem = time_drift(growing, burn_in=4, predicted=D)
    names = statistic_names(D.shape[0])
    band = [i for i, n in enumerate(names) if kind_of(n) == "band"]
    assert (mean[band].abs() > 4 * sem[band]).all()
