"""fct.linear_ring on synthetic inputs with a known answer: the statistic vector, the floored
frequencies, the relaxation fit and the comparison rules."""

import math

import pytest
import torch

from fct import linear_ring as lr
from fct.exp import preset
from fct.spectrum import bootstrap_members
from tests.spectrum_helpers import gaussian_field

F64 = torch.float64


def _gaussian_spectrum(N: int, ell: float) -> torch.Tensor:
    q = torch.fft.fftfreq(N, d=1.0 / N, dtype=F64)
    return torch.exp(-2 * (math.pi * ell * q / N) ** 2)


def test_the_statistic_list_pairs_each_kind_with_its_mode():
    stats = lr.statistic_list(2)
    assert [(s.kind, s.mode) for s in stats] == [
        ("lag1", 1),
        ("lag1", 2),
        ("lagl", 1),
        ("lagl", 2),
        ("band", 1),
        ("band", 2),
        ("ratio", None),
    ]


def test_ring_statistics_are_the_autocorrelation_band_share_and_ratio():
    N, ell = 64, 2
    D = torch.stack([_gaussian_spectrum(N, 2.0), 0.5 * _gaussian_spectrum(N, 1.0)])
    values = lr.ring_statistics(N, ell)(D)
    R = torch.fft.ifft(D, dim=-1).real
    in_band = torch.minimum(torch.arange(N), N - torch.arange(N)) <= 3
    torch.testing.assert_close(values[0:2], R[:, 1] / R[:, 0])
    torch.testing.assert_close(values[2:4], R[:, ell] / R[:, 0])
    torch.testing.assert_close(values[4:6], D[:, in_band].sum(-1) / D.sum(-1))
    torch.testing.assert_close(values[6], D[0, in_band].sum() / D[1, in_band].sum())


def test_ring_statistics_ignore_the_frequencies_outside_keep():
    N = 32
    D = torch.stack([_gaussian_spectrum(N, 2.0)] * 2)
    keep = torch.minimum(torch.arange(N), N - torch.arange(N)) <= 10
    noisy = D.clone()
    noisy[:, ~keep] += 5.0  # power the statistics must not see
    stats = lr.ring_statistics(N, 2, keep)
    torch.testing.assert_close(stats(noisy), stats(D))
    with pytest.raises(ValueError, match="N = 32"):
        stats(D[:, :16])


@pytest.mark.parametrize(("name", "kept"), [("linear_ring_tiny", 4), ("linear_ring", 214)])
def test_the_floored_frequencies_are_those_below_min_rel_eig_of_the_top_eigenvalue(name, kept):
    """linear_ring_tiny (N = 16, ell = 2 sites, min_rel_eig 1e-3): lambda_q / lambda_0 is 4.5e-4
    at |q| = 5, so |q| >= 5 are floored; linear_ring (N = 512, ell = 2, 1e-6): |q| >= 215."""
    law = lr.predicted_law(preset(name), 0)
    N = law.D.shape[-1]
    q = torch.minimum(torch.arange(N), N - torch.arange(N))
    assert torch.equal(law.keep, q <= kept)


@pytest.fixture(scope="module")
def tiny_law() -> lr.Law:
    return lr.predicted_law(preset("linear_ring_tiny"), 3)


def test_the_delta_method_se_falls_as_one_over_root_b(tiny_law):
    at_b = lr.delta_method_se(tiny_law)
    torch.testing.assert_close(lr.delta_method_se(tiny_law, B=4 * tiny_law.B), at_b / 2)
    torch.testing.assert_close(lr.delta_method_se(tiny_law, B=tiny_law.B), at_b)


def test_the_budget_is_one_se_at_the_follow_up_b(tiny_law):
    torch.testing.assert_close(
        lr.budget_se(tiny_law), lr.delta_method_se(tiny_law, B=lr.FOLLOW_UP_FACTOR * tiny_law.B)
    )
    predicted = lr.law_statistics(tiny_law)(tiny_law.D)
    budget = lr.budget_se(tiny_law)
    assert lr.regime_accurate(predicted + 0.9 * budget, tiny_law).all()
    assert not lr.regime_accurate(predicted - 1.1 * budget, tiny_law).any()


def test_the_pipeline_bounds_are_the_named_multiples_of_the_delta_method_se(tiny_law):
    se_max, tolerance = lr.pipeline_bounds(tiny_law)
    se = lr.delta_method_se(tiny_law)
    torch.testing.assert_close(se_max, lr.PIPELINE_SE_MARGIN * se)
    torch.testing.assert_close(tolerance, lr.PIPELINE_N_SE * lr.PIPELINE_SE_MARGIN * se)


def test_member_covariance_doubles_q_zero_and_nyquist_and_ties_q_to_minus_q():
    D = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0]], dtype=F64)  # even in q
    C = lr.member_covariance(D)[0]
    assert C[0, 0] == 2 * 5.0**2 and C[4, 4] == 2 * 1.0**2
    assert C[1, 1] == C[1, 7] == 4.0**2 and C[1, 2] == 0


def _relaxing_records(tau: float, depth: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(epochs (R,), records (R, B, 2, N)): independent draws, member by member, of a law whose
    mode-1 low-q power relaxes as 1 - depth exp(-t / tau)."""
    N, B, R = 32, 128, 40
    epochs = torch.linspace(0.0, 2000.0, R, dtype=F64)
    D = torch.stack([_gaussian_spectrum(N, 2.0), 0.3 * _gaussian_spectrum(N, 2.0)])
    q = torch.minimum(torch.arange(N), N - torch.arange(N))
    gen = torch.Generator().manual_seed(seed)
    records = []
    for t in epochs:
        Dt = D.clone()
        Dt[0, q <= 2] *= 1 - depth * math.exp(-float(t) / tau)
        t_maps = gaussian_field(Dt, B, gen)
        records.append(torch.fft.fft(t_maps, dim=-1, norm="ortho").abs().square())
    return epochs, torch.stack(records)


def _low_q_power(D: torch.Tensor) -> torch.Tensor:
    """Mode 1's power at |q| <= 2, (..., 2, N) -> (..., 1): linear in the spectrum, so its
    member mean relaxes exactly as the law does."""
    q = torch.minimum(torch.arange(D.shape[-1]), D.shape[-1] - torch.arange(D.shape[-1]))
    return D[..., 0, q <= 2].sum(-1, keepdim=True)


def test_the_relaxation_fit_recovers_a_known_time_constant():
    epochs, records = _relaxing_records(tau=300.0, depth=0.6, seed=0)
    fit = lr.relaxation_fit(epochs, records, _low_q_power, n_boot=50)
    assert fit.tau[0] == pytest.approx(300.0, rel=0.15)
    assert abs(float(fit.deficit[0])) < 4 * float(fit.deficit_se[0])


def test_the_relaxation_fit_of_a_stationary_series_has_no_deficit():
    epochs, records = _relaxing_records(tau=300.0, depth=0.0, seed=1)
    for statistics in (_low_q_power, lr.ring_statistics(32, 2)):
        fit = lr.relaxation_fit(epochs, records, statistics, n_boot=50)
        assert (fit.deficit.abs() <= 4 * fit.deficit_se + 1e-12).all()


def test_a_run_stopped_early_has_its_deficit_measured():
    """Stopped at 2000 epochs with tau = 1000, the low-q power still lacks 0.6 e^-2 of its
    prior-to-final change: the fit recovers that deficit within its SE, and the run is not
    stationary against the SE of the final record's member mean."""
    epochs, records = _relaxing_records(tau=1000.0, depth=0.6, seed=3)
    fit = lr.relaxation_fit(epochs, records, _low_q_power, n_boot=100)
    D_low = float(_low_q_power(records[-1].mean(0)).squeeze()) / (1 - 0.6 * math.exp(-2.0))
    true_deficit = 0.6 * math.exp(-2.0) * D_low
    assert abs(float(fit.deficit[0]) - true_deficit) < 3 * float(fit.deficit_se[0])
    final_se = _low_q_power(records[-1]).std(0) / records.shape[1] ** 0.5
    assert not lr.is_stationary(fit, final_se)[0]


def test_a_late_window_of_a_relaxed_run_is_stationary_and_conclusive():
    """Fitted from the middle of a run relaxed early (tau = 300 of 2000 epochs), the deficit is
    zero within an SE no larger than the window mean's, which is below the final record's SE:
    the diagnostic can tell."""
    epochs, records = _relaxing_records(tau=300.0, depth=0.6, seed=4)
    fit = lr.relaxation_fit(epochs, records, _low_q_power, start=1000.0, n_boot=100)
    final_se = _low_q_power(records[-1]).std(0) / records.shape[1] ** 0.5
    assert lr.is_stationary(fit, final_se)[0]


def test_the_deficit_se_is_never_below_the_window_means_se():
    """Fitted over all records of a run relaxed within its first record spacing, the deficit is
    0 in every resample; its SE is then the bootstrap SE of the window's mean statistic."""
    epochs, records = _relaxing_records(tau=10.0, depth=0.6, seed=5)
    fit = lr.relaxation_fit(epochs, records, _low_q_power, n_boot=50, seed=7)
    _, window_se = bootstrap_members(
        records.transpose(0, 1), lambda m: _low_q_power(m).mean(-2), 50, seed=7
    )
    assert float(fit.deficit_se[0]) >= float(window_se[0]) > 0


def test_a_zero_deficit_with_an_se_above_the_budget_is_not_stationary():
    fit = lr.RelaxationFit(*(torch.zeros(1, dtype=F64) for _ in range(4)), torch.ones(1, dtype=F64))
    assert not lr.is_stationary(fit, torch.full((1,), 0.5, dtype=F64))[0]
    assert lr.is_stationary(fit, torch.full((1,), 2.0, dtype=F64))[0]


def test_the_relaxation_fit_needs_four_records():
    epochs, records = _relaxing_records(tau=300.0, depth=0.0, seed=3)
    with pytest.raises(ValueError, match="at least 4"):
        lr.relaxation_fit(epochs[:3], records[:3], lr.ring_statistics(32, 2))


def test_disagreement_is_falsify_n_se_standard_errors():
    predicted = torch.zeros(3, dtype=F64)
    se = torch.ones(3, dtype=F64)
    n = lr.FALSIFY_N_SE
    values = torch.tensor([n - 0.1, -(n + 0.1), 0.0], dtype=F64)
    assert lr.disagrees(values, se, predicted).tolist() == [False, True, False]


def test_two_paths_disagree_beyond_equivalence_n_se_combined_ses():
    """Deviations from their own predictions (here 0 and 1) are compared, with the SEs combined
    in quadrature (sqrt(0.6^2 + 0.8^2) = 1)."""
    n = lr.EQUIVALENCE_N_SE
    zero, one = torch.zeros(2, dtype=F64), torch.ones(2, dtype=F64)
    first = (torch.tensor([n - 0.1, n + 0.1], dtype=F64), 0.6 * one, zero)
    second = (one, 0.8 * one, one)
    assert lr.paths_disagree(first, second).tolist() == [False, True]
