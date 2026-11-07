"""The measurement the linear-ring notebook draws (`fct.linear_ring.measure_final`,
`measure_relaxation` and their helpers), on synthetic data with a known ground truth.

Synthetic ensembles are drawn from the predicted law of a `linear_ring_tiny` setting (each
member's tuning maps a stationary Gaussian field with the predicted spectrum, exactly the law the
theory states), so the measurement must recover the prediction within its own standard errors;
the helpers are checked on inputs whose answer is known in closed form.
"""

import math

import pytest
import torch

from fct import linear_ring as lr
from fct.exp import preset
from fct.spectrum import (
    autocorrelation,
    fold_q,
    latent_basis,
    mode_spectra,
    theta_histogram,
    tuning_maps,
)
from tests.spectrum_helpers import gaussian_field, uniform_factors

PRESET = "linear_ring_tiny"
WHITENED = 3  # the whitened diagonal setting of the tiny preset
B = 256
# Agreement of a recovered statistic with the truth, in its own bootstrap SEs. Seeded draws, so
# the test is deterministic; 4 SEs keeps it far from a fluke across the ~30 values compared.
N_SE = 4.0
F64 = torch.float64


@pytest.fixture(scope="module")
def law() -> lr.Law:
    return lr.predicted_law(preset(PRESET), WHITENED)


def _members(law: lr.Law, n: int, seed: int) -> torch.Tensor:
    """(n, K + 1, N): `mode_spectra` (latent rows and the null row) of n members whose tuning
    maps are drawn from the predicted law; h = U t lies in col(Z), so the null row is 0."""
    t = gaussian_field(law.D, n, torch.Generator().manual_seed(seed))  # (n, K, N)
    return mode_spectra(law.U @ t, law.U)


@pytest.fixture(scope="module")
def measured(law) -> lr.FinalMeasurement:
    return lr.measure_final(law, _members(law, B, seed=1))


# ------------------------------------------------------------------ helpers with closed forms
def test_lag_profile_is_the_normalised_autocorrelation_up_to_half_the_ring():
    N = 8
    D = torch.rand(3, 2, N, dtype=F64, generator=torch.Generator().manual_seed(0))
    D = 0.5 * (D + D.roll(-1, -1).flip(-1))  # even in q
    profile = lr.lag_profile(D)
    R = autocorrelation(D)
    torch.testing.assert_close(profile, (R / R[..., :1])[..., : N // 2 + 1])
    assert torch.allclose(profile[..., 0], torch.ones(3, 2, dtype=F64))


def test_lag_profile_zeroes_the_excluded_frequencies_first():
    N = 8
    D = torch.ones(1, N, dtype=F64)
    keep = torch.tensor([True] + [False] * (N - 1))  # only q = 0: a flat profile
    torch.testing.assert_close(lr.lag_profile(D, keep), torch.ones(1, N // 2 + 1, dtype=F64))


def test_folded_mode_ratio_is_mode_1_over_mode_2_per_abs_q():
    D = torch.stack([torch.full((8,), 6.0), torch.full((8,), 2.0)]).to(F64)
    torch.testing.assert_close(lr.folded_mode_ratio(D), torch.full((5,), 3.0, dtype=F64))


def test_secondary_statistics_are_q_half_and_the_q0_share_per_mode():
    N = 8
    q = torch.minimum(torch.arange(N), N - torch.arange(N)).to(F64)
    D = torch.stack([torch.exp(-(q**2) / 2), torch.exp(-(q**2) / 8)])  # wider in mode 2
    values = lr.secondary_statistics(N)(D)
    assert values.shape == (4,)
    # q_1/2: exp(-q^2/2) >= 1/2 up to q = 1; exp(-q^2/8) up to q = 2 (floor d_4 is tiny)
    assert values[:2].tolist() == [1.0, 2.0]
    torch.testing.assert_close(values[2:], D[:, 0] / D.sum(-1))
    assert lr.secondary_names(2) == ["q_half_1", "q_half_2", "share0_1", "share0_2"]


def test_hexatic_estimate_is_the_modulus_of_the_member_mean_moment():
    phase = torch.tensor([0.3, 0.3, 0.3, 0.3], dtype=F64)
    moments = 0.5 * torch.stack([phase.cos(), phase.sin()], dim=-1)  # (B, 2)
    value, se = lr.hexatic_estimate(moments)
    assert float(value) == pytest.approx(0.5) and float(se) == pytest.approx(0.0, abs=1e-12)
    cancelling = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=F64)
    assert float(lr.hexatic_estimate(cancelling)[0]) == pytest.approx(0.0)


def test_latent_angles_are_where_a_neuron_tuned_to_one_latent_lies():
    """A neuron whose activity is latent j (h_i = +-z_j) has its contrast-plane angle at one of
    the six marks; the six are three opposite pairs."""
    factors = uniform_factors(P=64, K=3, seed=2)
    basis = latent_basis(factors)
    angles = lr.latent_angles(basis.Z, basis.U)
    assert angles.shape == (6,)
    h = torch.stack([s * basis.Z[:, j] for s in (1.0, -1.0) for j in range(3)], dim=1)  # (P, 6)
    t = tuning_maps(h, basis.U)  # (3, 6)
    theta = torch.atan2(t[2], t[1])
    torch.testing.assert_close(theta.sort().values, angles.sort().values)
    # the summed histogram of these six neurons puts one count in each mark's bin
    assert int(theta_histogram(t[1], t[2]).sum()) == 6


def test_latent_angles_need_three_latents():
    basis = latent_basis(uniform_factors(P=32, K=2))
    with pytest.raises(ValueError, match="K = 3"):
        lr.latent_angles(basis.Z, basis.U)


def test_relaxation_curve_evaluates_the_fitted_exponential():
    fit = lr.RelaxationFit(
        asymptote=torch.tensor([1.0, 2.0], dtype=F64),
        amplitude=torch.tensor([0.5, -1.0], dtype=F64),
        tau=torch.tensor([10.0, 20.0], dtype=F64),
        deficit=torch.zeros(2, dtype=F64),
        deficit_se=torch.zeros(2, dtype=F64),
    )
    t = torch.tensor([5.0, 15.0, 25.0], dtype=F64)
    curve = lr.relaxation_curve(fit, t, t0=5.0)
    expected = fit.asymptote - fit.amplitude * torch.exp(-(t[:, None] - 5.0) / fit.tau)
    torch.testing.assert_close(curve, expected)


def test_statistic_labels_name_the_kind_and_mode():
    labels = [lr.statistic_label(s) for s in lr.statistic_list(2)]
    assert labels == ["lag1_1", "lag1_2", "lagl_1", "lagl_2", "band_1", "band_2", "ratio"]


# ------------------------------------------------------------------ recovering the truth
def test_the_primary_statistics_recover_the_prediction(measured, law):
    predicted = lr.law_statistics(law)(law.D)
    torch.testing.assert_close(measured.predicted, predicted)
    torch.testing.assert_close(measured.values, lr.estimate(_members(law, B, 1)[:, :-1], law)[0])
    assert ((measured.values - predicted).abs() <= N_SE * measured.se).all()
    assert not measured.disagree.any()


def test_the_folded_spectra_recover_the_predicted_spectra(measured, law):
    K, N = law.D.shape
    assert measured.spectrum.shape == (K + 1, N // 2 + 1) == measured.spectrum_se.shape
    torch.testing.assert_close(measured.spectrum_theory, fold_q(law.D))
    deviation = (measured.spectrum[:K] - measured.spectrum_theory).abs()
    assert (deviation <= N_SE * measured.spectrum_se[:K]).all()
    assert float(measured.spectrum[K].abs().max()) < 1e-12 * float(law.D.max())  # null row 0
    assert measured.null_theory == 0.0  # sigma = 0


def test_the_excluded_frequencies_are_the_folded_complement_of_the_kept_band(measured, law):
    N = law.D.shape[-1]
    torch.testing.assert_close(measured.excluded, ~law.keep[: N // 2 + 1])
    assert measured.excluded.any() and not measured.excluded[0]  # tiny floors |q| >= 5


def test_the_ratio_and_lag_profiles_recover_their_predictions(measured, law):
    kept = ~measured.excluded
    ratio_dev = (measured.ratio - measured.ratio_theory)[kept].abs()
    assert (ratio_dev <= N_SE * measured.ratio_se[kept]).all()
    torch.testing.assert_close(measured.ratio_theory, lr.folded_mode_ratio(law.D))
    profile_dev = (measured.profile - measured.profile_theory)[:, 1:].abs()
    assert (profile_dev <= N_SE * measured.profile_se[:, 1:]).all()
    torch.testing.assert_close(measured.profile_theory, lr.lag_profile(law.D, law.keep))


def test_the_kernel_profile_is_that_of_the_prior_eigenvalues(measured, law):
    """At beta = 0.25 the prediction is the prior's shape: the kernel profile is close to the
    predicted profile of every mode (within 0.02 here)."""
    kernel = measured.kernel_profile
    assert float(kernel[0]) == pytest.approx(1.0)
    assert ((measured.profile_theory - kernel).abs() < 0.02).all()


def test_the_secondary_statistics_recover_the_prediction(measured, law):
    K = law.D.shape[0]
    assert measured.secondary_names == lr.secondary_names(K)
    q_half, share = measured.secondary[:K], measured.secondary[K:]
    assert ((q_half - measured.secondary_predicted[:K]).abs() <= lr.Q_HALF_TOLERANCE).all()
    share_dev = (share - measured.secondary_predicted[K:]).abs()
    assert (share_dev <= N_SE * measured.secondary_se[K:]).all()
    assert not measured.secondary_disagree.any()


def test_a_wrong_spectrum_is_flagged(law):
    """Both modes drawn at mode 1's predicted spectrum: the ratio is 1, not about 7."""
    wrong = law._replace(D=law.D[[0, 0]])
    measured = lr.measure_final(law, _members(wrong, B, seed=3))
    ratio = lr.statistic_list(2).index(lr.Statistic("ratio", None))
    assert measured.disagree[ratio]


def test_measure_final_refuses_spectra_without_the_null_row(law):
    with pytest.raises(ValueError, match="K \\+ 1"):
        lr.measure_final(law, _members(law, 4, seed=0)[:, :-1])


# ------------------------------------------------------------------ the relaxation view
def test_a_stationary_ensemble_shows_no_relaxation(law):
    """Records drawn independently from the predicted law: the normalised statistics scatter
    around 0 and the fit finds no deficit beyond its SE."""
    R, members = 24, 64
    records = torch.stack([_members(law, members, seed=10 + r) for r in range(R)])[:, :, :-1]
    epochs = torch.arange(R, dtype=F64) * 250
    view = lr.measure_relaxation(law, epochs, records, start=float(epochs[R // 2]))
    S = len(lr.statistic_list(law.D.shape[0]))
    assert view.values.shape == (R, S) and view.fitted.shape == (R - R // 2, S)
    torch.testing.assert_close(view.fit_epochs, epochs[R // 2 :])
    assert view.labels == [lr.statistic_label(s) for s in lr.statistic_list(law.D.shape[0])]
    assert (view.fit.deficit.abs() <= 3 * view.fit.deficit_se).all()
    raw = lr.law_statistics(law)(records.mean(1))
    torch.testing.assert_close(view.values, (raw - view.fit.asymptote) / view.budget)
    torch.testing.assert_close(view.budget, lr.budget_se(law))


def test_the_relaxation_window_defaults_to_the_second_half_of_the_run(law):
    R = 8
    records = torch.stack([_members(law, 8, seed=r) for r in range(R)])[:, :, :-1]
    epochs = torch.linspace(0, law.cfg.train.n_epochs, R, dtype=F64)
    view = lr.measure_relaxation(law, epochs, records)
    assert view.start == law.cfg.train.n_epochs / 2
    assert float(view.fit_epochs[0]) >= view.start


def test_hexatic_and_angle_inputs_are_optional_and_checked(law):
    measured = lr.measure_final(law, _members(law, 8, seed=0))
    assert measured.hexatic is None
    with pytest.raises(ValueError, match="moments"):
        lr.hexatic_estimate(torch.zeros(4, 3, dtype=F64))
    assert math.isfinite(float(measured.values[0]))
