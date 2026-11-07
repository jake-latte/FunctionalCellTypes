"""The measurement the linear-ring notebook draws (`fct.linear_ring.measure_final`,
`measure_relaxation` and their helpers), on synthetic data with a known ground truth.

Synthetic ensembles are drawn from the predicted law of a `linear_ring_tiny` setting (each
member's tuning maps a stationary Gaussian field with the predicted spectrum, exactly the law the
theory states), so the measurement must recover the prediction within its own standard errors;
the helpers are checked on inputs whose answer is known in closed form or computed another way.
"""

import pytest
import torch

from fct import linear_ring as lr
from fct.data import task_dims
from fct.exp import ExpConfig, preset
from fct.spatial import build_spaces
from fct.spectrum import (
    N_THETA_BINS,
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


@pytest.fixture(scope="module")
def law3() -> lr.Law:
    """The tiny ring with K = 3 latents, where the angle statistics are defined."""
    tree = preset(PRESET).model_dump()
    tree["linear_teacher"]["n_latents"] = 3
    return lr.predicted_law(ExpConfig.model_validate(tree), WHITENED)


def _maps(law: lr.Law, n: int, seed: int) -> torch.Tensor:
    """(n, K, N) tuning maps drawn from the predicted law."""
    return gaussian_field(law.D, n, torch.Generator().manual_seed(seed))


def _members(law: lr.Law, n: int, seed: int) -> torch.Tensor:
    """(n, K + 1, N): `mode_spectra` (latent rows and the null row) of n members whose tuning
    maps are drawn from the predicted law; h = U t lies in col(Z), so the null row is 0."""
    return mode_spectra(law.U @ _maps(law, n, seed), law.U)


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


def test_the_ring_statistics_lags_are_the_lag_profile(law):
    """One masking and one normalisation serve both."""
    stats = lr.law_statistics(law)(law.D)
    profile = lr.lag_profile(law.D, law.keep)
    K = law.D.shape[0]
    torch.testing.assert_close(stats[:K], profile[:, 1])
    torch.testing.assert_close(stats[K : 2 * K], profile[:, law.ell])


def test_folded_mode_ratio_is_mode_1_over_mode_2_per_abs_q():
    D = torch.stack([torch.full((8,), 6.0), torch.full((8,), 2.0)]).to(F64)
    torch.testing.assert_close(lr.folded_mode_ratio(D), torch.full((5,), 3.0, dtype=F64))


def test_q_half_and_q0_share_statistics():
    N = 8
    q = torch.minimum(torch.arange(N), N - torch.arange(N)).to(F64)
    D = torch.stack([torch.exp(-(q**2) / 2), torch.exp(-(q**2) / 8)])  # wider in mode 2
    # q_1/2: exp(-q^2/2) >= 1/2 up to q = 1; exp(-q^2/8) up to q = 2 (floor d_4 is tiny)
    assert lr.q_half_statistic()(D).tolist() == [1.0, 2.0]
    torch.testing.assert_close(lr.share0_statistic()(D), D[:, 0] / D.sum(-1))
    names = [lr.statistic_name(s, 2) for s in lr.secondary_list(2)]
    assert names == ["q_1/2", "q_1/2", "q = 0 share", "q = 0 share"]


def test_prior_profile_is_the_hidden_kernels_own_row(law):
    """R of the kernel's eigenvalues is its first row: computed here from the built kernel,
    not from its eigenvalues (with every frequency kept)."""
    everything = law._replace(keep=torch.ones_like(law.keep))
    m, n = task_dims(law.cfg.data)
    K = build_spaces(law.cfg.spatial, m=m, N=law.cfg.net.N, L=1, n_out=n)[0].post.K
    row = torch.as_tensor(K, dtype=F64)[0]
    N = law.cfg.net.N
    torch.testing.assert_close(lr.prior_profile(everything), row[: N // 2 + 1] / row[0])


def test_hexatic_estimate_is_the_modulus_of_the_member_mean_moment():
    phase = torch.tensor([0.3, 0.3, 0.3, 0.3], dtype=F64)
    moments = 0.5 * torch.stack([phase.cos(), phase.sin()], dim=-1)  # (B, 2)
    value, se = lr.hexatic_estimate(moments)
    assert float(value) == pytest.approx(0.5) and float(se) == pytest.approx(0.0, abs=1e-12)
    cancelling = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=F64)
    assert float(lr.hexatic_estimate(cancelling)[0]) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="moments"):
        lr.hexatic_estimate(torch.zeros(4, 3, dtype=F64))


def test_latent_angles_are_where_a_neuron_tuned_to_one_latent_lies():
    """A neuron whose activity is latent j (h_i = +-z_j) has its contrast-plane angle at one of
    the six marks; the six are three opposite pairs."""
    basis = latent_basis(uniform_factors(P=64, K=3, seed=2))
    angles = lr.latent_angles(basis.Z, basis.U)
    assert angles.shape == (6,)
    h = torch.stack([s * basis.Z[:, j] for s in (1.0, -1.0) for j in range(3)], dim=1)  # (P, 6)
    t = tuning_maps(h, basis.U)  # (3, 6)
    theta = torch.atan2(t[2], t[1])
    torch.testing.assert_close(theta.sort().values, angles.sort().values)


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


def test_one_naming_scheme_for_every_statistic(law):
    labels = [lr.statistic_label(s, law.ell) for s in lr.statistic_list(2)]
    assert labels == [
        "R(1)/R(0), mode 1",
        "R(1)/R(0), mode 2",
        "R(2)/R(0), mode 1",
        "R(2)/R(0), mode 2",
        "band share, mode 1",
        "band share, mode 2",
        "band power ratio, mode 1/2",
    ]


# ------------------------------------------------------------------ recovering the truth
def test_the_primary_statistics_recover_the_prediction(measured, law):
    p = measured.primary
    predicted = lr.law_statistics(law)(law.D)
    torch.testing.assert_close(p.predicted, predicted)
    torch.testing.assert_close(p.values, lr.estimate(_members(law, B, 1)[:, :-1], law)[0])
    assert ((p.values - predicted).abs() <= N_SE * p.se).all()
    assert not p.disagree.any()


def test_the_band_share_estimate_is_the_band_part_of_the_primary(measured, law):
    values, se, predicted = lr.band_share_estimate(law, _members(law, B, 1)[:, :-1])
    band = lr.band_indices(law.D.shape[0])
    torch.testing.assert_close(values, measured.primary.values[band])
    torch.testing.assert_close(se, measured.primary.se[band])
    torch.testing.assert_close(predicted, measured.primary.predicted[band])
    assert [measured.primary.stats[i].kind for i in band] == ["band", "band"]


def test_the_folded_spectra_recover_the_predicted_spectra(measured, law):
    s = measured.spectrum
    K, N = law.D.shape
    assert s.measured.shape == (K + 1, N // 2 + 1) == s.se.shape
    torch.testing.assert_close(s.theory, fold_q(law.D))
    assert ((s.measured[:K] - s.theory).abs() <= N_SE * s.se[:K]).all()
    assert float(s.measured[K].abs().max()) < 1e-12 * float(law.D.max())  # null row 0


def test_the_null_floor_is_the_predictions(measured, law):
    assert measured.spectrum.null_theory == law.null_floor == 0.0  # sigma = 0
    raised = lr.measure_final(law._replace(null_floor=0.25), _members(law, 8, seed=0))
    assert raised.spectrum.null_theory == 0.25


def test_the_excluded_frequencies_are_the_folded_complement_of_the_kept_band(measured, law):
    N = law.D.shape[-1]
    torch.testing.assert_close(measured.spectrum.excluded, ~law.keep[: N // 2 + 1])
    torch.testing.assert_close(measured.ratio.excluded, measured.spectrum.excluded)
    assert measured.spectrum.excluded.any() and not measured.spectrum.excluded[0]


def test_the_ratio_and_lag_profiles_recover_their_predictions(measured, law):
    kept = ~measured.ratio.excluded
    r = measured.ratio
    assert ((r.ratio - r.theory)[kept].abs() <= N_SE * r.se[kept]).all()
    torch.testing.assert_close(r.theory, lr.folded_mode_ratio(law.D))
    p = measured.profile
    assert ((p.measured - p.theory)[:, 1:].abs() <= N_SE * p.se[:, 1:]).all()
    torch.testing.assert_close(p.theory, lr.lag_profile(law.D, law.keep))
    torch.testing.assert_close(p.kernel, lr.prior_profile(law))


def test_the_secondary_statistics_recover_the_prediction(measured, law):
    s = measured.secondary
    K = law.D.shape[0]
    assert s.stats == lr.secondary_list(K) and s.unavailable == {}
    assert ((s.values[:K] - s.predicted[:K]).abs() <= lr.Q_HALF_TOLERANCE).all()
    assert ((s.values[K:] - s.predicted[K:]).abs() <= N_SE * s.se[K:]).all()
    assert not s.disagree.any()


def test_an_undefined_q_half_is_reported_without_stopping_the_measurement(law):
    """A spectrum rising to Nyquist (white noise above a tiny q = 0) has no q_1/2: the shares and
    every primary statistic are still measured."""
    everything = law._replace(keep=torch.ones_like(law.keep))
    final = _members(law, 8, seed=0)
    final[:, :-1, 0] = 1e-9  # q = 0 below Nyquist in every member
    m = lr.measure_final(everything, final)
    assert set(m.secondary.unavailable) == {"q_half"}
    assert "half-power frequency undefined" in m.secondary.unavailable["q_half"]
    assert [s.kind for s in m.secondary.stats] == ["share0", "share0"]
    assert m.primary.values.isfinite().all()


def test_only_the_rising_spectrum_refusal_is_reported_as_unavailable(law, monkeypatch):
    """Any other error of the q_1/2 statistic is a bug and propagates."""

    def broken(keep=None):
        def statistic(D):
            raise ValueError("a bug, not a rising spectrum")

        return statistic

    monkeypatch.setattr(lr, "q_half_statistic", broken)
    with pytest.raises(ValueError, match="a bug"):
        lr.measure_final(law, _members(law, 8, seed=0))


def test_a_wrong_spectrum_is_flagged(law):
    """Both modes drawn at mode 1's predicted spectrum: the ratio is 1, not about 7."""
    wrong = law._replace(D=law.D[[0, 0]])
    measured = lr.measure_final(law, _members(wrong, B, seed=3))
    ratio = lr.statistic_list(2).index(lr.Statistic("ratio", None))
    assert measured.primary.disagree[ratio]


def test_measure_final_refuses_spectra_without_the_null_row(law):
    with pytest.raises(ValueError, match="K \\+ 1"):
        lr.measure_final(law, _members(law, 4, seed=0)[:, :-1])


def test_measure_final_refuses_a_single_member(law):
    with pytest.raises(ValueError, match="at least 2 members"):
        lr.measure_final(law, _members(law, 1, seed=0))


# ------------------------------------------------------------------ the angles (K = 3 only)
def test_at_k_below_3_there_are_no_angles_whatever_is_given(law):
    m = lr.measure_final(
        law, _members(law, 8, seed=0), torch.zeros(8, 2), torch.ones(8, N_THETA_BINS)
    )
    assert m.angles is None


def test_at_k_3_the_angles_are_measured_from_the_members(law3):
    n = 64
    t = _maps(law3, n, seed=4)
    final = mode_spectra(law3.U @ t, law3.U)
    moments = torch.full((n, 2), 0.0, dtype=F64)
    moments[:, 0] = 0.2  # every member's moment is 0.2: |mean| = 0.2, SE 0
    counts = theta_histogram(t[:, 1], t[:, 2])  # (n, N_THETA_BINS)
    m = lr.measure_final(law3, final, moments, counts)
    assert m.angles is not None
    assert float(m.angles.hexatic) == pytest.approx(0.2)
    assert float(m.angles.hexatic_se) == pytest.approx(0.0, abs=1e-12)
    assert m.angles.n_samples == law3.cfg.net.N * n >= lr.HEXATIC_MIN_SAMPLES
    assert m.angles.flagged == (0.2 > lr.HEXATIC_MAX)
    torch.testing.assert_close(m.angles.histogram.counts, counts.sum(0).to(F64))
    assert float(m.angles.histogram.counts.sum()) == n * law3.cfg.net.N
    torch.testing.assert_close(m.angles.histogram.marks, lr.angle_marks(law3))


def test_at_k_3_the_angle_inputs_are_required(law3):
    final = _members(law3, 8, seed=0)
    with pytest.raises(ValueError, match="hexatic moments and angle histograms"):
        lr.measure_final(law3, final)
    with pytest.raises(ValueError, match="hexatic moments and angle histograms"):
        lr.measure_final(law3, final, hex_moments=torch.zeros(8, 2))
    with pytest.raises(ValueError, match=f"\\(B, {N_THETA_BINS}\\)"):
        lr.measure_final(law3, final, torch.zeros(8, 2), torch.ones(8, N_THETA_BINS - 1))
    with pytest.raises(ValueError, match="hexatic moments must be \\(B, 2\\) = \\(8, 2\\)"):
        lr.measure_final(law3, final, torch.zeros(7, 2), torch.ones(8, N_THETA_BINS))
    with pytest.raises(ValueError, match="angle histograms must be"):
        lr.measure_final(law3, final, torch.zeros(8, 2), torch.ones(9, N_THETA_BINS))


def test_below_the_neuron_sample_threshold_the_hexatic_is_not_flagged(law3):
    """N B = 16 x 8 = 128 < HEXATIC_MIN_SAMPLES: |mean m| = 0.5 is reported, not called."""
    t = _maps(law3, 8, seed=5)
    moments = torch.tensor([[0.5, 0.0]] * 8, dtype=F64)
    m = lr.measure_final(law3, mode_spectra(law3.U @ t, law3.U), moments,
                         theta_histogram(t[:, 1], t[:, 2]))  # fmt: skip
    assert m.angles.n_samples == 128 < lr.HEXATIC_MIN_SAMPLES
    assert m.angles.flagged is None and float(m.angles.hexatic) == pytest.approx(0.5)


# ------------------------------------------------------------------ the relaxation view
def test_a_stationary_ensemble_shows_no_relaxation(law):
    """Records drawn independently from the predicted law: the normalised statistics scatter
    around 0 and the fit finds no deficit beyond its SE."""
    R, members = 24, 64
    records = torch.stack([_members(law, members, seed=10 + r) for r in range(R)])[:, :, :-1]
    epochs = torch.arange(R, dtype=F64) * 250
    view = lr.measure_relaxation(law, epochs, records, law.B, start=float(epochs[R // 2]))
    c = view.curves
    S = len(lr.statistic_list(law.D.shape[0]))
    assert c.values.shape == (R, S) and c.fitted.shape == (R - R // 2, S)
    torch.testing.assert_close(c.fit_epochs, epochs[R // 2 :])
    assert c.labels == [lr.statistic_label(s, law.ell) for s in view.stats]
    assert c.band == lr.RELAXATION_BAND
    assert (view.fit.deficit.abs() <= 3 * view.fit.deficit_se).all()
    raw = lr.law_statistics(law)(records.mean(1))
    torch.testing.assert_close(c.values, (raw - view.fit.asymptote) / view.budget)
    torch.testing.assert_close(view.budget, lr.budget_se(law, law.B))


def test_the_relaxation_budget_is_at_the_base_b_given(law):
    R = 8
    records = torch.stack([_members(law, 8, seed=r) for r in range(R)])[:, :, :-1]
    epochs = torch.linspace(0, law.cfg.train.n_epochs, R, dtype=F64)
    view = lr.measure_relaxation(law, epochs, records, base_B=16)
    torch.testing.assert_close(view.budget, lr.budget_se(law, 16))
    assert view.curves.start == law.cfg.train.n_epochs / 2  # default: the second half
    assert float(view.curves.fit_epochs[0]) >= view.curves.start
