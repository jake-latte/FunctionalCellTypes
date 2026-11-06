"""Tests for fct/spectrum.py: tuning maps, their spectra and summaries, on synthetic inputs.

Every test builds its inputs with a known answer: maps drawn from a chosen spectrum, a null
complement drawn explicitly, a bivariate Gaussian whose cos 4 theta has a closed form, AR(1)
chains whose integrated autocorrelation time is known, and the limits note's d_q from the
script that produced its numbers (loaded by path: it is not part of the fct package).
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from fct import spectrum as sp

# The script that produced the limits note's numbers is the reference implementation of
# half_power_q, autocorrelation and lag-1; it is a standalone check outside the fct package,
# so it is loaded by path rather than imported.
SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "latex"
    / "notes"
    / "scripts"
    / "check_linear_soln_limits.py"
)
F64 = torch.float64


@pytest.fixture(scope="module")
def limits():
    spec = importlib.util.spec_from_file_location("check_linear_soln_limits", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def uniform_factors(P: int, K: int, a: float = 1.0, seed: int = 0) -> dict[str, torch.Tensor]:
    """Latents z0..z{K-1} iid uniform on [0, a], as the linear-teacher task draws them."""
    gen = torch.Generator().manual_seed(seed)
    z = a * torch.rand((P, K), dtype=F64, generator=gen)
    return {f"z{k}": z[:, k] for k in range(K)}


def gaussian_field(d: torch.Tensor, n_draws: int, gen: torch.Generator) -> torch.Tensor:
    """(n_draws, N) real stationary Gaussian fields on the ring with unitary-DFT power d_q.

    t = C^{1/2} xi for white xi and the circulant C of eigenvalues d_q: in Fourier space
    t~(q) = sqrt(d_q) xi~(q) with E|xi~(q)|^2 = 1 (random phases), so E|t~(q)|^2 = d_q. d must
    be even in q (d_q = d_{-q}) for t to be real.
    """
    xi = torch.randn((n_draws, d.shape[-1]), dtype=F64, generator=gen)
    return torch.fft.ifft(d.sqrt() * torch.fft.fft(xi, dim=-1), dim=-1).real


def smooth_spectrum(N: int, ell: float, floor: float) -> torch.Tensor:
    """d_q = sqrt(2 pi) ell exp(-2 (pi ell q / N)^2) + floor: a Gaussian kernel's, even in q."""
    q = torch.fft.fftfreq(N, d=1.0 / N, dtype=F64)
    return math.sqrt(2 * math.pi) * ell * torch.exp(-2 * (math.pi * ell * q / N) ** 2) + floor


# ------------------------------------------------------------------ latent_basis
def test_latent_basis_is_the_sign_fixed_thin_svd_of_the_stacked_latents():
    factors = uniform_factors(P=200, K=3)
    basis = sp.latent_basis(factors)
    Z = torch.stack([factors[f"z{k}"] for k in range(3)], dim=1)
    assert torch.equal(basis.Z, Z)
    assert basis.U.shape == (200, 3) and basis.s2.shape == (3,)
    torch.testing.assert_close(basis.U.T @ basis.U, torch.eye(3, dtype=F64), atol=1e-12, rtol=0)
    # U spans col(Z) with singular values s: U^T Z Z^T U = diag(s2)
    torch.testing.assert_close(
        basis.U.T @ Z @ Z.T @ basis.U, torch.diag(basis.s2), atol=1e-9, rtol=1e-12
    )
    assert torch.all(basis.s2[:-1] >= basis.s2[1:])
    assert torch.all(basis.U.sum(dim=0) >= 0)


def test_latent_basis_first_column_is_the_mean_direction_with_the_formula_s2():
    """Latents on [0, a] at P >> K: s_1^2 ~ P a^2 (3K+1)/12, s_k^2 ~ P a^2/12.

    u_1 is along Z 1 (the sum of the latents), whose cosine with the constant pattern vector is
    E[sum z] / sqrt(E[(sum z)^2]) = sqrt(3K / (3K + 1)).
    """
    P, K = 4000, 3
    basis = sp.latent_basis(uniform_factors(P, K))
    ones = torch.ones(P, dtype=F64) / math.sqrt(P)
    assert float(basis.U[:, 0] @ ones) == pytest.approx(math.sqrt(3 * K / (3 * K + 1)), rel=0.01)
    formula = torch.tensor([P * (3 * K + 1) / 12] + [P / 12] * (K - 1), dtype=F64)
    torch.testing.assert_close(basis.s2, formula, rtol=0.1, atol=0)


def test_latent_basis_orders_latents_by_index_not_by_string():
    """z10 must come after z9, not after z1."""
    P, K = 60, 11
    factors = uniform_factors(P, K)
    shuffled = {name: factors[name] for name in sorted(factors)}  # z0, z1, z10, z2, ...
    basis = sp.latent_basis(shuffled)
    assert torch.equal(basis.Z[:, 10], factors["z10"])


def test_latent_basis_rejects_symmetric_latents_without_a_mean_direction():
    """Latents on [-a, a] have K near-degenerate singular values: u_1 is a random direction."""
    gen = torch.Generator().manual_seed(15)
    z = 2 * torch.rand((256, 3), dtype=F64, generator=gen) - 1
    with pytest.raises(ValueError, match="symmetric.*mean direction is undefined"):
        sp.latent_basis({f"z{k}": z[:, k] for k in range(3)})


def test_latent_basis_accepts_one_latent_on_zero_to_a():
    basis = sp.latent_basis(uniform_factors(500, 1))
    assert basis.U.shape == (500, 1)


def test_latent_basis_rejects_one_symmetric_latent_by_its_cosine():
    z = 2 * torch.rand(256, dtype=F64, generator=torch.Generator().manual_seed(22)) - 1
    with pytest.raises(ValueError, match="symmetric.*mean direction is undefined") as error:
        sp.latent_basis({"z0": z})
    assert "degenerate" not in str(error.value)  # one latent has no second singular value


def frame_accepted(z: torch.Tensor) -> bool:
    try:
        sp.latent_basis({f"z{k}": z[:, k] for k in range(z.shape[1])})
    except ValueError as error:
        assert "mean direction is undefined" in str(error)
        return False
    return True


@pytest.mark.parametrize("P, K", [(16, 2), (16, 3), (32, 2)])
def test_latent_basis_frame_check_at_small_p(P, K):
    """Never rejects latents on [0, a]; accepts few symmetric draws even at P = 16.

    The cosine test alone accepts about 6% of symmetric draws at P = 16; adding the gap
    s_1^2 / s_2^2 >= 2 brings that to 1.7% at K = 2 and 0.7% at K = 3 (see latent_basis).
    """
    gen = torch.Generator().manual_seed(23)
    n = 2000
    positive = torch.rand((n, P, K), dtype=F64, generator=gen)
    assert all(frame_accepted(z) for z in positive)
    accepted = sum(frame_accepted(2 * z - 1) for z in positive)
    assert accepted / n < 0.03, f"{accepted} of {n} symmetric draws accepted"


@pytest.mark.parametrize(
    "factors, match",
    [
        ({}, "no latents"),
        ({name: torch.rand(5, dtype=F64) for name in ("z0", "z1", "z01")}, "collide"),
        ({"z0": torch.ones(6, dtype=F64), "z1": torch.ones(6, dtype=F64)}, "rank-deficient"),
        ({"z0": torch.zeros(5, dtype=F64), "z2": torch.zeros(5, dtype=F64)}, "z0..z"),
        ({"z0": torch.zeros(5, dtype=F64), "where": torch.zeros(5, dtype=F64)}, "where"),
        ({"z0": torch.zeros((5, 2), dtype=F64)}, "1-D"),
        ({"z0": torch.zeros(5, dtype=F64), "z1": torch.zeros(4, dtype=F64)}, "length"),
        ({f"z{k}": torch.rand(3, dtype=F64) for k in range(3)}, "K < P"),
        ({"z0": torch.tensor([1.0, float("nan"), 2.0], dtype=F64)}, "finite"),
    ],
)
def test_latent_basis_rejects_bad_factors(factors, match):
    with pytest.raises(ValueError, match=match):
        sp.latent_basis(factors)


def test_non_tensor_inputs_raise_type_error():
    with pytest.raises(TypeError, match="torch tensor"):
        sp.latent_basis({"z0": [1.0, 2.0, 3.0]})
    with pytest.raises(TypeError, match="torch tensor"):
        sp.power_spectrum(np.ones((2, 8)))


# ------------------------------------------------------------------ tuning_maps, spectra
def test_tuning_maps_project_activity_on_the_basis():
    gen = torch.Generator().manual_seed(1)
    U = sp.latent_basis(uniform_factors(30, 3)).U
    h = torch.randn((30, 16), dtype=F64, generator=gen)
    torch.testing.assert_close(sp.tuning_maps(h, U), U.T @ h, atol=0, rtol=0)
    batched = torch.randn((4, 30, 16), dtype=F64, generator=gen)
    torch.testing.assert_close(sp.tuning_maps(batched, U)[2], U.T @ batched[2])


def test_power_spectrum_is_unitary_and_keeps_the_mean():
    """Parseval with the unitary DFT, and a constant map puts all its power at q = 0."""
    gen = torch.Generator().manual_seed(2)
    t = torch.randn((3, 32), dtype=F64, generator=gen)
    d = sp.power_spectrum(t)
    torch.testing.assert_close(d.sum(dim=-1), t.square().sum(dim=-1), atol=1e-12, rtol=0)
    constant = torch.full((1, 32), 2.0, dtype=F64)
    expected = torch.zeros((1, 32), dtype=F64)
    expected[0, 0] = 32 * 4.0
    torch.testing.assert_close(sp.power_spectrum(constant), expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(sp.power_spectrum(constant[0]), expected[0], atol=1e-12, rtol=0)


def test_power_spectrum_recovers_a_known_spectrum_within_sampling_error():
    """h = sum_k u_k (x) t_k with t_k Gaussian fields of known d_q^(k), n = 10^4 networks.

    Each network gives one sample of |t~_k(q)|^2, exponential with mean d_q (chi^2_1 at
    q = 0, N/2: variance 2 d_q^2), so the mean over n has standard error d_q sqrt(c/n).
    """
    P, K, N, n = 40, 3, 64, 10_000
    gen = torch.Generator().manual_seed(3)
    U = sp.latent_basis(uniform_factors(P, K)).U
    d_true = torch.stack([smooth_spectrum(N, ell, 0.01) for ell in (2.0, 4.0, 8.0)])  # (K, N)
    t = torch.stack([gaussian_field(d_true[k], n, gen) for k in range(K)], dim=1)  # (n, K, N)
    h = U @ t  # (n, P, N): each network's activity lies in col(Z)
    d_hat = sp.power_spectrum(sp.tuning_maps(h, U)).mean(dim=0)
    var_factor = torch.ones(N, dtype=F64)
    var_factor[[0, N // 2]] = 2.0
    z = (d_hat - d_true) / (d_true * (var_factor / n).sqrt())
    assert float(z.abs().max()) < 4.5, f"max |z| = {float(z.abs().max()):.2f}"


def test_null_power_equals_the_explicit_projection_on_a_random_complement():
    P, K, N = 30, 3, 16
    gen = torch.Generator().manual_seed(4)
    U = sp.latent_basis(uniform_factors(P, K)).U
    h = torch.randn((P, N), dtype=F64, generator=gen)
    complete = torch.linalg.qr(U, mode="complete").Q[:, K:]  # (P, P-K), orthogonal to U
    rotation = torch.linalg.qr(torch.randn((P - K, P - K), dtype=F64, generator=gen)).Q
    U_perp = complete @ rotation  # any orthonormal basis of the complement gives the same mean
    explicit = sp.power_spectrum(U_perp.T @ h).mean(dim=0)
    torch.testing.assert_close(sp.null_power(h, U), explicit, atol=1e-12, rtol=0)


def test_null_power_is_zero_for_activity_in_the_latent_span():
    """sigma -> 0: h = U t has no null power; a subtraction of totals would leave round-off."""
    P, K, N = 40, 3, 64
    gen = torch.Generator().manual_seed(16)
    U = sp.latent_basis(uniform_factors(P, K)).U
    t = 30.0 * torch.randn((K, N), dtype=F64, generator=gen)  # latent power ~ 10^3
    assert float(sp.null_power(U @ t, U).abs().max()) < 1e-25
    assert float(sp.mode_spectra(U @ t, U)[K].abs().max()) < 1e-25


def test_null_power_resolves_a_small_floor_under_large_latent_power():
    P, K, N, floor = 40, 3, 64, 1e-6
    gen = torch.Generator().manual_seed(17)
    U = sp.latent_basis(uniform_factors(P, K)).U
    U_perp = torch.linalg.qr(U, mode="complete").Q[:, K:]
    t = 30.0 * torch.randn((K, N), dtype=F64, generator=gen)
    noise = math.sqrt(floor) * torch.randn((P - K, N), dtype=F64, generator=gen)
    expected = sp.power_spectrum(noise).mean(dim=0)  # the null directions' power, exactly
    null = sp.null_power(U @ t + U_perp @ noise, U)
    assert float(((null - expected) / expected).abs().max()) < 1e-9


def test_null_power_recovers_a_white_floor():
    """Null directions filled with white noise of variance sigma0^2 give a flat sigma0^2."""
    P, K, N, n, sigma0_2 = 40, 3, 64, 2000, 0.5
    gen = torch.Generator().manual_seed(5)
    U = sp.latent_basis(uniform_factors(P, K)).U
    U_perp = torch.linalg.qr(U, mode="complete").Q[:, K:]
    signal = U @ gaussian_field(smooth_spectrum(N, 4.0, 0.0), n * K, gen).reshape(n, K, N)
    noise = math.sqrt(sigma0_2) * U_perp @ torch.randn((n, P - K, N), dtype=F64, generator=gen)
    floor = sp.null_power(signal + noise, U).mean(dim=0)  # (N,), one sample per network
    # each network's null mean averages P - K exponentials: variance sigma0^4 / (P - K)
    se = sigma0_2 * math.sqrt(2 / ((P - K) * n))  # 2 covers the chi^2_1 bins at q = 0, N/2
    assert float((floor - sigma0_2).abs().max()) < 4.5 * se


def test_mode_spectra_stacks_the_modes_and_the_null_mean():
    P, K, N = 20, 2, 8
    gen = torch.Generator().manual_seed(6)
    U = sp.latent_basis(uniform_factors(P, K)).U
    h = torch.randn((5, P, N), dtype=F64, generator=gen)
    d = sp.mode_spectra(h, U)
    assert d.shape == (5, K + 1, N)
    torch.testing.assert_close(d[:, :K], sp.power_spectrum(sp.tuning_maps(h, U)))
    torch.testing.assert_close(d[:, K], sp.null_power(h, U))


def test_spectra_are_float64_safe_and_keep_float32():
    U = sp.latent_basis(uniform_factors(20, 2)).U
    h = torch.randn((20, 8), dtype=F64, generator=torch.Generator().manual_seed(18))
    assert sp.mode_spectra(h, U).dtype == F64
    assert sp.mode_spectra(h.float(), U.float()).dtype == torch.float32


@pytest.mark.parametrize(
    "call, match",
    [
        (lambda h, U: sp.tuning_maps(h[0], U), "at least 2-D"),
        (lambda h, U: sp.tuning_maps(h[:-1], U), "rows"),
        (lambda h, U: sp.tuning_maps(h, U[:, :0]), "1 <= K < P"),
        (lambda h, U: sp.tuning_maps(h, 2 * U), "orthonormal"),
        (lambda h, U: sp.tuning_maps(h * float("inf"), U), "finite"),
        (lambda h, U: sp.null_power(h, torch.eye(20, dtype=F64)), "K < P"),
        (lambda h, U: sp.power_spectrum(h[0, 0]), "scalar"),
        (lambda h, U: sp.power_spectrum(h.to(torch.int64)), "floating"),
    ],
)
def test_maps_and_spectra_reject_bad_inputs(call, match):
    U = sp.latent_basis(uniform_factors(20, 2)).U
    h = torch.randn((20, 8), dtype=F64, generator=torch.Generator().manual_seed(19))
    with pytest.raises(ValueError, match=match):
        call(h, U)


# ------------------------------------------------------------------ fold_q and summaries
def test_fold_q_averages_plus_and_minus_q():
    d = torch.arange(8, dtype=F64).reshape(1, 8)  # q order 0, 1, 2, 3, 4, -3, -2, -1
    expected = torch.tensor([[0.0, (1 + 7) / 2, (2 + 6) / 2, (3 + 5) / 2, 4.0]], dtype=F64)
    torch.testing.assert_close(sp.fold_q(d), expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="even"):
        sp.fold_q(torch.ones(7, dtype=F64))


def test_summaries_agree_with_the_script_on_random_spectra(limits):
    """half_power_q, autocorrelation and lag-1 of the module and the script, on random d."""
    rng = np.random.default_rng(7)
    for N in (8, 16, 64):
        d = rng.exponential(size=N)
        d = 0.5 * (d + np.roll(d[::-1], 1))  # even in q, as a spectrum of a real map
        d[0] = max(d[0], d[N // 2])  # q_1/2 needs d_0 >= d_{N/2}
        d_t = torch.from_numpy(d)
        summary = sp.spectrum_summary(d_t)
        np.testing.assert_allclose(sp.autocorrelation(d_t).numpy(), limits.autocorrelation(d))
        np.testing.assert_allclose(summary.R.numpy(), limits.autocorrelation(d))
        assert sp.half_power_q(d_t) == limits.half_power_q(d)
        assert summary.lag1 == pytest.approx(limits.autocorr_lag1(d), rel=1e-12)
        assert summary.share0 == pytest.approx(d[0] / d.sum(), rel=1e-12)
        assert (summary.d0, summary.d_nyq) == (d[0], d[N // 2])


@pytest.mark.parametrize(
    "name, mode1, mode2",
    [
        # (q_half, share0, lag1) of modes 1 and 2, as quoted in linear_solution_limits.tex
        ("P1 prior-shaped (weak target, no rate penalty)", (2, 0.157, 0.969), (2, 0.156, 0.966)),
        ("P2 rate-penalty whitening", (6, 0.077, 0.917), (4, 0.111, 0.944)),
        ("P3 target condensation", (1, 0.343, 0.985), (2, 0.164, 0.967)),
    ],
)
def test_spectrum_summary_gives_the_limits_note_numbers(limits, name, mode1, mode2):
    _, spectra = limits.prediction_spectra(*limits.PREDICTIONS[name])
    for d, (q_half, share0, lag1) in zip(spectra, (mode1, mode2)):
        summary = sp.spectrum_summary(torch.from_numpy(d))
        assert sp.half_power_q(torch.from_numpy(d)) == q_half
        assert summary.share0 == pytest.approx(share0, abs=5e-4)
        assert summary.lag1 == pytest.approx(lag1, abs=5e-4)


def test_summaries_batch_over_leading_dims():
    rng = np.random.default_rng(20)
    d = rng.exponential(size=(3, 2, 16))
    d = torch.from_numpy(0.5 * (d + np.roll(d[..., ::-1], 1, axis=-1)))
    d[..., 0] += 5.0  # peaked at q = 0, so every half-power frequency is defined
    batched = sp.spectrum_summary(d)
    assert batched.share0.shape == batched.lag1.shape == batched.d0.shape == (3, 2)
    assert batched.R.shape == (3, 2, 16)
    for i in range(3):
        for j in range(2):
            one = sp.spectrum_summary(d[i, j])
            assert isinstance(one.share0, float) and isinstance(one.lag1, float)
            assert isinstance(sp.half_power_q(d[i, j]), int)
            assert float(batched.share0[i, j]) == pytest.approx(one.share0, rel=1e-14)
            assert float(batched.lag1[i, j]) == pytest.approx(one.lag1, rel=1e-14)
            assert (float(batched.d0[i, j]), float(batched.d_nyq[i, j])) == (one.d0, one.d_nyq)
    q_half = sp.half_power_q(d)
    assert q_half.shape == (3, 2) and q_half.dtype == torch.int64
    assert q_half.tolist() == [[sp.half_power_q(d[i, j]) for j in range(2)] for i in range(3)]


def test_spectrum_summary_runs_on_noisy_white_null_rows():
    """One record of a white spectrum has d_0 < d_{N/2} half the time; the summary still runs."""
    gen = torch.Generator().manual_seed(24)
    d = -torch.log(torch.rand((4, 64, 64), dtype=F64, generator=gen))  # exponential power
    summary = sp.spectrum_summary(d)
    assert summary.lag1.shape == (4, 64) and "q_half" not in summary._fields
    assert bool((d[..., 0] < d[..., 32]).any())  # half_power_q would raise on these rows


def test_half_power_q_rejects_a_spectrum_rising_to_nyquist():
    d = torch.linspace(1.0, 2.0, 8, dtype=F64)  # d_0 = 1 < d_{N/2}
    with pytest.raises(ValueError, match="d_0 .* d_N/2"):
        sp.half_power_q(d)
    batch = torch.stack([d, d.flip(-1), d])  # rows 0 and 2 rise to Nyquist
    with pytest.raises(ValueError, match="2 of 3 spectra.*first at batch index \\[0\\]"):
        sp.half_power_q(batch)


def test_gaussian_width_inverts_a_gaussian_autocorrelation():
    N, width = 64, 5.0
    r = torch.arange(N, dtype=F64)
    R = torch.exp(-(torch.minimum(r, N - r) ** 2) / (2 * width**2))
    assert sp.gaussian_width(R, 4.0) == pytest.approx(width, rel=1e-12)
    with pytest.raises(ValueError, match="integer"):
        sp.gaussian_width(R, 2.5)
    with pytest.raises(ValueError, match="R\\(ell\\)/R\\(0\\)"):
        sp.gaussian_width(-R, 4.0)
    for lag in (0, N):
        with pytest.raises(ValueError, match="integer lag"):
            sp.gaussian_width(R, lag)
    with pytest.raises(ValueError, match="scalar"):
        sp.gaussian_width(R[0], 1)
    widths = sp.gaussian_width(torch.stack([R, R**4]), 4.0)  # R^4 is a Gaussian of width / 2
    torch.testing.assert_close(widths, torch.tensor([width, width / 2], dtype=F64))


@pytest.mark.parametrize(
    "call, match",
    [
        (lambda: sp.spectrum_summary(torch.tensor(1.0, dtype=F64)), "scalar"),
        (lambda: sp.spectrum_summary(torch.ones(7, dtype=F64)), "even"),
        (lambda: sp.spectrum_summary(torch.zeros(8, dtype=F64)), "positive"),
        (lambda: sp.half_power_q(torch.full((8,), float("nan"), dtype=F64)), "finite"),
        (lambda: sp.spectrum_summary(torch.ones((0, 8), dtype=F64)), "empty"),
        (lambda: sp.half_power_q(torch.ones((0, 8), dtype=F64)), "empty"),
        (lambda: sp.autocorrelation(torch.ones(8, dtype=torch.int64)), "floating"),
    ],
)
def test_summaries_reject_bad_inputs(call, match):
    with pytest.raises(ValueError, match=match):
        call()


# ------------------------------------------------------------------ angle statistic
def per_network_cos4(t2: torch.Tensor, t3: torch.Tensor) -> tuple[float, float]:
    """Mean over networks (leading axis) of the per-network statistic, and its standard error."""
    mean, sem = sp.member_mean_sem(sp.contrast_angle_statistic(t2, t3))
    return float(mean), float(sem)


def isotropic_networks(P: int, N: int, n: int, seed: int) -> torch.Tensor:
    """(n, P, N): every neuron's activity iid N(0, I_P), the null of prediction 6."""
    return torch.randn((n, P, N), dtype=F64, generator=torch.Generator().manual_seed(seed))


def test_angle_statistic_is_zero_in_the_orthonormal_frame():
    P, N, n = 100, 64, 500
    factors = uniform_factors(P, 3)
    t = sp.tuning_maps(isotropic_networks(P, N, n, seed=8), sp.latent_basis(factors).U)
    mean, se = per_network_cos4(t[:, 1], t[:, 2])
    assert abs(mean) < 3 * se and se < 0.01


def test_angle_statistic_is_negative_in_the_oblique_raw_latent_frame():
    """Tuning to the raw (z_2, z_3), unit-normalised, is the oblique frame the note warns about.

    For an isotropic cloud the pair is bivariate normal with unit variances and correlation rho
    (the cosine between z_2 and z_3, about 3/4 for latents on [0, 1]). Its angle phi from the
    bisector, doubled, is wrapped Cauchy with concentration
    r = (sqrt(1+rho) - sqrt(1-rho)) / (sqrt(1+rho) + sqrt(1-rho)), so <cos 4 phi> = r^2 and
    <cos 4 theta> = <cos(4 phi + pi)> = -r^2.
    """
    P, N, n = 100, 64, 500
    factors = uniform_factors(P, 3)
    v2, v3 = (factors[name] / factors[name].norm() for name in ("z1", "z2"))
    rho = float(v2 @ v3)
    r = (math.sqrt(1 + rho) - math.sqrt(1 - rho)) / (math.sqrt(1 + rho) + math.sqrt(1 - rho))
    h = isotropic_networks(P, N, n, seed=9)
    mean, se = per_network_cos4(v2 @ h, v3 @ h)
    assert mean + 3 * se < -0.1
    assert abs(mean - (-(r**2))) < 3 * se


def test_angle_statistic_is_one_on_the_axes():
    t2 = torch.tensor([1.0, 0.0, -2.0, 0.0], dtype=F64)
    t3 = torch.tensor([0.0, 3.0, 0.0, -1.0], dtype=F64)
    assert sp.contrast_angle_statistic(t2, t3) == pytest.approx(1.0, abs=1e-12)
    diagonal = torch.ones(4, dtype=F64)
    batched = sp.contrast_angle_statistic(torch.stack([t2, diagonal]), torch.stack([t3, diagonal]))
    torch.testing.assert_close(batched, torch.tensor([1.0, -1.0], dtype=F64))


def test_theta_histogram_uses_36_fixed_bins_on_minus_pi_to_pi():
    width = 2 * math.pi / 36
    centres = -math.pi + width * (torch.arange(36, dtype=F64) + 0.5)
    angles = torch.cat([centres, centres[[3]], torch.tensor([math.pi], dtype=F64)])
    hist = sp.theta_histogram(torch.cos(angles), torch.sin(angles))
    expected = torch.ones(36, dtype=torch.int64)
    expected[3] += 1
    expected[35] += 1  # theta = pi is in the last bin, (pi - width, pi]
    assert torch.equal(hist, expected)
    # atan2(-0.0, -1) = -pi is the same angle as pi, so it lands in the last bin too
    minus_pi = sp.theta_histogram(torch.tensor([-1.0], dtype=F64), torch.tensor([-0.0], dtype=F64))
    assert int(minus_pi[35]) == 1 and int(minus_pi.sum()) == 1
    batched = sp.theta_histogram(
        torch.stack([torch.cos(angles), torch.ones_like(angles)]),
        torch.stack([torch.sin(angles), torch.zeros_like(angles)]),
    )
    assert batched.shape == (2, 36)
    assert torch.equal(batched[0], expected)
    assert int(batched[1, 17]) == angles.numel()  # theta = 0 is the right edge of bin 17


@pytest.mark.parametrize("fn", [sp.contrast_angle_statistic, sp.theta_histogram])
def test_angle_functions_reject_bad_inputs(fn):
    with pytest.raises(ValueError, match="same shape"):
        fn(torch.ones(4, dtype=F64), torch.ones(5, dtype=F64))
    with pytest.raises(ValueError, match="non-empty"):
        fn(torch.ones((2, 0), dtype=F64), torch.ones((2, 0), dtype=F64))
    with pytest.raises(ValueError, match="finite"):
        fn(torch.full((4,), float("nan"), dtype=F64), torch.ones(4, dtype=F64))


# ------------------------------------------------------------------ records and members
def ar1_chains(phi: float, n_records: int, n_chains: int, seed: int) -> torch.Tensor:
    """(n_records, n_chains) stationary AR(1) chains with unit variance.

    Their integrated autocorrelation time is tau = (1 + phi) / (1 - phi).
    """
    gen = torch.Generator().manual_seed(seed)
    noise = math.sqrt(1 - phi**2) * torch.randn((n_records, n_chains), dtype=F64, generator=gen)
    x = torch.empty_like(noise)
    x[0] = torch.randn(n_chains, dtype=F64, generator=gen)
    for i in range(1, n_records):  # genuine sequential state: each record depends on the last
        x[i] = phi * x[i - 1] + noise[i]
    return x


def test_integrated_autocorr_time_of_ar1_chains():
    phi = 0.8
    tau = sp.integrated_autocorr_time(ar1_chains(phi, 4000, 200, seed=10), burn_in=0)
    assert tau.shape == (200,)
    assert float(tau.mean()) == pytest.approx((1 + phi) / (1 - phi), rel=0.05)


def test_record_sem_is_calibrated_on_correlated_chains():
    """The spread of chain means over independent chains matches the per-chain sem."""
    records = ar1_chains(0.8, 4000, 400, seed=11)
    sem = sp.record_sem(records, burn_in=0)
    spread = records.mean(dim=0).std()
    assert float(sem.square().mean().sqrt()) == pytest.approx(float(spread), rel=0.1)
    naive = records.std(dim=0) / math.sqrt(4000)  # what ignoring the correlation would give
    assert float((sem / naive).mean()) == pytest.approx(3.0, rel=0.1)  # sqrt(tau) = 3


def test_integrated_autocorr_time_of_anticorrelated_chains():
    """AR(1) with phi = -0.5 has tau = (1 + phi) / (1 - phi) = 1/3."""
    tau = sp.integrated_autocorr_time(ar1_chains(-0.5, 4000, 200, seed=21), burn_in=0)
    assert float(tau.mean()) == pytest.approx(1 / 3, rel=0.1)


@pytest.mark.parametrize("b", [0.8, 0.9])
def test_record_sem_is_calibrated_on_a_slow_plus_alternating_chain(b):
    """sqrt(1-b) AR(0.95) + sqrt(b) AR(-0.9): the alternation must not hide the slow part.

    tau = (1 - b) 39 + b (0.1 / 1.9): 7.8 at b = 0.8, 3.9 at b = 0.9. Sokal's window stops at
    W = 1 here and gives tau about 1.
    """
    slow = ar1_chains(0.95, 4000, 200, seed=25)
    alternating = ar1_chains(-0.9, 4000, 200, seed=26)
    records = math.sqrt(1 - b) * slow + math.sqrt(b) * alternating
    sem = sp.record_sem(records, burn_in=0)
    spread = records.mean(dim=0).std()
    assert float(sem.square().mean().sqrt()) == pytest.approx(float(spread), rel=0.2)


def test_record_sem_of_independent_records_is_the_naive_sem():
    records = torch.randn((4000, 200), dtype=F64, generator=torch.Generator().manual_seed(12))
    ratio = sp.record_sem(records, burn_in=0) / (records.std(dim=0) / math.sqrt(4000))
    assert float(ratio.mean()) == pytest.approx(1.0, rel=0.05)


def test_records_drop_the_burn_in_and_keep_trailing_shape():
    records = torch.randn((300, 4, 8), dtype=F64, generator=torch.Generator().manual_seed(13))
    records[:100] += 50.0  # a transient that burn-in must remove
    mean = sp.average_records(records, burn_in=100)
    assert mean.shape == (4, 8)
    torch.testing.assert_close(mean, records[100:].mean(dim=0))
    sem = sp.record_sem(records, burn_in=100)
    assert sem.shape == (4, 8)
    assert float(sem.max()) < 0.5


def test_record_sem_of_constant_records_is_zero():
    sem = sp.record_sem(torch.ones((200, 3), dtype=F64), burn_in=0)
    assert torch.equal(sem, torch.zeros(3, dtype=F64))


@pytest.mark.parametrize(
    "call, match",
    [
        (lambda r: sp.average_records(r, burn_in=-1), "burn_in"),
        (lambda r: sp.average_records(r, burn_in=200), "burn_in"),
        (lambda r: sp.record_sem(r[:, 0, 0] * float("nan"), 0), "finite"),
        (lambda r: sp.record_sem(r, burn_in=1), "at least 200"),
        (lambda r: sp.record_sem(torch.arange(200.0, dtype=F64), 0), "too short"),
    ],
)
def test_records_reject_bad_inputs(call, match):
    records = torch.randn((200, 2, 3), dtype=F64, generator=torch.Generator().manual_seed(14))
    with pytest.raises(ValueError, match=match):
        call(records)


def test_member_mean_sem_is_the_spread_across_networks():
    values = torch.tensor([[1.0, 10.0], [3.0, 10.0], [5.0, 10.0]], dtype=F64)
    mean, sem = sp.member_mean_sem(values)
    torch.testing.assert_close(mean, torch.tensor([3.0, 10.0], dtype=F64))
    torch.testing.assert_close(sem, torch.tensor([2.0 / math.sqrt(3), 0.0], dtype=F64))
    with pytest.raises(ValueError, match="at least 2"):
        sp.member_mean_sem(values[:1])
