"""Tests for fct.linear_theory: the per-mode linear solution and the ansatz-free kernel from a Config.

Ground truths, all synthetic:
  * the numbers quoted in latex/notes/fct_experiment_plan.tex (Sections sec:P1-P3, sec:smoke) at
    sigma^2 = 0.01, and their sigma^2 = 0 counterparts, computed from the formula singular values
    of the uniform latent task;
  * the reference script latex/notes/scripts/check_linear_soln_limits.py, loaded by path, for a
    direct comparison of the port on several parameter sets including sigma^2 = 0;
  * the limits note's N = 12 check (Section sec:rederive): the ansatz-free fixed point against a
    brute-force evaluation of the NP x NP master equation and against the per-mode roots;
  * latents built with exact singular values, so that `predicted_spectra` from a `Config` must
    reproduce the quoted numbers through the whole pipeline.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from fct import linear_theory as lt
from fct.config import Config, DataConfig, NetConfig, TrainConfig
from fct.spatial import KernelSpec, PopulationSpec, SpatialConfig
from fct.tasks import Dataset

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "latex"
    / "notes"
    / "scripts"
    / "check_linear_soln_limits.py"
)

# The common setting of the plan's dictionary (Section sec:dictionary) and its smoke version.
FULL = {"N": 64, "ell": 4.0, "P": 256, "m": 8, "K": 3}
SMOKE = {"N": 16, "ell": 2.0, "P": 32, "m": 4, "K": 2}
A_LATENT = 1.0  # latent range [0, a]
SIG2 = 0.01  # the quoted numbers' input noise
Q_MAX = 12  # the ratio d_q(mode 1) / d_q(mode 2) is quoted for |q| = 0..12


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("check_linear_soln_limits", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ synthetic builders


def rate_lambda_for(r: float, beta: float) -> float:
    """train.rate_lambda that makes the rate coefficient r (today's relation r = 2 beta rate_lambda)."""
    return r / (2.0 * beta)


def ring_config(
    N: int,
    ell: float,
    P: int,
    m: int,
    K: int,
    beta: float,
    r: float = 0.0,
    k0: float = 1.0,
    k1: float = 1.0,
    weight_lambda: float = 1.0,
    optimizer: str = "sgld",
) -> Config:
    """The `linear_ring` preset of the plan (Section sec:presets) with the fields that exist today."""
    return Config(
        data=DataConfig(task="linear_teacher", P=P, D_in=m, n_latents=K),
        net=NetConfig(N=N, L=1, phi="id"),
        train=TrainConfig(
            optimizer=optimizer,
            beta=beta,
            rate_lambda=rate_lambda_for(r, beta if optimizer == "sgld" else 1.0),
            weight_lambda=weight_lambda,
        ),
        spatial=SpatialConfig(
            d=1,
            box=float(N),  # length scales in sites
            input=PopulationSpec(kernel=KernelSpec(length_scale=0.0, amplitude=k0)),
            hidden=PopulationSpec(kernel=KernelSpec(length_scale=ell, amplitude=1.0)),
            output=PopulationSpec(kernel=KernelSpec(length_scale=0.0, amplitude=k1)),
        ),
        device="cpu",
        dtype="float64",
    )


def orthonormal_columns(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    return np.linalg.qr(rng.normal(size=(rows, cols)))[0]


def latent_dataset(Z: np.ndarray, W_X: np.ndarray, W_Y: np.ndarray) -> Dataset:
    """X = Z W_X^T, Y = Z W_Y^T with the latents as factors z0.., as the linear teacher lays them out."""
    Z_t = torch.as_tensor(Z, dtype=torch.float64)
    names = [f"z{i}" for i in range(Z.shape[1])]
    return Dataset(
        x=Z_t @ torch.as_tensor(W_X, dtype=torch.float64).T,
        y=Z_t @ torch.as_tensor(W_Y, dtype=torch.float64).T,
        Q=None,
        period=None,
        factors={n: Z_t[:, i].contiguous() for i, n in enumerate(names)},
        groups={n: [n] for n in names},
    )


def uniform_latent_dataset(rng: np.random.Generator, P: int, K: int, m: int, a: float) -> Dataset:
    """Z ~ U[0, a]^(P x K), orthonormal W_X, W_Y (the task of the limits note, Section sec:latent)."""
    Z = rng.uniform(0, a, (P, K))
    return latent_dataset(Z, orthonormal_columns(rng, m, K), orthonormal_columns(rng, m, K))


def exact_s2_dataset(rng: np.random.Generator, s2: np.ndarray, P: int, m: int) -> Dataset:
    """Latents whose singular values are exactly sqrt(s2): Z = U diag(sqrt s2), U orthonormal (P x K)."""
    K = s2.size
    Z = orthonormal_columns(rng, P, K) * np.sqrt(s2)
    return latent_dataset(Z, orthonormal_columns(rng, m, K), orthonormal_columns(rng, m, K))


def formula_numbers(setting: dict, sig2: float, r: float, beta: float, k0=1.0, k1=1.0):
    """(tau, summaries, ratio) of modes 1 and 2 from the formula s_k^2, as Appendix app:numbers."""
    cfg = ring_config(setting["N"], setting["ell"], setting["P"], setting["m"], setting["K"], beta)
    lam = lt.ring_spectrum(lt.hidden_kernel(cfg))
    s2 = lt.uniform_latent_s2(setting["P"], setting["K"], A_LATENT)
    delta = setting["P"] * setting["m"] / (beta * k1)
    modes = [
        lt.per_mode_spectrum(lam, k0 * s / setting["m"], s / k1, sig2, r, delta) for s in s2[:2]
    ]
    tau = [
        s / k1 * lt.h_of(lam, k0 * s / setting["m"], sig2, r).max() / c**2
        for s, (c, _) in zip(s2, modes)
    ]
    summaries = [lt.spectrum_summary(d) for _, d in modes]
    # Only the quoted |q| <= Q_MAX: at sigma = 0 the clipped Nyquist eigenvalue gives 0 / 0.
    ratio = modes[0][1][: Q_MAX + 1] / modes[1][1][: Q_MAX + 1]
    return tau, summaries, ratio


# ------------------------------------------------------------------ the port against the script


@pytest.mark.parametrize(
    "sx2, sy2, sig2, r, delta",
    [
        (26.67, 213.3, 0.01, 0.0, 8192.0),  # P1 mode 1
        (26.67, 213.3, 0.01, 0.0, 204.8),  # P3 mode 1, strong target
        (2.667, 21.33, 0.01, 0.1, 8192.0),  # P2 mode 2
        (26.67, 213.3, 0.0, 0.0, 204.8),  # sigma = 0, strong target
        (2.667, 21.33, 0.0, 0.5, 512.0),  # sigma = 0, rate penalty
        (1.0, 0.0, 0.0, 0.3, 0.5),  # no target
    ],
)
def test_per_mode_spectrum_matches_the_reference_script(script, sx2, sy2, sig2, r, delta):
    # The script's FFT carries a -9e-16 round-off eigenvalue that the module clips; both get the
    # clipped spectrum so that the comparison is of the solvers.
    lam = lt.ring_spectrum(script.ring_kernel(64, 4.0))
    c_ref, d_ref = script.spectrum_d(lam, sx2, sy2, sig2, r, delta)
    c, d = lt.per_mode_spectrum(lam, sx2, sy2, sig2, r, delta)
    assert c == pytest.approx(c_ref, rel=1e-12)
    np.testing.assert_allclose(d, d_ref, rtol=1e-10, atol=1e-300)


def test_spectrum_summary_matches_the_reference_script(script):
    lam = script.ring_spectrum(64, 4.0)
    _, d = script.spectrum_d(lam, 26.67, 213.3, 0.01, 0.1, 204.8)
    s = lt.spectrum_summary(d)
    assert s.q_half == script.half_power_q(d)
    assert s.share0 == pytest.approx(d[0] / d.sum(), rel=1e-12)
    assert s.lag1 == pytest.approx(script.autocorr_lag1(d), rel=1e-12)
    np.testing.assert_allclose(lt.autocorrelation(d), script.autocorrelation(d), rtol=1e-12)


def test_uniform_latent_s2_is_the_formula_of_the_limits_note(script):
    np.testing.assert_allclose(
        lt.uniform_latent_s2(256, 3, 1.0), [256 * 10 / 12, 256 / 12, 256 / 12], rtol=1e-14
    )
    sx2, sy2, delta = script.latent_modes(256, 3, 1.0, 8, 1.0, 1.0, 8, 0.25)
    np.testing.assert_allclose(lt.uniform_latent_s2(256, 3, 1.0) / 8, sx2, rtol=1e-14)
    np.testing.assert_allclose(lt.uniform_latent_s2(256, 3, 1.0), sy2, rtol=1e-14)
    assert delta == 8192.0


# ------------------------------------------------------------------ the quoted numbers


def test_p1_numbers_at_sigma2_0_01():
    tau, (m1, m2), ratio = formula_numbers(FULL, SIG2, r=0.0, beta=0.25)
    assert tau[0] == pytest.approx(0.00814, rel=1e-2)
    assert tau[1] < 1e-3
    assert (m1.q_half, m2.q_half) == (2, 2)
    assert (m1.share0, m2.share0) == pytest.approx((0.157, 0.156), abs=5e-4)
    assert (m1.lag1, m2.lag1) == pytest.approx((0.969, 0.966), abs=5e-4)
    quoted = [10.08, 10.07, 10.04, 10.01, 10.00, 9.98, 9.95, 9.86, 9.56, 8.54, 5.91, 2.73, 1.35]
    np.testing.assert_allclose(ratio, quoted, atol=5e-3)


def test_p2_numbers_at_sigma2_0_01():
    _, (m1, m2), ratio = formula_numbers(FULL, SIG2, r=0.1, beta=0.25)
    assert (m1.q_half, m2.q_half) == (6, 4)
    assert (m1.share0, m2.share0) == pytest.approx((0.077, 0.111), abs=5e-4)
    assert (m1.lag1, m2.lag1) == pytest.approx((0.917, 0.944), abs=5e-4)
    np.testing.assert_allclose(
        ratio[:8], [1.32, 1.35, 1.44, 1.63, 2.02, 2.84, 4.35, 6.49], atol=5e-3
    )
    assert ratio.max() == pytest.approx(8.2, abs=0.05) and ratio.argmax() in (8, 9)
    np.testing.assert_allclose(ratio[10:], [5.84, 2.72, 1.35], atol=5e-3)


def test_p3_numbers_at_sigma2_0_01():
    tau, (m1, m2), ratio = formula_numbers(FULL, SIG2, r=0.0, beta=10.0)
    assert tau == pytest.approx((0.834, 0.113), abs=1e-3)
    assert (m1.q_half, m2.q_half) == (1, 2)
    assert (m1.share0, m2.share0) == pytest.approx((0.343, 0.164), abs=5e-4)
    assert (m1.lag1, m2.lag1) == pytest.approx((0.985, 0.967), abs=5e-4)
    assert ratio[0] == pytest.approx(53.5, abs=0.05)


def test_p1_to_p3_without_input_noise():
    """The sigma^2 = 0 numbers of the plan ("Without eta")."""
    _, (m1, m2), ratio = formula_numbers(FULL, 0.0, r=0.0, beta=0.25)
    np.testing.assert_allclose(ratio[:6], [10.08, 10.07, 10.04, 10.02, 10.01, 10.00], atol=5e-3)
    np.testing.assert_allclose(ratio[6:], 10.0, atol=5e-3)  # no floor to reach first
    assert (m1.lag1, m2.lag1) == pytest.approx((0.969, 0.969), abs=5e-4)
    _, (m1, m2), ratio = formula_numbers(FULL, 0.0, r=0.1, beta=0.25)
    assert (m1.share0, m2.share0) == pytest.approx((0.077, 0.112), abs=5e-4)
    assert (m1.lag1, m2.lag1) == pytest.approx((0.922, 0.953), abs=5e-4)
    assert (np.diff(ratio) > 0).all() and ratio[9] > 7  # monotone, no interior maximum
    _, (m1, m2), ratio = formula_numbers(FULL, 0.0, r=0.0, beta=10.0)
    assert (m1.share0, m2.share0) == pytest.approx((0.343, 0.165), abs=5e-4)
    assert ratio[0] == pytest.approx(53.5, abs=0.05)


def test_p3_beta_sweep_without_input_noise():
    share1 = [formula_numbers(FULL, 0.0, 0.0, b)[1][0].share0 for b in (0.25, 1, 2, 5, 10)]
    np.testing.assert_allclose(share1, [0.157, 0.165, 0.188, 0.281, 0.343], atol=5e-4)


def test_smoke_numbers_at_sigma2_0_01():
    tau, (m1, m2), ratio = formula_numbers(SMOKE, SIG2, r=0.0, beta=0.25)  # S1
    assert tau == pytest.approx((0.0078, 1.7e-4), rel=2e-2)
    assert (m1.q_half, m2.q_half) == (1, 1)
    assert (m1.share0, m2.share0) == pytest.approx((0.314, 0.310), abs=5e-4)
    # Mode 2 is 0.86946; the plan quotes 0.870, so allow its rounding.
    assert (m1.lag1, m2.lag1) == pytest.approx((0.881, 0.870), abs=1e-3)
    np.testing.assert_allclose(
        ratio[:9], [7.04, 7.00, 6.94, 6.73, 5.24, 1.78, 1.03, 1.00, 1.00], atol=5e-3
    )
    tau, (m1, m2), ratio = formula_numbers(SMOKE, SIG2, r=0.5, beta=0.25)  # S2
    assert (m1.q_half, m2.q_half) == (2, 2)
    assert (m1.share0, m2.share0) == pytest.approx((0.173, 0.236), abs=5e-4)
    assert (m1.lag1, m2.lag1) == pytest.approx((0.746, 0.814), abs=5e-4)
    np.testing.assert_allclose(
        ratio[:9], [1.47, 1.62, 2.35, 4.30, 4.89, 1.77, 1.03, 1.00, 1.00], atol=5e-3
    )
    tau, (m1, m2), ratio = formula_numbers(SMOKE, SIG2, r=0.0, beta=10.0)  # S3
    assert tau == pytest.approx((0.73, 0.19), abs=5e-3)
    assert (m1.share0, m2.share0) == pytest.approx((0.539, 0.338), abs=5e-4)
    assert (m1.lag1, m2.lag1) == pytest.approx((0.933, 0.880), abs=5e-4)
    assert ratio[0] == pytest.approx(21.3, abs=0.05)


def test_smoke_s1_without_input_noise():
    """S1 at sigma^2 = 0, recomputed with check_linear_soln_limits.py at sig2 = 0."""
    _, (m1, m2), ratio = formula_numbers(SMOKE, 0.0, r=0.0, beta=0.25)
    assert (m1.share0, m2.share0) == pytest.approx((0.314, 0.313), abs=5e-4)
    assert (m1.lag1, m2.lag1) == pytest.approx((0.883, 0.883), abs=5e-4)
    np.testing.assert_allclose(ratio[:4], [7.05, 7.03, 7.00, 7.00], atol=5e-3)
    np.testing.assert_allclose(ratio[4:9], 7.0, atol=1e-3)  # sX2 ratio of the two modes, K = 2


# ------------------------------------------------------------------ sigma^2 = 0 as a first-class case


def test_null_mode_at_sigma_zero_is_exactly_zero():
    lam = lt.ring_spectrum(lt.hidden_kernel(ring_config(**FULL, beta=0.25)))
    c, d = lt.per_mode_spectrum(lam, 0.0, 0.0, 0.0, 0.3, 512.0)
    assert c == pytest.approx(512.0, rel=1e-14)  # c = delta + sigma_0^2 lam_bar with sigma_0 = 0
    assert d.shape == lam.shape and (d == 0.0).all()


def test_null_mode_at_positive_sigma_is_the_floor():
    lam = lt.ring_spectrum(lt.hidden_kernel(ring_config(**FULL, beta=0.25)))
    c, d = lt.per_mode_spectrum(lam, 0.0, 0.0, 0.2, 0.3, 0.5)
    assert c == pytest.approx(0.5 + 0.2 / 1.06, rel=1e-12)  # the limits note's 0.68867925
    np.testing.assert_allclose(d, lt.noise_floor(0.2, 0.3), rtol=1e-12)


def test_modes_with_zero_eigenvalue_get_zero_at_sigma_zero():
    lam = np.array([3.0, 1.0, 0.0, 0.0, 1.0, 0.5])
    c, d = lt.per_mode_spectrum(lam, 2.0, 5.0, 0.0, 0.1, 1.0)
    assert d[2] == 0.0 and d[3] == 0.0
    assert (d[lam > 0] > 0).all()
    assert c > math.sqrt(5.0 * lt.h_of(lam, 2.0, 0.0, 0.1).max())  # on the physical branch


def test_sigma_to_zero_limit_is_regular():
    """The root and spectrum at sigma^2 = 1e-8 are within 1e-7 of their sigma = 0 values (B1)."""
    lam = lt.ring_spectrum(lt.hidden_kernel(ring_config(**FULL, beta=0.25)))
    c_eps, d_eps = lt.per_mode_spectrum(lam, 1.0, 2.0, 1e-8, 0.2, 0.5)
    c0, d0 = lt.per_mode_spectrum(lam, 1.0, 2.0, 0.0, 0.2, 0.5)
    assert abs(c_eps - c0) < 1e-7
    assert np.abs(d_eps - d0).max() < 1e-7


def test_spectrum_summary_rejects_an_all_zero_spectrum():
    with pytest.raises(ValueError, match="zero"):
        lt.spectrum_summary(np.zeros(16))


# ------------------------------------------------------------------ theory_parameters from a Config


def test_theory_parameters_of_the_full_preset():
    cfg = ring_config(**FULL, beta=0.25, r=0.1)
    ds = uniform_latent_dataset(np.random.default_rng(0), FULL["P"], FULL["K"], FULL["m"], A_LATENT)
    tp = lt.theory_parameters(cfg, ds)
    assert tp.lam0 == pytest.approx(10.0265, abs=5e-5)
    assert tp.lam_bar == pytest.approx(1.0, rel=1e-12)
    assert tp.lam.shape == (64,) and tp.lam[1] == tp.lam[-1]  # DFT order, lam_q = lam_{-q}
    assert tp.s2[0] == pytest.approx(213.3, rel=0.1)
    assert tp.s2[1] == pytest.approx(21.33, rel=0.2) and tp.s2[2] == pytest.approx(21.33, rel=0.2)
    assert tp.U_Z.shape == (256, 3) and np.allclose(tp.U_Z.T @ tp.U_Z, np.eye(3), atol=1e-12)
    assert (tp.U_Z.sum(axis=0) >= 0).all()
    assert (tp.m, tp.n, tp.P, tp.N, tp.n_latents) == (8, 8, 256, 64, 3)
    assert (tp.k0, tp.k1, tp.beta) == (1.0, 1.0, 0.25)
    assert tp.delta == pytest.approx(8192.0)
    assert tp.r == pytest.approx(0.1)
    assert tp.sig2 == 0.0
    np.testing.assert_allclose(tp.sx2, tp.k0 * tp.s2 / tp.m)
    np.testing.assert_allclose(tp.sy2, tp.s2 / tp.k1)


def test_ring_spectrum_agrees_with_eigvalsh_and_clips_round_off():
    K = lt.hidden_kernel(ring_config(**FULL, beta=0.25))
    lam = lt.ring_spectrum(K)
    np.testing.assert_allclose(np.sort(lam), np.linalg.eigvalsh(K), atol=1e-13)
    assert lam.min() == 0.0  # the -9e-16 of the FFT is round-off, clipped
    with pytest.raises(ValueError, match="negative"):
        lt.ring_spectrum(np.eye(4) - 0.5 * np.ones((4, 4)))  # not positive semi-definite
    with pytest.raises(ValueError, match="circulant"):
        lt.ring_spectrum(np.diag([1.0, 2.0, 3.0]))


def test_rate_coefficient_and_temperature_follow_the_train_section():
    cfg = ring_config(**SMOKE, beta=0.25, r=0.5)
    assert lt.rate_coefficient(cfg) == pytest.approx(0.5)
    assert lt.rate_coefficient(cfg) == pytest.approx(2 * 0.25 * cfg.train.rate_lambda)
    adam = ring_config(**SMOKE, beta=0.25, r=0.5, optimizer="adam")
    assert lt.inverse_temperature(adam) == 1.0  # Adam has no temperature (prior_coefficient's rule)
    assert lt.rate_coefficient(adam) == pytest.approx(2 * adam.train.rate_lambda)


def test_weight_lambda_rescales_both_prior_scales():
    """Cov(W) = K / weight_lambda: k0 and k1 are the amplitudes over weight_lambda, delta follows."""
    cfg = ring_config(**SMOKE, beta=10.0, weight_lambda=0.1)
    ds = uniform_latent_dataset(np.random.default_rng(1), SMOKE["P"], SMOKE["K"], SMOKE["m"], 1.0)
    tp = lt.theory_parameters(cfg, ds)
    assert (tp.k0, tp.k1) == pytest.approx((10.0, 10.0))
    assert tp.delta == pytest.approx(32 * 4 / (10.0 * 10.0))


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("net.L", 2, "L"),
        ("net.phi", "relu", "phi"),
        ("data.task", "position", "linear_teacher"),
    ],
)
def test_theory_parameters_rejects_configs_outside_the_theory(field, value, message):
    cfg = ring_config(**SMOKE, beta=0.25)
    section, name = field.split(".")
    cfg = cfg.model_copy(update={section: getattr(cfg, section).model_copy(update={name: value})})
    ds = uniform_latent_dataset(np.random.default_rng(0), SMOKE["P"], SMOKE["K"], SMOKE["m"], 1.0)
    with pytest.raises(ValueError, match=message):
        lt.theory_parameters(cfg, ds)


def test_theory_parameters_needs_a_spatial_prior_with_scalar_input_and_output_kernels():
    ds = uniform_latent_dataset(np.random.default_rng(0), SMOKE["P"], SMOKE["K"], SMOKE["m"], 1.0)
    cfg = ring_config(**SMOKE, beta=0.25).model_copy(update={"spatial": None})
    with pytest.raises(ValueError, match="spatial"):
        lt.theory_parameters(cfg, ds)
    cfg = ring_config(**SMOKE, beta=0.25)
    smooth_input = cfg.spatial.model_copy(
        update={"input": PopulationSpec(kernel=KernelSpec(length_scale=1.0, amplitude=1.0))}
    )
    with pytest.raises(ValueError, match="input"):
        lt.theory_parameters(cfg.model_copy(update={"spatial": smooth_input}), ds)


def test_theory_parameters_rejects_a_dataset_without_the_latent_factors():
    cfg = ring_config(**SMOKE, beta=0.25)
    ds = uniform_latent_dataset(np.random.default_rng(0), SMOKE["P"], SMOKE["K"], SMOKE["m"], 1.0)
    with pytest.raises(ValueError, match="z1"):
        lt.theory_parameters(cfg, Dataset(ds.x, ds.y, None, None, {"z0": ds.factors["z0"]}))


# ------------------------------------------------------------------ predicted_spectra from a Config


def test_predicted_spectra_from_a_config_reproduces_the_quoted_shares():
    """Latents with exactly the formula singular values: the whole pipeline gives P1 and P3."""
    rng = np.random.default_rng(2)
    s2 = lt.uniform_latent_s2(FULL["P"], FULL["K"], A_LATENT)
    ds = exact_s2_dataset(rng, s2, FULL["P"], FULL["m"])
    p1 = lt.predicted_spectra(ring_config(**FULL, beta=0.25), ds, sig2=SIG2)
    assert p1.d.shape == (3, 64) and p1.c.shape == (3,) and p1.tau.shape == (3,)
    np.testing.assert_allclose(p1.s2, s2, rtol=1e-12)
    shares = [lt.spectrum_summary(d).share0 for d in p1.d]
    assert shares[:2] == pytest.approx((0.157, 0.156), abs=5e-4)
    assert shares[2] == pytest.approx(shares[1], rel=1e-10)  # the two contrasts are degenerate
    assert p1.null_floor == pytest.approx(SIG2)  # r = 0: sigma_0^2 = sigma^2
    p3 = lt.predicted_spectra(ring_config(**FULL, beta=10.0), ds, sig2=SIG2)
    assert lt.spectrum_summary(p3.d[0]).share0 == pytest.approx(0.343, abs=5e-4)
    assert p3.tau[0] == pytest.approx(0.834, abs=1e-3)
    assert (p3.d[0] / p3.d[1])[0] == pytest.approx(53.5, abs=0.05)
    assert p3.params.delta == pytest.approx(204.8)


def test_predicted_spectra_default_is_sigma_zero_on_the_smoke_config():
    ds = uniform_latent_dataset(np.random.default_rng(3), SMOKE["P"], SMOKE["K"], SMOKE["m"], 1.0)
    pred = lt.predicted_spectra(ring_config(**SMOKE, beta=0.25), ds)
    assert pred.params.sig2 == 0.0 and pred.null_floor == 0.0
    assert pred.d.shape == (2, 16) and np.isfinite(pred.d).all() and (pred.d >= 0).all()
    assert pred.lam.shape == (16,) and pred.U_Z.shape == (32, 2)
    np.testing.assert_allclose(pred.d[:, 1], pred.d[:, -1], rtol=1e-12)  # d_q = d_{-q}
    assert all(0 < t < 1 for t in pred.tau)


# ------------------------------------------------------------------ the ansatz-free kernel


def brute_force_kernel_map(C_tilde, K, A, YY, k1, sig2, r, delta):
    """delta I + (1/N) sum_ij K_ij Cov[h_i, h_j] from the NP x NP precision of eq:precision,
    (A (x) K + sig2 I)^{-1} - (B (x) K - r I), B = C~^{-1} Y Y^T C~^{-1} / k1; index order (mu, i)."""
    P, N = A.shape[0], K.shape[0]
    Ci = np.linalg.inv(C_tilde)
    B = Ci @ YY @ Ci / k1
    I = np.eye(P * N)
    precision = np.linalg.inv(np.kron(A, K) + sig2 * I) - (np.kron(B, K) - r * I)
    Cov = np.linalg.inv(precision).reshape(P, N, P, N)
    return delta * np.eye(P) + np.einsum("ij,uivj->uv", K, Cov) / N


# The limits note's section-A case: ring N = 12, ell = sqrt 2, P = 5, m = n = 6, K = 2, k0 = 0.7,
# k1 = 1.3, sigma^2 = 0.2, r = 0.3, delta = 0.5; the task is drawn exactly as the script draws it.
NOTE_N, NOTE_P, NOTE_M, NOTE_K = 12, 5, 6, 2
NOTE_K0, NOTE_K1, NOTE_SIG2, NOTE_R, NOTE_DELTA = 0.7, 1.3, 0.2, 0.3, 0.5
NOTE_EIGENVALUES = [0.68867925, 0.68867925, 0.68867925, 0.75074939, 3.85726387]


def note_case(wy_skew: np.ndarray | None = None):
    """(cfg, dataset) of the note's case; `wy_skew` (K x K) right-multiplies W_Y, as in the script,
    so the skewed task has the same Z, W_X, W_Y and only breaks the teacher's orthonormality."""
    beta = NOTE_P * NOTE_M / (NOTE_DELTA * NOTE_K1)  # delta = P n / (beta k1)
    cfg = ring_config(
        NOTE_N, math.sqrt(2.0), NOTE_P, NOTE_M, NOTE_K, beta, NOTE_R, k0=NOTE_K0, k1=NOTE_K1
    )
    rng = np.random.default_rng(0)
    Z = rng.uniform(0, 1.0, (NOTE_P, NOTE_K))
    W_X = orthonormal_columns(rng, NOTE_M, NOTE_K)
    W_Y = orthonormal_columns(rng, NOTE_M, NOTE_K)
    if wy_skew is not None:
        W_Y = W_Y @ wy_skew
    return cfg, latent_dataset(Z, W_X, W_Y)


def test_ansatz_free_kernel_reproduces_the_note_and_the_brute_force_master_equation():
    cfg, ds = note_case()
    C = lt.ansatz_free_kernel(cfg, ds, sig2=NOTE_SIG2)
    assert C.shape == (NOTE_P, NOTE_P) and np.allclose(C, C.T, atol=1e-13)
    np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(C)), NOTE_EIGENVALUES, atol=1e-8)
    K = lt.hidden_kernel(cfg)
    X, Y = ds.x.numpy(), ds.y.numpy()
    A = NOTE_K0 / NOTE_M * X @ X.T
    F = brute_force_kernel_map(C, K, A, Y @ Y.T, NOTE_K1, NOTE_SIG2, NOTE_R, NOTE_DELTA)
    assert np.abs(F - C).max() < 1e-10


def test_ansatz_free_kernel_equals_the_per_mode_roots_for_the_orthonormal_teacher():
    cfg, ds = note_case()
    C = lt.ansatz_free_kernel(cfg, ds, sig2=NOTE_SIG2)
    pred = lt.predicted_spectra(cfg, ds, sig2=NOTE_SIG2)
    U = pred.U_Z
    rot = U.T @ C @ U
    assert np.abs(rot - np.diag(np.diag(rot))).max() < 1e-10
    np.testing.assert_allclose(np.sort(np.diag(rot)), np.sort(pred.c), rtol=1e-10)
    null_c = NOTE_DELTA + lt.noise_floor(NOTE_SIG2, NOTE_R) * pred.lam.mean()
    complement = np.eye(NOTE_P) - U @ U.T
    np.testing.assert_allclose(complement @ C @ complement, null_c * complement, atol=1e-10)


@pytest.mark.parametrize("beta, r", [(0.25, 0.0), (0.25, 0.1), (10.0, 0.0)])
def test_ansatz_free_kernel_matches_per_mode_roots_on_the_full_ring(beta, r):
    """P1, P2, P3 at sigma = 0, with a strong target (tau = 0.83) in P3."""
    ds = uniform_latent_dataset(np.random.default_rng(4), FULL["P"], FULL["K"], FULL["m"], 1.0)
    cfg = ring_config(**FULL, beta=beta, r=r)
    C = lt.ansatz_free_kernel(cfg, ds)
    pred = lt.predicted_spectra(cfg, ds)
    rot = pred.U_Z.T @ C @ pred.U_Z
    assert np.abs(rot - np.diag(np.diag(rot))).max() < 1e-10 * np.abs(rot).max()
    np.testing.assert_allclose(np.sort(np.diag(rot)), np.sort(pred.c), rtol=1e-10)
    complement = np.eye(FULL["P"]) - pred.U_Z @ pred.U_Z.T
    np.testing.assert_allclose(
        complement @ C @ complement, pred.params.delta * complement, atol=1e-9
    )


def test_ansatz_free_kernel_sees_the_skew_of_a_non_orthonormal_teacher():
    """W_Y -> W_Y diag(1, 0.7): the Gram matrices no longer commute and the per-mode roots are
    off by up to 0.2% (the limits note's approximate case), which the ansatz-free kernel shows."""
    cfg, skewed = note_case(wy_skew=np.diag([1.0, 0.7]))
    C = lt.ansatz_free_kernel(cfg, skewed, sig2=NOTE_SIG2)
    tp = lt.theory_parameters(cfg, skewed, sig2=NOTE_SIG2)
    X, Y = skewed.x.numpy(), skewed.y.numpy()
    commutator = X @ X.T @ Y @ Y.T - Y @ Y.T @ X @ X.T
    assert np.abs(commutator).max() > 0.1  # the Gram matrices do not commute
    # The note's approximate roots: sY2 from the diagonal of U^T Y Y^T U, U the eigenbasis of X X^T.
    w, U = np.linalg.eigh(X @ X.T)
    sx2 = np.clip(tp.k0 * w / tp.m, 0, None)
    sy2 = np.clip(np.diag(U.T @ Y @ Y.T @ U), 0, None) / tp.k1
    roots = np.sort(
        [lt.physical_root(lt.h_of(tp.lam, a, tp.sig2, tp.r), b, tp.delta) for a, b in zip(sx2, sy2)]
    )
    rot = U.T @ C @ U
    assert np.abs(rot - np.diag(np.diag(rot))).max() > 1e-2  # the ansatz is only approximate
    rel = np.abs(np.sort(np.linalg.eigvalsh(C)) / roots - 1).max()
    assert 1e-4 < rel < 1e-2  # off by up to 0.2% in the note
