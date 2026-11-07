"""Tests for fct.linear_theory: the per-mode linear solution and the ansatz-free kernel from a Config.

Ground truths, all synthetic:
  * the numbers quoted in latex/notes/fct_experiment_plan.tex (Sections sec:P1-P3, sec:smoke) at
    sigma^2 = 0.01, and their sigma^2 = 0 counterparts, computed through a `Config` and a dataset
    whose latents have exactly the formula singular values of the uniform latent task;
  * the reference script latex/notes/scripts/check_linear_soln_limits.py, loaded by path, for a
    direct comparison of the port on several parameter sets including sigma^2 = 0;
  * the limits note's N = 12 check (Section sec:rederive): the ansatz-free fixed point and the
    per-mode prediction against a brute-force evaluation of the NP x NP master equation;
  * the training loss itself, for the rate coefficient r.

The quoted numbers are for the formula singular values of Z ~ U[0, 1] and an orthonormal teacher,
so they use `exact_s2_dataset`; `make_dataset` with `data.teacher = "orthonormal"` on [0, 1] is
checked end to end, and its Gaussian teacher is refused.
"""

import dataclasses
import importlib.util
import inspect
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from fct import linear_theory as lt
from fct import spectrum
from fct.config import Config, DataConfig, NetConfig, TrainConfig
from fct.data import make_dataset, task_dims
from fct.model import make_net
from fct.spatial import EmbeddingSpec, KernelSpec, PopulationSpec, SpatialConfig, build_spaces
from fct.tasks import Dataset
from fct.train import _step, make_weight_penalty

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
    """train.rate_lambda that makes the rate coefficient r under the loss's r = 2 beta rate_lambda."""
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
) -> Config:
    """The `linear_ring` preset of the plan (Section sec:presets): orthonormal teacher, latents on
    [0, 1], SGLD, float64; k0, k1 are the input and output kernel amplitudes."""
    return Config(
        data=DataConfig(
            task="linear_teacher",
            P=P,
            D_in=m,
            n_latents=K,
            latent_min=0.0,
            latent_max=1.0,
            teacher="orthonormal",
        ),
        net=NetConfig(N=N, L=1, phi="id"),
        train=TrainConfig(
            optimizer="sgld",
            beta=beta,
            rate_lambda=rate_lambda_for(r, beta),
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


def with_field(cfg: Config, field: str, value) -> Config:
    section, name = field.split(".")
    return cfg.model_copy(update={section: getattr(cfg, section).model_copy(update={name: value})})


def hidden_population(cfg: Config):
    """The hidden population of `build_spaces` for the configured one-layer network."""
    m, n = task_dims(cfg.data)
    return build_spaces(cfg.spatial, m=m, N=cfg.net.N, L=1, n_out=n)[0].post


def trainer_loss(cfg: Config, ds: Dataset, **train_fields):
    """(parts, net) of one `fct.train._step` at lr = 0 (the loss the trainer builds, before the
    step), for the config's net and the train section with `train_fields` overridden."""
    train = cfg.train.model_copy(update=train_fields)
    m, n = task_dims(cfg.data)
    net = make_net(cfg.net, m, n, torch.device("cpu"), torch.float64)
    optim = torch.optim.SGD(net.parameters(), lr=0.0)
    penalty = make_weight_penalty(cfg.model_copy(update={"train": train}), net)
    return _step(net, optim, ds, train, penalty), net


def circulant_kernel(lam: np.ndarray) -> np.ndarray:
    """The symmetric circulant matrix whose DFT-order spectrum is `lam`: the kernel the theory
    actually uses, so a brute-force check sees the same lam_q (floored or not) as the module."""
    row = np.fft.ifft(lam).real
    N = lam.size
    return row[(np.arange(N)[None, :] - np.arange(N)[:, None]) % N]


def orthonormal_columns(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    return np.linalg.qr(rng.normal(size=(rows, cols)))[0]


def latent_dataset(Z: np.ndarray, W_X: np.ndarray, W_Y: np.ndarray) -> Dataset:
    """X = Z W_X^T, Y = Z W_Y^T with the latents as factors z0.. and the teacher maps in extras,
    as the linear teacher lays them out."""
    Z_t = torch.as_tensor(Z, dtype=torch.float64)
    W_X_t = torch.as_tensor(W_X, dtype=torch.float64)
    W_Y_t = torch.as_tensor(W_Y, dtype=torch.float64)
    names = [f"z{i}" for i in range(Z.shape[1])]
    return Dataset(
        x=Z_t @ W_X_t.T,
        y=Z_t @ W_Y_t.T,
        Q=None,
        period=None,
        factors={n: Z_t[:, i].contiguous() for i, n in enumerate(names)},
        groups={n: [n] for n in names},
        extras={"W_X": W_X_t, "W_Y": W_Y_t},
    )


def uniform_latent_dataset(rng: np.random.Generator, P: int, K: int, m: int, a: float) -> Dataset:
    """Z ~ U[0, a]^(P x K), orthonormal W_X, W_Y (the task of the limits note, Section sec:latent)."""
    Z = rng.uniform(0, a, (P, K))
    return latent_dataset(Z, orthonormal_columns(rng, m, K), orthonormal_columns(rng, m, K))


def exact_s2_dataset(rng: np.random.Generator, s2: np.ndarray, P: int, m: int) -> Dataset:
    """Latents whose singular values are exactly sqrt(s2): Z = U diag(sqrt s2) with U orthonormal
    (P x K) and u_1 = 1 / sqrt(P), the mean direction of latents on [0, a]."""
    K = s2.size
    U = np.linalg.qr(np.column_stack([np.ones(P), rng.normal(size=(P, K - 1))]))[0]
    Z = U * np.where(U.sum(axis=0) < 0, -1.0, 1.0) * np.sqrt(s2)
    return latent_dataset(Z, orthonormal_columns(rng, m, K), orthonormal_columns(rng, m, K))


def config_numbers(setting: dict, sig2: float, r: float, beta: float, exact_kernel=False):
    """(tau, summaries, ratio) of modes 1 and 2, as Appendix app:numbers, but computed through a
    `Config` and `predicted_spectra` on latents with the formula singular values."""
    s2 = lt.uniform_latent_s2(setting["P"], setting["K"], A_LATENT)
    ds = exact_s2_dataset(np.random.default_rng(0), s2, setting["P"], setting["m"])
    cfg = ring_config(**setting, beta=beta, r=r)
    pred = lt.predicted_spectra(cfg, ds, sig2=sig2, exact_kernel=exact_kernel)
    np.testing.assert_allclose(pred.s2, s2, rtol=1e-12)
    summaries = [lt.predicted_summary(d) for d in pred.d[:2]]
    ratio = pred.d[0, : Q_MAX + 1] / pred.d[1, : Q_MAX + 1]
    return pred.tau, summaries, ratio


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


def test_predicted_summary_matches_the_reference_script(script):
    lam = script.ring_spectrum(64, 4.0)
    _, d = script.spectrum_d(lam, 26.67, 213.3, 0.01, 0.1, 204.8)
    s = lt.predicted_summary(d)
    assert s.q_half == script.half_power_q(d)
    assert s.share0 == pytest.approx(d[0] / d.sum(), rel=1e-12)
    assert s.lag1 == pytest.approx(script.autocorr_lag1(d), rel=1e-12)
    R = spectrum.autocorrelation(torch.as_tensor(d)).numpy()  # the module uses fct.spectrum's
    np.testing.assert_allclose(R, script.autocorrelation(d), rtol=1e-12, atol=1e-15)


def test_uniform_latent_s2_is_the_formula_of_the_limits_note(script):
    np.testing.assert_allclose(
        lt.uniform_latent_s2(256, 3, 1.0), [256 * 10 / 12, 256 / 12, 256 / 12], rtol=1e-14
    )
    sx2, sy2, delta = script.latent_modes(256, 3, 1.0, 8, 1.0, 1.0, 8, 0.25)
    np.testing.assert_allclose(lt.uniform_latent_s2(256, 3, 1.0) / 8, sx2, rtol=1e-14)
    np.testing.assert_allclose(lt.uniform_latent_s2(256, 3, 1.0), sy2, rtol=1e-14)
    assert delta == 8192.0


def test_physical_root_resolves_a_root_next_to_the_pole():
    """A root within 1e-13 (relative) of the pole: the bracket must start at the next float
    above the pole, not at a fixed relative margin that would already lie past the root."""
    # One frequency, sY2 = 1, h = p^2 with p = 2e-14: the pole is p and the residual
    # 1 - delta/c - c h / (c^2 - h) vanishes at c = p (1 + eps) with eps (2 + eps) = p (1 + eps),
    # eps = 1e-14 to first order: below where a bracket a relative 1e-13 above the pole would begin.
    p = 2e-14
    c = lt.physical_root(np.array([p**2]), 1.0, 1e-60)
    assert c / p - 1 == pytest.approx(1e-14, rel=5e-2)


def test_physical_root_names_the_bracket_when_it_fails():
    with pytest.raises(ValueError, match="delta"):
        lt.physical_root(np.ones(4), 1.0, 0.0)


def test_physical_root_refuses_a_root_within_one_ulp_of_the_pole():
    """p = 1e-17: the root is p (1 + 5e-18), closer to the pole than one ulp (1.1e-16)."""
    p = 1e-17
    with pytest.raises(ValueError, match="one ulp"):
        lt.physical_root(np.array([p**2]), 1.0, 1e-60)


def test_physical_root_refuses_an_overflowing_bracket():
    with pytest.raises(ValueError, match="overflow"):
        lt.physical_root(np.ones(4), 1.0, 1e300)


# ------------------------------------------------------------------ the quoted numbers, via a Config


def test_p1_numbers_at_sigma2_0_01():
    tau, (m1, m2), ratio = config_numbers(FULL, SIG2, r=0.0, beta=0.25)
    assert tau[0] == pytest.approx(0.00814, rel=1e-2)
    assert tau[1] < 1e-3
    assert (m1.q_half, m2.q_half) == (2, 2)
    assert (m1.share0, m2.share0) == pytest.approx((0.157, 0.156), abs=5e-4)
    assert (m1.lag1, m2.lag1) == pytest.approx((0.969, 0.966), abs=5e-4)
    quoted = [10.08, 10.07, 10.04, 10.01, 10.00, 9.98, 9.95, 9.86, 9.56, 8.54, 5.91, 2.73, 1.35]
    np.testing.assert_allclose(ratio, quoted, atol=5e-3)


def test_p2_numbers_at_sigma2_0_01():
    _, (m1, m2), ratio = config_numbers(FULL, SIG2, r=0.1, beta=0.25)
    assert (m1.q_half, m2.q_half) == (6, 4)
    assert (m1.share0, m2.share0) == pytest.approx((0.077, 0.111), abs=5e-4)
    assert (m1.lag1, m2.lag1) == pytest.approx((0.917, 0.944), abs=5e-4)
    np.testing.assert_allclose(
        ratio[:8], [1.32, 1.35, 1.44, 1.63, 2.02, 2.84, 4.35, 6.49], atol=5e-3
    )
    assert ratio.max() == pytest.approx(8.2, abs=0.05) and ratio.argmax() in (8, 9)
    np.testing.assert_allclose(ratio[10:], [5.84, 2.72, 1.35], atol=5e-3)


def test_p3_numbers_at_sigma2_0_01():
    tau, (m1, m2), ratio = config_numbers(FULL, SIG2, r=0.0, beta=10.0)
    assert tau[:2] == pytest.approx((0.834, 0.113), abs=1e-3)
    assert (m1.q_half, m2.q_half) == (1, 2)
    assert (m1.share0, m2.share0) == pytest.approx((0.343, 0.164), abs=5e-4)
    assert (m1.lag1, m2.lag1) == pytest.approx((0.985, 0.967), abs=5e-4)
    assert ratio[0] == pytest.approx(53.5, abs=0.05)


def test_p1_to_p3_without_input_noise():
    """The sigma^2 = 0 numbers of the plan ("Without eta")."""
    _, (m1, m2), ratio = config_numbers(FULL, 0.0, r=0.0, beta=0.25)
    np.testing.assert_allclose(ratio[:6], [10.08, 10.07, 10.04, 10.02, 10.01, 10.00], atol=5e-3)
    np.testing.assert_allclose(ratio[6:], 10.0, atol=5e-3)  # no floor to reach first
    assert (m1.lag1, m2.lag1) == pytest.approx((0.969, 0.969), abs=5e-4)
    _, (m1, m2), ratio = config_numbers(FULL, 0.0, r=0.1, beta=0.25)
    assert (m1.share0, m2.share0) == pytest.approx((0.077, 0.112), abs=5e-4)
    assert (m1.lag1, m2.lag1) == pytest.approx((0.922, 0.953), abs=5e-4)
    assert (np.diff(ratio) > 0).all() and ratio[9] > 7  # monotone, no interior maximum
    _, (m1, m2), ratio = config_numbers(FULL, 0.0, r=0.0, beta=10.0)
    assert (m1.share0, m2.share0) == pytest.approx((0.343, 0.165), abs=5e-4)
    assert ratio[0] == pytest.approx(53.5, abs=0.05)


def test_p3_beta_sweep_without_input_noise():
    share1 = [config_numbers(FULL, 0.0, 0.0, b)[1][0].share0 for b in (0.25, 1, 2, 5, 10)]
    np.testing.assert_allclose(share1, [0.157, 0.165, 0.188, 0.281, 0.343], atol=5e-4)


def test_smoke_numbers_at_sigma2_0_01():
    tau, (m1, m2), ratio = config_numbers(SMOKE, SIG2, r=0.0, beta=0.25)  # S1
    assert tau == pytest.approx((0.0078, 1.7e-4), rel=2e-2)
    assert (m1.q_half, m2.q_half) == (1, 1)
    assert (m1.share0, m2.share0) == pytest.approx((0.314, 0.310), abs=5e-4)
    # Mode 2 is 0.86946; the plan quotes 0.870, so allow its rounding.
    assert (m1.lag1, m2.lag1) == pytest.approx((0.881, 0.870), abs=1e-3)
    np.testing.assert_allclose(
        ratio[:9], [7.04, 7.00, 6.94, 6.73, 5.24, 1.78, 1.03, 1.00, 1.00], atol=5e-3
    )
    tau, (m1, m2), ratio = config_numbers(SMOKE, SIG2, r=0.5, beta=0.25)  # S2
    assert (m1.q_half, m2.q_half) == (2, 2)
    assert (m1.share0, m2.share0) == pytest.approx((0.173, 0.236), abs=5e-4)
    assert (m1.lag1, m2.lag1) == pytest.approx((0.746, 0.814), abs=5e-4)
    np.testing.assert_allclose(
        ratio[:9], [1.47, 1.62, 2.35, 4.30, 4.89, 1.77, 1.03, 1.00, 1.00], atol=5e-3
    )
    tau, (m1, m2), ratio = config_numbers(SMOKE, SIG2, r=0.0, beta=10.0)  # S3
    assert tau == pytest.approx((0.73, 0.19), abs=5e-3)
    assert (m1.share0, m2.share0) == pytest.approx((0.539, 0.338), abs=5e-4)
    assert (m1.lag1, m2.lag1) == pytest.approx((0.933, 0.880), abs=5e-4)
    assert ratio[0] == pytest.approx(21.3, abs=0.05)


def test_smoke_s1_without_input_noise():
    """S1 at sigma^2 = 0, recomputed with check_linear_soln_limits.py at sig2 = 0."""
    _, (m1, m2), ratio = config_numbers(SMOKE, 0.0, r=0.0, beta=0.25)
    assert (m1.share0, m2.share0) == pytest.approx((0.314, 0.313), abs=5e-4)
    assert (m1.lag1, m2.lag1) == pytest.approx((0.883, 0.883), abs=5e-4)
    np.testing.assert_allclose(ratio[:4], [7.05, 7.03, 7.00, 7.00], atol=5e-3)
    np.testing.assert_allclose(ratio[4:9], 7.0, atol=1e-3)  # sX2 ratio of the two modes, K = 2


def test_smoke_s1_without_input_noise_equals_the_reference_script(script):
    """The S1 recipe of Appendix app:numbers run through the script at sig2 = 0 and through a
    `Config` with the unfloored kernel (the script's): the same c, tau, spectra and summaries."""
    N, ell, P, m, K = SMOKE["N"], SMOKE["ell"], SMOKE["P"], SMOKE["m"], SMOKE["K"]
    beta = 0.25
    sx2, sy2, delta = script.latent_modes(P, K, A_LATENT, m, 1.0, 1.0, m, beta)
    lam = lt.ring_spectrum(script.ring_kernel(N, ell))  # clipped as the module clips
    reference = [script.spectrum_d(lam, sx2[k], sy2[k], 0.0, 0.0, delta) for k in range(K)]
    ds = exact_s2_dataset(np.random.default_rng(5), lt.uniform_latent_s2(P, K, A_LATENT), P, m)
    pred = lt.predicted_spectra(ring_config(**SMOKE, beta=beta), ds, exact_kernel=True)
    for k, (c_ref, d_ref) in enumerate(reference):
        assert pred.c[k] == pytest.approx(c_ref, rel=1e-12)
        np.testing.assert_allclose(pred.d[k], d_ref, rtol=1e-10, atol=1e-300)
        tau_ref = sy2[k] * script.h_of(lam, sx2[k], 0.0, 0.0).max() / c_ref**2
        assert pred.tau[k] == pytest.approx(tau_ref, rel=1e-10)
        s = lt.predicted_summary(pred.d[k])
        assert s.q_half == script.half_power_q(d_ref)
        assert s.share0 == pytest.approx(d_ref[0] / d_ref.sum(), rel=1e-10)
        assert s.lag1 == pytest.approx(script.autocorr_lag1(d_ref), rel=1e-10)
    assert pred.null_floor == 0.0


# ------------------------------------------------------------------ the floored prior


def test_default_spectrum_is_the_floored_prior_the_run_trains_with():
    """The run's prior is N(0, M^{-1}) with M's eigenvalues 1 / max(lam_q, min_rel_eig lam_0):
    by default the theory uses exactly that, and `exact_kernel=True` the unfloored K."""
    cfg = ring_config(**FULL, beta=0.25)
    ds = exact_s2_dataset(np.random.default_rng(0), lt.uniform_latent_s2(256, 3, 1.0), 256, 8)
    exact = lt.ring_spectrum(hidden_population(cfg).K)
    floor = cfg.spatial.min_rel_eig * exact.max()
    tail = exact < floor
    assert tail.sum() == 37  # of 64 frequencies
    trained = lt.theory_parameters(cfg, ds).lam
    np.testing.assert_allclose(trained, np.maximum(exact, floor), rtol=1e-8)
    np.testing.assert_allclose(
        np.sort(1 / trained), np.linalg.eigvalsh(hidden_population(cfg).M), rtol=1e-8
    )
    np.testing.assert_array_equal(lt.theory_parameters(cfg, ds, exact_kernel=True).lam, exact)
    pred = lt.predicted_spectra(cfg, ds)
    p = pred.params
    _, d_floored = lt.per_mode_spectrum(np.maximum(exact, floor), p.sx2[0], p.sy2[0], 0, 0, p.delta)
    np.testing.assert_allclose(pred.d[0][tail], d_floored[tail], rtol=1e-7)
    unfloored = lt.predicted_spectra(cfg, ds, exact_kernel=True)
    assert (pred.d[0][tail] > unfloored.d[0][tail]).all()  # the floor raises the whole tail
    assert pred.d[0][32] > 1e3 * unfloored.d[0][32]  # and dominates it at the Nyquist frequency
    np.testing.assert_allclose(pred.d[0][~tail], unfloored.d[0][~tail], rtol=1e-6)


# ------------------------------------------------------------------ sigma^2 = 0 as a first-class case


def test_null_mode_at_sigma_zero_is_exactly_zero():
    lam = lt.ring_spectrum(hidden_population(ring_config(**FULL, beta=0.25)).K)
    c, d = lt.per_mode_spectrum(lam, 0.0, 0.0, 0.0, 0.3, 512.0)
    assert c == pytest.approx(512.0, rel=1e-14)  # c = delta + sigma_0^2 lam_bar with sigma_0 = 0
    assert d.shape == lam.shape and (d == 0.0).all()


def test_null_mode_at_positive_sigma_is_the_floor():
    lam = lt.ring_spectrum(hidden_population(ring_config(**FULL, beta=0.25)).K)
    c, d = lt.per_mode_spectrum(lam, 0.0, 0.0, 0.2, 0.3, 0.5)
    assert c == pytest.approx(0.5 + 0.2 / 1.06, rel=1e-12)  # the limits note's 0.68867925
    np.testing.assert_allclose(d, lt.noise_floor(0.2, 0.3), rtol=1e-12)


def test_modes_with_zero_eigenvalue_get_zero_at_sigma_zero():
    lam = np.array([3.0, 1.0, 0.0, 0.0, 1.0, 0.5])
    c, d = lt.per_mode_spectrum(lam, 2.0, 5.0, 0.0, 0.1, 1.0)
    assert d[2] == 0.0 and d[3] == 0.0
    assert (d[lam > 0] > 0).all()
    assert c > math.sqrt(5.0 * lt.pi_lam(lam, 2.0, 0.0, 0.1).max())  # on the physical branch


def test_sigma_to_zero_limit_is_regular():
    """The root and spectrum at sigma^2 = 1e-8 are within 1e-7 of their sigma = 0 values (B1)."""
    lam = lt.ring_spectrum(hidden_population(ring_config(**FULL, beta=0.25)).K)
    c_eps, d_eps = lt.per_mode_spectrum(lam, 1.0, 2.0, 1e-8, 0.2, 0.5)
    c0, d0 = lt.per_mode_spectrum(lam, 1.0, 2.0, 0.0, 0.2, 0.5)
    assert abs(c_eps - c0) < 1e-7
    assert np.abs(d_eps - d0).max() < 1e-7


def test_a_latent_the_data_does_not_span():
    """A zero latent column has no frame (`fct.spectrum.latent_basis` refuses a rank-deficient
    Z), so the per-mode theory refuses it; at sigma = 0 `ansatz_free_kernel` puts c_0 = delta on
    every direction the data does not span, and the per-mode root on the one it does."""
    rng = np.random.default_rng(6)
    P, m = SMOKE["P"], SMOKE["m"]
    z = rng.uniform(0, 1, P)
    Z = np.stack([z, np.zeros(P)], axis=1)
    ds = latent_dataset(Z, orthonormal_columns(rng, m, 2), orthonormal_columns(rng, m, 2))
    cfg = ring_config(**SMOKE, beta=0.25, r=0.5)
    with pytest.raises(ValueError, match="rank-deficient"):
        lt.predicted_spectra(cfg, ds)
    net_params = lt.network_parameters(cfg, ds)
    C = lt.ansatz_free_kernel(cfg, ds)
    u = z / np.linalg.norm(z)
    s2 = z @ z
    c, _ = lt.per_mode_spectrum(
        net_params.lam, net_params.k0 * s2 / m, s2 / net_params.k1, 0.0, 0.5, net_params.delta
    )
    assert u @ C @ u == pytest.approx(c, rel=1e-10)
    complement = np.eye(P) - np.outer(u, u)
    np.testing.assert_allclose(complement @ C @ complement, net_params.delta * complement)


def test_floored_spectrum_survives_a_tiny_min_rel_eig():
    """At min_rel_eig = 1e-12 inverting M would lose 12 digits on the coarse modes; lam is the
    floor rule of `penalty_matrix` applied to K's spectrum, so it is exact and runs."""
    cfg = ring_config(**FULL, beta=0.25)
    cfg = cfg.model_copy(update={"spatial": cfg.spatial.model_copy(update={"min_rel_eig": 1e-12})})
    ds = exact_s2_dataset(np.random.default_rng(0), lt.uniform_latent_s2(256, 3, 1.0), 256, 8)
    hidden = hidden_population(cfg)
    exact = lt.ring_spectrum(hidden.K)
    lam = lt.theory_parameters(cfg, ds).lam
    np.testing.assert_array_equal(lam, np.maximum(exact, hidden.report.floor))
    assert hidden.report.floor == pytest.approx(1e-12 * exact.max(), rel=1e-12)
    assert np.isfinite(lt.predicted_spectra(cfg, ds).d).all()


def test_a_float32_run_is_refused():
    """A float32 run casts M to float32 and trains with a prior whose coarse lam_q differ by
    1-2%, so the theory refuses it, naming the field."""
    cfg = ring_config(**SMOKE, beta=0.25).model_copy(update={"dtype": "float32"})
    ds = uniform_latent_dataset(np.random.default_rng(0), SMOKE["P"], SMOKE["K"], SMOKE["m"], 1.0)
    for f in (lt.theory_parameters, lt.predicted_spectra, lt.ansatz_free_kernel):
        with pytest.raises(ValueError, match="dtype"):
            f(cfg, ds)


def test_orthonormal_teacher_from_make_dataset_end_to_end():
    """`make_dataset` of the preset's data section (orthonormal teacher, latents on [0, 1]):
    x = Z W_X^T, y = Z W_Y^T, and theory_parameters' s2 is the SVD of Z."""
    cfg = ring_config(**SMOKE, beta=0.25, r=0.5)
    ds = make_dataset(cfg.data, torch.device("cpu"), torch.float64)
    Z = np.stack([ds.factors[f"z{i}"].numpy() for i in range(SMOKE["K"])], axis=1)
    assert Z.min() >= 0 and Z.max() <= 1
    W_X, W_Y = ds.extras["W_X"].numpy(), ds.extras["W_Y"].numpy()
    np.testing.assert_allclose(ds.x.numpy(), Z @ W_X.T, atol=1e-14)
    np.testing.assert_allclose(ds.y.numpy(), Z @ W_Y.T, atol=1e-14)
    tp = lt.theory_parameters(cfg, ds)
    np.testing.assert_allclose(tp.s2, np.linalg.svd(Z, compute_uv=False) ** 2, rtol=1e-12)
    pred = lt.predicted_spectra(cfg, ds)
    assert pred.d.shape == (SMOKE["K"], SMOKE["N"]) and (pred.d > 0).all()


def test_data_not_generated_by_the_teacher_is_refused():
    """x and y must be Z W_X^T and Z W_Y^T for the W in extras: a rescaled x is refused."""
    cfg = ring_config(**SMOKE, beta=0.25)
    ds = uniform_latent_dataset(np.random.default_rng(0), SMOKE["P"], SMOKE["K"], SMOKE["m"], 1.0)
    with pytest.raises(ValueError, match=r"dataset.x is not Z W_X\^T.*\d"):
        lt.theory_parameters(cfg, dataclasses.replace(ds, x=1.5 * ds.x))
    with pytest.raises(ValueError, match=r"dataset.y is not Z W_Y\^T"):
        lt.predicted_spectra(cfg, dataclasses.replace(ds, y=ds.y + 1e-3))


def test_symmetric_latents_are_refused():
    """Latents on [-1, 1] have no mean direction (`fct.spectrum.latent_basis`)."""
    cfg = ring_config(**SMOKE, beta=0.25)
    rng = np.random.default_rng(0)
    Z = rng.uniform(-1, 1, (SMOKE["P"], SMOKE["K"]))
    ds = latent_dataset(Z, orthonormal_columns(rng, 4, 2), orthonormal_columns(rng, 4, 2))
    with pytest.raises(ValueError, match="symmetric"):
        lt.theory_parameters(cfg, ds)


def test_parameters_compare_by_identity_and_hash():
    ds = uniform_latent_dataset(np.random.default_rng(0), SMOKE["P"], SMOKE["K"], SMOKE["m"], 1.0)
    tp = lt.theory_parameters(ring_config(**SMOKE, beta=0.25), ds)
    same = tp
    assert tp == same and tp != dataclasses.replace(tp)  # identity, not value
    assert {tp: 1}[tp] == 1
    net_params = lt.network_parameters(ring_config(**SMOKE, beta=0.25), ds)
    alias = net_params
    assert net_params == alias and hash(net_params) == hash(alias)


def test_predicted_summary_rejects_an_all_zero_spectrum():
    with pytest.raises(ValueError, match="zero"):
        lt.predicted_summary(np.zeros(16))


def test_predicted_summary_names_the_negative_value():
    with pytest.raises(ValueError, match="-0.5"):
        lt.predicted_summary(np.array([1.0, -0.5, 0.2, -0.5]))


# ------------------------------------------------------------------ theory_parameters from a Config


def test_theory_parameters_of_the_full_preset():
    cfg = ring_config(**FULL, beta=0.25, r=0.1)
    ds = uniform_latent_dataset(np.random.default_rng(0), FULL["P"], FULL["K"], FULL["m"], A_LATENT)
    tp = lt.theory_parameters(cfg, ds)
    assert tp.lam0 == pytest.approx(10.0265, abs=5e-5)
    assert tp.lam_bar == pytest.approx(1.0, rel=1e-5)  # 1 up to the floor, 37 x 1e-5
    assert lt.theory_parameters(cfg, ds, exact_kernel=True).lam_bar == pytest.approx(1, rel=1e-12)
    assert tp.lam.shape == (64,) and tp.lam[1] == tp.lam[-1]  # DFT order, lam_q = lam_{-q}
    assert tp.s2[0] == pytest.approx(213.3, rel=0.1)
    assert tp.s2[1] == pytest.approx(21.33, rel=0.2) and tp.s2[2] == pytest.approx(21.33, rel=0.2)
    assert tp.U.shape == (256, 3) and np.allclose(tp.U.T @ tp.U, np.eye(3), atol=1e-12)
    assert (tp.U.sum(axis=0) >= 0).all()
    assert (tp.m, tp.n, tp.P, tp.N, tp.n_latents) == (8, 8, 256, 64, 3)
    assert (tp.k0, tp.k1, tp.beta) == (1.0, 1.0, 0.25)
    assert tp.delta == pytest.approx(8192.0)
    assert tp.r == pytest.approx(0.1)
    assert tp.sig2 == 0.0
    np.testing.assert_allclose(tp.sx2, tp.k0 * tp.s2 / tp.m)
    np.testing.assert_allclose(tp.sy2, tp.s2 / tp.k1)


def test_ring_spectrum_agrees_with_eigvalsh_and_clips_round_off():
    K = hidden_population(ring_config(**FULL, beta=0.25)).K
    lam = lt.ring_spectrum(K)
    np.testing.assert_allclose(np.sort(lam), np.linalg.eigvalsh(K), atol=1e-13)
    assert lam.min() == 0.0  # the -9e-16 of the FFT is round-off, clipped
    with pytest.raises(ValueError, match="negative eigenvalue -1"):
        lt.ring_spectrum(np.eye(4) - 0.5 * np.ones((4, 4)))  # not positive semi-definite
    with pytest.raises(ValueError, match=r"circulant.*\d"):
        lt.ring_spectrum(np.diag([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="symmetric"):
        lt.ring_spectrum(np.array([[1.0, 0.2], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="finite"):
        lt.ring_spectrum(np.array([[1.0, np.nan], [np.nan, 1.0]]))


def test_theory_rate_is_the_rate_part_of_the_training_loss():
    """beta times the rate part of the loss the run minimises equals (r/2) sum phi^2 with
    r = theory_rate(cfg): taken from `fct.train._step`, the call the trainer makes, so a change
    to how the loss scales the rate penalty fails here."""
    cfg = with_field(ring_config(**SMOKE, beta=0.25, r=0.5), "train.lr", 1e-6)  # step check
    ds = make_dataset(cfg.data, torch.device("cpu"), torch.float64)
    parts, net = trainer_loss(cfg, ds)
    without, _ = trainer_loss(cfg, ds, rate_lambda=0.0)
    rate_part = (parts["loss"] - without["loss"]).item()
    h, _ = net(ds.x)
    sum_phi2 = net.phi(h).square().sum().item()
    assert parts["rate_l2"].item() == pytest.approx(sum_phi2, rel=1e-12)
    r = lt.theory_rate(cfg)
    assert r == pytest.approx(0.5)
    assert cfg.train.beta * rate_part == pytest.approx(r / 2 * sum_phi2, rel=1e-10)


@pytest.mark.parametrize("beta", [0.25, 4.0])
def test_prior_and_data_parts_of_the_training_loss_match_the_theory(beta):
    """With k0, k1 != 1 and weight_lambda != 1, beta times the trainer's prior part is
    (1/2) tr(M_h U U^T) / k0 + (1/2) tr(V^T M_h V) / k1, with M_h the inverse of the circulant of
    the module's lam (so the floor, k0 and k1 are the trainer's), and beta times its data part is
    N ||y - f||^2 / (2 delta k1), which is what fixes delta = P n / (beta k1)."""
    cfg = ring_config(**SMOKE, beta=beta, k0=2.0, k1=0.5, weight_lambda=4.0)
    cfg = with_field(cfg, "train.lr", 1e-10)  # passes the SGLD step check
    ds = make_dataset(cfg.data, torch.device("cpu"), torch.float64)
    net_params = lt.network_parameters(cfg, ds)
    assert (net_params.k0, net_params.k1) == pytest.approx((0.5, 0.125))
    parts, net = trainer_loss(cfg, ds)
    doubled, _ = trainer_loss(cfg, ds, weight_lambda=2 * cfg.train.weight_lambda)
    prior_part = (doubled["loss"] - parts["loss"]).item()  # the prior term is linear in it
    U, V = net.U.detach().numpy(), net.V.detach().numpy()
    M_h = circulant_kernel(1.0 / net_params.lam)
    expected_prior = 0.5 * (
        np.trace(M_h @ U @ U.T) / net_params.k0 + np.trace(V.T @ M_h @ V) / net_params.k1
    )
    assert beta * prior_part == pytest.approx(expected_prior, rel=1e-8)
    data_part = parts["loss"].item() - prior_part  # rate_lambda = 0
    _, f = net(ds.x)
    sq_error = (ds.y - f).square().sum().item()
    expected_data = net_params.N * sq_error / (2 * net_params.delta * net_params.k1)
    assert beta * data_part == pytest.approx(expected_data, rel=1e-8)


def test_config_path_carries_k0_k1_and_weight_lambda(script):
    """Amplitudes 2 and 0.5 over weight_lambda = 4 are k0 = 0.5, k1 = 0.125: the Config path
    equals the reference script at those values (exact kernel, as the script)."""
    P, K, m = SMOKE["P"], SMOKE["K"], SMOKE["m"]
    beta, r = 0.25, 0.1
    cfg = ring_config(**SMOKE, beta=beta, r=r, k0=2.0, k1=0.5, weight_lambda=4.0)
    ds = exact_s2_dataset(np.random.default_rng(0), lt.uniform_latent_s2(P, K, 1.0), P, m)
    pred = lt.predicted_spectra(cfg, ds, sig2=SIG2, exact_kernel=True)
    sx2, sy2, delta = script.latent_modes(P, K, 1.0, m, 0.5, 0.125, m, beta)
    assert pred.params.delta == pytest.approx(delta, rel=1e-12)
    lam = lt.ring_spectrum(script.ring_kernel(SMOKE["N"], SMOKE["ell"]))
    for k in range(K):
        c_ref, d_ref = script.spectrum_d(lam, sx2[k], sy2[k], SIG2, r, delta)
        assert pred.c[k] == pytest.approx(c_ref, rel=1e-10)
        np.testing.assert_allclose(pred.d[k], d_ref, rtol=1e-8)


def test_weight_lambda_rescales_both_prior_scales():
    """Cov(W) = K / weight_lambda: k0 and k1 are the amplitudes over weight_lambda, delta follows."""
    cfg = ring_config(**SMOKE, beta=10.0, weight_lambda=0.1)
    ds = uniform_latent_dataset(np.random.default_rng(1), SMOKE["P"], SMOKE["K"], SMOKE["m"], 1.0)
    tp = lt.theory_parameters(cfg, ds)
    assert (tp.k0, tp.k1) == pytest.approx((10.0, 10.0))
    assert tp.delta == pytest.approx(32 * 4 / (10.0 * 10.0))


def test_every_signature_names_the_rate_coefficient_r():
    """The rate coefficient is `r` everywhere; the notes' g^2 never appears as a parameter."""
    functions = [
        f for f in vars(lt).values() if inspect.isfunction(f) and f.__module__ == lt.__name__
    ]
    params = {p for f in functions for p in inspect.signature(f).parameters}
    assert "r" in params
    assert not params & {"g", "g2", "rate", "rate_lambda"}
    for f in (lt.noise_floor, lt.prior_shrunk_variance, lt.pi_lam, lt.per_mode_spectrum):
        assert "r" in inspect.signature(f).parameters, f.__name__
    assert "r" in {f.name for f in dataclasses.fields(lt.TheoryParameters)}


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("net.L", 2, "L"),
        ("net.phi", "relu", "phi"),
        ("data.task", "position", "linear_teacher"),
        ("train.optimizer", "adam", "train.optimizer"),
    ],
)
def test_theory_parameters_rejects_configs_outside_the_theory(field, value, message):
    cfg = with_field(ring_config(**SMOKE, beta=0.25), field, value)
    ds = uniform_latent_dataset(np.random.default_rng(0), SMOKE["P"], SMOKE["K"], SMOKE["m"], 1.0)
    with pytest.raises(ValueError, match=message):
        lt.theory_parameters(cfg, ds)
    with pytest.raises(ValueError, match=message):
        lt.ansatz_free_kernel(cfg, ds)


def test_theory_rate_rejects_adam():
    with pytest.raises(ValueError, match="train.optimizer"):
        lt.theory_rate(with_field(ring_config(**SMOKE, beta=0.25), "train.optimizer", "adam"))


def test_theory_parameters_needs_a_one_dimensional_grid_ring():
    ds = uniform_latent_dataset(np.random.default_rng(0), SMOKE["P"], SMOKE["K"], SMOKE["m"], 1.0)
    cfg = ring_config(**SMOKE, beta=0.25)
    with pytest.raises(ValueError, match="spatial"):
        lt.theory_parameters(cfg.model_copy(update={"spatial": None}), ds)
    random_hidden = cfg.spatial.model_copy(
        update={"hidden": PopulationSpec(embedding=EmbeddingSpec(kind="random"))}
    )
    with pytest.raises(ValueError, match="grid"):
        lt.theory_parameters(cfg.model_copy(update={"spatial": random_hidden}), ds)
    plane = cfg.spatial.model_copy(
        update={
            "d": 2,
            "input": PopulationSpec(
                embedding=EmbeddingSpec(shape=[2, 2]), kernel=KernelSpec(length_scale=0.0)
            ),
            "hidden": PopulationSpec(embedding=EmbeddingSpec(shape=[4, 4])),
            "output": PopulationSpec(
                embedding=EmbeddingSpec(shape=[2, 2]), kernel=KernelSpec(length_scale=0.0)
            ),
        }
    )
    with pytest.raises(ValueError, match="spatial.d"):
        lt.theory_parameters(cfg.model_copy(update={"spatial": plane}), ds)


def test_theory_parameters_needs_scalar_input_and_output_kernels():
    ds = uniform_latent_dataset(np.random.default_rng(0), SMOKE["P"], SMOKE["K"], SMOKE["m"], 1.0)
    cfg = ring_config(**SMOKE, beta=0.25)
    smooth_input = cfg.spatial.model_copy(
        update={"input": PopulationSpec(kernel=KernelSpec(length_scale=1.0, amplitude=1.0))}
    )
    with pytest.raises(ValueError, match=r"input.*deviates by \d"):
        lt.theory_parameters(cfg.model_copy(update={"spatial": smooth_input}), ds)


def test_theory_parameters_checks_the_latent_factors():
    cfg = ring_config(**SMOKE, beta=0.25)
    ds = uniform_latent_dataset(np.random.default_rng(0), SMOKE["P"], SMOKE["K"], SMOKE["m"], 1.0)
    missing = dataclasses.replace(ds, factors={"z0": ds.factors["z0"]})
    with pytest.raises(ValueError, match="z1"):
        lt.theory_parameters(cfg, missing)
    extra = dataclasses.replace(ds, factors={**ds.factors, "z2": ds.factors["z0"]})
    with pytest.raises(ValueError, match="z2"):
        lt.theory_parameters(cfg, extra)


def test_theory_parameters_checks_the_dataset_shapes():
    cfg = ring_config(**SMOKE, beta=0.25)
    ds = uniform_latent_dataset(np.random.default_rng(0), SMOKE["P"], SMOKE["K"], SMOKE["m"], 1.0)
    with pytest.raises(ValueError, match="dataset.y"):
        lt.theory_parameters(cfg, dataclasses.replace(ds, y=ds.y[:, :3]))
    with pytest.raises(ValueError, match="data.P"):
        lt.theory_parameters(with_field(cfg, "data.P", 40), ds)
    nan_x = ds.x.clone()
    nan_x[3, 1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        lt.theory_parameters(cfg, dataclasses.replace(ds, x=nan_x))


def test_per_mode_spectrum_rejects_nan_and_negative_inputs():
    lam = np.array([3.0, 1.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="sx2"):
        lt.per_mode_spectrum(lam, float("nan"), 1.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="-2"):
        lt.per_mode_spectrum(np.array([3.0, -2.0, 1.0, -2.0]), 1.0, 1.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="finite"):
        lt.per_mode_spectrum(np.array([3.0, np.nan, 1.0, np.nan]), 1.0, 1.0, 0.0, 0.0, 1.0)


# ------------------------------------------------------------------ the teacher precondition


def test_a_gaussian_teacher_is_refused_by_the_per_mode_theory():
    """The per-mode weights from the SVD of Z are the theory only when W_X, W_Y have orthonormal
    columns; `make_dataset`'s Gaussian teacher does not, and the per-mode functions refuse it."""
    cfg = with_field(ring_config(**SMOKE, beta=0.25), "data.teacher", "gaussian")
    ds = make_dataset(cfg.data, torch.device("cpu"), torch.float64)
    with pytest.raises(ValueError, match=r'data.teacher="orthonormal".*deviates by \d'):
        lt.theory_parameters(cfg, ds)
    with pytest.raises(ValueError, match='data.teacher="orthonormal"'):
        lt.predicted_spectra(cfg, ds)


def test_a_dataset_without_the_teacher_maps_is_refused():
    cfg = ring_config(**SMOKE, beta=0.25)
    ds = uniform_latent_dataset(np.random.default_rng(0), SMOKE["P"], SMOKE["K"], SMOKE["m"], 1.0)
    with pytest.raises(ValueError, match="W_X"):
        lt.theory_parameters(cfg, dataclasses.replace(ds, extras={}))


def test_ansatz_free_kernel_is_exact_for_a_gaussian_teacher():
    """`ansatz_free_kernel` needs no orthonormal teacher: on `make_dataset`'s Gaussian teacher it
    is a fixed point of the brute-force master equation."""
    cfg = with_field(ring_config(**SMOKE, beta=0.25, r=0.5), "data.teacher", "gaussian")
    ds = make_dataset(cfg.data, torch.device("cpu"), torch.float64)
    tp = lt.network_parameters(cfg, ds, sig2=SIG2)
    C = lt.ansatz_free_kernel(cfg, ds, sig2=SIG2)
    X, Y = ds.x.numpy(), ds.y.numpy()
    F = brute_force_kernel_map(
        C, circulant_kernel(tp.lam), tp.k0 / tp.m * X @ X.T, Y @ Y.T, tp.k1, SIG2, tp.r, tp.delta
    )
    assert np.abs(F - C).max() < 1e-9 * np.abs(C).max()


# ------------------------------------------------------------------ predicted_spectra from a Config


def test_predicted_spectra_default_is_sigma_zero_on_the_smoke_config():
    ds = uniform_latent_dataset(np.random.default_rng(3), SMOKE["P"], SMOKE["K"], SMOKE["m"], 1.0)
    pred = lt.predicted_spectra(ring_config(**SMOKE, beta=0.25), ds)
    assert pred.params.sig2 == 0.0 and pred.null_floor == 0.0
    assert pred.d.shape == (2, 16) and np.isfinite(pred.d).all() and (pred.d >= 0).all()
    assert pred.lam.shape == (16,) and pred.U.shape == (32, 2)
    assert pred.c.shape == (2,) and pred.tau.shape == (2,)
    np.testing.assert_allclose(pred.d[:, 1], pred.d[:, -1], rtol=1e-12)  # d_q = d_{-q}
    assert all(0 < t < 1 for t in pred.tau)


def test_predicted_spectra_degenerate_contrasts_and_the_floor():
    s2 = lt.uniform_latent_s2(FULL["P"], FULL["K"], A_LATENT)
    ds = exact_s2_dataset(np.random.default_rng(2), s2, FULL["P"], FULL["m"])
    p1 = lt.predicted_spectra(ring_config(**FULL, beta=0.25), ds, sig2=SIG2)
    np.testing.assert_allclose(p1.d[2], p1.d[1], rtol=1e-10)  # the two contrasts are degenerate
    assert p1.null_floor == pytest.approx(SIG2)  # r = 0: sigma_0^2 = sigma^2
    assert p1.params.delta == pytest.approx(8192.0)


# ------------------------------------------------------------------ the ansatz-free kernel


def brute_force_covariance(C_tilde, K, A, YY, k1, sig2, r):
    """Cov[h_{mu i}, h_{nu j}] (P, N, P, N) of the NP x NP Gaussian of eq:precision, in the form
    S (I + S T S)^{-1} S with S = (A (x) K + sig2 I)^{1/2}, T = r I - B (x) K and
    B = C~^{-1} Y Y^T C~^{-1} / k1, which is the inverse of the precision
    (A (x) K + sig2 I)^{-1} + T when that exists and its limit at sig2 = 0; index order (mu, i)."""
    P, N = A.shape[0], K.shape[0]
    Ci = np.linalg.inv(C_tilde)
    B = Ci @ YY @ Ci / k1
    I = np.eye(P * N)
    w, V = np.linalg.eigh(np.kron(A, K) + sig2 * I)
    S = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    T = r * I - np.kron(B, K)
    return (S @ np.linalg.solve(I + S @ T @ S, S)).reshape(P, N, P, N)


def brute_force_kernel_map(C_tilde, K, A, YY, k1, sig2, r, delta):
    """delta I + (1/N) sum_ij K_ij Cov[h_i, h_j] (eq:master) from the brute-force covariance."""
    Cov = brute_force_covariance(C_tilde, K, A, YY, k1, sig2, r)
    return delta * np.eye(A.shape[0]) + np.einsum("ij,uivj->uv", K, Cov) / K.shape[0]


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


def note_operators(cfg, ds):
    """(K, A, Y Y^T) of the brute force for the note's case, K the circulant of the module's lam."""
    lam = lt.network_parameters(cfg, ds).lam
    X, Y = ds.x.numpy(), ds.y.numpy()
    return circulant_kernel(lam), NOTE_K0 / NOTE_M * X @ X.T, Y @ Y.T


def test_brute_force_covariance_is_the_inverse_precision_when_it_exists():
    cfg, ds = note_case()
    K, A, YY = note_operators(cfg, ds)
    C = np.eye(NOTE_P) * 2.0
    Ci = np.linalg.inv(C)
    B = Ci @ YY @ Ci / NOTE_K1
    I = np.eye(NOTE_P * NOTE_N)
    precision = np.linalg.inv(np.kron(A, K) + NOTE_SIG2 * I) - (np.kron(B, K) - NOTE_R * I)
    Cov = brute_force_covariance(C, K, A, YY, NOTE_K1, NOTE_SIG2, NOTE_R)
    np.testing.assert_allclose(
        Cov.reshape(I.shape), np.linalg.inv(precision), atol=1e-12, rtol=1e-10
    )


def test_ansatz_free_kernel_reproduces_the_note_and_the_brute_force_master_equation():
    cfg, ds = note_case()
    C = lt.ansatz_free_kernel(cfg, ds, sig2=NOTE_SIG2)
    assert C.shape == (NOTE_P, NOTE_P) and np.allclose(C, C.T, atol=1e-13)
    np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(C)), NOTE_EIGENVALUES, atol=1e-8)
    K, A, YY = note_operators(cfg, ds)
    F = brute_force_kernel_map(C, K, A, YY, NOTE_K1, NOTE_SIG2, NOTE_R, NOTE_DELTA)
    assert np.abs(F - C).max() < 1e-10


def test_ansatz_free_kernel_at_sigma_zero_solves_the_brute_force_master_equation():
    """The presets' case: at sig2 = 0, lam_q A is singular and only the S (I + S T S)^{-1} S form
    is finite; the kernel is a brute-force fixed point and equals the per-mode roots."""
    cfg, ds = note_case()
    C = lt.ansatz_free_kernel(cfg, ds)
    K, A, YY = note_operators(cfg, ds)
    F = brute_force_kernel_map(C, K, A, YY, NOTE_K1, 0.0, NOTE_R, NOTE_DELTA)
    assert np.abs(F - C).max() < 1e-10
    pred = lt.predicted_spectra(cfg, ds)
    rot = pred.U.T @ C @ pred.U
    np.testing.assert_allclose(np.sort(np.diag(rot)), np.sort(pred.c), rtol=1e-10)


def test_prediction_is_a_fixed_point_of_the_brute_force_master_equation():
    """Synthetic-data-first, N = 12: the kernel the prediction implies (U diag(c) U^T on the
    latents, c_0 on the null directions) is a fixed point of the brute-force NP x NP map, and the
    covariance of the tuning maps t_k = u_k^T h under that law has the predicted spectra d_q
    (unitary DFT: the eigenvalues of the circulant N x N covariance), no cross-mode covariance,
    and the flat floor sigma_0^2 on every null direction."""
    cfg, ds = note_case()
    pred = lt.predicted_spectra(cfg, ds, sig2=NOTE_SIG2)
    U = pred.U
    null = np.linalg.svd(np.eye(NOTE_P) - U @ U.T)[0][:, : NOTE_P - NOTE_K]
    c0 = NOTE_DELTA + pred.null_floor * pred.lam.mean()
    C = U @ np.diag(pred.c) @ U.T + c0 * null @ null.T
    K, A, YY = note_operators(cfg, ds)
    F = brute_force_kernel_map(C, K, A, YY, NOTE_K1, NOTE_SIG2, NOTE_R, NOTE_DELTA)
    assert np.abs(F - C).max() < 1e-10
    Cov = brute_force_covariance(C, K, A, YY, NOTE_K1, NOTE_SIG2, NOTE_R)
    basis = np.concatenate([U, null], axis=1)
    Ct = np.einsum("uk,uivj,vl->kilj", basis, Cov, basis)  # covariance of the projected maps
    spectra = np.fft.fft(Ct[np.arange(NOTE_P), 0, np.arange(NOTE_P), :], axis=-1).real
    np.testing.assert_allclose(spectra[:NOTE_K], pred.d, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(spectra[NOTE_K:], pred.null_floor, rtol=1e-10)
    off_diagonal = Ct.copy()
    off_diagonal[np.arange(NOTE_P), :, np.arange(NOTE_P), :] = 0.0
    assert np.abs(off_diagonal).max() < 1e-12


def test_ansatz_free_kernel_equals_the_per_mode_roots_for_the_orthonormal_teacher():
    cfg, ds = note_case()
    C = lt.ansatz_free_kernel(cfg, ds, sig2=NOTE_SIG2)
    pred = lt.predicted_spectra(cfg, ds, sig2=NOTE_SIG2)
    U = pred.U
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
    rot = pred.U.T @ C @ pred.U
    assert np.abs(rot - np.diag(np.diag(rot))).max() < 1e-10 * np.abs(rot).max()
    np.testing.assert_allclose(np.sort(np.diag(rot)), np.sort(pred.c), rtol=1e-10)
    complement = np.eye(FULL["P"]) - pred.U @ pred.U.T
    np.testing.assert_allclose(
        complement @ C @ complement, pred.params.delta * complement, atol=1e-9
    )


def test_ansatz_free_kernel_sees_the_skew_of_a_non_orthonormal_teacher():
    """W_Y -> W_Y diag(1, 0.7): the Gram matrices no longer commute and the per-mode roots are
    off by up to 0.2% (the limits note's approximate case), which the ansatz-free kernel shows;
    the per-mode theory refuses the skewed teacher outright."""
    cfg, skewed = note_case(wy_skew=np.diag([1.0, 0.7]))
    with pytest.raises(ValueError, match="orthonormal"):
        lt.theory_parameters(cfg, skewed, sig2=NOTE_SIG2)
    C = lt.ansatz_free_kernel(cfg, skewed, sig2=NOTE_SIG2)
    tp = lt.network_parameters(cfg, skewed, sig2=NOTE_SIG2)
    X, Y = skewed.x.numpy(), skewed.y.numpy()
    commutator = X @ X.T @ Y @ Y.T - Y @ Y.T @ X @ X.T
    assert np.abs(commutator).max() > 0.1  # the Gram matrices do not commute
    # The note's approximate roots: sY2 from the diagonal of U^T Y Y^T U, U the eigenbasis of X X^T.
    w, U = np.linalg.eigh(X @ X.T)
    sx2 = np.clip(tp.k0 * w / tp.m, 0, None)
    sy2 = np.clip(np.diag(U.T @ Y @ Y.T @ U), 0, None) / tp.k1
    roots = np.sort(
        [
            lt.physical_root(lt.pi_lam(tp.lam, a, tp.sig2, tp.r), b, tp.delta)
            for a, b in zip(sx2, sy2)
        ]
    )
    rot = U.T @ C @ U
    assert np.abs(rot - np.diag(np.diag(rot))).max() > 1e-2  # the ansatz is only approximate
    rel = np.abs(np.sort(np.linalg.eigvalsh(C)) / roots - 1).max()
    assert 1e-4 < rel < 1e-2  # off by up to 0.2% in the note
