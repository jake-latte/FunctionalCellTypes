"""Known-answer tests for the importance sampler, side by side with HMC.

The synthetic problem is a single site whose tilt is quadratic, so the tilted density is a
Gaussian with a closed-form covariance:

    psi = h,              U = 1/2 h^T M h                    ->  <psi psi^T> = (Sigma^-1 + M)^-1
    psi = phi = identity, U = 1/2 h^T M h + 1/2 lambda_r |h|^2 ->  (Sigma^-1 + M + lambda_r I)^-1

and the effective sample size of prior-proposal importance sampling is closed-form too (see
`exact_ess_frac`). Every expected value is computed here, never by a sampler.
"""

import math

import pytest
import torch

from fct.activations import identity
from fct.hmc import SingleSiteHMC, relative_fro
from fct.importance import ImportanceSampler

P = 4
F64 = {"device": "cpu", "dtype": torch.float64}
N_IS, N_BATCHES = 20_000, 10
N_CHAINS, WARMUP, N_TRANS = 2048, 300, 200


def seeded(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


@pytest.fixture(scope="module")
def Sigma():
    B = torch.randn(P, P, generator=seeded(200), **F64)
    return B @ B.T / P + 0.5 * torch.eye(P, **F64)


@pytest.fixture(scope="module")
def M():
    # Indefinite but mild: Sigma^-1 + M stays positive definite, the density is normalisable.
    C = torch.randn(P, P, generator=seeded(201), **F64)
    return 0.4 * (C @ C.T / P) - 0.1 * torch.eye(P, **F64)


# (psi_is_phi, phi, lambda_r): the readout case psi = h without a rate tilt, and a hidden
# layer with phi = identity, where the rate tilt adds lambda_r I to the precision.
CASES = {"readout_h": (False, torch.relu, 0.0), "identity_rate_tilt": (True, identity, 0.5)}


def exact_cov(Sigma, M, lambda_r):
    eye = torch.eye(P, **F64)
    return torch.linalg.inv(torch.linalg.inv(Sigma) + M + lambda_r * eye)


def exact_ess_frac(C, Q):
    """ESS / n of self-normalised IS with proposal N(0, Q) for the target N(0, C), as n -> inf:
    1 / E_q[(p/q)^2], and E_q[(p/q)^2] = det(C)^-1 det(Q)^1/2 det(2 C^-1 - Q^-1)^-1/2."""
    logdet = torch.logdet
    A = 2 * torch.linalg.inv(C) - torch.linalg.inv(Q)
    return math.exp(logdet(C).item() - 0.5 * logdet(Q).item() + 0.5 * logdet(A).item())


def expected_rel_fro_err(S, n_eff):
    # Relative Frobenius error of the sample covariance of n_eff draws from N(0, S).
    tr_S2 = (S @ S).trace().item()
    return math.sqrt(1 + S.trace().item() ** 2 / tr_S2) / math.sqrt(n_eff)


def importance(Sigma, M, case, seed, n=N_IS, scale=1.0):
    psi_is_phi, phi, lambda_r = CASES[case]
    return ImportanceSampler(
        Sigma, M, lambda_r, phi, psi_is_phi, n_samples=n, proposal_scale=scale,
        sample_dtype=torch.float64, generator=seeded(seed),
    )  # fmt: skip


def hmc(Sigma, M, case, seed):
    psi_is_phi, phi, lambda_r = CASES[case]
    return SingleSiteHMC(
        Sigma, M, lambda_r, phi, psi_is_phi, n_chains=N_CHAINS, sample_dtype=torch.float64,
        generator=seeded(seed),
    )  # fmt: skip


# ----------------------------------------------------------------------------- known answers
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("scale", [1.0, 1.5])
def test_importance_recovers_the_closed_form_kernel(Sigma, M, case, scale):
    s = importance(Sigma, M, case, seed=1, scale=scale)
    moments, diag = s.run(N_BATCHES)
    C = exact_cov(Sigma, M, CASES[case][2])
    err = relative_fro(moments["psi_psi"], C)
    n_eff = diag["ess_frac"] * N_IS * N_BATCHES
    print(f"\nIS {case} scale={scale}: err={err:.2e} mc_err={diag['psi_psi_rel_mc_err']:.2e}")
    assert err < 3 * expected_rel_fro_err(C, n_eff)
    # The sampler's own error estimate is the right size (it is what the solver trusts).
    assert err < 3 * diag["psi_psi_rel_mc_err"]
    assert 0.3 < diag["psi_psi_rel_mc_err"] / expected_rel_fro_err(C, n_eff) < 3
    # <psi psi^T> = <h h^T> here, and the Stein identity <h u^T> = <h h^T> (M + lambda_r I).
    torch.testing.assert_close(moments["h_h"], moments["psi_psi"])
    assert diag["stein_mismatch"] < 3 * expected_rel_fro_err(C, n_eff) + 1e-2


@pytest.mark.parametrize("case", list(CASES))
def test_hmc_recovers_the_same_closed_form_kernel(Sigma, M, case):
    s = hmc(Sigma, M, case, seed=2)
    s.run(WARMUP, collect=False)
    moments, diag = s.run(N_TRANS)
    C = exact_cov(Sigma, M, CASES[case][2])
    err = relative_fro(moments["psi_psi"], C)
    assert err < 3 * expected_rel_fro_err(C, N_CHAINS * N_TRANS * diag["ess_frac"])


@pytest.mark.parametrize("scale", [1.0, 1.5])
def test_importance_ess_matches_its_closed_form(Sigma, M, scale):
    s = importance(Sigma, M, "readout_h", seed=3, scale=scale)
    _, diag = s.run(N_BATCHES)
    expected = exact_ess_frac(exact_cov(Sigma, M, 0.0), scale**2 * s.Sigma)
    assert diag["ess_frac"] == pytest.approx(expected, rel=0.05)
    assert 0.0 < diag["max_weight"] < 1e-3


# ----------------------------------------------------------------------------- collapse
def test_extreme_tilt_collapses_the_weights_and_says_so():
    """A tilt 1e4 times the prior precision: the target is ~100x narrower than the proposal in
    each of 5 directions, so a handful of draws carry all the weight."""
    Sigma, M = torch.eye(5, **F64), 1e4 * torch.eye(5, **F64)
    s = ImportanceSampler(Sigma, M, 0.0, torch.relu, False, n_samples=4096, generator=seeded(4))
    _, diag = s.run(2)
    assert diag["ess_frac"] < 1e-3
    assert diag["max_weight"] > 0.1
    # The variance-based MC error cannot see a collapse (one dominant draw has no spread, so it
    # reads ~0), which is why the ESS, not the MC error, is what flags it.
    assert diag["psi_psi_rel_mc_err"] < 0.1


def test_mild_tilt_keeps_nearly_uniform_weights():
    Sigma, M = torch.eye(5, **F64), 0.05 * torch.eye(5, **F64)
    s = ImportanceSampler(Sigma, M, 0.0, torch.relu, False, n_samples=4096, generator=seeded(5))
    _, diag = s.run(2)
    assert diag["ess_frac"] > 0.95
    assert diag["max_weight"] < 1e-3


def test_non_finite_weights_fail_loudly(Sigma):
    M = torch.full((P, P), float("nan"), **F64)
    s = ImportanceSampler(Sigma, M, 0.0, torch.relu, False, n_samples=64, generator=seeded(6))
    with pytest.raises(FloatingPointError, match="finite"):
        s.run(1)


# ----------------------------------------------------------------------------- interface
def test_hmc_only_diagnostics_are_nan_and_moments_need_collect(Sigma, M):
    s = importance(Sigma, M, "readout_h", seed=7, n=256)
    moments, diag = s.run(3, collect=False)
    assert moments is None
    for key in ("accept", "eps", "n_divergent", "n_reset", "max_abs_dH", "rho1", "h2_max"):
        assert math.isnan(diag[key]), key
    assert 0 < diag["ess_frac"] <= 1 and diag["h2_mean"] > 0
    assert diag["prior_h2"] == pytest.approx(s.Sigma.diagonal().mean().item())
    assert "psi_psi_rel_mc_err" not in diag


def test_draws_are_reproducible_and_independent_of_state(Sigma, M):
    a = importance(Sigma, M, "readout_h", seed=8, n=128)
    b = importance(Sigma, M, "readout_h", seed=8, n=128)
    b.set_state(a.get_state())
    b.load_state_dict(a.state_dict())
    ma, _ = a.run(2)
    mb, _ = b.run(2)
    assert torch.equal(ma["psi_psi"], mb["psi_psi"])


def test_run_draws_n_trans_batches_with_or_without_moments(Sigma, M):
    s = importance(Sigma, M, "readout_h", seed=12, n=32)
    s.run(3, collect=False)
    s.run(2)
    assert s.n_transitions_total == 5


def test_hmc_load_state_dict_rejects_an_importance_state(Sigma, M):
    s = hmc(Sigma, M, "readout_h", seed=13)
    with pytest.raises(ValueError, match="importance"):
        s.load_state_dict(importance(Sigma, M, "readout_h", seed=13, n=16).state_dict())


def test_load_state_dict_rejects_an_hmc_state(Sigma, M):
    s = importance(Sigma, M, "readout_h", seed=9, n=16)
    with pytest.raises(ValueError, match="importance"):
        s.load_state_dict(hmc(Sigma, M, "readout_h", seed=9).state_dict())


def test_tilt_helpers_match_hmc(Sigma, M):
    """The solver projects tilts and reads the normalisability margin through the sampler;
    both samplers must agree on them for the same density."""
    for case in CASES:
        a, b = importance(Sigma, M, case, seed=10, n=16), hmc(Sigma, M, case, seed=10)
        torch.testing.assert_close(a.project_tilt(-3 * M, 0.05), b.project_tilt(-3 * M, 0.05))
        assert a.normalisability_margin() == b.normalisability_margin()
        h = torch.randn(5, P, generator=seeded(11), **F64)
        for x, y in zip(a.tilt_and_grad(h), b.tilt_and_grad(h)):
            torch.testing.assert_close(x, y)
