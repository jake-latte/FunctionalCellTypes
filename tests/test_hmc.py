"""Known-answer tests for SingleSiteHMC: every expected value is derived analytically
or by exact importance sampling from the prior, never from the sampler itself."""

import math

import pytest
import torch

from fct.hmc import SingleSiteHMC, kernel_alignment, project_pd, relative_fro, sym

P, N_CHAINS, WARMUP, N_TRANS = 20, 2048, 300, 200
F64 = dict(device="cpu", dtype=torch.float64)


@pytest.fixture(scope="module")
def gen():
    return torch.Generator(device="cpu").manual_seed(0)


@pytest.fixture(scope="module")
def Sigma(gen):
    B = torch.randn(P, P, generator=gen, **F64)
    return B @ B.T / P + 0.5 * torch.eye(P, **F64)


@pytest.fixture(scope="module")
def M(gen):
    C = torch.randn(P, P, generator=gen, **F64)
    return 0.3 * (C @ C.T / P)


def make_sampler(Sigma, M, lambda_r, phi, gen):
    return SingleSiteHMC(Sigma, M, lambda_r, phi, True, n_chains=N_CHAINS, generator=gen)


def warm_and_run(sampler):
    sampler.run(WARMUP, collect=False)
    return sampler.run(N_TRANS)


def mc_error(diag):
    return 1 / math.sqrt(N_CHAINS * N_TRANS * diag["ess_frac"])


@pytest.fixture(scope="module")
def gaussian(Sigma, gen):
    s = make_sampler(Sigma, torch.zeros(P, P, **F64), 0.0, torch.relu, gen)
    return (s, *warm_and_run(s))


@pytest.fixture(scope="module")
def linear(Sigma, M, gen):
    s = make_sampler(Sigma, M, 0.0, lambda h: h, gen)
    return (s, *warm_and_run(s))


@pytest.fixture(scope="module")
def relu(Sigma, M, gen):
    s = make_sampler(Sigma, M, 0.1, torch.relu, gen)
    return (s, *warm_and_run(s))


def test_untilted_gaussian_recovers_prior_covariance(gaussian, Sigma):
    _, moments, diag = gaussian
    assert relative_fro(moments["h_h"], Sigma) < 5 * mc_error(diag)


def test_linear_phi_with_tilt_recovers_gaussian_posterior(linear, Sigma, M):
    _, moments, diag = linear
    exact = torch.linalg.inv(torch.linalg.inv(Sigma) + M)
    assert relative_fro(moments["h_h"], exact) < 5 * mc_error(diag)


def importance_sampling_psi_psi(Sigma, M, gen, n_is=400_000):
    """<phi phi^T> under the tilted density, reweighting exact prior draws (lambda_r = 0.1)."""
    h = torch.randn(n_is, P, generator=gen, **F64) @ torch.linalg.cholesky(Sigma).T
    ph = torch.relu(h)
    logw = -0.5 * ((ph @ M) * ph).sum(-1) - 0.05 * (ph * ph).sum(-1)
    w = torch.softmax(logw, 0)
    return (ph * w[:, None]).T @ ph


def test_relu_with_tilt_satisfies_stein_identity(relu):
    _, _, diag = relu
    assert diag["stein_mismatch"] < 0.05


def test_relu_with_tilt_matches_importance_sampling(relu, Sigma, M, gen):
    _, moments, _ = relu
    reference = importance_sampling_psi_psi(Sigma, M, gen)
    assert relative_fro(moments["psi_psi"], reference) < 0.05


@pytest.mark.parametrize("name", ["gaussian", "linear", "relu"])
def test_acceptance_rate_near_target_after_warmup(name, request):
    sampler, _, diag = request.getfixturevalue(name)
    assert abs(diag["accept"] - sampler.target_accept) < 0.15


def test_run_reports_documented_keys(relu):
    _, moments, diag = relu
    assert set(moments) == {"psi_psi", "h_h", "h_u"}
    assert set(diag) == {
        "accept", "eps", "n_divergent", "n_reset", "max_abs_dH", "mean_abs_dH", "rho1",
        "ess_frac", "h2_mean", "h2_max", "prior_h2", "psi_psi_rel_mc_err", "stein_mismatch",
    }


def test_state_dict_round_trip_reproduces_chain_and_next_transition(relu, Sigma, M):
    original, _, _ = relu
    state = original.state_dict()
    assert all(isinstance(v, (torch.Tensor, int, float, type(None))) for v in state.values())

    fresh = make_sampler(Sigma, M, 0.1, torch.relu, torch.Generator().manual_seed(1))
    fresh.load_state_dict(state)
    torch.testing.assert_close(fresh.h(), original.h(), rtol=1e-5, atol=1e-5)
    assert fresh.eps == original.eps
    assert fresh.n_transitions_total == original.n_transitions_total

    original.gen.manual_seed(123)
    fresh.gen.manual_seed(123)
    original.transition()
    fresh.transition()
    torch.testing.assert_close(fresh.h(), original.h(), rtol=1e-3, atol=1e-3)


def test_state_dict_is_independent_of_later_transitions(relu):
    sampler, _, _ = relu
    state = sampler.state_dict()
    before = state["h"].clone()
    sampler.transition()
    torch.testing.assert_close(state["h"], before)


@pytest.mark.parametrize("delta", [0.1, 0.5])
def test_project_tilt_phi_is_normalisable(Sigma, delta):
    s = SingleSiteHMC(Sigma, torch.zeros(P, P, **F64), 0.1, torch.relu, True, n_chains=8)
    out = s.project_tilt(-3.0 * torch.eye(P, **F64), delta)
    floor = delta * s.lambda_r
    assert torch.linalg.eigvalsh(out + s.lambda_r * torch.eye(P, **F64)).min() >= floor - 1e-9


@pytest.mark.parametrize("delta", [0.1, 0.5])
def test_project_tilt_h_is_normalisable(Sigma, delta):
    s = SingleSiteHMC(Sigma, torch.zeros(P, P, **F64), 0.0, torch.relu, False, n_chains=8)
    out = s.project_tilt(-5.0 * torch.linalg.inv(s.Sigma), delta)
    root = torch.linalg.cholesky(s.Sigma)
    margin = torch.linalg.eigvalsh(root.T @ out @ root).min()
    assert margin >= -(1 - delta) - 1e-9


def test_project_tilt_leaves_admissible_tilt_unchanged(Sigma, M):
    s = SingleSiteHMC(Sigma, M, 0.0, torch.relu, False, n_chains=8)
    torch.testing.assert_close(s.project_tilt(M, 0.1), sym(M))


def test_project_pd_floors_eigenvalues_and_counts_clamped():
    A = torch.diag(torch.tensor([2.0, 1.0, -0.5, 1e-9], dtype=torch.float64))
    out, n_clamped, lam_min = project_pd(A, floor_rel=1e-3)
    assert n_clamped == 2
    assert lam_min == pytest.approx(-0.5)
    assert torch.linalg.eigvalsh(out).min() >= 2e-3 - 1e-12


def test_project_pd_is_noop_when_already_pd():
    A = torch.eye(3, dtype=torch.float64)
    out, n_clamped, _ = project_pd(A, floor_rel=1e-3)
    assert n_clamped == 0
    torch.testing.assert_close(out, A)


def test_relative_fro_and_kernel_alignment():
    B = torch.eye(3, dtype=torch.float64)
    assert relative_fro(2 * B, B) == pytest.approx(1.0)
    assert kernel_alignment(3 * B, B) == pytest.approx(1.0)
    assert kernel_alignment(B, torch.zeros_like(B)) == 0.0
