"""Known-answer tests for SingleSiteHMC: every expected value is derived analytically
or by exact importance sampling from the prior, never from the sampler itself.

Each sampler and reference gets its own seeded generator, so no test's statistics depend on
which other tests ran or in what order."""

import math

import pytest
import torch

from fct.hmc import SingleSiteHMC, kernel_alignment, project_pd, relative_fro, sym

P, N_CHAINS, WARMUP, N_TRANS = 20, 2048, 300, 200
N_SMALL = 64
F64 = {"device": "cpu", "dtype": torch.float64}


def seeded(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


@pytest.fixture(scope="module")
def Sigma():
    B = torch.randn(P, P, generator=seeded(100), **F64)
    return B @ B.T / P + 0.5 * torch.eye(P, **F64)


@pytest.fixture(scope="module")
def M():
    C = torch.randn(P, P, generator=seeded(101), **F64)
    return 0.3 * (C @ C.T / P)


def make_sampler(Sigma, M, lambda_r, phi, seed, n_chains=N_CHAINS, psi_is_phi=True):
    return SingleSiteHMC(
        Sigma, M, lambda_r, phi, psi_is_phi, n_chains=n_chains, generator=seeded(seed)
    )


def warm_and_run(sampler):
    sampler.run(WARMUP, collect=False)
    return sampler.run(N_TRANS)


def expected_rel_fro_err(S, n_eff):
    # Relative Frobenius error of the sample covariance of n_eff draws from N(0, S):
    # E||xx^T - S||_F^2 = tr(S)^2 + tr(S^2), divided by n_eff and normalised by ||S||_F^2.
    tr_S2 = (S @ S).trace().item()
    return math.sqrt(1 + S.trace().item() ** 2 / tr_S2) / math.sqrt(n_eff)


def n_eff_of(diag):
    return N_CHAINS * N_TRANS * diag["ess_frac"]


@pytest.fixture(scope="module")
def gaussian(Sigma):
    s = make_sampler(Sigma, torch.zeros(P, P, **F64), 0.0, torch.relu, seed=1)
    return (s, *warm_and_run(s))


@pytest.fixture(scope="module")
def linear(Sigma, M):
    s = make_sampler(Sigma, M, 0.0, lambda h: h, seed=2)
    return (s, *warm_and_run(s))


@pytest.fixture(scope="module")
def relu(Sigma, M):
    s = make_sampler(Sigma, M, 0.1, torch.relu, seed=3)
    return (s, *warm_and_run(s))


def test_untilted_gaussian_recovers_prior_covariance(gaussian, Sigma):
    _, moments, diag = gaussian
    err = relative_fro(moments["h_h"], Sigma)
    assert err < 3 * expected_rel_fro_err(Sigma, n_eff_of(diag))


def test_linear_phi_with_tilt_recovers_gaussian_posterior(linear, Sigma, M):
    _, moments, diag = linear
    exact = torch.linalg.inv(torch.linalg.inv(Sigma) + M)
    err = relative_fro(moments["h_h"], exact)
    assert err < 3 * expected_rel_fro_err(exact, n_eff_of(diag))


def importance_sampling_psi_psi(Sigma, M, n_is=400_000):
    """<phi phi^T> under the tilted density (lambda_r = 0.1), reweighting exact prior draws."""
    h = torch.randn(n_is, P, generator=seeded(200), **F64) @ torch.linalg.cholesky(Sigma).T
    ph = torch.relu(h)
    logw = -0.5 * ((ph @ M) * ph).sum(-1) - 0.05 * (ph * ph).sum(-1)
    w = torch.softmax(logw, 0)
    return (ph * w[:, None]).T @ ph


def test_relu_with_tilt_satisfies_stein_identity(relu):
    _, _, diag = relu
    assert diag["stein_mismatch"] < 0.05


def test_relu_with_tilt_matches_importance_sampling(relu, Sigma, M):
    _, moments, _ = relu
    reference = importance_sampling_psi_psi(Sigma, M)
    assert relative_fro(moments["psi_psi"], reference) < 0.05


@pytest.mark.parametrize("name", ["gaussian", "linear", "relu"])
def test_acceptance_rate_near_target_after_warmup(name, request):
    sampler, _, diag = request.getfixturevalue(name)
    assert abs(diag["accept"] - sampler.target_accept) < 0.15


def test_run_reports_documented_keys(relu):
    _, moments, diag = relu
    assert set(moments) == {"psi_psi", "h_h", "h_u"}
    assert set(diag) == {
        "accept",
        "eps",
        "n_divergent",
        "n_reset",
        "max_abs_dH",
        "mean_abs_dH",
        "rho1",
        "ess_frac",
        "h2_mean",
        "h2_max",
        "prior_h2",
        "psi_psi_rel_mc_err",
        "stein_mismatch",
    }


# ----------------------------------------------------------------------------- state_dict
@pytest.fixture
def warmed_small(Sigma, M):
    """A small sampler that has run long enough to have a non-trivial step size and summary."""
    s = make_sampler(Sigma, M, 0.1, torch.relu, seed=4, n_chains=N_SMALL)
    s.run(30, collect=False)
    return s


def test_state_dict_holds_only_tensors_and_scalars(warmed_small):
    state = warmed_small.state_dict()
    assert all(isinstance(v, torch.Tensor | int | float | bool | None) for v in state.values())
    assert state["h"].dtype == torch.float64


def test_state_dict_round_trip_reproduces_chain_and_next_transition(warmed_small, Sigma, M):
    state = warmed_small.state_dict()
    fresh = make_sampler(Sigma, M, 0.1, torch.relu, seed=5, n_chains=N_SMALL)
    fresh.load_state_dict(state)
    torch.testing.assert_close(fresh.h(), warmed_small.h(), rtol=1e-5, atol=1e-5)
    assert fresh.eps == warmed_small.eps
    assert fresh.n_transitions_total == warmed_small.n_transitions_total

    warmed_small.gen.manual_seed(123)
    fresh.gen.manual_seed(123)
    warmed_small.transition()
    fresh.transition()
    torch.testing.assert_close(fresh.h(), warmed_small.h(), rtol=1e-3, atol=1e-3)


def test_state_dict_loads_after_prior_and_tilt_refresh(warmed_small, Sigma, M):
    fresh = make_sampler(Sigma, M, 0.1, torch.relu, seed=5, n_chains=N_SMALL)
    fresh.set_prior(Sigma)
    fresh.set_tilt(M)
    fresh.load_state_dict(warmed_small.state_dict())
    torch.testing.assert_close(fresh.h(), warmed_small.h(), rtol=1e-5, atol=1e-5)


def test_state_dict_is_independent_of_later_transitions(warmed_small):
    state = warmed_small.state_dict()
    before = state["h"].clone()
    warmed_small.transition()
    torch.testing.assert_close(state["h"], before)


def test_state_dict_survives_torch_save_and_weights_only_load(warmed_small, Sigma, M, tmp_path):
    path = tmp_path / "chains.pt"
    torch.save(warmed_small.state_dict(), path)
    loaded = torch.load(path, weights_only=True)

    fresh = make_sampler(Sigma, M, 0.1, torch.relu, seed=5, n_chains=N_SMALL)
    fresh.load_state_dict(loaded)
    assert torch.equal(fresh.h(), warmed_small.h())
    assert fresh.log_eps == warmed_small.log_eps
    assert torch.equal(fresh._prev_summary, warmed_small._prev_summary)


def test_load_state_dict_restores_adapt_flag(warmed_small, Sigma, M):
    warmed_small.adapt = False
    fresh = make_sampler(Sigma, M, 0.1, torch.relu, seed=5, n_chains=N_SMALL)
    fresh.load_state_dict(warmed_small.state_dict())
    assert fresh.adapt is False


def test_load_state_dict_rejects_chain_shape_mismatch(warmed_small, Sigma, M):
    other = make_sampler(Sigma, M, 0.1, torch.relu, seed=5, n_chains=N_SMALL // 2)
    with pytest.raises(ValueError, match="chain shape"):
        other.load_state_dict(warmed_small.state_dict())


# ----------------------------------------------------------------------------- project_tilt
@pytest.mark.parametrize("delta", [0.1, 0.5])
def test_project_tilt_phi_is_normalisable(Sigma, delta):
    s = make_sampler(Sigma, torch.zeros(P, P, **F64), 0.1, torch.relu, seed=6, n_chains=8)
    out = s.project_tilt(-3.0 * torch.eye(P, **F64), delta)
    floor = delta * s.lambda_r
    assert torch.linalg.eigvalsh(out + s.lambda_r * torch.eye(P, **F64)).min() >= floor - 1e-9


@pytest.mark.parametrize("delta", [0.1, 0.5])
def test_project_tilt_h_is_normalisable(Sigma, delta):
    s = make_sampler(
        Sigma, torch.zeros(P, P, **F64), 0.0, torch.relu, seed=6, n_chains=8, psi_is_phi=False
    )
    out = s.project_tilt(-5.0 * torch.linalg.inv(s.Sigma), delta)
    root = torch.linalg.cholesky(s.Sigma)
    margin = torch.linalg.eigvalsh(root.T @ out @ root).min()
    assert margin >= -(1 - delta) - 1e-9


def test_project_tilt_leaves_admissible_tilt_unchanged(Sigma, M):
    s = make_sampler(Sigma, M, 0.0, torch.relu, seed=6, n_chains=8, psi_is_phi=False)
    torch.testing.assert_close(s.project_tilt(M, 0.1), sym(M))


# ----------------------------------------------------------------------------- helpers
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


def test_relative_fro_is_norm_of_difference_over_norm_of_reference():
    B = torch.eye(3, dtype=torch.float64)
    assert relative_fro(2 * B, B) == pytest.approx(1.0)


def test_kernel_alignment_is_cosine_similarity_with_zero_floor():
    B = torch.eye(3, dtype=torch.float64)
    assert kernel_alignment(3 * B, B) == pytest.approx(1.0)
    assert kernel_alignment(B, torch.zeros_like(B)) == 0.0


@pytest.mark.parametrize("psi_is_phi, lambda_r", [(False, 0.0), (True, 0.0), (True, 0.3)])
def test_tilt_and_grad_is_the_tilt_exponent_and_its_gradient(Sigma, M, psi_is_phi, lambda_r):
    """U(h) = 1/2 psi^T M psi + 1/2 lambda_r phi^T phi, u = dU/dh by autograd, for samples the
    caller brings (in float64 for a float64 sampler)."""
    s = SingleSiteHMC(
        Sigma, M, lambda_r, torch.tanh, psi_is_phi, n_chains=1, sample_dtype=torch.float64,
        generator=seeded(7),
    )  # fmt: skip
    h = torch.randn(5, P, generator=seeded(8), **F64).requires_grad_(True)
    psi = torch.tanh(h) if psi_is_phi else h
    U_ref = 0.5 * ((psi @ M) * psi).sum(-1) + 0.5 * lambda_r * (torch.tanh(h) ** 2).sum(-1)
    (u_ref,) = torch.autograd.grad(U_ref.sum(), h)
    U, u = s.tilt_and_grad(h.detach())
    torch.testing.assert_close(U, U_ref.detach())
    torch.testing.assert_close(u, u_ref)
