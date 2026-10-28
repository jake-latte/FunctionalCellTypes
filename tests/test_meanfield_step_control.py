"""Step control of the mean-field solver: the inner ascent on Φ̂ accepts a step only if the inner
residual does not rise (beyond its tolerance), and the outer step on Φ is scaled by the kernel.

The inner-step tests drive `_inner_layer_step` with a stand-in sampler that returns the exact
moments of a tilted Gaussian, so every residual is known in closed form:
with Σ = I, a tilt Φ̂ = x I gives ⟨ψψᵀ⟩ = I / (1 + x), and against Φ = ½ I the residual is
|2 / (1 + x) - 1|, the gradient ½(⟨ψψᵀ⟩ - Φ), and c² = (tr Φ / P)² = ¼.
"""

import math

import numpy as np
import pytest
import torch

import fct.meanfield as mf
from fct.activations import identity
from fct.config import MeanFieldConfig

F64 = {"device": "cpu", "dtype": torch.float64}
P = 3
EYE = torch.eye(P, **F64)


class ExactGaussianSampler:
    """Stands in for a layer's sampler on a linear layer without rate tilt: `run` returns the
    exact ⟨ψψᵀ⟩ = (Σ⁻¹ + M)⁻¹ with a fixed Monte Carlo error, and the state is a counter of
    runs, so a reverted step is visible."""

    def __init__(self, Sigma: torch.Tensor, mc_err: float):
        self.Sigma_inv = torch.linalg.inv(Sigma)
        self.M = torch.zeros_like(Sigma)
        self.mc_err = mc_err
        self.state = 0

    def set_tilt(self, M):
        self.M = M.clone()

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def project_tilt(self, M, delta):
        return M

    def run(self, n_trans, collect=True):
        self.state += 1
        moments = {"psi_psi": torch.linalg.inv(self.Sigma_inv + self.M)}
        diag = {
            "h2_mean": 1.0, "psi_psi_rel_mc_err": self.mc_err, "n_divergent": 0, "n_reset": 0,
            "eps": 1.0,
        }  # fmt: skip
        return moments, diag


def residual_at(x: float) -> float:
    """Inner residual of the tilt x I against Φ = ½ I (see the module docstring)."""
    return abs(2.0 / (1.0 + x) - 1.0)


def tilt_after_step(x: float, eta: float) -> float:
    """x + η ½(1/(1+x) - ½) / c², the tilt one inner step from x I reaches."""
    return x + eta * 0.5 * (1.0 / (1.0 + x) - 0.5) / 0.25


def layer(x0: float, eta: float, mc_err: float = 1e-3, **solver):
    """(problem, state, sampler) of one linear layer with Σ = I, Φ = ½ I and tilt x0 I."""
    cfg = MeanFieldConfig(
        alpha=1.0, prior_scale=1.0, rate_tilt=0.0, learning_prefactor=0.5,
        inner_tol=1e-2, inner_tol_mc=2.0, project_tilt=False, eta_phihat=eta, **solver,
    )  # fmt: skip
    X = torch.eye(P, **F64) * math.sqrt(P)  # Φ0 = X Xᵀ / P = I
    prob = mf._make_problem(X, torch.ones(P, 1, **F64), 1, 1.0, 0.0, identity, 1.0, cfg)
    sampler = ExactGaussianSampler(EYE, mc_err)
    sampler.set_tilt(x0 * EYE)
    st = mf._SolverState(
        outer=0, Phi=0.5 * EYE[None].clone(), Phi_hat=x0 * EYE[None].clone(), samplers=[sampler],
        gen=None, eta=0.05, eta_hat=np.array([eta]), h2_last=[1.0], eps_ref=[1.0],
        ess_ref=[float("nan")], history={}, mse_net=None, net_Phi=None, elapsed=0.0,
    )  # fmt: skip
    return prob, st, sampler


def two_steps(prob, st):
    """Run two inner steps on layer 0: the first steps from the start, the second validates it."""
    pending, res = [None], mf._new_inner_result(1)
    mf._inner_layer_step(prob, st, 0, 0, pending, res)
    mf._inner_layer_step(prob, st, 0, 1, pending, res)
    return pending, res


def tilt(st) -> float:
    return st.Phi_hat[0, 0, 0].item()


# ----------------------------------------------------------------------------- inner step
def test_a_step_that_raises_the_inner_residual_is_rejected_and_its_step_size_halved():
    prob, st, sampler = layer(0.8, eta=100.0)
    x1 = tilt_after_step(0.8, 100.0)
    assert residual_at(x1) > residual_at(0.8) + 0.5  # the overshoot the test is about
    pending, res = two_steps(prob, st)
    assert res.n_rejected[0] == 1 and res.n_accepted[0] == 0
    assert st.eta_hat[0] == 50.0
    # retried from the point before the step, with the chain state of that point
    assert tilt(st) == pytest.approx(tilt_after_step(0.8, 50.0), rel=1e-12)
    assert sampler.state == 1
    assert pending[0] is not None and pending[0].residual == pytest.approx(residual_at(0.8))
    assert res.residual[0] == pytest.approx(residual_at(0.8))  # the accepted point's


def test_health_only_step_control_accepts_the_same_step():
    prob, st, _ = layer(0.8, eta=100.0, inner_step_control="health")
    _, res = two_steps(prob, st)
    assert res.n_rejected[0] == 0 and res.n_accepted[0] == 1
    assert res.residual[0] == pytest.approx(residual_at(tilt_after_step(0.8, 100.0)))


def test_a_step_that_lowers_the_residual_is_accepted_and_the_step_size_grows():
    prob, st, _ = layer(0.8, eta=0.1)
    _, res = two_steps(prob, st)
    assert res.n_accepted[0] == 1 and res.n_rejected[0] == 0
    assert st.eta_hat[0] == pytest.approx(0.1 * prob.solver.eta_grow)


@pytest.mark.parametrize(("mc_err", "rejected"), [(1e-3, 1), (2e-2, 0)])
def test_a_rise_within_the_tolerance_of_the_point_before_is_noise(mc_err, rejected):
    """From x = 0.8 (residual 0.111) a step to x = 1.3 raises the residual by 0.019: more than
    a tolerance of 0.01 (inner_tol, small MC error), less than one of 0.04 (2 x MC error 0.02)."""
    eta = (1.3 - 0.8) / (0.5 * (1 / 1.8 - 0.5) / 0.25)
    assert tilt_after_step(0.8, eta) == pytest.approx(1.3)
    prob, st, _ = layer(0.8, eta=eta, mc_err=mc_err)
    _, res = two_steps(prob, st)
    assert res.n_rejected[0] == rejected


def test_the_step_size_floor_stops_the_halving():
    prob, st, _ = layer(0.8, eta=100.0, eta_min_rel=0.9, eta_max_phihat=1e3)  # floor 90
    pending, res = [None], mf._new_inner_result(1)
    for inner in range(3):
        mf._inner_layer_step(prob, st, 0, inner, pending, res)
    assert res.n_rejected[0] == 1 and res.n_accepted[0] == 1
    assert st.eta_hat[0] == pytest.approx(90.0 * prob.solver.eta_grow)


# ----------------------------------------------------------------------------- outer step
def random_pd(seed: int, scale: torch.Tensor) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(P, P, generator=g, **F64))
    return (Q * scale) @ Q.T


def test_the_kernel_step_scale_is_the_trace_scale_for_an_isotropic_kernel():
    grad = random_pd(1, torch.tensor([1.0, -0.5, 0.2], **F64))
    Phi = 0.7 * EYE
    kernel = mf._phi_step(MeanFieldConfig().solver, 1.0, grad, Phi, Phi)
    trace = mf._phi_step(MeanFieldConfig(phi_step_scale="trace").solver, 1.0, grad, Phi, Phi)
    torch.testing.assert_close(kernel, trace, rtol=1e-14, atol=1e-14)


def test_the_kernel_step_is_the_newton_step_of_a_gaussian_layer():
    """At Φ = (Σ⁻¹ + Φ̂)⁻¹, a Gaussian layer's tilt follows its kernel as Φ̂(Φ) = Φ⁻¹ - Σ⁻¹.
    For a target tilt T a little away, the step Φ - η Φ ∂S/∂Φ Φ with η = 2 and
    ∂S/∂Φ = ½(T - Φ̂) lands where Φ̂(Φ) = T up to second order; the trace scale, on a kernel
    with a spread of eigenvalues, does not."""
    Sigma = random_pd(2, torch.tensor([4.0, 1.0, 0.1], **F64))
    Sigma_inv = torch.linalg.inv(Sigma)
    hat = -0.2 * random_pd(3, torch.tensor([1.0, 0.5, 0.1], **F64))
    Phi = torch.linalg.inv(Sigma_inv + hat)
    T = hat + 1e-3 * random_pd(4, torch.tensor([1.0, -1.0, 0.5], **F64))
    grad = -0.5 * hat + 0.5 * T

    def tilt_error(scale: str) -> float:
        cfg = MeanFieldConfig(phi_step_scale=scale).solver
        new = Phi - mf._phi_step(cfg, 2.0, grad, Phi, Phi)
        return (torch.linalg.inv(new) - Sigma_inv - T).norm().item() / (T - hat).norm().item()

    assert tilt_error("kernel") < 1e-2
    assert tilt_error("trace") > 0.5
