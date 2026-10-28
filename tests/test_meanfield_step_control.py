"""Step control of the mean-field solver: the inner ascent on Φ̂ accepts a step only if the inner
residual does not rise (beyond its tolerance), the outer step on Φ is scaled by the kernel, and
the last layer's inner ascent starts at its Stein target.

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
from fct.hmc import relative_fro
from tests.pinned_runs import LEGACY_STEP_CONTROL
from tests.test_meanfield import LAMBDA_W, analytic_saddle, linear_cfg, make_linear_problem
from tests.weak_tanh import WEAK_CONSTANTS, WeakTanh, weak_tanh_problem

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
    assert pending[0] is not None and pending[0].tol == pytest.approx(1e-2)
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


def test_small_accepted_rises_cannot_add_up():
    """Tolerance 0.04 (MC error 0.02), a fixed step size: from x = 0.8 the residual goes 0.111,
    0.126 (a rise of 0.015, accepted), 0.152: within the tolerance of the last accepted
    residual, beyond that of the smallest, so it is rejected."""
    eta, x0 = 4.39, 0.8
    x1 = tilt_after_step(x0, eta)
    x2 = tilt_after_step(x1, eta)
    r0, r1, r2 = residual_at(x0), residual_at(x1), residual_at(x2)
    assert r0 < r1 < r0 + 0.04 < r2 < r1 + 0.04  # the creep the rule must stop
    prob, st, _ = layer(x0, eta=eta, mc_err=0.02, eta_grow=1.0, eta_max_phihat=1e3)
    pending, res = [None], mf._new_inner_result(1)
    for inner in range(3):
        mf._inner_layer_step(prob, st, 0, inner, pending, res)
    assert res.n_accepted[0] == 1 and res.n_rejected[0] == 1
    assert res.best_residual[0] == pytest.approx(r0) and res.residual[0] == pytest.approx(r1)


def test_the_monte_carlo_part_of_the_tolerance_is_capped():
    """A run whose MC error is 0.98 (pure noise) cannot call a residual of 0.9 converged."""
    diag = {"psi_psi_rel_mc_err": 0.98}
    progress, health = MeanFieldConfig().solver, MeanFieldConfig(inner_step_control="health").solver
    assert mf._inner_tol(progress, diag) == mf.INNER_TOL_MC_CAP == 0.5
    assert mf._inner_tol(health, diag) == pytest.approx(1.96)  # unchanged without step control
    assert mf._inner_tol(progress, {"psi_psi_rel_mc_err": 1e-3}) == progress.inner_tol


def test_the_log_says_when_an_inner_step_size_sits_at_its_floor():
    prob, st, _ = layer(0.8, eta=100.0, eta_min_rel=0.9, eta_max_phihat=1e3)  # floor 90
    st.eta_hat[0] = 90.0
    assert mf._eta_hat_floor(prob.solver) == 90.0
    rec = {"outer_step": 0, "time": 0.0, "train_mse_net": None, "train_mse_theory": 0.1,
           "picard_resid": 0.1}  # fmt: skip
    rec |= {k: np.zeros(1) for k in ("inner_residual", "outer_grad_rel", "phihat_fp_resid_stein",
            "hmc_accept", "hmc_eps", "hmc_rho1", "hmc_mc_err", "hmc_stein_mismatch",
            "hmc_h2_ratio", "norm_margin", "tilt_projected_rel")}  # fmt: skip
    rec["hmc_div"] = np.zeros(1, dtype=int)
    inner, step = mf._new_inner_result(1), mf._OuterStep([0], [1.0], [False], 0)
    line = mf._format_step(rec, inner, step, st, prob)
    assert "eta_phihat at its floor in layers [0]" in line


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


# ----------------------------------------------------------------------------- weak response
@pytest.fixture(scope="module")
def weak_tanh():
    return weak_tanh_problem()


def weak_cfg(**overrides) -> MeanFieldConfig:
    base = {
        **WEAK_CONSTANTS, "rate_tilt": 0.0, "n_outer": 30, "n_chains": 512, "n_trans": 5,
        "n_trans_outer": 10, "warmup_trans": 50, "init": "lazy", "sample_dtype": "float64",
        "seed": 0,
    }  # fmt: skip
    return MeanFieldConfig(**{**base, **overrides})


def solve_weak(problem: WeakTanh, cfg: MeanFieldConfig, on_outer=None) -> mf.SolveResult:
    return mf.solve(problem.X, problem.Y, 1, 1.0, 0.0, torch.tanh, 1.0, cfg, log=lambda _m: None,
                    on_outer=on_outer)  # fmt: skip


def test_the_weak_response_reference_is_a_weak_response_fixed_point(weak_tanh):
    p = weak_tanh
    assert relative_fro(p.moments(p.target(p.Phi_star)), p.Phi_star) < 1e-12
    eig = torch.linalg.eigvalsh(p.Phi_star).flip(0)
    assert eig[2] < 0.1 * eig[0]  # nearly rank two
    # the response to the tilt along the target, against the Gaussian one
    T = p.target(p.Phi_star)
    response = p.moments(2 * T) - p.moments(T)
    gaussian = -p.Phi_star @ T @ p.Phi_star
    assert (response * gaussian).sum() / (gaussian * gaussian).sum() < 0.2
    # far from the lazy start, so a solver that does not move fails
    assert relative_fro(p.moments(torch.zeros_like(T)), p.Phi_star) > 0.2


def test_a_weak_response_tanh_layer_converges_within_thirty_outer_steps(weak_tanh):
    """Under the default tilt tripwire (solve raises if Φ̂ ends far from its fixed point).
    history["picard_resid"], the last layer's residual at its Stein target, is the Picard
    residual ||⟨ψψᵀ⟩_T(Φ) - Φ|| / ||Φ|| of the kernel entering the step: near convergence it
    matches the quadrature value to its Monte Carlo error (0.02 here). Early on it reads low,
    as the chains equilibrated under the previous tilt lag the restart at the target."""
    p, entering = weak_tanh, []
    result = solve_weak(p, weak_cfg(), on_outer=lambda snap: entering.append(snap.Phi[0]))
    assert all(result.history["inner_converged"])
    assert relative_fro(result.Phi[0], p.Phi_star) < 0.07
    exact = np.array([relative_fro(p.moments(p.target(Phi)), Phi) for Phi in entering[:-1]])
    recorded = np.array(result.history["picard_resid"][1:])
    assert np.abs(recorded - exact)[-10:].max() < 0.04


def test_without_the_target_start_no_target_residual_is_recorded(weak_tanh):
    cfg = weak_cfg(n_outer=2, inner_start="previous", tilt_tripwire_resid=1e9)
    assert np.isnan(solve_weak(weak_tanh, cfg).history["picard_resid"]).all()


def test_the_legacy_step_control_does_not_converge_on_it(weak_tanh):
    result = solve_weak(weak_tanh, weak_cfg(**LEGACY_STEP_CONTROL, tilt_tripwire_resid=1e9))
    assert relative_fro(result.Phi[0], weak_tanh.Phi_star) > 0.1


# ----------------------------------------------------------------------------- Gaussian layer
def test_a_gaussian_layer_follows_the_legacy_path_to_the_same_saddle():
    """The L = 1 linear problem of tests/test_meanfield.py, whose saddle is known in closed
    form. The step-controlled path lags the legacy one (its outer step size is capped at 0.25,
    the legacy one grew to 1) by less than 0.1 at every outer step, and the two end within 0.01
    of each other on the analytic saddle."""
    X, Y = make_linear_problem()
    Phi_star, _, _, _ = analytic_saddle(X, Y)

    def distances(**solver) -> np.ndarray:
        path = []

        def record(snap):
            path.append(relative_fro(snap.Phi[0], Phi_star))

        mf.solve(X, Y, 1, LAMBDA_W, 0.0, identity, 1.0, linear_cfg(**solver), log=lambda _m: None,
                 on_outer=record)  # fmt: skip
        return np.array(path)

    new, legacy = distances(), distances(**LEGACY_STEP_CONTROL)
    print(f"\nGaussian layer, distance to the saddle: step-controlled {np.round(new, 3)}, "
          f"legacy {np.round(legacy, 3)}")  # fmt: skip
    assert (new - legacy).max() < 0.1
    assert new[-1] < 0.04 and legacy[-1] < 0.04
    assert abs(new[-1] - legacy[-1]) < 0.01
