"""Tests for the mean-field saddle-point solver.

The correctness test uses a problem whose saddle point is known in closed form (L=1, linear
phi, no rate tilt: the single-site density is Gaussian); the reference is computed here by
a deterministic float64 fixed-point iteration, never by the solver.  The other tests check
the packaging: checkpoint / resume, caching, staleness and the history layout.

Every solve gets its own seeded generator through `cfg.seed`, so no test depends on the
global RNG or on which other tests ran.
"""

import time

import numpy as np
import pytest
import torch

import fct.config
import fct.meanfield as mf
from fct.activations import identity
from fct.cache import StaleCacheError
from fct.config import Config, DataConfig, MeanFieldConfig, NetConfig, TrainConfig
from fct.data import Dataset
from fct.hmc import SingleSiteHMC, project_tilt, relative_fro
from fct.importance import ImportanceSampler
from fct.meanfield import (
    HISTORY_KEYS,
    SpatialPriorUnsupportedError,
    empirical_kernels,
    load_meanfield,
    net_kernels,
    readout_activities,
    run_meanfield,
    solve,
)
from fct.model import make_net
from fct.spatial import SpatialConfig

F64 = {"device": "cpu", "dtype": torch.float64}


def seeded(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def silent(_msg):
    pass


# The packaging tests run a few outer steps, far from converged; this lets them finish (the
# final-residual check has its own tests).
ANY_RESID = 1e9


# ----------------------------------------------------------------------------- analytic L=1
LAMBDA_W, ALPHA, PREFACTOR = 2.0, 0.5, 0.5
P_LIN, D_LIN = 8, 8


def make_linear_problem() -> tuple[torch.Tensor, torch.Tensor]:
    """X with Phi0 = X X^T / D_in = U diag(s) U^T, s in [0.8, 1.2] (well conditioned), and
    targets Y scaled so that the saddle point moves Phi ~20% away from the prior yet keeps
    Sigma^-1 + Phi_hat comfortably positive definite.  (Much larger Y also admits a second,
    unphysical fixed point with Sigma^-1 + Phi_hat indefinite.)"""
    U, _ = torch.linalg.qr(torch.randn(P_LIN, P_LIN, generator=seeded(10), **F64))
    s = torch.linspace(0.8, 1.2, P_LIN, **F64)
    X = (U * s.sqrt()) * D_LIN**0.5  # (P, D_in) with D_in = P
    Y = 0.4 * torch.randn(P_LIN, 2, generator=seeded(11), **F64)
    return X, Y


@pytest.fixture(scope="module")
def linear_problem():
    return make_linear_problem()


def analytic_saddle(X, Y, n_iter=2000):
    """Saddle point of the L=1 linear problem by fixed-point iteration in float64:
    inner  Phi = (Sigma^-1 + Phi_hat)^-1  with Sigma = Phi0 / lambda_w  (Gaussian density),
    outer  Phi_hat = -2c A Y Y^T A        with A = (alpha I + Phi)^-1."""
    P, D_in = X.shape
    eye = torch.eye(P, **F64)
    Sigma = X @ X.T / D_in / LAMBDA_W
    Sigma_inv = torch.linalg.inv(Sigma)
    Phi = Sigma.clone()
    for _ in range(n_iter):
        A = torch.linalg.inv(ALPHA * eye + Phi)
        Phi_hat = -2.0 * PREFACTOR * A @ Y @ Y.T @ A
        Phi = torch.linalg.inv(Sigma_inv + Phi_hat)
    A = torch.linalg.inv(ALPHA * eye + Phi)
    residual = relative_fro(Phi_hat, -2.0 * PREFACTOR * A @ Y @ Y.T @ A)
    return Phi, Phi_hat, Sigma, residual


def linear_cfg(**overrides):
    base = {
        "alpha": ALPHA,
        "learning_prefactor": PREFACTOR,
        "n_outer": 40,
        "eta_phi": 0.5,
        "n_chains": 256,
        "n_trans": 100,
        "n_trans_outer": 200,
        "warmup_trans": 200,
        "init": "lazy",
        "log_int": 10,
        "save_int": 10,
        "seed": 0,
    }
    return MeanFieldConfig(**{**base, **overrides})


@pytest.fixture(scope="module")
def linear_solution(linear_problem):
    X, Y = linear_problem
    start = time.perf_counter()
    result = solve(X, Y, 1, LAMBDA_W, 0.0, identity, 1.0, linear_cfg(), log=silent)
    return result, time.perf_counter() - start


def test_analytic_reference_is_a_fixed_point(linear_problem):
    X, Y = linear_problem
    Phi, Phi_hat, Sigma, residual = analytic_saddle(X, Y)
    assert residual < 1e-10
    # Far enough from the prior that "the solver did not move" fails the solve test, and on
    # the physical branch: the tilted density is normalisable, so Phi is positive definite.
    assert relative_fro(Sigma, Phi) > 0.15
    assert torch.linalg.eigvalsh(torch.linalg.inv(Sigma) + Phi_hat).min() > 1.0


def test_the_default_projection_keeps_the_analytic_saddle_tilt(linear_problem):
    """The learning tilt Phi_hat* = -2c A Y Y^T A is negative definite; for the linear (Gaussian)
    case the projection must leave it whole, since Sigma^-1 + Phi_hat* > 0."""
    X, Y = linear_problem
    _, Phi_hat_star, Sigma, _ = analytic_saddle(X, Y)
    s = SingleSiteHMC(Sigma, torch.zeros_like(Sigma), 0.0, identity, True, n_chains=4)
    assert linear_cfg().solver.project_tilt  # the default
    out = s.project_tilt(Phi_hat_star, linear_cfg().solver.tilt_floor)
    assert out.norm() / Phi_hat_star.norm() == pytest.approx(1.0, abs=1e-12)
    torch.testing.assert_close(out, mf.sym(Phi_hat_star), rtol=0, atol=0)  # unchanged


def test_the_linear_solve_is_not_changed_by_the_projection(linear_solution):
    h = linear_solution[0].history
    assert all(r[0] == 0.0 for r in h["tilt_projected_rel"][-5:])


def test_relu_projection_keeps_a_learning_tilt_the_all_active_orthant_admits():
    """A rank-2 negative learning tilt that takes 90 % of the prior precision in its plane:
    lambda_min(M) is far below -lambda_min(Sigma^-1) (the orthant floor would cut it), but
    Sigma^-1 + M = Sigma^-1/2 (I - 0.9 U U^T) Sigma^-1/2 > 0, so the relu rule keeps it whole."""
    X, Y = relu_problem(6, 8, seed=56)
    Sigma = X @ X.T / 8 + 0.1 * torch.eye(6, **F64)
    Sigma_inv = torch.linalg.inv(Sigma)
    w, V = torch.linalg.eigh(Sigma_inv)
    root_inv = (V * w.sqrt()) @ V.T
    U, _ = torch.linalg.qr(Y)
    Mh = -0.9 * root_inv @ U @ U.T @ root_inv
    assert torch.linalg.eigvalsh(Mh).min() < -2 * w.min()
    out = project_tilt(Mh, 0.05, 0.0, torch.relu, True, torch.linalg.cholesky(Sigma), Sigma_inv)
    torch.testing.assert_close(out, mf.sym(Mh), rtol=0, atol=0)


def test_a_solve_ending_unconverged_raises_and_writes_no_final(linear_problem, tmp_path):
    X, Y = linear_problem
    cfg = linear_cfg(n_outer=2, save_int=1, tilt_tripwire_resid=1e-12)
    with pytest.raises(mf.UnconvergedSolveError, match="residual"):
        solve(X, Y, 1, LAMBDA_W, 0.0, identity, 1.0, cfg, log=silent, checkpoint_dir=tmp_path,
              config_hash="h0")  # fmt: skip
    assert not (tmp_path / "final.pt").exists()
    assert list(tmp_path.glob("ckpt_*.pt"))  # kept for inspection


def _resid_history(*rows):
    return {"phihat_fp_resid_stein": [np.array(r) for r in rows], "tilt_projected_rel": [None]}


def test_the_tripwire_reads_the_largest_residual_of_the_last_steps():
    # oscillating around the tripwire (as a relu run did) trips it, though the last step is below
    with pytest.raises(mf.UnconvergedSolveError, match="last 3 outer steps"):
        mf._check_tilt_tripwire(_resid_history([0.1], [0.96], [1.04], [0.89]), 0.9, 3)
    # steps before the last k do not count; a steady unconverged 0.76 passes (not a convergence test)
    mf._check_tilt_tripwire(_resid_history([5.0], [0.76], [0.76], [0.76]), 0.9, 3)
    # fewer steps than k: all of them
    with pytest.raises(mf.UnconvergedSolveError):
        mf._check_tilt_tripwire(_resid_history([0.1, 0.95]), 0.9, 3)


def test_phihat_change_is_nan_after_a_step_from_a_zero_tilt(linear_problem):
    X, Y = linear_problem
    cfg = linear_cfg(n_outer=1, init="lazy", tilt_tripwire_resid=ANY_RESID)
    h = solve(X, Y, 1, LAMBDA_W, 0.0, identity, 1.0, cfg, log=silent).history
    assert np.isnan(h["phihat_change_rel"][0]).all()


def test_the_log_has_no_margin_for_a_bounded_phi():
    X, Y = relu_problem(6, 8, seed=58)
    cfg = MeanFieldConfig(
        beta=1e3, n_chains=16, n_outer=1, warmup_trans=5, n_inner_max=2, log_int=1,
        tilt_tripwire_resid=ANY_RESID,
    )  # fmt: skip
    lines = []
    solve(X, Y, 1, 1e-3, 0.0, torch.tanh, 1.0, cfg, log=lines.append)
    assert any("margin=n/a" in line for line in lines)


def test_a_projection_needs_a_known_activation():
    X, Y = relu_problem(6, 8, seed=57)
    with pytest.raises(ValueError, match="normalisability rule"):
        solve(X, Y, 1, 1e-3, 0.0, torch.nn.functional.softplus, 1.0, MeanFieldConfig(), log=silent)


def assert_the_scale_bound_never_binds(history) -> None:
    """Every outer-gradient run sampled h2/prior far below max_h2_prior_ratio, and no step was
    reverted for sampler health (on the scale bound or anything else)."""
    ratio = max(float(np.max(r)) for r in history["hmc_h2_ratio"])
    print(f"largest h2/prior of an outer-gradient run: {ratio:.3g}")
    assert ratio < 0.1 * MeanFieldConfig().solver.max_h2_prior_ratio
    assert sum(int(np.sum(b)) for b in history["n_backtrack_inner"]) == 0


def test_linear_solve_recovers_analytic_saddle(linear_problem, linear_solution):
    X, Y = linear_problem
    result, seconds = linear_solution
    Phi_star, Phi_hat_star, Sigma, _ = analytic_saddle(X, Y)
    h = result.history

    dist = relative_fro(result.Phi[0], Phi_star)
    dist_hat = relative_fro(result.Phi_hat[0], Phi_hat_star)
    dist_prior = relative_fro(Sigma, Phi_star)
    print(
        f"\nL=1 linear: |Phi-Phi*|/|Phi*|={dist:.3e} (prior {dist_prior:.3e}), "
        f"|Phi_hat-Phi_hat*|/|Phi_hat*|={dist_hat:.3e}, "
        f"fp_stein={h['phihat_fp_resid_stein'][-1][0]:.3e}, "
        f"inner_res={h['inner_residual'][-1][0]:.3e}, {seconds:.1f}s"
    )
    # Outer stationarity, Phi_hat = -2c A Y Y^T A, on the last few steps (not one lucky one).
    assert all(r[0] < 0.1 for r in h["phihat_fp_resid_stein"][-5:])
    # Inner stationarity: the ascent stopped because the residual is below its tolerance.
    assert all(h["inner_converged"][-5:])
    assert h["inner_residual"][-1][0] < max(1e-2, 2.0 * h["hmc_mc_err"][-1][0])
    # The solver moved far from its start and landed on the analytic saddle point; the
    # remaining distance is MC noise (it shrinks as 1/sqrt(n_trans): ~0.022 at n_trans=40,
    # ~0.013 at 100, ~0.009 at 200).
    assert dist < 0.2 * dist_prior
    assert dist < 0.04
    assert_the_scale_bound_never_binds(h)


def test_solve_without_checkpoint_dir_never_writes(linear_problem, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("solve wrote to disk without a checkpoint_dir")

    monkeypatch.setattr(mf, "save_artefact", forbidden)
    monkeypatch.setattr(mf, "write_checkpoint", forbidden)
    X, Y = linear_problem
    cfg = linear_cfg(
        n_outer=2, warmup_trans=10, n_trans=2, n_trans_outer=2, n_inner_max=2,
        tilt_tripwire_resid=ANY_RESID,
    )  # fmt: skip
    result = solve(X, Y, 1, LAMBDA_W, 0.0, identity, 1.0, cfg, log=silent)
    assert len(result.history["outer_step"]) == 2


def test_history_has_one_entry_per_step_and_no_net_keys_without_net(linear_solution):
    result, _ = linear_solution
    h = result.history
    assert set(h) == set(HISTORY_KEYS) - {"align_net", "dist_net"}
    assert all(len(h[k]) == 40 for k in h)
    assert h["outer_step"] == list(range(40))
    assert h["train_mse_net"] == [None] * 40


# ----------------------------------------------------------------------------- ReLU L=2 problems
def relu_problem(P, D_in, seed):
    X = torch.randn(P, D_in, generator=seeded(seed), **F64)
    Y = torch.randn(P, 2, generator=seeded(seed + 1), **F64)
    return X, Y


def small_net(D_in, L=2, N=32):
    return make_net(NetConfig(N=N, L=L, g=1.0, phi="relu", seed=3), D_in, 2, "cpu", torch.float64)


def test_smoke_relu_two_layers_with_net_is_fast_and_complete():
    X, Y = relu_problem(16, 8, seed=20)
    net = small_net(8)
    cfg = MeanFieldConfig(
        beta=1e3, n_chains=128, n_outer=3, warmup_trans=20, log_int=1, seed=1,
        tilt_tripwire_resid=ANY_RESID,
    )  # fmt: skip
    lines = []
    start = time.perf_counter()
    result = solve(X, Y, 2, 1e-3, 1e-3, torch.relu, 1.0, cfg, net=net, log=lines.append)
    seconds = time.perf_counter() - start
    print(f"\nsmoke L=2 relu P=16: {seconds:.1f}s")
    assert seconds < 60

    h = result.history
    assert set(h) == set(HISTORY_KEYS)
    assert all(len(h[k]) == 3 for k in HISTORY_KEYS)
    for key in ("align_net", "dist_net"):
        assert all(v is not None and v.shape == (2,) for v in h[key])
    assert all(v is not None for v in h["train_mse_net"])
    assert result.Phi.shape == result.Phi_hat.shape == (2, 16, 16)
    assert torch.isfinite(result.Phi).all() and torch.isfinite(result.Phi_hat).all()
    assert len(result.samplers) == 2
    assert any(line.startswith("[0000]") for line in lines)


def test_empirical_kernels_match_definition():
    X, _ = relu_problem(6, 8, seed=30)
    net = small_net(8, L=3, N=5)
    Phi, H = empirical_kernels(net, X)
    assert H.shape == (6, 3, 5) and Phi.shape == (3, 6, 6)
    for layer in range(3):  # phi(h) on every layer, the last (read out) one included
        psi = torch.relu(H[:, layer])
        torch.testing.assert_close(Phi[layer], psi @ psi.T / 5)


def test_readout_activities_are_phi_of_h_on_every_layer():
    H = torch.randn(6, 3, 5, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    psis = readout_activities(H, torch.tanh)
    assert len(psis) == 3 and all(psi.shape == (6, 5) for psi in psis)
    for layer in range(3):
        assert torch.equal(psis[layer], torch.tanh(H[:, layer]))


def test_empirical_kernels_are_built_from_the_readout_activities():
    X, _ = relu_problem(6, 8, seed=31)
    net = small_net(8, L=2, N=5)
    Phi, H = empirical_kernels(net, X)
    psis = readout_activities(H, net.phi)
    assert torch.equal(Phi, torch.stack([psi @ psi.T / 5 for psi in psis]))


def test_the_net_output_is_the_linear_readout_of_the_last_kernel_activities():
    X, _ = relu_problem(6, 8, seed=32)
    net = small_net(8, L=2, N=5)
    H, Z = net(X)
    psi_L = readout_activities(H, net.phi)[-1]
    torch.testing.assert_close(Z, psi_L @ net.V / 5)


def test_init_net_requires_a_net():
    X, Y = relu_problem(8, 8, seed=40)
    cfg = MeanFieldConfig(init="net", n_chains=16, n_outer=1)
    with pytest.raises(ValueError, match="init='net'"):
        solve(X, Y, 2, 1e-3, 1e-3, torch.relu, 1.0, cfg, log=silent)


def test_init_net_runs():
    X, Y = relu_problem(8, 8, seed=41)
    cfg = MeanFieldConfig(
        beta=1e3, init="net", n_chains=32, n_outer=1, warmup_trans=5, n_inner_max=2, seed=0,
        tilt_tripwire_resid=ANY_RESID,
    )  # fmt: skip
    result = solve(X, Y, 2, 1e-3, 1e-3, torch.relu, 1.0, cfg, net=small_net(8), log=silent)
    assert torch.isfinite(result.Phi).all()


def test_checkpoint_dir_needs_a_hash(tmp_path):
    X, Y = relu_problem(8, 8, seed=42)
    with pytest.raises(ValueError, match="config_hash"):
        solve(X, Y, 2, 1e-3, 1e-3, torch.relu, 1.0, MeanFieldConfig(), checkpoint_dir=tmp_path)


# ----------------------------------------------------------------------------- resume
class Interrupted(Exception):
    pass


def resume_cfg(sample_dtype="float64"):
    # The tilt and prefactor are pinned: these solver tests were tuned (importance-sampling ESS
    # margins, step-size reverts) at tilt 1e-3 and prefactor 1/2, not at the network's 2 beta
    # lambda_r and lambda_w / 2.
    return MeanFieldConfig(
        beta=1e3,
        rate_tilt=1e-3,
        learning_prefactor=0.5,
        n_chains=64,
        n_outer=4,
        save_int=2,
        warmup_trans=20,
        n_inner_max=5,
        sample_dtype=sample_dtype,
        log_int=1,
        seed=5,
        tilt_tripwire_resid=ANY_RESID,
    )


def interrupt_on_call(monkeypatch, n):
    """Make the n-th outer step raise Interrupted; returns a function that undoes the patch."""
    real_outer_step = mf._outer_step
    calls = []

    def outer_step(*args, **kwargs):
        calls.append(1)
        if len(calls) == n:
            raise Interrupted
        return real_outer_step(*args, **kwargs)

    monkeypatch.setattr(mf, "_outer_step", outer_step)
    return lambda: monkeypatch.setattr(mf, "_outer_step", real_outer_step)


def solve_relu(X, Y, cfg, net=None, **kwargs):
    return solve(X, Y, 2, 1e-3, 1e-3, torch.relu, 1.0, cfg, net=net, log=silent, **kwargs)


# The chain state is checkpointed in h-coordinates (float64) and re-whitened on load.  For
# float32 chains that round trip is exact, so a resumed run is bitwise identical; for float64
# chains it costs ~1e-16 per entry, which grows to ~1e-14 relative over the remaining steps.
# 1e-10 is still far below what a broken restore (e.g. of the generator) produces.
@pytest.mark.parametrize("sample_dtype", ["float32", "float64"])
def test_resume_after_interruption_matches_uninterrupted_run(tmp_path, monkeypatch, sample_dtype):
    X, Y = relu_problem(8, 8, seed=50)
    net = small_net(8)
    cfg = resume_cfg(sample_dtype)
    full = solve_relu(X, Y, cfg, net)

    restore = interrupt_on_call(monkeypatch, 3)
    with pytest.raises(Interrupted):
        solve_relu(X, Y, cfg, net, checkpoint_dir=tmp_path, config_hash="h0")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ckpt_00002.pt"]

    restore()
    resumed = solve_relu(X, Y, cfg, net, checkpoint_dir=tmp_path, config_hash="h0")

    # final.pt holds the same payload as the last checkpoint, which is therefore removed.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["final.pt"]
    assert resumed.history["outer_step"] == full.history["outer_step"] == [0, 1, 2, 3]
    assert relative_fro(resumed.Phi, full.Phi) < 1e-10
    assert relative_fro(resumed.Phi_hat, full.Phi_hat) < 1e-10
    for key in ("phihat_fp_resid_stein", "hmc_accept", "hmc_eps", "eta_phihat", "align_net"):
        for a, b in zip(resumed.history[key], full.history[key]):
            assert np.allclose(a, b, rtol=1e-10, atol=0, equal_nan=True), key


@pytest.mark.parametrize("net_when_written", [False, True])
def test_resume_refuses_a_checkpoint_with_the_other_net_setting(
    tmp_path, monkeypatch, net_when_written
):
    X, Y = relu_problem(8, 8, seed=51)
    cfg = resume_cfg()
    nets = {True: small_net(8), False: None}
    restore = interrupt_on_call(monkeypatch, 3)
    with pytest.raises(Interrupted):
        solve_relu(X, Y, cfg, nets[net_when_written], checkpoint_dir=tmp_path, config_hash="h0")
    restore()
    with pytest.raises(ValueError, match="ckpt_00002.pt"):
        solve_relu(X, Y, cfg, nets[not net_when_written], checkpoint_dir=tmp_path, config_hash="h0")


# ----------------------------------------------------------------------------- run / load
def small_config(beta=1e3):
    return Config(
        data=DataConfig(P=12, D_in=8),
        net=NetConfig(N=16, L=2, phi="relu"),
        train=TrainConfig(weight_lambda=1e-3, rate_lambda=1e-3),
        meanfield=MeanFieldConfig(
            n_points=10,
            beta=beta,
            n_chains=32,
            n_outer=2,
            save_int=1,
            warmup_trans=10,
            n_inner_max=3,
            seed=0,
            tilt_tripwire_resid=ANY_RESID,
        ),
        device="cpu",
    )


def small_dataset(cfg):
    x = torch.randn(cfg.data.P, cfg.data.D_in, generator=seeded(60), **F64)
    y = torch.randn(cfg.data.P, cfg.data.d, generator=seeded(61), **F64)
    return Dataset(x=x, y=y, Q=None, period=None)


@pytest.fixture
def cached_run(tmp_path):
    cfg = small_config()
    dataset = small_dataset(cfg)
    net = make_net(cfg.net, cfg.data.D_in, cfg.data.d, "cpu", torch.float64)
    result = run_meanfield(cfg, dataset, net, tmp_path, log=silent)
    return cfg, dataset, net, tmp_path, result


def test_run_meanfield_second_call_returns_cache_without_solving(cached_run, monkeypatch):
    cfg, dataset, net, cache_dir, first = cached_run

    def forbidden(*args, **kwargs):
        raise AssertionError("an outer step ran although final.pt exists")

    monkeypatch.setattr(mf, "_outer_step", forbidden)
    lines = []
    second = run_meanfield(cfg, dataset, net, cache_dir, log=lines.append)
    final = cache_dir / "meanfield" / "final.pt"
    assert lines == [f"loaded finished solve from {final} (2 outer steps)"]
    assert sorted(p.name for p in final.parent.iterdir()) == ["final.pt"]
    torch.testing.assert_close(second.Phi, first.Phi)
    torch.testing.assert_close(second.Phi_hat, first.Phi_hat)
    assert second.history["outer_step"] == first.history["outer_step"]
    assert len(second.samplers) == cfg.net.L


def test_load_meanfield_returns_numpy_history(cached_run):
    cfg, _, _, cache_dir, first = cached_run
    Phi, Phi_hat, history = load_meanfield(cache_dir, cfg, "cpu")
    torch.testing.assert_close(Phi, first.Phi)
    torch.testing.assert_close(Phi_hat, first.Phi_hat)
    assert set(history) == set(HISTORY_KEYS)
    for key in HISTORY_KEYS:
        assert len(history[key]) == len(first.history[key]) == 2
    for a, b in zip(history["hmc_accept"], first.history["hmc_accept"]):
        assert type(a) is type(b) and a.dtype == b.dtype and (a == b).all()
    assert history["step_clipped"][0].dtype == bool
    assert history["n_backtrack_inner"][0].dtype == first.history["n_backtrack_inner"][0].dtype


def test_load_meanfield_is_stale_when_beta_changes(cached_run):
    _, _, _, cache_dir, _ = cached_run
    with pytest.raises(StaleCacheError):
        load_meanfield(cache_dir, small_config(beta=2e3), "cpu")


def test_load_meanfield_rejects_a_result_from_another_model_version(tmp_path, monkeypatch):
    cfg = small_config()
    net = make_net(cfg.net, cfg.data.D_in, cfg.data.d, "cpu", torch.float64)
    with monkeypatch.context() as old:
        old.setattr(fct.config, "MODEL_VERSION", fct.config.MODEL_VERSION - 1)
        run_meanfield(cfg, small_dataset(cfg), net, tmp_path, log=silent)
    with pytest.raises(StaleCacheError):
        load_meanfield(tmp_path, cfg, "cpu")


def test_load_meanfield_rejects_a_result_from_another_loss_version(tmp_path, monkeypatch):
    cfg = small_config()
    net = make_net(cfg.net, cfg.data.D_in, cfg.data.d, "cpu", torch.float64)
    with monkeypatch.context() as old:
        old.setattr(fct.config, "LOSS_VERSION", fct.config.LOSS_VERSION - 1)
        run_meanfield(cfg, small_dataset(cfg), net, tmp_path, log=silent)
    with pytest.raises(StaleCacheError):
        load_meanfield(tmp_path, cfg, "cpu")


def test_load_meanfield_without_result_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_meanfield(tmp_path, small_config(), "cpu")


def test_run_meanfield_rejects_more_points_than_data(tmp_path):
    cfg = small_config()
    cfg = cfg.model_copy(update={"meanfield": cfg.meanfield.model_copy(update={"n_points": 13})})
    dataset = small_dataset(cfg)
    net = make_net(cfg.net, cfg.data.D_in, cfg.data.d, "cpu", torch.float64)
    with pytest.raises(ValueError, match="n_points"):
        run_meanfield(cfg, dataset, net, tmp_path, log=silent)


def test_net_kernels_are_the_empirical_kernels_on_the_solver_inputs():
    cfg = small_config()
    dataset = small_dataset(cfg)
    # A float32 net: net_kernels must evaluate it in its own dtype, then return float64.
    net = make_net(cfg.net, cfg.data.D_in, cfg.data.d, "cpu", torch.float32)
    K = net_kernels(net, dataset, cfg)
    expected, _ = empirical_kernels(net, dataset.x[:10].to(torch.float32))
    assert K.dtype == torch.float64 and K.shape == (2, 10, 10)
    torch.testing.assert_close(K, expected.to(torch.float64), rtol=0, atol=0)


def test_net_kernels_reject_more_points_than_data():
    cfg = small_config()
    dataset = small_dataset(cfg)
    small = Dataset(x=dataset.x[:5], y=dataset.y[:5], Q=None, period=None)
    net = make_net(cfg.net, cfg.data.D_in, cfg.data.d, "cpu", torch.float64)
    with pytest.raises(ValueError, match="n_points"):
        net_kernels(net, small, cfg)


@pytest.mark.parametrize(
    "rate_lambda, tilt, expected",
    [(1e-3, None, 2.0), (0.0, None, 0.0), (1e-3, 1e-4, 1e-4)],
    ids=["default-is-2-beta-rate_lambda", "no-penalty", "explicit-override"],
)
def test_run_meanfield_tilts_with_two_beta_rate_lambda_unless_overridden(
    tmp_path, monkeypatch, rate_lambda, tilt, expected
):
    cfg = small_config()
    cfg = cfg.model_copy(
        update={
            "train": cfg.train.model_copy(update={"rate_lambda": rate_lambda}),
            "meanfield": cfg.meanfield.model_copy(update={"rate_tilt": tilt}),
        }
    )
    used = []
    make_problem = mf._make_problem

    def spy(*args, **kwargs):
        prob = make_problem(*args, **kwargs)
        used.append(prob.rate_tilt)
        return prob

    monkeypatch.setattr(mf, "_make_problem", spy)
    net = make_net(cfg.net, cfg.data.D_in, cfg.data.d, "cpu", torch.float64)
    run_meanfield(cfg, small_dataset(cfg), net, tmp_path, log=silent)
    assert used == [expected]


def test_load_meanfield_accepts_an_unset_rate_tilt(cached_run):
    cfg, _, _, cache_dir, _ = cached_run
    assert cfg.meanfield.rate_tilt is None
    Phi, _, _ = load_meanfield(cache_dir, cfg, "cpu")
    assert Phi.shape[-1] == cfg.meanfield.n_points


# ----------------------------------------------------------------------------- importance sampler
HMC_ONLY_KEYS = ("hmc_accept", "hmc_eps", "hmc_div", "hmc_reset", "hmc_max_abs_dH", "hmc_rho1")


def test_importance_solve_recovers_analytic_saddle(linear_problem):
    """The same L=1 Gaussian problem as the HMC test, with fresh prior draws instead of chains."""
    X, Y = linear_problem
    cfg = linear_cfg(sampler="importance", n_importance_samples=4096, n_trans=25, n_trans_outer=50)
    result = solve(X, Y, 1, LAMBDA_W, 0.0, identity, 1.0, cfg, log=silent)
    Phi_star, _, Sigma, _ = analytic_saddle(X, Y)
    h = result.history
    dist = relative_fro(result.Phi[0], Phi_star)
    print(f"\nL=1 linear, importance: |Phi-Phi*|/|Phi*|={dist:.3e}")
    assert all(r[0] < 0.1 for r in h["phihat_fp_resid_stein"][-5:])
    assert all(h["inner_converged"][-5:])
    assert dist < 0.2 * relative_fro(Sigma, Phi_star)
    assert dist < 0.04
    assert all(np.isnan(v).all() for k in HMC_ONLY_KEYS for v in h[k])
    assert all(0.1 < v[0] <= 1 for v in h["hmc_ess_frac"])
    assert all(0 < v[0] < 1e-3 for v in h["is_max_weight"])
    assert_the_scale_bound_never_binds(h)


def test_hmc_history_has_no_importance_diagnostics(linear_solution):
    result, _ = linear_solution
    assert all(np.isnan(v).all() for v in result.history["is_max_weight"])


def importance_cfg(**overrides):
    base = {"sampler": "importance", "n_importance_samples": 256, "sample_dtype": "float64"}
    cfg = resume_cfg()
    return cfg.model_copy(update={"solver": cfg.solver.model_copy(update={**base, **overrides})})


def test_importance_solve_with_net_runs_and_resumes_exactly(tmp_path, monkeypatch):
    X, Y = relu_problem(8, 8, seed=52)
    net = small_net(8)
    cfg = importance_cfg()
    full = solve_relu(X, Y, cfg, net)
    assert set(full.history) == set(HISTORY_KEYS)
    assert all(isinstance(s, ImportanceSampler) for s in full.samplers)

    restore = interrupt_on_call(monkeypatch, 3)
    with pytest.raises(Interrupted):
        solve_relu(X, Y, cfg, net, checkpoint_dir=tmp_path, config_hash="h0")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ckpt_00002.pt"]
    restore()
    resumed = solve_relu(X, Y, cfg, net, checkpoint_dir=tmp_path, config_hash="h0")

    # No chain state to round-trip: the only state is the generator, so the resumed run is
    # the uninterrupted one exactly.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["final.pt"]
    assert torch.equal(resumed.Phi, full.Phi) and torch.equal(resumed.Phi_hat, full.Phi_hat)
    for key in ("phihat_fp_resid_stein", "hmc_ess_frac", "is_max_weight", "align_net"):
        for a, b in zip(resumed.history[key], full.history[key]):
            assert np.array_equal(a, b), key


def test_an_hmc_checkpoint_does_not_load_into_importance_samplers(tmp_path, monkeypatch):
    """Belt and braces under the cache hash, which already differs between the two samplers."""
    X, Y = relu_problem(8, 8, seed=53)
    restore = interrupt_on_call(monkeypatch, 3)
    with pytest.raises(Interrupted):
        solve_relu(X, Y, resume_cfg(), checkpoint_dir=tmp_path, config_hash="h0")
    restore()
    with pytest.raises(ValueError, match="importance"):
        solve_relu(X, Y, importance_cfg(), checkpoint_dir=tmp_path, config_hash="h0")


def test_an_ess_drop_relative_to_the_point_before_is_unhealthy():
    """The gate is step-relative, like max_h2_growth: the ESS of the prior proposal depends on
    the point, so only a fall relative to the accepted point before the step blames the step."""
    cfg = importance_cfg().solver
    nan = float("nan")

    def healthy(ess_after, ess_before):
        diag = {"h2_mean": 1.0, "prior_h2": 1.0, "ess_frac": ess_after, "psi_psi_rel_mc_err": 0.01}
        return mf._healthy(diag, 1.0, ess_before, cfg.n_trans, nan, cfg)

    assert healthy(0.5, 0.5)
    assert healthy(1.01 * cfg.min_ess_drop * 0.5, 0.5)
    assert not healthy(0.99 * cfg.min_ess_drop * 0.5, 0.5)
    assert healthy(1e-4, 1e-4)  # low but not lower than before: the floor's job, not the gate's


def extreme_tilt_cfg(**overrides):
    # gaussian init and a tolerance never met: the lazy init would make Phi_hat = 0 the inner
    # solution, and the loop would stop before validating any step.
    base = {
        "init": "gaussian", "eta_phihat": 1e4, "eta_max": 1e4, "eta_max_phihat": 1e4,
        "inner_tol": 1e-12, "inner_tol_mc": 1e-12, "n_inner_max": 5, "n_outer": 1,
        "max_backtrack": 50,
    }  # fmt: skip
    return importance_cfg(**{**base, **overrides})


def test_a_step_that_collapses_the_weights_is_reverted():
    """A huge inner step size drives the tilt where the prior proposal's weights collapse
    (ESS falls by far more than min_ess_drop); the solver reverts and halves the step.  The
    absolute floor is switched off so the step-relative gate is tested on its own."""
    X, Y = relu_problem(8, 8, seed=54)
    result = solve_relu(X, Y, extreme_tilt_cfg(min_ess_frac=0.0))
    assert result.history["n_backtrack_inner"][0].sum() > 0


def test_an_accepted_point_below_the_ess_floor_fails_loudly_even_with_one_inner_step(tmp_path):
    """One inner step: its run is at the accepted starting point and the step it takes is never
    validated by the inner loop; the outer-gradient run at the collapsed tilt must raise rather
    than let the solve finish."""
    X, Y = relu_problem(8, 8, seed=54)
    cfg = extreme_tilt_cfg(n_inner_max=1)
    # tanh: a bounded phi needs no tilt projection, so the huge step reaches a tilt the prior
    # proposal cannot represent (the relu certificate would keep the step small enough).
    with pytest.raises(mf.InadequateSamplerError, match="outer step 0"):
        solve(X, Y, 2, 1e-3, 1e-3, torch.tanh, 1.0, cfg, log=silent, checkpoint_dir=tmp_path,
              config_hash="h0")  # fmt: skip
    assert not (tmp_path / "final.pt").exists()


@pytest.mark.parametrize("n_points", [48, 100])
def test_importance_sampling_at_realistic_size_fails_loudly_instead_of_freezing(tmp_path, n_points):
    """The reviewer's case: default solver settings, relu, L=2, beta=1e3 (seed 0).  At P=100 the
    ESS fraction is below the floor already at the initial point (layer 0, ~5e-3); at P=48 the
    initial point passes (layer 1, ~0.013) and the floor is hit at the first outer-gradient run
    (layer 1, ~1e-3): one case per accept site.  Either way the solve must say the sampler is
    inadequate, not write a frozen final.pt."""
    cfg = small_config()
    cfg = cfg.model_copy(
        update={
            "data": cfg.data.model_copy(update={"P": n_points}),
            "meanfield": MeanFieldConfig(n_points=n_points, beta=1e3, sampler="importance"),
        }
    )
    dataset = small_dataset(cfg)
    net = make_net(cfg.net, cfg.data.D_in, cfg.data.d, "cpu", torch.float64)
    with pytest.raises(mf.InadequateSamplerError, match="inadequate.*sampler='hmc'"):
        run_meanfield(cfg, dataset, net, tmp_path, log=silent)
    assert not (tmp_path / "meanfield" / "final.pt").exists()


def test_init_from_net_matches_the_tilt_gradient_of_an_hmc_sampler():
    """init='net' evaluates the tilt gradient directly; the reference is the gradient of a
    SingleSiteHMC holding the same density."""
    X, Y = relu_problem(8, 8, seed=55)
    net = small_net(8, L=3)
    cfg = MeanFieldConfig(beta=1e3, init="net", n_chains=8)
    prob = mf._make_problem(X, Y, 3, 1e-3, 1e-2, torch.relu, 1.0, cfg)
    net_Phi, net_H = empirical_kernels(net, prob.X)
    L, P = 3, prob.P
    Phi, Phi_hat = torch.zeros(L, P, P, **F64), torch.zeros(L, P, P, **F64)
    mf._init_from_net(prob, Phi, Phi_hat, net_Phi, net_H)
    for l in range(L - 1):
        ref = SingleSiteHMC(
            mf._prior_cov(prob, l + 1, Phi), Phi_hat[l + 1], prob.rate_tilt, torch.relu,
            True, n_chains=1, sample_dtype=torch.float64, generator=seeded(0),
        )  # fmt: skip
        h = net_H[:, l + 1].T
        _, u = ref.tilt_and_grad(h)
        expected = mf.sym(torch.linalg.solve(Phi[l], h.T @ u / h.shape[0]))
        assert torch.equal(Phi_hat[l], expected)


def test_run_meanfield_records_the_full_meanfield_config(cached_run):
    cfg, _, _, cache_dir, _ = cached_run
    payload = torch.load(cache_dir / "meanfield" / "final.pt", weights_only=True)
    assert payload["meanfield_config"] == cfg.meanfield.model_dump()


def with_solver(cfg, **solver):
    mf_cfg = cfg.meanfield.model_copy(
        update={"solver": cfg.meanfield.solver.model_copy(update=solver)}
    )
    return cfg.model_copy(update={"meanfield": mf_cfg})


def test_load_meanfield_is_stale_when_the_sampler_changes(cached_run):
    cfg, _, _, cache_dir, _ = cached_run
    with pytest.raises(StaleCacheError):
        load_meanfield(cache_dir, with_solver(cfg, sampler="importance"), "cpu")


def test_load_meanfield_ignores_importance_settings_of_an_hmc_result(cached_run):
    cfg, _, _, cache_dir, first = cached_run
    Phi, _, _ = load_meanfield(cache_dir, with_solver(cfg, n_importance_samples=7), "cpu")
    torch.testing.assert_close(Phi, first.Phi)


def test_run_meanfield_refuses_a_spatial_config_before_writing_anything(tmp_path):
    cfg = small_config().model_copy(update={"spatial": SpatialConfig()})
    net = make_net(cfg.net, cfg.data.D_in, cfg.data.d, "cpu", torch.float64)
    with pytest.raises(SpatialPriorUnsupportedError, match="no spatial prior"):
        run_meanfield(cfg, small_dataset(cfg), net, tmp_path, log=silent)
    assert not (tmp_path / "meanfield").exists()
