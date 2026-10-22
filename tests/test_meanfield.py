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

import fct.meanfield as mf
from fct.cache import StaleCacheError
from fct.config import Config, DataConfig, MeanFieldConfig, NetConfig, TrainConfig
from fct.data import Dataset
from fct.hmc import relative_fro
from fct.meanfield import (
    HISTORY_KEYS,
    empirical_kernels,
    load_meanfield,
    net_kernels,
    run_meanfield,
    solve,
)
from fct.model import make_net

F64 = {"device": "cpu", "dtype": torch.float64}


def seeded(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def silent(_msg):
    pass


# ----------------------------------------------------------------------------- analytic L=1
LAMBDA_W, ALPHA, PREFACTOR = 2.0, 0.5, 0.5
P_LIN, D_LIN = 8, 8


def identity(h):
    return h


@pytest.fixture(scope="module")
def linear_problem():
    """X with Phi0 = X X^T / D_in = U diag(s) U^T, s in [0.8, 1.2] (well conditioned), and
    targets Y scaled so that the saddle point moves Phi ~20% away from the prior yet keeps
    Sigma^-1 + Phi_hat comfortably positive definite.  (Much larger Y also admits a second,
    unphysical fixed point with Sigma^-1 + Phi_hat indefinite.)"""
    U, _ = torch.linalg.qr(torch.randn(P_LIN, P_LIN, generator=seeded(10), **F64))
    s = torch.linspace(0.8, 1.2, P_LIN, **F64)
    X = (U * s.sqrt()) * D_LIN**0.5  # (P, D_in) with D_in = P
    Y = 0.4 * torch.randn(P_LIN, 2, generator=seeded(11), **F64)
    return X, Y


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


def test_solve_without_checkpoint_dir_never_writes(linear_problem, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("solve wrote to disk without a checkpoint_dir")

    monkeypatch.setattr(mf, "save_artefact", forbidden)
    monkeypatch.setattr(mf, "write_checkpoint", forbidden)
    X, Y = linear_problem
    cfg = linear_cfg(n_outer=2, warmup_trans=10, n_trans=2, n_trans_outer=2, n_inner_max=2)
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
    cfg = MeanFieldConfig(beta=1e3, n_chains=128, n_outer=3, warmup_trans=20, log_int=1, seed=1)
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
    Phi, H = empirical_kernels(net, X, "h")
    assert H.shape == (6, 3, 5) and Phi.shape == (3, 6, 6)
    torch.testing.assert_close(Phi[0], torch.relu(H[:, 0]) @ torch.relu(H[:, 0]).T / 5)
    torch.testing.assert_close(Phi[2], H[:, 2] @ H[:, 2].T / 5)
    Phi_phi, _ = empirical_kernels(net, X, "phi")
    torch.testing.assert_close(Phi_phi[2], torch.relu(H[:, 2]) @ torch.relu(H[:, 2]).T / 5)


def test_init_net_requires_a_net():
    X, Y = relu_problem(8, 8, seed=40)
    cfg = MeanFieldConfig(init="net", n_chains=16, n_outer=1)
    with pytest.raises(ValueError, match="init='net'"):
        solve(X, Y, 2, 1e-3, 1e-3, torch.relu, 1.0, cfg, log=silent)


def test_init_net_runs():
    X, Y = relu_problem(8, 8, seed=41)
    cfg = MeanFieldConfig(
        beta=1e3, init="net", n_chains=32, n_outer=1, warmup_trans=5, n_inner_max=2, seed=0
    )
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
    return MeanFieldConfig(
        beta=1e3,
        n_chains=64,
        n_outer=4,
        save_int=2,
        warmup_trans=20,
        n_inner_max=5,
        sample_dtype=sample_dtype,
        log_int=1,
        seed=5,
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


@pytest.mark.parametrize("readout", ["h", "phi"])
def test_net_kernels_are_the_empirical_kernels_on_the_solver_inputs(readout):
    cfg = small_config()
    mf_cfg = MeanFieldConfig(**{**cfg.meanfield.model_dump(), "readout": readout})
    cfg = Config(**{**cfg.model_dump(), "meanfield": mf_cfg.model_dump()})
    dataset = small_dataset(cfg)
    # A float32 net: net_kernels must evaluate it in its own dtype, then return float64.
    net = make_net(cfg.net, cfg.data.D_in, cfg.data.d, "cpu", torch.float32)
    K = net_kernels(net, dataset, cfg)
    expected, _ = empirical_kernels(net, dataset.x[:10].to(torch.float32), readout)
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
    [(1e-3, None, 1e-3), (0.0, None, 0.0), (1e-3, 1e-4, 1e-4)],
    ids=["default-is-rate_lambda", "no-penalty", "explicit-override"],
)
def test_run_meanfield_tilts_with_rate_lambda_unless_overridden(
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
