"""Mean-field saddle-point solver: Algorithm 1 of arXiv:2502.07998 with persistent-HMC
single-site expectations, plus checkpoint / resume of the outer loop.

The solver finds  min_Φ max_Φ̂ S(Φ, Φ̂)  for the action

    S = -½ Σ_l Tr Φ_l Φ̂_l + c Tr[Yᵀ(αI + Φ_L)⁻¹Y] - Σ_l ln Z_l,

where Z_l normalises the single-site density of layer l (see `fct.hmc`): prior
N(0, Σ_l), Σ_l = prior_scale · g² Φ_{l-1} (Σ_0 = prior_scale · Φ0, Φ0 = XXᵀ/D_in), tilted by
exp(-½ ψᵀΦ̂_l ψ - ½ λ_r φᵀφ).  ln Z is never estimated: every derivative of S is an
expectation under that density, which the per-layer `SingleSiteHMC` chains provide:

    ∂S/∂Φ̂_l = ½(⟨ψψᵀ⟩_l - Φ_l)
    ∂S/∂Φ_l = -½ Φ̂_l + ½ sym(⟨u hᵀ⟩_{l+1} Φ_l⁻¹)        (Stein form of Eq. 31b)
    ∂S/∂Φ_L = -½ Φ̂_L - c A Y Yᵀ A,  A = (αI + Φ_L)⁻¹      (Eq. 31c)

Mapping of the network's (β, λ_w, λ_r) onto the action (overridable in MeanFieldConfig):
prior_scale = 1/λ_w, α = λ_w P D_out / β, c = learning_prefactor = ½.  The rate tilt λ_r
multiplies φᵀφ summed over the P solved points, a per-sample coefficient; `solve` takes it as
`lambda_r`.  The trained network's rate penalty is summed over its samples, so its per-sample
coefficient is exactly train.rate_lambda, which `run_meanfield` passes as `lambda_r`;
`MeanFieldConfig.rate_tilt`, when set, overrides it.

Layers: `solve` is the computation (it touches disk only when given a checkpoint_dir),
`run_meanfield` / `load_meanfield` are the cache-directory I/O.  The outer loop mutates one
`_SolverState`; everything needed to continue it bit-for-bit (up to the float round trip of
the chain state) is in that object, which is what makes checkpoint / resume straightforward.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fct.activations import Activation, get_phi
from fct.cache import (
    cache_paths,
    latest_checkpoint,
    load_artefact,
    save_artefact,
    write_checkpoint,
)
from fct.config import Config, MeanFieldConfig, resolve_dtype
from fct.data import Dataset
from fct.hmc import SingleSiteHMC, kernel_alignment, project_pd, relative_fro, sym
from fct.model import MLP

F64 = torch.float64
CKPT_PREFIX, CKPT_WIDTH, FINAL_NAME = "ckpt_", 5, "final.pt"

# One entry per outer step for every key; plots read these names.  NET_KEYS are present
# only when `solve` is given a net: their entries are per-layer arrays, and a list of None
# in their place would break code that stacks them.  `train_mse_net` is a scalar per step and
# stays a per-step None without a net (None is weights_only-loadable, and
# `draw.fig_solver_history` treats an all-None column as absent).
HISTORY_KEYS = (
    "outer_step", "time", "inner_steps", "inner_residual", "inner_grad_rel", "inner_converged",
    "step_clipped", "eta_phi", "eta_phihat", "n_backtrack_inner", "n_backtrack_outer",
    "outer_grad_rel", "phihat_fp_resid_stein", "phihat_fp_resid_direct", "stein_vs_direct",
    "trace_term", "learning_term", "train_mse_theory", "train_mse_net",
    "phi_lambda_min", "n_pd_clamped", "norm_margin",
    "hmc_accept", "hmc_eps", "hmc_div", "hmc_reset", "hmc_max_abs_dH", "hmc_rho1", "hmc_ess_frac",
    "hmc_mc_err", "hmc_stein_mismatch", "hmc_h2_ratio",
    "align_net", "dist_net", "phi_change_rel", "phihat_change_rel",
)  # fmt: skip
NET_KEYS = ("align_net", "dist_net")
# Keys whose per-step entry is a per-layer numpy array (saved as a list, restored on load).
_ARRAY_KEYS = frozenset(HISTORY_KEYS) - {
    "outer_step", "time", "inner_steps", "inner_converged", "eta_phi", "n_backtrack_outer",
    "trace_term", "learning_term", "train_mse_theory", "train_mse_net",
}  # fmt: skip

History = dict[str, list]
Log = Callable[[str], Any]


# ----------------------------------------------------------------------------- public results
@torch.no_grad()
def empirical_kernels(net: MLP, X: torch.Tensor, readout: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Φ_l = ψ(h^l) ψ(h^l)ᵀ / N from the trained network at the solver's inputs, and the
    raw pre-activations (P, L, N) (the N neurons are samples from the net's own
    single-site density, which is what the theory predicts)."""
    H, _ = net(X)  # (P, L, N)
    L, N = H.shape[1], H.shape[2]
    psis = [net.phi(H[:, l]) if (l < L - 1 or readout == "phi") else H[:, l] for l in range(L)]
    return torch.stack([psi @ psi.T / N for psi in psis]), H


def _solver_inputs(dataset: Dataset, cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    """(X, Y) = the first cfg.meanfield.n_points training points, the solver's inputs."""
    n, P = cfg.meanfield.n_points, dataset.x.shape[0]
    if n > P:
        raise ValueError(f"meanfield.n_points={n} exceeds the dataset size P={P}")
    return dataset.x[:n], dataset.y[:n]


def net_kernels(net: MLP, dataset: Dataset, cfg: Config) -> torch.Tensor:
    """The trained net's empirical kernels (L, P, P), float64, on exactly the solver's inputs
    and readout, so they are directly comparable with the solver's Φ."""
    X, _ = _solver_inputs(dataset, cfg)
    Phi, _ = empirical_kernels(net, X.to(net.dtype), cfg.meanfield.readout)
    return Phi.to(F64)


@dataclass
class SolveResult:
    Phi: torch.Tensor  # (L, P, P)
    Phi_hat: torch.Tensor  # (L, P, P)
    history: History  # HISTORY_KEYS -> one entry per outer step
    samplers: list[SingleSiteHMC]  # live chains at the end of the run, for inspection


# ----------------------------------------------------------------------------- problem constants
@dataclass(frozen=True)
class _Problem:
    """Everything the iteration reads but never changes."""

    X: torch.Tensor  # (P, D_in) float64
    Y: torch.Tensor  # (P, D_out) float64
    L: int
    phi: Activation
    g: float
    cfg: MeanFieldConfig
    Phi0: torch.Tensor  # X Xᵀ / D_in
    eye: torch.Tensor
    alpha: float
    prior_scale: float
    rate_tilt: float
    lp: float  # learning_prefactor c
    psi_is_phi: tuple[bool, ...]
    sample_dtype: torch.dtype

    @property
    def P(self) -> int:
        return self.X.shape[0]

    @property
    def dev(self) -> torch.device:
        return self.X.device


def _make_problem(
    X: torch.Tensor,
    Y: torch.Tensor,
    L: int,
    lambda_w: float,
    lambda_r: float,
    phi: Activation,
    g: float,
    cfg: MeanFieldConfig,
) -> _Problem:
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError(f"X (P, D_in) and Y (P, D_out) must share P; got {X.shape}, {Y.shape}")
    if L < 1:
        raise ValueError(f"L must be >= 1, got {L}")
    X, Y = X.to(F64), Y.to(F64)
    P, D_in = X.shape
    D_out = Y.shape[1]
    return _Problem(
        X=X,
        Y=Y,
        L=L,
        phi=phi,
        g=g,
        cfg=cfg,
        Phi0=X @ X.T / D_in,
        eye=torch.eye(P, device=X.device, dtype=F64),
        alpha=lambda_w * P * D_out / cfg.beta if cfg.alpha is None else cfg.alpha,
        prior_scale=1.0 / lambda_w if cfg.prior_scale is None else cfg.prior_scale,
        rate_tilt=lambda_r if cfg.rate_tilt is None else cfg.rate_tilt,
        lp=cfg.learning_prefactor,
        psi_is_phi=(True,) * (L - 1) + (cfg.readout == "phi",),
        sample_dtype=resolve_dtype(cfg.sample_dtype),
    )


def _prior_cov(prob: _Problem, l: int, Phi: torch.Tensor) -> torch.Tensor:
    """Σ_l = prior_scale · Φ0 for l = 0, prior_scale · g² Φ_{l-1} otherwise."""
    return prob.prior_scale * (prob.Phi0 if l == 0 else (prob.g**2) * Phi[l - 1])


def _new_sampler(
    prob: _Problem, l: int, Sigma: torch.Tensor, M: torch.Tensor, generator: torch.Generator
) -> SingleSiteHMC:
    """Sampler for layer l with prior Σ, tilt M and the HMC settings of cfg."""
    c = prob.cfg
    return SingleSiteHMC(
        Sigma,
        M,
        prob.rate_tilt,
        prob.phi,
        prob.psi_is_phi[l],
        n_chains=c.n_chains,
        eps=c.eps0,
        n_leap=c.n_leap,
        target_accept=c.target_accept,
        adapt_rate=c.adapt_rate,
        jitter_rel=c.jitter_rel,
        sample_dtype=prob.sample_dtype,
        precondition=c.precondition,
        generator=generator,
    )


def _make_samplers(prob: _Problem, Phi, Phi_hat, gen) -> list[SingleSiteHMC]:
    """One persistent sampler per layer, prior Σ_l(Φ) and tilt Φ̂_l (chains drawn from gen)."""
    return [_new_sampler(prob, l, _prior_cov(prob, l, Phi), Phi_hat[l], gen) for l in range(prob.L)]


# ----------------------------------------------------------------------------- initial guesses
def _init_gaussian(prob: _Problem, Phi: torch.Tensor, gen) -> None:
    """The paper's lazy guess (Sec. G.2): Φ_l = ⟨ψψᵀ⟩ under the untilted Gaussian N(0, Σ_l),
    layer by layer, from exact samples.  Ignores the rate tilt, so Φ̂ = 0 is not the inner
    solution at step 0."""
    n = prob.cfg.n_init_samples
    for l in range(prob.L):
        Sigma = _prior_cov(prob, l, Phi)
        Lc = torch.linalg.cholesky(
            sym(Sigma) + prob.cfg.jitter_rel * Sigma.diagonal().mean() * prob.eye
        )
        h = torch.randn(n, prob.P, device=prob.dev, dtype=F64, generator=gen) @ Lc.T
        psi = prob.phi(h) if prob.psi_is_phi[l] else h
        Phi[l] = psi.T @ psi / n


def _init_lazy(prob: _Problem, Phi, Phi_hat, gen, log: Log) -> None:
    """Lazy guess consistent with the rate tilt: Φ_l = ⟨ψψᵀ⟩ under the Φ̂ = 0 density of each
    layer (sampled with HMC), layer by layer.  Then Φ̂ = 0 is exactly the inner solution at
    step 0 and the outer loop starts from a self-consistent point."""
    c = prob.cfg
    for l in range(prob.L):
        tmp = _new_sampler(prob, l, _prior_cov(prob, l, Phi), Phi_hat[l], gen)
        tmp.run(c.warmup_trans, collect=False)
        mom, d = tmp.run(max(c.n_init_samples // c.n_chains, 1))
        Phi[l] = mom["psi_psi"]
        log(
            f"init layer {l}: tr/P={Phi[l].diagonal().mean().item():.3e} "
            f"(untilted prior tr/P={_prior_cov(prob, l, Phi).diagonal().mean().item():.3e}) "
            f"acc={d['accept']:.2f} rho1={d['rho1']:.2f} mc_err={d['psi_psi_rel_mc_err']:.1e}"
        )


def _init_from_net(prob: _Problem, Phi, Phi_hat, net_Phi, net_H) -> None:
    """Φ = the net's empirical kernels; Φ̂ from the stationarity conditions using the net's
    own neurons as samples: Φ̂_L = -2c A Y Yᵀ A (Eq. 31c),
    Φ̂_l = sym(Φ_l⁻¹ ⟨h uᵀ⟩_{l+1}) (Eq. 31b, Stein form)."""
    Phi[:] = net_Phi
    A = torch.linalg.solve(prob.alpha * prob.eye + Phi[-1], prob.eye)
    Phi_hat[-1] = -2.0 * prob.lp * A @ prob.Y @ prob.Y.T @ A
    # The helper sampler only evaluates the tilt gradient; its own random chain draw is
    # discarded, so it gets a throwaway generator instead of consuming the solver's stream.
    scratch = torch.Generator(device=prob.dev).manual_seed(0)
    for l in range(prob.L - 2, -1, -1):
        tmp = SingleSiteHMC(
            _prior_cov(prob, l + 1, Phi),
            Phi_hat[l + 1],
            prob.rate_tilt,
            prob.phi,
            prob.psi_is_phi[l + 1],
            n_chains=1,
            sample_dtype=F64,
            jitter_rel=prob.cfg.jitter_rel,
            generator=scratch,
        )
        h = net_H[:, l + 1].T  # (N, P)
        _, u = tmp.tilt_and_grad(h)
        h_u = h.T @ u / h.shape[0]
        Phi_hat[l] = sym(torch.linalg.solve(Phi[l], h_u))


def _init_kernels(prob: _Problem, gen, net_Phi, net_H, log: Log):
    """Initial (Φ, Φ̂), (L, P, P) float64 each, by cfg.init in {gaussian, lazy, net}."""
    L, P = prob.L, prob.P
    Phi = torch.zeros(L, P, P, device=prob.dev, dtype=F64)
    Phi_hat = torch.zeros(L, P, P, device=prob.dev, dtype=F64)
    if prob.cfg.init == "gaussian":
        _init_gaussian(prob, Phi, gen)
    elif prob.cfg.init == "lazy":
        _init_lazy(prob, Phi, Phi_hat, gen, log)
    elif prob.cfg.init == "net":
        if net_Phi is None:
            raise ValueError("init='net' needs the trained network (pass net=...)")
        _init_from_net(prob, Phi, Phi_hat, net_Phi, net_H)
    else:
        raise ValueError(f"unknown init {prob.cfg.init!r}")
    return Phi, Phi_hat


# ----------------------------------------------------------------------------- mutable state
@dataclass
class _SolverState:
    """Everything the outer loop carries from one step to the next (= a checkpoint)."""

    outer: int  # next outer step to run
    Phi: torch.Tensor
    Phi_hat: torch.Tensor
    samplers: list[SingleSiteHMC]
    gen: torch.Generator
    eta: float  # outer step size (dimensionless)
    eta_hat: np.ndarray  # per-layer inner step sizes
    h2_last: list[float]  # last healthy chain scale ||h||²/P per layer (for `_healthy`)
    eps_ref: list[float]  # warm-up step size per layer (for the collapse test in `_healthy`)
    history: History
    mse_net: float | None
    net_Phi: torch.Tensor | None
    elapsed: float  # seconds spent in solve before this state, for history["time"]


def _empty_history(with_net: bool) -> History:
    return {k: [] for k in HISTORY_KEYS if with_net or k not in NET_KEYS}


def _fresh_state(prob: _Problem, net: MLP | None, gen, log: Log) -> _SolverState:
    """Initial kernels, samplers and their warm-up: the state before outer step 0."""
    net_Phi = net_H = mse_net = None
    if net is not None:
        # The net runs in its own dtype; the theory works in float64.
        X_net = prob.X.to(net.dtype)
        net_Phi, net_H = empirical_kernels(net, X_net, prob.cfg.readout)
        net_Phi, net_H = net_Phi.to(F64), net_H.to(F64)
        with torch.no_grad():
            _, Z = net(X_net)
            mse_net = ((prob.Y - Z.to(F64)) ** 2).mean().item()
    Phi, Phi_hat = _init_kernels(prob, gen, net_Phi, net_H, log)
    samplers = _make_samplers(prob, Phi, Phi_hat, gen)
    warm_diag = [s.run(prob.cfg.warmup_trans, collect=False)[1] for s in samplers]
    log(
        "warm-up: "
        + " | ".join(
            f"l={l} acc={d['accept']:.2f} eps={d['eps']:.3g} div={d['n_divergent']} "
            f"rho1={d['rho1']:.2f}"
            for l, d in enumerate(warm_diag)
        )
    )
    return _SolverState(
        outer=0,
        Phi=Phi,
        Phi_hat=Phi_hat,
        samplers=samplers,
        gen=gen,
        eta=float(prob.cfg.eta_phi),
        eta_hat=np.full(prob.L, float(prob.cfg.eta_phihat)),
        h2_last=[d["h2_mean"] for d in warm_diag],
        eps_ref=[d["eps"] for d in warm_diag],
        history=_empty_history(net is not None),
        mse_net=mse_net,
        net_Phi=net_Phi,
        elapsed=0.0,
    )


# ----------------------------------------------------------------------------- small pieces
def _c2(Phi_l: torch.Tensor) -> float:
    """c_l² with c_l = tr(Φ_l)/P: the scale that makes the step sizes dimensionless."""
    return (Phi_l.diagonal().mean().item()) ** 2


def _reg_inv(A: torch.Tensor, inv_floor_rel: float) -> torch.Tensor:
    """A⁻¹ with eigenvalues floored at inv_floor_rel · λ_max: Φ0 = XXᵀ/D_in is rank-deficient
    when D_in < P, and the Monte Carlo noise in ⟨u hᵀ⟩ would otherwise be amplified by the
    condition number."""
    w, V = torch.linalg.eigh(sym(A))
    w = torch.clamp(w, min=inv_floor_rel * w.max().clamp_min(1e-300))
    return (V / w) @ V.T


def _healthy(diag: dict, h2_before: float, n_tr: int, eps_ref_l: float, cfg) -> bool:
    """A sampler run is healthy if it has few divergences, no resets, no blow-up of the chain
    scale, no step-size collapse and (if moments were collected) a bounded MC error."""
    div_frac = diag["n_divergent"] / (cfg.n_chains * n_tr)
    growth = diag["h2_mean"] / max(h2_before, 1e-300)
    return (
        (div_frac <= cfg.max_div_frac)
        and (growth <= cfg.max_h2_growth)
        and diag["n_reset"] == 0
        and diag["eps"] >= cfg.eps_collapse * eps_ref_l
        and diag.get("psi_psi_rel_mc_err", 0.0) <= cfg.max_mc_err
    )


def _tilt_step(prob: _Problem, st: _SolverState, l, hat0, grad0, eta_l) -> torch.Tensor:
    """Φ̂_l ← Π(Φ̂_l + η̂_l ∂S/∂Φ̂_l / c_l²), Π the projection onto normalisable tilts."""
    new = sym(hat0 + eta_l * grad0 / _c2(st.Phi[l]))
    return st.samplers[l].project_tilt(new, prob.cfg.tilt_floor) if prob.cfg.project_tilt else new


def _learning_pieces(prob: _Problem, Phi_last: torch.Tensor):
    """A = (αI + Φ_L)⁻¹, AY, the learning term c Tr[Yᵀ A Y] and the train MSE of the
    theory's kernel predictor f = Φ_L A Y."""
    A = torch.linalg.solve(prob.alpha * prob.eye + Phi_last, prob.eye)
    AY = A @ prob.Y
    learning = prob.lp * (prob.Y * AY).sum().item()
    f = Phi_last @ AY
    mse = ((prob.Y - f) ** 2).mean().item()
    return A, AY, learning, mse


# ----------------------------------------------------------------------------- inner ascent
@dataclass
class _InnerResult:
    residual: np.ndarray  # ||⟨ψψᵀ⟩ - Φ|| / ||Φ|| per layer
    grad_rel: np.ndarray  # ||∂S/∂Φ̂|| / ||Φ|| per layer
    n_backtrack: np.ndarray  # reverted steps per layer
    converged: bool
    n_steps: int


def _inner_layer_step(prob, st: _SolverState, l, inner, pending, res: _InnerResult) -> dict:
    """One ascent step on Φ̂_l.  The HMC run at its top also validates the previous step
    (retrospective backtracking): if that step broke the sampler it is reverted and retried
    with half the step size.  Updates `pending[l]` and `res` in place; returns the run's
    diagnostics."""
    cfg, s = prob.cfg, st.samplers[l]
    moments, diag = s.run(cfg.n_trans)
    if pending[l] is not None and not _healthy(
        diag, st.h2_last[l], cfg.n_trans, st.eps_ref[l], cfg
    ):
        state0, hat0, grad0 = pending[l]
        res.n_backtrack[l] += 1
        st.eta_hat[l] *= 0.5
        st.Phi_hat[l] = _tilt_step(prob, st, l, hat0, grad0, st.eta_hat[l])
        s.set_tilt(st.Phi_hat[l])
        s.set_state(state0)
        if res.n_backtrack[l] > cfg.max_backtrack * (inner + 1):
            raise RuntimeError(
                f"layer {l}: inner step keeps breaking the sampler (eta_phihat={st.eta_hat[l]:.2e})"
            )
        return diag
    st.h2_last[l] = diag["h2_mean"]
    if pending[l] is not None:
        st.eta_hat[l] = min(st.eta_hat[l] * cfg.eta_grow, cfg.eta_max)
    grad_hat = 0.5 * (moments["psi_psi"] - st.Phi[l])
    res.residual[l] = relative_fro(moments["psi_psi"], st.Phi[l])
    res.grad_rel[l] = grad_hat.norm().item() / max(st.Phi[l].norm().item(), 1e-300)
    pending[l] = (s.get_state(), st.Phi_hat[l].clone(), grad_hat)
    st.Phi_hat[l] = _tilt_step(prob, st, l, st.Phi_hat[l], grad_hat, st.eta_hat[l])
    s.set_tilt(st.Phi_hat[l])
    return diag


def _inner_ascent(prob: _Problem, st: _SolverState) -> _InnerResult:
    """Gradient ascent on Φ̂ at fixed Φ, ∂S/∂Φ̂_l = ½(⟨ψψᵀ⟩_l - Φ_l), until every layer's
    residual is below max(inner_tol, inner_tol_mc × its own MC error) or n_inner_max steps."""
    cfg, L = prob.cfg, prob.L
    res = _InnerResult(
        residual=np.full(L, np.nan),
        grad_rel=np.full(L, np.nan),
        n_backtrack=np.zeros(L, dtype=int),
        converged=False,
        n_steps=0,
    )
    pending = [None] * L  # (state_before, hat_before, grad_before) of each layer's last step
    for inner in range(cfg.n_inner_max):
        diags = [_inner_layer_step(prob, st, l, inner, pending, res) for l in range(L)]
        res.n_steps = inner + 1
        tol = np.array(
            [max(cfg.inner_tol, cfg.inner_tol_mc * d["psi_psi_rel_mc_err"]) for d in diags]
        )
        if np.isfinite(res.residual).all() and (res.residual < tol).all():
            res.converged = True
            break
    return res


# ----------------------------------------------------------------------------- outer step
@dataclass
class _OuterGradients:
    grads: list[torch.Tensor]  # ∂S/∂Φ_l
    fp_stein: list[float]  # ||Φ̂_l - Φ̂_l^Stein|| / ||Φ̂_l^Stein||
    fp_direct: list[float]  # same for the direct form (nan for the last layer)
    stein_vs_direct: list[float]  # disagreement of the two forms (equilibration check)
    learning: float
    mse_theory: float
    hmc_diag: list[dict]  # diagnostics of the fresh n_trans_outer run


def _outer_gradients(prob: _Problem, st: _SolverState) -> _OuterGradients:
    """∂S/∂Φ_l from a fresh, longer run under the final Φ̂:
    l < L: -½Φ̂_l + ½ sym(⟨u hᵀ⟩_{l+1} Φ_l⁻¹)  (Eq. 31b, Stein form);
           the direct form s(Σ⁻¹ - Σ⁻¹⟨hhᵀ⟩_{l+1}Σ⁻¹), s = prior_scale g², is a diagnostic;
    l = L: -½Φ̂_L - c A Y Yᵀ A                 (Eq. 31c)."""
    cfg, L = prob.cfg, prob.L
    moments, hmc_diag = [], []
    for l, s in enumerate(st.samplers):
        m, d = s.run(cfg.n_trans_outer)
        st.h2_last[l] = d["h2_mean"]
        moments.append(m)
        hmc_diag.append(d)
    A, _, learning, mse_theory = _learning_pieces(prob, st.Phi[-1])
    scale = prob.prior_scale * prob.g**2
    out = _OuterGradients([], [], [], [], learning, mse_theory, hmc_diag)
    for l in range(L):
        if l < L - 1:
            Phi_inv = _reg_inv(st.Phi[l], cfg.inv_floor_rel)
            hat_stein = sym(moments[l + 1]["h_u"].T @ Phi_inv)
            Sig_inv = Phi_inv / scale
            hat_direct = sym(scale * (Sig_inv - Sig_inv @ moments[l + 1]["h_h"] @ Sig_inv))
            out.stein_vs_direct.append(relative_fro(hat_stein, hat_direct))
            out.fp_direct.append(relative_fro(st.Phi_hat[l], hat_direct))
        else:
            hat_stein = -2.0 * prob.lp * A @ prob.Y @ prob.Y.T @ A
            out.stein_vs_direct.append(float("nan"))
            out.fp_direct.append(float("nan"))
        out.fp_stein.append(relative_fro(st.Phi_hat[l], hat_stein))
        out.grads.append(-0.5 * st.Phi_hat[l] + 0.5 * hat_stein)
    return out


@dataclass
class _OuterStep:
    n_clamped: list[int]
    lam_min: list[float]
    clipped: list[bool]
    n_backtrack: int


def _trust_region_step(
    prob: _Problem, st: _SolverState, grads: list[torch.Tensor], Phi_prev: torch.Tensor
) -> tuple[list[int], list[float], list[bool]]:
    """Φ_l ← Π_PD(Φ_l^prev - η ∂S/∂Φ_l c_l²), the step clipped to max_rel_step ||Φ_l^prev||.
    c_l is read from the current Φ_l, i.e. the last attempted step when backtracking."""
    cfg = prob.cfg
    n_clamped, lam_min, clipped = [], [], []
    for l in range(prob.L):
        step = st.eta * grads[l] * _c2(st.Phi[l])
        cap = cfg.max_rel_step * Phi_prev[l].norm().item()
        sn = step.norm().item()
        clipped.append(sn > cap)
        if sn > cap:
            step = step * (cap / sn)
        Phi_l, nc, lm = project_pd(Phi_prev[l] - step, cfg.pd_floor_rel)
        st.Phi[l] = Phi_l
        n_clamped.append(nc)
        lam_min.append(lm)
    return n_clamped, lam_min, clipped


def _probe_samplers(prob, st: _SolverState) -> bool:
    """Move the priors of layers 1..L-1 to the new Φ and check every chain stays healthy."""
    ok = True
    for l in range(1, prob.L):
        st.samplers[l].set_prior(_prior_cov(prob, l, st.Phi))
        _, d = st.samplers[l].run(prob.cfg.n_trans, collect=False)
        ok = ok and _healthy(d, st.h2_last[l], prob.cfg.n_trans, st.eps_ref[l], prob.cfg)
    return ok


def _outer_descent(prob, st: _SolverState, grads, Phi_prev, outer, log: Log) -> _OuterStep:
    """One trust-region descent step on Φ.  If it breaks a sampler, revert (priors and chain
    states) and halve η, up to max_backtrack times or until η = eta_min_rel · eta_phi; an
    accepted first try grows η by eta_grow up to eta_max."""
    cfg = prob.cfg
    states0 = [s.get_state() for s in st.samplers]
    n_bt = 0
    while True:
        n_clamped, lam_min, clipped = _trust_region_step(prob, st, grads, Phi_prev)
        ok = _probe_samplers(prob, st)
        if ok or n_bt >= cfg.max_backtrack or st.eta <= cfg.eta_min_rel * cfg.eta_phi:
            if not ok:
                log(
                    f"[{outer:04d}] WARNING: outer step still unhealthy after {n_bt} halvings "
                    f"(eta_phi={st.eta:.2e})"
                )
            elif n_bt == 0:
                st.eta = min(st.eta * cfg.eta_grow, cfg.eta_max)
            return _OuterStep(n_clamped, lam_min, clipped, n_bt)
        n_bt += 1
        st.eta = max(st.eta * 0.5, cfg.eta_min_rel * cfg.eta_phi)
        for l in range(1, prob.L):
            st.samplers[l].set_prior(_prior_cov(prob, l, Phi_prev))
            st.samplers[l].set_state(states0[l])


# ----------------------------------------------------------------------------- diagnostics
def _record(
    prob: _Problem,
    st: _SolverState,
    outer: int,
    elapsed: float,
    inner: _InnerResult,
    og: _OuterGradients,
    step: _OuterStep,
    Phi_prev: torch.Tensor,
    Phi_hat_prev: torch.Tensor,
) -> dict:
    """The history entry of one outer step (per-layer quantities as numpy arrays)."""
    L, Phi, Phi_hat, diags = prob.L, st.Phi, st.Phi_hat, og.hmc_diag

    def per_layer(key):
        return np.array([d[key] for d in diags])

    rec = {
        "outer_step": outer,
        "time": elapsed,
        "inner_steps": inner.n_steps,
        "inner_residual": inner.residual.copy(),
        "inner_grad_rel": inner.grad_rel.copy(),
        "inner_converged": inner.converged,
        "step_clipped": np.array(step.clipped),
        "eta_phi": st.eta,
        "eta_phihat": st.eta_hat.copy(),
        "n_backtrack_inner": inner.n_backtrack.copy(),
        "n_backtrack_outer": step.n_backtrack,
        "outer_grad_rel": np.array(
            [
                og.grads[l].norm().item() * _c2(Phi[l]) / max(Phi_prev[l].norm().item(), 1e-300)
                for l in range(L)
            ]
        ),
        "phihat_fp_resid_stein": np.array(og.fp_stein),
        "phihat_fp_resid_direct": np.array(og.fp_direct),
        "stein_vs_direct": np.array(og.stein_vs_direct),
        "trace_term": -0.5 * sum((Phi[l] * Phi_hat[l]).sum().item() for l in range(L)),
        "learning_term": og.learning,
        "train_mse_theory": og.mse_theory,
        "train_mse_net": st.mse_net,
        "phi_lambda_min": np.array(step.lam_min),
        "n_pd_clamped": np.array(step.n_clamped),
        "norm_margin": np.array([s.normalisability_margin() for s in st.samplers]),
        "hmc_accept": per_layer("accept"),
        "hmc_eps": per_layer("eps"),
        "hmc_div": per_layer("n_divergent"),
        "hmc_reset": per_layer("n_reset"),
        "hmc_max_abs_dH": per_layer("max_abs_dH"),
        "hmc_rho1": per_layer("rho1"),
        "hmc_ess_frac": per_layer("ess_frac"),
        "hmc_mc_err": per_layer("psi_psi_rel_mc_err"),
        "hmc_stein_mismatch": per_layer("stein_mismatch"),
        "hmc_h2_ratio": np.array([d["h2_mean"] / d["prior_h2"] for d in diags]),
        "phi_change_rel": np.array([relative_fro(Phi[l], Phi_prev[l]) for l in range(L)]),
        "phihat_change_rel": np.array(
            [relative_fro(Phi_hat[l], Phi_hat_prev[l]) for l in range(L)]
        ),
    }
    if st.net_Phi is not None:
        rec["align_net"] = np.array([kernel_alignment(Phi[l], st.net_Phi[l]) for l in range(L)])
        rec["dist_net"] = np.array([relative_fro(Phi[l], st.net_Phi[l]) for l in range(L)])
    return rec


def _format_step(rec: dict, inner: _InnerResult, step: _OuterStep, st: _SolverState) -> str:
    def fmt(a):
        return "[" + " ".join(f"{v:.2e}" for v in a) + "]"

    mse_net = rec["train_mse_net"]
    msg = (
        f"[{rec['outer_step']:04d}] t={rec['time']:.0f}s "
        f"inner={inner.n_steps:3d}{'' if inner.converged else '*'} "
        f"res_in={fmt(inner.residual)} grad_out={fmt(rec['outer_grad_rel'])} "
        f"fp(Φ̂)={fmt(rec['phihat_fp_resid_stein'])} "
        f"mse={rec['train_mse_theory']:.3e}"
        + (f" (net {mse_net:.3e})" if mse_net is not None else "")
        + f" | eta={st.eta:.1e}/{fmt(st.eta_hat)} bt={inner.n_backtrack.tolist()}/{step.n_backtrack}"
        + f" | HMC acc={fmt(rec['hmc_accept'])} eps={fmt(rec['hmc_eps'])} "
        f"div={rec['hmc_div']} rho1={fmt(rec['hmc_rho1'])} mcerr={fmt(rec['hmc_mc_err'])} "
        f"stein={fmt(rec['hmc_stein_mismatch'])} h2/prior={fmt(rec['hmc_h2_ratio'])} "
        f"margin={fmt(rec['norm_margin'])}"
    )
    if "align_net" in rec:
        msg += f" | align={fmt(rec['align_net'])}"
    if sum(step.n_clamped):
        msg += f" | PD-clamped={step.n_clamped}"
    if any(step.clipped):
        msg += f" | step-clipped={[int(c) for c in step.clipped]}"
    return msg


def _outer_step(prob: _Problem, st: _SolverState, t_start: float, log: Log) -> None:
    """One iteration of Algorithm 1: inner ascent on Φ̂, one descent step on Φ, diagnostics.
    Advances st.outer by one."""
    outer = st.outer
    Phi_prev, Phi_hat_prev = st.Phi.clone(), st.Phi_hat.clone()
    inner = _inner_ascent(prob, st)
    og = _outer_gradients(prob, st)
    step = _outer_descent(prob, st, og.grads, Phi_prev, outer, log)
    rec = _record(prob, st, outer, time.time() - t_start, inner, og, step, Phi_prev, Phi_hat_prev)
    for k in st.history:
        st.history[k].append(rec[k])
    if outer % prob.cfg.log_int == 0:
        log(_format_step(rec, inner, step, st))
    st.outer = outer + 1


# ----------------------------------------------------------------------------- (de)serialisation
def _history_to_lists(history: History) -> History:
    """numpy entries -> nested lists of Python scalars, so the payload is weights_only-loadable."""
    return {
        k: [v.tolist() for v in vs] if k in _ARRAY_KEYS else list(vs) for k, vs in history.items()
    }


def _history_from_lists(history: History) -> History:
    return {
        k: [np.asarray(v) for v in vs] if k in _ARRAY_KEYS else list(vs)
        for k, vs in history.items()
    }


def _to_payload(st: _SolverState) -> dict:
    return {
        "outer": st.outer,
        "Phi": st.Phi,
        "Phi_hat": st.Phi_hat,
        "history": _history_to_lists(st.history),
        "eta": st.eta,
        "eta_hat": st.eta_hat.tolist(),
        "h2_last": list(st.h2_last),
        "eps_ref": list(st.eps_ref),
        "samplers": [s.state_dict() for s in st.samplers],
        "generator": st.gen.get_state(),
        "mse_net": st.mse_net,
        "net_Phi": st.net_Phi,
        "elapsed": st.elapsed,
    }


def _restore_state(
    prob: _Problem, payload: dict, gen: torch.Generator, path: Path, with_net: bool
) -> _SolverState:
    """Rebuild the samplers at the saved (Φ, Φ̂), load their chains and the generator state.
    No warm-up: the saved chains are already equilibrated."""
    Phi, Phi_hat = payload["Phi"].to(F64), payload["Phi_hat"].to(F64)
    expected = (prob.L, prob.P, prob.P)
    if tuple(Phi.shape) != expected or tuple(Phi_hat.shape) != expected:
        raise ValueError(f"{path} holds kernels of shape {tuple(Phi.shape)}, expected {expected}")
    # The net diagnostics are fixed when a solve starts; continuing with a different net
    # setting would silently drop them or leave history columns of different lengths.
    if (payload["net_Phi"] is not None) != with_net:
        written, given = (
            ("with", "without") if payload["net_Phi"] is not None else ("without", "with")
        )
        raise ValueError(
            f"{path} was written {written} a net but solve was called {given} one; "
            "pass the same net (or none) as the original run, or delete the checkpoint"
        )
    samplers = _make_samplers(prob, Phi, Phi_hat, gen)
    for s, state in zip(samplers, payload["samplers"], strict=True):
        s.load_state_dict(state)
    # After the samplers: their construction draws from gen, the saved state must win.
    gen.set_state(payload["generator"].cpu())
    return _SolverState(
        outer=payload["outer"],
        Phi=Phi,
        Phi_hat=Phi_hat,
        samplers=samplers,
        gen=gen,
        eta=payload["eta"],
        eta_hat=np.array(payload["eta_hat"], dtype=float),
        h2_last=list(payload["h2_last"]),
        eps_ref=list(payload["eps_ref"]),
        history=_history_from_lists(payload["history"]),
        mse_net=payload["mse_net"],
        net_Phi=payload["net_Phi"],
        elapsed=payload["elapsed"],
    )


def _saved_state_path(checkpoint_dir: Path | None) -> Path | None:
    """final.pt if present, else the latest ckpt_*.pt, else None."""
    if checkpoint_dir is None:
        return None
    final = Path(checkpoint_dir) / FINAL_NAME
    if final.exists():
        return final
    return latest_checkpoint(checkpoint_dir, CKPT_PREFIX) if Path(checkpoint_dir).exists() else None


# ----------------------------------------------------------------------------- entry points
def solve(
    X: torch.Tensor,
    Y: torch.Tensor,
    L: int,
    lambda_w: float,
    lambda_r: float,
    phi: Activation,
    g: float,
    cfg: MeanFieldConfig,
    net: MLP | None = None,
    log: Log = print,
    checkpoint_dir: Path | None = None,
    config_hash: str = "",
) -> SolveResult:
    """Solve  min_Φ max_Φ̂ S(Φ, Φ̂)  by Algorithm 1 of arXiv:2502.07998 with the
    single-site expectations estimated by persistent HMC.

    Step sizes are dimensionless: with c_l = tr(Φ_l)/P,
        Φ̂_l += eta_phihat_l · ∂S/∂Φ̂_l / c_l²      (Φ̂ has units of 1/Φ)
        Φ_l  -= eta_phi      · ∂S/∂Φ_l  · c_l²     (clipped to ||ΔΦ_l|| ≤ max_rel_step ||Φ_l||)
    and are adapted: every HMC run also validates the step taken before it.  If the
    chains show divergences (> max_div_frac of transitions), resets, or their scale
    ||h||²/P jumps by more than max_h2_growth, the step is reverted (kernel and chain
    state) and its step size halved; otherwise the step size grows by eta_grow up to
    eta_max.  This keeps the iteration on the normalisable side of the tilt.
    The inner loop stops when every layer's residual ||⟨ψψᵀ⟩-Φ||/||Φ|| is below
    max(inner_tol, inner_tol_mc × its own Monte Carlo error): below that it is noise.

    phi is a callable (any elementwise activation, not only the registered ones).  All
    randomness comes from one torch.Generator seeded with cfg.seed on X's device.

    With checkpoint_dir (requires config_hash): return its final.pt if present, else resume
    from its latest ckpt_*.pt if present; write ckpt_<outer:05d>.pt every cfg.save_int
    outer steps, and final.pt at the end (which then replaces the checkpoints).  A saved state
    must have been written with a net iff `net` is given.  Without checkpoint_dir, nothing
    touches disk.
    """
    if checkpoint_dir is not None and not config_hash:
        raise ValueError("checkpoint_dir needs a non-empty config_hash to guard against reuse")
    t_call = time.time()
    prob = _make_problem(X, Y, L, lambda_w, lambda_r, phi, g, cfg)
    gen = torch.Generator(device=prob.dev).manual_seed(cfg.seed)

    saved = _saved_state_path(checkpoint_dir)
    if saved is None:
        st = _fresh_state(prob, net, gen, log)
    else:
        payload = load_artefact(saved, config_hash, prob.dev)
        st = _restore_state(prob, payload, gen, saved, with_net=net is not None)
        if saved.name == FINAL_NAME:
            log(f"loaded finished solve from {saved} ({st.outer} outer steps)")
            return SolveResult(st.Phi, st.Phi_hat, st.history, st.samplers)
        log(f"resumed from {saved} at outer step {st.outer}")

    t_start = t_call - st.elapsed
    while st.outer < cfg.n_outer:
        _outer_step(prob, st, t_start, log)
        st.elapsed = time.time() - t_start
        if checkpoint_dir is not None and st.outer % cfg.save_int == 0:
            write_checkpoint(
                checkpoint_dir, CKPT_PREFIX, st.outer, _to_payload(st), config_hash, CKPT_WIDTH
            )
    if checkpoint_dir is not None:
        _save_final(Path(checkpoint_dir), _to_payload(st), config_hash)
    return SolveResult(st.Phi, st.Phi_hat, st.history, st.samplers)


def _save_final(checkpoint_dir: Path, payload: dict, config_hash: str) -> None:
    """Write final.pt, then delete the checkpoints: final.pt supersedes them (the last one
    holds the same payload, which is large at production size)."""
    save_artefact(checkpoint_dir / FINAL_NAME, payload, config_hash)
    for ckpt in checkpoint_dir.glob(f"{CKPT_PREFIX}*.pt"):
        ckpt.unlink()


def _meanfield_hash(cfg: Config) -> str:
    return cfg.section_hash("data", "net", "train", "meanfield")


def run_meanfield(
    cfg: Config, dataset: Dataset, net: MLP, cache_dir: Path, log: Log = print
) -> SolveResult:
    """Solve on the first cfg.meanfield.n_points training points, caching under
    CACHE_DIR/meanfield (returns final.pt if present, resumes from a checkpoint otherwise).
    The rate tilt is train.rate_lambda unless meanfield.rate_tilt overrides it."""
    X, Y = _solver_inputs(dataset, cfg)
    return solve(
        X,
        Y,
        cfg.net.L,
        cfg.train.weight_lambda,
        cfg.train.rate_lambda,
        get_phi(cfg.net.phi),
        cfg.net.g,
        cfg.meanfield,
        net=net,
        log=log,
        checkpoint_dir=cache_paths(cache_dir).meanfield,
        config_hash=_meanfield_hash(cfg),
    )


def load_meanfield(
    cache_dir: Path, cfg: Config, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor, History]:
    """(Φ, Φ̂, history) of a finished solve; history entries are numpy arrays again.
    Raises FileNotFoundError if there is no result, StaleCacheError if the config changed."""
    path = cache_paths(cache_dir).meanfield / FINAL_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run `python -m fct solve` first")
    payload = load_artefact(path, _meanfield_hash(cfg), device)
    return payload["Phi"], payload["Phi_hat"], _history_from_lists(payload["history"])
