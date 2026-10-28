"""Mean-field saddle-point solver: Algorithm 1 of arXiv:2502.07998 with single-site
expectations from persistent HMC or from importance sampling, plus checkpoint / resume of the
outer loop.

The solver finds  min_Φ max_Φ̂ S(Φ, Φ̂)  for the action

    S = -½ Σ_l Tr Φ_l Φ̂_l + c Tr[Yᵀ(αI + Φ_L)⁻¹Y] - Σ_l ln Z_l,

where Z_l normalises the single-site density of layer l (see `fct.hmc`): prior
N(0, Σ_l), Σ_l = prior_scale · g² Φ_{l-1} (Σ_0 = prior_scale · Φ0, Φ0 = XXᵀ/D_in), tilted by
exp(-½ ψᵀΦ̂_l ψ - ½ λ_r φᵀφ).  ln Z is never estimated: every derivative of S is an
expectation under that density, which a per-layer sampler provides: `SingleSiteHMC` chains
(solver.sampler="hmc") or an `ImportanceSampler` drawing from the prior ("importance"):

    ∂S/∂Φ̂_l = ½(⟨ψψᵀ⟩_l - Φ_l)
    ∂S/∂Φ_l = -½ Φ̂_l + ½ sym(⟨u hᵀ⟩_{l+1} Φ_l⁻¹)        (Stein form of Eq. 31b)
    ∂S/∂Φ_L = -½ Φ̂_L - c A Y Yᵀ A,  A = (αI + Φ_L)⁻¹      (Eq. 31c)

Mapping of the network's (β, λ_w, λ_r) onto the action (`network_constants`; each constant can be
overridden in MeanFieldConfig): prior_scale = 1/λ_w, α = λ_w P D_out / β, rate tilt 2βλ_r,
c = learning_prefactor = λ_w/2, with β = train.beta for SGLD and 1 for Adam
(`resolve_beta`).  The loss carries the weight prior as λ_w/(2β)‖θ‖² (`fct.train.prior_coefficient`),
so the stationary law exp(-βL) of SGLD has weights N(0, 1/λ_w) for every β; the readout is a
Gaussian likelihood of precision βN/(P D_out); the rate penalty λ_r Σφ² enters the law as
exp(-βλ_r Σφ²), a tilt of 2βλ_r (the tilt multiplies φᵀφ summed over the P solved points, a
per-sample coefficient, as the rate penalty is); and the kernels are unscaled (Σ = prior_scale Φ0),
so the data-term prefactor is 1/(2 prior_scale).  Under Adam there is no Gibbs distribution and
β = 1 is the convention of the loss.

Layers: `solve` is the computation (it touches disk only when given a checkpoint_dir),
`run_meanfield` / `load_meanfield` are the cache-directory I/O.  The outer loop mutates one
`_SolverState`; everything needed to continue it bit-for-bit (up to the float round trip of
the chain state) is in that object, which is what makes checkpoint / resume straightforward.
The problem constants come from `MeanFieldConfig`, everything about how it is solved from its
`solver` group (`SolverConfig`).
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch

from fct.activations import Activation, get_phi, known_tilt_class, phi_prime
from fct.cache import (
    cache_paths,
    latest_checkpoint,
    load_artefact,
    save_artefact,
    write_checkpoint,
)
from fct.config import Config, MeanFieldConfig, SolverConfig, resolve_dtype
from fct.data import Dataset
from fct.hmc import (
    SingleSiteHMC,
    kernel_alignment,
    project_pd,
    relative_fro,
    sym,
    tilt_and_grad,
    tilt_rule,
)
from fct.importance import ImportanceSampler
from fct.model import MLP

F64 = torch.float64
CKPT_PREFIX, CKPT_WIDTH, FINAL_NAME = "ckpt_", 5, "final.pt"

# One entry per outer step for every key; plots read these names.  NET_KEYS are present
# only when `solve` is given a net: their entries are per-layer arrays, and a list of None
# in their place would break code that stacks them.  `train_mse_net` is a scalar per step and
# stays a per-step None without a net (None is weights_only-loadable, and
# `draw.fig_solver_history` treats an all-None column as absent).  The hmc_* keys are filled
# by both samplers where they mean something (ess_frac, mc_err, stein_mismatch, h2_ratio) and
# are NaN under importance sampling where they do not (accept, eps, div, reset, max_abs_dH,
# rho1); the is_* keys are NaN under HMC.
HISTORY_KEYS = (
    "outer_step", "time", "inner_steps", "inner_residual", "inner_grad_rel", "inner_converged",
    "step_clipped", "eta_phi", "eta_phihat", "n_backtrack_inner", "n_accepted_inner",
    "n_rejected_inner", "n_backtrack_outer",
    "outer_grad_rel", "phihat_fp_resid_stein", "phihat_fp_resid_direct", "stein_vs_direct",
    "picard_resid",
    "trace_term", "learning_term", "train_mse_theory", "train_mse_net",
    "phi_lambda_min", "n_pd_clamped", "norm_margin",
    "hmc_accept", "hmc_eps", "hmc_div", "hmc_reset", "hmc_max_abs_dH", "hmc_rho1", "hmc_ess_frac",
    "hmc_mc_err", "hmc_stein_mismatch", "hmc_h2_ratio", "is_max_weight",
    "align_net", "dist_net", "phi_change_rel", "phihat_change_rel", "tilt_projected_rel",
)  # fmt: skip
NET_KEYS = ("align_net", "dist_net")
# Keys whose per-step entry is a per-layer numpy array (saved as a list, restored on load).
_ARRAY_KEYS = frozenset(HISTORY_KEYS) - {
    "outer_step", "time", "inner_steps", "inner_converged", "eta_phi", "n_backtrack_outer",
    "trace_term", "learning_term", "train_mse_theory", "train_mse_net", "picard_resid",
}  # fmt: skip

History = dict[str, list]
Log = Callable[[str], Any]
Sampler = SingleSiteHMC | ImportanceSampler


class OuterSnapshot(NamedTuple):
    """What an `on_outer` callback sees after outer step `outer`: the kernels and the history
    so far. The kernels and the history's lists are copies, so the solver cannot be reached
    through them; the history's entries (numpy arrays) are shared and must be treated as
    read-only."""

    outer: int
    history: History
    Phi: torch.Tensor  # (L, P, P)
    Phi_hat: torch.Tensor  # (L, P, P)


OnOuter = Callable[[OuterSnapshot], Any]


class SpatialPriorUnsupportedError(ValueError):
    """The config sets `spatial`, which the solver (iid weight prior only) cannot honour."""


class InadequateSamplerError(RuntimeError):
    """Importance sampling from the prior cannot represent the density at an accepted point."""


class UnconvergedSolveError(RuntimeError):
    """The tilt tripwire fired: over the last outer steps Φ̂ stayed at a fixed-point residual of
    order one (it has lost its tilt, e.g. to the projection), so the result is certainly not a
    saddle point. Not raising this does not mean the solve converged."""


# ----------------------------------------------------------------------------- public results
def readout_activities(
    H: torch.Tensor, phi: Callable[[torch.Tensor], torch.Tensor]
) -> list[torch.Tensor]:
    """ψ^l = φ(h^l) (P, N) of every hidden layer from the pre-activations H (P, L, N). The last
    one is what the output reads, Z = Vᵀ φ(h^L) / N."""
    return [phi(H[:, l]) for l in range(H.shape[1])]


@torch.no_grad()
def empirical_kernels(net: MLP, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Φ_l = φ(h^l) φ(h^l)ᵀ / N from the trained network at the solver's inputs, and the
    raw pre-activations (P, L, N) (the N neurons are samples from the net's own
    single-site density, which is what the theory predicts)."""
    H, _ = net(X)  # (P, L, N)
    N = H.shape[2]
    psis = readout_activities(H, net.phi)
    return torch.stack([psi @ psi.T / N for psi in psis]), H


def _solver_inputs(dataset: Dataset, cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    """(X, Y) = the first cfg.meanfield.n_points training points, the solver's inputs."""
    n, P = cfg.meanfield.n_points, dataset.x.shape[0]
    if n > P:
        raise ValueError(f"meanfield.n_points={n} exceeds the dataset size P={P}")
    return dataset.x[:n], dataset.y[:n]


def net_kernels(net: MLP, dataset: Dataset, cfg: Config) -> torch.Tensor:
    """The trained net's empirical kernels (L, P, P), float64, on exactly the solver's inputs
    so they are directly comparable with the solver's Φ."""
    X, _ = _solver_inputs(dataset, cfg)
    Phi, _ = empirical_kernels(net, X.to(net.dtype))
    return Phi.to(F64)


@dataclass
class SolveResult:
    Phi: torch.Tensor  # (L, P, P)
    Phi_hat: torch.Tensor  # (L, P, P)
    history: History  # HISTORY_KEYS -> one entry per outer step
    samplers: list[Sampler]  # live samplers at the end of the run, for inspection


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
    sample_dtype: torch.dtype

    @property
    def P(self) -> int:
        return self.X.shape[0]

    @property
    def solver(self) -> SolverConfig:
        return self.cfg.solver

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
    if cfg.solver.project_tilt:
        tilt_rule(phi, True)  # fail now, not at the first step, if phi has no projection rule
    X, Y = X.to(F64), Y.to(F64)
    P, D_in = X.shape
    D_out = Y.shape[1]
    # `run_meanfield` resolves an unset beta from the network; a direct `solve` has no network,
    # so it takes beta = 1, the Adam convention of the loss.
    derived = network_constants(1.0 if cfg.beta is None else cfg.beta, lambda_w, lambda_r, P, D_out)

    def pick(explicit: float | None, name: str) -> float:
        return derived[name] if explicit is None else explicit

    return _Problem(
        X=X,
        Y=Y,
        L=L,
        phi=phi,
        g=g,
        cfg=cfg,
        Phi0=X @ X.T / D_in,
        eye=torch.eye(P, device=X.device, dtype=F64),
        alpha=pick(cfg.alpha, "alpha"),
        prior_scale=pick(cfg.prior_scale, "prior_scale"),
        rate_tilt=pick(cfg.rate_tilt, "rate_tilt"),
        lp=pick(cfg.learning_prefactor, "learning_prefactor"),
        sample_dtype=resolve_dtype(cfg.solver.sample_dtype),
    )


def network_constants(
    beta: float, weight_lambda: float, rate_lambda: float, n_points: int, d_out: int
) -> dict[str, float]:
    """The action's constants (prior_scale, alpha, rate_tilt, learning_prefactor) under which the
    theory describes the stationary law exp(-beta L) of the repo loss L (see the module docstring
    and `fct.train.prior_coefficient`); `beta` is the network's (`resolve_beta`)."""
    return {
        "prior_scale": 1.0 / weight_lambda,
        "alpha": weight_lambda * n_points * d_out / beta,
        "rate_tilt": 2.0 * beta * rate_lambda,
        "learning_prefactor": 0.5 * weight_lambda,
    }


def resolve_beta(cfg: Config) -> float:
    """The inverse temperature of the action: `meanfield.beta` if set, else the network's
    (train.beta for SGLD, 1 for Adam)."""
    if cfg.meanfield.beta is not None:
        return cfg.meanfield.beta
    return cfg.train.beta if cfg.train.optimizer == "sgld" else 1.0


def _prior_cov(prob: _Problem, l: int, Phi: torch.Tensor) -> torch.Tensor:
    """Σ_l = prior_scale · Φ0 for l = 0, prior_scale · g² Φ_{l-1} otherwise."""
    return prob.prior_scale * (prob.Phi0 if l == 0 else (prob.g**2) * Phi[l - 1])


def _draws_per_transition(c: SolverConfig) -> int:
    """Samples one transition adds: HMC chains, or fresh importance draws."""
    return c.n_importance_samples if c.sampler == "importance" else c.n_chains


def _warmup_trans(c: SolverConfig) -> int:
    """Warm-up transitions: HMC chains must equilibrate; independent importance draws need no
    warm-up, so one transition just measures the starting point."""
    return 1 if c.sampler == "importance" else c.warmup_trans


def _new_sampler(
    prob: _Problem, l: int, Sigma: torch.Tensor, M: torch.Tensor, generator: torch.Generator
) -> Sampler:
    """Sampler for layer l with prior Σ and tilt M, of the kind and settings of the solver."""
    c = prob.solver
    if c.sampler == "importance":
        return ImportanceSampler(
            Sigma,
            M,
            prob.rate_tilt,
            prob.phi,
            True,  # ψ = φ on every layer, the read-out one included
            n_samples=c.n_importance_samples,
            proposal_scale=c.importance_proposal_scale,
            jitter_rel=c.jitter_rel,
            sample_dtype=prob.sample_dtype,
            generator=generator,
        )
    return SingleSiteHMC(
        Sigma,
        M,
        prob.rate_tilt,
        prob.phi,
        True,
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


def _make_samplers(prob: _Problem, Phi, Phi_hat, gen) -> list[Sampler]:
    """One persistent sampler per layer, prior Σ_l(Φ) and tilt Φ̂_l (draws from gen)."""
    return [_new_sampler(prob, l, _prior_cov(prob, l, Phi), Phi_hat[l], gen) for l in range(prob.L)]


# ----------------------------------------------------------------------------- initial guesses
def _init_gaussian(prob: _Problem, Phi: torch.Tensor, gen) -> None:
    """The paper's lazy guess (Sec. G.2): Φ_l = ⟨ψψᵀ⟩ under the untilted Gaussian N(0, Σ_l),
    layer by layer, from exact samples.  Ignores the rate tilt, so Φ̂ = 0 is not the inner
    solution at step 0."""
    n = prob.solver.n_init_samples
    for l in range(prob.L):
        Sigma = _prior_cov(prob, l, Phi)
        Lc = torch.linalg.cholesky(
            sym(Sigma) + prob.solver.jitter_rel * Sigma.diagonal().mean() * prob.eye
        )
        h = torch.randn(n, prob.P, device=prob.dev, dtype=F64, generator=gen) @ Lc.T
        psi = prob.phi(h)
        Phi[l] = psi.T @ psi / n


def _init_lazy(prob: _Problem, Phi, Phi_hat, gen, log: Log) -> None:
    """Lazy guess consistent with the rate tilt: Φ_l = ⟨ψψᵀ⟩ under the Φ̂ = 0 density of each
    layer (sampled with the solver's sampler), layer by layer.  Then Φ̂ = 0 is exactly the
    inner solution at step 0 and the outer loop starts from a self-consistent point."""
    c = prob.solver
    for l in range(prob.L):
        tmp = _new_sampler(prob, l, _prior_cov(prob, l, Phi), Phi_hat[l], gen)
        tmp.run(_warmup_trans(c), collect=False)
        mom, d = tmp.run(max(c.n_init_samples // _draws_per_transition(c), 1))
        Phi[l] = mom["psi_psi"]
        log(
            f"init layer {l}: tr/P={Phi[l].diagonal().mean().item():.3e} "
            f"(untilted prior tr/P={_prior_cov(prob, l, Phi).diagonal().mean().item():.3e}) "
            + _quality(d, c.sampler)
            + f" mc_err={d['psi_psi_rel_mc_err']:.1e}"
        )


def _init_from_net(prob: _Problem, Phi, Phi_hat, net_Phi, net_H) -> None:
    """Φ = the net's empirical kernels; Φ̂ from the stationarity conditions using the net's
    own neurons as samples: Φ̂_L = -2c A Y Yᵀ A (Eq. 31c),
    Φ̂_l = sym(Φ_l⁻¹ ⟨h uᵀ⟩_{l+1}) (Eq. 31b, Stein form)."""
    Phi[:] = net_Phi
    A = torch.linalg.solve(prob.alpha * prob.eye + Phi[-1], prob.eye)
    Phi_hat[-1] = -2.0 * prob.lp * A @ prob.Y @ prob.Y.T @ A
    dphi = phi_prime(prob.phi)
    for l in range(prob.L - 2, -1, -1):
        h = net_H[:, l + 1].T  # (N, P)
        M = sym(Phi_hat[l + 1].to(F64))
        _, u = tilt_and_grad(h, M, float(prob.rate_tilt), prob.phi, dphi, True)
        h_u = h.T @ u / h.shape[0]
        Phi_hat[l] = sym(torch.linalg.solve(Phi[l], h_u))


def _init_kernels(prob: _Problem, gen, net_Phi, net_H, log: Log):
    """Initial (Φ, Φ̂), (L, P, P) float64 each, by cfg.solver.init in {gaussian, lazy, net}."""
    L, P, init = prob.L, prob.P, prob.solver.init
    Phi = torch.zeros(L, P, P, device=prob.dev, dtype=F64)
    Phi_hat = torch.zeros(L, P, P, device=prob.dev, dtype=F64)
    if init == "gaussian":
        _init_gaussian(prob, Phi, gen)
    elif init == "lazy":
        _init_lazy(prob, Phi, Phi_hat, gen, log)
    elif init == "net":
        if net_Phi is None:
            raise ValueError("init='net' needs the trained network (pass net=...)")
        _init_from_net(prob, Phi, Phi_hat, net_Phi, net_H)
    else:
        raise ValueError(f"unknown init {init!r}")
    return Phi, Phi_hat


# ----------------------------------------------------------------------------- mutable state
@dataclass
class _SolverState:
    """Everything the outer loop carries from one step to the next (= a checkpoint)."""

    outer: int  # next outer step to run
    Phi: torch.Tensor
    Phi_hat: torch.Tensor
    samplers: list[Sampler]
    gen: torch.Generator
    eta: float  # outer step size (dimensionless)
    eta_hat: np.ndarray  # per-layer inner step sizes
    h2_last: list[float]  # last healthy sample scale ||h||²/P per layer (for `_healthy`)
    eps_ref: list[float]  # warm-up HMC step size per layer (collapse test; NaN for importance)
    ess_ref: list[float]  # ESS fraction of the last accepted point per layer (NaN for HMC)
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
        net_Phi, net_H = empirical_kernels(net, X_net)
        net_Phi, net_H = net_Phi.to(F64), net_H.to(F64)
        with torch.no_grad():
            _, Z = net(X_net)
            mse_net = ((prob.Y - Z.to(F64)) ** 2).mean().item()
    Phi, Phi_hat = _init_kernels(prob, gen, net_Phi, net_H, log)
    samplers = _make_samplers(prob, Phi, Phi_hat, gen)
    warm_diag = [s.run(_warmup_trans(prob.solver), collect=False)[1] for s in samplers]
    if prob.solver.sampler == "importance":
        summary = [f"l={l} {_quality(d, 'importance')}" for l, d in enumerate(warm_diag)]
    else:
        summary = [
            f"l={l} acc={d['accept']:.2f} eps={d['eps']:.3g} div={d['n_divergent']} "
            f"rho1={d['rho1']:.2f}"
            for l, d in enumerate(warm_diag)
        ]
    log("warm-up: " + " | ".join(summary))
    st = _SolverState(
        outer=0,
        Phi=Phi,
        Phi_hat=Phi_hat,
        samplers=samplers,
        gen=gen,
        eta=float(prob.solver.eta_phi),
        eta_hat=np.full(prob.L, float(prob.solver.eta_phihat)),
        h2_last=[d["h2_mean"] for d in warm_diag],
        eps_ref=[d["eps"] for d in warm_diag],
        ess_ref=[float("nan")] * prob.L,
        history=_empty_history(net is not None),
        mse_net=mse_net,
        net_Phi=net_Phi,
        elapsed=0.0,
    )
    for l, d in enumerate(warm_diag):
        _accept_point(prob, st, l, d, "the initial point")
    return st


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


def _quality(d: dict, sampler: str) -> str:
    """The sampler's own quality figures for a log line: HMC acceptance and autocorrelation, or
    the importance weights' ESS fraction and largest share."""
    if sampler == "importance":
        return f"ess={d['ess_frac']:.2f} max_w={d['max_weight']:.1e}"
    return f"acc={d['accept']:.2f} rho1={d['rho1']:.2f}"


def _accept_point(prob: _Problem, st: _SolverState, l: int, diag: dict, where: str) -> None:
    """Record the run `diag` as layer l's accepted point: its sample scale ||h||²/P and, under
    importance sampling, its ESS fraction, the references the next step is judged against.

    An accepted point whose ESS fraction is below min_ess_frac raises InadequateSamplerError:
    the ESS of the prior proposal is a property of the point (it falls with P and with the
    tilt), not of the step, so backtracking cannot cure it, and carrying on would only freeze
    the iteration at a point the sampler cannot represent."""
    st.h2_last[l] = diag["h2_mean"]
    c = prob.solver
    if c.sampler != "importance":
        return
    ess = diag["ess_frac"]
    if ess < c.min_ess_frac:
        raise InadequateSamplerError(
            f"importance sampling from the prior is inadequate for this problem: at {where}, "
            f"layer {l} has ESS fraction {ess:.2g} < min_ess_frac={c.min_ess_frac:g} (P={prob.P}; "
            "the prior proposal only suits small P and mild tilts). Use solver.sampler='hmc', "
            "reduce meanfield.n_points, or raise solver.n_importance_samples together with "
            "lowering solver.min_ess_frac (ESS = ess_frac x draws; a proposal scale other "
            "than 1 lowered the ESS fraction in every case measured)"
        )
    st.ess_ref[l] = ess


def _healthy(
    diag: dict,
    h2_before: float,
    ess_before: float,
    n_tr: int,
    eps_ref_l: float,
    cfg: SolverConfig,
) -> bool:
    """A sampler run is healthy if the sample scale ||h||²/P has not blown up and (if moments
    were collected) the MC error is bounded; and, for HMC, it has few divergences, no resets
    and no step-size collapse; for importance sampling, the ESS fraction has not dropped below
    min_ess_drop times that of the accepted point before the step (the MC error cannot see
    a weight collapse, see fct.importance)."""
    growth = diag["h2_mean"] / max(h2_before, 1e-300)
    common = (growth <= cfg.max_h2_growth) and diag.get("psi_psi_rel_mc_err", 0.0) <= cfg.max_mc_err
    if cfg.sampler == "importance":
        return common and diag["ess_frac"] >= cfg.min_ess_drop * ess_before
    div_frac = diag["n_divergent"] / (cfg.n_chains * n_tr)
    return (
        common
        and (div_frac <= cfg.max_div_frac)
        and diag["n_reset"] == 0
        and diag["eps"] >= cfg.eps_collapse * eps_ref_l
    )


def _tilt_step(
    prob: _Problem, st: _SolverState, l, hat0, grad0, eta_l
) -> tuple[torch.Tensor, float]:
    """Φ̂_l ← Π(Φ̂_l + η̂_l ∂S/∂Φ̂_l / c_l²), Π the projection onto normalisable tilts, and
    ||Π(x) - x|| / ||x||, how much the projection changed the step's result x."""
    new = sym(hat0 + eta_l * grad0 / _c2(st.Phi[l]))
    c = prob.solver
    if not c.project_tilt:
        return new, 0.0
    projected = st.samplers[l].project_tilt(new, c.tilt_floor)
    return projected, relative_fro(projected, new) if new.norm().item() > 0 else 0.0


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
    residual: np.ndarray  # ||⟨ψψᵀ⟩ - Φ|| / ||Φ|| per layer, at the last accepted point
    first_residual: np.ndarray  # the same at the ascent's starting tilt, per layer
    grad_rel: np.ndarray  # ||∂S/∂Φ̂|| / ||Φ|| per layer
    n_backtrack: np.ndarray  # steps reverted because they broke the sampler, per layer
    n_accepted: np.ndarray  # steps validated and kept, per layer
    n_rejected: np.ndarray  # steps reverted because they raised the residual, per layer
    best_residual: np.ndarray  # smallest residual accepted so far in this ascent, per layer
    projected_rel: np.ndarray  # largest relative change by the tilt projection, per layer
    converged: bool
    n_steps: int


class _PendingStep(NamedTuple):
    """An inner step taken on one layer and not yet validated, with the point it left."""

    state: Any  # sampler state before the step
    hat: torch.Tensor  # Φ̂_l before the step
    grad: torch.Tensor  # ∂S/∂Φ̂_l measured there
    tol: float  # the convergence tolerance there (`_inner_tol`)


# Under inner_step_control="progress", the Monte Carlo part of the inner tolerance is capped at
# this: a kernel off by half its size is not converged however noisy its estimate, and a run
# whose MC error is that large cannot resolve the residual, so the ascent runs on and reports
# the layer unconverged instead of calling a pure-noise layer converged.
INNER_TOL_MC_CAP = 0.5


def _inner_tol(cfg: SolverConfig, diag: dict) -> float:
    """The inner residual below which a layer counts as converged: below
    inner_tol_mc × its own Monte Carlo error the residual is noise (that part capped at
    INNER_TOL_MC_CAP under inner_step_control="progress")."""
    mc = cfg.inner_tol_mc * diag["psi_psi_rel_mc_err"]
    if cfg.inner_step_control == "progress":
        mc = min(mc, INNER_TOL_MC_CAP)
    return max(cfg.inner_tol, mc)


def _eta_hat_floor(cfg: SolverConfig) -> float:
    """The smallest inner step size inner_step_control="progress" halves down to."""
    return cfg.eta_min_rel * cfg.eta_phihat


def _no_progress(
    cfg: SolverConfig, residual: float, best: float, step: _PendingStep, eta: float
) -> bool:
    """Whether inner_step_control="progress" rejects a step after which the residual is
    `residual`: it exceeds `best`, the smallest residual accepted so far in this ascent, by
    more than the convergence tolerance of the point before the step. Against the smallest,
    not the last, so a run of small accepted rises cannot let the residual creep up. The
    tolerance is the margin because the residuals are Monte Carlo estimates (and `best` a
    lucky low one), so a strict "did not rise" would reject sound steps on noise alone. A step
    already at the floor `_eta_hat_floor` is never rejected, so noise cannot stall the ascent."""
    rose = residual > best + step.tol
    return cfg.inner_step_control == "progress" and rose and eta > _eta_hat_floor(cfg)


def _retry_half_step(prob: _Problem, st: _SolverState, l: int, step: _PendingStep, res) -> None:
    """Revert layer l's pending inner step (tilt and chain state) and take it again from the
    same point with half the step size. Under inner_step_control="progress" the halving stops
    at `_eta_hat_floor`, for a step reverted for sampler health as for one reverted for lack
    of progress: a step that still breaks the sampler at the floor is retried there until
    max_backtrack raises."""
    cfg, s = prob.solver, st.samplers[l]
    floor = _eta_hat_floor(cfg) if cfg.inner_step_control == "progress" else 0.0
    st.eta_hat[l] = max(st.eta_hat[l] * 0.5, floor)
    st.Phi_hat[l], rel = _tilt_step(prob, st, l, step.hat, step.grad, st.eta_hat[l])
    res.projected_rel[l] = max(res.projected_rel[l], rel)
    s.set_tilt(st.Phi_hat[l])
    s.set_state(step.state)


def _inner_layer_step(prob, st: _SolverState, l, inner, pending, res: _InnerResult) -> dict:
    """One ascent step on Φ̂_l.  The sampler run at its top also validates the pending step
    (retrospective backtracking): if that step broke the sampler, or (inner_step_control
    "progress", `_no_progress`) raised the residual ||⟨ψψᵀ⟩ - Φ_l|| / ||Φ_l||, it is reverted
    and retried with half the step size; the step size grows only after an accepted step.
    Updates `pending[l]` and `res` in place; returns the run's diagnostics."""
    cfg, s = prob.solver, st.samplers[l]
    moments, diag = s.run(cfg.n_trans)
    residual = relative_fro(moments["psi_psi"], st.Phi[l])
    if inner == 0:
        res.first_residual[l] = residual
    step = pending[l]
    if step is not None:
        if not _healthy(diag, st.h2_last[l], st.ess_ref[l], cfg.n_trans, st.eps_ref[l], cfg):
            res.n_backtrack[l] += 1
            _retry_half_step(prob, st, l, step, res)
            if res.n_backtrack[l] > cfg.max_backtrack * (inner + 1):
                raise RuntimeError(
                    f"layer {l}: inner step keeps breaking the sampler "
                    f"(eta_phihat={st.eta_hat[l]:.2e})"
                )
            return diag
        if _no_progress(cfg, residual, res.best_residual[l], step, st.eta_hat[l]):
            res.n_rejected[l] += 1
            _retry_half_step(prob, st, l, step, res)
            return diag
    _accept_point(prob, st, l, diag, f"outer step {st.outer}, inner step {inner}")
    if step is not None:
        res.n_accepted[l] += 1
        st.eta_hat[l] = min(st.eta_hat[l] * cfg.eta_grow, cfg.eta_max_phihat)
    grad_hat = 0.5 * (moments["psi_psi"] - st.Phi[l])
    res.residual[l] = residual
    res.best_residual[l] = min(res.best_residual[l], residual)
    res.grad_rel[l] = grad_hat.norm().item() / max(st.Phi[l].norm().item(), 1e-300)
    pending[l] = _PendingStep(s.get_state(), st.Phi_hat[l].clone(), grad_hat, _inner_tol(cfg, diag))
    st.Phi_hat[l], rel = _tilt_step(prob, st, l, st.Phi_hat[l], grad_hat, st.eta_hat[l])
    res.projected_rel[l] = max(res.projected_rel[l], rel)
    s.set_tilt(st.Phi_hat[l])
    return diag


def _new_inner_result(L: int) -> _InnerResult:
    """The record of an inner ascent before its first step."""
    return _InnerResult(
        residual=np.full(L, np.nan),
        first_residual=np.full(L, np.nan),
        grad_rel=np.full(L, np.nan),
        n_backtrack=np.zeros(L, dtype=int),
        n_accepted=np.zeros(L, dtype=int),
        n_rejected=np.zeros(L, dtype=int),
        best_residual=np.full(L, np.inf),
        projected_rel=np.zeros(L),
        converged=False,
        n_steps=0,
    )


def _inner_ascent(prob: _Problem, st: _SolverState) -> _InnerResult:
    """Gradient ascent on Φ̂ at fixed Φ, ∂S/∂Φ̂_l = ½(⟨ψψᵀ⟩_l - Φ_l), until every layer's
    residual is below max(inner_tol, inner_tol_mc × its own MC error) or n_inner_max steps."""
    cfg, L = prob.solver, prob.L
    res = _new_inner_result(L)
    pending: list[_PendingStep | None] = [None] * L
    for inner in range(cfg.n_inner_max):
        diags = [_inner_layer_step(prob, st, l, inner, pending, res) for l in range(L)]
        res.n_steps = inner + 1
        tol = np.array([_inner_tol(cfg, d) for d in diags])
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
    hmc_diag: list[dict]  # sampler diagnostics of the fresh n_trans_outer run


def _outer_gradients(prob: _Problem, st: _SolverState) -> _OuterGradients:
    """∂S/∂Φ_l from a fresh, longer run under the final Φ̂:
    l < L: -½Φ̂_l + ½ sym(⟨u hᵀ⟩_{l+1} Φ_l⁻¹)  (Eq. 31b, Stein form);
           the direct form s(Σ⁻¹ - Σ⁻¹⟨hhᵀ⟩_{l+1}Σ⁻¹), s = prior_scale g², is a diagnostic;
    l = L: -½Φ̂_L - c A Y Yᵀ A                 (Eq. 31c)."""
    cfg, L = prob.solver, prob.L
    moments, hmc_diag = [], []
    for l, s in enumerate(st.samplers):
        m, d = s.run(cfg.n_trans_outer)
        _accept_point(prob, st, l, d, f"outer step {st.outer} (outer-gradient run)")
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


def _phi_step(
    cfg: SolverConfig, eta: float, grad: torch.Tensor, Phi_at: torch.Tensor, Phi_now: torch.Tensor
) -> torch.Tensor:
    """The descent step of Φ_l, before clipping, from ∂S/∂Φ_l (units of Φ̂, i.e. 1/Φ) at Φ_at.
    "kernel": η Φ_at ∂S/∂Φ_l Φ_at, the Newton direction of a Gaussian layer (its tilt follows
    the kernel as dΦ̂ = -Φ⁻¹ dΦ Φ⁻¹, so η = 2 is a full Newton step), which moves every
    eigendirection of Φ at its own rate; "trace": η ∂S/∂Φ_l c_l² with c_l = tr(Φ_now)/P, the
    same step when Φ = c I, and far too small along the top eigenvectors of a low-rank kernel.
    Φ_now is the current Φ_l (the last attempted step when backtracking), which "trace" reads."""
    if cfg.phi_step_scale == "trace":
        return eta * grad * _c2(Phi_now)
    return eta * sym(Phi_at @ grad @ Phi_at)


def _phi_step_rel(
    cfg: SolverConfig, grad: torch.Tensor, Phi_prev: torch.Tensor, Phi_now: torch.Tensor
) -> float:
    """||ΔΦ_l|| / ||Φ_l|| per unit η before clipping (history["outer_grad_rel"])."""
    if cfg.phi_step_scale == "trace":
        size = grad.norm().item() * _c2(Phi_now)
    else:
        size = _phi_step(cfg, 1.0, grad, Phi_prev, Phi_now).norm().item()
    return size / max(Phi_prev.norm().item(), 1e-300)


def _trust_region_step(
    prob: _Problem, st: _SolverState, grads: list[torch.Tensor], Phi_prev: torch.Tensor
) -> tuple[list[int], list[float], list[bool]]:
    """Φ_l ← Π_PD(Φ_l^prev - ΔΦ_l) with ΔΦ_l = `_phi_step` at Φ_l^prev, clipped to
    max_rel_step ||Φ_l^prev||."""
    cfg = prob.solver
    n_clamped, lam_min, clipped = [], [], []
    for l in range(prob.L):
        step = _phi_step(cfg, st.eta, grads[l], Phi_prev[l], st.Phi[l])
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
        _, d = st.samplers[l].run(prob.solver.n_trans, collect=False)
        ok = ok and _healthy(
            d, st.h2_last[l], st.ess_ref[l], prob.solver.n_trans, st.eps_ref[l], prob.solver
        )
    return ok


def _outer_descent(prob, st: _SolverState, grads, Phi_prev, outer, log: Log) -> _OuterStep:
    """One trust-region descent step on Φ.  If it breaks a sampler, revert (priors and chain
    states) and halve η, up to max_backtrack times or until η = eta_min_rel · eta_phi; an
    accepted first try grows η by eta_grow up to eta_max."""
    cfg = prob.solver
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
        "n_accepted_inner": inner.n_accepted.copy(),
        "n_rejected_inner": inner.n_rejected.copy(),
        "outer_grad_rel": np.array(
            [_phi_step_rel(prob.solver, og.grads[l], Phi_prev[l], Phi[l]) for l in range(L)]
        ),
        "phihat_fp_resid_stein": np.array(og.fp_stein),
        "phihat_fp_resid_direct": np.array(og.fp_direct),
        "stein_vs_direct": np.array(og.stein_vs_direct),
        "picard_resid": (
            float(inner.first_residual[-1]) if prob.solver.inner_start == "target" else float("nan")
        ),
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
        "is_max_weight": np.array([d.get("max_weight", np.nan) for d in diags]),
        "phi_change_rel": np.array([relative_fro(Phi[l], Phi_prev[l]) for l in range(L)]),
        # nan, not a ratio against 1e-300, when the tilt was 0 before the step (lazy init)
        "phihat_change_rel": np.array(
            [
                relative_fro(Phi_hat[l], Phi_hat_prev[l])
                if Phi_hat_prev[l].norm().item() > 0
                else float("nan")
                for l in range(L)
            ]
        ),
        "tilt_projected_rel": inner.projected_rel.copy(),
    }
    if st.net_Phi is not None:
        rec["align_net"] = np.array([kernel_alignment(Phi[l], st.net_Phi[l]) for l in range(L)])
        rec["dist_net"] = np.array([relative_fro(Phi[l], st.net_Phi[l]) for l in range(L)])
    return rec


def _format_step(
    rec: dict, inner: _InnerResult, step: _OuterStep, st: _SolverState, prob: _Problem
) -> str:
    sampler = prob.solver.sampler
    # A bounded phi is normalisable for any tilt: its margin is only lambda_min(Sigma^-1).
    margin = "n/a" if known_tilt_class(prob.phi) == "bounded" else None

    def fmt(a):
        return "[" + " ".join(f"{v:.2e}" for v in a) + "]"

    mse_net = rec["train_mse_net"]
    msg = (
        f"[{rec['outer_step']:04d}] t={rec['time']:.0f}s "
        f"inner={inner.n_steps:3d}{'' if inner.converged else '*'} "
        f"res_in={fmt(inner.residual)} grad_out={fmt(rec['outer_grad_rel'])} "
        f"fp(Φ̂)={fmt(rec['phihat_fp_resid_stein'])} "
        + (f"picard={rec['picard_resid']:.2e} " if prob.solver.inner_start == "target" else "")
        + f"mse={rec['train_mse_theory']:.3e}"
        + (f" (net {mse_net:.3e})" if mse_net is not None else "")
        + f" | eta={st.eta:.1e}/{fmt(st.eta_hat)} bt={inner.n_backtrack.tolist()}/{step.n_backtrack}"
        + f" rej={inner.n_rejected.tolist()}"
        + (
            f" | IS ess={fmt(rec['hmc_ess_frac'])} max_w={fmt(rec['is_max_weight'])} "
            if sampler == "importance"
            else f" | HMC acc={fmt(rec['hmc_accept'])} eps={fmt(rec['hmc_eps'])} "
            f"div={rec['hmc_div']} rho1={fmt(rec['hmc_rho1'])} "
        )
        + f"mcerr={fmt(rec['hmc_mc_err'])} "
        f"stein={fmt(rec['hmc_stein_mismatch'])} h2/prior={fmt(rec['hmc_h2_ratio'])} "
        f"margin={margin or fmt(rec['norm_margin'])} projected={fmt(rec['tilt_projected_rel'])}"
    )
    if "align_net" in rec:
        msg += f" | align={fmt(rec['align_net'])}"
    if sum(step.n_clamped):
        msg += f" | PD-clamped={step.n_clamped}"
    if any(step.clipped):
        msg += f" | step-clipped={[int(c) for c in step.clipped]}"
    if prob.solver.inner_step_control == "progress":
        at_floor = [l for l, e in enumerate(st.eta_hat) if e <= _eta_hat_floor(prob.solver)]
        if at_floor:
            msg += f" | eta_phihat at its floor in layers {at_floor}"
    return msg


def _start_last_tilt_at_target(prob: _Problem, st: _SolverState) -> float:
    """Φ̂_L ← Π(T), T = -2c A Y Yᵀ A the last layer's Stein target at the current Φ_L (exact,
    a function of Φ_L only; Π the tilt projection), as the start of the inner ascent; returns
    ||Π(T) - T|| / ||T||.

    The inner residual pins Φ̂_L only along directions the kernel responds to. Along the others
    (most of them, for a low-rank kernel or a saturating φ) every inner step adds Monte Carlo
    noise that nothing removes, and continued from step to step it piles up. Restarted at T,
    it cannot. What the iteration then is: the ascent validates each step at the next run, so
    it ends with at least one step past its last measurement, and the outer gradient run sees
    Φ̂_L = T + Δ, Δ = the inner steps from T (one step, η̂ (⟨ψψᵀ⟩_T - Φ_L) / (2 c²), whenever
    the residual at T is already below tolerance). The outer step ΔΦ_L = (η/2) Φ_L Δ Φ_L is a
    damped stochastic Picard step towards ⟨ψψᵀ⟩ under T, and the recorded fixed-point residual
    ||Φ̂_L - T|| / ||T|| measures Δ, which scales with η̂: it is not a convergence measure for
    this layer. history["picard_resid"], the residual ||⟨ψψᵀ⟩_T - Φ_L|| / ||Φ_L|| of the
    ascent's first run, is."""
    A, _, _, _ = _learning_pieces(prob, st.Phi[-1])
    target = sym(-2.0 * prob.lp * A @ prob.Y @ prob.Y.T @ A)
    rel = 0.0
    if prob.solver.project_tilt:
        projected = st.samplers[-1].project_tilt(target, prob.solver.tilt_floor)
        rel = relative_fro(projected, target) if target.norm().item() > 0 else 0.0
        target = projected
    st.Phi_hat[-1] = target
    st.samplers[-1].set_tilt(target)
    return rel


def _outer_step(prob: _Problem, st: _SolverState, t_start: float, log: Log) -> None:
    """One iteration of Algorithm 1: inner ascent on Φ̂, one descent step on Φ, diagnostics.
    Advances st.outer by one."""
    outer = st.outer
    Phi_prev, Phi_hat_prev = st.Phi.clone(), st.Phi_hat.clone()
    start_projected = 0.0
    if prob.solver.inner_start == "target":
        start_projected = _start_last_tilt_at_target(prob, st)
    inner = _inner_ascent(prob, st)
    inner.projected_rel[-1] = max(inner.projected_rel[-1], start_projected)
    og = _outer_gradients(prob, st)
    step = _outer_descent(prob, st, og.grads, Phi_prev, outer, log)
    rec = _record(prob, st, outer, time.time() - t_start, inner, og, step, Phi_prev, Phi_hat_prev)
    for k in st.history:
        st.history[k].append(rec[k])
    if outer % prob.solver.log_int == 0:
        log(_format_step(rec, inner, step, st, prob))
    st.outer = outer + 1


# ----------------------------------------------------------------------------- (de)serialisation
def _snapshot(st: _SolverState) -> OuterSnapshot:
    """The state after the outer step just run, copied (see `OuterSnapshot`)."""
    history = {k: list(v) for k, v in st.history.items()}
    return OuterSnapshot(st.outer - 1, history, st.Phi.clone(), st.Phi_hat.clone())


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


def _to_payload(st: _SolverState, cfg: MeanFieldConfig) -> dict:
    """The checkpoint of `st`.  `meanfield_config` records the full section (every solver
    field, also those the cache hash leaves out) for provenance; nothing reads it back."""
    return {
        "meanfield_config": cfg.model_dump(),
        "outer": st.outer,
        "Phi": st.Phi,
        "Phi_hat": st.Phi_hat,
        "history": _history_to_lists(st.history),
        "eta": st.eta,
        "eta_hat": st.eta_hat.tolist(),
        "h2_last": list(st.h2_last),
        "eps_ref": list(st.eps_ref),
        "ess_ref": list(st.ess_ref),
        "samplers": [s.state_dict() for s in st.samplers],
        "generator": st.gen.get_state(),
        "mse_net": st.mse_net,
        "net_Phi": st.net_Phi,
        "elapsed": st.elapsed,
    }


def _restore_state(
    prob: _Problem, payload: dict, gen: torch.Generator, path: Path, with_net: bool
) -> _SolverState:
    """Rebuild the samplers at the saved (Φ, Φ̂), load their states and the generator state.
    No warm-up: saved HMC chains are already equilibrated (importance draws need none)."""
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
        # HMC checkpoints written before the importance sampler existed have no ess_ref, and
        # HMC never reads it.
        ess_ref=list(payload.get("ess_ref", [float("nan")] * prob.L)),
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
    on_outer: OnOuter | None = None,
) -> SolveResult:
    """Solve  min_Φ max_Φ̂ S(Φ, Φ̂)  by Algorithm 1 of arXiv:2502.07998 with the
    single-site expectations estimated by persistent HMC (cfg.solver.sampler="hmc") or by
    importance sampling from the prior ("importance").

    Step sizes are dimensionless: with c_l = tr(Φ_l)/P,
        Φ̂_l += eta_phihat_l · ∂S/∂Φ̂_l / c_l²      (Φ̂ has units of 1/Φ)
        Φ_l  -= eta_phi      · Φ_l ∂S/∂Φ_l Φ_l      (clipped to ||ΔΦ_l|| ≤ max_rel_step ||Φ_l||)
    (solver.phi_step_scale="trace": Φ_l -= eta_phi · ∂S/∂Φ_l · c_l², the same step when
    Φ_l = c_l I), and are adapted: every sampler run also validates the step taken before it.
    If the HMC chains show divergences (> max_div_frac of transitions) or resets, the
    importance ESS fraction drops below min_ess_drop times that of the point before, or the
    sample scale ||h||²/P jumps by more than max_h2_growth, the step is reverted (kernel and
    chain state) and its step size halved; so is an inner step that raised the layer's inner
    residual by more than the convergence tolerance of the point before it
    (inner_step_control="progress"; not below eta_min_rel · eta_phihat). Otherwise the step
    size grows by eta_grow up to eta_max (outer) or eta_max_phihat (inner).  This keeps the
    iteration on the normalisable side of the tilt.  With inner_start="target" every inner
    ascent of the last layer starts from its Stein target -2c A Y Yᵀ A at the current Φ_L,
    which makes the last layer's outer step a damped stochastic Picard step; its convergence
    measure is then history["picard_resid"] (`_start_last_tilt_at_target`).
    The settings are cfg.solver's.
    Under importance sampling every accepted point (the initial one, each validated inner
    step, each outer-gradient run) must have ESS fraction >= min_ess_frac, else
    InadequateSamplerError: the prior proposal does not suit the problem.
    The inner loop stops when every layer's residual ||⟨ψψᵀ⟩-Φ||/||Φ|| is below
    max(inner_tol, inner_tol_mc × its own Monte Carlo error): below that it is noise.

    phi is a callable: any elementwise activation without tilt projection; with it (the
    default) one whose normalisability rule is known (`fct.hmc.tilt_rule`: tanh, sigmoid,
    relu, `fct.activations.identity`), else ValueError. The tilt tripwire: a solve whose Φ̂
    fixed-point residual stays >= cfg.solver.tilt_tripwire_resid over the last
    cfg.solver.tilt_tripwire_steps outer steps (in some layer, at their largest) raises
    UnconvergedSolveError; it catches a lost tilt and is not a convergence test.  All
    randomness comes from one torch.Generator seeded with cfg.solver.seed on X's device.

    With checkpoint_dir (requires config_hash): return its final.pt if present, else resume
    from its latest ckpt_*.pt if present; write ckpt_<outer:05d>.pt every cfg.solver.save_int
    outer steps, and final.pt at the end (which then replaces the checkpoints); each holds
    cfg in full for provenance.  A saved state must have been written with a net iff `net`
    is given.  Without checkpoint_dir, nothing touches disk.

    `on_outer`, if given, is called with an `OuterSnapshot` after every outer step run in this
    call (not for steps restored from a checkpoint, and not at all for a loaded final.pt), for
    live plots.  It only observes: it receives copies, draws nothing from the generator, and
    runs after the step's history entry and before its checkpoint is written, so results are
    identical with and without it, except that the time it takes is charged to history["time"] of
    the following steps.  An exception it raises stops the solve.
    """
    if checkpoint_dir is not None and not config_hash:
        raise ValueError("checkpoint_dir needs a non-empty config_hash to guard against reuse")
    t_call = time.time()
    prob = _make_problem(X, Y, L, lambda_w, lambda_r, phi, g, cfg)
    gen = torch.Generator(device=prob.dev).manual_seed(cfg.solver.seed)

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
    while st.outer < cfg.solver.n_outer:
        _outer_step(prob, st, t_start, log)
        st.elapsed = time.time() - t_start
        if on_outer is not None:
            on_outer(_snapshot(st))
        if checkpoint_dir is not None and st.outer % cfg.solver.save_int == 0:
            write_checkpoint(
                checkpoint_dir, CKPT_PREFIX, st.outer, _to_payload(st, cfg), config_hash, CKPT_WIDTH
            )
    _check_tilt_tripwire(st.history, cfg.solver.tilt_tripwire_resid, cfg.solver.tilt_tripwire_steps)
    if checkpoint_dir is not None:
        _save_final(Path(checkpoint_dir), _to_payload(st, cfg), config_hash)
    return SolveResult(st.Phi, st.Phi_hat, st.history, st.samplers)


def _check_tilt_tripwire(history: History, resid_max: float, n_steps: int) -> None:
    """UnconvergedSolveError if, in some layer, the largest Stein fixed-point residual of Φ̂
    over the last n_steps outer steps is >= resid_max: a tilt lost (near 0, residual near 1) or
    far off its fixed point, which must not stand in for a saddle point (no final.pt is written;
    the checkpoints stay for inspection). The largest, so a residual oscillating around the
    tripwire trips it. Passing is not evidence of convergence."""
    last = np.asarray(history["phihat_fp_resid_stein"][-n_steps:], dtype=float)  # (k, L)
    resid = last.max(axis=0)
    if not (resid < resid_max).all():
        projected = history.get("tilt_projected_rel", [None])[-1]
        raise UnconvergedSolveError(
            f"tilt tripwire: the largest Stein fixed-point residual of Phi_hat over the last "
            f"{len(last)} outer steps is {np.array2string(resid, precision=3)} per layer "
            f"(tilt_tripwire_resid={resid_max:g}), so Phi_hat has lost its tilt or is far from "
            f"its fixed point; the tilt projection changed Phi_hat by {projected} (relative, per "
            "layer) in the last step. Check the sampler and step-size diagnostics, run more "
            "outer steps, or raise solver.tilt_tripwire_resid to accept the result"
        )


def _save_final(checkpoint_dir: Path, payload: dict, config_hash: str) -> None:
    """Write final.pt, then delete the checkpoints: final.pt supersedes them (the last one
    holds the same payload, which is large at production size)."""
    save_artefact(checkpoint_dir / FINAL_NAME, payload, config_hash)
    for ckpt in checkpoint_dir.glob(f"{CKPT_PREFIX}*.pt"):
        ckpt.unlink()


def meanfield_hash(cfg: Config) -> str:
    return cfg.section_hash("data", "net", "train", "meanfield")


def run_meanfield(
    cfg: Config,
    dataset: Dataset,
    net: MLP,
    cache_dir: Path,
    log: Log = print,
    on_outer: OnOuter | None = None,
) -> SolveResult:
    """Solve on the first cfg.meanfield.n_points training points, caching under
    CACHE_DIR/meanfield (returns final.pt if present, resumes from a checkpoint otherwise).
    The constants of the action follow the network (`network_constants`, beta from
    `resolve_beta`) unless `meanfield` sets them.
    `on_outer` is passed to `solve`.
    Raises SpatialPriorUnsupportedError if cfg.spatial is set (checked before any file access)."""
    if cfg.spatial is not None:
        raise SpatialPriorUnsupportedError(
            "the mean-field solver has no spatial prior; unset `spatial` in the config"
        )
    X, Y = _solver_inputs(dataset, cfg)
    meanfield = cfg.meanfield.model_copy(update={"beta": resolve_beta(cfg)})
    return solve(
        X,
        Y,
        cfg.net.L,
        cfg.train.weight_lambda,
        cfg.train.rate_lambda,
        get_phi(cfg.net.phi),
        cfg.net.g,
        meanfield,
        net=net,
        log=log,
        checkpoint_dir=cache_paths(cache_dir).meanfield,
        config_hash=meanfield_hash(cfg),
        on_outer=on_outer,
    )


def load_meanfield(
    cache_dir: Path, cfg: Config, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor, History]:
    """(Φ, Φ̂, history) of a finished solve; history entries are numpy arrays again.
    Raises FileNotFoundError if there is no result, StaleCacheError if the config changed."""
    path = cache_paths(cache_dir).meanfield / FINAL_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run `python -m fct solve` first")
    payload = load_artefact(path, meanfield_hash(cfg), device)
    return payload["Phi"], payload["Phi_hat"], _history_from_lists(payload["history"])
