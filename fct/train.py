"""Full-batch training of the MLP with periodic checkpoints and resume.

Layout under `cache_dir/train/`: `ckpt_<epoch>.pt` (only the latest kept) and `final.pt`.
Both share one payload shape; resuming and `load_trained` read them through the same
validating loader (`_load_payload`) and write the net with `load_state_dict`. The state dict
holds the trained parameters under the same names in both parametrisations of the spatial
prior: the weights under "penalty", xi under "whitened" (`make_net_from_config` rebuilds the
whitening factors from the config, as `make_weight_penalty` rebuilds the penalty matrices).
"""

import hashlib
import pickle
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from torch import Tensor

from fct.cache import (
    StaleCacheError,
    cache_paths,
    latest_checkpoint,
    load_artefact,
    read_field,
    save_artefact,
    write_checkpoint,
)
from fct.config import Config, TrainConfig, law_beta, resolve_device, resolve_dtype
from fct.data import Dataset
from fct.model import MLP, EnsembleMLP, Net, make_net
from fct.sgld import SGLD
from fct.spatial import (
    build_spaces,
    floored_whitening_factors,
    parameter_factors,
    weight_quadratic_form,
    whitening_factors,
)

# Monitor entries are lists over layers, the rest numbers; for an ensemble every entry but
# `epoch` and `monitor_epoch` has a member axis first (see `train`).
History = dict[str, list]
Monitor = Callable[[MLP], dict[str, list]]  # always called on a single network
# The prior's quadratic form, before its coefficient: 0-d for an MLP, (B,) for an ensemble.
WeightPenalty = Callable[[Net], Tensor]
MONITOR_PREFIX = "monitor_"  # history keys of monitor results: monitor_epoch, monitor_<key>
MONITOR_EPOCH = f"{MONITOR_PREFIX}epoch"
_HISTORY_KEYS = ("epoch", "mse", "weight_l2", "rate_l2", "loss")
# Extra history keys of SGLD runs: the gradient norm and the mean squared parameter
# displacement per step (see `SGLD.last_msd`), both of the step taken at the logged epoch.
_SGLD_KEYS = ("grad_norm", "param_msd")
_REQUIRED_KEYS = ("epoch", "net_state_dict", "optim_state_dict", "history", "rng_state")
# Lanczos for the run-start curvature check under whitening (`largest_hessian_eigenvalue`): at
# most this many Hessian-vector products, and convergence once the top Ritz pair's residual is
# below this fraction of the largest Ritz value's magnitude (or 64 machine epsilons of the
# parameters' dtype, if larger).
HESSIAN_MAX_ITERATIONS = 300
HESSIAN_RTOL = 1e-8


def make_net_from_config(
    cfg: Config,
    D_in: int,
    D_out: int,
    device: torch.device,
    dtype: torch.dtype,
    members: range | None = None,
) -> Net:
    """The net `cfg` describes for `D_in` inputs and `D_out` outputs. The one place a config
    turns into a net, for training and for loading alike. `members` builds only those members
    of the ensemble (`fct.model.make_net`), each exactly as in the whole one.

    Every path starts from the same standard-normal draw z of `make_net(cfg.net)`. Under
    `net.init = "standard"` the parameters are z (the only start before `net.init` existed).
    Under "prior" W is drawn from the prior the loss encodes, so every direction the data does
    not touch starts in equilibrium: whitened, xi = z / sqrt(weight_lambda) with the factors
    S = K^{1/2}; under the penalty, W = S_fl,rows z S_fl,cols / sqrt(weight_lambda) with the
    square roots of the floored kernels M^{-1} (`floored_whitening_factors`), which is the
    whitened start's W bit for bit when nothing is floored; without a spatial prior, W =
    z / sqrt(weight_lambda)."""
    prior = cfg.net.init == "prior"
    init_std = cfg.train.weight_lambda**-0.5 if prior else 1.0  # > 0: `Config` checks it
    if cfg.spatial is None or not prior:
        return make_net(cfg.net, D_in, D_out, device, dtype, init_std=init_std, members=members)
    spaces = build_spaces(cfg.spatial, m=D_in, N=cfg.net.N, L=cfg.net.L, n_out=D_out)
    if cfg.is_whitened:
        factors = whitening_factors(spaces)
        return make_net(
            cfg.net,
            D_in,
            D_out,
            device,
            dtype,
            whitening=factors,
            init_std=init_std,
            members=members,
        )
    net = make_net(cfg.net, D_in, D_out, device, dtype, init_std=init_std, members=members)
    return _correlate(net, floored_whitening_factors(spaces))


def _correlate(net: Net, factors: dict[str, tuple[Tensor, Tensor]]) -> Net:
    """`net` with every parameter p replaced, in place, by S_rows p S_cols: the product
    `Whitening.whiten` forms, so the same factors give the same weights."""
    with torch.no_grad():
        for name, p in net.named_parameters():
            rows, cols = (S.to(p.device, p.dtype) for S in factors[name])
            p.copy_(rows @ p @ cols)
    return net


def _check_parametrisation_matches(cfg: Config, net: Net) -> None:
    """A net with whitening factors is trained only under "whitened" and a net without only
    under the penalty (or no prior): the loss and the readers would otherwise silently take
    xi for W or W for xi."""
    has = net.whitening is not None
    if has != cfg.is_whitened:
        parametrisation = None if cfg.spatial is None else cfg.spatial.parametrisation
        raise ValueError(
            f"the net {'has' if has else 'has no'} whitening factors but the config's "
            f"spatial.parametrisation is {parametrisation!r}; build the net with "
            "make_net_from_config"
        )


def prior_coefficient(tc: TrainConfig) -> float:
    """weight_lambda / (2 beta): the coefficient of the weight prior's quadratic form in the loss.

    beta is the SGLD inverse temperature `tc.beta`, and 1 for Adam (which has no temperature).
    The SGLD stationary law is exp(-beta L), so dividing by beta here leaves the prior
    exp(-weight_lambda/2 * form), i.e. weights N(0, K / weight_lambda), for every beta; only
    the data term of the law carries the factor beta (the rate term is scaled the same way,
    `rate_coefficient`).
    """
    return tc.weight_lambda / (2.0 * law_beta(tc.optimizer, tc.beta))


def rate_coefficient(tc: TrainConfig) -> float:
    """rate_lambda / (2 beta): the coefficient of the rate penalty sum phi^2 in the loss.

    beta as in `prior_coefficient`. The law exp(-beta L) then carries the tilt
    exp(-rate_lambda/2 * sum phi^2) for every beta, so `tc.rate_lambda` is the notes' rate
    coefficient r (G = r I) and the mean-field solver's `rate_tilt`.
    """
    return tc.rate_lambda / (2.0 * law_beta(tc.optimizer, tc.beta))


@dataclass(frozen=True)
class LossCoefficients:
    """The loss's two penalty coefficients: `prior` multiplies the weight prior's quadratic form
    and `rate` the rate penalty sum phi^2. A type of its own, so a bare (weight_lambda,
    rate_lambda) pair cannot be passed to `compute_loss` unscaled: build it with
    `loss_coefficients`, or by name where a test wants chosen values."""

    prior: float
    rate: float


def loss_coefficients(tc: TrainConfig) -> LossCoefficients:
    """The coefficients the config's loss carries: `prior_coefficient` and `rate_coefficient`."""
    return LossCoefficients(prior=prior_coefficient(tc), rate=rate_coefficient(tc))


def compute_loss(
    net: Net,
    x: Tensor,
    y: Tensor,
    coefficients: LossCoefficients,
    weight_penalty: WeightPenalty | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """0.5*N*mse + coefficients.prior*weight_l2 + coefficients.rate*rate_l2, plus its parts.

    For an ensemble every part is (B,), member b's own value, and the returned loss is the
    SUM of the members' losses: member b's parameters enter only its own term, so the
    gradient with respect to them is exactly that of its loss (not 1/B of it).
    `weight_penalty` None is the net's own `weight_l2`.

    Callers pass `loss_coefficients(train_config)`: weight_lambda / (2 beta) and
    rate_lambda / (2 beta), never the raw `train.weight_lambda` or `train.rate_lambda`.

    `rate_l2` is the squared rates phi(h) summed over samples, neurons and layers, so it grows
    with the number of samples P. `weight_l2` is `weight_penalty(net)`: by default the plain
    sum of squares of every parameter, with a spatial prior the sum over layers of
    tr(M_post W M_pre W^T) (`make_weight_penalty` picks the one the config asks for).

    The parts are detached 0-d tensors: calling `.item()` here would force a host-device
    sync every epoch, so the caller converts only when it logs.
    """
    if not isinstance(coefficients, LossCoefficients):
        raise TypeError(
            f"coefficients must be LossCoefficients (`loss_coefficients(train_config)`), got "
            f"{coefficients!r}"
        )
    h, z = net(x)
    mse = per_member(net, (y - z).square(), torch.mean)
    rate_l2 = per_member(net, net.phi(h).square())
    weight_l2 = (type(net).weight_l2 if weight_penalty is None else weight_penalty)(net)
    if weight_l2.shape != mse.shape:
        raise ValueError(
            f"the weight penalty has shape {tuple(weight_l2.shape)} but the loss needs "
            f"{tuple(mse.shape)} (one value per member for an ensemble)"
        )
    loss = 0.5 * net.N * mse + coefficients.prior * weight_l2 + coefficients.rate * rate_l2
    parts = {"mse": mse, "weight_l2": weight_l2, "rate_l2": rate_l2, "loss": loss}
    return loss.sum(), {k: v.detach() for k, v in parts.items()}


def per_member(net: Net, t: Tensor, reduce: Callable[..., Tensor] = torch.sum) -> Tensor:
    """`reduce` (sum, mean) of `t` over every axis but the net's member axis: 0-d for a
    single network, (B,) for an ensemble, whose tensors carry the member axis first."""
    return reduce(t, dim=tuple(range(net.member_axes, t.ndim)))


def spatial_weight_penalty(net: Net, factors: dict[str, tuple[Tensor, Tensor]]) -> Tensor:
    """Sum over the parameters of tr(M_rows P M_cols P^T), `factors[name] = (M_rows, M_cols)`
    in each parameter's stored orientation (`fct.spatial.parameter_factors`); per member,
    (B,), for an ensemble. Summed in parameter order like `MLP.weight_l2`, so identity
    factors reproduce it bit for bit."""
    terms = [weight_quadratic_form(p, *factors[name]) for name, p in net.named_parameters()]
    return torch.stack(terms).sum(0)


def prior_stiffness(factors: dict[str, tuple[Tensor, Tensor]]) -> float:
    """lambda_max(M_rows) lambda_max(M_cols) of the stiffest parameter: the prior term
    c * tr(M_rows W M_cols W^T) has curvature 2 c times this along its stiffest direction. A
    wide kernel makes it large (up to 1 / min_rel_eig over the largest eigenvalue of K per
    factor). The whitened prior c |xi|^2 has stiffness 1 in every direction."""
    return max(
        torch.linalg.eigvalsh(rows)[-1].item() * torch.linalg.eigvalsh(cols)[-1].item()
        for rows, cols in factors.values()
    )


def _check_sgld_step_resolves_the_prior(stiffness: float, prior: str, tc: TrainConfig) -> None:
    """Raise if an SGLD step of size lr is unstable on the prior term alone, whose curvature
    along its stiffest direction is 2 c `stiffness` (c = `prior_coefficient(tc)` = weight_lambda
    / (2 beta); `prior_stiffness`): an Euler step on a quadratic of curvature k diverges once
    lr * k >= 2, so lr * c * stiffness >= 1 fails at the start instead of after the run has
    filled with NaN. `prior` names the prior in the message. Passing it is necessary, not
    sufficient: the data and rate terms add curvature of their own
    (`_check_sgld_step_resolves_the_loss`). Adam rescales its steps and needs no such check.
    """
    step = tc.lr * prior_coefficient(tc) * stiffness
    if step >= 1:
        raise ValueError(
            f"SGLD with lr={tc.lr:g} diverges on the {prior} alone: lr * weight_lambda / "
            f"(2 beta) * stiffness = {step:.3g} must be < 1 (stiffness {stiffness:.3g}); lower "
            "lr or weight_lambda, or raise beta"
        )


def largest_hessian_eigenvalue(
    loss_fn: Callable[[], Tensor],
    params: Iterable[Tensor],
    max_iter: int = HESSIAN_MAX_ITERATIONS,
    seed: int = 0,
) -> float:
    """The largest (most positive) eigenvalue of the Hessian H of `loss_fn()` with respect to
    `params`, by Lanczos with full reorthogonalisation on Hessian-vector products (double
    backward through the gradient of one evaluation), in float64.

    Lanczos builds the Krylov space of a random start vector and takes the top eigenvalue of
    the projected tridiagonal matrix (the top Ritz value). It stops when the top Ritz pair's
    residual |beta_k s_k| falls below `HESSIAN_RTOL` (or 64 epsilons of the parameters' dtype)
    times the largest Ritz value's magnitude, or when the Krylov space is invariant (beta_k ~ 0,
    at the latest after as many steps as there are parameters); the Ritz value is then within
    that residual of an eigenvalue of H and, from a random start, it is the top one. A run that
    has not converged after `max_iter` products raises ValueError instead of returning an
    estimate. For an ensemble the loss is the sum of the members' and H is block-diagonal, so
    this is the largest of the members' top eigenvalues. The start vector comes from a local
    generator, so the global RNG is untouched; the parameters and their `.grad` are left as
    they were."""
    params = list(params)
    grads = torch.autograd.grad(loss_fn(), params, create_graph=True)
    sizes = [p.numel() for p in params]
    n = sum(sizes)
    rtol = max(HESSIAN_RTOL, 64 * torch.finfo(params[0].dtype).eps)

    def hvp(v: Tensor) -> Tensor:
        """H v for v (n,) float64, through the parameters' own dtype and device."""
        parts = [
            x.reshape(p.shape).to(p.device, p.dtype)
            for x, p in zip(v.split(sizes), params, strict=True)
        ]
        dot = sum((g * x).sum() for g, x in zip(grads, parts, strict=True))
        hv = torch.autograd.grad(dot, params, retain_graph=True)
        out = torch.cat([h.detach().reshape(-1).to("cpu", torch.float64) for h in hv])
        if not torch.isfinite(out).all():
            raise ValueError("the loss Hessian has non-finite entries at the current net")
        return out

    start = torch.randn(n, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return _lanczos_top_eigenvalue(hvp, start, min(max_iter, n), rtol)


def _lanczos_top_eigenvalue(
    matvec: Callable[[Tensor], Tensor], start: Tensor, max_iter: int, rtol: float
) -> float:
    """The top eigenvalue of the symmetric operator `matvec` (n,) -> (n,), by `max_iter` or
    fewer Lanczos steps from `start`, with full reorthogonalisation (each new vector is
    orthogonalised twice against all previous ones, so the basis stays orthonormal and no
    spurious copies of converged Ritz values appear). See `largest_hessian_eigenvalue`."""
    basis = [start / torch.linalg.norm(start)]
    alphas: list[float] = []
    betas: list[float] = []
    for _ in range(max_iter):
        q = basis[-1]
        w = matvec(q)
        alphas.append(torch.dot(w, q).item())
        Q = torch.stack(basis)
        for _ in range(2):
            w = w - Q.T @ (Q @ w)
        beta = torch.linalg.norm(w).item()
        T = torch.diag(torch.tensor(alphas, dtype=torch.float64))
        if betas:
            off = torch.tensor(betas, dtype=torch.float64)
            T = T + torch.diag(off, 1) + torch.diag(off, -1)
        ritz, vectors = torch.linalg.eigh(T)
        scale = ritz.abs().max().item()
        residual = beta * abs(vectors[-1, -1].item())
        if residual <= rtol * scale or beta <= rtol * scale or scale == 0.0:
            return ritz[-1].item()
        betas.append(beta)
        basis.append(w / beta)
    raise ValueError(
        f"Lanczos for the largest Hessian eigenvalue did not converge in {max_iter} "
        f"Hessian-vector products (top Ritz value {ritz[-1].item():.6g}, residual "
        f"{residual:.3g} against a tolerance of {rtol * scale:.3g})"
    )


def _check_sgld_step_resolves_the_loss(
    net: Net,
    dataset: Dataset,
    tc: TrainConfig,
    weight_penalty: WeightPenalty,
    log: Callable[[str], None],
) -> None:
    """Raise if an SGLD step of size lr is unstable on the full loss at the current net: the
    data term adds, along the directions the data spans, a curvature the prior-only bound
    does not see. lr * lambda_max(Hessian) >= 1, half the divergence limit, is refused; the
    value is logged otherwise. A step with lr * lambda_max <= 0.1 inflates the stationary
    variance of the stiffest mode by at most 5% (1 / (1 - lr c / 2) on a quadratic of
    curvature c)."""
    coefficients = loss_coefficients(tc)

    def loss() -> Tensor:
        return compute_loss(net, dataset.x, dataset.y, coefficients, weight_penalty)[0]

    lam = largest_hessian_eigenvalue(loss, net.parameters())
    step = tc.lr * lam
    if step >= 1:
        raise ValueError(
            f"SGLD with lr={tc.lr:g} is unstable on the loss at the initial net: lr * "
            f"lambda_max(loss Hessian) = {step:.3g} must be < 1 (lambda_max = {lam:.3g} from "
            "Lanczos); lower lr"
        )
    log(f"SGLD step check: lr * lambda_max(loss Hessian) = {step:.3g} (lambda_max = {lam:.3g})")


def make_weight_penalty(cfg: Config, net: Net) -> WeightPenalty:
    """The weight term `cfg` asks for: the net's `weight_l2` (`MLP.weight_l2`, or
    `EnsembleMLP.weight_l2` per member) without a spatial prior and under its whitened
    parametrisation (where the parameters are xi and the prior is the plain |xi|^2), else the
    spatial prior (per member for an ensemble) with the penalty matrices built for `net`'s
    sizes (float64 on the CPU) and then cast to its dtype and device. They are built here, once
    per run, and are not parameters: a resumed run rebuilds the same matrices from the config.
    For SGLD it raises if `train.lr` cannot resolve the prior
    (`_check_sgld_step_resolves_the_prior`), on either path. ValueError if the net's
    parametrisation is not the config's.
    """
    _check_parametrisation_matches(cfg, net)
    if cfg.spatial is None:
        return type(net).weight_l2
    if cfg.is_whitened:
        if cfg.train.optimizer == "sgld":
            _check_sgld_step_resolves_the_prior(1.0, "whitened spatial prior", cfg.train)
        return type(net).weight_l2
    spaces = build_spaces(cfg.spatial, m=net.D_in, N=net.N, L=net.L, n_out=net.D_out)
    factors64 = parameter_factors(spaces)
    names = {name for name, _ in net.named_parameters()}
    if names != set(factors64):
        # A parameter without a factor (a bias, say) would otherwise silently drop out.
        raise ValueError(
            f"the spatial prior covers {sorted(factors64)} but the net has parameters "
            f"{sorted(names)}"
        )
    if cfg.train.optimizer == "sgld":
        _check_sgld_step_resolves_the_prior(
            prior_stiffness(factors64),
            "spatial prior (stiffness lambda_max(M_rows) lambda_max(M_cols), which a wider "
            "floor min_rel_eig lowers)",
            cfg.train,
        )
    factors = {
        name: (rows.to(net.device, net.dtype), cols.to(net.device, net.dtype))
        for name, (rows, cols) in factors64.items()
    }
    return partial(spatial_weight_penalty, factors=factors)


def _rng_state(device: torch.device) -> dict[str, Tensor]:
    state = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def _restore_rng(state: dict[str, Tensor], device: torch.device) -> str | None:
    """Restore the saved RNG state; return a note if a CUDA state could not be restored."""
    # set_rng_state needs CPU ByteTensors, but load_artefact maps tensors to the model device.
    torch.set_rng_state(state["cpu"].cpu())
    if "cuda" in state:
        if device.type != "cuda":
            raise ValueError("checkpoint holds a CUDA RNG state, which cannot be restored on CPU")
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)
        return None
    if device.type == "cuda":
        return "checkpoint has no CUDA RNG state, so the CUDA RNG was not restored"
    return None


def _optim_to_saveable(state: dict) -> dict:
    """Optimiser per-parameter state is keyed by int index; saved artefacts need str keys."""
    return {**state, "state": {str(i): v for i, v in state["state"].items()}}


def _optim_from_saved(state: dict) -> dict:
    return {**state, "state": {int(i): v for i, v in state["state"].items()}}


def _payload(net: Net, optim: torch.optim.Optimizer, history: History, epoch: int) -> dict:
    """`epoch` is the next epoch to run."""
    return {
        "epoch": epoch,
        "net_state_dict": dict(net.state_dict()),  # plain dict: the cache rejects OrderedDict
        "optim_state_dict": _optim_to_saveable(optim.state_dict()),
        "history": history,
        "rng_state": _rng_state(net.device),
    }


def _load_payload(path: Path, cfg: Config, device: torch.device) -> dict:
    """Load a checkpoint or final file, failing loudly (naming `path`) if it is inconsistent."""
    payload = load_artefact(path, cfg.section_hash("data", "net", "train"), device)
    missing = [k for k in _REQUIRED_KEYS if k not in payload]
    if missing:
        raise ValueError(f"{path} is missing keys {missing}")
    n_epochs = cfg.train.n_epochs
    epoch = payload["epoch"]
    if epoch > n_epochs or (path.name == "final.pt" and epoch != n_epochs):
        raise ValueError(f"{path} is at epoch {epoch}, inconsistent with n_epochs={n_epochs}")
    return payload


def _restore(
    path: Path, cfg: Config, net: Net, optim: torch.optim.Optimizer
) -> tuple[int, History, str | None]:
    """Load a checkpoint into `net` and `optim`; return next epoch, history, RNG note."""
    payload = _load_payload(path, cfg, net.device)
    net.load_state_dict(payload["net_state_dict"])
    optim.load_state_dict(_optim_from_saved(payload["optim_state_dict"]))
    try:
        note = _restore_rng(payload["rng_state"], net.device)
    except ValueError as err:
        raise ValueError(f"{path}: {err}") from err
    return payload["epoch"], payload["history"], note


def _log_line(epoch: int, n_epochs: int, parts: dict[str, float | list[float]]) -> str:
    """One line per logged epoch; for an ensemble the mean over members with (min..max)."""
    labels = {"mse": "mse", "weight_l2": "weight penalty", "rate_l2": "rate L2", "loss": "loss"}
    head = f"[{epoch:05d}/{n_epochs:05d}]"
    if not isinstance(parts["loss"], list):
        return f"{head} " + " ".join(f"{label}={parts[k]:.4f}" for k, label in labels.items())

    def summary(values: list[float]) -> str:
        mean = sum(values) / len(values)
        return f"{mean:.4f} ({min(values):.4f}..{max(values):.4f})"

    B = len(parts["loss"])
    return f"{head} B={B} mean (min..max): " + " ".join(
        f"{label}={summary(parts[k])}" for k, label in labels.items()
    )


def _to_host(value: Tensor) -> float | list[float]:
    """A logged part as a number, or for an ensemble a list over members."""
    return value.item() if value.ndim == 0 else value.tolist()


def _history_keys(optimizer: str) -> tuple[str, ...]:
    return _HISTORY_KEYS + (_SGLD_KEYS if optimizer == "sgld" else ())


def _record(history: History, epoch: int, parts: dict[str, float | list[float]]) -> None:
    history["epoch"].append(epoch)
    for key, value in parts.items():
        history[key].append(value)


def _make_optimiser(net: Net, tc: TrainConfig) -> torch.optim.Optimizer:
    if tc.optimizer == "sgld":
        member_axis = net.member_axes == 1
        return SGLD(net.parameters(), lr=tc.lr, beta=tc.beta, member_axis=member_axis)
    # Adam's moments are elementwise, so the members of an ensemble do not interact.
    return torch.optim.Adam(net.parameters(), lr=tc.lr)


def _grad_norm(net: Net) -> Tensor:
    """L2 norm of the full gradient; (B,), each member's, for an ensemble."""
    return torch.sqrt(sum(per_member(net, p.grad.square()) for p in net.parameters()))


def _step(
    net: Net,
    optim: torch.optim.Optimizer,
    dataset: Dataset,
    tc: TrainConfig,
    weight_penalty: WeightPenalty,
) -> dict[str, Tensor]:
    """One full-batch optimiser step; returns the loss parts from before the step (for SGLD
    also the gradient norm and the step's mean squared displacement), (B,) for an ensemble."""
    optim.zero_grad()
    loss, parts = compute_loss(net, dataset.x, dataset.y, loss_coefficients(tc), weight_penalty)
    loss.backward()
    if isinstance(optim, SGLD):
        parts["grad_norm"] = _grad_norm(net).detach()
    optim.step()
    if isinstance(optim, SGLD):
        parts["param_msd"] = optim.last_msd
    return parts


def _check_monitor_matches_config(cfg: Config, monitor: Monitor | None) -> None:
    if cfg.train.monitor_int > 0 and monitor is None:
        raise ValueError(f"train.monitor_int={cfg.train.monitor_int} needs a monitor, got None")
    if cfg.train.monitor_int == 0 and monitor is not None:
        raise ValueError("a monitor was passed but train.monitor_int=0 disables monitoring")


def _new_history(monitored: bool, optimizer: str) -> History:
    history: History = {key: [] for key in _history_keys(optimizer)}
    if monitored:
        history[MONITOR_EPOCH] = []
    return history


def _monitor_due(epoch: int, tc: TrainConfig) -> bool:
    """Whether the monitor runs at the start of `epoch` (the final call is separate)."""
    return tc.monitor_int > 0 and epoch % tc.monitor_int == 0


def _record_monitor(history: History, epoch: int, result: dict[str, list]) -> None:
    """Append one monitor call to `history` as `monitor_epoch` and `monitor_<key>` entries."""
    keys = {f"{MONITOR_PREFIX}{k}" for k in result}
    if MONITOR_EPOCH in keys:
        raise ValueError(f"monitor key 'epoch' is reserved: it would overwrite {MONITOR_EPOCH!r}")
    seen = {k for k in history if k.startswith(MONITOR_PREFIX) and k != MONITOR_EPOCH}
    if history[MONITOR_EPOCH] and keys != seen:
        raise ValueError(f"monitor returned keys {sorted(keys)}, earlier calls {sorted(seen)}")
    history[MONITOR_EPOCH].append(epoch)
    for key, value in result.items():
        history.setdefault(f"{MONITOR_PREFIX}{key}", []).append(value)


def _monitor_members(monitor: Monitor, net: EnsembleMLP) -> dict[str, list]:
    """`monitor` on every member, merged into one result whose values are lists over members."""
    results = [monitor(net.member(b)) for b in range(net.B)]
    keys = set(results[0])
    for b, result in enumerate(results):
        if set(result) != keys:
            raise ValueError(
                f"monitor returned keys {sorted(result)} for member {b}, "
                f"{sorted(keys)} for member 0"
            )
    return {key: [result[key] for result in results] for key in results[0]}


def _call_monitor(monitor: Monitor, net: Net) -> dict[str, list]:
    """Run `monitor` without gradients and without advancing the global RNG: the training
    stream must not depend on whether, or how often, the net is being monitored. An ensemble
    is monitored member by member, which puts a member axis first in every value."""
    cuda = [net.device] if net.device.type == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=cuda):
        if isinstance(net, EnsembleMLP):
            return _monitor_members(monitor, net)
        return monitor(net)


def _check_net_matches_config(net: Net, cfg: Config) -> None:
    if net.dtype != resolve_dtype(cfg.dtype):
        raise ValueError(f"net dtype {net.dtype} does not match config dtype {cfg.dtype!r}")
    if net.device.type != resolve_device(cfg).type:
        raise ValueError(f"net is on {net.device} but the config resolves to {resolve_device(cfg)}")
    if net.B != cfg.net.B:
        raise ValueError(
            f"the net is an ensemble of B={net.B} but the config has net.B={cfg.net.B}"
        )
    _check_parametrisation_matches(cfg, net)


def train(
    cfg: Config,
    dataset: Dataset,
    net: Net,
    cache_dir: Path,
    log: Callable[[str], None] = print,
    monitor: Monitor | None = None,
) -> tuple[Net, History]:
    """Train `net` in place on the full dataset (Adam or SGLD, `train.optimizer`) with the
    weight penalty of `make_weight_penalty`; resume from the latest checkpoint if any.

    If `final.pt` exists the net is loaded from it and nothing else happens (no optimiser,
    no RNG change). A fresh SGLD run under the whitened spatial prior first checks that `lr`
    resolves the loss's curvature at the initial net (`_check_sgld_step_resolves_the_loss`,
    which logs the value). A checkpoint with `epoch=s` is written before the step of epoch s; the
    step, history entry and log line follow. An interruption at epoch t therefore resumes at
    the last multiple of `save_int` at or before t, repeating the deterministic steps since.

    `monitor` (required exactly when `train.monitor_int > 0`) is called on the net at the
    start of every epoch e with e % monitor_int == 0, after e's checkpoint and before its
    step, and once more after the last step (e = n_epochs). Its results enter the history as
    `monitor_epoch` and `monitor_<key>`. A checkpoint therefore never holds the entry of the
    epoch it is written at; the resumed run records it, so resuming reproduces the history.

    An ensemble (`EnsembleMLP`, cfg.net.B > 1) trains on the sum of its members' losses, so
    each member follows exactly its own gradient (`compute_loss`); every logged part is then
    a list over members, and the monitor runs on each member (`net.member(b)`), so each
    `monitor_<key>` entry is a list over members of what the monitor returned. `epoch` and
    `monitor_epoch` have no member axis. `select_member` extracts one member's history.
    """
    return _train_in(cfg, dataset, net, cache_paths(cache_dir).train, log, monitor)


def _train_in(
    cfg: Config,
    dataset: Dataset,
    net: Net,
    train_dir: Path,
    log: Callable[[str], None],
    monitor: Monitor | None,
) -> tuple[Net, History]:
    """`train` with its checkpoints and final.pt in `train_dir`: CACHE_DIR/train for a run,
    a staging directory for the members an extension adds (`extend_trained`)."""
    _check_net_matches_config(net, cfg)
    _check_monitor_matches_config(cfg, monitor)
    tc = cfg.train
    config_hash = cfg.section_hash("data", "net", "train")
    final = train_dir / "final.pt"

    if final.exists():
        payload = _load_payload(final, cfg, net.device)
        net.load_state_dict(payload["net_state_dict"])
        return net, payload["history"]

    optim = _make_optimiser(net, tc)
    weight_penalty = make_weight_penalty(cfg, net)
    ckpt = latest_checkpoint(train_dir)
    if ckpt is None:
        # Adam draws nothing from the global RNG; SGLD draws its step noise from it, which is
        # why the RNG state is checkpointed.
        torch.manual_seed(tc.seed)
        start, history = 0, _new_history(monitor is not None, tc.optimizer)
        if tc.optimizer == "sgld" and cfg.is_whitened:
            # At the initial net only: a resumed run passed this when it started.
            _check_sgld_step_resolves_the_loss(net, dataset, tc, weight_penalty, log)
    else:
        start, history, note = _restore(ckpt, cfg, net, optim)
        log(f"resumed from {ckpt} at epoch {start}" + (f" ({note})" if note else ""))

    for epoch in range(start, tc.n_epochs):
        if epoch > start and epoch % tc.save_int == 0:
            # `epoch > start` avoids rewriting the checkpoint we have just loaded.
            write_checkpoint(
                train_dir, "ckpt_", epoch, _payload(net, optim, history, epoch), config_hash
            )
        if monitor is not None and _monitor_due(epoch, tc):
            _record_monitor(history, epoch, _call_monitor(monitor, net))
        parts = _step(net, optim, dataset, tc, weight_penalty)
        if epoch % tc.log_int == 0:
            values = {k: _to_host(v) for k, v in parts.items()}
            _record(history, epoch, values)
            log(_log_line(epoch, tc.n_epochs, values))

    if monitor is not None:
        _record_monitor(history, tc.n_epochs, _call_monitor(monitor, net))
    save_artefact(final, _payload(net, optim, history, tc.n_epochs), config_hash)
    return net, history


def train_from_config(
    cfg: Config,
    dataset: Dataset,
    cache_dir: Path,
    log: Callable[[str], None] = print,
    monitor: Monitor | None = None,
) -> tuple[Net, History]:
    """Build the net the config describes for `dataset` and `train` it (see `train`).

    The one place that turns a config and a dataset into a net (`make_net_from_config`), so
    the CLI and the experiment start every run identically.
    """
    net = make_net_from_config(
        cfg,
        dataset.x.shape[1],
        dataset.y.shape[1],
        resolve_device(cfg),
        resolve_dtype(cfg.dtype),
    )
    return train(cfg, dataset, net, cache_dir, log=log, monitor=monitor)


def _net_from_payload(payload: dict, cfg: Config, device: torch.device, dtype: torch.dtype) -> Net:
    """The net of a saved payload: built from the config (which also gives a whitened net its
    factors) at the sizes of the saved parameters, then loaded with them."""
    state = payload["net_state_dict"]
    # The last axis is D_in / D_out for a single net (N, D) and an ensemble (B, N, D) alike.
    net = make_net_from_config(cfg, state["U"].shape[-1], state["V"].shape[-1], device, dtype)
    net.load_state_dict(state)
    return net


def load_trained(
    cache_dir: Path, cfg: Config, device: torch.device, dtype: torch.dtype
) -> tuple[Net, History]:
    """Rebuild the trained net and its history from `final.pt`: an `EnsembleMLP` and an
    ensemble history when cfg.net.B > 1 (`select_member` picks one network)."""
    if dtype != resolve_dtype(cfg.dtype):
        raise ValueError(f"dtype {dtype} does not match config dtype {cfg.dtype!r}")
    final = cache_paths(cache_dir).train / "final.pt"
    if not final.exists():
        raise FileNotFoundError(f"no trained model at {final}; run training first")
    payload = _load_payload(final, cfg, device)
    return _net_from_payload(payload, cfg, device, dtype), payload["history"]


# History keys without a member axis in an ensemble history.
_SHARED_KEYS = ("epoch", MONITOR_EPOCH)


def member_history(history: History, member: int) -> History:
    """Member `member`'s history from an ensemble's: every entry but `epoch` and
    `monitor_epoch` indexed by the member, which gives a single network's history."""
    return {
        key: list(values) if key in _SHARED_KEYS else [entry[member] for entry in values]
        for key, values in history.items()
    }


def select_member(net: Net, history: History, member: int | None = None) -> tuple[MLP, History]:
    """One network of a trained run and its history, for readers that analyse one network.

    A single network passes through unchanged (`member` None or 0). An ensemble needs an
    explicit `member` (ValueError otherwise), so no reader ever looks at member 0 without
    being asked to; IndexError for a member out of range.
    """
    if isinstance(net, MLP):
        if member not in (None, 0):
            raise IndexError(f"member {member} out of range for a single network (B=1)")
        return net, history
    if member is None:
        raise ValueError(
            f"the run is an ensemble of B={net.B} networks and this reader takes one; "
            "select a member"
        )
    return net.member(member), member_history(history, member)


def load_weights(
    cache_dir: Path, cfg: Config, device: torch.device, dtype: torch.dtype, done: bool
) -> tuple[Net, History, int]:
    """The net, history and epochs completed of a setting that has weights: `final.pt` when
    `done` (all `cfg.train.n_epochs`), else the latest checkpoint of a run still going."""
    if done:
        net, history = load_trained(cache_dir, cfg, device, dtype)
        return net, history, cfg.train.n_epochs
    return load_latest_checkpoint(cache_dir, cfg, device, dtype)


# What torch.load, `_load_payload` and `load_state_dict` raise for a truncated, foreign or
# inconsistent file; anything else (a bug) is not a file problem and propagates.
_UNREADABLE = (StaleCacheError, ValueError, TypeError, KeyError, RuntimeError, EOFError)


def _read_latest(train_dir: Path, cfg: Config, device: torch.device) -> tuple[Path, dict]:
    """The latest checkpoint and its payload. A file that cannot be used (truncated, other
    config hash, inconsistent contents) raises a ValueError naming it; an OSError (the file
    vanished, a stale network-file-system handle) passes through, because it is retried."""
    ckpt = latest_checkpoint(train_dir)
    if ckpt is None:
        raise FileNotFoundError(f"no checkpoint in {train_dir}")
    try:
        return ckpt, _load_payload(ckpt, cfg, device)
    except OSError:
        raise
    except (*_UNREADABLE, pickle.UnpicklingError) as err:
        raise ValueError(f"cannot read {ckpt}: {type(err).__name__}: {err}") from err


def load_latest_checkpoint(
    cache_dir: Path, cfg: Config, device: torch.device, dtype: torch.dtype
) -> tuple[Net, History, int]:
    """Read-only: the net, history and epoch reached (epochs completed) of the latest
    checkpoint, for looking at a run that is still going.

    A running job writes the new checkpoint before deleting the old one, so a checkpoint
    that vanishes between listing and opening (or whose handle goes stale on a network file
    system) has been superseded: the read is repeated once. OSError if it fails twice or there
    is no checkpoint; ValueError naming the file if it cannot be used or does not match `cfg`.
    """
    if dtype != resolve_dtype(cfg.dtype):
        raise ValueError(f"dtype {dtype} does not match config dtype {cfg.dtype!r}")
    train_dir = cache_paths(cache_dir).train
    try:
        ckpt, payload = _read_latest(train_dir, cfg, device)
    except OSError:
        ckpt, payload = _read_latest(train_dir, cfg, device)
    try:
        net = _net_from_payload(payload, cfg, device, dtype)
    except _UNREADABLE as err:
        raise ValueError(
            f"cannot rebuild the net from {ckpt}: {type(err).__name__}: {err}"
        ) from err
    return net, payload["history"], payload["epoch"]


# ---------------------------------------------------------------------------- extending an ensemble
# An ensemble in final.pt grows without retraining its members: the members it lacks train as
# an ensemble of their own in a staging directory beside final.pt, and are then appended.
STAGING_PREFIX = "extend_"


def staging_dir(train_dir: Path, b0: int, b1: int) -> Path:
    """Where members b0 .. b1-1 (half-open, like a slice) of an extension train, with their own
    checkpoints and final.pt: `train_dir/extend_<b0>-<b1>/`."""
    return Path(train_dir) / f"{STAGING_PREFIX}{b0}-{b1}"


def remove_staging(train_dir: Path, keep: Path | None = None) -> None:
    """Delete every staging directory in `train_dir` but `keep`: what an extension that has
    been appended (or abandoned for another block) leaves behind."""
    for directory in Path(train_dir).glob(f"{STAGING_PREFIX}*"):
        if directory.is_dir() and directory != keep:
            shutil.rmtree(directory)


def extension_seed(seed: int, b0: int) -> int:
    """Seed of the global RNG, which draws the SGLD noise, while the members from b0 on of an
    extension train: the first 8 bytes (little-endian) of sha256("fct.train.extension:<seed>:
    <b0>"), with `seed` the run's `train.seed`. Reseeding with `seed` itself would replay the
    stream the original members drew, so the new members' noise would be the old members'
    noise rearranged; a stream of its own keeps every member independent."""
    if b0 < 1:
        raise ValueError(f"an extension adds members after the first, so b0 >= 1; got b0={b0}")
    digest = hashlib.sha256(f"fct.train.extension:{seed}:{b0}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def with_ensemble_size(cfg: Config, B: int) -> Config:
    """`cfg` with `net.B = B`: the config an ensemble of B of its members was trained under."""
    return cfg.model_copy(update={"net": cfg.net.model_copy(update={"B": B})})


def _staging_config(cfg: Config, b0: int) -> Config:
    """The config members b0 .. cfg.net.B-1 train under in staging: an ensemble of that many,
    with the extension's noise seed. Its hash stamps the staging checkpoints."""
    train_cfg = cfg.train.model_copy(update={"seed": extension_seed(cfg.train.seed, b0)})
    return with_ensemble_size(cfg, cfg.net.B - b0).model_copy(update={"train": train_cfg})


def trained_member_count(path: Path) -> int:
    """Members of the net in a train artefact (`final.pt`, a checkpoint): 1 for a single
    network, else the length of the member axis of its U. Only that entry is read; ValueError
    naming the file if it is unreadable or holds no net."""
    state = read_field(path, "net_state_dict")
    if not isinstance(state, dict) or not isinstance(state.get("U"), Tensor):
        # ValueError, like an unreadable file: either way the file holds no usable net.
        raise ValueError(f"{path} holds no net state dict with a parameter U")  # noqa: TRY004
    return _members_of(state)


def _members_of(state: dict) -> int:
    U = state["U"]
    return 1 if U.ndim == 2 else U.shape[0]


def _with_member_axis(t: Tensor, members: int) -> Tensor:
    """A per-member tensor with its member axis: a single network's gains one of length 1."""
    return t.unsqueeze(0) if members == 1 else t


def _member_entries(entry, members: int) -> list:
    """A per-member history entry as a list over members (a single network's is one value)."""
    if members == 1:
        return [entry]
    if len(entry) != members:
        raise ValueError(f"a history entry has {len(entry)} members, the net {members}")
    return list(entry)


def _append_state(old: dict, new: dict, b_old: int, k: int) -> dict:
    if old.keys() != new.keys():
        raise ValueError(f"state dict keys differ: {sorted(old)} and {sorted(new)}")
    merged = {}
    for key, value in old.items():
        a, b = _with_member_axis(value, b_old), _with_member_axis(new[key], k)
        if a.shape[1:] != b.shape[1:]:
            raise ValueError(
                f"state dict entry {key!r} has member shape {tuple(a.shape[1:])} in the old "
                f"payload, {tuple(b.shape[1:])} in the new one"
            )
        merged[key] = torch.cat([a, b])
    return merged


def _same(a, b) -> bool:
    if isinstance(a, Tensor) and isinstance(b, Tensor):
        return a.shape == b.shape and torch.equal(a, b)
    return a == b


def _append_param_state(
    old: dict, new: dict, params: tuple[Tensor, Tensor], b_old: int, k: int
) -> dict:
    """One parameter's optimiser state: an entry of the parameter's shape (Adam's moments) is
    per member and concatenated; any other (Adam's step count) must agree and is kept."""
    if old.keys() != new.keys():
        raise ValueError(f"optimiser state keys differ: {sorted(old)} and {sorted(new)}")
    merged = {}
    for name, a in old.items():
        b = new[name]
        per_member = isinstance(a, Tensor) and a.shape == params[0].shape
        if per_member and isinstance(b, Tensor) and b.shape == params[1].shape:
            merged[name] = torch.cat([_with_member_axis(a, b_old), _with_member_axis(b, k)])
        elif _same(a, b):
            merged[name] = a
        else:
            raise ValueError(f"optimiser state {name!r} differs and has no member axis")
    return merged


def _append_optim(old: dict, new: dict, b_old: int, k: int) -> dict:
    """The optimiser state dict. Its per-parameter state is indexed like the parameters, which
    are the state dict's entries in order (the state dict holds exactly the parameters)."""
    o, n = old["optim_state_dict"], new["optim_state_dict"]
    if o["param_groups"] != n["param_groups"]:
        raise ValueError("the optimiser param_groups differ (another optimiser or lr)")
    if o["state"].keys() != n["state"].keys():
        raise ValueError(
            f"optimiser state indices differ: {sorted(o['state'])} and {sorted(n['state'])}"
        )
    pairs = list(zip(old["net_state_dict"].values(), new["net_state_dict"].values(), strict=True))
    state = {
        i: _append_param_state(o["state"][i], n["state"][i], pairs[int(i)], b_old, k)
        for i in o["state"]
    }
    return {**o, "state": state}


def _append_history(old: History, new: History, b_old: int, k: int) -> History:
    if old.keys() != new.keys():
        raise ValueError(f"history keys differ: {sorted(old)} and {sorted(new)}")
    merged: History = {}
    for key, values in old.items():
        if key in _SHARED_KEYS:
            if values != new[key]:
                raise ValueError(f"history {key!r} differs: {values} and {new[key]}")
            merged[key] = list(values)
        elif len(values) != len(new[key]):
            raise ValueError(f"history {key!r} has {len(values)} entries and {len(new[key])}")
        else:
            merged[key] = [
                _member_entries(a, b_old) + _member_entries(b, k)
                for a, b in zip(values, new[key], strict=True)
            ]
    return merged


def append_members(old: dict, new: dict) -> dict:
    """The train payload of an ensemble whose members are `old`'s followed by `new`'s, both
    finished payloads of one config but for `net.B` (as `_payload` builds them: no config
    hash). Every per-member entry is concatenated along its member axis: the state dict (which
    holds exactly the parameters; a whitened net's factors are non-persistent buffers, rebuilt
    from the config), Adam's moments, and every history entry but `epoch` and `monitor_epoch`,
    which must agree. A single network (no member axis) counts as one member. The RNG state is
    `old`'s: a finished run is never resumed. ValueError if the payloads do not fit."""
    for name, payload in (("old", old), ("new", new)):
        if set(payload) != set(_REQUIRED_KEYS):
            raise ValueError(f"the {name} payload has keys {sorted(payload)}, not {_REQUIRED_KEYS}")
    if old["epoch"] != new["epoch"]:
        raise ValueError(f"the payloads end at epoch {old['epoch']} and epoch {new['epoch']}")
    b_old, k = _members_of(old["net_state_dict"]), _members_of(new["net_state_dict"])
    return {
        "epoch": old["epoch"],
        "net_state_dict": _append_state(old["net_state_dict"], new["net_state_dict"], b_old, k),
        "optim_state_dict": _append_optim(old, new, b_old, k),
        "history": _append_history(old["history"], new["history"], b_old, k),
        "rng_state": old["rng_state"],
    }


def _without_hash(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k != "config_hash"}


def extend_trained(
    cfg: Config,
    dataset: Dataset,
    cache_dir: Path,
    log: Callable[[str], None] = print,
    monitor: Monitor | None = None,
) -> tuple[Net, History]:
    """Grow the trained ensemble in CACHE_DIR/train/final.pt from its b0 members to cfg.net.B.

    final.pt must have been written under `cfg` with `net.B = b0` (StaleCacheError otherwise).
    Members b0 .. B-1 are built exactly as in a from-scratch run of B (`member_seed` does not
    depend on B) and trained by `train` as an ensemble of their own in `staging_dir(train, b0,
    B)`, so an interrupted extension resumes from its checkpoints. The staging run draws its
    SGLD noise from `extension_seed`, so under SGLD the new members are new samples of the same
    law, not a from-scratch run's members bit for bit (Adam draws no noise: those are equal).
    The monitor (built from `cfg`) sees each new member as in a from-scratch run. Then
    final.pt is replaced by `append_members` stamped with `cfg`'s hash, the staging directory
    is deleted, and the ensemble of B and its history are returned."""
    train_dir = cache_paths(cache_dir).train
    final = train_dir / "final.pt"
    b0, B = trained_member_count(final), cfg.net.B
    if b0 >= B:
        raise ValueError(f"{final} holds {b0} members; there is nothing to add for net.B={B}")
    cpu = torch.device("cpu")  # the payloads are merged on the CPU, the net built after
    old = _load_payload(final, with_ensemble_size(cfg, b0), cpu)
    staging, staging_cfg = staging_dir(train_dir, b0, B), _staging_config(cfg, b0)
    remove_staging(train_dir, keep=staging)
    device, dtype = resolve_device(cfg), resolve_dtype(cfg.dtype)
    D_in, D_out = dataset.x.shape[1], dataset.y.shape[1]
    net = make_net_from_config(cfg, D_in, D_out, device, dtype, members=range(b0, B))
    log(f"extending {final} from {b0} to {B} members (staging in {staging})")
    _train_in(staging_cfg, dataset, net, staging, log, monitor)
    staged = _load_payload(staging / "final.pt", staging_cfg, cpu)
    merged = append_members(_without_hash(old), _without_hash(staged))
    save_artefact(final, merged, cfg.section_hash("data", "net", "train"))
    shutil.rmtree(staging)
    return _net_from_payload(merged, cfg, device, dtype), merged["history"]
