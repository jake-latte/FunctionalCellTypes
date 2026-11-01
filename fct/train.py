"""Full-batch training of the MLP with periodic checkpoints and resume.

Layout under `cache_dir/train/`: `ckpt_<epoch>.pt` (only the latest kept) and `final.pt`.
Both share one payload shape; resuming and `load_trained` read them through the same
validating loader (`_load_payload`) and write the net with `load_state_dict`.
"""

import pickle
from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch
from torch import Tensor

from fct.cache import (
    StaleCacheError,
    cache_paths,
    latest_checkpoint,
    load_artefact,
    save_artefact,
    write_checkpoint,
)
from fct.config import Config, TrainConfig, resolve_device, resolve_dtype
from fct.data import Dataset
from fct.model import MLP, EnsembleMLP, Net, make_net
from fct.sgld import SGLD
from fct.spatial import build_spaces, parameter_factors, weight_quadratic_form

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


def prior_coefficient(tc: TrainConfig) -> float:
    """weight_lambda / (2 beta): the coefficient of the weight prior's quadratic form in the loss.

    beta is the SGLD inverse temperature `tc.beta`, and 1 for Adam (which has no temperature).
    The SGLD stationary law is exp(-beta L), so dividing by beta here leaves the prior
    exp(-weight_lambda/2 * form), i.e. weights N(0, K / weight_lambda), for every beta, while
    the data and rate terms of the law carry the factor beta.
    """
    beta = tc.beta if tc.optimizer == "sgld" else 1.0
    return tc.weight_lambda / (2.0 * beta)


def compute_loss(
    net: Net,
    x: Tensor,
    y: Tensor,
    prior_coefficient: float,
    rate_lambda: float,
    weight_penalty: WeightPenalty | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """0.5*N*mse + prior_coefficient*weight_l2 + rate_lambda*rate_l2, plus its parts.

    For an ensemble every part is (B,), member b's own value, and the returned loss is the
    SUM of the members' losses: member b's parameters enter only its own term, so the
    gradient with respect to them is exactly that of its loss (not 1/B of it).
    `weight_penalty` None is the net's own `weight_l2`.

    `prior_coefficient` multiplies the prior's quadratic form: callers pass
    `prior_coefficient(train_config)` (= weight_lambda / (2 beta)), never the raw
    `train.weight_lambda`.

    `rate_l2` is the squared rates phi(h) summed over samples, neurons and layers, so it grows
    with the number of samples P. `weight_l2` is `weight_penalty(net)`: by default the plain
    sum of squares of every parameter, with a spatial prior the sum over layers of
    tr(M_post W M_pre W^T) (`make_weight_penalty` picks the one the config asks for).

    The parts are detached 0-d tensors: calling `.item()` here would force a host-device
    sync every epoch, so the caller converts only when it logs.
    """
    h, z = net(x)
    mse = per_member(net, (y - z).square(), torch.mean)
    rate_l2 = per_member(net, net.phi(h).square())
    weight_l2 = (type(net).weight_l2 if weight_penalty is None else weight_penalty)(net)
    if weight_l2.shape != mse.shape:
        raise ValueError(
            f"the weight penalty has shape {tuple(weight_l2.shape)} but the loss needs "
            f"{tuple(mse.shape)} (one value per member for an ensemble)"
        )
    loss = 0.5 * net.N * mse + prior_coefficient * weight_l2 + rate_lambda * rate_l2
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


def _check_sgld_step_resolves_the_prior(
    factors: dict[str, tuple[Tensor, Tensor]], tc: TrainConfig
) -> None:
    """Raise if an SGLD step of size lr is unstable on the prior term alone.

    c * tr(M_rows W M_cols W^T), with c = `prior_coefficient(tc)` = weight_lambda / (2 beta),
    has curvature 2 c lambda_max(M_rows) lambda_max(M_cols) along its stiffest direction, and an
    Euler step on a quadratic of curvature k diverges once lr * k >= 2. A wide kernel makes lambda_max large (up to
    1 / min_rel_eig over the largest eigenvalue of K), so this fails at the start instead of
    after the run has filled with NaN. Passing it is necessary, not sufficient: the data and
    rate terms add curvature of their own. Adam rescales its steps and needs no such check.
    """
    stiffest = max(
        torch.linalg.eigvalsh(rows)[-1].item() * torch.linalg.eigvalsh(cols)[-1].item()
        for rows, cols in factors.values()
    )
    step = tc.lr * prior_coefficient(tc) * stiffest
    if step >= 1:
        raise ValueError(
            f"SGLD with lr={tc.lr:g} diverges on the spatial prior alone: lr * weight_lambda / "
            f"(2 beta) * lambda_max(M) = {step:.3g} must be < 1 (lambda_max(M) = "
            f"{stiffest:.3g}); lower lr or weight_lambda, raise beta, or widen the floor "
            "min_rel_eig"
        )


def make_weight_penalty(cfg: Config, net: Net) -> WeightPenalty:
    """The weight term `cfg` asks for: the net's `weight_l2` (`MLP.weight_l2`, or
    `EnsembleMLP.weight_l2` per member) without a spatial prior, else the spatial prior
    (per member for an ensemble) with the penalty matrices built for `net`'s sizes (float64
    on the CPU) and then cast to its dtype and device. They are built here, once per run, and
    are not parameters: a resumed run rebuilds the same matrices from the config. For SGLD it
    raises if `train.lr` cannot resolve the prior (`_check_sgld_step_resolves_the_prior`).
    """
    if cfg.spatial is None:
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
        _check_sgld_step_resolves_the_prior(factors64, cfg.train)
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
    loss, parts = compute_loss(
        net, dataset.x, dataset.y, prior_coefficient(tc), tc.rate_lambda, weight_penalty
    )
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
    no RNG change). A checkpoint with `epoch=s` is written before the step of epoch s; the
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
    _check_net_matches_config(net, cfg)
    _check_monitor_matches_config(cfg, monitor)
    tc = cfg.train
    train_dir = cache_paths(cache_dir).train
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

    The one place that turns a config and a dataset into a net, so the CLI and the experiment
    start every run identically.
    """
    net = make_net(
        cfg.net,
        dataset.x.shape[1],
        dataset.y.shape[1],
        resolve_device(cfg),
        resolve_dtype(cfg.dtype),
    )
    return train(cfg, dataset, net, cache_dir, log=log, monitor=monitor)


def _net_from_payload(payload: dict, cfg: Config, device: torch.device, dtype: torch.dtype) -> Net:
    state = payload["net_state_dict"]
    # The last axis is D_in / D_out for a single net (N, D) and an ensemble (B, N, D) alike.
    net = make_net(cfg.net, state["U"].shape[-1], state["V"].shape[-1], device, dtype)
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
