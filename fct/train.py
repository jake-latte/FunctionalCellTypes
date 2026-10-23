"""Full-batch training of the MLP with periodic checkpoints and resume.

Layout under `cache_dir/train/`: `ckpt_<epoch>.pt` (only the latest kept) and `final.pt`.
Both share one payload shape; resuming and `load_trained` read them through the same
validating loader (`_load_payload`) and write the net with `load_state_dict`.
"""

import pickle
from collections.abc import Callable
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
from fct.model import MLP, make_net
from fct.sgld import SGLD

History = dict[str, list]  # monitor entries are lists over layers, the rest numbers
Monitor = Callable[[MLP], dict[str, list]]
MONITOR_PREFIX = "monitor_"  # history keys of monitor results: monitor_epoch, monitor_<key>
MONITOR_EPOCH = f"{MONITOR_PREFIX}epoch"
_HISTORY_KEYS = ("epoch", "mse", "weight_l2", "rate_l2", "loss")
# Extra history keys of SGLD runs: the gradient norm and the mean squared parameter
# displacement per step (see `SGLD.last_msd`), both of the step taken at the logged epoch.
_SGLD_KEYS = ("grad_norm", "param_msd")
_REQUIRED_KEYS = ("epoch", "net_state_dict", "optim_state_dict", "history", "rng_state")


def compute_loss(
    net: MLP, x: Tensor, y: Tensor, weight_lambda: float, rate_lambda: float
) -> tuple[Tensor, dict[str, Tensor]]:
    """0.5*N*mse + weight_lambda*weight_l2 + rate_lambda*rate_l2, plus its parts.

    `rate_l2` is the squared rates phi(h) summed over samples, neurons and layers, so it grows
    with the number of samples P. `weight_l2` is the plain sum of squares of every parameter.

    The parts are detached 0-d tensors: calling `.item()` here would force a host-device
    sync every epoch, so the caller converts only when it logs.
    """
    h, z = net(x)
    mse = (y - z).square().mean()
    weight_l2 = net.weight_l2()
    rate_l2 = net.phi(h).square().sum()
    loss = 0.5 * net.N * mse + weight_lambda * weight_l2 + rate_lambda * rate_l2
    parts = {"mse": mse, "weight_l2": weight_l2, "rate_l2": rate_l2, "loss": loss}
    return loss, {k: v.detach() for k, v in parts.items()}


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


def _payload(net: MLP, optim: torch.optim.Optimizer, history: History, epoch: int) -> dict:
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
    path: Path, cfg: Config, net: MLP, optim: torch.optim.Optimizer
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


def _log_line(epoch: int, n_epochs: int, parts: dict[str, float]) -> str:
    return (
        f"[{epoch:05d}/{n_epochs:05d}] mse={parts['mse']:.4f} "
        f"weight L2={parts['weight_l2']:.4f} rate L2={parts['rate_l2']:.4f} "
        f"loss={parts['loss']:.4f}"
    )


def _history_keys(optimizer: str) -> tuple[str, ...]:
    return _HISTORY_KEYS + (_SGLD_KEYS if optimizer == "sgld" else ())


def _record(history: History, epoch: int, parts: dict[str, float]) -> None:
    history["epoch"].append(epoch)
    for key, value in parts.items():
        history[key].append(value)


def _make_optimiser(net: MLP, tc: TrainConfig) -> torch.optim.Optimizer:
    if tc.optimizer == "sgld":
        return SGLD(net.parameters(), lr=tc.lr, beta=tc.beta)
    return torch.optim.Adam(net.parameters(), lr=tc.lr)


def _grad_norm(net: MLP) -> Tensor:
    return torch.sqrt(sum(p.grad.square().sum() for p in net.parameters()))


def _step(
    net: MLP, optim: torch.optim.Optimizer, dataset: Dataset, tc: TrainConfig
) -> dict[str, Tensor]:
    """One full-batch optimiser step; returns the loss parts from before the step (for SGLD
    also the gradient norm and the step's mean squared displacement)."""
    optim.zero_grad()
    loss, parts = compute_loss(net, dataset.x, dataset.y, tc.weight_lambda, tc.rate_lambda)
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


def _call_monitor(monitor: Monitor, net: MLP) -> dict[str, list]:
    """Run `monitor` without gradients and without advancing the global RNG: the training
    stream must not depend on whether, or how often, the net is being monitored."""
    cuda = [net.device] if net.device.type == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=cuda):
        return monitor(net)


def _check_net_matches_config(net: MLP, cfg: Config) -> None:
    if net.dtype != resolve_dtype(cfg.dtype):
        raise ValueError(f"net dtype {net.dtype} does not match config dtype {cfg.dtype!r}")
    if net.device.type != resolve_device(cfg).type:
        raise ValueError(f"net is on {net.device} but the config resolves to {resolve_device(cfg)}")


def train(
    cfg: Config,
    dataset: Dataset,
    net: MLP,
    cache_dir: Path,
    log: Callable[[str], None] = print,
    monitor: Monitor | None = None,
) -> tuple[MLP, History]:
    """Train `net` in place on the full dataset (Adam or SGLD, `train.optimizer`); resume from
    the latest checkpoint if any.

    If `final.pt` exists the net is loaded from it and nothing else happens (no optimiser,
    no RNG change). A checkpoint with `epoch=s` is written before the step of epoch s; the
    step, history entry and log line follow. An interruption at epoch t therefore resumes at
    the last multiple of `save_int` at or before t, repeating the deterministic steps since.

    `monitor` (required exactly when `train.monitor_int > 0`) is called on the net at the
    start of every epoch e with e % monitor_int == 0, after e's checkpoint and before its
    step, and once more after the last step (e = n_epochs). Its results enter the history as
    `monitor_epoch` and `monitor_<key>`. A checkpoint therefore never holds the entry of the
    epoch it is written at; the resumed run records it, so resuming reproduces the history.
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
        parts = _step(net, optim, dataset, tc)
        if epoch % tc.log_int == 0:
            values = {k: v.item() for k, v in parts.items()}
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
) -> tuple[MLP, History]:
    """Build the net the config describes for `dataset` and `train` it (see `train`).

    The one place that turns a config and a dataset into a net, so the CLI and the sweep
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


def _net_from_payload(payload: dict, cfg: Config, device: torch.device, dtype: torch.dtype) -> MLP:
    state = payload["net_state_dict"]
    net = make_net(cfg.net, state["U"].shape[1], state["V"].shape[1], device, dtype)
    net.load_state_dict(state)
    return net


def load_trained(
    cache_dir: Path, cfg: Config, device: torch.device, dtype: torch.dtype
) -> tuple[MLP, History]:
    """Rebuild the trained net and its history from `final.pt`."""
    if dtype != resolve_dtype(cfg.dtype):
        raise ValueError(f"dtype {dtype} does not match config dtype {cfg.dtype!r}")
    final = cache_paths(cache_dir).train / "final.pt"
    if not final.exists():
        raise FileNotFoundError(f"no trained model at {final}; run training first")
    payload = _load_payload(final, cfg, device)
    return _net_from_payload(payload, cfg, device, dtype), payload["history"]


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
) -> tuple[MLP, History, int]:
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
