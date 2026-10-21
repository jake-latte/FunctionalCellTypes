"""Full-batch training of the MLP with periodic checkpoints and resume.

Layout under `cache_dir/train/`: `ckpt_<epoch>.pt` (only the latest kept) and `final.pt`.
Both share one payload shape; resuming and `load_trained` read them through the same
validating loader (`_load_payload`) and write the net with `load_state_dict`.
"""

from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor

from fct.cache import cache_paths, latest_checkpoint, load_artefact, save_artefact, write_checkpoint
from fct.config import Config, TrainConfig, resolve_device, resolve_dtype
from fct.data import Dataset
from fct.model import MLP, make_net

History = dict[str, list]  # monitor entries are lists over layers, the rest numbers
Monitor = Callable[[MLP], dict[str, list]]
MONITOR_PREFIX = "monitor_"  # history keys of monitor results: monitor_epoch, monitor_<key>
MONITOR_EPOCH = f"{MONITOR_PREFIX}epoch"
_HISTORY_KEYS = ("epoch", "mse", "weight_l2", "rate_l2", "loss")
_REQUIRED_KEYS = ("epoch", "net_state_dict", "optim_state_dict", "history", "rng_state")


def compute_loss(
    net: MLP, x: Tensor, y: Tensor, weight_lambda: float, rate_lambda: float
) -> tuple[Tensor, dict[str, Tensor]]:
    """0.5*N*mse + weight_lambda*weight_l2 + rate_lambda*rate_l2, plus its parts.

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


def _record(history: History, epoch: int, parts: dict[str, float]) -> None:
    history["epoch"].append(epoch)
    for key in _HISTORY_KEYS[1:]:
        history[key].append(parts[key])


def _step(
    net: MLP, optim: torch.optim.Optimizer, dataset: Dataset, tc: TrainConfig
) -> dict[str, Tensor]:
    """One full-batch Adam step; returns the loss parts from before the step."""
    optim.zero_grad()
    loss, parts = compute_loss(net, dataset.x, dataset.y, tc.weight_lambda, tc.rate_lambda)
    loss.backward()
    optim.step()
    return parts


def _check_monitor_matches_config(cfg: Config, monitor: Monitor | None) -> None:
    if cfg.train.monitor_int > 0 and monitor is None:
        raise ValueError(f"train.monitor_int={cfg.train.monitor_int} needs a monitor, got None")
    if cfg.train.monitor_int == 0 and monitor is not None:
        raise ValueError("a monitor was passed but train.monitor_int=0 disables monitoring")


def _new_history(monitored: bool) -> History:
    history: History = {key: [] for key in _HISTORY_KEYS}
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
    """Train `net` in place on the full dataset; resume from the latest checkpoint if any.

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

    optim = torch.optim.Adam(net.parameters(), lr=tc.lr)
    ckpt = latest_checkpoint(train_dir)
    if ckpt is None:
        # Nothing in the full-batch loop draws from the global RNG today; the state is kept
        # for future stochastic steps and for downstream code.
        torch.manual_seed(tc.seed)
        start, history = 0, _new_history(monitor is not None)
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
    state = payload["net_state_dict"]
    net = make_net(cfg.net, state["U"].shape[1], state["V"].shape[1], device, dtype)
    net.load_state_dict(state)
    return net, payload["history"]
