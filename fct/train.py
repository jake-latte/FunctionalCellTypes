"""Full-batch training of the MLP with periodic checkpoints and resume.

Layout under `cache_dir/train/`: `ckpt_<epoch>.pt` (only the latest kept) and `final.pt`.
Both share one payload shape, so resuming and `load_trained` use the same loader.
"""

from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor

from fct.cache import cache_paths, latest_checkpoint, load_artefact, save_artefact, write_checkpoint
from fct.config import Config
from fct.data import Dataset
from fct.model import MLP, make_net

History = dict[str, list[float]]
_HISTORY_KEYS = ("epoch", "mse", "weight_l2", "rate_l2", "loss")


def compute_loss(
    net: MLP, x: Tensor, y: Tensor, weight_lambda: float, rate_lambda: float
) -> tuple[Tensor, dict[str, float]]:
    """0.5*N*mse + weight_lambda*weight_l2 + rate_lambda*rate_l2, plus its parts as floats."""
    h, z = net(x)
    mse = (y - z).square().mean()
    weight_l2 = net.weight_l2()
    rate_l2 = net.phi(h).square().sum()
    loss = 0.5 * net.N * mse + weight_lambda * weight_l2 + rate_lambda * rate_l2
    parts = {
        "mse": mse.item(),
        "weight_l2": weight_l2.item(),
        "rate_l2": rate_l2.item(),
        "loss": loss.item(),
    }
    return loss, parts


def _rng_state(device: torch.device) -> dict[str, Tensor]:
    state = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def _restore_rng(state: dict[str, Tensor]) -> None:
    # set_rng_state needs CPU ByteTensors, but load_artefact maps tensors to the model device.
    torch.set_rng_state(state["cpu"].cpu())
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"].cpu())


def _optim_to_saveable(state: dict) -> dict:
    """Optimiser per-parameter state is keyed by int index; saved artefacts need str keys."""
    return {**state, "state": {str(i): v for i, v in state["state"].items()}}


def _optim_from_saved(state: dict) -> dict:
    return {**state, "state": {int(i): v for i, v in state["state"].items()}}


def _payload(net: MLP, optim, history: History, epoch: int) -> dict:
    """`epoch` is the next epoch to run."""
    return {
        "epoch": epoch,
        "net_state_dict": dict(net.state_dict()),  # plain dict: the cache rejects OrderedDict
        "optim_state_dict": _optim_to_saveable(optim.state_dict()),
        "history": history,
        "rng_state": _rng_state(net.device),
    }


def _restore(payload: dict, net: MLP, optim) -> tuple[int, History]:
    net.load_state_dict(payload["net_state_dict"])
    optim.load_state_dict(_optim_from_saved(payload["optim_state_dict"]))
    _restore_rng(payload["rng_state"])
    return payload["epoch"], payload["history"]


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


def train(
    cfg: Config,
    dataset: Dataset,
    net: MLP,
    cache_dir: Path,
    log: Callable[[str], None] = print,
) -> tuple[MLP, History]:
    """Train `net` in place on the full dataset; resume from the latest checkpoint if any.

    If `final.pt` exists the net is loaded from it and no epoch is run. A checkpoint with
    `epoch=s` is written before the step of epoch s, so the step, its history entry and
    its log line (in that order) all come after the save: an interruption inside `log` at
    epoch s resumes at s, repeating that deterministic step instead of skipping it.
    """
    tc = cfg.train
    train_dir = cache_paths(cache_dir).train
    config_hash = cfg.section_hash("data", "net", "train")
    final = train_dir / "final.pt"
    optim = torch.optim.Adam(net.parameters(), lr=tc.lr)

    if final.exists():
        _, history = _restore(load_artefact(final, config_hash, net.device), net, optim)
        return net, history

    ckpt = latest_checkpoint(train_dir)
    if ckpt is None:
        torch.manual_seed(tc.seed)
        start, history = 0, {key: [] for key in _HISTORY_KEYS}
    else:
        start, history = _restore(load_artefact(ckpt, config_hash, net.device), net, optim)
        log(f"resumed from {ckpt} at epoch {start}")

    for epoch in range(start, tc.n_epochs):
        if epoch > start and epoch % tc.save_int == 0:
            # `epoch > start` avoids rewriting the checkpoint we have just loaded.
            write_checkpoint(
                train_dir, "ckpt_", epoch, _payload(net, optim, history, epoch), config_hash
            )
        optim.zero_grad()
        loss, parts = compute_loss(net, dataset.x, dataset.y, tc.weight_lambda, tc.rate_lambda)
        loss.backward()
        optim.step()
        if epoch % tc.log_int == 0:
            _record(history, epoch, parts)
            log(_log_line(epoch, tc.n_epochs, parts))

    payload = _payload(net, optim, history, tc.n_epochs)
    save_artefact(final, payload, config_hash)
    return net, history


def load_trained(
    cache_dir: Path, cfg: Config, device: torch.device, dtype: torch.dtype
) -> tuple[MLP, History]:
    """Rebuild the trained net and its history from `final.pt`."""
    final = cache_paths(cache_dir).train / "final.pt"
    if not final.exists():
        raise FileNotFoundError(f"no trained model at {final}; run training first")
    payload = load_artefact(final, cfg.section_hash("data", "net", "train"), device)
    state = payload["net_state_dict"]
    net = make_net(cfg.net, state["U"].shape[1], state["V"].shape[1], device, dtype)
    net.load_state_dict(state)
    return net, payload["history"]
