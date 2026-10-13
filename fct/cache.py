"""Cache-directory layout and artefact I/O.

Artefacts are `.pt` files holding plain tensors / scalars / lists / dicts plus the hash of the
config sections they depend on, so loading with a different config fails loudly instead of
silently mixing results from different settings.
"""

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch


class StaleCacheError(RuntimeError):
    """An artefact was produced under a different config than the one now in use."""


@dataclass(frozen=True)
class CachePaths:
    config: Path
    data: Path
    train: Path
    meanfield: Path
    analysis: Path
    figures: Path


def cache_paths(cache_dir: Path) -> CachePaths:
    root = Path(cache_dir)
    return CachePaths(
        config=root / "config.json",
        data=root / "data",
        train=root / "train",
        meanfield=root / "meanfield",
        analysis=root / "analysis",
        figures=root / "figures",
    )


def save_artefact(path: Path, payload: dict, config_hash: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write then rename: an interrupted save never leaves a truncated file at `path`.
    tmp = path.with_name(path.name + ".tmp")
    torch.save({**payload, "config_hash": config_hash}, tmp)
    os.replace(tmp, path)


def load_artefact(path: Path, expected_hash: str, device) -> dict:
    path = Path(path)
    payload = torch.load(path, map_location=device, weights_only=True)
    found = payload.get("config_hash")
    if found != expected_hash:
        raise StaleCacheError(
            f"{path} was written with config hash {found!r} but the current config has hash "
            f"{expected_hash!r}. Use a fresh cache directory or delete this file."
        )
    return payload


def load_or_compute(
    path: Path, expected_hash: str, compute: Callable[[], dict], device
) -> dict:
    """Load the artefact at `path` if it exists, else compute, save and return it."""
    path = Path(path)
    if path.exists():
        return load_artefact(path, expected_hash, device)
    payload = compute()
    save_artefact(path, payload, expected_hash)
    return load_artefact(path, expected_hash, device)


def _step_of(path: Path, prefix: str) -> int | None:
    match = re.fullmatch(re.escape(prefix) + r"(\d+)\.pt", path.name)
    return int(match.group(1)) if match else None


def latest_checkpoint(dir: Path, prefix: str = "ckpt_") -> Path | None:
    """Checkpoint with the highest step number, or None if there are none."""
    steps = {p: _step_of(p, prefix) for p in Path(dir).glob(f"{prefix}*.pt")}
    steps = {p: s for p, s in steps.items() if s is not None}
    return max(steps, key=steps.get) if steps else None


def write_checkpoint(
    dir: Path, prefix: str, step: int, payload: dict, config_hash: str, width: int = 6
) -> Path:
    """Save `<prefix><step>.pt`, then delete older checkpoints with the same prefix.

    The new file is written first so an interruption never leaves zero checkpoints.
    `width` zero-pads the step in the file name (6 for train, 5 for meanfield).
    """
    dir = Path(dir)
    new = dir / f"{prefix}{step:0{width}d}.pt"
    save_artefact(new, payload, config_hash)
    for old in dir.glob(f"{prefix}*.pt"):
        if old != new and _step_of(old, prefix) is not None:
            old.unlink()
    return new
