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


# Exact types, not isinstance: np.float64 subclasses float but does not load with weights_only.
_SCALARS = (int, float, bool, str, type(None))


def _check_loadable(obj, where: str) -> None:
    """Raise TypeError unless `obj` survives torch.load(weights_only=True).

    A file that cannot be reloaded would poison the cache, so reject it before writing.
    """
    if isinstance(obj, torch.Tensor) or type(obj) in _SCALARS:
        return
    if type(obj) is dict:
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(f"{where}: dict keys must be str, got {key!r}")
            _check_loadable(value, f"{where}.{key}" if where else key)
    elif type(obj) in (list, tuple):
        for i, value in enumerate(obj):
            _check_loadable(value, f"{where}[{i}]")
    else:
        raise TypeError(f"{where}: {type(obj).__name__} is not weights_only-loadable")


def save_artefact(path: str | Path, payload: dict, config_hash: str) -> None:
    if "config_hash" in payload:
        raise ValueError("payload must not contain the reserved key 'config_hash'")
    _check_loadable(payload, "")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write then rename: an interrupted save never leaves a truncated file at `path`.
    tmp = path.with_name(path.name + ".tmp")
    torch.save({**payload, "config_hash": config_hash}, tmp)
    os.replace(tmp, path)


def load_artefact(path: str | Path, expected_hash: str, device: torch.device | str) -> dict:
    if not expected_hash:
        raise ValueError("expected_hash must be a non-empty string")
    path = Path(path)
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} holds a {type(payload).__name__}, not an artefact dict")
    found = payload.get("config_hash")
    if found != expected_hash:
        raise StaleCacheError(
            f"{path} was written with config hash {found!r} but the current config has hash "
            f"{expected_hash!r}. Use a fresh cache directory or delete this file."
        )
    return payload


def load_or_compute(
    path: str | Path,
    expected_hash: str,
    compute: Callable[[], dict],
    device: torch.device | str,
) -> dict:
    """Load the artefact at `path` if it exists, else compute, save and return it."""
    path = Path(path)
    if path.exists():
        return load_artefact(path, expected_hash, device)
    save_artefact(path, compute(), expected_hash)
    # Re-read after saving so a hit and a miss return identical types and devices, and a
    # payload that cannot be reloaded fails now rather than on the next run.
    try:
        return load_artefact(path, expected_hash, device)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _step_of(path: Path, prefix: str) -> int | None:
    match = re.fullmatch(re.escape(prefix) + r"(\d+)\.pt", path.name)
    return int(match.group(1)) if match else None


def latest_checkpoint(ckpt_dir: str | Path, prefix: str = "ckpt_") -> Path | None:
    """Checkpoint with the highest step number, or None if there are none."""
    steps = {p: _step_of(p, prefix) for p in Path(ckpt_dir).glob(f"{prefix}*.pt")}
    steps = {p: s for p, s in steps.items() if s is not None}
    return max(steps, key=steps.get) if steps else None


def write_checkpoint(
    ckpt_dir: str | Path,
    prefix: str,
    step: int,
    payload: dict,
    config_hash: str,
    width: int = 6,
) -> Path:
    """Save `<prefix><step>.pt`, then delete checkpoints with the same prefix and a lower step.

    The new file is written first so an interruption never leaves zero checkpoints, and
    higher-step files are kept so an out-of-order write cannot destroy newer progress.
    `width` zero-pads the step in the file name (6 for train, 5 for meanfield).
    """
    ckpt_dir = Path(ckpt_dir)
    new = ckpt_dir / f"{prefix}{step:0{width}d}.pt"
    save_artefact(new, payload, config_hash)
    for old in ckpt_dir.glob(f"{prefix}*.pt"):
        old_step = _step_of(old, prefix)
        if old != new and old_step is not None and old_step < step:
            old.unlink()
    return new
