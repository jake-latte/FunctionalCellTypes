"""Assemble sweep records into the arrays the grid figures draw."""

from collections.abc import Iterable, Sequence
from typing import NamedTuple

import numpy as np

# lambda values come back from JSON text, so equal settings can differ in the last digits
LAMBDA_RTOL = 1e-9


class MirRecord(NamedTuple):
    lambda_r: float
    lambda_w: float
    layer: int  # 0-based hidden layer
    mir: float


def _check_grid_axis(values: np.ndarray, what: str) -> None:
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{what} must be a non-empty 1-D sequence, got shape {values.shape}")
    close = np.isclose(values[:, None], values[None, :], rtol=LAMBDA_RTOL, atol=0)
    if close.sum() > values.size:  # more than the diagonal
        raise ValueError(f"{what} has near-duplicate values: {values.tolist()}")


def _index_of(values: np.ndarray, value: float, what: str) -> int:
    (matches,) = np.nonzero(np.isclose(values, value, rtol=LAMBDA_RTOL, atol=0))
    if matches.size == 0:
        raise ValueError(f"{what} {value} is not on the grid {values.tolist()}")
    return int(matches[0])


def _check_layer(layer: int, n_layers: int) -> None:
    if isinstance(layer, bool) or not isinstance(layer, int | np.integer):
        raise ValueError(f"layer must be an integer, got {layer!r}")
    if not 0 <= layer < n_layers:
        raise ValueError(f"layer {layer} is outside [0, {n_layers})")


def mir_grid(
    records: Iterable[MirRecord],
    lambda_r_values: Sequence[float],
    lambda_w_values: Sequence[float],
    n_layers: int,
) -> np.ndarray:
    """(n_layers, n_r, n_w) layer-average MIR from `MirRecord`s (0-based layers).

    A setting with no record is NaN (the heat-map draws it blank), including a whole layer
    with no records. Two records for one cell, a record off the grid or with a layer outside
    [0, n_layers), a non-finite MIR, a grid with near-duplicate values, or no records at all
    is an error.
    """
    records = list(records)
    if not records:
        raise ValueError("no records")
    lam_r = np.asarray(lambda_r_values, dtype=float)
    lam_w = np.asarray(lambda_w_values, dtype=float)
    _check_grid_axis(lam_r, "lambda_r_values")
    _check_grid_axis(lam_w, "lambda_w_values")
    grid = np.full((n_layers, lam_r.size, lam_w.size), np.nan)
    seen = set()
    for lr, lw, layer, value in records:
        _check_layer(layer, n_layers)
        if not np.isfinite(value):
            raise ValueError(f"non-finite mir {value} for lambda_r={lr}, lambda_w={lw}")
        cell = (layer, _index_of(lam_r, lr, "lambda_r"), _index_of(lam_w, lw, "lambda_w"))
        if cell in seen:
            raise ValueError(f"duplicate record for lambda_r={lr}, lambda_w={lw}, layer={layer}")
        seen.add(cell)
        grid[cell] = value
    return grid
