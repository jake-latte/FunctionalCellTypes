"""Assemble sweep records into the arrays the grid figures draw."""

from collections.abc import Iterable, Sequence

import numpy as np


def _index_of(values: Sequence[float], value: float, what: str) -> int:
    matches = [i for i, v in enumerate(values) if np.isclose(v, value, rtol=1e-9, atol=0)]
    if not matches:
        raise ValueError(f"{what} {value} is not on the grid {list(values)}")
    return matches[0]


def mir_grid(
    records: Iterable[tuple[float, float, int, float]],
    lambda_r_values: Sequence[float],
    lambda_w_values: Sequence[float],
) -> np.ndarray:
    """(L, n_r, n_w) layer-average MIR from `(lambda_r, lambda_w, layer, mir)` records.

    Layers are 0-based; L is the largest layer index + 1. A setting with no record is NaN
    (the heat-map draws it blank). Two records for one cell, a record off the grid, or no
    records at all is an error.
    """
    records = list(records)
    if not records:
        raise ValueError("no records")
    n_layers = max(layer for _, _, layer, _ in records) + 1
    grid = np.full((n_layers, len(lambda_r_values), len(lambda_w_values)), np.nan)
    seen = set()
    for lr, lw, layer, value in records:
        cell = (
            layer,
            _index_of(lambda_r_values, lr, "lambda_r"),
            _index_of(lambda_w_values, lw, "lambda_w"),
        )
        if cell in seen:
            raise ValueError(f"duplicate record for lambda_r={lr}, lambda_w={lw}, layer={layer}")
        seen.add(cell)
        grid[cell] = value
    return grid
