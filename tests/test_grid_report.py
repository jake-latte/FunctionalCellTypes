"""Assembling the (layer, lambda_r, lambda_w) MIR array from sweep records."""

import numpy as np
import pytest

from fct.grid_report import mir_grid

LR, LW = [0.1, 1.0], [0.01, 0.1, 1.0]


def _records(skip=None):
    out = []
    for i, lr in enumerate(LR):
        for j, lw in enumerate(LW):
            for layer in (0, 1):
                if (i, j, layer) != skip:
                    out.append((lr, lw, layer, 100 * layer + 10 * i + j))
    return out


def test_values_land_in_the_right_cells():
    grid = mir_grid(_records(), LR, LW)
    assert grid.shape == (2, 2, 3)
    assert grid[1, 1, 2] == 112
    assert grid[0, 0, 1] == 1
    assert not np.isnan(grid).any()


def test_missing_setting_is_nan_and_the_rest_intact():
    grid = mir_grid(_records(skip=(1, 0, 1)), LR, LW)
    assert np.isnan(grid[1, 1, 0])
    assert np.isnan(grid).sum() == 1
    assert grid[0, 1, 0] == 10


def test_duplicate_cell_is_an_error():
    with pytest.raises(ValueError, match="duplicate"):
        mir_grid(_records() + [(0.1, 0.01, 0, 0.5)], LR, LW)


def test_record_outside_the_grid_is_an_error():
    with pytest.raises(ValueError, match="not on the grid"):
        mir_grid([(0.2, 0.01, 0, 0.5)], LR, LW)


def test_no_records_is_an_error():
    with pytest.raises(ValueError, match="no records"):
        mir_grid([], LR, LW)
