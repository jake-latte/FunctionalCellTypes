"""Assembling the (layer, lambda_r, lambda_w) MIR array from sweep records."""

import numpy as np
import pytest

from fct.grid_report import MirRecord, mir_grid

LR, LW = [0.1, 1.0], [0.01, 0.1, 1.0]


def _records(skip=None):
    return [
        MirRecord(lr, lw, layer, 100 * layer + 10 * i + j)
        for i, lr in enumerate(LR)
        for j, lw in enumerate(LW)
        for layer in (0, 1)
        if (i, j, layer) != skip
    ]


def test_mir_grid_puts_values_in_the_right_cells():
    grid = mir_grid(_records(), LR, LW, n_layers=2)
    assert grid.shape == (2, 2, 3)
    assert grid[1, 1, 2] == 112
    assert grid[0, 0, 1] == 1
    assert not np.isnan(grid).any()


def test_mir_grid_gives_nan_for_a_missing_setting_and_keeps_the_rest():
    grid = mir_grid(_records(skip=(1, 0, 1)), LR, LW, n_layers=2)
    assert np.isnan(grid[1, 1, 0])
    assert np.isnan(grid).sum() == 1
    assert grid[0, 1, 0] == 10


def test_mir_grid_gives_a_nan_plane_for_a_layer_without_records():
    grid = mir_grid(_records(), LR, LW, n_layers=3)
    assert grid.shape[0] == 3
    assert np.isnan(grid[2]).all()


def test_mir_grid_matches_lambdas_through_a_float_round_trip():
    grid = mir_grid([MirRecord(0.1 * (1 + 1e-12), 0.01, 0, 0.5)], LR, LW, n_layers=1)
    assert grid[0, 0, 0] == 0.5


def test_mir_grid_rejects_a_duplicate_cell():
    with pytest.raises(ValueError, match="duplicate"):
        mir_grid(_records() + [MirRecord(0.1, 0.01, 0, 0.5)], LR, LW, n_layers=2)


@pytest.mark.parametrize(
    ("record", "match"),
    [
        (MirRecord(0.2, 0.01, 0, 0.5), "not on the grid"),
        (MirRecord(0.1, 0.01, 2, 0.5), "outside"),
        (MirRecord(0.1, 0.01, -1, 0.5), "outside"),
        (MirRecord(0.1, 0.01, 0, float("nan")), "non-finite"),
    ],
)
def test_mir_grid_rejects_a_bad_record(record, match):
    with pytest.raises(ValueError, match=match):
        mir_grid([record], LR, LW, n_layers=2)


@pytest.mark.parametrize("layer", [True, 1.0])
def test_mir_grid_rejects_a_non_integer_layer(layer):
    with pytest.raises(TypeError, match="integer"):
        mir_grid([MirRecord(0.1, 0.01, layer, 0.5)], LR, LW, n_layers=2)


def test_mir_grid_rejects_no_records_and_near_duplicate_grid_values():
    with pytest.raises(ValueError, match="no records"):
        mir_grid([], LR, LW, n_layers=1)
    with pytest.raises(ValueError, match="near-duplicate"):
        mir_grid(_records(), [0.1, 0.1 * (1 + 1e-12)], LW, n_layers=2)
