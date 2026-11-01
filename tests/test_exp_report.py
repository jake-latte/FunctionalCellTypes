"""Assembling the (layer, lambda_r, lambda_w) MIR array from experiment records."""

import numpy as np
import pytest

from fct.exp_report import MirRecord, mir_matrix

LR, LW = [0.1, 1.0], [0.01, 0.1, 1.0]


def _records(skip=None):
    return [
        MirRecord(lr, lw, layer, 100 * layer + 10 * i + j)
        for i, lr in enumerate(LR)
        for j, lw in enumerate(LW)
        for layer in (0, 1)
        if (i, j, layer) != skip
    ]


def test_mir_matrix_puts_values_in_the_right_cells():
    matrix = mir_matrix(_records(), LR, LW, n_layers=2)
    assert matrix.shape == (2, 2, 3)
    assert matrix[1, 1, 2] == 112
    assert matrix[0, 0, 1] == 1
    assert not np.isnan(matrix).any()


def test_mir_matrix_gives_nan_for_a_missing_setting_and_keeps_the_rest():
    matrix = mir_matrix(_records(skip=(1, 0, 1)), LR, LW, n_layers=2)
    assert np.isnan(matrix[1, 1, 0])
    assert np.isnan(matrix).sum() == 1
    assert matrix[0, 1, 0] == 10


def test_mir_matrix_gives_a_nan_plane_for_a_layer_without_records():
    matrix = mir_matrix(_records(), LR, LW, n_layers=3)
    assert matrix.shape[0] == 3
    assert np.isnan(matrix[2]).all()


def test_mir_matrix_matches_lambdas_through_a_float_round_trip():
    matrix = mir_matrix([MirRecord(0.1 * (1 + 1e-12), 0.01, 0, 0.5)], LR, LW, n_layers=1)
    assert matrix[0, 0, 0] == 0.5


def test_mir_matrix_rejects_a_duplicate_cell():
    with pytest.raises(ValueError, match="duplicate"):
        mir_matrix(_records() + [MirRecord(0.1, 0.01, 0, 0.5)], LR, LW, n_layers=2)


@pytest.mark.parametrize(
    ("record", "match"),
    [
        (MirRecord(0.2, 0.01, 0, 0.5), "not among the values"),
        (MirRecord(0.1, 0.01, 2, 0.5), "outside"),
        (MirRecord(0.1, 0.01, -1, 0.5), "outside"),
        (MirRecord(0.1, 0.01, 0, float("nan")), "non-finite"),
    ],
)
def test_mir_matrix_rejects_a_bad_record(record, match):
    with pytest.raises(ValueError, match=match):
        mir_matrix([record], LR, LW, n_layers=2)


@pytest.mark.parametrize("layer", [True, 1.0])
def test_mir_matrix_rejects_a_non_integer_layer(layer):
    with pytest.raises(TypeError, match="integer"):
        mir_matrix([MirRecord(0.1, 0.01, layer, 0.5)], LR, LW, n_layers=2)


def test_mir_matrix_rejects_no_records_and_near_duplicate_axis_values():
    with pytest.raises(ValueError, match="no records"):
        mir_matrix([], LR, LW, n_layers=1)
    with pytest.raises(ValueError, match="near-duplicate"):
        mir_matrix(_records(), [0.1, 0.1 * (1 + 1e-12)], LW, n_layers=2)
