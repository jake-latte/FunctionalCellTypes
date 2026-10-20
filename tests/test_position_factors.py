"""The position task keeps its numbers and gains per-coordinate factors."""

import pytest
import torch

from fct.config import Config, DataConfig
from fct.data import Dataset, get_dataset, make_dataset

CPU = torch.device("cpu")
F64 = torch.float64

# Values produced by the position task before factors existed; x and y of a fixed config must
# stay bit-identical so that existing experiments reproduce.
POSITION_Y = [
    [-0.20759961775316083, 1.1117716865964231],
    [1.339007960601311, -1.5759204280167223],
    [2.3911383270247164, -1.2654777227154987],
]
POSITION_X = {
    "linear": [
        [0.4180963228550089, -0.9238224490187025, 0.4070761463994133, 0.29183975559461334],
        [-1.3139190255154933, 1.5897448548511628, 0.06728591375409565, -0.13515994475819176],
        [-1.9635704256532072, 1.6295556758231127, 0.8655743380775202, 0.2422744596737085],
    ],
    "fourier": [
        [-0.1889294537044553, 0.9819906626454938, 0.9464290528790364, -0.32291182676775726],
        [-0.7987465738217839, -0.6016676082422598, -0.5226682118702537, 0.8525361812265516],
        [-0.4200655770183759, -0.9074937525979002, -0.7987465738217846, -0.601667608242259],
    ],
}


def _cfg(**kw) -> DataConfig:
    return DataConfig(**{"P": 50, "D_in": 8, "d": 2, "var": 2.0, **kw})


@pytest.mark.parametrize("embedding", ["linear", "fourier"])
def test_position_task_values_are_unchanged(embedding):
    ds = make_dataset(_cfg(P=3, D_in=4, embedding=embedding, seed=7), CPU, F64)
    assert ds.x.tolist() == POSITION_X[embedding]
    assert ds.y.tolist() == POSITION_Y


def test_position_factors_are_the_columns_of_y():
    ds = make_dataset(_cfg(d=3), CPU, F64)
    assert list(ds.factors) == ["y0", "y1", "y2"]
    for i in range(3):
        assert torch.equal(ds.factors[f"y{i}"], ds.y[:, i])
        assert ds.factors[f"y{i}"].dtype == F64
    assert ds.groups == {"y0": ["y0"], "y1": ["y1"], "y2": ["y2"]}


def test_dataset_factor_fields_default_to_none():
    x = torch.zeros(2, 3)
    ds = Dataset(x, x, None, None)
    assert ds.factors is None and ds.groups is None and ds.extras is None


def test_get_dataset_round_trips_factors_and_groups(tmp_path):
    cfg = Config(data=_cfg(), device="cpu", dtype="float64")
    first = get_dataset(tmp_path, cfg)
    second = get_dataset(tmp_path, cfg)
    assert second.groups == first.groups == {"y0": ["y0"], "y1": ["y1"]}
    assert set(second.factors) == set(first.factors)
    for name, value in first.factors.items():
        assert torch.equal(second.factors[name], value)
