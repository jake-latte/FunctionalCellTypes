"""Every preset keeps the hashes the code gave before `grid_hash` became `exp_hash`.

`fixtures/preset_hashes.json` was computed with the pre-rename code: for each preset the
experiment hash, and for each setting its train hash (`section_hash("data", "net", "train")`),
its mean-field (solve) hash and its directory. A renamed or restructured experiment must not
move any of them, or finished runs would be taken for stale.
"""

import json
from pathlib import Path

import pytest

from fct.exp import (
    LAMBDA_PRESETS,
    PRESET_NAMES,
    exp_hash,
    expand,
    preset,
    run_setting,
    save_exp,
    setting_config,
    with_axes,
)
from fct.meanfield import meanfield_hash

PINNED = json.loads((Path(__file__).parent / "fixtures" / "preset_hashes.json").read_text())
N_LAMBDAS = (1, 3)


def _pinned_key(name: str, n_lambda: int | None) -> str:
    return name if n_lambda is None else f"{name}:{n_lambda}"


CASES = [
    (name, n) for name in PRESET_NAMES for n in (N_LAMBDAS if name in LAMBDA_PRESETS else (None,))
]


def test_the_pins_cover_every_preset():
    assert set(PINNED) == {_pinned_key(*case) for case in CASES}


@pytest.mark.parametrize(("name", "n_lambda"), CASES, ids=lambda c: str(c))
def test_a_preset_keeps_its_hashes(name, n_lambda):
    pinned = PINNED[_pinned_key(name, n_lambda)]
    exp = preset(name, n_lambda)
    settings = expand(exp)
    configs = [setting_config(exp, s) for s in settings]
    assert exp_hash(exp) == pinned["exp_hash"]
    assert [str(s.relpath) for s in settings] == pinned["relpaths"]
    assert [c.section_hash("data", "net", "train") for c in configs] == pinned["train"]
    assert [meanfield_hash(c) for c in configs] == pinned["solve"]


def test_the_hash_is_the_same_for_the_same_axes_however_the_experiment_is_built():
    """`exp_hash` depends on what the axes set, not on how the experiment object was made."""
    built = with_axes(preset("tiny"), {"net.L": [1, 2]})
    rebuilt = type(built).model_validate_json(built.model_dump_json())
    assert exp_hash(built) == exp_hash(rebuilt)


def test_the_setting_record_stores_the_hash_under_exp_hash(tmp_path):
    exp = with_axes(preset("tiny"), {"net.L": [1, 2]})
    save_exp(exp, tmp_path)
    run_setting(exp, 0, tmp_path, log=lambda s: None)
    record = json.loads(next(tmp_path.glob("*/*/setting.json")).read_text())
    assert record["exp_hash"] == exp_hash(exp)
    assert "grid_hash" not in record
