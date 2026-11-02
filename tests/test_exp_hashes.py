"""Every preset file keeps the hashes the Python presets gave.

`fixtures/preset_hashes.json` was computed with the code that defined the presets in Python
(`lambda_exp` at its 7 log-spaced points per axis): for each preset the
experiment hash, and for each setting its train hash (`section_hash("data", "net", "train")`),
its mean-field (solve) hash and its directory. A renamed or restructured experiment must not
move any of them, or finished runs would be taken for stale.
"""

import json
from pathlib import Path

import pytest

from fct.exp import (
    ExpConfig,
    exp_hash,
    expand,
    preset,
    preset_names,
    run_setting,
    same_experiment,
    save_exp,
    setting_config,
    with_axes,
)
from fct.meanfield import meanfield_hash

PINNED = json.loads((Path(__file__).parent / "fixtures" / "preset_hashes.json").read_text())


def test_the_pins_cover_every_preset():
    assert set(PINNED) == set(preset_names())


@pytest.mark.parametrize("name", sorted(PINNED))
def test_a_preset_keeps_its_hashes(name):
    pinned = PINNED[name]
    exp = preset(name)
    settings = expand(exp)
    configs = [setting_config(exp, s) for s in settings]
    assert exp_hash(exp) == pinned["exp_hash"]
    assert len(settings) == pinned["n_settings"]
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


def test_a_shared_B_and_a_one_value_B_axis_hash_alike_but_name_different_directories():
    """Both spellings train the same networks, so train and solve hashes and Configs agree.
    The axis spelling puts `_B8` in the directory name and in `exp_hash`, so the experiments
    are different ones to `same_experiment` and the directories differ."""
    base = ExpConfig.model_validate({**preset("tiny").model_dump(), "B": 8})
    shared = with_axes(base, {"net.L": [1, 2]})
    axis = with_axes(preset("tiny"), {"net.B": [8], "net.L": [1, 2]})
    shared_configs = [setting_config(shared, s) for s in expand(shared)]
    axis_configs = [setting_config(axis, s) for s in expand(axis)]
    assert shared_configs == axis_configs
    assert [c.section_hash("data", "net", "train") for c in shared_configs] == [
        c.section_hash("data", "net", "train") for c in axis_configs
    ]
    assert [meanfield_hash(c) for c in shared_configs] == [meanfield_hash(c) for c in axis_configs]
    shared_paths = [s.relpath for s in expand(shared)]
    assert shared_paths != [s.relpath for s in expand(axis)]
    assert exp_hash(shared) != exp_hash(axis)
    assert not same_experiment(shared, axis)
