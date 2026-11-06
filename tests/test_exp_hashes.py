"""Every preset file keeps the hashes the Python presets gave.

`fixtures/preset_hashes.json` was computed with the code that defined the presets in Python
(`lambda_exp` at its 7 log-spaced points per axis): for each preset the
experiment hash, and for each setting its train hash (`section_hash("data", "net", "train")`),
its mean-field (solve) hash and its directory. The train and solve hashes were recomputed when
the data section's `latent_range` became `latent_min` / `latent_max` (a field rename, so every
hash holding the data section moved); the experiment hashes and directories are the original
ones. A renamed or restructured experiment must not move any of them, or finished runs would be
taken for stale.
"""

import hashlib
import json
from pathlib import Path

import pytest

from fct.config import Config, load_config
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
from tests.pinned_runs import LEGACY_CONFIG_FILE

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


# --- the linear teacher's map type (`data.teacher`) -----------------------------------------
HASHED_SECTIONS = [
    ("data",),
    ("data", "net", "train"),
    ("data", "net", "train", "meanfield"),
    ("data", "net", "train", "analysis"),
]


def _with_teacher(cfg: Config, teacher: str) -> Config:
    dump = cfg.model_dump()
    return Config.model_validate({**dump, "data": {**dump["data"], "teacher": teacher}})


def _data_hash_without_teacher(cfg: Config) -> str:
    """The data hash as computed before `teacher` existed: the canonical dump lacking the key."""
    hashed = cfg.model_dump(include={"data", "dtype"})
    del hashed["data"]["teacher"]
    return hashlib.sha256(json.dumps(hashed, sort_keys=True).encode()).hexdigest()[:12]


@pytest.mark.parametrize("name", [*sorted(PINNED), "legacy_config"])
def test_the_gaussian_teacher_hashes_as_a_data_section_without_the_field(name):
    # Fails if section_hash keeps `teacher` for the Gaussian teacher. The train, solve and
    # result hashes that include the data are guarded by the pins (preset_hashes.json,
    # meanfield_hashes.json), which hold no `teacher`.
    if name == "legacy_config":
        configs = [load_config(LEGACY_CONFIG_FILE)]
    else:
        configs = [setting_config(preset(name), s) for s in expand(preset(name))]
    for cfg in configs:
        assert cfg.data.teacher == "gaussian"
        assert cfg.section_hash("data") == _data_hash_without_teacher(cfg)


@pytest.mark.parametrize("sections", HASHED_SECTIONS)
def test_an_orthonormal_teacher_moves_every_hash_that_holds_the_data(sections):
    cfg = load_config(LEGACY_CONFIG_FILE)
    assert _with_teacher(cfg, "orthonormal").section_hash(*sections) != cfg.section_hash(*sections)


def test_the_teacher_does_not_enter_hashes_without_the_data():
    cfg = load_config(LEGACY_CONFIG_FILE)
    ortho = _with_teacher(cfg, "orthonormal")
    for sections in [("net",), ("train",), ("meanfield",), ("net", "train")]:
        assert ortho.section_hash(*sections) == cfg.section_hash(*sections)
