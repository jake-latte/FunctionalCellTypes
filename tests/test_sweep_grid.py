"""The sweep over an arbitrary grid of Config fields: expansion, validation, naming, the
phase-transition preset, setting.json lookup, and a tiny end-to-end run (CPU)."""

import json

import numpy as np
import pytest
from pydantic import ValidationError

from fct.cache import cache_paths
from fct.cli import main
from fct.config import load_config
from fct.sweep import (
    SweepConfig,
    expand,
    find_setting_dir,
    get_setting,
    load_sweep,
    phase_transition,
    preset,
    read_setting,
    run_setting,
    save_sweep,
    setting_config,
    setting_dir,
    setting_status,
)

QUIET = {"log": lambda s: None}


def _with_grid(grid: dict, **updates) -> SweepConfig:
    """The tiny preset's sizes with an explicit grid."""
    tree = {**preset("tiny").model_dump(), "grid": grid, **updates}
    return SweepConfig.model_validate(tree)


@pytest.fixture
def small() -> SweepConfig:
    return _with_grid({"net.L": [1, 2], "net.phi": ["tanh", "relu"]}, n_epochs=4)


# ---------------------------------------------------------------------------- expansion
def test_expand_is_the_cartesian_product_with_the_first_axis_slowest():
    sweep = _with_grid(
        {"net.L": [1, 2], "train.rate_lambda": [0.1, 0.2, 0.3], "net.phi": ["tanh", "relu"]}
    )
    settings = expand(sweep)
    assert len(settings) == 2 * 3 * 2
    assert [s.index for s in settings] == list(range(12))
    values = [(s["net.L"], s["train.rate_lambda"], s["net.phi"]) for s in settings]
    assert len(set(values)) == 12
    assert values[:3] == [(1, 0.1, "tanh"), (1, 0.1, "relu"), (1, 0.2, "tanh")]
    assert values[-1] == (2, 0.3, "relu")


def test_axis_order_survives_the_json_round_trip(tmp_path):
    sweep = _with_grid({"train.rate_lambda": [0.1, 0.2], "net.L": [1, 2]})
    save_sweep(sweep, tmp_path)
    loaded = load_sweep(tmp_path)
    assert list(loaded.grid) == ["train.rate_lambda", "net.L"]
    assert expand(loaded) == expand(sweep)


def test_a_grid_axis_overrides_the_shared_value():
    sweep = _with_grid({"net.N": [4, 6]})
    assert [setting_config(sweep, s).net.N for s in expand(sweep)] == [4, 6]
    assert preset("tiny").N == 8  # the shared value is the default of the unswept axes


def test_the_task_axis_selects_that_tasks_parameters():
    sweep = _with_grid({"data.task": ["object_action", "what_where"]})
    configs = [setting_config(sweep, s) for s in expand(sweep)]
    assert [c.data.task for c in configs] == ["object_action", "what_where"]
    assert configs[0].data.D_in == 2 * sweep.object_action.grid_size**2
    assert [s.task for s in expand(sweep)] == ["object_action", "what_where"]


def test_without_a_task_axis_the_base_task_is_used():
    sweep = _with_grid({"net.L": [1, 2]}, task="what_where")
    assert {setting_config(sweep, s).data.task for s in expand(sweep)} == {"what_where"}


def test_get_setting_rejects_bad_indices(small):
    assert get_setting(small, 3) == expand(small)[3]
    with pytest.raises(ValueError, match=r"index 4 .* 0\.\.3"):
        get_setting(small, 4)


# ---------------------------------------------------------------------------- directories
def test_directories_are_index_prefixed_labelled_and_distinct(small, tmp_path):
    dirs = [setting_dir(tmp_path, s) for s in expand(small)]
    assert len(set(dirs)) == 4
    assert [d.parent for d in dirs] == [tmp_path / "linear_teacher"] * 4
    assert [d.name.split("_")[0] for d in dirs] == ["s0000", "s0001", "s0002", "s0003"]
    assert dirs[1].name == "s0001_L1_phirelu"


def test_near_equal_floats_never_share_a_directory(tmp_path):
    # `:g` prints both as 1e-07; the index keeps the directories apart.
    sweep = _with_grid({"train.rate_lambda": [1.00000001e-7, 1.00000002e-7]})
    dirs = {setting_dir(tmp_path, s) for s in expand(sweep)}
    assert len(dirs) == 2


# ---------------------------------------------------------------------------- phase transition
def test_phase_transition_preset_with_three_lambdas():
    sweep = phase_transition(3)
    settings = expand(sweep)
    assert len(settings) == 2 * 2 * 3 * 3
    configs = [setting_config(sweep, s) for s in settings]
    assert {c.data.task for c in configs} == {"linear_teacher"}
    assert {c.net.L for c in configs} == {1, 2}
    assert {c.net.phi for c in configs} == {"tanh", "relu"}
    for name in ("weight_lambda", "rate_lambda"):
        got = sorted({getattr(c.train, name) for c in configs})
        assert got == pytest.approx([1e-6, 1e-3, 1.0], rel=1e-12)
    pairs = {(c.train.weight_lambda, c.train.rate_lambda) for c in configs}
    assert len(pairs) == 9  # every weight/rate combination is a setting


def test_phase_transition_lambdas_are_log_spaced():
    sweep = phase_transition(8)
    for axis in ("train.weight_lambda", "train.rate_lambda"):
        assert sweep.grid[axis] == np.logspace(-6, 0, 8).tolist()
    assert len(expand(sweep)) == 2 * 2 * 8 * 8


def test_presets_that_need_n_lambda_say_so():
    with pytest.raises(ValueError, match="n_lambda"):
        preset("phase_transition")
    with pytest.raises(ValueError, match="n_lambda"):
        preset("tiny", n_lambda=3)
    with pytest.raises(ValueError, match="n_lambda"):
        phase_transition(0)
    assert len(expand(preset("phase_transition", n_lambda=2))) == 16


# ---------------------------------------------------------------------------- validation
@pytest.mark.parametrize(
    ("grid", "axis"),
    [
        ({"net.nope": [1, 2]}, "net.nope"),
        ({"nosection.L": [1]}, "nosection.L"),
        ({"net": [1]}, "net"),  # a section, not a field
        ({"net.L": [1, "two"]}, "net.L"),
        ({"net.phi": ["tanh", "swish"]}, "net.phi"),
        ({"train.lr": [-1.0]}, "train.lr"),
        ({"data.task": ["linear_teacher", "nope"]}, "data.task"),
        ({"net.L": []}, "net.L"),
        ({"net.L": [1, 1]}, "net.L"),
    ],
)
def test_bad_axes_and_values_fail_at_load_naming_the_axis(grid, axis):
    with pytest.raises(ValidationError, match=axis.replace(".", r"\.")):
        _with_grid(grid)


def test_a_bad_combination_of_axes_fails_at_load():
    # Each value is fine alone; together there are more objects than grid cells.
    grid = {"data.task": ["object_action"], "data.max_objects": [2, 99]}
    with pytest.raises(ValidationError, match="max_objects"):
        _with_grid(grid)


def test_an_invalid_grid_in_sweep_json_fails_at_load(tmp_path):
    tree = {**preset("tiny").model_dump(), "grid": {"net.typo": [1]}}
    (tmp_path / "sweep.json").write_text(json.dumps(tree))
    with pytest.raises(ValidationError, match=r"net\.typo"):
        load_sweep(tmp_path)


# ---------------------------------------------------------------------------- setting.json
def test_run_setting_records_the_axis_values_and_they_find_the_setting(small, tmp_path):
    for i in range(4):
        run_setting(small, i, tmp_path, **QUIET)
    target = expand(small)[2]  # L=2, tanh
    directory = find_setting_dir(tmp_path, {"net.L": 2, "net.phi": "tanh"})
    assert directory == setting_dir(tmp_path, target)
    assert read_setting(directory) == {"index": 2, "axes": {"net.L": 2, "net.phi": "tanh"}}
    # a subset of the axes is enough when it is unambiguous
    assert find_setting_dir(tmp_path, {"net.L": 1, "net.phi": "relu"}).name.startswith("s0001_")


def test_find_setting_dir_is_loud_about_no_match_and_ambiguity(small, tmp_path):
    for i in range(4):
        run_setting(small, i, tmp_path, **QUIET)
    with pytest.raises(LookupError, match="no setting"):
        find_setting_dir(tmp_path, {"net.L": 3})
    with pytest.raises(LookupError, match="4 settings"):
        find_setting_dir(tmp_path, {})
    with pytest.raises(LookupError, match="net.typo"):
        find_setting_dir(tmp_path, {"net.typo": 1})


def test_find_setting_dir_matches_floats_by_value(tmp_path):
    sweep = _with_grid({"train.rate_lambda": np.logspace(-6, 0, 3).tolist()}, n_epochs=2)
    run_setting(sweep, 1, tmp_path, **QUIET)
    assert find_setting_dir(tmp_path, {"train.rate_lambda": 1e-3}).name.startswith("s0001_")


# ---------------------------------------------------------------------------- end to end
def _cli(capsys, *args) -> tuple[int, str, str]:
    code = main(["sweep", *map(str, args)])
    out = capsys.readouterr()
    return code, out.out, out.err


TINY_SETS = [
    "N=8", "P=32", "n_epochs=4", "log_int=2", "save_int=2", "monitor_int=2",
    "monitor_bins=4", "monitor_shuffles=2", "linear_teacher.n_features=8", "device=cpu",
]  # fmt: skip


def _init_2x2(capsys, root):
    args = ["init", root, "--preset", "phase_transition", "--n-lambda", 2]
    args += ["--axis", "net.L=[1]", "--axis", 'net.phi=["tanh"]']
    for item in TINY_SETS:
        args += ["--set", item]
    return _cli(capsys, *args)


def test_cli_2x2_grid_count_list_status_run_agree(tmp_path, capsys):
    assert _init_2x2(capsys, tmp_path)[0] == 0
    _, count, _ = _cli(capsys, "count", tmp_path)
    _, listing, _ = _cli(capsys, "list", tmp_path)
    assert int(count) == 4 == len(listing.strip().splitlines())
    assert listing.splitlines()[3].split()[1].startswith("linear_teacher/s0003_")

    assert _cli(capsys, "run", tmp_path, "--index", 3)[0] == 0
    _, status, _ = _cli(capsys, "status", tmp_path)
    lines = status.strip().splitlines()
    assert [ln.split()[1] for ln in lines[:4]] == ["missing", "missing", "missing", "done"]
    assert "1 done, 0 partial, 3 missing, 0 stale" in lines[-1]

    sweep = load_sweep(tmp_path)
    cfg = load_config(cache_paths(setting_dir(tmp_path, expand(sweep)[3])).config)
    assert (cfg.net.L, cfg.net.phi) == (1, "tanh")
    assert cfg.train.weight_lambda == pytest.approx(1.0)
    assert cfg.train.rate_lambda == pytest.approx(1.0)


def test_rerun_and_resume_behave_as_before_on_a_grid(small, tmp_path):
    run_setting(small, 0, tmp_path, **QUIET)
    final = cache_paths(setting_dir(tmp_path, expand(small)[0])).train / "final.pt"
    before = (final.read_bytes(), final.stat().st_mtime_ns)
    lines: list[str] = []
    run_setting(small, 0, tmp_path, log=lines.append)
    assert (final.read_bytes(), final.stat().st_mtime_ns) == before
    assert any("already done" in line for line in lines)
    assert setting_status(tmp_path, small, expand(small)[0]) == "done"


def test_cli_init_axis_errors_are_one_line(tmp_path, capsys):
    code, _, err = _cli(capsys, "init", tmp_path, "--preset", "phase_transition", "--n-lambda", 2,
                        "--axis", "net.typo=[1]")  # fmt: skip
    assert code == 1 and err.startswith("error:") and "net.typo" in err
    code, _, err = _cli(capsys, "init", tmp_path, "--preset", "phase_transition")
    assert code == 1 and "n_lambda" in err
    code, _, err = _cli(capsys, "init", tmp_path, "--preset", "tiny", "--axis", "net.L=[1]")
    assert code == 1 and "legacy" in err
    assert not (tmp_path / "sweep.json").exists()
