"""An experiment over arbitrary axes of Config fields: expansion, validation, naming, the
phase-transition preset, setting.json lookup, and a tiny end-to-end run (CPU)."""

import json

import numpy as np
import pytest
from pydantic import ValidationError

from fct.cache import cache_paths
from fct.cli import main
from fct.config import load_config
from fct.exp import (
    ExpConfig,
    ExpError,
    exp_hash,
    expand,
    find_setting_dir,
    get_setting,
    load_exp,
    preset,
    read_setting,
    run_setting,
    same_value,
    save_exp,
    setting_config,
    setting_dir,
    setting_status,
    with_axes,
)
from tests.report_helpers import QUIET


def _with_axes(axes: dict, **updates) -> ExpConfig:
    """The tiny preset's sizes (with `updates`) over `axes`."""
    return with_axes(preset("tiny").model_copy(update=updates), axes)


def _with_B(**updates) -> ExpConfig:
    """The tiny preset with `updates`, built through validation."""
    return ExpConfig.model_validate({**preset("tiny").model_dump(), **updates})


@pytest.fixture
def small() -> ExpConfig:
    return _with_axes({"net.L": [1, 2], "net.phi": ["tanh", "relu"]})


# ---------------------------------------------------------------------------- expansion
def test_expand_is_the_cartesian_product_with_the_first_axis_slowest():
    exp = _with_axes(
        {"net.L": [1, 2], "train.rate_lambda": [0.1, 0.2, 0.3], "net.phi": ["tanh", "relu"]}
    )
    settings = expand(exp)
    assert len(settings) == 2 * 3 * 2
    assert [s.index for s in settings] == list(range(12))
    values = [(s.axes["net.L"], s.axes["train.rate_lambda"], s.axes["net.phi"]) for s in settings]
    assert len(set(values)) == 12
    assert values[:3] == [(1, 0.1, "tanh"), (1, 0.1, "relu"), (1, 0.2, "tanh")]
    assert values[-1] == (2, 0.3, "relu")


def test_axis_order_survives_the_json_round_trip(tmp_path):
    exp = _with_axes({"train.rate_lambda": [0.1, 0.2], "net.L": [1, 2]})
    save_exp(exp, tmp_path)
    loaded = load_exp(tmp_path)
    assert list(loaded.axes) == ["train.rate_lambda", "net.L"]
    assert expand(loaded) == expand(exp)


def test_an_axis_overrides_the_shared_value():
    exp = _with_axes({"net.N": [4, 6]})
    assert [setting_config(exp, s).net.N for s in expand(exp)] == [4, 6]
    assert preset("tiny").N == 8  # the shared value is the default of the axes that do not vary


def test_a_shared_B_reaches_every_setting():
    exp = _with_B(B=8)
    assert [setting_config(exp, s).net.B for s in expand(exp)] == [8] * len(expand(exp))
    default = preset("tiny")
    assert {setting_config(default, s).net.B for s in expand(default)} == {1}


def test_a_B_axis_overrides_the_shared_B():
    exp = with_axes(_with_B(B=8), {"net.B": [2, 4]})
    assert [setting_config(exp, s).net.B for s in expand(exp)] == [2, 4]


def test_the_task_axis_selects_that_tasks_parameters():
    exp = _with_axes({"data.task": ["object_action", "what_where"]})
    configs = [setting_config(exp, s) for s in expand(exp)]
    assert [c.data.task for c in configs] == ["object_action", "what_where"]
    assert configs[0].data.D_in == 2 * exp.object_action.grid_size**2
    assert [s.task for s in expand(exp)] == ["object_action", "what_where"]


def test_without_a_task_axis_the_base_task_is_used():
    exp = _with_axes({"net.L": [1, 2]}, task="what_where")
    assert {setting_config(exp, s).data.task for s in expand(exp)} == {"what_where"}


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


def test_labels_keep_the_underscores_of_axis_names(tmp_path):
    name = setting_dir(tmp_path, expand(preset("lambda_exp"))[1]).name
    assert name == "s0001_L1_phitanh_rate_lambda1e-06_weight_lambda1e-05"


def test_near_equal_floats_never_share_a_directory(tmp_path):
    # `:g` prints both as 1e-07; the index keeps the directories apart.
    exp = _with_axes({"train.rate_lambda": [1.00000001e-7, 1.00000002e-7]})
    dirs = {setting_dir(tmp_path, s) for s in expand(exp)}
    assert len(dirs) == 2


# ---------------------------------------------------------------------------- phase transition
def test_lambda_exp_preset_has_seven_lambdas_per_axis():
    exp = preset("lambda_exp")
    settings = expand(exp)
    assert len(settings) == 2 * 2 * 7 * 7
    configs = [setting_config(exp, s) for s in settings]
    assert {c.data.task for c in configs} == {"linear_teacher"}
    assert {c.net.L for c in configs} == {1, 2}
    assert {c.net.phi for c in configs} == {"tanh", "relu"}
    for name in ("weight_lambda", "rate_lambda"):
        got = sorted({getattr(c.train, name) for c in configs})
        assert got == pytest.approx([10.0**k for k in range(-6, 1)], rel=1e-12)
    pairs = {(c.train.weight_lambda, c.train.rate_lambda) for c in configs}
    assert len(pairs) == 49  # every weight/rate combination is a setting


def test_lambda_exp_lambdas_are_seven_log_spaced_values_from_1e_6_to_1():
    exp = preset("lambda_exp")
    for axis in ("train.weight_lambda", "train.rate_lambda"):
        assert exp.axes[axis] == np.logspace(-6, 0, 7).tolist()


# ---------------------------------------------------------------------------- validation
@pytest.mark.parametrize(
    ("axes", "axis"),
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
        ({"train.rate_lambda": [0, 0.0]}, "train.rate_lambda"),  # the same value, twice
    ],
)
def test_bad_axes_and_values_fail_at_load_naming_the_axis(axes, axis):
    with pytest.raises(ValidationError, match=axis.replace(".", r"\.")):
        _with_axes(axes)


@pytest.mark.parametrize(
    "axis",
    ["data.P", "data.D_in", "data.seed", "net.seed", "train.seed", "train.monitor_int",
     "train.monitor_bins", "train.monitor_shuffles", "train.log_int", "train.save_int"],
)  # fmt: skip
def test_axes_the_experiment_owns_are_rejected_with_a_pointer_to_its_field(axis):
    with pytest.raises(ValidationError, match=rf"{axis}.* set by the experiment's own"):
        _with_axes({axis: [4, 6]})


def test_other_fields_may_be_axes_and_win_over_the_shared_value():
    exp = _with_axes({"train.lr": [0.5, 0.25], "train.n_epochs": [3]})
    configs = [setting_config(exp, s) for s in expand(exp)]
    assert [c.train.lr for c in configs] == [0.5, 0.25]
    assert {c.train.n_epochs for c in configs} == {3}


def test_a_bad_combination_of_axes_fails_at_load():
    # Each value is fine alone; together there are more objects than cells of the task's grid.
    axes = {"data.task": ["object_action"], "data.max_objects": [2, 99]}
    with pytest.raises(ValidationError, match="max_objects"):
        _with_axes(axes)


def test_empty_axes_are_rejected():
    with pytest.raises(ValidationError, match="no axes"):
        _with_axes({})


def test_invalid_axes_in_exp_json_fail_at_load(tmp_path):
    tree = {**preset("tiny").model_dump(), "axes": {"net.typo": [1]}}
    (tmp_path / "exp.json").write_text(json.dumps(tree))
    with pytest.raises(ValidationError, match=r"net\.typo"):
        load_exp(tmp_path)


# ---------------------------------------------------------------------------- setting.json
def _run_all(exp, root):
    for i in range(len(expand(exp))):
        run_setting(exp, i, root, **QUIET)


def test_run_setting_records_the_axis_values_and_they_find_the_setting(small, tmp_path):
    _run_all(small, tmp_path)
    directory = find_setting_dir(tmp_path, small, {"net.L": 2, "net.phi": "tanh"})
    assert directory == setting_dir(tmp_path, expand(small)[2])
    record = read_setting(directory, small)
    assert record["index"] == 2 and record["axes"] == {"net.L": 2, "net.phi": "tanh"}
    # a subset of the axes is enough when it is unambiguous
    one = find_setting_dir(tmp_path, small, {"net.L": 1, "net.phi": "relu"})
    assert one.name.startswith("s0001_")


def test_find_setting_dir_is_loud_about_no_match_and_ambiguity(small, tmp_path):
    _run_all(small, tmp_path)
    with pytest.raises(LookupError, match="no setting"):
        find_setting_dir(tmp_path, small, {"net.L": 3})
    with pytest.raises(LookupError, match="4 settings"):
        find_setting_dir(tmp_path, small, {})
    with pytest.raises(LookupError, match="net.typo"):
        find_setting_dir(tmp_path, small, {"net.typo": 1})
    with pytest.raises(LookupError, match="no setting of this experiment"):
        find_setting_dir(tmp_path / "empty", small, {})


def test_find_setting_dir_matches_floats_by_value(tmp_path):
    exp = _with_axes({"train.rate_lambda": np.logspace(-6, 0, 3).tolist()}, n_epochs=2)
    run_setting(exp, 1, tmp_path, **QUIET)
    found = find_setting_dir(tmp_path, exp, {"train.rate_lambda": 1e-3})
    assert found.name.startswith("s0001_")


def test_find_setting_dir_ignores_directories_the_exp_does_not_own(small, tmp_path):
    _run_all(small, tmp_path)
    stray = tmp_path / "linear_teacher" / "s0009_L2_phitanh"
    stray.mkdir()
    (stray / "setting.json").write_text(
        json.dumps({"index": 9, "exp_hash": "x", "axes": {"net.L": 2, "net.phi": "tanh"}})
    )
    found = find_setting_dir(tmp_path, small, {"net.L": 2, "net.phi": "tanh"})
    assert found == setting_dir(tmp_path, expand(small)[2])


def test_values_match_by_tolerance_but_a_bool_is_not_an_int():
    assert same_value(1e-3, 1e-3 * (1 + 1e-12)) and not same_value(1e-3, 1.1e-3)
    assert same_value([1, [2.0, 3]], (1, (2.0, 3 + 1e-12)))
    assert not same_value(True, 1) and not same_value(1, True) and same_value(True, True)
    assert not same_value([True], [1]) and not same_value([1, 2], [1])


AXES_A = {"train.rate_lambda": [1e-4, 1e-2]}
AXES_B = {"train.rate_lambda": [1e-2, 1e-4]}


def test_reordered_axes_find_their_own_directories_not_the_old_ones(tmp_path):
    first = _with_axes(AXES_A, n_epochs=2)
    second = _with_axes(AXES_B, n_epochs=2)
    _run_all(first, tmp_path)
    _run_all(second, tmp_path)  # the labels differ, so these are new directories beside the old
    for exp in (first, second):
        found = find_setting_dir(tmp_path, exp, {"train.rate_lambda": 1e-4})
        assert found == setting_dir(
            tmp_path, expand(exp)[exp.axes["train.rate_lambda"].index(1e-4)]
        )
    old_dir = setting_dir(tmp_path, expand(first)[0])
    with pytest.raises(ExpError, match=str(old_dir)):
        read_setting(old_dir, second)


def test_axes_that_move_a_value_under_the_same_label_are_refused(tmp_path):
    # `:g` gives both 1e-07, so index 0 has the same directory name under either set of axes.
    first = _with_axes({"train.rate_lambda": [1.00000001e-7, 1.0]}, n_epochs=2)
    second = _with_axes({"train.rate_lambda": [1.00000002e-7, 1.0]}, n_epochs=2)
    run_setting(first, 0, tmp_path, **QUIET)
    with pytest.raises(ExpError, match="differs"):
        run_setting(second, 0, tmp_path, **QUIET)
    directory = setting_dir(tmp_path, expand(first)[0])
    with pytest.raises(ExpError, match="another experiment"):
        find_setting_dir(tmp_path, second, {})
    assert read_setting(directory, _with_axes(first.axes, n_epochs=3))["index"] == 0


def test_cli_init_refuses_a_root_that_already_has_settings(tmp_path, capsys):
    run_setting(_with_axes(AXES_A, n_epochs=2), 0, tmp_path, **QUIET)
    args = ["init", tmp_path, "--preset", "tiny"]
    code, _, err = _cli(capsys, *args)
    assert code == 1 and "setting directories" in err and "--force" in err
    assert not (tmp_path / "exp.json").exists()
    assert _cli(capsys, *args, "--force")[0] == 0


# ---------------------------------------------------------------------------- end to end
def _cli(capsys, *args) -> tuple[int, str, str]:
    code = main(["exp", *map(str, args)])
    out = capsys.readouterr()
    return code, out.out, out.err


TINY_SETS = [
    "N=8", "P=32", "n_epochs=4", "log_int=2", "save_int=2", "monitor_int=2",
    "monitor_bins=4", "monitor_shuffles=2", "linear_teacher.n_features=8", "device=cpu",
]  # fmt: skip


def _init_2x2(capsys, root):
    args = ["init", root, "--preset", "lambda_exp"]
    args += ["--axis", "net.L=[1]", "--axis", 'net.phi=["tanh"]']
    args += ["--axis", "train.rate_lambda=[1e-6,1.0]", "--axis", "train.weight_lambda=[1e-6,1.0]"]
    for item in TINY_SETS:
        args += ["--set", item]
    return _cli(capsys, *args)


def test_cli_2x2_axes_count_list_status_run_agree(tmp_path, capsys):
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

    exp = load_exp(tmp_path)
    cfg = load_config(cache_paths(setting_dir(tmp_path, expand(exp)[3])).config)
    assert (cfg.net.L, cfg.net.phi) == (1, "tanh")
    assert cfg.train.weight_lambda == pytest.approx(1.0)
    assert cfg.train.rate_lambda == pytest.approx(1.0)


def test_a_done_setting_is_skipped_in_an_experiment(small, tmp_path):
    run_setting(small, 0, tmp_path, **QUIET)
    final = cache_paths(setting_dir(tmp_path, expand(small)[0])).train / "final.pt"
    before = (final.read_bytes(), final.stat().st_mtime_ns)
    lines: list[str] = []
    run_setting(small, 0, tmp_path, log=lines.append)
    assert (final.read_bytes(), final.stat().st_mtime_ns) == before
    assert any("already done" in line for line in lines)


class Interrupt(Exception):
    pass


def test_an_interrupted_setting_resumes_from_its_checkpoint(small, tmp_path):
    def stop_at_epoch_4(line: str) -> None:
        if line.startswith("[00004/"):
            raise Interrupt

    with pytest.raises(Interrupt):
        run_setting(small, 1, tmp_path, log=stop_at_epoch_4)
    assert setting_status(tmp_path, small, expand(small)[1]) == "partial"
    run_setting(small, 1, tmp_path, **QUIET)
    assert setting_status(tmp_path, small, expand(small)[1]) == "done"


def test_cli_set_cannot_edit_the_axes(tmp_path, capsys):
    code, _, err = _cli(capsys, "init", tmp_path, "--preset", "lambda_exp",
                        "--set", "axes.net.L=[1]")  # fmt: skip
    assert code == 1 and "--axis" in err
    code, _, err = _cli(capsys, "init", tmp_path, "--preset", "tiny", "--set", "axes=null")
    assert code == 1 and "--axis" in err


def test_cli_init_axis_errors_are_one_line(tmp_path, capsys):
    code, _, err = _cli(capsys, "init", tmp_path, "--preset", "lambda_exp",
                        "--axis", "net.typo=[1]")  # fmt: skip
    assert code == 1 and err.startswith("error:") and "net.typo" in err
    code, _, err = _cli(capsys, "init", tmp_path, "--preset", "tiny", "--axis", "net.L=[1]")
    assert code == 1 and "modularity" in err
    assert not (tmp_path / "exp.json").exists()


# ---------------------------------------------------------------------------- exp hash and guards
def test_the_exp_hash_is_that_of_the_built_values_not_the_json_text():
    ints = _with_axes({"train.rate_lambda": [0, 1e-3]})
    floats = _with_axes({"train.rate_lambda": [0.0, 1e-3]})
    assert exp_hash(ints) == exp_hash(floats)
    assert exp_hash(ints) != exp_hash(_with_axes({"train.rate_lambda": [1e-3, 0.0]}))


def test_the_base_task_is_in_the_hash_only_without_a_task_axis():
    no_axis = {"net.L": [1, 2]}
    assert exp_hash(_with_axes(no_axis)) != exp_hash(_with_axes(no_axis, task="what_where"))
    task_axis = {"data.task": ["linear_teacher", "what_where"]}
    assert exp_hash(_with_axes(task_axis)) == exp_hash(_with_axes(task_axis, task="what_where"))


def test_a_re_init_that_only_respells_the_axes_reuses_the_directories(tmp_path):
    first = _with_axes({"train.rate_lambda": [0, 1e-3]}, n_epochs=2)
    run_setting(first, 0, tmp_path, **QUIET)
    again = _with_axes({"train.rate_lambda": [0.0, 1e-3]}, n_epochs=2)
    run_setting(again, 0, tmp_path, **QUIET)
    found = find_setting_dir(tmp_path, again, {"train.rate_lambda": 0.0})
    assert found.name.startswith("s0000_")


def test_a_setting_json_without_an_exp_hash_is_an_exp_error(small, tmp_path):
    run_setting(small, 0, tmp_path, **QUIET)
    directory = setting_dir(tmp_path, expand(small)[0])
    record = json.loads((directory / "setting.json").read_text())
    del record["exp_hash"]
    (directory / "setting.json").write_text(json.dumps(record))
    with pytest.raises(ExpError, match=str(directory)):
        read_setting(directory, small)


def test_nothing_run_says_how_old_directories_get_a_setting_json(small, tmp_path):
    with pytest.raises(LookupError, match="next `exp run`"):
        find_setting_dir(tmp_path, small, {})


def test_axes_too_large_for_the_directory_index_are_refused():
    huge = {"train.lr": [float(i + 1) for i in range(100)], "net.N": list(range(1, 102))}
    with pytest.raises(ExpError, match="at most 10000"):
        expand(ExpConfig.model_construct(**{**preset("tiny").model_dump(), "axes": huge}))


def test_same_value_treats_numpy_scalars_like_python_ones():
    assert same_value(np.float64(1e-3), 1e-3) and same_value(np.int64(1), 1)
    assert not same_value(np.True_, 1) and not same_value(1, np.True_)
    assert same_value(np.True_, True)


def test_the_validated_path_refuses_axes_too_large_for_the_directory_index():
    huge = {"train.lr": [float(i + 1) for i in range(100)], "net.N": list(range(1, 102))}
    with pytest.raises(ValidationError, match="at most 10000"):
        _with_axes(huge)


def test_two_spellings_of_one_value_name_the_same_directory(tmp_path):
    floats = expand(_with_axes({"train.rate_lambda": [1000000]}))
    other = expand(_with_axes({"train.rate_lambda": [1000000.0]}))
    assert floats[0].relpath == other[0].relpath
    assert floats[0].relpath.name == "s0000_rate_lambda1e+06"
