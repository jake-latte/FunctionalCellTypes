"""`fct exp migrate ROOT`: a folder that holds a legacy `sweep.json` (a `grid` key, `grid_hash`
in every setting.json) becomes the layout the current code writes, safely: nothing is changed
unless the whole conversion is valid, artefacts are never touched, an interrupted run is
finished by running it again, and a finished folder refuses.

Most legacy folders here are built by running the current code and writing the old names back;
`test_a_folder_written_as_literal_legacy_text_migrates` builds one from literal text instead,
so the conversion is not only checked against its own inverse.
"""

import json
import os
import shutil
from pathlib import Path

import pytest

from fct.cli import main
from fct.exp import (
    ExpConfig,
    UnmigratedExpError,
    exp_hash,
    expand,
    load_exp,
    preset,
    run_setting,
    save_exp,
    setting_statuses,
    with_axes,
)
from fct.exp_migrate import MigrationError, migrate_exp
from tests.report_helpers import QUIET

FIXTURES = Path(__file__).parent / "fixtures"
AXES = {"net.L": [1, 2]}


def _tiny_exp() -> ExpConfig:
    return with_axes(preset("tiny"), AXES)


def _snapshot(root: Path) -> dict[str, bytes]:
    """Every file under `root` by relative path, so "nothing changed" is one equality."""
    files = sorted(p for p in root.rglob("*") if p.is_file())
    return {str(p.relative_to(root)): p.read_bytes() for p in files}


def _setting_files(root: Path) -> list[Path]:
    return sorted(root.glob("*/*/setting.json"))


def _make_legacy(root: Path) -> None:
    """Rewrite a folder written by the current code into the old layout."""
    tree = json.loads((root / "exp.json").read_text())
    legacy = {("grid" if key == "axes" else key): value for key, value in tree.items()}
    (root / "sweep.json").write_text(json.dumps(legacy, indent=2) + "\n")
    (root / "exp.json").unlink()
    for path in _setting_files(root):
        record = json.loads(path.read_text())
        record = {("grid_hash" if key == "exp_hash" else key): v for key, v in record.items()}
        path.write_text(json.dumps(record, indent=2) + "\n")


def _edit_first_setting(root: Path, edit) -> None:
    path = _setting_files(root)[0]
    path.write_text(json.dumps(edit(json.loads(path.read_text())), indent=2) + "\n")


@pytest.fixture(scope="module")
def fresh(tmp_path_factory) -> Path:
    """A folder written by the current code: two settings, both trained."""
    root = tmp_path_factory.mktemp("fresh")
    exp = _tiny_exp()
    save_exp(exp, root)
    for index in (0, 1):
        run_setting(exp, index, root, **QUIET)
    return root


@pytest.fixture
def legacy(fresh, tmp_path) -> Path:
    root = tmp_path / "old"
    shutil.copytree(fresh, root)
    _make_legacy(root)
    return root


def _works_afterwards(root: Path, fresh: Path, capsys) -> None:
    """The migrated folder is the one the current code writes, and its commands run."""
    assert not (root / "sweep.json").exists()
    assert load_exp(root) == load_exp(fresh)
    assert [s for *_, s in setting_statuses(root, load_exp(root))] == ["done", "done"]
    capsys.readouterr()
    assert main(["exp", "status", str(root)]) == 0
    assert "2 done" in capsys.readouterr().out
    assert main(["exp", "run", str(root), "--index", "0"]) == 0  # a done setting is skipped
    for path in _setting_files(root):
        record = json.loads(path.read_text())
        assert "grid_hash" not in record and record["exp_hash"] == exp_hash(load_exp(root))


# ---------------------------------------------------------------------------- the conversion
def test_a_migrated_folder_equals_one_written_fresh(legacy, fresh, capsys):
    lines = migrate_exp(legacy)
    _works_afterwards(legacy, fresh, capsys)
    migrated, written = _snapshot(legacy), _snapshot(fresh)
    assert migrated.keys() == written.keys()
    assert all(migrated[n] == c for n, c in written.items() if n.endswith("setting.json"))
    assert any("exp.json" in line for line in lines)
    assert any("setting.json" in line for line in lines)


def test_artefacts_and_everything_but_the_keys_are_untouched(legacy):
    before = _snapshot(legacy)
    migrate_exp(legacy)
    after = _snapshot(legacy)
    changed = {name for name in before.keys() & after.keys() if before[name] != after[name]}
    assert changed and all(name.endswith("/setting.json") for name in changed)
    assert set(before) - set(after) == {"sweep.json"}
    assert set(after) - set(before) == {"exp.json"}
    train_files = [n for n in before if "/train/" in n]
    assert train_files and all(before[n] == after[n] for n in train_files)


def test_a_folder_written_as_literal_legacy_text_migrates(tmp_path):
    base = json.loads((FIXTURES / "legacy_sweep_tiny.json").read_text())
    (tmp_path / "sweep.json").write_text(
        json.dumps({**base, "grid": {"net.L": [1, 2]}}, indent=2) + "\n"
    )
    exp = ExpConfig.model_validate({**base, "axes": {"net.L": [1, 2]}})
    expected = exp_hash(exp)
    for setting in expand(exp):
        directory = tmp_path / setting.relpath
        directory.mkdir(parents=True)
        (directory / "setting.json").write_text(
            f'{{\n  "index": {setting.index},\n  "grid_hash": "{expected}",\n'
            f'  "axes": {{\n    "net.L": {setting.axes["net.L"]}\n  }}\n}}\n'
        )
    migrate_exp(tmp_path)
    assert load_exp(tmp_path) == exp
    first = json.loads((tmp_path / expand(exp)[0].relpath / "setting.json").read_text())
    assert first == {"index": 0, "exp_hash": expected, "axes": {"net.L": 1}}


def test_the_axes_key_replaces_the_grid_key(legacy):
    migrate_exp(legacy)
    tree = json.loads((legacy / "exp.json").read_text())
    assert "grid" not in tree and tree["axes"] == AXES


def test_an_explicit_null_grid_migrates(tmp_path):
    tree = json.loads((FIXTURES / "modularity_exp.json").read_text())
    (tmp_path / "sweep.json").write_text(json.dumps({**tree, "grid": None}))
    migrate_exp(tmp_path)
    assert json.loads((tmp_path / "exp.json").read_text())["axes"] is None
    assert len(expand(load_exp(tmp_path))) == 36


def test_a_legacy_file_without_the_grid_key_migrates(tmp_path):
    (tmp_path / "sweep.json").write_text((FIXTURES / "modularity_exp.json").read_text())
    migrate_exp(tmp_path)
    expected = ExpConfig.model_validate_json((FIXTURES / "modularity_exp.json").read_text())
    assert load_exp(tmp_path) == expected


def test_a_real_legacy_file_migrates_and_validates(tmp_path):
    text = (FIXTURES / "legacy_sweep_tiny.json").read_text()
    (tmp_path / "sweep.json").write_text(text)
    migrate_exp(tmp_path)
    assert load_exp(tmp_path) == ExpConfig.model_validate_json(text)
    assert not (tmp_path / "sweep.json").exists()


def test_a_never_run_folder_with_no_setting_files_migrates(tmp_path):
    save_exp(_tiny_exp(), tmp_path)
    _make_legacy(tmp_path)
    assert _setting_files(tmp_path) == []
    lines = migrate_exp(tmp_path)
    assert any("0 setting.json" in line for line in lines)
    assert [s for *_, s in setting_statuses(tmp_path, load_exp(tmp_path))] == ["missing"] * 2


def test_both_a_grid_and_an_axes_key_is_a_value_error(legacy):
    tree = json.loads((legacy / "sweep.json").read_text())
    (legacy / "sweep.json").write_text(json.dumps({**tree, "axes": AXES}))
    before = _snapshot(legacy)
    with pytest.raises(ValueError, match="axes"):
        migrate_exp(legacy)
    assert _snapshot(legacy) == before


def test_only_the_experiments_own_setting_directories_are_touched(legacy, fresh):
    nested = legacy / "nested"
    shutil.copytree(fresh, nested)
    _make_legacy(nested)
    stray = legacy / "linear_teacher" / "stray"
    stray.mkdir()
    (stray / "setting.json").write_text('{"grid_hash": "not ours"}')

    def nested_files() -> dict[str, bytes]:
        return {k: v for k, v in _snapshot(legacy).items() if k.startswith("nested")}

    before = nested_files()
    lines = migrate_exp(legacy)
    assert nested_files() == before
    assert (nested / "sweep.json").exists() and not (nested / "exp.json").exists()
    # a record at setting depth is renamed whether or not the experiment owns the directory
    assert json.loads((stray / "setting.json").read_text()) == {"exp_hash": "not ours"}
    assert any("kept from an earlier experiment" in line and "stray" in line for line in lines)


# ---------------------------------------------------------------------------- refusals
def test_a_second_run_refuses_and_changes_nothing(legacy):
    migrate_exp(legacy)
    before = _snapshot(legacy)
    with pytest.raises(MigrationError, match="already migrated"):
        migrate_exp(legacy)
    assert _snapshot(legacy) == before


def test_it_refuses_a_folder_with_no_legacy_file(tmp_path):
    with pytest.raises(MigrationError, match="nothing to migrate"):
        migrate_exp(tmp_path)
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("content", ["{broken", None], ids=["unreadable", "other-experiment"])
def test_it_refuses_an_exp_json_that_is_not_the_conversion_naming_both_files(legacy, content):
    if content is None:
        save_exp(preset("tiny"), legacy)
    else:
        (legacy / "exp.json").write_text(content)
    before = _snapshot(legacy)
    with pytest.raises(MigrationError, match=r"both sweep\.json and exp\.json"):
        migrate_exp(legacy)
    assert _snapshot(legacy) == before


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda r: {k: v for k, v in r.items() if k != "grid_hash"}, "neither"),
        (lambda r: {**r, "exp_hash": r["grid_hash"]}, "both grid_hash and exp_hash"),
        (lambda r: {**r, "index": 99}, "index"),
        (lambda r: {**r, "index": True}, "index"),
        (lambda r: {**r, "index": 1 - r["index"]}, "index"),
        (lambda r: {**r, "axes": {"net.L": 7}}, "axis values"),
    ],
    ids=["no-hash", "both-keys", "index-out-of-range", "index-bool",
         "index-of-another-setting", "other-axis-values"],
)  # fmt: skip
def test_it_refuses_inconsistent_setting_files_and_changes_nothing(legacy, edit, match):
    _edit_first_setting(legacy, edit)
    before = _snapshot(legacy)
    with pytest.raises(MigrationError, match=match):
        migrate_exp(legacy)
    assert _snapshot(legacy) == before


@pytest.mark.parametrize("content", [b"{not json", b"[1, 2]", b"\xff\xfe\x00"])
def test_a_corrupt_setting_file_is_a_migration_error_naming_its_path(legacy, content):
    path = _setting_files(legacy)[0]
    path.write_bytes(content)
    before = _snapshot(legacy)
    with pytest.raises(MigrationError, match=str(path).replace(".", r"\.")):
        migrate_exp(legacy)
    assert _snapshot(legacy) == before


def test_it_refuses_a_legacy_file_that_does_not_validate(legacy):
    tree = json.loads((legacy / "sweep.json").read_text())
    tree["grid"]["net.L"] = []
    (legacy / "sweep.json").write_text(json.dumps(tree))
    before = _snapshot(legacy)
    with pytest.raises(MigrationError, match="does not validate"):
        migrate_exp(legacy)
    assert _snapshot(legacy) == before


def test_it_refuses_a_legacy_file_that_is_not_json(tmp_path):
    (tmp_path / "sweep.json").write_text("{nope")
    with pytest.raises(MigrationError, match="not valid JSON"):
        migrate_exp(tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["sweep.json"]


def test_a_setting_file_with_another_hash_and_a_wrong_index_is_kept_not_judged(legacy):
    """Only a record with the experiment's own hash is checked for index and axes."""
    _edit_first_setting(legacy, lambda r: {**r, "grid_hash": "0" * 12, "index": 99})
    lines = migrate_exp(legacy)
    first = json.loads(_setting_files(legacy)[0].read_text())
    assert first["exp_hash"] == "0" * 12 and first["index"] == 99
    assert any("kept from an earlier experiment (hash " + "0" * 12 in line for line in lines)


# ---------------------------------------------------------------------------- crashes
class _Crash(OSError):
    pass


@pytest.mark.parametrize("crash_at", [1, 2, 3], ids=["first-setting", "second-setting", "exp-json"])
def test_a_crash_while_writing_is_finished_by_running_again(
    legacy, fresh, monkeypatch, capsys, crash_at
):
    """`os.replace` commits each file; the nth call fails and leaves its temp file behind."""
    calls = []
    real_replace = os.replace

    def failing_replace(src, dst):
        calls.append(dst)
        if len(calls) == crash_at:
            raise _Crash("simulated crash")
        real_replace(src, dst)

    with monkeypatch.context() as patched:
        patched.setattr("fct.exp_migrate.os.replace", failing_replace)
        with pytest.raises(_Crash):
            migrate_exp(legacy)
    assert (legacy / "sweep.json").exists() and not (legacy / "exp.json").exists()
    lines = migrate_exp(legacy)
    assert crash_at == 1 or any("resuming" in line for line in lines)
    _works_afterwards(legacy, fresh, capsys)
    assert not list(legacy.rglob("*.tmp"))


def test_a_crash_before_the_legacy_file_is_removed_is_finished_by_running_again(
    legacy, fresh, monkeypatch, capsys
):
    real_unlink = Path.unlink

    def failing_unlink(self, *args, **kwargs):
        if self.name == "sweep.json":
            raise _Crash("simulated crash")
        real_unlink(self, *args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(Path, "unlink", failing_unlink)
        with pytest.raises(_Crash):
            migrate_exp(legacy)
    assert (legacy / "sweep.json").exists() and (legacy / "exp.json").exists()
    assert any("resuming" in line for line in migrate_exp(legacy))
    _works_afterwards(legacy, fresh, capsys)


# ---------------------------------------------------------------------------- unmigrated folders
def test_load_exp_on_an_unmigrated_folder_names_the_migration(legacy):
    with pytest.raises(UnmigratedExpError, match="fct exp migrate"):
        load_exp(legacy)


@pytest.mark.parametrize(
    "argv",
    [["count"], ["list"], ["status"], ["status", "--stage", "solve"], ["run", "--index", "0"],
     ["run"], ["init"]],
    ids=lambda a: "-".join(a),
)  # fmt: skip
def test_every_exp_command_on_an_unmigrated_folder_fails_naming_the_migration(legacy, argv, capsys):
    before = _snapshot(legacy)
    assert main(["exp", argv[0], str(legacy), *argv[1:]]) == 1
    assert "fct exp migrate" in capsys.readouterr().err
    assert _snapshot(legacy) == before


def test_the_cli_migrates_and_the_commands_then_work(legacy, capsys):
    assert main(["exp", "migrate", str(legacy)]) == 0
    out = capsys.readouterr().out
    assert "exp.json" in out and "sweep.json" in out
    assert main(["exp", "count", str(legacy)]) == 0
    assert capsys.readouterr().out.strip() == "2"
    assert main(["exp", "migrate", str(legacy)]) == 1
    assert "already migrated" in capsys.readouterr().err


def test_the_cli_reports_a_failed_write_as_one_error_line(legacy, monkeypatch, capsys):
    def disk_full(root):
        raise OSError("No space left on device")

    monkeypatch.setattr("fct.cli.migrate_exp", disk_full)
    assert main(["exp", "migrate", str(legacy)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "No space left" in err and "Traceback" not in err
    assert f"fct exp migrate {legacy}` again to finish" in err


# ---------------------------------------------------------------------------- init --force leftovers
def _forced_reinit(root: Path, first_axes: dict, second_axes: dict, run_index: int) -> None:
    """The layout the old `init --force` left: a run of one experiment, then another experiment
    written over exp.json, then a run of that one (settings of the first are never deleted)."""
    first = with_axes(preset("tiny"), first_axes)
    second = with_axes(preset("tiny"), second_axes)
    save_exp(first, root)
    run_setting(first, 0, root, **QUIET)
    save_exp(second, root)
    run_setting(second, run_index, root, **QUIET)
    _make_legacy(root)


def test_a_member_directory_kept_from_an_earlier_experiment_migrates_with_a_note(tmp_path, capsys):
    """B: net.L [1, 2] run at index 0, then --force with [1, 2, 3] run at index 2. s0000_L1 is a
    member directory of the new experiment but carries the old hash."""
    _forced_reinit(tmp_path, {"net.L": [1, 2]}, {"net.L": [1, 2, 3]}, 2)
    old_hash = json.loads(_setting_files(tmp_path)[0].read_text())["grid_hash"]
    lines = migrate_exp(tmp_path)
    assert sum("kept from an earlier experiment" in line for line in lines) == 1
    assert any(f"(hash {old_hash}); `exp run` refuses it, as before" in line for line in lines)
    assert all("exp_hash" in json.loads(p.read_text()) for p in _setting_files(tmp_path))
    assert not (tmp_path / "sweep.json").exists()
    kept = tmp_path / expand(load_exp(tmp_path))[0].relpath / "setting.json"
    assert json.loads(kept.read_text())["exp_hash"] == old_hash
    assert main(["exp", "status", str(tmp_path)]) == 0
    assert main(["exp", "run", str(tmp_path), "--index", "0"]) == 1  # refused, as before
    capsys.readouterr()


def test_a_non_member_directory_kept_from_an_earlier_experiment_migrates_with_a_note(tmp_path):
    """A: net.L [1, 2] run at index 0, then --force with [3] run at index 0 (another name)."""
    _forced_reinit(tmp_path, {"net.L": [1, 2]}, {"net.L": [3]}, 0)
    lines = migrate_exp(tmp_path)
    assert sum("kept from an earlier experiment" in line for line in lines) == 1
    assert all("exp_hash" in json.loads(p.read_text()) for p in _setting_files(tmp_path))
    assert not any("grid_hash" in p.read_text() for p in _setting_files(tmp_path))


# ---------------------------------------------------------------------------- a stale legacy file
def test_a_legacy_file_that_no_longer_validates_says_the_old_code_could_not_load_it(tmp_path):
    shutil.copy(FIXTURES / "legacy_sweep_readout.json", tmp_path / "sweep.json")
    before = _snapshot(tmp_path)
    with pytest.raises(MigrationError) as info:
        migrate_exp(tmp_path)
    message = str(info.value)
    assert "meanfield.readout" in message
    assert "old code could not load it either" in message
    assert "validate under the current config" in message and "stale" in message
    assert _snapshot(tmp_path) == before
