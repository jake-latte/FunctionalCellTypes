"""`fct exp migrate ROOT`: a folder written before the sweep -> experiment rename (`sweep.json`
with a `grid` key, `grid_hash` in every setting.json) becomes the layout the current code writes,
safely: nothing is changed unless the whole conversion is valid, artefacts are never touched,
and a second run refuses.

The legacy folders here are built by running the current code and then writing the old names
back, which is the only difference between the two layouts.
"""

import json
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
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def _make_legacy(root: Path) -> None:
    """Rewrite a folder written by the current code into the old layout."""
    tree = json.loads((root / "exp.json").read_text())
    legacy = {("grid" if key == "axes" else key): value for key, value in tree.items()}
    (root / "sweep.json").write_text(json.dumps(legacy, indent=2) + "\n")
    (root / "exp.json").unlink()
    for path in root.glob("*/*/setting.json"):
        record = json.loads(path.read_text())
        record = {("grid_hash" if key == "exp_hash" else key): v for key, v in record.items()}
        path.write_text(json.dumps(record, indent=2) + "\n")


@pytest.fixture(scope="module")
def fresh(tmp_path_factory) -> Path:
    """A folder written by the current code: two settings, the first trained."""
    root = tmp_path_factory.mktemp("fresh")
    exp = _tiny_exp()
    save_exp(exp, root)
    run_setting(exp, 0, root, **QUIET)
    return root


@pytest.fixture
def legacy(fresh, tmp_path) -> Path:
    root = tmp_path / "old"
    shutil.copytree(fresh, root)
    _make_legacy(root)
    return root


# ---------------------------------------------------------------------------- the conversion
def test_a_migrated_folder_equals_one_written_fresh(legacy, fresh):
    lines = migrate_exp(legacy)
    assert load_exp(legacy) == load_exp(fresh)
    exp = load_exp(legacy)
    assert setting_statuses(legacy, exp) == setting_statuses(fresh, load_exp(fresh))
    assert [s for *_, s in setting_statuses(legacy, exp)] == ["done", "missing"]
    assert _snapshot(legacy).keys() == _snapshot(fresh).keys()
    record = json.loads((legacy / expand(exp)[0].relpath / "setting.json").read_text())
    assert record["exp_hash"] == exp_hash(exp) and "grid_hash" not in record
    assert any("exp.json" in line for line in lines)
    assert any("setting.json" in line for line in lines)


def test_artefacts_and_setting_files_other_than_the_keys_are_untouched(legacy):
    before = _snapshot(legacy)
    migrate_exp(legacy)
    after = _snapshot(legacy)
    changed = {name for name in before.keys() & after.keys() if before[name] != after[name]}
    assert changed and all(name.endswith("/setting.json") for name in changed)
    assert set(before) - set(after) == {"sweep.json"}
    assert set(after) - set(before) == {"exp.json"}
    train_files = [n for n in before if "/train/" in n]
    assert train_files and all(before[n] == after[n] for n in train_files)


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
    assert load_exp(tmp_path) == ExpConfig.model_validate_json(
        (FIXTURES / "modularity_exp.json").read_text()
    )


def test_a_real_legacy_file_migrates_and_validates(tmp_path):
    text = (FIXTURES / "legacy_sweep_tiny.json").read_text()
    (tmp_path / "sweep.json").write_text(text)
    migrate_exp(tmp_path)
    assert load_exp(tmp_path) == ExpConfig.model_validate_json(text)
    assert not (tmp_path / "sweep.json").exists()


# ---------------------------------------------------------------------------- refusals
def test_a_second_run_refuses_and_changes_nothing(legacy):
    migrate_exp(legacy)
    before = _snapshot(legacy)
    with pytest.raises(MigrationError, match="already migrated"):
        migrate_exp(legacy)
    assert _snapshot(legacy) == before


def test_it_refuses_when_exp_json_exists_beside_the_legacy_file(legacy, fresh):
    shutil.copy(fresh / "exp.json", legacy / "exp.json")
    before = _snapshot(legacy)
    with pytest.raises(MigrationError, match="already migrated"):
        migrate_exp(legacy)
    assert _snapshot(legacy) == before


def test_it_refuses_a_folder_with_no_legacy_file(tmp_path):
    with pytest.raises(MigrationError, match="nothing to migrate"):
        migrate_exp(tmp_path)
    assert not any(tmp_path.iterdir())


def _setting_files(root: Path) -> list[Path]:
    return sorted(root.glob("*/*/setting.json"))


def _edit_first_setting(root: Path, edit) -> None:
    path = _setting_files(root)[0]
    path.write_text(json.dumps(edit(json.loads(path.read_text())), indent=2) + "\n")


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda r: {k: v for k, v in r.items() if k != "grid_hash"}, "no grid_hash"),
        (lambda r: {**r, "exp_hash": r["grid_hash"]}, "already has exp_hash"),
        (lambda r: {**r, "grid_hash": "0" * 12}, "differ"),
    ],
    ids=["missing-hash", "half-migrated", "other-experiment"],
)
def test_it_refuses_inconsistent_setting_files_and_changes_nothing(legacy, edit, match):
    _edit_first_setting(legacy, edit)
    before = _snapshot(legacy)
    with pytest.raises(MigrationError, match=match):
        migrate_exp(legacy)
    assert _snapshot(legacy) == before


def test_it_refuses_an_unreadable_setting_file(legacy):
    _setting_files(legacy)[0].write_text("{not json")
    before = _snapshot(legacy)
    with pytest.raises(MigrationError, match="setting.json"):
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


# ---------------------------------------------------------------------------- unmigrated folders
def test_load_exp_on_an_unmigrated_folder_names_the_migration(legacy):
    with pytest.raises(UnmigratedExpError, match="fct exp migrate"):
        load_exp(legacy)


@pytest.mark.parametrize(
    "argv",
    [
        ["count"],
        ["list"],
        ["status"],
        ["status", "--stage", "solve"],
        ["run", "--index", "0"],
        ["run"],
        ["init"],
    ],
    ids=lambda a: "-".join(a),
)
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
