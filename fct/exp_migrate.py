"""Convert an experiment folder written by older code (`sweep.json`, a `grid` key, `grid_hash` in
every setting.json) to the current layout (`exp.json`, `axes`, `exp_hash`), once, in place.

Only those names change; the values, the hashes and every train / solve artefact stay as they
are. `plan_migration` reads and checks everything and writes nothing; `apply_plan` is the only
writer. The converted exp.json is validated as an `ExpConfig` before any file is written, and the
legacy file is removed last, so a refusal never leaves a half-converted folder.

This is the one module that knows the legacy names (`fct.exp.LEGACY_EXP_FILE` and the keys below).
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from fct.exp import (
    EXP_FILE,
    LEGACY_EXP_FILE,
    SETTING_DIRS_GLOB,
    SETTING_FILE,
    ExpConfig,
    ExpError,
    exp_hash,
)

LEGACY_AXES_KEY = "grid"
LEGACY_HASH_KEY = "grid_hash"


class MigrationError(ExpError):
    """The folder cannot be migrated as it stands; nothing was changed."""


@dataclass(frozen=True)
class MigrationPlan:
    """The full result of a migration: the new exp.json and every rewritten setting.json."""

    exp_text: str
    setting_texts: dict[Path, str]


# ---------------------------------------------------------------------------- pure conversion
def convert_exp_tree(tree: dict[str, Any]) -> dict[str, Any]:
    """The experiment dict with its `grid` key renamed `axes` (in place of the key, so the
    order is kept); an absent or null `grid` stays absent or null."""
    if "axes" in tree:
        raise MigrationError(
            f"{LEGACY_EXP_FILE} already has an `axes` key; it is not a legacy file"
        )
    return {("axes" if key == LEGACY_AXES_KEY else key): value for key, value in tree.items()}


def convert_setting_record(record: dict[str, Any], path: Path) -> dict[str, Any]:
    """The setting record with `grid_hash` renamed `exp_hash`; refuses a record that is not
    plainly a legacy one."""
    if "exp_hash" in record:
        raise MigrationError(
            f"{path} already has exp_hash: a half-migrated folder, check it by hand"
        )
    if LEGACY_HASH_KEY not in record:
        raise MigrationError(f"{path} has no {LEGACY_HASH_KEY}, so it cannot be migrated")
    return {("exp_hash" if key == LEGACY_HASH_KEY else key): v for key, v in record.items()}


def _dump(tree: dict[str, Any]) -> str:
    return json.dumps(tree, indent=2) + "\n"


# ---------------------------------------------------------------------------- read, check
def _read_json(path: Path, what: str) -> dict[str, Any]:
    try:
        tree = json.loads(path.read_text())
    except json.JSONDecodeError as err:
        raise MigrationError(f"{path} is not valid JSON ({what}): {err}") from err
    if not isinstance(tree, dict):
        raise MigrationError(f"{path} must hold a JSON object ({what})")
    return tree


def _check_refusals(root: Path) -> None:
    if (root / EXP_FILE).exists():
        raise MigrationError(
            f"{root} is already migrated ({EXP_FILE} exists); nothing was changed"
            + (
                f", and {LEGACY_EXP_FILE} is still there: check it by hand"
                if _has_legacy(root)
                else ""
            )
        )
    if not _has_legacy(root):
        raise MigrationError(f"no {LEGACY_EXP_FILE} in {root}: nothing to migrate")


def _has_legacy(root: Path) -> bool:
    return (root / LEGACY_EXP_FILE).exists()


def _validated(tree: dict[str, Any], root: Path) -> ExpConfig:
    try:
        return ExpConfig.model_validate(tree)
    except ValidationError as err:
        raise MigrationError(
            f"{root / LEGACY_EXP_FILE} does not validate as an experiment once converted: {err}"
        ) from err


def _setting_texts(root: Path, exp: ExpConfig) -> dict[Path, str]:
    """The new text of every setting.json, after checking that they all carry the hash this
    experiment gives (a different one means the settings belong to another experiment)."""
    expected = exp_hash(exp)
    texts: dict[Path, str] = {}
    for path in sorted(root.glob(f"{SETTING_DIRS_GLOB}/{SETTING_FILE}")):
        record = _read_json(path, SETTING_FILE)
        converted = convert_setting_record(record, path)
        if converted["exp_hash"] != expected:
            raise MigrationError(
                f"{path} carries hash {converted['exp_hash']}, but {LEGACY_EXP_FILE} gives "
                f"{expected}: the hashes differ, so the settings were written under another "
                "experiment"
            )
        texts[path] = _dump(converted)
    return texts


def plan_migration(root: Path) -> MigrationPlan:
    """Everything `apply_plan` will write; MigrationError, with nothing written, if the folder
    is not a consistent legacy experiment."""
    root = Path(root)
    _check_refusals(root)
    tree = convert_exp_tree(_read_json(root / LEGACY_EXP_FILE, LEGACY_EXP_FILE))
    exp = _validated(tree, root)
    return MigrationPlan(_dump(tree), _setting_texts(root, exp))


# ---------------------------------------------------------------------------- write
def _write_atomically(path: Path, text: str) -> None:
    """Write beside the target and rename, so a crash leaves the old or the new file whole."""
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text)
    os.replace(partial, path)


def apply_plan(root: Path, plan: MigrationPlan) -> list[str]:
    """Write exp.json, then the setting files, then remove the legacy file. Returns the report."""
    root = Path(root)
    _write_atomically(root / EXP_FILE, plan.exp_text)
    for path, text in plan.setting_texts.items():
        _write_atomically(path, text)
    (root / LEGACY_EXP_FILE).unlink()
    return [
        f"wrote {root / EXP_FILE} ({LEGACY_AXES_KEY} -> axes)",
        f"rewrote {len(plan.setting_texts)} {SETTING_FILE} file(s) ({LEGACY_HASH_KEY} -> exp_hash)",
        f"removed {root / LEGACY_EXP_FILE}",
    ]


def migrate_exp(root: Path) -> list[str]:
    """Migrate the folder at `root`; the lines to print. MigrationError, with nothing changed,
    if it is already migrated, has no legacy file, or its files disagree."""
    return apply_plan(root, plan_migration(root))
