"""Convert an experiment folder that holds a legacy `sweep.json` (a `grid` key, `grid_hash` in
every setting.json) to the current layout (`exp.json`, `axes`, `exp_hash`), once, in place.

Only those names change; the values, the hashes and every train / solve artefact stay as they
are. The rename is cosmetic, so it keeps the old (in)consistency instead of judging it: a
setting.json that an earlier `init --force` experiment left behind (another hash, or a directory
the current experiment does not own) is renamed like the rest and reported, not refused. `plan_migration` reads and checks everything and writes nothing; `apply_plan` is the only
writer. The converted exp.json is validated as an `ExpConfig` before any file is written, and the
legacy file is removed last. A run that was interrupted after writing exp.json (both files
then exist) is finished by running the command again.

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
    Setting,
    exp_hash,
    expand,
)

LEGACY_AXES_KEY = "grid"
LEGACY_HASH_KEY = "grid_hash"


class MigrationError(ExpError):
    """The folder cannot be migrated as it stands; nothing was changed."""


@dataclass(frozen=True)
class MigrationPlan:
    """What `apply_plan` writes: the new exp.json (None when a half-finished earlier run already
    wrote the right one) and the setting.json files that still need converting; `notes` report
    the records kept from an earlier experiment."""

    exp_text: str | None
    setting_texts: dict[Path, str]
    resumed: bool
    notes: tuple[str, ...] = ()


# ---------------------------------------------------------------------------- pure conversion
def convert_exp_tree(tree: dict[str, Any]) -> dict[str, Any]:
    """The experiment dict with its `grid` key renamed `axes` (in place of the key, so the
    order is kept); an absent or null `grid` stays absent or null. MigrationError (an ExpError, so a
    ValueError) if the file already has `axes`, which would make the renamed key ambiguous."""
    if "axes" in tree:
        raise MigrationError(
            f"the legacy {LEGACY_EXP_FILE} has an `axes` key"
            + (f" as well as `{LEGACY_AXES_KEY}`" if LEGACY_AXES_KEY in tree else "")
            + ": it is not a plain legacy file"
        )
    return {("axes" if key == LEGACY_AXES_KEY else key): value for key, value in tree.items()}


def _dump(tree: dict[str, Any]) -> str:
    return json.dumps(tree, indent=2) + "\n"


# ---------------------------------------------------------------------------- read, check
def _read_json(path: Path, what: str) -> dict[str, Any]:
    try:
        tree = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise MigrationError(f"{path} is not valid JSON ({what}): {err}") from err
    if not isinstance(tree, dict):
        raise MigrationError(f"{path} must hold a JSON object ({what})")
    return tree


def _exp_json_is_the_conversion(root: Path, converted: dict[str, Any]) -> bool:
    try:
        return json.loads((root / EXP_FILE).read_text()) == converted
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False


def _validated(tree: dict[str, Any], root: Path) -> ExpConfig:
    try:
        return ExpConfig.model_validate(tree)
    except ValidationError as err:
        fields = sorted({".".join(str(part) for part in e["loc"]) for e in err.errors()})
        raise MigrationError(
            f"{root / LEGACY_EXP_FILE} does not validate as an experiment once converted "
            f"(offending: {', '.join(fields)}): {err}\n"
            "The old code could not load it either: a legacy file must first validate "
            "under the current config (fix the offending field), and the folder's "
            "results are stale under the current code."
        ) from err


def _hash_of(path: Path, record: dict[str, Any]) -> tuple[str, bool]:
    """The record's hash and whether it already carries the new key; MigrationError unless it
    has exactly one of the two keys."""
    has_new, has_old = "exp_hash" in record, LEGACY_HASH_KEY in record
    if has_new and has_old:
        raise MigrationError(f"{path} has both {LEGACY_HASH_KEY} and exp_hash: check it by hand")
    if not (has_new or has_old):
        raise MigrationError(f"{path} has neither {LEGACY_HASH_KEY} nor exp_hash")
    return (record["exp_hash"] if has_new else record[LEGACY_HASH_KEY]), has_new


def _check_member(path: Path, record: dict[str, Any], setting: Setting) -> None:
    """A record with the experiment's own hash must be the one `run_setting` writes for this
    setting (index, axes), so that no later command rejects the migrated folder."""
    if record.get("index") != setting.index or isinstance(record.get("index"), bool):
        raise MigrationError(f"{path} has index {record.get('index')!r}, expected {setting.index}")
    if record.get("axes") != json.loads(json.dumps(setting.axes)):
        raise MigrationError(f"{path} records axis values that differ from the experiment's")


def _setting_texts(root: Path, exp: ExpConfig) -> tuple[dict[Path, str], int, list[str]]:
    """The new text of every setting.json still carrying `grid_hash`, how many were converted
    already, and a note per record kept from an earlier experiment. Every setting.json at the
    setting depth `SETTING_DIRS_GLOB` is converted (nothing deeper, so a nested copy of an
    experiment is left alone); only a record with the experiment's own hash is checked against
    its setting, because the rename keeps an older experiment's records as they are."""
    expected = exp_hash(exp)
    members = {s.relpath: s for s in expand(exp)}
    texts: dict[Path, str] = {}
    notes: list[str] = []
    done = 0
    for path in sorted(root.glob(SETTING_DIRS_GLOB + "/" + SETTING_FILE)):
        record = _read_json(path, SETTING_FILE)
        found, converted = _hash_of(path, record)
        relpath = path.parent.relative_to(root)
        setting = members.get(relpath)
        if setting is not None and found == expected:
            _check_member(path, record, setting)
        elif setting is not None:
            notes.append(
                f"{relpath}: kept from an earlier experiment (hash {found}); "
                "`exp run` refuses it, as before"
            )
        else:
            notes.append(f"{relpath}: kept from an earlier experiment (hash {found})")
        if converted:
            done += 1
        else:
            renamed = {("exp_hash" if k == LEGACY_HASH_KEY else k): v for k, v in record.items()}
            texts[path] = _dump(renamed)
    return texts, done, notes


def plan_migration(root: Path) -> MigrationPlan:
    """Everything `apply_plan` will write; MigrationError, with nothing written, if the folder
    is not a consistent legacy experiment. A legacy file with no exp.json, or with an exp.json
    that is exactly its conversion, is a migration to finish (some setting files may be done)."""
    root = Path(root)
    has_exp, has_legacy = (root / EXP_FILE).exists(), (root / LEGACY_EXP_FILE).exists()
    if has_exp and not has_legacy:
        raise MigrationError(f"{root} is already migrated ({EXP_FILE} exists); nothing changed")
    if not has_legacy:
        raise MigrationError(f"no {LEGACY_EXP_FILE} in {root}: nothing to migrate")
    tree = convert_exp_tree(_read_json(root / LEGACY_EXP_FILE, LEGACY_EXP_FILE))
    if has_exp and not _exp_json_is_the_conversion(root, tree):
        raise MigrationError(
            f"{root} holds both {LEGACY_EXP_FILE} and {EXP_FILE}, and {EXP_FILE} is not the "
            f"conversion of {LEGACY_EXP_FILE} (already migrated?); nothing changed, check both"
        )
    texts, done, notes = _setting_texts(root, _validated(tree, root))
    plan_exp = None if has_exp else _dump(tree)
    return MigrationPlan(plan_exp, texts, resumed=has_exp or done > 0, notes=tuple(notes))


# ---------------------------------------------------------------------------- write
def _write_atomically(path: Path, text: str) -> None:
    """Write `<name>.tmp` beside the target and rename it over, so a crash leaves each file
    either old or new, never half-written."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def apply_plan(root: Path, plan: MigrationPlan) -> list[str]:
    """Convert the setting files, then write exp.json, then remove the legacy file last: at
    every point the folder is either still a legacy one (the command can be run again) or
    done."""
    root = Path(root)
    lines = [f"resuming an interrupted migration of {root}"] if plan.resumed else []
    for path, text in plan.setting_texts.items():
        _write_atomically(path, text)
    lines.extend(plan.notes)
    lines.append(
        f"rewrote {len(plan.setting_texts)} {SETTING_FILE} file(s) ({LEGACY_HASH_KEY} -> exp_hash)"
    )
    if plan.exp_text is not None:
        _write_atomically(root / EXP_FILE, plan.exp_text)
        lines.append(f"wrote {root / EXP_FILE} ({LEGACY_AXES_KEY} -> axes)")
    (root / LEGACY_EXP_FILE).unlink()
    lines.append(f"removed {root / LEGACY_EXP_FILE}")
    return lines


def migrate_exp(root: Path) -> list[str]:
    """Migrate the folder at `root`; the lines to print. MigrationError, with nothing changed,
    if it is already migrated, has no legacy file, or its files disagree."""
    return apply_plan(root, plan_migration(root))
