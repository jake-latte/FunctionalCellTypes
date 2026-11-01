"""The experiment: one place that defines the settings of a batch of runs and runs any one of them.

An `ExpConfig` (stored as ROOT/exp.json) fixes everything that is shared by the settings
(sizes, optimiser, monitor, seeds, per-task data parameters) and, in `axes`, the hyperparameters
that vary: each axis is a dotted `Config` path (`net.L`, `train.rate_lambda`, ...) with an explicit
list of values. `expand` lists the Cartesian product in a stable order, so a SLURM array index
and a local check name the same setting. Each setting is a normal fct cache directory that
`run_setting` fills by the same path as `python -m fct train`: config.json, dataset, training
with the MIR monitor.
`run_setting(..., stage="solve")` instead solves the mean-field saddle point of the same
setting into its `meanfield/` directory, with the experiment's `meanfield` section (problem fields
plus the `solver` group; it never enters a training hash, so the same exp.json defines both
stages and an exp.json without it keeps its training hashes and directory names).

An exp.json without `axes` is the 36-setting modularity experiment (task x layers x phi x
rate_lambda), kept with its original directory names ROOT/<task>/L{L}_{phi}_r{rate}/. An experiment
with `axes` uses ROOT/<task>/s{index:04d}_<label>/ and records the axis values in
setting.json beside config.json.

This module is the I/O edge of the experiment: `expand`, `setting_config` and the config models are
pure; `run_setting`, `setting_status`, `find_setting_dir` and the exp.json functions touch disk.
"""

import hashlib
import itertools
import json
import math
import numbers
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import numpy as np
from pydantic import Field, ValidationError, model_validator

from fct.cache import (
    CachePaths,
    StaleCacheError,
    cache_paths,
    latest_checkpoint,
    read_config_hash,
    read_field,
)
from fct.config import (
    Config,
    DataConfig,
    MeanFieldConfig,
    NetConfig,
    NonNegFloat,
    NonNegInt,
    Optimizer,
    Phi,
    PosFloat,
    PosInt,
    StrictModel,
    TrainConfig,
    load_config,
    resolve_device,
    resolve_dtype,
    save_config,
)
from fct.data import Dataset, get_dataset, task_dims
from fct.meanfield import meanfield_hash, run_meanfield
from fct.monitor import make_mir_monitor
from fct.spatial import SpatialConfig
from fct.train import load_trained, train_from_config

EXP_FILE = "exp.json"
# The file and keys that older code wrote for the same things. Only `load_exp` (to tell the user
# to migrate) and `fct.exp_migrate` read them.
LEGACY_EXP_FILE = "sweep.json"
SETTING_FILE = "setting.json"
TASK_AXIS = "data.task"
SPATIAL_PREFIX = "spatial."
# Length of the hash that stands for a dict-, list- or long-valued axis value in a directory name.
LABEL_HASH_LENGTH = 6
# A value whose text is longer than this is shown as a hash in a directory name.
MAX_LABEL_VALUE = 24
# Setting directories are ROOT/<task>/<name>: `expand` builds `relpath` that way, and the lookups
# below find them with this glob.
SETTING_DIRS_GLOB = "*/*"
# Index prefix of an indexed setting directory name; zero-padded so names sort in index order, which
# holds up to 10**INDEX_WIDTH settings (`expand` refuses more).
INDEX_WIDTH = 4
# A float typed in a notebook finds the one that went through a JSON round trip.
LOOKUP_REL_TOL = 1e-9
# The modularity axes, used when exp.json has no `axes`. Axis order is the order of `expand`
# (task slowest, rate_lambda fastest): changing it or a value list would move existing indices.
EXP_TASKS = ("linear_teacher", "object_action", "what_where")
LAYERS = (1, 3, 5)
PHIS = ("tanh", "relu")
RATE_LAMBDAS = (0.0, 1e-3)
MODULARITY_EXP_AXES: dict[str, list] = {
    TASK_AXIS: list(EXP_TASKS),
    "net.L": list(LAYERS),
    "net.phi": list(PHIS),
    "train.rate_lambda": list(RATE_LAMBDAS),
}
MODULARITY_AXES = ("net.L", "net.phi", "train.rate_lambda")
# Config fields the experiment sets from its own fields; the experiment field to change instead.
EXP_OWNED_AXES = {
    "data.P": "P",
    "data.D_in": "the task parameters",
    "data.seed": "seeds.data",
    "net.seed": "seeds.net",
    "train.seed": "seeds.train",
    "train.log_int": "log_int",
    "train.save_int": "save_int",
    "train.monitor_int": "monitor_int",
    "train.monitor_bins": "monitor_bins",
    "train.monitor_shuffles": "monitor_shuffles",
}

Status = Literal["done", "partial", "missing", "stale"]
Stage = Literal["train", "solve"]
STAGES: tuple[Stage, ...] = ("train", "solve")


class ExpError(ValueError):
    """An experiment-level failure the user can fix: a bad index or a changed experiment config."""


class UnmigratedExpError(ExpError):
    """ROOT holds the legacy experiment file but no exp.json: `fct exp migrate ROOT` converts it."""


@dataclass(frozen=True)
class Setting:
    """One point of the experiment: its position, the value of every axis, and its directory.

    `axes` maps each axis to this setting's value, in the experiment's axis order. `task` is the
    data task the setting trains on: the `data.task` axis if there is one, else the experiment's
    base task. `relpath` is the setting's directory relative to the experiment ROOT.
    """

    index: int
    axes: dict[str, Any]
    task: str
    relpath: Path

    # Axis values may be unhashable (lists); the index identifies a point of one experiment.
    def __hash__(self) -> int:
        return hash((self.index, self.relpath))


# ---------------------------------------------------------------------------- config
class LinearTeacherParams(StrictModel):
    n_features: PosInt  # m: dimension of both x and y
    n_latents: PosInt
    latent_range: PosFloat


class ObjectActionParams(StrictModel):
    grid_size: PosInt
    max_objects: NonNegInt


class WhatWhereParams(StrictModel):
    n_motifs: PosInt


class Seeds(StrictModel):
    data: int = 0
    net: int = 0
    train: int = 0


class ExpConfig(StrictModel):
    """Shared settings of the runs, and the axes they vary over; see `preset`.

    `description` is free text saying what the experiment is. It is documentation only: it never
    enters a hash or a directory name, `same_experiment` ignores it, and exp.json omits it when
    it is None.

    The same data, net and training seeds are used for every setting, so settings differ only
    in the axes. Everything else is the base `Config` a setting starts from (the shared
    network and loss fields `L`, `g`, `phi`, `weight_lambda`, ... are the library defaults unless set). `axes` maps a dotted `Config` path to the list
    of values that axis takes; None means the modularity axes. Any Config field may be an axis
    and then wins over the experiment's shared field (`net.N` over `N`, `train.lr` over `lr`), except
    the fields the experiment owns (`EXP_OWNED_AXES`: sizes of the data, seeds, log/save/monitor
    schedules), which are rejected. Task parameters and the whole experiment are validated at load time
    by building every setting's `Config`, so a bad axis name, value or combination fails when
    exp.json is read, naming the axis.

    `meanfield` is the mean-field problem and its solver (`MeanFieldConfig`, library defaults;
    the rate tilt follows `train.rate_lambda` unless set). It is read by the solve stage only:
    changing it never makes a trained setting stale.

    `spatial` is the spatial weight prior (`fct.spatial.SpatialConfig`) shared by every setting;
    a `spatial.<path>` axis varies one of its fields and needs the section to be set. The solve
    stage refuses a spatial setting (the solver knows only the iid prior).
    """

    description: str | None = None
    N: PosInt
    P: PosInt
    n_epochs: NonNegInt
    lr: PosFloat
    log_int: PosInt
    save_int: PosInt
    monitor_int: PosInt  # the experiment always monitors: MIR over training is its point
    monitor_bins: Annotated[int, Field(ge=2)]
    monitor_shuffles: PosInt
    linear_teacher: LinearTeacherParams
    object_action: ObjectActionParams
    what_where: WhatWhereParams
    seeds: Seeds = Seeds()
    dtype: Literal["float32", "float64"] = "float32"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    # The base task, used unless `data.task` is an axis.
    task: Literal["linear_teacher", "object_action", "what_where"] = "linear_teacher"
    # Shared network and loss fields, the library defaults unless set; any of them may be an axis.
    L: PosInt = NetConfig().L
    g: PosFloat = NetConfig().g
    phi: Phi = NetConfig().phi
    weight_lambda: NonNegFloat = TrainConfig().weight_lambda
    rate_lambda: NonNegFloat = TrainConfig().rate_lambda
    optimizer: Optimizer = TrainConfig().optimizer
    beta: PosFloat = TrainConfig().beta
    # The spatial weight prior of every setting (`Config.spatial`); None trains under the iid
    # prior (the plain weight L2). Its fields may be axes (`spatial.hidden.kernel.length_scale`).
    spatial: SpatialConfig | None = None
    meanfield: MeanFieldConfig = MeanFieldConfig()
    axes: dict[str, list[Any]] | None = None

    @model_validator(mode="after")
    def _task_parameters_are_consistent(self) -> Self:
        for task in EXP_TASKS:
            data_config(self, task)  # raises ValidationError naming the offending field
        return self

    @model_validator(mode="after")
    def _axes_are_valid(self) -> Self:
        if self.axes is not None:
            _validate_axes(self, self.axes)
        elif self.spatial is not None:
            # The built-in axes: every setting must still build a valid Config (the spatial
            # section ties weight_lambda to lambda_max). Experiments without one have no such
            # cross-field rule, so they are not checked here.
            _validate_combinations(self)
        return self


def _task_fields(exp: ExpConfig, task: str) -> dict:
    """The DataConfig fields set by the task's parameters (D_in only where the task has it)."""
    params = getattr(exp, task)
    if task == "linear_teacher":
        return {
            "D_in": params.n_features,
            "n_latents": params.n_latents,
            "latent_range": params.latent_range,
        }
    return params.model_dump()


def data_config(exp: ExpConfig, task: str) -> DataConfig:
    """DataConfig of `task` with `D_in` taken from `task_dims`, so the input size the task
    fixes (2*G*G, 81) is never repeated in the experiment."""
    fields = {"task": task, "P": exp.P, "seed": exp.seeds.data, **_task_fields(exp, task)}
    # Unvalidated copy: object_action and what_where reject the default D_in, which `task_dims`
    # ignores for them. The returned config is fully validated.
    d_in, _ = task_dims(DataConfig.model_construct(**fields))
    return DataConfig(**{**fields, "D_in": d_in})


def with_axes(exp: ExpConfig, axes: dict[str, list], description: str | None = None) -> ExpConfig:
    """A validated copy of `exp` over `axes`. The copy has `description` (None unless given): the
    description of `exp` describes its own axes, which no longer apply."""
    return ExpConfig.model_validate({**exp.model_dump(), "axes": axes, "description": description})


# ---------------------------------------------------------------------------- the presets
class PresetError(ExpError):
    """A preset that cannot be used: unknown, unreadable or invalid. The message names the file."""


# A preset is `<name>.json` in the presets folder; the stem must be a Python-style identifier so
# it works as a command-line value and in file names without quoting.
PRESET_NAME_RULE = "[A-Za-z_][A-Za-z0-9_]*"
_PRESET_STEM = re.compile(PRESET_NAME_RULE)
PRESET_SUFFIX = ".json"


def packaged_presets() -> Traversable:
    """The presets folder that ships in the library (a real folder, or a folder inside a zip)."""
    return resources.files("fct").joinpath("presets")


def _folder(directory: Path | str | None) -> Traversable:
    return packaged_presets() if directory is None else Path(directory)


def _is_preset_file(entry: Traversable) -> bool:
    """A regular file `<stem>.json`, not hidden, that is a candidate for a preset."""
    name = entry.name
    return entry.is_file() and name.endswith(PRESET_SUFFIX) and not name.startswith(".")


def _list_presets(directory: Path | str | None) -> tuple[list[str], list[str]]:
    """`(names, ignored)` in the folder: the names of the usable presets, sorted, and the
    `*.json` files ignored because their stem breaks `PRESET_NAME_RULE`. Subdirectories, hidden
    files and files of other kinds (a README) are neither."""
    folder = _folder(directory)
    try:
        entries = [e for e in folder.iterdir() if _is_preset_file(e)]
    except OSError as err:
        raise PresetError(f"cannot list the presets folder {folder}: {err}") from err
    stems = sorted(e.name.removesuffix(PRESET_SUFFIX) for e in entries)
    names = [stem for stem in stems if _PRESET_STEM.fullmatch(stem)]
    ignored = [stem + PRESET_SUFFIX for stem in stems if not _PRESET_STEM.fullmatch(stem)]
    return names, ignored


def preset_names(directory: Path | str | None = None) -> list[str]:
    """The presets available now, sorted: found when called, so a file added to the folder is a
    preset at once. `directory` replaces the packaged folder."""
    return _list_presets(directory)[0]


def _unknown_preset_message(
    name: str, folder: Traversable, names: list[str], ignored: list[str]
) -> str:
    message = f"unknown preset {name!r}; available in {folder}: {names}"
    if ignored:
        message += (
            f". Ignored (the name before {PRESET_SUFFIX} must match {PRESET_NAME_RULE}): {ignored}"
        )
    return message


def preset(name: str, directory: Path | str | None = None) -> ExpConfig:
    """The base experiment `<directory>/<name>.json` (the packaged `fct/presets` by default),
    validated. PresetError, naming the full path, if the name is unknown (the message lists
    the presets and the ignored files) or the file cannot be read or is not a valid experiment."""
    names, ignored = _list_presets(directory)
    folder = _folder(directory)
    if name not in names:
        raise PresetError(_unknown_preset_message(name, folder, names, ignored))
    path = folder.joinpath(name + PRESET_SUFFIX)
    try:
        return ExpConfig.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as err:
        raise PresetError(f"cannot read the preset file {path}: {err}") from err
    except ValidationError as err:  # also covers JSON that does not parse
        raise PresetError(f"the preset file {path} is not a valid experiment: {err}") from err


def exp_path(root: Path) -> Path:
    return Path(root) / EXP_FILE


def exp_text(exp: ExpConfig) -> str:
    """The text of exp.json. The description is left out when there is none, so an experiment
    without one has the text it had before the field existed."""
    exclude = {"description"} if exp.description is None else None
    return exp.model_dump_json(indent=2, exclude=exclude) + "\n"


def write_exp_text(path: Path, text: str) -> None:
    """Write `text` as the exp.json at `path`, the one way exp.json is written: UTF-8, through a
    `.tmp` sibling renamed over the target so a crash leaves the old file or the new, never half."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_exp(exp: ExpConfig, root: Path) -> None:
    path = exp_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_exp_text(path, exp_text(exp))


def same_experiment(a: ExpConfig, b: ExpConfig) -> bool:
    """Whether two experiments are the same one: every field equal except the description,
    which is documentation and may be edited without changing what runs."""
    return a.model_dump(exclude={"description"}) == b.model_dump(exclude={"description"})


def load_exp(root: Path) -> ExpConfig:
    """Read ROOT/exp.json; FileNotFoundError if absent, ValidationError if invalid,
    UnmigratedExpError if only the legacy file is there."""
    path = exp_path(root)
    if not path.exists():
        if (Path(root) / LEGACY_EXP_FILE).exists():
            raise UnmigratedExpError(
                f"{root} holds a legacy {LEGACY_EXP_FILE} and no {EXP_FILE}; "
                f"run `python -m fct exp migrate {root}` once to convert it"
            )
        raise FileNotFoundError(f"{path} not found")
    return ExpConfig.model_validate_json(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------- the axes
def axes_of(exp: ExpConfig) -> dict[str, list]:
    """The experiment's axes: its `axes`, or the modularity axes when it has none."""
    return MODULARITY_EXP_AXES if exp.axes is None else exp.axes


def _is_section(node: Any) -> bool:
    return isinstance(node, dict)


def _set_leaf(tree: dict, axis: str, value: Any) -> None:
    """Set the Config field at the dotted path `axis` in a dumped config tree."""
    *parents, last = axis.split(".")
    node = tree
    for part in parents:
        node = node.get(part)
        if not _is_section(node):
            raise ValueError(f"axis {axis!r}: {part!r} is not a config section")
    if last not in node:
        raise ValueError(f"axis {axis!r}: no config field {last!r}")
    if _is_section(node[last]):
        raise ValueError(f"axis {axis!r} is a config section; name one of its fields")
    node[last] = value


def _base_config(exp: ExpConfig, task: str) -> Config:
    """The shared part of every setting's `Config`; the axes are written over it."""
    return Config(
        data=data_config(exp, task),
        net=NetConfig(N=exp.N, L=exp.L, g=exp.g, phi=exp.phi, seed=exp.seeds.net),
        train=TrainConfig(
            n_epochs=exp.n_epochs,
            lr=exp.lr,
            weight_lambda=exp.weight_lambda,
            rate_lambda=exp.rate_lambda,
            optimizer=exp.optimizer,
            beta=exp.beta,
            log_int=exp.log_int,
            save_int=exp.save_int,
            seed=exp.seeds.train,
            monitor_int=exp.monitor_int,
            monitor_bins=exp.monitor_bins,
            monitor_shuffles=exp.monitor_shuffles,
        ),
        meanfield=exp.meanfield,
        spatial=exp.spatial,
        dtype=exp.dtype,
        device=exp.device,
    )


def _config_from(exp: ExpConfig, axes: Mapping[str, Any]) -> Config:
    """The experiment's base `Config` for the task, with each axis value written over it."""
    task = axes.get(TASK_AXIS, exp.task)
    if task not in EXP_TASKS:
        raise ValueError(f"axis {TASK_AXIS!r}: unknown task {task!r}; valid: {EXP_TASKS}")
    tree = _base_config(exp, task).model_dump()
    for axis, value in axes.items():
        if axis != TASK_AXIS:  # already chosen above, together with the task's own D_in
            _set_leaf(tree, axis, value)
    return Config.model_validate(tree)


def _config_or_value_error(exp: ExpConfig, axes: Mapping[str, Any], context: str) -> Config:
    """`_config_from`, with a pydantic ValidationError turned into a ValueError that starts
    with `context` (which axis or setting was being built)."""
    try:
        return _config_from(exp, axes)
    except ValidationError as err:
        raise ValueError(f"{context}: {err}") from err


def _validate_axis(exp: ExpConfig, axis: str, values: list) -> None:
    """One axis alone: experiment-owned fields are rejected, the values are non-empty, each builds
    a valid Config, and no two build the same one (so `[0, 0.0]` counts as a repeat).
    Each value is built under the experiment's base task, so a task-specific value that is valid
    only under a task named on a `data.task` axis is rejected here (loudly, naming the axis)."""
    if axis.startswith(SPATIAL_PREFIX) and exp.spatial is None:
        raise ValueError(f"axis {axis!r}: set the experiment's spatial section first")
    if axis in EXP_OWNED_AXES:
        raise ValueError(
            f"axis {axis!r} is set by the experiment's own {EXP_OWNED_AXES[axis]}; "
            "change that instead of varying it"
        )
    if not values:
        raise ValueError(f"axis {axis!r} has no values")
    configs = [_config_or_value_error(exp, {axis: v}, f"axis {axis!r} value {v!r}") for v in values]
    encoded = [cfg.model_dump_json() for cfg in configs]
    if len(set(encoded)) != len(encoded):
        raise ValueError(f"axis {axis!r} repeats a value: {values}")


def _validate_combinations(exp: ExpConfig) -> None:
    """Every setting of the product builds a valid Config (axes can be fine alone and wrong
    together, e.g. more objects than cells of the task's grid)."""
    for setting in expand(exp):
        context = f"setting {setting.index} {setting.axes}"
        _config_or_value_error(exp, setting.axes, context)


def _validate_axes(exp: ExpConfig, axes: dict[str, list]) -> None:
    """Axes first, one at a time, so an error names the culprit; then the combinations."""
    if not axes:
        raise ValueError("no axes (a product over no axes would be one bare setting)")
    for axis, values in axes.items():
        _validate_axis(exp, axis, values)
    _validate_combinations(exp)


def _label_part(axis: str, value: Any) -> str:
    """`<last part of the axis path><value>` for a directory name, e.g. `rate_lambda1e-06`; a
    dict or list value (or any text longer than MAX_LABEL_VALUE) is shown as a short hash of its
    contents, so the name stays within NAME_MAX.
    Characters outside letters, digits and `._+-` become `-` so the name is a safe path
    component. The label is for people; uniqueness comes from the index prefix."""
    text = f"{value:g}" if isinstance(value, float) else str(value)
    if isinstance(value, dict | list) or len(text) > MAX_LABEL_VALUE:
        digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        text = f"-{digest[:LABEL_HASH_LENGTH]}"
    return re.sub(r"[^A-Za-z0-9._+-]", "-", axis.rsplit(".", 1)[-1] + text)


def _indexed_name(index: int, axes: Mapping[str, Any]) -> str:
    """`s0007_L2_phitanh_...`: the index makes the name unique (`:g` would merge close floats),
    the label (the task is already the parent directory) is for people."""
    label = "_".join(_label_part(a, v) for a, v in axes.items() if a != TASK_AXIS)
    prefix = f"s{index:0{INDEX_WIDTH}d}"
    return f"{prefix}_{label}" if label else prefix


def _modularity_name(values: Mapping[str, Any]) -> str:
    """`:g` gives `r0` and `r0.001`, distinct per value of the modularity axes."""
    return f"L{values['net.L']}_{values['net.phi']}_r{values['train.rate_lambda']:g}"


def expand(exp: ExpConfig) -> list[Setting]:
    """All settings: the Cartesian product of the axes in the order they are written, the
    first axis slowest. The order depends only on the axes, not on the experiment's sizes, so
    index i names the same setting for every size of the same axes."""
    varied = axes_of(exp)
    n_settings = math.prod(len(values) for values in varied.values())
    if exp.axes is not None and n_settings > 10**INDEX_WIDTH:
        raise ExpError(
            f"the experiment has {n_settings} settings; directory names support at most "
            f"{10**INDEX_WIDTH} (INDEX_WIDTH={INDEX_WIDTH})"
        )
    built = built_values(exp)
    settings = []
    for index, combo in enumerate(itertools.product(*(enumerate(v) for v in varied.values()))):
        axes = {axis: varied[axis][i] for axis, (i, _) in zip(varied, combo, strict=True)}
        # Names come from the built values, the same ones the exp hash uses, so two spellings
        # of one value (`1000000`, `1000000.0`) cannot hash alike yet name different directories.
        named = {axis: built[axis][i] for axis, (i, _) in zip(varied, combo, strict=True)}
        task = axes.get(TASK_AXIS, exp.task)
        name = _modularity_name(named) if exp.axes is None else _indexed_name(index, named)
        settings.append(Setting(index, axes, task, Path(task) / name))
    return settings


def get_setting(exp: ExpConfig, index: int) -> Setting:
    settings = expand(exp)
    if not 0 <= index < len(settings):
        raise ExpError(f"index {index} is out of range; valid indices are 0..{len(settings) - 1}")
    return settings[index]


def setting_dir(root: Path, setting: Setting) -> Path:
    return Path(root) / setting.relpath


def setting_config(exp: ExpConfig, setting: Setting) -> Config:
    """The full fct `Config` of one setting."""
    return _config_from(exp, setting.axes)


def _dump_at(dump: dict, axis: str) -> Any:
    node = dump
    for part in axis.split("."):
        node = node[part]
    return node


def built_values(exp: ExpConfig) -> dict[str, list]:
    """Each axis's values as the built `Config` holds them (`0` becomes `0.0` for a float field),
    so two spellings of one value become one. This is the sameness `_validate_axis` enforces."""
    return {
        axis: [_dump_at(_config_from(exp, {axis: v}).model_dump(), axis) for v in values]
        for axis, values in axes_of(exp).items()
    }


def exp_hash(exp: ExpConfig) -> str:
    """Short hash of what the axes set: the built values of every axis. The base task counts
    only when `data.task` is not an axis. The sizes are not in it; config.json checks those."""
    base_task = None if TASK_AXIS in axes_of(exp) else exp.task
    canonical = json.dumps({"axes": built_values(exp), "task": base_task})
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------- disk
def _train_hash(cfg: Config) -> str:
    return cfg.section_hash("data", "net", "train")


def _check_stage(stage: str) -> None:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; valid: {list(STAGES)}")


def _artefact_status(directory: Path, current_hash: str) -> Status:
    """done = directory/final.pt exists and carries `current_hash`; stale = a final.pt with
    another hash, or one that cannot be read; partial = checkpoints but no final.pt; else
    missing. Only the hash of final.pt is read."""
    final = directory / "final.pt"
    if final.exists():
        try:
            return "done" if read_config_hash(final) == current_hash else "stale"
        except ValueError:  # an unreadable final.pt cannot be reused, whatever config it had
            return "stale"
    return "partial" if latest_checkpoint(directory) is not None else "missing"


def setting_status(root: Path, exp: ExpConfig, setting: Setting, stage: Stage = "train") -> Status:
    """State of one stage of a setting. "train": train/final.pt written under this setting's
    current data, net and train sections and the loss version. "solve": meanfield/final.pt
    written under those and the meanfield section. Otherwise as `_artefact_status`: stale
    is a final.pt written under another config (or unreadable), partial a checkpoint without a
    final.pt, else missing. A finished solve that ran without the net is also stale once the
    net is trained (`_solved_without_net`): it lacks the comparison with the net."""
    _check_stage(stage)
    paths = cache_paths(setting_dir(root, setting))
    cfg = setting_config(exp, setting)
    if stage == "train":
        return _artefact_status(paths.train, _train_hash(cfg))
    if _done_without_net(paths, cfg):
        return "stale"
    return _artefact_status(paths.meanfield, meanfield_hash(cfg))


def setting_statuses(
    root: Path, exp: ExpConfig, stage: Stage = "train"
) -> list[tuple[int, Setting, Status]]:
    """`(index, setting, status)` for every setting of the experiment, in index order."""
    return [(i, s, setting_status(root, exp, s, stage)) for i, s in enumerate(expand(exp))]


def _done_without_net(paths: CachePaths, cfg: Config) -> bool:
    """A solve finished under the CURRENT config that ran with no net, while a trained net now
    exists. The hash matches (so a solve of another config is never taken for this case); only
    the stored kernels of the net (absent) tell, and only that entry is read."""
    if not (paths.train / "final.pt").exists():
        return False
    if _artefact_status(paths.meanfield, meanfield_hash(cfg)) != "done":
        return False
    try:
        return read_field(paths.meanfield / "final.pt", "net_Phi") is None
    except ValueError:  # became unreadable since the hash was read: stale, not a redo
        return False


def _write_config_once(cfg: Config, directory: Path) -> None:
    """Write config.json for a new setting; fail if an existing one differs in anything but the
    meanfield section, because the artefacts beside it belong to that config. The meanfield
    section is the experiment's current solver (no training artefact depends on it, and `python -m
    fct solve` reads it from here), so it is brought up to date."""
    path = cache_paths(directory).config
    if not path.exists():
        save_config(cfg, path)
        return
    existing = load_config(path)
    if existing.model_copy(update={"meanfield": cfg.meanfield}) != cfg:
        raise ExpError(
            f"{path} differs from the config exp.json builds for this setting; "
            "use a fresh ROOT or delete the setting directory"
        )
    if existing.meanfield != cfg.meanfield:
        save_config(cfg, path)


def _setting_record(exp: ExpConfig, setting: Setting) -> dict:
    return {"index": setting.index, "exp_hash": exp_hash(exp), "axes": setting.axes}


def _write_setting_once(exp: ExpConfig, setting: Setting, directory: Path) -> None:
    """Record the setting's index, axis values and exp hash in setting.json; fail if an
    existing record differs, because the experiment then no longer puts this setting here."""
    path = directory / SETTING_FILE
    record = _setting_record(exp, setting)
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + "\n")
    elif json.loads(path.read_text()) != json.loads(json.dumps(record)):
        raise ExpError(
            f"{path} differs from what exp.json gives this setting (another experiment, or other "
            "axis values at this index); use a fresh ROOT or delete the setting directory"
        )


def _read_setting(directory: Path, current_hash: str) -> dict:
    record = json.loads((Path(directory) / SETTING_FILE).read_text())
    if record.get("exp_hash") != current_hash:
        raise ExpError(
            f"{directory} has no exp hash or was written under another experiment than this "
            "exp.json's; use a fresh ROOT or delete the directory"
        )
    return record


def read_setting(directory: Path, exp: ExpConfig) -> dict:
    """The `{"index", "exp_hash", "axes"}` record `run_setting` wrote in a setting directory.
    ExpError, naming the directory, if it was written under another experiment than `exp`'s (or
    carries no exp hash)."""
    return _read_setting(directory, exp_hash(exp))


def has_settings(root: Path) -> bool:
    """Whether ROOT already holds a setting directory (anything with a config.json)."""
    dirs = (d for d in Path(root).glob(SETTING_DIRS_GLOB) if d.is_dir())
    return any(cache_paths(d).config.exists() for d in dirs)


def same_value(a: Any, b: Any) -> bool:
    """Equality for looking a setting up by its values: numbers (also inside lists, and numpy
    scalars of float64 or integer type) agree to a relative `LOOKUP_REL_TOL`; a bool equals only
    a bool, so `True` never finds `L=1`. A float32 scalar carries only ~1e-7 precision, so it
    does not meet `LOOKUP_REL_TOL` against the float64 that was written."""
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, bool | np.bool_) or isinstance(b, bool | np.bool_):
        return isinstance(a, bool | np.bool_) and isinstance(b, bool | np.bool_) and a == b
    if isinstance(a, numbers.Real) and isinstance(b, numbers.Real):
        return math.isclose(a, b, rel_tol=LOOKUP_REL_TOL)
    return a == b


def find_setting_dir(root: Path, exp: ExpConfig, values: Mapping[str, Any]) -> Path:
    """The directory of the one already-run setting of `exp` whose axes equal `values`; a
    subset of the axes is enough when it is unambiguous. LookupError if none or several match,
    or an axis is not one of the experiment's; ExpError if a directory was written under another
    experiment. Only the experiment's own directories are considered, so strays in ROOT cannot
    match."""
    if unknown := sorted(set(values) - set(axes_of(exp))):
        raise LookupError(f"unknown axis {unknown}; the experiment has axes {list(axes_of(exp))}")
    directories = [setting_dir(root, s) for s in expand(exp)]
    run = [d for d in directories if (d / SETTING_FILE).exists()]
    if not run:
        raise LookupError(
            f"no setting of this experiment has been run under {root} (settings run before "
            f"{SETTING_FILE} existed get one on their next `exp run`)"
        )
    current = exp_hash(exp)  # once: it builds every setting's config
    records = [(directory, _read_setting(directory, current)["axes"]) for directory in run]
    matches = [
        directory
        for directory, axes in records
        if all(same_value(axes[a], v) for a, v in values.items())
    ]
    if not matches:
        raise LookupError(f"no setting under {root} has axes {dict(values)}")
    if len(matches) > 1:
        raise LookupError(f"{len(matches)} settings match {dict(values)}; give more axes")
    return matches[0]


_STAGE_DIRS = {"train": "train", "solve": "meanfield"}
_STAGE_SECTIONS = {
    "train": "(data, net or train sections, or model or loss definition)",
    "solve": "(data, net, train or meanfield sections, or model or loss definition)",
}


def run_setting(
    exp: ExpConfig,
    index: int,
    root: Path,
    log: Callable[[str], None] = print,
    stage: Stage = "train",
) -> None:
    """Run one stage of setting `index` in its cache directory; resumes a partial run, skips a
    done one. "train" trains the net; "solve" solves the mean-field saddle point, comparing
    with the trained net if there is one (see `_solve_setting`).

    StaleCacheError if the stage was finished under another config or loss definition: its
    results would silently stand in for a run of the current one. ExpError for the solve
    stage of an ensemble setting (net.B > 1), before anything is written, whether or not it is
    trained yet: the solve compares with one trained network."""
    _check_stage(stage)
    setting = get_setting(exp, index)
    directory = setting_dir(root, setting)
    cfg = setting_config(exp, setting)
    if stage == "solve" and cfg.net.B > 1:
        raise ExpError(
            f"setting {index}: the solve compares with one trained network, but the setting "
            f"is an ensemble of net.B={cfg.net.B}; solving ensembles is not supported"
        )
    _write_config_once(cfg, directory)
    _write_setting_once(exp, setting, directory)
    status = setting_status(root, exp, setting, stage)
    paths = cache_paths(directory)
    if stage == "solve" and _done_without_net(paths, cfg):
        # Finished under the current config, only lacking the net: solve again with it.
        log(f"setting {index}: solved without the net, but the net is trained now; solving again")
        (paths.meanfield / "final.pt").unlink(missing_ok=True)
        status = "missing"
    if status == "stale":
        raise StaleCacheError(
            f"setting {index}: {directory}/{_STAGE_DIRS[stage]}/final.pt was written under a "
            f"different config {_STAGE_SECTIONS[stage]} than the current one, or is unreadable, "
            "so its results are not those of this experiment. To rerun it, delete the experiment "
            "ROOT or use a fresh one."
        )
    if status == "done":
        log(f"setting {index} ({directory}) already done ({stage})")
        return
    if stage == "solve" and cfg.meanfield.n_points > cfg.data.P:
        raise ExpError(
            f"setting {index}: meanfield.n_points={cfg.meanfield.n_points} exceeds the dataset "
            f"size P={cfg.data.P}; lower it in exp.json"
        )
    log(f"setting {index}: {directory} ({stage})")
    dataset = get_dataset(directory, cfg)
    if stage == "train":
        train_from_config(cfg, dataset, directory, log=log, monitor=make_mir_monitor(dataset, cfg))
    else:
        _solve_setting(cfg, dataset, directory, log)


def _check_checkpoint_matches_net(directory: Path, net) -> None:
    """A solver checkpoint continues the run it came from, which fixed whether a net is
    compared; ExpError (one line, naming the file) if the net situation has changed since."""
    ckpt = latest_checkpoint(cache_paths(directory).meanfield)
    if ckpt is not None and (read_field(ckpt, "net_Phi") is not None) != (net is not None):
        was = "with the trained net, but there is no trained net now"
        if net is not None:
            was = "without the trained net, but there is a trained net now"
        raise ExpError(f"{ckpt} was written {was}; delete {ckpt.parent} to solve again")


def _solve_setting(
    cfg: Config, dataset: Dataset, directory: Path, log: Callable[[str], None]
) -> None:
    """Solve the setting's saddle point. The trained net, if train/final.pt exists, adds the
    comparison with the net to the history; without one the solve runs on its own. The choice
    is fixed when a solve starts: resuming a checkpoint needs the same (ExpError otherwise,
    see `_check_checkpoint_matches_net`). `run_setting` refuses ensembles (net.B > 1) before
    this. An InadequateSamplerError propagates and leaves no final.pt."""
    net = None
    if (cache_paths(directory).train / "final.pt").exists():
        net, _ = load_trained(directory, cfg, resolve_device(cfg), resolve_dtype(cfg.dtype))
    else:
        log("no trained net: solving without the comparison with the net")
    _check_checkpoint_matches_net(directory, net)
    run_meanfield(cfg, dataset, net, directory, log=log)
