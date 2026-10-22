"""The sweep: one place that defines the settings of a grid and runs any one of them.

A `SweepConfig` (stored as ROOT/sweep.json) fixes everything that is shared by the settings
(sizes, optimiser, monitor, seeds, per-task data parameters) and, in `grid`, the axes that
vary: each axis is a dotted `Config` path (`net.L`, `train.rate_lambda`, ...) with an explicit
list of values. `expand` lists the Cartesian product in a stable order, so a SLURM array index
and a local check name the same setting. Each setting is a normal fct cache directory that
`run_setting` fills by the same path as `python -m fct train`: config.json, dataset, training
with the MIR monitor.

A sweep.json without `grid` is the 36-setting modularity grid (task x layers x phi x
rate_lambda), kept with its original directory names ROOT/<task>/L{L}_{phi}_r{rate}/. A sweep
with a `grid` uses ROOT/<task>/s{index:04d}_<label>/ and records the axis values in
setting.json beside config.json.

This module is the I/O edge of the sweep: `expand`, `setting_config` and the config models are
pure; `run_setting`, `setting_status`, `find_setting_dir` and the sweep.json functions touch disk.
"""

import hashlib
import itertools
import json
import math
import numbers
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import numpy as np
from pydantic import Field, ValidationError, model_validator

from fct.cache import StaleCacheError, cache_paths, latest_checkpoint, read_config_hash
from fct.config import (
    Config,
    DataConfig,
    NetConfig,
    NonNegInt,
    PosFloat,
    PosInt,
    StrictModel,
    TrainConfig,
    load_config,
    save_config,
)
from fct.data import get_dataset, task_dims
from fct.monitor import make_mir_monitor
from fct.train import train_from_config

SWEEP_FILE = "sweep.json"
SETTING_FILE = "setting.json"
TASK_AXIS = "data.task"
# Setting directories are ROOT/<task>/<name>: `expand` builds `relpath` that way, and the lookups
# below find them with this glob.
SETTING_DIRS_GLOB = "*/*"
# Index prefix of a grid directory name; zero-padded so names sort in index order, which holds up
# to 10**INDEX_WIDTH settings (`expand` refuses more).
INDEX_WIDTH = 4
# A float typed in a notebook finds the one that went through a JSON round trip.
LOOKUP_REL_TOL = 1e-9
# The modularity grid, used when sweep.json has no `grid`. Axis order is the order of `expand`
# (task slowest, rate_lambda fastest): changing it or a value list would move existing indices.
SWEEP_TASKS = ("linear_teacher", "object_action", "what_where")
LAYERS = (1, 3, 5)
PHIS = ("tanh", "relu")
RATE_LAMBDAS = (0.0, 1e-3)
MODULARITY_GRID: dict[str, list] = {
    TASK_AXIS: list(SWEEP_TASKS),
    "net.L": list(LAYERS),
    "net.phi": list(PHIS),
    "train.rate_lambda": list(RATE_LAMBDAS),
}
MODULARITY_AXES = ("net.L", "net.phi", "train.rate_lambda")
# Config fields the sweep sets from its own fields; the sweep field to change instead.
SWEEP_OWNED_AXES = {
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


class SweepError(ValueError):
    """A sweep-level failure the user can fix: a bad index or a changed sweep config."""


@dataclass(frozen=True)
class Setting:
    """One point of the grid: its position, the value of every axis, and its directory.

    `axes` maps each grid axis to this setting's value, in the grid's axis order. `task` is the
    data task the setting trains on: the `data.task` axis if there is one, else the sweep's base
    task. `relpath` is the setting's directory relative to the sweep ROOT.
    """

    index: int
    axes: dict[str, Any]
    task: str
    relpath: Path

    # Axis values may be unhashable (lists); the index identifies a point of one grid.
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


class SweepConfig(StrictModel):
    """Shared settings of the runs, and the grid they are swept over; see `preset`.

    The same data, net and training seeds are used for every setting, so settings differ only
    in the grid axes. Everything else is the base `Config` a setting starts from (`weight_lambda`
    is the library default unless it is an axis). `grid` maps a dotted `Config` path to the list
    of values that axis takes; None means the modularity grid. Any Config field may be an axis
    and then wins over the sweep's shared field (`net.N` over `N`, `train.lr` over `lr`), except
    the fields the sweep owns (`SWEEP_OWNED_AXES`: sizes of the data, seeds, log/save/monitor
    schedules), which are rejected. Task parameters and the whole grid are validated at load time
    by building every setting's `Config`, so a bad axis name, value or combination fails when
    sweep.json is read, naming the axis.
    """

    N: PosInt
    P: PosInt
    n_epochs: NonNegInt
    lr: PosFloat
    log_int: PosInt
    save_int: PosInt
    monitor_int: PosInt  # the sweep always monitors: MIR over training is its point
    monitor_bins: Annotated[int, Field(ge=2)]
    monitor_shuffles: PosInt
    linear_teacher: LinearTeacherParams
    object_action: ObjectActionParams
    what_where: WhatWhereParams
    seeds: Seeds = Seeds()
    dtype: Literal["float32", "float64"] = "float32"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    # The base task, used unless `data.task` is a grid axis.
    task: Literal["linear_teacher", "object_action", "what_where"] = "linear_teacher"
    grid: dict[str, list[Any]] | None = None

    @model_validator(mode="after")
    def _task_parameters_are_consistent(self) -> Self:
        for task in SWEEP_TASKS:
            data_config(self, task)  # raises ValidationError naming the offending field
        return self

    @model_validator(mode="after")
    def _grid_is_valid(self) -> Self:
        if self.grid is not None:
            _validate_grid(self, self.grid)
        return self


def _task_fields(sweep: SweepConfig, task: str) -> dict:
    """The DataConfig fields set by the task's parameters (D_in only where the task has it)."""
    params = getattr(sweep, task)
    if task == "linear_teacher":
        return {
            "D_in": params.n_features,
            "n_latents": params.n_latents,
            "latent_range": params.latent_range,
        }
    return params.model_dump()


def data_config(sweep: SweepConfig, task: str) -> DataConfig:
    """DataConfig of `task` with `D_in` taken from `task_dims`, so the input size the task
    fixes (2*G*G, 81) is never repeated in the sweep."""
    fields = {"task": task, "P": sweep.P, "seed": sweep.seeds.data, **_task_fields(sweep, task)}
    # Unvalidated copy: object_action and what_where reject the default D_in, which `task_dims`
    # ignores for them. The returned config is fully validated.
    d_in, _ = task_dims(DataConfig.model_construct(**fields))
    return DataConfig(**{**fields, "D_in": d_in})


_SIZE_PRESETS: dict[str, SweepConfig] = {
    # Seconds per setting on a CPU: for checking the pipeline, not for results.
    "tiny": SweepConfig(
        N=8,
        P=32,
        n_epochs=6,
        lr=1e-2,
        log_int=2,
        save_int=2,
        monitor_int=2,
        monitor_bins=4,
        monitor_shuffles=2,
        linear_teacher=LinearTeacherParams(n_features=8, n_latents=3, latent_range=1.0),
        object_action=ObjectActionParams(grid_size=3, max_objects=2),
        what_where=WhatWhereParams(n_motifs=4),
        device="cpu",
    ),
    # Cluster sizes; N, P and n_epochs are the knobs to tune before submitting.
    "full": SweepConfig(
        N=256,
        P=4096,
        n_epochs=15_000,
        lr=1e-3,
        log_int=100,
        save_int=500,
        monitor_int=100,
        monitor_bins=10,
        monitor_shuffles=5,
        linear_teacher=LinearTeacherParams(n_features=64, n_latents=3, latent_range=1.0),
        object_action=ObjectActionParams(grid_size=5, max_objects=3),
        what_where=WhatWhereParams(n_motifs=8),
    ),
}


def with_grid(sweep: SweepConfig, grid: dict[str, list]) -> SweepConfig:
    """A validated copy of `sweep` over `grid`."""
    return SweepConfig.model_validate({**sweep.model_dump(), "grid": grid})


def phase_transition(n_lambda: int) -> SweepConfig:
    """Linear-teacher sweep over depth, activation and both regularisers: L in {1, 2}, phi in
    {tanh, relu}, rate_lambda and weight_lambda each `n_lambda` log-spaced values in [1e-6, 1].
    Cluster sizes; the shared sizes are the knobs to tune before submitting."""
    if n_lambda < 1:
        raise ValueError(f"n_lambda must be at least 1, got {n_lambda}")
    lambdas = np.logspace(-6, 0, n_lambda).tolist()
    grid = {
        "net.L": [1, 2],
        "net.phi": ["tanh", "relu"],
        "train.rate_lambda": lambdas,
        "train.weight_lambda": lambdas,
    }
    return with_grid(_SIZE_PRESETS["full"], grid)


# Presets with an explicit grid, by name; each takes n_lambda, the points per lambda axis.
GRID_PRESETS: dict[str, Callable[[int], SweepConfig]] = {"phase_transition": phase_transition}
PRESET_NAMES = ("tiny", "full", "modularity", *GRID_PRESETS)


def preset(name: str, n_lambda: int | None = None) -> SweepConfig:
    """A ready-made sweep. `tiny` and `full` are two sizes of the modularity grid (`modularity`
    is `full`); the grid presets need `n_lambda`, the others reject it."""
    if name not in PRESET_NAMES:
        raise ValueError(f"unknown preset {name!r}; valid: {sorted(PRESET_NAMES)}")
    if name in GRID_PRESETS:
        if n_lambda is None:
            raise ValueError(f"preset {name!r} needs n_lambda (points per lambda axis)")
        return GRID_PRESETS[name](n_lambda)
    if n_lambda is not None:
        raise ValueError(f"preset {name!r} has no lambda axes, so n_lambda does not apply")
    return _SIZE_PRESETS["tiny" if name == "tiny" else "full"]


def sweep_path(root: Path) -> Path:
    return Path(root) / SWEEP_FILE


def save_sweep(sweep: SweepConfig, root: Path) -> None:
    path = sweep_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sweep.model_dump_json(indent=2) + "\n")


def load_sweep(root: Path) -> SweepConfig:
    """Read ROOT/sweep.json; FileNotFoundError if absent, ValidationError if invalid."""
    path = sweep_path(root)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return SweepConfig.model_validate_json(path.read_text())


# ---------------------------------------------------------------------------- the grid
def grid_of(sweep: SweepConfig) -> dict[str, list]:
    """The sweep's axes: its `grid`, or the modularity grid when it has none."""
    return MODULARITY_GRID if sweep.grid is None else sweep.grid


def _is_section(node: Any) -> bool:
    return isinstance(node, dict)


def _set_leaf(tree: dict, axis: str, value: Any) -> None:
    """Set the Config field at the dotted path `axis` in a dumped config tree."""
    *parents, last = axis.split(".")
    node = tree
    for part in parents:
        node = node.get(part)
        if not _is_section(node):
            raise ValueError(f"grid axis {axis!r}: {part!r} is not a config section")
    if last not in node:
        raise ValueError(f"grid axis {axis!r}: no config field {last!r}")
    if _is_section(node[last]):
        raise ValueError(f"grid axis {axis!r} is a config section; name one of its fields")
    node[last] = value


def _base_config(sweep: SweepConfig, task: str) -> Config:
    """The shared part of every setting's `Config`; grid axes are written over it."""
    return Config(
        data=data_config(sweep, task),
        net=NetConfig(N=sweep.N, seed=sweep.seeds.net),
        train=TrainConfig(
            n_epochs=sweep.n_epochs,
            lr=sweep.lr,
            log_int=sweep.log_int,
            save_int=sweep.save_int,
            seed=sweep.seeds.train,
            monitor_int=sweep.monitor_int,
            monitor_bins=sweep.monitor_bins,
            monitor_shuffles=sweep.monitor_shuffles,
        ),
        dtype=sweep.dtype,
        device=sweep.device,
    )


def _config_from(sweep: SweepConfig, axes: Mapping[str, Any]) -> Config:
    """The sweep's base `Config` for the task, with each axis value written over it."""
    task = axes.get(TASK_AXIS, sweep.task)
    if task not in SWEEP_TASKS:
        raise ValueError(f"grid axis {TASK_AXIS!r}: unknown task {task!r}; valid: {SWEEP_TASKS}")
    tree = _base_config(sweep, task).model_dump()
    for axis, value in axes.items():
        if axis != TASK_AXIS:  # already chosen above, together with the task's own D_in
            _set_leaf(tree, axis, value)
    return Config.model_validate(tree)


def _config_or_value_error(sweep: SweepConfig, axes: Mapping[str, Any], context: str) -> Config:
    """`_config_from`, with a pydantic ValidationError turned into a ValueError that starts
    with `context` (which axis or setting was being built)."""
    try:
        return _config_from(sweep, axes)
    except ValidationError as err:
        raise ValueError(f"{context}: {err}") from err


def _validate_axis(sweep: SweepConfig, axis: str, values: list) -> None:
    """One axis alone: sweep-owned fields are rejected, the values are non-empty, each builds
    a valid Config, and no two build the same one (so `[0, 0.0]` counts as a repeat)."""
    if axis in SWEEP_OWNED_AXES:
        raise ValueError(
            f"grid axis {axis!r} is set by the sweep's own {SWEEP_OWNED_AXES[axis]}; "
            "change that instead of sweeping it"
        )
    if not values:
        raise ValueError(f"grid axis {axis!r} has no values")
    configs = [
        _config_or_value_error(sweep, {axis: v}, f"grid axis {axis!r} value {v!r}") for v in values
    ]
    encoded = [cfg.model_dump_json() for cfg in configs]
    if len(set(encoded)) != len(encoded):
        raise ValueError(f"grid axis {axis!r} repeats a value: {values}")


def _validate_combinations(sweep: SweepConfig) -> None:
    """Every setting of the product builds a valid Config (axes can be fine alone and wrong
    together, e.g. more objects than grid cells)."""
    for setting in expand(sweep):
        context = f"grid setting {setting.index} {setting.axes}"
        _config_or_value_error(sweep, setting.axes, context)


def _validate_grid(sweep: SweepConfig, grid: dict[str, list]) -> None:
    """Axes first, one at a time, so an error names the culprit; then the combinations."""
    for axis, values in grid.items():
        _validate_axis(sweep, axis, values)
    _validate_combinations(sweep)


def _label_part(axis: str, value: Any) -> str:
    """`<last part of the axis path><value>` for a directory name, e.g. `rate_lambda1e-06`.
    Characters outside letters, digits and `._+-` become `-` so the name is a safe path
    component. The label is for people; uniqueness comes from the index prefix."""
    text = f"{value:g}" if isinstance(value, float) else str(value)
    return re.sub(r"[^A-Za-z0-9._+-]", "-", axis.rsplit(".", 1)[-1] + text)


def _indexed_name(index: int, axes: Mapping[str, Any]) -> str:
    """`s0007_L2_phitanh_...`: the index makes the name unique (`:g` would merge close floats),
    the label (the task is already the parent directory) is for people."""
    label = "_".join(_label_part(a, v) for a, v in axes.items() if a != TASK_AXIS)
    prefix = f"s{index:0{INDEX_WIDTH}d}"
    return f"{prefix}_{label}" if label else prefix


def _modularity_name(values: Mapping[str, Any]) -> str:
    """`:g` gives `r0` and `r0.001`, distinct per value of the modularity grid."""
    return f"L{values['net.L']}_{values['net.phi']}_r{values['train.rate_lambda']:g}"


def expand(sweep: SweepConfig) -> list[Setting]:
    """All settings: the Cartesian product of the grid axes in the order they are written, the
    first axis slowest. The order depends only on the grid, not on the sweep's sizes, so
    index i names the same setting for every size of the same grid."""
    grid = grid_of(sweep)
    n_settings = math.prod(len(values) for values in grid.values())
    if sweep.grid is not None and n_settings > 10**INDEX_WIDTH:
        raise SweepError(
            f"the grid has {n_settings} settings; directory names support at most "
            f"{10**INDEX_WIDTH} (INDEX_WIDTH={INDEX_WIDTH})"
        )
    built = built_values(sweep)
    settings = []
    for index, combo in enumerate(itertools.product(*(enumerate(v) for v in grid.values()))):
        axes = {axis: grid[axis][i] for axis, (i, _) in zip(grid, combo, strict=True)}
        # Names come from the built values, the same ones the grid hash uses, so two spellings
        # of one value (`1000000`, `1000000.0`) cannot hash alike yet name different directories.
        named = {axis: built[axis][i] for axis, (i, _) in zip(grid, combo, strict=True)}
        task = axes.get(TASK_AXIS, sweep.task)
        name = _modularity_name(named) if sweep.grid is None else _indexed_name(index, named)
        settings.append(Setting(index, axes, task, Path(task) / name))
    return settings


def get_setting(sweep: SweepConfig, index: int) -> Setting:
    settings = expand(sweep)
    if not 0 <= index < len(settings):
        raise SweepError(f"index {index} is out of range; valid indices are 0..{len(settings) - 1}")
    return settings[index]


def setting_dir(root: Path, setting: Setting) -> Path:
    return Path(root) / setting.relpath


def setting_config(sweep: SweepConfig, setting: Setting) -> Config:
    """The full fct `Config` of one setting."""
    return _config_from(sweep, setting.axes)


def _dump_at(dump: dict, axis: str) -> Any:
    node = dump
    for part in axis.split("."):
        node = node[part]
    return node


def built_values(sweep: SweepConfig) -> dict[str, list]:
    """Each axis's values as the built `Config` holds them (`0` becomes `0.0` for a float field),
    so two spellings of one value become one. This is the sameness `_validate_axis` enforces."""
    return {
        axis: [_dump_at(_config_from(sweep, {axis: v}).model_dump(), axis) for v in values]
        for axis, values in grid_of(sweep).items()
    }


def grid_hash(sweep: SweepConfig) -> str:
    """Short hash of what the grid sets: the built values of every axis. The base task counts
    only when `data.task` is not an axis. The sizes are not in it; config.json checks those."""
    base_task = None if TASK_AXIS in grid_of(sweep) else sweep.task
    canonical = json.dumps({"axes": built_values(sweep), "task": base_task})
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------- disk
def _train_hash(cfg: Config) -> str:
    return cfg.section_hash("data", "net", "train")


def setting_status(root: Path, sweep: SweepConfig, setting: Setting) -> Status:
    """done = train/final.pt exists and was written under this setting's current config (data,
    net, train sections and the loss version); stale = a final.pt written under another, or one
    that cannot be read;
    partial = a checkpoint but no final.pt; else missing. Only the hash of final.pt is read."""
    train_dir = cache_paths(setting_dir(root, setting)).train
    final = train_dir / "final.pt"
    if final.exists():
        current = _train_hash(setting_config(sweep, setting))
        try:
            return "done" if read_config_hash(final) == current else "stale"
        except ValueError:  # an unreadable final.pt cannot be reused, whatever config it had
            return "stale"
    return "partial" if latest_checkpoint(train_dir) is not None else "missing"


def _write_config_once(cfg: Config, directory: Path) -> None:
    """Write config.json for a new setting; fail if an existing one differs, because the
    artefacts beside it belong to that config."""
    path = cache_paths(directory).config
    if not path.exists():
        save_config(cfg, path)
    elif load_config(path) != cfg:
        raise SweepError(
            f"{path} differs from the config sweep.json builds for this setting; "
            "use a fresh ROOT or delete the setting directory"
        )


def _setting_record(sweep: SweepConfig, setting: Setting) -> dict:
    return {"index": setting.index, "grid_hash": grid_hash(sweep), "axes": setting.axes}


def _write_setting_once(sweep: SweepConfig, setting: Setting, directory: Path) -> None:
    """Record the setting's index, axis values and grid hash in setting.json; fail if an
    existing record differs, because the grid then no longer puts this setting here."""
    path = directory / SETTING_FILE
    record = _setting_record(sweep, setting)
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + "\n")
    elif json.loads(path.read_text()) != json.loads(json.dumps(record)):
        raise SweepError(
            f"{path} differs from what sweep.json gives this setting (another grid, or other "
            "axis values at this index); use a fresh ROOT or delete the setting directory"
        )


def _read_setting(directory: Path, current_hash: str) -> dict:
    record = json.loads((Path(directory) / SETTING_FILE).read_text())
    if record.get("grid_hash") != current_hash:
        raise SweepError(
            f"{directory} has no grid hash or was written under another grid than this "
            "sweep.json's; use a fresh ROOT or delete the directory"
        )
    return record


def read_setting(directory: Path, sweep: SweepConfig) -> dict:
    """The `{"index", "grid_hash", "axes"}` record `run_setting` wrote in a setting directory.
    SweepError, naming the directory, if it was written under another grid than `sweep`'s (or
    carries no grid hash)."""
    return _read_setting(directory, grid_hash(sweep))


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


def find_setting_dir(root: Path, sweep: SweepConfig, values: Mapping[str, Any]) -> Path:
    """The directory of the one already-run setting of `sweep` whose axes equal `values`; a
    subset of the axes is enough when it is unambiguous. LookupError if none or several match,
    or an axis is not one of the sweep's; SweepError if a directory was written under another
    grid. Only the sweep's own directories are considered, so strays in ROOT cannot match."""
    if unknown := sorted(set(values) - set(grid_of(sweep))):
        raise LookupError(f"unknown axis {unknown}; the grid has axes {list(grid_of(sweep))}")
    directories = [setting_dir(root, s) for s in expand(sweep)]
    run = [d for d in directories if (d / SETTING_FILE).exists()]
    if not run:
        raise LookupError(
            f"no setting of this sweep has been run under {root} (settings run before "
            f"{SETTING_FILE} existed get one on their next `sweep run`)"
        )
    current = grid_hash(sweep)  # once: it builds every setting's config
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


def run_setting(
    sweep: SweepConfig, index: int, root: Path, log: Callable[[str], None] = print
) -> None:
    """Train setting `index` into its cache directory; resumes a partial run, skips a done one.

    StaleCacheError if the setting was finished under another config or loss definition: its
    results would silently stand in for a run of the current one."""
    setting = get_setting(sweep, index)
    directory = setting_dir(root, setting)
    cfg = setting_config(sweep, setting)
    _write_config_once(cfg, directory)
    _write_setting_once(sweep, setting, directory)
    status = setting_status(root, sweep, setting)
    if status == "stale":
        raise StaleCacheError(
            f"setting {index}: {directory}/train/final.pt was written under a different config "
            "or loss definition than the current one, or is unreadable, so its results are not "
            "those of this sweep. To rerun it, delete the sweep ROOT or use a fresh one."
        )
    if status == "done":
        log(f"setting {index} ({directory}) already done")
        return
    log(f"setting {index}: {directory}")
    dataset = get_dataset(directory, cfg)
    train_from_config(cfg, dataset, directory, log=log, monitor=make_mir_monitor(dataset, cfg))
