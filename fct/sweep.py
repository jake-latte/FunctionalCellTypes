"""The sweep: one place that defines the settings of a grid and runs any one of them.

A `SweepConfig` (stored as ROOT/sweep.json) fixes everything that is shared by the settings
(sizes, optimiser, monitor, seeds, per-task data parameters) and, in `grid`, the axes that
vary: each axis is a dotted `Config` path (`net.L`, `train.rate_lambda`, ...) with an explicit
list of values. `expand` lists the Cartesian product in a stable order, so a SLURM array index
and a local check name the same setting. Each setting is a normal fct cache directory that
`run_setting` fills by the same path as `python -m fct train`: config.json, dataset, training
with the MIR monitor.

A sweep.json without `grid` is the fixed 36-setting modularity grid (task x layers x phi x
rate_lambda), kept with its original directory names ROOT/<task>/L{L}_{phi}_r{rate}/. A sweep
with a `grid` uses ROOT/<task>/s{index:04d}_<label>/ and records the axis values in
setting.json beside config.json.

This module is the I/O edge of the sweep: `expand`, `setting_config` and the config models are
pure; `run_setting`, `setting_status`, `find_setting_dir` and the sweep.json functions touch disk.
"""

import itertools
import json
import math
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

Status = Literal["done", "partial", "missing", "stale"]


class SweepError(ValueError):
    """A sweep-level failure the user can fix: a bad index or a changed sweep config."""


@dataclass(frozen=True)
class Setting:
    """One point of the grid: its position, the value of every axis, and its directory.

    `axes` keeps the grid's axis order. `task` is the data task the setting trains on: the
    `data.task` axis if there is one, else the sweep's base task. `relpath` is the setting's
    directory relative to the sweep ROOT. `L`, `phi` and `rate_lambda` read the matching axes
    of a grid that has them (all of the modularity grid).
    """

    index: int
    axes: tuple[tuple[str, Any], ...]
    task: str
    relpath: Path

    def __hash__(
        self,
    ) -> int:  # axis values may be unhashable (lists); index identifies a grid point
        return hash((self.index, self.relpath))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.axes)

    def __getitem__(self, axis: str) -> Any:
        values = self.as_dict()
        if axis not in values:
            raise KeyError(f"setting {self.index} has no axis {axis!r}; axes: {list(values)}")
        return values[axis]

    @property
    def L(self) -> int:
        return self["net.L"]

    @property
    def phi(self) -> str:
        return self["net.phi"]

    @property
    def rate_lambda(self) -> float:
        return self["train.rate_lambda"]


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
    is the library default unless it is an axis); an axis overrides the base value. `grid` maps
    a dotted `Config` path to the list of values that axis takes; None means the fixed
    modularity grid. Task parameters and the whole grid are validated at load time by building
    every setting's `Config`, so a bad axis name, value or combination fails when sweep.json is
    read, naming the axis.
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
    task: Literal["linear_teacher", "object_action", "what_where"] = (
        "linear_teacher"  # unless `data.task` is an axis
    )
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
PRESET_NAMES = ("tiny", "full", "modularity", "phase_transition")


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


def preset(name: str, n_lambda: int | None = None) -> SweepConfig:
    """A ready-made sweep. `tiny` and `full` are two sizes of the modularity grid (`modularity`
    is `full`); `phase_transition` needs `n_lambda`, the others reject it."""
    if name not in PRESET_NAMES:
        raise ValueError(f"unknown preset {name!r}; valid: {sorted(PRESET_NAMES)}")
    if name == "phase_transition":
        if n_lambda is None:
            raise ValueError("preset 'phase_transition' needs n_lambda (points per lambda axis)")
        return phase_transition(n_lambda)
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
def _grid(sweep: SweepConfig) -> dict[str, list]:
    return MODULARITY_GRID if sweep.grid is None else sweep.grid


def _set_leaf(tree: dict, axis: str, value: Any) -> None:
    """Set the Config field at the dotted path `axis` in a dumped config tree."""
    *parents, last = axis.split(".")
    node = tree
    for part in parents:
        if not isinstance(node.get(part), dict):
            raise ValueError(f"grid axis {axis!r}: {part!r} is not a config section")  # noqa: TRY004 - a bad axis name is a bad value
        node = node[part]
    if last not in node:
        raise ValueError(f"grid axis {axis!r}: no config field {last!r}")
    if isinstance(node[last], dict):
        raise ValueError(f"grid axis {axis!r} is a config section; name one of its fields")  # noqa: TRY004
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


def _build(sweep: SweepConfig, axes: Mapping[str, Any], what: str) -> Config:
    try:
        return _config_from(sweep, axes)
    except ValidationError as err:
        raise ValueError(f"{what}: {err}") from err


def _validate_grid(sweep: SweepConfig, grid: dict[str, list]) -> None:
    """Every axis is a Config field, takes distinct valid values, and every combination builds
    a valid Config. Each axis is checked alone first so the error names the culprit."""
    for axis, values in grid.items():
        if not values:
            raise ValueError(f"grid axis {axis!r} has no values")
        encoded = [json.dumps(v, sort_keys=True) for v in values]
        if len(set(encoded)) != len(encoded):
            raise ValueError(f"grid axis {axis!r} repeats a value: {values}")
        for value in values:
            _build(sweep, {axis: value}, f"grid axis {axis!r} value {value!r}")
    for setting in expand(sweep):
        _build(sweep, setting.as_dict(), f"grid setting {setting.index} {setting.as_dict()}")


def _label_part(axis: str, value: Any) -> str:
    text = f"{value:g}" if isinstance(value, float) else str(value)
    return re.sub(r"[^A-Za-z0-9.+-]", "-", axis.rsplit(".", 1)[-1] + text)


def _indexed_name(index: int, axes: tuple[tuple[str, Any], ...]) -> str:
    """`s0007_L2_phitanh_...`: the index makes the name unique (`:g` would merge close floats),
    the label (the task is already the parent directory) is for people."""
    label = "_".join(_label_part(a, v) for a, v in axes if a != TASK_AXIS)
    return f"s{index:04d}_{label}" if label else f"s{index:04d}"


def _modularity_name(values: Mapping[str, Any]) -> str:
    """`:g` gives `r0` and `r0.001`, distinct per value of the fixed grid."""
    return f"L{values['net.L']}_{values['net.phi']}_r{values['train.rate_lambda']:g}"


def expand(sweep: SweepConfig) -> list[Setting]:
    """All settings: the Cartesian product of the grid axes in the order they are written, the
    first axis slowest. The order depends only on the grid, not on the sweep's sizes, so
    index i names the same setting for every size of the same grid."""
    grid = _grid(sweep)
    settings = []
    for index, combo in enumerate(itertools.product(*grid.values())):
        axes = tuple(zip(grid, combo, strict=True))
        task = dict(axes).get(TASK_AXIS, sweep.task)
        name = _modularity_name(dict(axes)) if sweep.grid is None else _indexed_name(index, axes)
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
    return _config_from(sweep, setting.as_dict())


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


def _setting_record(setting: Setting) -> dict:
    return {"index": setting.index, "axes": setting.as_dict()}


def _write_setting_once(setting: Setting, directory: Path) -> None:
    """Record the setting's axis values in setting.json; fail if an existing record differs,
    because the grid then no longer puts this setting at this index."""
    path = directory / SETTING_FILE
    record = _setting_record(setting)
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + "\n")
    elif json.loads(path.read_text()) != json.loads(json.dumps(record)):
        raise SweepError(
            f"{path} differs from the axis values sweep.json gives this setting; "
            "use a fresh ROOT or delete the setting directory"
        )


def read_setting(directory: Path) -> dict:
    """The `{"index", "axes"}` record that `run_setting` wrote in a setting directory."""
    return json.loads((Path(directory) / SETTING_FILE).read_text())


def _same_value(a: Any, b: Any) -> bool:
    """Equality, with numbers compared to a relative 1e-9 so a float typed in a notebook finds
    the one that was written to JSON."""
    if _is_number(a) and _is_number(b):
        return math.isclose(a, b, rel_tol=1e-9)
    return a == b


def _is_number(x: Any) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool)


def find_setting_dir(root: Path, values: Mapping[str, Any]) -> Path:
    """The directory of the one setting (already run) whose axes equal `values`; a subset of
    the axes is enough when it is unambiguous. LookupError if none or several match, or if an
    axis name is not one the settings record."""
    records = [
        (p.parent, read_setting(p.parent)) for p in sorted(Path(root).glob(f"*/*/{SETTING_FILE}"))
    ]
    if not records:
        raise LookupError(f"no {SETTING_FILE} under {root}; no setting has been run there")
    known = {axis for _, record in records for axis in record["axes"]}
    if unknown := sorted(set(values) - known):
        raise LookupError(f"unknown axis {unknown}; the settings have axes {sorted(known)}")
    matches = [
        directory
        for directory, record in records
        if all(_same_value(record["axes"].get(a), v) for a, v in values.items())
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
    _write_setting_once(setting, directory)
    status = setting_status(root, sweep, setting)
    if status == "stale":
        raise StaleCacheError(
            f"setting {index}: {directory}/train/final.pt was written under a different config "
            "or loss definition than the current one, or is unreadable, so its results are not those of this "
            "sweep. To rerun it, delete the sweep ROOT or use a fresh one."
        )
    if status == "done":
        log(f"setting {index} ({directory}) already done")
        return
    log(f"setting {index}: {directory}")
    dataset = get_dataset(directory, cfg)
    train_from_config(cfg, dataset, directory, log=log, monitor=make_mir_monitor(dataset, cfg))
