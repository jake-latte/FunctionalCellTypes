"""The sweep: one place that defines the 36 settings and runs any one of them.

A `SweepConfig` (stored as ROOT/sweep.json) fixes everything that is shared by the settings
(sizes, optimiser, monitor, seeds, per-task data parameters); the grid itself (task x hidden
layers x phi x rate_lambda) is fixed here. `expand` lists the settings in a stable order, so
a SLURM array index and a local check name the same setting. Each setting is a normal fct
cache directory ROOT/<task>/L{L}_{phi}_r{rate}/ that `run_setting` fills by the same path as
`python -m fct train`: config.json, dataset, training with the MIR monitor.

This module is the I/O edge of the sweep: `expand`, `setting_config` and the config models are
pure; `run_setting`, `setting_status` and the sweep.json functions touch disk.
"""

import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal, NamedTuple, Self

from pydantic import Field, model_validator

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
# Grid axes. The order here is the order of `expand`: task is the slowest axis, rate_lambda the
# fastest, so appending a value to an axis would change later indices; do not reorder casually.
SWEEP_TASKS = ("linear_teacher", "object_action", "what_where")
LAYERS = (1, 3, 5)
PHIS = ("tanh", "relu")
RATE_LAMBDAS = (0.0, 1e-3)

Status = Literal["done", "partial", "missing", "stale"]


class SweepError(ValueError):
    """A sweep-level failure the user can fix: a bad index or a changed sweep config."""


class Setting(NamedTuple):
    task: str
    L: int
    phi: str
    rate_lambda: float


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
    """Shared settings of the 36 runs; see `preset` for the two ready-made sizes.

    The same data, net and training seeds are used for every setting, so settings differ only
    in the swept axes. `weight_lambda` is the library default (TrainConfig); the swept
    regulariser is `rate_lambda`. Task parameters are validated at load time by building each
    task's `DataConfig`, so an impossible combination fails when sweep.json is read.
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

    @model_validator(mode="after")
    def _task_parameters_are_consistent(self) -> Self:
        for task in SWEEP_TASKS:
            data_config(self, task)  # raises ValidationError naming the offending field
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


PRESETS: dict[str, SweepConfig] = {
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


def preset(name: str) -> SweepConfig:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; valid: {sorted(PRESETS)}")
    return PRESETS[name]


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
def expand(sweep: SweepConfig) -> list[Setting]:
    """All settings in a fixed order: task (slowest), layers, phi, rate_lambda (fastest).

    The grid does not depend on the sweep's sizes, so index i names the same setting for every
    preset.
    """
    return [Setting(*s) for s in itertools.product(SWEEP_TASKS, LAYERS, PHIS, RATE_LAMBDAS)]


def get_setting(sweep: SweepConfig, index: int) -> Setting:
    settings = expand(sweep)
    if not 0 <= index < len(settings):
        raise SweepError(f"index {index} is out of range; valid indices are 0..{len(settings) - 1}")
    return settings[index]


def setting_dir(root: Path, setting: Setting) -> Path:
    """ROOT/<task>/L{L}_{phi}_r{rate}; `:g` gives `r0` and `r0.001`, distinct per grid value."""
    name = f"L{setting.L}_{setting.phi}_r{setting.rate_lambda:g}"
    return Path(root) / setting.task / name


def setting_config(sweep: SweepConfig, setting: Setting) -> Config:
    """The full fct `Config` of one setting."""
    return Config(
        data=data_config(sweep, setting.task),
        net=NetConfig(N=sweep.N, L=setting.L, phi=setting.phi, seed=sweep.seeds.net),
        train=TrainConfig(
            n_epochs=sweep.n_epochs,
            lr=sweep.lr,
            rate_lambda=setting.rate_lambda,
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
