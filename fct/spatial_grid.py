"""One setting of a spatial sweep, for the spatial-grid notebook: pick it by its axis values,
load its trained net, its training set and the held-out latent-lattice test set.

The sweep's own functions find and train the settings; this module only joins them for a
notebook. A grid setting that was never trained is a status, not an error, so the notebook can
say so in the cell; a setting that is not in the grid at all (a typo) is a LookupError.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

from fct.config import Config, TrainConfig, resolve_device, resolve_dtype
from fct.data import Dataset, get_dataset, get_latent_lattice_dataset
from fct.spatial_view import TrainRun
from fct.sweep import (
    SETTING_FILE,
    Setting,
    Status,
    expand,
    find_setting_dir,
    grid_of,
    load_sweep,
    same_value,
    setting_config,
    setting_dir,
    setting_status,
)
from fct.train import load_trained

# The test set has one point per element of {-a, 0, a}^k (`latent_lattice_dataset`).
TEST_POINTS_BASE = 3


class GridSetting(NamedTuple):
    setting: Setting
    directory: Path
    cfg: Config  # the setting's config with the device `select_setting` was given
    status: Status  # "done" is the only status with a trained net to show


def _unrun_matches(root: Path, values: Mapping[str, Any]) -> list[Setting]:
    """Settings of the grid whose axes equal `values` and that have never been run."""
    return [
        s
        for s in expand(load_sweep(root))
        if all(a in s.axes and same_value(s.axes[a], v) for a, v in values.items())
        and not (setting_dir(root, s) / SETTING_FILE).exists()
    ]


def select_setting(root: Path, values: Mapping[str, Any], device: str = "cpu") -> GridSetting:
    """The setting of the sweep at `root` whose axes equal `values` (a subset is enough when it
    is unambiguous), with its status. Found through `find_setting_dir`; a setting of the grid
    that was never run is returned with status "missing". LookupError for values or axes that
    are not in the grid, or that match several settings."""
    root = Path(root)
    sweep = load_sweep(root)
    try:
        directory = find_setting_dir(root, sweep, values)
    except LookupError:
        unrun = _unrun_matches(root, values)
        if len(unrun) != 1:
            raise
        directory = setting_dir(root, unrun[0])
    (setting,) = (s for s in expand(sweep) if setting_dir(root, s) == directory)
    cfg = setting_config(sweep, setting).model_copy(update={"device": device})
    return GridSetting(setting, directory, cfg, setting_status(root, sweep, setting))


def settings_table(root: Path) -> str:
    """A markdown table of the grid: index, the value of every axis, and the training status."""
    root = Path(root)
    sweep = load_sweep(root)
    axes = list(grid_of(sweep))
    rows = [
        "| " + " | ".join(["index", *axes, "status"]) + " |",
        "|" + "---|" * (len(axes) + 2),
    ]
    for i, s in enumerate(expand(sweep)):
        cells = [str(i), *(str(s.axes[a]) for a in axes), setting_status(root, sweep, s)]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def load_grid_run(chosen: GridSetting) -> tuple[TrainRun, Dataset]:
    """The trained run of `chosen` and its test set: the 3^k latent-lattice points of the same
    teacher. ValueError if the setting is not trained."""
    if chosen.status != "done":
        raise ValueError(f"setting {chosen.setting.axes} is not trained ({chosen.status})")
    cfg = chosen.cfg
    net, history = load_trained(
        chosen.directory, cfg, resolve_device(cfg), resolve_dtype(cfg.dtype)
    )
    run = TrainRun(chosen.directory, get_dataset(chosen.directory, cfg), net, history)
    return run, get_latent_lattice_dataset(chosen.directory, cfg)


def prior_coeff(train: TrainConfig) -> float:
    """The loss coefficient of the weight prior, lambda_w / (2 beta): beta is `train.beta`
    under SGLD and 1 under Adam. The spatial notebook's `PRIOR_COEFF`."""
    beta = train.beta if train.optimizer == "sgld" else 1.0
    return train.weight_lambda / (2 * beta)
