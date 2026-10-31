"""One setting of a spatial sweep, for the spatial-grid notebook: pick it by its axis values,
load its trained net, its training set and the held-out latent-lattice test set.

The sweep's own functions find and train the settings; this module only joins them for a
notebook. A grid setting that was never trained is a status, not an error, so the notebook can
say so in the cell; a setting that is not in the grid at all (a typo) is a LookupError.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

from fct.config import Config, resolve_device, resolve_dtype
from fct.data import Dataset, get_dataset, get_latent_lattice_dataset
from fct.spatial_view import TrainRun
from fct.sweep import (
    SETTING_FILE,
    Setting,
    Status,
    SweepConfig,
    expand,
    find_setting_dir,
    grid_of,
    load_sweep,
    same_value,
    setting_config,
    setting_dir,
    setting_status,
    setting_statuses,
)
from fct.train import load_trained, select_member


class GridSetting(NamedTuple):
    setting: Setting
    directory: Path
    cfg: Config  # the setting's config with the device `select_setting` was given
    status: Status  # "done" is the only status with a trained net to show


def _matching_settings(sweep: SweepConfig, values: Mapping[str, Any]) -> list[Setting]:
    """The settings of the grid whose axes equal `values`, run or not."""
    if unknown := sorted(set(values) - set(grid_of(sweep))):
        raise LookupError(f"unknown axis {unknown}; the grid has axes {list(grid_of(sweep))}")
    return [s for s in expand(sweep) if all(same_value(s.axes[a], v) for a, v in values.items())]


def select_setting(root: Path, values: Mapping[str, Any], device: str = "cpu") -> GridSetting:
    """The one setting of the sweep at `root` whose axes equal `values` (a subset of the axes is
    enough when it picks one), with its status. LookupError for an axis that is not in the grid
    and for no match or several, whether or not they were run, so an ambiguous request never
    resolves to a setting the caller did not name. A setting of the grid that was never run is
    returned with status "missing". A run setting's directory is confirmed by `find_setting_dir`
    (SweepError if it was written under another grid)."""
    root = Path(root)
    sweep = load_sweep(root)
    matches = _matching_settings(sweep, values)
    if len(matches) != 1:
        raise LookupError(
            f"{len(matches)} settings of the grid match {dict(values)}; give axes that pick one"
        )
    (setting,) = matches
    directory = setting_dir(root, setting)
    if (directory / SETTING_FILE).exists():
        directory = find_setting_dir(root, sweep, setting.axes)
    # Config validates the device name, unlike a model_copy update.
    cfg = Config.model_validate({**setting_config(sweep, setting).model_dump(), "device": device})
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
    for i, setting, status in setting_statuses(root, sweep):
        cells = [str(i), *(str(setting.axes[a]) for a in axes), status]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def load_grid_run(chosen: GridSetting, member: int | None = None) -> tuple[TrainRun, Dataset]:
    """The trained run of `chosen` and its test set: the 3^k latent-lattice points of the same
    teacher. ValueError if the setting is not trained. For an ensemble (net.B > 1) `member`
    picks the network (and its history) and is required (`fct.train.select_member`)."""
    if chosen.status != "done":
        raise ValueError(f"setting {chosen.setting.axes} is not trained ({chosen.status})")
    cfg = chosen.cfg
    net, history = select_member(
        *load_trained(chosen.directory, cfg, resolve_device(cfg), resolve_dtype(cfg.dtype)),
        member,
    )
    run = TrainRun(chosen.directory, get_dataset(chosen.directory, cfg), net, history)
    test = get_latent_lattice_dataset(chosen.directory, cfg)
    if len(test.x) != 3**cfg.data.n_latents:  # one point per sign pattern
        raise ValueError(f"the test set has {len(test.x)} points, not 3^{cfg.data.n_latents}")
    return run, test
