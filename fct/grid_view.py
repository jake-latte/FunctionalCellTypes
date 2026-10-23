"""From a sweep ROOT to the phase-transition figures: one chosen setting and the whole grid.

Three layers, as in `fct.report`:

- I/O: `load_setting_data` reads the net and dataset of one setting (its final.pt, or the latest
  checkpoint of one still running). The results themselves come from `report.load_results`.
- Analysis (pure): `snap_to_grid` and `choose_setting` turn requested values into the nearest
  grid setting; `grid_selection` assembles the (lambda_r, lambda_w, layer, MIR) records of one
  depth and activation; `layer_rates`, `neuron_mir_values` and `tuning_view` give the data of
  the tuning plot.
- Plotting: `fct.draw` draws; `tuning_items`, `matrix_items` and `ratio_items` assemble the
  items a notebook cell shows.

The sweep must have the four axes of `phase_transition` (depth, activation, lambda_r, lambda_w).
A setting still running is shown from its latest checkpoint and labelled "provisional"; a
setting with no usable data is a blank cell, never a made-up value.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from torch import Tensor

from fct.config import Config, resolve_dtype
from fct.data import get_dataset
from fct.draw import fig_mir_matrix, fig_mir_vs_ratio, fig_tuning
from fct.grid_report import MirRecord, mir_grid
from fct.model import MLP
from fct.modularity import layer_mir, neuron_mir
from fct.report import MIR_KINDS, Item, SettingResult, setting_label
from fct.sweep import load_sweep, same_value, setting_config, setting_dir
from fct.tasks import Dataset
from fct.train import load_latest_checkpoint, load_trained
from fct.tuning import ORDERS, TuningCurves, select_neurons, tuning_curves

DEPTH_AXIS = "net.L"
PHI_AXIS = "net.phi"
LAMBDA_R_AXIS = "train.rate_lambda"
LAMBDA_W_AXIS = "train.weight_lambda"
GRID_AXES = (DEPTH_AXIS, PHI_AXIS, LAMBDA_R_AXIS, LAMBDA_W_AXIS)
KINDS = ("raw", "corrected")  # which of the history's MIR the figures use


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


# ---------------------------------------------------------------------------- snapping
class Snapped(NamedTuple):
    requested: float
    value: float  # the grid value nearest to `requested`
    exact: bool


def snap_to_grid(value: float, grid: Sequence[float], log: bool = False) -> Snapped:
    """The grid value nearest to `value` (the first of two equally near ones).

    `log=True` measures distance in log10, the spacing of a lambda grid (0.1 is nearer 1.0
    than 0.001); it needs `value` and the whole grid positive. Otherwise distance is linear.
    """
    if len(grid) == 0:
        raise ValueError("grid is empty")
    if not math.isfinite(value):
        raise ValueError(f"value must be finite, got {value}")
    values = np.asarray(grid, dtype=float)
    target = float(value)
    if log:
        if target <= 0 or np.any(values <= 0):
            raise ValueError(
                f"a log-scale snap needs positive values; got {value} on grid {list(grid)} "
                "(non-positive)"
            )
        values, target = np.log10(values), math.log10(target)
    nearest = int(np.argmin(np.abs(values - target)))
    return Snapped(value, grid[nearest], same_value(grid[nearest], value))


def _unique(values: Sequence[Any]) -> list:
    out: list = []
    for v in values:
        if not any(same_value(v, u) for u in out):
            out.append(v)
    return out


def axis_values(results: Sequence[SettingResult]) -> dict[str, list]:
    """The values of each grid axis over all settings: numbers ascending, names in grid order.
    ValueError if the sweep lacks one of `GRID_AXES`."""
    out = {}
    for axis in GRID_AXES:
        if any(axis not in r.setting.axes for r in results):
            raise ValueError(
                f"the sweep has no axis {axis!r}; this view needs {GRID_AXES} (the "
                "phase_transition preset has them)"
            )
        values = _unique([r.setting.axes[axis] for r in results])
        numeric = all(isinstance(v, int | float) and not isinstance(v, bool) for v in values)
        out[axis] = sorted(values) if numeric else values
    return out


@dataclass(frozen=True)
class Choice:
    """The setting nearest a request, and how each requested value was snapped."""

    result: SettingResult
    snapped: dict[str, Snapped]


def choose_setting(
    results: Sequence[SettingResult], L: float, phi: str, lambda_r: float, lambda_w: float
) -> Choice:
    """The one setting at the grid values nearest `L`, `lambda_r` and `lambda_w` (lambdas on a
    log scale) with activation `phi` (exact). ValueError for an unknown `phi`; LookupError if
    the grid has axes beyond these four, so that the choice is not a single setting."""
    values = axis_values(results)
    if phi not in values[PHI_AXIS]:
        raise ValueError(f"phi {phi!r} is not one of {values[PHI_AXIS]}")
    snapped = {
        DEPTH_AXIS: snap_to_grid(L, values[DEPTH_AXIS]),
        PHI_AXIS: Snapped(phi, phi, True),
        LAMBDA_R_AXIS: snap_to_grid(lambda_r, values[LAMBDA_R_AXIS], log=True),
        LAMBDA_W_AXIS: snap_to_grid(lambda_w, values[LAMBDA_W_AXIS], log=True),
    }
    matches = [
        r
        for r in results
        if all(same_value(r.setting.axes[a], s.value) for a, s in snapped.items())
    ]
    if len(matches) != 1:
        raise LookupError(f"{len(matches)} settings match {[s.value for s in snapped.values()]}")
    return Choice(matches[0], snapped)


def snap_text(choice: Choice) -> str:
    """Which setting was selected, and every requested value that was snapped to the grid."""
    names = {DEPTH_AXIS: "L", PHI_AXIS: "phi", LAMBDA_R_AXIS: "lambda_r", LAMBDA_W_AXIS: "lambda_w"}
    moved = [
        f"{names[a]} {s.requested:g} snapped to {s.value:g}"
        for a, s in choice.snapped.items()
        if not s.exact
    ]
    head = f"Selected {setting_label(choice.result.setting)} ({choice.result.status})."
    return head + (" Nearest grid values: " + "; ".join(moved) + "." if moved else "")


# ---------------------------------------------------------------------------- global records
@dataclass(frozen=True)
class GridSelection:
    """MIR of every network of one depth and activation. `lambda_*_values` are the whole grid,
    so a setting without a record is a blank cell; `provisional` lists the (lambda_r, lambda_w)
    of the records read from a checkpoint of a run still going."""

    records: list[MirRecord]
    lambda_r_values: list[float]
    lambda_w_values: list[float]
    n_layers: int
    provisional: list[tuple[float, float]]
    n_settings: int  # settings of this depth and activation, with or without a record


def _last_mir(result: SettingResult, kind: str, depth: int) -> list[float] | None:
    """MIR per layer from the last monitor call, None for a setting that has none yet (not
    started, or a checkpoint before its first call). A finished setting without any raises."""
    history = result.history
    key = MIR_KINDS[kind]
    if history is None or not history.get(key):
        if result.status == "done":
            raise ValueError(
                f"{setting_label(result.setting)}: no monitor entries {key!r}; "
                "was the sweep run with monitor_int > 0?"
            )
        return None
    last = list(history[key][-1])
    if len(last) != depth:
        raise ValueError(
            f"{setting_label(result.setting)}: history has {len(last)} layers, expected {depth}"
        )
    return last


def grid_selection(
    results: Sequence[SettingResult], depth: int, phi: str, kind: str
) -> GridSelection:
    """The MIR records of the settings with `depth` hidden layers and activation `phi`.

    `kind` is "raw" or "corrected" (shuffle-bias-corrected). A finished setting gives its final
    MIR, a partial one the last monitor call of its checkpoint (listed in `provisional`); a
    setting with no MIR yet gives no record. ValueError if no setting has this depth and
    activation."""
    _check_kind(kind)
    values = axis_values(results)
    chosen = [
        r
        for r in results
        if same_value(r.setting.axes[DEPTH_AXIS], depth) and r.setting.axes[PHI_AXIS] == phi
    ]
    if not chosen:
        raise ValueError(
            f"no setting has L={depth} and phi={phi!r}; "
            f"depths {values[DEPTH_AXIS]}, activations {values[PHI_AXIS]}"
        )
    records, provisional = [], []
    for r in chosen:
        last = _last_mir(r, kind, depth)
        if last is None:
            continue
        lr, lw = r.setting.axes[LAMBDA_R_AXIS], r.setting.axes[LAMBDA_W_AXIS]
        records += [MirRecord(lr, lw, layer, mir) for layer, mir in enumerate(last)]
        if r.provisional:
            provisional.append((lr, lw))
    return GridSelection(
        records,
        values[LAMBDA_R_AXIS],
        values[LAMBDA_W_AXIS],
        depth,
        provisional,
        len(chosen),
    )


# ---------------------------------------------------------------------------- tuning
def layer_rates(net: MLP, x: Tensor, layer: int) -> Tensor:
    """(P, N) firing rates phi(H) of hidden layer `layer` (0-based): the rates the MIR is
    computed on. ValueError for a layer the net does not have."""
    if not 0 <= layer < net.L:
        raise ValueError(f"layer {layer} is outside the net's {net.L} hidden layer(s)")
    with torch.no_grad():
        return net.phi(net(x)[0][:, layer])


def factor_matrix(factors: Mapping[str, Tensor]) -> tuple[list[str], Tensor]:
    """Names and the (P, K) matrix of the latents. `tuning_curves` bins all latents alike, so
    all must be integer (categorical) or all floating; a mix is an error."""
    kinds = {f.is_floating_point() for f in factors.values()}
    if len(kinds) != 1:
        raise ValueError(
            f"factors {list(factors)} mix integer and floating latents; tuning curves need one kind"
        )
    return list(factors), torch.stack(list(factors.values()), dim=1)


def neuron_mir_values(
    rates: Tensor,
    factors: dict[str, Tensor],
    groups: dict[str, list[str]] | None,
    n_bins: int,
    kind: str,
    n_shuffles: int,
    seed: int,
) -> np.ndarray:
    """(N,) MIR of each neuron of one layer, whose mean is the layer's MIR: raw, or with the
    shuffle bias correction, with the monitor's bins, shuffles and seed."""
    _check_kind(kind)
    bias = "shuffle" if kind == "corrected" else "none"
    layer = layer_mir(rates, factors, groups, n_bins, bias, n_shuffles, seed)
    return neuron_mir(layer.r, len(layer.group_names)).numpy()


class TuningView(NamedTuple):
    curves: TuningCurves
    neuron_mir: np.ndarray  # (N,)
    latent_names: list[str]


def tuning_view(
    net: MLP, dataset: Dataset, layer: int, n_bins: int, kind: str, n_shuffles: int, seed: int
) -> TuningView:
    """Tuning curves of every neuron of `layer` to each latent, with each neuron's MIR."""
    if dataset.factors is None:
        raise ValueError("the dataset has no factors, so no latents to tune to")
    rates = layer_rates(net, dataset.x, layer)
    names, matrix = factor_matrix(dataset.factors)
    mir = neuron_mir_values(rates, dataset.factors, dataset.groups, n_bins, kind, n_shuffles, seed)
    return TuningView(tuning_curves(rates, matrix, n_bins), mir, names)


# ---------------------------------------------------------------------------- I/O
class SettingData(NamedTuple):
    net: MLP
    dataset: Dataset
    cfg: Config  # on the CPU


def load_setting_data(root: Path, result: SettingResult) -> SettingData:
    """The net and dataset of a setting that has data: a done one from final.pt, a partial one
    from its latest checkpoint. ValueError for one without (see `result.status`). The dataset
    file exists whenever `result` has data, so nothing is generated."""
    if result.history is None:
        raise ValueError(f"{setting_label(result.setting)} has no data ({result.status})")
    root = Path(root)
    cfg = setting_config(load_sweep(root), result.setting).model_copy(update={"device": "cpu"})
    directory = setting_dir(root, result.setting)
    cpu, dtype = torch.device("cpu"), resolve_dtype(cfg.dtype)
    if result.status == "done":
        net, _ = load_trained(directory, cfg, cpu, dtype)
    else:
        net, _, _ = load_latest_checkpoint(directory, cfg, cpu, dtype)
    return SettingData(net, get_dataset(directory, cfg), cfg)


# ---------------------------------------------------------------------------- items
_PROVISIONAL = "provisional (still running, shown from the latest checkpoint)"


def tuning_items(
    root: Path,
    choice: Choice,
    layer: int,
    n_neurons: int,
    order: str,
    seed: int,
    kind: str,
) -> list[Item]:
    """Prose and the tuning figure of `n_neurons` neurons of hidden layer `layer` (0-based) of
    the chosen setting, picked by `order` ("random", "mir_ascending", "mir_descending"; `seed`
    is for random). A setting with no data gives just a message."""
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}, got {order!r}")
    _check_kind(kind)
    result = choice.result
    label = f"{result.setting.task} {setting_label(result.setting)}"
    if result.history is None:
        return [f"**{label}: {result.status}**: no data, so no tuning curves."]
    data = load_setting_data(root, result)
    tc = data.cfg.train
    view = tuning_view(
        data.net, data.dataset, layer, tc.monitor_bins, kind, tc.monitor_shuffles, tc.seed
    )
    chosen = select_neurons(view.neuron_mir, n_neurons, order, seed)
    title = f"{label}, layer {layer + 1}: tuning of {len(chosen)} neurons ({order}, {kind} MIR)"
    note = ""
    if result.provisional:
        note = f" This is provisional: the net is the latest checkpoint ({result.progress})."
        title += f" - provisional, {result.progress}"
    text = (
        f"**{label}**, layer {layer + 1}: mean firing rate (phi) of {len(chosen)} neurons "
        f"chosen {order}, against each latent (other latents averaged out). Layer mean "
        f"{kind} MIR {view.neuron_mir.mean():.3f}.{note}"
    )
    fig = fig_tuning(view.curves, chosen, view.neuron_mir, view.latent_names)
    fig.suptitle(title)
    return [text, fig]


def _global_title(kind: str, depth: int, phi: str, sel: GridSelection) -> str:
    title = f"{kind} MIR of every network, L={depth}, {phi}"
    return title + (
        f" - provisional ({len(sel.provisional)} from checkpoints)" if sel.provisional else ""
    )


def _global_text(kind: str, depth: int, phi: str, sel: GridSelection) -> str:
    n_cells = len({(r.lambda_r, r.lambda_w) for r in sel.records})
    text = (
        f"L={depth}, {phi}: {kind} layer-average MIR of {n_cells} of {sel.n_settings} networks; "
        "the rest have none yet (blank in the matrix)."
    )
    if sel.provisional:
        text += f" Of these, {len(sel.provisional)} are {_PROVISIONAL}."
    return text


def matrix_items(results: Sequence[SettingResult], depth: int, phi: str, kind: str) -> list[Item]:
    """Prose and one heat-map per hidden layer, lambda_r against lambda_w, shared colour scale."""
    sel = grid_selection(results, depth, phi, kind)
    if not sel.records:
        return [f"L={depth}, {phi}: no MIR in any network yet, so no matrix."]
    grid = mir_grid(sel.records, sel.lambda_r_values, sel.lambda_w_values, sel.n_layers)
    fig = fig_mir_matrix(sel.lambda_r_values, sel.lambda_w_values, grid)
    fig.suptitle(_global_title(kind, depth, phi, sel))
    return [_global_text(kind, depth, phi, sel), fig]


def ratio_items(results: Sequence[SettingResult], depth: int, phi: str, kind: str) -> list[Item]:
    """Prose and the scatter of layer-average MIR against lambda_r / lambda_w."""
    sel = grid_selection(results, depth, phi, kind)
    if not sel.records:
        return [f"L={depth}, {phi}: no MIR in any network yet, so no scatter."]
    fig = fig_mir_vs_ratio(sel.records)
    fig.suptitle(_global_title(kind, depth, phi, sel))
    return [_global_text(kind, depth, phi, sel), fig]
