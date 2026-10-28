"""Everything the results notebook shows, so the notebook itself holds no logic.

Three layers, as in the rest of the package:

- I/O: `load_results` reads a sweep ROOT into one `SettingResult` per grid setting. A finished
  setting is read from final.pt; one still running from its latest checkpoint, and is then
  *provisional*: the same data, plus the epoch reached. A setting with no checkpoint, or whose
  checkpoint or dataset file cannot be used, is returned with its status (and a note) and no
  data, so a partial sweep still loads.
- Analysis (pure): `final_mir` collects the last MIR of every layer of a task's settings into
  arrays (provisional rows are flagged, and `FinalMIR.finished_only` and `summary_text` leave
  them out of every statistic); `mir_table_markdown`, `summary_text` and `setting_text` turn
  numbers into prose, so no claim about the results is ever typed by hand.
- Plotting: the figures come from `fct.draw`; `setting_items` and `task_items` assemble them.
  `show` is the only function that touches IPython.

The report supports the modularity grid (settings varying task, net.L, net.phi and
train.rate_lambda; other grids may add axes, as long as these exist): `load_results` raises on a
sweep without them, and `final_mir` and the figures assume a layer count of at most `max(LAYERS)`.

Loading is read-only: nothing is generated, cached or written under ROOT, which a running job
may be writing to. Settings are loaded on the CPU whatever device they were trained on, and are
validated against the config rebuilt from sweep.json (a stale final.pt raises, it is not shown;
a stale or unreadable checkpoint of a running setting is reported in its note).
"""

import dataclasses
import os
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from fct.analyse import ExampleOutputs, example_outputs, pick_examples, prediction_report
from fct.cache import cache_paths
from fct.config import Config, DataConfig, resolve_dtype
from fct.data import get_dataset
from fct.draw import (
    PROVISIONAL,
    fig_example_outputs,
    fig_loss_curves,
    fig_mir_heatmaps,
    fig_mir_history,
)
from fct.model import MLP
from fct.sweep import (
    LAYERS,
    MODULARITY_AXES,
    TASK_AXIS,
    Setting,
    Status,
    SweepConfig,
    expand,
    grid_of,
    load_sweep,
    same_value,
    setting_config,
    setting_dir,
    setting_status,
)
from fct.train import History, load_latest_checkpoint, load_trained

ROOT_ENV = "FCT_SWEEP_ROOT"
N_EXAMPLES = 6  # examples loaded per setting; fig_example_outputs draws up to its per-task default
EXAMPLE_SEED = 0
MIR_KINDS = {
    "raw": "monitor_mir",
    "corrected": "monitor_mir_corrected",
    "chance": "monitor_mir_chance",
}
_PANEL_TITLES = {"raw": "raw", "corrected": "corrected", "chance": "chance baseline"}
N_TIES_LISTED = 3
REQUIRED_HISTORY = ("epoch", "loss", "mse", "monitor_epoch", "monitor_n_dead_corrected")

Item = str | Figure  # what a notebook cell shows: markdown text or a figure


@dataclass(frozen=True)
class SettingResult:
    """One grid setting; `directory` is relative to ROOT. `history`, `examples`, `metrics`,
    `data_cfg`, `epoch` and `n_epochs` are None unless there is something to show: a done
    setting, or a partial one whose latest checkpoint was read (then `provisional`). `epoch` is
    the epochs completed (n_epochs for a done setting), `n_epochs` the sweep's total. `note`
    explains a setting that is not shown beyond what its status says (a dataset file that is
    missing, a checkpoint that cannot be used)."""

    setting: Setting
    status: Status
    directory: Path
    note: str | None = None
    history: dict | None = None
    examples: ExampleOutputs | None = None
    metrics: dict[str, float] | None = None
    data_cfg: DataConfig | None = None
    epoch: int | None = None
    n_epochs: int | None = None

    @property
    def provisional(self) -> bool:
        """Still running, with data read from its latest checkpoint: numbers are not final."""
        return self.status == "partial" and self.history is not None

    @property
    def progress(self) -> str:
        """ "epoch X of N" of a setting that has data; ValueError if it has none."""
        if self.epoch is None:
            raise ValueError(f"{setting_label(self.setting)} has no data, so no progress")
        return f"epoch {self.epoch} of {self.n_epochs}"

    @property
    def mir_epoch(self) -> int | None:
        """Epoch of the last monitor call in the history, None if there is none yet. A
        checkpoint at epoch s never holds the monitor call of epoch s, so this lags `epoch`
        by up to `monitor_int` epochs."""
        monitored = (self.history or {}).get("monitor_epoch")
        return monitored[-1] if monitored else None


@dataclass(frozen=True)
class FinalMIR:
    """Final MIR per layer of one task's settings, rows in grid order. Arrays are
    (n_settings, max layers); a layer the setting does not have, and every layer of a setting
    with no data, is NaN. A provisional setting's row holds the MIR of its last monitor call so
    far (NaN if there is none yet); `mir_epoch` is the epoch of that call and `epoch` /
    `n_epochs` the checkpoint's progress. All three are None for a setting that is not
    provisional.

    The arrays include provisional rows (the table and heatmap show them, marked): anything
    that computes a statistic must use `finished_only()`."""

    task: str
    labels: list[str]
    status: list[Status]
    mir_epoch: list[int | None]
    epoch: list[int | None]
    n_epochs: list[int | None]
    raw: np.ndarray
    corrected: np.ndarray
    chance: np.ndarray

    @property
    def provisional(self) -> list[bool]:
        return [
            s == "partial" and e is not None for s, e in zip(self.status, self.epoch, strict=True)
        ]

    def array(self, which: str) -> np.ndarray:
        if which not in MIR_KINDS:
            raise ValueError(f"which must be one of {sorted(MIR_KINDS)}, got {which!r}")
        return getattr(self, which)

    def finished_only(self) -> "FinalMIR":
        """A copy with the rows of provisional settings blanked (NaN), so no statistic over
        it can include them."""
        blank = np.array(self.provisional)[:, None]
        return dataclasses.replace(
            self, **{kind: np.where(blank, np.nan, self.array(kind)) for kind in MIR_KINDS}
        )

    def display_labels(self) -> list[str]:
        """Row labels; a provisional row is marked `*` with where its numbers come from."""
        return [
            f"{label} * ({_origin(m, e, n, short=True)})" if prov else label
            for label, prov, m, e, n in zip(
                self.labels,
                self.provisional,
                self.mir_epoch,
                self.epoch,
                self.n_epochs,
                strict=True,
            )
        ]

    def row_status(self) -> list[str]:
        """Status per row as `fct.draw` names it: provisional rows are not just "partial"."""
        return [
            PROVISIONAL if prov else s
            for s, prov in zip(self.status, self.provisional, strict=True)
        ]


def _origin(mir_epoch: int | None, epoch: int, n_epochs: int, short: bool = False) -> str:
    """Where a provisional row's MIR comes from: the epoch of its last monitor call, which is
    not the checkpoint's epoch."""
    if short:
        measured = f"MIR@{mir_epoch}" if mir_epoch is not None else "no MIR"
        return f"{measured}, ckpt {epoch}/{n_epochs}"
    measured = f"MIR from epoch {mir_epoch}" if mir_epoch is not None else "no MIR yet"
    return f"{measured}; checkpoint at epoch {epoch} of {n_epochs}"


# How the modularity axes read in a label: `L3 relu r=0.001`.
_LABEL_FORMATS = {"net.L": "L{}", "net.phi": "{}", "train.rate_lambda": "r={}"}


def _label_value(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def setting_label(setting: Setting) -> str:
    """Every axis except the task, in grid order, so two settings of a grid never share a label."""
    parts = []
    for axis, value in setting.axes.items():
        if axis == TASK_AXIS:
            continue
        template = _LABEL_FORMATS.get(axis, axis.rsplit(".", 1)[-1] + "={}")
        parts.append(template.format(_label_value(value)))
    return " ".join(parts)


# ---------------------------------------------------------------------------- I/O
def sweep_root_from_env() -> Path:
    """The sweep ROOT named by $FCT_SWEEP_ROOT; RuntimeError if it is not set."""
    root = os.environ.get(ROOT_ENV)
    if not root:
        raise RuntimeError(f"set ${ROOT_ENV} to a sweep ROOT (the directory with sweep.json)")
    return Path(root)


def _cpu_config(sweep: SweepConfig, setting: Setting) -> Config:
    """The setting's config with device cpu; the device is not part of any cache hash."""
    return setting_config(sweep, setting).model_copy(update={"device": "cpu"})


def _with_data(
    setting: Setting,
    status: Status,
    relative: Path,
    cfg: Config,
    net: MLP,
    history: History,
    dataset,
    epoch: int,
) -> SettingResult:
    """The result of a net (final or checkpointed): its history, example outputs and metrics."""
    # Predictions are scored on the whole training set: the sweep has no held-out data.
    everything = example_outputs(net, dataset, torch.arange(len(dataset.x)))
    shown = pick_examples(len(dataset.x), min(N_EXAMPLES, len(dataset.x)), EXAMPLE_SEED)
    metrics = prediction_report(setting.task, everything.y_true, everything.y_pred, cfg.data)
    return SettingResult(
        setting,
        status,
        relative,
        history=history,
        examples=example_outputs(net, dataset, shown),
        metrics=metrics,
        data_cfg=cfg.data,
        epoch=epoch,
        n_epochs=cfg.train.n_epochs,
    )


# What torch.load and the artefact loader raise for a truncated, foreign or stale file. Anything
# else (a shape mismatch in the model code, a bug) is not a file problem and propagates.
_UNREADABLE = (OSError, ValueError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError)


STALE_NOTE = (
    "train/final.pt was written under a different config, model or loss definition (an "
    "older MODEL_VERSION or LOSS_VERSION, or another weight_lambda or setting); rerun in a "
    "fresh sweep ROOT"
)


def _not_shown(setting: Setting, relative: Path, note: str) -> SettingResult:
    return SettingResult(setting, "partial", relative, note=note)


def _dataset_missing(directory: Path, relative: Path, setting: Setting, what: str):
    """A "partial" result with a note if the dataset file is absent (it is never generated
    here, because `get_dataset` would write it), else None."""
    if (cache_paths(directory).data / "dataset.pt").exists():
        return None
    note = f"{what} exists but data/dataset.pt does not; data is never generated here"
    return _not_shown(setting, relative, note)


def _load_partial(directory: Path, relative: Path, setting: Setting, cfg: Config) -> SettingResult:
    """A setting with a checkpoint but no final.pt, from its latest checkpoint. A dataset file
    or checkpoint that is unreadable, or from another config, is reported in `note` instead of
    raised: the notebook is built while jobs run, and one bad file must not hide the other 35
    settings."""
    missing = _dataset_missing(directory, relative, setting, "a checkpoint")
    if missing:
        return missing
    cpu, dtype = torch.device("cpu"), resolve_dtype(cfg.dtype)
    try:
        dataset = get_dataset(directory, cfg)
    except _UNREADABLE as err:
        return _not_shown(setting, relative, f"dataset not readable: {err}")
    try:
        net, history, epoch = load_latest_checkpoint(directory, cfg, cpu, dtype)
    except (OSError, ValueError) as err:
        return _not_shown(setting, relative, f"checkpoint not shown: {err}")
    return _with_data(setting, "partial", relative, cfg, net, history, dataset, epoch)


def load_setting_result(root: Path, sweep: SweepConfig, setting: Setting) -> SettingResult:
    """Load one setting: a done one from train/final.pt, a partial one from its latest
    checkpoint (then `provisional`, with the epoch reached), a missing one not at all, and a
    stale one (finished under another config or loss definition) not at all, with a note.

    Read-only: a setting whose data/dataset.pt is missing is reported as "partial" with a note,
    because loading it through `get_dataset` would generate and write the file. A finished
    setting whose dataset or other artefacts do not match the sweep config raises; for a partial
    one an unreadable or stale dataset or checkpoint file is reported in `note` instead, since its
    checkpoint may be replaced by a running job at any moment. Other exceptions (from the
    model code) propagate: they are bugs, not bad files."""
    directory = setting_dir(root, setting)
    relative = directory.relative_to(root)
    status = setting_status(root, sweep, setting)
    if status == "missing":
        return SettingResult(setting, status, relative)
    if status == "stale":
        return SettingResult(setting, status, relative, note=STALE_NOTE)
    cfg = _cpu_config(sweep, setting)
    if status == "partial":
        return _load_partial(directory, relative, setting, cfg)
    missing = _dataset_missing(directory, relative, setting, "train/final.pt")
    if missing:
        return missing
    dataset = get_dataset(directory, cfg)
    net, history = load_trained(directory, cfg, torch.device("cpu"), resolve_dtype(cfg.dtype))
    return _with_data(setting, status, relative, cfg, net, history, dataset, cfg.train.n_epochs)


def load_results(root: Path | str) -> list[SettingResult]:
    """All settings of the sweep at `root`, in grid order. FileNotFoundError if `root` has no
    sweep.json: with no sweep there is nothing to report. ValueError if the sweep's grid lacks
    the axes net.L, net.phi and train.rate_lambda (the report supports the modularity grid)."""
    root = Path(root)
    sweep = load_sweep(root)
    missing = [a for a in MODULARITY_AXES if a not in grid_of(sweep)]
    if missing:
        raise ValueError(
            f"{root}: the report supports the modularity grid, whose axes include {missing}; "
            f"this sweep's axes are {list(grid_of(sweep))}"
        )
    return list(map(partial(load_setting_result, root, sweep), expand(sweep)))


def _varying_axes(setting: Setting) -> dict:
    """The setting's axes other than the task."""
    return {a: v for a, v in setting.axes.items() if a != TASK_AXIS}


def find_result(results: list[SettingResult], task: str, axes: Mapping[str, Any]) -> SettingResult:
    """The one result of `task` whose axes equal `axes` (a dict of dotted path -> value, every
    axis that varies besides the task). LookupError if none or several match, so a notebook cell
    never silently shows another setting than the one it names."""
    matches = [
        r
        for r in results
        if r.setting.task == task
        and _varying_axes(r.setting).keys() == axes.keys()
        and all(same_value(_varying_axes(r.setting)[a], v) for a, v in axes.items())
    ]
    if len(matches) != 1:
        raise LookupError(f"{len(matches)} results for task {task!r} with axes {dict(axes)}")
    return matches[0]


# ---------------------------------------------------------------------------- analysis
def _last_per_layer(result: SettingResult, which: str) -> list[float]:
    history = result.history
    key = MIR_KINDS[which]
    if history is None or not history.get(key):
        raise ValueError(
            f"{setting_label(result.setting)}: history has no monitor entries {key!r}; "
            "was the sweep run with monitor_int > 0?"
        )
    return list(history[key][-1])


def _padded(values: list[float], width: int) -> np.ndarray:
    row = np.full(width, np.nan)
    row[: len(values)] = values
    return row


def _mir_row(result: SettingResult, which: str, width: int) -> np.ndarray:
    """Last MIR per layer of a setting, NaN-padded. A setting with no data, or a provisional
    one with no monitor call yet, gives all NaN; a done setting without monitor history raises."""
    if result.status == "done" or (result.provisional and result.history.get(MIR_KINDS[which])):
        return _padded(_last_per_layer(result, which), width)
    return np.full(width, np.nan)


def final_mir(results: list[SettingResult], task: str) -> FinalMIR:
    """Final (last monitor call) MIR per layer of every setting of `task`; for a provisional
    setting the last call so far, flagged as such."""
    rows = [r for r in results if r.setting.task == task]
    if not rows:
        raise ValueError(f"no settings of task {task!r} in the results")
    width = max(LAYERS)
    return FinalMIR(
        task=task,
        labels=[setting_label(r.setting) for r in rows],
        status=[r.status for r in rows],
        mir_epoch=[r.mir_epoch if r.provisional else None for r in rows],
        epoch=[r.epoch if r.provisional else None for r in rows],
        n_epochs=[r.n_epochs if r.provisional else None for r in rows],
        **{kind: np.stack([_mir_row(r, kind, width) for r in rows]) for kind in MIR_KINDS},
    )


def _fmt(value: float) -> str:
    return "n/a" if np.isnan(value) else f"{value:.3f}"


def _provisional_list(fm: FinalMIR) -> str:
    """ "A (MIR from epoch 2; checkpoint at epoch 4 of 6), ..." for the provisional rows."""
    return ", ".join(
        f"{label} ({_origin(m, e, n)})"
        for label, prov, m, e, n in zip(
            fm.labels, fm.provisional, fm.mir_epoch, fm.epoch, fm.n_epochs, strict=True
        )
        if prov
    )


def mir_table_markdown(fm: FinalMIR) -> str:
    """Markdown table: one row per setting, raw / corrected / chance MIR for each layer.
    A provisional row is marked `*` and a footnote says its values are not final."""
    n_layers = max(LAYERS)
    layer_cols = [f"{kind} L{layer + 1}" for kind in MIR_KINDS for layer in range(n_layers)]
    lines = [
        "| setting | " + " | ".join(layer_cols) + " |",
        "|---|" + "---:|" * len(layer_cols),
    ]
    for row, (label, status, prov) in enumerate(
        zip(fm.display_labels(), fm.status, fm.provisional, strict=True)
    ):
        name = label.replace("*", "\\*")
        name = f"{name} ({status})" if status != "done" and not prov else name
        cells = [
            _fmt(fm.array(kind)[row, layer]) for kind in MIR_KINDS for layer in range(n_layers)
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    if any(fm.provisional):
        lines.append(
            "\n\\* provisional: the setting is still running and these are the values of its "
            f"latest checkpoint, not final: {_provisional_list(fm)}."
        )
    return "\n".join(lines)


def _extreme_cells(fm: FinalMIR, highest: bool) -> tuple[float, list[tuple[str, int]]]:
    """The max (or min) finite corrected MIR and every (setting label, 1-based layer) that has
    exactly that value."""
    values = fm.corrected
    finite = np.isfinite(values)
    target = values[finite].max() if highest else values[finite].min()
    rows, layers = np.nonzero(finite & (values == target))
    return float(target), [(fm.labels[r], int(layer) + 1) for r, layer in zip(rows, layers)]


def _describe_extreme(word: str, value: float, cells: list[tuple[str, int]]) -> str:
    listed = "; ".join(f"{label}, layer {layer}" for label, layer in cells[:N_TIES_LISTED])
    if len(cells) == 1:
        return f"{word} at {listed} ({value:.3f})"
    more = len(cells) - N_TIES_LISTED
    tail = f"; and {more} more" if more > 0 else ""
    return f"{word} ({value:.3f}) tied across {len(cells)} layer-cells: {listed}{tail}"


def _finished_statistics(fm: FinalMIR) -> str:
    """Extremes and means of the finite values in `fm` (which holds finished settings only)."""
    if not np.isfinite(fm.corrected).any():
        return " No finite corrected MIR in the finished settings."
    hi = _describe_extreme("highest", *_extreme_cells(fm, highest=True))
    lo = _describe_extreme("lowest", *_extreme_cells(fm, highest=False))
    common = np.isfinite(fm.raw) & np.isfinite(fm.corrected) & np.isfinite(fm.chance)
    text = f" Final corrected MIR is {hi}, and {lo}."
    if not common.any():
        return text + " No layer-cell has all of raw, corrected and chance values."
    means = {kind: float(fm.array(kind)[common].mean()) for kind in MIR_KINDS}
    return (
        f"{text} Mean over the {int(common.sum())} layer-cells with all three values: "
        f"raw {means['raw']:.3f}, corrected {means['corrected']:.3f}, "
        f"chance baseline {means['chance']:.3f}."
    )


def summary_text(fm: FinalMIR) -> str:
    """Prose summary of a task's final MIR, computed from the arrays.

    Extremes are of the corrected MIR over layer-cells (one layer of one setting), with ties
    listed. The means pool the layer-cells where raw, corrected and chance are all finite, so a
    setting counts once per layer, not once. Every statistic is over FINISHED settings only;
    provisional ones (still running, values from a checkpoint) are counted and listed apart."""
    n_done = sum(s == "done" for s in fm.status)
    n_prov = sum(fm.provisional)
    text = f"**{fm.task}**: {n_done} of {len(fm.status)} settings finished"
    text += f", {n_prov} provisional (still running)." if n_prov else "."
    unshown = [
        f"{label} ({s})"
        for label, s, prov in zip(fm.labels, fm.status, fm.provisional, strict=True)
        if s != "done" and not prov
    ]
    if unshown:
        text += " Not shown: " + ", ".join(unshown) + "."
    if n_done == 0:
        text += " There is no finished setting to summarise."
    else:
        text += _finished_statistics(fm.finished_only())
    if n_prov:
        text += (
            f" Provisional, not part of the statistics above: {_provisional_list(fm)}; "
            "their numbers in the table and heatmap are marked and will change."
        )
    return text


def overview_text(results: list[SettingResult]) -> str:
    """Counts of done, partial (provisional or without data), missing and stale settings."""
    n = {s: sum(r.status == s for r in results) for s in ("done", "partial", "missing", "stale")}
    n_prov = sum(r.provisional for r in results)
    text = f"{len(results)} settings: {n['done']} done, {n_prov} partial (provisional)"
    if n["partial"] > n_prov:
        text += f", {n['partial'] - n_prov} partial without data (see their notes)"
    text += f", {n['missing']} missing"
    return text + (
        f", {n['stale']} stale (old loss or config; see their notes)." if n["stale"] else "."
    )


def _list_numbers(values: list[float], fmt: str = ".3f") -> str:
    return ", ".join(f"{v:{fmt}}" for v in values)


def _require_history(result: SettingResult) -> dict:
    """The history of a done setting; ValueError naming any key the text needs but lacks."""
    history = result.history or {}
    missing = [k for k in REQUIRED_HISTORY if not history.get(k)]
    if missing:
        raise ValueError(
            f"{setting_label(result.setting)}: history lacks {missing}; "
            "was the sweep run with monitor_int > 0?"
        )
    return history


def _provisional_text(result: SettingResult, label: str) -> str:
    """Prose for a setting read from its latest checkpoint. Everything is "so far", and what
    the checkpoint does not hold yet (no logged epoch, no monitor call) is said, not skipped."""
    h = result.history
    head = (
        f"**{label}: partial, {result.progress}** "
        "(provisional: the latest checkpoint of a run still going)"
    )
    if h.get("epoch"):
        loss = f"at the last logged epoch ({h['epoch'][-1]}) the loss is {h['loss'][-1]:.4g}"
        loss += f" (mse {h['mse'][-1]:.4g})."
    else:
        loss = "no epoch has been logged yet."
    if h.get("monitor_epoch") and h.get("monitor_mir"):
        last = {kind: _last_per_layer(result, kind) for kind in MIR_KINDS}
        mir = (
            f"MIR was last measured at epoch {h['monitor_epoch'][-1]}. MIR so far per layer: "
            f"raw {_list_numbers(last['raw'])}; corrected {_list_numbers(last['corrected'])}; "
            f"chance baseline {_list_numbers(last['chance'])}."
        )
    else:
        mir = "There is no MIR yet: the checkpoint holds no monitor call."
    metrics = ", ".join(f"{k} = {v:.3f}" for k, v in result.metrics.items())
    return (
        f"{head}: {loss} {mir} Prediction of the checkpointed net on the training set: {metrics}."
    )


def setting_text(result: SettingResult) -> str:
    """Prose for one setting: where it is, final loss, final MIR per layer, metrics. A
    provisional setting is headed "partial, epoch X of N" and its numbers are "so far"."""
    label = f"{result.setting.task} {setting_label(result.setting)}"
    if result.provisional:
        return _provisional_text(result, label)
    if result.status != "done":
        reason = (
            result.note
            or {
                "missing": "no checkpoint found",
                "partial": "a checkpoint exists but there is no final.pt",
                "stale": STALE_NOTE,
            }[result.status]
        )
        return f"**{label}: {result.status}** ({reason}; under `{result.directory}`); not shown."
    h = _require_history(result)
    logged, monitored = h["epoch"][-1], h["monitor_epoch"][-1]
    last = {kind: _last_per_layer(result, kind) for kind in MIR_KINDS}
    dead = h["monitor_n_dead_corrected"][-1]
    metrics = ", ".join(f"{k} = {v:.3f}" for k, v in result.metrics.items())
    return (
        f"**{label}**: at the last logged epoch ({logged}) the loss is {h['loss'][-1]:.4g} "
        f"(mse {h['mse'][-1]:.4g}). MIR is measured at epoch {monitored}. "
        f"Final MIR per layer: raw {_list_numbers(last['raw'])}; corrected "
        f"{_list_numbers(last['corrected'])}; chance baseline {_list_numbers(last['chance'])}. "
        f"Dead neurons (corrected) per layer: {', '.join(str(int(d)) for d in dead)}. "
        f"Prediction on the training set: {metrics}."
    )


# ---------------------------------------------------------------------------- figures
def _has_monitor_history(history: dict) -> bool:
    return bool(history.get("monitor_epoch")) and bool(history.get("monitor_mir"))


def setting_items(result: SettingResult) -> list[Item]:
    """What the notebook shows for one setting: its prose, then loss curves, example outputs
    and the MIR history; just the prose (stating what is missing) if it has no data.

    A provisional setting's figures all carry "partial, epoch X of N". A figure whose data does
    not exist yet (no logged epoch, no monitor call) is left out; the prose says so."""
    text = setting_text(result)
    if result.history is None:
        return [text]
    title = f"{result.setting.task} {setting_label(result.setting)}"
    if result.provisional:
        title += f" — partial, {result.progress}"
    items: list[Item] = [text]
    if result.status == "done" or result.history["epoch"]:
        items.append(fig_loss_curves(result.history, title=title))
    shown = title if result.provisional else ""
    items.append(
        fig_example_outputs(result.setting.task, result.examples, result.data_cfg, title=shown)
    )
    if result.status == "done" or _has_monitor_history(result.history):
        items.append(fig_mir_history(result.history, title=title))
    return items


def task_items(results: list[SettingResult], task: str) -> list[Item]:
    """Summary of one task: prose, the table and the heatmap figure."""
    fm = final_mir(results, task)
    heatmaps = fig_mir_heatmaps(
        {_PANEL_TITLES[kind]: fm.array(kind) for kind in MIR_KINDS},
        fm.display_labels(),
        fm.row_status(),
        title=f"{fm.task}: final MIR per layer",
    )
    return [summary_text(fm), mir_table_markdown(fm), heatmaps]


# ---------------------------------------------------------------------------- notebook edge
def show(items: Sequence[Item]) -> None:
    """Display markdown strings and figures in a notebook cell, then close the figures."""
    from IPython.display import Markdown, display

    for item in items:
        if isinstance(item, str):
            display(Markdown(item))
        else:
            display(item)
            plt.close(item)
