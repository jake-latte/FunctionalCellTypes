"""Everything the results notebook shows, so the notebook itself holds no logic.

Three layers, as in the rest of the package:

- I/O: `load_results` reads a sweep ROOT into one `SettingResult` per grid setting. A setting
  that has not finished is returned with its status and no data, so a partial sweep still loads.
- Analysis (pure): `final_mir` collects the last MIR of every layer of a task's settings into
  arrays; `mir_table_markdown`, `summary_text` and `setting_text` turn numbers into prose, so no
  claim about the results is ever typed by hand.
- Plotting: the figures come from `fct.draw`; `setting_items` and `task_items` assemble them.
  `show` is the only function that touches IPython.

Loading is read-only: nothing is generated or cached under ROOT. Settings are loaded on the CPU
whatever device they were trained on, and are validated against the config rebuilt from
sweep.json (a stale artefact raises, it is not shown).
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from fct.analyse import ExampleOutputs, example_outputs, pick_examples, prediction_report
from fct.cache import cache_paths
from fct.config import Config, DataConfig, resolve_dtype
from fct.data import get_dataset
from fct.draw import fig_example_outputs, fig_loss_curves, fig_mir_heatmaps, fig_mir_history
from fct.sweep import (
    LAYERS,
    Setting,
    Status,
    SweepConfig,
    expand,
    load_sweep,
    setting_config,
    setting_dir,
    setting_status,
)
from fct.train import load_trained

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
    """One grid setting; `directory` is relative to ROOT. `history`, `examples`, `metrics` and
    `data_cfg` are None unless the setting is done. `note` explains a setting that is not shown
    beyond what its status says (a finished run whose dataset file is missing)."""

    setting: Setting
    status: Status
    directory: Path
    note: str | None = None
    history: dict | None = None
    examples: ExampleOutputs | None = None
    metrics: dict[str, float] | None = None
    data_cfg: DataConfig | None = None


@dataclass(frozen=True)
class FinalMIR:
    """Final MIR per layer of one task's settings, rows in grid order. Arrays are
    (n_settings, max layers); a layer the setting does not have, and every layer of an
    unfinished setting, is NaN."""

    task: str
    labels: list[str]
    status: list[Status]
    raw: np.ndarray
    corrected: np.ndarray
    chance: np.ndarray

    def array(self, which: str) -> np.ndarray:
        if which not in MIR_KINDS:
            raise ValueError(f"which must be one of {sorted(MIR_KINDS)}, got {which!r}")
        return getattr(self, which)


def setting_label(setting: Setting) -> str:
    return f"L{setting.L} {setting.phi} r={setting.rate_lambda:g}"


# ---------------------------------------------------------------------------- explanation
# Markdown for the notebook (Jupyter renders $...$ and $$...$$). It restates what
# fct/modularity.py computes; tests/test_report.py checks its key terms and that every equation
# is balanced, but not its agreement with the code, so edit both together.
MIR_EXPLANATION = r"""## What the MIR numbers mean

All quantities are computed by `fct/modularity.py`, for one hidden layer at a time, at each
monitor call during training (`fct.monitor.make_mir_monitor`, which calls `layer_mir_report`).

### Setup: rates, bins, factors

For a layer of $N$ neurons and the $P$ training inputs $x_p$, the **rate** of neuron $i$ is its
post-activation value $r_{ip} = \phi(h_{i}(x_p))$. The task provides ground-truth **factors**
$F_m$ (one integer or real label per input, e.g. the latents $z_j$ of `linear_teacher`;
`where` and `what` for `what_where`; the agent position, one occupancy bit per grid cell and
one bit per action for `object_action`), collected into $K$ **factor groups** $g_1,\dots,g_K$
(`linear_teacher`: one group per latent; `what_where`: `where`, `what`; `object_action`:
`position`, `objects`, `actions`).

Rates are discretised into $B$ equal-frequency **rank bins** (`rank_bins`, $B$ =
`monitor_bins`, 10 by default). With the midrank
$m_{ip} = \tfrac12\big(\#\{q: r_{iq} < r_{ip}\} + \#\{q: r_{iq} \le r_{ip}\}\big)$ the bin is

$$b_{ip} = \Big\lfloor \frac{B\, m_{ip}}{P} \Big\rfloor \in \{0,\dots,B-1\},$$

so tied rates (a ReLU's mass at zero) share one bin and the bins are invariant to monotone
transforms of the rate. Integer factors are categorical with $C_m$ distinct labels; real-valued
factors are rank-binned the same way into $C_m = B$ labels (`factor_labels`).

### Mutual information, preference, MIR

The **plug-in mutual information** (`mutual_information`, in nats) between the binned rate of
neuron $i$ and factor $F_m$ uses the empirical joint frequencies $\hat p$ over the $P$ samples:

$$I_{im} = \sum_{b,c} \hat p(b,c)\,\log\frac{\hat p(b,c)}{\hat p(b)\,\hat p(c)} \;\ge 0 .$$

The information about a group is the **sum** over its members (the information about their
joint configuration cannot be estimated by a plug-in estimator):

$$I_{ik} = \sum_{m \in g_k} I_{im}.$$

The **preference** of neuron $i$ (`preference`; called `r` in the code) is the share of its
information that sits in its best group:

$$\rho_i = \frac{\max_k I_{ik}}{\sum_{k} I_{ik}} \in \Big[\tfrac1K, 1\Big].$$

A **dead neuron** is one whose total information $\sum_k I_{ik}$ is at most $10^{-12}$ (constant
firing, or a rate unrelated to every factor at floating-point level). Its preference is
undefined; it is set to exactly $\rho_i = 1/K$, so it adds nothing to the numerator below instead
of producing NaN. The layer's **modularity index** (`mir`, normalised so that chance preference
$1/K$ maps to 0 and perfect preference 1 maps to 1) is

$$\mathrm{MIR} = \frac{\bar\rho - 1/K}{1 - 1/K}, \qquad \bar\rho = \frac1N \sum_{i=1}^{N} \rho_i .$$

It is 1 when every neuron carries information about one group only and 0 when information is
spread evenly over the $K$ groups (or neurons are dead).

### Raw (plug-in) MIR and why it has a floor

The **raw MIR** is the formula above applied to the plain plug-in informations
(`layer_mir`, `bias_correction="none"`; the `monitor_mir` curve). The plug-in estimate is biased
upward: for a rate independent of a factor, with $B$ rate bins and $C$ factor labels,

$$\mathbb{E}[\hat I] \approx \frac{(B-1)(C-1)}{2P}\ \text{nats}.$$

The bias grows with the factor's cardinality $C$, so even completely unrelated neurons have
positive information, concentrated in the group with the most labels, and $\max_k / \sum_k$ of
positive numbers is never exactly $1/K$. The raw MIR of noise neurons therefore sits above 0 by
an amount that depends on the cardinalities of the factors, not on the network. For example at
$P = 8192$ and $B = 10$, noise neurons score a raw MIR of about 0.89 with one 25-way and two
binary factors, but about 0.10 with three five-way factors. A raw MIR is only interpretable
next to its chance baseline.

### Chance baseline

The **chance baseline** (`shuffled_mir`; `monitor_mir_chance`) is the raw MIR of the same layer
when the link between rates and factors is destroyed: all factors are permuted across the $P$
samples with **one shared random permutation** $\pi$ (so dependencies between the factors are
kept, rates are untouched), the plug-in informations are recomputed as $I_{im}^{\pi}$ and the
raw MIR formula is applied. The baseline is the mean of this raw MIR over $S$ permutations
($S$ = `monitor_shuffles`, 5 by default; a seeded generator, so it is deterministic):

$$\mathrm{MIR}_{\text{chance}} = \frac1S \sum_{s=1}^{S} \mathrm{MIR}\big[I^{\pi_s}\big].$$

A raw MIR is only evidence of modularity to the extent that it exceeds this baseline.

### Corrected MIR

The **corrected MIR** (`layer_mir`, `bias_correction="shuffle"`; `monitor_mir_corrected`)
removes the bias before the preferences are formed. `layer_mir_report` uses the same $S$
permutations as the baseline, so no extra shuffles are drawn (`n_shuffles` = $S$). For every
neuron and factor (member, not group) the mean shuffled information is the null
$\bar I^{0}_{im} = \frac1S\sum_s I^{\pi_s}_{im}$ and the corrected information is

$$\tilde I_{im} = \max\big(I_{im} - \bar I^{0}_{im},\, 0\big) .$$

A neuron whose raw total is not clearly above its null, $\sum_m I_{im} \le 2\sum_m \bar I^{0}_{im}$,
is indistinguishable from noise and all its $\tilde I_{im}$ are set to 0, which makes it dead
($\rho_i = 1/K$). (Without this step the leftover noise residuals, mostly in one factor, would
give noise neurons a spuriously high preference.) The corrected informations are summed within
groups, and $\rho_i$ and the MIR are computed exactly as above:

$$\mathrm{MIR}_{\text{corrected}} = \frac{\bar{\tilde\rho} - 1/K}{1 - 1/K}.$$

Noise neurons contribute exactly $1/K$, so the corrected MIR of an unrelated layer is about 0
whatever the cardinalities, and no baseline has to be subtracted from it. Because $\rho_i \ge 1/K$,
it cannot go below 0; it is a modularity of the neurons that carry detectable information,
diluted by those that do not.

### Reading the plots and tables

- **raw MIR panel**: one solid line per layer (marker o) is the raw MIR; the dotted line of
  the same colour (marker x) is that layer's chance baseline. Read the gap between them.
- **corrected MIR panel**: one line per layer; 0 means indistinguishable from unrelated
  neurons, 1 means every informative neuron is selective for one group.
- **dead-neuron panel**: the number of dead neurons of each layer under the corrected estimate
  (`monitor_n_dead_corrected`): neurons with constant firing or no information beyond the
  shuffled null. At most $N$ per layer.
- **heatmaps and table**: one row per setting, one column per layer, the value at the last
  monitor call, in three panels or column groups: `raw`, `corrected` and `chance baseline`.
  Grey cells are layers the setting does not have or settings that have not finished.
"""


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


def load_setting_result(root: Path, sweep: SweepConfig, setting: Setting) -> SettingResult:
    """Load one setting; only a finished one (train/final.pt) is read beyond its status.

    Read-only: a finished setting whose data/dataset.pt is missing is reported as "partial" with
    a note, because loading it through `get_dataset` would generate and write the file."""
    directory = setting_dir(root, setting)
    relative = directory.relative_to(root)
    status = setting_status(root, setting)
    if status != "done":
        return SettingResult(setting, status, relative)
    if not (cache_paths(directory).data / "dataset.pt").exists():
        note = "train/final.pt exists but data/dataset.pt does not; data is never generated here"
        return SettingResult(setting, "partial", relative, note=note)
    cfg = _cpu_config(sweep, setting)
    dataset = get_dataset(directory, cfg)
    net, history = load_trained(directory, cfg, torch.device("cpu"), resolve_dtype(cfg.dtype))
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
    )


def load_results(root: Path | str) -> list[SettingResult]:
    """All settings of the sweep at `root`, in grid order. FileNotFoundError if `root` has no
    sweep.json: with no sweep there is nothing to report."""
    root = Path(root)
    sweep = load_sweep(root)
    return list(map(partial(load_setting_result, root, sweep), expand(sweep)))


def find_result(
    results: list[SettingResult], task: str, L: int, phi: str, rate_lambda: float
) -> SettingResult:
    """The result of the named setting; ValueError if there is none, so a notebook cell never
    silently shows another setting than the one it names."""
    wanted = Setting(task, L, phi, rate_lambda)
    for result in results:
        if result.setting == wanted:
            return result
    raise ValueError(f"no result for setting {tuple(wanted)}")


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


def final_mir(results: list[SettingResult], task: str) -> FinalMIR:
    """Final (last monitor call) MIR per layer of every setting of `task`."""
    rows = [r for r in results if r.setting.task == task]
    if not rows:
        raise ValueError(f"no settings of task {task!r} in the results")
    width = max(LAYERS)

    def table(which: str) -> np.ndarray:
        return np.stack(
            [
                _padded(_last_per_layer(r, which), width)
                if r.status == "done"
                else np.full(width, np.nan)
                for r in rows
            ]
        )

    return FinalMIR(
        task=task,
        labels=[setting_label(r.setting) for r in rows],
        status=[r.status for r in rows],
        raw=table("raw"),
        corrected=table("corrected"),
        chance=table("chance"),
    )


def _fmt(value: float) -> str:
    return "n/a" if np.isnan(value) else f"{value:.3f}"


def mir_table_markdown(fm: FinalMIR) -> str:
    """Markdown table: one row per setting, raw / corrected / chance MIR for each layer."""
    n_layers = max(LAYERS)
    layer_cols = [f"{kind} L{layer + 1}" for kind in MIR_KINDS for layer in range(n_layers)]
    lines = [
        "| setting | " + " | ".join(layer_cols) + " |",
        "|---|" + "---:|" * len(layer_cols),
    ]
    for row, (label, status) in enumerate(zip(fm.labels, fm.status, strict=True)):
        name = label if status == "done" else f"{label} ({status})"
        cells = [
            _fmt(fm.array(kind)[row, layer]) for kind in MIR_KINDS for layer in range(n_layers)
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
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


def summary_text(fm: FinalMIR) -> str:
    """Prose summary of a task's final MIR, computed from the arrays.

    Extremes are of the corrected MIR over layer-cells (one layer of one setting), with ties
    listed. The means pool the layer-cells where raw, corrected and chance are all finite, so a
    setting counts once per layer, not once."""
    n_done = sum(s == "done" for s in fm.status)
    text = f"**{fm.task}**: {n_done} of {len(fm.status)} settings finished."
    unfinished = [
        f"{label} ({s})" for label, s in zip(fm.labels, fm.status, strict=True) if s != "done"
    ]
    if unfinished:
        text += " Not shown: " + ", ".join(unfinished) + "."
    if n_done == 0:
        return text + " There is no finished setting to summarise."
    if not np.isfinite(fm.corrected).any():
        return text + " No finite corrected MIR in the finished settings."
    hi = _describe_extreme("highest", *_extreme_cells(fm, highest=True))
    lo = _describe_extreme("lowest", *_extreme_cells(fm, highest=False))
    common = np.isfinite(fm.raw) & np.isfinite(fm.corrected) & np.isfinite(fm.chance)
    text += f" Final corrected MIR is {hi}, and {lo}."
    if not common.any():
        return text + " No layer-cell has all of raw, corrected and chance values."
    means = {kind: float(fm.array(kind)[common].mean()) for kind in MIR_KINDS}
    return (
        f"{text} Mean over the {int(common.sum())} layer-cells with all three values: "
        f"raw {means['raw']:.3f}, corrected {means['corrected']:.3f}, "
        f"chance baseline {means['chance']:.3f}."
    )


def overview_text(results: list[SettingResult]) -> str:
    """Counts of done, partial and missing settings."""
    counts = {s: sum(r.status == s for r in results) for s in ("done", "partial", "missing")}
    return (
        f"{len(results)} settings: {counts['done']} done, {counts['partial']} partial "
        f"(checkpoint but no final.pt), {counts['missing']} missing."
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


def setting_text(result: SettingResult) -> str:
    """Prose for one setting: where it is, final loss, final MIR per layer, metrics."""
    label = f"{result.setting.task} {setting_label(result.setting)}"
    if result.status != "done":
        reason = (
            result.note
            or {
                "missing": "no checkpoint found",
                "partial": "a checkpoint exists but there is no final.pt",
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
def setting_items(result: SettingResult) -> list[Item]:
    """What the notebook shows for one setting: its prose, then loss curves, example outputs
    and the MIR history; just the prose (stating what is missing) if it is not done."""
    text = setting_text(result)
    if result.status != "done":
        return [text]
    title = f"{result.setting.task} {setting_label(result.setting)}"
    return [
        text,
        fig_loss_curves(result.history, title=title),
        fig_example_outputs(result.setting.task, result.examples, result.data_cfg),
        fig_mir_history(result.history, title=title),
    ]


def task_items(results: list[SettingResult], task: str) -> list[Item]:
    """Summary of one task: prose, the table and the heatmap figure."""
    fm = final_mir(results, task)
    heatmaps = fig_mir_heatmaps(
        {_PANEL_TITLES[kind]: fm.array(kind) for kind in MIR_KINDS},
        fm.labels,
        fm.status,
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
