"""Figures for the learning-rate tuning table (plotting layer: takes rows, draws)."""

import itertools
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from fct.lrtune import Row, collapse_seeds

# One colour per phi (colour-blind safe) and one line style per rate regulariser.
PHI_COLOURS = {"tanh": "#0072B2", "relu": "#D55E00"}
RATE_STYLES = {0.0: "-", 1e-3: "--"}
FAILURE_MARKER_Y = 0.96  # axes fraction at which a diverged run is marked


def plot_heldout_vs_lr(
    ax: Axes, rows: Sequence[Row], task: str, L: int, best_lr: float | None = None
) -> Axes:
    """Held-out MSE against lr (log-log) for each phi x rate_lambda of one task and depth.

    Several seeds are collapsed to their median. A diverged run is a red cross at the top of
    the axes at its lr, since it has no value to draw; `best_lr` adds a vertical guide."""
    cells = [r for r in collapse_seeds(rows) if r["task"] == task and r["L"] == L]
    if not cells:
        raise ValueError(f"no rows for task {task!r}, L={L}")
    for phi, rate in itertools.product(PHI_COLOURS, RATE_STYLES):
        style = RATE_STYLES[rate]
        line = sorted(
            (r for r in cells if r["phi"] == phi and r["rate_lambda"] == rate),
            key=lambda r: r["lr"],
        )
        ok = [r for r in line if not r["diverged"]]
        ax.plot(
            [r["lr"] for r in ok],
            [r["heldout_mse"] for r in ok],
            style,
            marker="o",
            ms=3,
            color=PHI_COLOURS[phi],
            label=f"{phi}, r={rate:g}",
        )
        bad = [r["lr"] for r in line if r["diverged"]]
        ax.plot(
            bad,
            [FAILURE_MARKER_Y] * len(bad),
            "x",
            color="red",
            ms=7,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )
    if best_lr is not None:
        ax.axvline(best_lr, color="0.6", lw=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"{task}, L={L}", fontsize=9)
    return ax


def fig_heldout_vs_lr(
    rows: Sequence[Row], best_by_group: dict[tuple, float] | None = None
) -> Figure:
    """Grid of `plot_heldout_vs_lr`: one row per task, one column per depth in the table."""
    tasks = list(dict.fromkeys(r["task"] for r in rows))
    depths = sorted({r["L"] for r in rows})
    fig, axes = plt.subplots(
        len(tasks), len(depths), figsize=(3.6 * len(depths), 2.8 * len(tasks)), squeeze=False
    )
    for i, task in enumerate(tasks):
        for j, L in enumerate(depths):
            best = (best_by_group or {}).get((task, L))
            plot_heldout_vs_lr(axes[i, j], rows, task, L, best)
            axes[i, j].set_xlabel("learning rate" if i == len(tasks) - 1 else "")
            axes[i, j].set_ylabel("held-out MSE" if j == 0 else "")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def fig_gap_vs_lr(scores: Sequence, tasks: Sequence[str], depths: Sequence[int]) -> Figure:
    """Median relative held-out gap to each setting's best lr, per task and depth (failures
    counted as infinite gap and so clipped to the top of the axis)."""
    fig, axes = plt.subplots(
        1, len(depths), figsize=(3.6 * len(depths), 3), squeeze=False, sharey=True
    )
    for ax, L in zip(axes[0], depths):
        for task in tasks:
            cells = [s for s in scores if s.task == task and s.L == L]
            lrs = sorted({s.lr for s in cells})
            gaps = [np.median([s.gap for s in cells if s.lr == lr]) for lr in lrs]
            ax.plot(lrs, [min(g, 10.0) for g in gaps], marker="o", ms=3, label=task)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_title(f"L={L}", fontsize=9)
        ax.set_xlabel("learning rate")
        ax.axhline(0.1, color="0.6", lw=0.8)
    axes[0, 0].set_ylabel("median relative gap")
    axes[0, -1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig
