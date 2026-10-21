"""Figures for the learning-rate tuning table (plotting layer: takes rows, draws)."""

import itertools
from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from fct.lrtune import Row, collapse_seeds

# One colour per phi (colour-blind safe) and one line style per rate regulariser.
PHI_COLOURS = {"tanh": "#0072B2", "relu": "#D55E00"}
RATE_STYLES = {0.0: "-", 1e-3: "--"}
FAILURE_MARKER_Y = 0.96  # axes fraction at which a diverged run is marked


def plot_metric_vs_lr(
    ax: Axes,
    rows: Sequence[Row],
    task: str,
    L: int,
    key: str = "heldout_mse",
    best_lr: float | None = None,
) -> Axes:
    """Column `key` of the result table against lr (log x) for each phi x rate_lambda of one
    task and depth.

    Several seeds are collapsed to their median. A diverged run is a red cross at the top of
    the axes at its lr, since it has no value to draw; `best_lr` adds a vertical guide. At
    depth 1 phi acts only through the rate regulariser, so the tanh and relu lines coincide for
    rate_lambda = 0: tanh is drawn thick and relu thin on top so both stay visible."""
    cells = [r for r in collapse_seeds(rows, (key,)) if r["task"] == task and r["L"] == L]
    if not cells:
        raise ValueError(f"no rows for task {task!r}, L={L}")
    for phi, rate in itertools.product(PHI_COLOURS, RATE_STYLES):
        line = sorted(
            (r for r in cells if r["phi"] == phi and r["rate_lambda"] == rate),
            key=lambda r: r["lr"],
        )
        ok = [r for r in line if not r["diverged"]]
        ax.plot(
            [r["lr"] for r in ok],
            [r[key] for r in ok],
            RATE_STYLES[rate],
            marker="o",
            ms=3,
            lw=3.0 if phi == "tanh" else 1.2,
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
    ax.set_title(f"{task}, L={L}", fontsize=9)
    return ax


def fig_metric_vs_lr(
    rows: Sequence[Row],
    key: str = "heldout_mse",
    ylabel: str = "held-out MSE",
    logy: bool = False,
    best_by_group: dict[tuple, float] | None = None,
) -> Figure:
    """Grid of `plot_metric_vs_lr`: one row per task, one column per depth in the table."""
    tasks = list(dict.fromkeys(r["task"] for r in rows))
    depths = sorted({r["L"] for r in rows})
    fig, axes = plt.subplots(
        len(tasks),
        len(depths),
        figsize=(3.8 * len(depths), 2.6 * len(tasks)),
        squeeze=False,
        layout="constrained",
    )
    for i, task in enumerate(tasks):
        for j, L in enumerate(depths):
            best = (best_by_group or {}).get((task, L))
            ax = plot_metric_vs_lr(axes[i, j], rows, task, L, key, best)
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel("learning rate" if i == len(tasks) - 1 else "")
            ax.set_ylabel(ylabel if j == 0 else "")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), fontsize=8)
    return fig
