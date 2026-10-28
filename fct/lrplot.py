"""Figures for the learning-rate tuning table (plotting layer: takes rows, draws).

Text sizes and line widths come from the shared style of `fct.draw` (`STYLE_RC`).
"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from fct.draw import FIG_WIDTH, STYLE_RC, _styled
from fct.lrtune import Row, collapse_seeds

PHI_COLOURS = {"tanh": "#0072B2", "relu": "#D55E00", "id": "#009E73"}  # colour-blind safe
LINESTYLES = ("-", "--", ":", "-.")  # one per rate_lambda present, in increasing order
# Line width relative to the default. Lines of different activations can coincide (at depth 1 phi
# acts only through the rate regulariser); drawing the widest first and the thinnest last keeps
# every one visible.
WIDTH_FACTOR = {"tanh": 2.0, "relu": 1.5, "id": 1.0}
MIN_REL_SPAN = 0.05  # smallest y-range shown, as a fraction of the largest value
FAILURE_MARKER_Y = 0.96  # axes fraction at which a diverged run is marked
ROW_HEIGHT = 3.4  # inches per row of panels


def _draw_order(pair: tuple[str, float]) -> tuple:
    phi, rate = pair
    return (-WIDTH_FACTOR[phi], phi, rate)


def _keep_span(ax: Axes, values: Sequence[float], rel: float = MIN_REL_SPAN) -> None:
    """Make the y-range at least `rel` of the largest value, so that differences below that
    are not stretched to fill the panel; plain tick labels (no offset). All-zero data get a
    unit range."""
    lo, hi, top = min(values), max(values), max(abs(v) for v in values)
    pad = max(0.0, rel * top - (hi - lo)) / 2 + 0.05 * (hi - lo)
    if lo - pad == hi + pad:
        pad = 1.0
    ax.set_ylim(lo - pad, hi + pad)
    ax.ticklabel_format(axis="y", useOffset=False)


def plot_metric_vs_lr(
    rows: Sequence[Row],
    task: str,
    L: int,
    key: str = "heldout_mse",
    best_lr: float | None = None,
    logy: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Column `key` of the result table against lr (log x) for every phi and rate_lambda of one
    task and depth that is in `rows`; draws into `ax` or a new figure.

    Several seeds are collapsed to their median. A diverged run is a red cross at the top of
    the axes at its lr, since it has no value to draw; `best_lr` adds a vertical guide. At
    depth 1 phi acts only through the rate regulariser, so the lines of every phi coincide for
    rate_lambda = 0: they are drawn in decreasing width (tanh, relu, id) so all stay visible. ValueError for a task and
    depth without rows or a phi without a colour.
    """
    cells = [r for r in collapse_seeds(rows, (key,)) if r["task"] == task and r["L"] == L]
    if not cells:
        raise ValueError(f"no rows for task {task!r}, L={L}")
    unknown = {r["phi"] for r in cells} - set(PHI_COLOURS)
    if unknown:
        raise ValueError(f"no colour for phi {sorted(unknown)}; known: {sorted(PHI_COLOURS)}")
    rates = sorted({r["rate_lambda"] for r in cells})
    if len(rates) > len(LINESTYLES):
        raise ValueError(f"at most {len(LINESTYLES)} rate_lambda values can be drawn, got {rates}")
    if ax is None:
        with plt.rc_context(STYLE_RC):
            _, ax = plt.subplots(layout="constrained")
    width = plt.rcParams["lines.linewidth"]
    # widest first, so its line lies under a coinciding thinner one
    for phi, rate in sorted({(r["phi"], r["rate_lambda"]) for r in cells}, key=_draw_order):
        line = sorted(
            (r for r in cells if r["phi"] == phi and r["rate_lambda"] == rate),
            key=lambda r: r["lr"],
        )
        ok = [r for r in line if not r["diverged"]]
        ax.plot(
            [r["lr"] for r in ok],
            [r[key] for r in ok],
            LINESTYLES[rates.index(rate)],
            marker="o",
            lw=width * WIDTH_FACTOR[phi],
            color=PHI_COLOURS[phi],
            label=f"{phi}, r={rate:g}",
        )
        bad = [r["lr"] for r in line if r["diverged"]]
        if bad:
            ax.plot(
                bad,
                [FAILURE_MARKER_Y] * len(bad),
                "x",
                color="red",
                transform=ax.get_xaxis_transform(),
                clip_on=False,
            )
    if best_lr is not None:
        ax.axvline(best_lr, color="0.6", zorder=0)
    ax.set_xscale("log")
    lrs = sorted({r["lr"] for r in cells})
    ax.set_xticks(lrs, [f"{lr:.0e}".replace("e-0", "e-") for lr in lrs])
    ax.minorticks_off()
    values = [r[key] for r in cells if not r["diverged"]]
    if logy:
        ax.set_yscale("log")
    elif values:
        _keep_span(ax, values)
    ax.set_title(f"{task}, L={L}")
    return ax


@_styled
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
        figsize=(FIG_WIDTH, ROW_HEIGHT * len(tasks) + 0.8),
        squeeze=False,
        layout="constrained",
    )
    for i, task in enumerate(tasks):
        for j, L in enumerate(depths):
            best = (best_by_group or {}).get((task, L))
            ax = plot_metric_vs_lr(rows, task, L, key, best, logy, axes[i, j])
            ax.set_xlabel("learning rate" if i == len(tasks) - 1 else "")
            ax.set_ylabel(ylabel if j == 0 else "")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels))
    return fig
