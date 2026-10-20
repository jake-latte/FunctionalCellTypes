"""Figures: single-panel `plot_*` functions that draw into an axis, `fig_*` that assemble them.

Plotting is the outermost layer: nothing here computes beyond what is needed to draw, nothing
reads files, and nothing opens a window (the CLI saves figures from cached results on
machines without a display). A `plot_*` function creates a figure only when it is not given an
`ax`, so each one can be dropped into a larger figure unchanged.
"""

import math
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle
from matplotlib.ticker import MaxNLocator
from torch import Tensor

from fct.analyse import (
    ExampleOutputs,
    GridPredictions,
    LayerPCA,
    decode_object_action,
    decode_what_where,
    r_squared,
    split_what_where,
)
from fct.config import DataConfig
from fct.data import TASKS, Dataset
from fct.tasks import ACTION_NAMES, MOTIF_SIZE, N_MACRO

ArrayLike = Tensor | np.ndarray | Sequence[float]

ERROR_VMIN = 1e-6  # lower end of the log colour scale for squared errors
N_EXAMPLES = 5  # example points shown by fig_example

# One colour per layer, cycled when there are more layers than colours.
LAYER_COLORS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
_GREY = "#52514e"  # reference lines and the trained-network overlay


class Overlay(NamedTuple):
    """A second history column drawn on top of a panel."""

    key: str
    label: str
    color: str
    linestyle: str


class SolverPanel(NamedTuple):
    key: str
    title: str
    yscale: str
    hline: float | None = None
    overlay: Overlay | None = None


_NET_OVERLAY = Overlay("train_mse_net", "net", _GREY, "--")
_ETA_OVERLAY = Overlay("eta_phi", "η (Φ)", "#0b0b0b", "-")

# Display order of the solver dashboard.
SOLVER_PANELS: list[SolverPanel] = [
    SolverPanel("inner_residual", "inner residual  ||⟨ψψᵀ⟩-Φ||/||Φ||", "log"),
    SolverPanel("outer_grad_rel", "outer step size  ||ΔΦ||/||Φ|| (before clipping)", "log"),
    SolverPanel("phihat_fp_resid_stein", "fixed-point residual of Φ̂  (Eq. 31b/c)", "log"),
    SolverPanel("stein_vs_direct", "Stein vs direct estimate of Φ̂  (sampler equilibrium)", "log"),
    SolverPanel(
        "train_mse_theory", "train MSE: theory (lines) vs net (dashed)", "log", None, _NET_OVERLAY
    ),
    SolverPanel(
        "eta_phihat", "adaptive step sizes  η̂ (lines), η (black)", "log", None, _ETA_OVERLAY
    ),
    SolverPanel("hmc_accept", "HMC acceptance", "linear", 0.75),
    SolverPanel("hmc_eps", "HMC step size ε", "log"),
    SolverPanel("hmc_div", "divergent transitions per run", "symlog"),
    SolverPanel("hmc_mc_err", "relative Monte Carlo error of ⟨ψψᵀ⟩", "log"),
    SolverPanel("hmc_rho1", "lag-1 autocorrelation of ||h||²/P", "linear"),
    SolverPanel("hmc_h2_ratio", "chain scale  ⟨||h||²⟩ / prior diag", "log", 1.0),
    SolverPanel("hmc_stein_mismatch", "Stein identity mismatch", "log"),
    SolverPanel("norm_margin", "normalisability margin (sufficient cond.)", "symlog", 0.0),
    SolverPanel("align_net", "kernel alignment with the trained net", "linear"),
    SolverPanel("phi_lambda_min", "λ_min(Φ) before PD floor", "symlog", 0.0),
]
SOLVER_N_COLS = 4


# --------------------------------------------------------------------------- helpers


def _np(x: ArrayLike) -> np.ndarray:
    """Tensor (any device, any grad state) or array-like -> float numpy array."""
    if isinstance(x, Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def _get_ax(ax: Axes | None, **subplot_kw) -> Axes:
    """The given axis, or a new one on a new figure."""
    if ax is None:
        _, ax = plt.subplots(**subplot_kw)
    return ax


def _floor_error(error: np.ndarray) -> np.ndarray:
    """Clip errors up to ERROR_VMIN: LogNorm masks exact zeros, which would hide perfect points,
    so zeros sit at the bottom of the log scale instead."""
    return np.maximum(error, ERROR_VMIN)


def _error_norm(error: np.ndarray) -> LogNorm:
    vmax = float(error.max())
    if vmax <= ERROR_VMIN:
        # A (near-)perfect predictor has no error above the floor; LogNorm needs vmax > vmin,
        # so show a decade above the floor rather than crash.
        vmax = ERROR_VMIN * 10
    return LogNorm(vmin=ERROR_VMIN, vmax=vmax)


def _extent(half_width: float) -> list[float]:
    return [-half_width, half_width, -half_width, half_width]


# --------------------------------------------------------------------------- learning


def plot_test_error(test_error: ArrayLike, half_width: float, ax: Axes | None = None) -> Axes:
    """Per-point test error on the (p, p) latent grid, log colour scale."""
    ax = _get_ax(ax)
    err = _floor_error(_np(test_error))
    # Transpose + origin="lower": coordinate 0 runs horizontally, as in the scatter panels.
    im = ax.imshow(
        err.T,
        cmap="jet",
        interpolation="none",
        aspect="equal",
        origin="lower",
        extent=_extent(half_width),
        norm=_error_norm(err),
    )
    ax.figure.colorbar(im, ax=ax, orientation="horizontal", label="test error")
    ax.set(xlabel="x position", ylabel="y position")
    return ax


def plot_train_error(y: ArrayLike, train_error: ArrayLike, ax: Axes | None = None) -> Axes:
    """Training targets coloured by their per-point training error (log scale)."""
    ax = _get_ax(ax)
    y_np, err = _np(y), _floor_error(_np(train_error))
    sc = ax.scatter(y_np[:, 0], y_np[:, 1], c=err, cmap="jet", norm=_error_norm(err))
    ax.figure.colorbar(sc, ax=ax, orientation="horizontal", label="train error")
    ax.set(xlabel="x position", ylabel="y position")
    return ax


def fig_learning(grid: GridPredictions, y: ArrayLike) -> Figure:
    """Test error on the grid (left) and train error at the training targets (right)."""
    fig, (ax_test, ax_train) = plt.subplots(1, 2, figsize=(12, 8))
    plot_test_error(grid.test_error, grid.half_width, ax=ax_test)
    plot_train_error(y, grid.train_error, ax=ax_train)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- representation


def plot_pca_variance(explained_variance_l: ArrayLike, ax: Axes | None = None) -> Axes:
    """Bar chart of the explained-variance ratio of each principal component of one layer."""
    ax = _get_ax(ax)
    ev = _np(explained_variance_l)
    ax.bar(np.arange(len(ev)) + 1, ev)
    ax.set(ylabel="Explained Variance Ratio", xlabel="Principal Component")
    return ax


def plot_pca_similarity(sim_ij: ArrayLike, vmax: float, ax: Axes | None = None) -> Axes:
    """(k, k) component overlaps between two layers on a symmetric scale of +-vmax."""
    ax = _get_ax(ax)
    ax.imshow(
        _np(sim_ij),
        aspect="equal",
        vmin=-vmax,
        vmax=vmax,
        cmap="coolwarm",
        interpolation="none",
    )
    return ax


def fig_representation(pca: LayerPCA) -> Figure:
    """L explained-variance bar charts (left) next to an L x L grid of similarity blocks."""
    n_layers = pca.explained_variance.shape[0]
    similarity = _np(pca.similarity)
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(n_layers, 1 + n_layers, width_ratios=[1] + [1 / n_layers] * n_layers)
    for layer in range(n_layers):
        ax = plot_pca_variance(pca.explained_variance[layer], ax=fig.add_subplot(gs[layer, 0]))
        ax.set_title(f"PC in layer {layer + 1}")
    vmax = float(np.abs(similarity).max())  # one scale so blocks are comparable
    for i in range(n_layers):
        for j in range(n_layers):
            ax = plot_pca_similarity(similarity[i, :, j, :], vmax, ax=fig.add_subplot(gs[i, 1 + j]))
            ax.set_title(f"Layer #{i + 1} vs #{j + 1}")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- neuron maps


def plot_neuron_map(map2d: ArrayLike, half_width: float, ax: Axes | None = None) -> Axes:
    """One neuron's activity over the latent grid, with its own colour scale."""
    ax = _get_ax(ax)
    m = _np(map2d)
    # Transpose + origin="lower": coordinate 0 runs horizontally, as in the scatter panels.
    im = ax.imshow(
        m.T,
        vmin=m.min(),
        vmax=m.max(),
        cmap="coolwarm",
        extent=_extent(half_width),
        origin="lower",
        interpolation="none",
        aspect="equal",
    )
    ax.figure.colorbar(im, ax=ax, orientation="horizontal")
    return ax


def fig_neuron_maps(maps: ArrayLike, half_width: float, n_cols: int, title: str) -> Figure:
    """Grid of per-neuron activity maps, `n_cols` wide; surplus axes are switched off."""
    maps = _np(maps)
    n_rows = math.ceil(len(maps) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False)
    for k, ax in enumerate(axes.ravel()):
        if k < len(maps):
            plot_neuron_map(maps[k], half_width, ax=ax)
        else:
            ax.set_axis_off()
    fig.suptitle(title, fontsize=24)
    # tight_layout ignores the suptitle; reserve the top strip for it explicitly. The strip is a
    # fraction of the figure height, so it grows with the row count that causes the overlap
    # (the title is fixed in points); on short figures tight_layout's own pad already clears it.
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# --------------------------------------------------------------------------- example trial


def _require_2d_targets(dataset: Dataset) -> None:
    if dataset.y.shape[1] != 2:
        raise ValueError(f"the target scatter is 2-D, got d={dataset.y.shape[1]}")


def plot_example_trial(dataset: Dataset, ax: Axes | None = None) -> Axes:
    """Training targets y in the plane, coloured by sample index."""
    _require_2d_targets(dataset)
    ax = _get_ax(ax)
    y = _np(dataset.y)
    sc = ax.scatter(y[:, 0], y[:, 1], c=np.arange(len(y)), cmap="viridis")
    ax.figure.colorbar(sc, ax=ax, orientation="horizontal", label="sample index")
    ax.set(xlabel="y₁", ylabel="y₂", title="target positions", aspect="equal")
    return ax


def _plot_example_embedding(x: ArrayLike, ax: Axes) -> Axes:
    """Rows = example points, columns = input dimensions."""
    im = ax.imshow(_np(x), aspect="auto", cmap="coolwarm", interpolation="none")
    ax.figure.colorbar(im, ax=ax, orientation="horizontal", label="input activity")
    ax.set(xlabel="input dimension", ylabel="example point")
    return ax


def fig_example(dataset: Dataset, cfg: DataConfig) -> Figure:
    """Task illustration needing no training: targets y and the embedding x of a few points."""
    _require_2d_targets(dataset)
    fig, (ax_y, ax_x) = plt.subplots(1, 2, figsize=(12, 5))
    plot_example_trial(dataset, ax=ax_y)
    _plot_example_embedding(dataset.x[:N_EXAMPLES], ax_x)
    ax_x.set_title(f"{cfg.embedding} embedding, D_in={cfg.D_in}")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- kernels


def plot_kernel(
    Phi_l: ArrayLike,
    ax: Axes | None = None,
    title: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Axes:
    """One (P, P) kernel matrix; pass vmin/vmax to share a colour scale across panels."""
    ax = _get_ax(ax)
    im = ax.imshow(_np(Phi_l), vmin=vmin, vmax=vmax, cmap="viridis", interpolation="none")
    ax.figure.colorbar(im, ax=ax, orientation="horizontal")
    ax.set_title(title)
    return ax


def fig_kernels(Phi_theory: ArrayLike, Phi_net: ArrayLike, shared_scale: bool = True) -> Figure:
    """L rows x 2 columns: theory kernel and network kernel.

    With `shared_scale` both panels of a row use one colour scale; without it each panel has
    its own, so a kernel orders of magnitude smaller than its partner is not rendered flat.
    Every panel title names the scale in force.
    """
    theory, net = _np(Phi_theory), _np(Phi_net)
    if theory.shape != net.shape or theory.ndim != 3:
        raise ValueError(f"need two (L, P, P) stacks, got {theory.shape} and {net.shape}")
    n_layers = len(theory)
    scale_note = "shared scale" if shared_scale else "own scale"
    fig, axes = plt.subplots(n_layers, 2, figsize=(9, 4.5 * n_layers), squeeze=False)
    for layer in range(n_layers):
        # vmin/vmax None lets imshow use the panel's own range.
        vmin = min(theory[layer].min(), net[layer].min()) if shared_scale else None
        vmax = max(theory[layer].max(), net[layer].max()) if shared_scale else None
        for ax, kernel, name in zip(axes[layer], (theory, net), ("theory", "net")):
            plot_kernel(
                kernel[layer],
                ax=ax,
                title=f"{name} layer {layer + 1} ({scale_note})",
                vmin=vmin,
                vmax=vmax,
            )
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- histories


def plot_training_history(history: Mapping[str, ArrayLike], ax: Axes | None = None) -> Axes:
    """Training loss and its logged terms against epoch on a log axis (see `plot_loss_curves`)."""
    return plot_loss_curves(history, ax=ax)


def _present_columns(history: Mapping[str, ArrayLike | None]) -> dict[str, np.ndarray]:
    """The plotted history columns as float arrays.

    Absent keys, literal None and all-None columns (e.g. `train_mse_net` when no net was
    given) mean "not recorded" and are dropped; a column with only some Nones is corrupt.
    """
    wanted = {"outer_step", "eta_phi", "train_mse_net"} | {p.key for p in SOLVER_PANELS}
    cols = {}
    for key in wanted:
        values = history.get(key)
        if values is None or len(values) == 0:
            continue
        missing = [v is None for v in values]
        if all(missing):
            continue
        if any(missing):
            raise ValueError(f"history column {key!r} has missing (None) entries")
        cols[key] = _np(values)
    return cols


def _plot_solver_panel(ax: Axes, cols: dict[str, np.ndarray], panel: SolverPanel) -> None:
    """Draw one diagnostic against the outer step: a line per layer, or a single line."""
    t, v = cols["outer_step"], cols[panel.key]
    if v.ndim == 1:
        ax.plot(t, v, lw=1.5, color=LAYER_COLORS[0])
    else:
        for layer in range(v.shape[1]):
            color = LAYER_COLORS[layer % len(LAYER_COLORS)]
            ax.plot(t, v[:, layer], lw=1.5, color=color, label=f"layer {layer}")
    overlay = panel.overlay
    if overlay is not None and overlay.key in cols:
        ax.plot(
            t,
            cols[overlay.key],
            lw=1.5,
            ls=overlay.linestyle,
            color=overlay.color,
            label=overlay.label,
        )
    if panel.hline is not None:
        ax.axhline(panel.hline, color=_GREY, lw=1, ls=":")
    ax.set_yscale(panel.yscale)
    ax.set_title(panel.title, fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, frameon=False)


def fig_solver_history(
    history: Mapping[str, ArrayLike | None], figsize: tuple[float, float] = (16, 14)
) -> Figure:
    """Dashboard of solver and sampler diagnostics; one panel per recorded key."""
    cols = _present_columns(history)
    if "outer_step" not in cols:
        raise ValueError("solver history has no 'outer_step' entries")
    panels = [p for p in SOLVER_PANELS if p.key in cols]
    if not panels:
        raise ValueError("solver history contains none of the diagnostics to plot")
    n_rows = math.ceil(len(panels) / SOLVER_N_COLS)
    fig, axes = plt.subplots(n_rows, SOLVER_N_COLS, figsize=figsize, squeeze=False)
    axes = axes.ravel()
    for ax, panel in zip(axes, panels):
        _plot_solver_panel(ax, cols, panel)
    for ax in axes[len(panels) :]:
        ax.axis("off")
    for ax in axes[-SOLVER_N_COLS:]:
        ax.set_xlabel("outer step", fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- result figures

# (history key, legend label, colour, linestyle): four distinct line styles. weight_l2 and
# rate_l2 are the raw logged penalties, not the weighted terms that enter the loss.
LOSS_STYLES = (
    ("loss", "loss (total)", LAYER_COLORS[0], "-"),
    ("mse", "mse (raw)", LAYER_COLORS[1], "--"),
    ("weight_l2", "weight L2 (raw)", LAYER_COLORS[2], ":"),
    ("rate_l2", "rate L2 (raw)", LAYER_COLORS[3], "-."),
)
MIR_KEYS = {"raw": "monitor_mir", "corrected": "monitor_mir_corrected"}
MIR_YLIM = (-0.05, 1.05)
MIR_YMARGIN = 0.05  # headroom below the lowest corrected MIR, which can go negative
# Colours of the example figures: grid and agent marks, and true vs predicted values. They are
# not in LAYER_COLORS so a legend never has to say which meaning a colour has.
AGENT_COLOR = "#c2185b"
TRUE_COLOR = "#7f7f7f"
PRED_COLOR = "#0b6e75"


def _history_columns(history: Mapping, keys: Sequence[str], what: str) -> dict[str, list]:
    """The columns `keys` of a history after checking they exist, are non-empty and share the
    length of the first key (the x axis)."""
    missing = [k for k in keys if history.get(k) is None]
    if missing:
        raise ValueError(f"history lacks {what} entries {missing}")
    cols = {k: list(history[k]) for k in keys}
    n = len(cols[keys[0]])
    if n == 0:
        raise ValueError(f"history is empty: {keys[0]!r} has no entries")
    bad = [k for k, v in cols.items() if len(v) != n]
    if bad:
        raise ValueError(f"history columns {bad} differ in length from {keys[0]!r} ({n})")
    return cols


def plot_loss_curves(history: Mapping[str, ArrayLike], ax: Axes | None = None) -> Axes:
    """The logged training terms against epoch on a log axis.

    The total loss is 0.5*N*mse + lambda_w*weight_l2 + lambda_r*rate_l2, so the three raw
    terms are drawn as logged and do not add up to the loss curve. A series that is zero at
    every epoch cannot be seen on a log axis and is marked "(zero)" in the legend instead.
    """
    keys = ["epoch", *(key for key, *_ in LOSS_STYLES)]
    cols = _history_columns(history, keys, "training")
    ax = _get_ax(ax)
    epoch = _np(cols["epoch"])
    for key, label, color, linestyle in LOSS_STYLES:
        values = _np(cols[key])
        if not np.any(values > 0):
            label += " (zero)"
        ax.plot(epoch, values, lw=1.5, color=color, ls=linestyle, label=label)
    ax.set(xlabel="epoch", ylabel="logged value (raw terms, not additive)", yscale="log")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False)
    return ax


def _layer_series(history: Mapping, keys: Sequence[str], what: str) -> dict[str, np.ndarray]:
    """Monitor columns as (n_calls, n_layers) arrays, plus 'monitor_epoch' as (n_calls,)."""
    cols = _history_columns(history, ["monitor_epoch", *keys], what)
    series = {"monitor_epoch": _np(cols["monitor_epoch"])}
    for key in keys:
        n_layers = {len(row) for row in cols[key]}
        if len(n_layers) != 1:
            raise ValueError(f"history column {key!r} has differing numbers of layers {n_layers}")
        series[key] = _np(cols[key])
    return series


def _layer_color(layer: int) -> str:
    return LAYER_COLORS[layer % len(LAYER_COLORS)]


def _layer_legend(ax: Axes, n_layers: int, extra: list[Line2D] | None = None) -> None:
    """Legend beside the axis so it never covers the curves."""
    handles = [
        Line2D([], [], color=_layer_color(i), label=f"layer {i + 1}") for i in range(n_layers)
    ]
    ax.legend(
        handles=handles + (extra or []),
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
    )


def _format_epoch_axis(ax: Axes, epoch: np.ndarray) -> None:
    """Whole-epoch ticks, and a visible window when there is only one monitor call."""
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if epoch.min() == epoch.max():
        ax.set_xlim(epoch[0] - 1, epoch[0] + 1)


def _mir_ylim(mir: np.ndarray, which: str) -> tuple[float, float]:
    """Fixed MIR range for raw values; the corrected MIR can be negative, so extend the
    lower limit to show every point."""
    if which == "raw" or not np.isfinite(mir).any():
        return MIR_YLIM
    return min(MIR_YLIM[0], float(np.nanmin(mir)) - MIR_YMARGIN), MIR_YLIM[1]


def plot_mir_history(
    history: Mapping, ax: Axes | None = None, which: str = "raw", legend: bool = True
) -> Axes:
    """Modularity ratio of every hidden layer at each monitor call.

    `which="raw"` plots `monitor_mir` with the chance baseline `monitor_mir_chance` dotted in
    each layer's colour, on the fixed range [-0.05, 1.05]. `which="corrected"` plots
    `monitor_mir_corrected`, whose lower limit extends below -0.05 if the data do.
    """
    if which not in MIR_KEYS:
        raise ValueError(f"which must be one of {sorted(MIR_KEYS)}, got {which!r}")
    keys = [MIR_KEYS[which]] + (["monitor_mir_chance"] if which == "raw" else [])
    series = _layer_series(history, keys, "monitor")
    epoch, mir = series["monitor_epoch"], series[keys[0]]
    ax = _get_ax(ax)
    for layer in range(mir.shape[1]):  # markers keep a single monitor call visible
        ax.plot(epoch, mir[:, layer], lw=1.5, marker="o", ms=3, color=_layer_color(layer))
    extra = []
    if which == "raw":
        for layer in range(mir.shape[1]):
            chance = series["monitor_mir_chance"][:, layer]
            ax.plot(epoch, chance, ls=":", lw=1.5, marker="x", ms=3, color=_layer_color(layer))
        extra = [Line2D([], [], color=_GREY, ls=":", marker="x", ms=3, label="chance baseline")]
    ax.set(xlabel="epoch", ylabel="MIR" + (" (corrected)" if which == "corrected" else ""))
    ax.set_ylim(*_mir_ylim(mir, which))
    _format_epoch_axis(ax, epoch)
    ax.grid(alpha=0.25, lw=0.5)
    if legend:
        _layer_legend(ax, mir.shape[1], extra)
    return ax


def plot_dead_neurons(history: Mapping, ax: Axes | None = None, legend: bool = True) -> Axes:
    """Number of dead neurons per layer (`monitor_n_dead_corrected`) at each monitor call."""
    series = _layer_series(history, ["monitor_n_dead_corrected"], "monitor")
    counts = series["monitor_n_dead_corrected"]
    ax = _get_ax(ax)
    for layer in range(counts.shape[1]):
        ax.plot(
            series["monitor_epoch"],
            counts[:, layer],
            lw=1.5,
            marker="o",
            ms=3,
            color=_layer_color(layer),
        )
    ax.set(xlabel="epoch", ylabel="dead neurons")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.margins(y=0.15)
    _format_epoch_axis(ax, series["monitor_epoch"])
    ax.grid(alpha=0.25, lw=0.5)
    if legend:
        _layer_legend(ax, counts.shape[1])
    return ax


def fig_mir_history(history: Mapping, title: str = "") -> Figure:
    """Raw MIR with its chance baseline, corrected MIR, and dead-neuron counts, sharing epoch."""
    fig, axes = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(8, 8),
        gridspec_kw={"height_ratios": [3, 3, 1.6]},
        layout="constrained",
    )
    plot_mir_history(history, ax=axes[0], which="raw")
    plot_mir_history(history, ax=axes[1], which="corrected", legend=False)
    plot_dead_neurons(history, ax=axes[2], legend=False)
    axes[0].set_title("raw MIR (dotted: chance baseline)", fontsize=10)
    axes[1].set_title("corrected MIR", fontsize=10)
    for ax in axes[:2]:
        ax.set_xlabel("")
    if title:
        fig.suptitle(title)
    return fig


# --------------------------------------------------------------------------- example outputs


def plot_prediction_scatter(
    y_true: ArrayLike, y_pred: ArrayLike, ax: Axes | None = None, r2: float | None = None
) -> Axes:
    """Every predicted entry against its target, with the identity line.

    `r2`, computed by the caller (see `analyse.r_squared`), is shown in the legend when given.
    """
    t, p = _np(y_true).ravel(), _np(y_pred).ravel()
    ax = _get_ax(ax)
    lim = [min(t.min(), p.min()), max(t.max(), p.max())]
    label = "identity" if r2 is None else f"identity (R² on these examples = {r2:.2f})"
    ax.plot(lim, lim, "--", color=_GREY, lw=1, label=label)
    ax.scatter(t, p, s=12, alpha=0.6, color=PRED_COLOR)
    ax.set(xlabel="true output", ylabel="predicted output", aspect="equal")
    ax.legend(frameon=False, loc="upper left")
    return ax


def plot_paired_bars(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: Sequence,
    ax: Axes | None = None,
    threshold: float | None = None,
) -> Axes:
    """True and predicted value of each output side by side; optional horizontal threshold.

    The artists carry the labels "true", "predicted" and "threshold <value>"; the caller adds
    a legend (one per figure for a grid of panels, which avoids covering any panel's title).
    """
    ax = _get_ax(ax)
    x = np.arange(len(labels))
    ax.bar(x - 0.2, _np(y_true), width=0.4, color=TRUE_COLOR, label="true")
    ax.bar(x + 0.2, _np(y_pred), width=0.4, color=PRED_COLOR, label="predicted")
    if threshold is not None:
        ax.axhline(threshold, color=_GREY, ls="--", lw=1, label=f"threshold {threshold}")
    ax.set_xticks(x, [str(v) for v in labels], fontsize=8)
    ax.axhline(0, color=_GREY, lw=0.5)
    return ax


OBJECT_HALF_SIDE = 0.28  # object squares span 0.56 of a cell, so the agent ring fits around
AGENT_RING_RADIUS = 0.42
AGENT_DOT_RADIUS = 0.1


def plot_object_grid(agent_rc: ArrayLike, objects: ArrayLike, ax: Axes | None = None) -> Axes:
    """One object_action scene: objects as squares, the agent as a ring around a dot, so an
    agent standing on an object stays visible. Row 0 is at the top: 'up' points to the top.

    All sizes are in cell units, so the marks scale with the cell size.
    """
    obj = _np(objects).astype(bool)
    agent = _np(agent_rc)
    if obj.ndim != 2 or obj.shape[0] != obj.shape[1]:
        raise ValueError(f"objects must be a (G, G) grid, got shape {obj.shape}")
    n = obj.shape[0]
    if agent.shape != (2,) or not ((agent >= 0) & (agent < n)).all():
        raise ValueError(f"agent_rc must be a (row, col) inside the {n}x{n} grid, got {agent}")
    agent_row, agent_col = agent
    ax = _get_ax(ax)
    for row, col in zip(*np.nonzero(obj)):
        ax.add_patch(
            Rectangle(
                (col - OBJECT_HALF_SIDE, row - OBJECT_HALF_SIDE),
                2 * OBJECT_HALF_SIDE,
                2 * OBJECT_HALF_SIDE,
                color=_GREY,
            )
        )
    ax.add_patch(
        Circle((agent_col, agent_row), AGENT_RING_RADIUS, fill=False, ec=AGENT_COLOR, lw=2.5)
    )
    ax.add_patch(Circle((agent_col, agent_row), AGENT_DOT_RADIUS, color=AGENT_COLOR))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)  # inverted: row 0 on top
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", lw=0.6, color="#c8c8c8")
    ax.tick_params(which="minor", length=0)
    ax.set(aspect="equal", xlabel="column", ylabel="row (0 = top)")
    return ax


def _object_grid_handles() -> list:
    return [
        Patch(color=_GREY, label="object"),
        Line2D([], [], marker="o", mfc="none", mec=AGENT_COLOR, mew=2, ls="", label="agent"),
    ]


def plot_motif_image(
    image: ArrayLike,
    ax: Axes | None = None,
    n_macro: int = N_MACRO,
    motif_size: int = MOTIF_SIZE,
) -> Axes:
    """One what_where input (n_macro*motif_size pixels square) with the macro grid drawn
    between pixel blocks: pixel centres sit on integers, so the boundaries are at
    motif_size*k - 0.5."""
    img = _np(image)
    side = n_macro * motif_size
    if img.shape != (side, side):
        raise ValueError(f"image must have shape ({side}, {side}), got {img.shape}")
    ax = _get_ax(ax)
    ax.imshow(img, cmap="Greys", vmin=0, vmax=1, interpolation="none")
    for k in range(1, n_macro):
        boundary = motif_size * k - 0.5
        ax.axvline(boundary, color=AGENT_COLOR, lw=1.2)
        ax.axhline(boundary, color=AGENT_COLOR, lw=1.2)
    centres = [motif_size * i + motif_size // 2 for i in range(n_macro)]
    ax.set_xticks(centres, range(n_macro))
    ax.set_yticks(centres, range(n_macro))
    ax.set(xlabel="macro column", ylabel="macro row")
    return ax


def plot_example_outputs(
    task: str, outputs: ExampleOutputs, ax: Axes | None = None, r2: float | None = None
) -> Axes:
    """Predicted against true entries of the example outputs on one axis.

    This view is the same for every task (`task` is only validated); the per-example pictures
    that differ between tasks are in `fig_example_outputs`.
    """
    _renderer(task)
    return plot_prediction_scatter(outputs.y_true, outputs.y_pred, ax=ax, r2=r2)


def _pooled_r2(outputs: ExampleOutputs) -> float | None:
    """R² over the shown examples; None when the targets are constant, which is possible
    for a handful of examples and should not prevent the figure."""
    try:
        return r_squared(outputs.y_true, outputs.y_pred)
    except ValueError:
        return None


def _fig_linear_teacher(outputs: ExampleOutputs, cfg: DataConfig, n: int) -> Figure:
    fig = plt.figure(figsize=(11, 1.6 * n + 1.5), layout="constrained")
    left, right = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    examples = left.subgridspec(n, 1, hspace=0.3)
    bottoms = []
    for i in range(n):
        pair = examples[i].subgridspec(2, 1, hspace=0.05)  # true row directly above predicted
        top = fig.add_subplot(pair[0])
        bottom = fig.add_subplot(pair[1], sharex=top, sharey=top)
        for ax, values, color, name in (
            (top, outputs.y_true[i], TRUE_COLOR, "true"),
            (bottom, outputs.y_pred[i], PRED_COLOR, "pred"),
        ):
            ax.stem(_np(values), linefmt=color, markerfmt="o", basefmt=_GREY)
            ax.set_ylabel(name)
            ax.margins(y=0.2)  # keep end markers off the frame
        top.tick_params(labelbottom=False)
        top.set_title(f"example {int(outputs.indices[i])}", fontsize=9, loc="left")
        bottoms.append(bottom)
    bottoms[-1].set_xlabel("output index")
    scatter_ax = fig.add_subplot(right)
    plot_prediction_scatter(outputs.y_true, outputs.y_pred, ax=scatter_ax, r2=_pooled_r2(outputs))
    return fig


def _fig_object_action(outputs: ExampleOutputs, cfg: DataConfig, n: int) -> Figure:
    if outputs.y_true.shape[1] != len(ACTION_NAMES):
        raise ValueError(f"object_action outputs need {len(ACTION_NAMES)} columns")
    agent_rc, objects = decode_object_action(outputs.x[:n], cfg.grid_size)
    fig, axes = plt.subplots(
        n,
        2,
        figsize=(8, 2.2 * n + 0.8),
        gridspec_kw={"width_ratios": [1, 1.6]},
        squeeze=False,
        layout="constrained",
    )
    for i in range(n):
        plot_object_grid(agent_rc[i], objects[i], ax=axes[i, 0])
        plot_paired_bars(
            outputs.y_true[i], outputs.y_pred[i], ACTION_NAMES, ax=axes[i, 1], threshold=0.5
        )
        axes[i, 0].set_title(f"example {int(outputs.indices[i])}", fontsize=9)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(
        handles + _object_grid_handles(),
        labels + ["object", "agent"],
        loc="outside upper center",
        ncol=5,
        frameon=False,
    )
    return fig


def _decision_text(name: str, y_true: Tensor, y_pred: Tensor) -> str:
    pred, true = int(y_pred.argmax()), int(y_true.argmax())
    return f"{name} {pred} (true {true}): {'right' if pred == true else 'WRONG'}"


def _fig_what_where(outputs: ExampleOutputs, cfg: DataConfig, n: int) -> Figure:
    images = decode_what_where(outputs.x[:n])
    where_t, what_t = split_what_where(outputs.y_true[:n], cfg.n_motifs)
    where_p, what_p = split_what_where(outputs.y_pred[:n], cfg.n_motifs)
    fig, axes = plt.subplots(
        n,
        3,
        figsize=(12, 2.4 * n + 0.8),
        gridspec_kw={"width_ratios": [1, 1.3, 1.3]},
        squeeze=False,
        layout="constrained",
    )
    for i in range(n):
        plot_motif_image(images[i], ax=axes[i, 0])
        plot_paired_bars(where_t[i], where_p[i], range(where_t.shape[1]), ax=axes[i, 1])
        plot_paired_bars(what_t[i], what_p[i], range(cfg.n_motifs), ax=axes[i, 2])
        axes[i, 0].set_title(
            f"{_decision_text('where', where_t[i], where_p[i])}\n"
            f"{_decision_text('what', what_t[i], what_p[i])}",
            fontsize=8,
        )
        axes[i, 1].set_ylabel("output")
    axes[0, 1].set_title("where (macro-cell)", fontsize=9)
    axes[0, 2].set_title("what (motif)", fontsize=9)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    return fig


def _fig_position(outputs: ExampleOutputs, cfg: DataConfig, n: int) -> Figure:
    fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
    plot_prediction_scatter(outputs.y_true, outputs.y_pred, ax=ax, r2=_pooled_r2(outputs))
    return fig


# task -> (figure builder, examples shown by default). The default counts keep each figure
# about one screen tall: scenes and bar pairs need more room per example than scatters.
_EXAMPLE_FIGURES = {
    "linear_teacher": (_fig_linear_teacher, 4),
    "object_action": (_fig_object_action, 6),
    "what_where": (_fig_what_where, 4),
    "position": (_fig_position, 4),
}


def _renderer(task: str):
    """The (builder, default n) of `task`; ValueError if the task is not in the registry or
    has no example figure."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {sorted(TASKS)}")
    if task not in _EXAMPLE_FIGURES:
        raise ValueError(f"no example figure for task {task!r}")
    return _EXAMPLE_FIGURES[task]


def _check_outputs(outputs: ExampleOutputs) -> None:
    n = len(outputs.indices)
    if n == 0:
        raise ValueError("no examples to draw: ExampleOutputs is empty")
    if not (len(outputs.x) == len(outputs.y_true) == len(outputs.y_pred) == n):
        raise ValueError("ExampleOutputs fields have different numbers of rows")


def fig_example_outputs(
    task: str, outputs: ExampleOutputs, data_cfg: DataConfig, n_show: int | None = None
) -> Figure:
    """Per-task picture of the first `n_show` examples (default 4, 6, 4, 4 for linear_teacher,
    object_action, what_where, position).

    linear_teacher: true and predicted output vectors as stems, plus a scatter of all entries.
    object_action: the scene next to true vs predicted action values. what_where: the input
    image next to true vs predicted `where` and `what` outputs, titled with the argmax
    decisions. position: true vs predicted scatter. `data_cfg` must be the config the outputs
    came from (its task, grid size and motif count are checked against the outputs).
    """
    build, default_n = _renderer(task)
    if data_cfg.task != task:
        raise ValueError(f"data_cfg is for task {data_cfg.task!r}, not {task!r}")
    _check_outputs(outputs)
    n_show = default_n if n_show is None else n_show
    if n_show < 1:
        raise ValueError(f"n_show must be at least 1, got {n_show}")
    return build(outputs, data_cfg, min(n_show, len(outputs.indices)))
