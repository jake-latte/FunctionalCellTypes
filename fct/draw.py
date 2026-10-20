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
from matplotlib.ticker import MaxNLocator
from torch import Tensor, as_tensor

from fct.analyse import (
    ExampleOutputs,
    GridPredictions,
    LayerPCA,
    decode_object_action,
    decode_what_where,
    r_squared,
)
from fct.config import DataConfig
from fct.data import Dataset
from fct.tasks import ACTION_NAMES, N_MACRO

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
    """MSE and total loss against epoch on a log axis."""
    ax = _get_ax(ax)
    epoch = _np(history["epoch"])
    for key, color in (("mse", LAYER_COLORS[0]), ("loss", LAYER_COLORS[1])):
        ax.plot(epoch, _np(history[key]), lw=1.5, color=color, label=key)
    ax.set(xlabel="epoch", yscale="log")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False)
    return ax


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

LOSS_STYLES = (  # (history key, legend label, colour, linestyle): four distinct line styles
    ("loss", "loss", LAYER_COLORS[0], "-"),
    ("mse", "mse", LAYER_COLORS[1], "--"),
    ("weight_l2", "weight L2", LAYER_COLORS[2], ":"),
    ("rate_l2", "rate L2", LAYER_COLORS[3], "-."),
)
MIR_KEYS = {"raw": "monitor_mir", "corrected": "monitor_mir_corrected"}
MIR_YLIM = (-0.05, 1.05)
N_SHOW = {"linear_teacher": 4, "object_action": 6, "what_where": 4, "position": 4}
EXAMPLE_TASKS = tuple(N_SHOW)
_TRUE_COLOR, _PRED_COLOR = LAYER_COLORS[0], LAYER_COLORS[1]


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
    """Training loss, mse and the weight and rate penalty terms against epoch, log y axis."""
    keys = ["epoch", *(key for key, *_ in LOSS_STYLES)]
    cols = _history_columns(history, keys, "training")
    ax = _get_ax(ax)
    for key, label, color, linestyle in LOSS_STYLES:
        ax.plot(_np(cols["epoch"]), _np(cols[key]), lw=1.5, color=color, ls=linestyle, label=label)
    ax.set(xlabel="epoch", ylabel="value", yscale="log")
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


def plot_mir_history(
    history: Mapping, ax: Axes | None = None, which: str = "raw", legend: bool = True
) -> Axes:
    """Modularity ratio of every hidden layer at each monitor call.

    `which="raw"` plots `monitor_mir` with the chance baseline `monitor_mir_chance` dotted in
    each layer's colour; `which="corrected"` plots `monitor_mir_corrected`.
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
            ax.plot(
                epoch, series["monitor_mir_chance"][:, layer], ls=":", lw=1.5, marker="x", ms=3,
                color=_layer_color(layer),
            )  # fmt: skip
        extra = [Line2D([], [], color=_GREY, ls=":", marker="x", ms=3, label="chance baseline")]
    ax.set(xlabel="epoch", ylabel="MIR" + (" (corrected)" if which == "corrected" else ""))
    ax.set_ylim(*MIR_YLIM)
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
            series["monitor_epoch"], counts[:, layer], lw=1.5, marker="o", ms=3,
            color=_layer_color(layer),
        )  # fmt: skip
    ax.set(xlabel="epoch", ylabel="dead neurons")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.25, lw=0.5)
    if legend:
        _layer_legend(ax, counts.shape[1])
    return ax


def fig_mir_history(history: Mapping, title: str = "") -> Figure:
    """Raw MIR with its chance baseline, corrected MIR, and dead-neuron counts, sharing epoch."""
    fig, axes = plt.subplots(
        3, 1, sharex=True, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 3, 1.6]},
        layout="constrained",
    )  # fmt: skip
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


def plot_prediction_scatter(y_true: ArrayLike, y_pred: ArrayLike, ax: Axes | None = None) -> Axes:
    """Every predicted entry against its target with the identity line and the pooled R²."""
    t, p = _np(y_true).ravel(), _np(y_pred).ravel()
    ax = _get_ax(ax)
    lim = [min(t.min(), p.min()), max(t.max(), p.max())]
    ax.plot(
        lim,
        lim,
        "--",
        color=_GREY,
        lw=1,
        label=f"identity (R² = {r_squared(as_tensor(t), as_tensor(p)):.2f})",
    )
    ax.scatter(t, p, s=12, alpha=0.6, color=_PRED_COLOR)
    ax.set(xlabel="true output", ylabel="predicted output", aspect="equal")
    ax.legend(frameon=False, loc="upper left")
    return ax


def plot_paired_bars(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: Sequence[str],
    ax: Axes | None = None,
    threshold: float | None = None,
    legend: bool = True,
) -> Axes:
    """True and predicted value of each output side by side; optional horizontal threshold."""
    ax = _get_ax(ax)
    x = np.arange(len(labels))
    ax.bar(x - 0.2, _np(y_true), width=0.4, color=_TRUE_COLOR, label="true")
    ax.bar(x + 0.2, _np(y_pred), width=0.4, color=_PRED_COLOR, label="predicted")
    if threshold is not None:
        ax.axhline(threshold, color=_GREY, ls="--", lw=1, label=f"threshold {threshold}")
    ax.set_xticks(x, labels, fontsize=8)
    ax.axhline(0, color=_GREY, lw=0.5)
    if legend:
        # Above the axes: bars of value 1 reach the top of the plot.
        ax.legend(frameon=False, fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3)
    return ax


def plot_object_grid(agent_rc: ArrayLike, objects: ArrayLike, ax: Axes | None = None) -> Axes:
    """One object_action scene: objects as squares, the agent as a circle; row 0 is at the top,
    so 'up' points toward the top of the picture."""
    obj = _np(objects).astype(bool)
    n = obj.shape[0]
    rows, cols = np.nonzero(obj)
    agent_row, agent_col = (float(v) for v in _np(agent_rc))
    ax = _get_ax(ax)
    ax.scatter(cols, rows, marker="s", s=(80 / n) ** 2, color=_GREY, label="object")
    ax.scatter([agent_col], [agent_row], marker="o", s=(50 / n) ** 2, color=LAYER_COLORS[7],
               edgecolor="white", linewidth=1, zorder=3, label="agent")  # fmt: skip
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)  # inverted: row 0 on top
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", lw=0.6, color="#c8c8c8")
    ax.tick_params(which="minor", length=0)
    ax.set(aspect="equal", xlabel="column", ylabel="row")
    return ax


def plot_motif_image(image: ArrayLike, ax: Axes | None = None) -> Axes:
    """One 9x9 what_where input with the 3x3 macro grid drawn between pixel blocks."""
    ax = _get_ax(ax)
    ax.imshow(_np(image), cmap="Greys", vmin=0, vmax=1, interpolation="none")
    for edge in range(1, N_MACRO):
        boundary = N_MACRO * edge - 0.5  # pixel centres sit on integers
        ax.axvline(boundary, color=LAYER_COLORS[7], lw=1.2)
        ax.axhline(boundary, color=LAYER_COLORS[7], lw=1.2)
    centres = [N_MACRO * i + N_MACRO // 2 for i in range(N_MACRO)]
    ax.set_xticks(centres, range(N_MACRO))
    ax.set_yticks(centres, range(N_MACRO))
    ax.set(xlabel="macro column", ylabel="macro row")
    return ax


def _check_task(task: str) -> None:
    if task not in EXAMPLE_TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {EXAMPLE_TASKS}")


def plot_example_outputs(
    task: str,
    outputs: ExampleOutputs,
    data_cfg: DataConfig,
    extras: Mapping[str, Tensor] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """One-axis summary of the example outputs: predicted against true entries, any task.

    The per-example pictures of each task need several axes: see `fig_example_outputs`.
    `data_cfg` and `extras` are accepted so both entry points share one signature.
    """
    _check_task(task)
    return plot_prediction_scatter(outputs.y_true, outputs.y_pred, ax=ax)


def _fig_linear_teacher(outputs: ExampleOutputs, n: int) -> Figure:
    fig = plt.figure(figsize=(11, 1.6 * n + 1.5), layout="constrained")
    left, right = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    examples = left.subgridspec(n, 1, hspace=0.3)
    for i in range(n):
        pair = examples[i].subgridspec(2, 1, hspace=0.05)  # true row directly above predicted
        top = fig.add_subplot(pair[0])
        bottom = fig.add_subplot(pair[1], sharex=top, sharey=top)
        for ax, values, color, name in (
            (top, outputs.y_true[i], _TRUE_COLOR, "true"),
            (bottom, outputs.y_pred[i], _PRED_COLOR, "pred"),
        ):
            ax.stem(_np(values), linefmt=color, markerfmt="o", basefmt=_GREY)
            ax.set_ylabel(name)
            ax.margins(y=0.2)  # keep end markers off the frame
        top.tick_params(labelbottom=False)
        top.set_title(f"example {int(outputs.indices[i])}", fontsize=9, loc="left")
    bottom.set_xlabel("output index")
    plot_prediction_scatter(outputs.y_true, outputs.y_pred, ax=fig.add_subplot(right))
    return fig


def _fig_object_action(outputs: ExampleOutputs, cfg: DataConfig, n: int) -> Figure:
    agent_rc, objects = decode_object_action(outputs.x[:n], cfg.grid_size)
    fig, axes = plt.subplots(
        n, 2, figsize=(8, 2.2 * n + 0.6), gridspec_kw={"width_ratios": [1, 1.6]},
        squeeze=False, layout="constrained",
    )  # fmt: skip
    for i in range(n):
        plot_object_grid(agent_rc[i], objects[i], ax=axes[i, 0])
        plot_paired_bars(
            outputs.y_true[i], outputs.y_pred[i], ACTION_NAMES, ax=axes[i, 1], threshold=0.5,
            legend=i == 0,
        )  # fmt: skip
        axes[i, 0].set_title(f"example {int(outputs.indices[i])}", fontsize=9)
    fig.suptitle(
        "squares: objects, circle: agent, row 0 at the top (up = toward row 0)", fontsize=9
    )
    return fig


def _decision_text(name: str, y_true: Tensor, y_pred: Tensor) -> str:
    pred, true = int(y_pred.argmax()), int(y_true.argmax())
    return f"{name} {pred} (true {true}): {'right' if pred == true else 'WRONG'}"


def _fig_what_where(outputs: ExampleOutputs, n: int) -> Figure:
    images = decode_what_where(outputs.x[:n])
    m = outputs.y_true.shape[1] - N_MACRO**2
    fig, axes = plt.subplots(
        n, 3, figsize=(12, 2.4 * n + 0.6), gridspec_kw={"width_ratios": [1, 1.3, 1.3]},
        squeeze=False, layout="constrained",
    )  # fmt: skip
    for i in range(n):
        plot_motif_image(images[i], ax=axes[i, 0])
        where, what = slice(0, N_MACRO**2), slice(N_MACRO**2, None)
        t, p = outputs.y_true[i], outputs.y_pred[i]
        plot_paired_bars(t[where], p[where], range(N_MACRO**2), ax=axes[i, 1], legend=i == 0)
        plot_paired_bars(t[what], p[what], range(m), ax=axes[i, 2], legend=False)
        axes[i, 0].set_title(
            f"{_decision_text('where', t[where], p[where])}\n"
            f"{_decision_text('what', t[what], p[what])}",
            fontsize=8,
        )
        axes[i, 1].set_ylabel("output")
    axes[0, 1].set_title("where (macro-cell)", fontsize=9)
    axes[0, 2].set_title("what (motif)", fontsize=9)
    return fig


def fig_example_outputs(
    task: str,
    outputs: ExampleOutputs,
    data_cfg: DataConfig,
    extras: Mapping[str, Tensor] | None = None,
    n_show: int | None = None,
) -> Figure:
    """Per-task picture of the first `n_show` examples (default per task: 4, 6, 4, 4).

    linear_teacher: true and predicted output vectors as stems, plus a scatter of all entries.
    object_action: the scene next to true vs predicted action values. what_where: the input
    image next to true vs predicted `where` and `what` outputs. position: true vs predicted
    scatter. `extras` is reserved for the dataset's `extras`; no task needs it to draw today.
    """
    _check_task(task)
    n = min(n_show or N_SHOW[task], len(outputs.indices))
    if task == "linear_teacher":
        return _fig_linear_teacher(outputs, n)
    if task == "object_action":
        return _fig_object_action(outputs, data_cfg, n)
    if task == "what_where":
        return _fig_what_where(outputs, n)
    fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
    plot_prediction_scatter(outputs.y_true, outputs.y_pred, ax=ax)
    return fig
