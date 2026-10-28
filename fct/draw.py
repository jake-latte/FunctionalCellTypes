"""Figures: single-panel `plot_*` functions that draw into an axis, `fig_*` that assemble them.

Plotting is the outermost layer: nothing here computes beyond what is needed to draw, nothing
reads files, and nothing opens a window (the CLI saves figures from cached results on
machines without a display). A `plot_*` function creates a figure only when it is not given an
`ax`, so each one can be dropped into a larger figure unchanged.
"""

import functools
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
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
from fct.grid_report import MirRecord, check_finite_mir, check_layer
from fct.tasks import ACTION_NAMES, MOTIF_SIZE, N_MACRO
from fct.tuning import TuningCurves

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

# Fonts and line widths of the results figures, in one place. A notebook is read at full width,
# so the figures are wide and their text is large enough to read without zooming. `STYLE_RC` is
# applied while a results figure is built (see `_styled`), so every title, label, tick and legend
# in it takes its size from here and nothing below hard-codes a font size. Each `fig_*` keeps its
# own geometry (heights grow with the number of rows of panels).
FIG_WIDTH = 13.0  # inches; the width of the full-width results figures
HEATMAP_WIDTH = FIG_WIDTH + 1  # the shared colourbar sits outside the panels
ANNOTATION_SIZE = 12  # numbers written inside heatmap cells
PROVISIONAL_COLOR = "#e69f00"  # orange: distinct from viridis and from the grey of missing cells
PROVISIONAL_HATCH = "//"  # hatch pattern over the cells of a provisional heatmap row
STYLE_RC = {
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 13,
    "figure.titlesize": 18,
    "lines.linewidth": 2.5,
    "lines.markersize": 8,
    "axes.linewidth": 1.2,
}
LINE_WIDTH = STYLE_RC["lines.linewidth"]
MARKER_SIZE = STYLE_RC["lines.markersize"]


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
_BACKTRACK_OVERLAY = Overlay("n_backtrack_outer", "outer", "#0b0b0b", "-")

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
    SolverPanel(
        "n_backtrack_inner",
        "backtracking: inner (lines), outer (black)",
        "symlog",
        None,
        _BACKTRACK_OVERLAY,
    ),
    SolverPanel("hmc_accept", "HMC acceptance", "linear", 0.75),
    SolverPanel("hmc_eps", "HMC step size ε", "log"),
    SolverPanel("hmc_div", "divergent transitions per run", "symlog"),
    SolverPanel("hmc_mc_err", "relative Monte Carlo error of ⟨ψψᵀ⟩", "log"),
    SolverPanel("hmc_ess_frac", "sampler ESS / samples", "log"),
    SolverPanel("is_max_weight", "largest importance weight share", "log"),
    SolverPanel("hmc_rho1", "lag-1 autocorrelation of ||h||²/P", "linear"),
    SolverPanel("hmc_h2_ratio", "chain scale  ⟨||h||²⟩ / prior diag", "log", 1.0),
    SolverPanel("hmc_stein_mismatch", "Stein identity mismatch", "log"),
    SolverPanel("norm_margin", "normalisability margin (all-active orthant cond.)", "symlog", 0.0),
    SolverPanel("align_net", "kernel alignment with the trained net", "linear"),
    SolverPanel("n_pd_clamped", "eigenvalues raised to the PD floor", "symlog"),
    SolverPanel("phi_lambda_min", "λ_min(Φ) before PD floor", "symlog", 0.0),
]
SOLVER_N_COLS = 4
# The two groups of the live view, in dashboard order: is the saddle-point iteration
# converging, and is the sampler under it trustworthy. Panels a sampler does not record
# (all-NaN columns) are left out, so one list serves both samplers.
SOLVER_HEALTH_KEYS = (
    "inner_residual", "outer_grad_rel", "phihat_fp_resid_stein", "train_mse_theory",
    "eta_phihat", "n_backtrack_inner", "n_pd_clamped", "norm_margin", "align_net",
)  # fmt: skip
SAMPLER_HEALTH_KEYS = (
    "hmc_accept", "hmc_eps", "hmc_div", "hmc_ess_frac", "is_max_weight", "hmc_mc_err",
)  # fmt: skip
HEALTH_N_COLS = 3
HEALTH_ROW_HEIGHT = 3.2  # inches per row of panels


# --------------------------------------------------------------------------- helpers


def _styled[**P](build: Callable[P, Figure]) -> Callable[P, Figure]:
    """Build a figure under `STYLE_RC`; the text sizes are fixed when each artist is created,
    so the figure keeps them after the context is left."""

    @functools.wraps(build)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Figure:
        with plt.rc_context(STYLE_RC):
            return build(*args, **kwargs)

    return wrapper


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
    All-NaN columns are dropped too: they are the diagnostics the solver's sampler does not
    have (the HMC-only ones under importance sampling, `is_max_weight` under HMC).
    """
    wanted = {"outer_step"} | {p.key for p in SOLVER_PANELS}
    wanted |= {p.overlay.key for p in SOLVER_PANELS if p.overlay is not None}
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
        column = _np(values)
        if np.isnan(column).all():
            continue
        cols[key] = column
    return cols


def _plot_solver_panel(
    ax: Axes, cols: dict[str, np.ndarray], panel: SolverPanel, first_layer: int
) -> None:
    """Draw one diagnostic against the outer step: a line per layer (labelled from
    `first_layer`), or a single line."""
    t, v = cols["outer_step"], cols[panel.key]
    # A single point has no line to draw, so it gets a marker (the first live refresh).
    marker = "o" if len(t) == 1 else None
    if v.ndim == 1:
        ax.plot(t, v, lw=1.5, color=LAYER_COLORS[0], marker=marker)
    else:
        for layer in range(v.shape[1]):
            color = LAYER_COLORS[layer % len(LAYER_COLORS)]
            ax.plot(
                t,
                v[:, layer],
                lw=1.5,
                color=color,
                marker=marker,
                label=f"layer {layer + first_layer}",
            )
    overlay = panel.overlay
    if overlay is not None and overlay.key in cols:
        ax.plot(
            t,
            cols[overlay.key],
            lw=1.5,
            ls=overlay.linestyle,
            color=overlay.color,
            marker=marker,
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


def _fig_solver_panels(
    history: Mapping[str, ArrayLike | None],
    keys: Sequence[str] | None,
    n_cols: int,
    figsize: tuple[float, float] | None,
    first_layer: int,
) -> Figure:
    """The solver panels for `keys` (all of them if None) that the history records, with
    layers labelled from `first_layer`."""
    cols = _present_columns(history)
    if "outer_step" not in cols:
        raise ValueError("solver history has no 'outer_step' entries")
    panels = [p for p in SOLVER_PANELS if p.key in cols and (keys is None or p.key in keys)]
    if not panels:
        raise ValueError("solver history contains none of the diagnostics to plot")
    n_rows = math.ceil(len(panels) / n_cols)
    figsize = figsize or (FIG_WIDTH, HEALTH_ROW_HEIGHT * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.ravel()
    for ax, panel in zip(axes, panels):
        _plot_solver_panel(ax, cols, panel, first_layer)
    for ax in axes[len(panels) :]:
        ax.axis("off")
    for ax in axes[-n_cols:]:
        ax.set_xlabel("outer step", fontsize=8)
    fig.tight_layout()
    return fig


def fig_solver_history(
    history: Mapping[str, ArrayLike | None], figsize: tuple[float, float] = (16, 14)
) -> Figure:
    """Dashboard of solver and sampler diagnostics; one panel per recorded key. Layers are
    labelled from 0 here, as in the CLI's figure; the live-view figures count from 1."""
    return _fig_solver_panels(history, None, SOLVER_N_COLS, figsize, first_layer=0)


# --------------------------------------------------------------------------- live solver view


def fig_solver_health(history: Mapping[str, ArrayLike | None]) -> Figure:
    """Is the saddle-point iteration converging: residuals, step sizes, backtracking and the
    train MSE of the theory (against the net's, when the solve had one). Layers count from 1,
    as in `fig_layer_kernels`."""
    return _fig_solver_panels(history, SOLVER_HEALTH_KEYS, HEALTH_N_COLS, None, first_layer=1)


def fig_sampler_health(history: Mapping[str, ArrayLike | None]) -> Figure:
    """Is the sampler under it trustworthy: HMC acceptance, step size, divergences and ESS, or
    the importance ESS and largest weight share; the Monte Carlo error under either."""
    return _fig_solver_panels(history, SAMPLER_HEALTH_KEYS, HEALTH_N_COLS, None, first_layer=1)


def fig_layer_kernels(kernels: Mapping[str, ArrayLike], title: str = "") -> Figure:
    """One row of (P, P) panels per named (L, P, P) stack, one column per layer.

    Each panel has its own colour scale: Φ and Φ̂ (or a theory and a net kernel) can differ by
    orders of magnitude, and a shared scale would render the smaller one flat."""
    if not kernels:
        raise ValueError("no kernels to draw")
    stacks = {name: _np(stack) for name, stack in kernels.items()}
    shapes = {stack.shape for stack in stacks.values()}
    if len(shapes) != 1:
        raise ValueError(f"kernel stacks must have the same shape, got {sorted(shapes)}")
    if any(stack.ndim != 3 for stack in stacks.values()):
        raise ValueError(f"need (L, P, P) stacks, got {sorted(shapes)}")
    n_layers = next(iter(stacks.values())).shape[0]
    fig, axes = plt.subplots(
        len(stacks), n_layers, figsize=(3.6 * n_layers, 3.8 * len(stacks)), squeeze=False
    )
    for row, (name, stack) in zip(axes, stacks.items()):
        for layer, ax in enumerate(row):
            plot_kernel(stack[layer], ax=ax, title=f"{name}, layer {layer + 1}")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- result figures

# (history key, legend label, colour, linestyle): four distinct line styles. weight_l2 and
# rate_l2 are the raw logged penalties, not the weighted terms that enter the loss; rate_l2 is
# summed over samples, neurons and layers, as it is in the loss.
LOSS_STYLES = (
    ("loss", "loss (total)", LAYER_COLORS[0], "-"),
    ("mse", "mse (raw)", LAYER_COLORS[1], "--"),
    ("weight_l2", "weight penalty (raw)", LAYER_COLORS[2], ":"),
    ("rate_l2", "rate L2 (raw)", LAYER_COLORS[3], "-."),
)
MIR_KEYS = {"raw": "monitor_mir", "corrected": "monitor_mir_corrected"}
MIR_YLIM = (-0.05, 1.05)
MIR_YMARGIN = 0.05  # headroom below the lowest corrected MIR, which is >= 0 up to round-off
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

    The total loss is 0.5*N*mse + lambda_w/(2 beta)*weight_l2 + lambda_r*rate_l2 (rate_l2
    summed over the samples; beta = 1 for Adam), so the three raw terms are drawn as logged and do not add up to the
    loss curve. A series that is zero at every epoch cannot be seen on a log axis and is marked
    "(zero)" in the legend instead.

    The legend sits outside the axis, to its right: that assumes a constrained-layout figure
    such as `fig_loss_curves` makes; with a caller-supplied `ax` it may be clipped.
    """
    keys = ["epoch", *(key for key, *_ in LOSS_STYLES)]
    cols = _history_columns(history, keys, "training")
    ax = _get_ax(ax)
    epoch = _np(cols["epoch"])
    for key, label, color, linestyle in LOSS_STYLES:
        values = _np(cols[key])
        if not np.any(values > 0):
            label += " (zero)"
        ax.plot(epoch, values, lw=LINE_WIDTH, color=color, ls=linestyle, label=label)
    ax.set(xlabel="epoch", ylabel="logged value (raw terms, not additive)", yscale="log")
    ax.grid(alpha=0.25, lw=0.5)
    # Beside the axis, so it never covers a curve.
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    return ax


@_styled
def fig_loss_curves(history: Mapping[str, ArrayLike], title: str = "") -> Figure:
    """`plot_loss_curves` as a full-width figure with an optional title."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 5.5), layout="constrained")
    plot_loss_curves(history, ax=ax)
    ax.set_title(title)
    return fig


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


def _layer_legend(
    ax: Axes,
    n_layers: int,
    extra: list[Line2D] | None = None,
    layers: Sequence[int] | None = None,
) -> None:
    """Legend beside the axis so it never covers the curves; lists `layers` (default all)."""
    handles = [
        Line2D([], [], color=_layer_color(i), label=f"layer {i + 1}")
        for i in (range(n_layers) if layers is None else layers)
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
    """Fixed MIR range for raw values. The corrected MIR is >= 0 up to round-off, so its range
    is the same; the lower limit is still extended defensively to show every point."""
    if which == "raw" or not np.isfinite(mir).any():
        return MIR_YLIM
    return min(MIR_YLIM[0], float(np.nanmin(mir)) - MIR_YMARGIN), MIR_YLIM[1]


def plot_mir_history(
    history: Mapping, ax: Axes | None = None, which: str = "raw", legend: bool = True
) -> Axes:
    """Modularity ratio of every hidden layer at each monitor call.

    `which="raw"` plots `monitor_mir` with the chance baseline `monitor_mir_chance` dotted in
    each layer's colour, on the fixed range [-0.05, 1.05]. `which="corrected"` plots
    `monitor_mir_corrected` on the same range (the corrected MIR is >= 0 up to round-off; the
    lower limit extends below -0.05 only if the data do).
    """
    if which not in MIR_KEYS:
        raise ValueError(f"which must be one of {sorted(MIR_KEYS)}, got {which!r}")
    keys = [MIR_KEYS[which]] + (["monitor_mir_chance"] if which == "raw" else [])
    series = _layer_series(history, keys, "monitor")
    epoch, mir = series["monitor_epoch"], series[keys[0]]
    ax = _get_ax(ax)
    for layer in range(mir.shape[1]):  # markers keep a single monitor call visible
        ax.plot(epoch, mir[:, layer], lw=LINE_WIDTH, marker="o", color=_layer_color(layer))
    extra = []
    if which == "raw":
        for layer in range(mir.shape[1]):
            chance = series["monitor_mir_chance"][:, layer]
            ax.plot(epoch, chance, ls=":", lw=LINE_WIDTH, marker="x", color=_layer_color(layer))
        extra = [Line2D([], [], color=_GREY, ls=":", marker="x", label="chance baseline")]
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
            lw=LINE_WIDTH,
            marker="o",
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


@_styled
def fig_mir_history(history: Mapping, title: str = "") -> Figure:
    """Raw MIR with its chance baseline, corrected MIR, and dead-neuron counts, sharing epoch."""
    fig, axes = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(FIG_WIDTH, 10),
        gridspec_kw={"height_ratios": [3, 3, 1.6]},
        layout="constrained",
    )
    plot_mir_history(history, ax=axes[0], which="raw")
    plot_mir_history(history, ax=axes[1], which="corrected", legend=False)
    plot_dead_neurons(history, ax=axes[2], legend=False)
    axes[0].set_title("raw MIR (dotted: chance baseline)")
    axes[1].set_title("corrected MIR")
    axes[2].set_title("dead neurons (corrected estimate)")
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
    The legend sits below the axis, which assumes a constrained-layout figure (the styled
    `fig_*` functions); with a caller-supplied `ax` it may be clipped.
    """
    t, p = _np(y_true).ravel(), _np(y_pred).ravel()
    ax = _get_ax(ax)
    lim = [min(t.min(), p.min()), max(t.max(), p.max())]
    label = "identity" if r2 is None else f"identity (R² on these examples = {r2:.2f})"
    ax.plot(lim, lim, "--", color=_GREY, lw=1.5, label=label)
    ax.scatter(t, p, s=36, alpha=0.7, color=PRED_COLOR)
    ax.set(xlabel="true output", ylabel="predicted output", aspect="equal")
    # Below the axis: a long R² label would run over the points inside a narrow panel.
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
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
        ax.axhline(threshold, color=_GREY, ls="--", lw=1.5, label=f"threshold {threshold}")
    ax.set_xticks(x, [str(v) for v in labels])
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
        Circle((agent_col, agent_row), AGENT_RING_RADIUS, fill=False, ec=AGENT_COLOR, lw=3)
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


def _plot_output_vector(true: ArrayLike, pred: ArrayLike, ax: Axes) -> Axes:
    """True and predicted value of every output index on one axis: the true value as a hollow
    ring on a stem from zero, the prediction as a filled diamond at the same x, so the gap
    between them is read directly and a good prediction sits inside its ring."""
    t, p = _np(true), _np(pred)
    x = np.arange(len(t))
    ax.vlines(x, 0, t, color=TRUE_COLOR, lw=LINE_WIDTH)
    ax.axhline(0, color=_GREY, lw=1)
    ax.plot(
        x, t, "o", ms=1.75 * MARKER_SIZE, mfc="none", mec=TRUE_COLOR, mew=LINE_WIDTH, label="true"
    )
    ax.plot(x, p, "D", color=PRED_COLOR, label="predicted")
    ax.set_xticks(x)
    ax.margins(y=0.25)  # keep end markers off the frame
    return ax


def _legend_and_title(
    fig: Figure, handles: Sequence[Artist], labels: Sequence[str], ncol: int, title: str
) -> None:
    """The figure legend on top, or, when there is a title, below the figure: constrained layout
    lets a title and a legend outside the axes overlap at the top."""
    if title:
        fig.suptitle(title)
    where = "lower" if title else "upper"
    fig.legend(handles, labels, loc=f"outside {where} center", ncol=ncol, frameon=False)


def _fig_linear_teacher(outputs: ExampleOutputs, cfg: DataConfig, n: int, title: str) -> Figure:
    fig = plt.figure(figsize=(FIG_WIDTH, 2.4 * n + 1.5), layout="constrained")
    left, right = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    rows = left.subgridspec(n, 1)
    example_axes: list[Axes] = []
    for i in range(n):
        # One shared y scale, so examples are comparable; x is the output index.
        first = example_axes[0] if example_axes else None
        ax = fig.add_subplot(rows[i], sharex=first, sharey=first)
        _plot_output_vector(outputs.y_true[i], outputs.y_pred[i], ax)
        ax.set_title(f"example {int(outputs.indices[i])}", loc="left")
        ax.set_ylabel("output value")
        if i < n - 1:
            ax.tick_params(labelbottom=False)
        example_axes.append(ax)
    example_axes[-1].set_xlabel("output index")
    handles, labels = example_axes[0].get_legend_handles_labels()
    _legend_and_title(fig, handles, labels, 2, title)
    plot_prediction_scatter(
        outputs.y_true, outputs.y_pred, ax=fig.add_subplot(right), r2=_pooled_r2(outputs)
    )
    return fig


def _fig_object_action(outputs: ExampleOutputs, cfg: DataConfig, n: int, title: str) -> Figure:
    if outputs.y_true.shape[1] != len(ACTION_NAMES):
        raise ValueError(f"object_action outputs need {len(ACTION_NAMES)} columns")
    agent_rc, objects = decode_object_action(outputs.x[:n], cfg.grid_size)
    fig, axes = plt.subplots(
        n,
        2,
        figsize=(FIG_WIDTH, 2.9 * n + 1.0),
        gridspec_kw={"width_ratios": [1, 1.8]},
        squeeze=False,
        layout="constrained",
    )
    for i in range(n):
        plot_object_grid(agent_rc[i], objects[i], ax=axes[i, 0])
        plot_paired_bars(
            outputs.y_true[i], outputs.y_pred[i], ACTION_NAMES, ax=axes[i, 1], threshold=0.5
        )
        axes[i, 0].set_title(f"example {int(outputs.indices[i])}")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    _legend_and_title(fig, handles + _object_grid_handles(), labels + ["object", "agent"], 5, title)
    return fig


def _decision_text(name: str, y_true: Tensor, y_pred: Tensor) -> str:
    pred, true = int(y_pred.argmax()), int(y_true.argmax())
    return f"{name} {pred} (true {true}): {'right' if pred == true else 'WRONG'}"


def _fig_what_where(outputs: ExampleOutputs, cfg: DataConfig, n: int, title: str) -> Figure:
    images = decode_what_where(outputs.x[:n])
    where_t, what_t = split_what_where(outputs.y_true[:n], cfg.n_motifs)
    where_p, what_p = split_what_where(outputs.y_pred[:n], cfg.n_motifs)
    fig, axes = plt.subplots(
        n,
        3,
        figsize=(FIG_WIDTH, 3.0 * n + 1.2),
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
            fontsize="small",  # two lines of text per panel title
        )
        axes[i, 1].set_ylabel("output")
    axes[0, 1].set_title("where (macro-cell)")
    axes[0, 2].set_title("what (motif)")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    _legend_and_title(fig, handles, labels, 2, title)
    return fig


def _fig_position(outputs: ExampleOutputs, cfg: DataConfig, n: int, title: str) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 7), layout="constrained")
    plot_prediction_scatter(outputs.y_true, outputs.y_pred, ax=ax, r2=_pooled_r2(outputs))
    if title:
        fig.suptitle(title)
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


@_styled
def fig_example_outputs(
    task: str,
    outputs: ExampleOutputs,
    data_cfg: DataConfig,
    n_show: int | None = None,
    title: str = "",
) -> Figure:
    """Per-task picture of the first `n_show` examples (default 4, 6, 4, 4 for linear_teacher,
    object_action, what_where, position).

    linear_teacher: one axis per example with the true (ring) and predicted (diamond) output
    vector on the same axis, plus a scatter of all entries.
    object_action: the scene next to true vs predicted action values. what_where: the input
    image next to true vs predicted `where` and `what` outputs, titled with the argmax
    decisions. position: true vs predicted scatter. `data_cfg` must be the config the outputs
    came from (its task, grid size and motif count are checked against the outputs). A `title`
    heads the figure and moves the legend to its bottom.
    """
    build, default_n = _renderer(task)
    if data_cfg.task != task:
        raise ValueError(f"data_cfg is for task {data_cfg.task!r}, not {task!r}")
    _check_outputs(outputs)
    n_show = default_n if n_show is None else n_show
    if n_show < 1:
        raise ValueError(f"n_show must be at least 1, got {n_show}")
    return build(outputs, data_cfg, min(n_show, len(outputs.indices)), title)


# --------------------------------------------------------------------------- final MIR heatmaps
_MISSING_GREY = "#d9d9d9"  # cells with no value: a layer the setting lacks, or an unfinished run


def _add_mir_colorbar(fig: Figure, image, axes: Axes | Sequence[Axes], **kwargs) -> None:
    """The one colourbar code path of the MIR heatmaps."""
    fig.colorbar(image, ax=axes, label="MIR", **kwargs)


def _annotate_cells(
    ax: Axes, values: np.ndarray, shades: np.ndarray, vmin: float, vmax: float
) -> None:
    """Write each finite value in its cell, in white on dark and black on light cells; `shades`
    are the values the cells are coloured by (they differ from `values` only where a
    provisional value is clipped for colour)."""
    middle = (vmin + vmax) / 2
    for (row, col), value in np.ndenumerate(values):
        if np.isfinite(value):
            colour = "white" if shades[row, col] < middle else "black"
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=ANNOTATION_SIZE,
                color=colour,
            )


PROVISIONAL = "provisional"  # row status of a running setting whose cells hold checkpoint values
PROVISIONAL_NOTE = (
    "hatched rows (*) are provisional: values from the latest checkpoint, not final.\n"
    "Their colours are clipped to the scale of the finished rows; the printed numbers are exact.\n"
    "MIR@m: epoch of the last MIR measurement; ckpt e/N: checkpoint epoch of N."
)


def _label_unfinished_rows(ax: Axes, row_status: Sequence[str], n_cols: int) -> None:
    """Name the status ("partial", "missing", "stale") across the row of an unfinished setting
    that has no values; a provisional row keeps its values (see `_hatch_provisional_rows`)."""
    for row, status in enumerate(row_status):
        if status not in ("done", PROVISIONAL):
            ax.text(
                (n_cols - 1) / 2,
                row,
                status,
                ha="center",
                va="center",
                fontsize=ANNOTATION_SIZE,
                style="italic",
                color=_GREY,
            )


def _hatch_provisional_rows(ax: Axes, row_status: Sequence[str], n_cols: int) -> None:
    """Hatch the cells of provisional rows, so a number from a running setting is never read
    as a final one."""
    for row, status in enumerate(row_status):
        if status == PROVISIONAL:
            ax.add_patch(
                Rectangle(
                    (-0.5, row - 0.5),  # imshow cells are centred on integers
                    n_cols,
                    1,
                    fill=False,
                    hatch=PROVISIONAL_HATCH,
                    edgecolor=PROVISIONAL_COLOR,
                    lw=0,
                )
            )


def _draw_mir_cells(
    ax: Axes,
    values: np.ndarray,
    shades: np.ndarray,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    vmin: float,
    vmax: float,
    origin: str = "upper",
) -> AxesImage:
    """Cells of a MIR heatmap: `shades` coloured on [vmin, vmax] (NaN in the missing-cell grey),
    labelled ticks, each finite value written in its cell. Returns the image for a colourbar."""
    cmap = plt.get_cmap("viridis").with_extremes(bad=_MISSING_GREY)
    image = ax.imshow(
        np.ma.masked_invalid(shades), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin=origin
    )
    ax.set_xticks(range(len(x_labels)), list(x_labels))
    ax.set_yticks(range(len(y_labels)), list(y_labels))
    ax.tick_params(length=0)
    _annotate_cells(ax, values, shades, vmin, vmax)
    return image


def _mir_colour_range(arrays: Iterable[np.ndarray]) -> tuple[float, float]:
    """Shared MIR colour scale [min(0, data), max(1, data)] over the finite entries, so a value
    outside [0, 1] (round-off) is shown, not clipped."""
    finite = np.concatenate([a[np.isfinite(a)] for a in arrays])
    if not finite.size:
        return 0.0, 1.0
    return min(0.0, float(finite.min())), max(1.0, float(finite.max()))


def _check_in_range(values: np.ndarray, vmin: float, vmax: float) -> None:
    finite = values[np.isfinite(values)]
    if finite.size and (finite.min() < vmin or finite.max() > vmax):
        raise ValueError(
            f"values span [{finite.min():.3f}, {finite.max():.3f}], outside the colour range "
            f"[{vmin}, {vmax}]; they would be clipped"
        )


def plot_mir_heatmap(
    values: ArrayLike,
    row_labels: Sequence[str],
    row_status: Sequence[str],
    title: str = "",
    ax: Axes | None = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    colorbar: bool = True,
) -> Axes:
    """Settings (rows) by layer (columns) heatmap of a final MIR array, NaN where there is no
    value (grey). Each cell carries its value and an unfinished setting's row its status; a row
    with status "provisional" keeps its values and is hatched.
    A finished value outside [vmin, vmax] is an error, not silently clipped; a provisional
    row's colours are clipped to the range (its printed numbers are not), so a running setting
    cannot dictate the scale. A figure that shows
    several heatmaps on one scale passes `colorbar=False` and adds one for all of them."""
    values = _np(values)
    if values.ndim != 2 or not values.shape[0] == len(row_labels) == len(row_status):
        raise ValueError(
            f"values {values.shape} do not match {len(row_labels)} labels / "
            f"{len(row_status)} statuses"
        )
    provisional = np.array([s == PROVISIONAL for s in row_status], dtype=bool)
    _check_in_range(values[~provisional], vmin, vmax)
    shades = np.where(provisional[:, None], np.clip(values, vmin, vmax), values)
    ax = _get_ax(ax, figsize=(7, 6))
    layer_labels = [f"layer {i + 1}" for i in range(values.shape[1])]
    image = _draw_mir_cells(ax, values, shades, layer_labels, row_labels, vmin, vmax)
    _label_unfinished_rows(ax, row_status, values.shape[1])
    _hatch_provisional_rows(ax, row_status, values.shape[1])
    ax.set_title(title)
    if colorbar:
        _add_mir_colorbar(ax.figure, image, ax, fraction=0.046, pad=0.03)
    return ax


@_styled
def fig_mir_heatmaps(
    panels: Mapping[str, ArrayLike],
    row_labels: Sequence[str],
    row_status: Sequence[str],
    title: str = "",
) -> Figure:
    """One heatmap per named panel side by side on a shared colour scale and colourbar. The
    scale is [min(0, data), max(1, data)] over the finished rows, so a finished value outside
    [0, 1] (round-off) is shown, not clipped; provisional rows do not enter the scale."""
    arrays = {name: _np(a) for name, a in panels.items()}
    finished = np.array([s != PROVISIONAL for s in row_status], dtype=bool)
    vmin, vmax = _mir_colour_range([a[finished] for a in arrays.values()])
    fig, axes = plt.subplots(
        1,
        len(arrays),
        figsize=(HEATMAP_WIDTH, 0.55 * len(row_labels) + 2.6),
        sharey=True,
        layout="constrained",
    )
    axes = list(np.atleast_1d(axes))
    for ax, (name, values) in zip(axes, arrays.items(), strict=True):
        plot_mir_heatmap(values, row_labels, row_status, name, ax, vmin, vmax, colorbar=False)
    _add_mir_colorbar(fig, axes[-1].images[0], axes, shrink=0.6)
    if title:
        fig.suptitle(title)
    if PROVISIONAL in row_status:
        fig.supxlabel(PROVISIONAL_NOTE)
    return fig


# --------------------------------------------------------------------------- tuning and grid MIR
TUNING_N_COLS = 4  # tuning panels per row
# One colour per latent, none of them in LAYER_COLORS: a page can show layer k and latent k.
LATENT_COLORS = ["#7b3294", "#8c510a", "#00a6b5", "#5c5c5c", "#b8a800", "#000000"]


def _latent_color(latent: int) -> str:
    return LATENT_COLORS[latent % len(LATENT_COLORS)]


def _check_positive(values: np.ndarray, what: str) -> None:
    if not (np.isfinite(values) & (values > 0)).all():
        raise ValueError(f"{what} must all be positive and finite for a log axis, got {values}")


def plot_neuron_tuning(
    curves: TuningCurves,
    neuron: int,
    mir: float,
    latent_names: Sequence[str],
    ax: Axes | None = None,
) -> Axes:
    """One neuron's tuning: a mean rate (+- standard error) curve per latent against the
    latent's bin centre; the title is the neuron index and its MIR."""
    mean, sem, centres = _np(curves.mean), _np(curves.sem), _np(curves.centres)
    if not 0 <= neuron < mean.shape[0]:
        raise ValueError(f"neuron {neuron} is outside [0, {mean.shape[0]})")
    if len(latent_names) != mean.shape[1]:
        raise ValueError(
            f"latent_names has {len(latent_names)} entries for {mean.shape[1]} latents"
        )
    ax = _get_ax(ax)
    for k, name in enumerate(latent_names):
        ax.errorbar(
            centres[k],
            mean[neuron, k],
            yerr=sem[neuron, k],
            color=_latent_color(k),
            marker="o",
            capsize=2,
            label=name,
        )
    ax.set_title(f"neuron {neuron}, MIR {mir:.2f}")
    ax.grid(alpha=0.25, lw=0.5)
    return ax


@_styled
def fig_tuning(
    curves: TuningCurves,
    neuron_indices: Sequence[int],
    mir_per_neuron: ArrayLike,
    latent_names: Sequence[str],
) -> Figure:
    """One panel per chosen neuron, one curve per latent, a common rate axis. Latents have
    their own scales, so the x axis is each curve's own bin centres ("latent value")."""
    mir = _np(mir_per_neuron)
    n_neurons = curves.mean.shape[0]
    if len(neuron_indices) == 0:
        raise ValueError("neuron_indices is empty")
    if len(mir) != n_neurons:
        raise ValueError(f"mir_per_neuron has {len(mir)} entries for {n_neurons} neurons")
    if not all(0 <= i < n_neurons for i in neuron_indices):
        raise ValueError(f"neuron_indices must lie in [0, {n_neurons}), got {list(neuron_indices)}")
    n_cols = min(TUNING_N_COLS, len(neuron_indices))
    n_rows = math.ceil(len(neuron_indices) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(FIG_WIDTH, 3.2 * n_rows + 0.8),
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    flat = list(axes.ravel())
    for ax, neuron in zip(flat, neuron_indices, strict=False):
        plot_neuron_tuning(curves, neuron, mir[neuron], latent_names, ax)
    for ax in flat[len(neuron_indices) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("latent value")
    for ax in axes[:, 0]:
        ax.set_ylabel("mean rate")
    flat[0].legend(loc="best", frameon=False)
    return fig


def _log_tick_label(value: float) -> str:
    """10^k for a power of ten, else three significant figures."""
    exponent = math.log10(value)
    if math.isclose(exponent, round(exponent)):
        return f"$10^{{{round(exponent)}}}$"
    return f"{value:.3g}"


def plot_mir_matrix(
    values: ArrayLike,
    lambda_r_values: ArrayLike,
    lambda_w_values: ArrayLike,
    title: str = "",
    ax: Axes | None = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> Axes:
    """One layer's MIR over the (lambda_r, lambda_w) grid; `values` is (n_r, n_w), NaN cells grey.

    It is a phase diagram: both lambdas increase from the bottom-left (`origin="lower"`),
    lambda_w along x and lambda_r up y, ticks labelled in powers of ten. The image is on
    `ax.images[0]` for a colourbar.
    """
    values, lam_r, lam_w = _np(values), _np(lambda_r_values), _np(lambda_w_values)
    _check_positive(lam_r, "lambda_r_values")
    _check_positive(lam_w, "lambda_w_values")
    if values.shape != (len(lam_r), len(lam_w)):
        raise ValueError(f"values {values.shape} do not match the grid {(len(lam_r), len(lam_w))}")
    _check_in_range(values, vmin, vmax)
    ax = _get_ax(ax, figsize=(7, 6))
    _draw_mir_cells(
        ax,
        values,
        values,
        [_log_tick_label(v) for v in lam_w],
        [_log_tick_label(v) for v in lam_r],
        vmin,
        vmax,
        origin="lower",
    )
    ax.set(xlabel=r"$\lambda_w$", ylabel=r"$\lambda_r$", title=title)
    return ax


@_styled
def fig_mir_matrix(
    lambda_r_values: ArrayLike, lambda_w_values: ArrayLike, mir_by_layer: ArrayLike
) -> Figure:
    """One `plot_mir_matrix` per hidden layer, `mir_by_layer` (L, n_r, n_w), on one shared
    colour scale (`_mir_colour_range`) and colourbar."""
    mir = _np(mir_by_layer)
    n_r, n_w = len(_np(lambda_r_values)), len(_np(lambda_w_values))
    if mir.ndim != 3 or mir.shape[1:] != (n_r, n_w):
        raise ValueError(f"mir_by_layer must have shape (L, {n_r}, {n_w}), got {mir.shape}")
    vmin, vmax = _mir_colour_range([mir])
    fig, axes = plt.subplots(
        1,
        len(mir),
        figsize=(HEATMAP_WIDTH, 0.5 * n_r + 3.2),
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    axes = list(axes[0])
    for layer, (ax, values) in enumerate(zip(axes, mir, strict=True)):
        plot_mir_matrix(
            values, lambda_r_values, lambda_w_values, f"layer {layer + 1}", ax, vmin, vmax
        )
        if layer:
            ax.set_ylabel("")
    _add_mir_colorbar(fig, axes[-1].images[0], axes, shrink=0.6)
    return fig


def plot_mir_vs_ratio(
    records: Iterable[MirRecord], ax: Axes | None = None, legend: bool = True
) -> Axes:
    """MIR of every network (one dot per record) against lambda_r / lambda_w on a log axis,
    coloured by hidden layer."""
    records = list(records)
    if not records:
        raise ValueError("no records")
    n_layers = max(rec.layer for rec in records) + 1
    for lr, lw, layer, value in records:
        check_layer(layer, n_layers)
        check_finite_mir(value, lr, lw)
    lam_r, lam_w, layer, mir = (np.array(col, dtype=float) for col in zip(*records, strict=True))
    _check_positive(lam_r, "lambda_r")
    _check_positive(lam_w, "lambda_w")
    ax = _get_ax(ax, figsize=(8, 5))
    present = np.unique(layer).astype(int)
    for each in present:
        keep = layer == each
        ax.scatter(
            lam_r[keep] / lam_w[keep],
            mir[keep],
            s=MARKER_SIZE**2,
            alpha=0.8,
            color=_layer_color(each),
        )
    ax.set_xscale("log")
    ax.set(xlabel=r"$\lambda_r / \lambda_w$", ylabel="MIR")
    ax.grid(alpha=0.25, lw=0.5)
    if legend:
        _layer_legend(ax, n_layers, layers=present.tolist())
    return ax


@_styled
def fig_mir_vs_ratio(records: Iterable[MirRecord]) -> Figure:
    """`plot_mir_vs_ratio` on its own figure."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 0.6, 5), layout="constrained")
    plot_mir_vs_ratio(records, ax)
    return fig
