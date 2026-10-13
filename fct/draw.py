"""Figures: single-panel `plot_*` functions that draw into an axis, `fig_*` that assemble them.

Plotting is the outermost layer: nothing here computes beyond what is needed to draw, nothing
reads files, and nothing opens a window (the CLI saves figures from cached results on
machines without a display). A `plot_*` function creates a figure only when it is not given an
`ax`, so each one can be dropped into a larger figure unchanged.
"""

import math
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from torch import Tensor

from fct.analyse import GridPredictions, LayerPCA
from fct.config import DataConfig
from fct.data import Dataset

ERROR_VMIN = 1e-6  # lower end of the log colour scale for squared errors
N_EXAMPLES = 5  # example points shown by fig_example

LAYER_COLORS = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]  # fmt: skip
_GREY = "#52514e"

# (key, title, yscale, hline) in display order
SOLVER_PANELS: list[tuple[str, str, str, float | None]] = [
    ("inner_residual", "inner residual  ||⟨ψψᵀ⟩-Φ||/||Φ||", "log", None),
    ("outer_grad_rel", "outer step size  ||ΔΦ||/||Φ|| (before clipping)", "log", None),
    ("phihat_fp_resid_stein", "fixed-point residual of Φ̂  (Eq. 31b/c)", "log", None),
    ("stein_vs_direct", "Stein vs direct estimate of Φ̂  (sampler equilibrium)", "log", None),
    ("train_mse_theory", "train MSE: theory (lines) vs net (dashed)", "log", None),
    ("eta_phihat", "adaptive step sizes  η̂ (lines), η (black)", "log", None),
    ("hmc_accept", "HMC acceptance", "linear", 0.75),
    ("hmc_eps", "HMC step size ε", "log", None),
    ("hmc_div", "divergent transitions per run", "symlog", None),
    ("hmc_mc_err", "relative Monte Carlo error of ⟨ψψᵀ⟩", "log", None),
    ("hmc_rho1", "lag-1 autocorrelation of ||h||²/P", "linear", None),
    ("hmc_h2_ratio", "chain scale  ⟨||h||²⟩ / prior diag", "log", 1.0),
    ("hmc_stein_mismatch", "Stein identity mismatch", "log", None),
    ("norm_margin", "normalisability margin (sufficient cond.)", "symlog", 0.0),
    ("align_net", "kernel alignment with the trained net", "linear", None),
    ("phi_lambda_min", "λ_min(Φ) before PD floor", "symlog", 0.0),
]
SOLVER_N_COLS = 4


# --------------------------------------------------------------------------- helpers


def _np(x) -> np.ndarray:
    """Tensor (any device, any grad state) or array-like -> float numpy array."""
    if isinstance(x, Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def _get_ax(ax: Axes | None, **subplot_kw) -> Axes:
    """The given axis, or a new one on a new figure."""
    if ax is None:
        _, ax = plt.subplots(**subplot_kw)
    return ax


def _error_norm(error: np.ndarray) -> LogNorm:
    vmax = float(error.max())
    if vmax <= ERROR_VMIN:
        # A (near-)perfect predictor has no error above the floor; LogNorm needs vmax > vmin,
        # so show a decade above the floor rather than crash.
        vmax = ERROR_VMIN * 10
    return LogNorm(vmin=ERROR_VMIN, vmax=vmax)


def _extent(half_width: float) -> list[float]:
    return [-half_width, half_width, half_width, -half_width]


# --------------------------------------------------------------------------- learning


def plot_test_error(test_error, half_width: float, ax: Axes | None = None) -> Axes:
    """Per-point test error on the (p, p) latent grid, log colour scale."""
    ax = _get_ax(ax)
    err = _np(test_error)
    im = ax.imshow(
        err, cmap="jet", interpolation="none", aspect="equal",
        extent=_extent(half_width), norm=_error_norm(err),
    )  # fmt: skip
    ax.figure.colorbar(im, ax=ax, orientation="horizontal", label="test error")
    ax.set(xlabel="x position", ylabel="y position")
    return ax


def plot_train_error(y, train_error, ax: Axes | None = None) -> Axes:
    """Training targets coloured by their per-point training error (log scale)."""
    ax = _get_ax(ax)
    y_np, err = _np(y), _np(train_error)
    sc = ax.scatter(y_np[:, 0], y_np[:, 1], c=err, cmap="jet", norm=_error_norm(err))
    ax.figure.colorbar(sc, ax=ax, orientation="horizontal", label="train error")
    ax.set(xlabel="x position", ylabel="y position")
    return ax


def fig_learning(grid: GridPredictions, y) -> Figure:
    """Test error on the grid (left) and train error at the training targets (right)."""
    fig, (ax_test, ax_train) = plt.subplots(1, 2, figsize=(12, 8))
    plot_test_error(grid.test_error, grid.half_width, ax=ax_test)
    plot_train_error(y, grid.train_error, ax=ax_train)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- representation


def plot_pca_variance(explained_variance_l, ax: Axes | None = None) -> Axes:
    """Bar chart of the explained-variance ratio of each principal component of one layer."""
    ax = _get_ax(ax)
    ev = _np(explained_variance_l)
    ax.bar(np.arange(len(ev)) + 1, ev)
    ax.set(ylabel="Explained Variance Ratio", xlabel="Principal Component")
    return ax


def plot_pca_similarity(sim_ij, vmax: float, ax: Axes | None = None) -> Axes:
    """(k, k) component overlaps between two layers on a symmetric scale of +-vmax."""
    ax = _get_ax(ax)
    ax.imshow(_np(sim_ij), aspect="equal", vmin=-vmax, vmax=vmax, cmap="coolwarm",
              interpolation="none")  # fmt: skip
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


def plot_neuron_map(map2d, half_width: float, ax: Axes | None = None) -> Axes:
    """One neuron's activity over the latent grid, with its own colour scale."""
    ax = _get_ax(ax)
    m = _np(map2d)
    im = ax.imshow(
        m, vmin=m.min(), vmax=m.max(), cmap="coolwarm", extent=_extent(half_width),
        interpolation="none", aspect="equal",
    )  # fmt: skip
    ax.figure.colorbar(im, ax=ax, orientation="horizontal")
    return ax


def fig_neuron_maps(maps, half_width: float, n_cols: int, title: str) -> Figure:
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
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- example trial


def plot_example_trial(dataset: Dataset, ax: Axes | None = None) -> Axes:
    """Training targets y in the plane, coloured by sample index."""
    if dataset.y.shape[1] != 2:
        raise ValueError(f"the target scatter is 2-D, got d={dataset.y.shape[1]}")
    ax = _get_ax(ax)
    y = _np(dataset.y)
    sc = ax.scatter(y[:, 0], y[:, 1], c=np.arange(len(y)), cmap="viridis")
    ax.figure.colorbar(sc, ax=ax, orientation="horizontal", label="sample index")
    ax.set(xlabel="y₁", ylabel="y₂", title="target positions", aspect="equal")
    return ax


def _plot_example_embedding(x, ax: Axes) -> Axes:
    """Rows = example points, columns = input dimensions."""
    im = ax.imshow(_np(x), aspect="auto", cmap="coolwarm", interpolation="none")
    ax.figure.colorbar(im, ax=ax, orientation="horizontal", label="input activity")
    ax.set(xlabel="input dimension", ylabel="example point")
    return ax


def fig_example(dataset: Dataset, cfg: DataConfig) -> Figure:
    """Task illustration needing no training: targets y and the embedding x of a few points."""
    fig, (ax_y, ax_x) = plt.subplots(1, 2, figsize=(12, 5))
    plot_example_trial(dataset, ax=ax_y)
    _plot_example_embedding(dataset.x[:N_EXAMPLES], ax_x)
    ax_x.set_title(f"{cfg.embedding} embedding, D_in={cfg.D_in}")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- kernels


def plot_kernel(Phi_l, ax: Axes | None = None, title: str = "",
                vmin: float | None = None, vmax: float | None = None) -> Axes:  # fmt: skip
    """One (P, P) kernel matrix; pass vmin/vmax to share a colour scale across panels."""
    ax = _get_ax(ax)
    im = ax.imshow(_np(Phi_l), vmin=vmin, vmax=vmax, cmap="viridis", interpolation="none")
    ax.figure.colorbar(im, ax=ax, orientation="horizontal")
    ax.set_title(title)
    return ax


def fig_kernels(Phi_theory, Phi_net) -> Figure:
    """L rows x 2 columns: theory kernel and network kernel, sharing a colour scale per row."""
    theory, net = _np(Phi_theory), _np(Phi_net)
    if theory.shape != net.shape or theory.ndim != 3:
        raise ValueError(f"need two (L, P, P) stacks, got {theory.shape} and {net.shape}")
    n_layers = len(theory)
    fig, axes = plt.subplots(n_layers, 2, figsize=(9, 4.5 * n_layers), squeeze=False)
    for layer in range(n_layers):
        vmin = min(theory[layer].min(), net[layer].min())
        vmax = max(theory[layer].max(), net[layer].max())
        for ax, kernel, name in zip(axes[layer], (theory, net), ("theory", "net")):
            plot_kernel(kernel[layer], ax=ax, title=f"{name} layer {layer + 1}",
                        vmin=vmin, vmax=vmax)  # fmt: skip
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- histories


def plot_training_history(history: Mapping[str, Sequence[float]], ax: Axes | None = None) -> Axes:
    """MSE and total loss against epoch on a log axis."""
    ax = _get_ax(ax)
    epoch = _np(history["epoch"])
    for key, color in (("mse", LAYER_COLORS[0]), ("loss", LAYER_COLORS[1])):
        ax.plot(epoch, _np(history[key]), lw=1.5, color=color, label=key)
    ax.set(xlabel="epoch", yscale="log")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False)
    return ax


def _present_columns(history: Mapping[str, Sequence]) -> dict[str, np.ndarray]:
    """History columns as float arrays; absent keys and all-None columns are dropped."""
    return {
        k: _np(v) for k, v in history.items() if len(v) > 0 and v[0] is not None
    }  # an all-None column (e.g. train_mse_net without a net) means "not recorded"


def _plot_solver_panel(ax: Axes, cols: dict[str, np.ndarray], key: str, title: str,
                       yscale: str, hline: float | None) -> None:  # fmt: skip
    t, v = cols["outer_step"], cols[key]
    if v.ndim == 1:
        ax.plot(t, v, lw=1.5, color=LAYER_COLORS[0])
        if key == "train_mse_theory" and "train_mse_net" in cols:
            ax.plot(t, cols["train_mse_net"], lw=1.5, ls="--", color=_GREY, label="net")
    else:
        for layer in range(v.shape[1]):
            color = LAYER_COLORS[layer % len(LAYER_COLORS)]
            ax.plot(t, v[:, layer], lw=1.5, color=color, label=f"layer {layer}")
        if key == "eta_phihat" and "eta_phi" in cols:
            ax.plot(t, cols["eta_phi"], lw=1.5, color="#0b0b0b", label="η (Φ)")
    if hline is not None:
        ax.axhline(hline, color=_GREY, lw=1, ls=":")
    ax.set_yscale(yscale)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, frameon=False)


def fig_solver_history(history: Mapping[str, Sequence], figsize=(16, 14)) -> Figure:
    """Dashboard of solver and sampler diagnostics; one panel per recorded key."""
    cols = _present_columns(history)
    if "outer_step" not in cols:
        raise ValueError("solver history has no 'outer_step' entries")
    panels = [p for p in SOLVER_PANELS if p[0] in cols]
    if not panels:
        raise ValueError("solver history contains none of the diagnostics to plot")
    n_rows = math.ceil(len(panels) / SOLVER_N_COLS)
    fig, axes = plt.subplots(n_rows, SOLVER_N_COLS, figsize=figsize, squeeze=False)
    axes = axes.ravel()
    for ax, (key, title, yscale, hline) in zip(axes, panels):
        _plot_solver_panel(ax, cols, key, title, yscale, hline)
    for ax in axes[len(panels) :]:
        ax.axis("off")
    for ax in axes[-SOLVER_N_COLS:]:
        ax.set_xlabel("outer step", fontsize=8)
    fig.tight_layout()
    return fig
