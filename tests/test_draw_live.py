"""The figures of the live solver view: kernel stacks, solver health and sampler health, drawn
from synthetic solver histories that carry exactly the key sets either sampler produces."""

import numpy as np
import pytest
import torch

from fct import draw
from fct.meanfield import HISTORY_KEYS, NET_KEYS
from tests.draw_helpers import assert_png

L, P = 2, 5
HMC_ONLY = ("hmc_accept", "hmc_eps", "hmc_div", "hmc_reset", "hmc_max_abs_dH", "hmc_rho1")
SCALAR_KEYS = {
    "outer_step", "time", "inner_steps", "inner_converged", "eta_phi", "n_backtrack_outer",
    "trace_term", "learning_term", "train_mse_theory", "train_mse_net",
}  # fmt: skip


def synthetic_history(sampler: str, n_steps: int = 4, with_net: bool = True) -> dict:
    """A full solver history: random positive values, the keys the other sampler lacks NaN,
    per-layer columns (n_steps, L) and scalar ones (n_steps,), as `solve` records them."""
    rng = np.random.default_rng(0)
    history: dict = {}
    for key in HISTORY_KEYS:
        if key in NET_KEYS and not with_net:
            continue
        if key == "outer_step":
            history[key] = list(range(n_steps))
        elif key in SCALAR_KEYS:
            history[key] = list(rng.random(n_steps) + 0.1)
        else:
            history[key] = list(rng.random((n_steps, L)) + 0.1)
    if not with_net:
        history["train_mse_net"] = [None] * n_steps
    nan = [np.full(L, np.nan)] * n_steps
    for key in HMC_ONLY if sampler == "importance" else ("is_max_weight",):
        history[key] = nan
    return history


def titles(fig) -> list[str]:
    return [ax.get_title() for ax in fig.axes if ax.axison]


@pytest.fixture
def kernels():
    return torch.rand(L, P, P, generator=torch.Generator().manual_seed(1))


# --------------------------------------------------------------------------- kernel stacks
def test_layer_kernels_has_a_row_per_name_and_a_column_per_layer(kernels, tmp_path):
    fig = draw.fig_layer_kernels({"Φ": kernels, "Φ̂": -kernels}, title="step 3")
    images = [ax for ax in fig.axes if ax.images]
    assert len(images) == 2 * L
    assert fig.get_suptitle() == "step 3"
    assert_png(fig, tmp_path)


def test_layer_kernels_colour_scales_are_per_panel(kernels):
    """Φ and Φ̂ differ by orders of magnitude, so a shared scale would render one flat."""
    fig = draw.fig_layer_kernels({"Φ": kernels, "Φ̂": 1e-6 * kernels})
    limits = [ax.images[0].get_clim() for ax in fig.axes if ax.images]
    assert limits[0] != limits[L]


def test_layer_kernels_rejects_non_stacks_and_mismatched_rows(kernels):
    with pytest.raises(ValueError, match=r"\(L, P, P\)"):
        draw.fig_layer_kernels({"Φ": kernels[0]})
    with pytest.raises(ValueError, match="same shape"):
        draw.fig_layer_kernels({"Φ": kernels, "Φ̂": kernels[:1]})
    with pytest.raises(ValueError, match="no kernels"):
        draw.fig_layer_kernels({})


# --------------------------------------------------------------------------- health figures
@pytest.mark.parametrize("with_net", [False, True])
@pytest.mark.parametrize("sampler", ["hmc", "importance"])
def test_solver_health_shows_the_same_panels_under_either_sampler(sampler, with_net, tmp_path):
    fig = draw.fig_solver_health(synthetic_history(sampler, with_net=with_net))
    shown = titles(fig)
    for word in ("inner residual", "fixed-point", "train MSE", "step sizes", "backtracking"):
        assert any(word in t for t in shown), word
    assert not any("ESS" in t or "HMC" in t for t in shown)
    assert_png(fig, tmp_path)


def test_solver_health_overlays_the_net_mse_only_when_there_is_a_net():
    def mse_axis(fig):
        return next(ax for ax in fig.axes if "train MSE" in ax.get_title())

    with_net = mse_axis(draw.fig_solver_health(synthetic_history("hmc", with_net=True)))
    without = mse_axis(draw.fig_solver_health(synthetic_history("hmc", with_net=False)))
    assert len(with_net.lines) == len(without.lines) + 1


def test_sampler_health_of_hmc(tmp_path):
    fig = draw.fig_sampler_health(synthetic_history("hmc"))
    shown = titles(fig)
    for word in ("HMC acceptance", "HMC step size", "divergent", "ESS"):
        assert any(word in t for t in shown), word
    assert not any("importance weight" in t for t in shown)
    assert_png(fig, tmp_path)


def test_sampler_health_of_importance_sampling_drops_the_nan_hmc_columns(tmp_path):
    fig = draw.fig_sampler_health(synthetic_history("importance"))
    shown = titles(fig)
    assert any("ESS" in t for t in shown) and any("importance weight" in t for t in shown)
    assert not any("HMC" in t or "divergent" in t for t in shown)
    assert_png(fig, tmp_path)


@pytest.mark.parametrize("figure", [draw.fig_solver_health, draw.fig_sampler_health])
@pytest.mark.parametrize("sampler", ["hmc", "importance"])
def test_health_figures_draw_after_a_single_outer_step(figure, sampler, tmp_path):
    """The first live refresh has one point per line; it must still be visible and not fail."""
    fig = figure(synthetic_history(sampler, n_steps=1))
    assert all(ax.lines[0].get_marker() != "None" for ax in fig.axes if ax.axison and ax.lines)
    assert_png(fig, tmp_path)


def test_health_figures_fail_loudly_without_their_diagnostics():
    history = synthetic_history("hmc")
    for key in ("hmc_accept", "hmc_eps", "hmc_div", "hmc_ess_frac", "hmc_mc_err"):
        history[key] = [np.full(L, np.nan)] * 4
    history["is_max_weight"] = history["hmc_eps"]
    with pytest.raises(ValueError, match="none of the diagnostics"):
        draw.fig_sampler_health(history)
    with pytest.raises(ValueError, match="outer_step"):
        draw.fig_solver_health({k: v for k, v in history.items() if k != "outer_step"})


def test_the_full_dashboard_keeps_its_panels_and_zero_based_layers():
    """Literal titles, in order, so a change to the shared panel list is a visible edit here."""
    fig = draw.fig_solver_history(synthetic_history("hmc"))
    assert titles(fig) == [
        "inner residual  ||⟨ψψᵀ⟩-Φ||/||Φ||",
        "outer step size  ||ΔΦ||/||Φ|| (before clipping)",
        "fixed-point residual of Φ̂  (Eq. 31b/c)",
        "Stein vs direct estimate of Φ̂  (sampler equilibrium)",
        "train MSE: theory (lines) vs net (dashed)",
        "adaptive step sizes  η̂ (lines), η (black)",
        "backtracking: inner (lines), outer (black)",
        "HMC acceptance",
        "HMC step size ε",
        "divergent transitions per run",
        "relative Monte Carlo error of ⟨ψψᵀ⟩",
        "sampler ESS / samples",
        "lag-1 autocorrelation of ||h||²/P",
        "chain scale  ⟨||h||²⟩ / prior diag",
        "Stein identity mismatch",
        "normalisability margin (all-active orthant cond.)",
        "kernel alignment with the trained net",
        "eigenvalues raised to the PD floor",
        "λ_min(Φ) before PD floor",
    ]
    assert legend_labels(fig)[0] == "layer 0"


def legend_labels(fig) -> list[str]:
    return [
        t.get_text()
        for ax in fig.axes
        for t in (ax.get_legend().get_texts() if ax.get_legend() else [])
    ]


def test_the_live_figures_all_count_layers_from_one():
    """The kernel panels say layer 1 for the first layer, so the health panels must too."""
    kernels = torch.rand(L, P, P)
    history = synthetic_history("hmc")
    for fig in (draw.fig_solver_health(history), draw.fig_sampler_health(history)):
        labels = [x for x in legend_labels(fig) if x.startswith("layer")]
        assert "layer 0" not in labels and "layer 1" in labels
    kernel_titles = titles(draw.fig_layer_kernels({"Φ": kernels}))
    assert kernel_titles[0].endswith("layer 1")
