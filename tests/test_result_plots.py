"""Smoke and behaviour tests for the loss, MIR-history and example-output figures."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.figure import Figure

from fct import draw
from fct.analyse import example_outputs
from fct.config import DataConfig, NetConfig
from fct.data import make_dataset, task_dims
from fct.model import make_net

CPU, F64 = torch.device("cpu"), torch.float64
G = 3
CONFIGS = {
    "object_action": DataConfig(
        task="object_action", P=30, grid_size=G, max_objects=3, D_in=2 * G * G, seed=1
    ),
    "what_where": DataConfig(task="what_where", P=30, n_motifs=4, D_in=81, seed=2),
    "linear_teacher": DataConfig(task="linear_teacher", P=30, D_in=6, n_latents=2, seed=3),
    "position": DataConfig(task="position", P=30, D_in=8, d=2, embedding="linear", seed=4),
}


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _loss_history(n=6, **drop):
    rng = np.random.default_rng(0)
    h = {
        "epoch": list(range(n)),
        "mse": list(rng.uniform(0.1, 1, n)),
        "weight_l2": list(rng.uniform(1, 2, n)),
        "rate_l2": list(rng.uniform(0.5, 1, n)),
        "loss": list(rng.uniform(1, 3, n)),
    }
    return {k: v for k, v in h.items() if k not in drop}


def _monitor_history(n_layers=3, n_calls=4, with_monitor=True):
    h = _loss_history()
    if not with_monitor:
        return h
    rng = np.random.default_rng(1)

    def per_call(scale=1.0):
        return [list(scale * rng.uniform(0, 1, n_layers)) for _ in range(n_calls)]

    h["monitor_epoch"] = [10 * i for i in range(n_calls)]
    h["monitor_mir"] = per_call()
    h["monitor_mir_corrected"] = per_call()
    h["monitor_mir_chance"] = per_call(0.2)
    h["monitor_n_dead"] = [[int(v * 5) for v in row] for row in per_call()]
    h["monitor_n_dead_corrected"] = [[int(v * 5) for v in row] for row in per_call()]
    return h


# ----------------------------------------------------------------- loss curves


def test_plot_loss_curves_draws_four_distinguishable_log_lines():
    ax = draw.plot_loss_curves(_loss_history())
    assert ax.get_yscale() == "log" and ax.get_xlabel() == "epoch" and ax.get_ylabel()
    assert len(ax.lines) == 4
    assert len({line.get_linestyle() for line in ax.lines}) == 4
    assert {t.get_text() for t in ax.get_legend().get_texts()} >= {"loss", "mse"}


def test_plot_loss_curves_uses_the_given_axis_and_creates_no_figure():
    _fig, ax = plt.subplots()
    n_before = len(plt.get_fignums())
    assert draw.plot_loss_curves(_loss_history(), ax=ax) is ax
    assert len(plt.get_fignums()) == n_before


@pytest.mark.parametrize("bad", [{"rate_l2": None}, {"epoch": None}])
def test_plot_loss_curves_rejects_missing_keys(bad):
    with pytest.raises(ValueError, match="history"):
        draw.plot_loss_curves(_loss_history(**bad))


def test_plot_loss_curves_rejects_empty_and_mismatched_histories():
    with pytest.raises(ValueError, match="empty"):
        draw.plot_loss_curves(_loss_history(n=0))
    h = _loss_history()
    h["mse"] = h["mse"][:-1]
    with pytest.raises(ValueError, match="length"):
        draw.plot_loss_curves(h)


# ----------------------------------------------------------------- MIR history


@pytest.mark.parametrize("n_layers", [1, 3, 5])
@pytest.mark.parametrize("which", ["raw", "corrected"])
def test_plot_mir_history_has_one_line_per_layer_plus_baselines_for_raw(n_layers, which):
    h = _monitor_history(n_layers)
    ax = draw.plot_mir_history(h, which=which)
    expected = n_layers * (2 if which == "raw" else 1)
    assert len(ax.lines) == expected
    assert ax.get_ylim() == pytest.approx((-0.05, 1.05))
    assert ax.get_xlabel() and ax.get_ylabel()
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels[:n_layers] == [f"layer {i + 1}" for i in range(n_layers)]
    assert (len(labels) == n_layers + 1) == (which == "raw")


def test_plot_mir_history_colours_follow_the_palette_and_baseline_is_dotted_same_colour():
    ax = draw.plot_mir_history(_monitor_history(3), which="raw")
    solid = [ln for ln in ax.lines if ln.get_linestyle() == "-"]
    dotted = [ln for ln in ax.lines if ln.get_linestyle() == ":"]
    assert [ln.get_color() for ln in solid] == draw.LAYER_COLORS[:3]
    assert [ln.get_color() for ln in dotted] == draw.LAYER_COLORS[:3]


def test_plot_mir_history_with_one_monitor_call_still_draws_visible_points():
    ax = draw.plot_mir_history(_monitor_history(2, n_calls=1))
    assert all(ln.get_marker() not in (None, "None", "") for ln in ax.lines[:2])


def test_plot_mir_history_uses_the_given_axis():
    _fig, ax = plt.subplots()
    n_before = len(plt.get_fignums())
    assert draw.plot_mir_history(_monitor_history(), ax=ax) is ax
    assert len(plt.get_fignums()) == n_before


def test_plot_mir_history_without_monitor_keys_raises_a_clear_error():
    with pytest.raises(ValueError, match="monitor"):
        draw.plot_mir_history(_monitor_history(with_monitor=False))
    with pytest.raises(ValueError, match="monitor"):
        draw.fig_mir_history(_monitor_history(with_monitor=False))


def test_plot_mir_history_rejects_empty_mismatched_and_unknown_which():
    h = _monitor_history()
    with pytest.raises(ValueError, match="which"):
        draw.plot_mir_history(h, which="both")
    empty = {**h, "monitor_epoch": [], "monitor_mir": []}
    with pytest.raises(ValueError, match="empty"):
        draw.plot_mir_history(empty)
    short = {**h, "monitor_mir": h["monitor_mir"][:-1]}
    with pytest.raises(ValueError, match="length"):
        draw.plot_mir_history(short)
    ragged = {**h, "monitor_mir": [[0.1, 0.2], [0.1], [0.3, 0.4], [0.1, 0.2]]}
    with pytest.raises(ValueError, match="layers"):
        draw.plot_mir_history(ragged)


def test_plot_dead_neurons_one_line_per_layer():
    ax = draw.plot_dead_neurons(_monitor_history(3))
    assert len(ax.lines) == 3 and ax.get_xlabel() and ax.get_ylabel()


@pytest.mark.parametrize("n_layers", [1, 3, 5])
def test_fig_mir_history_has_three_panels_with_shared_x(n_layers):
    fig = draw.fig_mir_history(_monitor_history(n_layers, n_calls=1), title="setting A")
    assert isinstance(fig, Figure)
    axes = fig.axes
    assert len(axes) == 3
    assert axes[0].get_shared_x_axes().joined(axes[0], axes[1])
    assert axes[0].get_shared_x_axes().joined(axes[0], axes[2])
    assert fig._suptitle.get_text() == "setting A"


# ----------------------------------------------------------------- example outputs


def _outputs(task, n=6):
    cfg = CONFIGS[task]
    ds = make_dataset(cfg, CPU, F64)
    d_in, d_out = task_dims(cfg)
    net = make_net(NetConfig(N=5, L=2, phi="relu"), d_in, d_out, CPU, F64)
    return example_outputs(net, ds, list(range(n))), cfg


@pytest.mark.parametrize("task", list(CONFIGS))
def test_fig_example_outputs_renders_every_task(task):
    outputs, cfg = _outputs(task)
    fig = draw.fig_example_outputs(task, outputs, cfg)
    assert isinstance(fig, Figure) and fig.axes
    assert all(ax.has_data() for ax in fig.axes)


@pytest.mark.parametrize("task", list(CONFIGS))
def test_plot_example_outputs_draws_into_the_given_axis(task):
    outputs, cfg = _outputs(task)
    _fig, ax = plt.subplots()
    n_before = len(plt.get_fignums())
    assert draw.plot_example_outputs(task, outputs, cfg, ax=ax) is ax
    assert len(plt.get_fignums()) == n_before
    assert ax.has_data()
    draw.plot_example_outputs(task, outputs, cfg)  # creates its own figure
    assert len(plt.get_fignums()) == n_before + 1


def test_example_outputs_reject_an_unknown_task():
    outputs, cfg = _outputs("linear_teacher")
    with pytest.raises(ValueError, match="unknown task"):
        draw.fig_example_outputs("nope", outputs, cfg)
    with pytest.raises(ValueError, match="unknown task"):
        draw.plot_example_outputs("nope", outputs, cfg)


def test_fig_example_outputs_shows_at_most_the_requested_number_of_examples():
    outputs, cfg = _outputs("object_action", n=6)
    fig = draw.fig_example_outputs("object_action", outputs, cfg, n_show=2)
    assert len(fig.axes) == 2 * 2


def test_plot_prediction_scatter_has_identity_line_and_r2_in_the_legend():
    y = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    ax = draw.plot_prediction_scatter(y, y)
    assert any("R" in t.get_text() and "1.00" in t.get_text() for t in ax.get_legend().get_texts())
    assert ax.get_xlabel() and ax.get_ylabel()


def test_plot_object_grid_puts_row_zero_at_the_top_and_marks_agent_and_objects():
    objects = torch.zeros(G, G, dtype=torch.bool)
    objects[0, 2] = True
    ax = draw.plot_object_grid(torch.tensor([2, 0]), objects)
    assert ax.yaxis_inverted() and not ax.xaxis_inverted()
    assert ax.get_xlim() == pytest.approx((-0.5, G - 0.5))
    offsets = np.concatenate([c.get_offsets() for c in ax.collections])
    assert {(0.0, 2.0), (2.0, 0.0)} <= {tuple(map(float, o)) for o in offsets}  # (col, row)


def test_plot_motif_image_draws_the_macro_grid_on_three_pixel_boundaries():
    image = torch.zeros(9, 9)
    ax = draw.plot_motif_image(image)
    xs = sorted(ln.get_xdata()[0] for ln in ax.lines if ln.get_xdata()[0] == ln.get_xdata()[-1])
    ys = sorted(ln.get_ydata()[0] for ln in ax.lines if ln.get_ydata()[0] == ln.get_ydata()[-1])
    assert xs == [2.5, 5.5] and ys == [2.5, 5.5]
    assert ax.yaxis_inverted()
