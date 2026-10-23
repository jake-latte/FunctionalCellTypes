"""Smoke and behaviour tests for the loss, MIR-history and example-output figures."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

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
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        "loss (total)",
        "mse (raw)",
        "weight penalty (raw)",
        "rate L2 (raw)",
    ]
    assert "not additive" in ax.get_ylabel()


def test_plot_loss_curves_marks_all_zero_series_in_the_legend():
    h = _loss_history()
    h["rate_l2"] = [0.0] * len(h["epoch"])
    texts = [t.get_text() for t in draw.plot_loss_curves(h).get_legend().get_texts()]
    assert texts[-1] == "rate L2 (raw) (zero)"
    assert "(zero)" not in " ".join(texts[:-1])


def test_plot_training_history_is_the_loss_curves_plot():
    ax = draw.plot_training_history(_loss_history())
    assert ax.get_legend().get_texts()[0].get_text() == "loss (total)"


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
    assert fig.get_suptitle() == "setting A"


def test_plot_mir_history_corrected_extends_below_the_fixed_range_for_negative_values():
    h = _monitor_history(2)
    h["monitor_mir_corrected"][1][0] = -0.3
    low, high = draw.plot_mir_history(h, which="corrected").get_ylim()
    assert low < -0.3 and high == pytest.approx(1.05)
    # raw keeps the fixed range whatever the data
    h["monitor_mir"][1][0] = -0.3
    assert draw.plot_mir_history(h, which="raw").get_ylim() == pytest.approx((-0.05, 1.05))


def test_plot_mir_history_corrected_keeps_the_fixed_range_for_ordinary_values():
    h = _monitor_history(2)
    h["monitor_mir_corrected"] = [[0.1, 0.2]] * 4
    assert draw.plot_mir_history(h, which="corrected").get_ylim() == pytest.approx((-0.05, 1.05))


def test_single_monitor_call_axes_have_integer_ticks_and_a_visible_window():
    h = _monitor_history(2, n_calls=1)
    h["monitor_epoch"] = [7]
    for ax in (draw.plot_mir_history(h), draw.plot_dead_neurons(h)):
        low, high = ax.get_xlim()
        assert low < 7 < high
        assert all(float(t).is_integer() for t in ax.get_xticks())
    assert all(float(t).is_integer() for t in draw.plot_dead_neurons(h).get_yticks())


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
    outputs, _ = _outputs(task)
    _fig, ax = plt.subplots()
    n_before = len(plt.get_fignums())
    assert draw.plot_example_outputs(task, outputs, ax=ax) is ax
    assert len(plt.get_fignums()) == n_before
    assert ax.has_data()
    draw.plot_example_outputs(task, outputs)  # creates its own figure
    assert len(plt.get_fignums()) == n_before + 1


def test_example_outputs_reject_an_unknown_task():
    outputs, cfg = _outputs("linear_teacher")
    with pytest.raises(ValueError, match="unknown task"):
        draw.fig_example_outputs("nope", outputs, cfg)
    with pytest.raises(ValueError, match="unknown task"):
        draw.plot_example_outputs("nope", outputs)


def test_fig_example_outputs_rejects_a_config_for_another_task():
    outputs, cfg = _outputs("what_where")
    with pytest.raises(ValueError, match="not 'object_action'"):
        draw.fig_example_outputs("object_action", outputs, cfg)


def test_fig_example_outputs_rejects_inconsistent_motif_count():
    outputs, cfg = _outputs("what_where")
    wrong = cfg.model_copy(update={"n_motifs": cfg.n_motifs + 1})
    with pytest.raises(ValueError, match="columns"):
        draw.fig_example_outputs("what_where", outputs, wrong)


@pytest.mark.parametrize("n_show", [0, -1])
def test_fig_example_outputs_rejects_nonpositive_n_show(n_show):
    outputs, cfg = _outputs("object_action")
    with pytest.raises(ValueError, match="n_show"):
        draw.fig_example_outputs("object_action", outputs, cfg, n_show=n_show)


@pytest.mark.parametrize("task", list(CONFIGS))
def test_fig_example_outputs_rejects_empty_outputs(task):
    outputs, cfg = _outputs(task, n=0)
    with pytest.raises(ValueError, match="empty"):
        draw.fig_example_outputs(task, outputs, cfg)


def test_fig_example_outputs_shows_at_most_the_requested_number_of_examples():
    outputs, cfg = _outputs("object_action", n=6)
    fig = draw.fig_example_outputs("object_action", outputs, cfg, n_show=2)
    assert len(fig.axes) == 2 * 2


def test_example_figures_use_one_figure_legend_not_per_panel_legends():
    outputs, cfg = _outputs("what_where")
    fig = draw.fig_example_outputs("what_where", outputs, cfg)
    assert len(fig.legends) == 1
    assert {t.get_text() for t in fig.legends[0].get_texts()} == {"true", "predicted"}
    assert all(ax.get_legend() is None for ax in fig.axes)


def test_constant_targets_do_not_prevent_the_figure():
    outputs, cfg = _outputs("position", n=3)
    flat = type(outputs)(
        outputs.indices, outputs.x, torch.ones_like(outputs.y_true), outputs.y_pred
    )
    fig = draw.fig_example_outputs("position", flat, cfg)
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == ["identity"]


def test_plot_prediction_scatter_shows_the_given_r2_in_the_legend():
    y = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    ax = draw.plot_prediction_scatter(y, y, r2=0.9)
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert texts == ["identity (R² on these examples = 0.90)"]
    assert ax.get_xlabel() and ax.get_ylabel()
    assert [t.get_text() for t in draw.plot_prediction_scatter(y, y).get_legend().get_texts()] == [
        "identity"
    ]


def _grid(agent, *cells, n=G):
    objects = torch.zeros(n, n, dtype=torch.bool)
    for row, col in cells:
        objects[row, col] = True
    return torch.tensor(agent), objects


def test_plot_object_grid_puts_row_zero_at_the_top_and_marks_agent_and_objects():
    agent, objects = _grid([2, 0], (0, 2))
    ax = draw.plot_object_grid(agent, objects)
    assert ax.yaxis_inverted() and not ax.xaxis_inverted()
    assert ax.get_xlim() == pytest.approx((-0.5, G - 0.5))
    squares = [p for p in ax.patches if isinstance(p, Rectangle)]
    rings = [p for p in ax.patches if isinstance(p, Circle)]
    # data coordinates are (column, row)
    assert [(s.get_x() + s.get_width() / 2, s.get_y() + s.get_height() / 2) for s in squares] == [
        (2.0, 0.0)
    ]
    assert {c.center for c in rings} == {(0.0, 2.0)}


def test_agent_on_an_object_is_ringed_by_the_object_square_fit_inside():
    agent, objects = _grid([1, 1], (1, 1))
    ax = draw.plot_object_grid(agent, objects)
    square = next(p for p in ax.patches if isinstance(p, Rectangle))
    ring = max((p for p in ax.patches if isinstance(p, Circle)), key=lambda c: c.get_radius())
    assert not ring.get_fill()
    assert ring.get_radius() > np.hypot(square.get_width(), square.get_height()) / 2 * 0.9
    assert ring.get_radius() < 0.5  # inside the cell


def test_plot_object_grid_rejects_bad_shapes_and_agent_outside_the_grid():
    with pytest.raises(ValueError, match="grid"):
        draw.plot_object_grid(torch.tensor([0, 0]), torch.zeros(3, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match="agent_rc"):
        draw.plot_object_grid(torch.tensor([3, 0]), torch.zeros(3, 3, dtype=torch.bool))


def _boundaries(ax):
    xs = sorted(ln.get_xdata()[0] for ln in ax.lines if ln.get_xdata()[0] == ln.get_xdata()[-1])
    ys = sorted(ln.get_ydata()[0] for ln in ax.lines if ln.get_ydata()[0] == ln.get_ydata()[-1])
    return xs, ys


def test_plot_motif_image_draws_the_macro_grid_on_three_pixel_boundaries():
    ax = draw.plot_motif_image(torch.zeros(9, 9))
    assert _boundaries(ax) == ([2.5, 5.5], [2.5, 5.5])
    assert ax.yaxis_inverted()


@pytest.mark.parametrize(
    ("n_macro", "motif_size", "expected"),
    [(2, 4, [3.5]), (4, 2, [1.5, 3.5, 5.5]), (3, 2, [1.5, 3.5])],
)
def test_motif_image_grid_follows_macro_count_and_motif_size(n_macro, motif_size, expected):
    # Swapping the two constants would move the lines, so a mixed-up constant fails here.
    side = n_macro * motif_size
    ax = draw.plot_motif_image(torch.zeros(side, side), n_macro=n_macro, motif_size=motif_size)
    assert _boundaries(ax) == (expected, expected)
    assert len(ax.get_xticks()) == n_macro


def test_plot_motif_image_rejects_a_wrong_image_shape():
    with pytest.raises(ValueError, match="shape"):
        draw.plot_motif_image(torch.zeros(8, 9))


# ----------------------------------------------------------------- size and style


def _tick_sizes(fig) -> list[float]:
    fig.canvas.draw()  # tick labels exist only after a draw
    return [
        t.get_fontsize() for ax in fig.axes for t in ax.get_xticklabels() + ax.get_yticklabels()
    ]


def _full_width_figures():
    values = np.array([[0.1, 0.2], [0.3, np.nan]])
    figs = {
        "loss": draw.fig_loss_curves(_loss_history()),
        "mir": draw.fig_mir_history(_monitor_history(2)),
        "heatmap": draw.fig_mir_heatmaps({"raw": values}, ["a", "b"], ["done", "done"]),
    }
    for task in ("linear_teacher", "object_action", "what_where"):
        outputs, cfg = _outputs(task)
        figs[task] = draw.fig_example_outputs(task, outputs, cfg)
    return figs


def test_results_figures_are_full_width_with_readable_ticks_without_caller_rcparams():
    for name, fig in _full_width_figures().items():
        assert fig.get_figwidth() >= 12, name
        assert min(_tick_sizes(fig)) >= 11, name


def test_position_figure_is_a_large_square_scatter():
    outputs, cfg = _outputs("position")
    fig = draw.fig_example_outputs("position", outputs, cfg)
    assert fig.get_figwidth() >= 6 and fig.get_figheight() >= 6
    assert min(_tick_sizes(fig)) >= 11


def test_fig_loss_curves_is_titled_and_keeps_the_legend_right_of_the_axis():
    fig = draw.fig_loss_curves(_loss_history(), title="setting A")
    (ax,) = fig.axes
    assert ax.get_title() == "setting A"
    fig.canvas.draw()
    assert ax.get_legend().get_window_extent().x0 >= ax.get_window_extent().x1


def test_prediction_scatter_legend_is_below_the_axis_in_a_styled_figure():
    outputs, cfg = _outputs("position")
    fig = draw.fig_example_outputs("position", outputs, cfg)
    ax = fig.axes[0]
    fig.canvas.draw()
    assert ax.get_legend().get_window_extent().y1 <= ax.get_window_extent().y0


def test_linear_teacher_overlays_true_and_predicted_on_one_axis_per_example():
    outputs, cfg = _outputs("linear_teacher", n=4)
    fig = draw.fig_example_outputs("linear_teacher", outputs, cfg, n_show=3)
    *examples, scatter = fig.axes
    assert len(examples) == 3  # one axis per example, not a true/pred pair
    assert not scatter.get_title() and scatter.get_xlabel() == "true output"
    for ax, idx in zip(examples, outputs.indices[:3]):
        assert f"example {int(idx)}" in ax.get_title("left")
        labels = {a.get_label() for a in ax.lines + ax.collections + ax.containers}
        assert {"true", "predicted"} <= labels
        # both series have one marker per output index at the same x positions
        xs = {
            a.get_label(): list(a.get_xdata())
            for a in ax.lines
            if a.get_label() in {"true", "predicted"}
        }
        assert xs["true"] == xs["predicted"] == list(range(outputs.y_true.shape[1]))
    texts = [t.get_text() for t in fig.legends[0].get_texts()]
    assert texts == ["true", "predicted"]
    # shared y scale across the example axes
    assert all(ax.get_shared_y_axes().joined(ax, examples[0]) for ax in examples)
