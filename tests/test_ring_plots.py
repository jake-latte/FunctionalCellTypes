"""The linear-ring figures of fct.draw on synthetic inputs with known values (Agg backend, CPU).

Every `plot_*` here is checked for the contract of the module: it draws into a given axis and
returns it without making a figure, and makes its own figure only when it is given none. The
drawn values are read back from the artists and compared with the inputs.
"""

import math

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.container import ErrorbarContainer

from fct import draw
from fct.spectrum import N_THETA_BINS
from tests.draw_helpers import assert_png

K, N = 2, 16
Q = N // 2 + 1  # folded |q| = 0..N/2


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _spectra(rows: int = K) -> np.ndarray:
    """(rows, Q) strictly positive Gaussian-shaped spectra, row k scaled by 10^-k."""
    q = np.arange(Q)
    return np.stack([10.0**-k * np.exp(-0.2 * q**2) + 1e-6 for k in range(rows)])


def _errorbars(ax) -> list[ErrorbarContainer]:
    return [c for c in ax.containers if isinstance(c, ErrorbarContainer)]


def _points(container: ErrorbarContainer) -> tuple[np.ndarray, np.ndarray]:
    line = container.lines[0]
    return np.asarray(line.get_xdata(), float), np.asarray(line.get_ydata(), float)


def _line_labelled(ax, label: str):
    (line,) = [ln for ln in ax.lines if ln.get_label() == label]
    return line


# ------------------------------------------------------------------ the shared contract
def _call(name: str, ax=None):
    """Each plotting function on a small valid input."""
    s = _spectra()
    lags = np.exp(-(np.arange(Q)[None, :] ** 2) / 8.0) * np.ones((K, 1))
    calls = {
        "plot_spectrum": lambda: draw.plot_spectrum(s, 0.1 * s, s, ax=ax),
        "plot_spectrum_ratio": lambda: draw.plot_spectrum_ratio(
            s[0] / s[1], 0.1 * np.ones(Q), s[0] / s[1], ax=ax
        ),
        "plot_autocorrelation": lambda: draw.plot_autocorrelation(
            lags, 0.01 * lags, lags, kernel=lags[0], ax=ax
        ),
        "plot_share_sweep": lambda: draw.plot_share_sweep(
            [1.0, 2.0, 3.0],
            np.full((3, K), 0.5),
            np.full((3, K), 0.05),
            np.full((3, K), 0.5),
            ax=ax,
        ),
        "plot_angle_histogram": lambda: draw.plot_angle_histogram(np.ones(N_THETA_BINS), ax=ax),
        "plot_relaxation": lambda: draw.plot_relaxation(
            np.arange(10.0),
            np.zeros((10, 3)),
            np.arange(5.0, 10.0),
            np.zeros((5, 3)),
            ["a", "b", "c"],
            ax=ax,
        ),
        "plot_ensemble_loss": lambda: draw.plot_ensemble_loss(
            np.arange(1.0, 4.0), {"loss": np.ones(3)}, {"loss": 0.1 * np.ones(3)}, ax=ax
        ),
    }
    return calls[name]()


RING_PLOTS = (
    "plot_spectrum",
    "plot_spectrum_ratio",
    "plot_autocorrelation",
    "plot_share_sweep",
    "plot_angle_histogram",
    "plot_relaxation",
    "plot_ensemble_loss",
)


@pytest.mark.parametrize("name", RING_PLOTS)
def test_draws_into_the_given_axis_without_making_a_figure(name):
    _, ax = plt.subplots()
    n_figures = len(plt.get_fignums())
    assert _call(name, ax) is ax
    assert len(plt.get_fignums()) == n_figures
    assert ax.lines or ax.containers or ax.patches


@pytest.mark.parametrize("name", RING_PLOTS)
def test_makes_its_own_figure_without_an_axis(name, tmp_path):
    n_figures = len(plt.get_fignums())
    ax = _call(name)
    assert len(plt.get_fignums()) == n_figures + 1
    assert_png(ax.figure, tmp_path)


# ------------------------------------------------------------------ spectrum
def test_spectrum_draws_each_mode_as_points_with_errors_and_its_theory_as_a_line():
    measured, theory = _spectra(), 1.1 * _spectra()
    ax = draw.plot_spectrum(measured, 0.1 * measured, theory)
    assert ax.get_yscale() == "log" and ax.get_xlabel() == "|q|"
    bars = _errorbars(ax)
    assert len(bars) == K
    for k, bar in enumerate(bars):
        x, y = _points(bar)
        np.testing.assert_array_equal(x, np.arange(Q))
        np.testing.assert_allclose(y, measured[k])
        line = _line_labelled(ax, f"mode {k + 1} theory")
        np.testing.assert_allclose(line.get_ydata(), theory[k])
        assert line.get_color() == bar.lines[0].get_color()  # one colour per mode
    assert bars[0].lines[0].get_color() != bars[1].lines[0].get_color()


def test_spectrum_draws_a_null_row_dashed_and_its_floor():
    measured = np.vstack([_spectra(), np.full((1, Q), 1e-3)])
    ax = draw.plot_spectrum(measured, 0.1 * measured, _spectra(), null_theory=2e-3)
    bars = _errorbars(ax)
    assert len(bars) == K + 1
    null = bars[-1]
    np.testing.assert_allclose(_points(null)[1], measured[-1])
    assert null.lines[0].get_linestyle() == "--"
    floor = _line_labelled(ax, "null theory")
    assert floor.get_linestyle() == "--"
    np.testing.assert_allclose(floor.get_ydata(), 2e-3)


def test_spectrum_marks_a_null_row_of_zeros_instead_of_drawing_it():
    """At sigma = 0 the null row is round-off or exactly zero: a log axis cannot show it."""
    measured = np.vstack([_spectra(), np.zeros((1, Q))])
    ax = draw.plot_spectrum(measured, np.zeros_like(measured), _spectra(), null_theory=0.0)
    assert len(_errorbars(ax)) == K
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "null (zero)" in labels


def test_spectrum_names_a_null_row_below_the_latent_spectra_instead_of_stretching_the_axis():
    """Round-off null power (1e-32) would put thirty decades on the axis; it ends one decade
    below the smallest latent value."""
    measured = np.vstack([_spectra(), np.full((1, Q), 1e-32)])
    ax = draw.plot_spectrum(measured, 0.1 * measured, _spectra(), null_theory=1e-30)
    assert len(_errorbars(ax)) == K
    assert ax.get_ylim()[0] == pytest.approx(_spectra().min() / 10)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "null (at most 1e-32, below the axis)" in labels
    assert "null theory (at most 1e-30, below the axis)" in labels


def test_spectrum_drops_the_non_positive_points_of_a_mode():
    measured = _spectra()
    measured[0, 3] = 0.0
    ax = draw.plot_spectrum(measured, 0.1 * measured, _spectra())
    x, _ = _points(_errorbars(ax)[0])
    assert 3 not in x and len(x) == Q - 1


def test_spectrum_shades_the_excluded_frequencies():
    excluded = np.arange(Q) >= 5
    ax = draw.plot_spectrum(_spectra(), 0.1 * _spectra(), _spectra(), excluded=excluded)
    (span,) = [p for p in ax.patches if p.get_label() == "excluded |q|"]
    x0, x1 = span.get_x(), span.get_x() + span.get_width()
    assert x0 == pytest.approx(4.5) and x1 == pytest.approx(Q - 0.5)


def test_spectrum_takes_tensors():
    s = torch.from_numpy(_spectra())
    ax = draw.plot_spectrum(s, 0.1 * s, s)
    np.testing.assert_allclose(_points(_errorbars(ax)[0])[1], _spectra()[0])


@pytest.mark.parametrize(
    "measured, se, theory",
    [
        (_spectra(), _spectra()[:, :-1], _spectra()),  # se of another shape
        (_spectra(), _spectra(), _spectra()[:, :-1]),  # theory on other frequencies
        (_spectra(4), _spectra(4), _spectra()),  # two rows more than the theory's modes
        (_spectra()[0], _spectra()[0], _spectra()[0]),  # not (modes, |q|)
    ],
)
def test_spectrum_refuses_inconsistent_shapes(measured, se, theory):
    with pytest.raises(ValueError):
        draw.plot_spectrum(measured, se, theory)


# ------------------------------------------------------------------ ratio
def test_ratio_draws_the_measured_ratio_with_errors_against_the_theory():
    ratio, theory = np.linspace(1.5, 7.0, Q), np.full(Q, 7.0)
    ax = draw.plot_spectrum_ratio(ratio, 0.2 * np.ones(Q), theory, excluded=np.arange(Q) >= 5)
    (bar,) = _errorbars(ax)
    x, y = _points(bar)
    np.testing.assert_array_equal(x, np.arange(5))  # the excluded |q| are not drawn
    np.testing.assert_allclose(y, ratio[:5])
    np.testing.assert_allclose(_line_labelled(ax, "theory").get_ydata(), theory)
    assert "mode 1" in ax.get_ylabel() and ax.get_xlabel() == "|q|"


def test_ratio_refuses_mismatched_lengths():
    with pytest.raises(ValueError):
        draw.plot_spectrum_ratio(np.ones(Q), np.ones(Q - 1), np.ones(Q))


# ------------------------------------------------------------------ autocorrelation
def test_autocorrelation_draws_each_mode_against_its_theory_and_the_kernel():
    lags = np.arange(Q)
    theory = np.stack([np.exp(-(lags**2) / 8.0), np.exp(-(lags**2) / 4.0)])
    measured = theory + 0.01
    kernel = np.exp(-(lags**2) / 8.0)
    ax = draw.plot_autocorrelation(measured, 0.01 * np.ones_like(theory), theory, kernel=kernel)
    bars = _errorbars(ax)
    assert len(bars) == K
    for k in range(K):
        np.testing.assert_allclose(_points(bars[k])[1], measured[k])
        np.testing.assert_allclose(
            _line_labelled(ax, f"mode {k + 1} theory").get_ydata(), theory[k]
        )
    prior = _line_labelled(ax, "prior kernel")
    np.testing.assert_allclose(prior.get_ydata(), kernel)
    assert prior.get_linestyle() == ":"
    assert ax.get_xlabel() == "Δ (sites)" and "R(Δ)" in ax.get_ylabel()


def test_autocorrelation_without_a_kernel_draws_no_kernel_line():
    theory = np.ones((K, Q))
    ax = draw.plot_autocorrelation(theory, 0 * theory, theory)
    assert not [ln for ln in ax.lines if ln.get_label() == "prior kernel"]


# ------------------------------------------------------------------ band share sweep
def test_share_sweep_draws_each_mode_against_the_swept_value_with_its_theory():
    values = np.array([10.0, 0.25, 3.0])  # unsorted, as settings may come
    shares = np.array([[0.5, 0.3], [0.3, 0.31], [0.4, 0.32]])
    theory = shares + 0.01
    ax = draw.plot_share_sweep(values, shares, 0.02 * np.ones_like(shares), theory,
                               xlabel="β", xscale="log")  # fmt: skip
    assert ax.get_xscale() == "log" and ax.get_xlabel() == "β"
    assert "band share" in ax.get_ylabel()
    bars = _errorbars(ax)
    assert len(bars) == K
    order = np.argsort(values)
    for k in range(K):
        x, y = _points(bars[k])
        np.testing.assert_allclose(x, values)
        np.testing.assert_allclose(y, shares[:, k])
        line = _line_labelled(ax, f"mode {k + 1} theory")
        np.testing.assert_allclose(line.get_xdata(), values[order])  # a line in x order
        np.testing.assert_allclose(line.get_ydata(), theory[order, k])


def test_share_sweep_refuses_shares_not_one_row_per_value():
    with pytest.raises(ValueError):
        draw.plot_share_sweep([1.0, 2.0], np.ones((3, K)), np.ones((3, K)), np.ones((3, K)))


# ------------------------------------------------------------------ angle histogram
def test_angle_histogram_draws_the_density_with_the_uniform_level_and_the_marks():
    half = N_THETA_BINS // 2
    counts = np.zeros(N_THETA_BINS)
    counts[:half] = 2.0  # all mass on (-pi, 0]: density 1 / (half x w) there, 0 elsewhere
    marks = [-math.pi / 3, math.pi / 2]
    ax = draw.plot_angle_histogram(counts, marks=marks)
    heights = np.array([p.get_height() for p in ax.containers[0]])
    width = 2 * math.pi / N_THETA_BINS
    np.testing.assert_allclose(heights[:half], 1 / (half * width))
    np.testing.assert_allclose(heights[half:], 0.0)
    np.testing.assert_allclose(sum(heights) * width, 1.0)
    lefts = np.array([p.get_x() for p in ax.containers[0]])
    np.testing.assert_allclose(lefts, -math.pi + width * np.arange(N_THETA_BINS))
    assert len(ax.containers[0]) == N_THETA_BINS
    np.testing.assert_allclose(_line_labelled(ax, "uniform").get_ydata(), 1 / (2 * math.pi))
    mark_x = sorted(ln.get_xdata()[0] for ln in ax.lines if ln.get_gid() == "latent axis")
    np.testing.assert_allclose(mark_x, sorted(marks))
    assert ax.get_xlim() == pytest.approx((-math.pi, math.pi))


@pytest.mark.parametrize(
    "counts", [np.ones(N_THETA_BINS - 1), np.zeros(N_THETA_BINS), -np.ones(N_THETA_BINS)]
)
def test_angle_histogram_refuses_wrong_bins_and_empty_or_negative_counts(counts):
    with pytest.raises(ValueError):
        draw.plot_angle_histogram(counts)


# ------------------------------------------------------------------ relaxation
def test_relaxation_draws_the_records_the_fit_the_window_and_the_band():
    epochs = np.arange(0.0, 100.0, 10.0)
    values = np.stack([np.exp(-epochs / 20.0), -np.exp(-epochs / 30.0)], axis=1)
    fit_epochs = epochs[5:]
    fitted = values[5:] * 0.9
    ax = draw.plot_relaxation(
        epochs, values, fit_epochs, fitted, ["lag1_1", "band_1"], start=50.0, band=1.0
    )
    measured = _line_labelled(ax, "lag1_1")
    np.testing.assert_allclose(measured.get_xdata(), epochs)
    np.testing.assert_allclose(measured.get_ydata(), values[:, 0])
    (fit,) = [ln for ln in ax.lines if ln.get_gid() == "lag1_1 fit"]
    np.testing.assert_allclose(fit.get_xdata(), fit_epochs)
    np.testing.assert_allclose(fit.get_ydata(), fitted[:, 0])
    assert fit.get_linestyle() == "--" and fit.get_color() == measured.get_color()
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "fitted relaxation" in labels and "lag1_1 fit" not in labels  # one entry for all fits
    start = _line_labelled(ax, "fit window start")
    np.testing.assert_allclose(start.get_xdata(), 50.0)
    (span,) = [p for p in ax.patches if p.get_label() == "± budget"]
    assert span.get_y() == pytest.approx(-1.0) and span.get_height() == pytest.approx(2.0)


def test_relaxation_refuses_labels_not_one_per_statistic():
    with pytest.raises(ValueError):
        draw.plot_relaxation(
            np.arange(3.0), np.zeros((3, 2)), np.arange(3.0), np.zeros((3, 2)), ["a"]
        )


# ------------------------------------------------------------------ ensemble loss
def test_ensemble_loss_draws_the_member_mean_with_a_sem_band():
    epoch = np.array([100.0, 200.0, 300.0])
    mean = {"loss": np.array([3.0, 2.0, 1.0]), "mse": np.array([0.3, 0.2, 0.1])}
    sem = {"loss": np.array([0.1, 0.1, 0.1]), "mse": np.array([0.01, 0.01, 0.01])}
    ax = draw.plot_ensemble_loss(epoch, mean, sem)
    assert ax.get_yscale() == "log" and ax.get_xlabel() == "epoch"
    loss = _line_labelled(ax, "loss (total)")
    np.testing.assert_allclose(loss.get_ydata(), mean["loss"])
    assert len(ax.collections) == 2  # one band per series
    band = ax.collections[0].get_paths()[0].vertices[:, 1]
    assert band.min() == pytest.approx(0.9) and band.max() == pytest.approx(3.1)


def test_ensemble_loss_marks_an_all_zero_series():
    epoch = np.array([1.0, 2.0])
    ax = draw.plot_ensemble_loss(
        epoch,
        {"loss": np.ones(2), "rate_l2": np.zeros(2)},
        {"loss": np.ones(2), "rate_l2": np.zeros(2)},
    )
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "rate L2 (raw) (zero)" in labels


def test_ensemble_loss_refuses_an_unknown_series():
    with pytest.raises(ValueError, match="unknown"):
        draw.plot_ensemble_loss(np.ones(2), {"other": np.ones(2)}, {"other": np.ones(2)})


# ------------------------------------------------------------------ sweep scale, colour, panels
def test_a_single_settings_theory_is_visible_as_a_marker():
    ax = draw.plot_share_sweep([3.0], [[0.5, 0.4]], [[0.02, 0.02]], [[0.55, 0.45]])
    line = _line_labelled(ax, "mode 1 theory")
    assert line.get_marker() == "x" and line.get_xdata().tolist() == [3.0]


@pytest.mark.parametrize(
    "values, scale",
    [
        ([0.25, 1.0, 3.0, 10.0], "log"),  # a beta sweep: positive, 40-fold
        ([0.0, 3.0], "linear"),  # setting indices start at 0
        ([1.0, 2.0, 4.0], "linear"),  # under LOG_SWEEP_SPAN
        ([-1.0, 20.0], "linear"),
        ([5.0], "linear"),
    ],
)
def test_the_sweep_scale_is_chosen_at_the_drawing_edge(values, scale):
    assert draw.sweep_scale(values) == scale
    n = len(values)
    ax = draw.plot_share_sweep(values, np.full((n, 1), 0.5), np.zeros((n, 1)), np.full((n, 1), 0.5))
    assert ax.get_xscale() == scale


def test_the_angle_histogram_is_grey_not_a_modes_colour():
    ax = draw.plot_angle_histogram(np.ones(N_THETA_BINS))
    color = ax.containers[0][0].get_facecolor()[:3]
    assert np.allclose(color, matplotlib.colors.to_rgb(draw._GREY))
    for mode_color in draw.LATENT_COLORS[:3]:
        assert not np.allclose(color, matplotlib.colors.to_rgb(mode_color))


def test_fig_panels_draws_one_titled_panel_per_entry_side_by_side():
    s = _spectra()
    panels = [
        {"measured": s, "se": 0.1 * s, "theory": s},
        {"measured": 2 * s, "se": s, "theory": s},
    ]
    fig = draw.fig_panels(draw.plot_spectrum, panels, ["a", "b"], null_theory=0.0)
    assert [ax.get_title() for ax in fig.axes] == ["a", "b"]
    width, height = draw.PANEL_SIZE
    assert tuple(fig.get_size_inches()) == pytest.approx((2 * width, height))
    np.testing.assert_allclose(_points(_errorbars(fig.axes[1])[0])[1], 2 * s[0])
    with pytest.raises(ValueError, match="one title per panel"):
        draw.fig_panels(draw.plot_spectrum, panels, ["a"])


def test_ensemble_loss_puts_its_legend_beside_the_axis_like_plot_loss_curves():
    history = {"epoch": [1, 2], "loss": [1.0, 1.0], "mse": [1.0, 1.0],
               "weight_l2": [1.0, 1.0], "rate_l2": [1.0, 1.0]}  # fmt: skip
    reference = draw.plot_loss_curves(history).get_legend()
    legend = draw.plot_ensemble_loss(
        np.arange(1.0, 3.0), {"loss": np.ones(2)}, {"loss": np.ones(2)}
    ).get_legend()
    assert legend._loc == reference._loc
