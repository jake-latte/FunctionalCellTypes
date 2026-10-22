"""The kernel figure renders from a synthetic spatial spec (Agg backend, CPU)."""

import matplotlib.pyplot as plt
import pytest
from conftest import assert_png

from fct import draw
from fct.spatial import EmbeddingSpec, PopulationSpec, SpatialConfig, build_spaces


def _random_cfg(d):
    spec = PopulationSpec(embedding=EmbeddingSpec(kind="random", seed=1))
    return SpatialConfig(d=d, input=spec, hidden=spec, output=spec)


def test_fig_kernel_renders_a_ring(tmp_path):
    space = build_spaces(SpatialConfig(), m=6, N=8, L=2, n_out=4)[1]
    fig = draw.fig_kernel(space)
    assert_png(fig, tmp_path)
    ring = next(ax for ax in fig.axes if ax.get_title() == "positions")
    assert ring.collections


def test_fig_kernel_draws_points_and_analytic_curves_against_distance():
    space = build_spaces(SpatialConfig(), m=6, N=8, L=2, n_out=4)[1]
    fig = draw.fig_kernel(space)
    ax = next(ax for ax in fig.axes if "distance" in ax.get_xlabel())
    assert len(ax.collections) == 2 and len(ax.lines) == 2  # pre and post, each points + curve


def test_fig_kernel_renders_a_plane(tmp_path):
    space = build_spaces(_random_cfg(d=2), m=5, N=7, L=1, n_out=3)[0]
    assert_png(draw.fig_kernel(space), tmp_path)


def test_fig_kernel_rejects_d3_without_opening_a_figure():
    space = build_spaces(_random_cfg(d=3), m=4, N=4, L=1, n_out=2)[0]
    n_before = len(plt.get_fignums())
    with pytest.raises(ValueError, match="d = 3"):
        draw.fig_kernel(space)
    assert len(plt.get_fignums()) == n_before
