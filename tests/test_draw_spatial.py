"""The kernel figure renders from a synthetic spatial spec (Agg backend, CPU)."""

import matplotlib.pyplot as plt
import pytest

from fct import draw
from fct.spatial import EmbeddingSpec, PopulationSpec, SpatialConfig, build_spaces


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_fig_kernel_ring():
    space = build_spaces(SpatialConfig(), m=6, N=8, L=2, n_out=4)[1]
    fig = draw.fig_kernel(space)
    assert len(fig.axes) >= 4
    ring = next(ax for ax in fig.axes if ax.get_title() == "positions")
    assert ring.collections or ring.lines
    scatter = next(ax for ax in fig.axes if "distance" in ax.get_xlabel())
    assert scatter.collections and scatter.lines  # K_ij points and the analytic curves


def test_fig_kernel_plane():
    spec = PopulationSpec(embedding=EmbeddingSpec(kind="random", seed=1))
    cfg = SpatialConfig(d=2, input=spec, hidden=spec, output=spec)
    fig = draw.fig_kernel(build_spaces(cfg, m=5, N=7, L=1, n_out=3)[0])
    assert len(fig.axes) >= 4


def test_fig_kernel_rejects_d3():
    spec = PopulationSpec(embedding=EmbeddingSpec(kind="random"))
    cfg = SpatialConfig(d=3, input=spec, hidden=spec, output=spec)
    with pytest.raises(ValueError, match="d = 3"):
        draw.fig_kernel(build_spaces(cfg, m=4, N=4, L=1, n_out=2)[0])
