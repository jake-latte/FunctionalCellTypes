"""Suite-wide setup."""

import matplotlib
import matplotlib.pyplot as plt
import pytest

# One headless backend for every test, set before any test module imports pyplot.
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_figures():
    """Close every figure a test opened, so figures never pile up across tests."""
    yield
    plt.close("all")


def assert_png(fig, tmp_path):
    """The figure renders to a non-empty PNG."""
    out = tmp_path / "fig.png"
    fig.savefig(out)
    assert out.stat().st_size > 0
