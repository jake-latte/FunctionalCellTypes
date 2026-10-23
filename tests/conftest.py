"""Suite-wide setup."""

import matplotlib

# One headless backend for every test, set before pyplot is imported.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def close_figures():
    """Close every figure a test opened, so figures never pile up across tests."""
    yield
    plt.close("all")
