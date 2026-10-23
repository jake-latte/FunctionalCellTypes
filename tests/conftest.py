"""Suite-wide setup."""

import matplotlib

# One headless backend for every test, set before pyplot is imported.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Keep pytest's temp dir inside the project: its default is under /tmp, and the project
    forbids files outside its own directory (todo.md). The parent must exist: pytest does
    not create it."""
    if config.option.basetemp is None:
        tmp = config.rootpath / "tmp"
        tmp.mkdir(exist_ok=True)
        config.option.basetemp = tmp / "pytest"


@pytest.fixture(autouse=True)
def close_figures():
    """Close every figure a test opened, so figures never pile up across tests."""
    yield
    plt.close("all")
