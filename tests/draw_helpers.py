"""Helpers shared by the figure tests."""


def assert_png(fig, tmp_path):
    """The figure renders to a non-empty PNG."""
    out = tmp_path / "fig.png"
    fig.savefig(out)
    assert out.stat().st_size > 0
