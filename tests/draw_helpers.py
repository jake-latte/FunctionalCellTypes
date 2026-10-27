"""Helpers shared by the figure tests."""

import torch


def assert_png(fig, tmp_path):
    """The figure renders to a non-empty PNG."""
    out = tmp_path / "fig.png"
    fig.savefig(out)
    assert out.stat().st_size > 0


class FakeDeviceTensor(torch.Tensor):
    """Stands in for a CUDA tensor on a CPU-only host: like one, `.numpy()` raises TypeError until
    `.cpu()` is called. Subclass-ness survives arithmetic, so a value computed from one still
    refuses `.numpy()`, as a value computed on the GPU would. `.cpu()` is its only escape: a
    refactor to `.to("cpu")` or `.numpy(force=True)` turning a test red is this fake, not the code."""

    def numpy(self, *args, **kwargs):
        raise TypeError("can't convert cuda:0 device type tensor to numpy.")

    def cpu(self, *args, **kwargs):
        return self.as_subclass(torch.Tensor)


def on_fake_device(t: torch.Tensor) -> FakeDeviceTensor:
    return t.as_subclass(FakeDeviceTensor)
