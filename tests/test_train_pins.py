"""A single-network run (net.B = 1) reproduces the pinned runs of tests/pinned_training.py bit
for bit: config hash, every checkpoint and the final file."""

import pytest
import torch

from tests.pinned_training import FIXTURES, run_pinned


def _assert_identical(got, want, where: str) -> None:
    if isinstance(want, torch.Tensor):
        assert isinstance(got, torch.Tensor), where
        assert got.dtype == want.dtype and got.shape == want.shape, where
        assert torch.equal(got, want), where
    elif isinstance(want, dict):
        assert set(got) == set(want), where
        for key in want:
            _assert_identical(got[key], want[key], f"{where}.{key}")
    elif isinstance(want, list | tuple):
        assert type(got) is type(want) and len(got) == len(want), where
        for i, (g, w) in enumerate(zip(got, want, strict=True)):
            _assert_identical(g, w, f"{where}[{i}]")
    else:
        assert type(got) is type(want) and got == want, where


@pytest.mark.parametrize("name", ["adam", "sgld"])
def test_single_network_run_is_bitwise_pinned(tmp_path, name):
    want = torch.load(FIXTURES / f"train_{name}.pt", weights_only=True)
    got = run_pinned(name, tmp_path)
    assert set(want["files"]) == {"ckpt_000003.pt", "ckpt_000006.pt", "final.pt"}
    _assert_identical(got, want, name)
