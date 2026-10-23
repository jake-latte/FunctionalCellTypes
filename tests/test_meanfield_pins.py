"""Regression pins against outputs written by the solver before the sampler choice existed
(see tests/pinned_runs.py for how and when the fixtures were made).

Bit-for-bit equality holds on the machine and torch build that wrote the fixtures; another
BLAS or thread count may reorder float reductions, which these tests would report.
"""

import json
import math

import pytest
import torch

from fct.config import default_config, load_config
from tests.pinned_runs import (
    FIXTURES,
    HASHES_FILE,
    LEGACY_CONFIG_FILE,
    MEANFIELD_RESULT_SECTIONS,
    PINNED_SOLVES,
    legacy_config,
    pinned_solve,
    pinned_solve_payload,
)


def _same(a, b) -> bool:
    """Exact equality of nested lists / scalars, with NaN equal to NaN."""
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return type(a) is type(b) and a == b


@pytest.mark.parametrize("name", list(PINNED_SOLVES))
def test_hmc_solve_reproduces_the_pinned_output_bit_for_bit(name):
    pinned = torch.load(FIXTURES / f"hmc_solve_{name}.pt", weights_only=True)
    now = pinned_solve_payload(pinned_solve(name))
    assert torch.equal(now["Phi"], pinned["Phi"])
    assert torch.equal(now["Phi_hat"], pinned["Phi_hat"])
    # Columns added since the pin are allowed; every pinned column must be unchanged.
    for key, column in pinned["history"].items():
        assert _same(now["history"][key], column), key


def _pinned_hashes() -> dict:
    return json.loads(HASHES_FILE.read_text())


def test_default_meanfield_hashes_are_unchanged():
    cfg, pins = default_config(), _pinned_hashes()["default"]
    assert cfg.section_hash("meanfield") == pins["meanfield"]
    assert cfg.section_hash(*MEANFIELD_RESULT_SECTIONS) == pins["result"]


def test_legacy_config_file_loads_and_keeps_its_hashes():
    cfg, pins = load_config(LEGACY_CONFIG_FILE), _pinned_hashes()["legacy"]
    assert cfg == legacy_config()
    assert cfg.section_hash("meanfield") == pins["meanfield"]
    assert cfg.section_hash(*MEANFIELD_RESULT_SECTIONS) == pins["result"]
