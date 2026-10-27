"""The notebook explanation must agree with the code it describes: numbers come from the code's
own symbols, and the quoted examples are recomputed with the real functions."""

import re

import pytest
import torch

from fct.config import TrainConfig
from fct.explanations import MIR_EXPLANATION
from fct.modularity import (
    DEAD_TOL,
    NOISE_FACTOR,
    layer_mir,
    neuron_factor_information,
    preference,
)

P, BINS, N_NOISE_NEURONS = 8192, 10, 64


def test_quoted_defaults_are_the_code_defaults():
    assert f"{TrainConfig().monitor_bins} by default" in MIR_EXPLANATION
    assert f"{TrainConfig().monitor_shuffles} by default" in MIR_EXPLANATION
    assert "@" not in MIR_EXPLANATION  # every placeholder was filled


def test_quoted_dead_threshold_is_the_one_preference_uses():
    exponent = re.search(r"at most \$10\^\{(-?\d+)\}\$", MIR_EXPLANATION)
    assert exponent and 10.0 ** int(exponent.group(1)) == DEAD_TOL
    infos = torch.tensor([[DEAD_TOL, 0.0], [10 * DEAD_TOL, 0.0]], dtype=torch.float64)
    _, dead = preference(infos)
    assert dead.tolist() == [True, False]


def test_quoted_noise_rule_factor_is_the_one_the_correction_applies():
    quoted = re.search(r"\\le (\d+(?:\.\d+)?)\\,\\bar", MIR_EXPLANATION)
    assert quoted and float(quoted.group(1)) == NOISE_FACTOR
    # Neuron 0 is informative about the factor, neuron 1 is noise: the rule zeroes only neuron 1's member.
    g = torch.Generator().manual_seed(0)
    labels = torch.randint(0, 2, (P,), generator=g)
    informative = labels.double() + 0.1 * torch.randn(P, generator=g, dtype=torch.float64)
    noise = torch.randn(P, generator=g, dtype=torch.float64)
    info = neuron_factor_information(
        torch.stack([informative, noise], dim=1),
        {"a": labels},
        n_bins=BINS,
        bias_correction="shuffle",
    )
    assert info.member_mi[0].sum() > 0 and info.member_mi[1].sum() == 0


def _noise_raw_mir(cardinalities: list[int]) -> float:
    g = torch.Generator().manual_seed(1)
    rates = torch.randn(P, N_NOISE_NEURONS, generator=g, dtype=torch.float64)
    factors = {f"f{i}": torch.randint(0, c, (P,), generator=g) for i, c in enumerate(cardinalities)}
    return layer_mir(rates, factors, n_bins=BINS).mir


@pytest.mark.parametrize(("cardinalities", "quoted"), [([25, 2, 2], "0.89"), ([5, 5, 5], "0.10")])
def test_quoted_floor_examples_are_what_noise_neurons_score(cardinalities, quoted):
    assert quoted in MIR_EXPLANATION
    assert _noise_raw_mir(cardinalities) == pytest.approx(float(quoted), abs=0.04)


def test_equations_are_balanced():
    assert MIR_EXPLANATION.count("$$") % 2 == 0
    display = re.findall(r"\$\$(.+?)\$\$", MIR_EXPLANATION, flags=re.DOTALL)
    assert len(display) >= 8
    prose = re.sub(r"\$\$.+?\$\$", "", MIR_EXPLANATION, flags=re.DOTALL)
    assert prose.count("$") % 2 == 0  # every inline equation is closed
    inline = re.findall(r"\$(.+?)\$", prose)
    assert inline
    for equation in display + inline:
        for opening, closing in ("{}", "[]"):
            assert equation.count(opening) == equation.count(closing), equation
        for size in ("\\big", "\\Big"):
            assert equation.count(f"{size}(") == equation.count(f"{size})"), equation
