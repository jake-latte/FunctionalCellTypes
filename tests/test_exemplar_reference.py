"""The Picard reference of notebooks/exemplar_reference.py recovers a fixed point known by
quadrature (the weak-response tanh problem of tests/weak_tanh.py), and is wired to the
solver's own problem.

The bound is 0.015. Over seeds 0-19 `exemplar_reference` lands 0.0016 to 0.0112 from the
quadrature fixed point (seeds 0-9 of `picard_reference` on the same problem 0.002 to 0.007),
while with the learning prefactor 20 % low it lands 0.025 to 0.038 from it, so the bound tells
a correct target from a slightly wrong one. Each run takes about 1.5 s at this budget, far
below the script's own (8192 chains, 100 + 100 transitions, 30 iterations), which this
8-point problem does not need."""

import pytest
import torch

from fct.config import Config, MeanFieldConfig, NetConfig
from fct.data import Dataset
from fct.hmc import relative_fro
from notebooks.exemplar_reference import exemplar_reference, picard_reference
from tests.weak_tanh import WEAK_CONSTANTS, WEAK_P, weak_tanh_problem

BUDGET = {"n_chains": 1024, "warmup_trans": 30, "n_trans": 30, "n_iter": 12}
BOUND = 0.015


@pytest.fixture(scope="module")
def weak_tanh():
    return weak_tanh_problem()


def weak_reference(p, scale_target: float = 1.0):
    Sigma = WEAK_CONSTANTS["prior_scale"] * p.X @ p.X.T / p.X.shape[1]
    return picard_reference(Sigma, lambda Phi: scale_target * p.target(Phi), 0.0, torch.tanh,
                            **BUDGET)  # fmt: skip


def weak_config(**meanfield) -> Config:
    constants = {**WEAK_CONSTANTS, "rate_tilt": 0.0, **meanfield}
    return Config(net=NetConfig(L=1, phi="tanh"),
                  meanfield=MeanFieldConfig(n_points=WEAK_P, **constants))  # fmt: skip


def weak_dataset(p) -> Dataset:
    return Dataset(x=p.X, y=p.Y, Q=None, period=None)


def test_the_picard_reference_recovers_the_quadrature_fixed_point(weak_tanh):
    Phi, change = weak_reference(weak_tanh)
    assert relative_fro(Phi, weak_tanh.Phi_star) < BOUND
    assert change < 0.02


def test_a_learning_prefactor_off_by_a_fifth_misses_the_bound(weak_tanh):
    Phi, _ = weak_reference(weak_tanh, scale_target=0.8)
    assert relative_fro(Phi, weak_tanh.Phi_star) > BOUND


def test_the_exemplar_reference_solves_the_configured_problem(weak_tanh):
    """The config's explicit constants reach the prior and the Stein target: the reference of
    the weak problem's config is its quadrature fixed point, and with the learning prefactor
    set 20 % low it is not."""
    Phi, change = exemplar_reference(weak_config(), weak_dataset(weak_tanh), **BUDGET)
    assert relative_fro(Phi, weak_tanh.Phi_star) < BOUND
    assert change < 0.02
    low_c = weak_config(learning_prefactor=0.8 * WEAK_CONSTANTS["learning_prefactor"])
    Phi_low, _ = exemplar_reference(low_c, weak_dataset(weak_tanh), **BUDGET)
    assert relative_fro(Phi_low, weak_tanh.Phi_star) > BOUND


@pytest.mark.parametrize(
    ("net", "n_points", "match"),
    [
        (NetConfig(L=2, phi="tanh"), WEAK_P, "one hidden layer"),
        (NetConfig(L=1, phi="relu"), WEAK_P, "needs a bounded phi"),
        (NetConfig(L=1, phi="id"), WEAK_P, "needs a bounded phi"),
        (NetConfig(L=1, phi="tanh"), WEAK_P + 1, "exceeds the dataset size"),
    ],
)
def test_the_exemplar_reference_rejects_what_it_cannot_solve(weak_tanh, net, n_points, match):
    cfg = weak_config().model_copy(
        update={"net": net, "meanfield": MeanFieldConfig(n_points=n_points, **WEAK_CONSTANTS)}
    )
    with pytest.raises(ValueError, match=match):
        exemplar_reference(cfg, weak_dataset(weak_tanh), **BUDGET)
