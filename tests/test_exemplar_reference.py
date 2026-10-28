"""The Picard reference of notebooks/exemplar_reference.py recovers a fixed point known by
quadrature (the weak-response tanh problem of tests/test_meanfield_step_control.py)."""

import torch

from fct.hmc import relative_fro
from notebooks.exemplar_reference import picard_reference
from tests.test_meanfield_step_control import WEAK_CONSTANTS, weak_tanh_problem


def test_the_picard_reference_recovers_the_quadrature_fixed_point():
    """A far smaller sampler budget than the script's default (8192 chains, 100 + 100
    transitions, 30 iterations) suffices on this 8-point problem: it ends 0.004 from the
    quadrature fixed point, against 0.26 for the untilted start, in about a second."""
    p = weak_tanh_problem()
    Sigma = WEAK_CONSTANTS["prior_scale"] * p.X @ p.X.T / p.X.shape[1]
    Phi, change = picard_reference(Sigma, p.target, 0.0, torch.tanh, n_chains=1024,
                                   warmup_trans=30, n_trans=30, n_iter=12)  # fmt: skip
    assert relative_fro(p.moments(torch.zeros_like(Sigma)), p.Phi_star) > 0.2  # far to go
    assert relative_fro(Phi, p.Phi_star) < 0.02
    assert change < 0.02
