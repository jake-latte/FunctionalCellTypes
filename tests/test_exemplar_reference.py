"""The Picard reference of notebooks/exemplar_reference.py recovers a fixed point known by
quadrature (the weak-response tanh problem of tests/test_meanfield_step_control.py)."""

import torch

from fct.hmc import relative_fro
from notebooks.exemplar_reference import picard_reference
from tests.test_meanfield_step_control import WEAK_CONSTANTS, weak_tanh_problem


def test_the_picard_reference_recovers_the_quadrature_fixed_point():
    p = weak_tanh_problem()
    Sigma = WEAK_CONSTANTS["prior_scale"] * p.X @ p.X.T / p.X.shape[1]
    Phi, change = picard_reference(Sigma, p.target, 0.0, torch.tanh, n_chains=2048,
                                   warmup_trans=50, n_trans=50, n_iter=15)  # fmt: skip
    untilted = p.moments(torch.zeros_like(Sigma))
    print(f"\nPicard reference: {relative_fro(Phi, p.Phi_star):.4f} from the quadrature fixed "
          f"point (untilted start {relative_fro(untilted, p.Phi_star):.3f}), last change {change:.1e}")  # fmt: skip
    assert relative_fro(Phi, p.Phi_star) < 0.02
    assert change < 0.02
