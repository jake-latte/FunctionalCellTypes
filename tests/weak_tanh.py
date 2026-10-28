"""The weak-response tanh problem shared by the step-control and Picard-reference tests: a
layer whose saddle point is known by quadrature, never by the solver."""

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import torch

F64 = {"device": "cpu", "dtype": torch.float64}

# A tanh layer on two-dimensional inputs, h = sqrt(prior_scale / 2) X w with w ~ N(0, I_2):
# every expectation of its single-site density is a two-dimensional integral, done here by
# Gauss-Hermite quadrature, and the saddle point by a damped Picard iteration on Φ, never by
# the solver. The kernel is nearly rank two and responds to the tilt with a fifth or less of
# the Gaussian response Φ dΦ̂ Φ: the case the health-only, trace-scaled solver did not solve.
WEAK_P, WEAK_CONSTANTS = 8, {"prior_scale": 5.0, "alpha": 2.0, "learning_prefactor": 0.2}


class WeakTanh(NamedTuple):
    X: torch.Tensor
    Y: torch.Tensor
    moments: Callable[[torch.Tensor], torch.Tensor]  # tilt Φ̂ -> ⟨ψψᵀ⟩, by quadrature
    target: Callable[[torch.Tensor], torch.Tensor]  # Φ -> Stein target -2c A Y Yᵀ A
    Phi_star: torch.Tensor


def weak_tanh_problem(n_nodes: int = 60) -> WeakTanh:
    gen = torch.Generator().manual_seed(0)
    X = torch.randn(WEAK_P, 2, generator=gen, **F64)
    Y = X @ torch.randn(2, 3, generator=gen, **F64)  # a linear teacher
    nodes, weights = np.polynomial.hermite_e.hermegauss(n_nodes)
    w = torch.tensor(np.stack(np.meshgrid(nodes, nodes, indexing="ij"), -1).reshape(-1, 2), **F64)
    q = torch.tensor(np.outer(weights, weights).reshape(-1), **F64)
    psi = torch.tanh(w @ X.T * (WEAK_CONSTANTS["prior_scale"] / 2) ** 0.5)
    eye = torch.eye(WEAK_P, **F64)

    def moments(M):
        log_tilt = -0.5 * ((psi @ M) * psi).sum(1)
        p = q * torch.exp(log_tilt - log_tilt.max())
        return (psi * (p / p.sum())[:, None]).T @ psi

    def target(Phi):
        A = torch.linalg.inv(WEAK_CONSTANTS["alpha"] * eye + Phi)
        return -2.0 * WEAK_CONSTANTS["learning_prefactor"] * A @ Y @ Y.T @ A

    Phi = moments(torch.zeros_like(eye))
    for _ in range(300):
        Phi = 0.5 * Phi + 0.5 * moments(target(Phi))
    return WeakTanh(X, Y, moments, target, Phi)
