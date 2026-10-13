"""Activation-function registry and derivatives."""

from collections.abc import Callable

import torch

Activation = Callable[[torch.Tensor], torch.Tensor]

PHI: dict[str, Activation] = {"relu": torch.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}


def get_phi(name: str) -> Activation:
    if name not in PHI:
        raise KeyError(f"unknown activation {name!r}; valid names: {sorted(PHI)}")
    return PHI[name]


def _relu_prime(h: torch.Tensor) -> torch.Tensor:
    return (h > 0).to(h.dtype)


def _tanh_prime(h: torch.Tensor) -> torch.Tensor:
    return 1 - torch.tanh(h) ** 2


def _sigmoid_prime(h: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(h)
    return s * (1 - s)


def _autograd_prime(phi: Activation) -> Activation:
    def prime(h: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            h = h.detach().requires_grad_(True)
            (grad,) = torch.autograd.grad(phi(h).sum(), h)
        return grad

    return prime


_ANALYTIC_PRIME = {
    torch.relu: _relu_prime,
    torch.nn.functional.relu: _relu_prime,
    torch.tanh: _tanh_prime,
    torch.nn.functional.tanh: _tanh_prime,
    torch.sigmoid: _sigmoid_prime,
    torch.nn.functional.sigmoid: _sigmoid_prime,
}


def phi_prime(phi: Activation) -> Activation:
    """Elementwise derivative of phi: analytic for the registered activations, else autograd.

    The autograd fallback assumes phi acts elementwise (it differentiates the sum).
    """
    return _ANALYTIC_PRIME.get(phi) or _autograd_prime(phi)
