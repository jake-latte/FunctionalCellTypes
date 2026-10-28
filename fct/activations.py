"""Activation-function registry and derivatives."""

from collections.abc import Callable
from typing import Literal

import torch

Activation = Callable[[torch.Tensor], torch.Tensor]


def identity(h: torch.Tensor) -> torch.Tensor:
    """The linear activation, registered as "id": a net with it is linear in its input. It is also
    the Gaussian case of the single-site density, whose moments are known in closed form."""
    return h


PHI: dict[str, Activation] = {
    "relu": torch.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "id": identity,
}


def get_phi(name: str) -> Activation:
    if name not in PHI:
        raise KeyError(f"unknown activation {name!r}; valid names: {sorted(PHI)}")
    return PHI[name]


def _identity_prime(h: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(h)


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
    identity: _identity_prime,
}


# How the tilt -1/2 phi(h)^T M phi(h) behaves for large h, which decides when the single-site
# density is normalisable (see `fct.hmc.tilt_rule`): "bounded" phi keeps the tilt
# bounded, "linear" makes the density Gaussian, "relu" is positively 1-homogeneous (the tilt
# grows quadratically, but only on the orthant where units are active).
TiltClass = Literal["bounded", "linear", "relu"]
_TILT_CLASS: dict[Activation, TiltClass] = {
    torch.tanh: "bounded",
    torch.nn.functional.tanh: "bounded",
    torch.sigmoid: "bounded",
    torch.nn.functional.sigmoid: "bounded",
    identity: "linear",
    torch.relu: "relu",
    torch.nn.functional.relu: "relu",
}


def known_tilt_class(phi: Activation) -> TiltClass | None:
    """The normalisability class of phi, None for an activation without a known rule."""
    return _TILT_CLASS.get(phi)


def tilt_class(phi: Activation) -> TiltClass:
    """The normalisability class of phi; ValueError for an activation without a known rule."""
    if known_tilt_class(phi) is None:
        raise ValueError(
            f"no normalisability rule for the activation {phi!r}; known: tanh, sigmoid, relu "
            "and fct.activations.identity"
        )
    return _TILT_CLASS[phi]


def phi_prime(phi: Activation) -> Activation:
    """Elementwise derivative of phi: analytic for the registered activations, else autograd.

    The autograd fallback assumes phi acts elementwise (it differentiates the sum).
    """
    return _ANALYTIC_PRIME.get(phi) or _autograd_prime(phi)
