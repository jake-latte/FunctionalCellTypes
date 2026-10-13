import pytest
import torch

from fct.activations import PHI, get_phi, phi_prime


def test_relu_is_torch_relu():
    assert get_phi("relu") is torch.relu


def test_unknown_name_lists_valid_names():
    with pytest.raises(KeyError) as err:
        get_phi("gelu")
    for name in PHI:
        assert name in str(err.value)


@pytest.mark.parametrize("name", ["relu", "tanh", "sigmoid"])
def test_phi_prime_matches_autograd(name):
    phi = get_phi(name)
    h = torch.randn(200, dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad(phi(h).sum(), h)
    assert torch.allclose(phi_prime(phi)(h.detach()), grad, atol=1e-12, rtol=0)


def test_phi_prime_autograd_fallback_for_custom_function():
    def cube(h):
        return h**3

    h = torch.randn(50, dtype=torch.float64)
    assert torch.allclose(phi_prime(cube)(h), 3 * h**2, atol=1e-12)


def test_phi_prime_recognises_functional_relu():
    h = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float64)
    assert phi_prime(torch.nn.functional.relu)(h).tolist() == [0.0, 0.0, 1.0]
