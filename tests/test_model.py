import math

import pytest
import torch

from fct.activations import get_phi
from fct.config import NetConfig
from fct.model import MLP, make_net

P, D_IN, D_OUT, N = 5, 3, 2, 4
CPU, F64 = torch.device("cpu"), torch.float64


def _net(L: int, seed: int = 0, phi: str = "tanh", g: float = 1.5) -> MLP:
    return make_net(NetConfig(N=N, L=L, g=g, phi=phi, seed=seed), D_IN, D_OUT, CPU, F64)


def _x() -> torch.Tensor:
    return torch.randn(P, D_IN, dtype=F64, generator=torch.Generator().manual_seed(123))


def _x_wrong() -> torch.Tensor:
    gen = torch.Generator().manual_seed(1)
    return torch.randn(P, D_IN + 1, dtype=F64, generator=gen)


@pytest.mark.parametrize("L", [1, 3])
def test_forward_shapes(L):
    H, Z = _net(L)(_x())
    assert H.shape == (P, L, N)
    assert Z.shape == (P, D_OUT)


def test_single_layer_has_no_J():
    net = _net(1)
    assert len(net.J) == 0
    assert {name for name, _ in net.named_parameters()} == {"U", "V"}


def test_wrong_input_dim_raises():
    with pytest.raises(ValueError, match="D_in"):
        _net(3)(_x_wrong())


@pytest.mark.parametrize("L", [1, 3])
def test_input_scaling_and_readout_of_the_post_activations(L):
    net, x = _net(L), _x()
    H, Z = net(x)
    assert torch.allclose(H[:, 0], x @ net.U.T / math.sqrt(D_IN))
    assert torch.allclose(Z, net.phi(H[:, -1]) @ net.V / N)  # post-activation of the last layer


def test_every_hidden_layer_uses_its_own_J_and_scaling():
    net, x = _net(3, phi="tanh", g=1.5), _x()
    H, _ = net(x)
    for layer in range(2):
        expected = 1.5 / math.sqrt(N) * torch.tanh(H[:, layer]) @ net.J[layer].T
        assert torch.allclose(H[:, layer + 1], expected)


def test_weight_l2_matches_hand_computed_value():
    L, (a, b, c) = 3, (2.0, 3.0, 0.5)
    net = _net(L)
    with torch.no_grad():
        net.U.fill_(a)
        net.V.fill_(b)
        for J in net.J:
            J.fill_(c)
    expected = N * D_IN * a**2 + N * D_OUT * b**2 + (L - 1) * N * N * c**2
    assert net.weight_l2().item() == pytest.approx(expected)


def test_weight_l2_is_differentiable_and_does_not_mutate_parameters():
    net = _net(3)
    before = [p.detach().clone() for p in net.parameters()]
    net.weight_l2().backward()
    assert all(torch.equal(b, p) for b, p in zip(before, net.parameters()))
    assert all(p.grad is not None for p in net.parameters())


def test_make_net_seed_determines_parameters():
    a, b, c = _net(3, seed=7), _net(3, seed=7), _net(3, seed=8)
    assert all(torch.equal(pa, pb) for pa, pb in zip(a.parameters(), b.parameters()))
    assert not any(torch.equal(pa, pc) for pa, pc in zip(a.parameters(), c.parameters()))


def test_make_net_leaves_global_cpu_rng_untouched():
    torch.manual_seed(99)
    state = torch.get_rng_state()
    _net(3, seed=7)
    assert torch.equal(torch.get_rng_state(), state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cpu_build_does_not_touch_cuda_rng():
    before = [s.clone() for s in torch.cuda.get_rng_state_all()]
    was_initialized = torch.cuda.is_initialized()
    _net(3, seed=7)
    assert torch.cuda.is_initialized() == was_initialized
    assert all(torch.equal(a, b) for a, b in zip(before, torch.cuda.get_rng_state_all()))


@pytest.mark.parametrize("bad", [{"L": 0}, {"N": 0}, {"D_in": 0}, {"D_out": 0}])
def test_nonpositive_sizes_raise(bad):
    sizes = {"D_in": D_IN, "D_out": D_OUT, "N": N, "L": 3} | bad
    with pytest.raises(ValueError, match=next(iter(bad))):
        MLP(**sizes, g=1.0, phi=torch.tanh, device=CPU, dtype=F64)


def test_device_and_dtype_follow_the_parameters():
    net = _net(3).to(torch.float32)
    assert net.dtype == torch.float32
    assert net.device.type == "cpu"


def test_make_net_resolves_activation_by_name():
    assert _net(3, phi="relu").phi is get_phi("relu")


def test_make_net_unknown_activation_raises():
    cfg = NetConfig.model_construct(N=N, L=3, g=1.0, phi="nope", seed=0)
    with pytest.raises(KeyError, match="nope"):
        make_net(cfg, D_IN, D_OUT, CPU, F64)


def test_state_dict_round_trips_through_weights_only_load(tmp_path):
    net, x = _net(3, seed=4), _x()
    path = tmp_path / "net.pt"
    torch.save(net.state_dict(), path)
    fresh = _net(3, seed=5)
    fresh.load_state_dict(torch.load(path, weights_only=True))
    for out_a, out_b in zip(net(x), fresh(x)):
        assert torch.equal(out_a, out_b)
