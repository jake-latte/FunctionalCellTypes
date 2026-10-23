"""The spatial weight prior in training: the factorised penalty against brute force, the
config/hash contract, smoothness of trained weights (ground truth) and training end to end."""

import torch

from fct.config import Config, DataConfig, NetConfig, TrainConfig, default_config
from fct.data import make_dataset
from fct.model import make_net
from fct.train import compute_loss

CPU, F64 = torch.device("cpu"), torch.float64

# ---------------------------------------------------------------- pins of runs without the prior

# Section hashes of three configs, written by the code before `Config.spatial` existed: a run
# without a spatial prior must keep its caches.
SMALL = Config(
    data=DataConfig(P=16, D_in=4, d=2, embedding="linear", seed=1),
    net=NetConfig(N=8, L=2, g=1.0, phi="tanh", seed=2),
    train=TrainConfig(n_epochs=20, lr=1e-2, log_int=5, save_int=5, rate_lambda=1e-3),
    dtype="float64",
    device="cpu",
)
TRAIN = ("data", "net", "train")
ANALYSIS = ("data", "net", "train", "analysis")
HASH_PINS = [
    (default_config(), {TRAIN: "1628f5950dd5", ANALYSIS: "89bef42003a8"}),
    (
        Config(train=TrainConfig(optimizer="sgld", beta=2.0)),
        {TRAIN: "612e50816922", ANALYSIS: "084652f7164b"},
    ),
    (SMALL, {TRAIN: "b03d959af82e", ANALYSIS: "314b5c6191a3"}),
]
# compute_loss on the SMALL net (weight_lambda=0.1, rate_lambda=1e-3) and the squared norm of
# each parameter's gradient, as float.hex, from the same code.
LOSS_PIN = "0x1.f2fd0476077f9p+5"
GRAD_SQ_PINS = {
    "U": "0x1.2c8058b160712p+2",
    "V": "0x1.e79a6655c4412p+2",
    "J.0": "0x1.a72dff2cbbd3ep+3",
}


def _small_net_and_data():
    ds = make_dataset(SMALL.data, CPU, F64)
    net = make_net(SMALL.net, ds.x.shape[1], ds.y.shape[1], CPU, F64)
    return net, ds


def test_hashes_without_a_spatial_prior_are_unchanged():
    for cfg, pins in HASH_PINS:
        for sections, want in pins.items():
            assert cfg.section_hash(*sections) == want, sections


def test_default_loss_and_gradient_are_bit_for_bit_unchanged():
    net, ds = _small_net_and_data()
    loss, _ = compute_loss(net, ds.x, ds.y, 0.1, 1e-3)
    loss.backward()
    assert loss.item().hex() == LOSS_PIN
    for name, p in net.named_parameters():
        assert p.grad.square().sum().item().hex() == GRAD_SQ_PINS[name], name
