"""Ensembles of B non-interacting networks (net.B > 1): construction, loss, training, resume,
checkpoints, the monitor's member axis and member selection."""

import pytest
import torch

from fct.cache import cache_paths
from fct.config import Config, DataConfig, NetConfig, TrainConfig
from fct.data import make_dataset
from fct.model import MLP, EnsembleMLP, make_net, member_seed
from fct.monitor import MIR_MONITOR_KEYS, make_mir_monitor
from fct.sgld import SGLD
from fct.train import (
    MONITOR_EPOCH,
    _grad_norm,
    _step,
    compute_loss,
    load_latest_checkpoint,
    load_trained,
    make_weight_penalty,
    member_history,
    prior_coefficient,
    select_member,
    spatial_weight_penalty,
    train,
)
from tests.pinned_training import pinned_config, usum_monitor

CPU, F64 = torch.device("cpu"), torch.float64
QUIET = {"log": lambda s: None}
D_IN, D_OUT = 4, 2


class Interrupt(Exception):
    pass


def _net_cfg(B: int, L: int = 2, seed: int = 2) -> NetConfig:
    return NetConfig(N=6, L=L, g=1.0, phi="tanh", seed=seed, B=B)


def _ensemble(B: int = 3, L: int = 2) -> EnsembleMLP:
    return make_net(_net_cfg(B, L), D_IN, D_OUT, CPU, F64)


def _x(P: int = 5) -> torch.Tensor:
    return torch.randn(P, D_IN, dtype=F64, generator=torch.Generator().manual_seed(11))


def _setup(cfg: Config):
    ds = make_dataset(cfg.data, CPU, F64)
    return ds, make_net(cfg.net, ds.x.shape[1], ds.y.shape[1], CPU, F64)


def _with_B(cfg: Config, B: int) -> Config:
    return cfg.model_copy(update={"net": cfg.net.model_copy(update={"B": B})})


def _assert_same_state(a: dict, b: dict) -> None:
    assert set(a) == set(b)
    for key, value in a.items():
        assert torch.equal(value, b[key]), key


# ------------------------------------------------------------------ config


def test_B_defaults_to_one_and_must_be_positive():
    assert NetConfig().B == 1
    with pytest.raises(ValueError, match="B"):
        NetConfig(B=0)


def test_B_one_keeps_every_hash_and_B_above_one_changes_the_net_hashes():
    sections = [("net",), ("data", "net", "train"), ("data", "net", "train", "meanfield")]
    plain, one, two = Config(), Config(net=NetConfig(B=1)), Config(net=NetConfig(B=2))
    for s in sections:
        assert one.section_hash(*s) == plain.section_hash(*s)
        assert two.section_hash(*s) != plain.section_hash(*s)
    assert two.section_hash("data", "train") == plain.section_hash("data", "train")


# ------------------------------------------------------------------ model


def test_B_one_builds_the_single_network_and_B_above_one_an_ensemble():
    assert type(make_net(_net_cfg(1), D_IN, D_OUT, CPU, F64)) is MLP
    ens = _ensemble(3)
    assert isinstance(ens, EnsembleMLP) and ens.B == 3
    assert ens.U.shape == (3, 6, D_IN) and ens.V.shape == (3, 6, D_OUT)
    assert ens.J[0].shape == (3, 6, 6)
    assert {n for n, _ in ens.named_parameters()} == {"U", "J.0", "V"}


def test_member_zero_has_the_single_network_weights_and_members_are_distinct():
    single = make_net(_net_cfg(1), D_IN, D_OUT, CPU, F64)
    ens = _ensemble(4)
    _assert_same_state(ens.member(0).state_dict(), single.state_dict())
    members = [ens.member(b).state_dict() for b in range(4)]
    for a in range(4):
        for b in range(a + 1, 4):
            assert all(not torch.equal(members[a][k], members[b][k]) for k in members[a])


def test_a_member_does_not_depend_on_the_ensemble_size():
    small, large = _ensemble(2), _ensemble(5)
    _assert_same_state(small.member(1).state_dict(), large.member(1).state_dict())


def test_member_seeds_are_documented_streams():
    assert member_seed(7, 0) == 7
    seeds = {member_seed(s, b) for s in (-1, 0, 1, 7) for b in range(1, 6)}
    assert len(seeds) == 20 and all(0 <= s < 2**64 for s in seeds)


def test_member_is_an_independent_copy_and_from_members_inverts_it():
    ens = _ensemble(3)
    m = ens.member(1)
    assert type(m) is MLP and (m.N, m.L, m.g, m.D_in, m.D_out) == (6, 2, 1.0, D_IN, D_OUT)
    with torch.no_grad():
        m.U.add_(1.0)
    assert not torch.equal(ens.U[1], m.U)
    rebuilt = EnsembleMLP([ens.member(b) for b in range(3)])
    _assert_same_state(rebuilt.state_dict(), ens.state_dict())


def test_member_draws_no_random_numbers():
    ens = _ensemble(3)
    state = torch.get_rng_state()
    ens.member(2)
    assert torch.equal(torch.get_rng_state(), state)


def test_from_weights_holds_the_given_weights_and_rejects_misfits():
    U, J, V = torch.randn(5, 3), [torch.randn(5, 5)], torch.randn(5, 2)
    net = MLP.from_weights(U, J, V, 1.5, torch.tanh)
    assert (net.N, net.L, net.D_in, net.D_out, net.g) == (5, 2, 3, 2, 1.5)
    assert torch.equal(net.U, U) and torch.equal(net.J[0], J[0]) and torch.equal(net.V, V)
    with pytest.raises(ValueError, match="do not make an MLP"):
        MLP.from_weights(U, [torch.randn(4, 4)], V, 1.0, torch.tanh)


@pytest.mark.parametrize("b", [-1, 3])
def test_member_out_of_range_raises(b):
    with pytest.raises(IndexError, match="member"):
        _ensemble(3).member(b)


def test_from_members_rejects_mismatched_members():
    a = make_net(_net_cfg(1), D_IN, D_OUT, CPU, F64)
    b = make_net(NetConfig(N=7, L=2, phi="tanh"), D_IN, D_OUT, CPU, F64)
    with pytest.raises(ValueError, match="N"):
        EnsembleMLP([a, b])


@pytest.mark.parametrize("L", [1, 3])
def test_forward_is_each_members_forward(L):
    ens, x = _ensemble(3, L), _x()
    H, Z = ens(x)
    assert H.shape == (3, 5, L, 6) and Z.shape == (3, 5, D_OUT)
    for b in range(3):
        h, z = ens.member(b)(x)
        torch.testing.assert_close(H[b], h, rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(Z[b], z, rtol=1e-13, atol=1e-13)


def test_forward_rejects_the_wrong_input_width():
    with pytest.raises(ValueError, match="D_in"):
        _ensemble(2)(torch.zeros(5, D_IN + 1, dtype=F64))


# ------------------------------------------------------------------ loss


def _single_loss_and_grads(net: MLP, x, y, penalty=None):
    net.zero_grad()
    loss, parts = compute_loss(net, x, y, 0.3, 0.1, penalty or MLP.weight_l2)
    loss.backward()
    return parts, {n: p.grad.clone() for n, p in net.named_parameters()}


def test_ensemble_loss_is_the_sum_of_member_losses_with_member_gradients():
    ens, x = _ensemble(3), _x()
    y = torch.randn(5, D_OUT, dtype=F64, generator=torch.Generator().manual_seed(3))
    loss, parts = compute_loss(ens, x, y, 0.3, 0.1)
    loss.backward()
    assert all(v.shape == (3,) for v in parts.values())
    torch.testing.assert_close(loss.detach(), parts["loss"].sum(), rtol=0, atol=0)
    for b in range(3):
        single_parts, grads = _single_loss_and_grads(ens.member(b), x, y)
        for key, value in single_parts.items():
            torch.testing.assert_close(parts[key][b], value, rtol=1e-12, atol=0)
        for name, p in ens.named_parameters():
            torch.testing.assert_close(p.grad[b], grads[name], rtol=1e-12, atol=1e-14)


def test_a_single_network_penalty_on_an_ensemble_fails_loudly():
    ens, x = _ensemble(2), _x()
    with pytest.raises(ValueError, match="per member"):
        compute_loss(ens, x, torch.zeros(5, D_OUT, dtype=F64), 0.3, 0.1, MLP.weight_l2)


def test_spatial_prior_of_an_ensemble_is_the_members_penalties():
    cfg = pinned_config("sgld", B=3)  # a dense spatial prior
    ens = make_net(cfg.net, D_IN, D_OUT, CPU, F64)
    penalty = make_weight_penalty(cfg, ens)
    single_penalty = make_weight_penalty(_with_B(cfg, 1), ens.member(0))
    per_member = penalty(ens)
    assert per_member.shape == (3,)
    for b in range(3):
        torch.testing.assert_close(per_member[b], single_penalty(ens.member(b)), rtol=1e-13, atol=0)
    assert penalty.func is spatial_weight_penalty


def test_grad_norm_of_an_ensemble_is_each_members_with_the_spatial_prior():
    cfg = pinned_config("sgld", B=3)
    ens = make_net(cfg.net, D_IN, D_OUT, CPU, F64)
    singles = [ens.member(b) for b in range(3)]
    x, y = _x(), torch.randn(5, D_OUT, dtype=F64, generator=torch.Generator().manual_seed(3))
    compute_loss(ens, x, y, 0.3, 0.1, make_weight_penalty(cfg, ens))[0].backward()
    norms = _grad_norm(ens)
    assert norms.shape == (3,)
    single_penalty = make_weight_penalty(_with_B(cfg, 1), singles[0])
    for b, single in enumerate(singles):
        compute_loss(single, x, y, 0.3, 0.1, single_penalty)[0].backward()
        torch.testing.assert_close(norms[b], _grad_norm(single), rtol=1e-12, atol=0)


# ------------------------------------------------------------------ training


def _train_cfg(B: int, optimizer: str = "adam", n_epochs: int = 6, **train_kw) -> Config:
    train_kw = {"log_int": 2, "save_int": 3, "lr": 1e-2, "seed": 4, **train_kw}
    if optimizer == "sgld":
        train_kw |= {"optimizer": "sgld", "beta": 50.0, "lr": 1e-3}
    return Config(
        data=DataConfig(P=12, D_in=D_IN, d=2, embedding="linear", seed=1),
        net=_net_cfg(B),
        train=TrainConfig(n_epochs=n_epochs, **train_kw),
        dtype="float64",
        device="cpu",
    )


def test_adam_ensemble_equals_its_members_trained_separately(tmp_path):
    cfg = _train_cfg(3)
    ds, ens = _setup(cfg)
    starts = [ens.member(b) for b in range(3)]
    ens, history = train(cfg, ds, ens, tmp_path / "ens", **QUIET)
    for b, start in enumerate(starts):
        single, single_history = train(_with_B(cfg, 1), ds, start, tmp_path / f"m{b}", **QUIET)
        for name, p in ens.named_parameters():
            torch.testing.assert_close(
                p[b], dict(single.named_parameters())[name], rtol=1e-12, atol=1e-14
            )
        for key in ("mse", "loss", "weight_l2", "rate_l2"):
            got = [entry[b] for entry in history[key]]
            torch.testing.assert_close(
                torch.tensor(got), torch.tensor(single_history[key]), rtol=1e-12, atol=0
            )
    assert history["epoch"] == single_history["epoch"]


def _member_noise(gens):
    """SGLD noise for an ensemble: member b's slice is drawn from generator `gens[b]`."""

    def noise(p):
        return torch.stack([torch.randn(p.shape[1:], generator=g, dtype=p.dtype) for g in gens])

    return noise


def _single_noise(gen):
    return lambda p: torch.randn(p.shape, generator=gen, dtype=p.dtype)


def _sgld_steps(net, optim, x, y, k):
    for _ in range(k):
        optim.zero_grad()
        compute_loss(net, x, y, 0.01, 0.1)[0].backward()
        optim.step()


def test_sgld_ensemble_with_member_noise_streams_equals_separate_members():
    ens, x = _ensemble(3), _x()
    y = torch.randn(5, D_OUT, dtype=F64, generator=torch.Generator().manual_seed(3))
    singles = [ens.member(b) for b in range(3)]
    gens = [torch.Generator().manual_seed(100 + b) for b in range(3)]
    optim = SGLD(ens.parameters(), lr=1e-3, beta=50.0, member_axis=True, noise=_member_noise(gens))
    _sgld_steps(ens, optim, x, y, 5)
    assert optim.last_msd.shape == (3,)
    for b, single in enumerate(singles):
        gen = torch.Generator().manual_seed(100 + b)
        single_optim = SGLD(single.parameters(), lr=1e-3, beta=50.0, noise=_single_noise(gen))
        _sgld_steps(single, single_optim, x, y, 5)
        for name, p in ens.named_parameters():
            torch.testing.assert_close(
                p[b], dict(single.named_parameters())[name], rtol=1e-12, atol=1e-14
            )
        torch.testing.assert_close(optim.last_msd[b], single_optim.last_msd, rtol=1e-10, atol=0)


def test_sgld_training_steps_with_the_spatial_prior_equal_separate_members():
    """Through `fct.train._step` (what `train` runs each epoch) with the dense spatial prior;
    `train` itself builds its SGLD with the global noise stream, so the fixed member streams
    are injected by building the optimisers here."""
    cfg = pinned_config("sgld", B=3)
    ds, ens = _setup(cfg)
    singles = [ens.member(b) for b in range(3)]
    tc = cfg.train
    gens = [torch.Generator().manual_seed(100 + b) for b in range(3)]
    optim = SGLD(ens.parameters(), tc.lr, tc.beta, member_axis=True, noise=_member_noise(gens))
    penalty = make_weight_penalty(cfg, ens)
    parts = [_step(ens, optim, ds, tc, penalty) for _ in range(5)]
    single_cfg = _with_B(cfg, 1)
    for b, single in enumerate(singles):
        gen = torch.Generator().manual_seed(100 + b)
        single_optim = SGLD(single.parameters(), tc.lr, tc.beta, noise=_single_noise(gen))
        single_penalty = make_weight_penalty(single_cfg, single)
        single_parts = [_step(single, single_optim, ds, tc, single_penalty) for _ in range(5)]
        for name, p in ens.named_parameters():
            want = dict(single.named_parameters())[name]
            torch.testing.assert_close(p[b], want, rtol=1e-12, atol=1e-14)
        for got, want in zip(parts, single_parts, strict=True):
            for key in ("loss", "weight_l2", "grad_norm", "param_msd"):
                torch.testing.assert_close(got[key][b], want[key], rtol=1e-10, atol=0)
    assert prior_coefficient(tc) > 0  # the prior is in the loss


def _lambda_max_config(B: int) -> Config:
    """The pinned SGLD run with every kernel scaled to top eigenvalue lambda_max (random
    embeddings, unit amplitudes, weight_lambda 1)."""
    cfg = pinned_config("sgld", B=B)
    spatial = cfg.spatial.model_copy(update={"lambda_max": 0.5})
    train = cfg.train.model_copy(update={"weight_lambda": 1.0})
    return Config.model_validate(
        {**cfg.model_dump(), "spatial": spatial.model_dump(), "train": train.model_dump()}
    )


def test_an_ensemble_trains_under_lambda_max_with_member_penalties(tmp_path):
    cfg = _lambda_max_config(3)
    assert cfg.spatial.lambda_max == 0.5 and cfg.spatial.input.embedding.kind == "random"
    ds, ens = _setup(cfg)
    ens, history = train(cfg, ds, ens, tmp_path, **QUIET)
    assert all(len(entry) == 3 for entry in history["weight_l2"])
    assert torch.isfinite(torch.tensor(history["loss"])).all()
    per_member = make_weight_penalty(cfg, ens)(ens)
    single = make_weight_penalty(_lambda_max_config(1), ens.member(0))
    for b in range(3):
        torch.testing.assert_close(per_member[b], single(ens.member(b)), rtol=1e-13, atol=0)


def test_sgld_member_noises_are_independent():
    """Zero gradients: one step moves each parameter by its noise alone. The empirical
    correlation of two members' noises over n entries has standard error 1 / sqrt(n)."""
    torch.manual_seed(0)
    ens = make_net(NetConfig(N=64, L=2, phi="tanh", B=4), 32, 8, CPU, F64)
    before = [p.detach().clone() for p in ens.parameters()]
    optim = SGLD(ens.parameters(), lr=0.5, beta=1.0, member_axis=True)
    for p in ens.parameters():
        p.grad = torch.zeros_like(p)
    optim.step()
    noise = torch.cat(
        [(p.detach() - p0).reshape(4, -1) for p, p0 in zip(ens.parameters(), before, strict=True)],
        dim=1,
    )
    n = noise.shape[1]
    corr = torch.corrcoef(noise)
    off_diagonal = corr[~torch.eye(4, dtype=torch.bool)]
    assert off_diagonal.abs().max() < 4 / n**0.5
    torch.testing.assert_close(
        noise.var(dim=1), torch.full((4,), 1.0, dtype=F64), rtol=0.05, atol=0
    )


@pytest.mark.parametrize("optimizer", ["adam", "sgld"])
def test_ensemble_history_has_a_member_axis(tmp_path, optimizer):
    cfg = _train_cfg(3, optimizer, monitor_int=3)
    ds, ens = _setup(cfg)
    lines = []
    _, history = train(cfg, ds, ens, tmp_path, log=lines.append, monitor=usum_monitor)
    assert history["epoch"] == [0, 2, 4]
    member_keys = ["mse", "weight_l2", "rate_l2", "loss"]
    member_keys += ["grad_norm", "param_msd"] if optimizer == "sgld" else []
    for key in member_keys:
        assert all(isinstance(v, list) and len(v) == 3 for v in history[key]), key
    assert history[MONITOR_EPOCH] == [0, 3, 6]
    # monitor_<key>: a list over calls of lists over members of the monitor's value
    assert all(len(call) == 3 and len(call[0]) == 2 for call in history["monitor_usum"])
    assert all("B=3" in line for line in lines if line.startswith("["))


def test_monitor_of_an_ensemble_monitors_each_member(tmp_path):
    cfg = _train_cfg(2, monitor_int=3)
    ds, ens = _setup(cfg)
    starts = [ens.member(b) for b in range(2)]
    _, history = train(cfg, ds, ens, tmp_path, monitor=usum_monitor, **QUIET)
    assert history["monitor_usum"][0] == [usum_monitor(m)["usum"] for m in starts]


def test_mir_monitor_runs_on_members_of_an_ensemble(tmp_path):
    cfg = Config(
        data=DataConfig(task="linear_teacher", P=16, D_in=6, n_latents=2, seed=1),
        net=_net_cfg(2),
        train=TrainConfig(n_epochs=2, lr=1e-2, log_int=1, save_int=1, monitor_int=1),
        dtype="float64",
        device="cpu",
    )
    ds, ens = _setup(cfg)
    _, history = train(cfg, ds, ens, tmp_path, monitor=make_mir_monitor(ds, cfg), **QUIET)
    for key in MIR_MONITOR_KEYS:
        calls = history[f"monitor_{key}"]
        assert len(calls) == 3 and all(len(c) == 2 and len(c[0]) == 2 for c in calls), key


@pytest.mark.parametrize("optimizer", ["adam", "sgld"])
def test_resumed_ensemble_equals_an_uninterrupted_one(tmp_path, optimizer):
    cfg = _train_cfg(3, optimizer, n_epochs=8, monitor_int=3)
    ds, ref = _setup(cfg)
    ref, ref_history = train(cfg, ds, ref, tmp_path / "ref", monitor=usum_monitor, **QUIET)

    def stop_at_five(line):
        if line.startswith("[00004"):
            raise Interrupt

    ds, net = _setup(cfg)
    with pytest.raises(Interrupt):
        train(cfg, ds, net, tmp_path / "run", log=stop_at_five, monitor=usum_monitor)
    assert (cache_paths(tmp_path / "run").train / "ckpt_000003.pt").exists()
    ds, net = _setup(cfg)
    net, history = train(cfg, ds, net, tmp_path / "run", monitor=usum_monitor, **QUIET)
    _assert_same_state(net.state_dict(), ref.state_dict())
    assert history == ref_history


def test_load_trained_and_the_latest_checkpoint_return_the_ensemble(tmp_path):
    cfg = _train_cfg(3)
    ds, ens = _setup(cfg)
    ens, history = train(cfg, ds, ens, tmp_path, **QUIET)
    loaded, loaded_history = load_trained(tmp_path, cfg, CPU, F64)
    assert isinstance(loaded, EnsembleMLP) and loaded.B == 3
    _assert_same_state(loaded.state_dict(), ens.state_dict())
    assert loaded_history == history
    ckpt_net, _, epoch = load_latest_checkpoint(tmp_path, cfg, CPU, F64)
    assert isinstance(ckpt_net, EnsembleMLP) and epoch == 3


def test_an_ensemble_checkpoint_does_not_load_as_another_B(tmp_path):
    cfg = _train_cfg(3)
    ds, ens = _setup(cfg)
    train(cfg, ds, ens, tmp_path, **QUIET)
    # Another B is another hash, so the file is stale for it.
    with pytest.raises(Exception, match="hash|stale"):
        load_trained(tmp_path, _with_B(cfg, 2), CPU, F64)


def test_train_rejects_a_net_of_another_ensemble_size(tmp_path):
    cfg = _train_cfg(3)
    ds, _ = _setup(cfg)
    with pytest.raises(ValueError, match="B"):
        train(cfg, ds, make_net(_net_cfg(2), D_IN, 2, CPU, F64), tmp_path, **QUIET)


# ------------------------------------------------------------------ member selection


def _trained_ensemble(tmp_path, B=3):
    cfg = _train_cfg(B, monitor_int=3)
    ds, ens = _setup(cfg)
    return train(cfg, ds, ens, tmp_path, monitor=usum_monitor, **QUIET)


def test_select_member_gives_one_network_and_its_history(tmp_path):
    ens, history = _trained_ensemble(tmp_path)
    net, h = select_member(ens, history, 2)
    _assert_same_state(net.state_dict(), ens.member(2).state_dict())
    assert h["epoch"] == history["epoch"] and h[MONITOR_EPOCH] == history[MONITOR_EPOCH]
    assert h["mse"] == [entry[2] for entry in history["mse"]]
    assert h["monitor_usum"] == [call[2] for call in history["monitor_usum"]]
    assert member_history(history, 2) == h


def test_select_member_refuses_an_ensemble_without_a_member(tmp_path):
    ens, history = _trained_ensemble(tmp_path)
    with pytest.raises(ValueError, match="ensemble of B=3"):
        select_member(ens, history)


def test_select_member_passes_a_single_network_through(tmp_path):
    cfg = _train_cfg(1)
    ds, net = _setup(cfg)
    net, history = train(cfg, ds, net, tmp_path, **QUIET)
    assert select_member(net, history) == (net, history)
    assert select_member(net, history, 0) == (net, history)
    with pytest.raises(IndexError, match="member"):
        select_member(net, history, 1)
