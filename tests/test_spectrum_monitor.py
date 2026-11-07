"""The spectrum monitor (`fct.monitor.make_spectrum_monitor`) and its wiring by
`train.monitor_kind`: what one record holds, the hash rule, and a monitored ensemble run.

The recovery test builds a net whose hidden activity is chosen exactly: with x = I_P and one
layer, H[p, i] = U[i, p] / sqrt(P), so setting U = sqrt(P) h^T makes the pre-activation h.
"""

import math

import pytest
import torch

from fct import spectrum as sp
from fct.cache import cache_paths
from fct.config import Config, DataConfig, NetConfig, TrainConfig
from fct.data import make_dataset
from fct.exp import (
    EXP_JSON_OMITTED_AT_DEFAULT,
    ExpConfig,
    exp_text,
    expand,
    load_exp,
    preset,
    run_setting,
    setting_config,
    setting_dir,
    with_axes,
)
from fct.lrtune import evaluate_setting
from fct.model import make_net
from fct.monitor import (
    SPECTRUM_MONITOR_KEYS,
    make_mir_monitor,
    make_monitor,
    make_spectrum_monitor,
)
from fct.report import load_results, load_setting_result
from fct.tasks import Dataset
from fct.train import MONITOR_EPOCH, MONITOR_PREFIX, train

CPU, F64 = torch.device("cpu"), torch.float64
QUIET = {"log": lambda s: None}
TRAIN = ("data", "net", "train")


def _teacher(P=64, K=3, latent_min=0.0, **kw) -> DataConfig:
    return DataConfig(
        task="linear_teacher", P=P, D_in=8, n_latents=K, latent_min=latent_min, seed=3, **kw
    )


def _cfg(data: DataConfig, N=8, L=2, B=1, monitor_int=1, **train_kw) -> Config:
    return Config(
        data=data,
        net=NetConfig(N=N, L=L, B=B, phi="tanh", seed=1),
        train=TrainConfig(monitor_kind="spectrum", monitor_int=monitor_int, **train_kw),
        dtype="float64",
        device="cpu",
    )


def _uniform_factors(P: int, K: int, seed: int = 0) -> dict[str, torch.Tensor]:
    z = torch.rand((P, K), dtype=F64, generator=torch.Generator().manual_seed(seed))
    return {f"z{k}": z[:, k] for k in range(K)}


def _cosine_map(N: int, q: int, amplitude: float) -> torch.Tensor:
    return amplitude * torch.cos(2 * math.pi * q * torch.arange(N, dtype=F64) / N)


def _cosine_power(N: int, q: int, amplitude: float) -> torch.Tensor:
    """Unitary-DFT power of amplitude * cos(2 pi q i / N), 0 < q < N/2: a^2 N / 4 at +-q."""
    d = torch.zeros(N, dtype=F64)
    d[q] = d[N - q] = amplitude**2 * N / 4
    return d


def _identity_input_net(h: torch.Tensor, factors: dict[str, torch.Tensor]):
    """(dataset, cfg, net) whose single hidden layer's pre-activation on dataset.x is h (P, N)."""
    P, N = h.shape
    dataset = Dataset(
        x=torch.eye(P, dtype=F64),
        y=torch.zeros(P, 1, dtype=F64),
        Q=None,
        period=None,
        factors=factors,
        groups={name: [name] for name in factors},
    )
    cfg = _cfg(_teacher(), N=N, L=1)  # the monitor reads the dataset, not cfg.data
    net = make_net(cfg.net, P, 1, CPU, F64)
    with torch.no_grad():
        net.U.copy_(math.sqrt(P) * h.T)
    return dataset, cfg, net


# ------------------------------------------------------------------ one record
def test_one_record_recovers_a_known_spectrum_and_null_floor():
    """h = sum_k u_k (x) t_k + w (x) c, t_k a cosine at q_k, w a unit vector orthogonal to
    col(Z): the latent rows are the cosines' power and the null row is |c~(q)|^2 / (P - K)."""
    P, K, N = 64, 3, 32
    factors = _uniform_factors(P, K)
    U = sp.latent_basis(factors).U
    modes = [(2, 3.0), (5, 1.5), (7, 0.5)]  # (q_k, amplitude_k)
    t = torch.stack([_cosine_map(N, q, a) for q, a in modes])  # (K, N)
    w = torch.linalg.qr(U, mode="complete").Q[:, K]  # a unit null direction
    c = _cosine_map(N, 11, 0.2) + 0.1  # a null map with power at q = 0 and +-11
    h = U @ t + torch.outer(w, c)
    dataset, cfg, net = _identity_input_net(h, factors)

    record = make_spectrum_monitor(dataset, cfg)(net)

    assert set(record) == set(SPECTRUM_MONITOR_KEYS)
    (spectrum,) = record["spectrum"]  # one layer
    assert spectrum.shape == (K + 1, N) and spectrum.dtype == torch.float32
    expected_null = _cosine_power(N, 11, 0.2)
    expected_null[0] = N * 0.1**2
    expected = torch.stack([*(_cosine_power(N, q, a) for q, a in modes), expected_null / (P - K)])
    # float32 storage keeps each power to relative 6e-8; the null row is ~5e-3, so atol 1e-8
    # still resolves it, and the zeros are float64 round-off of order 1e-14 x 72
    torch.testing.assert_close(spectrum.double(), expected, atol=1e-8, rtol=1e-6)


def test_angle_keys_match_direct_calls_and_are_stored_compactly():
    """hex is (Re, Im) of hexatic_moment and theta_hist the 36 counts, of rows 1 and 2 of t."""
    P, K, N = 64, 3, 48
    factors = _uniform_factors(P, K, seed=4)
    h = torch.randn((P, N), dtype=F64, generator=torch.Generator().manual_seed(5))
    dataset, cfg, net = _identity_input_net(h, factors)
    record = make_spectrum_monitor(dataset, cfg)(net)

    t = sp.tuning_maps(net(dataset.x)[0][:, 0].detach(), sp.latent_basis(factors).U)
    moment = sp.hexatic_moment(t[1], t[2])
    (hexatic,) = record["hex"]
    assert hexatic.shape == (2,) and hexatic.dtype == torch.float32
    torch.testing.assert_close(
        hexatic, torch.stack([moment.real, moment.imag]).float(), atol=1e-6, rtol=1e-6
    )
    (hist,) = record["theta_hist"]
    assert torch.equal(hist, sp.theta_histogram(t[1], t[2]))
    assert int(hist.sum()) == N


@pytest.mark.parametrize("K, keys", [(2, {"spectrum"}), (3, set(SPECTRUM_MONITOR_KEYS))])
def test_every_layer_is_recorded_and_the_angle_keys_need_three_latents(K, keys):
    cfg = _cfg(_teacher(K=K), L=3)
    dataset = make_dataset(cfg.data, CPU, F64)
    net = make_net(cfg.net, dataset.x.shape[1], dataset.y.shape[1], CPU, F64)
    record = make_spectrum_monitor(dataset, cfg)(net)

    assert set(record) == keys
    assert all(len(per_layer) == 3 for per_layer in record.values())
    H = net(dataset.x)[0].detach()  # (P, L, N) pre-activations, not the rates
    U = sp.latent_basis(dataset.factors).U
    direct = sp.mode_spectra(H.transpose(0, 1), U).float()  # (L, K + 1, N)
    torch.testing.assert_close(torch.stack(record["spectrum"]), direct)


def test_a_record_is_deterministic_and_leaves_the_rng_alone():
    cfg = _cfg(_teacher())
    dataset = make_dataset(cfg.data, CPU, F64)
    net = make_net(cfg.net, dataset.x.shape[1], dataset.y.shape[1], CPU, F64)
    monitor = make_spectrum_monitor(dataset, cfg)
    state = torch.get_rng_state()
    first, second = monitor(net), monitor(net)
    assert torch.equal(torch.get_rng_state(), state)
    assert all(
        torch.equal(a, b) for key in first for a, b in zip(first[key], second[key], strict=True)
    )
    assert all(p.grad is None for p in net.parameters())


# ------------------------------------------------------------------ failing at the start
def test_symmetric_latents_fail_when_the_monitor_is_built():
    cfg = _cfg(_teacher(latent_min=-1.0))
    dataset = make_dataset(cfg.data, CPU, F64)
    with pytest.raises(ValueError, match="symmetric.*mean direction is undefined"):
        make_spectrum_monitor(dataset, cfg)


def test_a_task_without_latents_fails_when_the_monitor_is_built():
    cfg = _cfg(DataConfig(task="what_where", P=32, D_in=81, n_motifs=4))
    dataset = make_dataset(cfg.data, CPU, F64)
    with pytest.raises(ValueError, match="not a latent"):
        make_spectrum_monitor(dataset, cfg)
    no_factors = Dataset(x=dataset.x, y=dataset.y, Q=None, period=None)
    with pytest.raises(ValueError, match="factors"):
        make_spectrum_monitor(no_factors, cfg)


def test_make_monitor_picks_the_monitor_by_kind():
    cfg = _cfg(_teacher())
    dataset = make_dataset(cfg.data, CPU, F64)
    net = make_net(cfg.net, dataset.x.shape[1], dataset.y.shape[1], CPU, F64)
    assert set(make_monitor(dataset, cfg)(net)) == set(SPECTRUM_MONITOR_KEYS)
    mir = cfg.model_copy(update={"train": cfg.train.model_copy(update={"monitor_kind": "mir"})})
    assert make_monitor(dataset, mir)(net) == make_mir_monitor(dataset, mir)(net)


# ------------------------------------------------------------------ the hash rule
def test_monitor_kind_enters_the_hash_only_as_spectrum():
    default = Config()
    explicit = Config(train=TrainConfig(monitor_kind="mir"))
    spectrum = Config(train=TrainConfig(monitor_kind="spectrum", monitor_int=1))
    assert explicit.section_hash(*TRAIN) == default.section_hash(*TRAIN)
    assert spectrum.section_hash(*TRAIN) != default.section_hash(*TRAIN)
    assert spectrum.section_hash("data", "net") == default.section_hash("data", "net")


def test_mir_hashes_are_those_written_before_monitor_kind_existed():
    """Literal hashes computed by the code just before `monitor_kind` was added: the default
    config, a MIR-monitored one whose bins and shuffles enter the hash, and a rate-free one.
    The first two have rate_lambda > 0 and so fold in LOSS_VERSION; the rate-free one folds in
    RATE_FREE_LOSS_VERSION and is also the hash from before LOSS_VERSION 5."""
    assert Config().section_hash(*TRAIN) == "629c7b68d953"
    monitored = Config(train=TrainConfig(monitor_int=5, monitor_bins=7))
    assert monitored.section_hash(*TRAIN) == "19d79cce4911"
    rate_free = Config(train=TrainConfig(rate_lambda=0.0))
    assert rate_free.section_hash(*TRAIN) == "bfb7ec92fda0"


def test_bins_and_shuffles_do_not_enter_a_spectrum_hash():
    """They configure the MIR monitor only, so they cannot change a spectrum run's history."""
    def hashed(**train_kw) -> str:
        return Config(train=TrainConfig(monitor_int=1, **train_kw)).section_hash(*TRAIN)

    other = {"monitor_bins": 3, "monitor_shuffles": 9}
    assert hashed(monitor_kind="spectrum", **other) == hashed(monitor_kind="spectrum")
    assert hashed(**other) != hashed()  # under the MIR monitor they do


def test_the_spectrum_monitor_needs_a_positive_monitor_int():
    with pytest.raises(ValueError, match="monitor_int is 0"):
        TrainConfig(monitor_kind="spectrum", monitor_int=0)


def test_exp_json_carries_monitor_kind_only_as_spectrum():
    tiny = preset("tiny")
    assert tiny.monitor_kind == "mir"
    assert EXP_JSON_OMITTED_AT_DEFAULT["monitor_kind"] == "mir"
    assert "monitor_kind" not in exp_text(tiny)
    spectrum = tiny.model_copy(update={"monitor_kind": "spectrum"})
    assert '"monitor_kind": "spectrum"' in exp_text(spectrum)
    assert {setting_config(spectrum, s).train.monitor_kind for s in expand(spectrum)} == {
        "spectrum"
    }


def test_monitor_kind_is_owned_by_the_experiment_not_an_axis():
    with pytest.raises(ValueError, match="monitor_kind"):
        with_axes(preset("tiny"), {"train.monitor_kind": ["mir", "spectrum"]})


# ------------------------------------------------------------------ monitored runs
def _spectrum_history(history: dict, key: str) -> torch.Tensor:
    """monitor_<key> of an ensemble, (records, members, layers, ...), stacked from the lists."""
    return torch.stack(
        [torch.stack([torch.stack(layers) for layers in members]) for members in history[key]]
    )


def test_an_ensemble_run_records_a_member_axis(tmp_path):
    B, L, N, K = 2, 2, 8, 3
    cfg = _cfg(_teacher(K=K), N=N, L=L, B=B, n_epochs=4, monitor_int=2, log_int=2, save_int=2)
    dataset = make_dataset(cfg.data, CPU, F64)
    net = make_net(cfg.net, dataset.x.shape[1], dataset.y.shape[1], CPU, F64)
    net, history = train(cfg, dataset, net, tmp_path, monitor=make_monitor(dataset, cfg), **QUIET)

    assert history[MONITOR_EPOCH] == [0, 2, 4]
    spectrum = _spectrum_history(history, f"{MONITOR_PREFIX}spectrum")
    assert spectrum.shape == (3, B, L, K + 1, N) and spectrum.dtype == torch.float32
    assert _spectrum_history(history, f"{MONITOR_PREFIX}hex").shape == (3, B, L, 2)
    assert _spectrum_history(history, f"{MONITOR_PREFIX}theta_hist").shape == (3, B, L, 36)
    # the last record is of the trained members, each seen as a single network
    member = make_spectrum_monitor(dataset, cfg)(net.member(1))
    torch.testing.assert_close(spectrum[-1, 1], torch.stack(member["spectrum"]))
    assert not torch.equal(spectrum[-1, 0], spectrum[-1, 1])  # the members differ


def test_run_setting_uses_the_spectrum_monitor_of_the_experiment(tmp_path):
    tiny = preset("tiny")
    exp = with_axes(
        tiny.model_copy(
            update={
                "monitor_kind": "spectrum",
                "B": 2,
                "linear_teacher": tiny.linear_teacher.model_copy(update={"latent_min": 0.0}),
            }
        ),
        {"net.L": [2]},
    )
    run_setting(exp, 0, tmp_path, **QUIET)
    directory = setting_dir(tmp_path, expand(exp)[0])
    history = torch.load(cache_paths(directory).train / "final.pt", weights_only=True)["history"]
    spectrum = _spectrum_history(history, f"{MONITOR_PREFIX}spectrum")
    n_records = len([*range(0, exp.n_epochs, exp.monitor_int)]) + 1
    K = exp.linear_teacher.n_latents
    assert spectrum.shape == (n_records, 2, 2, K + 1, exp.N)
    assert f"{MONITOR_PREFIX}mir" not in history


def _spectrum_tiny(latent_min: float = 0.0) -> ExpConfig:
    """The tiny preset under the spectrum monitor, unvalidated (model_copy), latents on
    [latent_min, 1]."""
    tiny = preset("tiny")
    return tiny.model_copy(
        update={
            "monitor_kind": "spectrum",
            "linear_teacher": tiny.linear_teacher.model_copy(update={"latent_min": latent_min}),
        }
    )


def test_an_experiment_with_symmetric_latents_is_rejected_when_built():
    """The rule is latent_basis's own, applied to each setting's dataset, so the error is
    raised at load, before any setting runs, and names the setting."""
    with pytest.raises(ValueError, match=r"setting 0 .*symmetric.*mean direction is undefined"):
        with_axes(_spectrum_tiny(latent_min=-1.0), {"net.L": [1]})
    with_axes(_spectrum_tiny(), {"net.L": [1]})  # latents on [0, 1] have a mean direction


@pytest.mark.parametrize(
    "axes", [{"data.task": ["linear_teacher", "what_where"]}, None]  # None: the built-in axes
)
def test_an_experiment_with_a_task_without_latents_is_rejected_when_built(axes):
    exp = _spectrum_tiny().model_copy(update={"axes": axes})
    with pytest.raises(ValueError, match=r"needs the latents of the linear_teacher task"):
        ExpConfig.model_validate(exp.model_dump())


def test_the_mir_readers_refuse_a_spectrum_experiment(tmp_path):
    """The report and lrtune read the MIR monitor's history; on a spectrum experiment they
    name that cause instead of reporting a missing MIR."""
    modularity_axes = {"net.L": [1], "net.phi": ["tanh"], "train.rate_lambda": [0.0]}
    exp = with_axes(_spectrum_tiny(), modularity_axes)
    (tmp_path / "exp.json").write_text(exp_text(exp))
    setting = expand(exp)[0]
    cause = "records spectra, not MIR"
    with pytest.raises(ValueError, match=cause):
        load_results(tmp_path)
    with pytest.raises(ValueError, match=cause):
        load_setting_result(tmp_path, exp, setting)
    with pytest.raises(ValueError, match=cause):
        evaluate_setting(tmp_path, exp, setting, held_out_seed=1000)


def test_a_spectrum_experiment_is_checked_when_loaded(tmp_path):
    (tmp_path / "exp.json").write_text(exp_text(_spectrum_tiny(latent_min=-1.0)))
    with pytest.raises(ValueError, match="symmetric"):
        load_exp(tmp_path)
