"""Growing an experiment's ensembles in place: `fct exp extend ROOT --by K`, then `exp run`
trains the K missing members of each setting in a staging directory and appends them to
train/final.pt (tiny experiments, CPU). b0 is the member count before, B after, K = B - b0."""

import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

import fct.train
from fct import phase_view, report
from fct.cache import StaleCacheError, cache_paths, latest_checkpoint, save_artefact
from fct.cli import main
from fct.config import NetConfig, load_config
from fct.exp import (
    ExpConfig,
    ExpError,
    expand,
    extend_experiment,
    load_exp,
    preset,
    run_setting,
    save_exp,
    setting_config,
    setting_dir,
    setting_members,
    setting_status,
    with_axes,
)
from fct.model import make_net
from fct.sgld import SGLD
from fct.train import (
    MONITOR_EPOCH,
    MONITOR_PREFIX,
    SHARED_HISTORY_KEYS,
    append_members,
    extension_seed,
    load_trained,
    member_history,
    staging_dir,
    trained_member_count,
)

PROJECT = Path(__file__).resolve().parents[1]
QUIET = {"log": lambda s: None}
# SGLD runs here take beta = 1e3, where the noise dominates the drift: at tiny's lr = 0.01 a
# step's noise has variance 2 lr / beta = 2e-5 per parameter, while its drift lr * grad has a
# squared norm of ~2e-6 summed over all parameters (measured).
SGLD_BETA = 1e3


class Interrupt(Exception):
    pass


@contextmanager
def interrupted_after_checkpoint(monkeypatch, epoch: int) -> Iterator[None]:
    """Raise `Interrupt` right after the training checkpoint of `epoch` is written, as a job
    killed then would; the checkpoint is on disk, the step of `epoch` not taken."""
    write = fct.train.write_checkpoint

    def write_then_stop(ckpt_dir, prefix, step, payload, config_hash, **kw):
        path = write(ckpt_dir, prefix, step, payload, config_hash, **kw)
        if step == epoch:
            raise Interrupt
        return path

    with monkeypatch.context() as patch:
        patch.setattr(fct.train, "write_checkpoint", write_then_stop)
        with pytest.raises(Interrupt):
            yield


def _exp(B: int, optimizer: str = "adam", **fields) -> ExpConfig:
    """Two tiny settings (net.L in {1, 2}) of ensembles of B members."""
    tree = {**preset("tiny").model_dump(), "B": B, "optimizer": optimizer, **fields}
    if optimizer == "sgld":
        tree["beta"] = SGLD_BETA
    return ExpConfig.model_validate({**tree, "axes": {"net.L": [1, 2]}, "description": None})


def _train_dir(root: Path, index: int = 0) -> Path:
    return cache_paths(setting_dir(root, expand(load_exp(root))[index])).train


def _final(root: Path, index: int = 0) -> dict:
    return torch.load(_train_dir(root, index) / "final.pt", weights_only=True)


def _status(root: Path, index: int = 0) -> str:
    exp = load_exp(root)
    return setting_status(root, exp, expand(exp)[index])


def _run(root: Path, exp: ExpConfig, index: int = 0, **kw) -> None:
    save_exp(exp, root)
    run_setting(exp, index, root, **(kw or QUIET))


def _run_extended(root: Path, B: int, by: int, optimizer: str = "adam", **fields) -> dict:
    """Train setting 0 with B members, extend the experiment by `by`, run it again; the
    payload of final.pt before the extension."""
    _run(root, _exp(B, optimizer, **fields))
    old = _final(root)
    extend_experiment(root, by)
    run_setting(load_exp(root), 0, root, **QUIET)
    return old


def _scratch(root: Path, B: int, optimizer: str = "adam", **fields) -> dict:
    _run(root, _exp(B, optimizer, **fields))
    return _final(root)


def _member(payload: dict, b: int) -> tuple[dict, dict]:
    """Member b's state dict and history; a single network's are its own."""
    state, history = payload["net_state_dict"], payload["history"]
    if state["U"].ndim == 2:
        assert b == 0
        return state, history
    return {k: v[b] for k, v in state.items()}, member_history(history, b)


def _same_state(x: dict, y: dict) -> bool:
    return x.keys() == y.keys() and all(torch.equal(v, y[k]) for k, v in x.items())


def _same_tree(x, y) -> bool:
    """Equality of histories whose entries may hold tensors (the spectrum monitor's)."""
    if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
        return isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and torch.equal(x, y)
    if isinstance(x, dict) and isinstance(y, dict):
        return x.keys() == y.keys() and all(_same_tree(x[k], y[k]) for k in x)
    if isinstance(x, list | tuple) and isinstance(y, list | tuple):
        return len(x) == len(y) and all(_same_tree(u, v) for u, v in zip(x, y, strict=True))
    return x == y


def _close_state(x: dict, y: dict) -> None:
    assert x.keys() == y.keys()
    for key, value in x.items():
        torch.testing.assert_close(value, y[key], rtol=1e-5, atol=1e-6)


def _per_member_keys(payload: dict) -> list[str]:
    return [k for k in payload["history"] if k not in SHARED_HISTORY_KEYS]


# ---------------------------------------------------------------------------- Adam: 2 -> 5
@pytest.fixture(scope="module")
def adam(tmp_path_factory) -> dict:
    """Setting 0 trained with B = 2 under Adam, extended by 3; and a from-scratch B = 5."""
    root = tmp_path_factory.mktemp("adam") / "ext"
    lines: list[str] = []
    _run(root, _exp(2))
    old = _final(root)
    old_checkpoints = sorted(p.name for p in _train_dir(root).glob("ckpt_*.pt"))
    extend_experiment(root, 3)
    run_setting(load_exp(root), 0, root, log=lines.append)
    scratch = _scratch(tmp_path_factory.mktemp("adam") / "scratch", B=5)
    return {
        "root": root,
        "old": old,
        "new": _final(root),
        "scratch": scratch,
        "log": lines,
        "old_checkpoints": old_checkpoints,
    }


def test_the_extended_final_holds_B_members(adam):
    assert adam["new"]["net_state_dict"]["U"].shape[0] == 5
    assert trained_member_count(_train_dir(adam["root"]) / "final.pt") == 5


def test_the_old_members_are_unchanged_bit_for_bit(adam):
    for b in range(2):
        state, history = _member(adam["new"], b)
        old_state, old_history = _member(adam["old"], b)
        assert _same_state(state, old_state), b
        assert history == old_history, b


def test_under_adam_the_new_members_are_a_scratch_runs_bit_for_bit(adam):
    for b in range(2, 5):
        assert _same_state(_member(adam["new"], b)[0], _member(adam["scratch"], b)[0]), b


def test_under_adam_the_new_members_histories_are_a_scratch_runs(adam):
    for b in range(2, 5):
        assert _member(adam["new"], b)[1] == _member(adam["scratch"], b)[1], b


def test_every_per_member_history_entry_has_B_members(adam):
    for key in _per_member_keys(adam["new"]):
        assert all(len(entry) == 5 for entry in adam["new"]["history"][key]), key
    assert adam["new"]["history"]["epoch"] == adam["scratch"]["history"]["epoch"]


def test_the_monitor_records_have_a_member_axis_of_B(adam):
    history = adam["new"]["history"]
    monitored = [k for k in history if k.startswith(MONITOR_PREFIX) and k != MONITOR_EPOCH]
    assert monitored
    for key in monitored:
        assert all(len(entry) == 5 for entry in history[key]), key


def test_the_extended_final_carries_the_hash_of_a_scratch_run_of_B(adam):
    assert adam["new"]["config_hash"] == adam["scratch"]["config_hash"]


def test_the_extended_setting_is_done_and_its_old_checkpoints_and_staging_are_gone(adam):
    train = _train_dir(adam["root"])
    assert _status(adam["root"]) == "done"
    assert adam["old_checkpoints"] and latest_checkpoint(train) is None
    assert not list(train.glob("extend_*"))
    for name in adam["old_checkpoints"]:
        assert any("removed the checkpoint" in s and name in s for s in adam["log"]), name
    assert any("removed the staging directory" in s for s in adam["log"])


def test_config_json_follows_the_grown_B(adam):
    config = load_config(cache_paths(setting_dir(adam["root"], expand(_exp(5))[0])).config)
    assert config.net.B == 5


def test_the_extended_ensemble_loads_and_its_members_are_selectable(adam):
    exp = load_exp(adam["root"])
    cfg = setting_config(exp, expand(exp)[0])
    directory = setting_dir(adam["root"], expand(exp)[0])
    net, history = load_trained(directory, cfg, torch.device("cpu"), torch.float32)
    assert net.B == 5 and len(history["loss"][0]) == 5


def test_under_adam_a_block_of_one_equals_a_scratch_member_to_round_off(tmp_path):
    """K = 1 trains as a single `MLP`, whose forward pass is not the batched one."""
    _run_extended(tmp_path / "ext", B=2, by=1)
    scratch = _scratch(tmp_path / "scratch", B=3)
    _close_state(_member(_final(tmp_path / "ext"), 2)[0], _member(scratch, 2)[0])


def test_under_adam_the_prior_draw_through_the_penalty_extends_bit_for_bit(tmp_path):
    """net.init = "prior" under the spatial penalty correlates the draw (`_correlate`)."""
    base = preset("spatial_tiny")
    tree = {**base.model_dump(), "B": 2, "optimizer": "adam", "init": "prior"}
    exp = with_axes(ExpConfig.model_validate(tree), {"net.N": [4]})
    assert setting_config(exp, expand(exp)[0]).spatial.parametrisation == "penalty"
    _run(tmp_path / "ext", exp)
    extend_experiment(tmp_path / "ext", 2)
    run_setting(load_exp(tmp_path / "ext"), 0, tmp_path / "ext", **QUIET)
    scratch = with_axes(ExpConfig.model_validate({**tree, "B": 4}), {"net.N": [4]})
    _run(tmp_path / "scratch", scratch)
    for b in range(2, 4):
        got, want = _final(tmp_path / "ext"), _final(tmp_path / "scratch")
        assert _same_state(_member(got, b)[0], _member(want, b)[0]), b


def test_a_whitened_ensemble_is_extended_and_reloads_with_its_factors(tmp_path):
    """Under the whitened parametrisation the factors are non-persistent buffers rebuilt from
    the config, so only the parameters (xi) are appended."""
    base = preset("spatial_tiny")
    spatial = base.spatial.model_copy(update={"parametrisation": "whitened"})
    fields = {"spatial": spatial.model_dump(), "init": "prior"}
    tree = {**base.model_dump(), "B": 2, "optimizer": "adam", **fields}
    _run(tmp_path / "ext", with_axes(ExpConfig.model_validate(tree), {"net.N": [4]}))
    extend_experiment(tmp_path / "ext", 2)
    run_setting(load_exp(tmp_path / "ext"), 0, tmp_path / "ext", **QUIET)
    scratch = with_axes(ExpConfig.model_validate({**tree, "B": 4}), {"net.N": [4]})
    _run(tmp_path / "scratch", scratch)

    got, want = _final(tmp_path / "ext"), _final(tmp_path / "scratch")
    for b in range(2, 4):
        assert _same_state(_member(got, b)[0], _member(want, b)[0]), b
    cfg = setting_config(scratch, expand(scratch)[0])
    directory = setting_dir(tmp_path / "ext", expand(scratch)[0])
    net, _ = load_trained(directory, cfg, torch.device("cpu"), torch.float32)
    assert net.B == 4 and net.whitening is not None


def test_the_spectrum_monitor_records_gain_the_member_axis(tmp_path):
    """Spectrum records are per-layer lists of each member, like the MIR's."""
    # The spectrum monitor needs latents on [0, a] (`fct.spectrum.latent_basis`).
    teacher = preset("tiny").linear_teacher.model_dump() | {"latent_min": 0.0}
    spectrum = {"monitor_kind": "spectrum", "linear_teacher": teacher}
    _run_extended(tmp_path / "ext", B=2, by=2, **spectrum)
    new = _final(tmp_path / "ext")
    scratch = _scratch(tmp_path / "scratch", B=4, **spectrum)
    history = new["history"]
    monitored = [k for k in history if k.startswith(MONITOR_PREFIX) and k != MONITOR_EPOCH]
    assert monitored and not any("mir" in k for k in monitored)
    for key in monitored:
        assert all(len(entry) == 4 for entry in history[key]), key
        assert all(len(member) == 1 for entry in history[key] for member in entry), key
    for b in range(2, 4):
        assert _same_tree(_member(new, b)[1], _member(scratch, b)[1]), b


# ---------------------------------------------------------------------------- SGLD
@pytest.fixture(scope="module")
def sgld(tmp_path_factory) -> dict:
    """Setting 0 trained with B = 2 under SGLD, extended by 3; and a from-scratch B = 5."""
    root = tmp_path_factory.mktemp("sgld") / "ext"
    old = _run_extended(root, B=2, by=3, optimizer="sgld")
    scratch = _scratch(tmp_path_factory.mktemp("sgld") / "scratch", B=5, optimizer="sgld")
    return {"root": root, "old": old, "new": _final(root), "scratch": scratch}


def test_under_sgld_the_old_members_are_unchanged_bit_for_bit(sgld):
    for b in range(2):
        state, history = _member(sgld["new"], b)
        assert _same_state(state, _member(sgld["old"], b)[0]), b
        assert history == _member(sgld["old"], b)[1], b


def test_under_sgld_everything_before_the_first_step_is_a_scratch_runs(sgld):
    """The default SGLD noise is one global stream over the whole ensemble, so a scratch run of
    B gives members b0 .. B-1 slices of draws of another shape: they are equal in distribution
    only. Everything recorded before the first noise draw is equal bit for bit."""
    for b in range(2, 5):
        history, scratch = _member(sgld["new"], b)[1], _member(sgld["scratch"], b)[1]
        for key in ("loss", "mse", "weight_l2", "rate_l2", "grad_norm"):
            assert history[key][0] == scratch[key][0], (b, key)
        assert history[f"{MONITOR_PREFIX}mir"][0] == scratch[f"{MONITOR_PREFIX}mir"][0], b


def test_under_sgld_the_new_members_are_another_noise_realisation(sgld):
    new_U, scratch_U = (
        sgld["new"]["net_state_dict"]["U"][2:],
        sgld["scratch"]["net_state_dict"]["U"][2:],
    )
    assert not torch.equal(new_U, scratch_U)


def test_under_sgld_the_new_members_step_with_the_noise_of_the_same_law(sgld):
    """param_msd ~ 2 lr / beta when the noise dominates the drift (SGLD_BETA). The mean over
    the 3 new members and 3 logged steps spreads by ~10% between seeds (measured), so 25%
    catches only gross errors: another lr or beta, or no noise at all."""
    new = torch.tensor(sgld["new"]["history"]["param_msd"])[:, 2:].mean()
    scratch = torch.tensor(sgld["scratch"]["history"]["param_msd"])[:, 2:].mean()
    assert new.item() == pytest.approx(scratch.item(), rel=0.25)


class _RecordingSGLD(SGLD):
    """SGLD that keeps every noise draw, in order, in `draws` (set by the test)."""

    draws: list[torch.Tensor]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, noise=self._record, **kwargs)

    def _record(self, p: torch.Tensor) -> torch.Tensor:
        z = torch.randn_like(p)
        _RecordingSGLD.draws.append(z.clone())
        return z


def _first_draws(monkeypatch, tmp_path: Path, replay: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """The first SGLD noise draw (of U) of a B = 2 run and of its extension by K = 2."""
    with monkeypatch.context() as patch:
        patch.setattr(fct.train, "SGLD", _RecordingSGLD)
        if replay:  # the extension reseeds with train.seed itself
            patch.setattr(fct.train, "extension_seed", lambda seed, b0: seed)
        _RecordingSGLD.draws = []
        _run(tmp_path, _exp(2, "sgld"))
        old = _RecordingSGLD.draws[0]
        extend_experiment(tmp_path, 2)
        _RecordingSGLD.draws = []
        run_setting(load_exp(tmp_path), 0, tmp_path, **QUIET)
        return old, _RecordingSGLD.draws[0]


def test_the_new_members_never_replay_the_old_members_noise(tmp_path, monkeypatch):
    """With K = b0 the extension's first draw has the old run's shape: reseeding with
    train.seed would hand the new members exactly the old members' first noise (the replay arm
    shows this test would catch it)."""
    old, new = _first_draws(monkeypatch, tmp_path / "own", replay=False)
    assert old.shape == new.shape and not torch.equal(old, new)
    old, new = _first_draws(monkeypatch, tmp_path / "replay", replay=True)
    assert torch.equal(old, new)


def test_the_extension_noise_stream_is_its_own():
    seeds = {extension_seed(0, b0) for b0 in range(1, 20)}
    assert len(seeds) == 19 and 0 not in seeds
    assert extension_seed(0, 2) != extension_seed(1, 2)
    with pytest.raises(ValueError, match="b0"):
        extension_seed(0, 0)


@pytest.mark.parametrize(("b0", "K"), [(2, 1), (1, 2)], ids=["K=1", "B=1 to 3"])
def test_under_sgld_a_single_network_on_either_side_extends(tmp_path, b0, K):
    """A block of one trains as an `MLP`; a single network gains its member axis on append."""
    old = _run_extended(tmp_path, B=b0, by=K, optimizer="sgld")
    new = _final(tmp_path)
    assert new["net_state_dict"]["U"].shape[0] == b0 + K
    for b in range(b0):
        assert _same_state(_member(new, b)[0], _member(old, b)[0]), b
    for key in _per_member_keys(new):
        assert all(len(entry) == b0 + K for entry in new["history"][key]), key
    assert _status(tmp_path) == "done"


def test_an_interrupted_extension_resumes_from_its_staging_checkpoint(tmp_path, monkeypatch):
    whole = tmp_path / "whole"
    _run_extended(whole, B=2, by=3, optimizer="sgld")
    root = tmp_path / "broken"
    _run(root, _exp(2, "sgld"))
    extend_experiment(root, 3)
    with interrupted_after_checkpoint(monkeypatch, epoch=4):
        run_setting(load_exp(root), 0, root, **QUIET)
    staging = staging_dir(_train_dir(root), 2, 5)
    assert list(staging.glob("ckpt_*.pt")) and _status(root) == "partial"
    assert trained_member_count(_train_dir(root) / "final.pt") == 2
    lines: list[str] = []
    run_setting(load_exp(root), 0, root, log=lines.append)
    assert any(line.startswith("resumed from") and "extend_2-5" in line for line in lines)
    resumed, uninterrupted = _final(root), _final(whole)
    assert _same_state(resumed["net_state_dict"], uninterrupted["net_state_dict"])
    assert resumed["history"] == uninterrupted["history"]
    assert not staging.exists()


def test_a_single_network_grows_into_an_ensemble_one_member_at_a_time(tmp_path):
    """B = 1 has no member axis on disk; a block of K = 1 trains a single network too. Both
    gain one on append."""
    old = _run_extended(tmp_path, B=1, by=1)
    assert old["net_state_dict"]["U"].ndim == 2
    two = _final(tmp_path)
    assert torch.equal(two["net_state_dict"]["U"][0], old["net_state_dict"]["U"])
    assert two["history"]["loss"][0] == [old["history"]["loss"][0], two["history"]["loss"][0][1]]
    extend_experiment(tmp_path, 1)
    run_setting(load_exp(tmp_path), 0, tmp_path, **QUIET)
    three = _final(tmp_path)
    assert three["net_state_dict"]["U"].shape[0] == 3
    assert torch.equal(three["net_state_dict"]["U"][:2], two["net_state_dict"]["U"])
    assert all(len(entry) == 3 for entry in three["history"][f"{MONITOR_PREFIX}mir"])
    assert _status(tmp_path) == "done"


# ---------------------------------------------------------------------------- the pure append
def _payload(B: int, epochs=(0, 2), value: float = 0.0) -> dict:
    shape = (B, 3, 2) if B > 1 else (3, 2)
    entry = [value] * B if B > 1 else value
    return {
        "epoch": 4,
        "net_state_dict": {"U": torch.full(shape, value)},
        "optim_state_dict": {
            "state": {"0": {"step": torch.tensor(4.0), "exp_avg": torch.full(shape, value)}},
            "param_groups": [{"lr": 0.1, "params": [0]}],
        },
        "history": {"epoch": list(epochs), "loss": [entry for _ in epochs]},
        "rng_state": {"cpu": torch.zeros(3, dtype=torch.uint8)},
    }


def test_append_members_concatenates_every_per_member_entry():
    merged = append_members(_payload(2, value=1.0), _payload(3, value=2.0))
    U = merged["net_state_dict"]["U"]
    assert U.shape == (5, 3, 2) and U[:2].eq(1).all() and U[2:].eq(2).all()
    moments = merged["optim_state_dict"]["state"]["0"]
    assert moments["exp_avg"].shape == (5, 3, 2) and moments["step"] == 4.0
    assert merged["history"] == {"epoch": [0, 2], "loss": [[1.0, 1.0, 2.0, 2.0, 2.0]] * 2}


def test_append_members_adds_the_member_axis_of_a_single_network():
    merged = append_members(_payload(1, value=1.0), _payload(1, value=2.0))
    assert merged["net_state_dict"]["U"].shape == (2, 3, 2)
    assert merged["history"]["loss"] == [[1.0, 2.0], [1.0, 2.0]]


def _moment(shape):
    def change(payload):
        payload["optim_state_dict"]["state"]["0"]["exp_avg"] = torch.zeros(shape)

    return change


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda p: p.update(epoch=3), "epoch"),
        (lambda p: p["history"].update(epoch=[0, 4]), "epoch"),
        (lambda p: p["history"].pop("loss"), "history keys"),
        (lambda p: p["net_state_dict"].update(V=torch.zeros(3, 3, 2)), "state dict keys"),
        (lambda p: p["net_state_dict"].update(U=torch.zeros(3, 4, 2)), "U"),
        (lambda p: p.update(extra=1), "keys"),
        (_moment((3, 3, 3)), "exp_avg.*does not fit its parameter"),
        (lambda p: p["optim_state_dict"]["param_groups"][0].update(params=[0, 1]), "param"),
    ],
)
def test_append_members_refuses_payloads_that_do_not_fit(change, match):
    new = _payload(3)
    change(new)
    with pytest.raises(ValueError, match=match):
        append_members(_payload(2), new)


def test_append_members_needs_a_state_dict_entry_per_optimiser_parameter():
    old, new = _payload(2), _payload(3)
    for payload in (old, new):
        payload["optim_state_dict"]["param_groups"][0]["params"] = [0, 1]
    with pytest.raises(ValueError, match="exactly the parameters"):
        append_members(old, new)


# ---------------------------------------------------------------------------- building members
def test_make_net_builds_a_block_of_members_exactly_as_in_the_whole_ensemble():
    cfg = NetConfig(N=5, L=2, B=5, seed=3)
    whole = make_net(cfg, 4, 2, torch.device("cpu"), torch.float64)
    block = make_net(cfg, 4, 2, torch.device("cpu"), torch.float64, members=range(2, 5))
    single = make_net(cfg, 4, 2, torch.device("cpu"), torch.float64, members=range(4, 5))
    assert block.B == 3 and single.B == 1
    for name, p in block.named_parameters():
        assert torch.equal(p, dict(whole.named_parameters())[name][2:5]), name
    assert torch.equal(single.U, whole.U[4])


@pytest.mark.parametrize("members", [range(3, 3), range(6), range(0, 4, 2), range(-1, 2)])
def test_make_net_refuses_members_outside_the_ensemble(members):
    with pytest.raises(ValueError, match="members"):
        make_net(NetConfig(N=5, B=5), 4, 2, torch.device("cpu"), torch.float64, members=members)


# ---------------------------------------------------------------------------- status and refusals
def test_status_and_members_through_an_extension(tmp_path):
    save_exp(_exp(2), tmp_path)
    exp = load_exp(tmp_path)
    assert setting_members(tmp_path, exp, expand(exp)[0]) is None
    run_setting(exp, 0, tmp_path, **QUIET)
    assert setting_members(tmp_path, exp, expand(exp)[0]) == 2
    extended = extend_experiment(tmp_path, 3)
    assert extended.B == 5 and load_exp(tmp_path) == extended
    statuses = [setting_status(tmp_path, extended, s) for s in expand(extended)]
    assert statuses == ["partial", "missing"]
    assert setting_members(tmp_path, extended, expand(extended)[0]) == 2


def test_a_final_with_the_current_hash_but_the_wrong_member_count_is_stale(tmp_path):
    exp = _exp(5)
    setting = expand(exp)[0]
    cfg = setting_config(exp, setting)
    final = cache_paths(setting_dir(tmp_path, setting)).train / "final.pt"
    payload = {"net_state_dict": {"U": torch.zeros(3, 8, 8)}}
    save_artefact(final, payload, cfg.section_hash("data", "net", "train"))
    assert setting_status(tmp_path, exp, setting) == "stale"


def test_extending_a_partial_setting_is_refused_and_leaves_exp_json(tmp_path, monkeypatch):
    save_exp(_exp(2), tmp_path)
    with interrupted_after_checkpoint(monkeypatch, epoch=4):
        run_setting(load_exp(tmp_path), 1, tmp_path, **QUIET)
    before = (tmp_path / "exp.json").read_text()
    with pytest.raises(ExpError, match=r"partial.*\[1\]"):
        extend_experiment(tmp_path, 3)
    assert (tmp_path / "exp.json").read_text() == before


def test_extending_twice_before_running_is_refused(tmp_path):
    _run(tmp_path, _exp(2))
    extend_experiment(tmp_path, 1)
    with pytest.raises(ExpError, match=r"partial.*\[0\]"):
        extend_experiment(tmp_path, 1)


@pytest.mark.parametrize("by", [0, -1])
def test_extending_by_less_than_one_is_refused(tmp_path, by):
    save_exp(_exp(2), tmp_path)
    with pytest.raises(ExpError, match="at least 1"):
        extend_experiment(tmp_path, by)


@pytest.mark.parametrize("values", [[2, 3], [8]], ids=["two values", "one value"])
def test_an_experiment_with_a_B_axis_cannot_be_extended(tmp_path, values):
    """Even one value (as in the phase_transition preset): it names the directories."""
    save_exp(with_axes(preset("tiny"), {"net.B": values}), tmp_path)
    before = (tmp_path / "exp.json").read_text()
    with pytest.raises(ExpError, match="net.B is an axis.*directory name"):
        extend_experiment(tmp_path, 1)
    assert (tmp_path / "exp.json").read_text() == before


def test_a_B_smaller_than_the_cached_member_count_fails_loudly(tmp_path):
    _run(tmp_path, _exp(3))
    save_exp(_exp(2), tmp_path)
    exp = load_exp(tmp_path)
    assert setting_status(tmp_path, exp, expand(exp)[0]) == "stale"
    with pytest.raises(ExpError, match="3 members but exp.json gives net.B=2"):
        run_setting(exp, 0, tmp_path, **QUIET)
    assert trained_member_count(_train_dir(tmp_path) / "final.pt") == 3


def test_an_interrupted_extension_can_be_backed_out(tmp_path, monkeypatch):
    """Setting B back to b0 after an extension crashed: config.json follows exp.json, the
    setting is done again and the abandoned staging directory is removed (and logged)."""
    _run(tmp_path, _exp(2))
    before = _final(tmp_path)
    extend_experiment(tmp_path, 3)
    with interrupted_after_checkpoint(monkeypatch, epoch=2):
        run_setting(load_exp(tmp_path), 0, tmp_path, **QUIET)
    save_exp(_exp(2), tmp_path)
    assert _status(tmp_path) == "done"
    lines: list[str] = []
    run_setting(load_exp(tmp_path), 0, tmp_path, log=lines.append)
    assert any("already done" in line for line in lines)
    assert any("removed the staging directory" in line for line in lines)
    assert not list(_train_dir(tmp_path).glob("extend_*"))
    config = load_config(cache_paths(setting_dir(tmp_path, expand(_exp(2))[0])).config)
    assert config.net.B == 2 and _same_state(
        _final(tmp_path)["net_state_dict"], before["net_state_dict"]
    )


def test_a_leftover_staging_directory_of_a_done_setting_is_removed(tmp_path):
    _run_extended(tmp_path, B=2, by=1)
    leftover = staging_dir(_train_dir(tmp_path), 2, 3)
    leftover.mkdir()
    lines: list[str] = []
    run_setting(load_exp(tmp_path), 0, tmp_path, log=lines.append)
    assert not leftover.exists() and _status(tmp_path) == "done"
    assert any("removed the staging directory" in s and "extend_2-3" in s for s in lines)


# ---------------------------------------------------------------------------- readers
def test_the_report_shows_the_members_a_pending_extension_has(tmp_path):
    """The report reads one network: a B = 1 setting extended to 2, before the new member
    trains, shows final.pt's network, provisional, with a note."""
    _run(tmp_path, _exp(1))
    extend_experiment(tmp_path, 1)
    exp = load_exp(tmp_path)
    result = report.load_setting_result(tmp_path, exp, expand(exp)[0])
    assert result.status == "partial" and result.provisional
    assert result.epoch == exp.n_epochs and "pending" in result.note


def _phase_exp(B: int) -> ExpConfig:
    """phase_transition at tiny sizes with B as the shared field (not an axis), one setting."""
    base = preset("phase_transition")
    fields = {
        "N": 6, "P": 16, "n_epochs": 4, "log_int": 2, "save_int": 2, "monitor_int": 2,
        "monitor_bins": 4, "monitor_shuffles": 2, "device": "cpu", "B": B,
        "linear_teacher": preset("tiny").linear_teacher.model_dump() | {"n_features": 8},
    }  # fmt: skip
    exp = ExpConfig.model_validate({**base.model_dump(), **fields})
    return with_axes(exp, {"spatial.d": [2], "spatial.lambda_max": [0.5]})


def test_phase_view_shows_the_members_a_pending_extension_has(tmp_path):
    _run(tmp_path, _phase_exp(2))
    extend_experiment(tmp_path, 2)
    run = phase_view.load_phase_run(tmp_path, {"spatial.d": 2})
    assert run.setting.status == "partial"
    assert run.net.B == 2 and run.epoch == run.setting.cfg.train.n_epochs
    assert len(run.history["loss"][0]) == 2


# ---------------------------------------------------------------------------- B changes and crashes
def test_a_B_changed_under_a_partial_original_run_names_net_B_and_the_fix(tmp_path, monkeypatch):
    save_exp(_exp(2), tmp_path)
    with interrupted_after_checkpoint(monkeypatch, epoch=2):
        run_setting(load_exp(tmp_path), 0, tmp_path, **QUIET)
    save_exp(_exp(3), tmp_path)
    fix = r"net.B is 2, exp.json gives 3; .*set B back to 2 .*let the run finish.*exp extend"
    with pytest.raises(ExpError, match=fix):
        run_setting(load_exp(tmp_path), 0, tmp_path, **QUIET)
    config = load_config(cache_paths(setting_dir(tmp_path, expand(_exp(2))[0])).config)
    assert config.net.B == 2


def test_a_B_change_beside_an_unreadable_final_is_refused(tmp_path):
    _run(tmp_path, _exp(2))
    (_train_dir(tmp_path) / "final.pt").write_bytes(b"truncated")
    save_exp(_exp(3), tmp_path)
    with pytest.raises(ExpError, match="net.B is 2.*member count is unknown"):
        run_setting(load_exp(tmp_path), 0, tmp_path, **QUIET)


def test_B_and_another_net_field_changed_together_are_refused(tmp_path):
    _run(tmp_path, _exp(2))
    save_exp(_exp(3, g=1.5), tmp_path)
    with pytest.raises(ExpError, match="differs from the config exp.json builds"):
        run_setting(load_exp(tmp_path), 0, tmp_path, **QUIET)


def test_a_crash_after_the_append_leaves_stale_checkpoints_that_the_next_run_removes(
    tmp_path, monkeypatch
):
    _run(tmp_path, _exp(2))
    old_checkpoints = sorted(_train_dir(tmp_path).glob("ckpt_*.pt"))
    extend_experiment(tmp_path, 1)

    def crash(*args, **kwargs):
        raise Interrupt

    with monkeypatch.context() as patch:
        patch.setattr(fct.train, "remove_stale_checkpoints", crash)
        with pytest.raises(Interrupt):
            run_setting(load_exp(tmp_path), 0, tmp_path, **QUIET)
    assert old_checkpoints and all(p.exists() for p in old_checkpoints)
    assert _status(tmp_path) == "done"
    lines: list[str] = []
    run_setting(load_exp(tmp_path), 0, tmp_path, log=lines.append)
    assert not any(p.exists() for p in old_checkpoints)
    assert not list(_train_dir(tmp_path).glob("extend_*"))
    for ckpt in old_checkpoints:
        assert any("removed the checkpoint" in s and ckpt.name in s for s in lines), ckpt


def test_a_finished_runs_own_checkpoint_is_kept_by_the_done_path(tmp_path):
    _run(tmp_path, _exp(2))
    kept = sorted(_train_dir(tmp_path).glob("ckpt_*.pt"))
    run_setting(load_exp(tmp_path), 0, tmp_path, **QUIET)
    assert kept and all(p.exists() for p in kept)


@pytest.mark.parametrize("error", [StaleCacheError("replaced"), ValueError("truncated")])
def test_the_report_notes_a_pending_final_it_cannot_read(tmp_path, monkeypatch, error):
    _run(tmp_path, _exp(1))
    extend_experiment(tmp_path, 1)

    def unreadable(*args, **kwargs):
        raise error

    monkeypatch.setattr(report, "load_trained_members", unreadable)
    exp = load_exp(tmp_path)
    result = report.load_setting_result(tmp_path, exp, expand(exp)[0])
    assert result.status == "partial" and result.history is None
    assert "final.pt not shown" in result.note and str(error) in result.note


def test_a_pending_rows_origin_names_final_pt_not_a_checkpoint():
    assert "final.pt" in report._origin(4, 6, 6) and "checkpoint" not in report._origin(4, 6, 6)
    assert "ckpt" not in report._origin(4, 6, 6, short=True)
    assert "checkpoint at epoch 4 of 6" in report._origin(2, 4, 6)


def test_phase_view_notes_a_pending_final_it_cannot_read(tmp_path, monkeypatch):
    _run(tmp_path, _phase_exp(2))
    extend_experiment(tmp_path, 2)

    def unreadable(*args, **kwargs):
        raise StaleCacheError("replaced by the append")

    monkeypatch.setattr(phase_view, "load_trained_members", unreadable)
    run = phase_view.load_phase_run(tmp_path, {"spatial.d": 2})
    assert run.net is None and run.history is None and run.epoch == 0
    assert "replaced by the append" in run.note


# ---------------------------------------------------------------------------- the command line
def _cli(capsys, *args) -> tuple[int, str, str]:
    code = main(["exp", *map(str, args)])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_cli_extend_then_run_then_status(tmp_path, capsys):
    save_exp(_exp(2), tmp_path)
    assert _cli(capsys, "run", tmp_path, "--index", 0)[0] == 0
    _, status, _ = _cli(capsys, "status", tmp_path)
    assert status.splitlines()[0].split()[:3] == ["0", "done", "2/2"]
    assert status.splitlines()[1].split()[:3] == ["1", "missing", "-/2"]

    code, out, _ = _cli(capsys, "extend", tmp_path, "--by", 3)
    assert code == 0 and "B 2 -> 5" in out
    _, status, _ = _cli(capsys, "status", tmp_path)
    assert status.splitlines()[0].split()[:3] == ["0", "partial", "2/5"]

    assert _cli(capsys, "run", tmp_path, "--index", 0)[0] == 0
    _, status, _ = _cli(capsys, "status", tmp_path)
    assert status.splitlines()[0].split()[:3] == ["0", "done", "5/5"]
    assert "1 done, 0 partial, 1 missing, 0 stale" in status


def test_cli_extend_refusal_is_one_line(tmp_path, capsys):
    save_exp(_exp(2), tmp_path)
    code, _, err = _cli(capsys, "extend", tmp_path, "--by", 0)
    assert code == 1 and err.startswith("error:") and len(err.strip().splitlines()) == 1


def test_cli_run_with_a_shrunk_B_is_one_line(tmp_path, capsys):
    _run(tmp_path, _exp(3))
    save_exp(_exp(2), tmp_path)
    code, _, err = _cli(capsys, "run", tmp_path, "--index", 0)
    assert code == 1 and "3 members" in err and "Traceback" not in err


def test_the_slurm_array_task_trains_the_new_members_unchanged(tmp_path):
    """jobs/fct_exp.sh runs `exp run ROOT --index $SLURM_ARRAY_TASK_ID` as before; on an
    extended experiment that run is the extension."""
    _run(tmp_path, _exp(2))
    extend_experiment(tmp_path, 2)
    env = {**os.environ, "FCT_PYTHON": sys.executable, "FCT_PROJECT": str(PROJECT)}
    env.update(SLURM_ARRAY_TASK_ID="0")
    env.pop("FCT_EXP_ROOT", None)
    script = PROJECT / "jobs" / "fct_exp.sh"
    run = subprocess.run(
        ["bash", str(script), str(tmp_path)], env=env, capture_output=True, text=True, check=False
    )
    assert run.returncode == 0, run.stderr
    assert _status(tmp_path) == "done"
    assert trained_member_count(_train_dir(tmp_path) / "final.pt") == 4


# ---------------------------------------------------------------- the recorded base B
def test_the_first_extension_records_the_base_b_and_later_ones_keep_it(tmp_path):
    exp = preset("tiny").model_copy(update={"B": 3})
    save_exp(exp, tmp_path)
    assert load_exp(tmp_path).base_B is None
    assert extend_experiment(tmp_path, by=2).base_B == 3
    twice = extend_experiment(tmp_path, by=4)
    assert (twice.B, twice.base_B) == (9, 3)
    assert load_exp(tmp_path).base_B == 3  # written to exp.json


def test_an_unextended_exp_json_omits_base_b_and_keeps_its_text(tmp_path):
    """Files written before base_B existed load as unextended, and are written back unchanged."""
    save_exp(preset("tiny"), tmp_path)
    text = (tmp_path / "exp.json").read_text()
    assert "base_B" not in text
    loaded = load_exp(tmp_path)
    assert loaded.base_B is None
    save_exp(loaded, tmp_path)
    assert (tmp_path / "exp.json").read_text() == text


def test_a_base_b_must_be_below_b_and_not_beside_a_b_axis():
    with pytest.raises(ValueError, match="must be below"):
        ExpConfig.model_validate({**preset("tiny").model_dump(), "B": 4, "base_B": 4})
    exp = with_axes(preset("tiny"), {"net.B": [2, 3]})
    with pytest.raises(ValueError, match="base_B is recorded only by extending"):
        ExpConfig.model_validate({**exp.model_dump(), "B": 4, "base_B": 2})


def test_exp_init_refuses_a_base_b(tmp_path, capsys):
    assert main(["exp", "init", str(tmp_path), "--preset", "tiny", "--set", "base_B=2"]) != 0
    assert "recorded by `exp extend`" in capsys.readouterr().err
