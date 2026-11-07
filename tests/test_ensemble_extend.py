"""Growing an experiment's ensembles in place: `fct exp extend ROOT --by K`, then `exp run`
trains the K missing members of each setting in a staging directory and appends them to
train/final.pt (tiny experiments, CPU)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from fct.cache import cache_paths, save_artefact
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
from fct.train import (
    MONITOR_EPOCH,
    MONITOR_PREFIX,
    append_members,
    extension_seed,
    load_trained,
    staging_dir,
    trained_member_count,
)

PROJECT = Path(__file__).resolve().parents[1]
QUIET = {"log": lambda s: None}
SHARED = ("epoch", MONITOR_EPOCH)


class Interrupt(Exception):
    pass


def _exp(B: int, optimizer: str = "adam", **fields) -> ExpConfig:
    """Two tiny settings (net.L in {1, 2}) of ensembles of B members."""
    tree = {**preset("tiny").model_dump(), "B": B, "optimizer": optimizer, **fields}
    if optimizer == "sgld":
        tree["beta"] = 1e3
    return with_axes(ExpConfig.model_validate(tree), {"net.L": [1, 2]})


def _final(root: Path, exp: ExpConfig, index: int = 0) -> dict:
    train = cache_paths(setting_dir(root, expand(exp)[index])).train
    return torch.load(train / "final.pt", weights_only=True)


def _status(root: Path, index: int = 0) -> str:
    exp = load_exp(root)
    return setting_status(root, exp, expand(exp)[index])


def _run_extended(root: Path, B: int, by: int, optimizer: str = "adam", **fields) -> dict:
    """Train setting 0 with B members, extend the experiment by `by`, run it again; the
    old final.pt's payload."""
    save_exp(_exp(B, optimizer, **fields), root)
    run_setting(load_exp(root), 0, root, **QUIET)
    old = _final(root, load_exp(root))
    extend_experiment(root, by)
    run_setting(load_exp(root), 0, root, **QUIET)
    return old


def _scratch(root: Path, B: int, optimizer: str = "adam", **fields) -> dict:
    exp = _exp(B, optimizer, **fields)
    save_exp(exp, root)
    run_setting(exp, 0, root, **QUIET)
    return _final(root, exp)


def _members(payload: dict, members: range) -> tuple[dict, dict]:
    """The state dict and the per-member history of `members` of an ensemble payload."""
    state = {k: v[members.start : members.stop] for k, v in payload["net_state_dict"].items()}
    history = {
        k: v if k in SHARED else [entry[members.start : members.stop] for entry in v]
        for k, v in payload["history"].items()
    }
    return state, history


def _assert_same_state(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for key, value in a.items():
        assert torch.equal(value, b[key]), key


# ---------------------------------------------------------------------------- the append
def test_extending_from_2_to_5_appends_three_members_equal_to_a_scratch_run(tmp_path):
    """Adam draws no noise, so members 2-4 are a from-scratch B = 5 run's bit for bit."""
    old = _run_extended(tmp_path / "ext", B=2, by=3)
    new = _final(tmp_path / "ext", load_exp(tmp_path / "ext"))
    scratch = _scratch(tmp_path / "scratch", B=5)

    assert new["net_state_dict"]["U"].shape[0] == 5
    assert trained_member_count(_train_dir(tmp_path / "ext") / "final.pt") == 5
    old_state, old_history = _members(old, range(2))
    _assert_same_state(_members(new, range(2))[0], old_state)
    assert _members(new, range(2))[1] == old_history
    new_state, new_history = _members(new, range(2, 5))
    scratch_state, scratch_history = _members(scratch, range(2, 5))
    _assert_same_state(new_state, scratch_state)
    assert new_history == scratch_history

    for key, values in new["history"].items():
        if key not in SHARED:
            assert all(len(entry) == 5 for entry in values), key
    monitored = [k for k in new["history"] if k.startswith(MONITOR_PREFIX) and k != MONITOR_EPOCH]
    assert monitored and all(len(new["history"][k][0]) == 5 for k in monitored)
    assert new["history"]["epoch"] == scratch["history"]["epoch"]
    assert new["config_hash"] == scratch["config_hash"]
    assert _status(tmp_path / "ext") == "done"
    assert not list(_train_dir(tmp_path / "ext").glob("extend_*"))
    config = load_config(cache_paths(setting_dir(tmp_path / "ext", expand(_exp(5))[0])).config)
    assert config.net.B == 5


def _train_dir(root: Path, index: int = 0) -> Path:
    return cache_paths(setting_dir(root, expand(load_exp(root))[index])).train


def test_the_extended_ensemble_loads_and_its_members_are_selectable(tmp_path):
    _run_extended(tmp_path, B=2, by=3)
    exp = load_exp(tmp_path)
    cfg = setting_config(exp, expand(exp)[0])
    directory = setting_dir(tmp_path, expand(exp)[0])
    net, history = load_trained(directory, cfg, torch.device("cpu"), torch.float32)
    assert net.B == 5 and len(history["loss"][0]) == 5


def test_under_sgld_the_new_members_are_equal_in_distribution_not_bit_for_bit(tmp_path):
    """The default SGLD noise is one global stream over the whole ensemble, so a from-scratch
    B = 5 run gives members 2-4 slices of draws of another shape. The new members start from
    the same weights (everything recorded before the first step agrees bit for bit) and draw
    their noise from a stream of their own, so they are other samples of the same law."""
    old = _run_extended(tmp_path / "ext", B=2, by=3, optimizer="sgld")
    new = _final(tmp_path / "ext", load_exp(tmp_path / "ext"))
    scratch = _scratch(tmp_path / "scratch", B=5, optimizer="sgld")

    _assert_same_state(_members(new, range(2))[0], _members(old, range(2))[0])
    assert _members(new, range(2))[1] == _members(old, range(2))[1]
    _, new_history = _members(new, range(2, 5))
    _, scratch_history = _members(scratch, range(2, 5))
    for key in ("loss", "mse", "weight_l2", "rate_l2", "grad_norm"):
        assert new_history[key][0] == scratch_history[key][0], key  # before any noise
    assert new_history[f"{MONITOR_PREFIX}mir"][0] == scratch_history[f"{MONITOR_PREFIX}mir"][0]
    new_U, scratch_U = new["net_state_dict"]["U"][2:], scratch["net_state_dict"]["U"][2:]
    assert not torch.equal(new_U, scratch_U)  # another noise realisation
    # The step size the noise sets is the same law's: param_msd ~ 2 lr / beta when the noise
    # dominates, averaged over the three new members and every logged step.
    msd_new = torch.tensor(new_history["param_msd"]).mean()
    msd_scratch = torch.tensor(scratch_history["param_msd"]).mean()
    assert msd_new == pytest.approx(msd_scratch.item(), rel=0.25)


def test_the_extension_noise_stream_is_its_own():
    """Seeded apart from the original run's `train.seed`, so the new members never replay the
    noise the old ones drew, and apart from every other block of members."""
    seeds = {extension_seed(0, b0) for b0 in range(1, 20)}
    assert len(seeds) == 19 and 0 not in seeds
    assert extension_seed(0, 2) != extension_seed(1, 2)
    with pytest.raises(ValueError, match="b0"):
        extension_seed(0, 0)


def test_an_interrupted_extension_resumes_from_its_staging_checkpoint(tmp_path):
    whole = tmp_path / "whole"
    _run_extended(whole, B=2, by=3, optimizer="sgld")
    root = tmp_path / "broken"
    save_exp(_exp(2, "sgld"), root)
    run_setting(load_exp(root), 0, root, **QUIET)
    extend_experiment(root, 3)

    def stop_at_epoch_4(line: str) -> None:
        if line.startswith("[00004/"):
            raise Interrupt

    with pytest.raises(Interrupt):
        run_setting(load_exp(root), 0, root, log=stop_at_epoch_4)
    staging = staging_dir(_train_dir(root), 2, 5)
    assert list(staging.glob("ckpt_*.pt")) and _status(root) == "partial"
    assert trained_member_count(_train_dir(root) / "final.pt") == 2
    lines: list[str] = []
    run_setting(load_exp(root), 0, root, log=lines.append)
    assert any(line.startswith("resumed from") and "extend_2-5" in line for line in lines)
    resumed, uninterrupted = _final(root, load_exp(root)), _final(whole, load_exp(whole))
    _assert_same_state(resumed["net_state_dict"], uninterrupted["net_state_dict"])
    assert resumed["history"] == uninterrupted["history"]
    assert not staging.exists()


def test_a_single_network_grows_into_an_ensemble_one_member_at_a_time(tmp_path):
    """B = 1 has no member axis on disk; a block of K = 1 trains a single network too. Both
    gain one on append."""
    old = _run_extended(tmp_path, B=1, by=1)
    assert old["net_state_dict"]["U"].ndim == 2
    two = _final(tmp_path, load_exp(tmp_path))
    assert torch.equal(two["net_state_dict"]["U"][0], old["net_state_dict"]["U"])
    assert two["history"]["loss"][0] == [old["history"]["loss"][0], two["history"]["loss"][0][1]]
    extend_experiment(tmp_path, 1)
    run_setting(load_exp(tmp_path), 0, tmp_path, **QUIET)
    three = _final(tmp_path, load_exp(tmp_path))
    assert three["net_state_dict"]["U"].shape[0] == 3
    assert torch.equal(three["net_state_dict"]["U"][:2], two["net_state_dict"]["U"])
    assert all(len(entry) == 3 for entry in three["history"][f"{MONITOR_PREFIX}mir"])
    assert _status(tmp_path) == "done"


def test_a_whitened_ensemble_is_extended_and_reloads_with_its_factors(tmp_path):
    """Under the whitened parametrisation the factors are non-persistent buffers rebuilt from
    the config, so only the parameters (xi) are appended."""
    base = preset("spatial_tiny")
    spatial = base.spatial.model_copy(update={"parametrisation": "whitened"})
    fields = {"spatial": spatial.model_dump(), "init": "prior"}
    tree = {**base.model_dump(), "B": 2, "optimizer": "adam", **fields}
    exp = with_axes(ExpConfig.model_validate(tree), {"net.N": [4]})
    save_exp(exp, tmp_path / "ext")
    run_setting(exp, 0, tmp_path / "ext", **QUIET)
    extend_experiment(tmp_path / "ext", 2)
    run_setting(load_exp(tmp_path / "ext"), 0, tmp_path / "ext", **QUIET)
    scratch = with_axes(ExpConfig.model_validate({**tree, "B": 4}), {"net.N": [4]})
    save_exp(scratch, tmp_path / "scratch")
    run_setting(scratch, 0, tmp_path / "scratch", **QUIET)

    got, want = _final(tmp_path / "ext", scratch), _final(tmp_path / "scratch", scratch)
    _assert_same_state(_members(got, range(2, 4))[0], _members(want, range(2, 4))[0])
    cfg = setting_config(scratch, expand(scratch)[0])
    directory = setting_dir(tmp_path / "ext", expand(scratch)[0])
    net, _ = load_trained(directory, cfg, torch.device("cpu"), torch.float32)
    assert net.B == 4 and net.whitening is not None


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
    assert merged["net_state_dict"]["U"].shape == (5, 3, 2)
    U = merged["net_state_dict"]["U"]
    assert U[:2].eq(1).all() and U[2:].eq(2).all()
    moments = merged["optim_state_dict"]["state"]["0"]
    assert moments["exp_avg"].shape == (5, 3, 2) and moments["step"] == 4.0
    assert merged["history"] == {"epoch": [0, 2], "loss": [[1.0, 1.0, 2.0, 2.0, 2.0]] * 2}


def test_append_members_adds_the_member_axis_of_a_single_network():
    merged = append_members(_payload(1, value=1.0), _payload(1, value=2.0))
    assert merged["net_state_dict"]["U"].shape == (2, 3, 2)
    assert merged["history"]["loss"] == [[1.0, 2.0], [1.0, 2.0]]


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda p: p.update(epoch=3), "epoch"),
        (lambda p: p["history"].update(epoch=[0, 4]), "epoch"),
        (lambda p: p["history"].pop("loss"), "history keys"),
        (lambda p: p["net_state_dict"].update(V=torch.zeros(3, 3, 2)), "state dict keys"),
        (lambda p: p["net_state_dict"].update(U=torch.zeros(3, 4, 2)), "U"),
        (lambda p: p.update(extra=1), "keys"),
    ],
)
def test_append_members_refuses_payloads_that_do_not_fit(change, match):
    new = _payload(3)
    change(new)
    with pytest.raises(ValueError, match=match):
        append_members(_payload(2), new)


# ---------------------------------------------------------------------------- status and refusals
def test_status_and_members_through_an_extension(tmp_path):
    save_exp(_exp(2), tmp_path)
    exp = load_exp(tmp_path)
    assert setting_members(tmp_path, exp, expand(exp)[0]) is None
    run_setting(exp, 0, tmp_path, **QUIET)
    assert setting_members(tmp_path, exp, expand(exp)[0]) == 2
    extended = extend_experiment(tmp_path, 3)
    assert extended.B == 5 and load_exp(tmp_path) == extended
    assert [setting_status(tmp_path, extended, s) for s in expand(extended)] == [
        "partial",
        "missing",
    ]
    assert setting_members(tmp_path, extended, expand(extended)[0]) == 2


def test_a_final_with_the_current_hash_but_the_wrong_member_count_is_stale(tmp_path):
    exp = _exp(5)
    setting = expand(exp)[0]
    cfg = setting_config(exp, setting)
    final = cache_paths(setting_dir(tmp_path, setting)).train / "final.pt"
    payload = {"net_state_dict": {"U": torch.zeros(3, 8, 8)}}
    save_artefact(final, payload, cfg.section_hash("data", "net", "train"))
    assert setting_status(tmp_path, exp, setting) == "stale"


def test_extending_a_partial_setting_is_refused_and_leaves_exp_json(tmp_path):
    save_exp(_exp(2), tmp_path)

    def stop(line: str) -> None:
        if line.startswith("[00004/"):
            raise Interrupt

    with pytest.raises(Interrupt):
        run_setting(load_exp(tmp_path), 1, tmp_path, log=stop)
    before = (tmp_path / "exp.json").read_text()
    with pytest.raises(ExpError, match=r"partial.*\[1\]"):
        extend_experiment(tmp_path, 3)
    assert (tmp_path / "exp.json").read_text() == before


def test_extending_twice_before_running_is_refused(tmp_path):
    save_exp(_exp(2), tmp_path)
    run_setting(load_exp(tmp_path), 0, tmp_path, **QUIET)
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
    save_exp(_exp(3), tmp_path)
    run_setting(load_exp(tmp_path), 0, tmp_path, **QUIET)
    save_exp(_exp(2), tmp_path)
    exp = load_exp(tmp_path)
    assert setting_status(tmp_path, exp, expand(exp)[0]) == "stale"
    with pytest.raises(ExpError, match="B=2.*3 members|3 members.*B=2"):
        run_setting(exp, 0, tmp_path, **QUIET)
    assert trained_member_count(_train_dir(tmp_path) / "final.pt") == 3


def test_a_leftover_staging_directory_of_a_done_setting_is_removed(tmp_path):
    _run_extended(tmp_path, B=2, by=1)
    leftover = staging_dir(_train_dir(tmp_path), 2, 3)
    leftover.mkdir()
    run_setting(load_exp(tmp_path), 0, tmp_path, **QUIET)
    assert not leftover.exists() and _status(tmp_path) == "done"


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
    save_exp(_exp(3), tmp_path)
    run_setting(load_exp(tmp_path), 0, tmp_path, **QUIET)
    save_exp(_exp(2), tmp_path)
    code, _, err = _cli(capsys, "run", tmp_path, "--index", 0)
    assert code == 1 and "3 members" in err and "Traceback" not in err


def test_the_slurm_array_task_trains_the_new_members_unchanged(tmp_path):
    """jobs/fct_exp.sh runs `exp run ROOT --index $SLURM_ARRAY_TASK_ID` as before; on an
    extended experiment that run is the extension."""
    save_exp(_exp(2), tmp_path)
    run_setting(load_exp(tmp_path), 0, tmp_path, **QUIET)
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
