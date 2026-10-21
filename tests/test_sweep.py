"""The sweep grid, its on-disk layout, status, and running one setting end to end (tiny, CPU)."""

import json

import pytest
import torch
from pydantic import ValidationError

from fct.cache import cache_paths
from fct.cli import main
from fct.config import load_config
from fct.monitor import MIR_MONITOR_KEYS
from fct.sweep import (
    SWEEP_TASKS,
    SweepConfig,
    expand,
    get_setting,
    load_sweep,
    preset,
    run_setting,
    save_sweep,
    setting_dir,
    setting_status,
    sweep_path,
)
from fct.train import MONITOR_EPOCH, MONITOR_PREFIX

QUIET = {"log": lambda s: None}


@pytest.fixture
def tiny() -> SweepConfig:
    return preset("tiny")


def _index_of(sweep, task, L=1, phi="tanh", rate=1e-3) -> int:
    wanted = (task, L, phi, rate)
    settings = expand(sweep)
    return next(i for i, s in enumerate(settings) if (s.task, s.L, s.phi, s.rate_lambda) == wanted)


def _final(root, sweep, i):
    train_dir = cache_paths(setting_dir(root, expand(sweep)[i])).train
    return torch.load(train_dir / "final.pt", weights_only=True)


# ---------------------------------------------------------------------------- the grid
def test_expand_gives_36_distinct_settings_and_dirs(tiny, tmp_path):
    settings = expand(tiny)
    assert len(settings) == 36
    assert len(set(settings)) == 36
    assert len({setting_dir(tmp_path, s) for s in settings}) == 36
    assert {s.task for s in settings} == set(SWEEP_TASKS)


def test_expand_order_is_fixed_and_independent_of_the_preset(tiny):
    assert expand(tiny) == expand(preset("full")) == expand(tiny)
    first, last = expand(tiny)[0], expand(tiny)[-1]
    assert (first.task, first.L, first.phi, first.rate_lambda) == (SWEEP_TASKS[0], 1, "tanh", 0.0)
    assert (last.task, last.L, last.phi, last.rate_lambda) == (SWEEP_TASKS[-1], 5, "relu", 1e-3)


def test_setting_dir_layout(tiny, tmp_path):
    s = expand(tiny)[_index_of(tiny, "what_where", L=3, phi="relu", rate=1e-3)]
    assert setting_dir(tmp_path, s) == tmp_path / "what_where" / "L3_relu_r0.001"
    s0 = expand(tiny)[_index_of(tiny, "what_where", L=3, phi="relu", rate=0.0)]
    assert setting_dir(tmp_path, s0).name == "L3_relu_r0"


def test_get_setting_rejects_bad_indices(tiny):
    for bad in (-1, 36, 100):
        with pytest.raises(ValueError, match=r"index .* 0\.\.35"):
            get_setting(tiny, bad)
    assert get_setting(tiny, 35) == expand(tiny)[-1]


# ---------------------------------------------------------------------------- the config
def test_presets_differ_in_size_and_validate():
    tiny, full = preset("tiny"), preset("full")
    assert tiny.n_epochs < full.n_epochs and tiny.N < full.N and tiny.P < full.P
    assert tiny.dtype == full.dtype == "float32"
    assert tiny.device == "cpu"
    with pytest.raises(ValueError, match="unknown preset"):
        preset("huge")


def test_sweep_round_trips_through_json(tiny, tmp_path):
    save_sweep(tiny, tmp_path)
    assert sweep_path(tmp_path) == tmp_path / "sweep.json"
    assert load_sweep(tmp_path) == tiny


def test_sweep_config_is_strict(tiny):
    tree = tiny.model_dump()
    with pytest.raises(ValidationError):
        SweepConfig.model_validate({**tree, "typo": 1})
    with pytest.raises(ValidationError):
        SweepConfig.model_validate({**tree, "N": "8"})
    with pytest.raises(ValidationError):
        SweepConfig.model_validate({**tree, "n_epochs": -1})


def test_load_missing_sweep_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="sweep.json"):
        load_sweep(tmp_path)


def test_invalid_task_parameters_fail_at_load(tiny):
    tree = tiny.model_dump()
    tree["object_action"]["max_objects"] = 99  # more objects than grid cells
    with pytest.raises(ValidationError, match="max_objects"):
        SweepConfig.model_validate(tree)


# ---------------------------------------------------------------------------- status
def test_status_done_partial_missing(tiny, tmp_path):
    settings = expand(tiny)
    train = cache_paths(setting_dir(tmp_path, settings[0])).train
    train.mkdir(parents=True)
    (train / "final.pt").touch()
    train = cache_paths(setting_dir(tmp_path, settings[1])).train
    train.mkdir(parents=True)
    (train / "ckpt_000002.pt").touch()
    other = cache_paths(setting_dir(tmp_path, settings[2]))
    other.data.mkdir(parents=True)  # data alone is not training progress
    states = [setting_status(tmp_path, s) for s in settings[:4]]
    assert states == ["done", "partial", "missing", "missing"]


def test_done_wins_over_leftover_checkpoint(tiny, tmp_path):
    train = cache_paths(setting_dir(tmp_path, expand(tiny)[0])).train
    train.mkdir(parents=True)
    (train / "final.pt").touch()
    (train / "ckpt_000002.pt").touch()
    assert setting_status(tmp_path, expand(tiny)[0]) == "done"


# ---------------------------------------------------------------------------- running
@pytest.mark.parametrize("task", SWEEP_TASKS)
def test_run_setting_trains_and_records_monitor_per_task(tiny, tmp_path, task):
    i = _index_of(tiny, task, L=3, phi="relu", rate=1e-3)
    run_setting(tiny, i, tmp_path, **QUIET)
    setting = expand(tiny)[i]
    assert setting_status(tmp_path, setting) == "done"

    cfg = load_config(cache_paths(setting_dir(tmp_path, setting)).config)
    assert (cfg.data.task, cfg.net.L, cfg.net.phi, cfg.train.rate_lambda) == (task, 3, "relu", 1e-3)
    assert cfg.train.weight_lambda == 1e-3
    assert cfg.dtype == "float32"

    history = _final(tmp_path, tiny, i)["history"]
    n_calls = tiny.n_epochs // tiny.monitor_int + 1
    assert history[MONITOR_EPOCH][0] == 0 and history[MONITOR_EPOCH][-1] == tiny.n_epochs
    assert len(history[MONITOR_EPOCH]) == n_calls
    for key in MIR_MONITOR_KEYS:
        calls = history[f"{MONITOR_PREFIX}{key}"]
        assert len(calls) == n_calls and all(len(layers) == 3 for layers in calls)


def test_run_setting_sets_D_in_from_the_task(tiny, tmp_path):
    i = _index_of(tiny, "object_action")
    run_setting(tiny, i, tmp_path, **QUIET)
    cfg = load_config(cache_paths(setting_dir(tmp_path, expand(tiny)[i])).config)
    assert cfg.data.D_in == 2 * tiny.object_action.grid_size**2


def test_rerun_is_a_no_op(tiny, tmp_path):
    i = _index_of(tiny, "linear_teacher")
    run_setting(tiny, i, tmp_path, **QUIET)
    final = cache_paths(setting_dir(tmp_path, expand(tiny)[i])).train / "final.pt"
    before = (final.read_bytes(), final.stat().st_mtime_ns)
    lines: list[str] = []
    run_setting(tiny, i, tmp_path, log=lines.append)
    assert (final.read_bytes(), final.stat().st_mtime_ns) == before
    assert any("already done" in line for line in lines)


class Interrupt(Exception):
    pass


def test_interrupt_and_resume_gives_the_uninterrupted_history(tiny, tmp_path):
    i = _index_of(tiny, "what_where")
    run_setting(tiny, i, tmp_path / "straight", **QUIET)

    def stop_at_epoch_4(line: str) -> None:
        if line.startswith("[00004/"):
            raise Interrupt

    with pytest.raises(Interrupt):
        run_setting(tiny, i, tmp_path / "broken", log=stop_at_epoch_4)
    assert setting_status(tmp_path / "broken", expand(tiny)[i]) == "partial"
    run_setting(tiny, i, tmp_path / "broken", **QUIET)

    resumed = _final(tmp_path / "broken", tiny, i)["history"]
    assert resumed == _final(tmp_path / "straight", tiny, i)["history"]


def test_changed_sweep_does_not_overwrite_a_setting(tiny, tmp_path):
    i = _index_of(tiny, "linear_teacher")
    run_setting(tiny, i, tmp_path, **QUIET)
    changed = tiny.model_copy(update={"lr": tiny.lr * 2})
    with pytest.raises(ValueError, match="differs"):
        run_setting(changed, i, tmp_path, **QUIET)


# ---------------------------------------------------------------------------- CLI
def _cli(capsys, *args) -> tuple[int, str, str]:
    code = main(["sweep", *map(str, args)])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_cli_init_count_list_agree(tmp_path, capsys):
    assert _cli(capsys, "init", tmp_path, "--preset", "tiny")[0] == 0
    assert (tmp_path / "sweep.json").exists()
    _, count, _ = _cli(capsys, "count", tmp_path)
    _, listing, _ = _cli(capsys, "list", tmp_path)
    lines = listing.strip().splitlines()
    assert int(count) == 36 == len(lines)
    assert lines[0].split()[0] == "0"
    assert "linear_teacher/L1_tanh_r0" in lines[0]


def test_cli_init_refuses_overwrite_and_applies_set(tmp_path, capsys):
    assert _cli(capsys, "init", tmp_path, "--preset", "tiny", "--set", "n_epochs=4")[0] == 0
    assert json.loads((tmp_path / "sweep.json").read_text())["n_epochs"] == 4
    code, _, err = _cli(capsys, "init", tmp_path, "--preset", "tiny")
    assert code == 1 and "--force" in err
    assert _cli(capsys, "init", tmp_path, "--preset", "tiny", "--force")[0] == 0


def test_cli_missing_sweep_json_and_bad_index_fail_clearly(tmp_path, capsys):
    for sub in ("list", "count", "status", "run"):
        code, _, err = _cli(capsys, sub, tmp_path)
        assert code == 1 and "sweep.json" in err and "sweep init" in err
    _cli(capsys, "init", tmp_path, "--preset", "tiny")
    code, _, err = _cli(capsys, "run", tmp_path, "--index", 36)
    assert code == 1 and "0..35" in err


def test_cli_run_index_then_status(tmp_path, capsys):
    _cli(capsys, "init", tmp_path, "--preset", "tiny")
    code, _, _ = _cli(capsys, "run", tmp_path, "--index", 0)
    assert code == 0
    _, status, _ = _cli(capsys, "status", tmp_path)
    lines = status.strip().splitlines()
    assert lines[0].split()[:2] == ["0", "done"]
    assert lines[1].split()[:2] == ["1", "missing"]
    assert "1 done, 0 partial, 35 missing" in lines[-1]
