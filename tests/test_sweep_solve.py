"""Solving the saddle points of a sweep's settings: the solver config in sweep.json, the
per-stage status, and train-only / solve-only / train-then-solve runs on a tiny grid (CPU)."""

import json

import pytest
import torch

import fct.meanfield as mf
from fct.cache import StaleCacheError, cache_paths, save_artefact
from fct.cli import main
from fct.config import MeanFieldConfig, SolverConfig, load_config
from fct.meanfield import InadequateSamplerError, load_meanfield
from fct.sweep import (
    SweepConfig,
    SweepError,
    expand,
    load_sweep,
    preset,
    run_setting,
    save_sweep,
    setting_config,
    setting_dir,
    setting_status,
    with_grid,
)
from tests.report_helpers import QUIET

# A solve of a few outer steps at ten points: seconds on a CPU.
TINY_SOLVER = SolverConfig(n_chains=32, n_outer=3, save_int=1, log_int=1, warmup_trans=10)
TINY_MEANFIELD = MeanFieldConfig(n_points=10, beta=1e3, solver=TINY_SOLVER)


@pytest.fixture
def small() -> SweepConfig:
    """Two settings (L = 1, 2) with the tiny solve."""
    base = preset("tiny").model_copy(update={"meanfield": TINY_MEANFIELD})
    return with_grid(base, {"net.L": [1, 2]})


def _with_beta(sweep: SweepConfig, beta: float) -> SweepConfig:
    return sweep.model_copy(update={"meanfield": sweep.meanfield.model_copy(update={"beta": beta})})


def _dirs(root, sweep, i):
    return cache_paths(setting_dir(root, expand(sweep)[i]))


def _solve_hash(sweep, i):
    return setting_config(sweep, expand(sweep)[i]).section_hash("data", "net", "train", "meanfield")


def _write_solved(root, sweep, i, config_hash=None):
    """A minimal meanfield/final.pt carrying the solve hash of setting i (or `config_hash`)."""
    save_artefact(
        _dirs(root, sweep, i).meanfield / "final.pt",
        {"x": 0},
        config_hash or _solve_hash(sweep, i),
    )


def _status(root, sweep, i, stage):
    return setting_status(root, sweep, expand(sweep)[i], stage)


class Interrupt(Exception):
    pass


# ---------------------------------------------------------------------------- config
def test_sweep_config_carries_a_default_meanfield_and_a_legacy_file_loads(tmp_path):
    tree = preset("tiny").model_dump()
    del tree["meanfield"]  # a sweep.json written before the solver was part of it
    (tmp_path / "sweep.json").write_text(json.dumps(tree))
    assert load_sweep(tmp_path).meanfield == MeanFieldConfig()


def test_meanfield_round_trips_through_sweep_json(small, tmp_path):
    save_sweep(small, tmp_path)
    assert load_sweep(tmp_path).meanfield == TINY_MEANFIELD


def test_the_solver_config_reaches_every_setting_config(small):
    for setting in expand(small):
        assert setting_config(small, setting).meanfield == TINY_MEANFIELD


def test_the_meanfield_section_never_changes_training_hashes_or_directories():
    legacy = preset("tiny")
    changed = _with_beta(legacy, 2e3)
    assert expand(changed) == expand(legacy)
    for setting in expand(legacy):
        hashes = [
            setting_config(sweep, setting).section_hash("data", "net", "train")
            for sweep in (legacy, changed)
        ]
        assert hashes[0] == hashes[1]
    assert setting_config(legacy, expand(legacy)[0]).meanfield == MeanFieldConfig()


def test_rate_tilt_follows_the_library_default(small):
    assert setting_config(small, expand(small)[0]).meanfield.rate_tilt is None


def test_the_solve_hash_is_the_one_run_meanfield_writes(small):
    cfg = setting_config(small, expand(small)[0])
    assert _solve_hash(small, 0) == mf._meanfield_hash(cfg)


# ---------------------------------------------------------------------------- status
def test_solve_status_done_partial_missing(small, tmp_path):
    _write_solved(tmp_path, small, 0)
    assert _status(tmp_path, small, 0, "solve") == "done"
    assert _status(tmp_path, small, 1, "solve") == "missing"
    ckpt_dir = _dirs(tmp_path, small, 1).meanfield
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "ckpt_00001.pt").touch()
    assert _status(tmp_path, small, 1, "solve") == "partial"


def test_solve_status_done_wins_over_leftover_checkpoint(small, tmp_path):
    _write_solved(tmp_path, small, 0)
    (_dirs(tmp_path, small, 0).meanfield / "ckpt_00001.pt").touch()
    assert _status(tmp_path, small, 0, "solve") == "done"


def test_solve_status_is_stale_for_a_foreign_hash_or_an_unreadable_file(small, tmp_path):
    _write_solved(tmp_path, small, 0, config_hash="not-this-config")
    assert _status(tmp_path, small, 0, "solve") == "stale"
    _write_solved(tmp_path, small, 1)
    assert _status(tmp_path, small, 1, "solve") == "done"
    (_dirs(tmp_path, small, 1).meanfield / "final.pt").write_bytes(b"truncated")
    assert _status(tmp_path, small, 1, "solve") == "stale"


def test_solve_status_is_stale_when_the_solver_or_the_training_config_changed(small, tmp_path):
    _write_solved(tmp_path, small, 0)
    assert _status(tmp_path, _with_beta(small, 2e3), 0, "solve") == "stale"
    assert _status(tmp_path, small.model_copy(update={"lr": small.lr * 2}), 0, "solve") == "stale"


def test_stages_are_independent_and_train_is_the_default(small, tmp_path):
    _write_solved(tmp_path, small, 0)
    setting = expand(small)[0]
    assert setting_status(tmp_path, small, setting) == "missing"
    assert setting_status(tmp_path, small, setting, "train") == "missing"
    assert setting_status(tmp_path, small, setting, "solve") == "done"


def test_an_unknown_stage_is_rejected(small, tmp_path):
    with pytest.raises(ValueError, match="stage"):
        setting_status(tmp_path, small, expand(small)[0], "plot")
    with pytest.raises(ValueError, match="stage"):
        run_setting(small, 0, tmp_path, stage="plot", **QUIET)


# ---------------------------------------------------------------------------- running
def test_train_only_leaves_no_meanfield_directory(small, tmp_path):
    run_setting(small, 0, tmp_path, **QUIET)
    assert _status(tmp_path, small, 0, "train") == "done"
    assert _status(tmp_path, small, 0, "solve") == "missing"
    assert not _dirs(tmp_path, small, 0).meanfield.exists()


def _history(root, sweep, i):
    setting = expand(sweep)[i]
    return load_meanfield(setting_dir(root, setting), setting_config(sweep, setting), "cpu")[2]


def test_solve_only_runs_without_a_trained_net(small, tmp_path):
    run_setting(small, 0, tmp_path, stage="solve", **QUIET)
    assert _status(tmp_path, small, 0, "solve") == "done"
    assert _status(tmp_path, small, 0, "train") == "missing"
    history = _history(tmp_path, small, 0)
    assert "align_net" not in history  # no net, so no net comparison
    assert len(history["train_mse_theory"]) == TINY_SOLVER.n_outer


def test_train_then_solve_compares_with_the_trained_net(small, tmp_path):
    run_setting(small, 0, tmp_path, **QUIET)
    run_setting(small, 0, tmp_path, stage="solve", **QUIET)
    assert "align_net" in _history(tmp_path, small, 0)
    assert _status(tmp_path, small, 0, "train") == "done"
    assert _status(tmp_path, small, 0, "solve") == "done"


def test_solving_leaves_the_trained_net_untouched(small, tmp_path):
    run_setting(small, 0, tmp_path, **QUIET)
    final = _dirs(tmp_path, small, 0).train / "final.pt"
    before = (final.read_bytes(), final.stat().st_mtime_ns)
    run_setting(small, 0, tmp_path, stage="solve", **QUIET)
    assert (final.read_bytes(), final.stat().st_mtime_ns) == before


def test_a_rerun_of_either_stage_is_a_no_op(small, tmp_path):
    for stage in ("train", "solve"):
        run_setting(small, 0, tmp_path, stage=stage, **QUIET)
    paths = _dirs(tmp_path, small, 0)
    finals = [paths.train / "final.pt", paths.meanfield / "final.pt"]
    before = [(f.read_bytes(), f.stat().st_mtime_ns) for f in finals]
    for stage in ("train", "solve"):
        lines: list[str] = []
        run_setting(small, 0, tmp_path, stage=stage, log=lines.append)
        assert any("already done" in line for line in lines)
    assert [(f.read_bytes(), f.stat().st_mtime_ns) for f in finals] == before


def test_an_interrupted_solve_resumes_to_the_uninterrupted_result(small, tmp_path):
    run_setting(small, 0, tmp_path / "straight", stage="solve", **QUIET)

    def stop_at_outer_step_1(line: str) -> None:
        if line.startswith("[0001]"):
            raise Interrupt

    with pytest.raises(Interrupt):
        run_setting(small, 0, tmp_path / "broken", stage="solve", log=stop_at_outer_step_1)
    assert _status(tmp_path / "broken", small, 0, "solve") == "partial"
    run_setting(small, 0, tmp_path / "broken", stage="solve", **QUIET)
    assert _status(tmp_path / "broken", small, 0, "solve") == "done"

    def phi(root):
        final = _dirs(root, small, 0).meanfield / "final.pt"
        return torch.load(final, weights_only=True)["Phi"]

    torch.testing.assert_close(phi(tmp_path / "broken"), phi(tmp_path / "straight"))


def test_a_solve_finished_under_another_solver_config_is_stale_and_refused(small, tmp_path):
    run_setting(small, 0, tmp_path, stage="solve", **QUIET)
    changed = _with_beta(small, 2e3)
    assert _status(tmp_path, changed, 0, "solve") == "stale"
    with pytest.raises(StaleCacheError, match="meanfield"):
        run_setting(changed, 0, tmp_path, stage="solve", **QUIET)


def test_changing_the_solver_does_not_block_training_and_config_json_follows_it(small, tmp_path):
    run_setting(small, 0, tmp_path, **QUIET)
    changed = _with_beta(small, 2e3)
    run_setting(changed, 0, tmp_path, stage="solve", **QUIET)
    config_path = _dirs(tmp_path, small, 0).config
    assert load_config(config_path).meanfield.beta == 2e3
    assert _status(tmp_path, changed, 0, "train") == "done"


def test_solve_refuses_a_setting_whose_training_config_changed(small, tmp_path):
    run_setting(small, 0, tmp_path, **QUIET)
    changed = small.model_copy(update={"lr": small.lr * 2})
    with pytest.raises(SweepError, match="differs"):  # config.json belongs to the old training
        run_setting(changed, 0, tmp_path, stage="solve", **QUIET)


def test_an_inadequate_sampler_fails_the_setting_and_leaves_no_final(small, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise InadequateSamplerError("importance sampler inadequate at the test")

    monkeypatch.setattr(mf, "_outer_step", fail)
    with pytest.raises(InadequateSamplerError, match="inadequate"):
        run_setting(small, 0, tmp_path, stage="solve", **QUIET)
    assert not (_dirs(tmp_path, small, 0).meanfield / "final.pt").exists()


# ---------------------------------------------------------------------------- CLI
def _cli(capsys, *args) -> tuple[int, str, str]:
    code = main(["sweep", *map(str, args)])
    out = capsys.readouterr()
    return code, out.out, out.err


@pytest.fixture
def root(small, tmp_path):
    save_sweep(small, tmp_path)
    return tmp_path


def test_cli_status_and_run_take_a_stage(small, root, capsys):
    assert _cli(capsys, "run", root, "--index", 0, "--stage", "solve")[0] == 0
    _, solve_status, _ = _cli(capsys, "status", root, "--stage", "solve")
    assert solve_status.splitlines()[0].split()[:2] == ["0", "done"]
    assert "1 done, 0 partial, 1 missing, 0 stale" in solve_status
    _, train_status, _ = _cli(capsys, "status", root)  # default stage: train
    assert "0 done, 0 partial, 2 missing, 0 stale" in train_status
    assert _cli(capsys, "status", root, "--stage", "train")[1] == train_status


def test_cli_run_all_solves_every_setting(root, capsys):
    assert _cli(capsys, "run", root, "--stage", "solve")[0] == 0
    _, status, _ = _cli(capsys, "status", root, "--stage", "solve")
    assert "2 done, 0 partial, 0 missing, 0 stale" in status


def test_cli_rejects_an_unknown_stage(root, capsys):
    with pytest.raises(SystemExit):
        main(["sweep", "status", str(root), "--stage", "plot"])
    capsys.readouterr()


def test_cli_inadequate_sampler_exits_nonzero_with_a_message(small, root, capsys, monkeypatch):
    def fail(*args, **kwargs):
        raise InadequateSamplerError("the prior proposal is inadequate here")

    monkeypatch.setattr(mf, "_outer_step", fail)
    code, _, err = _cli(capsys, "run", root, "--index", 0, "--stage", "solve")
    assert code == 1
    assert "inadequate" in err and "Traceback" not in err
    assert not (_dirs(root, small, 0).meanfield / "final.pt").exists()


def test_cli_run_all_reports_an_inadequate_setting_and_continues(small, root, capsys, monkeypatch):
    real = mf._outer_step
    calls = []

    def fail_first_call(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise InadequateSamplerError("inadequate once")
        return real(*args, **kwargs)

    monkeypatch.setattr(mf, "_outer_step", fail_first_call)
    code, _, err = _cli(capsys, "run", root, "--stage", "solve")
    assert code == 1
    assert "setting 0 failed: InadequateSamplerError" in err
    assert _status(root, small, 1, "solve") == "done"


def test_list_and_count_are_unchanged_by_the_solver_section(small, root, capsys):
    assert _cli(capsys, "count", root)[1].strip() == "2"
    names = [line.split()[1] for line in _cli(capsys, "list", root)[1].splitlines()]
    assert names == [str(setting_dir(root, s).relative_to(root)) for s in expand(small)]
