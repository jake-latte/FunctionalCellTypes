"""jobs/fct_sweep.sh: only --submit may reach sbatch, and then only a test double.

A fake `sbatch` first on PATH records its arguments and exits `FAKE_SBATCH_EXIT`; the
non-submit actions run with it exiting 99, so any call to it fails the test loudly.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "jobs" / "fct_sweep.sh"


@pytest.fixture
def fake_sbatch(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls = tmp_path / "sbatch_calls.txt"
    sbatch = bindir / "sbatch"
    sbatch.write_text(f'#!/bin/bash\necho "$@" >> "{calls}"\nexit "${{FAKE_SBATCH_EXIT:-99}}"\n')
    sbatch.chmod(0o755)
    return bindir, calls


@pytest.fixture
def sweep_root(tmp_path):
    root = tmp_path / "sweep"
    subprocess.run(
        [sys.executable, "-m", "fct", "sweep", "init", str(root)],
        cwd=PROJECT, check=True, capture_output=True,
    )  # fmt: skip
    return root


def run_script(fake_sbatch, *args, sbatch_exit=99, env_extra=None):
    bindir, _ = fake_sbatch
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "FAKE_SBATCH_EXIT": str(sbatch_exit),
        "FCT_PYTHON": sys.executable,  # the worktree may have no .venv
        **(env_extra or {}),
    }
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        env=env,
        capture_output=True,
        text=True,
        cwd=PROJECT,
        check=False,
    )


def calls(fake_sbatch):
    path = fake_sbatch[1]
    return path.read_text().splitlines() if path.exists() else []


def test_syntax_is_valid():
    assert subprocess.run(["bash", "-n", str(SCRIPT)], check=False).returncode == 0


def test_dry_run_prints_the_exact_sbatch_command_without_calling_it(fake_sbatch, sweep_root):
    r = run_script(fake_sbatch, str(sweep_root), "--dry-run")
    assert r.returncode == 0, r.stderr
    assert "sbatch" in r.stdout
    assert "--array=0-35%4" in r.stdout
    assert f"--output={PROJECT}/tmp/logs/fct_sweep_%A_%a.out" in r.stdout
    assert str(sweep_root) in r.stdout
    assert "-m fct sweep run" in r.stdout and "--index $SLURM_ARRAY_TASK_ID" in r.stdout
    assert calls(fake_sbatch) == []


def test_concurrency_cap_is_configurable(fake_sbatch, sweep_root):
    r = run_script(fake_sbatch, str(sweep_root), "--dry-run", env_extra={"FCT_MAX_CONCURRENT": "7"})
    assert "--array=0-35%7" in r.stdout


def test_dry_run_with_index_targets_one_setting(fake_sbatch, sweep_root):
    r = run_script(fake_sbatch, str(sweep_root), "--dry-run", "--index", "5")
    assert r.returncode == 0, r.stderr
    assert "--array=5 " in r.stdout
    assert calls(fake_sbatch) == []


def test_index_out_of_range_fails(fake_sbatch, sweep_root):
    r = run_script(fake_sbatch, str(sweep_root), "--dry-run", "--index", "36")
    assert r.returncode != 0
    assert "out of range" in r.stderr


def test_list_prints_settings_without_calling_sbatch(fake_sbatch, sweep_root):
    r = run_script(fake_sbatch, str(sweep_root), "--list")
    assert r.returncode == 0, r.stderr
    assert len(r.stdout.strip().splitlines()) == 36
    assert calls(fake_sbatch) == []


def test_status_prints_summary_without_calling_sbatch(fake_sbatch, sweep_root):
    r = run_script(fake_sbatch, str(sweep_root), "--status")
    assert r.returncode == 0, r.stderr
    assert "0 done, 0 partial, 36 missing" in r.stdout
    assert calls(fake_sbatch) == []


def test_root_may_come_from_the_environment(fake_sbatch, sweep_root):
    r = run_script(fake_sbatch, "--status", env_extra={"FCT_SWEEP_ROOT": str(sweep_root)})
    assert r.returncode == 0, r.stderr
    assert "36 missing" in r.stdout


def test_submit_calls_sbatch_with_the_expected_arguments(fake_sbatch, sweep_root):
    r = run_script(fake_sbatch, str(sweep_root), "--submit", sbatch_exit=0)
    assert r.returncode == 0, r.stderr
    (line,) = calls(fake_sbatch)
    assert "--array=0-35%4" in line
    assert f"--output={PROJECT}/tmp/logs/fct_sweep_%A_%a.out" in line
    assert f"--error={PROJECT}/tmp/logs/fct_sweep_%A_%a.err" in line
    assert f"FCT_PROJECT={PROJECT}" in line
    assert line.endswith(f"{SCRIPT} {sweep_root}")


def test_submit_single_index(fake_sbatch, sweep_root):
    run_script(fake_sbatch, str(sweep_root), "--submit", "--index", "3", sbatch_exit=0)
    (line,) = calls(fake_sbatch)
    assert "--array=3 " in line


def test_failing_sbatch_propagates(fake_sbatch, sweep_root):
    r = run_script(fake_sbatch, str(sweep_root), "--submit", sbatch_exit=3)
    assert r.returncode != 0


@pytest.mark.parametrize("args", [["--bogus"], ["--status", "--list"]])
def test_unknown_or_repeated_action_fails_clearly(fake_sbatch, sweep_root, args):
    r = run_script(fake_sbatch, str(sweep_root), *args)
    assert r.returncode != 0
    assert "Unrecognised" in r.stderr or "one action" in r.stderr
    assert calls(fake_sbatch) == []


def test_missing_action_fails(fake_sbatch, sweep_root):
    r = run_script(fake_sbatch, str(sweep_root))
    assert r.returncode != 0
    assert "Use:" in r.stderr


def test_missing_root_fails(fake_sbatch):
    r = run_script(fake_sbatch, "--list")
    assert r.returncode != 0
    assert "ROOT" in r.stderr
    assert calls(fake_sbatch) == []


def test_nonexistent_root_fails(fake_sbatch, tmp_path):
    r = run_script(fake_sbatch, str(tmp_path / "nope"), "--list")
    assert r.returncode != 0
    assert "does not exist" in r.stderr
    assert calls(fake_sbatch) == []
