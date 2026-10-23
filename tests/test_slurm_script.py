"""jobs/fct_sweep.sh and jobs/fct_solve.sh: only --submit may reach sbatch, and then only a test double.

A fake `sbatch` first on PATH records its arguments and exits `FAKE_SBATCH_EXIT`; the
non-submit actions run with it exiting 99, so any call to it fails the test loudly.
Most tests use a stub interpreter (`FCT_PYTHON`) instead of importing torch through the
real `fct`; a few integration tests at the end run the real CLI.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "jobs" / "fct_sweep.sh"
SOLVE_SCRIPT = PROJECT / "jobs" / "fct_solve.sh"
STUB_COUNT = 5
LOG_PATTERN = "tmp/logs/fct_sweep_%A_%a"


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
def stub_python(tmp_path):
    """An interpreter stand-in: records its args and answers the sweep subcommands."""
    log = tmp_path / "python_calls.txt"
    stub = tmp_path / "stub_python"
    stub.write_text(
        "#!/bin/bash\n"
        f'echo "$@" >> "{log}"\n'
        'case "$4" in\n'
        f"  count) echo {STUB_COUNT} ;;\n"
        f"  list) seq 0 {STUB_COUNT - 1} ;;\n"
        '  status) echo "0 done, 0 partial, 5 missing" ;;\n'
        '  run) exit "${STUB_RUN_EXIT:-0}" ;;\n'
        "esac\n"
    )
    stub.chmod(0o755)
    return stub, log


@pytest.fixture(scope="module")
def real_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("real") / "sweep"
    subprocess.run(
        [sys.executable, "-m", "fct", "sweep", "init", str(root)],
        cwd=PROJECT,
        check=True,
        capture_output=True,
    )
    return root


@pytest.fixture
def root(tmp_path):
    path = tmp_path / "sweep"
    path.mkdir()
    return path


def run_script(
    fake_sbatch, python, *args, sbatch_exit=99, env_extra=None, cwd=PROJECT, script=SCRIPT
):
    env = {
        **os.environ,
        "PATH": f"{fake_sbatch[0]}:{os.environ['PATH']}",
        "FAKE_SBATCH_EXIT": str(sbatch_exit),
        "FCT_PYTHON": str(python),
        **(env_extra or {}),
    }
    for name in ("FCT_SWEEP_ROOT", "FCT_PROJECT", "SLURM_ARRAY_TASK_ID", "FCT_MAX_CONCURRENT"):
        if not (env_extra and name in env_extra):
            env.pop(name, None)
    return subprocess.run(
        ["bash", str(script), *args],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def sbatch_calls(fake_sbatch):
    path = fake_sbatch[1]
    return path.read_text().splitlines() if path.exists() else []


def last_index(n=STUB_COUNT):
    return n - 1


def test_syntax_is_valid():
    assert subprocess.run(["bash", "-n", str(SCRIPT)], check=False).returncode == 0


# ------------------------------------------------------------ non-submit actions
def test_dry_run_prints_the_exact_sbatch_command_without_calling_it(fake_sbatch, stub_python, root):
    r = run_script(fake_sbatch, stub_python[0], str(root), "--dry-run")
    assert r.returncode == 0, r.stderr
    assert f"--array=0-{last_index()}%4" in r.stdout
    assert f"--output={PROJECT}/{LOG_PATTERN}.out" in r.stdout
    assert str(root) in r.stdout
    assert "sweep run" in r.stdout and "--index $SLURM_ARRAY_TASK_ID" in r.stdout
    assert sbatch_calls(fake_sbatch) == []


def test_concurrency_cap_is_configurable(fake_sbatch, stub_python, root):
    r = run_script(
        fake_sbatch, stub_python[0], str(root), "--dry-run", env_extra={"FCT_MAX_CONCURRENT": "7"}
    )
    assert f"--array=0-{last_index()}%7" in r.stdout


@pytest.mark.parametrize("given, array", [("3", "3"), ("010", "10"), ("08", "8"), ("0004", "4")])
def test_index_is_normalised_to_base_ten(fake_sbatch, stub_python, root, given, array):
    r = run_script(fake_sbatch, stub_python[0], str(root), "--dry-run", f"--index={given}")
    expected_ok = int(array) < STUB_COUNT
    assert (r.returncode == 0) == expected_ok, r.stderr
    if expected_ok:
        assert f"--array={array}" in r.stdout.split()
    else:
        assert "out of range" in r.stderr
    assert sbatch_calls(fake_sbatch) == []


@pytest.mark.parametrize("given", [str(STUB_COUNT), "0099", "00" + str(STUB_COUNT)])
def test_index_out_of_range_fails(fake_sbatch, stub_python, root, given):
    r = run_script(fake_sbatch, stub_python[0], str(root), "--dry-run", "--index", given)
    assert r.returncode != 0
    assert "out of range" in r.stderr
    assert sbatch_calls(fake_sbatch) == []


@pytest.mark.parametrize(
    "args, message",
    [
        (["--submit", "--index="], "non-negative integer"),
        (["--submit", "--index", ""], "non-negative integer"),
        (["--submit", "--index", "--submit"], "non-negative integer"),
        (["--submit", "--index", "-1"], "non-negative integer"),
        (["--submit", "--index", "x"], "non-negative integer"),
        (["--submit", "--index"], "requires a value"),
        (["--list", "--index", "1"], "only applies to --submit and --dry-run"),
        (["--status", "--index=1"], "only applies to --submit and --dry-run"),
        (["--index", "1"], "only applies to --submit and --dry-run"),
        (["--bogus"], "Unrecognised argument: '--bogus'"),
        (["--status", "--list"], "only one action"),
        ([], "no action given"),
    ],
)
def test_bad_arguments_fail_clearly(fake_sbatch, stub_python, root, args, message):
    r = run_script(fake_sbatch, stub_python[0], str(root), *args)
    assert r.returncode != 0
    assert message in r.stderr
    assert sbatch_calls(fake_sbatch) == []


def test_second_root_fails(fake_sbatch, stub_python, root, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    r = run_script(fake_sbatch, stub_python[0], str(root), str(other), "--list")
    assert r.returncode != 0
    assert "more than one ROOT" in r.stderr


def test_list_and_status_pass_through_without_calling_sbatch(fake_sbatch, stub_python, root):
    for action, expected in [("--list", STUB_COUNT), ("--status", 1)]:
        r = run_script(fake_sbatch, stub_python[0], str(root), action)
        assert r.returncode == 0, r.stderr
        assert len(r.stdout.strip().splitlines()) == expected
    assert sbatch_calls(fake_sbatch) == []


def test_root_may_come_from_the_environment(fake_sbatch, stub_python, root):
    r = run_script(fake_sbatch, stub_python[0], "--status", env_extra={"FCT_SWEEP_ROOT": str(root)})
    assert r.returncode == 0, r.stderr


def test_missing_root_fails(fake_sbatch, stub_python):
    r = run_script(fake_sbatch, stub_python[0], "--list")
    assert r.returncode != 0
    assert "no sweep ROOT" in r.stderr


def test_nonexistent_root_fails(fake_sbatch, stub_python, tmp_path):
    r = run_script(fake_sbatch, stub_python[0], str(tmp_path / "nope"), "--list")
    assert r.returncode != 0
    assert "does not exist" in r.stderr
    assert sbatch_calls(fake_sbatch) == []


def test_comma_in_python_path_fails(fake_sbatch, stub_python, root, tmp_path):
    comma = tmp_path / "a,b"
    comma.mkdir()
    py = comma / "python"
    py.write_text(stub_python[0].read_text())
    py.chmod(0o755)
    r = run_script(fake_sbatch, py, str(root), "--dry-run")
    assert r.returncode != 0
    assert "comma" in r.stderr


def test_relative_python_is_resolved_before_changing_directory(fake_sbatch, stub_python, root):
    stub, log = stub_python
    r = run_script(fake_sbatch, f"./{stub.name}", str(root), "--list", cwd=stub.parent)
    assert r.returncode == 0, r.stderr
    assert log.read_text().startswith("-m fct sweep list")


# ----------------------------------------------------------------------- --submit
def test_submit_calls_sbatch_with_the_expected_arguments(fake_sbatch, stub_python, root):
    stub, _ = stub_python
    r = run_script(fake_sbatch, stub, str(root), "--submit", sbatch_exit=0)
    assert r.returncode == 0, r.stderr
    (line,) = sbatch_calls(fake_sbatch)
    assert f"--array=0-{last_index()}%4" in line
    assert f"--output={PROJECT}/{LOG_PATTERN}.out" in line
    assert f"--error={PROJECT}/{LOG_PATTERN}.err" in line
    assert f"--export=ALL,FCT_PROJECT={PROJECT},FCT_PYTHON={stub}" in line
    assert line.endswith(f"{SCRIPT} {root}")


def test_submit_single_index(fake_sbatch, stub_python, root):
    run_script(fake_sbatch, stub_python[0], str(root), "--submit", "--index", "3", sbatch_exit=0)
    (line,) = sbatch_calls(fake_sbatch)
    assert "--array=3" in line.split()


def test_failing_sbatch_propagates_its_exit_code(fake_sbatch, stub_python, root):
    r = run_script(fake_sbatch, stub_python[0], str(root), "--submit", sbatch_exit=3)
    assert r.returncode == 3


# ---------------------------------------------------------------- per-task mode
def task_env(tmp_path, **extra):
    return {"SLURM_ARRAY_TASK_ID": "2", "FCT_PROJECT": str(PROJECT), **extra}


@pytest.mark.parametrize("exit_code", [0, 7])
def test_task_runs_the_setting_and_propagates_the_exit_code(
    fake_sbatch, stub_python, root, tmp_path, exit_code
):
    stub, log = stub_python
    env = task_env(tmp_path, STUB_RUN_EXIT=str(exit_code))
    r = run_script(fake_sbatch, stub, str(root), env_extra=env)
    assert r.returncode == exit_code, r.stderr
    assert log.read_text().splitlines() == [f"-m fct sweep run {root} --index 2 --stage train"]
    assert sbatch_calls(fake_sbatch) == []


def test_task_without_project_fails(fake_sbatch, stub_python, root):
    r = run_script(fake_sbatch, stub_python[0], str(root), env_extra={"SLURM_ARRAY_TASK_ID": "2"})
    assert r.returncode != 0
    assert "FCT_PROJECT" in r.stderr
    assert not stub_python[1].exists()


def test_task_with_missing_python_fails(fake_sbatch, root, tmp_path):
    r = run_script(fake_sbatch, tmp_path / "no_python", str(root), env_extra=task_env(tmp_path))
    assert r.returncode != 0
    assert "not an executable file" in r.stderr


# ------------------------------------------------------------ the solve script
def run_solve(fake_sbatch, python, *args, **kwargs):
    return run_script(fake_sbatch, python, *args, script=SOLVE_SCRIPT, **kwargs)


def test_solve_script_syntax_is_valid():
    assert subprocess.run(["bash", "-n", str(SOLVE_SCRIPT)], check=False).returncode == 0


def test_solve_dry_run_prints_the_solve_sbatch_command_without_calling_it(
    fake_sbatch, stub_python, root
):
    r = run_solve(fake_sbatch, stub_python[0], str(root), "--dry-run")
    assert r.returncode == 0, r.stderr
    assert f"--array=0-{last_index()}%4" in r.stdout
    assert f"--output={PROJECT}/tmp/logs/fct_solve_%A_%a.out" in r.stdout
    assert "fct_sweep_" not in r.stdout
    assert str(SOLVE_SCRIPT) in r.stdout.split()
    assert "--index $SLURM_ARRAY_TASK_ID --stage solve" in r.stdout
    assert sbatch_calls(fake_sbatch) == []


def test_sweep_dry_run_names_the_train_stage(fake_sbatch, stub_python, root):
    r = run_script(fake_sbatch, stub_python[0], str(root), "--dry-run")
    assert "--index $SLURM_ARRAY_TASK_ID --stage train" in r.stdout


def test_solve_submit_resubmits_the_wrapper_not_the_train_script(fake_sbatch, stub_python, root):
    r = run_solve(fake_sbatch, stub_python[0], str(root), "--submit", "--index", "1", sbatch_exit=0)
    assert r.returncode == 0, r.stderr
    (line,) = sbatch_calls(fake_sbatch)
    assert "--array=1" in line.split()
    assert line.endswith(f"{SOLVE_SCRIPT} {root}")
    assert f"--error={PROJECT}/tmp/logs/fct_solve_%A_%a.err" in line


def test_solve_task_runs_the_solve_stage_and_propagates_the_exit_code(
    fake_sbatch, stub_python, root, tmp_path
):
    stub, log = stub_python
    r = run_solve(fake_sbatch, stub, str(root), env_extra=task_env(tmp_path, STUB_RUN_EXIT="7"))
    assert r.returncode == 7, r.stderr
    assert log.read_text().splitlines() == [f"-m fct sweep run {root} --index 2 --stage solve"]
    assert sbatch_calls(fake_sbatch) == []


def test_solve_status_asks_for_the_solve_stage(fake_sbatch, stub_python, root):
    stub, log = stub_python
    r = run_solve(fake_sbatch, stub, str(root), "--status")
    assert r.returncode == 0, r.stderr
    assert log.read_text().strip() == f"-m fct sweep status {root} --stage solve"
    assert sbatch_calls(fake_sbatch) == []


def test_solve_script_requests_one_gpu_per_task():
    header = [line for line in SOLVE_SCRIPT.read_text().splitlines() if line.startswith("#SBATCH")]
    assert "#SBATCH --gres=gpu:1" in header
    assert "#SBATCH -J fct-solve" in header


def test_an_unknown_stage_in_the_environment_fails(fake_sbatch, stub_python, root):
    r = run_script(
        fake_sbatch, stub_python[0], str(root), "--dry-run", env_extra={"FCT_STAGE": "plot"}
    )
    assert r.returncode != 0
    assert "FCT_STAGE" in r.stderr


# ------------------------------------------------------- integration (real fct)
def test_real_cli_dry_run_uses_the_sweep_size(fake_sbatch, real_root):
    count = subprocess.run(
        [sys.executable, "-m", "fct", "sweep", "count", str(real_root)],
        cwd=PROJECT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    r = run_script(fake_sbatch, sys.executable, str(real_root), "--dry-run")
    assert r.returncode == 0, r.stderr
    assert f"--array=0-{int(count) - 1}%4" in r.stdout
    assert sbatch_calls(fake_sbatch) == []


def test_real_cli_list_and_status(fake_sbatch, real_root):
    listed = run_script(fake_sbatch, sys.executable, str(real_root), "--list")
    status = run_script(fake_sbatch, sys.executable, str(real_root), "--status")
    assert listed.returncode == 0 and status.returncode == 0, listed.stderr + status.stderr
    n = len(listed.stdout.strip().splitlines())
    assert f"{n} missing" in status.stdout
    assert sbatch_calls(fake_sbatch) == []


def test_real_cli_solve_dry_run_and_status(fake_sbatch, real_root):
    dry = run_solve(fake_sbatch, sys.executable, str(real_root), "--dry-run")
    status = run_solve(fake_sbatch, sys.executable, str(real_root), "--status")
    assert dry.returncode == 0, dry.stderr
    assert status.returncode == 0, status.stderr
    assert "--stage solve" in dry.stdout
    assert "36 missing" in status.stdout
    assert sbatch_calls(fake_sbatch) == []
