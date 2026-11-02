"""The playground notebook runs top to bottom: it defines, saves and trains a small experiment
under $FCT_PLAYGROUND_ROOT, skips finished settings when run again, loads a setting on any load
device, and refuses a ROOT that belongs to another experiment."""

import hashlib
import re
import shutil
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from fct.exp import expand, load_exp, setting_status
from notebooks.build_spatial import EXECUTE_TIMEOUT_S, set_parameters

REPO = Path(__file__).parents[1]
NOTEBOOK = REPO / "notebooks" / "playground.ipynb"
# Shrinks the notebook's own small defaults further; the notebook's AXES are kept.
FAST = {
    "OVERRIDES": {"n_epochs": 4, "N": 8, "P": 32, "log_int": 2, "save_int": 2, "monitor_int": 2}
}


def _run(root: Path, overrides: dict | None = None):
    """Execute a copy of the notebook (ROOT from $FCT_PLAYGROUND_ROOT, which the caller sets);
    the kernel starts in the repo root so it imports this checkout's fct."""
    nb = set_parameters(nbformat.read(NOTEBOOK, 4), overrides or FAST)
    client = NotebookClient(
        nb,
        timeout=EXECUTE_TIMEOUT_S,
        kernel_name="python3",
        record_timing=False,
        resources={"metadata": {"path": str(REPO)}},
    )
    return client.execute()


def _snapshot(root: Path) -> dict[str, tuple[int, str]]:
    """{path: (mtime, md5)} of every file under ROOT except exp.json, which a run may rewrite."""
    return {
        str(p.relative_to(root)): (p.stat().st_mtime_ns, hashlib.md5(p.read_bytes()).hexdigest())
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "exp.json"
    }


@pytest.fixture
def playground_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "playground"
    monkeypatch.setenv("FCT_PLAYGROUND_ROOT", str(root))
    return root


@pytest.fixture(scope="module")
def trained_root(tmp_path_factory) -> Path:
    """One ROOT trained once by the notebook; the tests below only read it or re-run on it."""
    root = tmp_path_factory.mktemp("trained") / "playground"
    mp = pytest.MonkeyPatch()
    mp.setenv("FCT_PLAYGROUND_ROOT", str(root))
    try:
        _run(root)
    finally:
        mp.undo()
    return root


def test_source_notebook_validates_and_has_no_outputs_or_absolute_paths():
    nb = nbformat.read(NOTEBOOK, 4)
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert all(not c.outputs and c.execution_count is None for c in code)
    assert not re.search(r"""["']/[A-Za-z]""", "".join(c.source for c in code))
    parameter = [c for c in nb.cells if "parameters" in c.metadata.get("tags", [])]
    assert len(parameter) == 1
    for name in ("ROOT =", "PRESET =", "OVERRIDES =", "AXES =", "LOAD_DEVICE =", "SETTING ="):
        assert name in parameter[0].source


def test_writes_exp_json_and_trains_every_setting(trained_root):
    exp = load_exp(trained_root)
    settings = expand(exp)
    assert settings
    assert all(setting_status(trained_root, exp, s) == "done" for s in settings)


def test_a_second_run_trains_nothing_new(trained_root, monkeypatch):
    monkeypatch.setenv("FCT_PLAYGROUND_ROOT", str(trained_root))
    before = _snapshot(trained_root)
    assert before
    _run(trained_root)
    assert _snapshot(trained_root) == before


def test_loads_a_setting_given_as_axes_on_the_auto_device(trained_root, monkeypatch):
    monkeypatch.setenv("FCT_PLAYGROUND_ROOT", str(trained_root))
    _run(trained_root, {**FAST, "LOAD_DEVICE": "auto", "SETTING": {"net.L": 2}})


def test_a_different_experiment_in_root_is_refused_not_overwritten(trained_root, monkeypatch):
    monkeypatch.setenv("FCT_PLAYGROUND_ROOT", str(trained_root))
    before = (trained_root / "exp.json").read_text()
    other = {"OVERRIDES": {**FAST["OVERRIDES"], "n_epochs": 6}}
    with pytest.raises(CellExecutionError, match="different experiment"):
        _run(trained_root, other)
    assert (trained_root / "exp.json").read_text() == before


def test_setting_directories_without_exp_json_are_refused(trained_root, playground_root):
    shutil.copytree(trained_root, playground_root)
    (playground_root / "exp.json").unlink()
    with pytest.raises(CellExecutionError, match="setting directories"):
        _run(playground_root)
    assert not (playground_root / "exp.json").exists()
