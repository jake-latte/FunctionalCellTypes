"""The playground notebook runs top to bottom: it defines, saves and trains a small experiment
under $FCT_PLAYGROUND_ROOT, skips finished settings when run again, and refuses to overwrite a
different experiment already in ROOT."""

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from fct.exp import expand, load_exp, setting_status
from notebooks.build_spatial import set_parameters

NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "playground.ipynb"
# Shrinks the notebook's own small defaults further; the notebook's AXES are kept.
FAST = {
    "OVERRIDES": {"n_epochs": 4, "N": 8, "P": 32, "log_int": 2, "save_int": 2, "monitor_int": 2}
}


def _run(root: Path, monkeypatch, overrides: dict | None = None):
    """Execute a copy of the notebook with ROOT taken from the environment; return the cells."""
    monkeypatch.setenv("FCT_PLAYGROUND_ROOT", str(root))
    nb = set_parameters(nbformat.read(NOTEBOOK, 4), overrides or FAST)
    return NotebookClient(nb, timeout=300, kernel_name="python3", record_timing=False).execute()


def _stream(nb) -> str:
    return "".join(o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.outputs)


def test_source_notebook_validates_and_has_no_outputs():
    nb = nbformat.read(NOTEBOOK, 4)
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert all(not c.outputs and c.execution_count is None for c in code)
    assert "/ceph" not in "".join(c.source for c in code)
    parameter = [c for c in nb.cells if "parameters" in c.metadata.get("tags", [])]
    assert len(parameter) == 1
    for name in ("ROOT =", "PRESET =", "OVERRIDES =", "AXES =", "DEVICE =", "SETTING ="):
        assert name in parameter[0].source


def test_trains_every_setting_and_a_second_run_trains_nothing(tmp_path, monkeypatch):
    root = tmp_path / "playground"
    first = _run(root, monkeypatch)
    exp = load_exp(root)  # exp.json was written
    settings = expand(exp)
    assert settings
    assert all(setting_status(root, exp, s) == "done" for s in settings)
    assert "already done" not in _stream(first)

    again = _stream(_run(root, monkeypatch))
    assert again.count("already done") == len(settings)


def test_a_different_experiment_in_root_is_refused_not_overwritten(tmp_path, monkeypatch):
    root = tmp_path / "playground"
    _run(root, monkeypatch)
    before = (root / "exp.json").read_text()
    other = {"OVERRIDES": {**FAST["OVERRIDES"], "n_epochs": 6}}
    with pytest.raises(CellExecutionError, match="different experiment"):
        _run(root, monkeypatch, other)
    assert (root / "exp.json").read_text() == before
