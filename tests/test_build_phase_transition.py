"""The phase-transition notebook (the viewer for the `lambda_exp` preset): structure of the source, and an end-to-end execution on a tiny
phase-transition experiment with a done, a partial and many missing settings."""

from pathlib import Path

import nbformat
import pytest

from fct.exp import expand, run_setting, save_exp, setting_dir
from notebooks.build_modularity import execute_notebook, with_root
from notebooks.build_phase_transition import build_notebook
from tests.report_helpers import QUIET, tiny_phase_exp

N_SETTINGS = 16
FIGURES_ONE_SETTING = 3  # loss curves, example outputs, MIR history


def _code(nb) -> str:
    return "\n".join(c.source for c in nb.cells if c.cell_type == "code")


def _text_outputs(nb) -> str:
    chunks = []
    for cell in nb.cells:
        for out in cell.get("outputs", []):
            if out.output_type == "error":
                chunks.append(f"ERROR {out.ename}: {out.evalue}")
            elif out.output_type == "display_data":
                chunks.append(out.data.get("text/markdown", ""))
            elif out.output_type == "stream":
                chunks.append(out.text)
    return "\n".join(chunks)


def _n_figures(nb) -> int:
    return sum(
        "image/png" in out.get("data", {}) for cell in nb.cells for out in cell.get("outputs", [])
    )


def test_source_notebook_validates_has_no_outputs_and_no_path():
    nb = build_notebook()
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert all(not c.outputs and c.execution_count is None for c in code)
    assert "/ceph" not in _code(nb) and "/nfs" not in _code(nb)
    parameter = [c for c in nb.cells if "parameters" in c.metadata.get("tags", [])]
    assert len(parameter) == 1 and "ROOT" in parameter[0].source


def test_source_notebook_exposes_the_selector_and_plot_parameters():
    code = _code(build_notebook())
    for name in (
        "L =",
        "PHI =",
        "LAMBDA_R =",
        "LAMBDA_W =",
        "N_PLOT_NEURONS =",
        "PLOT_LAYER =",
        "NEURON_ORDER =",
        "NEURON_SEED =",
        "MIR_KIND =",
    ):
        assert name in code
    for order in ("random", "mir_ascending", "mir_descending"):
        assert order in code


def test_with_root_injects_only_the_parameter_cell(tmp_path):
    nb = build_notebook()
    before = nbformat.writes(nb)
    injected = with_root(nb, tmp_path)
    assert nbformat.writes(nb) == before
    assert repr(str(tmp_path)) in injected.cells[1].source


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    """Setting 0 done, 1 stopped at its checkpoint, the other 14 never run."""
    root = tmp_path_factory.mktemp("phase")
    exp = tiny_phase_exp()
    save_exp(exp, root)
    for index in (0, 1):
        run_setting(exp, index, root, **QUIET)
    (setting_dir(root, expand(exp)[1]) / "train" / "final.pt").unlink()
    return root


def test_executes_on_a_done_a_partial_and_missing_settings(root):
    nb = execute_notebook(build_notebook(), root)
    text = _text_outputs(nb)
    assert "ERROR" not in text
    assert f"1 done, 1 partial (provisional), {N_SETTINGS - 2} missing" in text
    assert "snapped to" in text  # the default 1e-3 is off the 2-value lambda axes
    assert "provisional" in text  # the global figures say which cell is a checkpoint
    # the setting's three figures, the tuning figure, the matrix and the scatter
    assert _n_figures(nb) == FIGURES_ONE_SETTING + 3


def test_a_missing_selection_gives_a_message_and_no_setting_plots(root):
    source = build_notebook()
    selector = next(c for c in source.cells if c.source.startswith("L = 1"))
    selector.source = selector.source.replace("L = 1", "L = 2")  # no L=2 setting has run
    nb = execute_notebook(source, root)
    text = _text_outputs(nb)
    assert "ERROR" not in text
    assert "(missing)" in text and "no data, so no tuning curves" in text
    assert "no MIR in any network yet" in text
    assert _n_figures(nb) == 0


def test_the_committed_notebook_is_what_the_builder_produces():
    """The source notebook is generated; a stale copy would call an outdated API."""
    committed = nbformat.read(Path(__file__).parents[1] / "notebooks" / "phase_transition.ipynb", 4)
    assert [(c.cell_type, c.source) for c in committed.cells] == [
        (c.cell_type, c.source) for c in build_notebook().cells
    ]
