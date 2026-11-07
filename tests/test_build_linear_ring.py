"""The linear-ring notebook: the structure of its source, and end-to-end executions on a smoke ROOT
of `linear_ring_tiny` trained through the CLI on the CPU (its penalty and whitened diagonal
settings, 0 and 3, at B = 32 so the training takes well under a minute), with both settings
side by side, with one, and with a setting that is not trained.

The executions train and are marked slow, like tests/test_linear_ring_smoke.py."""

import ast
import re
from pathlib import Path

import nbformat
import pytest
import torch

from fct import draw
from fct.cli import main
from notebooks.build_linear_ring import build_notebook
from notebooks.build_spatial import execute_notebook
from tests.test_build_spatial import _code, _n_figures_in, _text_outputs

PRESET = "linear_ring_tiny"
# The figures of a trained run of K = 2 latents: loss, relaxation, spectra, band share sweep,
# ratio, autocorrelation (the angle histogram needs K = 3).
N_FIGURES = 6
DRAW_CALL = re.compile(r"draw\.(plot_\w+)\(")


# ---------------------------------------------------------------- the source notebook


def test_source_notebook_validates_has_no_outputs_and_no_path():
    nb = build_notebook()
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert all(not c.outputs and c.execution_count is None for c in code)
    assert "/ceph" not in _code(nb) and "/nfs" not in _code(nb)
    parameter = [c for c in nb.cells if "parameters" in c.metadata.get("tags", [])]
    assert len(parameter) == 1
    for name in ("ROOT =", "SETTING =", "TWIN_ROOT =", "TWIN_SETTING =", "BURN_IN =", "EXACT ="):
        assert name in parameter[0].source


def test_every_cell_parses():
    for cell in build_notebook().cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)


def test_every_figure_is_drawn_by_a_plot_function_of_fct_draw_and_all_of_them_are_used():
    called = set(DRAW_CALL.findall(_code(build_notebook())))
    assert called == {
        "plot_ensemble_loss",
        "plot_relaxation",
        "plot_spectrum",
        "plot_share_sweep",
        "plot_spectrum_ratio",
        "plot_autocorrelation",
        "plot_angle_histogram",
    }
    assert all(callable(getattr(draw, name)) for name in called)


def test_every_figure_cell_is_introduced_by_an_equation():
    nb = build_notebook()
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code" and DRAW_CALL.search(cell.source):
            assert "$" in nb.cells[i - 1].source, cell.source


def test_the_notebook_calls_no_mir_reader():
    """The MIR readers refuse spectrum experiments; only the $FCT_EXP_ROOT reader is taken."""
    code = _code(build_notebook())
    assert "fct.exp_report" not in code and "fct.exp_view" not in code
    imports = re.findall(r"from fct\.report import (.+)", code)
    assert [line.split("#")[0].strip() for line in imports] == ["exp_root_from_env"]


def test_the_notebook_states_the_limit_of_the_theory_and_the_lr_bias():
    intro = build_notebook().cells[0].source
    assert r"$N \to \infty$ mean-field limit" in intro
    assert "biases mode 1" in intro and "description" in intro


def test_the_committed_notebook_is_what_the_builder_produces():
    committed = nbformat.read(Path(__file__).parents[1] / "notebooks" / "linear_ring.ipynb", 4)
    assert [(c.cell_type, c.source) for c in committed.cells] == [
        (c.cell_type, c.source) for c in build_notebook().cells
    ]


# ---------------------------------------------------------------- execution


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    """`linear_ring_tiny` at B = 32 with its diagonal settings 0 (penalty) and 3 (whitened)
    trained through the CLI; settings 1 and 2 are not run."""
    root = tmp_path_factory.mktemp("linear_ring_tiny")
    threads = torch.get_num_threads()
    # The smoke ring's matrices are 32 x 16: one thread is faster than many here.
    torch.set_num_threads(1)
    try:
        assert main(["exp", "init", str(root), "--preset", PRESET, "--set", "B=32"]) == 0
        for index in (0, 3):
            assert main(["exp", "run", str(root), "--index", str(index)]) == 0
    finally:
        torch.set_num_threads(threads)
    return root


def _execute(root: Path, **overrides):
    return execute_notebook(build_notebook(), root, overrides, root_name="ROOT")


@pytest.mark.slow
def test_executes_on_the_two_parametrisations_side_by_side(root):
    nb = _execute(root)
    text = _text_outputs(nb)
    assert "ERROR" not in text
    assert _n_figures_in(nb) == N_FIGURES
    assert "| 0 | penalty | 0.0006 | done |" in text and "| 3 | whitened | 0.005 | done |" in text
    assert text.count("Trained; 41 records, the last at epoch 10000") == 2
    assert "made from the preset `linear_ring_tiny` with overrides" in text
    assert "K = 2 latents and no recorded angle: skipped" in text
    assert text.count("| R(1)/R(0) | 1 | primary |") == 2  # a summary table per setting
    assert "**Equivalence of the two settings**" in text
    assert "7 statistics compared" in text
    for i, cell in enumerate(nb.cells):
        if any("image/png" in o.get("data", {}) for o in cell.get("outputs", [])):
            assert "$" in nb.cells[i - 1].source


@pytest.mark.slow
def test_executes_on_one_setting_swept_along_an_axis(root):
    nb = _execute(
        root,
        SETTING={"spatial.parametrisation": "whitened", "train.lr": 0.005},
        TWIN_SETTING=None,
        SWEEP_AXIS="train.lr",
        BURN_IN=2000,
    )
    text = _text_outputs(nb)
    assert "ERROR" not in text
    assert _n_figures_in(nb) == N_FIGURES
    assert text.count("| R(1)/R(0) | 1 | primary |") == 1
    assert "Equivalence" not in text
    assert "fit from epoch 2000" in text


@pytest.mark.slow
def test_a_setting_that_is_not_trained_is_reported_and_its_figures_skipped(root):
    nb = _execute(
        root,
        SETTING={"spatial.parametrisation": "whitened", "train.lr": 0.0006},
        TWIN_SETTING=None,
    )
    text = _text_outputs(nb)
    assert "ERROR" not in text
    assert "**Not trained** (status: missing)" in text
    assert _n_figures_in(nb) == 0
    # the measurement, the seven figures and the summary
    assert text.count("No shown setting is trained") == 9
