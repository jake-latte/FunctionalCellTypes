"""The spatial-experiment notebook: structure of the source, the guard that skips figures of an untrained
setting, and end-to-end executions on a tiny spatial experiment (CPU) with one trained and one
never-run setting."""

import ast
from pathlib import Path

import nbformat
import numpy as np
import pytest
import torch
from nbclient.exceptions import CellExecutionError

from fct.config import DataConfig
from fct.data import Dataset
from fct.exp import preset, run_setting, save_exp
from fct.tasks import latent_lattice_dataset
from notebooks import build_spatial as sp
from notebooks.build_spatial import execute_notebook
from notebooks.build_spatial_exp import build_notebook, when_trained
from tests.test_build_spatial import (
    N_FIGURES,
    PLOTTING_FUNCTIONS,
    _code,
    _n_figures_in,
    _text_outputs,
)

# Weight layers 1 and 2 of the one-hidden-layer net: the only figures that need no trained net.
N_PRIOR_FIGURES = 2
# The code cells of the spatial notebook that need the trained run (everything but the prior).
RUN_CELLS = (
    sp.LOSS,
    sp.EXAMPLES,
    sp.EMPIRICAL,
    sp.WEIGHTS,
    sp.DATA_KERNEL,
    sp.TRIALS,
    sp.PHI_PAIRS,
    sp.TUNING,
    sp.OVERLAP,
)


# ---------------------------------------------------------------- the source notebook


def test_source_notebook_validates_has_no_outputs_and_no_path():
    nb = build_notebook()
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert all(not c.outputs and c.execution_count is None for c in code)
    assert "/ceph" not in _code(nb) and "/nfs" not in _code(nb)
    parameter = [c for c in nb.cells if "parameters" in c.metadata.get("tags", [])]
    assert len(parameter) == 1
    for name in ("ROOT =", "SETTING =", "DEVICE ="):
        assert name in parameter[0].source


def test_every_figure_of_the_spatial_notebook_is_drawn_by_the_same_function_here():
    """The plotting functions are the spatial notebook's, defined in the cell that draws them."""
    exp_code = _code(build_notebook())
    for name in PLOTTING_FUNCTIONS:
        assert f"def {name}(" in exp_code, name
    # a cell's source up to its last function definition is the spatial notebook's, verbatim
    for source in RUN_CELLS:
        end = max(n.end_lineno for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
        assert "\n".join(source.splitlines()[:end]) in exp_code


def test_every_figure_cell_is_introduced_by_an_equation_and_keeps_its_function_at_top_level():
    nb = build_notebook()
    drawing = [
        i
        for i, c in enumerate(nb.cells)
        if c.cell_type == "code" and ("plt.subplots(" in c.source or "plt.figure(" in c.source)
    ]
    assert len(drawing) == len(PLOTTING_FUNCTIONS)
    for i in drawing:
        assert "$$" in nb.cells[i - 1].source, nb.cells[i].source
        top_level = {
            n.name for n in ast.parse(nb.cells[i].source).body if isinstance(n, ast.FunctionDef)
        }
        assert top_level & PLOTTING_FUNCTIONS, nb.cells[i].source


def test_the_phi_and_example_cells_use_the_test_set():
    code = {c.source: c.source for c in build_notebook().cells if c.cell_type == "code"}
    for source in code:
        if "plot_data_kernels" in source or "plot_trial_latents" in source:
            assert "TEST_DATASET" in source and "RUN.dataset" not in source
        if "plot_example_outputs" in source:
            assert "TEST_DATASET" in source and "RUN.dataset" not in source
    assert "load_exp_run" in _code(build_notebook())


def test_set_parameters_takes_the_setting_as_a_dict_literal():
    from notebooks.build_spatial import set_parameters

    out = set_parameters(build_notebook(), {"SETTING": {"net.N": 256}, "ROOT": "experiments/a"})
    code = _code(out)
    assert "SETTING = {'net.N': 256}  # " in code and "ROOT = 'experiments/a'  # " in code


# ---------------------------------------------------------------- the guard


def test_when_trained_guards_everything_after_the_last_function():
    source = "N = 3  # n\n\n\ndef f(x):\n    return x\n\n\nf(N)\nprint(1)"
    guarded = when_trained(source)
    tree = ast.parse(guarded)
    assert isinstance(tree.body[0], ast.Assign) and isinstance(tree.body[1], ast.FunctionDef)
    guard = tree.body[2]
    assert isinstance(guard, ast.If) and len(tree.body) == 3
    assert ast.unparse(guard.test) == "TRAINED" and len(guard.body) == 2
    assert "NOT_TRAINED" in ast.unparse(guard.orelse)


def test_when_trained_guards_a_cell_without_a_function_whole():
    guarded = when_trained("x = 1\ny = 2")
    assert isinstance(ast.parse(guarded).body[0], ast.If)


def test_the_guarded_cell_runs_its_body_only_when_trained():
    guarded = when_trained("def f():\n    return 1\n\n\nresult = f()")
    shown: list = []
    namespace = {
        "TRAINED": True,
        "NOT_TRAINED": "skipped",
        "display": shown.append,
        "Markdown": str,
    }
    exec(guarded, namespace)  # noqa: S102 - the notebook's own generated source
    assert namespace["result"] == 1 and shown == []
    namespace = {
        "TRAINED": False,
        "NOT_TRAINED": "skipped",
        "display": shown.append,
        "Markdown": str,
    }
    exec(guarded, namespace)  # noqa: S102
    assert "result" not in namespace and shown == ["skipped"]


# ---------------------------------------------------------------- the test-set picks


def test_trial_picks_spread_over_the_sign_patterns_sorted_by_the_chosen_latent():
    """On the 3^k lattice sorted by z0 the picks run from its smallest value to its largest and
    are distinct sign patterns."""
    ns: dict = {"np": np, "torch": torch}
    for source in (sp.DATA_KERNEL, sp.TRIALS):
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef):
                exec(ast.unparse(node), ns)  # noqa: S102 - the notebook's own function
    k, a = 2, 1.0
    cfg = DataConfig(task="linear_teacher", P=4, D_in=3, n_latents=k, latent_min=-a, latent_max=a)
    maps = torch.eye(3, k, dtype=torch.float64)
    reference = Dataset(
        x=torch.zeros(4, 3, dtype=torch.float64),
        y=torch.zeros(4, 3, dtype=torch.float64),
        Q=None,
        period=None,
        factors={},
        groups={},
        extras={"W_X": maps, "W_Y": maps},
    )
    test = latent_lattice_dataset(cfg, reference)
    order = ns["sample_order"](test.factors, "z0", len(test.x))
    picks = ns["spread_trials"](order, 4)
    z = torch.stack([test.factors["z0"], test.factors["z1"]], dim=1)[picks]
    assert len(set(map(tuple, z.tolist()))) == 4  # four different sign patterns
    assert z[0, 0] == -a and z[-1, 0] == a  # smallest to largest z0
    assert bool((z[1:, 0] >= z[:-1, 0]).all())  # in the order of the sorting latent


# ---------------------------------------------------------------- execution


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    """spatial_tiny (net.N in {4, 8}) with only N = 8 trained."""
    root = tmp_path_factory.mktemp("exp_nb")
    exp = preset("spatial_tiny")
    save_exp(exp, root)
    run_setting(exp, 1, root, log=lambda _: None)
    return root


def _execute(root: Path, setting: dict):
    return execute_notebook(build_notebook(), root, {"SETTING": setting}, root_name="ROOT")


def test_executes_on_a_trained_setting_and_draws_every_figure(root):
    nb = _execute(root, {"net.N": 8})
    text = _text_outputs(nb)
    assert "ERROR" not in text
    assert _n_figures_in(nb) == N_FIGURES
    assert "| 0 | 4 | missing |" in text and "| 1 | 8 | done |" in text  # the settings table
    assert "**Setting 1** (net.N = 8)" in text
    assert "Test set: 9 latent-lattice points" in text  # 3^k, k = 2
    assert "test set: 9 points, data kernels A of shape (9, 9) per layer" in text
    for i, cell in enumerate(nb.cells):
        if any("image/png" in o.get("data", {}) for o in cell.get("outputs", [])):
            assert "$" in nb.cells[i - 1].source  # an equation introduces every figure


def test_a_setting_that_is_not_trained_is_reported_and_only_its_prior_is_drawn(root):
    nb = _execute(root, {"net.N": 4})
    text = _text_outputs(nb)
    assert "ERROR" not in text
    assert "is not trained** (status: missing)" in text
    assert text.count("is not trained") == 1 + len(RUN_CELLS)  # the loader and every skipped cell
    assert _n_figures_in(nb) == N_PRIOR_FIGURES  # the prior kernels need no trained net
    assert "SGLD needs < 1" in text  # the floor tables are there


def test_a_setting_outside_the_experiment_fails_loudly(root):
    with pytest.raises(CellExecutionError, match="net.N"):
        _execute(root, {"net.N": 5})


def test_the_committed_notebook_is_what_the_builder_produces():
    committed = nbformat.read(Path(__file__).parents[1] / "notebooks" / "spatial_exp.ipynb", 4)
    assert [(c.cell_type, c.source) for c in committed.cells] == [
        (c.cell_type, c.source) for c in build_notebook().cells
    ]
