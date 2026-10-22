"""The notebook builder: structure of the source, and an end-to-end execution on a partial tiny
sweep (most settings missing) that must render a message for each missing setting."""

import ast

import nbformat
import pytest

from fct.sweep import SWEEP_TASKS, expand, preset, run_setting, save_sweep, setting_dir
from notebooks.build_modularity import (
    build_notebook,
    execute_notebook,
    with_root,
    write_notebook,
)
from tests.report_helpers import QUIET

N_SETTINGS = 36
FIGURES_PER_SETTING = 3  # loss curves, example outputs, MIR history


def _code(nb) -> str:
    return "\n".join(c.source for c in nb.cells if c.cell_type == "code")


def _show_cells(nb) -> list:
    return [c for c in nb.cells if c.cell_type == "code" and c.source.startswith("show(")]


def test_source_notebook_has_a_cell_per_setting_and_task_and_no_outputs():
    nb = build_notebook()
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert all(not c.outputs and c.execution_count is None for c in code)
    assert _code(nb).count("find_result(RESULTS,") == N_SETTINGS
    assert _code(nb).count("task_items(RESULTS,") == len(SWEEP_TASKS)


def test_explanation_cell_sits_between_the_setup_and_the_first_task():
    from fct.explanations import MIR_EXPLANATION

    nb = build_notebook()
    (index,) = [i for i, c in enumerate(nb.cells) if c.source == MIR_EXPLANATION]
    assert nb.cells[index].cell_type == "markdown"
    assert "RESULTS = load_results" in nb.cells[index - 1].source
    assert nb.cells[index + 1].source == f"## {SWEEP_TASKS[0]}"


def test_running_the_setup_cell_sets_a_legible_dpi(tmp_path):
    import matplotlib as mpl

    save_sweep(preset("tiny"), tmp_path)
    setup = next(c.source for c in build_notebook().cells if "load_results(" in c.source)
    with mpl.rc_context():  # restores the caller's rcParams afterwards
        exec(setup, {"ROOT": tmp_path})  # noqa: S102  (runs the notebook source itself)
        assert mpl.rcParams["figure.dpi"] == 100


def _named_setting(line: str) -> tuple:
    """(task, L, phi, rate_lambda) named by a cell `show(setting_items(find_result(...)))`."""
    find = ast.parse(line).body[0].value.args[0].args[0]
    task, axes = find.args[1], ast.literal_eval(find.args[2])
    return (ast.literal_eval(task), axes["net.L"], axes["net.phi"], axes["train.rate_lambda"])


def test_each_setting_cell_names_one_distinct_grid_setting():
    """Cells select settings by name; together they name exactly the grid."""
    named = {
        _named_setting(line)
        for line in _code(build_notebook()).splitlines()
        if "find_result(" in line
    }
    expected = {
        (s.task, s.axes["net.L"], s.axes["net.phi"], s.axes["train.rate_lambda"])
        for s in expand(preset("tiny"))
    }
    assert named == expected


def test_source_notebook_takes_the_root_from_one_parameter_cell_without_a_path():
    nb = build_notebook()
    parameter = [c for c in nb.cells if "parameters" in c.metadata.get("tags", [])]
    assert len(parameter) == 1 and "ROOT" in parameter[0].source
    assert "/ceph" not in _code(nb) and "/nfs" not in _code(nb)


def test_report_cells_are_a_single_show_call():
    """No computation in cells: apart from the parameter and setup cells, each code cell is
    one `show(...)` of a fct.report function."""
    nb = build_notebook()
    show_cells = _show_cells(nb)
    other = [c for c in nb.cells if c.cell_type == "code" and c not in show_cells]
    assert len(other) == 2  # the parameter cell and the setup cell
    assert len(show_cells) == len(SWEEP_TASKS) + N_SETTINGS
    assert all("\n" not in c.source for c in show_cells)


def test_with_root_sets_only_the_parameter_cell_and_leaves_the_source_alone(tmp_path):
    nb = build_notebook()
    before = nbformat.writes(nb)
    injected = with_root(nb, tmp_path)
    assert nbformat.writes(nb) == before
    changed = [a.id for a, b in zip(nb.cells, injected.cells, strict=True) if a.source != b.source]
    assert changed == [c.id for c in nb.cells if "parameters" in c.metadata.get("tags", [])]
    assert repr(str(tmp_path)) in injected.cells[1].source


@pytest.fixture(scope="module")
def executed(tmp_path_factory):
    root = tmp_path_factory.mktemp("sweep")
    sweep = preset("tiny")
    save_sweep(sweep, root)
    run_setting(sweep, 0, root, **QUIET)
    source = build_notebook()
    return source, execute_notebook(source, root), root


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


def test_executes_end_to_end_with_a_message_for_each_missing_setting(executed):
    _, nb, _ = executed
    text = _text_outputs(nb)
    assert "ERROR" not in text
    assert text.count(": missing**") == N_SETTINGS - 1  # one visible message per missing setting
    assert "1 done" in text and f"{N_SETTINGS - 1} missing" in text


def test_executed_notebook_holds_exactly_the_expected_figures(executed):
    _, nb, _ = executed
    png = sum(
        "image/png" in out.get("data", {}) for cell in nb.cells for out in cell.get("outputs", [])
    )
    assert png == FIGURES_PER_SETTING + len(SWEEP_TASKS)  # plus one heatmap figure per task


def test_executing_leaves_the_source_notebook_output_free(executed):
    source, _, _ = executed
    assert all(
        not c.outputs and c.execution_count is None for c in source.cells if c.cell_type == "code"
    )


def test_two_executions_of_the_same_root_agree_apart_from_image_bytes(executed):
    source, first, root = executed
    second = execute_notebook(source, root)

    def skeleton(nb):
        return [(c.id, c.cell_type, c.source, dict(c.metadata)) for c in nb.cells]

    assert skeleton(first) == skeleton(second)
    assert _text_outputs(first) == _text_outputs(second)


def test_write_notebook_round_trips(tmp_path, executed):
    _, nb, _ = executed
    path = write_notebook(nb, tmp_path / "out.ipynb")
    assert nbformat.read(path, as_version=4).cells[0].source == nb.cells[0].source


def test_executes_on_a_mix_of_done_partial_and_missing_settings(tmp_path):
    sweep = preset("tiny")
    save_sweep(sweep, tmp_path)
    for index in (0, 1):
        run_setting(sweep, index, tmp_path, **QUIET)
    # an interrupted run: its checkpoint stays, final.pt never appeared
    (setting_dir(tmp_path, expand(sweep)[1]) / "train" / "final.pt").unlink()
    nb = execute_notebook(build_notebook(), tmp_path)
    text = _text_outputs(nb)
    assert "ERROR" not in text
    assert f"1 done, 1 partial (provisional), {N_SETTINGS - 2} missing" in text
    assert "partial, epoch 4 of 6" in text and "1 provisional (still running)" in text
    png = sum(
        "image/png" in out.get("data", {}) for cell in nb.cells for out in cell.get("outputs", [])
    )
    assert png == 2 * FIGURES_PER_SETTING + len(SWEEP_TASKS)
