"""The notebook builder: structure of the source, and an end-to-end execution on a partial tiny
sweep (most settings missing) that must render a message for each missing setting."""

import nbformat
import pytest

from fct.sweep import SWEEP_TASKS, expand, preset, run_setting, save_sweep
from notebooks.build_modularity import (
    build_notebook,
    execute_notebook,
    with_root,
    write_notebook,
)

QUIET = {"log": lambda s: None}
N_SETTINGS = 36


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


def test_each_setting_cell_names_one_distinct_grid_setting():
    """Cells select settings by name; together they name exactly the grid."""
    named = {
        line.split("find_result(RESULTS, ")[1].split("))")[0]
        for line in _code(build_notebook()).splitlines()
        if "find_result(" in line
    }
    expected = {f'"{s.task}", {s.L}, "{s.phi}", {s.rate_lambda!r}' for s in expand(preset("tiny"))}
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
    per_done_setting = 3  # loss curves, example outputs, MIR history
    assert png == per_done_setting + len(SWEEP_TASKS)  # plus one heatmap figure per task


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
