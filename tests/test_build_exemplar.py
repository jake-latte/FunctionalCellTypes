"""The exemplar notebook: structure of the source, the parameter rewriting, and an end-to-end
execution on a tiny configuration with each sampler (executed copies stay in pytest's temp dir)."""

from pathlib import Path

import nbformat
import pytest
from nbformat.v4 import new_code_cell, new_notebook

from notebooks.build_exemplar import build_notebook, execute_notebook, set_parameters

TINY = {
    "N_EPOCHS": 300,
    "TRAIN_LOG_INT": 100,
    "N_OUTER": 6,
    "PLOT_EVERY": 2,
    "N_CHAINS": 32,
    "WARMUP_TRANS": 10,
    "N_IMPORTANCE_SAMPLES": 1024,
}
# loss curves, example outputs and empirical kernels; the final live frame (kernels, solver
# health, sampler health), which replaces earlier frames; the kernel comparison
EXPECTED_FIGURES = 3 + 3 + 1


def _code(nb) -> str:
    return "\n".join(c.source for c in nb.cells if c.cell_type == "code")


def _outputs(nb):
    return [out for cell in nb.cells for out in cell.get("outputs", [])]


def _text(nb) -> str:
    chunks = []
    for out in _outputs(nb):
        if out.output_type == "error":
            chunks.append(f"ERROR {out.ename}: {out.evalue}")
        elif out.output_type == "display_data":
            chunks.append(out.data.get("text/markdown", ""))
        elif out.output_type == "stream":
            chunks.append(out.text)
    return "\n".join(chunks)


def _n_figures(nb) -> int:
    return sum("image/png" in out.get("data", {}) for out in _outputs(nb))


def test_source_notebook_validates_has_no_outputs_and_no_path():
    nb = build_notebook()
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert all(not c.outputs and c.execution_count is None for c in code)
    assert "/ceph" not in _code(nb) and "/nfs" not in _code(nb)
    parameter = [c for c in nb.cells if "parameters" in c.metadata.get("tags", [])]
    assert len(parameter) == 1 and "CACHE_ROOT" in parameter[0].source


def test_every_task_and_net_setting_is_in_one_cell_and_each_stage_has_its_parameters():
    nb = build_notebook()
    task = next(c for c in nb.cells if "parameters" in c.metadata.get("tags", [])).source
    for name in (
        "P =", "M_IN =", "K_LATENTS =", "N =", "L =", "PHI =", "G =", "DATA_SEED =", "NET_SEED =",
        "WEIGHT_LAMBDA =", "RATE_LAMBDA =",
    ):  # fmt: skip
        assert name in task
    code = _code(nb)
    for name in ("OPTIMIZER =", "BETA =", "LR =", "N_EPOCHS =", "SAMPLER =", "N_OUTER =",
                 "N_CHAINS =", "N_IMPORTANCE_SAMPLES =", "PLOT_EVERY ="):  # fmt: skip
        assert code.count(name) == 1, name


def test_the_text_says_why_the_sizes_are_small_and_what_importance_sampling_needs():
    intro = build_notebook().cells[0].source
    assert "InadequateSamplerError" in intro and "small-$P$" in intro


def test_set_parameters_rewrites_one_assignment_and_keeps_its_comment():
    nb = build_notebook()
    out = set_parameters(nb, {"SAMPLER": "importance", "N_OUTER": 3})
    code = _code(out)
    assert "SAMPLER = 'importance'  # " in code and "N_OUTER = 3  # " in code
    assert 'SAMPLER = "hmc"' in _code(nb)  # the source is untouched


def test_set_parameters_rejects_unknown_and_ambiguous_names():
    nb = build_notebook()
    with pytest.raises(ValueError, match="N_OUTRE.*0 times"):
        set_parameters(nb, {"N_OUTRE": 3})
    twice = new_notebook(cells=[new_code_cell("X = 1"), new_code_cell("X = 2")])
    with pytest.raises(ValueError, match="X.*2 times"):
        set_parameters(twice, {"X": 3})


@pytest.mark.parametrize("sampler", ["hmc", "importance"])
def test_executes_end_to_end_with_each_sampler(sampler, tmp_path):
    nb = execute_notebook(build_notebook(), tmp_path / "runs", {**TINY, "SAMPLER": sampler})
    text = _text(nb)
    assert "ERROR" not in text
    assert f"with the {sampler} sampler" in text
    assert _n_figures(nb) == EXPECTED_FIGURES
    # the live view keeps the last log lines under the final frame
    assert "[0005]" in text
    # a second execution reloads the cached training and solve and shows the same frames
    again = execute_notebook(build_notebook(), tmp_path / "runs", {**TINY, "SAMPLER": sampler})
    assert "ERROR" not in _text(again) and _n_figures(again) == EXPECTED_FIGURES


def test_the_committed_notebook_is_what_the_builder_produces():
    """The source notebook is generated; a stale copy would call an outdated API."""
    committed = nbformat.read(Path(__file__).parents[1] / "notebooks" / "exemplar.ipynb", 4)
    assert [(c.cell_type, c.source) for c in committed.cells] == [
        (c.cell_type, c.source) for c in build_notebook().cells
    ]
