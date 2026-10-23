"""The spatial-prior notebook: structure of the source, parameter injection, and an end-to-end
execution on a tiny configuration for each optimiser (CPU)."""

from pathlib import Path

import nbformat
import pytest

from notebooks.build_spatial import build_notebook, execute_notebook, set_parameters

# Hidden layers L = 1: kernel figures for its L + 1 = 2 weight layers, loss curves, example
# outputs, the two first-layer figures (empirical kernels, weights) and one tuning figure.
N_FIGURES = 2 + 1 + 1 + 2 + 1
TINY = {"P": 32, "M_IN": 6, "K_LATENTS": 2, "N": 8, "N_EPOCHS": 40}


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
    assert len(parameter) == 1 and "CACHE_ROOT" in parameter[0].source


def test_source_notebook_exposes_task_net_training_and_spatial_parameters():
    nb = build_notebook()
    cells = [c.source for c in nb.cells if c.cell_type == "code"]
    # task and net hyperparameters are one cell at the top
    for name in ("P =", "M_IN =", "K_LATENTS =", "N =", "L =", "PHI =", "G =", "WEIGHT_LAMBDA ="):
        assert name in cells[0]
    assert "OPTIMIZER =" in cells[1] and "BETA =" in cells[1] and "N_EPOCHS =" in cells[1]
    spatial = next(c for c in cells if "INPUT =" in c)
    for name in ("HIDDEN =", "OUTPUT =", "MIN_REL_EIG =", "D =", "population_spec", "kernel_items"):
        assert name in spatial
    assert '"sgld"' in cells[1] and '"adam"' in cells[1]


def test_set_parameters_replaces_the_value_and_keeps_the_comment():
    nb = build_notebook()
    before = nbformat.writes(nb)
    out = set_parameters(nb, {"N": 4, "OPTIMIZER": "adam"})
    assert nbformat.writes(nb) == before
    code = _code(out)
    assert "\nN = 4  # " in code and 'OPTIMIZER = "adam"' not in code
    assert "OPTIMIZER = 'adam'  # " in code


def test_set_parameters_fails_on_a_name_the_notebook_does_not_set():
    with pytest.raises(ValueError, match="NOT_A_PARAMETER"):
        set_parameters(build_notebook(), {"NOT_A_PARAMETER": 1})


@pytest.mark.parametrize("optimizer", ["adam", "sgld"])
def test_executes_on_a_tiny_configuration(tmp_path, optimizer):
    nb = execute_notebook(build_notebook(), tmp_path, {**TINY, "OPTIMIZER": optimizer})
    text = _text_outputs(nb)
    assert "ERROR" not in text
    assert _n_figures(nb) == N_FIGURES
    assert list(tmp_path.glob("spatial_*/train/final.pt"))  # trained through the normal cache
    assert (
        "Eigenvalue floor" in text
        and f"Trained with {'SGLD' if optimizer == 'sgld' else 'Adam'}" in text
    )


def test_the_committed_notebook_is_what_the_builder_produces():
    """The source notebook is generated; a stale copy would call an outdated API."""
    committed = nbformat.read(Path(__file__).parents[1] / "notebooks" / "spatial.ipynb", 4)
    assert [(c.cell_type, c.source) for c in committed.cells] == [
        (c.cell_type, c.source) for c in build_notebook().cells
    ]
