"""Build and execute `phase_transition.ipynb`, the results notebook of a phase-transition sweep.

One viewer for a sweep made with `--preset phase_transition`: pick a setting by (L, phi,
lambda_r, lambda_w) and see its loss curves, example outputs, MIR over training and the tuning
curves of its neurons; then the MIR of the whole grid as one heat-map per hidden layer and as a
scatter against lambda_r / lambda_w. All logic is in `fct.grid_view` and `fct.report`; each cell
only calls it. Run

    python notebooks/build_phase_transition.py ROOT            # execute, write tmp/phase_transition.ipynb
    python notebooks/build_phase_transition.py --no-execute    # source only: notebooks/phase_transition.ipynb

ROOT is the sweep directory (the one holding sweep.json); `$FCT_SWEEP_ROOT` works too. Only the
source notebook is committed; an executed one records the ROOT path and its figures are large, so it
goes to the git-ignored tmp/ unless --out says otherwise.
"""

import argparse
import os
from pathlib import Path

from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from fct.explanations import MIR_EXPLANATION
from fct.report import ROOT_ENV
from notebooks.build_modularity import execute_notebook, write_notebook

SOURCE_OUT = Path(__file__).with_name("phase_transition.ipynb")
EXECUTED_OUT = Path(__file__).parents[1] / "tmp" / "phase_transition.ipynb"

INTRO = """# Phase transition in modularity: one setting, then the whole grid

Set the sweep ROOT in the first code cell, then choose a setting in the selector cell. Requested
values are snapped to the nearest grid value (lambdas on a log scale) and the value used is
printed. A setting still running is shown from its latest checkpoint and labelled
"provisional"; a setting with no checkpoint gets a message and no plots. The last two figures
show every network of one depth and activation: layer-average MIR as a (lambda_r, lambda_w)
matrix per hidden layer, and against lambda_r / lambda_w. `MIR_KIND` picks the
shuffle-bias-corrected MIR (default) or the raw one for the tuning order and both global figures."""

PARAMETERS = """ROOT = None  # the sweep ROOT (holds sweep.json); None reads $FCT_SWEEP_ROOT"""

SETUP = """import matplotlib as mpl

from fct.grid_view import (
    DEPTH_AXIS,
    PHI_AXIS,
    choose_setting,
    matrix_items,
    ratio_items,
    snap_text,
    tuning_items,
)
from fct.report import load_results, overview_text, setting_items, show, sweep_root_from_env

mpl.rcParams["figure.dpi"] = 100
ROOT = ROOT or sweep_root_from_env()
RESULTS = load_results(ROOT)
show([overview_text(RESULTS)])"""

SELECTOR = """L = 1  # hidden layers
PHI = "tanh"  # activation ("tanh", "relu", "sigmoid" or "id"), exactly as in the sweep
LAMBDA_R = 1e-3  # rate regulariser, snapped to the nearest grid value
LAMBDA_W = 1e-3  # weight regulariser, snapped to the nearest grid value
MIR_KIND = "corrected"  # "corrected" (shuffle-bias-corrected) or "raw"; used below

CHOICE = choose_setting(RESULTS, L=L, phi=PHI, lambda_r=LAMBDA_R, lambda_w=LAMBDA_W)
show([snap_text(CHOICE)])"""

SETTING_PLOTS = """show(setting_items(CHOICE.result))"""

TUNING = """N_PLOT_NEURONS = 6
PLOT_LAYER = 1  # hidden layer, counted from 1
NEURON_ORDER = "mir_descending"  # "random", "mir_ascending" or "mir_descending"
NEURON_SEED = 0  # only for "random"

show(
    tuning_items(ROOT, CHOICE, PLOT_LAYER - 1, N_PLOT_NEURONS, NEURON_ORDER, NEURON_SEED, MIR_KIND)
)"""

MATRIX = """# Depth and activation of the global figures; by default those of the chosen setting.
GLOBAL_L = CHOICE.snapped[DEPTH_AXIS].value
GLOBAL_PHI = CHOICE.snapped[PHI_AXIS].value

show(matrix_items(RESULTS, GLOBAL_L, GLOBAL_PHI, MIR_KIND))"""

RATIO = """show(ratio_items(RESULTS, GLOBAL_L, GLOBAL_PHI, MIR_KIND))"""


def build_notebook() -> NotebookNode:
    """The notebook without outputs."""
    nb = new_notebook(
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        }
    )
    cells = [
        new_markdown_cell(INTRO),
        new_code_cell(PARAMETERS, metadata={"tags": ["parameters"]}),
        new_code_cell(SETUP),
        new_markdown_cell(MIR_EXPLANATION),
        new_markdown_cell("## Choose a setting"),
        new_code_cell(SELECTOR),
        new_markdown_cell("## The chosen setting: loss, example outputs, MIR over training"),
        new_code_cell(SETTING_PLOTS),
        new_markdown_cell("## Tuning of the neurons of one layer"),
        new_code_cell(TUNING),
        new_markdown_cell("## MIR over the (lambda_r, lambda_w) grid, one matrix per layer"),
        new_code_cell(MATRIX),
        new_markdown_cell("## MIR against lambda_r / lambda_w, one dot per network"),
        new_code_cell(RATIO),
    ]
    # Fixed ids, so rebuilding the notebook does not churn the diff.
    for i, cell in enumerate(cells, start=1):
        cell.id = f"cell-{i:03d}"
    nb.cells = cells
    return nb


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", nargs="?", type=Path, help=f"sweep ROOT (default ${ROOT_ENV})")
    parser.add_argument(
        "--out", type=Path, help="default: the source notebook, or tmp/ if executed"
    )
    parser.add_argument("--no-execute", action="store_true", help="write the source only")
    args = parser.parse_args(argv)
    nb = build_notebook()
    if not args.no_execute:
        root = args.root or os.environ.get(ROOT_ENV)
        if not root:
            parser.error(f"give a sweep ROOT or set ${ROOT_ENV}")
        nb = execute_notebook(nb, Path(root))
    out = args.out or (SOURCE_OUT if args.no_execute else EXECUTED_OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"wrote {write_notebook(nb, out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
