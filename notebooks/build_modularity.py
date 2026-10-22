"""Build and execute `modularity.ipynb`, the results notebook of a sweep.

The notebook source lives here, not in the .ipynb: `build_notebook` makes it from a few lines per
cell, each of which only calls a function of `fct.report` (where the logic is tested). Run

    python notebooks/build_modularity.py ROOT            # build, execute, write tmp/modularity.ipynb
    python notebooks/build_modularity.py --no-execute    # source only: notebooks/modularity.ipynb

ROOT is the sweep directory (the one holding sweep.json); `$FCT_SWEEP_ROOT` works too. Only the
source notebook is committed; an executed one records the ROOT path and its figures are large, so it
goes to the git-ignored tmp/ unless --out says otherwise.
"""

import argparse
import copy
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from fct.explanations import MIR_EXPLANATION
from fct.report import ROOT_ENV
from fct.sweep import MODULARITY_AXES, SWEEP_TASKS, expand, preset

SOURCE_OUT = Path(__file__).with_name("modularity.ipynb")
EXECUTED_OUT = Path(__file__).parents[1] / "tmp" / "modularity.ipynb"
EXECUTE_TIMEOUT_S = 1800

INTRO = """# Modularity of hidden layers across the sweep

Per task and setting: loss curves, example outputs of the trained net (of the latest checkpoint
for a setting still running), MIR per layer over training (raw with its chance baseline,
shuffle-corrected, dead-neuron counts) and prediction metrics. A summary table and heatmap of the
final MIR per layer open each task. All numbers and sentences below are computed from the sweep
ROOT named in the parameter cell. A setting that is still running is shown from its latest
checkpoint and labelled "partial, epoch X of N"; its numbers are provisional (marked `*` in the
table, hatched in the heatmap) and are left out of the summary statistics. A setting with no
checkpoint, or whose checkpoint or dataset file is unreadable, stale or missing, is listed with a
message instead of being drawn."""

PARAMETERS = """ROOT = None  # the sweep ROOT (holds sweep.json); None reads $FCT_SWEEP_ROOT"""

SETUP = """import matplotlib as mpl

from fct.report import (
    find_result,
    load_results,
    overview_text,
    setting_items,
    show,
    sweep_root_from_env,
    task_items,
)

mpl.rcParams["figure.dpi"] = 100  # about 1300 px for a 13 in figure
RESULTS = load_results(ROOT or sweep_root_from_env())
show([overview_text(RESULTS)])"""


class _Ids:
    """Deterministic cell ids, so rebuilding the notebook does not churn the diff."""

    def __init__(self) -> None:
        self.n = 0

    def next(self) -> str:
        self.n += 1
        return f"cell-{self.n:03d}"


def build_notebook() -> NotebookNode:
    """The notebook without outputs: one summary cell per task, then one cell per setting."""
    ids = _Ids()
    nb = new_notebook(
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        }
    )
    nb.cells = [
        new_markdown_cell(INTRO, id=ids.next()),
        new_code_cell(PARAMETERS, id=ids.next(), metadata={"tags": ["parameters"]}),
        new_code_cell(SETUP, id=ids.next()),
        new_markdown_cell(MIR_EXPLANATION, id=ids.next()),
    ]
    # The grid does not depend on the sweep's sizes, so any preset gives the same order.
    settings = expand(preset("tiny"))
    for task in SWEEP_TASKS:
        nb.cells.append(new_markdown_cell(f"## {task}", id=ids.next()))
        nb.cells.append(new_markdown_cell("### Summary of the final MIR", id=ids.next()))
        nb.cells.append(new_code_cell(f'show(task_items(RESULTS, "{task}"))', id=ids.next()))
        for setting in settings:
            if setting.task != task:
                continue
            L, phi, rate = (setting.axes[a] for a in MODULARITY_AXES)
            title = f"### L{L}, {phi}, rate_lambda {rate:g}"
            nb.cells.append(new_markdown_cell(title, id=ids.next()))
            # Selected by name, not position: a reordered grid fails here instead of
            # silently showing another setting under this heading.
            axes = dict(zip(MODULARITY_AXES, (L, phi, rate), strict=True))
            select = f'find_result(RESULTS, "{task}", {axes!r})'
            nb.cells.append(new_code_cell(f"show(setting_items({select}))", id=ids.next()))
    return nb


def with_root(nb: NotebookNode, root: Path) -> NotebookNode:
    """A deep copy of `nb` whose parameter cell sets ROOT to `root`; `nb` is left unchanged."""
    out = copy.deepcopy(nb)
    (cell,) = [c for c in out.cells if "parameters" in c.metadata.get("tags", [])]
    cell.source = f"ROOT = {str(root)!r}"
    return out


def execute_notebook(nb: NotebookNode, root: Path) -> NotebookNode:
    """Run a copy of `nb` in a fresh kernel with ROOT injected into the parameter cell; the
    source `nb` gains no outputs. Timing metadata is off so equal runs give equal notebooks.
    A failing cell raises (nbclient.exceptions.CellExecutionError)."""
    client = NotebookClient(
        with_root(nb, root),
        timeout=EXECUTE_TIMEOUT_S,
        kernel_name="python3",
        record_timing=False,
    )
    return client.execute()


def write_notebook(nb: NotebookNode, path: Path) -> Path:
    path = Path(path)
    nbformat.validate(nb)
    nbformat.write(nb, path)
    return path


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
