"""Build and execute `spatial_grid.ipynb`: every figure of `spatial.ipynb` for one setting of a
spatial sweep.

Give a sweep ROOT (made with `python -m fct sweep init ROOT --preset spatial_n_scan`, or any sweep
with a spatial section on a linear teacher) and the axis values of one setting, `SETTING =
{"net.N": 256}`. The notebook lists the grid, finds that setting through `fct.sweep.find_setting_dir`
and loads its trained net and training set. It then shows the figures of the spatial notebook (the
prior kernels, the eigenvalue floor, the loss curves, example outputs, the first-layer kernels and
weights, the data kernels A next to the latent kernel B, the chosen trials' latents, the Phi pair
grids, the tuning of every neuron and the information and rate overlaps). The Phi kernels and the
example outputs come from the held-out latent-lattice test set (3^k points, one per sign pattern of
the latents), the tuning and the loss curves from the training samples and history. A setting that
is not trained is reported in the cells and only its prior is drawn.

The cells are those of `build_spatial.py` (imported, so the two notebooks cannot drift) with
the training and configuration cells replaced by a loader. Run

    python notebooks/build_spatial_grid.py ROOT --set 'SETTING={"net.N": 256}'   # execute, write tmp/spatial_grid.ipynb
    python notebooks/build_spatial_grid.py --no-execute                           # source only: notebooks/spatial_grid.ipynb

ROOT is the sweep directory (the one holding sweep.json); `$FCT_SWEEP_ROOT` works in the notebook
too. Only the source notebook is committed; an executed one records the ROOT path and its figures
are large, so it goes to the git-ignored tmp/ unless --out says otherwise.
"""

import ast
import textwrap
from pathlib import Path

from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from notebooks import build_spatial as sp

SOURCE_OUT = Path(__file__).with_name("spatial_grid.ipynb")
EXECUTED_OUT = Path(__file__).parents[1] / "tmp" / "spatial_grid.ipynb"

INTRO = (
    r"""# The spatial grid: every figure of the spatial notebook, for one setting

`ROOT` is a sweep over spatially embedded nets (`python -m fct sweep init ROOT --preset
spatial_n_scan`, say) and `SETTING` the axis values of one of its settings, e.g. `{"net.N": 256}`; a
subset of the axes is enough when it picks one setting. The notebook lists the grid, loads that
setting's trained net and shows what `spatial.ipynb` shows for a net it trains itself: the prior,
the loss curves, example outputs, the first-layer weights and the spatially indexed kernels
$\Phi^{(l)}$, and the tuning of every neuron.

Two things differ from the spatial notebook. The kernels $\Phi^{(l)}$ (the data kernels $A^{(l)}$,
the chosen trials and the pair grids) and the example outputs are computed on the **test set**: the
$3^k$ points of the latent lattice $\{-a, 0, a\}^k$, one per sign pattern of the $k$ latents, with
inputs and targets from the training samples' teacher, so every pair of patterns appears exactly
once and none was trained on. The loss curves are the training history, and the tuning and
information overlaps use the training samples, as in the spatial notebook. The readout is
$\phi(h^{(L)})$ and $\psi^{(l)} = \phi(h^{(l)})$ on every hidden layer.

A setting of the grid that was never trained (or is only partly trained) is reported in its cells
and only its prior is drawn.

"""
    + sp.NOTATION
)

PARAMETERS = """ROOT = None  # the sweep ROOT (holds sweep.json); None reads $FCT_SWEEP_ROOT
SETTING = {"net.N": 8}  # axis values of the setting to show: a subset of the grid's axes that picks one
DEVICE = "cpu"  # "cpu", "cuda" or "auto": where the net and the test activities are computed"""

GRID_IMPORTS = """from fct.report import sweep_root_from_env
from fct.spatial_grid import load_grid_run, select_setting, settings_table
from fct.train import prior_coefficient

SWEEP_ROOT = ROOT or sweep_root_from_env()"""

GRID_TEXT = """## The grid

Every setting of the sweep with its axis values and the state of its training (`done`, `partial`,
`missing` or `stale`), as `python -m fct sweep status ROOT` prints it. Choose one with `SETTING`."""

GRID = """display(Markdown(settings_table(SWEEP_ROOT)))"""

SETTING_TEXT = """## The chosen setting

The setting's config gives the spatial prior (`SPACES`, as in the spatial notebook) whether or not
it is trained; its trained net, training samples, training history and test set are loaded only
when it is done. The notation and training text above tell the reader to set `PRIOR_COEFF`,
`DATA_SEED`, `RATE_LAMBDA`, `BETA` and `OPTIMIZER`; in this notebook they are read from the setting's
config instead."""

LOADER = r"""CHOSEN = select_setting(SWEEP_ROOT, SETTING, DEVICE)
CFG = CHOSEN.cfg
SPACES = layer_spaces(CFG)  # weight layer l = 1 .. L + 1 is SPACES[l - 1]
tc = CFG.train
# the names the spatial notebook's cells read
OPTIMIZER, BETA, LR, TRAIN_SEED = tc.optimizer, tc.beta, tc.lr, tc.seed
PRIOR_COEFF = prior_coefficient(tc)
D, BOX, MIN_REL_EIG = CFG.spatial.d, CFG.spatial.box, CFG.spatial.min_rel_eig
TRAINED = CHOSEN.status == "done"
where = ", ".join(f"{axis} = {value}" for axis, value in CHOSEN.setting.axes.items())
if TRAINED:
    RUN, TEST_DATASET = load_grid_run(CHOSEN)
    NOT_TRAINED = ""
    sampler = f"SGLD, beta = {tc.beta:g}" if tc.optimizer == "sgld" else "Adam"
    last = {key: RUN.history[key][-1] for key in ("epoch", "loss", "mse", "weight_l2", "rate_l2")}
    display(
        Markdown(
            f"**Setting {CHOSEN.setting.index}** ({where}), trained with {sampler}, lr = {tc.lr:g}, "
            f"{tc.n_epochs} epochs; run in `{RUN.cache_dir.name}`. At the last logged epoch "
            f"({last['epoch']}): loss {last['loss']:.4g}, mse {last['mse']:.4g}, weight penalty "
            f"{last['weight_l2']:.4g}, rate penalty {last['rate_l2']:.4g}. Test set: "
            f"{len(TEST_DATASET.x)} latent-lattice points."
        )
    )
else:
    RUN = TEST_DATASET = None
    NOT_TRAINED = (
        f"**Setting {CHOSEN.setting.index} ({where}) is not trained** (status: {CHOSEN.status}): "
        "only its prior is drawn; the figures of the trained net are skipped."
    )
    display(Markdown(NOT_TRAINED))"""


def when_trained(source: str) -> str:
    """`source` with everything after its last function definition run only for a trained
    setting, else a note that the figure is skipped. The plotting functions stay at the top
    level of the cell, where a reader (and the tests) find them; a cell without a function
    is guarded whole."""
    tree = ast.parse(source)
    defs = [n.end_lineno for n in tree.body if isinstance(n, ast.FunctionDef)]
    head, rest = source.splitlines(), []
    if defs:
        head, rest = head[: max(defs)], head[max(defs) :]
    else:
        head, rest = [], head
    body = textwrap.indent("\n".join(rest).strip("\n"), "    ")
    guard = f"if TRAINED:\n{body}\nelse:\n    display(Markdown(NOT_TRAINED))"
    return "\n".join([*head, guard]) if head else guard


def build_notebook() -> NotebookNode:
    """The notebook without outputs: parameters, the grid, the chosen setting, then the
    figures of the spatial notebook (a markdown cell with the equation, a code cell with the
    plotting function)."""
    nb = new_notebook(
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        }
    )
    cells = [
        new_markdown_cell(INTRO),
        new_code_cell(PARAMETERS, metadata={"tags": ["parameters"]}),
        new_markdown_cell(sp.SETUP_TEXT),
        new_code_cell(sp.IMPORTS),
        new_code_cell(GRID_IMPORTS),
        new_markdown_cell(GRID_TEXT),
        new_code_cell(GRID),
        new_markdown_cell(SETTING_TEXT),
        new_code_cell(LOADER),
        new_markdown_cell(sp.PRIOR_TEXT),
        new_markdown_cell(sp.FLOOR_TEXT),
        new_code_cell(sp.FLOOR),
        new_markdown_cell(sp.KERNELS_TEXT),
        new_code_cell(sp.KERNELS),
        new_markdown_cell(sp.TRAINING_TEXT),
        new_markdown_cell(sp.LOSS_TEXT),
        new_code_cell(when_trained(sp.LOSS)),
        new_markdown_cell(sp.EXAMPLES_TEXT),
        new_code_cell(when_trained(sp.EXAMPLES)),
        new_markdown_cell(sp.EMPIRICAL_TEXT),
        new_code_cell(when_trained(sp.EMPIRICAL)),
        new_markdown_cell(sp.WEIGHTS_TEXT),
        new_code_cell(when_trained(sp.WEIGHTS)),
        new_markdown_cell(sp.DATA_KERNEL_TEXT),
        new_code_cell(when_trained(sp.DATA_KERNEL)),
        new_markdown_cell(sp.TRIALS_TEXT),
        new_code_cell(when_trained(sp.TRIALS)),
        new_markdown_cell(sp.PHI_PAIRS_TEXT),
        new_code_cell(when_trained(sp.PHI_PAIRS)),
        new_markdown_cell(sp.TUNING_TEXT),
        new_code_cell(when_trained(sp.TUNING)),
        new_markdown_cell(sp.OVERLAP_TEXT),
        new_code_cell(when_trained(sp.OVERLAP)),
    ]
    # Fixed ids, so rebuilding the notebook does not churn the diff.
    for i, cell in enumerate(cells, start=1):
        cell.id = f"cell-{i:03d}"
    nb.cells = cells
    return nb


def main(argv: list[str] | None = None) -> int:
    return sp.build_main(
        build_notebook,
        SOURCE_OUT,
        EXECUTED_OUT,
        __doc__.split("\n\n")[0],
        "'SETTING={\"net.N\": 256}'",
        argv,
        root_name="ROOT",
        root_help="the sweep ROOT (the directory holding sweep.json)",
    )


if __name__ == "__main__":
    raise SystemExit(main())
