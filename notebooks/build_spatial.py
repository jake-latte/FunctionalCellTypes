"""Build and execute `spatial.ipynb`: train a net under a spatial weight prior and look at it.

One linear-teacher run. The first cells set the task, the network, the training and the spatial
prior (an embedding and a kernel per population); the notebook then plots those kernels, trains
through the normal `fct` cache (a re-run reloads or resumes), and shows the loss curves, example
outputs, the first-layer weights over the embedding and the tuning of every neuron to the
latents, one panel per neuron placed by its position. All logic is in `fct.spatial_view` and
`fct.spatial_report`; each cell only calls it. Run

    python notebooks/build_spatial.py CACHE_ROOT                  # execute, write tmp/spatial.ipynb
    python notebooks/build_spatial.py CACHE_ROOT --set OPTIMIZER=adam --set N_EPOCHS=200
    python notebooks/build_spatial.py --no-execute                # source only: notebooks/spatial.ipynb

CACHE_ROOT holds the cached runs (`$FCT_CACHE_ROOT` works in the notebook too). Only the source
notebook is committed; an executed one records the CACHE_ROOT path and its figures are large, so
it goes to the git-ignored tmp/ unless --out says otherwise.
"""

import argparse
import ast
import copy
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nbclient import NotebookClient
from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from notebooks.build_modularity import EXECUTE_TIMEOUT_S, write_notebook

SOURCE_OUT = Path(__file__).with_name("spatial.ipynb")
EXECUTED_OUT = Path(__file__).parents[1] / "tmp" / "spatial.ipynb"

INTRO = """# A spatial weight prior: kernels, training, and the tuning of every cell

Every unit of the input, the hidden layer(s) and the output sits at a position on a ring (or a
torus); the weight prior of each layer is a Gaussian kernel over those positions, so units that
are close in space get similar weights. The notebook sets the prior, trains a net under it on a
linear teacher ($x = Z W_X^\\top$, $y = Z W_Y^\\top$, $k$ latents $Z$), and looks at what
training produced: the tuning of each neuron to each latent, laid out by the neuron's position.

Set `CACHE_ROOT` in the first cell. The run is cached under it in a directory named by a hash of
the task, net, training and spatial settings, so re-running reloads a finished run, resumes an
interrupted one, and changing any setting starts a new one.

**Why the default sizes are small.** The prior's penalty matrix is the inverse of the kernel with
its eigenvalues floored at `MIN_REL_EIG` times the largest. A smooth kernel has eigenvalues down to
round-off, so the library default floor of 1e-6 gives a prior term of order 1e9 at the start, and
SGLD refuses to run unless `LR * WEIGHT_LAMBDA * lambda_max(M) < 1` for the stiffest direction of
the prior. With 16 hidden units on a ring, kernel length scales of 0.15 (about 2.4 neighbour
spacings) and a floor of 1e-2, that product is about 0.01 at `LR = 1e-2`, both optimisers are stable,
and 2000 epochs take seconds on a CPU. SGLD takes plain gradient steps plus noise, so in the same
number of epochs it fits the teacher less closely than Adam does. A larger network or a lower floor needs a smaller `LR`
(for SGLD) or `WEIGHT_LAMBDA`."""

TASK_AND_NET = """CACHE_ROOT = None  # directory for the cached runs; None reads $FCT_CACHE_ROOT
# task: a linear teacher with m features and k latents
P = 256  # samples
M_IN = 12  # m = D_in: features of the input x and of the output y
K_LATENTS = 3  # k latents, fewer than m
DATA_SEED = 0
# network
N = 16  # hidden units per layer
L = 1  # hidden layers
PHI = "tanh"  # "tanh", "relu" or "sigmoid"
G = 1.0  # hidden-to-hidden gain
NET_SEED = 0
# loss weights
WEIGHT_LAMBDA = 3e-3  # multiplies the spatial weight penalty
RATE_LAMBDA = 1e-4  # multiplies the rate penalty
DEVICE = "cpu"  # "cpu", "cuda" or "auto"; these nets are tiny"""

TRAINING = """OPTIMIZER = "sgld"  # "sgld" (Langevin noise of variance 2 * LR / BETA per step) or "adam"
BETA = 1e4  # inverse temperature, read by SGLD only
LR = 1e-2
N_EPOCHS = 2000
TRAIN_SEED = 0"""

SPATIAL = """from fct.spatial import SpatialConfig
from fct.spatial_view import population_spec

D = 1  # embedding dimension: a ring for 1, a torus for 2 (a grid then needs a shape, d <= 2)
BOX = 1.0  # side of the torus; length scales are in the same units
MIN_REL_EIG = 1e-2  # eigenvalue floor of the penalty matrices, relative to the largest
# one embedding and kernel per population: "grid" (evenly spaced) or "random" (uniform)
INPUT = population_spec("grid", length_scale=0.15, amplitude=1.0)
HIDDEN = population_spec("grid", length_scale=0.15, amplitude=1.0)  # shared by all hidden layers
OUTPUT = population_spec("grid", length_scale=0.15, amplitude=1.0)

SPATIAL = SpatialConfig(
    d=D, box=BOX, input=INPUT, hidden=HIDDEN, output=OUTPUT, min_rel_eig=MIN_REL_EIG
)
CFG = make_config(
    P=P,
    m=M_IN,
    k=K_LATENTS,
    N=N,
    L=L,
    phi=PHI,
    g=G,
    data_seed=DATA_SEED,
    net_seed=NET_SEED,
    weight_lambda=WEIGHT_LAMBDA,
    rate_lambda=RATE_LAMBDA,
    optimizer=OPTIMIZER,
    beta=BETA,
    lr=LR,
    n_epochs=N_EPOCHS,
    train_seed=TRAIN_SEED,
    spatial=SPATIAL,
    device=DEVICE,
)
SPACES = layer_spaces(CFG)
show(kernel_items(SPACES))"""

SETUP = """import matplotlib as mpl

from fct.report import show
from fct.spatial_report import (
    kernel_items,
    training_items,
    tuning_by_position_items,
    weight_items,
)
from fct.spatial_view import cache_root_from_env, layer_spaces, make_config, train_run

mpl.rcParams["figure.dpi"] = 100
RUNS_ROOT = CACHE_ROOT or cache_root_from_env()"""

TRAIN = """RUN = train_run(CFG, RUNS_ROOT, log=print)
show([f"Run cached in `{RUN.cache_dir.name}`."])"""

RESULTS = """show(training_items(RUN, CFG))"""

WEIGHTS = """show(weight_items(RUN.net, SPACES))"""

TUNING = """N_BINS = 8  # rank bins of each latent and of the rates, as in the MIR
N_SHUFFLES = 5  # shuffles for the bias-corrected MIR in each title
MIR_KIND = "corrected"  # "corrected" or "raw"
N_COLS = 4  # panels per row on a ring; a plane lattice uses its own rows and columns

show(tuning_by_position_items(RUN, SPACES, N_BINS, MIR_KIND, N_SHUFFLES, TRAIN_SEED, N_COLS))"""


def build_notebook() -> NotebookNode:
    """The notebook without outputs."""
    nb = new_notebook(
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        }
    )
    cells = [
        new_markdown_cell(INTRO),
        new_markdown_cell("## Task and network"),
        new_code_cell(TASK_AND_NET, metadata={"tags": ["parameters"]}),
        new_markdown_cell("## Training"),
        new_code_cell(TRAINING),
        new_markdown_cell(
            "## Spatial prior\n\nEmbedding and kernel of each population, then the kernels and "
            "positions of every weight layer (`K_post (x) K_pre`)."
        ),
        new_code_cell(SETUP),
        new_code_cell(SPATIAL),
        new_markdown_cell("## Train"),
        new_code_cell(TRAIN),
        new_markdown_cell("## Loss, and the trained network's outputs"),
        new_code_cell(RESULTS),
        new_markdown_cell("## First-layer weights over the embedding"),
        new_code_cell(WEIGHTS),
        new_markdown_cell(
            "## Tuning of every neuron to the latents\n\nOne panel per neuron, placed by its "
            "position in space (ring order along the rows for d = 1, its lattice cell for d = 2), "
            "one curve per latent: the neuron's mean rate against that latent, the other latents "
            "averaged out."
        ),
        new_code_cell(TUNING),
    ]
    # Fixed ids, so rebuilding the notebook does not churn the diff.
    for i, cell in enumerate(cells, start=1):
        cell.id = f"cell-{i:03d}"
    nb.cells = cells
    return nb


def _assignment(name: str) -> re.Pattern[str]:
    """`NAME = value  # comment` on one line: the value is group 2, the comment group 3."""
    return re.compile(rf"^({re.escape(name)} = )([^#\n]*?)(\s+#.*)?$", re.MULTILINE)


def set_parameters(nb: NotebookNode, values: Mapping[str, Any]) -> NotebookNode:
    """A deep copy of `nb` with the one-line assignment `NAME = ...` of each parameter replaced
    by `NAME = repr(value)`, keeping its trailing comment; `nb` is left unchanged. A name that
    is not assigned exactly once is an error, so a typo cannot silently run the defaults."""
    out = copy.deepcopy(nb)
    code = [c for c in out.cells if c.cell_type == "code"]
    for name, value in values.items():
        pattern = _assignment(name)
        hits = [c for c in code if pattern.search(c.source)]
        n_matches = sum(len(pattern.findall(c.source)) for c in hits)
        if n_matches != 1:
            raise ValueError(f"parameter {name} is assigned {n_matches} times, expected once")
        (cell,) = hits
        cell.source = pattern.sub(
            lambda m, v=value: f"{m[1]}{v!r}{m[3] or ''}", cell.source, count=1
        )
    return out


def execute_notebook(
    nb: NotebookNode, cache_root: Path, overrides: Mapping[str, Any] | None = None
) -> NotebookNode:
    """Run a copy of `nb` in a fresh kernel with CACHE_ROOT and the `overrides` set; the source
    `nb` gains no outputs. A failing cell raises (nbclient.exceptions.CellExecutionError)."""
    values = {"CACHE_ROOT": str(cache_root), **(overrides or {})}
    client = NotebookClient(
        set_parameters(nb, values),
        timeout=EXECUTE_TIMEOUT_S,
        kernel_name="python3",
        record_timing=False,
    )
    return client.execute()


def _parse_override(text: str) -> tuple[str, Any]:
    name, sep, literal = text.partition("=")
    if not sep or not name:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {text!r}")
    try:
        return name, ast.literal_eval(literal)
    except (ValueError, SyntaxError) as err:
        raise argparse.ArgumentTypeError(f"{literal!r} is not a Python literal") from err


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cache_root", nargs="?", type=Path, help="directory for the cached runs")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        type=_parse_override,
        metavar="NAME=VALUE",
        help="set a parameter of the notebook (a Python literal), e.g. OPTIMIZER='adam'",
    )
    parser.add_argument("--out", type=Path, help="default: the source notebook, or tmp/ if run")
    parser.add_argument("--no-execute", action="store_true", help="write the source only")
    args = parser.parse_args(argv)
    nb = build_notebook()
    if not args.no_execute:
        if args.cache_root is None:
            parser.error("give a CACHE_ROOT")
        nb = execute_notebook(nb, args.cache_root, dict(args.overrides))
    out = args.out or (SOURCE_OUT if args.no_execute else EXECUTED_OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"wrote {write_notebook(nb, out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
