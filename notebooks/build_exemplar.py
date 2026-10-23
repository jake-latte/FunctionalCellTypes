"""Build and execute `exemplar.ipynb`: a task, SGLD training, and the mean-field solve with live
diagnostics.

One linear-teacher run end to end. The first cell sets the task and the network, the next
sets the training and trains through the normal `fct` cache (a re-run reloads or resumes); then
the solver is configured and run with live plots of the layer kernels and of solver and sampler
health, once with HMC or with importance sampling (`SAMPLER`). All logic is in `fct.exemplar` and
`fct.draw`; each cell only calls it. Run

    python notebooks/build_exemplar.py CACHE_ROOT                  # execute, write tmp/exemplar.ipynb
    python notebooks/build_exemplar.py CACHE_ROOT --set "SAMPLER='importance'" --set N_OUTER=10
    python notebooks/build_exemplar.py --no-execute                # source only: notebooks/exemplar.ipynb

CACHE_ROOT holds the cached runs (`$FCT_CACHE_ROOT` works in the notebook too). `--set NAME=VALUE`
(a Python literal) rewrites the one-line assignment `NAME = ...` of a parameter cell. Only the
source notebook is committed; an executed one records the CACHE_ROOT path and its figures are
large, so it goes to the git-ignored tmp/ unless --out says otherwise.
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

SOURCE_OUT = Path(__file__).with_name("exemplar.ipynb")
EXECUTED_OUT = Path(__file__).parents[1] / "tmp" / "exemplar.ipynb"

INTRO = """# Exemplar: a task, SGLD training, and the mean-field solve, watched live

A linear teacher ($x = Z W_X^\\top$, $y = Z W_Y^\\top$ for $k$ latents $Z$) is learned by a small
network trained with SGLD at inverse temperature $\\beta$, so the trained net is a sample from
the distribution $e^{-\\beta \\mathcal L}$. The mean-field theory of that distribution is a saddle
point in the layer kernels $\\Phi_l$ and their conjugates $\\hat\\Phi_l$, found by an outer
descent on $\\Phi$ and an inner ascent on $\\hat\\Phi$, with the single-site expectations
estimated by a sampler. The solve is shown live: the kernels, how well the iteration is
converging, and whether the sampler can be trusted.

Set `CACHE_ROOT` in the first code cell. Training and the solve are cached under it in
directories named by hashes of the settings: re-running reloads a finished step, resumes an
interrupted one, and changing a solver setting re-solves without retraining.

**Why the default sizes are tiny.** Everything runs on a CPU in about a minute per sampler:
$P = 24$ points, 6 features, 2 latents and one hidden layer of 128 units. The prior-proposal
importance sampler (`SAMPLER = "importance"`) is a small-$P$, mild-tilt tool: its effective
sample size collapses as $P$ or $\\beta$ grow, and the solver then raises
`InadequateSamplerError` rather than continue on a degenerate estimate. At $P = 24$ and
$\\beta = 10$ it works; with two hidden layers, at $P = 100$ or at $\\beta = 10^3$ it did not in the trial runs,
and HMC (`SAMPLER = "hmc"`) is the sampler to use there. The net is therefore a hot one
($\\beta = 10$), so its train MSE sits well above the theory's, which describes the mean
predictor of the distribution; the quantities to compare are the kernels."""

TASK_AND_NET = """CACHE_ROOT = None  # directory for the cached runs; None reads $FCT_CACHE_ROOT
# task: a linear teacher with m features and k latents
P = 24  # samples; all of them enter the solve
M_IN = 6  # m = D_in: features of the input x and of the output y
K_LATENTS = 2  # k latents, fewer than m
DATA_SEED = 0
# network
N = 128  # hidden units per layer
L = 1  # hidden layers
PHI = "tanh"  # "tanh", "relu" or "sigmoid"
G = 1.0  # hidden-to-hidden gain
NET_SEED = 0
# loss weights
WEIGHT_LAMBDA = 1e-2  # weight penalty
RATE_LAMBDA = 1e-3  # rate penalty"""

TRAINING = """OPTIMIZER = "sgld"  # "sgld" (Langevin noise of variance 2 * LR / BETA per step) or "adam"
BETA = 10.0  # inverse temperature: read by SGLD, and the beta of the mean-field theory
LR = 3e-2
N_EPOCHS = 10_000
TRAIN_SEED = 0
TRAIN_LOG_INT = 500  # epochs between logged losses"""

SETUP = """import matplotlib as mpl

from fct.draw import fig_example_outputs, fig_loss_curves
from fct.exemplar import (
    LiveView,
    cache_root_from_env,
    empirical_kernel_figure,
    example_outputs_of,
    exemplar_config,
    kernel_comparison,
    solve_exemplar,
    solve_summary,
    train_exemplar,
    with_solver,
)
from fct.report import show

mpl.rcParams["figure.dpi"] = 100
RUNS_ROOT = CACHE_ROOT or cache_root_from_env()"""

TRAIN = """CFG = exemplar_config(
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
    log_int=TRAIN_LOG_INT,
    save_int=TRAIN_LOG_INT,
)
DATASET, NET, HISTORY, CACHE_DIR = train_exemplar(CFG, RUNS_ROOT)
show([f"Trained net cached in `{CACHE_DIR.name}`."])"""

TRAIN_PLOTS = """show(
    [
        fig_loss_curves(HISTORY, title=f"{OPTIMIZER}, beta = {BETA:g}"),
        fig_example_outputs("linear_teacher", example_outputs_of(NET, DATASET, P), CFG.data),
        empirical_kernel_figure(NET, DATASET, CFG),
    ]
)"""

SOLVER = """SAMPLER = "hmc"  # "hmc" (persistent chains) or "importance" (draws from the prior)
N_OUTER = 40  # outer descent steps on the kernels
PLOT_EVERY = 5  # refresh the live figures every this many outer steps
SOLVER_SEED = 0
SOLVE_LOG_INT = 1  # outer steps between log lines
# HMC only
N_CHAINS = 512
WARMUP_TRANS = 50  # transitions that let the chains equilibrate before the first estimate
# importance sampling only
N_IMPORTANCE_SAMPLES = 4096  # draws per transition"""

SOLVE = """SOLVE_CFG = with_solver(
    CFG,
    sampler=SAMPLER,
    n_outer=N_OUTER,
    seed=SOLVER_SEED,
    log_int=SOLVE_LOG_INT,
    n_chains=N_CHAINS,
    warmup_trans=WARMUP_TRANS,
    n_importance_samples=N_IMPORTANCE_SAMPLES,
    sample_dtype="float64",
)
VIEW = LiveView(tail=6)  # replaces the figures at every refresh, keeps the last log lines
RESULT = solve_exemplar(SOLVE_CFG, DATASET, NET, RUNS_ROOT, PLOT_EVERY, VIEW.show, log=VIEW.log)
show([solve_summary(RESULT.history, SAMPLER)])"""

COMPARE = """show([kernel_comparison(RESULT, NET, DATASET, SOLVE_CFG)])"""


def build_notebook() -> NotebookNode:
    """The notebook without outputs."""
    nb = new_notebook(
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        }
    )
    cells = [
        new_markdown_cell(INTRO),
        new_markdown_cell(
            "## 1. Task and network\n\nEvery setting of the task and the network, and the loss "
            "weights, in one cell."
        ),
        new_code_cell(TASK_AND_NET, metadata={"tags": ["parameters"]}),
        new_markdown_cell(
            "## 2. Training\n\nSGLD takes a gradient step plus Gaussian noise; `BETA` sets the "
            "noise. The net is trained through the `fct` cache, so a re-run is instant."
        ),
        new_code_cell(TRAINING),
        new_code_cell(SETUP),
        new_code_cell(TRAIN),
        new_markdown_cell(
            "The training loss, the net's outputs on a few training points, and its empirical "
            "kernels $\\Phi_l = \\langle \\psi\\psi^\\top \\rangle$ at the end of training."
        ),
        new_code_cell(TRAIN_PLOTS),
        new_markdown_cell(
            "## 3. Mean-field solve\n\nThe solver settings; the problem itself (task, net, "
            "$\\beta$) is fixed above."
        ),
        new_code_cell(SOLVER),
        new_markdown_cell(
            "The live figures, refreshed every `PLOT_EVERY` outer steps: the kernels $\\Phi_l$ and "
            "$\\hat\\Phi_l$; solver health (residuals, step sizes, backtracking, train MSE of the "
            "theory against the net); and sampler health (HMC acceptance, step size, divergences "
            "and ESS, or the importance ESS and the largest weight share)."
        ),
        new_code_cell(SOLVE),
        new_markdown_cell("## The solved kernels next to the trained net's"),
        new_code_cell(COMPARE),
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
        help="set a parameter of the notebook (a Python literal), e.g. SAMPLER='importance'",
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
