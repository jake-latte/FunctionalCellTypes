"""Build and execute `exemplar.ipynb`: a task, SGLD training, and the mean-field solve with live
diagnostics.

One linear-teacher run end to end. The first cell sets the task and the network, the next
sets the training and trains through the normal `fct` cache (a re-run reloads or resumes); then
the solver is configured and run with live plots of the layer kernels and of solver and sampler
health, with HMC or with importance sampling (`SAMPLER`). All logic is in `fct.exemplar` and
`fct.draw`; each cell only calls it. Run

    python notebooks/build_exemplar.py CACHE_ROOT                  # execute, write tmp/exemplar.ipynb
    python notebooks/build_exemplar.py CACHE_ROOT --set "SAMPLER='importance'" --set N_OUTER_IMPORTANCE=10
    python notebooks/build_exemplar.py --no-execute                # source only: notebooks/exemplar.ipynb

CACHE_ROOT holds the cached runs (`$FCT_CACHE_ROOT` works in the notebook too). `--set NAME=VALUE`
(a Python literal) rewrites the one-line assignment `NAME = ...` of a parameter cell. Only the
source notebook is committed; an executed one records the CACHE_ROOT path and its figures are
large, so it goes to the git-ignored tmp/ unless --out says otherwise. The command line and the
parameter rewriting are `notebooks.build_spatial`'s.
"""

from pathlib import Path

from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from notebooks.build_spatial import build_main

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

**What the theory is compared with.** `PRIOR_COEFF` is the coefficient of the squared weights in the loss, and the library's
`weight_lambda` is $\\lambda_w = 2\\beta\\,\\texttt{PRIOR\\_COEFF}$ ($\\beta = 1$ for Adam), so SGLD and Adam train the same loss.
The loss puts the prior $\\frac{\\lambda_w}{2\\beta}\\sum W^2$
on the squared weights, so at equilibrium each weight has variance $1 / \\lambda_w$
for every $\\beta$; the readout acts as a Gaussian likelihood whose ridge is
$\\alpha = \\lambda_w\\,P\\,m / \\beta$, the rate penalty is a tilt
$2\\beta\\,\\texttt{RATE\\_LAMBDA}$, and the data term of the action has prefactor
$\\lambda_w / 2$ (one half only for unit prior scale). The solver's defaults are exactly these four constants (`fct.meanfield.network_constants`), so
the theory is that of the distribution the net was sampled from.

**Is the solve converged?** Judge it by the numbers printed under the final figures, not by the
figures alone: the residual of the $\\hat\\Phi$ fixed point over the last outer steps, whether the
inner loop converged at every outer step (the solver-health panels), the sampler's effective sample
size, and the kernel scales of the theory and the net. At the defaults the residual settles at 0.06
to 0.2 under HMC and mostly below 0.1 under importance sampling (with isolated rises to 0.3 or 0.4),
after a transient in the first steps while the iteration leaves the net's kernels; that is its Monte
Carlo floor with 512 chains, not 0.01. The default `INIT = "net"` starts at the net's own kernels,
but `INIT = "lazy"` ends at the same point: the agreement of scale is not inherited from the start.
The train MSE of the theory is that of the mean predictor, which is below that of one sampled net,
whose weights fluctuate at temperature $1/\\beta$.

**Sizes.** The defaults are tiny, so that training and a solve each take seconds on a CPU: $P = 20$
points, 6 features, 2 latents, one hidden layer of 128 units, $\\beta = 10$. The importance
sampler (`SAMPLER = "importance"`) draws from the prior, which makes it a small-$P$, mild-tilt
tool: its effective sample size falls as the solve moves the kernels away from the prior, and
the solver raises `InadequateSamplerError` when it drops below its floor instead of continuing on
a degenerate estimate. With these settings, and at $P = 24$, it stays far above the floor for
its 80 outer steps (an effective sample size of at least 70 % of the draws). HMC
(`SAMPLER = "hmc"`) does not draw from the prior and has no such floor."""

TASK_AND_NET = """CACHE_ROOT = None  # directory for the cached runs; None reads $FCT_CACHE_ROOT
# task: a linear teacher with m features and k latents
P = 20  # samples; all of them enter the solve
M_IN = 6  # m = D_in: features of the input x and of the output y
K_LATENTS = 2  # k latents, fewer than m
DATA_SEED = 0
# network
N = 128  # hidden units per layer
L = 1  # hidden layers
PHI = "tanh"  # "tanh", "relu", "sigmoid" or "id" (linear)
G = 1.0  # hidden-to-hidden gain
NET_SEED = 0
# loss weights
PRIOR_COEFF = 1e-2  # coefficient of the sum of squared weights in the loss
RATE_LAMBDA = 1e-3  # rate penalty"""

TRAINING = """OPTIMIZER = "sgld"  # "sgld" (Langevin noise of variance 2 * LR / BETA per step) or "adam"
BETA = 10.0  # inverse temperature: read by SGLD, and the beta of the mean-field theory
LR = 3e-2
N_EPOCHS = 10_000  # until the loss and the parameter step level off
TRAIN_SEED = 0"""

SETUP = """import matplotlib as mpl

from fct.cache import cache_root_from_env
from fct.draw import fig_example_outputs, fig_loss_curves
from fct.exemplar import (
    LiveView,
    empirical_kernel_figure,
    example_outputs_of,
    exemplar_config,
    kernel_comparison,
    solve_exemplar,
    solve_summary,
    train_exemplar,
    with_solver,
)
from fct.meanfield import net_kernels
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
    weight_lambda=PRIOR_COEFF * 2 * (1.0 if OPTIMIZER != "sgld" else BETA),
    rate_lambda=RATE_LAMBDA,
    optimizer=OPTIMIZER,
    beta=BETA,
    lr=LR,
    n_epochs=N_EPOCHS,
    train_seed=TRAIN_SEED,
)
RUN = train_exemplar(CFG, RUNS_ROOT)
DATASET, NET = RUN.dataset, RUN.net
show([f"Trained net cached in `{RUN.cache_dir.name}`."])"""

TRAIN_PLOTS = """show(
    [
        fig_loss_curves(RUN.history, title=f"{OPTIMIZER}, beta = {BETA:g}"),
        fig_example_outputs("linear_teacher", example_outputs_of(NET, DATASET), CFG.data),
        empirical_kernel_figure(NET, DATASET, CFG),
    ]
)"""

SOLVER = """SAMPLER = "hmc"  # "hmc" (persistent chains) or "importance" (draws from the prior)
INIT = "net"  # initial kernels: "net" (the trained net's), "lazy" or "gaussian"
N_OUTER_HMC = 80  # outer descent steps on the kernels
N_OUTER_IMPORTANCE = 80  # outer descent steps under importance sampling
PLOT_EVERY = 5  # refresh the live figures every this many outer steps
SOLVER_SEED = 0
SOLVE_LOG_INT = 1  # outer steps between log lines
# The solve raises if the fixed-point residual of Phi_hat stays at least this over its last
# outer steps (Phi_hat has lost its tilt); a smaller residual does not mean convergence.
TILT_TRIPWIRE_RESID = 0.9
# HMC only
N_CHAINS = 512
WARMUP_TRANS = 50  # transitions that let the chains equilibrate before the first estimate
N_TRANS = 5  # transitions per inner-loop sampler run
N_TRANS_OUTER = 10  # transitions per outer-gradient run
# importance sampling only
N_IMPORTANCE_SAMPLES = 4096  # draws per transition"""

SOLVE = """SAMPLER_SETTINGS = {
    "hmc": {
        "n_outer": N_OUTER_HMC,
        "n_chains": N_CHAINS,
        "warmup_trans": WARMUP_TRANS,
        "n_trans": N_TRANS,
        "n_trans_outer": N_TRANS_OUTER,
    },
    "importance": {"n_outer": N_OUTER_IMPORTANCE, "n_importance_samples": N_IMPORTANCE_SAMPLES},
}
SOLVE_CFG = with_solver(
    CFG,
    sampler=SAMPLER,
    init=INIT,
    seed=SOLVER_SEED,
    log_int=SOLVE_LOG_INT,
    sample_dtype="float64",
    tilt_tripwire_resid=TILT_TRIPWIRE_RESID,
    **SAMPLER_SETTINGS[SAMPLER],
)
VIEW = LiveView(tail=6)  # replaces the figures at every refresh, keeps the last log lines
RESULT = solve_exemplar(
    SOLVE_CFG, DATASET, NET, RUNS_ROOT, PLOT_EVERY, VIEW.show, log=VIEW.log, on_failure=VIEW.flush
)
NET_PHI = net_kernels(NET, DATASET, SOLVE_CFG)
show([solve_summary(RESULT.history, SAMPLER, RESULT.Phi, NET_PHI)])"""

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
            "kernels $\\Phi_l = \\langle \\psi\\psi^\\top \\rangle$, $\\psi = \\phi(h^{(l)})$, at the "
            "end of training; the output reads the last one, $\\hat y = V^\\top \\phi(h^{(L)}) / N$."
        ),
        new_code_cell(TRAIN_PLOTS),
        new_markdown_cell(
            "## 3. Mean-field solve\n\nThe solver settings; the problem itself (task, net, "
            "$\\beta$) is fixed above."
        ),
        new_code_cell(SOLVER),
        new_markdown_cell(
            "The live figures, refreshed every `PLOT_EVERY` outer steps: the kernels $\\Phi_l$ and "
            "$\\hat\\Phi_l$; solver health (residuals, step sizes, backtracking, PD clamping, "
            "train MSE of the theory against the net); and sampler health (HMC acceptance, step "
            "size, divergences and ESS, or the importance ESS and the largest weight share). "
            "Layers are counted from 1 in all of them."
        ),
        new_code_cell(SOLVE),
        new_markdown_cell(
            "## The solved kernels next to the trained net's\n\nEach panel has its own colour "
            "scale, so the shapes can be compared; the scales themselves are in the summary above."
        ),
        new_code_cell(COMPARE),
    ]
    # Fixed ids, so rebuilding the notebook does not churn the diff.
    for i, cell in enumerate(cells, start=1):
        cell.id = f"cell-{i:03d}"
    nb.cells = cells
    return nb


def main(argv: list[str] | None = None) -> int:
    description = __doc__.split("\n\n")[0]
    return build_main(
        build_notebook, SOURCE_OUT, EXECUTED_OUT, description, "SAMPLER='importance'", argv
    )


if __name__ == "__main__":
    raise SystemExit(main())
