"""Build and execute `spatial.ipynb`: train a net under a spatial weight prior and look at it.

One linear-teacher run. The first cells set the task, the network, the training and the spatial
prior (an embedding and a kernel per population); the notebook then shows the eigenvalue floor
and the prior kernels, trains through the normal `fct` cache (a re-run reloads or resumes), and
shows the loss curves, example outputs, the first-layer weights next to the prior, the
spatially indexed activity kernels Phi of every hidden layer (their data factor over all samples
and their blocks between a few chosen samples) and the tuning of every neuron to the latents,
one panel per neuron placed by its position.

The library computes (the kernels and penalty matrices, the training run, rates, tuning curves
and MI); every figure is drawn by a plotting function written out in full in the code cell that
draws it, so a reader sees exactly what is plotted. Each figure cell is preceded by a markdown
cell that defines the plotted quantity with an equation. A new figure is one more such pair
in `build_notebook`: the markdown first, then a code cell with any settings, the plotting
function, and the call at the bottom. Run

    python notebooks/build_spatial.py CACHE_ROOT                  # execute, write tmp/spatial.ipynb
    python notebooks/build_spatial.py CACHE_ROOT --set "OPTIMIZER='adam'" --set N_EPOCHS=200   # Python literals
    python notebooks/build_spatial.py --no-execute                # source only: notebooks/spatial.ipynb

CACHE_ROOT holds the cached runs (`$FCT_CACHE_ROOT` works in the notebook too). Only the source
notebook is committed; an executed one records the CACHE_ROOT path and its figures are large, so
it goes to the git-ignored tmp/ unless --out says otherwise.
"""

import argparse
import ast
import copy
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from nbclient import NotebookClient
from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from notebooks.build_modularity import EXECUTE_TIMEOUT_S, write_notebook

SOURCE_OUT = Path(__file__).with_name("spatial.ipynb")
EXECUTED_OUT = Path(__file__).parents[1] / "tmp" / "spatial.ipynb"

# Cell sources are raw strings: a backslash in them (LaTeX in the markdown, an escape inside a
# string in the code) reaches the notebook unchanged.

INTRO = r"""# A spatial weight prior: kernels, training, and the tuning of every cell

Every unit of the input, the hidden layer(s) and the output sits at a position on a ring (or a
torus); the weight prior of each layer is a Gaussian kernel over those positions, so units that
are close in space get similar weights. The notebook sets the prior, trains a net under it on a
linear teacher, and looks at what training produced: the weights next to the prior, and the
tuning of each neuron to each latent, laid out by the neuron's position. Every figure is drawn by
a function written out in the cell that draws it; the markdown before it gives the equation of
what is plotted.

**Notation.**
- Task (`fct/tasks.py`, linear teacher): latents $z_\mu \in \mathbb{R}^k$ with i.i.d. entries
  uniform on $[-1, 1]$, inputs $x_\mu = W_X z_\mu \in \mathbb{R}^m$ and targets
  $y_\mu = W_Y z_\mu \in \mathbb{R}^m$, $\mu = 1, \dots, P$; $W_X$ and $W_Y$ are $(m, k)$ with
  i.i.d. $\mathcal{N}(0, 1/k)$ entries fixed by `DATA_SEED`.
- Network (`fct/model.py`), $L$ hidden layers of $N$ units with activation $\phi$:
  $h^{(1)}_\mu = U x_\mu / \sqrt{m}$, $h^{(l)}_\mu = (g / \sqrt{N})\, J_{l-1}\, \phi(h^{(l-1)}_\mu)$
  for $l = 2, \dots, L$, output $\hat y_\mu = V^\top \phi(h^{(L)}_\mu) / N$, a linear map of the last layer's post-activations. The weight layers, each
  stored (post, pre), are $W_1 = U$ of shape $(N, m)$, $W_l = J_{l-1}$, and $W_{L+1} = V^\top$.
  The rates are $r^{(l)}_{i\mu} = \phi(h^{(l)}_{i\mu})$.
- Loss (`fct/train.py`):
  $\mathcal{L} = \frac{N}{2}\,\mathrm{mse} + \frac{\lambda_w}{2\beta} \sum_l \mathrm{tr}(M^{\mathrm{post}}_l W_l M^{\mathrm{pre}}_l W_l^\top) + \lambda_r \sum_{l, i, \mu} (r^{(l)}_{i\mu})^2$
  with $\mathrm{mse} = \frac{1}{P m} \sum_\mu \lVert y_\mu - \hat y_\mu \rVert^2$,
  $\lambda_w = 2\beta\,$`PRIOR_COEFF` (derived from the loss coefficient `PRIOR_COEFF` $= \lambda_w / 2\beta$, so SGLD and Adam train the same loss), $\lambda_r$ = `RATE_LAMBDA`, $\beta$ = `BETA` under SGLD and 1 under Adam; the penalty matrices $M$ are
  defined in the spatial prior section.

Set `CACHE_ROOT` in the first cell. The run is cached under it in a directory named by a hash of
the task, net, training and spatial settings, so re-running reloads a finished run, resumes an
interrupted one, and changing any setting starts a new one."""

# The notation block, shared with the spatial-experiment notebook so the two cannot drift.
NOTATION = INTRO[INTRO.index("**Notation.**") : INTRO.index("Set `CACHE_ROOT`")].rstrip()

TASK_AND_NET = """CACHE_ROOT = None  # directory for the cached runs; None reads $FCT_CACHE_ROOT
# task: a linear teacher with m features and k latents
P = 256  # samples
M_IN = 12  # m = D_in: features of the input x and of the output y
K_LATENTS = 3  # k latents, fewer than m
DATA_SEED = 0
# network
N = 16  # hidden units per layer
L = 1  # hidden layers
PHI = "tanh"  # "tanh", "relu", "sigmoid" or "id" (linear)
G = 1.0  # hidden-to-hidden gain
NET_SEED = 0
# loss weights
PRIOR_COEFF = 3e-3  # coefficient of the spatial weight penalty in the loss
RATE_LAMBDA = 1e-4  # multiplies the rate penalty
DEVICE = "cpu"  # "cpu", "cuda" or "auto"; these nets are tiny"""

TRAINING_TEXT = r"""## Training

With `OPTIMIZER = "sgld"` every epoch is one full-batch Langevin step on all weights $\theta$,
$$\theta \leftarrow \theta - \mathrm{lr}\, \nabla_\theta \mathcal{L} + \sqrt{2\, \mathrm{lr} / \beta}\; \xi, \qquad \xi \sim \mathcal{N}(0, I),$$
whose stationary law for a small step is $p(\theta) \propto e^{-\beta \mathcal{L}(\theta)}$, with $\beta$ =
`BETA`. With `"adam"` the net minimises $\mathcal{L}$ and $\beta$ is unused. SGLD takes plain gradient steps
plus noise, so in the same number of epochs it fits the teacher less closely than Adam does."""

TRAINING = """OPTIMIZER = "sgld"  # "sgld" (Langevin noise of variance 2 * LR / BETA per step) or "adam"
BETA = 1e4  # inverse temperature, read by SGLD only
LR = 1e-2
N_EPOCHS = 2000
TRAIN_SEED = 0"""

SETUP_TEXT = """## Setup

The library computes; the cells below only draw, with plain matplotlib on the arrays it returns."""

IMPORTS = """import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import Markdown, display

from fct.analyse import example_outputs, pick_examples, r_squared
from fct.exp_view import tuning_view
from fct.meanfield import readout_activities
from fct.modularity import information_overlap, rate_overlap
from fct.spatial import (
    data_kernel,
    latent_kernel,
    spatial_kernel_pairs,
    torus_distance,
    wrapped_gaussian_profile,
)
from fct.spatial_view import (
    layer_spaces,
    order_weights,
    ordered_kernel_comparison,
    panel_grid,
    position_order,
)

mpl.rcParams.update({"figure.dpi": 100, "font.size": 12, "axes.titlesize": 12})"""

# What only the notebook that trains its own run needs; the spatial-experiment notebook has its own
# extras.
TRAIN_IMPORTS = """from fct.cache import cache_root_from_env
from fct.data import get_latent_lattice_dataset
from fct.spatial_view import make_config, train_run

RUNS_ROOT = CACHE_ROOT or cache_root_from_env()"""

PRIOR_TEXT = r"""## Spatial prior

Every unit $i$ of population $p$ (the input, hidden layer $l$, the output) has a position
$x^p_i$ on the torus $[0, \mathrm{box})^d$ (`D`, `BOX`), placed on a grid or uniformly at random.
The population's kernel is the wrapped Gaussian of `fct/spatial.py`, with amplitude $A_p$ and
length scale $\ell_p$ set by `population_spec` (if `lambda_max` is set, the kernel is rescaled so its
top eigenvalue is `lambda_max`, which multiplies it by `Population.scale`; $A_p$ below is then
`amplitude * scale`, its diagonal):
$$K^p_{ij} = A_p \prod_{a=1}^{d} \frac{G(\delta_a)}{G(0)}, \qquad G(\delta) = \sum_{|n| \le n_\mathrm{img}} e^{-(\delta + n\, \mathrm{box})^2 / 2 \ell_p^2},$$
where $\delta = x^p_i - x^p_j$ reduced to $[-\mathrm{box}/2, \mathrm{box}/2]$ on every axis $a$.
Weight layer $l$ maps population pre to population post, and its prior covariance is
$K^{\mathrm{post}}_l \otimes K^{\mathrm{pre}}_l$. The penalty uses the inverse with floored
eigenvalues: from $K^p = \sum_j \lambda_j v_j v_j^\top$,
$$M^p = \sum_j \max(\lambda_j,\, \epsilon\, \lambda_{\max})^{-1}\, v_j v_j^\top, \qquad \epsilon = \texttt{MIN\_REL\_EIG},$$
and the loss term is $\frac{\lambda_w}{2\beta} \sum_l \mathrm{tr}(M^{\mathrm{post}}_l W_l M^{\mathrm{pre}}_l W_l^\top)$ ($\beta = 1$ for Adam).
Under SGLD at inverse temperature $\beta$ the stationary law $\propto e^{-\beta \mathcal{L}}$ makes the
prior alone
$$\mathrm{vec}(W_l) \sim \mathcal{N}\big(0,\ \lambda_w^{-1}\, K^{\mathrm{post}}_l \otimes K^{\mathrm{pre}}_l\big)$$
up to the floor (exactly, with $K^p$ replaced by $(M^p)^{-1}$). These kernels are over *units*:
they are a prior on the weights, not the mean-field kernels $\Phi^{(l)}$ over *samples* of
`fct/meanfield.py`. Under this prior those kernels gain a pair of unit indices; they are shown,
built from the trained activity and these $K$, in their own section below."""

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
    weight_lambda=PRIOR_COEFF * 2 * (1.0 if OPTIMIZER != "sgld" else BETA),
    rate_lambda=RATE_LAMBDA,
    optimizer=OPTIMIZER,
    beta=BETA,
    lr=LR,
    n_epochs=N_EPOCHS,
    train_seed=TRAIN_SEED,
    spatial=SPATIAL,
    device=DEVICE,
)
SPACES = layer_spaces(CFG)  # weight layer l = 1 .. L + 1 is SPACES[l - 1]"""

FLOOR_TEXT = r"""### The eigenvalue floor and the SGLD step size

A smooth kernel has eigenvalues down to round-off, so the floor $\epsilon$ is what keeps $M^p$
finite. It sets the largest penalty eigenvalue of a population, and that of a layer is the
product of those of its two factors, so the floor enters squared:
$$\lambda_{\max}(M^p) = \frac{1}{\epsilon\, \lambda_{\max}(K^p)}, \qquad \lambda_{\max}(M_l) = \lambda_{\max}(M^{\mathrm{post}}_l)\, \lambda_{\max}(M^{\mathrm{pre}}_l) = \frac{1}{\epsilon^2\, \lambda_{\max}(K^{\mathrm{post}}_l)\, \lambda_{\max}(K^{\mathrm{pre}}_l)}.$$
The prior term has curvature $(\lambda_w / \beta)\, \lambda_{\max}(M_l)$ along its stiffest direction, and
an Euler step of size lr on a quadratic diverges once lr times its curvature reaches 2, so SGLD
needs (and `fct.train` refuses to start otherwise)
$$\mathrm{lr}\, \frac{\lambda_w}{2\beta}\, \lambda_{\max}(M_l) < 1 \quad \text{for every layer } l.$$
At the defaults (rings of 12 input, 16 hidden and 12 output units, $\ell = 0.15$, $A = 1$,
$\epsilon = 10^{-2}$), $\lambda_{\max}(K) \approx 4.51$ for 12 units and $6.02$ for 16, so
$\lambda_{\max}(M_l) \approx 1 / (\epsilon^2 \cdot 4.51 \cdot 6.02) = 1 / (10^{-4} \cdot 4.51 \cdot 6.02) \approx 368$ and
$\mathrm{lr}\, \frac{\lambda_w}{2\beta}\, \lambda_{\max}(M_l) \approx 10^{-2} \cdot 3 \cdot 10^{-3} \cdot 368 \approx 0.011$ (`PRIOR_COEFF` $= 3 \cdot 10^{-3}$):
both optimisers are stable and 2000 epochs take seconds on a CPU. The library default
$\epsilon = 10^{-6}$ would make both $10^8$ times larger ($\lambda_{\max}(M_l) \approx 3.7 \cdot 10^{10}$,
$\mathrm{lr}\, \frac{\lambda_w}{2\beta}\, \lambda_{\max}(M_l) \approx 1.1 \cdot 10^6$); a larger network or a lower floor needs a
smaller `LR` or `PRIOR_COEFF`. The tables give the numbers for the current settings."""

FLOOR = r"""def top_eigenvalue(matrix):
    return torch.linalg.eigvalsh(matrix)[-1].item()


def stiffness(space):
    # lambda_max(M_l): the Kronecker product's largest eigenvalue is the product of its factors'
    return top_eigenvalue(space.M_post) * top_eigenvalue(space.M_pre)


populations = [SPACES[0].pre, *(space.post for space in SPACES)]
population_table = [
    r"| population | eigenvalues floored | $\lambda_{\max}(K^p)$ | $\lambda_{\max}(M^p)$ |",
    "|---|---|---|---|",
    *(
        f"| {p.name} | {p.report.n_floored} of {len(p.K)} | {top_eigenvalue(p.K):.3g} "
        f"| {top_eigenvalue(p.M):.3g} |"
        for p in populations
    ),
]
layer_table = [
    (
        r"| layer $l$ | $\lambda_{\max}(M_l)$ "
        r"| $\mathrm{lr}\, \frac{\lambda_w}{2\beta}\, \lambda_{\max}(M_l)$ (SGLD needs < 1) |"
    ),
    "|---|---|---|",
    *(
        f"| {s.layer + 1} ({s.pre.name} to {s.post.name}) | {stiffness(s):.3g} "
        f"| {LR * PRIOR_COEFF * stiffness(s):.3g} |"
        for s in SPACES
    ),
]
display(Markdown("\n".join(population_table) + "\n\n" + "\n".join(layer_table)))"""

KERNELS_TEXT = r"""### The prior kernels of each weight layer

One figure per weight layer $l$, showing the two factors of its prior covariance
$K^{\mathrm{post}}_l \otimes K^{\mathrm{pre}}_l$. Top: $K^{\mathrm{pre}}_l$ and
$K^{\mathrm{post}}_l$ as matrices, units in index order. Bottom left: every entry $K^p_{ij}$
against the torus distance $\lVert x^p_i - x^p_j \rVert$ (points), with the kernel along one axis
$$\kappa_p(\delta) = A_p\, \frac{G(\delta)}{G(0)}, \qquad 0 \le \delta \le \mathrm{box} / 2,$$
as a line. Bottom right: the unit positions (for $d = 1$ the pre ring outside, the post ring
inside). For $d = 1$ the points lie on the line; for $d = 2$ they scatter around it, because
$K$ depends on both axis differences, not only on the distance."""

KERNELS = r"""def plot_layer_prior(space):
    # The prior of one weight layer: K_pre and K_post as matrices, K_ij against torus distance
    # with the profile kappa(delta), and the unit positions.
    d = space.pre.embedding.positions.shape[1]
    if d > 2:
        raise ValueError(f"can only draw positions for d = 1 or 2, got d = {d}")
    factors = [("pre", space.pre, "C0", 1.0), ("post", space.post, "C1", 0.8)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), layout="constrained")
    for ax, (which, pop, _, _) in zip(axes[0], factors):
        image = ax.imshow(pop.K.numpy(), cmap="viridis", interpolation="none")
        fig.colorbar(image, ax=ax, shrink=0.8)
        ax.set(title=rf"$K^{{\rm {which}}}$ ({pop.name})", xlabel="unit $j$", ylabel="unit $i$")

    ax = axes[1, 0]
    for which, pop, color, _ in factors:
        x, box = pop.embedding.positions, pop.embedding.box
        dist = torus_distance(x, x, box).ravel()
        ax.scatter(dist.numpy(), pop.K.ravel().numpy(), s=6, alpha=0.4, color=color)
        # kappa(delta) is a function of one axis difference, defined up to box / 2
        delta = torch.linspace(0, min(dist.max().item(), box / 2), 200, dtype=x.dtype)
        diagonal = pop.kernel.amplitude * pop.scale  # K_ii, with or without lambda_max
        kappa = wrapped_gaussian_profile(delta, pop.kernel.length_scale, box, diagonal)
        label = rf"{which}: {pop.name}, $\ell$ = {pop.kernel.length_scale:g}"
        ax.plot(delta.numpy(), kappa.numpy(), color=color, label=label)
    ax.set(xlabel=r"torus distance $\Vert x_i - x_j \Vert$", ylabel="$K_{ij}$")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    for which, pop, color, radius in factors:
        xy = pop.embedding.positions.numpy()
        if d == 1:  # around a ring of the given radius
            angle = 2 * np.pi * xy[:, 0] / pop.embedding.box
            xy = radius * np.stack([np.cos(angle), np.sin(angle)], axis=1)
        ax.scatter(xy[:, 0], xy[:, 1], s=14, color=color, label=f"{which}: {pop.name}")
    if d == 1:
        ax.set_axis_off()
    else:
        ax.set(xlim=(0, space.pre.embedding.box), ylim=(0, space.pre.embedding.box))
    ax.set_aspect("equal")
    ax.set_title("positions")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    layer = space.layer + 1
    kron = rf"$K^{{\rm post}}_{layer} \otimes K^{{\rm pre}}_{layer}$"
    fig.suptitle(f"weight layer {layer}: prior covariance {kron}")
    return fig


for space in SPACES:
    plot_layer_prior(space)
plt.show()"""

TRAIN = """RUN = train_run(CFG, RUNS_ROOT, log=print)
# held-out set of the same teacher: one point per sign pattern of the k latents, 3^k in all
TEST_DATASET = get_latent_lattice_dataset(RUN.cache_dir, CFG)
tc = CFG.train
sampler = f"SGLD, beta = {tc.beta:g}" if tc.optimizer == "sgld" else "Adam"
last = {key: RUN.history[key][-1] for key in ("epoch", "loss", "mse", "weight_l2", "rate_l2")}
display(
    Markdown(
        f"**Trained with {sampler}**, lr = {tc.lr:g}, {tc.n_epochs} epochs; run cached in "
        f"`{RUN.cache_dir.name}`. At the last logged epoch ({last['epoch']}): loss "
        f"{last['loss']:.4g}, mse {last['mse']:.4g}, weight penalty {last['weight_l2']:.4g}, "
        f"rate penalty {last['rate_l2']:.4g}."
    )
)"""

LOSS_TEXT = r"""## Loss curves

The loss and its three raw terms, as logged during training, against epoch on a log axis:
$$\mathcal{L} = \frac{N}{2}\, \mathrm{mse} + \frac{\lambda_w}{2\beta}\, \Omega_w + \lambda_r\, \Omega_r, \qquad \Omega_w = \sum_l \mathrm{tr}(M^{\mathrm{post}}_l W_l M^{\mathrm{pre}}_l W_l^\top), \qquad \Omega_r = \sum_{l, i, \mu} (r^{(l)}_{i\mu})^2.$$
The terms are drawn unweighted, so they do not add up to the $\mathcal{L}$ curve; a term that is zero at
every epoch has no point on a log axis and is marked "(zero)" in the legend. Under SGLD the
curves settle to a noisy stationary level rather than to a minimum."""

LOSS = r"""def plot_loss_curves(history, title):
    # The logged loss L and its raw terms against epoch, on a log axis.
    terms = {
        "loss": r"loss $\mathcal{L}$",
        "mse": "mse",
        "weight_l2": r"weight penalty $\Omega_w$",
        "rate_l2": r"rate penalty $\Omega_r$",
    }
    fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
    for key, label in terms.items():
        values = np.asarray(history[key], dtype=float)
        if not np.any(values > 0):
            label += " (zero)"
        ax.plot(history["epoch"], values, label=label)
    ax.set(xlabel="epoch", ylabel="logged value", yscale="log", title=title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    return fig


plot_loss_curves(RUN.history, f"{OPTIMIZER}: training")
plt.show()"""

EXAMPLES_TEXT = r"""## Example outputs

The trained network's output $\hat y_\mu = V^\top \phi(h^{(L)}_\mu) / N$ against the target
$y_\mu = W_Y z_\mu$ for `N_EXAMPLES` points $\mu$ drawn at random from the **test set**: the $3^k$
points of the latent lattice $z \in \{-a, 0, a\}^k$ ($a$ the latent range, one point per sign pattern
of the latents, with inputs and targets from the training samples' own $W_X$ and $W_Y$), which the
net was not trained on. Left: one panel per sample
with $y_{\mu a}$ (ring on a stem) and $\hat y_{\mu a}$ (diamond) for every output $a = 1, \dots, m$.
Right: every entry $\hat y_{\mu a}$ against $y_{\mu a}$ of **all** $3^k$ test points (the examples
are only a subset of them), with the identity line and the pooled
$$R^2 = 1 - \frac{\sum_{\mu a} (y_{\mu a} - \hat y_{\mu a})^2}{\sum_{\mu a} (y_{\mu a} - \bar y)^2}$$
over all test points and outputs, $\bar y$ their mean. A good fit puts every diamond inside its ring
and every point on the identity line."""

EXAMPLES = r"""N_EXAMPLES = 4  # samples shown


def plot_example_outputs(outputs, title, everything=None):
    # True and predicted output vector of each example (`outputs`), then the entries of
    # `everything` (default: the examples) against each other with their pooled R^2.
    everything = outputs if everything is None else everything
    y_true, y_pred = outputs.y_true.numpy(), outputs.y_pred.numpy()
    n, m = y_true.shape
    fig = plt.figure(figsize=(12, 2.2 * n + 1.5), layout="constrained")
    left, right = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    rows = left.subgridspec(n, 1)
    index = np.arange(m)
    first = None
    for row, (mu, true, pred) in enumerate(zip(outputs.indices.tolist(), y_true, y_pred)):
        ax = fig.add_subplot(rows[row], sharex=first, sharey=first)
        first = ax if first is None else first
        ax.vlines(index, 0, true, color="C0")
        ax.plot(index, true, "o", ms=12, mfc="none", mec="C0", label=r"true $y_{\mu a}$")
        ax.plot(index, pred, "D", color="C1", label=r"predicted $\hat y_{\mu a}$")
        ax.axhline(0, color="grey", lw=1)
        ax.set_title(f"example {mu}", loc="left")
        ax.set_ylabel("output")
        ax.margins(y=0.25)
        ax.tick_params(labelbottom=row == n - 1)  # index labels on the last row only
    ax.set_xlabel("output index $a$")
    ax.set_xticks(index)
    first.legend(frameon=False, loc="upper right", ncol=2)

    ax = fig.add_subplot(right)
    all_true, all_pred = everything.y_true.numpy(), everything.y_pred.numpy()
    r2 = r_squared(everything.y_true, everything.y_pred)
    lim = [min(all_true.min(), all_pred.min()), max(all_true.max(), all_pred.max())]
    ax.plot(lim, lim, "--", color="grey", label=f"identity; $R^2$ = {r2:.2f}")
    ax.scatter(all_true.ravel(), all_pred.ravel(), s=30, alpha=0.7, color="C1")
    ax.set(xlabel=r"true $y_{\mu a}$", ylabel=r"predicted $\hat y_{\mu a}$", aspect="equal")
    ax.legend(frameon=False, loc="upper left")
    fig.suptitle(title)
    return fig


n_samples = len(TEST_DATASET.x)
picks = pick_examples(n_samples, min(N_EXAMPLES, n_samples), seed=0)
test_outputs = example_outputs(RUN.net, TEST_DATASET, list(range(n_samples)))
shown = example_outputs(RUN.net, TEST_DATASET, picks)
plot_example_outputs(shown, f"trained network, test set ({n_samples} points)", test_outputs)
plt.show()"""

EMPIRICAL_TEXT = r"""## First-layer weights: their kernels next to the prior

The trained first-layer weight $U$, of shape $(N, m)$ (hidden unit $i$, input unit $a$), has a
second-moment matrix over its hidden units and one over its input units,
$$\hat K^{\mathrm{post}} = U U^\top / m, \qquad \hat K^{\mathrm{pre}} = U^\top U / N,$$
shown next to the prior's $K^{\mathrm{post}}$ and $K^{\mathrm{pre}}$, every matrix with its units
sorted by ring position. Under the prior alone (SGLD, ignoring the floor)
$\mathrm{vec}(U) \sim \mathcal{N}(0, c\, K^{\mathrm{post}} \otimes K^{\mathrm{pre}})$ with
$c = \lambda_w^{-1}$, so
$\mathbb{E}[\hat K^{\mathrm{post}}] = c\, A_{\mathrm{pre}}\, K^{\mathrm{post}}$ and
$\mathbb{E}[\hat K^{\mathrm{pre}}] = c\, A_{\mathrm{post}}\, K^{\mathrm{pre}}$ ($A$ the diagonal of $K$:
the kernel amplitude times `Population.scale`): the comparison holds up to that scale, so each panel has its own
colour scale (Adam samples no such law: its prior is the $\beta = 1$ case, and only the shapes compare). These are second moments of the
trained *weights* over *units*, not the mean-field kernels $\Phi^{(l)}$ of `fct/meanfield.py`, which
are activity kernels between *samples* (with a pair of unit indices under a spatial prior), shown
in their own section below.
A smooth prior shows as a band along the diagonal; one weight matrix is a single sample, so
$\hat K$ is noisy, and the data term pulls it away from the prior. Drawn for $d = 1$ only, since
sorting by position needs a ring."""

EMPIRICAL = r"""def plot_kernel_comparison(comparison, space):
    # Prior K_post and K_pre beside the second moments of the weights, all sorted around the
    # ring; each panel has its own colour scale.
    panels = [
        (comparison.post_theory, rf"prior $K^{{\rm post}}$ ({space.post.name})"),
        (comparison.post_empirical, r"weights $\hat K^{\rm post}$"),
        (comparison.pre_theory, rf"prior $K^{{\rm pre}}$ ({space.pre.name})"),
        (comparison.pre_empirical, r"weights $\hat K^{\rm pre}$"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), layout="constrained")
    for ax, (kernel, title) in zip(axes.ravel(), panels, strict=True):
        image = ax.imshow(kernel.cpu().numpy(), cmap="viridis", interpolation="none")
        fig.colorbar(image, ax=ax, shrink=0.8)
        ax.set(title=title, xlabel="unit, by ring position", ylabel="unit, by ring position")
    fig.suptitle(f"weight layer {space.layer + 1}: prior kernels and those of the trained weights")
    return fig


if D == 1:
    plot_kernel_comparison(ordered_kernel_comparison(SPACES[0], RUN.net.U), SPACES[0])
    plt.show()
else:
    display(Markdown(f"Sorting units by position needs a ring (d = 1); here d = {D}."))"""

WEIGHTS_TEXT = r"""## First-layer weights $U$, sorted by position

The trained $U$ itself as an image, its rows (hidden units) and columns (input units) sorted
around their rings:
$$\text{pixel } (r, c) = U_{\pi(r)\, \sigma(c)}, \qquad x^{\mathrm{hidden}}_{\pi(1)} \le x^{\mathrm{hidden}}_{\pi(2)} \le \dots, \qquad x^{\mathrm{input}}_{\sigma(1)} \le x^{\mathrm{input}}_{\sigma(2)} \le \dots,$$
on a colour scale symmetric about zero, the ticks giving the ring positions. Under the prior,
neighbouring rows are correlated by $K^{\mathrm{post}}$ and neighbouring columns by
$K^{\mathrm{pre}}$, so a long length scale shows as smooth bands. Drawn for $d = 1$ only."""

WEIGHTS = r"""def ring_ticks(pop, order, n_ticks=8):
    # At most n_ticks ticks along a sorted axis, labelled by the ring position of the unit.
    ticks = np.arange(0, len(order), max(1, int(np.ceil(len(order) / n_ticks))))
    positions = pop.embedding.positions[:, 0].numpy()
    return ticks, [f"{positions[order[t]]:.2f}" for t in ticks]


def plot_weights_by_position(W, space):
    # A (post, pre) weight matrix with rows and columns sorted around their rings.
    pos_post, pos_pre = space.post.embedding.positions, space.pre.embedding.positions
    if pos_post.shape[1] != 1:
        raise ValueError(f"sorting units around a ring needs d = 1, got d = {pos_post.shape[1]}")
    ordered = order_weights(W.detach().cpu(), pos_post, pos_pre)
    values = ordered.W.numpy()
    bound = np.abs(values).max()
    fig, ax = plt.subplots(figsize=(8, 6.5), layout="constrained")
    image = ax.imshow(values, cmap="RdBu_r", vmin=-bound, vmax=bound, interpolation="none")
    fig.colorbar(image, ax=ax, label="weight")
    ax.set_xticks(*ring_ticks(space.pre, ordered.pre_order))
    ax.set_yticks(*ring_ticks(space.post, ordered.post_order))
    ax.set(
        xlabel=f"{space.pre.name} unit, by ring position",
        ylabel=f"{space.post.name} unit, by ring position",
        title=f"weight layer {space.layer + 1}, sorted by position",
    )
    return fig


if D == 1:
    plot_weights_by_position(RUN.net.U, SPACES[0])
    plt.show()
else:
    display(Markdown(f"Sorting units by position needs a ring (d = 1); here d = {D}."))"""

DATA_KERNEL_TEXT = r"""## Spatially indexed kernels $\Phi^{(l)}$ of the trained network

The kernel through which hidden layer $l$ drives the next layer carries two indices of each
kind: samples $\mu, \nu$ and post-synaptic units $i, j$ (pre-synaptic units are $a, b$). For the
weight layer $W_{l+1}$ that reads from hidden layer $l$, with the factorised prior
$K_{iajb} = K^{\mathrm{post}}_{ij} K^{\mathrm{pre}}_{ab}$ of the section above,
$$\Phi^{(l)\,\mu\nu}_{ij} = \frac{1}{N} \sum_{a,b=1}^{N} \psi^{(l)}_{a\mu}\, K_{iajb}\, \psi^{(l)}_{b\nu} = K^{\mathrm{post}}_{ij}\, A^{(l)}_{\mu\nu}, \qquad A^{(l)}_{\mu\nu} = \frac{1}{N} \sum_{a,b=1}^{N} \psi^{(l)}_{a\mu}\, K^{\mathrm{pre}}_{ab}\, \psi^{(l)}_{b\nu}, \qquad \psi^{(l)} = \phi\big(h^{(l)}\big).$$
$K^{\mathrm{pre}}$ is the kernel of hidden layer $l$ and $K^{\mathrm{post}}$ that of hidden layer
$l + 1$, or of the output for the last hidden layer. So $\psi$ is the rate $r$ above on every
layer, the last one included, since the output reads $\phi(h^{(L)})$; this is the paper's
eq. kernel-def and what `fct.meanfield.empirical_kernels` computes
(`fct.meanfield.readout_activities` gives these $\psi$). The post-synaptic kernel only multiplies: all the
dependence on the data is in the $(P_\mathrm{test} \times P_\mathrm{test})$ data kernel $A^{(l)}$, the pre-synaptic index
summed out against its prior. Writing $K^{\mathrm{pre}} = \sum_q \lambda_q v_q v_q^\top$ and
$\tilde\psi_{q\mu} = v_q^\top \psi_\mu$, $A_{\mu\nu} = \frac{1}{N} \sum_q \lambda_q \tilde\psi_{q\mu} \tilde\psi_{q\nu}$:
each activity pattern is weighted by the prior variance of its spatial mode, so a smooth
$K^{\mathrm{pre}}$ keeps the patterns that are smooth over the ring, and with
$K^{\mathrm{pre}} = I$ it is the plain $\psi^\top \psi / N$ (in the code $\psi$ is stored
$(P, N)$, samples by rows, so the same matrix is `psi @ psi.T / n_pre` there). The input kernel $\Phi^{(0)}$
(with $1/m$, over the input units) is not drawn.

The activities $\psi$ are those of the **test set**: the $P_\mathrm{test} = 3^k$ points of the latent
lattice $z \in \{-a, 0, a\}^k$ (one point per sign pattern of the $k$ latents, inputs from the
training samples' own teacher map $W_X$; `fct.data.get_latent_lattice_dataset`), so $A^{(l)}$ is
$(3^k \times 3^k)$ and every pair of sign patterns appears once, instead of whatever pairs the random
training samples happen to give.

One panel per hidden layer: $A^{(l)}$ over the $P_\mathrm{test}$ points, with its own colour scale
(symmetric about zero, since $A$ is signed) and titled with the kernel scale
$\mathrm{tr}\,A^{(l)} / P_\mathrm{test}$. The first panel is the latent kernel of the same points,
$$B_{\mu\nu} = \frac{1}{k} \sum_{f=1}^{k} z_{f\mu}\, z_{f\nu} = \frac{1}{k}\, z_\mu \cdot z_\nu,$$
which a layer that is linear in the latents would have as its data kernel up to scale
($A \propto B$), so how much each $A^{(l)}$ resembles $B$ shows how far that layer is from linear.
The points are **sorted** by the latent $z_f$ with $f$ =
`PHI_SORT_LATENT` (0-based; `None` keeps lattice order), the ties of the three values $-a, 0, a$
keeping lattice order. How to read it: a first layer that is
nearly linear in $x = W_X z$ gives a dot-product-like kernel, $A_{\mu\nu} \propto z_\mu \cdot z_\nu$,
which sorted by $z_0$ shows a rank-one $z_0 z_0^\top$ pattern: the same sign in the top-left and
bottom-right quadrants, the opposite sign in the two off-diagonal ones. A band along the diagonal
would instead mean that similarity falls with the distance $|z_{f\mu} - z_{f\nu}|$, as for
localised tuning. This is the trained net's empirical kernel only: the library's solver has no
spatial prior, so no theory $\Phi$ is computed here."""

DATA_KERNEL = r"""PHI_SORT_LATENT = 0  # 0-based index f of the latent z_f (z0 .. z{k-1}) that orders the test points; None: lattice order


def sample_order(factors, name, n_samples):
    # Sample indices sorted by the latent called `name`; dataset order for None.
    if name is None:
        return np.arange(n_samples)
    return np.argsort(factors[name].detach().cpu().numpy(), kind="stable")


def data_kernels(H, phi, spaces):
    # A^(l) (P, P) of every hidden layer l (0-based) from the pre-activations H (P, L, N): its
    # psi = phi(h) against the K_pre of the weight layer it feeds, spaces[l + 1]. The K are
    # float64 on the CPU, so psi is brought there from the training device.
    psis = readout_activities(H, phi)
    kernels = [spaces[l + 1].pre.K for l in range(len(psis))]
    return torch.stack(
        [data_kernel(psi.detach().cpu().to(K.dtype), K) for psi, K in zip(psis, kernels)]
    )


def plot_data_kernels(A, order, sorted_by, B=None):
    # One panel per hidden layer: the (P, P) data kernel with samples reordered by `order`, its
    # own colour scale symmetric about zero, titled with psi = phi(h) and the kernel scale
    # tr(A) / P (unchanged by the reordering). With B, the latent kernel Z Z^T / k of the same
    # samples is drawn first, in the same order.
    A = A.detach().cpu().numpy()
    n_layers, P = A.shape[0], A.shape[1]
    if sorted(order.tolist()) != list(range(P)):
        raise ValueError(f"order is not a permutation of the {P} samples")
    panels = [
        (A[layer], rf"hidden layer {layer + 1}, $\psi = \phi(h)$: $\mathrm{{tr}}\,A / P$")
        for layer in range(n_layers)
    ]
    if B is not None:
        B = B.detach().cpu().numpy()
        if B.shape != (P, P):
            raise ValueError(f"B is {B.shape}, but A is over {P} samples")
        panels.insert(0, (B, r"latent kernel $B = Z Z^\top / k$: $\mathrm{tr}\,B / P$"))
    fig, axes = plt.subplots(
        1, len(panels), figsize=(5 * len(panels) + 1, 5), squeeze=False, layout="constrained"
    )
    for ax, (kernel, title) in zip(axes[0], panels, strict=True):
        shown = kernel[np.ix_(order, order)]
        bound = np.abs(shown).max()
        image = ax.imshow(shown, cmap="RdBu_r", vmin=-bound, vmax=bound, interpolation="none")
        fig.colorbar(image, ax=ax, shrink=0.8)
        ax.set(
            title=f"{title} = {np.trace(kernel) / P:.3g}",
            xlabel=rf"sample $\nu$, sorted by {sorted_by}" if sorted_by else r"sample $\nu$",
            ylabel=r"sample $\mu$",
        )
    return fig


with torch.no_grad():
    H, _ = RUN.net(TEST_DATASET.x.to(RUN.net.dtype))  # (3^k, L, N), on the training device
A = data_kernels(H, RUN.net.phi, SPACES)
by = None if PHI_SORT_LATENT is None else f"z{PHI_SORT_LATENT}"
order = sample_order(TEST_DATASET.factors, by, len(TEST_DATASET.x))
print(f"test set: {len(order)} points, data kernels A of shape {tuple(A.shape[1:])} per layer")
plot_data_kernels(A, order, by, latent_kernel(TEST_DATASET.factors))
plt.show()"""

TRIALS_TEXT = r"""### The trials whose kernels are drawn

The full $\Phi^{(l)}$ has $P^2$ blocks of $(n_\mathrm{post} \times n_\mathrm{post})$, too many
to draw, so the grids below show the $S$ = `PHI_TRIALS` test points at evenly spaced ranks of the
sorting latent, from its smallest value to its largest:
$$\mu_s = \pi\big(\mathrm{round}(s\, (P_\mathrm{test} - 1) / (S - 1))\big), \qquad s = 0, \dots, S - 1,$$
with $\pi$ the sorting order of the figure above ($\mu_0 = \pi(0)$ for $S = 1$). With
`PHI_SORT_LATENT = None`, $\pi$ is the dataset order, so the picks are evenly spaced dataset
indices and their latents follow no order from bar group to bar group. Each test point is a sign
pattern of the latents ($z_f \in \{-a, 0, a\}$), so the picks spread over the sign patterns, ordered by
the sorting latent. The figure
shows the latents $z_{f\mu_s}$ of the chosen samples, one group of $k$ bars per sample, labelled
by its index $\mu_s$."""

TRIALS = r"""PHI_TRIALS = 4  # S: samples whose kernel pairs are drawn, spread over the sorted order


def spread_trials(order, n_trials):
    # n_trials sample indices at evenly spaced positions of `order`, its first and last included.
    if not 1 <= n_trials <= len(order):
        raise ValueError(f"need 1 to {len(order)} trials, got {n_trials}")
    ranks = np.round(np.linspace(0, len(order) - 1, n_trials)).astype(int)
    return np.asarray(order)[ranks]


def plot_trial_latents(factors, trials):
    # One group of bars per trial, one bar per latent: z_{f mu} of the chosen samples.
    names = list(factors)
    values = np.stack([factors[name].detach().cpu().numpy()[trials] for name in names], axis=1)
    n_trials, k = values.shape
    width = 0.8 / k
    x = np.arange(n_trials)
    fig, ax = plt.subplots(figsize=(1.5 * n_trials + 3, 3.2), layout="constrained")
    for f, name in enumerate(names):
        ax.bar(x + (f - (k - 1) / 2) * width, values[:, f], width, color=f"C{f}", label=name)
    ax.axhline(0, color="grey", lw=1)
    ax.set_xticks(x, [rf"$\mu$ = {mu}" for mu in trials])
    ax.set(ylabel=r"latent $z_{f\mu}$", title="latents of the trials drawn below")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    return fig


trials = spread_trials(order, PHI_TRIALS)
plot_trial_latents(TEST_DATASET.factors, trials)
plt.show()"""

PHI_PAIRS_TEXT = r"""### $\Phi^{(l)}$ between pairs of trials

One figure per hidden layer $l$: an $S \times S$ grid whose panel in row $s$ and column $t$ is
the $(n_\mathrm{post} \times n_\mathrm{post})$ matrix over the post-synaptic units $i, j$,
$$\big(\Phi^{(l)\,\mu_s \mu_t}_{ij}\big)_{ij} = A^{(l)}_{\mu_s \mu_t}\, K^{\mathrm{post}}_{ij},$$
units in ring order for $d = 1$ and in index order otherwise, on **one** colour scale for the
whole grid, symmetric about zero; the row and column headers give $\mu_s$ and $\mu_t$. Under a
factorised prior every panel is the same $K^{\mathrm{post}}$ scaled by the number
$A^{(l)}_{\mu_s \mu_t}$ (eq. phi-diagonalised of the paper): the prior on the post-synaptic
weights shapes how the input to the next layer is spread *over space* ($i, j$), the pre-synaptic
prior and the activity shape how it is spread *over data* ($\mu, \nu$). So the pattern inside a
panel never changes; its strength and sign from panel to panel read off $A^{(l)}$ at the chosen
pairs, the diagonal panels being $A^{(l)}_{\mu\mu} \ge 0$ times $K^{\mathrm{post}}$."""

PHI_PAIRS = r"""def layer_kernel_pairs(A, spaces, layer, trials):
    # The population that hidden layer `layer` (0-based) drives, the post side of spaces[layer + 1]
    # (the output for the last hidden layer), and the (S, S, n, n) blocks A[mu, nu] K_post.
    post = spaces[layer + 1].post
    return post, spatial_kernel_pairs(A[layer], post.K, [int(t) for t in trials])


def plot_spatial_kernel_pairs(pairs, trials, unit_order, title):
    # The S x S grid of (n, n) blocks Phi^{mu nu}_ij, units in `unit_order`, one colour scale
    # for the whole grid, symmetric about zero.
    values = pairs.detach().cpu().numpy()
    if values.ndim != 4 or values.shape[0] != values.shape[1] or values.shape[2] != values.shape[3]:
        raise ValueError(f"pairs must be (S, S, n, n), got shape {values.shape}")
    n_trials, n_units = values.shape[0], values.shape[2]
    if len(trials) != n_trials:
        raise ValueError(f"{len(trials)} trials for a {n_trials} x {n_trials} grid")
    if len(unit_order) != n_units:
        raise ValueError(f"the unit order has {len(unit_order)} entries for {n_units} units")
    shown = values[:, :, unit_order][:, :, :, unit_order]
    # An all-zero grid would give an empty colour range; show it on -1..1.
    bound = np.abs(shown).max() or 1.0
    size = 2.4 * n_trials
    fig, axes = plt.subplots(
        n_trials, n_trials, figsize=(size + 1.2, size), squeeze=False, layout="constrained"
    )
    for (s, t), ax in np.ndenumerate(axes):
        image = ax.imshow(shown[s, t], cmap="RdBu_r", vmin=-bound, vmax=bound, interpolation="none")
        ax.set_xticks([])
        ax.set_yticks([])
    for t, ax in enumerate(axes[0]):
        ax.set_title(rf"$\nu$ = {trials[t]}")
    for s, ax in enumerate(axes[:, 0]):
        ax.set_ylabel(rf"$\mu$ = {trials[s]}")
    fig.colorbar(image, ax=axes, shrink=0.6, label=r"$\Phi^{\mu\nu}_{ij}$")
    fig.suptitle(title)
    return fig


units = "ring position" if D == 1 else "index"
for layer in range(RUN.net.L):
    post, pairs = layer_kernel_pairs(A, SPACES, layer, trials)
    unit_order = position_order(post.embedding.positions) if D == 1 else np.arange(len(post.K))
    title = (
        rf"hidden layer {layer + 1} into {post.name}: $\Phi^{{({layer + 1})\,\mu\nu}}_{{ij}}$, "
        f"units $i, j$ of {post.name} by {units}"
    )
    plot_spatial_kernel_pairs(pairs, trials, unit_order, title)
plt.show()"""


TUNING_TEXT = r"""## Tuning of every neuron to the latents

For hidden layer $l$, neuron $i$ and latent $f$, the samples are split into $B$ = `N_BINS`
equal-count bins $b$ by the rank of $z_{f\mu}$, and the curve is the bin mean of the rate against
the bin mean of the latent, the other latents averaged out:
$$\bar r_{if}(b) = \frac{1}{|b|} \sum_{\mu \in b} r^{(l)}_{i\mu}, \qquad \bar z_f(b) = \frac{1}{|b|} \sum_{\mu \in b} z_{f\mu},$$
with error bars of one standard error. Each title gives the neuron's position and its MIR:
$I_{if}$ is the plug-in mutual information (nats) of the rank-binned rate with the rank-binned
latent, both in $B$ bins. With `MIR_KIND = "corrected"`, $I_{if}$ is that minus its mean over
`N_SHUFFLES` label shuffles, clamped at 0; and an $I_{if}$ whose raw value is not above
`NOISE_FACTOR` (2, in `fct/modularity.py`) times its shuffled-null mean counts as noise and is set
to 0, each latent on its own (the library tests groups of factors on their summed information;
here every latent is its own group). A neuron with every $I_{if} = 0$ gets $\rho_i = 1/k$. Then
$$\rho_i = \frac{\max_f I_{if}}{\sum_f I_{if}}, \qquad \mathrm{MIR}_i = \frac{\rho_i - 1/k}{1 - 1/k},$$
with $k$ the number of latents; the layer's MIR is the mean over its neurons. One panel per neuron,
placed by its position (ring order along rows of `N_COLS` for $d = 1$, its lattice cell for
$d = 2$), all on one rate axis: a modular neuron has one sloped curve and the others flat.

Beside each neuron's curves, a column chart gives its information with every latent: one bar per
latent $f$ at $x = f$, in the colour of that latent's curve. The pale bar is the raw plug-in
$I_{if}$, the full-colour bar is the estimate the MIR uses (the same for `"raw"`), and for
`"corrected"` a black mark is the shuffled-null mean of that latent. All bar panels share one axis
in nats.

The plug-in estimate is biased upward even for a neuron independent of the latent: about
$$\mathbb{E}[\hat I] \approx \frac{(B-1)^2}{2P}\ \text{nats},$$
with $P$ the number of samples. Before each layer's figure a line gives $P$, $B$, this bias and the
mean shuffled null over neurons and latents. When the null marks are close to the raw bars, $B$ is
too large for $P$: the informations are mostly bias, and a neuron none of whose latents has raw
information above `NOISE_FACTOR` times its null gets MIR 0."""

TUNING = r"""N_BINS = 8  # B: rank bins of each latent and of the rates, as in the MIR
N_SHUFFLES = 5  # label shuffles of the bias-corrected MIR
MIR_KIND = "corrected"  # "corrected" or "raw"
N_COLS = 4  # panels per row on a ring; a plane lattice uses its own rows and columns
BAR_WIDTH = 0.8  # width of the raw information bar of a latent (x spacing is 1)


def _draw_information_bars(ax, raw, used, null, names, bar_width):
    # One bar per latent at x = f: the raw plug-in information pale, the estimate the MIR uses
    # (`used`) full colour and half as wide, and the shuffled null mean as a black mark spanning
    # the raw bar (`null` is None when no null was computed).
    x = np.arange(len(names))
    colours = [f"C{f}" for f in x]
    ax.bar(x, raw, width=bar_width, color=colours, alpha=0.35)
    ax.bar(x, used, width=bar_width / 2, color=colours, alpha=1.0)
    if null is not None:
        ax.hlines(null, x - bar_width / 2, x + bar_width / 2, color="black", linewidth=2)
    ax.set_xticks(x, names)
    ax.grid(alpha=0.25, axis="y")


def plot_tuning_by_position(view, positions, n_cols, bar_width):
    # Two panels per neuron at its place in space. Left: one curve per latent, the bin means of
    # the rate against the bin means of the latent, +- one standard error. Right: the neuron's
    # information with each latent (`_draw_information_bars`). `view` is a `TuningView`.
    curves, mir, names = view.curves, view.neuron_mir, view.latent_names
    mean = curves.mean.detach().cpu().numpy()  # (N, K, B)
    sem = curves.sem.detach().cpu().numpy()  # (N, K, B)
    centres = curves.centres.detach().cpu().numpy()  # (K, B)
    xy = positions.detach().cpu().numpy()  # (N, d)
    raw = view.raw_mi.detach().cpu().numpy()  # (N, K) nats
    used = view.member_mi.detach().cpu().numpy()  # (N, K)
    null = None if view.null_mi is None else view.null_mi.detach().cpu().numpy()  # (N, K)
    if not len(mean) == len(xy) == len(mir):
        raise ValueError(f"{len(mean)} neurons but {len(xy)} positions and {len(mir)} MIR values")
    if len(names) != mean.shape[1]:
        raise ValueError(f"{len(names)} latent names for {mean.shape[1]} latents")
    informations = [raw, used] if null is None else [raw, used, null]
    if any(m.shape != mean.shape[:2] for m in informations):
        raise ValueError(
            f"informations of shape {[m.shape for m in informations]}, need {mean.shape[:2]}"
        )
    grid = panel_grid(positions, n_cols)
    fig, axes = plt.subplots(
        grid.n_rows,
        2 * grid.n_cols,
        figsize=(15, 3.2 * grid.n_rows + 0.8),
        squeeze=False,
        layout="constrained",
    )
    for ax in axes.ravel():
        ax.set_visible(False)  # a ring of N units may leave the last row part empty
    # The neuron at cell (row, col) has its curves at grid column 2 col and its bars next to them.
    curve_axes = [axes[row, 2 * col] for row, col in grid.cells]
    bar_axes = [axes[row, 2 * col + 1] for row, col in grid.cells]
    for group in (curve_axes, bar_axes):  # one rate axis, and one information axis in nats
        for ax in group[1:]:
            ax.sharey(group[0])
    for neuron, (row, col) in enumerate(grid.cells):
        ax, bars = curve_axes[neuron], bar_axes[neuron]
        ax.set_visible(True)
        bars.set_visible(True)
        for f, name in enumerate(names):
            ax.errorbar(
                centres[f],
                mean[neuron, f],
                yerr=sem[neuron, f],
                color=f"C{f}",
                marker="o",
                capsize=2,
                label=name,
            )
        _draw_information_bars(
            bars, raw[neuron], used[neuron], None if null is None else null[neuron], names, bar_width
        )
        where = ", ".join(f"{c:.2f}" for c in xy[neuron])
        ax.set_title(f"neuron {neuron} at ({where})\nMIR {mir[neuron]:.2f}")
        ax.grid(alpha=0.25)
        if col > 0:  # the shared y axes are labelled on the first neuron column only
            ax.tick_params(labelleft=False)
            bars.tick_params(labelleft=False)
    for ax in axes[-1, ::2]:
        ax.set_xlabel(r"latent $\bar z_f(b)$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"rate $\bar r_{if}(b)$")
    for ax in axes[:, 1]:
        ax.set_ylabel(r"$I_{if}$ (nats)")
    axes[min(grid.cells)[0], 2 * min(grid.cells)[1]].legend(frameon=False)
    return fig


for layer in range(RUN.net.L):
    view = tuning_view(RUN.net, RUN.dataset, layer, N_BINS, MIR_KIND, N_SHUFFLES, TRAIN_SEED)
    n_samples = len(RUN.dataset.x)
    null_note = (
        ""
        if view.null_mi is None
        else f", mean shuffled null {view.null_mi.mean().item():.3f} nats"
    )
    print(
        f"layer {layer + 1}: P = {n_samples}, B = {N_BINS}, expected plug-in bias "
        f"(B-1)^2 / (2P) = {(N_BINS - 1) ** 2 / (2 * n_samples):.3f} nats per latent{null_note}"
    )
    positions = SPACES[layer].post.embedding.positions
    fig = plot_tuning_by_position(view, positions, N_COLS, BAR_WIDTH)
    fig.suptitle(
        f"hidden layer {layer + 1}: tuning of every neuron, by position "
        f"({MIR_KIND} MIR, layer mean {np.mean(view.neuron_mir):.3f})"
    )
plt.show()"""


OVERLAP_TEXT = r"""## Overlap of the neurons' information and of their mean rates

How much two neurons of a hidden layer are tuned to the same latents, measured two ways. With
$I_{if}$ the mutual information (nats) of neuron $i$ with latent $f$ exactly as in the tuning
section (the `MIR_KIND` estimate with `N_BINS` and `N_SHUFFLES`, so shuffle-corrected for
`"corrected"`), the information overlap of neurons $i$ and $j$ of layer $l$ is
$$C^{I}_{ij} = \frac{1}{k} \sum_{f=1}^{k} I_{if}\, I_{jf},$$
with $k$ the number of latents, in nats$^2$. With $\bar r_{if}(b)$ the mean rate of neuron $i$ in
bin $b$ of latent $f$ (the tuning curves above), the rate overlap is
$$C^{r}_{ij} = \frac{1}{n_\mathrm{valid}} \sum_{f,b} \bar r_{if}(b)\, \bar r_{jf}(b),$$
summed over the $n_\mathrm{valid}$ pairs $(f, b)$ with a non-empty bin (all $kB$ of them for
equal-count rank bins), in rate$^2$. Both are symmetric. $C^I_{ii}$ is large for a neuron that is
informative about some latent, and an off-diagonal entry is large only when both neurons are
informative about a common latent, so a modular layer shows blocks of neurons sharing a latent. A
neuron whose latents all count as noise in the tuning section has all $I_{if} = 0$ (for
`"corrected"`), hence an all-zero row and column in $C^I$. $C^r$ does not use the information
estimate: it is signed (tanh rates), positive for neurons tuned alike and negative for opposite
tuning, and it also has weight
where the tuning is a constant offset, since $\bar r$ is not centred. Compare the two: blocks that
appear in $C^r$ but not in $C^I$ are neurons with similar tuning whose information the estimate did
not resolve. One figure per hidden layer, $C^I$ left and $C^r$ right, neurons in ring order for
$d = 1$ and in index order otherwise (the ticks give the neuron index, as in the titles of the
tuning panels). $C^I$ is on one colour scale starting at 0, $C^r$ on one scale symmetric about 0,
each shared by all layers. If every $C^I$ is zero its scale is 0 to 1 and the title says "(no
information)", so an empty matrix is not read as data; if every $C^r$ is zero its scale is $-1$ to
1."""

OVERLAP = r"""def plot_overlaps(C_info, C_rate, order, info_max, rate_max, layer, kind):
    # The information overlap (viridis, 0 to info_max) and the rate overlap (diverging, +-rate_max)
    # of one layer side by side, both (N, N) with their neurons shown in `order`.
    info, rate = (C.detach().cpu().numpy() for C in (C_info, C_rate))
    for values in (info, rate):
        if values.ndim != 2 or values.shape[0] != values.shape[1]:
            raise ValueError(f"C must be a square matrix, got shape {values.shape}")
    if info.shape != rate.shape:
        raise ValueError(
            f"the two overlaps must be of the same neurons: {info.shape} vs {rate.shape}"
        )
    if len(order) != len(info):
        raise ValueError(f"need one order entry per neuron: {len(order)} for {len(info)}")
    informative = info_max > 0
    # A layer with no information (all C = 0) would give an empty colour range; show 0..1.
    info_top = info_max if informative else 1.0
    # Likewise for a layer whose rate overlap is all zero: the scale is -1 to 1.
    rate_top = rate_max if rate_max > 0 else 1.0
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    shown = np.ix_(order, order)
    left = axes[0].imshow(info[shown], cmap="viridis", vmin=0, vmax=info_top)
    right = axes[1].imshow(rate[shown], cmap="RdBu_r", vmin=-rate_top, vmax=rate_top)
    fig.colorbar(left, ax=axes[0], label=r"$C^I_{ij}$  (nats$^2$)")
    fig.colorbar(right, ax=axes[1], label=r"$C^r_{ij}$  (rate$^2$)")
    ticks = np.arange(0, len(order), max(1, int(np.ceil(len(order) / 8))))
    label = "neuron, by ring position" if kind == "ring" else "neuron, by index"
    titles = (
        "information overlap" + ("" if informative else " (no information)"),
        "rate overlap",
    )
    for ax, title in zip(axes, titles):
        ax.set_xticks(ticks, np.asarray(order)[ticks])  # the neuron index at each shown position
        ax.set_yticks(ticks, np.asarray(order)[ticks])
        ax.set(xlabel=label, ylabel=label, title=title)
    fig.suptitle(f"hidden layer {layer + 1}")
    return fig


views = [
    tuning_view(RUN.net, RUN.dataset, layer, N_BINS, MIR_KIND, N_SHUFFLES, TRAIN_SEED)
    for layer in range(RUN.net.L)
]
info_overlaps = [information_overlap(view.member_mi) for view in views]
rate_overlaps = [rate_overlap(view.curves.mean) for view in views]
orders = []
for layer in range(RUN.net.L):
    positions = SPACES[layer].post.embedding.positions
    orders.append(position_order(positions) if D == 1 else np.arange(len(positions)))
info_max = max(float(C.max()) for C in info_overlaps)
rate_max = max(float(C.abs().max()) for C in rate_overlaps)
for layer, (C_info, C_rate, order) in enumerate(zip(info_overlaps, rate_overlaps, orders)):
    plot_overlaps(C_info, C_rate, order, info_max, rate_max, layer, "ring" if D == 1 else "index")
plt.show()"""


def build_notebook() -> NotebookNode:
    """The notebook without outputs: settings, the prior, training, then one markdown cell with
    the equation and one code cell with the plotting function per figure."""
    nb = new_notebook(
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        }
    )
    cells = [
        new_markdown_cell(INTRO),
        new_markdown_cell("## Task and network"),
        new_code_cell(TASK_AND_NET, metadata={"tags": ["parameters"]}),
        new_markdown_cell(TRAINING_TEXT),
        new_code_cell(TRAINING),
        new_markdown_cell(SETUP_TEXT),
        new_code_cell(IMPORTS),
        new_code_cell(TRAIN_IMPORTS),
        new_markdown_cell(PRIOR_TEXT),
        new_code_cell(SPATIAL),
        new_markdown_cell(FLOOR_TEXT),
        new_code_cell(FLOOR),
        new_markdown_cell(KERNELS_TEXT),
        new_code_cell(KERNELS),
        new_markdown_cell("## Train"),
        new_code_cell(TRAIN),
        new_markdown_cell(LOSS_TEXT),
        new_code_cell(LOSS),
        new_markdown_cell(EXAMPLES_TEXT),
        new_code_cell(EXAMPLES),
        new_markdown_cell(EMPIRICAL_TEXT),
        new_code_cell(EMPIRICAL),
        new_markdown_cell(WEIGHTS_TEXT),
        new_code_cell(WEIGHTS),
        new_markdown_cell(DATA_KERNEL_TEXT),
        new_code_cell(DATA_KERNEL),
        new_markdown_cell(TRIALS_TEXT),
        new_code_cell(TRIALS),
        new_markdown_cell(PHI_PAIRS_TEXT),
        new_code_cell(PHI_PAIRS),
        new_markdown_cell(TUNING_TEXT),
        new_code_cell(TUNING),
        new_markdown_cell(OVERLAP_TEXT),
        new_code_cell(OVERLAP),
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
    nb: NotebookNode,
    cache_root: Path,
    overrides: Mapping[str, Any] | None = None,
    root_name: str = "CACHE_ROOT",
) -> NotebookNode:
    """Run a copy of `nb` in a fresh kernel with the parameter `root_name` (CACHE_ROOT, or the
    experiment ROOT of the spatial-experiment notebook) and the `overrides` set; the source `nb`
    gains no outputs.
    A failing cell raises (nbclient.exceptions.CellExecutionError)."""
    values = {root_name: str(cache_root), **(overrides or {})}
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


def build_main(
    build: Callable[[], NotebookNode],
    source_out: Path,
    executed_out: Path,
    description: str,
    set_example: str,
    argv: list[str] | None = None,
    root_name: str = "CACHE_ROOT",
    root_help: str = "directory for the cached runs",
) -> int:
    """The command line shared by the notebook builders that take a root directory (the
    notebook parameter `root_name`) and `--set` overrides: build the notebook, execute it
    unless --no-execute, write it."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("cache_root", nargs="?", type=Path, metavar=root_name, help=root_help)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        type=_parse_override,
        metavar="NAME=VALUE",
        help=f"set a parameter of the notebook (a Python literal), e.g. {set_example}",
    )
    parser.add_argument("--out", type=Path, help="default: the source notebook, or tmp/ if run")
    parser.add_argument("--no-execute", action="store_true", help="write the source only")
    args = parser.parse_args(argv)
    nb = build()
    if not args.no_execute:
        if args.cache_root is None:
            parser.error(f"give a {root_name}")
        nb = execute_notebook(nb, args.cache_root, dict(args.overrides), root_name)
    out = args.out or (source_out if args.no_execute else executed_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"wrote {write_notebook(nb, out)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    description = __doc__.split("\n\n")[0]
    return build_main(
        build_notebook, SOURCE_OUT, EXECUTED_OUT, description, "OPTIMIZER='adam'", argv
    )


if __name__ == "__main__":
    raise SystemExit(main())
