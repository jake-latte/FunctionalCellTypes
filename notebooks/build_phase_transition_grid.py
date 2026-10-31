"""Build and execute `phase_transition_grid.ipynb`: the figures of `spatial_grid.ipynb` for one
setting of a `phase_transition` sweep, an ensemble of B networks with random embeddings in d
dimensions.

Give a sweep ROOT (made with `python -m fct sweep init ROOT --preset phase_transition`) and the
axis values of one setting, `SETTING = {"spatial.d": 2, "spatial.lambda_max": 0.5}`. The notebook
lists the grid, loads that setting through `fct.phase_view.load_phase_run` and draws the prior
(covariance, effective covariance and penalty, with their spectra and the eigenvalue floor), the
ensemble's loss curves, example outputs, the trained weight kernels, the data kernels A, the tuning
of one member and the information and rate overlaps. A, the weight kernels and the overlaps come
for member `MEMBER` and for the ensemble average. A setting that is not trained is reported in the
cells and only its prior is drawn.

The computation is `fct/phase_view.py`; the figures are drawn by plotting functions written out in
the cells that draw them. The plotting functions of the data-kernel figure are those of
`build_spatial.py` (taken verbatim, so the notebooks cannot drift); the others are written for the
ensemble. Run

    python notebooks/build_phase_transition_grid.py ROOT --set 'SETTING={"spatial.d": 2}'   # execute, write tmp/phase_transition_grid.ipynb
    python notebooks/build_phase_transition_grid.py --no-execute                              # source only: notebooks/phase_transition_grid.ipynb

ROOT is the sweep directory (the one holding sweep.json); `$FCT_SWEEP_ROOT` works in the notebook
too. Only the source notebook is committed; an executed one records the ROOT path and its figures
are large, so it goes to the git-ignored tmp/ unless --out says otherwise.
"""

import ast
from pathlib import Path

from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from notebooks import build_spatial as sp
from notebooks.build_spatial_grid import GRID, GRID_TEXT, when_trained

SOURCE_OUT = Path(__file__).with_name("phase_transition_grid.ipynb")
EXECUTED_OUT = Path(__file__).parents[1] / "tmp" / "phase_transition_grid.ipynb"


def function_source(source: str, *names: str) -> str:
    """The definitions of the functions `names` in `source` (a cell's code), verbatim and in the
    order of `source`. ValueError if one is missing."""
    lines = source.splitlines()
    found = {
        node.name: "\n".join(lines[node.lineno - 1 : node.end_lineno])
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    }
    missing = set(names) - set(found)
    if missing:
        raise ValueError(f"no function {sorted(missing)} in the cell")
    return "\n\n\n".join(found[n] for n in names if n in found)


INTRO = (
    r"""# The phase-transition grid: the figures of the spatial grid, for an ensemble

`ROOT` is a sweep made with `python -m fct sweep init ROOT --preset phase_transition`: settings
that differ in the embedding dimension $d$ and in the top eigenvalue of the kernels
$\lambda_{\max}$, each an ensemble of $B$ networks that share their random embeddings and differ in
their initial weights and Langevin noise. `SETTING` is the axis values of one setting, e.g.
`{"spatial.d": 2, "spatial.lambda_max": 0.5}`; a subset of the axes is enough when it picks one.
The notebook lists the grid, loads that setting and draws what `spatial_grid.ipynb` draws for one
network, adapted to an ensemble:

- **the prior** of every weight layer: the covariance $K / \lambda_w$, the effective covariance the
  networks were trained under (the covariance with its small eigenvalues raised to the floor) and
  the penalty $\lambda_w M$, side by side, and the leading eigenvalues of their factors with the
  floor drawn;
- **the loss curves** as the mean over members with a band of one standard error;
- **example outputs** of every member on the same axes;
- **the trained weight kernels**, the data kernels $A^{(l)}$ and the **information and rate
  overlaps** of the neurons, each for member `MEMBER` and for the ensemble average;
- **the tuning of the neurons of member `MEMBER`** to the latents, to the first `N_NEURONS`.

Three things differ from the spatial grid. Tuning and the overlaps use a fresh random test set of
`P_TEST` samples of the training set's own teacher, not the training samples (a larger set lowers
the plug-in noise floor of the mutual information); the data kernels keep the held-out latent
lattice of $3^k$ points. Every view that orders units uses their coordinate along axis `SORT_DIM`
(see below). The first-layer weight image and the trial and pair grids of the spatial notebook
are not drawn.

**Ordering units.** Units are sorted by coordinate `SORT_DIM` of their random position in $d$
dimensions. For $d = 1$ that is the position on the ring. For $d > 1$ it is one coordinate, **not a
ring position**: two units that are neighbours in this order can be far apart along the other
axes, so a band along the diagonal of a sorted matrix is weaker evidence of a smooth kernel than
on a ring, and a kernel that depends on the full distance looks blurred. General distance measures
come later. `SORT_DIM` outside $0, \dots, d - 1$ raises an error.

A setting that was never trained is reported in the cells and only its prior is drawn; a setting
that is still running is drawn from its latest checkpoint and labelled provisional.

"""
    + sp.NOTATION
)

PARAMETERS = """ROOT = None  # the sweep ROOT (holds sweep.json); None reads $FCT_SWEEP_ROOT
SETTING = {"spatial.d": 2, "spatial.lambda_max": 0.5}  # axis values of the setting: a subset of the grid's axes that picks one
MEMBER = 0  # the ensemble member of the single-network figures (0 .. B - 1)
SORT_DIM = 0  # units are ordered by this coordinate of their position (0 .. d - 1)
P_TEST = 2000  # samples of the random test set used for tuning and the overlaps
N_EIG = 20  # eigenvalues drawn per spectrum
N_NEURONS = 16  # neurons of the tuning figure: the first N_NEURONS in SORT_DIM order
DEVICE = "cpu"  # "cpu", "cuda" or "auto": where the nets and the test activities are computed
# the tuning and the information estimate
N_BINS = 8  # rank bins of each latent and of the rates, as in the MIR
N_SHUFFLES = 5  # label shuffles of the bias-corrected MIR
MIR_KIND = "corrected"  # "corrected" or "raw"
N_COLS = 4  # neurons per row of the tuning figure
N_EXAMPLES = 4  # samples of the example-output panels
PHI_SORT_LATENT = 0  # 0-based latent z_f that orders the lattice points of the A figure; None: lattice order"""

IMPORTS = """import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import Markdown, display

from fct.analyse import pick_examples, r_squared
from fct.grid_view import tuning_view
from fct.phase_view import (
    average_views,
    ensemble_example_outputs,
    ensemble_statistics,
    ensemble_views,
    layer_priors,
    load_phase_run,
    random_test_set,
    unit_order,
)
from fct.report import sweep_root_from_env
from fct.spatial import latent_kernel
from fct.spatial_grid import settings_table
from fct.tasks import latent_lattice_dataset
from fct.train import prior_coefficient

mpl.rcParams.update({"figure.dpi": 100, "font.size": 12, "axes.titlesize": 12})

SWEEP_ROOT = ROOT or sweep_root_from_env()"""

SETTING_TEXT = """## The chosen setting

The setting's config gives the spatial prior (`SPACES`) whether or not it is trained; its ensemble
of nets, training history, the random test set and the lattice test set are built only when
there are weights. `MEMBER` and `SORT_DIM` are checked here, against the ensemble size and the
embedding dimension, so a bad value stops the notebook before any figure. The notation and
training text above tell the reader to set `PRIOR_COEFF`, `DATA_SEED`, `RATE_LAMBDA`, `BETA` and
`OPTIMIZER`; in this notebook they are read from the setting's config instead."""

LOADER = r"""RUN = load_phase_run(SWEEP_ROOT, SETTING, DEVICE)
CHOSEN, CFG, SPACES, NET = RUN.setting, RUN.setting.cfg, RUN.spaces, RUN.net
tc = CFG.train
# the names the spatial notebook's cells read
OPTIMIZER, BETA, LR, TRAIN_SEED = tc.optimizer, tc.beta, tc.lr, tc.seed
PRIOR_COEFF = prior_coefficient(tc)
WEIGHT_LAMBDA = tc.weight_lambda
D, BOX, MIN_REL_EIG = CFG.spatial.d, CFG.spatial.box, CFG.spatial.min_rel_eig
N_MEMBERS = CFG.net.B
if not 0 <= MEMBER < N_MEMBERS:
    raise ValueError(f"MEMBER must lie in 0..{N_MEMBERS - 1} for B = {N_MEMBERS}, got {MEMBER}")
# populations in layer order (input, hidden layers, output), each with its units' order along
# SORT_DIM; unit_order raises for a SORT_DIM outside 0..d-1
POPULATIONS = [SPACES[0].pre, *(space.post for space in SPACES)]
ORDERS = [unit_order(pop.embedding.positions, SORT_DIM) for pop in POPULATIONS]
print(f"units sorted by coordinate {SORT_DIM} of d = {D}")
PRIORS = layer_priors(SPACES, WEIGHT_LAMBDA, N_EIG)
TRAINED = NET is not None
where = ", ".join(f"{axis} = {value}" for axis, value in CHOSEN.setting.axes.items())
if TRAINED:
    # the held-out lattice of the A figure, and fresh samples of the training teacher for tuning
    LATTICE = latent_lattice_dataset(CFG.data, RUN.dataset)
    TEST = random_test_set(CFG.data, RUN.dataset, P_TEST)
    print(f"random test set: {P_TEST} points (data seed {TEST.seed})")
    print(f"lattice test set: {len(LATTICE.x)} points")
    VIEWS = ensemble_views(NET, LATTICE, TEST.dataset, SPACES, N_BINS, MIR_KIND, N_SHUFFLES, TRAIN_SEED)
    MEMBER_VIEW, MEAN_VIEW = VIEWS[MEMBER], average_views(VIEWS)
    NOT_TRAINED = ""
    STATS = ensemble_statistics(RUN.history)
    last = {key: STATS.mean[key][-1] for key in ("loss", "mse", "weight_l2", "rate_l2")}
    sampler = f"SGLD, beta = {tc.beta:g}" if tc.optimizer == "sgld" else "Adam"
    provisional = (
        ""
        if CHOSEN.status == "done"
        else f" **Provisional: the setting is still running, drawn from its checkpoint at epoch {RUN.epoch}.**"
    )
    display(
        Markdown(
            f"**Setting {CHOSEN.setting.index}** ({where}), an ensemble of {N_MEMBERS} nets trained "
            f"with {sampler}, lr = {tc.lr:g}, {tc.n_epochs} epochs. Single-network figures show "
            f"member {MEMBER}. Mean over members at the last logged epoch "
            f"({STATS.epoch[-1]}): loss {last['loss']:.4g}, mse {last['mse']:.4g}, weight penalty "
            f"{last['weight_l2']:.4g}, rate penalty {last['rate_l2']:.4g}." + provisional
        )
    )
else:
    VIEWS = MEMBER_VIEW = MEAN_VIEW = STATS = LATTICE = TEST = None
    NOT_TRAINED = (
        f"**Setting {CHOSEN.setting.index} ({where}) is not trained** (status: {CHOSEN.status}): "
        "only its prior is drawn; the figures of the trained nets are skipped."
    )
    display(Markdown(NOT_TRAINED))"""

PRIOR_TEXT = (
    sp.PRIOR_TEXT
    + r"""

### What the prior figures plot

Weight layer $l$ has prior covariance $K^{\mathrm{post}}_l \otimes K^{\mathrm{pre}}_l$ and
penalty $\frac{\lambda_w}{2\beta}\,\mathrm{tr}(M^{\mathrm{post}}_l W_l M^{\mathrm{pre}}_l W_l^\top)$.
Three Kronecker products are drawn, each as its two factors, with the overall scale $\lambda_w$
split as $\sqrt{\lambda_w}$ on each factor (so the product of the plotted factors carries
$\lambda_w$ exactly once; under `lambda_max` the scale is 1):

| matrix | factors drawn (post, pre) | their Kronecker product |
|---|---|---|
| covariance | $K^p / \sqrt{\lambda_w}$ | $K^{\mathrm{post}} \otimes K^{\mathrm{pre}} / \lambda_w$ |
| effective covariance | $(M^p)^{-1} / \sqrt{\lambda_w}$ | $(\lambda_w M^{\mathrm{post}} \otimes M^{\mathrm{pre}})^{-1}$ |
| penalty | $\sqrt{\lambda_w}\, M^p$ | $\lambda_w\, M^{\mathrm{post}} \otimes M^{\mathrm{pre}}$ |

The covariance is the prior as configured. The penalty is the loss's quadratic form without its
$1 / 2\beta$, everything else about the scale included. Because the penalty floors the small
eigenvalues of $K$, the covariance the networks were **trained under** is the effective one, the
covariance with every eigenvalue below the floor $\epsilon\, \lambda_{\max}(K^p)$ raised to it; the
two differ exactly where the floor acts. The weight kernels below are compared with the effective
covariance for that reason."""
)

PRIOR_MATRICES = r"""def plot_prior_matrices(prior, space, order_post, order_pre, sort_dim):
    # One weight layer's prior: rows are the pre and the post factor, columns the covariance, the
    # effective covariance (the floored one the nets were trained under) and the penalty. Units
    # are in the given orders; each panel has its own colour scale.
    columns = [
        ("covariance", prior.covariance, r"$K^{{\rm {w}}} / \sqrt{{\lambda_w}}$"),
        (
            "effective covariance (floored, as trained)",
            prior.effective_covariance,
            r"$(M^{{\rm {w}}})^{{-1}} / \sqrt{{\lambda_w}}$",
        ),
        ("penalty", prior.penalty, r"$\sqrt{{\lambda_w}}\, M^{{\rm {w}}}$"),
    ]
    rows = [("pre", space.pre, order_pre), ("post", space.post, order_post)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    for row, (which, pop, order) in enumerate(rows):
        for col, (name, pair, symbol) in enumerate(columns):
            ax = axes[row, col]
            matrix = getattr(pair, which).numpy()
            image = ax.imshow(matrix[np.ix_(order, order)], cmap="viridis", interpolation="none")
            fig.colorbar(image, ax=ax, shrink=0.8)
            ax.set(
                title=f"{name}, {which} ({pop.name})\n" + symbol.format(w=which),
                xlabel=f"unit, by coordinate {sort_dim}",
                ylabel=f"unit, by coordinate {sort_dim}",
            )
    fig.suptitle(f"weight layer {space.layer + 1}: prior factors, side by side")
    return fig


for space, prior in zip(SPACES, PRIORS):
    plot_prior_matrices(prior, space, ORDERS[space.layer + 1], ORDERS[space.layer], SORT_DIM)
plt.show()"""

SPECTRA_TEXT = r"""### Spectra of the prior factors

The leading eigenvalues of each factor (the Kronecker product's eigenvalues are products of one
eigenvalue of each factor), $j = 1, \dots, $ `N_EIG`. Top row: the covariance factor
$K^p / \sqrt{\lambda_w}$ and the effective one, the `N_EIG` **largest** eigenvalues, largest first,
on a log axis; the dotted line is the floor
$$\frac{\epsilon\, \lambda_{\max}(K^p)}{\sqrt{\lambda_w}},$$
below which the effective covariance has no eigenvalue, so the covariance's spectrum bends away from
the effective one where it crosses the line (eigenvalues of $K$ that round-off made non-positive
are not drawn on a log axis). Bottom row: the penalty factor $\sqrt{\lambda_w}\, M^p$, whose
**largest** eigenvalues are all equal to the floor's value, the ceiling
$\sqrt{\lambda_w} / (\epsilon\, \lambda_{\max}(K^p))$ (a flat line, the dotted one), so the
informative end is the other one: the figure shows the `N_EIG` **smallest** eigenvalues of the
penalty, smallest first (the axis says so), which are the reciprocals of the covariance's largest.
Each title gives how many of the population's eigenvalues the floor raised."""

PRIOR_SPECTRA = r"""def plot_prior_spectra(prior, space):
    # Top row: the covariance and effective covariance spectra (largest eigenvalues, descending)
    # with the floor; bottom row: the penalty spectrum (smallest eigenvalues, ascending) with its
    # ceiling, which is where the floor acts on the penalty. Columns: pre and post factor.
    populations = [
        ("pre", space.pre, prior.covariance_spectra.pre, prior.effective_covariance_spectra.pre,
         prior.penalty_spectra.pre, prior.floors.pre),
        ("post", space.post, prior.covariance_spectra.post, prior.effective_covariance_spectra.post,
         prior.penalty_spectra.post, prior.floors.post),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for col, (which, pop, cov, eff, pen, floor) in enumerate(populations):
        top, bottom = axes[0, col], axes[1, col]
        j = np.arange(1, len(cov) + 1)
        top.plot(j, cov.numpy(), "o-", label=r"covariance $K / \sqrt{\lambda_w}$")
        top.plot(j, eff.numpy(), "s--", label=r"effective covariance $(M)^{-1} / \sqrt{\lambda_w}$")
        top.axhline(floor.covariance_floor, color="C3", ls=":", label="floor")
        top.set(
            yscale="log",
            xlabel=r"eigenvalue index $j$ (largest first)",
            ylabel="eigenvalue",
            title=f"{which} ({pop.name}): {floor.n_floored} of {len(pop.K)} eigenvalues floored",
        )
        bottom.plot(j, pen.numpy(), "o-", color="C2", label=r"penalty $\sqrt{\lambda_w}\, M$")
        bottom.axhline(floor.penalty_ceiling, color="C3", ls=":", label="ceiling (the floor)")
        bottom.set(
            yscale="log",
            xlabel=r"eigenvalue index $j$ (smallest first)",
            ylabel="eigenvalue",
            title=f"{which} ({pop.name}): penalty, smallest eigenvalues",
        )
        for ax in (top, bottom):
            ax.grid(alpha=0.25)
            ax.legend(frameon=False)
    fig.suptitle(f"weight layer {space.layer + 1}: first {len(populations[0][2])} eigenvalues")
    return fig


for space, prior in zip(SPACES, PRIORS):
    shown = len(prior.covariance_spectra.post)
    print(
        f"weight layer {space.layer + 1}: {shown} of {len(space.post.K)} eigenvalues drawn "
        f"for post, {len(prior.covariance_spectra.pre)} of {len(space.pre.K)} for pre"
    )
    plot_prior_spectra(prior, space)
plt.show()"""

LOSS_TEXT = r"""## Loss curves

The loss and its three raw terms as logged during training, against epoch on a log axis, each the
**mean over the $B$ members** with a band of one standard error of that mean,
$$\bar f(t) = \frac{1}{B} \sum_{b=1}^{B} f_b(t), \qquad \mathrm{sem}(t) = \frac{1}{\sqrt{B}} \sqrt{\frac{1}{B - 1} \sum_{b=1}^{B} \big(f_b(t) - \bar f(t)\big)^2},$$
for $f \in \{\mathcal{L}, \mathrm{mse}, \Omega_w, \Omega_r\}$ with
$$\mathcal{L} = \frac{N}{2}\, \mathrm{mse} + \frac{\lambda_w}{2\beta}\, \Omega_w + \lambda_r\, \Omega_r.$$
The terms are drawn unweighted, so they do not add up to the $\mathcal{L}$ curve; a term that is zero at
every epoch has no point on a log axis and is marked "(zero)" in the legend. A band that reaches
zero is cut off at the bottom of a log axis. Under SGLD the curves settle to a noisy stationary level
rather than to a minimum."""

LOSS = r"""def plot_ensemble_loss(stats, title):
    # The mean over members of the logged loss and its raw terms, each with a +- sem band.
    terms = {
        "loss": r"loss $\mathcal{L}$",
        "mse": "mse",
        "weight_l2": r"weight penalty $\Omega_w$",
        "rate_l2": r"rate penalty $\Omega_r$",
    }
    fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
    for key, label in terms.items():
        mean, sem = stats.mean[key], stats.sem[key]
        if not np.any(mean > 0):
            label += " (zero)"
        (line,) = ax.plot(stats.epoch, mean, label=label)
        ax.fill_between(
            stats.epoch, mean - sem, mean + sem, color=line.get_color(), alpha=0.25, linewidth=0
        )
    ax.set(xlabel="epoch", ylabel=r"logged value, mean $\pm$ sem", yscale="log", title=title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    return fig


plot_ensemble_loss(STATS, f"{OPTIMIZER}: training, mean $\\pm$ sem over {N_MEMBERS} members")
plt.show()"""

EXAMPLES_TEXT = r"""## Example outputs

Every member's output $\hat y^{b}_\mu = V_b^\top \phi(h^{(L)}_{b\mu}) / N$ against the target
$y_\mu = W_Y z_\mu$ for `N_EXAMPLES` points $\mu$ drawn at random from the **lattice test set**: the
$3^k$ points of the latent lattice $z \in \{-a, 0, a\}^k$, which the nets were not trained on.
All members are on the same axes, **one palette colour per member** (member $b$ has the $b$-th
colour of matplotlib's default cycle, the same in both panels); there are at most as many members
as the palette has colours, and a larger ensemble raises an error rather than repeat a colour.
Left: one panel per sample, the target $y_{\mu a}$ as a ring on a stem and each member's
$\hat y^{b}_{\mu a}$ as a diamond for every output $a = 1, \dots, m$, the members spread a little
sideways so they do not hide each other. Right: every entry $\hat y^{b}_{\mu a}$ against $y_{\mu a}$
over **all** $3^k$ test points, one colour per member, with the identity line and each member's
pooled
$$R^2_b = 1 - \frac{\sum_{\mu a} (y_{\mu a} - \hat y^{b}_{\mu a})^2}{\sum_{\mu a} (y_{\mu a} - \bar y)^2}$$
in the legend, $\bar y$ the mean target."""

EXAMPLES = r"""def plot_ensemble_outputs(outputs, everything, colours, title):
    # True output and every member's prediction on the examples (`outputs`), one colour per
    # member; then every member's entries of `everything` against the targets with their R^2.
    n_members = outputs.y_pred.shape[0]
    if n_members > len(colours):
        raise ValueError(
            f"{n_members} members but the palette has {len(colours)} colours: one colour per member"
        )
    y_true, y_pred = outputs.y_true.numpy(), outputs.y_pred.numpy()  # (n, m), (B, n, m)
    n, m = y_true.shape
    fig = plt.figure(figsize=(13, 2.2 * n + 1.5), layout="constrained")
    left, right = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    rows = left.subgridspec(n, 1)
    index = np.arange(m)
    spread = 0.6 / n_members
    first = None
    for row, mu in enumerate(outputs.indices.tolist()):
        ax = fig.add_subplot(rows[row], sharex=first, sharey=first)
        first = ax if first is None else first
        ax.vlines(index, 0, y_true[row], color="black", lw=1)
        ax.plot(index, y_true[row], "o", ms=12, mfc="none", mec="black", label=r"true $y_{\mu a}$")
        for b in range(n_members):
            offset = (b - (n_members - 1) / 2) * spread
            ax.plot(index + offset, y_pred[b, row], "D", ms=5, color=colours[b])
        ax.axhline(0, color="grey", lw=1)
        ax.set_title(f"example {mu}", loc="left")
        ax.set_ylabel("output")
        ax.margins(y=0.25)
        ax.tick_params(labelbottom=row == n - 1)
    ax.set_xlabel("output index $a$")
    ax.set_xticks(index)
    first.legend(frameon=False, loc="upper right")

    ax = fig.add_subplot(right)
    all_true = everything.y_true.numpy()
    lim = [all_true.min(), all_true.max()]
    for b in range(n_members):
        r2 = r_squared(everything.y_true, everything.y_pred[b])
        pred = everything.y_pred[b].numpy()
        ax.scatter(
            all_true.ravel(), pred.ravel(), s=14, alpha=0.6, color=colours[b],
            label=rf"member {b}: $R^2$ = {r2:.2f}",
        )
        lim = [min(lim[0], pred.min()), max(lim[1], pred.max())]
    ax.plot(lim, lim, "--", color="grey")
    ax.set(xlabel=r"true $y_{\mu a}$", ylabel=r"predicted $\hat y_{\mu a}$", aspect="equal")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.suptitle(title)
    return fig


n_samples = len(LATTICE.x)
picks = pick_examples(n_samples, min(N_EXAMPLES, n_samples), seed=0)
shown = ensemble_example_outputs(NET, LATTICE, picks)
test_outputs = ensemble_example_outputs(NET, LATTICE, list(range(n_samples)))
colours = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
plot_ensemble_outputs(
    shown, test_outputs, colours, f"{N_MEMBERS} members, lattice test set ({n_samples} points)"
)
plt.show()"""

KERNELS_TEXT = r"""## Trained weight kernels next to the prior

Every trained weight $W_l$, of shape (post, pre) with the readout stored as $V^\top$, has a
second-moment matrix over its post units and one over its pre units,
$$\hat K^{\mathrm{post}} = W W^\top / n_{\mathrm{pre}}, \qquad \hat K^{\mathrm{pre}} = W^\top W / n_{\mathrm{post}}.$$
One figure per weight layer, rows the post and the pre factor, columns: the **effective prior
covariance** the nets were trained under (see the prior section), the kernels of the weights of
**member `MEMBER`**, and the **ensemble average**, the elementwise mean over the $B$ members of
their $\hat K$ (the members share their embeddings, so unit $i$ is the same unit in all of them).
Under the prior alone (SGLD, ignoring the data term)
$\mathbb{E}[\hat K^{\mathrm{post}}] = c\, A_{\mathrm{pre}}\, K^{\mathrm{post}}$ and
$\mathbb{E}[\hat K^{\mathrm{pre}}] = c\, A_{\mathrm{post}}\, K^{\mathrm{pre}}$ with
$c = \lambda_w^{-1}$ and $A$ the diagonal of the other side's kernel, so the comparison holds up
to a scale and every panel has its own colour scale. One member is a single sample of the weights
and is noisy; the average over members is less so, and the data term pulls both away from the
prior. Units are in order of coordinate `SORT_DIM` (not a ring position when $d > 1$). These are
second moments of the *weights* over *units*, not the mean-field kernels $\Phi^{(l)}$ over
*samples*."""

KERNELS = r"""def plot_weight_kernels(panels, space, order_post, order_pre, sort_dim):
    # `panels` is [(title, kernel pair)], one column each: rows are the post and the pre kernel,
    # units in the given orders, every panel on its own colour scale.
    fig, axes = plt.subplots(
        2, len(panels), figsize=(5 * len(panels) + 1, 9), squeeze=False, layout="constrained"
    )
    for col, (title, pair) in enumerate(panels):
        for row, (which, pop, order) in enumerate(
            [("post", space.post, order_post), ("pre", space.pre, order_pre)]
        ):
            ax = axes[row, col]
            matrix = getattr(pair, which).detach().cpu().numpy()
            image = ax.imshow(matrix[np.ix_(order, order)], cmap="viridis", interpolation="none")
            fig.colorbar(image, ax=ax, shrink=0.8)
            ax.set(
                title=f"{title}: {which} ({pop.name})",
                xlabel=f"unit, by coordinate {sort_dim}",
                ylabel=f"unit, by coordinate {sort_dim}",
            )
    fig.suptitle(f"weight layer {space.layer + 1}: kernels of the trained weights and the prior")
    return fig


for space, prior in zip(SPACES, PRIORS):
    panels = [
        ("effective prior", prior.effective_covariance),
        (f"member {MEMBER}", MEMBER_VIEW.weight_kernels[space.layer]),
        (f"average of {N_MEMBERS} members", MEAN_VIEW.weight_kernels[space.layer]),
    ]
    plot_weight_kernels(panels, space, ORDERS[space.layer + 1], ORDERS[space.layer], SORT_DIM)
plt.show()"""

DATA_KERNEL_TEXT = (
    sp.DATA_KERNEL_TEXT
    + r"""

Two things are specific to this notebook. $P_\mathrm{test}$ in this section is the lattice's
$3^k$, not the parameter `P_TEST`, which sizes the random test set of the tuning and overlap
sections. And the figure is drawn twice: for the network of member `MEMBER`, and for the ensemble
average $\bar A^{(l)} = \frac{1}{B} \sum_b A^{(l)}_b$, the elementwise mean of the members' data
kernels (each computed from that member's own activities, and linear in them, so this is the
data kernel of the ensemble's mean second moment)."""
)

DATA_KERNEL = (
    function_source(sp.DATA_KERNEL, "sample_order", "plot_data_kernels")
    + r"""


by = None if PHI_SORT_LATENT is None else f"z{PHI_SORT_LATENT}"
order = sample_order(LATTICE.factors, by, len(LATTICE.x))
print(
    f"lattice test set: {len(order)} points, data kernels A of shape "
    f"{tuple(MEAN_VIEW.A.shape[1:])} per layer"
)
B_latent = latent_kernel(LATTICE.factors)
for name, view in [
    (f"member {MEMBER}", MEMBER_VIEW),
    (f"ensemble average of {N_MEMBERS} members", MEAN_VIEW),
]:
    fig = plot_data_kernels(view.A, order, by, B_latent)
    fig.suptitle(name)
plt.show()"""
)

TUNING_TEXT = r"""## Tuning of the neurons of member `MEMBER` to the latents

For hidden layer $l$, neuron $i$ and latent $f$, the **random test set** of `P_TEST` fresh samples
of the training teacher (not the training samples, and not the lattice) is split into $B_\mathrm{bin}$ =
`N_BINS` equal-count bins $b$ by the rank of $z_{f\mu}$, and the curve is the bin mean of the rate
against the bin mean of the latent, the other latents averaged out:
$$\bar r_{if}(b) = \frac{1}{|b|} \sum_{\mu \in b} r^{(l)}_{i\mu}, \qquad \bar z_f(b) = \frac{1}{|b|} \sum_{\mu \in b} z_{f\mu},$$
with error bars of one standard error. The information is the plug-in mutual information $I_{if}$
(nats) of the rank-binned rate with the rank-binned latent; with `MIR_KIND = "corrected"` it is that
minus its mean over `N_SHUFFLES` label shuffles, clamped at 0, and an $I_{if}$ whose raw value is not
above `NOISE_FACTOR` (2, in `fct/modularity.py`) times its shuffled-null mean counts as noise and is
set to 0. A neuron with every $I_{if} = 0$ gets $\rho_i = 1/k$. Then
$$\rho_i = \frac{\max_f I_{if}}{\sum_f I_{if}}, \qquad \mathrm{MIR}_i = \frac{\rho_i - 1/k}{1 - 1/k},$$
with $k$ the number of latents; the layer mean in the title is over **all** neurons of the layer.
Only the **first `N_NEURONS` neurons in order of coordinate `SORT_DIM`** are drawn (not a ring
position when $d > 1$; the title gives the neuron's index and its coordinate), laid out in rows
of `N_COLS`, one panel per neuron, all on one rate axis. Beside each neuron's curves a column chart
gives its information with every latent: the pale bar is the raw plug-in $I_{if}$, the full-colour
bar the estimate the MIR uses, and for `"corrected"` a black mark is the shuffled-null mean.

The plug-in estimate is biased upward even for a neuron independent of the latent: about
$$\mathbb{E}[\hat I] \approx \frac{(B_\mathrm{bin}-1)^2}{2P}\ \text{nats},$$
with $P$ = `P_TEST` the number of samples, so a larger random test set lowers the noise floor.
Before each layer's figure a line gives $P$, $B_\mathrm{bin}$, this bias and the mean shuffled null
over neurons and latents."""

TUNING = (
    function_source(sp.TUNING, "_draw_information_bars")
    + r"""


def plot_tuning_by_order(view, positions, order, sort_dim, n_cols, bar_width):
    # Two panels per neuron of `order` (the neurons drawn, in this order), in rows of n_cols.
    # Left: one curve per latent, the bin means of the rate against the bin means of the latent,
    # +- one standard error. Right: the neuron's information with each latent. `view` is a
    # `TuningView` of all the layer's neurons; `positions` is (N, d).
    curves, mir, names = view.curves, view.neuron_mir, view.latent_names
    mean = curves.mean.detach().cpu().numpy()  # (N, K, B)
    sem = curves.sem.detach().cpu().numpy()
    centres = curves.centres.detach().cpu().numpy()  # (K, B)
    raw = view.raw_mi.detach().cpu().numpy()  # (N, K) nats
    used = view.member_mi.detach().cpu().numpy()
    null = None if view.null_mi is None else view.null_mi.detach().cpu().numpy()
    coordinate = positions.detach().cpu().numpy()[:, sort_dim]
    if not len(mean) == len(coordinate) == len(mir):
        raise ValueError(f"{len(mean)} neurons but {len(coordinate)} positions and {len(mir)} MIR")
    if len(names) != mean.shape[1]:
        raise ValueError(f"{len(names)} latent names for {mean.shape[1]} latents")
    order = np.asarray(order)
    n_rows = -(-len(order) // n_cols)
    fig, axes = plt.subplots(
        n_rows,
        2 * n_cols,
        figsize=(15, 3.2 * n_rows + 0.8),
        squeeze=False,
        layout="constrained",
    )
    for ax in axes.ravel():
        ax.set_visible(False)  # the last row may be part empty
    curve_axes = [axes[k // n_cols, 2 * (k % n_cols)] for k in range(len(order))]
    bar_axes = [axes[k // n_cols, 2 * (k % n_cols) + 1] for k in range(len(order))]
    for group in (curve_axes, bar_axes):  # one rate axis, and one information axis in nats
        for ax in group[1:]:
            ax.sharey(group[0])
    for k, neuron in enumerate(order):
        ax, bars = curve_axes[k], bar_axes[k]
        ax.set_visible(True)
        bars.set_visible(True)
        for f, name in enumerate(names):
            ax.errorbar(
                centres[f], mean[neuron, f], yerr=sem[neuron, f],
                color=f"C{f}", marker="o", capsize=2, label=name,
            )
        _draw_information_bars(
            bars, raw[neuron], used[neuron], None if null is None else null[neuron], names, bar_width
        )
        ax.set_title(
            f"neuron {neuron}, coordinate {sort_dim} = {coordinate[neuron]:.2f}\n"
            f"MIR {mir[neuron]:.2f}"
        )
        ax.grid(alpha=0.25)
        if k % n_cols > 0:  # the shared y axes are labelled on the first neuron column only
            ax.tick_params(labelleft=False)
            bars.tick_params(labelleft=False)
    for ax in axes[-1, ::2]:
        ax.set_xlabel(r"latent $\bar z_f(b)$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"rate $\bar r_{if}(b)$")
    for ax in axes[:, 1]:
        ax.set_ylabel(r"$I_{if}$ (nats)")
    curve_axes[0].legend(frameon=False)
    return fig


for layer in range(NET.L):
    view = tuning_view(NET.member(MEMBER), TEST.dataset, layer, N_BINS, MIR_KIND, N_SHUFFLES, TRAIN_SEED + MEMBER)
    null_note = (
        "" if view.null_mi is None else f", mean shuffled null {view.null_mi.mean().item():.3f} nats"
    )
    print(
        f"layer {layer + 1}: P = {P_TEST}, B = {N_BINS}, expected plug-in bias "
        f"(B-1)^2 / (2P) = {(N_BINS - 1) ** 2 / (2 * P_TEST):.3f} nats per latent{null_note}"
    )
    drawn = ORDERS[layer + 1][:N_NEURONS]
    print(
        f"tuning: first {len(drawn)} of {len(ORDERS[layer + 1])} neurons, in order of "
        f"coordinate {SORT_DIM} of d = {D}"
    )
    positions = SPACES[layer].post.embedding.positions
    fig = plot_tuning_by_order(view, positions, drawn, SORT_DIM, N_COLS, 0.8)
    fig.suptitle(
        f"hidden layer {layer + 1}, member {MEMBER}: tuning of the first {len(drawn)} neurons "
        f"({MIR_KIND} MIR, layer mean over all neurons {np.mean(view.neuron_mir):.3f})"
    )
plt.show()"""
)

OVERLAP_TEXT = r"""## Overlap of the neurons' information and of their mean rates

How much two neurons of a hidden layer are tuned to the same latents, measured two ways, on the
random test set of `P_TEST` samples. With $I_{if}$ the mutual information (nats) of neuron $i$ with
latent $f$ exactly as in the tuning section (the `MIR_KIND` estimate with `N_BINS` and
`N_SHUFFLES`), the information overlap of neurons $i$ and $j$ of layer $l$ is
$$C^{I}_{ij} = \frac{1}{k} \sum_{f=1}^{k} I_{if}\, I_{jf},$$
with $k$ the number of latents, in nats$^2$. With $\bar r_{if}(b)$ the mean rate of neuron $i$ in
bin $b$ of latent $f$ (the tuning curves), the rate overlap is
$$C^{r}_{ij} = \frac{1}{n_\mathrm{valid}} \sum_{f,b} \bar r_{if}(b)\, \bar r_{jf}(b),$$
summed over the $n_\mathrm{valid}$ pairs $(f, b)$ with a non-empty bin, in rate$^2$. Both are
symmetric. $C^I_{ii}$ is large for a neuron informative about some latent, and an off-diagonal entry
is large only when both neurons are informative about a common latent, so a modular layer shows
blocks of neurons sharing a latent. A neuron whose latents all count as noise has an all-zero row
and column in $C^I$ (for `"corrected"`). $C^r$ does not use the information estimate: it is signed
(tanh rates), positive for neurons tuned alike and negative for opposite tuning, and it also has
weight where the tuning is a constant offset, since $\bar r$ is not centred.

One figure per hidden layer and per variant, $C^I$ left and $C^r$ right, neurons in order of
coordinate `SORT_DIM` (not a ring position when $d > 1$; the ticks give the neuron index). The
figures come twice: for **member `MEMBER`**, and for the **ensemble average**. The ensemble average
is the **mean over members of each member's own overlap matrix**,
$$\bar C^{I}_{ij} = \frac{1}{B} \sum_{b=1}^{B} \frac{1}{k} \sum_f I^{b}_{if} I^{b}_{jf},$$
and likewise for $C^r$: a second moment over members, **not** the overlap of the mean tuning (the
overlap of averaged informations would shrink wherever members disagree). With the `"corrected"`
MIR each member's informations are clamped at zero before the products, which biases the average
upward. Unit $i$ is the same unit in every member, since they share their embeddings. $C^I$ is on
one colour scale starting at 0 and $C^r$ on one scale symmetric about 0, each shared by the layers
of a variant; if every $C^I$ is zero its scale is 0 to 1 and the title says "(no information)"."""

OVERLAP = r"""def plot_overlaps(C_info, C_rate, order, info_max, rate_max, title, axis_label):
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
    titles = (
        "information overlap" + ("" if informative else " (no information)"),
        "rate overlap",
    )
    for ax, name in zip(axes, titles):
        ax.set_xticks(ticks, np.asarray(order)[ticks])  # the neuron index at each shown position
        ax.set_yticks(ticks, np.asarray(order)[ticks])
        ax.set(xlabel=axis_label, ylabel=axis_label, title=name)
    fig.suptitle(title)
    return fig


axis_label = f"neuron, by coordinate {SORT_DIM}"
for name, view in [
    (f"member {MEMBER}", MEMBER_VIEW),
    (f"ensemble average of {N_MEMBERS} members", MEAN_VIEW),
]:
    info_max = float(view.info_overlap.max())
    rate_max = float(view.rate_overlap.abs().max())
    for layer in range(NET.L):
        plot_overlaps(
            view.info_overlap[layer],
            view.rate_overlap[layer],
            ORDERS[layer + 1],
            info_max,
            rate_max,
            f"hidden layer {layer + 1}: {name}",
            axis_label,
        )
plt.show()"""


def build_notebook() -> NotebookNode:
    """The notebook without outputs: parameters, the grid, the chosen setting, the prior (always
    drawn), then the figures of the trained ensemble (a markdown cell with the equation, a code
    cell with the plotting function), guarded for an untrained setting."""
    nb = new_notebook(
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        }
    )
    cells = [
        new_markdown_cell(INTRO),
        new_code_cell(PARAMETERS, metadata={"tags": ["parameters"]}),
        new_markdown_cell(sp.SETUP_TEXT),
        new_code_cell(IMPORTS),
        new_markdown_cell(GRID_TEXT),
        new_code_cell(GRID),
        new_markdown_cell(SETTING_TEXT),
        new_code_cell(LOADER),
        new_markdown_cell(PRIOR_TEXT),
        new_code_cell(PRIOR_MATRICES),
        new_markdown_cell(SPECTRA_TEXT),
        new_code_cell(PRIOR_SPECTRA),
        new_markdown_cell(sp.FLOOR_TEXT),
        new_code_cell(sp.FLOOR),
        new_markdown_cell(sp.TRAINING_TEXT),
        new_markdown_cell(LOSS_TEXT),
        new_code_cell(when_trained(LOSS)),
        new_markdown_cell(EXAMPLES_TEXT),
        new_code_cell(when_trained(EXAMPLES)),
        new_markdown_cell(KERNELS_TEXT),
        new_code_cell(when_trained(KERNELS)),
        new_markdown_cell(DATA_KERNEL_TEXT),
        new_code_cell(when_trained(DATA_KERNEL)),
        new_markdown_cell(TUNING_TEXT),
        new_code_cell(when_trained(TUNING)),
        new_markdown_cell(OVERLAP_TEXT),
        new_code_cell(when_trained(OVERLAP)),
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
        "'SETTING={\"spatial.d\": 2}'",
        argv,
        root_name="ROOT",
        root_help="the sweep ROOT (the directory holding sweep.json)",
    )


if __name__ == "__main__":
    raise SystemExit(main())
