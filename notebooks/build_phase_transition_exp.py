"""Build and execute `phase_transition_exp.ipynb`: the figures of `spatial_exp.ipynb` for one
setting of a `phase_transition` experiment, an ensemble of B networks with random embeddings in d
dimensions.

Give an experiment ROOT (made with `python -m fct exp init ROOT --preset phase_transition`) and the
axis values of one setting, `SETTING = {"spatial.d": 2, "spatial.lambda_max": 0.5}`. The notebook
lists the experiment's settings, loads that setting through `fct.phase_view.load_phase_run` and
draws the prior (covariance, effective covariance and penalty, with their spectra and the eigenvalue
floor), the ensemble's loss curves, example outputs, the trained weight kernels, the data kernels A,
the tuning of one member and the information and rate overlaps. A, the weight kernels and the
overlaps come
for member `MEMBER` and for the ensemble average. A setting that is not trained is reported in the
cells and only its prior is drawn.

The computation is `fct/phase_view.py`; the figures are drawn by plotting functions written out in
the cells that draw them. The plotting functions of the data-kernel figure are those of
`build_spatial.py` (taken verbatim, so the notebooks cannot drift); the others are written for the
ensemble. Run

    python notebooks/build_phase_transition_exp.py ROOT --set 'SETTING={"spatial.d": 2}'   # execute, write tmp/phase_transition_exp.ipynb
    python notebooks/build_phase_transition_exp.py --no-execute                              # source only: notebooks/phase_transition_exp.ipynb

ROOT is the experiment directory (the one holding exp.json); `$FCT_EXP_ROOT` works in the notebook
too. Only the source notebook is committed; an executed one records the ROOT path and its figures
are large, so it goes to the git-ignored tmp/ unless --out says otherwise.
"""

import ast
from pathlib import Path

from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from notebooks import build_spatial as sp
from notebooks.build_spatial_exp import SETTINGS_TABLE, SETTINGS_TEXT, when_trained

SOURCE_OUT = Path(__file__).with_name("phase_transition_exp.ipynb")
EXECUTED_OUT = Path(__file__).parents[1] / "tmp" / "phase_transition_exp.ipynb"


def function_source(source: str, *names: str) -> str:
    """The definitions of the functions `names` in `source` (a cell's code), verbatim, in the order
    of `names`. ValueError if one is missing."""
    lines = source.splitlines()
    found = {
        node.name: "\n".join(lines[node.lineno - 1 : node.end_lineno])
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    }
    missing = set(names) - set(found)
    if missing:
        raise ValueError(f"no function {sorted(missing)} in the cell")
    return "\n\n\n".join(found[n] for n in names)


INTRO = (
    r"""# The phase-transition experiment: the figures of the spatial experiment, for an ensemble

`ROOT` is an experiment made with `python -m fct exp init ROOT --preset phase_transition`: settings
that differ in the embedding dimension $d$ and in the top eigenvalue of the kernels
$\lambda_{\max}$, each an ensemble of $B$ networks that share their random embeddings and differ in
their initial weights and Langevin noise. `SETTING` is the axis values of one setting, e.g.
`{"spatial.d": 2, "spatial.lambda_max": 0.5}`; a subset of the axes is enough when it picks one.
The notebook lists the experiment's settings, loads that setting and draws what `spatial_exp.ipynb`
draws for one network, adapted to an ensemble:

- **the prior** of every weight layer: the covariance $K / \lambda_w$, the effective covariance the
  networks were trained under (the covariance with its small eigenvalues raised to the floor) and
  the penalty $\lambda_w M$, side by side, and the leading eigenvalues of their factors with the
  floor drawn;
- **the loss curves** as the mean over members with a band of one standard error;
- **example outputs** of every member on the same axes;
- **the trained weight kernels**, the data kernels $A^{(l)}$ and the **information and rate
  overlaps** of the neurons, each for member `MEMBER` and for the ensemble average;
- **the tuning of the neurons of member `MEMBER`** to the latents, to the first `N_NEURONS`.

Three things differ from the spatial experiment. Tuning and the overlaps use a fresh random test set
of `P_TEST` samples of the training set's own teacher, not the training samples (a larger set lowers
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

PARAMETERS = """ROOT = None  # the experiment ROOT (holds exp.json); None reads $FCT_EXP_ROOT
SETTING = {"spatial.d": 2, "spatial.lambda_max": 0.5}  # axis values of the setting: a subset of the experiment's axes that picks one
MEMBER = 0  # the ensemble member of the single-network figures (0 .. B - 1)
SORT_DIM = 0  # units are ordered by this coordinate of their position (0 .. d - 1)
P_TEST = 2000  # samples of the random test set used for tuning and the overlaps
N_EIG = 20  # eigenvalues drawn per spectrum
N_NEURONS = 16  # neurons of the tuning figure
NEURON_PICK = "first"  # "first": the first N_NEURONS in SORT_DIM order; "spread": N_NEURONS evenly spaced along that order
DEVICE = "cpu"  # "cpu", "cuda" or "auto": where the nets and the test activities are computed
# the tuning and the information estimate
N_BINS = 8  # rank bins of each latent and of the rates, as in the MIR
N_SHUFFLES = 5  # label shuffles of the bias-corrected MIR
MIR_KIND = "corrected"  # "corrected" or "raw"
N_COLS = 4  # neurons per row of the tuning figure
N_EXAMPLES = 4  # samples of the example-output panels
N_OUTPUT_DIMS = 8  # output dimensions drawn in the example panels, evenly spaced (the scatter uses all)
PHI_SORT_LATENT = 0  # 0-based latent z_f that orders the lattice points of the A figure; None: lattice order"""

IMPORTS = """import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import Markdown, display

from fct.analyse import pick_examples, r_squared
from fct.exp_view import tuning_view
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
from fct.report import exp_root_from_env
from fct.spatial import latent_kernel
from fct.spatial_exp import settings_table
from fct.tasks import latent_lattice_dataset
from fct.train import prior_coefficient

mpl.rcParams.update({"figure.dpi": 100, "font.size": 12, "axes.titlesize": 12})

EXP_ROOT = ROOT or exp_root_from_env()"""

SETTING_TEXT = """## The chosen setting

The setting's config gives the spatial prior (`SPACES`) whether or not it is trained; its ensemble
of nets, training history, the random test set and the lattice test set are built only when
there are weights. `MEMBER` and `SORT_DIM` are checked here, against the ensemble size and the
embedding dimension, so a bad value stops the notebook before any figure. The notation and
training text above tell the reader to set `PRIOR_COEFF`, `DATA_SEED`, `RATE_LAMBDA`, `BETA` and
`OPTIMIZER`; in this notebook they are read from the setting's config instead."""

LOADER = r"""RUN = load_phase_run(EXP_ROOT, SETTING, DEVICE)
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
PRIORS = layer_priors(SPACES, WEIGHT_LAMBDA, N_EIG, whitened=CFG.is_whitened)
TRAINED = NET is not None
# appended to every figure title of a setting that is still running
STATUS_TAG = (
    f" [PROVISIONAL: checkpoint at epoch {RUN.epoch} of {tc.n_epochs}]"
    if CHOSEN.status == "partial"
    else ""
)
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
        else f" **PROVISIONAL: the setting is still running, drawn from its checkpoint at epoch {RUN.epoch} of {tc.n_epochs}.**"
    )
    display(
        Markdown(
            f"**Setting {CHOSEN.setting.index}** ({where}), an ensemble of {N_MEMBERS} nets trained "
            f"with {sampler}, lr = {tc.lr:g}, {RUN.epoch} epochs. Single-network figures show "
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

FLOOR_TEXT = sp.FLOOR_TEXT[: sp.FLOOR_TEXT.index("At the defaults")] + (
    r"In this experiment $\lambda_w = 1$ and every kernel is rescaled so that its top "
    r"eigenvalue is `spatial.lambda_max`, so $\lambda_{\max}(K^p)$ is the same for every population and the floor "
    r"$\epsilon\, \lambda_{\max}(K^p)$ is set by $\epsilon$ (`MIN_REL_EIG`) and $\lambda_{\max}$ alone; "
    r"what changes from setting to setting is how many eigenvalues lie below it. The tables below give, for "
    r"this setting, how many eigenvalues of each population the floor raised, $\lambda_{\max}(K^p)$, "
    r"$\lambda_{\max}(M^p)$ and the SGLD stability number of each layer."
)

PRIOR_MATRICES = r"""def check_permutation(order, n_units, what):
    # `order` must list each of the n_units units exactly once.
    if sorted(np.asarray(order).tolist()) != list(range(n_units)):
        raise ValueError(f"{what}: the order is not a permutation of the {n_units} units")


def coordinate_ticks(ax, coordinate, order, n_ticks=6):
    # Ticks on both axes of a matrix whose units are in `order`, each labelled with the unit's
    # coordinate along SORT_DIM (`coordinate` is indexed by unit) at that rank.
    ranks = np.unique(np.linspace(0, len(order) - 1, min(n_ticks, len(order))).round().astype(int))
    labels = [f"{coordinate[order[r]]:.2f}" for r in ranks]
    ax.set_xticks(ranks, labels)
    ax.set_yticks(ranks, labels)


def draw_sorted_matrix(fig, ax, matrix, order, coordinate, sort_dim, what, bound=None):
    # A square unit-by-unit matrix with its units in `order`, on a diverging colour scale centred
    # at 0 and symmetric to `bound` (default: its own largest entry), ticked with the coordinate.
    # Antialiased, so an N x N matrix larger than its panel does not show moire patterns.
    check_permutation(order, len(matrix), what)
    shown = np.asarray(matrix)[np.ix_(order, order)]
    bound = float(np.abs(shown).max()) if bound is None else float(bound)
    bound = bound or 1.0  # an all-zero matrix would give an empty colour range
    image = ax.imshow(
        shown, cmap="RdBu_r", vmin=-bound, vmax=bound, interpolation="antialiased"
    )
    fig.colorbar(image, ax=ax, shrink=0.8)
    coordinate_ticks(ax, coordinate, order)
    label = f"unit, sorted by coordinate {sort_dim} (value)"
    ax.set(xlabel=label, ylabel=label)
    return image


def plot_prior_matrices(prior, space, order_post, order_pre, sort_dim, tag=""):
    # One weight layer's prior: rows are the pre and the post factor, columns the covariance, the
    # effective covariance (the one the nets were trained under: floored under the penalty, K
    # itself under the whitened parametrisation) and the penalty. The covariance and the
    # effective covariance share one colour scale per row; the penalty has its own. Units are
    # in the given orders.
    if prior.whitened:
        effective = ("exact, as trained", r"$K^{{\rm {w}}} / \sqrt{{\lambda_w}}$")
    else:
        effective = ("floored, as trained", r"$(M^{{\rm {w}}})^{{-1}} / \sqrt{{\lambda_w}}$")
    columns = [
        ("covariance", prior.covariance, r"$K^{{\rm {w}}} / \sqrt{{\lambda_w}}$"),
        (f"effective covariance ({effective[0]})", prior.effective_covariance, effective[1]),
        ("penalty", prior.penalty, r"$\sqrt{{\lambda_w}}\, M^{{\rm {w}}}$"),
    ]
    rows = [("pre", space.pre, order_pre), ("post", space.post, order_post)]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), layout="constrained")
    for row, (which, pop, order) in enumerate(rows):
        coordinate = pop.embedding.positions[:, sort_dim].numpy()
        matrices = [getattr(pair, which).numpy() for _, pair, _ in columns]
        shared = max(np.abs(matrices[0]).max(), np.abs(matrices[1]).max())
        for col, (name, _, symbol) in enumerate(columns):
            ax = axes[row, col]
            bound = shared if col < 2 else None
            draw_sorted_matrix(fig, ax, matrices[col], order, coordinate, sort_dim, f"{name}, {which}", bound)
            ax.set_title(f"{name}, {which} ({pop.name})\n" + symbol.format(w=which))
    fig.suptitle(f"weight layer {space.layer + 1}: prior factors, side by side{tag}")
    return fig


for space, prior in zip(SPACES, PRIORS):
    plot_prior_matrices(
        prior, space, ORDERS[space.layer + 1], ORDERS[space.layer], SORT_DIM, STATUS_TAG
    )
plt.show()"""

SPECTRA_TEXT = r"""### Spectra of the prior factors

The leading eigenvalues of each factor (the Kronecker product's eigenvalues are products of one
eigenvalue of each factor), $j = 1, \dots, $ `N_EIG`. Top row: the covariance factor
$K^p / \sqrt{\lambda_w}$ and the effective one, the `N_EIG` **largest** eigenvalues, largest first,
on a log y-axis; the dotted line is the floor
$$\frac{\epsilon\, \lambda_{\max}(K^p)}{\sqrt{\lambda_w}},$$
below which the effective covariance has no eigenvalue. Only the eigenvalues of $K^p$ below the floor
are changed, and they are the *smallest* ones, so the two spectra coincide until the index passes
the number of unfloored eigenvalues, $N_p - n_{\mathrm{floored}}$: **the floor only appears once
`N_EIG` exceeds that number** (the cell prints it per population, and each title gives
$n_{\mathrm{floored}}$). Eigenvalues of $K$ that round-off made non-positive are not drawn on a log
y-axis.

Bottom row: the penalty factor $\sqrt{\lambda_w}\, M^p$, the **reciprocal of the effective
covariance** eigenvalue by eigenvalue. Its **largest** eigenvalues are all equal to the ceiling
$\sqrt{\lambda_w} / (\epsilon\, \lambda_{\max}(K^p))$, the floor seen from the penalty (the dotted
line, a flat part of the full spectrum), so the informative end is the other one: the figure shows the
`N_EIG` **smallest** eigenvalues of the penalty, smallest first (the axis says so); the $j$-th
smallest is $1 / $ the $j$-th largest eigenvalue of the effective covariance."""

PRIOR_SPECTRA = r"""def plot_prior_spectra(prior, space, tag=""):
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
        effective = (
            r"effective covariance (exact) $K / \sqrt{\lambda_w}$"
            if prior.whitened
            else r"effective covariance (floored) $M^{-1} / \sqrt{\lambda_w}$"
        )
        top.plot(j, eff.numpy(), "s--", label=effective)
        top.axhline(floor.covariance_floor, color="C3", ls=":", label="floor")
        top.set(
            yscale="log",
            xlabel=r"eigenvalue index $j$ (largest first)",
            ylabel="eigenvalue",
            title=f"{which} ({pop.name}): {floor.n_floored} of {len(pop.K)} eigenvalues floored",
        )
        bottom.plot(j, pen.numpy(), "o-", color="C2", label=r"penalty $\sqrt{\lambda_w}\, M$ = 1 / effective covariance")
        bottom.axhline(floor.penalty_ceiling, color="C3", ls=":", label="ceiling (the floor)")
        bottom.set(
            yscale="log",
            xlabel=r"eigenvalue index $j$ (smallest first)",
            ylabel="eigenvalue",
            title=f"{which} ({pop.name}): penalty, smallest eigenvalues",
        )
        for ax in (top, bottom):
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
            ax.grid(alpha=0.25)
            ax.legend(frameon=False)
    shown = {which: len(cov) for which, _, cov, *_ in populations}
    fig.suptitle(
        f"weight layer {space.layer + 1}: eigenvalues drawn: pre {shown['pre']}, post {shown['post']}{tag}"
    )
    return fig


for space, prior in zip(SPACES, PRIORS):
    print(
        f"weight layer {space.layer + 1}: {len(prior.covariance_spectra.post)} of {len(space.post.K)} "
        f"eigenvalues drawn for post, {len(prior.covariance_spectra.pre)} of {len(space.pre.K)} for pre"
    )
    for which, pop, floor in [("pre", space.pre, prior.floors.pre), ("post", space.post, prior.floors.post)]:
        print(
            f"  {which}: {floor.n_floored} of {len(pop.K)} floored, so the floor shows from "
            f"eigenvalue {len(pop.K) - floor.n_floored + 1} on (N_EIG = {N_EIG})"
        )
    plot_prior_spectra(prior, space, STATUS_TAG)
plt.show()"""

LOSS_TEXT = r"""## Loss curves

The loss and its three raw terms as logged during training, against epoch on a log y-axis, each the
**mean over the $B$ members** with a band of one standard error of that mean,
$$\bar f(t) = \frac{1}{B} \sum_{b=1}^{B} f_b(t), \qquad \mathrm{sem}(t) = \frac{1}{\sqrt{B}} \sqrt{\frac{1}{B - 1} \sum_{b=1}^{B} \big(f_b(t) - \bar f(t)\big)^2},$$
for $f \in \{\mathcal{L}, \mathrm{mse}, \Omega_w, \Omega_r\}$ with
$$\mathcal{L} = \frac{N}{2}\, \mathrm{mse} + \frac{\lambda_w}{2\beta}\, \Omega_w + \frac{\lambda_r}{2\beta}\, \Omega_r,$$
where $\Omega_w = \sum_l \mathrm{tr}(M^{\mathrm{post}}_l W_l M^{\mathrm{pre}}_l W_l^\top)$ and $\Omega_r = \sum_{l, i, \mu} (r^{(l)}_{i\mu})^2$.
The terms are drawn unweighted, so they do not add up to the $\mathcal{L}$ curve; a term that is zero at
every epoch has no point on a log y-axis and is marked "(zero)" in the legend. A band that reaches
zero is cut off at the bottom of the log y-axis. Under SGLD the curves settle to a noisy stationary level
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


plot_ensemble_loss(
    STATS, f"{OPTIMIZER}: training, mean $\\pm$ sem over {N_MEMBERS} members{STATUS_TAG}"
)
plt.show()"""

EXAMPLES_TEXT = r"""## Example outputs

Every member's output $\hat y^{b}_\mu = V_b^\top \phi(h^{(L)}_{b\mu}) / N$ against the target
$y_\mu = W_Y z_\mu$ for `N_EXAMPLES` points $\mu$ drawn at random from the **lattice test set**: the
$3^k$ points of the latent lattice $z \in \{z_\mathrm{lo}, z_\mathrm{mid}, z_\mathrm{hi}\}^k$
(`latent_min`, the midpoint and `latent_max`), which the nets were not trained on.
All members are on the same axes, **one palette colour per member** (member $b$ has the $b$-th
colour of matplotlib's default cycle, the same in both panels, named in the legend); there are at most
as many members as the palette has colours, and a larger ensemble raises an error rather than repeat
a colour. Left: one panel per sample, the target $y_{\mu a}$ as a ring on a stem and each member's
$\hat y^{b}_{\mu a}$ as a diamond, the members spread a little sideways so they do not hide each
other. With $m$ outputs the panels would be unreadable, so only `N_OUTPUT_DIMS` output dimensions $a$,
evenly spaced over $1, \dots, m$, are drawn (their index is on the axis). Right: every entry
$\hat y^{b}_{\mu a}$ against $y_{\mu a}$ over **all** $3^k$ test points and **all** outputs, one colour per
member, with the identity line and each member's pooled
$$R^2_b = 1 - \frac{\sum_{\mu a} (y_{\mu a} - \hat y^{b}_{\mu a})^2}{\sum_{\mu a} (y_{\mu a} - \bar y)^2}$$
in the legend, $\bar y$ the mean target."""

EXAMPLES = r"""def plot_ensemble_outputs(outputs, everything, colours, title, n_dims):
    # True output and every member's prediction on the examples (`outputs`), one colour per
    # member, for n_dims evenly spaced output dimensions; then every member's entries of
    # `everything` (all outputs) against the targets with their R^2.
    n_members = outputs.y_pred.shape[0]
    if n_members > len(colours):
        raise ValueError(
            f"{n_members} members but the palette has {len(colours)} colours: one colour per member"
        )
    y_true, y_pred = outputs.y_true.numpy(), outputs.y_pred.numpy()  # (n, m), (B, n, m)
    n, m = y_true.shape
    dims = np.unique(np.linspace(0, m - 1, min(n_dims, m)).round().astype(int))
    index = np.arange(len(dims))
    fig = plt.figure(figsize=(13, 2.4 * n + 1.5), layout="constrained")
    left, right = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    rows = left.subgridspec(n, 1)
    spread = 0.6 / n_members
    first = None
    for row, mu in enumerate(outputs.indices.tolist()):
        ax = fig.add_subplot(rows[row], sharex=first, sharey=first)
        first = ax if first is None else first
        ax.vlines(index, 0, y_true[row, dims], color="black", lw=1)
        ax.plot(index, y_true[row, dims], "o", ms=12, mfc="none", mec="black")
        for b in range(n_members):
            offset = (b - (n_members - 1) / 2) * spread
            ax.plot(index + offset, y_pred[b, row, dims], "D", ms=5, color=colours[b])
        ax.axhline(0, color="grey", lw=1)
        ax.set_title(f"example {mu}", loc="left")
        ax.set_ylabel("output")
        ax.margins(y=0.4)
        ax.tick_params(labelbottom=row == n - 1)
    ax.set_xlabel("output index $a$")
    ax.set_xticks(index, dims)
    handles = [
        mpl.lines.Line2D([], [], marker="o", ms=10, mfc="none", mec="black", ls="", label=r"true $y_{\mu a}$")
    ] + [
        mpl.lines.Line2D([], [], marker="D", ms=5, color=colours[b], ls="", label=f"member {b}")
        for b in range(n_members)
    ]
    first.legend(handles=handles, frameon=False, loc="upper right", ncol=5, fontsize=9)

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
    shown,
    test_outputs,
    colours,
    f"{N_MEMBERS} members, lattice test set ({n_samples} points){STATUS_TAG}",
    N_OUTPUT_DIMS,
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
$c = \lambda_w^{-1}$ and $A$ the diagonal of the other side's kernel, so the comparison with the prior
holds up to a scale: the prior panel has its own colour scale, while the member and the average of a
row share one, so their strengths compare. Every scale is diverging and symmetric about 0 (the
off-diagonal entries are signed). One member is a single sample of the weights and is noisy; the
average over members is less so, and the data term pulls both away from the prior. Units are in order
of coordinate `SORT_DIM` (not a ring position when $d > 1$), the ticks giving that coordinate. These
are second moments of the *weights* over *units*, not the mean-field kernels $\Phi^{(l)}$ over
*samples*."""

KERNELS = r"""def plot_weight_kernels(panels, space, order_post, order_pre, sort_dim, tag=""):
    # `panels` is [(title, kernel pair)], one column each: rows are the post and the pre kernel,
    # units in the given orders. The first panel (the prior) has its own colour scale; the others
    # share one per row.
    fig, axes = plt.subplots(
        2, len(panels), figsize=(5.5 * len(panels) + 1, 10), squeeze=False, layout="constrained"
    )
    rows = [("post", space.post, order_post), ("pre", space.pre, order_pre)]
    for row, (which, pop, order) in enumerate(rows):
        coordinate = pop.embedding.positions[:, sort_dim].numpy()
        matrices = [getattr(pair, which).detach().cpu().numpy() for _, pair in panels]
        shared = max((np.abs(m).max() for m in matrices[1:]), default=None)
        for col, (title, _) in enumerate(panels):
            ax = axes[row, col]
            bound = None if col == 0 else shared
            draw_sorted_matrix(fig, ax, matrices[col], order, coordinate, sort_dim, f"{title}: {which}", bound)
            ax.set_title(f"{title}: {which} ({pop.name})")
    fig.suptitle(f"weight layer {space.layer + 1}: kernels of the trained weights and the prior{tag}")
    return fig


for space, prior in zip(SPACES, PRIORS):
    panels = [
        (f"effective prior ({'exact' if prior.whitened else 'floored'})", prior.effective_covariance),
        (f"member {MEMBER}", MEMBER_VIEW.weight_kernels[space.layer]),
        (f"average of {N_MEMBERS} members", MEAN_VIEW.weight_kernels[space.layer]),
    ]
    plot_weight_kernels(
        panels, space, ORDERS[space.layer + 1], ORDERS[space.layer], SORT_DIM, STATUS_TAG
    )
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
    fig.suptitle(name + STATUS_TAG)
plt.show()"""
)

TUNING_TEXT = r"""## Tuning of the neurons of member `MEMBER` to the latents

For hidden layer $l$, neuron $i$ and latent $f$, the **random test set** of `P_TEST` fresh samples
of the training teacher (not the training samples, and not the lattice) is split into $n_\mathrm{bins}$ =
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
Only `N_NEURONS` neurons are drawn: with `NEURON_PICK = "first"` the **first `N_NEURONS` in order of
coordinate `SORT_DIM`**, with `"spread"` `N_NEURONS` neurons evenly spaced along that order, from
the smallest coordinate to the largest (the coordinate is not a ring position when $d > 1$; each title
gives the neuron's index and its coordinate). They are laid out in rows of `N_COLS`, one panel per
neuron, all on one rate axis. Beside each neuron's curves a column chart gives its information with
every latent: the pale bar is the raw plug-in $I_{if}$, the full-colour bar the estimate the MIR
uses, and for `"corrected"` a black mark is the shuffled-null mean.

The plug-in estimate is biased upward even for a neuron independent of the latent: about
$$\mathbb{E}[\hat I] \approx \frac{(n_\mathrm{bins}-1)^2}{2P}\ \text{nats},$$
with $P$ = `P_TEST` the number of samples, so a larger random test set lowers the noise floor.
Before each layer's figure a line gives $P$, $n_\mathrm{bins}$, this bias and the mean shuffled null
over neurons and latents."""

TUNING = (
    function_source(sp.TUNING, "_draw_information_bars")
    + r"""


def pick_neurons(order, n_neurons, how):
    # The neurons to draw out of `order` (all the layer's neurons, sorted): the first n_neurons,
    # or n_neurons evenly spaced along the order, its first and last included.
    if how not in ("first", "spread"):
        raise ValueError(f"NEURON_PICK must be 'first' or 'spread', got {how!r}")
    order = np.asarray(order)
    n = min(n_neurons, len(order))
    if how == "first":
        return order[:n]
    return order[np.linspace(0, len(order) - 1, n).round().astype(int)]


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
        f"layer {layer + 1}: P = {P_TEST}, n_bins = {N_BINS}, expected plug-in bias "
        f"(n_bins-1)^2 / (2P) = {(N_BINS - 1) ** 2 / (2 * P_TEST):.3f} nats per latent{null_note}"
    )
    drawn = pick_neurons(ORDERS[layer + 1], N_NEURONS, NEURON_PICK)
    how = f"first {len(drawn)}" if NEURON_PICK == "first" else f"{len(drawn)} evenly spaced"
    print(
        f"tuning: {how} of {len(ORDERS[layer + 1])} neurons, in order of "
        f"coordinate {SORT_DIM} of d = {D}"
    )
    positions = SPACES[layer].post.embedding.positions
    fig = plot_tuning_by_order(view, positions, drawn, SORT_DIM, N_COLS, 0.8)
    fig.suptitle(
        f"hidden layer {layer + 1}, member {MEMBER}: tuning of {len(drawn)} neurons "
        f"({MIR_KIND} MIR, layer mean over all neurons {np.mean(view.neuron_mir):.3f}){STATUS_TAG}"
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
and column in $C^I$ (for `"corrected"`). $C^r$ does not use the information estimate: it is signed,
positive for neurons tuned alike and negative for opposite tuning, and it also has weight where the
tuning is a constant offset, since $\bar r$ is not centred.

One figure per hidden layer and per variant, $C^I$ left and $C^r$ right, neurons in order of
coordinate `SORT_DIM` (not a ring position when $d > 1$; the ticks give that coordinate, as in the
weight-kernel figures). The figures come twice: for **member `MEMBER`**, and for the **ensemble
average**. The ensemble average is the **mean over members of each member's own overlap matrix**,
$$\bar C^{I}_{ij} = \frac{1}{B} \sum_{b=1}^{B} \frac{1}{k} \sum_f I^{b}_{if} I^{b}_{jf},$$
and likewise for $C^r$: a second moment over members, **not** the overlap of the mean tuning (the
overlap of averaged informations would shrink wherever members disagree). With the `"corrected"`
MIR each member's informations are clamped at zero before the products, which biases the average
upward. Unit $i$ is the same unit in every member, since they share their embeddings. $C^I$ is on
one colour scale starting at 0 and $C^r$ on one scale symmetric about 0; each scale is shared by
all the layers **and by the member and average figures**, so their strengths compare. If every $C^I$
is zero its scale is 0 to 1 and the title says "(no information)"."""

OVERLAP = r"""def plot_overlaps(C_info, C_rate, order, coordinate, info_max, rate_max, title, sort_dim):
    # The information overlap (viridis, 0 to info_max) and the rate overlap (diverging, +-rate_max)
    # of one layer side by side, both (N, N) with their neurons shown in `order`, ticked with the
    # neurons' coordinate along sort_dim (`coordinate` is indexed by neuron).
    info, rate = (C.detach().cpu().numpy() for C in (C_info, C_rate))
    for values in (info, rate):
        if values.ndim != 2 or values.shape[0] != values.shape[1]:
            raise ValueError(f"C must be a square matrix, got shape {values.shape}")
    if info.shape != rate.shape:
        raise ValueError(
            f"the two overlaps must be of the same neurons: {info.shape} vs {rate.shape}"
        )
    check_permutation(order, len(info), "overlap")
    informative = info_max > 0
    # A layer with no information (all C = 0) would give an empty colour range; show 0..1.
    info_top = info_max if informative else 1.0
    # Likewise for a layer whose rate overlap is all zero: the scale is -1 to 1.
    rate_top = rate_max if rate_max > 0 else 1.0
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    shown = np.ix_(order, order)
    kw = {"interpolation": "antialiased"}
    left = axes[0].imshow(info[shown], cmap="viridis", vmin=0, vmax=info_top, **kw)
    right = axes[1].imshow(rate[shown], cmap="RdBu_r", vmin=-rate_top, vmax=rate_top, **kw)
    fig.colorbar(left, ax=axes[0], label=r"$C^I_{ij}$  (nats$^2$)")
    fig.colorbar(right, ax=axes[1], label=r"$C^r_{ij}$  (rate$^2$)")
    titles = (
        "information overlap" + ("" if informative else " (no information)"),
        "rate overlap",
    )
    label = f"neuron, sorted by coordinate {sort_dim} (value)"
    for ax, name in zip(axes, titles):
        coordinate_ticks(ax, coordinate, order)
        ax.set(xlabel=label, ylabel=label, title=name)
    fig.suptitle(title)
    return fig


variants = [
    (f"member {MEMBER}", MEMBER_VIEW),
    (f"ensemble average of {N_MEMBERS} members", MEAN_VIEW),
]
# one scale for the member and the average, so the two figures compare
info_max = max(float(view.info_overlap.max()) for _, view in variants)
rate_max = max(float(view.rate_overlap.abs().max()) for _, view in variants)
for name, view in variants:
    for layer in range(NET.L):
        coordinate = SPACES[layer].post.embedding.positions[:, SORT_DIM].numpy()
        plot_overlaps(
            view.info_overlap[layer],
            view.rate_overlap[layer],
            ORDERS[layer + 1],
            coordinate,
            info_max,
            rate_max,
            f"hidden layer {layer + 1}: {name}{STATUS_TAG}",
            SORT_DIM,
        )
plt.show()"""


def build_notebook() -> NotebookNode:
    """The notebook without outputs: parameters, the settings table, the chosen setting, the prior
    (always drawn), then the figures of the trained ensemble (a markdown cell with the equation, a
    code
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
        new_markdown_cell(SETTINGS_TEXT),
        new_code_cell(SETTINGS_TABLE),
        new_markdown_cell(SETTING_TEXT),
        new_code_cell(LOADER),
        new_markdown_cell(PRIOR_TEXT),
        new_code_cell(PRIOR_MATRICES),
        new_markdown_cell(SPECTRA_TEXT),
        new_code_cell(PRIOR_SPECTRA),
        new_markdown_cell(FLOOR_TEXT),
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
        root_help="the experiment ROOT (the directory holding exp.json)",
    )


if __name__ == "__main__":
    raise SystemExit(main())
