"""Build and execute `linear_ring.ipynb`: one setting of a linear-ring experiment against the
prediction of the linear theory, and beside it the same point under the other parametrisation.

Give an experiment ROOT (made with `python -m fct exp init ROOT --preset linear_ring`, or
`linear_ring_tiny`, `linear_ring_equivalence`) and the axis values of one setting,
`SETTING = {"train.beta": 10.0}`. `TWIN_SETTING` (in `TWIN_ROOT`, or in ROOT itself) is shown
beside it: the penalty twin of a whitened setting, say. Per setting the notebook draws the loss
curves, the relaxation diagnostic of the records, the member-averaged final-state spectra
against `predicted_spectra`, the band share across the experiment's trained settings, the mode
ratio and R(Delta)/R(0), then tabulates every statistic with its bootstrap SE over members
against the prediction, and the two settings' equivalence.

The computation is `fct/linear_ring.py` (the analysis) and `fct/ring_view.py` (loading, tables);
the figures are `fct.draw`'s `plot_*`. Run

    python notebooks/build_linear_ring.py ROOT --set 'SETTING={"train.beta": 10.0}' --set 'TWIN_SETTING=None'   # execute, write tmp/linear_ring.ipynb
    python notebooks/build_linear_ring.py --no-execute                                                          # source only: notebooks/linear_ring.ipynb

ROOT is the experiment directory (the one holding exp.json); `$FCT_EXP_ROOT` works in the notebook
too. Only the source notebook is committed; an executed one records the ROOT path, so it goes to
the git-ignored tmp/ unless --out says otherwise.
"""

from pathlib import Path

from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from notebooks import build_spatial as sp
from notebooks.build_spatial_exp import when_trained

SOURCE_OUT = Path(__file__).with_name("linear_ring.ipynb")
EXECUTED_OUT = Path(__file__).parents[1] / "tmp" / "linear_ring.ipynb"

INTRO = r"""# The linear ring: measured tuning-map spectra against the linear theory

`ROOT` is a linear-ring experiment (`python -m fct exp init ROOT --preset linear_ring`, or
`linear_ring_tiny`, `linear_ring_equivalence`): one hidden layer of $N$ neurons on a ring, the
identity activation, a linear teacher with an orthonormal map and latents $z \in [0, 1]^K$, a
Gaussian spatial prior of length scale $\ell$ sites on the hidden weights, trained by SGLD at
inverse temperature $\beta$. `SETTING` picks one of its settings by its axis values, and
`TWIN_SETTING` (in `TWIN_ROOT`, or in `ROOT` itself) the setting shown beside it, normally the
same point under the other parametrisation of the prior (`"penalty"` or `"whitened"`).

**What is measured.** The tuning map of latent mode $k$ is $t_{k,i} = u_k^\top h_i$, with $h_i$
the activity of neuron $i$ over the $P$ patterns and $u_k$ the $k$-th left singular vector of the
latent matrix $Z$ ($u_1$ the mean direction, $u_2, \dots$ the contrasts); its power spectrum is
$d^{(k)}_q = |\tilde t_k(q)|^2$ in the unitary DFT over the ring. Every member of the ensemble
contributes **one sample, its final state**; every statistic is read off the
**member-averaged** final-state spectrum $D^{(k)}_q = \frac1B \sum_b d^{(k)}_{q,b}$, and its error
bar is the **bootstrap over members** (resample the $B$ members with replacement, average,
recompute). A ratio of means is not a mean of ratios, so per-member values are never averaged
into an estimate; the records along each chain serve only the relaxation diagnostic.

**The statistics.** The falsifiers are ring-averaged: $R_k(1)/R_k(0)$ and $R_k(\ell)/R_k(0)$ with
$R_k(\Delta) = \frac1N \sum_q D^{(k)}_q e^{2\pi i q \Delta / N}$, the band share
$\sum_{|q| \le q_b} D^{(k)}_q / \sum_q D^{(k)}_q$ with $q_b = \mathrm{round}(3N/64)$, and the band
power ratio of modes 1 and 2. The frequencies the penalty path's eigenvalue floor raises are
removed from both paths' spectra before any statistic is formed. An agreement test fails when
$|\text{measured} - \text{predicted}| > 3\,\mathrm{SE}$, and a failure counts only if it persists
after `exp extend` to $4B$ members. The half-power frequency $q_{1/2}$ and the $q = 0$ share rest
on one or two frequencies and are secondary.

**What the theory is.** `predicted_spectra` is the $N \to \infty$ mean-field limit at fixed
output dimension $n$: it leaves out the readout's $\frac n2 \log\det$ term, an $O(n/N)$
correction, so it is accurate only in the regime a preset documents (an exact posterior within
one SE at $4B$ of the prediction). The SGLD step adds its own bias: at `linear_ring`'s base point
the whitened path's lr biases mode 1's statistics by more than that accuracy budget, so there a
disagreement in mode 1 or in the ratio is not evidence against the theory. Each preset's
description states its regime and these numbers; the notebook quotes it below rather than
restating them.
"""

PARAMETERS = """ROOT = None  # the experiment ROOT (holds exp.json); None reads $FCT_EXP_ROOT
SETTING = {"spatial.parametrisation": "penalty", "train.lr": 0.0006}  # axis values of the setting to show
TWIN_ROOT = None  # the ROOT of the setting shown beside it; None: ROOT
TWIN_SETTING = {"spatial.parametrisation": "whitened", "train.lr": 0.005}  # its axis values; None: SETTING alone
BURN_IN = None  # epoch from which the relaxation fit tests the end of each run; None: the second half
SWEEP_AXIS = None  # a numeric axis to draw the band share against (train.beta, say); None: the setting index
EXACT = None  # the exact posterior's statistics of SETTING, in statistic_list order; None: regime not assessed
TWIN_EXACT = None  # the same for TWIN_SETTING
SEED = 0  # of the bootstrap over members"""

SETUP_TEXT = """## Setup

The library computes (`fct.linear_ring`, loaded through `fct.ring_view`); the cells below only
draw, with the `plot_*` functions of `fct.draw`. The loss curves use
`fct.phase_view.ensemble_statistics` (the member mean and its standard error at each logged
epoch) and nothing else of that module: its prior figures label the penalty's factors, which a
whitened run does not train with."""

IMPORTS = """import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

from fct import draw
from fct import linear_ring as lr
from fct.phase_view import ensemble_statistics
from fct.report import exp_root_from_env  # the $FCT_EXP_ROOT reader only
from fct.ring_view import (
    band_share_sweep,
    describe_run,
    equivalence_table,
    load_ring_setting,
    regime_markdown,
    relaxation_table,
    run_title,
    summary_table,
)
from fct.spatial_exp import settings_table
from fct.spectrum import latent_basis

mpl.rcParams.update({"figure.dpi": 100, "font.size": 12, "axes.titlesize": 12})

EXP_ROOT = ROOT or exp_root_from_env()
TWIN_EXP_ROOT = TWIN_ROOT or EXP_ROOT"""

SETTINGS_TEXT = """## The experiments' settings

Every setting with its axis values and the state of its training (`done`, `partial`, `missing` or
`stale`), as `python -m fct exp status ROOT` prints it; the twin's experiment too when it has its
own ROOT."""

SETTINGS_TABLE = """display(Markdown(settings_table(EXP_ROOT)))
if TWIN_SETTING is not None and TWIN_EXP_ROOT != EXP_ROOT:
    display(Markdown(f"The twin's experiment, `{TWIN_EXP_ROOT}`:"))
    display(Markdown(settings_table(TWIN_EXP_ROOT)))"""

LOADER_TEXT = """## The shown settings

Each setting's predicted law (`predicted_law`: its config, its data, `predicted_spectra` with the
$\\lambda_q$ of the parametrisation it trains with, and the frequencies kept by the floor) is built
whether or not it is trained; its records are loaded only when it is done. A setting that is not
trained is reported here and left out of every figure."""

LOADER = """RUNS = {"setting": load_ring_setting(EXP_ROOT, SETTING)}
if TWIN_SETTING is not None:
    RUNS["twin"] = load_ring_setting(TWIN_EXP_ROOT, TWIN_SETTING)
for run in RUNS.values():
    display(Markdown(describe_run(run)))
SHOWN = {name: run for name, run in RUNS.items() if run.trained}
EXACT_OF = {"setting": EXACT, "twin": TWIN_EXACT}
TRAINED = bool(SHOWN)
NOT_TRAINED = "**No shown setting is trained**: this cell is skipped."


def panels(height=4.5):
    \"\"\"One axis per shown setting, side by side, titled with the setting.\"\"\"
    fig, axes = plt.subplots(
        1, len(SHOWN), figsize=(7 * len(SHOWN), height), squeeze=False, layout="constrained"
    )
    for ax, run in zip(axes[0], SHOWN.values()):
        ax.set_title(run_title(run))
    return fig, dict(zip(SHOWN, axes[0]))"""

REGIME_TEXT = """## The regime, in the preset's words

The sentences of each experiment's description that state its regime (the accuracy budget, and
the bias its lr brings), then the whole description. An experiment made from a preset with
overrides (`exp init --set`, `--axis`) has only a note of them as its description, so the
preset's own is quoted, with that caveat. The numbers are the description's own; the notebook
does not restate them."""

REGIME = """for root in dict.fromkeys(run.root for run in RUNS.values()):
    run = next(r for r in RUNS.values() if r.root == root)
    display(Markdown(regime_markdown(root, run.description)))"""

MEASURE_TEXT = r"""## The measurement

For each shown setting: the final states of its $B$ members (the last record, taken after the
last step) give the member-averaged spectrum and everything read off it, with the bootstrap over
members (`lr.measure_final`); the records before the last give the relaxation diagnostic
(`lr.measure_relaxation`)."""

MEASURE = """FINAL = {
    name: lr.measure_final(run.law, run.records[-1], run.hex_final, SEED) for name, run in SHOWN.items()
}
RELAXATION = {
    name: lr.measure_relaxation(run.law, run.epochs[:-1], run.records[:-1, :, :-1], BURN_IN, SEED)
    for name, run in SHOWN.items()
}
for name, m in FINAL.items():
    display(Markdown(f"**{run_title(SHOWN[name])}**: {m.B} members, {int(m.disagree.sum())} of {len(m.stats)} primary statistics disagree."))"""

LOSS_TEXT = r"""## Loss curves

The logged training terms, the member mean $\pm$ its standard error at each logged epoch. The
loss is $\frac N2\,\mathrm{mse} + \frac{\lambda_w}{2\beta}\,\mathrm{weight\_l2} +
\frac{r}{2\beta}\,\mathrm{rate\_l2}$; the raw terms are drawn as logged, so they do not add up to
it, and a term that is zero throughout (the rate penalty at $r = 0$) is marked "(zero)"."""

LOSS = """fig, axes = panels()
for name, run in SHOWN.items():
    stats = ensemble_statistics(run.history)
    draw.plot_ensemble_loss(stats.epoch, stats.mean, stats.sem, ax=axes[name])
plt.show()"""

RELAXATION_TEXT = r"""## Burn-in: the relaxation of the records

Each statistic $y(t)$ of the member-averaged record at epoch $t$, relative to the asymptote $A$ of
the relaxation fitted from `BURN_IN` on (default: the second half of the run),
$$y(t) = A - C\,e^{-(t - t_0)/\tau},$$
in units of its accuracy budget (one delta-method SE at $4B$ members, `budget_se`). The fit is
dashed and the grey band is $\pm 1$ budget. The run is stationary on a statistic when the deficit
still to relax at the last record, $C e^{-(t_\mathrm{end} - t_0)/\tau}$, and its bootstrap SE are
both below the budget (`is_stationary`). Chains started from the prior show no relaxation in the
directions the data does not touch whether or not they mix, so this diagnostic is weak where the
prediction is close to the prior."""

RELAXATION = """fig, axes = panels(5.0)
for name, view in RELAXATION.items():
    draw.plot_relaxation(
        view.epochs, view.values, view.fit_epochs, view.fitted, view.labels, ax=axes[name],
        start=view.start, band=1.0, ylabel="(statistic - A) / budget",
    )
plt.show()
for name, view in RELAXATION.items():
    display(Markdown(f"**{run_title(SHOWN[name])}**, fit from epoch {view.start:g}:\\n\\n" + relaxation_table(view)))"""

SPECTRUM_TEXT = r"""## Spectra: the member-averaged final state against the theory

The folded spectra $D^{(k)}_{|q|} = \frac12\big(D^{(k)}_q + D^{(k)}_{-q}\big)$ of every latent mode
(points, bootstrap SE over members) against `predicted_spectra` (lines); the null row, the mean
spectrum of the $P - K$ directions orthogonal to the latents, dashed, with its predicted floor
$\sigma_0^2 = \sigma^2/(1 + r\sigma^2)$, which is $0$ at $\sigma = 0$: there the null row checks the
projection, not the theory, and a row of zeros is named "(zero)" since a log axis cannot show it.
The shaded $|q|$ are those the floor raises, left out of every statistic."""

SPECTRUM = """fig, axes = panels(5.0)
for name, m in FINAL.items():
    draw.plot_spectrum(
        m.spectrum, m.spectrum_se, m.spectrum_theory, ax=axes[name],
        null_theory=m.null_theory, excluded=m.excluded,
    )
plt.show()"""

SWEEP_TEXT = r"""## Band share across the experiment's settings

The band share $\sum_{|q| \le q_b} D^{(k)}_q / \sum_q D^{(k)}_q$, $q_b = \mathrm{round}(3N/64)$,
of every trained setting of each shown experiment that matches the shown setting on every axis
but `SWEEP_AXIS`, against its value of `SWEEP_AXIS` (every trained setting against its index
when `SWEEP_AXIS` is None): measured with its bootstrap SE (points), predicted (lines)."""

SWEEP = """SWEEPS = {}
for name, run in SHOWN.items():
    if run.root not in SWEEPS:
        SWEEPS[run.root] = band_share_sweep(run.root, SWEEP_AXIS, run.chosen.setting.axes, SEED)
fig, axes = plt.subplots(1, len(SWEEPS), figsize=(7 * len(SWEEPS), 4.5), squeeze=False, layout="constrained")
for ax, (root, sweep) in zip(axes[0], SWEEPS.items()):
    draw.plot_share_sweep(
        sweep.values, sweep.shares, sweep.se, sweep.theory, ax=ax,
        xlabel=sweep.xlabel, xscale=sweep.xscale,
    )
    ax.set_title(f"{root.name}: settings {sweep.indices}")
plt.show()"""

RATIO_TEXT = r"""## The mode ratio

$D^{(1)}_{|q|} / D^{(2)}_{|q|}$ against $|q|$ (points, bootstrap SE over members) and its
prediction (line). With no noise floor both modes end on the same $\lambda_q$ tail, so the
predicted ratio tends to $\hat s_{X,1}^2/\hat s_{X,2}^2$ at large $|q|$; the frequencies the floor
raises are left out. Single-$|q|$ ratios rest on one frequency and are noisy; the falsifier is
the band power ratio in the summary table."""

RATIO = """fig, axes = panels()
for name, m in FINAL.items():
    draw.plot_spectrum_ratio(m.ratio, m.ratio_se, m.ratio_theory, ax=axes[name], excluded=m.excluded)
plt.show()"""

AUTOCORRELATION_TEXT = r"""## Autocorrelation along the ring

$$\frac{R_k(\Delta)}{R_k(0)}, \qquad R_k(\Delta) = \frac1N \sum_q D^{(k)}_q e^{2\pi i q\Delta/N},$$
for $\Delta = 0, \dots, N/2$ over the kept frequencies, measured (points, bootstrap SE over
members) and predicted (lines), with the prior's own profile, $R$ of the hidden kernel's
eigenvalues $\lambda_q$ (dotted): the shape every mode would have if the data did not act on it."""

AUTOCORRELATION = """fig, axes = panels()
for name, m in FINAL.items():
    draw.plot_autocorrelation(
        m.profile, m.profile_se, m.profile_theory, ax=axes[name], kernel=m.kernel_profile
    )
plt.show()"""

ANGLE_TEXT = r"""## The contrast-plane angle

At $K = 3$ the angle $\theta_i = \mathrm{atan2}(t_{3,i}, t_{2,i})$ of each neuron in the contrast
plane $(u_2, u_3)$, summed over neurons and members of the final states, as a density over
$(-\pi, \pi]$, with the uniform density $1/(2\pi)$ (dashed) and the six directions in which a
neuron tuned to a single latent $\pm z_j$ lies (dotted). The statistic is
$|\bar m|$, $\bar m = \frac1B\sum_b \langle e^{6i\theta_i}\rangle_i$, in the summary table: near
0 when the tuning does not single out the latents. The angle is recorded only for $K \ge 3$."""

ANGLE = """WITH_ANGLES = {name: run for name, run in SHOWN.items() if run.theta_final is not None}
if WITH_ANGLES:
    fig, axes = plt.subplots(1, len(WITH_ANGLES), figsize=(7 * len(WITH_ANGLES), 4.5), squeeze=False, layout="constrained")
    for ax, (name, run) in zip(axes[0], WITH_ANGLES.items()):
        basis = latent_basis(run.law.dataset.factors)
        draw.plot_angle_histogram(run.theta_final.sum(0), ax=ax, marks=lr.latent_angles(basis.Z, basis.U))
        ax.set_title(run_title(run))
    plt.show()
else:
    K = next(iter(SHOWN.values())).law.D.shape[0]
    display(Markdown(f"The shown settings have K = {K} latents and no recorded angle: skipped."))"""

SUMMARY_TEXT = r"""## Summary: every statistic against the prediction

Per shown setting, every statistic of the member-averaged final-state spectrum with its bootstrap
SE over members against the prediction at the run's $N$, floor and parametrisation. `disagrees`
is the experiments' rule for the primary statistics ($> 3$ SE; a candidate falsification until it
persists at $4B$ members), $|\Delta q_{1/2}| > 1$ for $q_{1/2}$, $> 3$ SE for the $q = 0$ shares
and $|\bar m| > 0.1$ for the hexatic modulus. `regime accurate` is `regime_accurate` of the exact
posterior's statistics (`EXACT`, `TWIN_EXACT`, from the preset's description) when they are given:
the prediction is trusted on a statistic only where the exact posterior lies within one SE at
$4B$ of it. With two settings, the last table is their equivalence: their deviations from their
own predictions must agree within 3 combined SEs."""

SUMMARY = """for name, m in FINAL.items():
    run = SHOWN[name]
    display(Markdown(f"**{run_title(run)}** ({m.B} members)\\n\\n" + summary_table(m, run.law, EXACT_OF[name])))
if len(FINAL) == 2:
    display(Markdown("**Equivalence of the two settings**\\n\\n" + equivalence_table({f"{n} ({SHOWN[n].law.cfg.spatial.parametrisation})": m for n, m in FINAL.items()})))"""


def build_notebook() -> NotebookNode:
    """The notebook without outputs: parameters, the settings, the shown settings and their
    regime, then the measurement, a figure per markdown cell with its equation, and the tables."""
    nb = new_notebook(
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        }
    )
    cells = [
        new_markdown_cell(INTRO),
        new_code_cell(PARAMETERS, metadata={"tags": ["parameters"]}),
        new_markdown_cell(SETUP_TEXT),
        new_code_cell(IMPORTS),
        new_markdown_cell(SETTINGS_TEXT),
        new_code_cell(SETTINGS_TABLE),
        new_markdown_cell(LOADER_TEXT),
        new_code_cell(LOADER),
        new_markdown_cell(REGIME_TEXT),
        new_code_cell(REGIME),
        new_markdown_cell(MEASURE_TEXT),
        new_code_cell(when_trained(MEASURE)),
        new_markdown_cell(LOSS_TEXT),
        new_code_cell(when_trained(LOSS)),
        new_markdown_cell(RELAXATION_TEXT),
        new_code_cell(when_trained(RELAXATION)),
        new_markdown_cell(SPECTRUM_TEXT),
        new_code_cell(when_trained(SPECTRUM)),
        new_markdown_cell(SWEEP_TEXT),
        new_code_cell(when_trained(SWEEP)),
        new_markdown_cell(RATIO_TEXT),
        new_code_cell(when_trained(RATIO)),
        new_markdown_cell(AUTOCORRELATION_TEXT),
        new_code_cell(when_trained(AUTOCORRELATION)),
        new_markdown_cell(ANGLE_TEXT),
        new_code_cell(when_trained(ANGLE)),
        new_markdown_cell(SUMMARY_TEXT),
        new_code_cell(when_trained(SUMMARY)),
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
        "'SETTING={\"train.beta\": 10.0}'",
        argv,
        root_name="ROOT",
        root_help="the experiment ROOT (the directory holding exp.json)",
    )


if __name__ == "__main__":
    raise SystemExit(main())
