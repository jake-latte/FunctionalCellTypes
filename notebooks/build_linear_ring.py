"""Build and execute `linear_ring.ipynb`: one setting of a linear-ring experiment against the
prediction of the linear theory, and beside it the same point under the other parametrisation.

Give an experiment ROOT (made with `python -m fct exp init ROOT --preset linear_ring`, or
`linear_ring_tiny`, `linear_ring_equivalence`) and the axis values of one setting,
`SETTING = {"train.beta": 10.0}`. `TWIN_SETTING` (in `TWIN_ROOT`, or in ROOT itself) is shown
beside it: the penalty twin of a whitened setting, say. Per setting the notebook draws the loss
curves, the relaxation diagnostic of the records, the member-averaged final-state spectra
against `predicted_spectra`, the band share across the experiment's trained settings, the mode
ratio and R(Delta)/R(0), at K = 3 the angle histogram, then tabulates every statistic with its
bootstrap SE over members against the prediction, and the two settings' equivalence.

The analysis is `fct/linear_ring.py`, the loading and the tables `fct/ring_view.py`, the
figures `fct.draw`'s `plot_*`; every code cell is a few library calls. Run

    python notebooks/build_linear_ring.py ROOT --set 'SETTING={"train.beta": 10.0}' --set 'TWIN_SETTING=None'   # execute, write tmp/linear_ring.ipynb
    python notebooks/build_linear_ring.py --no-execute                                                          # source only: notebooks/linear_ring.ipynb

ROOT is the experiment directory (the one holding exp.json); `$FCT_EXP_ROOT` works in the notebook
too. Only the source notebook is committed; an executed one records the ROOT path, so it goes to
the git-ignored tmp/ unless --out says otherwise.
"""

import re
from fractions import Fraction
from pathlib import Path

from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from fct import linear_ring as lr
from fct import spectrum
from notebooks import build_spatial as sp
from notebooks.build_spatial_exp import when_trained

SOURCE_OUT = Path(__file__).with_name("linear_ring.ipynb")
EXECUTED_OUT = Path(__file__).parents[1] / "tmp" / "linear_ring.ipynb"
PLACEHOLDER = re.compile(r"<[A-Z_]+>")  # a threshold in the text, filled in by `fill`


def thresholds() -> dict[str, str]:
    """The rules' numbers as the text states them, read from the library when the notebook is
    built, so the text cannot drift from the code."""
    band = Fraction(spectrum.BAND_FRACTION).limit_denominator()
    return {
        "<FALSIFY_N_SE>": f"{lr.FALSIFY_N_SE:g}",
        "<FOLLOW_UP_FACTOR>": f"{lr.FOLLOW_UP_FACTOR:g}",
        "<EQUIVALENCE_N_SE>": f"{lr.EQUIVALENCE_N_SE:g}",
        "<Q_HALF_TOLERANCE>": f"{lr.Q_HALF_TOLERANCE:g}",
        "<HEXATIC_MAX>": f"{lr.HEXATIC_MAX:g}",
        "<ANGLE_K>": f"{lr.ANGLE_K}",
        "<BAND_RULE>": rf"\mathrm{{round}}({band.numerator}N/{band.denominator})",
    }


def fill(text: str) -> str:
    """`text` with every threshold placeholder replaced by its value; ValueError if one is left."""
    for token, value in thresholds().items():
        text = text.replace(token, value)
    if left := PLACEHOLDER.findall(text):
        raise ValueError(f"unknown threshold placeholders {left} in {text[:80]!r}")
    return text


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
$\sum_{|q| \le q_b} D^{(k)}_q / \sum_q D^{(k)}_q$ with $q_b = <BAND_RULE>$, and the band
power ratio of modes 1 and 2. The frequencies the penalty path's eigenvalue floor raises are
removed from both paths' spectra before any statistic is formed. An agreement test fails when
$|\text{measured} - \text{predicted}| > <FALSIFY_N_SE>\,\mathrm{SE}$, and a failure counts only if it
persists after `exp extend` to $<FOLLOW_UP_FACTOR>B$ members. The half-power frequency $q_{1/2}$
and the $q = 0$ share rest on one or two frequencies and are secondary.

**What the theory is.** `predicted_spectra` is the $N \to \infty$ mean-field limit at fixed
output dimension $n$: it leaves out the readout's $\frac n2 \log\det$ term, an $O(n/N)$
correction, so it is accurate only in the regime a preset documents (an exact posterior within
one SE at $<FOLLOW_UP_FACTOR>B_0$ of the prediction, $B_0$ the experiment's base $B$). The SGLD
step adds its own bias: at `linear_ring`'s base point the whitened path's lr biases mode 1's
statistics by more than that accuracy budget, so there a disagreement in mode 1 or in the ratio
is not evidence against the theory. Each preset's description states its regime and these
numbers; the notebook quotes it below rather than restating them.
"""

PARAMETERS = """ROOT = None  # the experiment ROOT (holds exp.json); None reads $FCT_EXP_ROOT
SETTING = {"spatial.parametrisation": "penalty", "train.lr": 0.0006}  # axis values of the setting to show
TWIN_ROOT = None  # the ROOT of the setting shown beside it; None: ROOT
TWIN_SETTING = {"spatial.parametrisation": "whitened", "train.lr": 0.005}  # its axis values; None: SETTING alone
BURN_IN = None  # epoch from which the relaxation fit tests the end of each run; None: the second half
BASE_B = None  # the B before any exp extend, for the budget; None: from the preset each ROOT was made from
SWEEP_AXIS = None  # a numeric axis to draw the band share against (train.beta, say); None: the setting index
EXACT = None  # the exact posterior's statistics of SETTING, in statistic_list order; None: regime not assessed
TWIN_EXACT = None  # the same for TWIN_SETTING
SEED = 0  # of the bootstrap over members"""

SETUP_TEXT = """## Setup

The library computes (`fct.linear_ring`, loaded and tabulated by `fct.ring_view`); the cells
below only call it and draw with `fct.draw`. The loss curves use
`fct.phase_view.ensemble_statistics` (the member mean and its standard error at each logged
epoch) and nothing else of that module: its prior figures label the penalty's factors, which a
whitened run does not train with."""

IMPORTS = """import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

from fct import draw
from fct.report import exp_root_from_env  # the $FCT_EXP_ROOT reader only
from fct.ring_view import (
    angle_panels,
    describe_runs,
    load_shown,
    load_sweeps,
    measure_runs,
    measurement_note,
    panels,
    regimes_markdown,
    relaxation_tables,
    settings_tables,
    summary_tables,
    titles,
    trained,
)

mpl.rcParams.update({"figure.dpi": 100, "font.size": 12, "axes.titlesize": 12})

EXP_ROOT = ROOT or exp_root_from_env()
TWIN_EXP_ROOT = TWIN_ROOT or EXP_ROOT"""

SETTINGS_TEXT = """## The experiments' settings

Every setting with its axis values and the state of its training (`done`, `partial`, `missing` or
`stale`), as `python -m fct exp status ROOT` prints it; the twin's experiment too when it has its
own ROOT."""

SETTINGS_TABLE = """display(Markdown(settings_tables([EXP_ROOT, TWIN_EXP_ROOT])))"""

LOADER_TEXT = """## The shown settings

Each setting's predicted law (`predicted_law`: its config, its data, `predicted_spectra` with the
$\\lambda_q$ of the parametrisation it trains with, and the frequencies kept by the floor) is built
whether or not it is trained; its records are loaded only when it is done. A setting that is not
trained is reported here and left out of every figure."""

LOADER = """RUNS = load_shown(EXP_ROOT, SETTING, TWIN_EXP_ROOT, TWIN_SETTING)
display(Markdown(describe_runs(RUNS)))
SHOWN = trained(RUNS)
TITLES = titles(SHOWN)
EXACT_OF = {"setting": EXACT, "twin": TWIN_EXACT}
TRAINED = bool(SHOWN)
NOT_TRAINED = "**No shown setting is trained**: this cell is skipped.\""""

REGIME_TEXT = """## The regime, in the preset's words

The sentences of each experiment's description that state its regime (the accuracy budget, and
the bias its lr brings), then the whole description. An experiment made from a preset with
overrides (`exp init --set`, `--axis`) has only a note of them as its description, so the
preset's own is quoted, with that caveat. The numbers are the description's own; the notebook
does not restate them."""

REGIME = """display(Markdown(regimes_markdown(RUNS)))"""

MEASURE_TEXT = r"""## The measurement

For each shown setting: the final states of its $B$ members (the last record, taken after the
last step) give the member-averaged spectrum and everything read off it, with the bootstrap over
members (`lr.measure_final`); the records before the last give the relaxation diagnostic
(`lr.measure_relaxation`), judged against the budget at the base $B_0$ (`BASE_B`, by default the
$B$ of the preset the ROOT was made from, never exp.json's $B$, which `exp extend` grows)."""

MEASURE = """MEASURED = measure_runs(SHOWN, BURN_IN, BASE_B, SEED)
display(Markdown(measurement_note(SHOWN, MEASURED)))"""

LOSS_TEXT = r"""## Loss curves

The logged training terms, the member mean $\pm$ its standard error at each logged epoch. The
loss is $\frac N2\,\mathrm{mse} + \frac{\lambda_w}{2\beta}\,\mathrm{weight\_l2} +
\frac{r}{2\beta}\,\mathrm{rate\_l2}$; the raw terms are drawn as logged, so they do not add up to
it, and a term that is zero throughout is marked "(zero)"."""

LOSS = """draw.fig_panels(draw.plot_ensemble_loss, panels(MEASURED, "loss"), TITLES)
plt.show()"""

RELAXATION_TEXT = r"""## Burn-in: the relaxation of the records

Each statistic $y(t)$ of the member-averaged record at epoch $t$, relative to the asymptote $A$ of
the relaxation fitted from `BURN_IN` on (default: the second half of the run),
$$y(t) = A - C\,e^{-(t - t_0)/\tau},$$
in units of its accuracy budget (one delta-method SE at $<FOLLOW_UP_FACTOR>B_0$ members,
`budget_se`). The fit is dashed and the grey band is $\pm 1$ budget. The run is stationary on a
statistic when the deficit still to relax at the last record, $C e^{-(t_\mathrm{end} - t_0)/\tau}$,
and its bootstrap SE are both below the budget (`is_stationary`). Chains started from the prior
show no relaxation in the directions the data does not touch whether or not they mix, so this
diagnostic is weak where the prediction is close to the prior."""

RELAXATION = """draw.fig_panels(
    draw.plot_relaxation, panels(MEASURED, "relaxation.curves"), TITLES,
    ylabel="(statistic - A) / budget",
)
plt.show()
display(Markdown(relaxation_tables(SHOWN, MEASURED)))"""

SPECTRUM_TEXT = r"""## Spectra: the member-averaged final state against the theory

The folded spectra $D^{(k)}_{|q|} = \frac12\big(D^{(k)}_q + D^{(k)}_{-q}\big)$ of every latent mode
(points, bootstrap SE over members) against `predicted_spectra` (lines); the null row, the mean
spectrum of the $P - K$ directions orthogonal to the latents, dashed, with its predicted floor
$\sigma_0^2 = \sigma^2/(1 + r\sigma^2)$, which is $0$ at $\sigma = 0$: there the null row checks the
projection, not the theory, and is named in the legend when it lies below the axis or is zero.
The shaded $|q|$ are those the floor raises, left out of every statistic."""

SPECTRUM = """draw.fig_panels(draw.plot_spectrum, panels(MEASURED, "final.spectrum"), TITLES)
plt.show()"""

SWEEP_TEXT = r"""## Band share across the experiment's settings

The band share $\sum_{|q| \le q_b} D^{(k)}_q / \sum_q D^{(k)}_q$, $q_b = <BAND_RULE>$, of every
trained setting of a shown setting's experiment that matches it on every axis but `SWEEP_AXIS`,
against its value of `SWEEP_AXIS` (every trained setting against its index when `SWEEP_AXIS` is
None): measured with its bootstrap SE (points), predicted (lines with crosses). Two shown
settings that differ in another axis get a panel each."""

SWEEP = """SWEEP_TITLES, SWEEP_PANELS = load_sweeps(SHOWN, SWEEP_AXIS, SEED)
draw.fig_panels(draw.plot_share_sweep, SWEEP_PANELS, SWEEP_TITLES)
plt.show()"""

RATIO_TEXT = r"""## The mode ratio

$D^{(1)}_{|q|} / D^{(2)}_{|q|}$ against $|q|$ (points, bootstrap SE over members) and its
prediction (line). With no noise floor both modes end on the same $\lambda_q$ tail, so the
predicted ratio tends to $\hat s_{X,1}^2/\hat s_{X,2}^2$ at large $|q|$; the frequencies the floor
raises are left out. Single-$|q|$ ratios rest on one frequency and are noisy; the falsifier is
the band power ratio in the summary table."""

RATIO = """draw.fig_panels(draw.plot_spectrum_ratio, panels(MEASURED, "final.ratio"), TITLES)
plt.show()"""

AUTOCORRELATION_TEXT = r"""## Autocorrelation along the ring

$$\frac{R_k(\Delta)}{R_k(0)}, \qquad R_k(\Delta) = \frac1N \sum_q D^{(k)}_q e^{2\pi i q\Delta/N},$$
for $\Delta = 0, \dots, N/2$ over the kept frequencies, measured (points, bootstrap SE over
members) and predicted (lines), with the prior's own profile, $R$ of the hidden kernel's
eigenvalues $\lambda_q$ (dotted): the shape every mode would have if the data did not act on it."""

AUTOCORRELATION = """draw.fig_panels(draw.plot_autocorrelation, panels(MEASURED, "final.profile"), TITLES)
plt.show()"""

ANGLE_TEXT = r"""## The contrast-plane angle (K = <ANGLE_K> only)

At $K = <ANGLE_K>$ the angle $\theta_i = \mathrm{atan2}(t_{3,i}, t_{2,i})$ of each neuron in the
contrast plane $(u_2, u_3)$, summed over neurons and members of the final states, as a density
over $(-\pi, \pi]$, with the uniform density $1/(2\pi)$ (dashed) and the six directions in which a
neuron tuned to a single latent $\pm z_j$ lies (dotted). The statistic is $|\bar m|$,
$\bar m = \frac1B\sum_b \langle e^{6i\theta_i}\rangle_i$, in the summary table: near 0 when the
tuning does not single out the latents. At any other $K$ the plane $(u_2, u_3)$ is not the whole
contrast space, so neither the histogram nor $|\bar m|$ is shown."""

ANGLE = """ANGLES = angle_panels(SHOWN, MEASURED)
if ANGLES.panels:
    draw.fig_panels(draw.plot_angle_histogram, ANGLES.panels, ANGLES.titles)
    plt.show()
else:
    display(Markdown(ANGLES.note))"""

SUMMARY_TEXT = r"""## Summary: every statistic against the prediction

Per shown setting, every statistic of the member-averaged final-state spectrum with its bootstrap
SE over members against the prediction at the run's $N$, floor and parametrisation. `disagrees`
is the experiments' rule for the primary statistics ($> <FALSIFY_N_SE>$ SE; a candidate
falsification until it persists at $<FOLLOW_UP_FACTOR>B$ members), $|\Delta q_{1/2}| >
<Q_HALF_TOLERANCE>$ for $q_{1/2}$, $> <FALSIFY_N_SE>$ SE for the $q = 0$ shares and, at
$K = <ANGLE_K>$, $|\bar m| > <HEXATIC_MAX>$ for the hexatic modulus; a secondary statistic that
cannot be formed (a $q_{1/2}$ of a spectrum rising to Nyquist) is listed as unavailable with the
reason. `regime accurate` is `regime_accurate` of the exact posterior's statistics (`EXACT`,
`TWIN_EXACT`, from the preset's description) when they are given: the prediction is trusted on a
statistic only where the exact posterior lies within one SE at $<FOLLOW_UP_FACTOR>B_0$ of it.
With two settings, the last table is their equivalence: their deviations from their own
predictions must agree within <EQUIVALENCE_N_SE> combined SEs."""

SUMMARY = """display(Markdown(summary_tables(SHOWN, MEASURED, EXACT_OF)))"""


def build_notebook() -> NotebookNode:
    """The notebook without outputs: parameters, the settings, the shown settings and their
    regime, then the measurement, a figure per markdown cell with its equation, and the tables."""
    nb = new_notebook(
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        }
    )
    cells = [
        new_markdown_cell(fill(INTRO)),
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
        new_markdown_cell(fill(RELAXATION_TEXT)),
        new_code_cell(when_trained(RELAXATION)),
        new_markdown_cell(SPECTRUM_TEXT),
        new_code_cell(when_trained(SPECTRUM)),
        new_markdown_cell(fill(SWEEP_TEXT)),
        new_code_cell(when_trained(SWEEP)),
        new_markdown_cell(RATIO_TEXT),
        new_code_cell(when_trained(RATIO)),
        new_markdown_cell(AUTOCORRELATION_TEXT),
        new_code_cell(when_trained(AUTOCORRELATION)),
        new_markdown_cell(fill(ANGLE_TEXT)),
        new_code_cell(when_trained(ANGLE)),
        new_markdown_cell(fill(SUMMARY_TEXT)),
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
