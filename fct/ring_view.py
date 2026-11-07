"""One setting of a linear-ring experiment, for the linear-ring notebook: load its predicted law and
its members' records, measure the band share across the experiment's settings, and write the
summary tables.

The I/O layer of the notebook: the analysis is `fct.linear_ring` (pure functions of the law and
the records), the figures `fct.draw`. A setting that is not trained is a status, not an error
(`RingRun.trained`), as in `fct.spatial_exp`, whose `select_setting` finds it. The MIR readers
(`fct.report`, `fct.exp_report`) refuse experiments with the spectrum monitor by design, and
nothing here calls them.
"""

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor

from fct import linear_ring as lr
from fct.exp import axes_of, load_exp, same_value, setting_statuses
from fct.monitor import spectrum_records
from fct.spatial_exp import ExpSetting, select_setting
from fct.train import MONITOR_EPOCH, History, load_trained

CPU, F64 = torch.device("cpu"), torch.float64
# A sweep is drawn on a log axis when its positive values span at least this factor.
LOG_SWEEP_SPAN = 10.0
# Sentences of a preset description that state its regime: what the accuracy budget allows.
_REGIME_WORDS = re.compile(r"accura|bias", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=\.)\s+(?=[A-Z(`])")


class RingRun(NamedTuple):
    """One setting: its predicted law and, when it is trained, its records. The records are of
    the one hidden layer, float64; the last record is the final state, one sample per member."""

    chosen: ExpSetting
    law: lr.Law
    description: str | None  # the experiment's (the preset's) description
    history: History | None
    epochs: Tensor | None  # (R,) the records' epochs
    records: Tensor | None  # (R, B, K + 1, N), the null row last
    hex_final: Tensor | None  # (B, 2) the final hexatic moments (K = 3 only)
    theta_final: Tensor | None  # (B, 36) the final angle histograms (K = 3 only)

    @property
    def trained(self) -> bool:
        return self.chosen.status == "done"


def final_member_values(history: Mapping[str, Any], key: str) -> Tensor | None:
    """(B, ...) the values of monitor column `key` at the last record, of the first hidden layer,
    one row per member (B = 1 for a single network's history); None if the run did not record
    `key` (the angle keys exist only at K >= 3)."""
    entries = history.get(key)
    if not entries:
        return None
    last = entries[-1]
    if isinstance(last[0], Tensor):  # a single network: a list over layers
        last = [last]
    return torch.stack([layers[0] for layers in last])


def load_ring_setting(root: Path, values: Mapping[str, Any]) -> RingRun:
    """The setting of the experiment at `root` whose axes equal `values` (a subset that picks one,
    `select_setting`), with its predicted law and, if it is done, its records."""
    root = Path(root)
    exp = load_exp(root)
    chosen = select_setting(root, values, "cpu")
    law = lr.predicted_law(exp, chosen.setting.index)
    if chosen.status != "done":
        return RingRun(chosen, law, exp.description, None, None, None, None, None)
    if chosen.cfg.net.L != 1:
        raise ValueError(
            f"the linear ring has one hidden layer, the setting has {chosen.cfg.net.L}"
        )
    _, history = load_trained(chosen.directory, chosen.cfg, CPU, F64)
    return RingRun(
        chosen=chosen,
        law=law,
        description=exp.description,
        history=history,
        epochs=torch.tensor(history[MONITOR_EPOCH], dtype=F64),
        records=spectrum_records(history)[:, :, 0].to(F64),
        hex_final=final_member_values(history, "monitor_hex"),
        theta_final=final_member_values(history, "monitor_theta_hist"),
    )


def run_title(run: RingRun) -> str:
    """A short title: the setting's index, parametrisation and axis values."""
    axes = ", ".join(f"{a} = {v}" for a, v in run.chosen.setting.axes.items())
    return f"setting {run.chosen.setting.index} ({run.law.cfg.spatial.parametrisation}): {axes}"


def describe_run(run: RingRun) -> str:
    """A markdown paragraph: which setting, how it was trained, and its state."""
    cfg = run.law.cfg
    head = (
        f"**{run_title(run)}** of `{run.chosen.directory.parent.name}`: N = {cfg.net.N}, "
        f"P = {cfg.data.P}, K = {run.law.D.shape[0]}, ell = {run.law.ell} sites, "
        f"beta = {cfg.train.beta:g}, r = {cfg.train.rate_lambda:g}, lr = {cfg.train.lr:g}, "
        f"{cfg.train.n_epochs} epochs, B = {cfg.net.B} members, "
        f"{int((~run.law.keep).sum())} of {cfg.net.N} frequencies excluded by the floor "
        f"(min_rel_eig {cfg.spatial.min_rel_eig:g})."
    )
    if not run.trained:
        return f"{head} **Not trained** (status: {run.chosen.status}): its figures are skipped."
    return (
        f"{head} Trained; {run.records.shape[0]} records, the last at epoch "
        f"{int(run.epochs[-1])} (the final state, one sample per member)."
    )


def regime_sentences(description: str | None) -> list[str]:
    """The sentences of a description that speak of accuracy or bias: where a preset states
    its regime and what its lr does to it."""
    if not description:
        return []
    return [s for s in _SENTENCE_END.split(description) if _REGIME_WORDS.search(s)]


# ------------------------------------------------------------------ the band share sweep
class Sweep(NamedTuple):
    values: Tensor  # (n,) the swept value of each setting (its index without an axis)
    shares: Tensor  # (n, K) band shares of the member-averaged final spectra
    se: Tensor  # (n, K) bootstrap SEs over members
    theory: Tensor  # (n, K)
    indices: list[int]  # the settings' indices
    xlabel: str
    xscale: str


def _swept_value(setting, axis: str | None) -> float:
    if axis is None:
        return float(setting.index)
    value = setting.axes[axis]
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"the sweep axis {axis} must be numeric, got {value!r}")
    return float(value)


def band_share_sweep(
    root: Path, axis: str | None, values: Mapping[str, Any], seed: int = 0
) -> Sweep:
    """The band shares of every trained setting of the experiment at `root` that matches
    `values` on every axis but `axis`, against its value of `axis` (every trained setting
    against its index when `axis` is None)."""
    root = Path(root)
    exp = load_exp(root)
    if axis is not None and axis not in axes_of(exp):
        raise LookupError(f"unknown sweep axis {axis}; the experiment has {list(axes_of(exp))}")
    others = {a: v for a, v in values.items() if a != axis} if axis is not None else {}
    settings = [
        s
        for _, s, status in setting_statuses(root, exp)
        if status == "done" and all(same_value(s.axes[a], v) for a, v in others.items())
    ]
    if not settings:
        raise ValueError(f"no trained setting of {root} matches {others}")
    runs = [load_ring_setting(root, s.axes) for s in settings]
    statistics = lr.statistic_list(runs[0].law.D.shape[0])
    band = [i for i, s in enumerate(statistics) if s.kind == "band"]
    measured = [lr.estimate(r.records[-1, :, :-1], r.law, seed) for r in runs]
    x = torch.tensor([_swept_value(s, axis) for s in settings], dtype=F64)
    log = axis is not None and bool((x > 0).all()) and float(x.max() / x.min()) >= LOG_SWEEP_SPAN
    return Sweep(
        values=x,
        shares=torch.stack([v[band] for v, _ in measured]),
        se=torch.stack([se[band] for _, se in measured]),
        theory=torch.stack([lr.law_statistics(r.law)(r.law.D)[band] for r in runs]),
        indices=[s.index for s in settings],
        xlabel="setting index" if axis is None else axis,
        xscale="log" if log else "linear",
    )


# ------------------------------------------------------------------ tables
def _name(stat: lr.Statistic, ell: int) -> str:
    names = {
        "lag1": "R(1)/R(0)",
        "lagl": f"R({ell})/R(0)",
        "band": "band share (|q| <= q_b)",
        "ratio": "band power ratio, mode 1 / mode 2",
    }
    return names[stat.kind]


def _flag(flagged: bool) -> str:
    return "**yes**" if flagged else "no"


def _row(cells: Sequence[str]) -> str:
    """A markdown table row; a "|" inside a cell (|q|, |mean m|) is escaped, or it would end it."""
    return "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    return "\n".join([_row(header), "|" + "---|" * len(header), *map(_row, rows)])


def _z(value: float, predicted: float, se: float) -> str:
    return f"{(value - predicted) / se:+.2f}" if se > 0 else "n/a"


SUMMARY_HEADER = (
    "statistic", "mode", "role", "predicted", "measured", "bootstrap SE",
    "(measured - predicted) / SE", "disagrees", "regime accurate",
)  # fmt: skip


def summary_table(m: lr.FinalMeasurement, law: lr.Law, exact: Sequence[float] | None = None) -> str:
    """Every statistic of the member-averaged final-state spectrum against its prediction, with
    its bootstrap SE: the primary, ring-averaged falsifiers (`disagrees`: beyond FALSIFY_N_SE
    SEs), then the secondary q_1/2 (beyond Q_HALF_TOLERANCE) and q = 0 shares, and the hexatic
    modulus when measured (above HEXATIC_MAX). `exact`, the exact posterior's statistics in
    `statistic_list` order, states `regime_accurate` per primary statistic."""
    if exact is not None and len(exact) != len(m.stats):
        raise ValueError(f"exact has {len(exact)} statistics, expected {len(m.stats)}")
    accurate = (
        None if exact is None else lr.regime_accurate(torch.tensor(exact, dtype=F64), law).tolist()
    )
    rows = []
    for i, stat in enumerate(m.stats):
        v, se, p = float(m.values[i]), float(m.se[i]), float(m.predicted[i])
        regime = "not assessed" if accurate is None else ("yes" if accurate[i] else "no")
        mode = "1/2" if stat.mode is None else str(stat.mode)
        rows.append([_name(stat, law.ell), mode, "primary", f"{p:.6g}", f"{v:.6g}", f"{se:.4g}",
                     _z(v, p, se), _flag(bool(m.disagree[i])), regime])  # fmt: skip
    K = law.D.shape[0]
    for i, name in enumerate(m.secondary_names):
        v, se = float(m.secondary[i]), float(m.secondary_se[i])
        p = float(m.secondary_predicted[i])
        label = "q_1/2" if name.startswith("q_half") else "q = 0 share"
        rows.append([label, str(i % K + 1), "secondary", f"{p:.6g}", f"{v:.6g}", f"{se:.4g}",
                     _z(v, p, se), _flag(bool(m.secondary_disagree[i])), "-"])  # fmt: skip
    if m.hexatic is not None:
        v, se = (float(x) for x in m.hexatic)
        rows.append(["|mean m| (hexatic)", "2, 3", "secondary", "0", f"{v:.6g}", f"{se:.4g}",
                     "n/a", _flag(v > lr.HEXATIC_MAX), "-"])  # fmt: skip
    return _table(SUMMARY_HEADER, rows)


def equivalence_table(measurements: Mapping[str, lr.FinalMeasurement]) -> str:
    """Two paths' deviations from their own predictions on every primary statistic, and
    `paths_disagree` (beyond EQUIVALENCE_N_SE combined SEs), with the count compared."""
    if len(measurements) != 2:
        raise ValueError(f"the equivalence compares two settings, got {len(measurements)}")
    (name_a, a), (name_b, b) = measurements.items()
    disagree = lr.paths_disagree((a.values, a.se, a.predicted), (b.values, b.se, b.predicted))
    dev_a, dev_b = a.values - a.predicted, b.values - b.predicted
    combined = (a.se.square() + b.se.square()).sqrt()
    header = ["statistic", "mode", f"deviation {name_a}", f"SE {name_a}", f"deviation {name_b}",
              f"SE {name_b}", "|difference| / combined SE", "disagree"]  # fmt: skip
    rows = [
        [lr.statistic_label(stat), "1/2" if stat.mode is None else str(stat.mode),
         f"{float(dev_a[i]):+.4g}", f"{float(a.se[i]):.4g}", f"{float(dev_b[i]):+.4g}",
         f"{float(b.se[i]):.4g}", f"{float((dev_a[i] - dev_b[i]).abs() / combined[i]):.2f}",
         _flag(bool(disagree[i]))]
        for i, stat in enumerate(a.stats)
    ]  # fmt: skip
    note = (
        f"\n\n{len(a.stats)} statistics compared, each at {lr.EQUIVALENCE_N_SE:g} combined SEs; "
        f"an exceedance counts only if it persists after `exp extend` to "
        f"{lr.FOLLOW_UP_FACTOR} x B."
    )
    return _table(header, rows) + note


def relaxation_table(view: lr.RelaxationView) -> str:
    """Per statistic, the deficit still to relax at the last record before the final state, its
    SE, the fitted time constant, the accuracy budget and `is_stationary`."""
    header = ["statistic", "deficit", "deficit SE", "tau (epochs)", "budget", "stationary"]
    fit = view.fit
    rows = [
        [label, f"{float(fit.deficit[i]):.4g}", f"{float(fit.deficit_se[i]):.4g}",
         f"{float(fit.tau[i]):.4g}", f"{float(view.budget[i]):.4g}",
         "yes" if view.stationary[i] else "**no**"]
        for i, label in enumerate(view.labels)
    ]  # fmt: skip
    return _table(header, rows)
