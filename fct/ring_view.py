"""The linear-ring notebook's view of an experiment: load settings and their records (the I/O
layer), hand them to `fct.linear_ring` (the analysis), and write the notebook's markdown tables
and panel arguments for `fct.draw` (the tables live here because they are text for this one
notebook, not figures).

A setting that is not trained is a status, not an error (`RingRun.trained`). The MIR readers
(`fct.report`, `fct.exp_report`) refuse experiments with the spectrum monitor by design, and
nothing here calls them.
"""

import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor

from fct import linear_ring as lr
from fct.exp import (
    B_AXIS,
    ExpConfig,
    axes_of,
    expand,
    load_exp,
    parse_derived_description,
    preset,
    same_value,
)
from fct.monitor import spectrum_records
from fct.phase_view import ensemble_statistics
from fct.spatial_exp import ExpSetting, select_setting, settings_table
from fct.train import MONITOR_EPOCH, History, load_trained

CPU, F64 = torch.device("cpu"), torch.float64
# Sentences of a preset description that state its regime: what the accuracy budget allows.
_REGIME_WORDS = re.compile(r"accura|bias", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=\.)\s+(?=[A-Z(`])")
_HEX, _THETA = "monitor_hex", "monitor_theta_hist"


# ------------------------------------------------------------------ loading
class RingRun(NamedTuple):
    """One setting: its predicted law and, when it is trained, its records of the one hidden
    layer in float64 (R records, B members, K latent rows and the null row, N frequencies).
    The last record is the final state, each member's one sample; the records before it feed
    only the relaxation diagnostic. Read them through the properties, not by slicing."""

    root: Path  # the experiment ROOT
    chosen: ExpSetting
    law: lr.Law
    description: str | None  # the experiment's description
    history: History | None
    epochs: Tensor | None  # (R,)
    records: Tensor | None  # (R, B, K + 1, N)
    hex_final: Tensor | None  # (B, 2) final hexatic moments, recorded at K >= 3
    theta_final: Tensor | None  # (B, N_THETA_BINS) final angle histograms, recorded at K >= 3
    base_B: int  # `recorded_base_B`: the B before any `exp extend`

    @property
    def trained(self) -> bool:
        return self.chosen.status == "done"

    def _require_trained(self) -> None:
        if not self.trained:
            raise ValueError(f"{run_title(self)} is not trained (status {self.chosen.status})")

    @property
    def K(self) -> int:
        return self.law.D.shape[0]

    @property
    def final_spectra(self) -> Tensor:
        """(B, K + 1, N): every member's final state, the null row last."""
        self._require_trained()
        return self.records[-1]

    @property
    def final_latent(self) -> Tensor:
        """(B, K, N): the latent rows of the final state."""
        return self.final_spectra[:, : self.K]

    @property
    def relaxation_epochs(self) -> Tensor:
        """(R - 1,): the epochs of the records before the final state."""
        self._require_trained()
        return self.epochs[:-1]

    @property
    def relaxation_records(self) -> Tensor:
        """(R - 1, B, K, N): the latent rows of the records before the final state, so the
        diagnostic does not reuse the sample whose stationarity it vouches for."""
        self._require_trained()
        return self.records[:-1, :, : self.K]


def final_member_values(history: Mapping[str, Any], key: str, ensemble: bool) -> Tensor:
    """(B, ...): monitor column `key` at the last record, of the first hidden layer, one row per
    member (B = 1 for a single network, `ensemble` False). ValueError if the run did not record
    it."""
    entries = history.get(key)
    if not entries:
        raise ValueError(f"the history has no {key!r} records; was the spectrum monitor on?")
    last = entries[-1] if ensemble else [entries[-1]]  # members, each a list over layers
    return torch.stack([layers[0] for layers in last])


def _load(root: Path, exp: ExpConfig, index: int, values: Mapping[str, Any]) -> RingRun:
    chosen = select_setting(root, values, "cpu")
    law = lr.predicted_law(exp, index)
    base_B = recorded_base_B(exp, chosen.cfg.net.B, B_AXIS in axes_of(exp))
    if chosen.status != "done":
        return RingRun(root, chosen, law, exp.description, None, None, None, None, None, base_B)
    if chosen.cfg.net.L != 1:
        raise ValueError(
            f"the linear ring has one hidden layer, the setting has {chosen.cfg.net.L}"
        )
    _, history = load_trained(chosen.directory, chosen.cfg, CPU, F64)
    ensemble = chosen.cfg.net.B > 1
    angles = law.D.shape[0] >= lr.ANGLE_K  # the monitor records them from K = 3 on
    return RingRun(
        root=root,
        chosen=chosen,
        law=law,
        description=exp.description,
        history=history,
        epochs=torch.tensor(history[MONITOR_EPOCH], dtype=F64),
        records=spectrum_records(history)[:, :, 0].to(F64),
        hex_final=final_member_values(history, _HEX, ensemble) if angles else None,
        theta_final=final_member_values(history, _THETA, ensemble) if angles else None,
        base_B=base_B,
    )


def load_ring_settings(
    root: Path, values: Mapping[str, Any] | None = None, loaded: Iterable[RingRun] = ()
) -> list[RingRun]:
    """Every setting of the experiment at `root` whose axes equal `values` (a subset of the
    axes; all settings if None), in index order, trained or not. A run in `loaded` (same root
    and index) is reused instead of read again. LookupError for an axis the experiment lacks."""
    root = Path(root)
    exp = load_exp(root)
    values = dict(values or {})
    if unknown := sorted(set(values) - set(axes_of(exp))):
        raise LookupError(f"unknown axis {unknown}; the experiment has axes {list(axes_of(exp))}")
    have = {(run.root.resolve(), run.chosen.setting.index): run for run in loaded}
    return [
        have.get((root.resolve(), s.index)) or _load(root, exp, s.index, s.axes)
        for s in expand(exp)
        if all(same_value(s.axes[a], v) for a, v in values.items())
    ]


def load_ring_setting(root: Path, values: Mapping[str, Any]) -> RingRun:
    """The one setting whose axes equal `values`; LookupError for none or several."""
    runs = load_ring_settings(root, values)
    if len(runs) != 1:
        raise LookupError(
            f"{len(runs)} settings of {root} match {dict(values)}; give axes that pick one"
        )
    return runs[0]


def load_shown(
    root: Path,
    values: Mapping[str, Any],
    twin_root: Path | None = None,
    twin_values: Mapping[str, Any] | None = None,
) -> dict[str, RingRun]:
    """The notebook's settings: {"setting": ...} and, when `twin_values` is given, "twin" in
    `twin_root` (default `root`)."""
    shown = {"setting": load_ring_setting(root, values)}
    if twin_values is not None:
        shown["twin"] = load_ring_setting(twin_root or root, twin_values)
    return shown


def trained(runs: Mapping[str, RingRun]) -> dict[str, RingRun]:
    return {name: run for name, run in runs.items() if run.trained}


# ------------------------------------------------------------------ the base B
def recorded_base_B(exp: ExpConfig, cfg_B: int, B_is_axis: bool) -> int:
    """The B a setting was designed with: its own net.B when B is an axis (`exp extend` refuses
    such experiments, so it never grew), else the base_B `exp extend` recorded, else the
    experiment's B (it was never extended)."""
    if B_is_axis:
        return cfg_B
    return exp.B if exp.base_B is None else exp.base_B


def check_base_B(value: Any) -> int:
    """A base B given by hand: a positive int (not a float, which would truncate silently, nor
    a bool)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"a base B must be an int, got {value!r}")
    if value < 1:
        raise ValueError(f"a base B must be a positive number of members, got {value}")
    return value


def base_B_of(run: RingRun, given: int | None = None) -> int:
    """The B the accuracy budget of `run` is computed at (`lr.budget_se`): `given` (checked by
    `check_base_B`) if not None, else the one recorded with the experiment (`run.base_B`)."""
    return run.base_B if given is None else check_base_B(given)


# ------------------------------------------------------------------ measuring
class LossCurves(NamedTuple):
    """`fct.draw.plot_ensemble_loss`'s arguments: member mean and sem at each logged epoch."""

    epoch: Any
    mean: dict
    sem: dict


class RunMeasurement(NamedTuple):
    final: lr.FinalMeasurement
    relaxation: lr.RelaxationView
    loss: LossCurves
    base_B: int


def measure_run(
    run: RingRun, burn_in: float | None = None, base_B: int | None = None, seed: int = 0
) -> RunMeasurement:
    """Everything the notebook shows of one trained setting: its final states, the relaxation
    of its records from `burn_in` (default the second half), judged against the budget at its
    base B (`base_B_of`: `base_B` if given, else the one its experiment recorded), and its loss
    curves."""
    B0 = base_B_of(run, base_B)
    stats = ensemble_statistics(run.history)
    return RunMeasurement(
        final=lr.measure_final(run.law, run.final_spectra, run.hex_final, run.theta_final, seed),
        relaxation=lr.measure_relaxation(
            run.law, run.relaxation_epochs, run.relaxation_records, B0, burn_in, seed
        ),
        loss=LossCurves(stats.epoch, stats.mean, stats.sem),
        base_B=B0,
    )


def measure_runs(
    runs: Mapping[str, RingRun],
    burn_in: float | None,
    base_B: Mapping[str, int] | None = None,
    seed: int = 0,
) -> dict[str, RunMeasurement]:
    """`measure_run` of each of `runs`, with `base_B[name]` where given (the setting and its
    twin may be different experiments with different base Bs); a name not in `runs` is an
    error, so a typo cannot silently fall back to the recorded B."""
    given = dict(base_B or {})
    if unknown := sorted(set(given) - set(runs)):
        raise KeyError(f"BASE_B names {unknown}, which are not among {sorted(runs)}")
    return {name: measure_run(run, burn_in, given.get(name), seed) for name, run in runs.items()}


def panels(measured: Mapping[str, Any], part: str) -> list[dict]:
    """The arguments of one figure per measurement: the part at dotted path `part`
    ("final.spectrum", say) of each, as keyword arguments of the matching `fct.draw.plot_*`."""

    def get(m: Any) -> dict:
        for name in part.split("."):
            m = getattr(m, name)
        return m._asdict()

    return [get(m) for m in measured.values()]


class AnglePanels(NamedTuple):
    titles: list[str]
    panels: list[dict]  # `plot_angle_histogram`'s arguments
    note: str  # what to say when there are none


def angle_panels(
    runs: Mapping[str, RingRun], measured: Mapping[str, RunMeasurement]
) -> AnglePanels:
    """The angle histograms of the settings at K = 3 (`lr.ANGLE_K`), the only K they exist at."""
    have = {n: m.final.angles for n, m in measured.items() if m.final.angles is not None}
    Ks = sorted({run.K for run in runs.values()})
    note = (
        f"The angle statistics are defined only at K = {lr.ANGLE_K}; the shown settings have "
        f"K = {', '.join(map(str, Ks))}: skipped."
    )
    return AnglePanels(
        [run_title(runs[n]) for n in have], [a.histogram._asdict() for a in have.values()], note
    )


# ------------------------------------------------------------------ the band share sweep
class Sweep(NamedTuple):
    """`fct.draw.plot_share_sweep`'s arguments."""

    values: Tensor  # (n,) the swept value of each setting (its index without an axis)
    shares: Tensor  # (n, K)
    se: Tensor  # (n, K)
    theory: Tensor  # (n, K)
    xlabel: str


def sweep_filter(run: RingRun, axis: str | None) -> dict[str, Any]:
    """The axes a sweep holds fixed at `run`'s values: every axis but `axis` (none when `axis`
    is None, which sweeps every trained setting against its index)."""
    if axis is None:
        return {}
    axes = run.chosen.setting.axes
    if axis not in axes:
        raise LookupError(f"unknown sweep axis {axis}; the experiment has axes {list(axes)}")
    return {a: v for a, v in axes.items() if a != axis}


def sweep_groups(
    runs: Mapping[str, RingRun], axis: str | None
) -> dict[tuple, tuple[Path, dict[str, Any]]]:
    """One sweep per distinct (ROOT, fixed axes) among `runs`: two shown settings of one ROOT
    that differ in an axis other than `axis` (the parametrisation, say) get one sweep each."""
    groups = {}
    for run in runs.values():
        fixed = sweep_filter(run, axis)
        key = (run.root.resolve(), tuple(sorted((a, repr(v)) for a, v in fixed.items())))
        groups.setdefault(key, (run.root, fixed))
    return groups


def _swept_value(run: RingRun, axis: str | None) -> float:
    if axis is None:
        return float(run.chosen.setting.index)
    value = run.chosen.setting.axes[axis]
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"the sweep axis {axis} must be numeric, got {value!r}")
    return float(value)


def band_share_sweep(runs: Sequence[RingRun], axis: str | None, seed: int = 0) -> Sweep:
    """The band shares (`lr.band_share_estimate`) of the trained `runs` against their value of
    `axis` (their index if None). ValueError if none is trained."""
    done = [run for run in runs if run.trained]
    if not done:
        raise ValueError("no trained setting to sweep")
    estimates = [lr.band_share_estimate(run.law, run.final_latent, seed) for run in done]
    return Sweep(
        values=torch.tensor([_swept_value(run, axis) for run in done], dtype=F64),
        shares=torch.stack([values for values, _, _ in estimates]),
        se=torch.stack([se for _, se, _ in estimates]),
        theory=torch.stack([predicted for _, _, predicted in estimates]),
        xlabel="setting index" if axis is None else axis,
    )


def load_sweeps(
    shown: Mapping[str, RingRun], axis: str | None, seed: int = 0
) -> tuple[list[str], list[dict]]:
    """(titles, `plot_share_sweep` arguments): one sweep per `sweep_groups` group, over the
    settings of its ROOT with its fixed axes (the shown runs reused, not reloaded)."""
    titles, sweeps = [], []
    for root, fixed in sweep_groups(shown, axis).values():
        runs = [r for r in load_ring_settings(root, fixed, shown.values()) if r.trained]
        held = ", ".join(f"{a.rsplit('.', 1)[-1]} = {v}" for a, v in fixed.items())
        indices = [r.chosen.setting.index for r in runs]
        titles.append(f"{root.name}: settings {indices}" + (f" ({held})" if held else ""))
        sweeps.append(band_share_sweep(runs, axis, seed)._asdict())
    return titles, sweeps


# ------------------------------------------------------------------ text
def run_title(run: RingRun) -> str:
    """A short title: the setting's index, parametrisation and axis values (by their last name)."""
    axes = ", ".join(f"{a.rsplit('.', 1)[-1]} = {v}" for a, v in run.chosen.setting.axes.items())
    return f"setting {run.chosen.setting.index} ({run.law.cfg.spatial.parametrisation}): {axes}"


def titles(runs: Mapping[str, RingRun]) -> list[str]:
    return [run_title(run) for run in runs.values()]


def settings_tables(roots: Iterable[Path]) -> str:
    """`settings_table` of each distinct experiment ROOT, headed by its path."""
    distinct = {Path(root).resolve(): Path(root) for root in roots}
    return "\n\n".join(f"`{root}`:\n\n{settings_table(root)}" for root in distinct.values())


def describe_run(run: RingRun) -> str:
    """A markdown paragraph: which setting, how it was trained, and its state."""
    cfg = run.law.cfg
    head = (
        f"**{run_title(run)}** of `{run.root.name}`: N = {cfg.net.N}, "
        f"P = {cfg.data.P}, K = {run.K}, ell = {run.law.ell} sites, "
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


def describe_runs(runs: Mapping[str, RingRun]) -> str:
    return "\n\n".join(describe_run(run) for run in runs.values())


def measurement_note(runs: Mapping[str, RingRun], measured: Mapping[str, RunMeasurement]) -> str:
    return "\n\n".join(
        f"**{run_title(runs[n])}**: {m.final.B} members, base B {m.base_B} for the budget, "
        f"{int(m.final.primary.disagree.sum())} of {len(m.final.primary.stats)} primary "
        "statistics disagree."
        for n, m in measured.items()
    )


def regime_sentences(description: str | None) -> list[str]:
    """The sentences of a description that speak of accuracy or bias: where a preset states
    its regime and what its lr does to it."""
    if not description:
        return []
    return [s for s in _SENTENCE_END.split(description) if _REGIME_WORDS.search(s)]


def regime_markdown(root: Path, description: str | None) -> str:
    """What an experiment's description says of its regime (`regime_sentences`), then the
    whole description folded. An experiment made from a preset with overrides has only a note
    of them as its description: the preset's own is quoted instead, with a warning that the
    overrides may have moved the experiment out of the regime it documents."""
    overrides = parse_derived_description(description)
    if overrides is None:
        head, text = f"**`{root}`**, from its description:", description
    else:
        text = preset(overrides.preset).description
        given = "; ".join(
            [f"--set {s}" for s in overrides.sets] + [f"--axis {a}" for a in overrides.axes]
        )
        head = (
            f"**`{root}`** was made from the preset `{overrides.preset}` with overrides "
            f"({given}); they may move it out of the regime the preset documents. The preset's "
            "description:"
        )
    said = "\n".join(f"- {s}" for s in regime_sentences(text)) or "- (it states no regime)"
    whole = text or "(no description)"
    return (
        f"{head}\n\n{said}\n\n<details><summary>the whole description</summary>\n\n"
        f"{whole}\n\n</details>"
    )


def regimes_markdown(runs: Mapping[str, RingRun]) -> str:
    """`regime_markdown` of each distinct ROOT among `runs`."""
    by_root = {run.root.resolve(): run for run in runs.values()}
    return "\n\n".join(regime_markdown(run.root, run.description) for run in by_root.values())


# ------------------------------------------------------------------ tables
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


def _primary_rows(p: lr.Primary, ell: int, accurate: list[bool] | None) -> list[list[str]]:
    rows = []
    for i, stat in enumerate(p.stats):
        v, se, pred = float(p.values[i]), float(p.se[i]), float(p.predicted[i])
        regime = "not assessed" if accurate is None else ("yes" if accurate[i] else "no")
        rows.append([lr.statistic_name(stat, ell), lr.statistic_mode(stat), "primary",
                     f"{pred:.6g}", f"{v:.6g}", f"{se:.4g}", _z(v, pred, se),
                     _flag(bool(p.disagree[i])), regime])  # fmt: skip
    return rows


def _secondary_rows(s: lr.Secondary, ell: int) -> list[list[str]]:
    rows = []
    for i, stat in enumerate(s.stats):
        v, se, pred = float(s.values[i]), float(s.se[i]), float(s.predicted[i])
        rows.append([lr.statistic_name(stat, ell), lr.statistic_mode(stat), "secondary",
                     f"{pred:.6g}", f"{v:.6g}", f"{se:.4g}", _z(v, pred, se),
                     _flag(bool(s.disagree[i])), "-"])  # fmt: skip
    for kind, reason in s.unavailable.items():
        name = lr.statistic_name(lr.Statistic(kind, None), ell)
        rows.append([name, "all", "secondary", "-", "unavailable", "-", "-", reason, "-"])
    return rows


def _angle_rows(a: lr.Angles | None, ell: int) -> list[list[str]]:
    if a is None:
        return []
    stat = lr.Statistic("hexatic", None)
    verdict = (
        f"below N·B threshold ({a.n_samples} < {lr.HEXATIC_MIN_SAMPLES})"
        if a.flagged is None
        else _flag(a.flagged)
    )
    return [[lr.statistic_name(stat, ell), lr.statistic_mode(stat), "secondary", "0",
             f"{float(a.hexatic):.6g}", f"{float(a.hexatic_se):.4g}", "n/a", verdict,
             "-"]]  # fmt: skip


def summary_table(
    m: lr.FinalMeasurement, law: lr.Law, base_B: int, exact: Sequence[float] | None = None
) -> str:
    """Every statistic of the member-averaged final-state spectrum against its prediction, with
    its bootstrap SE: the primary falsifiers, then the secondary q_1/2 and q = 0 shares (an
    unavailable one with its reason) and, at K = 3, the hexatic modulus. `exact`, the exact
    posterior's statistics in `statistic_list` order, states `regime_accurate` at `base_B`."""
    if exact is not None and len(exact) != len(m.primary.stats):
        raise ValueError(f"exact has {len(exact)} statistics, expected {len(m.primary.stats)}")
    accurate = (
        None
        if exact is None
        else lr.regime_accurate(torch.tensor(exact, dtype=F64), law, base_B).tolist()
    )
    rows = (
        _primary_rows(m.primary, law.ell, accurate)
        + _secondary_rows(m.secondary, law.ell)
        + _angle_rows(m.angles, law.ell)
    )
    return _table(SUMMARY_HEADER, rows)


def equivalence_table(measurements: Mapping[str, lr.Primary], ell: int) -> str:
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
        [lr.statistic_name(stat, ell), lr.statistic_mode(stat),
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


def relaxation_table(view: lr.RelaxationView, ell: int) -> str:
    """Per statistic, the deficit still to relax at the last record before the final state, its
    SE, the fitted time constant, the accuracy budget and `is_stationary`."""
    header = ["statistic", "mode", "deficit", "deficit SE", "tau (epochs)", "budget",
              "stationary"]  # fmt: skip
    fit = view.fit
    rows = [
        [lr.statistic_name(stat, ell), lr.statistic_mode(stat),
         f"{float(fit.deficit[i]):.4g}", f"{float(fit.deficit_se[i]):.4g}",
         f"{float(fit.tau[i]):.4g}", f"{float(view.budget[i]):.4g}",
         "yes" if view.stationary[i] else "**no**"]
        for i, stat in enumerate(view.stats)
    ]  # fmt: skip
    return _table(header, rows)


def relaxation_tables(runs: Mapping[str, RingRun], measured: Mapping[str, RunMeasurement]) -> str:
    return "\n\n".join(
        f"**{run_title(runs[n])}**, fit from epoch {m.relaxation.curves.start:g}, budget at base "
        f"B {m.base_B}:\n\n" + relaxation_table(m.relaxation, runs[n].law.ell)
        for n, m in measured.items()
    )


def summary_tables(
    runs: Mapping[str, RingRun],
    measured: Mapping[str, RunMeasurement],
    exact: Mapping[str, Sequence[float] | None],
) -> str:
    """`summary_table` of every measured setting (with `exact[name]`), then, for two, their
    `equivalence_table`, each path named by its role and parametrisation."""
    parts = [
        f"**{run_title(runs[n])}** ({m.final.B} members)\n\n"
        + summary_table(m.final, runs[n].law, m.base_B, exact.get(n))
        for n, m in measured.items()
    ]
    if len(measured) == 2:
        paths = {
            f"{n} ({runs[n].law.cfg.spatial.parametrisation})": m.final.primary
            for n, m in measured.items()
        }
        ell = next(iter(runs.values())).law.ell
        parts.append("**Equivalence of the two settings**\n\n" + equivalence_table(paths, ell))
    return "\n\n".join(parts)
