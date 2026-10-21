"""Learning-rate tuning of the sweep: score trained settings on held-out data, then pick a rate.

Layers (kept apart so the logic is testable without a disk or a GPU):
- pure: `loss_curve_stats`, the table logic (`collapse_seeds`,
  `rank_learning_rates`, `select`, `select_by`, `global_verdict`);
- I/O edge: `evaluate_setting` (reads a finished setting, writes nothing), the CSV functions,
  `tuning_template`, `run_trials`, `collect_rows` and the command line (`python -m fct.lrtune`).

Every trial is a normal sweep: one ROOT per (learning rate, seed) filled by `fct.sweep.run_setting`,
so tuning exercises the code path of the real run. A result row has the keys of `COLUMNS`.
The held-out MSE and the regularised loss are the criteria; the MIR is recorded for context and
never used to choose. Held-out data comes from `fct.tasks.resample`.

This is a separate entry point (`python -m fct.lrtune`), not a subcommand of `python -m fct`.
"""

import argparse
import csv
import itertools
import math
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from fct.cache import cache_paths
from fct.config import Config, resolve_device, resolve_dtype
from fct.data import get_dataset
from fct.model import MLP
from fct.sweep import (
    LAYERS,
    SWEEP_TASKS,
    Seeds,
    Setting,
    SweepConfig,
    expand,
    get_setting,
    load_sweep,
    preset,
    run_setting,
    save_sweep,
    setting_config,
    setting_dir,
    setting_status,
)
from fct.tasks import Dataset, resample
from fct.train import compute_loss, load_trained

Row = dict[str, object]

# Result-row columns and their types (the CSV schema).
COLUMNS: dict[str, type] = {
    "task": str,
    "L": int,
    "phi": str,
    "rate_lambda": float,
    "lr": float,
    "seed": int,
    "n_epochs": int,
    "diverged": bool,
    "train_mse": float,
    "heldout_mse": float,
    "null_mse": float,  # held-out MSE of the zero network, the collapsed solution
    "loss": float,
    "max_spike": float,
    "tail_drop": float,
    "mir_corrected": str,  # final corrected MIR of each hidden layer, ';'-joined
}
SETTING_KEYS = ("task", "L", "phi", "rate_lambda")
CRITERIA = {  # rankable columns -> axis label
    "loss": "final regularised training loss",
    "heldout_mse": "held-out MSE",
    "train_mse": "training MSE",
}
HELD_OUT_SEED_OFFSET = 1000  # held-out seed = training data seed + this, never a training seed
SPIKE_START = 0.25  # fraction of the logged curve after which a loss spike counts
# Values closer than this (relative) are tied in a ranking. Final MSEs of nets that ended at the
# same solution agree to about 1e-4 (float32 summation order, different checkpoints); 1e-3 is the
# next decade up, so such nets tie while a 1% difference still ranks.
TIE_RTOL = 1e-3
LEARNED_MARGIN = 0.9  # a net has learned if its held-out MSE is below this x the zero net's
TAIL = 0.1  # the final fraction of the curve over which "still decreasing" is judged
MAX_MEDIAN_GAP = 0.10  # verdict: median relative gap to each setting's best lr at most this ...
MAX_WORST_RATIO = 2.0  # ... and no setting more than this many times worse than its best
DEFAULT_LAYERS = (min(LAYERS), max(LAYERS))  # the shallow and deep extremes of the grid
LOGGED_POINTS = 100  # history entries per run (log_int = n_epochs / this)
CHECKPOINTS = 10  # checkpoints per run
MONITOR_CALLS = 4  # MIR monitor calls per run, besides the final one
REQUIRED_HISTORY = ("epoch", "loss", "mse", "monitor_mir_corrected")
TUNING_FILE = "template.json"
TIMINGS_FILE = "timings.csv"
FIGURE_DPI = 150


# ---------------------------------------------------------------------------- scoring
def mse(net: MLP, data: Dataset) -> float:
    """Mean squared error over all outputs (the `mse` of `compute_loss`)."""
    with torch.no_grad():
        return (data.y - net(data.x)[1]).square().mean().item()


# ---------------------------------------------------------------------------- loss curve
def loss_curve_stats(history: dict) -> dict:
    """Stability and convergence of one logged loss curve (at least two logged points).

    - `diverged`: a non-finite loss or mse was logged;
    - `max_spike`: the largest ratio of a logged loss to the best loss since the first quarter
      of the curve, at least 1 (1 = no spike; inf if diverged);
    - `tail_drop`: relative fall of the loss over the final tenth of the curve; clearly positive
      means the run was still improving, i.e. under-trained (nan if diverged).

    The regularised loss is positive; a non-positive loss in a finite curve is an error, since
    the ratios would be meaningless.
    """
    loss = np.asarray(history["loss"], dtype=float)
    if len(loss) < 2:
        raise ValueError(f"need at least 2 logged loss values, got {len(loss)}")
    if not (np.isfinite(loss).all() and np.isfinite(np.asarray(history["mse"], dtype=float)).all()):
        return {"diverged": True, "max_spike": math.inf, "tail_drop": math.nan}
    if (loss <= 0).any():
        raise ValueError(f"the regularised loss must be positive, got min {loss.min()}")
    start = int(SPIKE_START * len(loss))
    ratios = [loss[i] / loss[start:i].min() for i in range(start + 1, len(loss))]
    tail_start = int((1 - TAIL) * (len(loss) - 1))
    return {
        "diverged": False,
        "max_spike": float(max([1.0, *ratios])),
        "tail_drop": float(1 - loss[-1] / loss[tail_start]),
    }


# ---------------------------------------------------------------------------- ranking
def average_ranks(values: np.ndarray, rtol: float = 0.0) -> np.ndarray:
    """Ranks 1.. in ascending order, ties sharing their mean rank; non-finite values rank last
    (tied with each other). Values within a relative `rtol` of the smallest value of their
    cluster are tied, so differences below the resolution of the measurement do not rank."""
    if len(values) == 0:
        raise ValueError("cannot rank an empty set of values")
    keys = np.where(np.isfinite(values), values, np.inf)
    order = np.argsort(keys, kind="stable")
    cluster = np.zeros(len(keys), dtype=int)
    start = keys[order[0]]
    for rank, i in enumerate(order[1:], start=1):
        if keys[i] > start * (1 + rtol):
            start = keys[i]
            cluster[rank] = cluster[rank - 1] + 1
        else:
            cluster[rank] = cluster[rank - 1]
    ranks = np.empty(len(keys))
    for c in np.unique(cluster):
        members = np.flatnonzero(cluster == c)
        ranks[order[members]] = members.mean() + 1
    return ranks


def setting_key(row: Row) -> tuple:
    return tuple(row[k] for k in SETTING_KEYS)


def collapse_seeds(rows: Iterable[Row], keys: Sequence[str] = ("heldout_mse",)) -> list[Row]:
    """One row per (setting, lr): the median over seeds of each column in `keys`.

    A collapsed row holds only the setting, `lr`, the `keys` medians, `n_seeds` (how many seeds
    it summarises) and `diverged`; if any seed diverged the medians are NaN and `diverged` is
    True. It has no `seed` and is not a result row. Raises ValueError on two rows for the same
    (setting, lr, seed), e.g. from two run lengths: select one with `filter_rows` first.
    """
    groups: dict[tuple, list[Row]] = defaultdict(list)
    seen = set()
    for row in rows:
        key = (*setting_key(row), row["lr"], row["seed"])
        if key in seen:
            raise ValueError(
                f"duplicate result row for {key}: rows of more than one run length? "
                "select one with filter_rows(n_epochs=...)"
            )
        seen.add(key)
        groups[(*setting_key(row), row["lr"])].append(row)
    collapsed = []
    for (*setting, lr), group in groups.items():
        diverged = any(r["diverged"] for r in group)
        medians = {
            k: math.nan if diverged else statistics.median(float(r[k]) for r in group) for k in keys
        }
        collapsed.append(
            dict(zip(SETTING_KEYS, setting))
            | {"lr": lr, "n_seeds": len(group), "diverged": diverged}
            | medians
        )
    return collapsed


def filter_rows(
    rows: Iterable[Row],
    seeds: Sequence[int] | None = None,
    n_epochs: int | None = None,
    lrs: Sequence[float] | None = None,
) -> list[Row]:
    """The rows of the given seeds, run length and learning rates (None keeps all)."""
    return [
        r
        for r in rows
        if (seeds is None or r["seed"] in seeds)
        and (n_epochs is None or r["n_epochs"] == n_epochs)
        and (lrs is None or r["lr"] in lrs)
    ]


def learned_settings(rows: Sequence[Row], margin: float = LEARNED_MARGIN) -> list[Row]:
    """The rows of the settings whose converged net has learned something.

    Rule: for each setting only its longest run length is considered (the converged value; a
    transient of a shorter run does not count); the setting has learned if, at that length, some
    lr that did not diverge gives a median-over-seeds held-out MSE below `margin` times the
    median held-out MSE of the zero network. In the other settings every lr ends at the collapsed
    solution, so the learning rate cannot be judged by what the net has learned. The returned
    rows are those longest-length rows of the learned settings.
    """
    longest: dict[tuple, int] = {}
    for r in rows:
        key = setting_key(r)
        longest[key] = max(longest.get(key, 0), r["n_epochs"])
    converged = [r for r in rows if r["n_epochs"] == longest[setting_key(r)]]
    learned: dict[tuple, bool] = defaultdict(bool)
    for r in collapse_seeds(converged, ("heldout_mse", "null_mse")):
        if not r["diverged"]:
            learned[setting_key(r)] |= r["heldout_mse"] < margin * r["null_mse"]
    return [r for r in converged if learned[setting_key(r)]]


class Score(NamedTuple):
    """One (setting, lr) cell: its value of the ranked column, the rank of this lr among the
    setting's lrs (1 = lowest, ties share a mean rank, failures last) and the relative gap of
    the value to the setting's best."""

    task: str
    L: int
    phi: str
    rate_lambda: float
    lr: float
    value: float
    rank: float
    gap: float
    failed: bool


def rank_learning_rates(
    rows: Iterable[Row], tie_rtol: float = TIE_RTOL, key: str = "heldout_mse"
) -> list[Score]:
    """Rank the learning rates inside each setting on column `key` (lower is better; the
    held-out MSE by default).

    A run that diverged, or whose MSE is not finite, is a failure: it ranks last and its gap is
    infinite. Every setting must have been run with the same set of learning rates.
    """
    by_setting: dict[tuple, list[Row]] = defaultdict(list)
    for row in collapse_seeds(rows, (key,)):
        by_setting[setting_key(row)].append(row)
    lr_sets = {frozenset(r["lr"] for r in group) for group in by_setting.values()}
    if len(lr_sets) != 1:
        raise ValueError(f"settings were run with different learning rates: {lr_sets}")
    scores = []
    for setting, group in by_setting.items():
        values = np.array([float(r[key]) for r in group])
        failed = np.array([bool(r["diverged"]) for r in group]) | ~np.isfinite(values)
        values = np.where(failed, np.inf, values)
        best = values.min()
        if best <= 0:
            raise ValueError(f"setting {setting} has {key} <= 0: {best}")
        ranks = average_ranks(values, tie_rtol)
        for r, v, rank, bad in zip(group, values, ranks, failed):
            gap = math.inf if bad else v / best - 1
            scores.append(Score(*setting, r["lr"], float(v), float(rank), float(gap), bool(bad)))
    return scores


class LrSummary(NamedTuple):
    lr: float
    mean_rank: float
    median_gap: float
    worst_ratio: float  # largest ratio of the ranked value to a setting's best (inf if any failure)
    n_failed: int
    n_settings: int


class Selection(NamedTuple):
    best: float  # lowest-lr member of `tied`
    tied: tuple[float, ...]  # all lrs equally good by (mean rank, median gap), after failures
    summaries: tuple[LrSummary, ...]  # one per lr, ascending


def _summarise(lr: float, cells: list[Score]) -> LrSummary:
    gaps = [c.gap for c in cells]
    return LrSummary(
        lr=lr,
        mean_rank=float(np.mean([c.rank for c in cells])),
        median_gap=float(np.median(gaps)),
        worst_ratio=1 + max(gaps),
        n_failed=sum(c.failed for c in cells),
        n_settings=len(cells),
    )


def _tie_key(summary: LrSummary) -> tuple:
    """Rates with the same key are tied: ranks are averages of halves, so they agree exactly when
    equal (6 decimals only absorb float summation noise); the median gap is compared at the
    resolution of the ranking, `TIE_RTOL`."""
    gap = summary.median_gap
    return (round(summary.mean_rank, 6), gap if math.isinf(gap) else round(gap / TIE_RTOL))


def select(scores: Sequence[Score]) -> Selection:
    """The learning rate with the lowest mean rank over `scores` (ties: lowest median gap).

    Only learning rates with the fewest failures are eligible, so a rate that diverged where
    another did not is never chosen. Rates equal on both criteria are all listed in `tied`.
    """
    if not scores:
        raise ValueError("no scores to select from")
    cells: dict[float, list[Score]] = defaultdict(list)
    for s in scores:
        cells[s.lr].append(s)
    summaries = tuple(_summarise(lr, cells[lr]) for lr in sorted(cells))
    fewest = min(s.n_failed for s in summaries)
    eligible = [s for s in summaries if s.n_failed == fewest]
    top = min(_tie_key(s) for s in eligible)
    tied = tuple(s.lr for s in eligible if _tie_key(s) == top)
    return Selection(best=tied[0], tied=tied, summaries=summaries)


def select_by(scores: Sequence[Score], keys: Sequence[str]) -> dict[tuple, Selection]:
    """`select` within each group of `scores` that share the values of the Score fields `keys`."""
    groups: dict[tuple, list[Score]] = defaultdict(list)
    for s in scores:
        groups[tuple(getattr(s, k) for k in keys)].append(s)
    return {key: select(group) for key, group in sorted(groups.items())}


class Verdict(NamedTuple):
    lr: float
    tied: tuple[float, ...]
    median_gap: float
    worst_ratio: float
    n_failed: int
    single_suffices: bool


def global_verdict(
    scores: Sequence[Score],
    max_median_gap: float = MAX_MEDIAN_GAP,
    max_ratio: float = MAX_WORST_RATIO,
) -> Verdict:
    """Whether the globally best lr can serve every setting: no failure, the median relative
    held-out gap to each setting's best lr at most `max_median_gap`, and no setting more than
    `max_ratio` times worse than its best."""
    selection = select(scores)
    chosen = next(s for s in selection.summaries if s.lr == selection.best)
    ok = (
        chosen.n_failed == 0
        and chosen.median_gap <= max_median_gap
        and chosen.worst_ratio <= max_ratio
    )
    return Verdict(
        lr=chosen.lr,
        tied=selection.tied,
        median_gap=chosen.median_gap,
        worst_ratio=chosen.worst_ratio,
        n_failed=chosen.n_failed,
        single_suffices=ok,
    )


# ---------------------------------------------------------------------------- I/O: evaluation
def _cpu_config(sweep: SweepConfig, setting: Setting) -> Config:
    """The setting's config on the CPU (the device is in no cache hash); evaluation is cheap."""
    return setting_config(sweep, setting).model_copy(update={"device": "cpu"})


def evaluate_setting(root: Path, sweep: SweepConfig, setting: Setting, held_out_seed: int) -> Row:
    """Score one finished setting: final train and held-out MSE, loss curve stats, final MIR.

    Read-only on ROOT: the training dataset is loaded from its cache (never generated), and
    the held-out data is built in memory. FileNotFoundError if the setting is not finished;
    ValueError if its history lacks the loss curve or the MIR monitor entries.
    """
    directory = setting_dir(root, setting)
    if setting_status(root, setting) != "done":
        raise FileNotFoundError(f"{directory} is not finished (no train/final.pt)")
    if not (cache_paths(directory).data / "dataset.pt").exists():
        raise FileNotFoundError(f"{directory} has no data/dataset.pt; refusing to generate it")
    cfg = _cpu_config(sweep, setting)
    device, dtype = resolve_device(cfg), resolve_dtype(cfg.dtype)
    train = get_dataset(directory, cfg)
    net, history = load_trained(directory, cfg, device, dtype)
    missing = [k for k in REQUIRED_HISTORY if not history.get(k)]
    if missing:
        raise ValueError(
            f"{directory}: the training history lacks {missing}; was the run monitored?"
        )
    held = resample(cfg.data, train, held_out_seed, device, dtype)
    tc = cfg.train
    with torch.no_grad():
        loss, parts = compute_loss(net, train.x, train.y, tc.weight_lambda, tc.rate_lambda)
    stats = loss_curve_stats(history)
    train_mse, held_mse = parts["mse"].item(), mse(net, held)
    null_mse = held.y.square().mean().item()
    final = [train_mse, held_mse, loss.item()]
    mir = history["monitor_mir_corrected"][-1]
    return {
        "task": setting.task,
        "L": setting.L,
        "phi": setting.phi,
        "rate_lambda": setting.rate_lambda,
        "lr": sweep.lr,
        "seed": sweep.seeds.train,
        "n_epochs": sweep.n_epochs,
        "diverged": stats["diverged"] or not all(math.isfinite(v) for v in final),
        "train_mse": train_mse,
        "heldout_mse": held_mse,
        "null_mse": null_mse,
        "loss": final[2],
        "max_spike": stats["max_spike"],
        "tail_drop": stats["tail_drop"],
        "mir_corrected": ";".join(f"{v:.4f}" for v in mir),
    }


# ---------------------------------------------------------------------------- I/O: CSV
def write_rows_csv(rows: Sequence[Row], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(
            {k: repr(float(row[k])) if isinstance(row[k], float) else row[k] for k in COLUMNS}
            for row in rows
        )


def _parse(kind: type, text: str):
    if kind is bool:
        if text not in ("True", "False"):
            raise ValueError(f"not a boolean: {text!r}")
        return text == "True"
    return kind(text)


def read_rows_csv(path: Path) -> list[Row]:
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != list(COLUMNS):
            raise ValueError(f"{path} has columns {reader.fieldnames}, expected {list(COLUMNS)}")
        return [{k: _parse(COLUMNS[k], v) for k, v in rec.items()} for rec in reader]


# ---------------------------------------------------------------------------- I/O: trials
def tuning_template(
    N: int, P: int, n_epochs: int, n_checks: int = MONITOR_CALLS, base: str = "full"
) -> SweepConfig:
    """The `base` preset with tuning sizes, on the GPU, logging `LOGGED_POINTS` points and
    saving `CHECKPOINTS` checkpoints per run and monitoring MIR `n_checks` times (cheap, but the
    final MIR is always recorded)."""
    log_int = max(1, n_epochs // LOGGED_POINTS)
    return preset(base).model_copy(
        update={
            "N": N,
            "P": P,
            "n_epochs": n_epochs,
            "log_int": log_int,
            "save_int": max(log_int, n_epochs // CHECKPOINTS),
            "monitor_int": max(1, n_epochs // n_checks),
            "device": "cuda",
        }
    )


def trial_sweep(template: SweepConfig, lr: float, seed: int) -> SweepConfig:
    """The template with learning rate `lr`; data, net and training all use `seed`."""
    return template.model_copy(update={"lr": lr, "seeds": Seeds(data=seed, net=seed, train=seed)})


def trial_root(base: Path, lr: float, seed: int) -> Path:
    return Path(base) / f"lr_{lr:g}_seed{seed}"


def subset_indices(sweep: SweepConfig, tasks: Sequence[str], layers: Sequence[int]) -> list[int]:
    """Grid indices of the settings with the given tasks and layer counts."""
    unknown = (set(tasks) - set(SWEEP_TASKS)) | (set(layers) - set(LAYERS))
    if unknown:
        raise ValueError(f"unknown tasks or layer counts: {sorted(unknown, key=str)}")
    return [i for i, s in enumerate(expand(sweep)) if s.task in tasks and s.L in layers]


def _prepare_root(sweep: SweepConfig, root: Path) -> None:
    """Write ROOT/sweep.json, or check that the one there is the same sweep."""
    if (root / "sweep.json").exists():
        if load_sweep(root) != sweep:
            raise ValueError(
                f"{root}/sweep.json differs from the requested trial; use a fresh base"
            )
    else:
        save_sweep(sweep, root)


def _device_name(sweep: SweepConfig) -> str:
    """Name of the device a trial runs on; fails if cuda was asked for and is absent."""
    if sweep.device == "cpu":
        return "cpu"
    cfg = Config(device=sweep.device)
    return torch.cuda.get_device_name(0) if resolve_device(cfg).type == "cuda" else "cpu"


def run_trials(
    template: SweepConfig,
    base: Path,
    lrs: Sequence[float],
    seeds: Sequence[int],
    indices: Sequence[int],
    log: Callable[[str], None] = print,
) -> None:
    """Train every (lr, seed, setting); a finished setting is skipped, a partial one resumes.
    Appends the wall time of each newly finished setting to BASE/timings.csv."""
    name = _device_name(template)
    log(f"device: {name}")
    for lr, seed in itertools.product(lrs, seeds):
        sweep, root = trial_sweep(template, lr, seed), trial_root(base, lr, seed)
        _prepare_root(sweep, root)
        for i in indices:
            was_done = setting_status(root, get_setting(sweep, i)) == "done"
            start = time.perf_counter()
            run_setting(sweep, i, root, log=log)
            if not was_done:
                _append_timing(base, lr, seed, i, time.perf_counter() - start, name)


def _append_timing(base: Path, lr: float, seed: int, index: int, seconds: float, gpu: str) -> None:
    path = Path(base) / TIMINGS_FILE
    new = not path.exists()
    with path.open("a") as f:
        if new:
            f.write("lr,seed,index,seconds,device\n")
        f.write(f"{lr:g},{seed},{index},{seconds:.1f},{gpu}\n")


def collect_rows(
    template: SweepConfig,
    base: Path,
    lrs: Sequence[float],
    seeds: Sequence[int],
    indices: Sequence[int],
) -> list[Row]:
    """Evaluate every (lr, seed, setting); FileNotFoundError if any is not finished."""
    rows = []
    for lr, seed in itertools.product(lrs, seeds):
        sweep, root = trial_sweep(template, lr, seed), trial_root(base, lr, seed)
        held_out_seed = seed + HELD_OUT_SEED_OFFSET
        rows += [
            evaluate_setting(root, sweep, get_setting(sweep, i), held_out_seed) for i in indices
        ]
    return rows


# ---------------------------------------------------------------------------- report
def _format_gap(value: float) -> str:
    return "fail" if math.isinf(value) else f"{100 * value:.0f}%"


def selection_markdown(title: str, selection: Selection) -> str:
    """A markdown table of one `Selection`: per lr the mean rank, median and worst gap to the
    setting's best lr, and failures; the chosen lr (and any tie) is named under the table."""
    lines = [
        f"**{title}**",
        "",
        "| lr | mean rank | median gap | worst gap | failed |",
        "|---|---|---|---|---|",
    ]
    for s in selection.summaries:
        worst = _format_gap(s.worst_ratio - 1)
        failed = f"{s.n_failed}/{s.n_settings}"
        median = _format_gap(s.median_gap)
        lines.append(f"| {s.lr:g} | {s.mean_rank:.2f} | {median} | {worst} | {failed} |")
    tied = ", ".join(f"{lr:g}" for lr in selection.tied)
    tie = f" (tied: {tied})" if len(selection.tied) > 1 else ""
    return "\n".join([*lines, "", f"best: {selection.best:g}{tie}"])


def group_table_markdown(scores: Sequence[Score], keys: Sequence[str] = ("task", "L")) -> str:
    """One table, a row per group of `keys` and a column per lr: `mean rank / median gap` of
    that lr in that group, with the group's selected lr last."""
    by_group = select_by(scores, keys)
    lrs = sorted({s.lr for s in scores})
    header = "| " + " | ".join([*keys, *(f"{lr:g}" for lr in lrs), "best"]) + " |"
    lines = [header, "|" + "---|" * (len(keys) + len(lrs) + 1)]
    for group, selection in by_group.items():
        by_lr = {s.lr: s for s in selection.summaries}
        cells = [f"{by_lr[lr].mean_rank:.1f} / {_format_gap(by_lr[lr].median_gap)}" for lr in lrs]
        best = "/".join(f"{lr:g}" for lr in selection.tied)
        lines.append("| " + " | ".join([*map(str, group), *cells, best]) + " |")
    return "\n".join(lines)


def _ranked_rows(rows: Sequence[Row], learned_only: bool) -> list[Row]:
    """The rows a report ranks: all, or only those of the settings that learned."""
    rows = learned_settings(rows) if learned_only else list(rows)
    if not rows:
        raise ValueError("no settings to report on (none learned anything?)")
    return rows


def report_markdown(
    rows: Sequence[Row],
    key: str = "loss",
    learned_only: bool = False,
    max_median_gap: float = MAX_MEDIAN_GAP,
    max_ratio: float = MAX_WORST_RATIO,
) -> str:
    """The selection report of a result table ranked on column `key`: all settings, per depth,
    per task, per task and depth, and the verdict on a single global lr. `learned_only` first
    drops the settings in which no lr beats the zero network (see `learned_settings`).

    The verdict is printed only for the regularised loss or for the settings that learned: a
    held-out or training MSE ranks the learning rates of a collapsed setting (every converged
    lr at the zero network) by how little they have trained, so a verdict from it would mislead.
    """
    scores = rank_learning_rates(_ranked_rows(rows, learned_only), key=key)
    parts = [
        "**mean rank / median gap by task and depth**\n\n" + group_table_markdown(scores),
        selection_markdown("all settings", select(scores)),
    ]
    for name, keys in (("depth", ("L",)), ("task", ("task",)), ("task and depth", ("task", "L"))):
        for group, selection in select_by(scores, keys).items():
            parts.append(selection_markdown(f"{name} {group}", selection))
    if key == "loss" or learned_only:
        verdict = global_verdict(scores, max_median_gap, max_ratio)
        parts.append(
            f"single global lr {verdict.lr:g}: median gap {_format_gap(verdict.median_gap)}, "
            f"worst gap {_format_gap(verdict.worst_ratio - 1)}, failed {verdict.n_failed}; "
            f"suffices (median gap <= {max_median_gap:.0%}, worst <= {max_ratio:g}x): "
            f"{verdict.single_suffices}"
        )
    else:
        parts.append(
            f"no verdict: ranking on {key} over settings that may have collapsed is confounded; "
            "use --key loss or --learned-only"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------- command line
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m fct.lrtune", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="write BASE/template.json")
    init.add_argument("base", type=Path)
    for name in ("N", "P", "epochs"):
        init.add_argument(f"--{name}", type=int, required=True)
    init.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    for name in ("run", "collect"):
        cmd = sub.add_parser(name)
        cmd.add_argument("base", type=Path)
        cmd.add_argument("--lrs", type=float, nargs="+", required=True)
        cmd.add_argument("--seeds", type=int, nargs="+", default=[0])
        cmd.add_argument("--tasks", nargs="+", default=list(SWEEP_TASKS))
        cmd.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
        if name == "collect":
            cmd.add_argument("--csv", type=Path, required=True)
    report = sub.add_parser("report", help="print the selection report of result CSVs")
    report.add_argument("csv", type=Path, nargs="+")
    report.add_argument("--fig-dir", type=Path, help="also save the figures here")
    report.add_argument("--key", default="loss", choices=list(CRITERIA))
    report.add_argument("--learned-only", action="store_true")
    report.add_argument("--max-median-gap", type=float, default=MAX_MEDIAN_GAP)
    report.add_argument("--max-ratio", type=float, default=MAX_WORST_RATIO)
    report.add_argument("--seeds", type=int, nargs="+", help="use only these seeds")
    report.add_argument("--n-epochs", type=int, help="use only runs of this length")
    report.add_argument("--lrs", type=float, nargs="+", help="use only these learning rates")
    return parser


def _load_template(base: Path) -> SweepConfig:
    path = base / TUNING_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `init` first")
    return SweepConfig.model_validate_json(path.read_text())


def _report(args: argparse.Namespace) -> int:
    rows = [row for path in args.csv for row in read_rows_csv(path)]
    rows = filter_rows(rows, args.seeds, args.n_epochs, args.lrs)
    print(report_markdown(rows, args.key, args.learned_only, args.max_median_gap, args.max_ratio))
    if args.fig_dir is not None:
        from fct.lrplot import fig_metric_vs_lr  # plotting is not needed to tune

        shown = _ranked_rows(rows, args.learned_only)
        scores = rank_learning_rates(shown, key=args.key)
        best = {k: sel.best for k, sel in select_by(scores, ("task", "L")).items()}
        args.fig_dir.mkdir(parents=True, exist_ok=True)
        fig = fig_metric_vs_lr(shown, args.key, CRITERIA[args.key], args.key == "loss", best)
        fig.savefig(args.fig_dir / f"{args.key}_vs_lr.png", dpi=FIGURE_DPI)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init":
        template = tuning_template(args.N, args.P, args.epochs).model_copy(
            update={"device": args.device}
        )
        args.base.mkdir(parents=True, exist_ok=True)
        (args.base / TUNING_FILE).write_text(template.model_dump_json(indent=2) + "\n")
        return 0
    if args.command == "report":
        return _report(args)
    template = _load_template(args.base)
    indices = subset_indices(template, args.tasks, args.layers)
    if args.command == "run":
        run_trials(template, args.base, args.lrs, args.seeds, indices)
    else:
        rows = collect_rows(template, args.base, args.lrs, args.seeds, indices)
        write_rows_csv(rows, args.csv)
        print(f"wrote {len(rows)} rows to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
