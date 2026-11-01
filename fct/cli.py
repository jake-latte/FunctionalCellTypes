"""Command-line entry point: `python -m fct init|train|solve|plot|plot-example CACHE_DIR` and
`python -m fct exp init|list|count|status|run ROOT`.

Each subcommand is a thin wiring of library functions; all computation lives elsewhere.
Expected user errors (missing config or prerequisites, stale cache) become one stderr line
and exit code 1; anything else propagates with its traceback.
"""

import argparse
import functools
import json
import sys
import traceback
from collections.abc import Callable, Sequence
from functools import cached_property
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pydantic import ValidationError
from torch import Tensor

from fct.activations import Activation, get_phi
from fct.analyse import GridPredictions, get_grid_predictions, get_layer_pca, neuron_maps
from fct.cache import StaleCacheError, cache_paths
from fct.config import (
    Config,
    default_config,
    load_config,
    resolve_device,
    resolve_dtype,
    save_config,
)
from fct.data import Dataset, get_dataset
from fct.draw import (
    fig_example,
    fig_kernels,
    fig_learning,
    fig_neuron_maps,
    fig_representation,
    fig_solver_history,
    plot_training_history,
)
from fct.exp import (
    FIXED_AXES_PRESETS,
    LAMBDA_PRESETS,
    LEGACY_EXP_FILE,
    PRESET_NAMES,
    STAGES,
    ExpConfig,
    ExpError,
    Stage,
    UnmigratedExpError,
    exp_path,
    expand,
    has_settings,
    load_exp,
    preset,
    run_setting,
    save_exp,
    setting_dir,
    setting_statuses,
)
from fct.exp_migrate import MigrationError, migrate_exp
from fct.meanfield import (
    InadequateSamplerError,
    SpatialPriorUnsupportedError,
    UnconvergedSolveError,
    UnhealthySolveError,
    load_meanfield,
    net_kernels,
    run_meanfield,
)
from fct.model import MLP
from fct.monitor import make_mir_monitor
from fct.train import History, load_trained, select_member, train_from_config

FIGURE_DPI = 150


class CliError(Exception):
    """A user-facing failure: printed as one line, exit code 1."""


# ---------------------------------------------------------------------------- config handling
def _parse_value(text: str) -> Any:
    """JSON if it parses (`2`, `[[1,0],[0,1]]`, `null`), else the raw string (`tanh`)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _set_path(tree: dict, dotted_key: str, value: Any) -> None:
    """Assign `value` at a dotted path of an existing nested dict.

    Intermediate sections must already exist; only the last key may be new, and then
    pydantic's extra="forbid" reports it when the config is validated.
    """
    *parents, last = dotted_key.split(".")
    node = tree
    for part in parents:
        if not isinstance(node, dict) or part not in node:
            raise CliError(f"--set {dotted_key}: no config section {part!r}")
        node = node[part]
    if not isinstance(node, dict):
        raise CliError(f"--set {dotted_key}: {'.'.join(parents)!r} is not a config section")
    node[last] = value


def _override_tree(tree: dict, overrides: Sequence[str]) -> dict:
    """`tree` with each `key=value` override applied (in place)."""
    for item in overrides:
        key, sep, text = item.partition("=")
        if not sep or not key:
            raise CliError(f"--set expects key=value, got {item!r}")
        _set_path(tree, key, _parse_value(text))
    return tree


def apply_overrides(cfg: Config, overrides: Sequence[str]) -> Config:
    """A validated copy of `cfg` with each `key=value` override applied."""
    return Config.model_validate(_override_tree(cfg.model_dump(), overrides))


def _load_cfg(cache_dir: Path) -> Config:
    path = cache_paths(cache_dir).config
    if not path.exists():
        raise CliError(f"{path} not found; run `python -m fct init {cache_dir}` first")
    try:
        return load_config(path)
    except json.JSONDecodeError as err:
        raise CliError(f"{path} is not valid JSON: {err}") from err
    except ValidationError as err:
        raise CliError(f"{path} is not a valid config: {err}") from err


# ---------------------------------------------------------------------------- subcommands
def cmd_init(args: argparse.Namespace) -> int:
    path = cache_paths(args.cache_dir).config
    if path.exists() and not args.force:
        raise CliError(f"{path} already exists; pass --force to overwrite it")
    try:
        cfg = apply_overrides(default_config(), args.set)
    except ValidationError as err:
        raise CliError(str(err)) from err
    paths = cache_paths(args.cache_dir)
    stale = [d.name for d in (paths.data, paths.train, paths.meanfield) if d.exists()]
    save_config(cfg, path)
    print(f"wrote {path}")
    if stale:
        print(f"warning: {', '.join(stale)}/ hold artefacts from the old config and are now stale")
    return 0


def _load_inputs(cache_dir: Path) -> tuple[Config, Dataset]:
    cfg = _load_cfg(cache_dir)
    return cfg, get_dataset(cache_dir, cfg)


def _load_net(cache_dir: Path, cfg: Config) -> tuple[MLP, History]:
    """The trained single network; CliError for an ensemble (net.B > 1), which the analysis,
    plot and solve commands do not support."""
    try:
        net, history = load_trained(cache_dir, cfg, resolve_device(cfg), resolve_dtype(cfg.dtype))
    except FileNotFoundError as err:
        raise CliError(f"{err}; run `python -m fct train {cache_dir}` first") from err
    try:
        return select_member(net, history)
    except ValueError as err:
        raise CliError(f"{cache_dir}: {err}; this command does not support ensembles") from err


def _lazy_mir_monitor(dataset: Dataset, cfg: Config) -> Callable[[MLP], dict[str, list]]:
    """A MIR monitor built on its first call, so a finished run (which `train` returns without
    monitoring) loads even when the dataset has no factors; a fresh run still fails at once."""

    @functools.cache
    def build() -> Callable[[MLP], dict[str, list]]:
        return make_mir_monitor(dataset, cfg)

    def monitor(net: MLP) -> dict[str, list]:
        return build()(net)

    return monitor


def cmd_train(args: argparse.Namespace) -> int:
    cfg, dataset = _load_inputs(args.cache_dir)
    final = cache_paths(args.cache_dir).train / "final.pt"
    if final.exists():
        print(f"loaded trained model from {final}")
    monitor = _lazy_mir_monitor(dataset, cfg) if cfg.train.monitor_int > 0 else None
    train_from_config(cfg, dataset, args.cache_dir, log=print, monitor=monitor)
    return 0


# Errors the user fixes by changing a config or the experiment: reported as one line, no traceback.
USER_FIXABLE_ERRORS = (
    ExpError,
    InadequateSamplerError,
    SpatialPriorUnsupportedError,
    UnconvergedSolveError,
    UnhealthySolveError,
)


def cmd_solve(args: argparse.Namespace) -> int:
    cfg, dataset = _load_inputs(args.cache_dir)
    n_points, P = cfg.meanfield.n_points, dataset.x.shape[0]
    if n_points > P:
        raise CliError(f"meanfield.n_points={n_points} exceeds the dataset size data.P={P}")
    net, _ = _load_net(args.cache_dir, cfg)
    try:
        run_meanfield(cfg, dataset, net, args.cache_dir, log=print)
    except USER_FIXABLE_ERRORS as err:  # fixable by the user, no traceback
        raise CliError(str(err)) from err
    return 0


def cmd_plot_example(args: argparse.Namespace) -> int:
    cfg, dataset = _load_inputs(args.cache_dir)
    _save_figure(fig_example(dataset, cfg.data), args.cache_dir, "example")
    return 0


# ---------------------------------------------------------------------------- plot
class _Inputs:
    """Everything a figure may need, loaded from the cache; only what `needs` names is read."""

    def __init__(self, cache_dir: Path, cfg: Config, needs: set[str]):
        self.cache_dir = cache_dir
        self.cfg = cfg
        self.dataset: Dataset = get_dataset(cache_dir, cfg)
        self.net: MLP | None = None
        self.train_history: History | None = None
        self.Phi: Tensor | None = None
        self.solver_history: History | None = None
        if "train" in needs:
            self.net, self.train_history = _load_net(cache_dir, cfg)
        if "meanfield" in needs:
            self.Phi, _, self.solver_history = _load_meanfield(cache_dir, cfg)

    @property
    def phi(self) -> Activation:
        return get_phi(self.cfg.net.phi)

    @cached_property
    def grid(self) -> GridPredictions:
        return get_grid_predictions(self.cache_dir, self.cfg, self.net, self.dataset)


def _load_meanfield(cache_dir: Path, cfg: Config) -> tuple[Tensor, Tensor, History]:
    try:
        return load_meanfield(cache_dir, cfg, resolve_device(cfg))
    except FileNotFoundError as err:
        raise CliError(f"{err}; run `python -m fct solve {cache_dir}` first") from err


def _draw_learning(i: _Inputs) -> Figure:
    return fig_learning(i.grid, i.dataset.y)


def _draw_representation(i: _Inputs) -> Figure:
    return fig_representation(get_layer_pca(i.cache_dir, i.cfg, i.grid, i.phi))


def _draw_neurons(i: _Inputs) -> Figure:
    a = i.cfg.analysis
    maps = neuron_maps(i.grid.h_grid, i.phi, a.plot_layer, a.n_grid)
    title = f"layer {a.plot_layer} neuron maps"
    return fig_neuron_maps(maps, i.grid.half_width, a.n_cols, title)


def _draw_training(i: _Inputs) -> Figure:
    return plot_training_history(i.train_history).figure


def _draw_solver(i: _Inputs) -> Figure:
    return fig_solver_history(i.solver_history)


def _draw_kernels(i: _Inputs) -> Figure:
    return fig_kernels(i.Phi, net_kernels(i.net, i.dataset, i.cfg))


class _Figure(NamedTuple):
    needs: frozenset[str]  # artefacts whose final.pt must exist
    draw: Callable[[_Inputs], Figure]


FIGURES: dict[str, _Figure] = {
    "learning": _Figure(frozenset({"train"}), _draw_learning),
    "representation": _Figure(frozenset({"train"}), _draw_representation),
    "neurons": _Figure(frozenset({"train"}), _draw_neurons),
    "training": _Figure(frozenset({"train"}), _draw_training),
    "solver": _Figure(frozenset({"meanfield"}), _draw_solver),
    "kernels": _Figure(frozenset({"train", "meanfield"}), _draw_kernels),
}


def _artefact_path(cache_dir: Path, artefact: str) -> Path:
    paths = cache_paths(cache_dir)
    return {"train": paths.train, "meanfield": paths.meanfield}[artefact] / "final.pt"


def _missing_files(cache_dir: Path, needs: frozenset[str]) -> list[str]:
    missing = (_artefact_path(cache_dir, a) for a in sorted(needs))
    return [str(p.relative_to(cache_dir)) for p in missing if not p.exists()]


def _save_figure(fig: Figure, cache_dir: Path, name: str) -> None:
    out = cache_paths(cache_dir).figures
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)  # matplotlib warns once more than 20 figures stay open
    print(f"saved {path}")


def cmd_plot(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args.cache_dir)
    wanted = list(FIGURES) if args.which == "all" else [args.which]
    missing = {name: _missing_files(args.cache_dir, FIGURES[name].needs) for name in wanted}
    drawable = [name for name in wanted if not missing[name]]
    skipped = [name for name in wanted if missing[name]]
    if not drawable:
        reasons = "; ".join(f"{n} needs {', '.join(missing[n])}" for n in skipped)
        raise CliError(f"nothing to draw: {reasons}")
    needs = set().union(*(FIGURES[name].needs for name in drawable))
    inputs = _Inputs(args.cache_dir, cfg, needs)
    for name in drawable:
        _save_figure(FIGURES[name].draw(inputs), args.cache_dir, name)
    for name in skipped:
        print(f"skipped {name}: missing {', '.join(missing[name])}")
    return 0


# ---------------------------------------------------------------------------- experiment
def _load_exp(root: Path) -> ExpConfig:
    path = exp_path(root)
    try:
        return load_exp(root)
    except UnmigratedExpError as err:  # names `exp migrate`
        raise CliError(str(err)) from err
    except FileNotFoundError as err:
        raise CliError(f"{err}; run `python -m fct exp init {root}` first") from err
    except ValidationError as err:  # also covers invalid JSON
        raise CliError(f"{path} is not a valid experiment config: {err}") from err


def _override_axes(tree: dict, axes: Sequence[str], preset_name: str) -> None:
    """Replace (or add) axes in a dumped experiment tree: each item is `config.path=[v1,v2]`."""
    if axes and tree["axes"] is None:
        raise CliError(
            f"preset {preset_name!r} is the modularity experiment, which has no editable axes; "
            f"--axis needs one of {[*LAMBDA_PRESETS, *FIXED_AXES_PRESETS]}"
        )
    for item in axes:
        key, sep, text = item.partition("=")
        values = _parse_value(text)
        if not key or not sep or not isinstance(values, list):
            raise CliError(f"--axis expects config.path=[value, ...], got {item!r}")
        tree["axes"][key] = values


def _reject_axes_overrides(overrides: Sequence[str]) -> None:
    for item in overrides:
        key = item.partition("=")[0]
        if key == "axes" or key.startswith("axes."):
            raise CliError(f"--set {key}: the axes are edited with --axis PATH=[values]")


def cmd_exp_init(args: argparse.Namespace) -> int:
    path = exp_path(args.root)
    if (args.root / LEGACY_EXP_FILE).exists():
        raise CliError(
            f"{args.root} holds a {LEGACY_EXP_FILE} from older code; "
            f"run `python -m fct exp migrate {args.root}` instead of starting over"
        )
    if path.exists() and not args.force:
        raise CliError(f"{path} already exists; pass --force to overwrite it")
    if has_settings(args.root) and not args.force:
        raise CliError(
            f"{args.root} already holds setting directories, which belong to the old exp.json; "
            "use a fresh ROOT, or pass --force to write a new exp.json beside them"
        )
    _reject_axes_overrides(args.set)
    try:
        base = preset(args.preset, args.n_lambda)
    except ValueError as err:  # a missing or unwanted --n-lambda
        raise CliError(str(err)) from err
    try:
        tree = _override_tree(base.model_dump(), args.set)
        _override_axes(tree, args.axis, args.preset)
        exp = ExpConfig.model_validate(tree)
    except ValidationError as err:
        raise CliError(str(err)) from err
    save_exp(exp, args.root)
    print(f"wrote {path}")
    return 0


def _setting_name(root: Path, setting) -> str:
    return str(setting_dir(root, setting).relative_to(root))


def cmd_exp_count(args: argparse.Namespace) -> int:
    print(len(expand(_load_exp(args.root))))
    return 0


def cmd_exp_list(args: argparse.Namespace) -> int:
    for i, setting in enumerate(expand(_load_exp(args.root))):
        print(f"{i} {_setting_name(args.root, setting)}")
    return 0


def cmd_exp_status(args: argparse.Namespace) -> int:
    exp = _load_exp(args.root)
    states = [
        (i, state, _setting_name(args.root, s))
        for i, s, state in setting_statuses(args.root, exp, args.stage)
    ]
    for i, state, name in states:
        print(f"{i} {state} {name}")
    counts = [
        sum(state == kind for _, state, _ in states)
        for kind in ("done", "partial", "missing", "stale")
    ]
    print("{} done, {} partial, {} missing, {} stale".format(*counts))
    return 0


def _run_all(exp: ExpConfig, root: Path, stage: Stage) -> None:
    """Run every setting, continuing past failures; raise CliError listing the failed indices.

    A failure is printed with its index, exception type and message (plus the traceback for
    anything that is not a user-fixable error), so nothing is swallowed silently.
    """
    failed = []
    for i in range(len(expand(exp))):
        try:
            run_setting(exp, i, root, log=print, stage=stage)
        except Exception as err:  # noqa: BLE001 - reported with type, message and traceback
            failed.append(i)
            print(f"error: setting {i} failed: {type(err).__name__}: {err}", file=sys.stderr)
            if not isinstance(err, USER_FIXABLE_ERRORS):
                traceback.print_exc()
    if failed:
        raise CliError(f"{len(failed)} of {len(expand(exp))} settings failed: {failed}")


def cmd_exp_run(args: argparse.Namespace) -> int:
    """One setting with --index (a SLURM array task), else every setting in order.

    Only the USER_FIXABLE_ERRORS (bad index or config mismatch, an inadequate sampler, a spatial
    config the solver refuses, a tripped tilt tripwire) become CliError; anything else raised by training or solving keeps
    its traceback.
    """
    exp = _load_exp(args.root)
    if args.index is None:
        _run_all(exp, args.root, args.stage)
        return 0
    try:
        run_setting(exp, args.index, args.root, log=print, stage=args.stage)
    except USER_FIXABLE_ERRORS as err:  # fixable by the user, no traceback
        raise CliError(str(err)) from err
    return 0


EXP_HELP = "an experiment: a batch of runs over varied hyperparameters (its axes)"


def cmd_exp_migrate(args: argparse.Namespace) -> int:
    try:
        lines = migrate_exp(args.root)
    except MigrationError as err:
        raise CliError(str(err)) from err
    print("\n".join(lines))
    return 0


def _add_exp_parser(sub) -> None:
    p_exp = sub.add_parser(
        "exp", help="an experiment: runs over varied hyperparameters", description=EXP_HELP
    )
    actions = p_exp.add_subparsers(dest="action", required=True)

    def add(name: str, func, help_: str) -> argparse.ArgumentParser:
        p = actions.add_parser(name, help=help_, description=help_)
        p.add_argument("root", metavar="ROOT", type=Path)
        p.set_defaults(func=func)
        return p

    p_init = add("init", cmd_exp_init, "write ROOT/exp.json from a preset")
    p_init.add_argument("--preset", choices=PRESET_NAMES, default="tiny")
    p_init.add_argument(
        "--n-lambda", type=int, default=None, metavar="N_LAMBDA",
        help=f"points per log-spaced lambda axis (required by {list(LAMBDA_PRESETS)}, else not allowed)",
    )  # fmt: skip
    p_init.add_argument(
        "--axis", action="append", default=[], metavar="PATH=[V,...]",
        help="replace or add an axis, e.g. --axis net.L=[1,2,3] (repeatable)",
    )  # fmt: skip
    p_init.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="override an experiment entry by dotted key, e.g. --set n_epochs=100 (repeatable)",
    )  # fmt: skip
    p_init.add_argument(
        "--force", action="store_true",
        help="overwrite an existing exp.json, and write one beside existing setting directories",
    )  # fmt: skip
    add("migrate", cmd_exp_migrate, "convert a folder written by older code to the exp.json layout")
    add("list", cmd_exp_list, "print `index directory` for every setting")
    add("count", cmd_exp_count, "print the number of settings")
    p_status = add(
        "status", cmd_exp_status, "print done / partial / missing / stale for every setting"
    )
    p_run = add("run", cmd_exp_run, "train or solve one setting (--index) or all of them in order")
    for p in (p_status, p_run):
        p.add_argument(
            "--stage", choices=STAGES, default="train",
            help="train the nets, or solve the mean-field saddle points (default: train)",
        )  # fmt: skip
    p_run.add_argument("--index", type=int, default=None, help="setting index (see `exp list`)")


# ---------------------------------------------------------------------------- entry point
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m fct", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add(
        name: str, func: Callable[[argparse.Namespace], int], help_: str
    ) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_, description=help_)
        p.add_argument("cache_dir", metavar="CACHE_DIR", type=Path)
        p.set_defaults(func=func)
        return p

    p_init = add("init", cmd_init, "write CACHE_DIR/config.json from the defaults")
    p_init.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="override a config entry by dotted key, e.g. --set net.L=2 (repeatable)",
    )  # fmt: skip
    p_init.add_argument("--force", action="store_true", help="overwrite an existing config.json")
    add("train", cmd_train, "train the network (resumes; reuses a finished run)")
    add("solve", cmd_solve, "solve the mean-field equations (resumes; reuses a finished run)")
    p_plot = add("plot", cmd_plot, "draw the figures that need trained / solved results")
    p_plot.add_argument("--which", choices=["all", *FIGURES], default="all")
    add("plot-example", cmd_plot_example, "draw the example figure (needs no results)")
    _add_exp_parser(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    # Line-buffer so progress reaches `> log` files and SLURM logs while a run is going,
    # not only when the process exits (matters for killed or long jobs).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (CliError, StaleCacheError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
