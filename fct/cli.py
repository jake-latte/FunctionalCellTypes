"""Command-line entry point: `python -m fct init|train|solve|plot|plot-example CACHE_DIR`.

Each subcommand is a thin wiring of library functions; all computation lives elsewhere.
Expected user errors (missing config or prerequisites, stale cache) become one stderr line
and exit code 1; anything else propagates with its traceback.
"""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pydantic import ValidationError

from fct.activations import get_phi
from fct.analyse import get_grid_predictions, get_layer_pca, neuron_maps
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
from fct.meanfield import empirical_kernels, load_meanfield, run_meanfield
from fct.model import MLP, make_net
from fct.train import load_trained, train

FIGURE_DPI = 150


class CliError(Exception):
    """A user-facing failure: printed as one line, exit code 1."""


# ---------------------------------------------------------------------------- config handling
def _parse_value(text: str):
    """JSON if it parses (`2`, `[[1,0],[0,1]]`, `null`), else the raw string (`tanh`)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _set_path(tree: dict, dotted_key: str, value) -> None:
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


def apply_overrides(cfg: Config, overrides: Sequence[str]) -> Config:
    """A validated copy of `cfg` with each `key=value` override applied."""
    tree = cfg.model_dump()
    for item in overrides:
        key, sep, text = item.partition("=")
        if not sep or not key:
            raise CliError(f"--set expects key=value, got {item!r}")
        _set_path(tree, key, _parse_value(text))
    return Config.model_validate(tree)


def _load_cfg(cache_dir: Path) -> Config:
    path = cache_paths(cache_dir).config
    if not path.exists():
        raise CliError(f"{path} not found; run `python -m fct init {cache_dir}` first")
    return load_config(path)


# ---------------------------------------------------------------------------- subcommands
def cmd_init(args: argparse.Namespace) -> int:
    path = cache_paths(args.cache_dir).config
    if path.exists() and not args.force:
        raise CliError(f"{path} already exists; pass --force to overwrite it")
    try:
        cfg = apply_overrides(default_config(), args.set)
    except ValidationError as err:
        raise CliError(str(err)) from err
    save_config(cfg, path)
    print(f"wrote {path}")
    return 0


def _load_inputs(cache_dir: Path) -> tuple[Config, Dataset]:
    cfg = _load_cfg(cache_dir)
    return cfg, get_dataset(cache_dir, cfg)


def cmd_train(args: argparse.Namespace) -> int:
    cfg, dataset = _load_inputs(args.cache_dir)
    net = make_net(
        cfg.net,
        dataset.x.shape[1],
        dataset.y.shape[1],
        resolve_device(cfg),
        resolve_dtype(cfg.dtype),
    )
    train(cfg, dataset, net, args.cache_dir, log=print)
    return 0


def cmd_solve(args: argparse.Namespace) -> int:
    cfg, dataset = _load_inputs(args.cache_dir)
    net, _ = load_trained(args.cache_dir, cfg, resolve_device(cfg), resolve_dtype(cfg.dtype))
    run_meanfield(cfg, dataset, net, args.cache_dir, log=print)
    return 0


def cmd_plot_example(args: argparse.Namespace) -> int:
    cfg, dataset = _load_inputs(args.cache_dir)
    _save_figure(fig_example(dataset, cfg.data), args.cache_dir, "example")
    return 0


# ---------------------------------------------------------------------------- plot
class _Inputs:
    """Everything a figure may need, loaded from the cache."""

    def __init__(self, cache_dir: Path, needs: set[str]):
        self.cache_dir = cache_dir
        self.cfg = _load_cfg(cache_dir)
        self.dataset = get_dataset(cache_dir, self.cfg)
        self.device = resolve_device(self.cfg)
        self.dtype = resolve_dtype(self.cfg.dtype)
        self.net: MLP | None = None
        self.train_history = None
        self.Phi = None
        self.solver_history = None
        if "train" in needs:
            self.net, self.train_history = load_trained(
                cache_dir, self.cfg, self.device, self.dtype
            )
        if "meanfield" in needs:
            self.Phi, _, self.solver_history = load_meanfield(cache_dir, self.cfg, self.device)

    @property
    def phi(self):
        return get_phi(self.cfg.net.phi)

    @property
    def grid(self):
        return get_grid_predictions(self.cache_dir, self.cfg, self.net, self.dataset)


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
    # The theory works in float64 on the first n_points inputs; the net is evaluated on the
    # same inputs in its own dtype.
    X = i.dataset.x[: i.cfg.meanfield.n_points]
    net_Phi, _ = empirical_kernels(i.net, X.to(i.net.dtype), i.cfg.meanfield.readout)
    return fig_kernels(i.Phi, net_Phi.to(i.Phi.dtype))


# figure name -> (artefacts it needs, drawing function)
FIGURES: dict[str, tuple[frozenset[str], Callable[[_Inputs], Figure]]] = {
    "learning": (frozenset({"train"}), _draw_learning),
    "representation": (frozenset({"train"}), _draw_representation),
    "neurons": (frozenset({"train"}), _draw_neurons),
    "training": (frozenset({"train"}), _draw_training),
    "solver": (frozenset({"meanfield"}), _draw_solver),
    "kernels": (frozenset({"train", "meanfield"}), _draw_kernels),
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
    _load_cfg(args.cache_dir)  # fail on a missing config before looking at artefacts
    wanted = list(FIGURES) if args.which == "all" else [args.which]
    missing = {name: _missing_files(args.cache_dir, FIGURES[name][0]) for name in wanted}
    drawable = [name for name in wanted if not missing[name]]
    skipped = [name for name in wanted if missing[name]]
    if not drawable:
        reasons = "; ".join(f"{n} needs {', '.join(missing[n])}" for n in skipped)
        raise CliError(f"nothing to draw: {reasons}")
    needs = set().union(*(FIGURES[name][0] for name in drawable))
    inputs = _Inputs(args.cache_dir, needs)
    for name in drawable:
        _save_figure(FIGURES[name][1](inputs), args.cache_dir, name)
    for name in skipped:
        print(f"skipped {name}: missing {', '.join(missing[name])}")
    return 0


# ---------------------------------------------------------------------------- entry point
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m fct", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func: Callable[[argparse.Namespace], int], help_: str):
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (CliError, FileNotFoundError, StaleCacheError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
