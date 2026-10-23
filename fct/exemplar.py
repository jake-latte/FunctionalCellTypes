"""The exemplar notebook's logic: a config from flat notebook settings, the cache layout, the
cached train and solve steps, and the live plot refresh that hangs on `solve(on_outer=...)`.

The notebook (`notebooks/exemplar.ipynb`) only sets variables and calls this module and
`fct.draw`. Training and solving go through the ordinary `fct` cache path, so a rerun resumes
from the checkpoints, and a finished step is loaded instead of redone.

Cache layout under CACHE_ROOT:

    exemplar-<hash of data, net, train>/           data/, train/, config.json
        solve-<hash of data, net, train, meanfield>/   meanfield/

so changing a solver setting re-solves but never retrains, and changing the task or training
starts a new directory. The hashes are `Config.section_hash`, the ones the artefacts are
themselves guarded with.
"""

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from fct.analyse import ExampleOutputs, example_outputs, pick_examples
from fct.cache import cache_paths
from fct.config import (
    Config,
    DataConfig,
    MeanFieldConfig,
    NetConfig,
    SolverConfig,
    TrainConfig,
    save_config,
)
from fct.data import Dataset, get_dataset
from fct.draw import fig_kernels, fig_layer_kernels, fig_sampler_health, fig_solver_health
from fct.meanfield import (
    History,
    Log,
    OnOuter,
    OuterSnapshot,
    SolveResult,
    net_kernels,
    run_meanfield,
)
from fct.model import MLP
from fct.train import train_from_config

CACHE_ROOT_ENV = "FCT_CACHE_ROOT"
N_EXAMPLE_OUTPUTS = 4
EXAMPLE_SEED = 0

Show = Callable[[Sequence[Figure]], None]


def exemplar_config(
    *,
    P: int,
    m: int,
    k: int,
    N: int,
    L: int,
    phi: str,
    g: float,
    data_seed: int,
    net_seed: int,
    weight_lambda: float,
    rate_lambda: float,
    optimizer: str,
    beta: float,
    lr: float,
    n_epochs: int,
    train_seed: int,
    log_int: int,
    save_int: int,
) -> Config:
    """The linear-teacher config of the notebook's settings, on the CPU in float64.

    `beta` is both the SGLD inverse temperature of the training and the beta of the mean-field
    action, so the solved theory describes the distribution the net was sampled from; with
    Adam it only sets the theory. All P training points enter the solve. The solver settings
    are left at their defaults, `with_solver` sets them."""
    return Config(
        data=DataConfig(task="linear_teacher", P=P, D_in=m, n_latents=k, seed=data_seed),
        net=NetConfig(N=N, L=L, g=g, phi=phi, seed=net_seed),
        train=TrainConfig(
            n_epochs=n_epochs,
            lr=lr,
            weight_lambda=weight_lambda,
            rate_lambda=rate_lambda,
            log_int=log_int,
            save_int=save_int,
            seed=train_seed,
            optimizer=optimizer,
            beta=beta,
        ),
        meanfield=MeanFieldConfig(n_points=P, beta=beta),
        device="cpu",
        dtype="float64",
    )


def with_solver(cfg: Config, **solver_fields) -> Config:
    """`cfg` with these `SolverConfig` fields set (validated; unknown names raise)."""
    solver = SolverConfig(**{**cfg.meanfield.solver.model_dump(), **solver_fields})
    return cfg.model_copy(update={"meanfield": cfg.meanfield.model_copy(update={"solver": solver})})


@dataclass(frozen=True)
class ExemplarDirs:
    train: Path  # holds data/, train/ and config.json
    solve: Path  # inside `train`; holds meanfield/


def exemplar_dirs(cache_root: Path, cfg: Config) -> ExemplarDirs:
    train = Path(cache_root) / f"exemplar-{cfg.section_hash('data', 'net', 'train')}"
    solve = train / f"solve-{cfg.section_hash('data', 'net', 'train', 'meanfield')}"
    return ExemplarDirs(train, solve)


def train_exemplar(
    cfg: Config, cache_root: Path, log: Log = print
) -> tuple[Dataset, MLP, History, Path]:
    """Dataset and trained net of `cfg` through the cache (training resumes from its
    checkpoints, a finished run is loaded); also the cache directory used."""
    directory = exemplar_dirs(cache_root, cfg).train
    save_config(cfg, cache_paths(directory).config)
    dataset = get_dataset(directory, cfg)
    net, history = train_from_config(cfg, dataset, directory, log=log)
    return dataset, net, history, directory


def example_outputs_of(net: MLP, dataset: Dataset, P: int) -> ExampleOutputs:
    """The net's outputs on a few fixed training points."""
    return example_outputs(net, dataset, pick_examples(P, N_EXAMPLE_OUTPUTS, EXAMPLE_SEED))


def live_figures(snapshot: OuterSnapshot) -> list[Figure]:
    """The three live figures of one snapshot: the kernels Φ_l and Φ̂_l, solver health and
    sampler health."""
    step = snapshot.outer
    return [
        fig_layer_kernels({"Φ": snapshot.Phi, "Φ̂": snapshot.Phi_hat}, title=f"outer step {step}"),
        fig_solver_health(snapshot.history),
        fig_sampler_health(snapshot.history),
    ]


def live_plotter(every: int, show: Show) -> OnOuter:
    """An `on_outer` callback that calls `show(figures)` after every `every`-th outer step
    (steps 0, every, 2 * every, ...). `show` owns the drawing surface and closing the figures:
    in a notebook it clears the cell output and displays them."""
    if every < 1:
        raise ValueError(f"every must be at least 1, got {every}")

    def refresh(snapshot: OuterSnapshot) -> None:
        if snapshot.outer % every == 0:
            show(live_figures(snapshot))

    return refresh


def solve_exemplar(
    cfg: Config,
    dataset: Dataset,
    net: MLP,
    cache_root: Path,
    plot_every: int,
    show: Show,
    log: Log = print,
) -> SolveResult:
    """Solve the mean-field problem of `cfg` for the trained net (through the cache), showing
    the live figures every `plot_every` outer steps and once more at the end."""
    result = run_meanfield(
        cfg,
        dataset,
        net,
        exemplar_dirs(cache_root, cfg).solve,
        log=log,
        on_outer=live_plotter(plot_every, show),
    )
    show(final_figures(result))
    return result


def final_figures(result: SolveResult) -> list[Figure]:
    """The live figures of the finished solve (also what a loaded, already finished solve shows,
    since it runs no steps and so never calls the callback)."""
    step = int(result.history["outer_step"][-1])
    return live_figures(OuterSnapshot(step, result.history, result.Phi, result.Phi_hat))


def kernel_comparison(result: SolveResult, net: MLP, dataset: Dataset, cfg: Config) -> Figure:
    """Solved Φ_l next to the trained net's empirical kernels, layer by layer."""
    return fig_kernels(result.Phi, net_kernels(net, dataset, cfg))


def empirical_kernel_figure(net: MLP, dataset: Dataset, cfg: Config) -> Figure:
    """The trained net's empirical kernels, one panel per layer."""
    return fig_layer_kernels({"Φ (net)": net_kernels(net, dataset, cfg)}, title="trained net")


def cache_root_from_env() -> Path:
    """The directory named by the environment variable CACHE_ROOT_ENV."""
    root = os.environ.get(CACHE_ROOT_ENV)
    if not root:
        raise RuntimeError(
            f"set CACHE_ROOT in the notebook or the environment variable {CACHE_ROOT_ENV}"
        )
    return Path(root)


def solve_summary(history: History, sampler: str) -> str:
    """One markdown paragraph on a finished solve: its length and final train MSE (theory and,
    if the solve had a net, the net's), the final kernel alignment with the net per layer, and
    the smallest sampler ESS fraction seen over the run."""
    n_steps = len(history["outer_step"])
    mse = f"theory {history['train_mse_theory'][-1]:.3g}"
    net_mse = history.get("train_mse_net", [None])[-1]
    if net_mse is not None:
        mse += f", trained net {net_mse:.3g}"
    text = f"{n_steps} outer steps with the {sampler} sampler. Final train MSE: {mse}."
    if "align_net" in history:
        align = ", ".join(f"{a:.3f}" for a in history["align_net"][-1])
        text += f" Kernel alignment with the net per layer: {align}."
    ess = np.min(np.asarray(history["hmc_ess_frac"], dtype=float))
    return text + f" Smallest sampler ESS fraction over the run: {ess:.3g}."


def _ipython_clear(wait: bool) -> None:
    from IPython.display import clear_output

    clear_output(wait=wait)


def _ipython_display(figure: Figure) -> None:
    from IPython.display import display

    display(figure)


class LiveView:
    """The notebook cell output of a running solve: each `show` replaces the previous frame.

    `clear_output(wait=True)` empties the whole cell output, log lines included, so the view
    keeps the last `tail` log lines itself and prints them under every frame. The clearing,
    displaying and printing are injected (defaults: IPython's and `print`) so the logic is
    testable without a kernel."""

    def __init__(
        self,
        tail: int = 8,
        clear: Callable[[bool], None] = _ipython_clear,
        display: Callable[[Figure], None] = _ipython_display,
        out: Callable[[str], None] = print,
    ) -> None:
        if tail < 1:
            raise ValueError(f"tail must be at least 1, got {tail}")
        self.tail, self._clear, self._display, self._out = tail, clear, display, out
        self.lines: list[str] = []

    def log(self, line: str) -> None:
        self.lines = [*self.lines, line][-self.tail :]

    def show(self, figures: Sequence[Figure]) -> None:
        self._clear(True)
        for figure in figures:
            self._display(figure)
            plt.close(figure)
        for line in self.lines:
            self._out(line)
