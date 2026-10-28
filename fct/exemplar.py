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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from fct.analyse import ExampleOutputs, example_outputs, pick_examples
from fct.config import Config, SolverConfig
from fct.data import Dataset
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
from fct.spatial_view import TrainRun, make_config, train_in

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
) -> Config:
    """The linear-teacher config of the notebook's settings, on the CPU in float64
    (`fct.spatial_view.make_config` without a spatial prior). All P training points enter the
    solve. The mean-field constants are left to follow the network (`fct.meanfield.
    network_constants`), so under SGLD the theory describes the distribution the net was
    sampled from (under Adam the solver's beta is 1: there is no Gibbs law). The solver settings are `with_solver`'s."""
    cfg = make_config(
        P=P, m=m, k=k, N=N, L=L, phi=phi, g=g, data_seed=data_seed, net_seed=net_seed,
        weight_lambda=weight_lambda, rate_lambda=rate_lambda, optimizer=optimizer, beta=beta,
        lr=lr, n_epochs=n_epochs, train_seed=train_seed, spatial=None, device="cpu",
    )  # fmt: skip
    meanfield = cfg.meanfield.model_copy(update={"n_points": P})
    return cfg.model_copy(update={"meanfield": meanfield})


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


def train_exemplar(cfg: Config, cache_root: Path, log: Log = print) -> TrainRun:
    """Dataset and trained net of `cfg` through the cache (training resumes from its
    checkpoints, a finished run is loaded), with the cache directory used."""
    return train_in(cfg, exemplar_dirs(cache_root, cfg).train, log)


def example_outputs_of(net: MLP, dataset: Dataset) -> ExampleOutputs:
    """The net's outputs on a few fixed training points."""
    indices = pick_examples(len(dataset.x), N_EXAMPLE_OUTPUTS, EXAMPLE_SEED)
    return example_outputs(net, dataset, indices)


def live_figures(snapshot: OuterSnapshot) -> list[Figure]:
    """The three live figures of one snapshot: the kernels Φ_l and Φ̂_l, solver health and
    sampler health (layers counted from 1 in all three)."""
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
    on_failure: Callable[[], None] = lambda: None,
) -> SolveResult:
    """Solve the mean-field problem of `cfg` for the trained net (through the cache), showing
    the live figures every `plot_every` outer steps and once more at the end.

    If the solve raises (an inadequate sampler, say), `on_failure` is called before the
    exception propagates: a live view uses it to print the log lines it was still holding."""
    try:
        result = run_meanfield(
            cfg,
            dataset,
            net,
            exemplar_dirs(cache_root, cfg).solve,
            log=log,
            on_outer=live_plotter(plot_every, show),
        )
    except BaseException:
        on_failure()
        raise
    show(final_figures(result))
    return result


def final_figures(result: SolveResult) -> list[Figure]:
    """The live figures of the finished solve (also what a loaded, already finished solve shows,
    since it runs no steps and so never calls the callback)."""
    step = int(result.history["outer_step"][-1])
    return live_figures(OuterSnapshot(step, result.history, result.Phi, result.Phi_hat))


def kernel_comparison(result: SolveResult, net: MLP, dataset: Dataset, cfg: Config) -> Figure:
    """Solved Φ_l next to the trained net's empirical kernels, layer by layer, each panel on
    its own colour scale (a shared scale would hide whichever kernel is smaller)."""
    return fig_kernels(result.Phi, net_kernels(net, dataset, cfg), shared_scale=False)


def empirical_kernel_figure(net: MLP, dataset: Dataset, cfg: Config) -> Figure:
    """The trained net's empirical kernels, one panel per layer."""
    return fig_layer_kernels({"Φ (net)": net_kernels(net, dataset, cfg)}, title="trained net")


def _per_layer(values, fmt: str = ".3g") -> str:
    return ", ".join(f"{v:{fmt}}" for v in values)


def _trend(first: np.ndarray, last: np.ndarray) -> str:
    """Whether the mean over layers fell, rose or stayed."""
    a, b = float(np.mean(first)), float(np.mean(last))
    return "fell" if b < a else "rose" if b > a else "was unchanged"


def _kernel_scales(name: str, kernels: torch.Tensor) -> str:
    """Largest eigenvalue and mean diagonal of each layer's kernel."""
    kernels = kernels.double()
    top = torch.linalg.eigvalsh(kernels)[:, -1].numpy()
    diag = torch.diagonal(kernels, dim1=-2, dim2=-1).mean(-1).numpy()
    return f"{name} kernels: largest eigenvalue {_per_layer(top)}, mean diagonal {_per_layer(diag)}"


def solve_summary(
    history: History,
    sampler: str,
    Phi: torch.Tensor | None = None,
    net_Phi: torch.Tensor | None = None,
) -> str:
    """One markdown paragraph on a finished solve, every number measured from it (layers
    counted from 1): the fixed-point residual of Φ̂ at the first and last outer step and
    whether it fell, the last layer's residual at its Stein target (if recorded), the smallest sampler ESS fraction and where it occurred, the largest
    eigenvalue and mean diagonal of the solved kernels `Phi` (and of the net's `net_Phi`, if
    given), the final train MSE (theory and, if the solve had a net, the net's) and the final
    kernel alignment with the net."""
    first = np.asarray(history["phihat_fp_resid_stein"][0], dtype=float)
    last = np.asarray(history["phihat_fp_resid_stein"][-1], dtype=float)
    parts = [
        f"{len(history['outer_step'])} outer steps with the {sampler} sampler.",
        (
            f"Fixed-point residual of Φ̂ per layer: {_per_layer(first)} at the first step, "
            f"{_per_layer(last)} at the last, so it {_trend(first, last)}."
        ),
    ]
    picard = np.asarray(history.get("picard_resid", []), dtype=float)
    if picard.size and np.isfinite(picard).all():
        parts.append(
            "Residual of the last layer's kernel at its Stein target (its convergence measure): "
            f"{picard[0]:.3g} at the first step, {picard[-1]:.3g} at the last, largest over the "
            f"last three {picard[-3:].max():.3g}."
        )
    ess = np.asarray(history["hmc_ess_frac"], dtype=float)
    step, layer = np.unravel_index(np.nanargmin(ess), ess.shape)
    parts.append(
        f"Smallest sampler ESS fraction {ess[step, layer]:.3g} (outer step "
        f"{history['outer_step'][step]}, layer "
        f"{layer + 1}); at the last step {_per_layer(ess[-1])}."
    )
    if Phi is not None:
        scales = [_kernel_scales("solved", Phi)]
        if net_Phi is not None:
            scales.append(_kernel_scales("net", net_Phi))
        parts.append("; ".join(scales) + ".")
    mse = f"theory {history['train_mse_theory'][-1]:.3g}"
    net_mse = history.get("train_mse_net", [None])[-1]
    if net_mse is not None:
        mse += f", trained net {net_mse:.3g}"
    parts.append(f"Final train MSE: {mse}.")
    if "align_net" in history:
        align = _per_layer(np.asarray(history["align_net"][-1]), ".3f")
        parts.append(f"Kernel alignment with the net per layer: {align}.")
    return " ".join(parts)


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

    def flush(self) -> None:
        """Print the held log lines: the last lines of a solve that failed, including one that
        failed before its first frame."""
        for line in self.lines:
            self._out(line)

    def show(self, figures: Sequence[Figure]) -> None:
        self._clear(True)
        for figure in figures:
            self._display(figure)
            plt.close(figure)
        self.flush()
