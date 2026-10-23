"""Wiring for the spatial-prior notebook: the config it trains, the resumable training run, and the
pure layout helpers that arrange neurons and weights by their position in space.

Layers follow `fct.spatial`: weight l is (post, pre), so the first-layer weight `MLP.U` is (hidden
unit, input unit) and its rows live at the hidden positions, its columns at the input positions.
Only positions on a ring (d = 1) can be ordered; a two-dimensional layout needs a lattice.
"""

import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
from torch import Tensor

from fct.cache import cache_paths
from fct.config import Config, DataConfig, NetConfig, TrainConfig, save_config
from fct.data import Dataset, get_dataset
from fct.model import MLP
from fct.spatial import (
    EmbeddingSpec,
    KernelSpec,
    LayerSpace,
    PopulationSpec,
    SpatialConfig,
    build_spaces,
)
from fct.train import History, train_from_config

CACHE_ROOT_ENV = "FCT_CACHE_ROOT"
# Decimals kept when deciding that two lattice coordinates are the same point.
LATTICE_DECIMALS = 9
# Log lines and checkpoints per run, so any n_epochs logs ~50 times and can resume from a quarter.
LOG_LINES, CHECKPOINTS = 50, 4


# --------------------------------------------------------------------------- configuration


def population_spec(
    kind: Literal["grid", "random"],
    length_scale: float,
    amplitude: float = 1.0,
    shape: list[int] | None = None,
    seed: int = 0,
) -> PopulationSpec:
    """One population's embedding and Gaussian kernel, the arguments the notebook edits."""
    return PopulationSpec(
        embedding=EmbeddingSpec(kind=kind, shape=shape, seed=seed),
        kernel=KernelSpec(length_scale=length_scale, amplitude=amplitude),
    )


def make_config(
    *,
    P: int,
    m: int,
    k: int,
    N: int,
    L: int,
    phi: Literal["relu", "tanh", "sigmoid"],
    g: float,
    data_seed: int,
    net_seed: int,
    weight_lambda: float,
    rate_lambda: float,
    optimizer: Literal["adam", "sgld"],
    beta: float,
    lr: float,
    n_epochs: int,
    train_seed: int,
    spatial: SpatialConfig | None,
    device: Literal["auto", "cpu", "cuda"],
) -> Config:
    """The linear-teacher run of the notebook: m features, k latents, P samples, an L-layer net
    of N units, trained by `optimizer` under the spatial prior `spatial`. Every field is
    validated by the config models, so an impossible combination fails here."""
    return Config(
        data=DataConfig(task="linear_teacher", P=P, D_in=m, n_latents=k, seed=data_seed),
        net=NetConfig(N=N, L=L, g=g, phi=phi, seed=net_seed),
        train=TrainConfig(
            n_epochs=n_epochs,
            lr=lr,
            weight_lambda=weight_lambda,
            rate_lambda=rate_lambda,
            log_int=max(1, n_epochs // LOG_LINES),
            save_int=max(1, n_epochs // CHECKPOINTS),
            seed=train_seed,
            optimizer=optimizer,
            beta=beta,
        ),
        spatial=spatial,
        device=device,
    )


def layer_spaces(cfg: Config) -> list[LayerSpace]:
    """The L + 1 layers' spatial factors for the network `cfg` describes."""
    if cfg.spatial is None:
        raise ValueError("the config has no spatial section")
    if cfg.data.task != "linear_teacher":
        raise ValueError(f"only the linear teacher is supported here, got {cfg.data.task!r}")
    d_in = cfg.data.D_in  # linear teacher: the output has as many features as the input
    return build_spaces(cfg.spatial, m=d_in, N=cfg.net.N, L=cfg.net.L, n_out=d_in)


def cache_root_from_env() -> Path:
    """The directory holding the notebook's runs, named by $FCT_CACHE_ROOT; RuntimeError if unset."""
    root = os.environ.get(CACHE_ROOT_ENV)
    if not root:
        raise RuntimeError(f"set ${CACHE_ROOT_ENV} to the directory that holds the cached runs")
    return Path(root)


def cache_dir_for(cfg: Config, root: Path) -> Path:
    """A run's cache directory under `root`, named by the hash of its data, net, train and
    spatial settings: changing any of them gives a new directory, an unchanged setup finds its
    earlier run."""
    return Path(root) / f"spatial_{cfg.section_hash('data', 'net', 'train')}"


# --------------------------------------------------------------------------- the run (I/O)


class TrainRun(NamedTuple):
    cache_dir: Path
    dataset: Dataset
    net: MLP
    history: History


def train_run(cfg: Config, root: Path, log: Callable[[str], None] = print) -> TrainRun:
    """Train the run of `cfg` through the normal cache (dataset, checkpoints, final.pt in
    `cache_dir_for(cfg, root)`, plus its config.json so `python -m fct plot` reads it too).
    A finished run is only loaded; an interrupted one resumes from its latest checkpoint."""
    cache_dir = cache_dir_for(cfg, root)
    save_config(cfg, cache_paths(cache_dir).config)
    dataset = get_dataset(cache_dir, cfg)
    net, history = train_from_config(cfg, dataset, cache_dir, log=log)
    return TrainRun(cache_dir, dataset, net, history)


# --------------------------------------------------------------------------- layout (pure)


def _positions(positions: Tensor) -> np.ndarray:
    if positions.ndim != 2:
        raise ValueError(f"positions must be (n, d), got shape {tuple(positions.shape)}")
    return positions.detach().cpu().numpy()


def position_order(positions: Tensor) -> np.ndarray:
    """Unit indices sorted by their coordinate around the ring (d = 1); ties keep unit order."""
    xy = _positions(positions)
    if xy.shape[1] != 1:
        raise ValueError(f"ordering by position needs d = 1, got d = {xy.shape[1]}")
    return np.argsort(xy[:, 0], kind="stable")


class OrderedWeights(NamedTuple):
    W: Tensor  # rows sorted by post position, columns by pre position
    post_order: np.ndarray  # post_order[r] is the unit shown in row r
    pre_order: np.ndarray  # pre_order[c] is the unit shown in column c


def order_weights(W: Tensor, pos_post: Tensor, pos_pre: Tensor) -> OrderedWeights:
    """`W` (post, pre) with rows and columns sorted around their rings."""
    if W.shape[0] != len(pos_post) or W.shape[1] != len(pos_pre):
        raise ValueError(
            f"W is {tuple(W.shape)} but there are {len(pos_post)} post rows and "
            f"{len(pos_pre)} pre columns of positions"
        )
    post_order, pre_order = position_order(pos_post), position_order(pos_pre)
    return OrderedWeights(W[post_order][:, pre_order], post_order, pre_order)


class PanelGrid(NamedTuple):
    n_rows: int
    n_cols: int
    cells: list[tuple[int, int]]  # cells[u] is the (row, column) of unit u


def _ring_grid(n: int, order: np.ndarray, n_cols: int) -> PanelGrid:
    """Units in ring order along wrapped rows of `n_cols`."""
    n_cols = min(n_cols, n)
    cells: list[tuple[int, int]] = [(0, 0)] * n
    for rank, unit in enumerate(order):
        cells[unit] = (rank // n_cols, rank % n_cols)
    return PanelGrid(math.ceil(n / n_cols), n_cols, cells)


def _lattice_grid(xy: np.ndarray) -> PanelGrid:
    """Units on a full regular lattice: column by the x coordinate, row by y with y increasing
    upwards, as in the plane plot of the positions."""
    rounded = xy.round(LATTICE_DECIMALS)
    xs, cols = np.unique(rounded[:, 0], return_inverse=True)
    ys, rows = np.unique(rounded[:, 1], return_inverse=True)
    distinct = len({tuple(p) for p in rounded.tolist()})
    if len(xs) * len(ys) != len(xy) or distinct != len(xy):
        raise ValueError(
            f"{len(xy)} positions do not form a lattice ({len(xs)} x values, {len(ys)} y "
            "values); use a grid embedding to lay the panels out in two dimensions"
        )
    cells = [(len(ys) - 1 - int(r), int(c)) for r, c in zip(rows, cols, strict=True)]
    return PanelGrid(len(ys), len(xs), cells)


def panel_grid(positions: Tensor, n_cols: int) -> PanelGrid:
    """Where each unit's panel goes: along wrapped rows of `n_cols` in ring order for d = 1,
    at its lattice coordinates for d = 2 (`n_cols` is ignored). ValueError for d > 2 or for
    scattered positions in the plane."""
    xy = _positions(positions)
    if xy.shape[1] == 1:
        return _ring_grid(len(xy), position_order(positions), n_cols)
    if xy.shape[1] == 2:
        return _lattice_grid(xy)
    raise ValueError(f"panels can be laid out for d <= 2, got d = {xy.shape[1]}")


# --------------------------------------------------------------------------- empirical kernels


def empirical_kernels(W: Tensor) -> tuple[Tensor, Tensor]:
    """The kernels `W` (post, pre) shows across its rows and across its columns:
    W W^T / n_pre between post units and W^T W / n_post between pre units. For one weight matrix
    they are noisy estimates (a rank-min(n_post, n_pre) sample covariance), compared with the
    prior's K_post and K_pre."""
    n_post, n_pre = W.shape
    return W @ W.T / n_pre, W.T @ W / n_post


class KernelComparison(NamedTuple):
    """Theoretical and empirical kernels of one layer with units sorted around their rings."""

    post_theory: Tensor
    post_empirical: Tensor
    pre_theory: Tensor
    pre_empirical: Tensor


def ordered_kernel_comparison(space: LayerSpace, W: Tensor) -> KernelComparison:
    """K_post, K_pre of `space` next to the empirical kernels of `W` (post, pre), every
    matrix indexed by the same ring order of its population."""
    ordered = order_weights(
        W.detach().to(space.post.K), space.post.embedding.positions, space.pre.embedding.positions
    )
    po, qo = ordered.post_order, ordered.pre_order
    post_hat, pre_hat = empirical_kernels(ordered.W)
    return KernelComparison(space.post.K[po][:, po], post_hat, space.pre.K[qo][:, qo], pre_hat)
