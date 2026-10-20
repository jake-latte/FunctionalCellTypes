"""Analysis of a trained network: test-grid predictions, per-layer PCA, neuron maps.

Everything except the `get_*` wrappers is pure (no file access, no plotting). The wrappers
cache the expensive results under CACHE_DIR/analysis/ so figures can be redrawn in seconds.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import Tensor

from fct.activations import Activation, get_phi
from fct.cache import StaleCacheError, cache_paths, load_or_compute
from fct.config import Config, DataConfig, resolve_device
from fct.data import TASKS, Dataset, embed, grid_targets
from fct.model import MLP
from fct.tasks import ACTION_NAMES, N_MACRO, N_MICRO


@dataclass
class GridPredictions:
    half_width: float
    n_grid: int
    y_grid: Tensor  # (p^2, 2) test targets, 'ij' meshgrid order
    h_grid: Tensor  # (p^2, L, N) pre-activations on the grid
    z_grid: Tensor  # (p^2, d) network outputs on the grid
    test_error: Tensor  # (p, p) per-point MSE over output dimensions
    train_error: Tensor  # (P,) per-training-point MSE over output dimensions
    net_weight_l2: float  # identifies the network the predictions came from


@dataclass
class LayerPCA:
    explained_variance: Tensor  # (L, k) explained-variance ratios
    components: Tensor  # (L, k, p^2) unit-norm spatial modes
    similarity: Tensor  # (L, k, L, k) component overlaps divided by p^2


def grid_predictions(
    net: MLP, dataset: Dataset, data_cfg: DataConfig, n_grid: int
) -> GridPredictions:
    """Evaluate `net` on an n_grid x n_grid square of latent positions and on the training set.

    The square is [-var/2, var/2]^2: `var` is a variance, but it is used as the test
    square's full width so the square matches the scale of the training latents.
    """
    if data_cfg.d != 2:
        raise ValueError(f"the test grid is 2-D, so the config needs d == 2, got d={data_cfg.d}")
    if net.D_out != data_cfg.d:
        raise ValueError(
            f"net.D_out ({net.D_out}) must equal the latent dimension d ({data_cfg.d})"
        )
    half_width = data_cfg.var / 2
    y_grid = grid_targets(half_width, n_grid, net.device, net.dtype)
    x_grid = embed(y_grid, dataset, data_cfg)
    with torch.no_grad():  # analysis only; no autograd graph is kept in the results
        _, z_train = net(dataset.x)
        h_grid, z_grid = net(x_grid)
    return GridPredictions(
        half_width=half_width,
        n_grid=n_grid,
        y_grid=y_grid,
        h_grid=h_grid,
        z_grid=z_grid,
        test_error=(y_grid - z_grid).square().mean(dim=1).reshape(n_grid, n_grid),
        train_error=(dataset.y - z_train).square().mean(dim=1),
        net_weight_l2=float(net.weight_l2().detach()),
    )


def layer_pca(h_grid: Tensor, phi: Activation, n_components: int) -> LayerPCA:
    """PCA per layer of the N neuron maps phi(h) (each a vector over the p^2 grid points).

    Neurons are the samples and grid points the features, so components are spatial
    modes. `similarity[l, i, L, j] = <c_li, c_Lj> / p^2`; its diagonal `[l, i, l, i]` is
    1/p^2 because sklearn components have unit norm.
    """
    n_points, n_layers, n_neurons = h_grid.shape
    if not 1 <= n_components <= min(n_neurons, n_points):
        raise ValueError(
            f"n_components must be in [1, min(N={n_neurons}, p^2={n_points})], got {n_components}"
        )
    ratios, comps = [], []
    for layer in range(n_layers):
        activity = phi(h_grid[:, layer].T).detach().cpu().numpy()  # (N, p^2)
        # Exact solver: the default may pick a randomized one, and a cached result must be
        # determined by the config alone.
        pca = PCA(n_components=n_components, svd_solver="full").fit(activity)
        ratios.append(pca.explained_variance_ratio_)
        comps.append(pca.components_)
    kwargs = {"device": h_grid.device, "dtype": h_grid.dtype}
    explained_variance = torch.as_tensor(np.stack(ratios), **kwargs)
    components = torch.as_tensor(np.stack(comps), **kwargs)
    similarity = torch.einsum("lnp,LNp->lnLN", components, components) / n_points
    return LayerPCA(explained_variance, components, similarity)


def neuron_maps(h_grid: Tensor, phi: Activation, layer: int, n_grid: int) -> Tensor:
    """(N, p, p) activity maps of every neuron in `layer`."""
    if not 0 <= layer < h_grid.shape[1]:
        raise ValueError(f"layer must be in [0, {h_grid.shape[1]}), got {layer}")
    if h_grid.shape[0] != n_grid**2:
        raise ValueError(
            f"h_grid has {h_grid.shape[0]} points but n_grid={n_grid} needs {n_grid**2}"
        )
    return phi(h_grid[:, layer]).T.reshape(-1, n_grid, n_grid)


def kernel_scales(Phi: Tensor) -> Tensor:
    """tr(Phi_l) / P for a stack of kernels (L, P, P)."""
    return Phi.diagonal(dim1=-2, dim2=-1).mean(dim=-1)


def get_grid_predictions(
    cache_dir: Path, cfg: Config, net: MLP, dataset: Dataset
) -> GridPredictions:
    """Load CACHE_DIR/analysis/grid.pt, computing it on first use.

    Expects the trained net (from `load_trained`). The file is keyed on the config, so a
    hit is checked against the net's weight norm: StaleCacheError if the config sections
    data/net/train/analysis differ from the writer's, or if `net` is not the net that
    produced the file.
    """

    def compute() -> dict:
        return vars(grid_predictions(net, dataset, cfg.data, cfg.analysis.n_grid))

    path = cache_paths(cache_dir).analysis / "grid.pt"
    payload = load_or_compute(path, _analysis_hash(cfg), compute, resolve_device(cfg))
    if not math.isclose(payload["net_weight_l2"], float(net.weight_l2().detach()), rel_tol=1e-9):
        raise StaleCacheError(
            f"{path} was computed from a different network (weight norm "
            f"{payload['net_weight_l2']!r} vs {float(net.weight_l2().detach())!r}). Delete the file "
            "to recompute it from this net."
        )
    return GridPredictions(**_fields(payload))


def get_layer_pca(cache_dir: Path, cfg: Config, grid: GridPredictions, phi: Activation) -> LayerPCA:
    """Load CACHE_DIR/analysis/pca.pt, fitting it on first use.

    Expects `grid` from `get_grid_predictions` and `phi` equal to the configured activation
    (ValueError otherwise). StaleCacheError if the config sections data/net/train/analysis
    differ from those the file was written with.
    """
    if phi is not get_phi(cfg.net.phi):
        raise ValueError(f"phi must be the configured activation {cfg.net.phi!r}")

    def compute() -> dict:
        return vars(layer_pca(grid.h_grid, phi, cfg.analysis.n_components))

    path = cache_paths(cache_dir).analysis / "pca.pt"
    payload = load_or_compute(path, _analysis_hash(cfg), compute, resolve_device(cfg))
    return LayerPCA(**_fields(payload))


def _fields(payload: dict) -> dict:
    """The artefact payload without the cache's own bookkeeping key."""
    return {k: v for k, v in payload.items() if k != "config_hash"}


def _analysis_hash(cfg: Config) -> str:
    return cfg.section_hash("data", "net", "train", "analysis")


# --------------------------------------------------------------------------- example outputs


@dataclass
class ExampleOutputs:
    """A few dataset rows with the net's predictions for them, all on the CPU."""

    indices: Tensor  # (n,) rows of the dataset
    x: Tensor  # (n, D_in)
    y_true: Tensor  # (n, D_out)
    y_pred: Tensor  # (n, D_out)


def pick_examples(P: int, n: int, seed: int) -> Tensor:
    """(n,) distinct dataset rows chosen uniformly; a CPU generator makes the choice
    independent of the device."""
    if n > P:
        raise ValueError(f"cannot pick {n} examples from only {P} samples")
    return torch.randperm(P, generator=torch.Generator().manual_seed(seed))[:n]


def example_outputs(net: MLP, dataset: Dataset, indices: Tensor | list[int]) -> ExampleOutputs:
    """The net's outputs and the dataset's targets on the rows `indices` (integers)."""
    idx = torch.as_tensor(indices).cpu()
    if idx.numel() == 0:  # an empty list has no dtype of its own
        idx = idx.long()
    if idx.is_floating_point() or idx.is_complex() or idx.dtype == torch.bool:
        raise ValueError(f"indices must be integers, got dtype {idx.dtype}")
    idx = idx.long()
    n_rows = len(dataset.x)
    if idx.ndim != 1 or (idx.numel() > 0 and (idx.min() < 0 or idx.max() >= n_rows)):
        raise ValueError(f"indices must be a 1-D list of rows in [0, {n_rows}), got {idx}")
    rows = idx.to(dataset.x.device)
    with torch.no_grad():
        _, z = net(dataset.x[rows])
    return ExampleOutputs(
        indices=idx,
        x=dataset.x[rows].cpu(),
        y_true=dataset.y[rows].cpu(),
        y_pred=z.cpu(),
    )


def decode_object_action(x: Tensor, grid_size: int) -> tuple[Tensor, Tensor]:
    """Invert the object_action input: (agent_rc (n, 2) int, objects (n, G, G) bool)."""
    cells = grid_size**2
    if x.ndim != 2 or x.shape[1] != 2 * cells:
        raise ValueError(f"x must have shape (n, 2*G*G = {2 * cells}), got {tuple(x.shape)}")
    agent = x[:, :cells].argmax(dim=1)
    agent_rc = torch.stack([agent // grid_size, agent % grid_size], dim=1)
    return agent_rc, (x[:, cells:] > 0.5).reshape(-1, grid_size, grid_size)


def decode_what_where(x: Tensor) -> Tensor:
    """Invert the what_where input: the (n, 9, 9) micro-grid images (row-major pixels)."""
    if x.ndim != 2 or x.shape[1] != N_MICRO**2:
        raise ValueError(f"x must have shape (n, {N_MICRO**2} = 81), got {tuple(x.shape)}")
    return x.reshape(-1, N_MICRO, N_MICRO)


def split_what_where(y: Tensor, n_motifs: int) -> tuple[Tensor, Tensor]:
    """The `where` (first 9) and `what` (last n_motifs) columns of what_where outputs."""
    n_where = N_MACRO**2
    if y.ndim != 2 or y.shape[1] != n_where + n_motifs:
        raise ValueError(
            f"what_where outputs need {n_where} + n_motifs = {n_where + n_motifs} columns, "
            f"got shape {tuple(y.shape)}"
        )
    return y[:, :n_where], y[:, n_where:]


def _check_pair(y_true: Tensor, y_pred: Tensor) -> None:
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        raise ValueError(
            f"y_true and y_pred need one 2-D shape, got {y_true.shape}, {y_pred.shape}"
        )


def r_squared(y_true: Tensor, y_pred: Tensor) -> float:
    """Pooled R^2: 1 - SS_res / SS_tot over all entries, SS_tot about the grand mean.

    Outputs with different means or scales are not weighted equally; see `r_squared_per_output`.
    """
    _check_pair(y_true, y_pred)
    ss_tot = (y_true - y_true.mean()).square().sum()
    if ss_tot == 0:
        raise ValueError("R^2 is undefined: the targets have zero variance")
    return float(1 - (y_true - y_pred).square().sum() / ss_tot)


def r_squared_per_output(y_true: Tensor, y_pred: Tensor) -> float:
    """R^2 of every output column about its own mean, averaged uniformly (sklearn's default)."""
    _check_pair(y_true, y_pred)
    ss_tot = (y_true - y_true.mean(dim=0)).square().sum(dim=0)
    if (ss_tot == 0).any():
        raise ValueError("R^2 is undefined: an output has zero variance across the samples")
    ss_res = (y_true - y_pred).square().sum(dim=0)
    return float((1 - ss_res / ss_tot).mean())


def _regression_report(y_true: Tensor, y_pred: Tensor, cfg: DataConfig) -> dict[str, float]:
    return {
        "r2_pooled": r_squared(y_true, y_pred),
        "r2_per_output": r_squared_per_output(y_true, y_pred),
    }


def _object_action_report(y_true: Tensor, y_pred: Tensor, cfg: DataConfig) -> dict[str, float]:
    _check_outputs(y_true, len(ACTION_NAMES))
    target = y_true > 0.5
    exact = ((y_pred > 0.5) == target).all(dim=1)
    # A tied maximum is not a decision (an all-equal prediction would otherwise always pick
    # 'stay', the first column), so only a unique maximum can be optimal.
    top = y_pred.argmax(dim=1, keepdim=True)
    unique_max = (y_pred == y_pred.gather(1, top)).sum(dim=1) == 1
    optimal = target.gather(1, top).squeeze(1) & unique_max
    only_stay = target[:, 0] & ~target[:, 1:].any(dim=1)
    return {
        "exact_match": float(exact.float().mean()),
        "argmax_optimal": float(optimal.float().mean()),
        "stay_rate": float(only_stay.float().mean()),
    }


def _top1_accuracy(y_true: Tensor, y_pred: Tensor) -> float:
    return float((y_pred.argmax(dim=1) == y_true.argmax(dim=1)).float().mean())


def _what_where_report(y_true: Tensor, y_pred: Tensor, cfg: DataConfig) -> dict[str, float]:
    where_t, what_t = split_what_where(y_true, cfg.n_motifs)
    where_p, what_p = split_what_where(y_pred, cfg.n_motifs)
    return {
        "where_accuracy": _top1_accuracy(where_t, where_p),
        "what_accuracy": _top1_accuracy(what_t, what_p),
    }


def _check_outputs(y: Tensor, expected: int) -> None:
    if y.shape[1] != expected:
        raise ValueError(f"expected {expected} outputs for this task, got {y.shape[1]}")


_REPORTS = {
    "linear_teacher": _regression_report,
    "position": _regression_report,
    "object_action": _object_action_report,
    "what_where": _what_where_report,
}


def prediction_report(
    task: str, y_true: Tensor, y_pred: Tensor, cfg: DataConfig
) -> dict[str, float]:
    """Task-appropriate accuracy summary of predictions `y_pred` against targets `y_true`.

    linear_teacher and position: {"r2_pooled", "r2_per_output"}.
    object_action: {"exact_match" (pred > 0.5 equals the target set), "argmax_optimal" (the
    largest prediction, if unique, is an optimal action), "stay_rate" (fraction of samples
    whose only optimal action is 'stay', the share a constant 'stay' output would get right)}.
    what_where: {"where_accuracy", "what_accuracy"} (top-1 over the first 9 / last n_motifs
    outputs).
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {sorted(TASKS)}")
    if task not in _REPORTS:
        raise ValueError(f"no prediction report for task {task!r}")
    _check_pair(y_true, y_pred)
    return _REPORTS[task](y_true, y_pred, cfg)
