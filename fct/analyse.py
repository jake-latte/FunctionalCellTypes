"""Analysis of a trained network: test-grid predictions, per-layer PCA, neuron maps.

Everything except the `get_*` wrappers is pure (no file access, no plotting). The wrappers
cache the expensive results under CACHE_DIR/analysis/ so figures can be redrawn in seconds.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import Tensor

from fct.cache import cache_paths, load_or_compute
from fct.config import Config, DataConfig, resolve_device
from fct.data import Dataset, embed, grid_targets
from fct.model import MLP


@dataclass
class GridPredictions:
    half_width: float
    n_grid: int
    y_grid: Tensor  # (p^2, 2) test targets, 'ij' meshgrid order
    h_grid: Tensor  # (p^2, L, N) pre-activations on the grid
    z_grid: Tensor  # (p^2, d) network outputs on the grid
    test_error: Tensor  # (p, p) per-point MSE over output dimensions
    train_error: Tensor  # (P,) per-training-point MSE over output dimensions


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
    )


def layer_pca(h_grid: Tensor, phi: Callable[[Tensor], Tensor], n_components: int) -> LayerPCA:
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
        pca = PCA(n_components=n_components).fit(activity)
        ratios.append(pca.explained_variance_ratio_)
        comps.append(pca.components_)
    kwargs = {"device": h_grid.device, "dtype": torch.float64}
    explained_variance = torch.as_tensor(np.stack(ratios), **kwargs)
    components = torch.as_tensor(np.stack(comps), **kwargs)
    similarity = torch.einsum("lnp,LNp->lnLN", components, components) / n_points
    return LayerPCA(explained_variance, components, similarity)


def neuron_maps(
    h_grid: Tensor, phi: Callable[[Tensor], Tensor], layer: int, n_grid: int
) -> Tensor:
    """(N, p, p) activity maps of every neuron in `layer`."""
    if h_grid.shape[0] != n_grid**2:
        raise ValueError(f"h_grid has {h_grid.shape[0]} points but n_grid={n_grid} needs {n_grid**2}")
    return phi(h_grid[:, layer]).T.reshape(-1, n_grid, n_grid)


def kernel_scales(Phi: Tensor) -> Tensor:
    """tr(Phi_l) / P for a stack of kernels (L, P, P)."""
    return Phi.diagonal(dim1=-2, dim2=-1).mean(dim=-1)


def get_grid_predictions(cache_dir: Path, cfg: Config, net: MLP, dataset: Dataset) -> GridPredictions:
    """Load CACHE_DIR/analysis/grid.pt, computing it on first use.

    Raises StaleCacheError if it was written under a different data/net/train/analysis config.
    """
    n_grid = cfg.analysis.n_grid

    def compute() -> dict:
        g = grid_predictions(net, dataset, cfg.data, n_grid)
        return {**vars(g)}

    path = cache_paths(cache_dir).analysis / "grid.pt"
    payload = load_or_compute(path, _analysis_hash(cfg), compute, resolve_device(cfg))
    return GridPredictions(**{k: v for k, v in payload.items() if k != "config_hash"})


def get_layer_pca(
    cache_dir: Path, cfg: Config, grid: GridPredictions, phi: Callable[[Tensor], Tensor]
) -> LayerPCA:
    """Load CACHE_DIR/analysis/pca.pt, fitting it on first use (same staleness rule as above)."""

    def compute() -> dict:
        return {**vars(layer_pca(grid.h_grid, phi, cfg.analysis.n_components))}

    path = cache_paths(cache_dir).analysis / "pca.pt"
    payload = load_or_compute(path, _analysis_hash(cfg), compute, resolve_device(cfg))
    return LayerPCA(**{k: v for k, v in payload.items() if k != "config_hash"})


def _analysis_hash(cfg: Config) -> str:
    return cfg.section_hash("data", "net", "train", "analysis")
