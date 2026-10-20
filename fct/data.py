"""Task data: latent positions `y` and their high-dimensional embedding `x`.

`make_dataset` and the helpers are pure given a config, device and dtype; `get_dataset` is
the only function that touches disk. The generated numbers depend on the data config and
dtype only, not on the device.
"""

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from fct.cache import cache_paths, load_or_compute
from fct.config import Config, DataConfig, resolve_device, resolve_dtype
from fct.tasks import GENERATORS, task_input_dim, task_output_dim


@dataclass(frozen=True)
class Dataset:
    x: Tensor  # (P, D_in)
    y: Tensor  # (P, d)
    Q: Tensor | None  # (D_in, d) orthonormal map for embedding="linear", else None
    period: float | None  # y.max() - y.min() at generation for "fourier", else None
    # Ground-truth task variables, for scoring what the hidden units encode. int64 = categorical
    # labels 0..C-1, floating = continuous. `groups` maps a group name to the factor names it
    # contains (a factor that is its own group is a singleton group); `extras` holds tensors
    # needed to draw further examples of the task.
    factors: dict[str, Tensor] | None = None
    groups: dict[str, list[str]] | None = None
    extras: dict[str, Tensor] | None = None


def _canonical_lattice(d: int, K: int, device: torch.device) -> Tensor:
    """Non-zero integer vectors in [-K, K]^d whose first non-zero component is positive
    (k and -k give the same cos/sin pair), in lexicographic order."""
    axis = torch.arange(-K, K + 1, device=device)
    k = torch.cartesian_prod(*[axis] * d).reshape(-1, d)
    k = k[(k != 0).any(dim=1)]
    first = k[torch.arange(k.shape[0], device=device), (k != 0).long().argmax(dim=1)]
    return k[first > 0]


def _lowest_wavevectors(d: int, M: int, device: torch.device) -> Tensor:
    """The M canonical lattice vectors of lowest |k|^2, ties in lexicographic order.

    Any vector outside the box [-K, K]^d has |k|^2 >= (K+1)^2, so the box holds the true M
    lowest once its M-th smallest |k|^2 is below that bound; until then, grow the box.
    """
    K = 1
    while True:
        k = _canonical_lattice(d, K, device)
        k2 = (k**2).sum(dim=1)
        # stable: equal-|k|^2 vectors keep lexicographic order, so the kept set is the same
        # on every call (generation and embedding of test points must share one basis).
        order = torch.argsort(k2, stable=True)[:M]
        if order.numel() == M and k2[order[-1]] < (K + 1) ** 2:
            return k[order]
        K += 1


def fourier_embedding(y: Tensor, D: int, period: float | Tensor) -> tuple[Tensor, Tensor]:
    """Real Fourier features [cos(k1.y), sin(k1.y), cos(k2.y), ...] of shape (P, D).

    Uses the D//2 integer wavevectors of lowest |k|^2 (first non-zero component positive,
    ties broken in lexicographic order); `period` (scalar or (d,), finite and positive) sets
    the length over which each coordinate completes one cycle at |k| = 1. Also returns the
    (D//2, d) wavevectors.
    """
    if D % 2 != 0:
        raise ValueError("D must be even when keeping both sine and cosine.")
    period = torch.as_tensor(period, device=y.device, dtype=y.dtype)
    if not (torch.isfinite(period).all() and (period > 0).all()):
        raise ValueError(f"period must be finite and > 0, got {period.tolist()}")
    P, d = y.shape
    theta = 2 * math.pi * y / period

    k = _lowest_wavevectors(d, D // 2, y.device)
    phase = theta @ k.to(y.dtype).T  # (P, D//2)
    features = torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1).reshape(P, D)
    return features, k


def _covariance(cfg: DataConfig, device: torch.device, dtype: torch.dtype) -> Tensor:
    if cfg.cov is None:
        return torch.eye(cfg.d, device=device, dtype=dtype)
    cov = torch.tensor(cfg.cov, device=device, dtype=dtype)
    if cov.shape != (cfg.d, cfg.d):
        raise ValueError(f"cov must be {cfg.d}x{cfg.d} to match d, got shape {tuple(cov.shape)}")
    # cholesky reads only the lower triangle, so an asymmetric cov would be silently altered.
    if not torch.allclose(cov, cov.T):
        raise ValueError("cov must be symmetric")
    return cov


def _position_factors(y: Tensor) -> tuple[dict[str, Tensor], dict[str, list[str]]]:
    """Each coordinate of y is a continuous factor in its own singleton group.

    Columns are copied so that saving one factor does not write the whole of y again.
    """
    names = [f"y{i}" for i in range(y.shape[1])]
    return {n: y[:, i].contiguous() for i, n in enumerate(names)}, {n: [n] for n in names}


def task_dims(cfg: DataConfig) -> tuple[int, int]:
    """(D_in, D_out) of the network that learns the configured task."""
    if cfg.task == "position":
        return cfg.D_in, cfg.d
    return task_input_dim(cfg) or cfg.D_in, task_output_dim(cfg)


def _make_task_dataset(cfg: DataConfig, device: torch.device, dtype: torch.dtype) -> Dataset:
    fixed = task_input_dim(cfg)
    if fixed is not None and cfg.D_in != fixed:
        raise ValueError(
            f"task {cfg.task!r} fixes the input dimension at {fixed}; set data.D_in={fixed} "
            f"(got D_in={cfg.D_in})"
        )
    t = GENERATORS[cfg.task](cfg, device, dtype)
    return Dataset(
        x=t.x, y=t.y, Q=None, period=None, factors=t.factors, groups=t.groups, extras=t.extras
    )


def make_dataset(cfg: DataConfig, device: torch.device, dtype: torch.dtype) -> Dataset:
    """Generate the configured task; the non-position tasks live in `fct.tasks`.

    Position task: Gaussian latents y ~ N(0, var * cov) embedded linearly (random orthonormal
    Q) or by Fourier features.

    The random draws come from a CPU generator seeded with `cfg.seed` and are moved to
    `device` afterwards: CUDA and CPU generators give different streams, and the cache key
    does not include the device, so the data must not depend on it.
    """
    if cfg.task != "position":
        return _make_task_dataset(cfg, device, dtype)
    cov = _covariance(cfg, device, dtype)
    gen = torch.Generator().manual_seed(cfg.seed)
    chol = torch.linalg.cholesky(cfg.var * cov)
    z = torch.randn((cfg.P, cfg.d), dtype=dtype, generator=gen).to(device)
    y = z @ chol.T
    factors, groups = _position_factors(y)

    if cfg.embedding == "linear":
        U = torch.randn((cfg.D_in, cfg.d), dtype=dtype, generator=gen).to(device)
        Q = torch.linalg.qr(U, mode="reduced")[0]
        return Dataset(x=y @ Q.T, y=y, Q=Q, period=None, factors=factors, groups=groups)
    # The period spans the generated values so the lowest mode covers the whole range of y.
    period = (y.max() - y.min()).item()
    x, _ = fourier_embedding(y, D=cfg.D_in, period=period)
    return Dataset(x=x, y=y, Q=None, period=period, factors=factors, groups=groups)


def embed(y_new: Tensor, dataset: Dataset, cfg: DataConfig) -> Tensor:
    """Embed new latent points exactly as `dataset` was embedded."""
    if dataset.x.shape[1] != cfg.D_in:
        raise ValueError(f"dataset has D_in={dataset.x.shape[1]} but config has D_in={cfg.D_in}")
    if cfg.embedding == "linear":
        if dataset.Q is None:
            raise ValueError('config embedding is "linear" but the dataset has no Q')
        return y_new @ dataset.Q.T
    if dataset.period is None:
        raise ValueError('config embedding is "fourier" but the dataset has no period')
    x, _ = fourier_embedding(y_new, D=cfg.D_in, period=dataset.period)
    return x


def grid_targets(
    half_width: float, n_grid: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """(n_grid**2, 2) points covering [-h, h]^2; reshape(n, n) of a column matches meshgrid 'ij'."""
    axis = torch.linspace(-half_width, half_width, n_grid, device=device, dtype=dtype)
    mesh = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([m.flatten() for m in mesh], dim=1)


def get_dataset(cache_dir: Path, cfg: Config) -> Dataset:
    """Load CACHE_DIR/data/dataset.pt, generating it on first use.

    Raises StaleCacheError if the file was written under a different data config.
    """
    device = resolve_device(cfg)
    dtype = resolve_dtype(cfg.dtype)

    def compute() -> dict:
        ds = make_dataset(cfg.data, device, dtype)
        return {f.name: getattr(ds, f.name) for f in dataclasses.fields(ds)}

    path = cache_paths(cache_dir).data / "dataset.pt"
    payload = load_or_compute(path, cfg.section_hash("data"), compute, device)
    return Dataset(**{f.name: payload[f.name] for f in dataclasses.fields(Dataset)})
