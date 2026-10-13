"""Task data: latent positions `y` and their high-dimensional embedding `x`.

`make_dataset` and the helpers are pure given a config, device and dtype; `get_dataset` is
the only function that touches disk.
"""

import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from fct.cache import cache_paths, load_or_compute
from fct.config import Config, DataConfig, resolve_device, resolve_dtype


@dataclass
class Dataset:
    x: Tensor  # (P, D_in)
    y: Tensor  # (P, d)
    Q: Tensor | None  # (D_in, d) orthonormal map for embedding="linear", else None
    period: float | None  # y.max() - y.min() at generation for "fourier", else None


def _canonical_lattice(d: int, min_count: int, device: torch.device) -> Tensor:
    """Integer vectors in Z^d, non-zero, first non-zero component positive (k ~ -k)."""
    K = 1
    while True:
        k = torch.tensor(
            list(itertools.product(range(-K, K + 1), repeat=d)), device=device, dtype=torch.long
        )
        k = k[(k != 0).any(dim=1)]
        first = k[torch.arange(k.shape[0], device=device), (k != 0).long().argmax(dim=1)]
        k = k[first > 0]
        if k.shape[0] >= min_count:
            return k
        K += 1


def fourier_embedding(y: Tensor, D: int, period: float | Tensor) -> tuple[Tensor, Tensor]:
    """Real Fourier features [cos(k1.y), sin(k1.y), cos(k2.y), ...] of shape (P, D).

    Uses the D//2 lattice wavevectors of lowest |k|^2; `period` (scalar or (d,)) sets the
    length over which each coordinate completes one cycle at |k| = 1. Also returns the
    (D//2, d) integer wavevectors.
    """
    if D % 2 != 0:
        raise ValueError("D must be even when keeping both sine and cosine.")
    P, d = y.shape
    M = D // 2
    period = torch.as_tensor(period, device=y.device, dtype=y.dtype)
    theta = 2 * math.pi * y / period

    k = _canonical_lattice(d, M, y.device)
    k = k[torch.argsort((k**2).sum(dim=1))[:M]]
    phase = theta @ k.to(y.dtype).T  # (P, M)
    features = torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1).reshape(P, D)
    return features, k


def _covariance(cfg: DataConfig, device: torch.device, dtype: torch.dtype) -> Tensor:
    if cfg.cov is None:
        return torch.eye(cfg.d, device=device, dtype=dtype)
    cov = torch.tensor(cfg.cov, device=device, dtype=dtype)
    if cov.shape != (cfg.d, cfg.d):
        raise ValueError(f"cov must be {cfg.d}x{cfg.d} to match d, got shape {tuple(cov.shape)}")
    return cov


def make_dataset(cfg: DataConfig, device: torch.device, dtype: torch.dtype) -> Dataset:
    """Gaussian latents y ~ N(0, var * cov) embedded linearly (random orthonormal Q) or by
    Fourier features.

    All randomness comes from one generator seeded with `cfg.seed` (created on `device`), so
    the result depends on the config alone and no global RNG state is touched.
    """
    cov = _covariance(cfg, device, dtype)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    chol = torch.linalg.cholesky(cfg.var * cov)
    z = torch.randn((cfg.P, cfg.d), device=device, dtype=dtype, generator=gen)
    y = z @ chol.T

    if cfg.embedding == "linear":
        U = torch.randn((cfg.D_in, cfg.d), device=device, dtype=dtype, generator=gen)
        Q = torch.linalg.qr(U, mode="reduced")[0]
        return Dataset(x=y @ Q.T, y=y, Q=Q, period=None)
    # The period spans the generated values so the lowest mode covers the whole range of y.
    period = (y.max() - y.min()).item()
    x, _ = fourier_embedding(y, D=cfg.D_in, period=period)
    return Dataset(x=x, y=y, Q=None, period=period)


def embed(y_new: Tensor, dataset: Dataset, cfg: DataConfig) -> Tensor:
    """Embed new latent points exactly as `dataset` was embedded."""
    if cfg.embedding == "linear":
        return y_new @ dataset.Q.T
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
        return {"x": ds.x, "y": ds.y, "Q": ds.Q, "period": ds.period}

    path = cache_paths(cache_dir).data / "dataset.pt"
    payload = load_or_compute(path, cfg.section_hash("data"), compute, device)
    return Dataset(x=payload["x"], y=payload["y"], Q=payload["Q"], period=payload["period"])
