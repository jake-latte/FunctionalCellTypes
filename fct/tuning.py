"""Tuning curves of a layer's neurons to each latent, and choosing which neurons to show.

Pure functions on plain inputs; plotting lives in `fct.draw.fig_tuning`.
"""

from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor

from fct.modularity import rank_bins

ORDERS = ("random", "mir_ascending", "mir_descending")


class TuningCurves(NamedTuple):
    mean: Tensor  # (N, K, n_bins) mean rate of each neuron in each bin of each latent
    sem: Tensor  # (N, K, n_bins) standard error of that mean; NaN with fewer than 2 samples
    centres: Tensor  # (K, n_bins) mean latent value in each bin; NaN for an empty bin
    count: Tensor  # (K, n_bins) samples per bin


def _check_inputs(rates: Tensor, factors: Tensor, n_bins: int) -> None:
    if rates.ndim != 2 or factors.ndim != 2:
        raise ValueError(
            f"rates and factors must be 2-D, got {tuple(rates.shape)} and {tuple(factors.shape)}"
        )
    if rates.shape[0] != factors.shape[0]:
        raise ValueError(f"rates has {rates.shape[0]} samples but factors has {factors.shape[0]}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    if rates.isnan().any() or factors.isnan().any():
        raise ValueError("rates or factors contain NaN")


def tuning_curves(rates: Tensor, factors: Tensor, n_bins: int) -> TuningCurves:
    """Mean rate of every neuron in `n_bins` equal-count bins of every latent.

    `rates` is (P, N), `factors` (P, K). Each latent is binned on its own by rank (the same
    `rank_bins` the MIR uses), so the other latents are marginalised over. Tied latent values
    share a bin, so bins are only exactly equal-count for continuous latents; an empty bin
    gives NaN rather than a made-up value.
    """
    _check_inputs(rates, factors, n_bins)
    rates, factors = rates.double(), factors.double()
    # (P, K, B) one-hot membership of each sample in each latent's bins
    onehot = torch.nn.functional.one_hot(rank_bins(factors, n_bins), n_bins).double()
    count = onehot.sum(0)
    total = torch.einsum("pn,pkb->nkb", rates, onehot)
    total_sq = torch.einsum("pn,pkb->nkb", rates**2, onehot)
    mean = total / count
    # sample variance via sums of squares: (sum x^2 - n mean^2) / (n - 1); clamp round-off
    variance = ((total_sq - count * mean**2) / (count - 1)).clamp(min=0)
    sem = (variance / count).sqrt()
    centres = torch.einsum("pk,pkb->kb", factors, onehot) / count
    return TuningCurves(mean, sem, centres, count)


def select_neurons(
    mir_per_neuron: np.ndarray | Tensor, n: int, order: str, seed: int
) -> np.ndarray:
    """Indices of `n` neurons: random (seeded), or the lowest/highest per-neuron MIR first."""
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}, got {order!r}")
    mir = np.asarray(mir_per_neuron, dtype=float)
    if mir.ndim != 1:
        raise ValueError(f"mir_per_neuron must be 1-D, got shape {mir.shape}")
    if n > len(mir):
        raise ValueError(f"asked for {n} neurons but there are only {len(mir)} neurons")
    if np.isnan(mir).any():
        raise ValueError("mir_per_neuron contains NaN")
    if order == "random":
        return np.random.default_rng(seed).choice(len(mir), size=n, replace=False)
    ranked = np.argsort(mir, kind="stable")
    return ranked[:n] if order == "mir_ascending" else ranked[::-1][:n]
