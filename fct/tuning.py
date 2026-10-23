"""Tuning curves of a layer's neurons to each latent, and choosing which neurons to show.

Pure functions on plain inputs; plotting lives in `fct.draw.fig_tuning`.
"""

from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor

from fct.modularity import factor_labels

ORDERS = ("random", "mir_ascending", "mir_descending")


class TuningCurves(NamedTuple):
    # B is the largest bin count over the latents (n_bins for float latents, levels for integer)
    mean: Tensor  # (N, K, B) mean rate of each neuron in each bin of each latent
    sem: Tensor  # (N, K, B) standard error of that mean; NaN with fewer than 2 samples
    centres: Tensor  # (K, B) mean latent value in each bin; NaN for an empty bin
    count: Tensor  # (K, B) samples per bin


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


def _binned(factors: Tensor, n_bins: int) -> Tensor:
    """(P, K, B) one-hot bin membership of every latent on the MIR's bins.

    Each column goes through `modularity.factor_labels`: integer latents are categorical (one
    bin per level), floating ones are rank-binned into `n_bins`. B is the largest bin count;
    a latent with fewer bins has empty (NaN) trailing ones.
    """
    labelled = [
        factor_labels(factors[:, k], n_bins, f"latent {k}") for k in range(factors.shape[1])
    ]
    width = max(n for _, n in labelled)
    labels = torch.stack([lab for lab, _ in labelled], dim=1)
    return torch.nn.functional.one_hot(labels, width).double()


def tuning_curves(rates: Tensor, factors: Tensor, n_bins: int) -> TuningCurves:
    """Mean rate of every neuron in the bins of every latent, other latents marginalised.

    `rates` is (P, N), `factors` (P, K). The bins are those the MIR uses
    (`modularity.factor_labels`): a floating latent is split into `n_bins` rank bins, equal-count
    when its values are distinct (tied values share a bin); an integer latent is categorical,
    one bin per level, and its `centres` are the level values. The arrays are as wide as the
    largest latent's bin count; an empty bin is NaN rather than a made-up value, and a bin with
    fewer than 2 samples has a NaN standard error. `factors` is one tensor, so a call is all
    categorical (integer dtype) or all continuous (float): casting integer factors to float makes
    them rank-binned, with interior empty bins, so pass integer factors as integers.
    """
    _check_inputs(rates, factors, n_bins)
    onehot = _binned(factors, n_bins)
    count = onehot.sum(0)
    # Sums of squares are taken about each neuron's global mean: with a large offset the raw
    # sum x^2 - n mean^2 cancels catastrophically and would give SEM 0.
    rates = rates.double()
    offset = rates.mean(0)
    centred = rates - offset
    mean_shift = torch.einsum("pn,pkb->nkb", centred, onehot) / count
    total_sq = torch.einsum("pn,pkb->nkb", centred**2, onehot)
    # The clamp only absorbs round-off that makes a tiny variance slightly negative.
    variance = ((total_sq - count * mean_shift**2) / (count - 1)).clamp(min=0)
    sem = (variance / count).sqrt()
    mean = mean_shift + offset[:, None, None]
    centres = torch.einsum("pk,pkb->kb", factors.double(), onehot) / count
    return TuningCurves(mean, sem, centres, count)


def _to_numpy(x: np.ndarray | Tensor) -> np.ndarray:
    if isinstance(x, Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def select_neurons(
    mir_per_neuron: np.ndarray | Tensor, n: int, order: str, seed: int
) -> np.ndarray:
    """Indices of `n` neurons: random (seeded), or the lowest/highest per-neuron MIR first."""
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}, got {order!r}")
    mir = _to_numpy(mir_per_neuron)
    if mir.ndim != 1:
        raise ValueError(f"mir_per_neuron must be 1-D, got shape {mir.shape}")
    if not 1 <= n <= len(mir):
        raise ValueError(f"n must be between 1 and the {len(mir)} neurons available, got {n}")
    if np.isnan(mir).any():
        raise ValueError("mir_per_neuron contains NaN")
    if order == "random":
        return np.random.default_rng(seed).choice(len(mir), size=n, replace=False)
    ranked = np.argsort(mir, kind="stable")
    return ranked[:n] if order == "mir_ascending" else ranked[::-1][:n]
