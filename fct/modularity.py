"""Modularity of a network's neurons for the task factors: mutual information, preference, MIR.

For hidden layer l, neuron n and factor group k, I[l, n, k] is the plug-in mutual information
(nats) between the neuron's firing rate and the factor. A neuron's preference is
r = max_k I / sum_k I, and the layer's modularity index is
MIR = (mean_n r - 1/K) / (1 - 1/K): 1 when every neuron carries information about one group
only, ~0 when information is spread evenly over the K groups.

Everything here is pure (no file access, no plotting) and vectorised over neurons.
"""

import torch
from torch import Tensor


def rank_bins(values: Tensor, n_bins: int) -> Tensor:
    """Equal-frequency bin index (int64, in [0, n_bins)) of every entry, per column.

    bin = floor(midrank / P * n_bins) with midrank = (#(v < x) + #(v <= x)) / 2, so tied values
    share one midrank and therefore one bin (a ReLU's mass at zero is one bin, not spread over
    several), and the result is invariant to monotone transforms of a column.
    """
    if values.ndim != 2:
        raise ValueError(f"values must be 2-D (P, N), got shape {tuple(values.shape)}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    P = values.shape[0]
    columns = values.T.contiguous()  # searchsorted needs the sorted axis last
    sorted_columns = columns.sort(dim=1).values
    below = torch.searchsorted(sorted_columns, columns, right=False)
    at_most = torch.searchsorted(sorted_columns, columns, right=True)
    # Integer arithmetic keeps the floor exact: floor((below + at_most) * n_bins / (2 P)).
    return ((below + at_most) * n_bins // (2 * P)).T.contiguous()


def factor_labels(factor: Tensor, n_bins: int) -> tuple[Tensor, int]:
    """Dense int64 labels in [0, C) for one factor, and C.

    Integer factors are categorical and are relabelled to 0..C-1; floating factors are
    continuous and are binned into `n_bins` equal-frequency bins.
    """
    if factor.ndim != 1:
        raise ValueError(f"a factor must be 1-D (P,), got shape {tuple(factor.shape)}")
    if factor.dtype == torch.bool or factor.is_complex():
        raise ValueError(f"a factor must have an integer or floating dtype, got {factor.dtype}")
    if factor.is_floating_point():
        return rank_bins(factor[:, None], n_bins)[:, 0], n_bins
    _, labels = factor.unique(return_inverse=True)
    return labels, int(labels.max()) + 1


def mutual_information(bins: Tensor, n_bins: int, labels: Tensor, n_labels: int) -> Tensor:
    """Plug-in mutual information (nats) between each column of `bins` and `labels`, shape (N,).

    All neurons share one flat bincount over n * B * C + b * C + c, so there is no loop over
    neurons. The estimate is biased upward by about (B - 1)(C - 1) / (2 P) nats; round-off
    negatives are clamped to zero.
    """
    P, N = bins.shape
    if labels.shape != (P,):
        raise ValueError(f"bins has {P} samples but labels has shape {tuple(labels.shape)}")
    if bins.min() < 0 or bins.max() >= n_bins or labels.min() < 0 or labels.max() >= n_labels:
        raise ValueError("bin or label codes outside the range [0, n_bins) x [0, n_labels)")
    neuron = torch.arange(N, device=bins.device) * (n_bins * n_labels)
    flat = neuron[None, :] + bins * n_labels + labels[:, None]
    counts = torch.bincount(flat.reshape(-1), minlength=N * n_bins * n_labels)
    joint = counts.reshape(N, n_bins, n_labels).double() / P
    marginal = joint.sum(dim=2, keepdim=True) * joint.sum(dim=1, keepdim=True)
    # xlogy(0, .) = 0 handles the empty cells without NaNs.
    mi = (torch.xlogy(joint, joint) - torch.xlogy(joint, marginal)).sum(dim=(1, 2))
    return mi.clamp(min=0.0)
