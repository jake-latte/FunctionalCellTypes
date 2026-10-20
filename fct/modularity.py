"""Modularity of a network's neurons for the task factors: mutual information, preference, MIR.

For hidden layer l, neuron n and factor group k, I[l, n, k] is the plug-in mutual information
(nats) between the neuron's firing rate and the factor. A neuron's preference is
r = max_k I / sum_k I, and the layer's modularity index is
MIR = (mean_n r - 1/K) / (1 - 1/K): 1 when every neuron carries information about one group
only, ~0 when information is spread evenly over the K groups.

Everything here is pure (no file access, no plotting) and vectorised over neurons.
"""

from typing import NamedTuple

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


class FactorInformation(NamedTuple):
    member_mi: Tensor  # (N, n_members) one column per factor, in the order of `factors`
    group_mi: Tensor  # (N, K) sum of the member columns of each group
    group_names: list[str]


class LayerMIR(NamedTuple):
    mir: float
    r: Tensor  # (N,) preference of every neuron; dead neurons sit at 1/K
    group_mi: Tensor  # (N, K)
    n_dead: int
    group_names: list[str]


def _check_samples(rates: Tensor, factors: dict[str, Tensor], n_bins: int) -> None:
    if rates.ndim != 2:
        raise ValueError(f"rates must be 2-D (P, N), got shape {tuple(rates.shape)}")
    if not factors:
        raise ValueError("need at least one factor")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    for name, factor in factors.items():
        if factor.shape != (rates.shape[0],):
            raise ValueError(
                f"factor {name!r} has shape {tuple(factor.shape)} but rates has "
                f"{rates.shape[0]} samples"
            )


def _membership(factors: dict[str, Tensor], groups: dict[str, list[str]] | None) -> Tensor:
    """(n_members, K) 0/1 matrix: entry [m, k] is 1 when factor m belongs to group k."""
    groups = {name: [name] for name in factors} if groups is None else groups
    members = list(factors)
    matrix = torch.zeros(len(members), len(groups), dtype=torch.float64)
    for k, (group, names) in enumerate(groups.items()):
        unknown = [name for name in names if name not in factors]
        if unknown or not names:
            raise ValueError(f"group {group!r} is empty or has unknown factors: {unknown}")
        for name in names:
            matrix[members.index(name), k] = 1.0
    return matrix


def neuron_factor_information(
    rates: Tensor,
    factors: dict[str, Tensor],
    groups: dict[str, list[str]] | None = None,
    n_bins: int = 10,
) -> FactorInformation:
    """Mutual information (nats) of every neuron's rate with every factor and factor group.

    A group's information is the SUM of its members' informations: the total the neuron
    carries about the group's members. The information about their joint configuration is not
    used because a plug-in estimator cannot resolve it (the joint table grows exponentially
    with the number of members). `groups=None` makes every factor its own group.
    """
    _check_samples(rates, factors, n_bins)
    membership = _membership(factors, groups)
    bins = rank_bins(rates, n_bins)
    columns = []
    for factor in factors.values():
        labels, n_labels = factor_labels(factor, n_bins)
        columns.append(mutual_information(bins, n_bins, labels.to(bins.device), n_labels))
    member_mi = torch.stack(columns, dim=1)
    group_names = list(factors) if groups is None else list(groups)
    return FactorInformation(member_mi, member_mi @ membership.to(member_mi.device), group_names)


def preference(group_mi: Tensor, tol: float = 1e-12) -> tuple[Tensor, Tensor]:
    """Preference r = max / sum of each neuron's group informations, and a dead-neuron mask.

    A neuron whose total information is <= tol (constant firing, or a rate unrelated to every
    factor at floating-point level) has no defined preference. It is deliberately assigned
    r = 1/K, exactly chance, so it adds zero to the MIR numerator instead of NaN or a spurious
    preference; callers count the dead neurons separately.
    """
    K = group_mi.shape[1]
    total = group_mi.sum(dim=1)
    dead = total <= tol
    ratio = group_mi.max(dim=1).values / total.where(~dead, torch.ones_like(total))
    return torch.where(dead, torch.full_like(ratio, 1 / K), ratio), dead


def mir(r: Tensor, K: int) -> float:
    """Modularity index (mean r - 1/K) / (1 - 1/K): 1 fully modular, ~0 at chance."""
    if K < 2:
        raise ValueError(f"the modularity index needs K >= 2 factor groups, got K={K}")
    return float((r.double().mean() - 1 / K) / (1 - 1 / K))


def layer_mir(
    rates: Tensor,
    factors: dict[str, Tensor],
    groups: dict[str, list[str]] | None = None,
    n_bins: int = 10,
) -> LayerMIR:
    """Modularity of one layer's firing rates `rates` (P, N) for the factors or factor groups."""
    info = neuron_factor_information(rates, factors, groups, n_bins)
    r, dead = preference(info.group_mi)
    return LayerMIR(
        mir(r, len(info.group_names)), r, info.group_mi, int(dead.sum()), info.group_names
    )


def network_mir(
    rates: Tensor,
    factors: dict[str, Tensor],
    groups: dict[str, list[str]] | None = None,
    n_bins: int = 10,
) -> list[LayerMIR]:
    """`layer_mir` of every hidden layer of `rates` (P, L, N); rates are phi(H), not H."""
    if rates.ndim != 3:
        raise ValueError(f"rates must be 3-D (P, L, N), got shape {tuple(rates.shape)}")
    return [layer_mir(rates[:, layer], factors, groups, n_bins) for layer in range(rates.shape[1])]


def shuffle_factors(factors: dict[str, Tensor], generator: torch.Generator) -> dict[str, Tensor]:
    """Permute the samples of every factor with ONE shared permutation (keeps dependencies)."""
    P = next(iter(factors.values())).shape[0]
    perm = torch.randperm(P, generator=generator)
    return {name: factor[perm.to(factor.device)] for name, factor in factors.items()}


def shuffled_mir(
    rates: Tensor,
    factors: dict[str, Tensor],
    groups: dict[str, list[str]] | None = None,
    n_bins: int = 10,
    seed: int = 0,
    n_shuffles: int = 1,
) -> float:
    """Chance level of `layer_mir`: its MIR with factors jointly permuted across samples.

    The plug-in information is biased upward and max/sum has a positive floor even for
    unrelated neurons, so an MIR curve is only meaningful next to this baseline. The mean over
    `n_shuffles` permutations from a seeded CPU generator is returned.
    """
    if n_shuffles < 1:
        raise ValueError(f"n_shuffles must be >= 1, got {n_shuffles}")
    _check_samples(rates, factors, n_bins)
    generator = torch.Generator().manual_seed(seed)
    values = [
        layer_mir(rates, shuffle_factors(factors, generator), groups, n_bins).mir
        for _ in range(n_shuffles)
    ]
    return sum(values) / n_shuffles
