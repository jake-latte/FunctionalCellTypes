"""Modularity of a network's neurons for the task factors: mutual information, preference, MIR.

For hidden layer l, neuron n and factor group k, I[l, n, k] is the plug-in mutual information
(nats) between the neuron's firing rate and the factor. A neuron's preference is
r = max_k I / sum_k I, and the layer's modularity index is
MIR = (mean_n r - 1/K) / (1 - 1/K): 1 when every neuron carries information about one group
only, ~0 when information is spread evenly over the K groups.

Everything here is pure (no file access, no plotting) and vectorised over neurons.

Conventions:
- Integer factors are categorical, floating ones continuous (rank-binned). A group's
  information is the SUM of its members' (the total the neuron carries about them; the
  information about their joint configuration cannot be estimated by a plug-in estimator).
  `groups=None` makes each factor its own group; a factor in no group is ignored, a factor in
  two groups counts in both, and constant factors are accepted (zero information, but they
  still count towards K).
- Dead neurons (total information <= tol, e.g. constant firing) get r = 1/K so they add zero
  to the MIR numerator; `n_dead` reports them.
- The raw (plug-in) MIR has a floor above zero for unrelated neurons that depends on the
  factors' cardinalities: the plug-in information is biased upward by ~(B-1)(C-1)/(2P) nats
  and max/sum of positive numbers is never 1/K. At P = 8192 and 10 rate bins, noise neurons
  score MIR ~ 0.89 with one 25-way and two binary factors but ~ 0.10 with three five-way
  factors. `bias_correction="shuffle"` removes this (corrected noise MIR ~ 0), and
  `shuffled_mir` gives the raw chance level; plot raw, corrected and baseline together.
"""

from typing import Literal, NamedTuple

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
    if P == 0:
        raise ValueError("values has no samples (P = 0)")
    columns = values.T.contiguous()  # searchsorted needs the sorted axis last
    sorted_columns = columns.sort(dim=1).values
    below = torch.searchsorted(sorted_columns, columns, right=False)
    at_most = torch.searchsorted(sorted_columns, columns, right=True)
    # Integer arithmetic keeps the floor exact: floor((below + at_most) * n_bins / (2 P)).
    return ((below + at_most) * n_bins // (2 * P)).T.contiguous()


def factor_labels(factor: Tensor, n_bins: int, name: str = "factor") -> tuple[Tensor, int]:
    """Dense int64 labels in [0, C) for one factor, and C.

    Integer factors are categorical and are relabelled to 0..C-1; floating factors are
    continuous and are binned into `n_bins` equal-frequency bins. `name` only labels errors.
    """
    if factor.ndim != 1:
        raise ValueError(f"{name!r} must be 1-D (P,), got shape {tuple(factor.shape)}")
    if factor.dtype == torch.bool or factor.is_complex():
        raise ValueError(f"{name!r} must have an integer or floating dtype, got {factor.dtype}")
    if factor.is_floating_point():
        if factor.isnan().any():
            raise ValueError(f"{name!r} contains NaN")
        return rank_bins(factor[:, None], n_bins)[:, 0], n_bins
    _, labels = factor.unique(return_inverse=True)
    return labels, int(labels.max()) + 1


def mutual_information(bins: Tensor, n_bins: int, labels: Tensor, n_labels: int) -> Tensor:
    """Plug-in mutual information (nats) between each column of `bins` and `labels`, shape (N,).

    All neurons share one flat bincount over n * B * C + b * C + c, so there is no loop over
    neurons. The estimate is biased upward by about (B - 1)(C - 1) / (2 P) nats; round-off
    negatives are clamped to zero.
    """
    if bins.ndim != 2:
        raise ValueError(f"bins must be 2-D (P, N), got shape {tuple(bins.shape)}")
    P, N = bins.shape
    if P == 0:
        raise ValueError("bins has no samples (P = 0)")
    if bins.is_floating_point() or labels.is_floating_point() or bins.dtype == torch.bool:
        raise ValueError(f"bins and labels must be integer, got {bins.dtype} and {labels.dtype}")
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
    if rates.is_floating_point() and rates.isnan().any():
        raise ValueError("rates contains NaN")
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


def _member_information(bins: Tensor, n_bins: int, encoded: list[tuple[Tensor, int]]) -> Tensor:
    """(N, n_members) mutual information of the binned rates with each encoded factor."""
    return torch.stack([mutual_information(bins, n_bins, lab, C) for lab, C in encoded], dim=1)


def neuron_factor_information(
    rates: Tensor,
    factors: dict[str, Tensor],
    groups: dict[str, list[str]] | None = None,
    n_bins: int = 10,
    bias_correction: Literal["none", "shuffle"] = "none",
    n_shuffles: int = 5,
    seed: int = 0,
) -> FactorInformation:
    """Mutual information (nats) of every neuron's rate with every factor and factor group.

    A group's information is the SUM of its members' informations: the total the neuron
    carries about the group's members (see the module docstring for the group conventions).

    bias_correction="shuffle" subtracts from every MEMBER information (before the group sum)
    its mean over `n_shuffles` joint label shuffles (one shared permutation for all factors,
    from a seeded CPU generator) and clamps at 0. A neuron whose raw total information is not
    above twice its shuffled-null total is indistinguishable from noise: its row is zeroed, so
    it is dead (r = 1/K). Without that step the leftover noise residuals, which are mostly
    one-member, would give noise neurons a spuriously high preference. "none" is the plain
    plug-in estimate.
    """
    if bias_correction not in ("none", "shuffle"):
        raise ValueError(f"bias_correction must be 'none' or 'shuffle', got {bias_correction!r}")
    if n_shuffles < 1:
        raise ValueError(f"n_shuffles must be >= 1, got {n_shuffles}")
    _check_samples(rates, factors, n_bins)
    membership = _membership(factors, groups)
    bins = rank_bins(rates, n_bins)
    encoded = []
    for name, factor in factors.items():
        labels, n_labels = factor_labels(factor, n_bins, name)
        encoded.append((labels.to(bins.device), n_labels))
    member_mi = _member_information(bins, n_bins, encoded)
    if bias_correction == "shuffle":
        generator = torch.Generator().manual_seed(seed)
        P = bins.shape[0]
        null = torch.zeros_like(member_mi)
        for _ in range(n_shuffles):
            perm = torch.randperm(P, generator=generator).to(bins.device)
            shuffled = [(labels[perm], C) for labels, C in encoded]
            null += _member_information(bins, n_bins, shuffled) / n_shuffles
        noise_like = member_mi.sum(dim=1) <= 2 * null.sum(dim=1)
        member_mi = (member_mi - null).clamp(min=0.0)
        member_mi[noise_like] = 0.0
    group_names = list(factors) if groups is None else list(groups)
    return FactorInformation(member_mi, member_mi @ membership.to(member_mi.device), group_names)


def preference(group_mi: Tensor, tol: float = 1e-12) -> tuple[Tensor, Tensor]:
    """Preference r = max / sum per neuron: (N, K) group informations -> r (N,), dead (N,) bool.

    Needs K >= 2 groups (with one group the preference is trivially 1). A neuron whose total information is <= tol (constant firing, or a rate unrelated to every
    factor at floating-point level) has no defined preference. It is deliberately assigned
    r = 1/K, exactly chance, so it adds zero to the MIR numerator instead of NaN or a spurious
    preference; callers count the dead neurons separately.
    """
    if group_mi.ndim != 2:
        raise ValueError(f"group_mi must be 2-D (N, K), got shape {tuple(group_mi.shape)}")
    K = group_mi.shape[1]
    if K < 2:
        raise ValueError(f"preference needs K >= 2 factor groups, got K={K}")
    total = group_mi.sum(dim=1)
    dead = total <= tol
    ratio = group_mi.max(dim=1).values / total.where(~dead, torch.ones_like(total))
    return torch.where(dead, torch.full_like(ratio, 1 / K), ratio), dead


def mir(r: Tensor, K: int) -> float:
    """Modularity index (mean r - 1/K) / (1 - 1/K) of preferences `r` (N,): 1 fully modular.

    0 is chance only for dead or perfectly evenly spread neurons: at finite P the raw index of
    unrelated neurons sits above 0 (see the module docstring).
    """
    if K < 2:
        raise ValueError(f"the modularity index needs K >= 2 factor groups, got K={K}")
    return float((r.double().mean() - 1 / K) / (1 - 1 / K))


def layer_mir(
    rates: Tensor,
    factors: dict[str, Tensor],
    groups: dict[str, list[str]] | None = None,
    n_bins: int = 10,
    bias_correction: Literal["none", "shuffle"] = "none",
    n_shuffles: int = 5,
    seed: int = 0,
) -> LayerMIR:
    """Modularity of one layer's firing rates `rates` (P, N) for the factors or factor groups.

    Informations are in nats; `group_mi` sums member informations within each group; dead
    neurons (no information) have r = 1/K and are counted in `n_dead`. The bias-correction
    arguments are those of `neuron_factor_information`. See the module docstring.
    """
    info = neuron_factor_information(
        rates, factors, groups, n_bins, bias_correction, n_shuffles, seed
    )
    r, dead = preference(info.group_mi)
    return LayerMIR(
        mir(r, len(info.group_names)), r, info.group_mi, int(dead.sum()), info.group_names
    )


def network_mir(
    rates: Tensor,
    factors: dict[str, Tensor],
    groups: dict[str, list[str]] | None = None,
    n_bins: int = 10,
    bias_correction: Literal["none", "shuffle"] = "none",
    n_shuffles: int = 5,
    seed: int = 0,
) -> list[LayerMIR]:
    """`layer_mir` of every hidden layer of `rates` (P, L, N); rates are phi(H), not H.

    Same units, dead-neuron handling and group-sum convention as `layer_mir` (see the module
    docstring); the shuffles use the same seed for every layer.
    """
    if rates.ndim != 3:
        raise ValueError(f"rates must be 3-D (P, L, N), got shape {tuple(rates.shape)}")
    return [
        layer_mir(rates[:, layer], factors, groups, n_bins, bias_correction, n_shuffles, seed)
        for layer in range(rates.shape[1])
    ]


def shuffle_factors(factors: dict[str, Tensor], generator: torch.Generator) -> dict[str, Tensor]:
    """Permute the samples of every factor with ONE shared permutation (keeps dependencies)."""
    if not factors:
        raise ValueError("need at least one factor to shuffle")
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
