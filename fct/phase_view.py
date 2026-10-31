"""The computation behind the phase-transition notebook: one setting of a `phase_transition`
sweep (an ensemble of B networks, random embeddings in d dimensions) turned into the arrays its
figures draw.

Layers (nothing here plots, and only `load_phase_run` reads disk):

- loading: `load_phase_run` -> `PhaseRun`;
- geometry: `unit_order` sorts units by one coordinate of their position;
- test data: `random_test_set` draws new samples of the training set's own teacher;
- the prior: `layer_priors` gives each weight layer's covariance and penalty factors (with the
  scale lambda_w applied) and their leading spectra;
- training curves: `ensemble_statistics` gives mean and standard error over members;
- matrices: `member_view` (one network), `ensemble_views` (every member) and `average_views`
  (their mean in unit-index space: members share the embedding, so unit i is the same position
  in every member);
- examples: `ensemble_example_outputs`.

Layers follow `fct.spatial`: weight l = 0..L maps population l (pre) to l + 1 (post), in (post,
pre) orientation; the readout `V` is stored (pre, post), so layer L's weight is `V.T`.
"""

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor

from fct.config import resolve_device, resolve_dtype
from fct.data import Dataset, get_dataset, make_dataset
from fct.grid_view import tuning_view
from fct.meanfield import readout_activities
from fct.model import MLP, EnsembleMLP
from fct.modularity import information_overlap, rate_overlap
from fct.spatial import LayerSpace, data_kernel
from fct.spatial_grid import GridSetting, select_setting
from fct.spatial_view import empirical_kernels, layer_spaces
from fct.tasks import resample
from fct.train import History, load_latest_checkpoint, load_trained

# History entries with a member axis that the notebook shows as mean +- sem.
ENSEMBLE_KEYS = ("loss", "mse", "weight_l2", "rate_l2")
# The random test set's data seed is the training set's plus this, so it is new but recorded.
TEST_SEED_OFFSET = 1_000_003


# --------------------------------------------------------------------------- loading (I/O)


class PhaseRun(NamedTuple):
    """One setting of the sweep. `net` and `history` are None for a setting with no weights
    (status "missing" or "stale"); a "partial" one holds the latest checkpoint, `epoch` epochs
    in. `dataset` is the training set (generated in memory, not written, when none is cached)."""

    setting: GridSetting
    net: EnsembleMLP | None
    history: History | None
    epoch: int  # epochs completed by `net`; 0 without weights
    spaces: list[LayerSpace]  # built from the config alone, so available for any setting
    dataset: Dataset


def _load_weights(chosen: GridSetting) -> tuple[EnsembleMLP | None, History | None, int]:
    cfg, device, dtype = chosen.cfg, resolve_device(chosen.cfg), resolve_dtype(chosen.cfg.dtype)
    if chosen.status == "done":
        net, history = load_trained(chosen.directory, cfg, device, dtype)
        return net, history, cfg.train.n_epochs
    if chosen.status == "partial":
        return load_latest_checkpoint(chosen.directory, cfg, device, dtype)
    return None, None, 0


def load_phase_run(root, values, device: str = "cpu") -> PhaseRun:
    """The setting of the sweep at `root` whose axes equal `values` (`select_setting`: LookupError
    for an unknown axis or value, or several matches). ValueError unless it is an ensemble
    (net.B > 1), since every view here reads a member axis."""
    chosen = select_setting(root, values, device)
    cfg = chosen.cfg
    if cfg.net.B < 2:
        raise ValueError(f"phase_view reads ensembles, but net.B = {cfg.net.B} for {dict(values)}")
    net, history, epoch = _load_weights(chosen)
    if net is None:
        # Nothing was trained, so no dataset file need exist: the data are a pure function of the
        # config, and generating them must not write into the cache of a setting never run.
        dataset = make_dataset(cfg.data, resolve_device(cfg), resolve_dtype(cfg.dtype))
    else:
        dataset = get_dataset(chosen.directory, cfg)
    return PhaseRun(chosen, net, history, epoch, layer_spaces(cfg), dataset)


# --------------------------------------------------------------------------- unit order


def unit_order(positions: Tensor, sort_dim: int) -> np.ndarray:
    """Unit indices sorted by coordinate `sort_dim` of their positions (n, d); ties keep unit
    order (stable). ValueError unless 0 <= sort_dim < d."""
    if positions.ndim != 2:
        raise ValueError(f"positions must be (n, d), got shape {tuple(positions.shape)}")
    d = positions.shape[1]
    if not 0 <= sort_dim < d:
        raise ValueError(f"sort_dim must lie in 0..{d - 1} for d = {d}, got {sort_dim}")
    return np.argsort(positions[:, sort_dim].detach().cpu().numpy(), kind="stable")


# --------------------------------------------------------------------------- random test set


class RandomTestSet(NamedTuple):
    dataset: Dataset
    seed: int  # the data seed of these samples, recorded so the set can be redrawn


def random_test_set(
    data_cfg, train: Dataset, n_points: int, seed: int | None = None
) -> RandomTestSet:
    """`n_points` new samples of the linear teacher `train` was drawn from (same W_X, W_Y,
    different latents), for the tuning and MIR calculation: more points lower the MIR's
    plug-in noise floor. `seed` defaults to the training data seed + `TEST_SEED_OFFSET`;
    ValueError if it equals the training seed or `n_points` < 1. Pure: nothing is written."""
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}")
    seed = data_cfg.seed + TEST_SEED_OFFSET if seed is None else seed
    cfg = data_cfg.model_copy(update={"P": n_points})
    return RandomTestSet(resample(cfg, train, seed, train.x.device, train.x.dtype), seed)


# --------------------------------------------------------------------------- the prior


class FactorPair(NamedTuple):
    post: Tensor
    pre: Tensor


def _check_lambda(weight_lambda: float) -> float:
    if not weight_lambda > 0:
        raise ValueError(f"weight_lambda must be > 0 to rescale the factors, got {weight_lambda}")
    return math.sqrt(weight_lambda)


def covariance_factors(space: LayerSpace, weight_lambda: float) -> FactorPair:
    """K^post / sqrt(lambda_w) and K^pre / sqrt(lambda_w). lambda_w multiplies the Kronecker
    product once, so each factor carries its square root: the factors' Kronecker product is
    K^post (x) K^pre / lambda_w, the variance of the weights once the prior strength is absorbed."""
    root = _check_lambda(weight_lambda)
    return FactorPair(space.post.K / root, space.pre.K / root)


def penalty_factors(space: LayerSpace, weight_lambda: float) -> FactorPair:
    """sqrt(lambda_w) M^post and sqrt(lambda_w) M^pre, whose Kronecker product is
    lambda_w M^post (x) M^pre: the loss penalises W by (lambda_w / 2 beta) tr(M^post W M^pre W^T),
    so this is that quadratic form without the 1 / (2 beta)."""
    root = _check_lambda(weight_lambda)
    return FactorPair(root * space.post.M, root * space.pre.M)


def descending_spectrum(matrix: Tensor, n_eig: int) -> Tensor:
    """The `n_eig` largest eigenvalues of the symmetric `matrix`, largest first (all of them if
    the matrix is smaller)."""
    if n_eig < 1:
        raise ValueError(f"n_eig must be >= 1, got {n_eig}")
    return torch.linalg.eigvalsh(matrix).flip(0)[:n_eig]


class LayerPrior(NamedTuple):
    layer: int
    covariance: FactorPair
    penalty: FactorPair
    covariance_spectra: FactorPair
    penalty_spectra: FactorPair


def _spectra(factors: FactorPair, n_eig: int) -> FactorPair:
    return FactorPair(
        descending_spectrum(factors.post, n_eig), descending_spectrum(factors.pre, n_eig)
    )


def layer_priors(
    spaces: Sequence[LayerSpace], weight_lambda: float, n_eig: int
) -> list[LayerPrior]:
    """Every weight layer's covariance and penalty factors (see `covariance_factors`,
    `penalty_factors`) with the leading `n_eig` eigenvalues of each factor."""

    def prior(space: LayerSpace) -> LayerPrior:
        cov, pen = covariance_factors(space, weight_lambda), penalty_factors(space, weight_lambda)
        return LayerPrior(space.layer, cov, pen, _spectra(cov, n_eig), _spectra(pen, n_eig))

    return [prior(space) for space in spaces]


# --------------------------------------------------------------------------- ensemble statistics


class EnsembleStatistics(NamedTuple):
    epoch: np.ndarray  # (E,) logged epochs
    mean: dict[str, np.ndarray]  # key -> (E,) mean over members
    sem: dict[str, np.ndarray]  # key -> (E,) standard error of that mean


def _member_matrix(history: History, key: str) -> np.ndarray:
    if key not in history:
        raise ValueError(f"the history has no {key!r}; it has {sorted(history)}")
    values = np.asarray(history[key], dtype=float)
    if values.ndim != 2:
        raise ValueError(
            f"history[{key!r}] must be (logged epochs, members), got shape {values.shape}; "
            "is this a single network's history, with no member axis?"
        )
    return values


def ensemble_statistics(
    history: History, keys: Sequence[str] = ENSEMBLE_KEYS
) -> EnsembleStatistics:
    """Mean and standard error of the mean over members at every logged epoch, for each of
    `keys` (default loss, mse, weight penalty, rate penalty). sem = std(ddof = 1) / sqrt(B); for
    B = 1 it is NaN, not 0: one member says nothing about the spread, and 0 would draw as certain."""
    matrices = {key: _member_matrix(history, key) for key in keys}

    def sem(values: np.ndarray) -> np.ndarray:
        if values.shape[1] < 2:
            return np.full(values.shape[0], np.nan)
        return values.std(axis=1, ddof=1) / math.sqrt(values.shape[1])

    return EnsembleStatistics(
        np.asarray(history["epoch"]),
        {key: values.mean(axis=1) for key, values in matrices.items()},
        {key: sem(values) for key, values in matrices.items()},
    )


# --------------------------------------------------------------------------- matrices


class KernelPair(NamedTuple):
    post: Tensor  # (n_post, n_post)  W W^T / n_pre
    pre: Tensor  # (n_pre, n_pre)  W^T W / n_post


class MemberView(NamedTuple):
    A: Tensor  # (L, P_lattice, P_lattice) data kernel of each hidden layer, on the lattice set
    info_overlap: Tensor  # (L, N, N) information overlap, on the random test set
    rate_overlap: Tensor  # (L, N, N) rate overlap, on the random test set
    weight_kernels: list[KernelPair]  # trained weight kernels of layers 0..L, (post, pre) weights


def _data_kernels(net: MLP, lattice: Dataset, spaces: Sequence[LayerSpace]) -> Tensor:
    """A^(l) = psi^T K_pre psi / n_pre of every hidden layer l (0-based), psi = phi(h^l) and
    K_pre that of the weight layer it feeds. The K are float64 on the CPU, so psi goes there."""
    with torch.no_grad():
        H, _ = net(lattice.x.to(net.dtype))
    psis = readout_activities(H, net.phi)
    kernels = [spaces[layer + 1].pre.K for layer in range(len(psis))]
    return torch.stack(
        [
            data_kernel(psi.detach().cpu().to(K.dtype), K)
            for psi, K in zip(psis, kernels, strict=True)
        ]
    )


def _overlaps(
    net: MLP, test: Dataset, n_bins: int, kind: str, n_shuffles: int, seed: int
) -> tuple[Tensor, Tensor]:
    views = [
        tuning_view(net, test, layer, n_bins, kind, n_shuffles, seed) for layer in range(net.L)
    ]
    info = torch.stack([information_overlap(v.member_mi) for v in views])
    rate = torch.stack([rate_overlap(v.curves.mean) for v in views])
    return info, rate


def _weight_kernels(net: MLP) -> list[KernelPair]:
    weights = [net.U, *net.J, net.V.T]
    return [KernelPair(*empirical_kernels(W.detach().cpu())) for W in weights]


def member_view(
    net: MLP,
    lattice: Dataset,
    test: Dataset,
    spaces: Sequence[LayerSpace],
    n_bins: int,
    kind: str,
    n_shuffles: int,
    seed: int,
) -> MemberView:
    """The matrices of one network: data kernels A^(l) on the `lattice` test set; information
    and rate overlaps from the tuning of every neuron on the (random) `test` set, with the
    tuning's `n_bins`, MIR `kind` ("raw" or "corrected"), `n_shuffles` and `seed`
    (`fct.grid_view.tuning_view`); and the empirical kernels of the trained weights."""
    info, rate = _overlaps(net, test, n_bins, kind, n_shuffles, seed)
    return MemberView(_data_kernels(net, lattice, spaces), info, rate, _weight_kernels(net))


def ensemble_views(net: EnsembleMLP, *args, **kwargs) -> list[MemberView]:
    """`member_view` of every member of `net` (arguments as there, after the net)."""
    return [member_view(net.member(b), *args, **kwargs) for b in range(net.B)]


def _mean(tensors: Sequence[Tensor]) -> Tensor:
    return torch.stack(list(tensors)).mean(0)


def average_views(views: Sequence[MemberView]) -> MemberView:
    """The elementwise mean of members' views, in unit-index space (members share their
    embeddings, so unit i is at the same position in every one)."""
    if not views:
        raise ValueError("no views to average")
    n_layers = len(views[0].weight_kernels)
    kernels = [
        KernelPair(
            _mean([v.weight_kernels[layer].post for v in views]),
            _mean([v.weight_kernels[layer].pre for v in views]),
        )
        for layer in range(n_layers)
    ]
    return MemberView(
        _mean([v.A for v in views]),
        _mean([v.info_overlap for v in views]),
        _mean([v.rate_overlap for v in views]),
        kernels,
    )


# --------------------------------------------------------------------------- examples


class EnsembleExamples(NamedTuple):
    indices: Tensor  # (n,) rows of the dataset
    x: Tensor  # (n, D_in)
    y_true: Tensor  # (n, D_out)
    y_pred: Tensor  # (B, n, D_out) every member's output


def ensemble_example_outputs(
    net: EnsembleMLP, dataset: Dataset, indices: Tensor | Sequence[int]
) -> EnsembleExamples:
    """Every member's outputs on the dataset rows `indices`, with the inputs and targets, on the CPU."""
    idx = torch.as_tensor(indices).cpu()
    if idx.numel() == 0 or idx.is_floating_point() or idx.dtype == torch.bool:
        raise ValueError(f"indices must be a non-empty list of integers, got {idx}")
    idx = idx.long()
    n_rows = len(dataset.x)
    if idx.ndim != 1 or idx.min() < 0 or idx.max() >= n_rows:
        raise ValueError(f"indices must be a 1-D list of rows in [0, {n_rows}), got {idx}")
    rows = idx.to(dataset.x.device)
    with torch.no_grad():
        _, Z = net(dataset.x[rows])  # (B, n, D_out)
    return EnsembleExamples(idx, dataset.x[rows].cpu(), dataset.y[rows].cpu(), Z.cpu())
