"""The computation behind the phase-transition notebook: one setting of a `phase_transition`
experiment (an ensemble of B networks, random embeddings in d dimensions) turned into the arrays its
figures draw. Single-network experiments go through `fct.spatial_exp` instead.

Layers (nothing here plots, and only `load_phase_run` reads disk):

- loading: `load_phase_run` -> `PhaseRun`;
- geometry: `unit_order` sorts units by one coordinate of their position;
- test data: `random_test_set` draws new samples of the training set's own teacher;
- the prior: `layer_priors` gives each weight layer's covariance, effective covariance and
  penalty factors (with the scale lambda_w applied), their spectra and the eigenvalue floor;
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

from fct.cache import cache_paths
from fct.config import DataConfig, resolve_device, resolve_dtype
from fct.data import Dataset, get_dataset, make_dataset
from fct.exp_view import tuning_view
from fct.model import MLP, EnsembleMLP
from fct.modularity import information_overlap, rate_overlap
from fct.spatial import LayerSpace
from fct.spatial_exp import ExpSetting, select_setting
from fct.spatial_view import data_kernels, empirical_kernels, layer_spaces
from fct.tasks import resample
from fct.train import History, load_weights

# History entries with a member axis that the notebook shows as mean +- sem.
ENSEMBLE_KEYS = ("loss", "mse", "weight_l2", "rate_l2")
# The random test set's data seed is the training set's plus this, so it is new but recorded.
TEST_SEED_OFFSET = 1_000_003


# --------------------------------------------------------------------------- loading (I/O)


class PhaseRun(NamedTuple):
    """One setting of the experiment. `net` and `history` are None for a setting with no weights
    (status "missing" or "stale"); a "partial" one holds the latest checkpoint, `epoch` epochs
    in. `dataset` is the training set (generated in memory, never written, when no usable file
    is cached)."""

    setting: ExpSetting
    net: EnsembleMLP | None
    history: History | None
    epoch: int  # epochs completed by `net`; 0 without weights
    spaces: list[LayerSpace]  # built from the config alone, so available for any setting
    dataset: Dataset


def _load_weights(chosen: ExpSetting) -> tuple[EnsembleMLP | None, History | None, int]:
    if chosen.status not in ("done", "partial"):
        return None, None, 0
    cfg = chosen.cfg
    device, dtype = resolve_device(cfg), resolve_dtype(cfg.dtype)
    return load_weights(chosen.directory, cfg, device, dtype, chosen.status == "done")


def _training_set(chosen: ExpSetting, has_weights: bool) -> Dataset:
    """The cached training set when the setting has weights and its file exists, else the same
    data generated in memory: they are a pure function of the config, and reading must not
    write into a setting's cache directory."""
    cfg = chosen.cfg
    if has_weights and (cache_paths(chosen.directory).data / "dataset.pt").exists():
        return get_dataset(chosen.directory, cfg)
    return make_dataset(cfg.data, resolve_device(cfg), resolve_dtype(cfg.dtype))


def load_phase_run(root, values, device: str = "cpu") -> PhaseRun:
    """The setting of the experiment at `root` whose axes equal `values` (`select_setting`:
    LookupError for an unknown axis or value, or several matches). Read-only. ValueError unless
    it is an ensemble (net.B > 1), since every view here reads a member axis; an experiment of
    single
    networks is read with `fct.spatial_exp`."""
    chosen = select_setting(root, values, device)
    cfg = chosen.cfg
    if cfg.net.B < 2:
        raise ValueError(
            f"phase_view reads ensembles, but net.B = {cfg.net.B} for {dict(values)}; use "
            "fct.spatial_exp for single networks"
        )
    net, history, epoch = _load_weights(chosen)
    dataset = _training_set(chosen, net is not None)
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


def _check_matches_training(data_cfg: DataConfig, train: Dataset) -> None:
    """ValueError unless `train` is exactly the set `data_cfg` generates: the same inputs and
    targets (the data depend on the config and dtype only, so they are regenerated in memory on
    the CPU and compared), which fixes P, D_in, n_latents, the latent interval and seed at once."""
    expected = make_dataset(data_cfg, torch.device("cpu"), train.x.dtype)
    same = (
        expected.x.shape == train.x.shape
        and expected.y.shape == train.y.shape
        and torch.equal(expected.x, train.x.cpu())
        and torch.equal(expected.y, train.y.cpu())
    )
    if not same:
        raise ValueError(
            "data_cfg does not match the training set: regenerating it from data_cfg "
            f"(P={data_cfg.P}, D_in={data_cfg.D_in}, n_latents={data_cfg.n_latents}, "
            f"latent_min={data_cfg.latent_min}, latent_max={data_cfg.latent_max}, "
            f"seed={data_cfg.seed}) gives different data"
        )


def random_test_set(
    data_cfg: DataConfig, train: Dataset, n_points: int, seed: int | None = None
) -> RandomTestSet:
    """`n_points` new samples of the linear teacher `train` was drawn from (same W_X, W_Y,
    fresh latents uniform on [latent_min, latent_max]), for the tuning and MIR
    calculation: more points lower the MIR's plug-in noise floor. `seed` defaults to the
    training data seed + `TEST_SEED_OFFSET`. ValueError if `data_cfg` is not the config of
    `train` (see `_check_matches_training`), if the seed equals the training seed, or if
    `n_points` < 1. Pure: nothing is written."""
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}")
    _check_matches_training(data_cfg, train)
    seed = data_cfg.seed + TEST_SEED_OFFSET if seed is None else seed
    cfg = data_cfg.model_copy(update={"P": n_points})
    return RandomTestSet(resample(cfg, train, seed, train.x.device, train.x.dtype), seed)


# --------------------------------------------------------------------------- the prior


class FactorPair(NamedTuple):
    post: Tensor
    pre: Tensor


class PopulationFloor(NamedTuple):
    """What the penalty's eigenvalue floor did to one population's kernel K. Under the floor
    every eigenvalue of K below `covariance_floor` is raised to it, so the penalty M never
    exceeds `penalty_ceiling` and the effective covariance never falls below the floor. Both
    are on the scale of the plotted factors (sqrt(lambda_w) applied)."""

    n_floored: int
    fraction_floored: float
    covariance_floor: float
    penalty_ceiling: float


class FloorPair(NamedTuple):
    post: PopulationFloor
    pre: PopulationFloor


def _check_lambda(weight_lambda: float) -> float:
    if not weight_lambda > 0:
        raise ValueError(f"weight_lambda must be > 0 to rescale the factors, got {weight_lambda}")
    return math.sqrt(weight_lambda)


def covariance_factors(space: LayerSpace, weight_lambda: float) -> FactorPair:
    """K^post / sqrt(lambda_w) and K^pre / sqrt(lambda_w). lambda_w multiplies the Kronecker
    product once, so each factor carries its square root: the factors' Kronecker product is
    K^post (x) K^pre / lambda_w. This is the prior covariance as configured, NOT what the
    networks were trained under: the penalty floors K's small eigenvalues (see
    `effective_covariance_factors`). Under `spatial.lambda_max` weight_lambda is forced to 1, so
    the split is the identity there."""
    root = _check_lambda(weight_lambda)
    return FactorPair(space.post.K / root, space.pre.K / root)


def effective_covariance_factors(space: LayerSpace, weight_lambda: float) -> FactorPair:
    """M^-1 / sqrt(lambda_w) on each side, whose Kronecker product is (lambda_w M^post (x)
    M^pre)^-1: the weight covariance the training actually samples, since the loss penalises W
    by exactly that quadratic form. It is K with its eigenvalues raised to the penalty's floor."""
    root = _check_lambda(weight_lambda)
    return FactorPair(space.post.M.inverse() / root, space.pre.M.inverse() / root)


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


def ascending_spectrum(matrix: Tensor, n_eig: int) -> Tensor:
    """The `n_eig` smallest eigenvalues of the symmetric `matrix`, smallest first (all of them
    if the matrix is smaller)."""
    if n_eig < 1:
        raise ValueError(f"n_eig must be >= 1, got {n_eig}")
    return torch.linalg.eigvalsh(matrix)[:n_eig]


class LayerPrior(NamedTuple):
    """Weight layer `layer`'s prior, as factor pairs and their spectra.

    `covariance_spectra` and `effective_covariance_spectra` hold the n_eig LARGEST eigenvalues
    of each factor, descending. `penalty_spectra` holds the n_eig SMALLEST eigenvalues of the
    penalty factors, ascending: the penalty's largest eigenvalues are all the floor value
    (a flat line), while its smallest are the reciprocals of the covariance's largest.
    `floors` says how many eigenvalues were floored and where the floor lies."""

    layer: int
    covariance: FactorPair
    effective_covariance: FactorPair
    penalty: FactorPair
    covariance_spectra: FactorPair
    effective_covariance_spectra: FactorPair
    penalty_spectra: FactorPair
    floors: FloorPair


def _pair(f, factors: FactorPair) -> FactorPair:
    return FactorPair(f(factors.post), f(factors.pre))


def _population_floor(population, weight_lambda: float) -> PopulationFloor:
    root = _check_lambda(weight_lambda)
    report = population.report
    floor = report.floor
    return PopulationFloor(report.n_floored, report.fraction_floored, floor / root, root / floor)


def layer_priors(
    spaces: Sequence[LayerSpace], weight_lambda: float, n_eig: int
) -> list[LayerPrior]:
    """Every weight layer's covariance, effective covariance and penalty factors
    (`covariance_factors`, `effective_covariance_factors`, `penalty_factors`) with their
    leading spectra (see `LayerPrior` for which end each keeps) and the floor of each side."""

    def prior(space: LayerSpace) -> LayerPrior:
        cov = covariance_factors(space, weight_lambda)
        eff = effective_covariance_factors(space, weight_lambda)
        pen = penalty_factors(space, weight_lambda)
        return LayerPrior(
            space.layer,
            cov,
            eff,
            pen,
            _pair(lambda m: descending_spectrum(m, n_eig), cov),
            _pair(lambda m: descending_spectrum(m, n_eig), eff),
            _pair(lambda m: ascending_spectrum(m, n_eig), pen),
            FloorPair(
                _population_floor(space.post, weight_lambda),
                _population_floor(space.pre, weight_lambda),
            ),
        )

    return [prior(space) for space in spaces]


# --------------------------------------------------------------------------- ensemble statistics


class EnsembleStatistics(NamedTuple):
    epoch: np.ndarray  # (E,) logged epochs
    mean: dict[str, np.ndarray]  # key -> (E,) mean over members
    sem: dict[str, np.ndarray]  # key -> (E,) standard error of that mean


def _member_matrix(history: History, key: str, n_epochs: int) -> np.ndarray:
    if key not in history:
        raise ValueError(f"the history has no {key!r}; it has {sorted(history)}")
    values = np.asarray(history[key], dtype=float)
    if values.ndim != 2:
        raise ValueError(
            f"history[{key!r}] must be (logged epochs, members), got shape {values.shape}; "
            "is this a single network's history, with no member axis?"
        )
    if values.shape[0] != n_epochs:
        raise ValueError(
            f"history[{key!r}] has {values.shape[0]} rows but history['epoch'] has {n_epochs} epochs"
        )
    return values


def ensemble_statistics(
    history: History, keys: Sequence[str] = ENSEMBLE_KEYS
) -> EnsembleStatistics:
    """Mean and standard error of the mean over members at every logged epoch, for each of
    `keys` (default loss, mse, weight penalty, rate penalty). sem = std(ddof = 1) / sqrt(B).
    For B = 1 (a synthetic or foreign history; `load_phase_run` only gives B >= 2) the sem is
    NaN, not 0: one member says nothing about the spread, and 0 would draw as certain."""
    epochs = np.asarray(history["epoch"])
    matrices = {key: _member_matrix(history, key, len(epochs)) for key in keys}

    def sem(values: np.ndarray) -> np.ndarray:
        if values.shape[1] < 2:
            return np.full(values.shape[0], np.nan)
        return values.std(axis=1, ddof=1) / math.sqrt(values.shape[1])

    return EnsembleStatistics(
        epochs,
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


def _overlaps(
    net: MLP, test: Dataset, n_bins: int, kind: str, n_shuffles: int, seed: int
) -> tuple[Tensor, Tensor]:
    views = [
        tuning_view(net, test, layer, n_bins, kind, n_shuffles, seed) for layer in range(net.L)
    ]
    info = torch.stack([information_overlap(v.member_mi) for v in views])
    rate = torch.stack([rate_overlap(v.curves.mean) for v in views])
    return info, rate


def weight_kernels(net: MLP) -> list[KernelPair]:
    """`empirical_kernels` of every weight of `net` in (post, pre) orientation (layers 0..L; the
    readout is `V.T`), in float64 on the CPU like the prior kernels they are compared with."""
    weights = [net.U, *net.J, net.V.T]
    return [KernelPair(*empirical_kernels(W.detach().cpu().to(torch.float64))) for W in weights]


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
    (`fct.exp_view.tuning_view`); and the empirical kernels of the trained weights."""
    info, rate = _overlaps(net, test, n_bins, kind, n_shuffles, seed)
    return MemberView(data_kernels(net, lattice.x, spaces), info, rate, weight_kernels(net))


def ensemble_views(
    net: EnsembleMLP,
    lattice: Dataset,
    test: Dataset,
    spaces: Sequence[LayerSpace],
    n_bins: int,
    kind: str,
    n_shuffles: int,
    seed: int,
) -> list[MemberView]:
    """`member_view` of every member of `net`. Member b shuffles its MIR null with seed
    `seed + b`, so the shuffle noise differs between members and averages down in
    `average_views`. TypeError unless `net` is an `EnsembleMLP`."""
    if not isinstance(net, EnsembleMLP):
        raise TypeError(f"ensemble_views needs an EnsembleMLP, got {type(net).__name__}")
    return [
        member_view(net.member(b), lattice, test, spaces, n_bins, kind, n_shuffles, seed + b)
        for b in range(net.B)
    ]


def _mean(tensors: Sequence[Tensor]) -> Tensor:
    return torch.stack(list(tensors)).mean(0)


def average_views(views: Sequence[MemberView]) -> MemberView:
    """The elementwise mean of members' views, in unit-index space (members share their
    embeddings, so unit i is at the same position in every one).

    For the overlaps this is the mean over members of each member's own overlap matrix, i.e.
    of products of that member's informations (rates): a second moment over members, not the
    overlap of the mean tuning. For the "corrected" MIR each member's informations are clamped
    at zero before the products, which biases the average upward."""
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
    """Every member's outputs on the dataset rows `indices`, with the inputs and targets, all on
    the CPU. The inputs are moved to the net's device and dtype."""
    idx = torch.as_tensor(indices).cpu()
    if idx.numel() == 0 or idx.is_floating_point() or idx.dtype == torch.bool:
        raise ValueError(f"indices must be a non-empty list of integers, got {idx}")
    idx = idx.long()
    n_rows = len(dataset.x)
    if idx.ndim != 1 or idx.min() < 0 or idx.max() >= n_rows:
        raise ValueError(f"indices must be a 1-D list of rows in [0, {n_rows}), got {idx}")
    rows = idx.to(dataset.x.device)
    with torch.no_grad():
        _, Z = net(dataset.x[rows].to(net.device, net.dtype))  # (B, n, D_out)
    return EnsembleExamples(idx, dataset.x[rows].cpu(), dataset.y[rows].cpu(), Z.cpu())
