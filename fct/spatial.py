"""Spatial embeddings and factorised periodic Gaussian weight kernels.

Every unit of every population (the input, each hidden layer, the output) gets a position in
R^d on the torus [0, box)^d. The weight prior of layer l is K^l = K^post (x) K^pre. Each factor
is a Gaussian kernel on the torus, and the penalty is M = K^{-1} built per factor (the inverse
of a Kronecker product is the Kronecker product of the inverses), so the (n_post n_pre)^2
matrix is never formed.

Populations and layers. Population 0 is the input (size m), populations 1..L are the hidden
layers ("hidden 1".."hidden L", size N) and population L + 1 is the output (size n_out). Layer
l = 0..L maps population l (pre) to population l + 1 (post), matching `fct.model.MLP`: U is
layer 0, J_1..J_{L-1} are the hidden to hidden layers, the readout is layer L. Consecutive
layers share a population object. Orientation: U and J are stored (post, pre), but the readout
`MLP.V` is stored (N, D_out) = (pre, post), so for layer L the weight in (post, pre)
orientation is `V.T`; `M_post` acts on its rows and `M_pre` on its columns.

The kernel. A Gaussian of the minimum-image distance is not positive definite on the torus
once length_scale exceeds roughly box / 10 (the default length_scale is 0.1 box), so the
kernel is the wrapped (periodised) Gaussian,
which is positive semi-definite for every length_scale:

    K_ij = amplitude * prod_axes W(delta_a) / W(0),  delta = x_i - x_j reduced to [-box/2, box/2],
    W(delta) = sum_k exp(-(delta + k box)^2 / (2 length_scale^2)),  |k| <= n_images.

It factorises over axes, has diagonal exactly `amplitude`, and reduces to the plain Gaussian
for length_scale << box. The image sum is cut at n_images = ceil(N_IMAGE_SIGMAS * length_scale
/ box) + 1, so every dropped image lies at least N_IMAGE_SIGMAS * length_scale + 1.5 box from
the point (the +1 is a margin) and contributes less than exp(-32) ~ 1e-14 of the central term.
The cut has to be that deep: the eigenvalues on a ring decay like
exp(-2 pi^2 length_scale^2 j^2 / box^2), so a shallower cut leaves negative eigenvalues.

length_scale = 0 is the limit of K as length_scale -> 0: K_ij = amplitude if x_i = x_j (mod box)
on every axis and 0 otherwise. "Equal" means delta reduces to exactly 0 in floating point, with
no tolerance, so positions shifted by a whole box coincide only when the arithmetic is exact
(e.g. a dyadic box). For distinct positions, which grid embeddings have by construction and
random ones almost surely, K = amplitude * I (an unstructured prior, the plain weight L2 up to
scale) and the penalty is I / amplitude; under `lambda_max`, K = lambda_max * I and
M = I / lambda_max. Positions from `make_embedding` lie in [0, box), so within a population only
the diagonal coincides.

The penalty. M comes from the symmetric eigendecomposition with eigenvalues floored at
`min_rel_eig` times the largest. A smooth kernel has many eigenvalues far below that (83%
of a 100-unit ring at the defaults), so the floor is not just a round-off guard: it caps the
prior's near-zero-variance directions, and so it defines M on that subspace. `FloorReport`
gives the floored fraction. A kernel with a clearly negative eigenvalue is an error, not
something to floor (`penalty_matrix` raises).

The whitened parametrisation (`SpatialConfig.parametrisation = "whitened"`). Instead of
penalising W with M, the net trains xi with W = S^post xi S^pre, S = K^{1/2} the symmetric PSD
square root (`whitening_matrix`: eigenvalues clamped at 0, no inversion, no floor), and the
penalty is the plain |xi|_F^2. Then Cov(vec W) = (K^post (x) K^pre) / weight_lambda exactly, for
every spectrum, where the penalty path only has it above the floor; and the prior's curvature is
the same in every direction of xi, so SGLD is not stiff. `Population.S` and `whitening_factors`
carry the factors; `fct.model` applies them.

Everything is pure float64 on the CPU, so inversions are accurate and independent of the
training device.
"""

import math
import numbers
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal, Self

import torch
from pydantic import Field, model_validator

from fct.schema import NonNegFloat, PosFloat, PosInt, StrictModel

DTYPE = torch.float64
# Dropped images lie beyond this many length scales; exp(-N_IMAGE_SIGMAS^2 / 2) = exp(-32) ~ 1e-14.
N_IMAGE_SIGMAS = 8
# Relative tolerance on the asymmetry of a kernel handed to penalty_matrix.
SYMMETRY_RTOL = 1e-10
# Eigenvalues above -NEGATIVE_RTOL * max count as round-off rather than as negative.
NEGATIVE_RTOL = 1e-10
# Populations per network are limited to this, so (seed, index) pairs map to distinct seeds.
SEED_STRIDE = 1000
RelEig = Annotated[float, Field(gt=0, lt=1)]


# --------------------------------------------------------------------------- config


class EmbeddingSpec(StrictModel):
    """How to place a population's units. `grid` is a regular lattice with `shape[k]` points
    along axis k (the product must equal the population size); for d = 1 `shape` may be omitted
    and the n points are evenly spaced. `random` is uniform in the box, seeded by the generator seed
    `seed * SEED_STRIDE + population index` (input 0, hidden k = k, output L + 1), so populations
    differ and no two (seed, index) pairs share a stream."""

    kind: Literal["grid", "random"] = "grid"
    shape: list[PosInt] | None = None
    seed: int = 0

    @model_validator(mode="after")
    def _shape_only_for_grid(self) -> Self:
        if self.kind == "random" and self.shape is not None:
            raise ValueError("a random embedding takes no shape")
        return self


class KernelSpec(StrictModel):
    # Same units as `box`. 0 gives the identity kernel (amplitude * I), the limit of the wrapped
    # Gaussian as length_scale -> 0: an unstructured prior on that population's factor.
    length_scale: NonNegFloat = 0.1
    amplitude: PosFloat = 1.0


class PopulationSpec(StrictModel):
    embedding: EmbeddingSpec = EmbeddingSpec()
    kernel: KernelSpec = KernelSpec()


class SpatialConfig(StrictModel):
    """Spatial settings. All hidden populations (hidden 1..L) share the `hidden` spec.
    `min_rel_eig` is the eigenvalue floor (relative to the largest) of the penalty matrices.
    A grid embedding needs a `shape` with d entries, except in d = 1 where it may be omitted."""

    d: PosInt = 1
    box: PosFloat = 1.0
    input: PopulationSpec = PopulationSpec()
    hidden: PopulationSpec = PopulationSpec()
    output: PopulationSpec = PopulationSpec()
    min_rel_eig: RelEig = 1e-6
    # When set, every population kernel is rescaled to this top eigenvalue (see `_make_population`)
    # and `kernel.amplitude` is not used, so it must stay 1; `Config` also requires weight_lambda = 1.
    lambda_max: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    # "penalty": the weights are the parameters and the loss carries tr(M W M W^T) (`min_rel_eig`
    # floors M). "whitened": the parameters are xi, W = S xi S with S = K^{1/2}, and the loss
    # carries |xi|^2 (see the module docstring); `min_rel_eig` is then unused. The field enters
    # the hash only as "whitened", so every penalty hash is unchanged.
    parametrisation: Literal["penalty", "whitened"] = "penalty"

    @model_validator(mode="after")
    def _amplitude_is_one_under_lambda_max(self) -> Self:
        if self.lambda_max is None:
            return self
        for name in ("input", "hidden", "output"):
            amplitude = getattr(self, name).kernel.amplitude
            if amplitude != 1.0:
                raise ValueError(
                    f"{name}.kernel.amplitude is {amplitude!r} but lambda_max is set, which "
                    "overwrites the amplitude; leave it at 1"
                )
        return self

    @model_validator(mode="after")
    def _grid_shapes_match_d(self) -> Self:
        for name in ("input", "hidden", "output"):
            emb = getattr(self, name).embedding
            if emb.kind == "grid" and emb.shape is None and self.d > 1:
                raise ValueError(f"{name}: a grid embedding in d = {self.d} needs a shape")
            if emb.shape is not None and len(emb.shape) != self.d:
                raise ValueError(
                    f"{name}.embedding.shape has {len(emb.shape)} entries but d = {self.d}"
                )
        return self


# --------------------------------------------------------------------------- embeddings


@dataclass(frozen=True)
class Embedding:
    """Positions of n units on the torus [0, box)^d: `positions` is (n, d), float64."""

    positions: torch.Tensor
    box: float


def _grid_positions(shape: list[int], box: float) -> torch.Tensor:
    axes = [torch.arange(s, dtype=DTYPE) * (box / s) for s in shape]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([g.reshape(-1) for g in mesh], dim=1)


def _grid_shape(spec: EmbeddingSpec, n: int, d: int) -> list[int]:
    """The lattice shape for n units in d dimensions, checked against the spec."""
    shape = spec.shape
    if shape is None:
        if d != 1:
            raise ValueError(f"a grid embedding in d = {d} needs an explicit shape")
        return [n]
    if len(shape) != d:
        raise ValueError(f"grid shape {shape} has {len(shape)} entries but d = {d}")
    if math.prod(shape) != n:
        raise ValueError(f"grid shape {shape} must multiply to the population size {n}")
    return shape


def make_embedding(spec: EmbeddingSpec, n: int, d: int, box: float, index: int = 0) -> Embedding:
    """Positions for n units; `index` is the population index (random kind only: it enters the
    generator seed `spec.seed * SEED_STRIDE + index`). The draw uses a local generator, so the
    global RNG is untouched."""
    if spec.kind == "random":
        if not 0 <= index < SEED_STRIDE:
            raise ValueError(f"population index must lie in [0, {SEED_STRIDE}), got {index}")
        gen = torch.Generator().manual_seed(spec.seed * SEED_STRIDE + index)
        return Embedding(torch.rand(n, d, generator=gen, dtype=DTYPE) * box, box)
    return Embedding(_grid_positions(_grid_shape(spec, n, d), box), box)


# --------------------------------------------------------------------------- kernels


def torus_distance(x: torch.Tensor, y: torch.Tensor, box: float) -> torch.Tensor:
    """Euclidean distance of the per-axis minimum images: (n, d), (m, d) -> (n, m)."""
    delta = (x[:, None, :] - y[None, :, :]).abs() % box
    return torch.minimum(delta, box - delta).norm(dim=-1)


def _wrapped_gaussian_1d(delta: torch.Tensor, length_scale: float, box: float) -> torch.Tensor:
    """W(delta) of the module docstring, elementwise; `delta` must lie in [0, box/2]."""
    n_images = math.ceil(N_IMAGE_SIGMAS * length_scale / box) + 1
    shifts = box * torch.arange(-n_images, n_images + 1, dtype=delta.dtype)
    return torch.exp(-((delta[..., None] + shifts) ** 2) / (2 * length_scale**2)).sum(-1)


def wrapped_gaussian_profile(
    delta: torch.Tensor, length_scale: float, box: float, amplitude: float
) -> torch.Tensor:
    """The kernel along one axis, amplitude * W(delta) / W(0), for any signed `delta`: K itself
    for d = 1, and K along a coordinate axis for d > 1. length_scale = 0 gives the limit of the
    profile: `amplitude` where delta reduces to exactly 0 (a whole number of boxes, in exact
    floating point arithmetic), 0 elsewhere."""
    # Reduce to [0, box/2] (W is even): the images are centred, and K_ij and K_ji sum the same
    # terms in the same order, so K is symmetric to the bit.
    delta = (delta - box * torch.round(delta / box)).abs()
    if length_scale == 0:  # the Gaussian formula would divide by zero
        return amplitude * (delta == 0).to(delta.dtype)
    zero = _wrapped_gaussian_1d(torch.zeros(1, dtype=delta.dtype), length_scale, box)
    return amplitude * _wrapped_gaussian_1d(delta, length_scale, box) / zero


def periodic_gaussian_kernel(
    x_post: torch.Tensor, x_pre: torch.Tensor, length_scale: float, box: float, amplitude: float
) -> torch.Tensor:
    """The wrapped Gaussian kernel of the module docstring, shape (n_post, n_pre). Positions
    may lie outside [0, box): only differences modulo `box` matter."""
    delta = x_post[:, None, :] - x_pre[None, :, :]  # (n_post, n_pre, d)
    per_axis = wrapped_gaussian_profile(delta, length_scale, box, 1.0)
    return amplitude * per_axis.prod(dim=-1)


# --------------------------------------------------------------------------- penalty


@dataclass(frozen=True)
class FloorReport:
    """What the eigenvalue floor did to one kernel (see the module docstring).

    n_floored: eigenvalues raised to the floor (all those below min_rel_eig * the largest).
    fraction_floored: n_floored over the number of eigenvalues.
    n_negative: eigenvalues below -NEGATIVE_RTOL * the largest (clearly negative, not round-off).
    cond_before, cond_after: largest over smallest eigenvalue before and after (inf if <= 0).
    floor: the eigenvalue floor, min_rel_eig times the largest eigenvalue (whether or not any
    eigenvalue was below it).
    """

    n_floored: int
    fraction_floored: float
    n_negative: int
    cond_before: float
    cond_after: float
    floor: float


def _check_kernel(K: torch.Tensor) -> None:
    """Raise unless K is a finite, square, symmetric matrix."""
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be square, got shape {tuple(K.shape)}")
    if K.dtype != DTYPE:
        raise ValueError(f"K must be {DTYPE}, got {K.dtype}")
    if not torch.isfinite(K).all():
        raise ValueError("K must be finite")
    if not torch.allclose(K, K.T, rtol=0, atol=SYMMETRY_RTOL * K.abs().max().item()):
        raise ValueError("K must be symmetric")


def _psd_eigh(K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The symmetric eigendecomposition of a kernel, checked to be a covariance: finite,
    square, symmetric, with a positive top eigenvalue and none clearly negative (below
    -NEGATIVE_RTOL times the largest; smaller negatives are round-off)."""
    _check_kernel(K)
    eigvals, eigvecs = torch.linalg.eigh(K)
    top = eigvals[-1].item()
    if top <= 0:
        raise ValueError("K needs a positive largest eigenvalue")
    n_negative = int((eigvals < -NEGATIVE_RTOL * top).sum().item())
    if n_negative:
        raise ValueError(
            f"K has {n_negative} clearly negative eigenvalues (below -{NEGATIVE_RTOL:g} times "
            "the largest); it is not a covariance"
        )
    return eigvals, eigvecs


def whitening_matrix(K: torch.Tensor) -> torch.Tensor:
    """S = K^{1/2}, the symmetric PSD square root from the eigendecomposition with eigenvalues
    clamped at 0 (round-off negatives of a smooth kernel; a clearly negative one raises). No
    inversion and no floor: S S = K to round-off, however small K's eigenvalues."""
    eigvals, eigvecs = _psd_eigh(K)
    return (eigvecs * eigvals.clamp(min=0).sqrt()) @ eigvecs.T


def _floored_eigh(
    K: torch.Tensor, min_rel_eig: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """K's eigenvalues, eigenvectors, the eigenvalues floored at min_rel_eig * (largest), and
    that floor: the spectrum of the kernel the penalty encodes, M^{-1}."""
    if not 0 < min_rel_eig < 1:
        raise ValueError(f"min_rel_eig must lie in (0, 1), got {min_rel_eig}")
    eigvals, eigvecs = _psd_eigh(K)
    floor = min_rel_eig * eigvals[-1].item()
    return eigvals, eigvecs, eigvals.clamp(min=floor), floor


def floored_square_root(K: torch.Tensor, min_rel_eig: float) -> torch.Tensor:
    """(M^{-1})^{1/2}, M = `penalty_matrix(K, min_rel_eig)`: the symmetric square root of the
    floored kernel the penalty encodes, which draws the penalty's prior. With nothing floored
    it is `whitening_matrix(K)` bit for bit (the same decomposition and the same eigenvalues)."""
    _, eigvecs, floored, _ = _floored_eigh(K, min_rel_eig)
    return (eigvecs * floored.sqrt()) @ eigvecs.T


def penalty_matrix(K: torch.Tensor, min_rel_eig: float) -> tuple[torch.Tensor, FloorReport]:
    """M ~ K^{-1} from the symmetric eigendecomposition with eigenvalues floored at
    min_rel_eig * (largest eigenvalue); returns M and the `FloorReport`."""
    eigvals, eigvecs, floored, floor = _floored_eigh(K, min_rel_eig)
    top, bottom = eigvals[-1].item(), eigvals[0].item()
    M = (eigvecs / floored) @ eigvecs.T
    n_floored = int((eigvals < floor).sum().item())
    report = FloorReport(
        n_floored=n_floored,
        fraction_floored=n_floored / len(eigvals),
        n_negative=0,  # `_psd_eigh` raises on any clearly negative eigenvalue
        cond_before=top / bottom if bottom > 0 else float("inf"),
        cond_after=top / floored[0].item(),
        floor=floor,
    )
    return M, report


# --------------------------------------------------------------------------- layers


@dataclass(frozen=True)
class Population:
    """One population's embedding, kernel K (n, n), penalty M (n, n) with its floor report,
    square root S = K^{1/2} (n, n) for the whitened parametrisation, and S_floored, the square
    root of the floored kernel M^{-1} that draws the penalty's prior. `scale` is the factor
    applied to the unit-amplitude kernel G under `lambda_max` (lambda_max / top eigenvalue of
    G); it is 1.0 when `lambda_max` is unset, where K carries the spec's amplitude instead."""

    name: str
    embedding: Embedding
    kernel: KernelSpec
    K: torch.Tensor
    M: torch.Tensor
    report: FloorReport
    S: torch.Tensor
    S_floored: torch.Tensor
    scale: float = 1.0


@dataclass(frozen=True)
class LayerSpace:
    """Weight matrix l maps `pre` to `post`; its prior is K^post (x) K^pre. The weight must be in
    (post, pre) orientation: for layer L that is `MLP.V.T`, since V is stored (pre, post)."""

    layer: int
    pre: Population
    post: Population

    @property
    def M_pre(self) -> torch.Tensor:
        return self.pre.M

    @property
    def M_post(self) -> torch.Tensor:
        return self.post.M


def _make_population(
    name: str, spec: PopulationSpec, n: int, cfg: SpatialConfig, index: int
) -> Population:
    """Population number `index` (0 input, 1..L hidden, L + 1 output), which enters the seed."""
    emb = make_embedding(spec.embedding, n, cfg.d, cfg.box, index=index)
    K = periodic_gaussian_kernel(
        emb.positions, emb.positions, spec.kernel.length_scale, cfg.box, spec.kernel.amplitude
    )
    scale = 1.0
    if cfg.lambda_max is not None:
        # Per population: the embedding (and so the top eigenvalue) differs between them.
        scale = cfg.lambda_max / torch.linalg.eigvalsh(K)[-1].item()
        K = scale * K
    M, report = penalty_matrix(K, cfg.min_rel_eig)
    S, S_floored = whitening_matrix(K), floored_square_root(K, cfg.min_rel_eig)
    return Population(name, emb, spec.kernel, K, M, report, S, S_floored, scale)


def build_spaces(cfg: SpatialConfig, m: int, N: int, L: int, n_out: int) -> list[LayerSpace]:
    """The L + 1 layers' pre and post factors for a network with input size m, L hidden layers
    of N units and n_out outputs."""
    for name, size in {"m": m, "N": N, "L": L, "n_out": n_out}.items():
        if size < 1:
            raise ValueError(f"{name} must be >= 1, got {size}")
    pops = [_make_population("input", cfg.input, m, cfg, index=0)]
    pops += [_make_population(f"hidden {k}", cfg.hidden, N, cfg, index=k) for k in range(1, L + 1)]
    pops.append(_make_population("output", cfg.output, n_out, cfg, index=L + 1))
    return [LayerSpace(layer, pops[layer], pops[layer + 1]) for layer in range(L + 1)]


# --------------------------------------------------------------------------- the prior on weights


def weight_quadratic_form(
    W: torch.Tensor, M_rows: torch.Tensor, M_cols: torch.Tensor
) -> torch.Tensor:
    """tr(M_rows W M_cols W^T) = sum_{i,a,j,b} W_ia (M_rows)_ij (M_cols)_ab W_jb for W (r, c);
    for W (..., r, c) with leading batch axes (an ensemble's members), one value per matrix.

    This is vec(W)^T (M_rows (x) M_cols) vec(W) with vec row-major, without forming the
    (r c, r c) Kronecker matrix. With identity factors it is exactly W.square().sum(): the
    products by 1 and sums with 0 are exact, so no round-off separates it from the L2 norm.
    Works in any dtype and on any device (the factors must match W's).
    """
    return ((M_rows @ W @ M_cols) * W).sum(dim=(-2, -1))


# --------------------------------------------------------------------------- kernels over samples


def _check_square(name: str, matrix: torch.Tensor) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {tuple(matrix.shape)}")


def data_kernel(psi: torch.Tensor, K_pre: torch.Tensor) -> torch.Tensor:
    """A_{mu nu} = psi_mu^T K_pre psi_nu / n_pre for psi (P, n_pre): the (P, P) data factor of
    the spatially indexed kernel Phi^{mu nu}_ij = K^post_ij A_{mu nu}, with the pre-synaptic
    index summed out against its prior kernel. With K_pre = I it is the plain psi psi^T / n_pre.
    psi and K_pre must share a dtype and a device."""
    if psi.ndim != 2:
        raise ValueError(f"psi must be (P, n_pre), got shape {tuple(psi.shape)}")
    _check_square("K_pre", K_pre)
    n_pre = psi.shape[1]
    if K_pre.shape[0] != n_pre:
        raise ValueError(f"psi has {n_pre} pre-synaptic units but K_pre is {tuple(K_pre.shape)}")
    return psi @ K_pre @ psi.T / n_pre


def latent_kernel(factors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """B_{mu nu} = z_mu . z_nu / k for latents given as `{name: (P,)}`: the (P, P) kernel a
    layer that is linear in the latents would have as its data kernel A (up to scale)."""
    if not factors:
        raise ValueError("no latents")
    Z = torch.stack(list(factors.values()), dim=1)
    return Z @ Z.T / Z.shape[1]


def spatial_kernel_pairs(
    A: torch.Tensor, K_post: torch.Tensor, trials: Sequence[int]
) -> torch.Tensor:
    """Phi^{mu nu}_ij = A_{mu nu} K_post_ij for every pair of the S samples `trials`, as an
    (S, S, n_post, n_post) tensor whose [s, t] block belongs to (trials[s], trials[t])."""
    _check_square("A", A)
    _check_square("K_post", K_post)
    P = A.shape[0]
    trials = list(trials)
    if not trials:
        raise ValueError("need a non-empty sequence of trial indices")
    # bool is an Integral too, but True as a sample index is a bug, not sample 1
    if any(isinstance(t, bool) or not isinstance(t, numbers.Integral) for t in trials):
        raise ValueError(f"trial indices must be integers, got {trials!r}")
    index = torch.as_tensor(trials, dtype=torch.long)
    if index.min() < 0 or index.max() >= P:
        raise ValueError(f"trial indices must lie in 0 .. {P - 1}, got {index.tolist()}")
    index = index.to(A.device)
    return A[index][:, index][:, :, None, None] * K_post[None, None]


def parameter_factors(spaces: list[LayerSpace]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """(M_rows, M_cols) of every `fct.model.MLP` weight in its stored orientation, keyed by the
    parameter name ("U", "J.0".."J.{L-2}", "V") of the L + 1 layer `spaces`.

    U and J are stored (post, pre), so their rows take M_post; the readout V is stored
    (pre, post), so its rows take M_pre. `weight_quadratic_form(param, *factors[name])` is
    then the layer's tr(M_post W M_pre W^T) with W in (post, pre) orientation.
    """
    return _factors_by_parameter(spaces, lambda pop: pop.M)


def whitening_factors(spaces: list[LayerSpace]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """(S_rows, S_cols) of every `fct.model.MLP` weight in its stored orientation, keyed like
    `parameter_factors`: W = S_rows xi S_cols is the layer's S^post xi S^pre for U and J, and
    for V (stored (pre, post)) its transpose, rows S^pre and columns S^post."""
    return _factors_by_parameter(spaces, lambda pop: pop.S)


def floored_whitening_factors(
    spaces: list[LayerSpace],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """(S_floored_rows, S_floored_cols) of every parameter, keyed and oriented like
    `whitening_factors`: S_floored_rows z S_floored_cols / sqrt(weight_lambda), z standard
    normal, is a draw from the penalty's (floored) prior."""
    return _factors_by_parameter(spaces, lambda pop: pop.S_floored)


def _factors_by_parameter(
    spaces: list[LayerSpace], of: Callable[[Population], torch.Tensor]
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """`of(population)` for the (rows, cols) populations of every parameter, by name."""
    L = len(spaces) - 1
    names = ["U", *[f"J.{k}" for k in range(L - 1)], "V"]
    factors = {
        name: (of(s.post), of(s.pre)) for name, s in zip(names[:-1], spaces[:-1], strict=True)
    }
    factors["V"] = (of(spaces[L].pre), of(spaces[L].post))
    return factors
