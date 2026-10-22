"""Spatial embeddings and factorised periodic Gaussian weight kernels.

Every population (the input, each hidden layer, the output) gets a position in R^d for each of
its units, on a torus [0, box)^d. The weight prior of layer l is K^l = K^post (x) K^pre with a
Gaussian kernel on the torus for each factor; the penalty matrix is M = K^{-1}, built per
factor (inverse of a Kronecker product is the Kronecker product of the inverses), so the
(n_post n_pre)^2 matrix is never formed.

Everything here is pure: positions and kernels are float64 CPU tensors, so the inversions are
accurate and independent of the training device.

Layer l = 0..L maps population l to population l + 1, where population 0 is the input (size m),
populations 1..L are the hidden layers (size N) and population L + 1 is the output (size
n_out). This matches `fct.model.MLP`: U (N, D_in) is layer 0, J_1..J_{L-1} are the hidden to
hidden layers, and the readout V (N, D_out) is layer L. Consecutive layers share a population
object, so the post factor of layer l is the very same object as the pre factor of layer l + 1.
"""

from dataclasses import dataclass
from typing import Annotated, Literal, Self

import torch
from pydantic import Field, model_validator

from fct.config import PosFloat, PosInt, StrictModel

DTYPE = torch.float64
RelEig = Annotated[float, Field(gt=0, lt=1)]


# --------------------------------------------------------------------------- config


class EmbeddingSpec(StrictModel):
    """How to place a population's units. `grid` is a regular lattice with `shape[k]` points
    along axis k (the product must equal the population size); for d = 1 `shape` may be omitted
    and the n points are evenly spaced. `random` is uniform in the box, seeded by `seed`."""

    kind: Literal["grid", "random"] = "grid"
    shape: list[PosInt] | None = None
    seed: int = 0


class KernelSpec(StrictModel):
    length_scale: PosFloat = 0.1  # in units of the box
    amplitude: PosFloat = 1.0


class PopulationSpec(StrictModel):
    embedding: EmbeddingSpec = EmbeddingSpec()
    kernel: KernelSpec = KernelSpec()


class SpatialConfig(StrictModel):
    """Spatial settings. All hidden layers share the `hidden` spec; a random hidden embedding
    uses seed `hidden.embedding.seed + k` for hidden layer k = 0..L-1 so the layers differ.
    `min_rel_eig` is the eigenvalue floor (relative to the largest) of the penalty matrices."""

    d: PosInt = 1
    box: PosFloat = 1.0
    input: PopulationSpec = PopulationSpec()
    hidden: PopulationSpec = PopulationSpec()
    output: PopulationSpec = PopulationSpec()
    min_rel_eig: RelEig = 1e-6

    @model_validator(mode="after")
    def _grid_shapes_match_d(self) -> Self:
        for name in ("input", "hidden", "output"):
            shape = getattr(self, name).embedding.shape
            if shape is not None and len(shape) != self.d:
                raise ValueError(
                    f"{name}.embedding.shape has {len(shape)} entries but d = {self.d}"
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


def make_embedding(
    spec: EmbeddingSpec, n: int, d: int, box: float, seed_offset: int = 0
) -> Embedding:
    """Positions for n units. `seed_offset` is added to `spec.seed` (random kind only)."""
    if spec.kind == "random":
        gen = torch.Generator().manual_seed(spec.seed + seed_offset)
        return Embedding(torch.rand(n, d, generator=gen, dtype=DTYPE) * box, box)
    shape = spec.shape
    if shape is None:
        if d != 1:
            raise ValueError(f"a grid embedding in d = {d} needs an explicit shape")
        shape = [n]
    if len(shape) != d:
        raise ValueError(f"grid shape {shape} has {len(shape)} entries but d = {d}")
    if torch.tensor(shape).prod().item() != n:
        raise ValueError(f"grid shape {shape} must multiply to the population size {n}")
    return Embedding(_grid_positions(shape, box), box)


# --------------------------------------------------------------------------- kernels


def torus_distance(x: torch.Tensor, y: torch.Tensor, box: float) -> torch.Tensor:
    """Euclidean distance of the per-axis minimum images: (n, d), (m, d) -> (n, m)."""
    delta = (x[:, None, :] - y[None, :, :]).abs() % box
    return torch.minimum(delta, box - delta).norm(dim=-1)


def periodic_gaussian_kernel(
    x_post: torch.Tensor, x_pre: torch.Tensor, length_scale: float, box: float, amplitude: float
) -> torch.Tensor:
    """K_ij = amplitude * exp(-d_torus(x_i, x_j)^2 / (2 length_scale^2)), shape (n_post, n_pre)."""
    dist = torus_distance(x_post, x_pre, box)
    return amplitude * torch.exp(-dist.square() / (2 * length_scale**2))


# --------------------------------------------------------------------------- penalty


@dataclass(frozen=True)
class FloorReport:
    """What the eigenvalue floor did: `n_negative` of the floored eigenvalues were below zero
    (the minimum-image Gaussian is not positive definite when the length scale is a sizeable
    fraction of the box), `cond_before` is the raw max/min eigenvalue (inf if the smallest is
    not positive), `cond_after` the same after flooring."""

    n_floored: int
    n_negative: int
    cond_before: float
    cond_after: float


def penalty_matrix(K: torch.Tensor, min_rel_eig: float) -> tuple[torch.Tensor, FloorReport]:
    """M ~ K^{-1} from the symmetric eigendecomposition, eigenvalues floored at
    min_rel_eig * (largest eigenvalue). The floor is the only approximation: a Gaussian
    kernel is numerically singular, and without it M would amplify round-off."""
    if not 0 < min_rel_eig < 1:
        raise ValueError(f"min_rel_eig must lie in (0, 1), got {min_rel_eig}")
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be square, got shape {tuple(K.shape)}")
    if not torch.allclose(K, K.T, rtol=0, atol=1e-10 * K.abs().max().item()):
        raise ValueError("K must be symmetric")
    eigvals, eigvecs = torch.linalg.eigh(K)
    top, bottom = eigvals[-1].item(), eigvals[0].item()
    if top <= 0:
        raise ValueError("K needs a positive largest eigenvalue")
    floor = min_rel_eig * top
    floored = eigvals.clamp(min=floor)
    M = (eigvecs / floored) @ eigvecs.T
    report = FloorReport(
        n_floored=int((eigvals < floor).sum().item()),
        n_negative=int((eigvals < 0).sum().item()),
        cond_before=top / bottom if bottom > 0 else float("inf"),
        cond_after=top / floored[0].item(),
    )
    return M, report


# --------------------------------------------------------------------------- layers


@dataclass(frozen=True)
class Population:
    """One population's embedding, kernel K (n, n), penalty M (n, n) and floor report."""

    name: str
    embedding: Embedding
    kernel: KernelSpec
    K: torch.Tensor
    M: torch.Tensor
    report: FloorReport


@dataclass(frozen=True)
class LayerSpace:
    """Weight matrix l maps `pre` to `post`; its prior is K^post (x) K^pre."""

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
    name: str, spec: PopulationSpec, n: int, cfg: SpatialConfig, seed_offset: int = 0
) -> Population:
    emb = make_embedding(spec.embedding, n, cfg.d, cfg.box, seed_offset)
    K = periodic_gaussian_kernel(
        emb.positions, emb.positions, spec.kernel.length_scale, cfg.box, spec.kernel.amplitude
    )
    M, report = penalty_matrix(K, cfg.min_rel_eig)
    return Population(name, emb, spec.kernel, K, M, report)


def build_spaces(cfg: SpatialConfig, m: int, N: int, L: int, n_out: int) -> list[LayerSpace]:
    """The L + 1 layers' pre and post factors for a network with input size m, L hidden layers
    of N units and n_out outputs."""
    for name, size in {"m": m, "N": N, "L": L, "n_out": n_out}.items():
        if size < 1:
            raise ValueError(f"{name} must be >= 1, got {size}")
    pops = [_make_population("input", cfg.input, m, cfg)]
    pops += [
        _make_population(f"hidden {k + 1}", cfg.hidden, N, cfg, seed_offset=k) for k in range(L)
    ]
    pops.append(_make_population("output", cfg.output, n_out, cfg))
    return [LayerSpace(layer, pops[layer], pops[layer + 1]) for layer in range(L + 1)]
