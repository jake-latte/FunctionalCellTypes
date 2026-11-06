"""Task specs and pure generators for the linear-teacher, object-action and what-where tasks.

Each task is a `TaskSpec` (input size, output size, config validation, generator) so that the
sizes the config validates, `task_dims` reports and the generator produces come from one place.
Generators return a `Dataset` carrying the ground-truth `factors`, their `groups` for
modularity scoring and the `extras` needed to draw further examples. All randomness comes from
a CPU generator seeded with `cfg.seed` and is moved to `device` afterwards, because CPU and
CUDA generators give different streams and the cache key does not include the device.

This module must not import `fct.config` at run time: the config imports `TASK_SPECS` to
validate itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from fct.config import DataConfig

ACTION_NAMES = ("stay", "up", "down", "left", "right")
# (d_row, d_col) of the four moves, in the column order of ACTION_NAMES[1:].
MOVE_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))

MOTIF_SIZE = 3  # motifs are 3x3 binary patterns
N_MACRO = 3  # the micro-grid holds N_MACRO x N_MACRO macro-cells
N_MICRO = MOTIF_SIZE * N_MACRO  # micro-grid side: 9, so 81 input pixels
MOTIF_LIBRARY_SIZE = 2 ** (MOTIF_SIZE**2) - 1  # 511 non-empty patterns


@dataclass(frozen=True)
class Dataset:
    x: Tensor  # (P, D_in)
    y: Tensor  # (P, D_out); position task: (P, d)
    Q: Tensor | None  # (D_in, d) orthonormal map for embedding="linear", else None
    period: float | None  # y.max() - y.min() at generation for "fourier", else None
    # Ground-truth task variables, for scoring what the hidden units encode. Each factor is a
    # (P,) tensor: int64 = categorical labels 0..C-1, floating = continuous. `groups` maps a
    # group name to the factor names it contains (a factor that is its own group is a singleton
    # group); `extras` holds tensors needed to draw further examples of the task.
    factors: dict[str, Tensor] | None = None
    groups: dict[str, list[str]] | None = None
    extras: dict[str, Tensor] | None = None


@dataclass(frozen=True)
class TaskSpec:
    input_dim: Callable[[DataConfig], int]
    output_dim: Callable[[DataConfig], int]
    # Raises ValueError for a config the task cannot generate (called by DataConfig and by
    # the generator, which also protects direct calls).
    validate: Callable[[DataConfig], None]
    generate: Callable[[DataConfig, torch.device, torch.dtype], Dataset]


def _singleton_groups(names: list[str]) -> dict[str, list[str]]:
    return {n: [n] for n in names}


def _validate_linear_teacher(cfg: DataConfig) -> None:
    if cfg.D_in <= cfg.n_latents:
        raise ValueError(
            f"D_in (the number of features m) must exceed n_latents; "
            f"got D_in={cfg.D_in}, n_latents={cfg.n_latents}"
        )


def _check_input_dim(cfg: DataConfig, expected: int) -> None:
    if cfg.D_in != expected:
        raise ValueError(
            f"task {cfg.task!r} fixes the input dimension at {expected}; set data.D_in={expected} "
            f"(got D_in={cfg.D_in})"
        )


def linear_teacher(cfg: DataConfig, device: torch.device, dtype: torch.dtype) -> Dataset:
    """x = Z W_X^T and y = Z W_Y^T for latents Z (P, k) iid uniform on [latent_min, latent_max].

    W_X and W_Y are (m, k) with iid N(0, 1/k) entries, fixed by the seed; m = cfg.D_in is the
    number of features of both x and y. Every latent is its own factor and its own group.
    """
    _validate_linear_teacher(cfg)
    k, m, lo, hi = cfg.n_latents, cfg.D_in, cfg.latent_min, cfg.latent_max
    gen = torch.Generator().manual_seed(cfg.seed)
    # With the default [-1, 1] this is -1 + 2 * rand, bit for bit the symmetric 2 * rand - 1.
    z = lo + (hi - lo) * torch.rand((cfg.P, k), dtype=dtype, generator=gen)
    w_x = torch.randn((m, k), dtype=dtype, generator=gen) / k**0.5
    w_y = torch.randn((m, k), dtype=dtype, generator=gen) / k**0.5
    z, w_x, w_y = z.to(device), w_x.to(device), w_y.to(device)
    names = [f"z{i}" for i in range(k)]
    return Dataset(
        x=z @ w_x.T,
        y=z @ w_y.T,
        Q=None,
        period=None,
        factors={n: z[:, i].contiguous() for i, n in enumerate(names)},
        groups=_singleton_groups(names),
        extras={"W_X": w_x, "W_Y": w_y},
    )


def _min_distance_to_objects(objects: Tensor) -> Tensor:
    """(P, G, G) Manhattan distance from every cell to the nearest object of that sample.

    Samples without objects get a distance larger than any real one (2G), so callers must
    test for emptiness separately. Memory is P * G**4 entries: fine for the small grids used.
    """
    P, G, _ = objects.shape
    rows, cols = torch.meshgrid(torch.arange(G), torch.arange(G), indexing="ij")
    rows, cols = rows.flatten().to(objects.device), cols.flatten().to(objects.device)
    between = (rows[:, None] - rows[None]).abs() + (cols[:, None] - cols[None]).abs()  # (G², G²)
    far = 2 * G
    dist = torch.where(objects.reshape(P, 1, G * G), between[None], far)
    return dist.amin(dim=2).reshape(P, G, G)


def optimal_actions(agent_rc: Tensor, objects: Tensor) -> Tensor:
    """Optimal action of an agent on a G x G grid, as (P, 5) float32 columns
    [stay, up, down, left, right] (up = row-1, down = row+1, left = col-1, right = col+1).

    `agent_rc` is (P, 2) integer (row, col); `objects` is (P, G, G) bool. With Manhattan
    distance: stay = 1 iff there are no objects or the agent stands on one; otherwise stay = 0
    and each move is 1 iff it stays on the grid and lowers the distance to the nearest object
    by exactly one. Several equally good moves are all marked (diagonal and tied objects).
    """
    P, G, _ = objects.shape
    dist = _min_distance_to_objects(objects)
    batch = torch.arange(P, device=objects.device)
    here = dist[batch, agent_rc[:, 0], agent_rc[:, 1]]
    stay = ~objects.flatten(1).any(dim=1) | (here == 0)
    moves = []
    for d_row, d_col in MOVE_STEPS:
        row, col = agent_rc[:, 0] + d_row, agent_rc[:, 1] + d_col
        inside = (row >= 0) & (row < G) & (col >= 0) & (col < G)
        # Clamp so the lookup is valid; out-of-grid moves are masked by `inside`.
        there = dist[batch, row.clamp(0, G - 1), col.clamp(0, G - 1)]
        moves.append(inside & ~stay & (there == here - 1))
    return torch.stack([stay, *moves], dim=1).to(torch.float32)


def _object_action_input_dim(cfg: DataConfig) -> int:
    return 2 * cfg.grid_size**2


def _validate_object_action(cfg: DataConfig) -> None:
    cells = cfg.grid_size**2
    if cfg.max_objects > cells:
        raise ValueError(
            f"max_objects={cfg.max_objects} exceeds the {cells} cells of a "
            f"{cfg.grid_size}x{cfg.grid_size} grid"
        )
    _check_input_dim(cfg, _object_action_input_dim(cfg))


def _random_object_maps(cfg: DataConfig, gen: torch.Generator) -> Tensor:
    """(P, G*G) bool: a uniform number in 0..A of distinct cells per sample, chosen uniformly.

    Ranking iid uniform scores and keeping the n lowest ranks selects a uniformly random
    n-subset of the cells.
    """
    n_cells = cfg.grid_size**2
    n_objects = torch.randint(0, cfg.max_objects + 1, (cfg.P,), generator=gen)
    ranks = torch.rand((cfg.P, n_cells), generator=gen).argsort(dim=1).argsort(dim=1)
    return ranks < n_objects[:, None]


def object_action(cfg: DataConfig, device: torch.device, dtype: torch.dtype) -> Dataset:
    """An agent on a G x G grid (cells row-major) with up to A objects; y is the optimal action.

    The agent's cell is uniform, the object count is uniform on 0..A, and object cells are
    distinct and drawn from all G*G cells, so an object can sit under the agent. x is the
    one-hot agent cell concatenated with the binary object map, shape (P, 2*G*G).
    """
    _validate_object_action(cfg)
    G = cfg.grid_size
    gen = torch.Generator().manual_seed(cfg.seed)
    agent = torch.randint(0, G * G, (cfg.P,), generator=gen)
    objects = _random_object_maps(cfg, gen)
    agent, objects = agent.to(device), objects.to(device)

    agent_rc = torch.stack([agent // G, agent % G], dim=1)
    actions = optimal_actions(agent_rc, objects.reshape(cfg.P, G, G))
    onehot = torch.nn.functional.one_hot(agent, G * G)
    factors = {"position": agent}
    factors |= {f"object_{i}": objects[:, i].long() for i in range(G * G)}
    factors |= {f"action_{n}": actions[:, j].long() for j, n in enumerate(ACTION_NAMES)}
    return Dataset(
        x=torch.cat([onehot, objects], dim=1).to(dtype),
        y=actions.to(dtype),
        Q=None,
        period=None,
        factors=factors,
        groups={
            "position": ["position"],
            "objects": [f"object_{i}" for i in range(G * G)],
            "actions": [f"action_{n}" for n in ACTION_NAMES],
        },
        extras={},
    )


def _motif_library(device: torch.device) -> Tensor:
    """(511, 9) every non-empty binary 3x3 pattern: bit i of the code (least significant
    first) is cell i of the row-major pattern, for codes 1..511."""
    codes = torch.arange(1, MOTIF_LIBRARY_SIZE + 1, device=device)
    return ((codes[:, None] >> torch.arange(MOTIF_SIZE**2, device=device)) & 1).to(torch.int64)


def _place_motifs(where: Tensor, motifs: Tensor) -> Tensor:
    """(P, 81) micro-grids: motif (P, 9) copied into macro-cell `where` (P,), zero elsewhere."""
    macro = torch.nn.functional.one_hot(where, N_MACRO**2).reshape(-1, N_MACRO, N_MACRO)
    patterns = motifs.reshape(-1, MOTIF_SIZE, MOTIF_SIZE)
    # indices [sample, macro row R, pixel row r, macro col C, pixel col c]: flattening them in
    # this order gives the row-major micro-grid with pixel (3R + r, 3C + c).
    grid = torch.einsum("pR,pC,prc->pRrCc", macro.sum(2), macro.sum(1), patterns)
    return grid.reshape(-1, N_MICRO**2)


def _validate_what_where(cfg: DataConfig) -> None:
    if cfg.n_motifs > MOTIF_LIBRARY_SIZE:
        raise ValueError(
            f"n_motifs={cfg.n_motifs} exceeds the {MOTIF_LIBRARY_SIZE} non-empty 3x3 binary "
            "patterns"
        )
    _check_input_dim(cfg, N_MICRO**2)


def what_where(cfg: DataConfig, device: torch.device, dtype: torch.dtype) -> Dataset:
    """A 3x3 motif placed in one of the 3x3 macro-cells of a 9x9 micro-grid.

    The motif library is all 2^9 - 1 = 511 non-empty binary 3x3 patterns (the all-on pattern
    included); the `n_motifs` motifs in use are chosen uniformly without replacement. `where`
    (macro-cell, 9 values, row-major) and `what` (motif index) are drawn independently and
    uniformly. y is the one-hot `where` followed by the one-hot `what`, shape (P, 9 + m).
    """
    _validate_what_where(cfg)
    m = cfg.n_motifs
    gen = torch.Generator().manual_seed(cfg.seed)
    chosen = torch.randperm(MOTIF_LIBRARY_SIZE, generator=gen)[:m]
    where = torch.randint(0, N_MACRO**2, (cfg.P,), generator=gen)
    what = torch.randint(0, m, (cfg.P,), generator=gen)
    chosen, where, what = chosen.to(device), where.to(device), what.to(device)

    motifs = _motif_library(device)[chosen]
    onehot = torch.nn.functional.one_hot
    return Dataset(
        x=_place_motifs(where, motifs[what]).to(dtype),
        y=torch.cat([onehot(where, N_MACRO**2), onehot(what, m)], dim=1).to(dtype),
        Q=None,
        period=None,
        factors={"where": where, "what": what},
        groups=_singleton_groups(["where", "what"]),
        extras={"motifs": motifs.to(dtype)},
    )


TASK_SPECS: dict[str, TaskSpec] = {
    "linear_teacher": TaskSpec(
        input_dim=lambda cfg: cfg.D_in,
        output_dim=lambda cfg: cfg.D_in,
        validate=_validate_linear_teacher,
        generate=linear_teacher,
    ),
    "object_action": TaskSpec(
        input_dim=_object_action_input_dim,
        output_dim=lambda cfg: len(ACTION_NAMES),
        validate=_validate_object_action,
        generate=object_action,
    ),
    "what_where": TaskSpec(
        input_dim=lambda cfg: N_MICRO**2,
        output_dim=lambda cfg: N_MACRO**2 + cfg.n_motifs,
        validate=_validate_what_where,
        generate=what_where,
    ),
}


def _redefine_linear_teacher(cfg: DataConfig, new: Dataset, reference: Dataset) -> Dataset:
    z = torch.stack([new.factors[f"z{i}"] for i in range(cfg.n_latents)], dim=1)
    w_x, w_y = reference.extras["W_X"], reference.extras["W_Y"]
    return Dataset(
        x=z @ w_x.T,
        y=z @ w_y.T,
        Q=None,
        period=None,
        factors=new.factors,
        groups=new.groups,
        extras=reference.extras,
    )


def _redefine_what_where(cfg: DataConfig, new: Dataset, reference: Dataset) -> Dataset:
    motifs = reference.extras["motifs"]
    x = _place_motifs(new.factors["where"], motifs[new.factors["what"]]).to(new.x.dtype)
    return Dataset(
        x=x,
        y=new.y,
        Q=None,
        period=None,
        factors=new.factors,
        groups=new.groups,
        extras=reference.extras,
    )


# object_action has no task definition beyond the config: a new seed alone gives new samples.
_REDEFINE: dict[str, Callable[[DataConfig, Dataset, Dataset], Dataset]] = {
    "linear_teacher": _redefine_linear_teacher,
    "object_action": lambda cfg, new, reference: new,
    "what_where": _redefine_what_where,
}


def resample(
    cfg: DataConfig, reference: Dataset, seed: int, device: torch.device, dtype: torch.dtype
) -> Dataset:
    """Fresh samples of the task `reference` was generated from, drawn with data seed `seed`.

    A generator draws the task definition (the linear teacher's W_X and W_Y, the what-where
    motif set) from the same seeded stream as the samples, so generating with a new seed alone
    would be a different task. The new samples are re-expressed with the definition held in
    `reference.extras`, so they are held-out data of the same task. Pure: generator outputs
    are unchanged and nothing is read or written on disk.
    """
    if cfg.task not in _REDEFINE:
        raise ValueError(f"resample is defined for tasks {sorted(_REDEFINE)}, not {cfg.task!r}")
    if seed == cfg.seed:
        raise ValueError(f"seed must differ from the reference data seed {cfg.seed}")
    new = TASK_SPECS[cfg.task].generate(cfg.model_copy(update={"seed": seed}), device, dtype)
    return _REDEFINE[cfg.task](cfg, new, reference)


def _teacher_maps(cfg: DataConfig, reference: Dataset) -> tuple[Tensor, Tensor]:
    """W_X and W_Y of `reference`, each checked to be (D_in, n_latents)."""
    extras = reference.extras or {}
    missing = [key for key in ("W_X", "W_Y") if key not in extras]
    if missing:
        raise ValueError(f"reference.extras lacks the teacher maps {missing}")
    expected = (cfg.D_in, cfg.n_latents)
    shapes = {key: tuple(extras[key].shape) for key in ("W_X", "W_Y")}
    if any(shape != expected for shape in shapes.values()):
        raise ValueError(
            f"teacher maps must have shape (D_in, n_latents) = {expected}; got {shapes}"
        )
    return extras["W_X"], extras["W_Y"]


def latent_lattice_dataset(cfg: DataConfig, reference: Dataset) -> Dataset:
    """One sample per point of the lattice {lo, (lo + hi) / 2, hi}^k of a linear teacher's
    latents.

    k = cfg.n_latents, lo = cfg.latent_min and hi = cfg.latent_max, giving 3^k rows. Row i is
    the i-th element of the Cartesian product of (lo, mid, hi) taken k times: z0 varies
    slowest, the last latent fastest, so row i is reproducible from i alone (digit j of i in
    base 3, most significant first, selects latent j's value from (lo, mid, hi)). x and y use the W_X and W_Y in
    `reference.extras`, so the points are held-out data of the same teacher. Pure: nothing is
    read or written on disk.

    `cfg` must be the DataConfig `reference` was generated from. Its D_in and n_latents are
    checked against the shapes of the maps; `latent_min` and `latent_max` cannot be verified
    from the reference and are taken on trust. The maps are assumed to share `reference.x`'s dtype and
    device, which the result inherits.
    """
    if cfg.task != "linear_teacher":
        raise ValueError(
            f"latent_lattice_dataset is defined for task 'linear_teacher', not {cfg.task!r}"
        )
    w_x, w_y = _teacher_maps(cfg, reference)
    k, lo, hi = cfg.n_latents, cfg.latent_min, cfg.latent_max
    values = torch.tensor(
        [lo, (lo + hi) / 2, hi], dtype=reference.x.dtype, device=reference.x.device
    )
    z = torch.cartesian_prod(*[values] * k).reshape(-1, k)  # reshape: k = 1 yields a 1-D tensor
    names = [f"z{i}" for i in range(k)]
    return Dataset(
        x=z @ w_x.T,
        y=z @ w_y.T,
        Q=None,
        period=None,
        factors={n: z[:, i].contiguous() for i, n in enumerate(names)},
        groups=_singleton_groups(names),
        extras=reference.extras,
    )
