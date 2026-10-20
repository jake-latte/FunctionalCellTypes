"""Pure generators for the linear-teacher, object-action and what-where tasks.

Each generator returns a `TaskData`: the inputs `x`, the targets `y`, the ground-truth
`factors` (int64 = categorical labels, floating = continuous), how those factors are `groups`
for modularity scoring, and `extras` needed to draw further examples. All randomness comes
from a CPU generator seeded with `cfg.seed` and is moved to `device` afterwards, because CPU
and CUDA generators give different streams and the cache key does not include the device.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from fct.config import DataConfig

ACTION_NAMES = ("stay", "up", "down", "left", "right")
# (d_row, d_col) of the four moves, in the column order of ACTION_NAMES[1:].
MOVE_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))

MOTIF_SIZE = 3  # motifs are 3x3 binary patterns
N_MACRO = 3  # the micro-grid holds N_MACRO x N_MACRO macro-cells
N_MICRO = MOTIF_SIZE * N_MACRO  # micro-grid side: 9, so 81 input pixels
MOTIF_LIBRARY_SIZE = 2 ** (MOTIF_SIZE**2) - 1  # 511 non-empty patterns


@dataclass(frozen=True)
class TaskData:
    x: Tensor
    y: Tensor
    factors: dict[str, Tensor]
    groups: dict[str, list[str]]
    extras: dict[str, Tensor]


def _singleton_groups(names: list[str]) -> dict[str, list[str]]:
    return {n: [n] for n in names}


def linear_teacher(cfg: DataConfig, device: torch.device, dtype: torch.dtype) -> TaskData:
    """x = Z W_X^T and y = Z W_Y^T for latents Z (P, k) iid uniform on [-a, a].

    W_X and W_Y are (m, k) with iid N(0, 1/k) entries, fixed by the seed; m = cfg.D_in is the
    number of features of both x and y. Every latent is its own factor and its own group.
    """
    k, m, a = cfg.n_latents, cfg.D_in, cfg.latent_range
    gen = torch.Generator().manual_seed(cfg.seed)
    z = a * (2 * torch.rand((cfg.P, k), dtype=dtype, generator=gen) - 1)
    w_x = torch.randn((m, k), dtype=dtype, generator=gen) / k**0.5
    w_y = torch.randn((m, k), dtype=dtype, generator=gen) / k**0.5
    z, w_x, w_y = z.to(device), w_x.to(device), w_y.to(device)
    names = [f"z{i}" for i in range(k)]
    return TaskData(
        x=z @ w_x.T,
        y=z @ w_y.T,
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


def _random_object_maps(cfg: DataConfig, gen: torch.Generator) -> Tensor:
    """(P, G*G) bool: a uniform number in 0..A of distinct cells per sample, chosen uniformly.

    Ranking iid uniform scores and keeping the n lowest ranks selects a uniformly random
    n-subset of the cells.
    """
    n_cells = cfg.grid_size**2
    n_objects = torch.randint(0, cfg.max_objects + 1, (cfg.P,), generator=gen)
    ranks = torch.rand((cfg.P, n_cells), generator=gen).argsort(dim=1).argsort(dim=1)
    return ranks < n_objects[:, None]


def object_action(cfg: DataConfig, device: torch.device, dtype: torch.dtype) -> TaskData:
    """An agent on a G x G grid (cells row-major) with up to A objects; y is the optimal action.

    The agent's cell is uniform, the object count is uniform on 0..A, and object cells are
    distinct and drawn from all G*G cells, so an object can sit under the agent. x is the
    one-hot agent cell concatenated with the binary object map, shape (P, 2*G*G).
    """
    G, A = cfg.grid_size, cfg.max_objects
    if A > G * G:
        raise ValueError(f"max_objects={A} exceeds the {G * G} cells of a {G}x{G} grid")
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
    return TaskData(
        x=torch.cat([onehot, objects], dim=1).to(dtype),
        y=actions.to(dtype),
        factors=factors,
        groups={
            "position": ["position"],
            "objects": [f"object_{i}" for i in range(G * G)],
            "actions": [f"action_{n}" for n in ACTION_NAMES],
        },
        extras={},
    )


def _motif_library(device: torch.device) -> Tensor:
    """(511, 9) every non-empty binary 3x3 pattern (row-major bits of 1..511)."""
    codes = torch.arange(1, MOTIF_LIBRARY_SIZE + 1, device=device)
    return (codes[:, None] >> torch.arange(MOTIF_SIZE**2, device=device) & 1).to(torch.int64)


def _place_motifs(where: Tensor, motifs: Tensor) -> Tensor:
    """(P, 81) micro-grids: motif (P, 9) copied into macro-cell `where` (P,), zero elsewhere."""
    macro = torch.nn.functional.one_hot(where, N_MACRO**2).reshape(-1, N_MACRO, N_MACRO)
    patterns = motifs.reshape(-1, MOTIF_SIZE, MOTIF_SIZE)
    # indices [sample, macro row R, pixel row r, macro col C, pixel col c]: flattening them in
    # this order gives the row-major micro-grid with pixel (3R + r, 3C + c).
    grid = torch.einsum("pR,pC,prc->pRrCc", macro.sum(2), macro.sum(1), patterns)
    return grid.reshape(-1, N_MICRO**2)


def what_where(cfg: DataConfig, device: torch.device, dtype: torch.dtype) -> TaskData:
    """A 3x3 motif placed in one of the 3x3 macro-cells of a 9x9 micro-grid.

    The motif library is all 2^9 - 1 = 511 non-empty binary 3x3 patterns (the all-on pattern
    included); the `n_motifs` motifs in use are chosen uniformly without replacement. `where`
    (macro-cell, 9 values, row-major) and `what` (motif index) are drawn independently and
    uniformly. y is the one-hot `where` followed by the one-hot `what`, shape (P, 9 + m).
    """
    m = cfg.n_motifs
    if m > MOTIF_LIBRARY_SIZE:
        raise ValueError(
            f"n_motifs={m} exceeds the {MOTIF_LIBRARY_SIZE} non-empty 3x3 binary patterns"
        )
    gen = torch.Generator().manual_seed(cfg.seed)
    chosen = torch.randperm(MOTIF_LIBRARY_SIZE, generator=gen)[:m]
    where = torch.randint(0, N_MACRO**2, (cfg.P,), generator=gen)
    what = torch.randint(0, m, (cfg.P,), generator=gen)
    chosen, where, what = chosen.to(device), where.to(device), what.to(device)

    motifs = _motif_library(device)[chosen]
    onehot = torch.nn.functional.one_hot
    return TaskData(
        x=_place_motifs(where, motifs[what]).to(dtype),
        y=torch.cat([onehot(where, N_MACRO**2), onehot(what, m)], dim=1).to(dtype),
        factors={"where": where, "what": what},
        groups=_singleton_groups(["where", "what"]),
        extras={"motifs": motifs.to(dtype)},
    )


def task_input_dim(cfg: DataConfig) -> int | None:
    """The input dimension a task fixes by itself, or None where D_in is a free choice."""
    if cfg.task == "object_action":
        return 2 * cfg.grid_size**2
    if cfg.task == "what_where":
        return N_MICRO**2
    return None


def task_output_dim(cfg: DataConfig) -> int:
    """The output dimension of every non-position task."""
    if cfg.task == "linear_teacher":
        return cfg.D_in
    if cfg.task == "object_action":
        return len(ACTION_NAMES)
    if cfg.task == "what_where":
        return N_MACRO**2 + cfg.n_motifs
    raise ValueError(f"task {cfg.task!r} has no generator in fct.tasks")


GENERATORS = {
    "linear_teacher": linear_teacher,
    "object_action": object_action,
    "what_where": what_where,
}
