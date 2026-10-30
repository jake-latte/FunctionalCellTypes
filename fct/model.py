"""The mean-field-scaled MLP, ensembles of it, and their seeded constructor."""

import hashlib
import math
from collections.abc import Sequence

import torch

from fct.activations import Activation, get_phi
from fct.config import NetConfig


class MLP(torch.nn.Module):
    """L-layer network storing pre-activations H, with a 1/N linear readout of the last
    layer's post-activations, Z = V^T phi(h^L) / N.

    Parameters are U (N, D_in), J_1..J_{L-1} (N, N) and V (N, D_out), all standard normal;
    the 1/sqrt(D_in), g/sqrt(N) and 1/N factors live in `forward`.
    """

    B = 1  # a single network: the ensemble size `fct.train` checks against `net.B`

    def __init__(
        self,
        D_in: int,
        D_out: int,
        N: int,
        L: int,
        g: float,
        phi: Activation,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ):
        """generator: source of the initial weights (must live on `device`); None -> global RNG."""
        super().__init__()
        for name, size in {"D_in": D_in, "D_out": D_out, "N": N, "L": L}.items():
            if size < 1:
                raise ValueError(f"{name} must be >= 1, got {size}")
        self.D_in, self.D_out = D_in, D_out
        self.N, self.g = N, g
        self.L = L
        self.phi = phi

        def init(*shape: int) -> torch.nn.Parameter:
            w = torch.randn(shape, device=device, dtype=dtype, generator=generator)
            return torch.nn.Parameter(w)

        self.U = init(N, D_in)
        self.V = init(N, D_out)
        self.J = torch.nn.ParameterList([init(N, N) for _ in range(1, L)])

    # Derived from the parameters so they stay correct after net.to(...).
    @property
    def device(self) -> torch.device:
        return self.U.device

    @property
    def dtype(self) -> torch.dtype:
        return self.U.dtype

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (P, D_in) -> (H (P, L, N) pre-activations, Z (P, D_out) outputs)."""
        if x.ndim != 2 or x.shape[1] != self.D_in:
            raise ValueError(f"x must have shape (P, D_in={self.D_in}), got {tuple(x.shape)}")

        H = [(1 / math.sqrt(self.D_in)) * torch.einsum("id, pd -> pi", self.U, x)]
        for layer in range(self.L - 1):
            z_prev = self.phi(H[-1])
            J = self.J[layer]
            h_next = (self.g / math.sqrt(self.N)) * torch.einsum("ij, pj -> pi", J, z_prev)
            H.append(h_next)

        Z = (1 / self.N) * torch.einsum("id, pi -> pd", self.V, self.phi(H[-1]))  # P, D_out
        return torch.stack(H, dim=1), Z

    def weight_l2(self) -> torch.Tensor:
        """Sum of squares of every parameter."""
        return torch.stack([p.square().sum() for p in self.parameters()]).sum()


class EnsembleMLP(torch.nn.Module):
    """B independent `MLP`s batched over a leading member axis: U (B, N, D_in), J_k (B, N, N),
    V (B, N, D_out), with member b's weights at index b. The members share nothing but their
    sizes, g and phi; `forward` runs every member on the same inputs.

    Built from its members (`EnsembleMLP([mlp_0, ..., mlp_{B-1}])`, which copies their weights)
    and read back one at a time with `member(b)`, so every function written for a single `MLP`
    applies to a member unchanged. Parameter names match `MLP`'s ("U", "J.<k>", "V").
    """

    def __init__(self, members: Sequence[MLP]):
        super().__init__()
        if len(members) < 2:
            raise ValueError(f"an ensemble needs at least 2 members, got {len(members)}")
        first = members[0]
        for b, m in enumerate(members[1:], start=1):
            for name in ("D_in", "D_out", "N", "L", "g", "phi"):
                if getattr(m, name) != getattr(first, name):
                    raise ValueError(
                        f"member {b} has {name}={getattr(m, name)!r}, member 0 "
                        f"{getattr(first, name)!r}; ensemble members must share their sizes"
                    )
        self.D_in, self.D_out = first.D_in, first.D_out
        self.N, self.g = first.N, first.g
        self.L = first.L
        self.phi = first.phi

        def stack(get) -> torch.nn.Parameter:
            return torch.nn.Parameter(torch.stack([get(m).detach() for m in members]))

        self.U = stack(lambda m: m.U)
        self.V = stack(lambda m: m.V)
        self.J = torch.nn.ParameterList([stack(lambda m, k=k: m.J[k]) for k in range(first.L - 1)])

    @property
    def B(self) -> int:
        return self.U.shape[0]

    @property
    def device(self) -> torch.device:
        return self.U.device

    @property
    def dtype(self) -> torch.dtype:
        return self.U.dtype

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (P, D_in) -> (H (B, P, L, N) pre-activations, Z (B, P, D_out) outputs); member
        b's slices are `self.member(b)(x)`."""
        if x.ndim != 2 or x.shape[1] != self.D_in:
            raise ValueError(f"x must have shape (P, D_in={self.D_in}), got {tuple(x.shape)}")

        H = [(1 / math.sqrt(self.D_in)) * torch.einsum("bid, pd -> bpi", self.U, x)]
        for layer in range(self.L - 1):
            z_prev = self.phi(H[-1])
            J = self.J[layer]
            h_next = (self.g / math.sqrt(self.N)) * torch.einsum("bij, bpj -> bpi", J, z_prev)
            H.append(h_next)

        Z = (1 / self.N) * torch.einsum("bid, bpi -> bpd", self.V, self.phi(H[-1]))
        return torch.stack(H, dim=2), Z

    def weight_l2(self) -> torch.Tensor:
        """(B,): each member's sum of squares of its parameters (`MLP.weight_l2` per member)."""
        return torch.stack([p.square().flatten(1).sum(1) for p in self.parameters()]).sum(0)

    def member(self, b: int) -> MLP:
        """A plain `MLP` holding a copy of member b's weights (same device and dtype); changing
        it leaves the ensemble untouched."""
        if not 0 <= b < self.B:
            raise IndexError(f"member {b} out of range for an ensemble of B={self.B}")
        # The throwaway generator keeps the global RNG untouched; the draw is overwritten.
        scratch = torch.Generator(device=self.device)
        net = MLP(
            self.D_in,
            self.D_out,
            self.N,
            self.L,
            self.g,
            self.phi,
            self.device,
            self.dtype,
            scratch,
        )
        net.load_state_dict({name: p[b] for name, p in self.state_dict().items()})
        return net


Net = MLP | EnsembleMLP


def member_seed(seed: int, b: int) -> int:
    """Seed of member b's initial-weight generator: `seed` itself for member 0, so a member 0
    is the single network of the same seed, and for b > 0 the first 8 bytes (little-endian)
    of sha256("fct.model.member:<seed>:<b>"), a stream unrelated to any small seed. It does
    not depend on B, so member b is the same network in every ensemble of more than b."""
    if b < 0:
        raise ValueError(f"member index must be >= 0, got {b}")
    if b == 0:
        return seed
    digest = hashlib.sha256(f"fct.model.member:{seed}:{b}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _make_member(cfg: NetConfig, b: int, D_in: int, D_out: int, dtype: torch.dtype) -> MLP:
    """Member b of the net `cfg` describes, built on the CPU."""
    generator = torch.Generator().manual_seed(member_seed(cfg.seed, b))
    phi = get_phi(cfg.phi)
    return MLP(D_in, D_out, cfg.N, cfg.L, cfg.g, phi, torch.device("cpu"), dtype, generator)


def make_net(
    cfg: NetConfig, D_in: int, D_out: int, device: torch.device, dtype: torch.dtype
) -> Net:
    """Build the net `cfg` describes: an `MLP` for cfg.B = 1, else an `EnsembleMLP` of cfg.B
    members, member b drawn from a generator seeded with `member_seed(cfg.seed, b)`. Initial
    weights depend only on cfg.seed (and b), independent of device.

    Weights are drawn on the CPU from local generators and then moved to `device`, so no
    global RNG (CPU or CUDA) is read or reseeded and CPU and CUDA builds agree. The
    activation is resolved by name, so a checkpoint needs only config.json plus a
    state_dict (nothing unpicklable is stored).
    """
    members = [_make_member(cfg, b, D_in, D_out, dtype) for b in range(cfg.B)]
    net = members[0] if cfg.B == 1 else EnsembleMLP(members)
    return net.to(device)
