"""The mean-field-scaled MLP, ensembles of it, and their seeded constructor.

Whitened weights. With `whitening` factors (`fct.spatial.whitening_factors`) a net's parameters
are xi, not the weights: the forward pass uses W = S_rows xi S_cols (`effective_weights`), and
the trained prior on xi is the plain |xi|^2. The factors live in a `Whitening` submodule as
non-persistent buffers, so the state dict holds exactly the parameters ("U", "J.<k>", "V") in
both parametrisations and a loader rebuilds the factors from the config. Readers that analyse
trained weights read `effective_weights(net)`; `net.U` is xi under whitening.
"""

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence

import torch

from fct.activations import Activation, get_phi
from fct.config import NetConfig

# (S_rows, S_cols) of every parameter, by parameter name, each in the parameter's stored
# orientation: W = S_rows xi S_cols.
Factors = Mapping[str, tuple[torch.Tensor, torch.Tensor]]


class Whitening(torch.nn.Module):
    """The square-root factors of a whitened net, (S_rows, S_cols) per parameter name, as
    non-persistent buffers: they follow the net to a device and dtype (`like` gives both) but
    are not in its state dict. `whiten(name, xi)` is the effective weight S_rows xi S_cols,
    batched over any leading axes of xi (an ensemble's member axis)."""

    def __init__(self, factors: Factors, like: torch.Tensor):
        super().__init__()
        self.names = tuple(factors)
        for name, (rows, cols) in factors.items():
            for side, S in (("rows", rows), ("cols", cols)):
                if S.ndim != 2 or S.shape[0] != S.shape[1]:
                    raise ValueError(
                        f"the {side} whitening factor of {name!r} must be square, got "
                        f"shape {tuple(S.shape)}"
                    )
                copy = S.detach().clone().to(like.device, like.dtype)
                self.register_buffer(_buffer_name(name, side), copy, persistent=False)

    def factors(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        if name not in self.names:
            raise ValueError(f"no whitening factors for {name!r}; have {list(self.names)}")
        return getattr(self, _buffer_name(name, "rows")), getattr(self, _buffer_name(name, "cols"))

    def whiten(self, name: str, xi: torch.Tensor) -> torch.Tensor:
        """S_rows xi S_cols for xi (..., r, c)."""
        rows, cols = self.factors(name)
        return rows @ xi @ cols

    def as_dict(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return {name: self.factors(name) for name in self.names}

    def equals(self, other: "Whitening") -> bool:
        if self.names != other.names:
            return False
        return all(
            torch.equal(a, b)
            for name in self.names
            for a, b in zip(self.factors(name), other.factors(name), strict=True)
        )


def _buffer_name(name: str, side: str) -> str:
    return f"{side}_{name.replace('.', '_')}"  # buffer names may not contain dots


def _check_factors_fit(factors: Factors, params: Iterable[tuple[str, torch.Tensor]]) -> None:
    """Every parameter has factors of its own (last two) dimensions and no factor is spare."""
    params = dict(params)
    if set(factors) != set(params):
        raise ValueError(
            f"whitening factors cover {sorted(factors)} but the net has parameters {sorted(params)}"
        )
    for name, p in params.items():
        rows, cols = factors[name]
        if rows.shape != (p.shape[-2],) * 2 or cols.shape != (p.shape[-1],) * 2:
            raise ValueError(
                f"whitening factors of {name!r} have shapes {tuple(rows.shape)}, "
                f"{tuple(cols.shape)} but the parameter is {tuple(p.shape)}"
            )


def _same_whitening(a: Whitening | None, b: Whitening | None) -> bool:
    if a is None or b is None:
        return a is b
    return a.equals(b)


class MLP(torch.nn.Module):
    """L-layer network storing pre-activations H, with a 1/N linear readout of the last
    layer's post-activations, Z = V^T phi(h^L) / N.

    Parameters are U (N, D_in), J_1..J_{L-1} (N, N) and V (N, D_out), drawn N(0, init_std^2);
    the 1/sqrt(D_in), g/sqrt(N) and 1/N factors live in `forward`. With `whitening` the
    parameters are xi and the forward pass uses `effective_weights` (module docstring).
    """

    B = 1  # a single network: the ensemble size `fct.train` checks against `net.B`
    member_axes = 0  # leading axes of parameters and outputs that index members (none)

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
        whitening: Factors | None = None,
        init_std: float = 1.0,
    ):
        """generator: source of the initial weights (must live on `device`); None -> global RNG."""
        super().__init__()
        for name, size in {"D_in": D_in, "D_out": D_out, "N": N, "L": L}.items():
            if size < 1:
                raise ValueError(f"{name} must be >= 1, got {size}")
        if not init_std > 0:
            raise ValueError(f"init_std must be positive, got {init_std}")

        def init(*shape: int) -> torch.Tensor:
            w = torch.randn(shape, device=device, dtype=dtype, generator=generator)
            # Left untouched at 1.0 so the standard draw is reproduced bit for bit.
            return w if init_std == 1.0 else init_std * w

        # Drawn in this order (U, V, then J) so a seed always gives the same weights.
        U, V = init(N, D_in), init(N, D_out)
        self._set(U, [init(N, N) for _ in range(1, L)], V, g, phi, whitening)

    def _set(
        self,
        U: torch.Tensor,
        J: list[torch.Tensor],
        V: torch.Tensor,
        g: float,
        phi: Activation,
        whitening: Factors | None,
    ) -> None:
        """Hold U (N, D_in), each J (N, N) and V (N, D_out) as parameters, with g and phi, and
        the whitening factors (checked against the parameters) if any."""
        (self.N, self.D_in), self.D_out = U.shape, V.shape[1]
        self.L, self.g, self.phi = len(J) + 1, g, phi
        self.U = torch.nn.Parameter(U)
        self.V = torch.nn.Parameter(V)
        self.J = torch.nn.ParameterList([torch.nn.Parameter(w) for w in J])
        self.whitening: Whitening | None = None
        if whitening is not None:
            _check_factors_fit(whitening, self.named_parameters())
            self.whitening = Whitening(whitening, like=self.U)

    @classmethod
    def from_weights(
        cls,
        U: torch.Tensor,
        J: list[torch.Tensor],
        V: torch.Tensor,
        g: float,
        phi: Activation,
        whitening: Factors | None = None,
    ) -> "MLP":
        """An MLP holding the given parameters as they are (no copy, no random draw): U (N, D_in),
        each J (N, N), V (N, D_out), which are xi when `whitening` is given. ValueError for
        shapes that do not fit together."""
        N = U.shape[0]
        if U.ndim != 2 or V.ndim != 2 or V.shape[0] != N or any(w.shape != (N, N) for w in J):
            shapes = [tuple(w.shape) for w in (U, *J, V)]
            raise ValueError(f"weights of shapes {shapes} do not make an MLP")
        net = cls.__new__(cls)
        torch.nn.Module.__init__(net)
        net._set(U, list(J), V, g, phi, whitening)
        return net

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
        W = effective_weights(self)

        H = [(1 / math.sqrt(self.D_in)) * torch.einsum("id, pd -> pi", W["U"], x)]
        for layer in range(self.L - 1):
            z_prev = self.phi(H[-1])
            J = W[f"J.{layer}"]
            h_next = (self.g / math.sqrt(self.N)) * torch.einsum("ij, pj -> pi", J, z_prev)
            H.append(h_next)

        Z = (1 / self.N) * torch.einsum("id, pi -> pd", W["V"], self.phi(H[-1]))  # P, D_out
        return torch.stack(H, dim=1), Z

    def weight_l2(self) -> torch.Tensor:
        """Sum of squares of every parameter (of xi under whitening)."""
        return torch.stack([p.square().sum() for p in self.parameters()]).sum()


class EnsembleMLP(torch.nn.Module):
    """B independent `MLP`s batched over a leading member axis: U (B, N, D_in), J_k (B, N, N),
    V (B, N, D_out), with member b's weights at index b. The members share nothing but their
    sizes, g, phi and whitening factors; `forward` runs every member on the same inputs.

    Built from its members (`EnsembleMLP([mlp_0, ..., mlp_{B-1}])`, which copies their weights)
    and read back one at a time with `member(b)`, so every function written for a single `MLP`
    applies to a member unchanged. Parameter names match `MLP`'s ("U", "J.<k>", "V").
    """

    member_axes = 1  # the leading axis of parameters and outputs indexes the members

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
            if not _same_whitening(m.whitening, first.whitening):
                raise ValueError(
                    f"member {b} and member 0 differ in their whitening factors (or one has "
                    "none); ensemble members must share them"
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
        self.whitening: Whitening | None = None
        if first.whitening is not None:
            self.whitening = Whitening(first.whitening.as_dict(), like=self.U)

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
        W = effective_weights(self)

        # Every product is a batched matmul with a contiguous (B, P, .) result, and each scale
        # factor multiplies the smaller of weight and result: elementwise passes over large
        # tensors are what an ensemble pays for most on a CPU (they leave the cache).
        H = [_scaled_matmul(x, W["U"].transpose(1, 2), 1 / math.sqrt(self.D_in))]
        for layer in range(self.L - 1):
            J = W[f"J.{layer}"].transpose(1, 2)
            H.append(_scaled_matmul(self.phi(H[-1]), J, self.g / math.sqrt(self.N)))

        Z = _scaled_matmul(self.phi(H[-1]), W["V"], 1 / self.N)
        return torch.stack(H, dim=2) if self.L > 1 else H[0].unsqueeze(2), Z

    def weight_l2(self) -> torch.Tensor:
        """(B,): each member's sum of squares of its parameters (`MLP.weight_l2` per member)."""
        return torch.stack([p.square().flatten(1).sum(1) for p in self.parameters()]).sum(0)

    def member(self, b: int) -> MLP:
        """A plain `MLP` holding a copy of member b's parameters and of the whitening factors
        (same device and dtype); changing it leaves the ensemble untouched."""
        if not 0 <= b < self.B:
            raise IndexError(f"member {b} out of range for an ensemble of B={self.B}")
        factors = None if self.whitening is None else self.whitening.as_dict()
        with torch.no_grad():
            J = [w[b].clone() for w in self.J]
            return MLP.from_weights(
                self.U[b].clone(), J, self.V[b].clone(), self.g, self.phi, whitening=factors
            )


Net = MLP | EnsembleMLP


def effective_weights(net: Net) -> dict[str, torch.Tensor]:
    """The weights the forward pass uses, by parameter name ("U", "J.<k>", "V"): the parameters
    themselves for a plain net (the same tensors, nothing copied or rounded) and
    S_rows xi S_cols for a whitened one, whose parameters are xi. Differentiable with respect
    to the parameters. Readers that analyse trained weights read these, never `net.U`."""
    if net.whitening is None:
        return dict(net.named_parameters())
    return {name: net.whitening.whiten(name, xi) for name, xi in net.named_parameters()}


def _scaled_matmul(a: torch.Tensor, w: torch.Tensor, c: float) -> torch.Tensor:
    """c * (a @ w) for activations a (P, n) or (B, P, n) and weights w (B, n, m), with c
    applied to whichever of w and the (B, P, m) result is smaller."""
    if w.numel() < w.shape[0] * a.shape[-2] * w.shape[-1]:
        return a @ (c * w)
    return c * (a @ w)


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


def _make_member(
    cfg: NetConfig,
    b: int,
    D_in: int,
    D_out: int,
    dtype: torch.dtype,
    whitening: Factors | None,
    init_std: float,
) -> MLP:
    """Member b of the net `cfg` describes, built on the CPU."""
    generator = torch.Generator().manual_seed(member_seed(cfg.seed, b))
    phi = get_phi(cfg.phi)
    cpu = torch.device("cpu")
    return MLP(D_in, D_out, cfg.N, cfg.L, cfg.g, phi, cpu, dtype, generator, whitening, init_std)


def make_net(
    cfg: NetConfig,
    D_in: int,
    D_out: int,
    device: torch.device,
    dtype: torch.dtype,
    whitening: Factors | None = None,
    init_std: float = 1.0,
) -> Net:
    """Build the net `cfg` describes: an `MLP` for cfg.B = 1, else an `EnsembleMLP` of cfg.B
    members, member b drawn from a generator seeded with `member_seed(cfg.seed, b)`. Initial
    parameters are N(0, init_std^2), depending only on cfg.seed (and b), independent of device;
    with `whitening` they are xi (`fct.train.make_net_from_config` draws them from the prior).

    Weights are drawn on the CPU from local generators and then moved to `device`, so no
    global RNG (CPU or CUDA) is read or reseeded and CPU and CUDA builds agree. The
    activation is resolved by name, so a checkpoint needs only config.json plus a
    state_dict (nothing unpicklable is stored).
    """
    members = [_make_member(cfg, b, D_in, D_out, dtype, whitening, init_std) for b in range(cfg.B)]
    net = members[0] if cfg.B == 1 else EnsembleMLP(members)
    return net.to(device)
