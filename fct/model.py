"""The mean-field-scaled MLP and its seeded constructor."""

import math

import torch

from fct.activations import Activation, get_phi
from fct.config import NetConfig


class MLP(torch.nn.Module):
    """L-layer network storing pre-activations H, with a 1/N linear readout of the last
    layer's post-activations, Z = V^T phi(h^L) / N.

    Parameters are U (N, D_in), J_1..J_{L-1} (N, N) and V (N, D_out), all standard normal;
    the 1/sqrt(D_in), g/sqrt(N) and 1/N factors live in `forward`.
    """

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


def make_net(
    cfg: NetConfig, D_in: int, D_out: int, device: torch.device, dtype: torch.dtype
) -> MLP:
    """Build an MLP whose initial weights depend only on cfg.seed, independent of device.

    Weights are drawn on the CPU from a local generator and then moved to `device`, so no
    global RNG (CPU or CUDA) is read or reseeded and CPU and CUDA builds agree. The
    activation is resolved by name, so a checkpoint needs only config.json plus a
    state_dict (nothing unpicklable is stored).
    """
    phi = get_phi(cfg.phi)
    generator = torch.Generator().manual_seed(cfg.seed)
    net = MLP(D_in, D_out, cfg.N, cfg.L, cfg.g, phi, torch.device("cpu"), dtype, generator)
    return net.to(device)
