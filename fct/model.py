"""The mean-field-scaled MLP and its seeded constructor."""

import math

import torch

from fct.activations import Activation, get_phi
from fct.config import NetConfig


class MLP(torch.nn.Module):
    """L-layer network storing pre-activations H, with a 1/N readout from the last one.

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
    ):
        super().__init__()
        self.D_in, self.D_out = D_in, D_out
        self.N, self.g = N, g
        self.L = L
        self.phi = phi
        self.device, self.dtype = device, dtype

        self.U = torch.nn.Parameter(torch.randn((N, D_in), device=device, dtype=dtype))
        self.V = torch.nn.Parameter(torch.randn((N, D_out), device=device, dtype=dtype))
        self.J = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.randn((N, N), device=device, dtype=dtype))
                for _ in range(1, L)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (P, D_in) -> (H (P, L, N) pre-activations, Z (P, D_out) outputs)."""
        if x.ndim != 2 or x.shape[1] != self.D_in:
            raise ValueError(f"x must have shape (P, D_in={self.D_in}), got {tuple(x.shape)}")

        H = [(1 / math.sqrt(self.D_in)) * torch.einsum("id, pd -> pi", self.U, x)]
        for l in range(self.L - 1):
            z_prev = self.phi(H[-1])
            h_next = (self.g / math.sqrt(self.N)) * torch.einsum("ij, pj -> pi", self.J[l], z_prev)
            H.append(h_next)

        Z = (1 / self.N) * torch.einsum("id, pi -> pd", self.V, H[-1])  # P, D_out
        return torch.stack(H, dim=1), Z  # P, L, N

    def weight_l2(self) -> torch.Tensor:
        """Sum of squares of every parameter."""
        return torch.stack([p.square().sum() for p in self.parameters()]).sum()


def _cuda_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def make_net(
    cfg: NetConfig, D_in: int, D_out: int, device: torch.device, dtype: torch.dtype
) -> MLP:
    """Build an MLP whose initial weights depend only on cfg.seed.

    The seed is scoped with fork_rng so the caller's global RNG stream is left untouched.
    The activation is resolved by name, so a checkpoint needs only config.json plus a
    state_dict (nothing unpicklable is stored).
    """
    phi = get_phi(cfg.phi)
    # Fork only the CUDA device in use; devices=[] keeps the CPU path off CUDA entirely.
    devices = [_cuda_index(device)] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(cfg.seed)
        return MLP(D_in, D_out, cfg.N, cfg.L, cfg.g, phi, device, dtype)
