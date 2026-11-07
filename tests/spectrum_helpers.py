"""Synthetic latents shared by the spectrum and spectrum-monitor tests."""

import torch


def uniform_factors(P: int, K: int, a: float = 1.0, seed: int = 0) -> dict[str, torch.Tensor]:
    """Latents z0..z{K-1} iid uniform on [0, a] in float64, as the linear-teacher task draws
    them: the factors `fct.spectrum.latent_basis` takes."""
    gen = torch.Generator().manual_seed(seed)
    z = a * torch.rand((P, K), dtype=torch.float64, generator=gen)
    return {f"z{k}": z[:, k] for k in range(K)}
