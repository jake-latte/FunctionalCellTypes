"""Synthetic latents and tuning maps shared by the spectrum, spectrum-monitor and linear-ring
tests."""

import torch


def uniform_factors(P: int, K: int, a: float = 1.0, seed: int = 0) -> dict[str, torch.Tensor]:
    """Latents z0..z{K-1} iid uniform on [0, a] in float64, as the linear-teacher task draws
    them: the factors `fct.spectrum.latent_basis` takes."""
    gen = torch.Generator().manual_seed(seed)
    z = a * torch.rand((P, K), dtype=torch.float64, generator=gen)
    return {f"z{k}": z[:, k] for k in range(K)}


def gaussian_field(d: torch.Tensor, n_draws: int, gen: torch.Generator) -> torch.Tensor:
    """(n_draws, ..., N) real stationary Gaussian fields on the ring with unitary-DFT power d_q,
    for d (..., N) in float64, one independent field per draw and per leading index of d.

    t = C^{1/2} xi for white xi and the circulant C of eigenvalues d_q: in Fourier space
    t~(q) = sqrt(d_q) xi~(q) with E|xi~(q)|^2 = 1 (random phases), so E|t~(q)|^2 = d_q. d must
    be even in q (d_q = d_{-q}) for t to be real. |t~(q)|^2 is then d_q times a chi^2_1 variate
    at q = 0 and N/2 and d_q times an Exp(1) variate otherwise, shared by q and -q.
    """
    xi = torch.randn((n_draws, *d.shape), dtype=torch.float64, generator=gen)
    return torch.fft.ifft(d.sqrt() * torch.fft.fft(xi, dim=-1), dim=-1).real
