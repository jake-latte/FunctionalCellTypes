"""Stochastic gradient Langevin dynamics as a `torch.optim.Optimizer`.

One full-batch Euler-Maruyama step of d(theta) = -grad L dt + sqrt(2 / beta) dW with `dt = lr`:

    theta <- theta - lr * grad L + sqrt(2 * lr / beta) * N(0, I)

whose stationary distribution is p(theta) ~ exp(-beta L) as lr -> 0. The noise is drawn from
the global torch generator (CPU or the parameter's CUDA generator), so the RNG state that
`fct.train` checkpoints already makes a resumed run reproduce an uninterrupted one.
"""

import math
from collections.abc import Callable, Iterable

import torch
from torch import Tensor


class SGLD(torch.optim.Optimizer):
    """SGLD with step size `lr` and inverse temperature `beta` (noise variance 2 * lr / beta).

    The optimiser keeps no per-parameter state, so `state_dict()` holds only `lr` and `beta`.
    After each step `last_msd` is the squared displacement of that step averaged over all
    parameters (a 0-d tensor, so reading it costs no host-device sync until it is converted).
    With a small `lr` and a loss that is flat on the noise scale it approaches 2 * lr / beta,
    which makes `last_msd / (2 * lr)` an effective temperature to compare with 1 / beta.
    """

    def __init__(self, params: Iterable[Tensor], lr: float, beta: float) -> None:
        if not lr > 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not beta > 0:
            raise ValueError(f"beta must be positive, got {beta}")
        super().__init__(params, {"lr": lr, "beta": beta})
        self.last_msd: Tensor | None = None

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        sq_disp, n_params = None, 0
        for group in self.param_groups:
            noise_scale = math.sqrt(2 * group["lr"] / group["beta"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                delta = noise_scale * torch.randn_like(p) - group["lr"] * p.grad
                p.add_(delta)
                total = delta.square().sum()
                sq_disp = total if sq_disp is None else sq_disp + total
                n_params += p.numel()
        if sq_disp is None:
            raise ValueError("SGLD.step found no parameter with a gradient")
        self.last_msd = sq_disp / n_params
        return loss
