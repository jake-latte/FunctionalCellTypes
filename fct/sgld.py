"""Stochastic gradient Langevin dynamics as a `torch.optim.Optimizer`.

One full-batch Euler-Maruyama step of d(theta) = -grad L dt + sqrt(2 / beta) dW with `dt = lr`:

    theta <- theta - lr * grad L + sqrt(2 * lr / beta) * N(0, I)

whose stationary distribution is p(theta) ~ exp(-beta L) as lr -> 0. By default the noise is
drawn from the global torch generator (CPU or the parameter's CUDA generator), so the RNG state
that `fct.train` checkpoints already makes a resumed run reproduce an uninterrupted one.
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

    `member_axis=True` is for the parameters of an ensemble (`fct.model.EnsembleMLP`), whose
    leading axis indexes the members: the update is the same elementwise one with iid noise
    per element, so members do not interact, and `last_msd` is (B,), each member's own average.
    `noise(p)` returns the standard normal draw for parameter `p`; the default
    `torch.randn_like` draws from the global generator, and tests inject fixed streams.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float,
        beta: float,
        member_axis: bool = False,
        noise: Callable[[Tensor], Tensor] = torch.randn_like,
    ) -> None:
        if not lr > 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not beta > 0:
            raise ValueError(f"beta must be positive, got {beta}")
        super().__init__(params, {"lr": lr, "beta": beta})
        self.member_axis = member_axis
        self.noise = noise
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
                delta = noise_scale * self.noise(p) - group["lr"] * p.grad
                p.add_(delta)
                if self.member_axis:  # per member: sum over all but the leading axis
                    total = delta.square().flatten(1).sum(1)
                    n_params += p[0].numel()
                else:
                    total = delta.square().sum()
                    n_params += p.numel()
                sq_disp = total if sq_disp is None else sq_disp + total
        if sq_disp is None:
            raise ValueError("SGLD.step found no parameter with a gradient")
        self.last_msd = sq_disp / n_params
        return loss
