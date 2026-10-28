"""Self-normalised importance sampling of the single-site density, a drop-in alternative to
`fct.hmc.SingleSiteHMC` for the mean-field solver.

The density of one layer (see `fct.hmc`) is

    p(h) ∝ N(h; 0, Σ) exp(-U(h)),   U(h) = ½ ψ(h)ᵀ M ψ(h) + ½ λ_r φ(h)ᵀ φ(h).

Every transition draws n fresh, independent samples from the proposal q = N(0, s² Σ) (s the
`proposal_scale`; s = 1 is the prior itself) and weights them by

    log w(h) = log N(h; 0, Σ) - log q(h) - U(h) = ½ (1 - s²) |z|² - U(h) + const,   h = s L z,

with Σ = L Lᵀ.  Expectations are the weighted averages Σ_i w_i f(h_i) / Σ_i w_i over all the
draws of a run.  There is no chain and nothing to equilibrate, so the estimates carry no
autocorrelation; their quality is measured by the Kish effective sample size
ESS = (Σ w)² / Σ w², which collapses towards 1 when the tilt moves the density far from the
proposal.  That collapse is the failure mode to watch, and `ess_frac` and `max_weight` are
what report it: the Monte Carlo error is computed with n_eff = ESS, but its variance factor is
itself estimated from the weighted draws and reads ~0 once a single draw dominates, so it
cannot detect a collapse on its own.

The ESS fraction is a property of the density, not of the sample size: it falls roughly
exponentially with P and with the strength of the tilt, so the prior proposal suits small P and
mild tilts only (at the solver's default n_points=100 it is ~1e-3 or less; use HMC there).
"""

import math
from dataclasses import dataclass
from typing import Any

import torch

from fct.activations import Activation, phi_prime
from fct.hmc import (
    jittered_prior,
    normalisability_margin,
    project_tilt,
    psi_psi_rel_mc_err,
    stein_mismatch,
    sym,
    tilt_and_grad,
)

KIND = "importance"
NAN = float("nan")
# Diagnostics that only an HMC chain has; reported as NaN so both samplers fill the same keys.
HMC_ONLY_DIAGNOSTICS = (
    "accept", "eps", "n_divergent", "n_reset", "max_abs_dH", "mean_abs_dH", "rho1", "h2_max",
)  # fmt: skip
MOMENT_KEYS = ("psi_psi", "psi2_psi2", "h_h", "h_u")


@dataclass
class _Batch:
    """One transition's draws, reduced: log Σw, log Σw², max log w, and the self-normalised
    weighted moments of the batch (empty when moments are not collected)."""

    log_sum_w: float
    log_sum_w2: float
    max_log_w: float
    h2_mean: float
    moments: dict[str, torch.Tensor]


def _logsumexp(values: list[float]) -> float:
    return torch.logsumexp(torch.tensor(values, dtype=torch.float64), 0).item()


class ImportanceSampler:
    """Importance sampling of the single-site density of one layer with the Gaussian prior
    (optionally widened) as the proposal.  The interface is the one the solver uses on
    `SingleSiteHMC`: `run`, `set_prior`, `set_tilt`, `get_state` / `set_state`,
    `state_dict` / `load_state_dict`, `tilt_and_grad`, `project_tilt`,
    `normalisability_margin`.

    Parameters
    ----------
    Sigma : (P, P) prior covariance (stored as float64 with a jitter of `jitter_rel` times the
        mean diagonal, as in SingleSiteHMC).  Also fixes the device.
    M : (P, P) symmetric tilt matrix (Φ̂ of the solver).
    lambda_r : coefficient of the ½ λ_r φ(h)ᵀφ(h) rate tilt (>= 0).
    phi : elementwise activation.
    psi_is_phi : True for ψ = φ (the solver's case); False for ψ = h, a Gaussian tilt.
    n_samples : independent draws per transition.
    proposal_scale : s > 0, the proposal is N(0, s² Σ); s > 1 widens it.
    jitter_rel : relative diagonal jitter added to Σ.
    sample_dtype : dtype of the draws and of the tilt evaluation; log weights and moments are
        accumulated in float64.
    generator : torch.Generator driving every draw (None: the global generator).
    """

    def __init__(
        self,
        Sigma: torch.Tensor,
        M: torch.Tensor,
        lambda_r: float,
        phi: Activation,
        psi_is_phi: bool,
        n_samples: int = 4096,
        proposal_scale: float = 1.0,
        jitter_rel: float = 1e-6,
        sample_dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ):
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if not proposal_scale > 0:
            raise ValueError(f"proposal_scale must be > 0, got {proposal_scale}")
        self.phi = phi
        self.phi_prime = phi_prime(phi)
        self.psi_is_phi = psi_is_phi
        self.lambda_r = float(lambda_r)
        self.n_samples = n_samples
        self.proposal_scale = float(proposal_scale)
        self.jitter_rel = jitter_rel
        self.sample_dtype = sample_dtype
        self.device = Sigma.device
        self.gen = generator
        self.P = Sigma.shape[0]
        self.n_transitions_total = 0
        self.set_tilt(M)
        self.set_prior(Sigma)

    # -- density ---------------------------------------------------------------
    def set_prior(self, Sigma: torch.Tensor) -> None:
        """Σ (P,P) prior covariance; the proposal follows it."""
        self.Sigma, self._Sigma_chol, self.Sigma_inv = jittered_prior(Sigma, self.jitter_rel)
        self._chol = self._Sigma_chol.to(self.sample_dtype)

    def set_tilt(self, M: torch.Tensor) -> None:
        self.M64 = sym(M.to(torch.float64))
        self.M = self.M64.to(self.sample_dtype)

    def tilt_and_grad(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """U(h) per row and u = ∇U, for samples h (n, P) in sample_dtype."""
        return tilt_and_grad(h, self.M, self.lambda_r, self.phi, self.phi_prime, self.psi_is_phi)

    def project_tilt(self, M: torch.Tensor, delta: float) -> torch.Tensor:
        """`fct.hmc.project_tilt` of M under this sampler's prior and rate tilt."""
        return project_tilt(
            M, delta, self.lambda_r, self.phi, self.psi_is_phi, self._Sigma_chol, self.Sigma_inv
        )

    def normalisability_margin(self) -> float:
        """`fct.hmc.normalisability_margin` of the current prior and tilt."""
        return normalisability_margin(
            self.M64, self.Sigma_inv, self.lambda_r, self.phi, self.psi_is_phi
        )

    # -- state: there is none beyond the generator, which the caller owns ---------
    def get_state(self) -> None:
        """Independent draws carry no state between runs (the generator is the caller's)."""

    def set_state(self, state: None) -> None:
        if state is not None:
            raise ValueError(f"an importance sampler has no state to set, got {type(state)}")

    def state_dict(self) -> dict[str, Any]:
        return {"sampler": KIND, "n_transitions_total": self.n_transitions_total}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        if d.get("sampler") != KIND:
            raise ValueError(
                f"state was saved by a {d.get('sampler', 'hmc')!r} sampler, not an importance one"
            )
        self.n_transitions_total = d["n_transitions_total"]

    # -- one transition: a fresh weighted batch ---------------------------------
    def _log_weights(self, z: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """log w per draw (float64) from the whitened draws z and the tilt U at h = s L z:
        prior over proposal, times exp(-U).  Raises if any is not finite: a NaN or +inf
        weight means the tilt is broken or not normalisable, and a -inf one that U overflowed
        in sample_dtype; either way the estimate would be wrong."""
        z2 = (z.double() ** 2).sum(-1)
        log_w = 0.5 * (1.0 - self.proposal_scale**2) * z2 - U.double()
        if not torch.isfinite(log_w).all():
            n_bad = int((~torch.isfinite(log_w)).sum().item())
            raise FloatingPointError(
                f"{n_bad} of {log_w.numel()} importance log-weights are not finite "
                "(non-normalisable tilt, NaN in the tilt, or overflow in sample_dtype)"
            )
        return log_w

    def _draw_batch(self, collect: bool) -> _Batch:
        z = torch.randn(
            self.n_samples, self.P, device=self.device, dtype=self.sample_dtype, generator=self.gen
        )
        h = self.proposal_scale * (z @ self._chol.T)
        U, u = self.tilt_and_grad(h)
        log_w = self._log_weights(z, U)
        m = log_w.max()
        w = torch.exp(log_w - m)  # in (0, 1], the largest is exactly 1
        sum_w = w.sum()
        wn = w / sum_w
        h64 = h.double()
        batch = _Batch(
            log_sum_w=(m + sum_w.log()).item(),
            log_sum_w2=(2 * m + (w * w).sum().log()).item(),
            max_log_w=m.item(),
            h2_mean=(wn * (h64 * h64).mean(-1)).sum().item(),
            moments={},
        )
        if collect:
            psi = (self.phi(h) if self.psi_is_phi else h).double()
            psi2 = psi * psi
            batch.moments = {
                "psi_psi": psi.T @ (wn[:, None] * psi),
                "psi2_psi2": psi2.T @ (wn[:, None] * psi2),
                "h_h": h64.T @ (wn[:, None] * h64),
                "h_u": h64.T @ (wn[:, None] * u.double()),
            }
        return batch

    def run(
        self, n_trans: int, collect: bool = True
    ) -> tuple[dict[str, torch.Tensor] | None, dict[str, float]]:
        """Weighted expectations over n_trans fresh batches of n_samples draws each, all
        self-normalised together.  Returns (moments, diagnostics) like `SingleSiteHMC.run`;
        moments is None when collect is False.

        With collect=False only the diagnostics are computed (still from n_trans batches).

        diagnostics: ess_frac (Kish ESS / number of draws), ess, max_weight (largest normalised
        weight), h2_mean (weighted ⟨||h||²/P⟩), prior_h2 (mean diagonal of Σ), and NaN for the
        HMC-only keys; if collect, psi_psi_rel_mc_err (with n_eff = ESS) and stein_mismatch.
        """
        if n_trans < 1:
            raise ValueError(f"n_trans must be >= 1, got {n_trans}")
        batches = [self._draw_batch(collect) for _ in range(n_trans)]
        self.n_transitions_total += len(batches)
        log_total = _logsumexp([b.log_sum_w for b in batches])
        ess = math.exp(2 * log_total - _logsumexp([b.log_sum_w2 for b in batches]))
        share = [math.exp(b.log_sum_w - log_total) for b in batches]  # each batch's weight mass
        diag = dict.fromkeys(HMC_ONLY_DIAGNOSTICS, NAN)
        diag.update(
            ess=ess,
            ess_frac=ess / (len(batches) * self.n_samples),
            max_weight=math.exp(max(b.max_log_w for b in batches) - log_total),
            h2_mean=sum(a * b.h2_mean for a, b in zip(share, batches)),
            prior_h2=self.Sigma.diagonal().mean().item(),
        )
        if not collect:
            return None, diag
        moments = {k: sum(a * b.moments[k] for a, b in zip(share, batches)) for k in MOMENT_KEYS}
        diag["psi_psi_rel_mc_err"] = psi_psi_rel_mc_err(
            moments["psi_psi"], moments["psi2_psi2"], ess
        )
        diag["stein_mismatch"] = stein_mismatch(moments["h_h"], moments["h_u"], self.Sigma)
        return {k: moments[k] for k in ("psi_psi", "h_h", "h_u")}, diag
