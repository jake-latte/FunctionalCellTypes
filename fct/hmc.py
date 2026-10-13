"""Persistent single-site HMC sampler for the mean-field saddle point, plus linear-algebra helpers.

Conventions (layer index l = 0..L-1):
  prior at layer l : h ~ N(0, Sigma_l)
  tilt at layer l  : exp( -1/2 psi(h)^T M psi(h) - 1/2 lambda_r phi(h)^T phi(h) )
                     psi = phi for hidden layers; for the last layer psi = identity
                     (readout from the pre-activation) or psi = phi.
Every derivative of the action is an expectation under the tilted single-site density;
with u = grad_h U, U(h) = 1/2 psi^T M psi + 1/2 lambda_r phi^T phi, the Stein identity
<h h^T> = Sigma - Sigma <u h^T> holds in equilibrium and is reported as `stein_mismatch`.
"""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from fct.activations import phi_prime


# ----------------------------------------------------------------------------- helpers
def sym(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.transpose(-1, -2))


def project_pd(A: torch.Tensor, floor_rel: float) -> tuple[torch.Tensor, int, float]:
    """Symmetrise and floor the eigenvalues of A at floor_rel * max eigenvalue.
    Returns (A_projected, number_of_eigenvalues_clamped, lambda_min_before)."""
    A = sym(A)
    w, V = torch.linalg.eigh(A)
    lam_min = w.min().item()
    floor = floor_rel * w.max().clamp_min(0.0)
    n_clamped = int((w < floor).sum().item())
    if n_clamped == 0:
        return A, 0, lam_min
    w = torch.clamp(w, min=floor)
    return (V * w) @ V.T, n_clamped, lam_min


def relative_fro(A: torch.Tensor, B: torch.Tensor) -> float:
    """||A - B||_F / ||B||_F (with a floor on the denominator)."""
    return (A - B).norm().item() / max(B.norm().item(), 1e-300)


def kernel_alignment(A: torch.Tensor, B: torch.Tensor) -> float:
    return (A * B).sum().item() / max(A.norm().item() * B.norm().item(), 1e-300)


# ----------------------------------------------------------------------------- HMC
class SingleSiteHMC:
    """Persistent HMC chains for the single-site density of one layer

        p(h) ∝ N(h; 0, Σ) exp( -½ ψ(h)ᵀ M ψ(h) - ½ λ_r φ(h)ᵀ φ(h) ),

    sampled in whitened coordinates z, with h = z Lᵀ and Σ = L Lᵀ, so that the
    potential has O(1) curvature in every direction (C = (Σ⁻¹ + Q)⁻¹ with Q a PSD estimate
    of the tilt's mean curvature; see _refresh).  All chains advance in parallel.
    Expensive linear algebra (cholesky, inverse) runs in float64; the chains run in
    `sample_dtype` (float32 by default; 30-60x faster than float64 on most GPUs).
    """

    def __init__(
        self,
        Sigma: torch.Tensor,
        M: torch.Tensor,
        lambda_r: float,
        phi: Callable,
        psi_is_phi: bool,
        n_chains: int = 2048,
        eps: float = 0.05,
        n_leap: int = 10,
        target_accept: float = 0.75,
        adapt_rate: float = 0.05,
        jitter_rel: float = 1e-6,
        sample_dtype: torch.dtype = torch.float32,
        div_threshold: float = 1e3,
        precondition: bool = True,
        generator: torch.Generator | None = None,
    ):
        self.precondition = precondition
        self.phi, self.phi_prime = phi, phi_prime(phi)
        self.psi_is_phi = psi_is_phi
        self.lambda_r = float(lambda_r)
        self.n_chains, self.n_leap = n_chains, n_leap
        self.log_eps = math.log(eps)
        self.target_accept, self.adapt_rate = target_accept, adapt_rate
        self.adapt = True
        self.jitter_rel = jitter_rel
        self.sample_dtype = sample_dtype
        self.div_threshold = div_threshold
        self.device = Sigma.device
        self.gen = generator
        self.P = Sigma.shape[0]

        self.M64 = sym(M.to(torch.float64)); self.M = self.M64.to(sample_dtype)
        self.set_prior(Sigma)                              # also builds L, G via _refresh()
        Lp = torch.linalg.cholesky(self.Sigma)             # exact draw from the untilted prior
        h0 = self._randn(n_chains, self.P).double() @ Lp.T
        self.z = torch.linalg.solve_triangular(self.L64, h0.T, upper=False).T.to(sample_dtype)
        self._prev_summary = None
        self.n_resets_total = 0
        self.n_transitions_total = 0

    # -- state -----------------------------------------------------------------
    def _randn(self, *shape):
        return torch.randn(*shape, device=self.device, dtype=self.sample_dtype, generator=self.gen)

    def _rand(self, *shape):
        return torch.rand(*shape, device=self.device, dtype=self.sample_dtype, generator=self.gen)

    def set_prior(self, Sigma: torch.Tensor):
        """Σ (P,P) float64 prior covariance."""
        Sigma = sym(Sigma.to(torch.float64))
        jitter = self.jitter_rel * (Sigma.diagonal().mean().item() + 1e-300)
        eye = torch.eye(self.P, device=Sigma.device, dtype=Sigma.dtype)
        self.Sigma = Sigma + jitter * eye
        self.Sigma_inv = torch.cholesky_inverse(torch.linalg.cholesky(self.Sigma))
        self._refresh()

    def set_tilt(self, M: torch.Tensor):
        self.M64 = sym(M.to(torch.float64))
        self.M = self.M64.to(self.sample_dtype)
        self._refresh()

    def _mean_curvature(self) -> torch.Tensor:
        """PSD estimate Q of the average Hessian of the tilt exponent U(h), used only to
        precondition the chains.  ψ = h: the ½hᵀMh part is exactly quadratic, so Q = M₊.
        ψ = φ (ReLU-like, each unit active with prob. ½): E[D (M+λ_r I) D] with D = diag(φ'),
        ≈ ¼(M+λ_r I) + ¼ diag(M+λ_r I), then projected onto the PSD cone."""
        eye = torch.eye(self.P, device=self.device, dtype=torch.float64)
        if self.psi_is_phi:
            T = self.M64 + self.lambda_r * eye
            Q = 0.25 * T + 0.25 * torch.diag(T.diagonal())
        else:
            Q = self.M64 + 0.25 * self.lambda_r * eye
        w, V = torch.linalg.eigh(sym(Q))
        return (V * w.clamp_min(0.0)) @ V.T

    def _refresh(self):
        """Recompute the whitening transform h = z Lᵀ with L Lᵀ = C = (Σ⁻¹ + Q)⁻¹, i.e. the
        chains move in coordinates where the total curvature of the potential is O(1)
        (preconditioned HMC with mass matrix C⁻¹).  The chain state h is preserved."""
        if not hasattr(self, "Sigma_inv") or not hasattr(self, "M64"):
            return
        h64 = (self.z.double() @ self.L64.T) if hasattr(self, "L64") else None
        C = torch.linalg.inv(self.Sigma_inv + self._mean_curvature()) if self.precondition else self.Sigma
        self.L64 = torch.linalg.cholesky(sym(C))
        self.L = self.L64.to(self.sample_dtype)
        G = self.L64.T @ self.Sigma_inv @ self.L64                 # Gaussian part in z-coords, 0 ≼ G ≼ I
        self.G = sym(G).to(self.sample_dtype)
        if h64 is not None:
            self.z = torch.linalg.solve_triangular(self.L64, h64.T, upper=False).T.to(self.sample_dtype)

    @property
    def eps(self) -> float:
        return math.exp(self.log_eps)

    def get_state(self):
        """Chain state in h-coordinates (independent of the whitening) plus step size."""
        return (self.z.double() @ self.L64.T, self.log_eps,
                None if self._prev_summary is None else self._prev_summary.clone())

    def set_state(self, state):
        h64, self.log_eps, self._prev_summary = state
        self.z = torch.linalg.solve_triangular(self.L64, h64.T, upper=False).T.to(self.sample_dtype)

    def state_dict(self) -> dict[str, Any]:
        """Checkpointable state: tensors and Python scalars only, independent of the
        whitening, so it can be loaded into a sampler whose Σ, M have since been refreshed."""
        h64, log_eps, prev_summary = self.get_state()
        return dict(
            h=h64,
            log_eps=log_eps,
            prev_summary=prev_summary,
            n_resets_total=self.n_resets_total,
            n_transitions_total=self.n_transitions_total,
        )

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """Restore a state_dict; Σ and M must already be set to the values it was saved under."""
        if d["h"].shape != (self.n_chains, self.P):
            raise ValueError(
                f"state has chain shape {tuple(d['h'].shape)}, sampler expects {(self.n_chains, self.P)}"
            )
        self.set_state((d["h"], d["log_eps"], d["prev_summary"]))
        self.n_resets_total = d["n_resets_total"]
        self.n_transitions_total = d["n_transitions_total"]

    def h(self) -> torch.Tensor:
        return self.z @ self.L.T

    # -- energy ----------------------------------------------------------------
    def _tilt_and_grad(self, h):
        """U(h) per chain and u = ∇_h U (n_chains, P)."""
        ph = self.phi(h)
        if self.psi_is_phi:
            psi, psi_p = ph, self.phi_prime(h)
        else:
            psi, psi_p = h, None
        Mpsi = psi @ self.M
        U = 0.5 * (psi * Mpsi).sum(-1)
        u = Mpsi if psi_p is None else psi_p * Mpsi
        if self.lambda_r != 0.0:
            U = U + 0.5 * self.lambda_r * (ph * ph).sum(-1)
            u = u + self.lambda_r * self.phi_prime(h) * ph
        return U, u

    def _potential_and_grad(self, z):
        h = z @ self.L.T
        U, u = self._tilt_and_grad(h)
        Gz = z @ self.G
        V = 0.5 * (z * Gz).sum(-1) + U
        return V, Gz + u @ self.L

    # -- one HMC transition ----------------------------------------------------
    def transition(self) -> dict[str, float]:
        z0 = self.z
        eps = self.eps
        p = self._randn(*z0.shape)
        V0, g = self._potential_and_grad(z0)
        H0 = V0 + 0.5 * (p * p).sum(-1)

        z, p = z0, p - 0.5 * eps * g
        for i in range(self.n_leap):
            z = z + eps * p
            V, g = self._potential_and_grad(z)
            p = p - (eps if i < self.n_leap - 1 else 0.5 * eps) * g
        H1 = V + 0.5 * (p * p).sum(-1)

        dH = H1 - H0
        finite = torch.isfinite(dH)
        divergent = (~finite) | (dH > self.div_threshold)
        log_u = torch.log(self._rand(z0.shape[0]))
        accept = finite & (log_u < -dH)
        self.z = torch.where(accept[:, None], z, z0)

        # chains whose state has become non-finite are re-drawn from the prior
        bad = ~torch.isfinite(self.z).all(-1)
        n_bad = int(bad.sum().item())
        if n_bad:
            Lp = torch.linalg.cholesky(self.Sigma)
            h0 = self._randn(n_bad, self.P).double() @ Lp.T
            self.z[bad] = torch.linalg.solve_triangular(self.L64, h0.T, upper=False).T.to(self.sample_dtype)
            self.n_resets_total += n_bad

        acc = accept.float().mean().item()
        if self.adapt:
            self.log_eps += self.adapt_rate * (acc - self.target_accept)
            self.log_eps = min(max(self.log_eps, math.log(1e-6)), math.log(10.0))
        self.n_transitions_total += 1

        dH_f = dH[finite].abs()
        return dict(
            accept=acc,
            n_divergent=int(divergent.sum().item()),
            max_abs_dH=dH_f.max().item() if dH_f.numel() else float("nan"),
            mean_abs_dH=dH_f.mean().item() if dH_f.numel() else float("nan"),
            n_reset=n_bad,
            eps=eps,
        )

    # -- run n transitions, accumulate moments ----------------------------------
    def run(self, n_trans: int, collect: bool = True):
        """Advance all chains by n_trans transitions.  Returns (moments, diagnostics).

        moments (float64, averaged over chains and the n_trans states):
            psi_psi : ⟨ψψᵀ⟩        (the kernel this layer produces)
            h_h     : ⟨hhᵀ⟩
            h_u     : ⟨h uᵀ⟩       (Stein term for the previous layer's Φ-gradient)
            psi_psi_rel_mc_err : ||MC s.e. of psi_psi||_F / ||psi_psi||_F
        diagnostics: acceptance, eps, divergences, |ΔH|, lag-1 autocorrelation of a
            chain summary, ESS fraction, chain-state scale, Stein-identity mismatch.
        """
        P = self.P
        f64 = dict(device=self.device, dtype=torch.float64)
        psi_psi = torch.zeros(P, P, **f64)
        psi2_psi2 = torch.zeros(P, P, **f64)
        h_h = torch.zeros(P, P, **f64)
        h_u = torch.zeros(P, P, **f64)
        acc, ndiv, nreset, maxdH, meandH = [], 0, 0, 0.0, []
        rho1, h2_mean, h2_max = [], [], 0.0

        for _ in range(n_trans):
            d = self.transition()
            acc.append(d["accept"]); ndiv += d["n_divergent"]; nreset += d["n_reset"]
            maxdH = max(maxdH, d["max_abs_dH"]) if math.isfinite(d["max_abs_dH"]) else maxdH
            meandH.append(d["mean_abs_dH"])

            h = self.h()
            s = (h * h).mean(-1)                                  # per-chain summary
            if self._prev_summary is not None:
                sp = self._prev_summary
                c = torch.corrcoef(torch.stack([sp, s]).double())[0, 1].item()
                rho1.append(c if math.isfinite(c) else 1.0)
            self._prev_summary = s.detach().clone()
            h2_mean.append(s.mean().item()); h2_max = max(h2_max, s.max().item())

            if collect:
                _, u = self._tilt_and_grad(h)
                psi = self.phi(h) if self.psi_is_phi else h
                n = h.shape[0]
                psi_psi += (psi.T @ psi).double() / n
                psi2_psi2 += ((psi * psi).T @ (psi * psi)).double() / n
                h_h += (h.T @ h).double() / n
                h_u += (h.T @ u).double() / n

        rho = float(np.mean(rho1)) if rho1 else float("nan")
        ess_frac = (1 - rho) / (1 + rho) if math.isfinite(rho) else float("nan")
        diag = dict(
            accept=float(np.mean(acc)),
            eps=self.eps,
            n_divergent=ndiv,
            n_reset=nreset,
            max_abs_dH=maxdH,
            mean_abs_dH=float(np.nanmean(meandH)) if np.isfinite(meandH).any() else float("nan"),
            rho1=rho,
            ess_frac=ess_frac,
            h2_mean=float(np.mean(h2_mean)),       # mean ||h||²/P over chains
            h2_max=h2_max,                         # max  ||h||²/P over chains (drift detector)
            prior_h2=self.Sigma.diagonal().mean().item(),
        )
        if not collect:
            return None, diag

        psi_psi /= n_trans; psi2_psi2 /= n_trans; h_h /= n_trans; h_u /= n_trans
        n_eff = self.n_chains * n_trans * (ess_frac if math.isfinite(ess_frac) else 1.0)
        var = (psi2_psi2 - psi_psi ** 2).clamp_min(0.0) / max(n_eff, 1.0)
        diag["psi_psi_rel_mc_err"] = (var.sqrt().norm() / psi_psi.norm().clamp_min(1e-300)).item()
        # Stein identity ⟨hhᵀ⟩ = Σ - Σ⟨u hᵀ⟩ holds only in equilibrium: a mismatch
        # means the chains are not equilibrated (or the density is not normalisable).
        stein_rhs = self.Sigma - 0.5 * (h_u @ self.Sigma + self.Sigma @ h_u.T)
        diag["stein_mismatch"] = relative_fro(h_h, stein_rhs)
        moments = dict(psi_psi=psi_psi, h_h=h_h, h_u=h_u)
        return moments, diag

    def project_tilt(self, M: torch.Tensor, delta: float) -> torch.Tensor:
        """Project a tilt matrix onto a (sufficient) normalisable region.
        ψ = φ : the effective quadratic tilt M + λ_r I must stay ⪰ δ λ_r I, so that the
                 orthant-restricted exponent is bounded below whatever the prior does.
        ψ = h : Σ⁻¹ + M ≻ 0, enforced as Σ^{1/2} M Σ^{1/2} ⪰ -(1-δ) I."""
        M = sym(M.to(torch.float64))
        eye = torch.eye(self.P, device=self.device, dtype=torch.float64)
        if self.psi_is_phi:
            T = M + self.lambda_r * eye
            w, V = torch.linalg.eigh(T)
            w = torch.clamp(w, min=delta * self.lambda_r)
            return (V * w) @ V.T - self.lambda_r * eye
        Lp = torch.linalg.cholesky(self.Sigma)
        W = Lp.T @ M @ Lp
        w, V = torch.linalg.eigh(sym(W))
        if w.min().item() >= -(1 - delta):
            return M
        w = torch.clamp(w, min=-(1 - delta))
        Linv = torch.linalg.solve_triangular(Lp, eye, upper=False)
        return sym(Linv.T @ ((V * w) @ V.T) @ Linv)

    def normalisability_margin(self) -> float:
        """Sufficient condition (for ReLU / 2-homogeneous φ) that the tilted density is
        normalisable: Σ⁻¹ + restriction of the tilt is positive definite on every orthant.
        Positive margin => normalisable.  Negative margin is a warning, not a proof."""
        if self.psi_is_phi:
            Q = self.Sigma_inv
            lam_M = torch.linalg.eigvalsh(self.M64 + self.lambda_r * torch.eye(self.P, device=self.device, dtype=torch.float64)).min().item()
            return torch.linalg.eigvalsh(Q).min().item() + min(0.0, lam_M)
        return torch.linalg.eigvalsh(self.Sigma_inv + self.M64).min().item()
