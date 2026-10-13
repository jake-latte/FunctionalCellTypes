"""Persistent single-site HMC sampler for the mean-field saddle point, plus linear-algebra helpers.

The mean-field solver extremises an action S[{Phi_l}, {Phi_hat_l}] over layer kernels Phi_l and
conjugate kernels Phi_hat_l; every derivative of S is an expectation under a tilted single-site
density, which this sampler estimates.  Symbols map to the solver's as

    Sigma_l = g^2 Phi_{l-1} / lambda_w   (Sigma_0 = Phi_0 / lambda_w),    M = Phi_hat_l.

Single-site density of layer l (h in R^P, one entry per input pattern):

    p(h) ∝ N(h; 0, Sigma) exp( -1/2 psi(h)^T M psi(h) - 1/2 lambda_r phi(h)^T phi(h) ),

with psi = phi for hidden layers, and psi = identity (readout from the pre-activation) or
psi = phi for the last layer.  With u = grad_h U, U(h) = 1/2 psi^T M psi + 1/2 lambda_r phi^T phi,
the Stein identity <h h^T> = Sigma - Sigma <u h^T> holds in equilibrium and is reported as
`stein_mismatch`.
"""

import math
from typing import Any

import numpy as np
import torch

from fct.activations import Activation, phi_prime


# ----------------------------------------------------------------------------- helpers
def sym(A: torch.Tensor) -> torch.Tensor:
    """Symmetric part of A (over the last two dimensions)."""
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
    """||A - B||_F / ||B||_F.  The denominator is floored at 1e-300 (deliberately, to avoid a
    division by zero when B is exactly zero; the result is then huge rather than NaN)."""
    return (A - B).norm().item() / max(B.norm().item(), 1e-300)


def kernel_alignment(A: torch.Tensor, B: torch.Tensor) -> float:
    """Cosine similarity <A, B>_F / (||A||_F ||B||_F).  The denominator is floored at 1e-300
    (deliberately, so an all-zero kernel gives alignment 0 rather than NaN)."""
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

    Parameters
    ----------
    Sigma : (P, P) prior covariance (any float dtype; stored as float64, with a small jitter
        of `jitter_rel` * mean diagonal added).  Also fixes the device of all chains.
    M : (P, P) symmetric tilt matrix (Φ̂ of the solver), converted to float64.
    lambda_r : coefficient of the extra ½ λ_r φ(h)ᵀφ(h) penalty (>= 0; 0 switches it off).
    phi : elementwise activation; its derivative comes from `fct.activations.phi_prime`.
    psi_is_phi : True for ψ = φ (hidden layers, or readout="phi"); False for ψ = identity
        (last layer with readout="h").
    n_chains : number of parallel chains.
    eps : initial leapfrog step size in whitened coordinates (adapted unless `adapt` is False).
    n_leap : leapfrog steps per transition.
    target_accept : acceptance rate the step-size adaptation aims for.
    adapt_rate : log-step-size increment per unit of (acceptance - target_accept).
    jitter_rel : relative diagonal jitter added to Σ.
    sample_dtype : dtype of the chains and of the HMC arithmetic.
    div_threshold : a proposal with energy error ΔH above this counts as divergent.
    precondition : whiten with C = (Σ⁻¹ + Q)⁻¹ if True, with Σ itself if False.
    generator : torch.Generator driving every random draw (None: the global generator).

    Attributes
    ----------
    adapt : public switch for step-size adaptation (default True); it is part of `state_dict`.
    """

    def __init__(
        self,
        Sigma: torch.Tensor,
        M: torch.Tensor,
        lambda_r: float,
        phi: Activation,
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
        self.phi = phi
        self.phi_prime = phi_prime(phi)
        self.psi_is_phi = psi_is_phi
        self.lambda_r = float(lambda_r)
        self.n_chains = n_chains
        self.n_leap = n_leap
        self.log_eps = math.log(eps)
        self.target_accept = target_accept
        self.adapt_rate = adapt_rate
        self.adapt = True
        self.jitter_rel = jitter_rel
        self.sample_dtype = sample_dtype
        self.div_threshold = div_threshold
        self.device = Sigma.device
        self.gen = generator
        self.P = Sigma.shape[0]

        self.M64 = sym(M.to(torch.float64))
        self.M = self.M64.to(sample_dtype)
        self.set_prior(Sigma)  # also builds L, G via _refresh()
        self.z = self._whitened_prior_draw(n_chains)  # exact draw from the untilted prior
        self._prev_summary = None
        self.n_resets_total = 0
        self.n_transitions_total = 0

    # -- state -----------------------------------------------------------------
    def _randn(self, *shape) -> torch.Tensor:
        return torch.randn(*shape, device=self.device, dtype=self.sample_dtype, generator=self.gen)

    def _rand(self, *shape) -> torch.Tensor:
        return torch.rand(*shape, device=self.device, dtype=self.sample_dtype, generator=self.gen)

    def _whitened_prior_draw(self, n: int) -> torch.Tensor:
        """n exact draws from N(0, Σ), returned in whitened coordinates (sample_dtype)."""
        h0 = self._randn(n, self.P).double() @ self._Sigma_chol.T
        return torch.linalg.solve_triangular(self.L64, h0.T, upper=False).T.to(self.sample_dtype)

    def set_prior(self, Sigma: torch.Tensor) -> None:
        """Σ (P,P) float64 prior covariance."""
        Sigma = sym(Sigma.to(torch.float64))
        jitter = self.jitter_rel * (Sigma.diagonal().mean().item() + 1e-300)
        eye = torch.eye(self.P, device=Sigma.device, dtype=Sigma.dtype)
        self.Sigma = Sigma + jitter * eye
        self._Sigma_chol = torch.linalg.cholesky(self.Sigma)
        self.Sigma_inv = torch.cholesky_inverse(self._Sigma_chol)
        self._refresh()

    def set_tilt(self, M: torch.Tensor) -> None:
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

    def _refresh(self) -> None:
        """Recompute the whitening transform h = z Lᵀ with L Lᵀ = C = (Σ⁻¹ + Q)⁻¹, i.e. the
        chains move in coordinates where the total curvature of the potential is O(1)
        (preconditioned HMC with mass matrix C⁻¹).  The chain state h is preserved."""
        # L64 exists once a whitening has been built; before that there is no chain state (z)
        # to carry over.  Sigma_inv and M64 are always set before the first call.
        h64 = (self.z.double() @ self.L64.T) if hasattr(self, "L64") else None
        if self.precondition:
            C = torch.linalg.inv(self.Sigma_inv + self._mean_curvature())
        else:
            C = self.Sigma
        self.L64 = torch.linalg.cholesky(sym(C))
        self.L = self.L64.to(self.sample_dtype)
        G = self.L64.T @ self.Sigma_inv @ self.L64  # Gaussian part in z-coords, 0 ≼ G ≼ I
        self.G = sym(G).to(self.sample_dtype)
        if h64 is not None:
            self.z = torch.linalg.solve_triangular(self.L64, h64.T, upper=False).T.to(
                self.sample_dtype
            )

    @property
    def eps(self) -> float:
        return math.exp(self.log_eps)

    def get_state(self) -> tuple[torch.Tensor, float, torch.Tensor | None]:
        """In-process fast path: (h float64, log_eps, prev_summary clone).  Use `state_dict`
        for anything that is saved to disk or passed between processes."""
        return (
            self.z.double() @ self.L64.T,
            self.log_eps,
            None if self._prev_summary is None else self._prev_summary.clone(),
        )

    def set_state(self, state: tuple[torch.Tensor, float, torch.Tensor | None]) -> None:
        """Inverse of `get_state` (chain state is re-whitened with the current L)."""
        h64, self.log_eps, self._prev_summary = state
        self.z = torch.linalg.solve_triangular(self.L64, h64.T, upper=False).T.to(self.sample_dtype)

    def state_dict(self) -> dict[str, Any]:
        """Serialisable checkpoint (tensors and Python scalars only; safe for
        `torch.load(weights_only=True)`).  The chain state `h` is float64 in h-coordinates,
        independent of the whitening, so it can be loaded into a sampler whose Σ, M have since
        been refreshed."""
        h64, log_eps, prev_summary = self.get_state()
        return {
            "h": h64,
            "log_eps": log_eps,
            "prev_summary": prev_summary,
            "n_resets_total": self.n_resets_total,
            "n_transitions_total": self.n_transitions_total,
            "adapt": self.adapt,
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """Restore a state_dict; Σ and M must already be set to the values it was saved under."""
        if d["h"].shape != (self.n_chains, self.P):
            raise ValueError(
                f"state has chain shape {tuple(d['h'].shape)}, "
                f"sampler expects {(self.n_chains, self.P)}"
            )
        self.set_state((d["h"], d["log_eps"], d["prev_summary"]))
        self.n_resets_total = d["n_resets_total"]
        self.n_transitions_total = d["n_transitions_total"]
        self.adapt = d["adapt"]

    def h(self) -> torch.Tensor:
        """Current chain state in h-coordinates, (n_chains, P) in `sample_dtype` (unlike
        `state_dict()["h"]`, which is float64)."""
        return self.z @ self.L.T

    # -- energy ----------------------------------------------------------------
    def _tilt_and_grad(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    def _potential_and_grad(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = z @ self.L.T
        U, u = self._tilt_and_grad(h)
        Gz = z @ self.G
        V = 0.5 * (z * Gz).sum(-1) + U
        return V, Gz + u @ self.L

    # -- one HMC transition ----------------------------------------------------
    def _leapfrog(
        self, z0: torch.Tensor, p: torch.Tensor, g0: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """n_leap leapfrog steps from (z0, p) with potential gradient g0 at z0.
        Returns (z, V, p) at the end point."""
        z, p = z0, p - 0.5 * eps * g0
        for i in range(self.n_leap):
            z = z + eps * p
            V, g = self._potential_and_grad(z)
            p = p - (eps if i < self.n_leap - 1 else 0.5 * eps) * g
        return z, V, p

    def _reset_nonfinite_chains(self) -> int:
        """Re-draw chains whose state has become non-finite from the prior; returns how many."""
        bad = ~torch.isfinite(self.z).all(-1)
        n_bad = int(bad.sum().item())
        if n_bad:
            self.z[bad] = self._whitened_prior_draw(n_bad)
            self.n_resets_total += n_bad
        return n_bad

    def _adapt_step_size(self, acc: float) -> None:
        """Robbins-Monro update of log eps towards `target_accept`, clipped to [1e-6, 10]."""
        self.log_eps += self.adapt_rate * (acc - self.target_accept)
        self.log_eps = min(max(self.log_eps, math.log(1e-6)), math.log(10.0))

    def transition(self) -> dict[str, float]:
        """One HMC transition of every chain, adapting the step size if `adapt` is set.
        Returns: accept (fraction of chains accepted), n_divergent (count), max_abs_dH and
        mean_abs_dH (|energy error| over finite proposals, nan if none), n_reset (chains
        re-drawn after going non-finite), eps (the step size used, whitened coordinates)."""
        z0 = self.z
        eps = self.eps
        p = self._randn(*z0.shape)
        V0, g = self._potential_and_grad(z0)
        H0 = V0 + 0.5 * (p * p).sum(-1)

        z, V, p = self._leapfrog(z0, p, g, eps)
        H1 = V + 0.5 * (p * p).sum(-1)

        dH = H1 - H0
        finite = torch.isfinite(dH)
        divergent = (~finite) | (dH > self.div_threshold)
        log_u = torch.log(self._rand(z0.shape[0]))
        accept = finite & (log_u < -dH)
        self.z = torch.where(accept[:, None], z, z0)

        n_bad = self._reset_nonfinite_chains()

        acc = accept.float().mean().item()
        if self.adapt:
            self._adapt_step_size(acc)
        self.n_transitions_total += 1

        dH_f = dH[finite].abs()
        return {
            "accept": acc,
            "n_divergent": int(divergent.sum().item()),
            "max_abs_dH": dH_f.max().item() if dH_f.numel() else float("nan"),
            "mean_abs_dH": dH_f.mean().item() if dH_f.numel() else float("nan"),
            "n_reset": n_bad,
            "eps": eps,
        }

    # -- run n transitions, accumulate moments ----------------------------------
    def _update_chain_summary(self, h: torch.Tensor) -> tuple[torch.Tensor, float | None]:
        """Per-chain summary ||h||²/P and its lag-1 correlation with the previous call's
        summary (None on the first call).  Stores the summary for the next call."""
        s = (h * h).mean(-1)
        rho = None
        if self._prev_summary is not None:
            c = torch.corrcoef(torch.stack([self._prev_summary, s]).double())[0, 1].item()
            # Non-finite (e.g. constant chains) counts as perfectly correlated: this drives
            # ess_frac to 0, so the MC error is over- rather than under-stated.
            rho = c if math.isfinite(c) else 1.0
        self._prev_summary = s.detach().clone()
        return s, rho

    def _accumulate_moments(self, sums: dict[str, torch.Tensor], h: torch.Tensor) -> None:
        """Add this state's chain-averaged moments (float64) into `sums` in place."""
        _, u = self._tilt_and_grad(h)
        psi = self.phi(h) if self.psi_is_phi else h
        n = h.shape[0]
        sums["psi_psi"] += (psi.T @ psi).double() / n
        sums["psi2_psi2"] += ((psi * psi).T @ (psi * psi)).double() / n
        sums["h_h"] += (h.T @ h).double() / n
        sums["h_u"] += (h.T @ u).double() / n

    @staticmethod
    def _psi_psi_rel_mc_err(psi_psi: torch.Tensor, psi2_psi2: torch.Tensor, n_eff: float) -> float:
        """||MC standard error of psi_psi||_F / ||psi_psi||_F from the elementwise variance."""
        var = (psi2_psi2 - psi_psi**2).clamp_min(0.0) / max(n_eff, 1.0)
        return (var.sqrt().norm() / psi_psi.norm().clamp_min(1e-300)).item()

    def _stein_mismatch(self, h_h: torch.Tensor, h_u: torch.Tensor) -> float:
        """Relative Frobenius gap between ⟨hhᵀ⟩ and its Stein-identity value Σ - Σ⟨u hᵀ⟩.
        The identity holds only in equilibrium: a mismatch means the chains are not
        equilibrated (or the density is not normalisable)."""
        stein_rhs = self.Sigma - 0.5 * (h_u @ self.Sigma + self.Sigma @ h_u.T)
        return relative_fro(h_h, stein_rhs)

    def run(
        self, n_trans: int, collect: bool = True
    ) -> tuple[dict[str, torch.Tensor] | None, dict[str, float]]:
        """Advance all chains by n_trans transitions.  Returns (moments, diagnostics);
        moments is None when collect is False.

        moments (float64 (P, P), averaged over chains and the n_trans states):
            psi_psi : ⟨ψψᵀ⟩        (the kernel this layer produces)
            h_h     : ⟨hhᵀ⟩
            h_u     : ⟨h uᵀ⟩       (Stein term for the previous layer's Φ-gradient)
        diagnostics: accept, eps, n_divergent, n_reset, max_abs_dH, mean_abs_dH, rho1 (lag-1
            autocorrelation of a chain summary), ess_frac, h2_mean / h2_max (||h||²/P over
            chains; max is a drift detector), prior_h2 (mean diagonal of Σ); and, if collect,
            psi_psi_rel_mc_err (||MC s.e. of psi_psi||_F / ||psi_psi||_F) and stein_mismatch.
        """
        P = self.P
        f64 = {"device": self.device, "dtype": torch.float64}
        sums = {k: torch.zeros(P, P, **f64) for k in ("psi_psi", "psi2_psi2", "h_h", "h_u")}
        acc, ndiv, nreset, maxdH, meandH = [], 0, 0, 0.0, []
        rho1, h2_mean, h2_max = [], [], 0.0

        for _ in range(n_trans):
            d = self.transition()
            acc.append(d["accept"])
            ndiv += d["n_divergent"]
            nreset += d["n_reset"]
            maxdH = max(maxdH, d["max_abs_dH"]) if math.isfinite(d["max_abs_dH"]) else maxdH
            meandH.append(d["mean_abs_dH"])

            h = self.h()
            s, rho = self._update_chain_summary(h)
            if rho is not None:
                rho1.append(rho)
            h2_mean.append(s.mean().item())
            h2_max = max(h2_max, s.max().item())

            if collect:
                self._accumulate_moments(sums, h)

        rho = float(np.mean(rho1)) if rho1 else float("nan")
        ess_frac = (1 - rho) / (1 + rho) if math.isfinite(rho) else float("nan")
        diag = {
            "accept": float(np.mean(acc)),
            "eps": self.eps,
            "n_divergent": ndiv,
            "n_reset": nreset,
            "max_abs_dH": maxdH,
            "mean_abs_dH": float(np.nanmean(meandH)) if np.isfinite(meandH).any() else float("nan"),
            "rho1": rho,
            "ess_frac": ess_frac,
            "h2_mean": float(np.mean(h2_mean)),  # mean ||h||²/P over chains
            "h2_max": h2_max,  # max  ||h||²/P over chains (drift detector)
            "prior_h2": self.Sigma.diagonal().mean().item(),
        }
        if not collect:
            return None, diag

        psi_psi, psi2_psi2, h_h, h_u = (sums[k] / n_trans for k in sums)
        # Too few transitions to estimate the autocorrelation (ess_frac is nan): fall back to
        # the optimistic n_eff = n_chains * n_trans, so the reported MC error is a lower bound.
        n_eff = self.n_chains * n_trans * (ess_frac if math.isfinite(ess_frac) else 1.0)
        diag["psi_psi_rel_mc_err"] = self._psi_psi_rel_mc_err(psi_psi, psi2_psi2, n_eff)
        diag["stein_mismatch"] = self._stein_mismatch(h_h, h_u)
        moments = {"psi_psi": psi_psi, "h_h": h_h, "h_u": h_u}
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
        Lp = self._Sigma_chol
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
            eye = torch.eye(self.P, device=self.device, dtype=torch.float64)
            Q = self.Sigma_inv
            lam_M = torch.linalg.eigvalsh(self.M64 + self.lambda_r * eye).min().item()
            return torch.linalg.eigvalsh(Q).min().item() + min(0.0, lam_M)
        return torch.linalg.eigvalsh(self.Sigma_inv + self.M64).min().item()
