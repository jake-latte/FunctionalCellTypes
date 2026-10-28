"""Persistent single-site HMC sampler for the mean-field saddle point, plus linear-algebra helpers.

The mean-field solver extremises an action S[{Phi_l}, {Phi_hat_l}] over layer kernels Phi_l and
conjugate kernels Phi_hat_l; every derivative of S is an expectation under a tilted single-site
density, which this sampler estimates.  Symbols map to the solver's as

    Sigma_l = g^2 Phi_{l-1} / lambda_w   (Sigma_0 = Phi_0 / lambda_w),    M = Phi_hat_l.

Single-site density of layer l (h in R^P, one entry per input pattern):

    p(h) ∝ N(h; 0, Sigma) exp( -1/2 psi(h)^T M psi(h) - 1/2 lambda_r phi(h)^T phi(h) ),

with psi = phi (what the solver uses on every layer, since the net reads its output from
phi(h^L)); psi = identity, the Gaussian tilt, is kept for closed-form checks of the sampler.  With u = grad_h U, U(h) = 1/2 psi^T M psi + 1/2 lambda_r phi^T phi,
the Stein identity <h h^T> = Sigma - Sigma <u h^T> holds in equilibrium and is reported as
`stein_mismatch`.
"""

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

from fct.activations import Activation, known_tilt_class, phi_prime, tilt_class


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


# ----------------------------------------------------------------------------- single-site density
# Pure pieces of the single-site density shared by every sampler of it (`SingleSiteHMC` here,
# `fct.importance.ImportanceSampler`), so the samplers cannot disagree on what they sample.
def jittered_prior(
    Sigma: torch.Tensor, jitter_rel: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(Σ + jitter I, its Cholesky factor, its inverse), float64; the jitter is jitter_rel times
    the mean diagonal, so a rank-deficient Σ (Φ0 with D_in < P) still factorises."""
    Sigma = sym(Sigma.to(torch.float64))
    jitter = jitter_rel * (Sigma.diagonal().mean().item() + 1e-300)
    eye = torch.eye(Sigma.shape[0], device=Sigma.device, dtype=Sigma.dtype)
    Sigma = Sigma + jitter * eye
    chol = torch.linalg.cholesky(Sigma)
    return Sigma, chol, torch.cholesky_inverse(chol)


def tilt_and_grad(
    h: torch.Tensor,
    M: torch.Tensor,
    lambda_r: float,
    phi: Activation,
    phi_prime_fn: Activation,
    psi_is_phi: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tilt exponent U(h) = ½ ψᵀMψ + ½ λ_r φᵀφ per row of h (n, P), and u = ∇_h U (n, P)."""
    ph = phi(h)
    if psi_is_phi:
        psi, psi_p = ph, phi_prime_fn(h)
    else:
        psi, psi_p = h, None
    Mpsi = psi @ M
    U = 0.5 * (psi * Mpsi).sum(-1)
    u = Mpsi if psi_p is None else psi_p * Mpsi
    if lambda_r != 0.0:
        U = U + 0.5 * lambda_r * (ph * ph).sum(-1)
        u = u + lambda_r * phi_prime_fn(h) * ph
    return U, u


def psi_psi_rel_mc_err(psi_psi: torch.Tensor, psi2_psi2: torch.Tensor, n_eff: float) -> float:
    """||MC standard error of psi_psi||_F / ||psi_psi||_F from the elementwise variance."""
    var = (psi2_psi2 - psi_psi**2).clamp_min(0.0) / max(n_eff, 1.0)
    return (var.sqrt().norm() / psi_psi.norm().clamp_min(1e-300)).item()


def stein_mismatch(h_h: torch.Tensor, h_u: torch.Tensor, Sigma: torch.Tensor) -> float:
    """Relative Frobenius gap between ⟨hhᵀ⟩ and its Stein-identity value Σ - Σ⟨u hᵀ⟩.
    The identity holds only in equilibrium: a mismatch means the samples do not follow the
    density (or the density is not normalisable)."""
    stein_rhs = Sigma - 0.5 * (h_u @ Sigma + Sigma @ h_u.T)
    return relative_fro(h_h, stein_rhs)


TiltRule = Literal["gaussian", "bounded", "relu"]


def tilt_rule(phi: Activation, psi_is_phi: bool) -> TiltRule:
    """Which normalisability rule the tilt U(h) = ½ψᵀMψ + ½λ_r φᵀφ follows.
    "gaussian": ψ = h, or ψ = φ with a linear φ; the density is Gaussian in the quadratic part
        (the rate tilt of a nonlinear φ under ψ = h is ≥ 0 and only helps).
    "bounded": ψ = φ with a bounded φ (tanh, sigmoid); the tilt is bounded, so the density is
        normalisable whatever M is.
    "relu": ψ = φ = relu; the tilt grows quadratically on the orthant of active units. The rule
        enforces positive definiteness of Σ⁻¹ + T, T = M + λ_r I: exactly the condition for the
        Gaussian density that equals the relu one on the all-active orthant. It is neither
        necessary nor sufficient for the relu density: the all-active orthant itself needs only
        strict copositivity of Σ⁻¹ + T, so the rule over-cuts tilts whose offending direction
        has mixed signs; the other orthants are not checked, so it under-cuts, and admits
        non-normalisable tilts (`relu_orthant_check` is the per-orthant oracle).
    ValueError (from `tilt_class`) for an activation without a known rule."""
    if not psi_is_phi:
        return "gaussian"
    kind = tilt_class(phi)
    return "gaussian" if kind == "linear" else kind


def _quadratic_tilt(M: torch.Tensor, lambda_r: float, psi_is_phi: bool) -> torch.Tensor:
    """The quadratic form Q of the condition Σ⁻¹ + Q ≻ 0: M, plus λ_r I when ψ = φ (for a linear
    φ the rate tilt is then ½ λ_r hᵀh; for relu, on the all-active orthant). Under ψ = h, λ_r I
    is left out whatever φ is, also a linear one: the rate tilt ½ λ_r φᵀφ is ≥ 0, so leaving it
    out only makes the condition conservative."""
    if not psi_is_phi:
        return M
    return M + lambda_r * torch.eye(M.shape[0], device=M.device, dtype=M.dtype)


def project_tilt(
    M: torch.Tensor,
    delta: float,
    lambda_r: float,
    phi: Activation,
    psi_is_phi: bool,
    Sigma_chol: torch.Tensor,
    Sigma_inv: torch.Tensor,
) -> torch.Tensor:
    """Project a tilt matrix onto a normalisable region, by `tilt_rule`:
    "gaussian": Σ⁻¹ + Q ≻ 0 (Q = `_quadratic_tilt`), necessary and sufficient, enforced as
        Σ^{1/2} Q Σ^{1/2} ⪰ -(1-δ) I (Σ = Sigma_chol Sigma_cholᵀ);
    "relu": the same condition with Q = M + λ_r I, the all-active orthant's (see `tilt_rule`);
    "bounded": none needed, M is returned (symmetrised).
    An admissible M is returned unchanged. Sigma_inv is unused: the signature is the samplers'
    (`project_tilt` methods) and stays the same for every rule."""
    del Sigma_inv
    M = sym(M.to(torch.float64))
    if tilt_rule(phi, psi_is_phi) == "bounded":
        return M
    Q = _quadratic_tilt(M, lambda_r, psi_is_phi)
    Lp = Sigma_chol
    w, V = torch.linalg.eigh(sym(Lp.T @ Q @ Lp))
    if w.min().item() >= -(1 - delta):
        return M
    eye = torch.eye(M.shape[0], device=M.device, dtype=torch.float64)
    Linv = torch.linalg.solve_triangular(Lp, eye, upper=False)
    Q_proj = sym(Linv.T @ ((V * torch.clamp(w, min=-(1 - delta))) @ V.T) @ Linv)
    return Q_proj - (Q - M)


def normalisability_margin(
    M64: torch.Tensor, Sigma_inv: torch.Tensor, lambda_r: float, phi: Activation, psi_is_phi: bool
) -> float:
    """How far the tilted density is inside the region of `tilt_rule`; positive means inside.
    "gaussian": λ_min(Σ⁻¹ + Q), exact. "relu": λ_min(Σ⁻¹ + M + λ_r I), the margin of the
    all-active orthant's Gaussian condition, neither necessary nor sufficient (see `tilt_rule`):
    a positive value does not prove normalisability, a negative one does not disprove it. "bounded": λ_min(Σ⁻¹), the prior's own (any M is fine). NaN for ψ = φ with an
    activation that has no rule (allowed when the solver does not project)."""
    if psi_is_phi and known_tilt_class(phi) is None:
        return float("nan")
    if tilt_rule(phi, psi_is_phi) == "bounded":
        return torch.linalg.eigvalsh(Sigma_inv).min().item()
    return (
        torch.linalg.eigvalsh(Sigma_inv + _quadratic_tilt(M64, lambda_r, psi_is_phi)).min().item()
    )


@dataclass(frozen=True)
class OrthantCheck:
    """`relu_orthant_check`'s result over the 2^P orthants (sign patterns) of h."""

    n_orthants: int  # 2^P, the all-inactive one (the prior alone, always fine) included
    n_violated: int  # orthants whose eigenvalue condition fails (λ_min ≤ 0)
    worst_min_eig: float  # the smallest λ_min over the orthants with an active unit

    @property
    def fraction_violated(self) -> float:
        return self.n_violated / self.n_orthants


def relu_orthant_check(
    Sigma_inv: torch.Tensor,
    M: torch.Tensor,
    lambda_r: float,
    max_P: int = 20,
    chunk: int = 1 << 14,
) -> OrthantCheck:
    """Per-orthant normalisability check of the relu tilt (ψ = φ = relu), a diagnostic and test
    oracle that the solver never calls. On the orthant whose active set is S the exponent is
    hᵀΣ⁻¹h + h_Sᵀ T_SS h_S (T = M + λ_r I); its minimum over the inactive coordinates is
    h_Sᵀ K_S h_S with K_S = (Σ_SS)⁻¹ + T_SS. The check is K_S ≻ 0 for every S ≠ ∅.

    Passing every orthant proves normalisability. A violated orthant is a warning, not a proof:
    the minimum over h_{S^c} and the sign of h_S are not constrained to the orthant, so a
    copositive but indefinite K_S (positive on the orthant itself) is also flagged. The
    all-active orthant's check, S = all units, is the condition `project_tilt` enforces for relu.

    Cost: 2^P eigen-decompositions of P x P matrices, batched `chunk` at a time (each K_S is
    padded to P x P by a diagonal larger than any of its eigenvalues); about 15 s at P = 20 on
    a CPU, doubling with every unit, so P > max_P raises ValueError."""
    P = M.shape[0]
    if P > max_P:
        raise ValueError(
            f"relu_orthant_check: P={P} has 2^{P} orthants; the check is limited to "
            f"P <= max_P={max_P} (raise max_P knowingly: the cost doubles with every unit)"
        )
    Sigma_inv = sym(Sigma_inv.to(torch.float64))
    T = sym(M.to(torch.float64)) + lambda_r * torch.eye(P, device=M.device, dtype=torch.float64)
    Sigma = torch.cholesky_inverse(torch.linalg.cholesky(Sigma_inv))
    # (Σ_SS)⁻¹ ⪯ λ_max(Σ⁻¹) I by interlacing, so `pad` exceeds every eigenvalue of every K_S and
    # the padding never sets λ_min of a non-empty S.
    pad = (
        torch.linalg.eigvalsh(Sigma_inv).max().item()
        + torch.linalg.matrix_norm(T, ord=2).item()
        + 1.0
    )
    bits = 1 << torch.arange(P, device=M.device)
    min_eigs = [
        _padded_orthant_min_eigs(Sigma, T, pad, bits, start, min(start + chunk, 1 << P))
        for start in range(1, 1 << P, chunk)
    ]
    min_eig = torch.cat(min_eigs)
    return OrthantCheck(
        n_orthants=1 << P,
        n_violated=int((min_eig <= 0).sum().item()),
        worst_min_eig=min_eig.min().item(),
    )


def _padded_orthant_min_eigs(
    Sigma: torch.Tensor, T: torch.Tensor, pad: float, bits: torch.Tensor, start: int, stop: int
) -> torch.Tensor:
    """λ_min(K_S) for the active sets S whose bit masks are start..stop-1 (see
    `relu_orthant_check`): with D the 0/1 mask, D Σ D + (I - D) inverts to (Σ_SS)⁻¹ on S and I
    off it, so blockdiag(K_S, pad I) is built without gathering sub-matrices."""
    masks = torch.arange(start, stop, device=Sigma.device)
    D = ((masks[:, None] & bits) != 0).to(torch.float64)  # (n, P)
    DD = D[:, :, None] * D[:, None, :]
    B = DD * Sigma + torch.diag_embed(1.0 - D)
    K = DD * (torch.cholesky_inverse(torch.linalg.cholesky(B)) + T) + torch.diag_embed(
        (1.0 - D) * pad
    )
    return torch.linalg.eigvalsh(K)[:, 0]


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
    psi_is_phi : True for ψ = φ (the solver's case on every layer); False for ψ = identity,
        a Gaussian tilt with closed-form moments.
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
        self.Sigma, self._Sigma_chol, self.Sigma_inv = jittered_prior(Sigma, self.jitter_rel)
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
        if "sampler" in d:  # only other samplers' states carry a kind; HMC's never did
            raise ValueError(f"state was saved by a {d['sampler']!r} sampler, not an HMC one")
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
        return tilt_and_grad(h, self.M, self.lambda_r, self.phi, self.phi_prime, self.psi_is_phi)

    def tilt_and_grad(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """U(h) per chain and u = ∇U, for callers that bring their own samples h (n, P)."""
        return self._tilt_and_grad(h)

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
        diag["psi_psi_rel_mc_err"] = psi_psi_rel_mc_err(psi_psi, psi2_psi2, n_eff)
        diag["stein_mismatch"] = stein_mismatch(h_h, h_u, self.Sigma)
        moments = {"psi_psi": psi_psi, "h_h": h_h, "h_u": h_u}
        return moments, diag

    def project_tilt(self, M: torch.Tensor, delta: float) -> torch.Tensor:
        """`project_tilt` of M under this sampler's prior and rate tilt."""
        return project_tilt(
            M, delta, self.lambda_r, self.phi, self.psi_is_phi, self._Sigma_chol, self.Sigma_inv
        )

    def normalisability_margin(self) -> float:
        """`normalisability_margin` of the current prior and tilt."""
        return normalisability_margin(
            self.M64, self.Sigma_inv, self.lambda_r, self.phi, self.psi_is_phi
        )
