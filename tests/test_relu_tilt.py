"""The relu tilt rule on the exemplar's relu problem: the all-active orthant condition that
`fct.hmc.project_tilt` enforces, checked against the per-orthant oracle
`fct.hmc.relu_orthant_check` and against an exact test of normalisability.

The problem is the exemplar notebook's default (notebooks/build_exemplar.py) with PHI = "relu":
a linear teacher with m = 6 features and k = 2 latents, P = 20, beta = 10, prior coefficient
1e-2 (weight_lambda 0.2, prior_scale 5), rate_lambda 1e-3 (rate_tilt 0.02). Its inputs span
k = 2 dimensions, so the prior Sigma = prior_scale X X^T / m has rank 2 (plus the solver's
jitter): h lives on a plane, and scanning the exponent over the directions of that plane
decides normalisability (see `plane_scan` for the bound that makes a positive scan a proof).
The oracle cannot use that structure and only certifies.
"""

import math
from typing import NamedTuple

import numpy as np
import pytest
import torch

import fct.meanfield as mf
from fct.activations import phi_prime
from fct.data import make_dataset
from fct.exemplar import exemplar_config, with_solver
from fct.hmc import jittered_prior, project_tilt, relu_orthant_check, tilt_and_grad
from fct.meanfield import solve
from tests.pinned_runs import LEGACY_STEP_CONTROL

F64 = torch.float64
BETA, WEIGHT_LAMBDA, RATE_LAMBDA = 10.0, 0.2, 1e-3
DELTA = 0.05  # the default solver.tilt_floor
JITTER_REL = 1e-6  # the default solver.jitter_rel
N_ANGLES = 200_001


def silent(_msg):
    pass


class Exemplar(NamedTuple):
    cfg: object  # fct.config.Config of the notebook's settings with phi relu
    X: torch.Tensor
    Y: torch.Tensor
    prob: mf._Problem  # the solver's problem: alpha, lp (learning prefactor), rate_tilt, ...
    Sigma: torch.Tensor  # the jittered prior, as the samplers hold it
    chol: torch.Tensor
    Sigma_inv: torch.Tensor


@pytest.fixture(scope="module")
def exemplar() -> Exemplar:
    """The exemplar's relu problem as the solver builds it; no training and no cache."""
    cfg = exemplar_config(
        P=20, m=6, k=2, N=128, L=1, phi="relu", g=1.0, data_seed=0, net_seed=0,
        weight_lambda=WEIGHT_LAMBDA, rate_lambda=RATE_LAMBDA, optimizer="sgld", beta=BETA,
        lr=3e-2, n_epochs=10_000, train_seed=0,
    )  # fmt: skip
    dataset = make_dataset(cfg.data, torch.device("cpu"), F64)
    X, Y = dataset.x, dataset.y
    meanfield = cfg.meanfield.model_copy(update={"beta": BETA})
    prob = mf._make_problem(X, Y, 1, WEIGHT_LAMBDA, RATE_LAMBDA, torch.relu, 1.0, meanfield)
    prior = jittered_prior(mf._prior_cov(prob, 0, None), JITTER_REL)
    return Exemplar(cfg, X, Y, prob, *prior)


class PlaneScan(NamedTuple):
    min_exponent: (
        float  # min over unit directions of the plane of a^T Lam^-1 a + relu(h)^T T relu(h)
    )
    h: torch.Tensor  # the unit-norm h = U a at the minimum


def plane_scan(ex: Exemplar, M: torch.Tensor) -> PlaneScan:
    """The exponent hᵀΣ⁻¹h + relu(h)ᵀ T relu(h) (T = M + rate_tilt I) over the directions
    h = U a of the plane the rank-2 prior spans, Σ ≈ U Lam Uᵀ. It is 2-homogeneous in the
    radius, so a negative minimum makes the density non-normalisable outright. A positive
    minimum m proves normalisability only if the off-plane directions, of prior variance at most
    eps, cannot tip it: m > ||T||² / (1/eps - ||T||) is asserted (it holds by orders of
    magnitude here: eps ~ 1e-6)."""
    w, V = torch.linalg.eigh(ex.Sigma)
    U, lam, eps = V[:, -2:], w[-2:], w[-3].item()
    theta = torch.linspace(0.0, 2.0 * math.pi, N_ANGLES, dtype=F64)
    a = torch.stack([torch.cos(theta), torch.sin(theta)], 1)
    h = a @ U.T
    r = torch.relu(h)
    T = M + ex.prob.rate_tilt * torch.eye(M.shape[0], dtype=F64)
    exponent = (a * a / lam).sum(1) + ((r @ T) * r).sum(1)
    i = int(exponent.argmin())
    m = exponent[i].item()
    if m > 0:
        norm_T = torch.linalg.matrix_norm(T, ord=2).item()
        assert m > norm_T**2 / (1.0 / eps - norm_T), "the plane scan does not decide this tilt"
    return PlaneScan(m, h[i])


def learning_tilt(ex: Exemplar, Phi: torch.Tensor) -> torch.Tensor:
    """The read-out layer's stationary tilt -2c A Y Yᵀ A at the kernel Phi."""
    _, AY, _, _ = mf._learning_pieces(ex.prob, Phi)
    return -2.0 * ex.prob.lp * AY @ AY.T


def orthant_floor_projection(ex: Exemplar, M: torch.Tensor) -> torch.Tensor:
    """The relu projection this rule replaced, kept here as the reference it is compared with:
    lambda(M + rate_tilt I) floored at -(1 - delta) lambda_min(Sigma^-1)."""
    eye = torch.eye(M.shape[0], dtype=F64)
    w, V = torch.linalg.eigh(M + ex.prob.rate_tilt * eye)
    floor = -(1 - DELTA) * torch.linalg.eigvalsh(ex.Sigma_inv).min()
    return (V * w.clamp(min=floor)) @ V.T - ex.prob.rate_tilt * eye


def kept_share(projected: torch.Tensor, M: torch.Tensor) -> float:
    """The share of the most negative eigenvalue of M that the projection keeps."""
    return (torch.linalg.eigvalsh(projected).min() / torch.linalg.eigvalsh(M).min()).item()


def project(ex: Exemplar, M: torch.Tensor) -> torch.Tensor:
    """The solver's projection of the relu tilt M under the exemplar's prior (default delta)."""
    return project_tilt(M, DELTA, ex.prob.rate_tilt, torch.relu, True, ex.chol, ex.Sigma_inv)


def margin(ex: Exemplar, M: torch.Tensor) -> float:
    """The rule's margin, lambda_min(Sigma^-1 + M + rate_tilt I); > 0 means admitted."""
    T = M + ex.prob.rate_tilt * torch.eye(M.shape[0], dtype=F64)
    return torch.linalg.eigvalsh(ex.Sigma_inv + T).min().item()


def relu_solve(ex: Exemplar, n_outer: int, **solver) -> list[mf.OuterSnapshot]:
    """The snapshots of n_outer outer steps of the exemplar's relu solve from the lazy start, at
    the notebook's HMC sizes (the tripwire is off: the solve is not converged)."""
    cfg = with_solver(
        ex.cfg, init="lazy", n_outer=n_outer, n_chains=512, warmup_trans=50, n_trans=5,
        n_trans_outer=10, sample_dtype="float64", seed=0, tilt_tripwire_resid=1e9, **solver,
    )  # fmt: skip
    meanfield = cfg.meanfield.model_copy(update={"beta": BETA})
    snapshots = []
    solve(ex.X, ex.Y, 1, WEIGHT_LAMBDA, RATE_LAMBDA, torch.relu, 1.0, meanfield, log=silent,
          on_outer=snapshots.append)  # fmt: skip
    return snapshots


def test_the_exemplar_prior_has_rank_two(exemplar):
    w = torch.linalg.eigvalsh(exemplar.Sigma)
    assert (w > 1e-3 * w.max()).sum() == 2  # the premise of the plane scan


# ----------------------------------------------------------------------------- the analytic tilt
@pytest.fixture(scope="module")
def analytic_tilt(exemplar):
    """The learning tilt at the untilted prior's kernel, which for relu is the arc-cosine
    kernel <relu(h_i) relu(h_j)> = sqrt(S_ii S_jj) (sin t + (pi - t) cos t) / (2 pi)."""
    d = exemplar.Sigma.diagonal().sqrt()
    cos = (exemplar.Sigma / torch.outer(d, d)).clamp(-1.0, 1.0)
    t = torch.arccos(cos)
    Phi = torch.outer(d, d) * (torch.sin(t) + (math.pi - t) * cos) / (2.0 * math.pi)
    return learning_tilt(exemplar, Phi)


def test_at_the_analytic_tilt_the_rule_cuts_a_non_normalisable_tilt_to_a_normalisable_one(
    exemplar, analytic_tilt
):
    assert plane_scan(exemplar, analytic_tilt).min_exponent < 0  # it has to be cut
    assert margin(exemplar, analytic_tilt) < 0  # and the rule does cut it
    projected = project(exemplar, analytic_tilt)
    assert plane_scan(exemplar, projected).min_exponent > 0
    # it keeps twice what the orthant floor kept (0.25 against 0.12 of the leading eigenvalue)
    floored = orthant_floor_projection(exemplar, analytic_tilt)
    assert kept_share(projected, analytic_tilt) > 2 * kept_share(floored, analytic_tilt)


def test_the_oracle_flags_few_orthants_of_the_projected_analytic_tilt(exemplar, analytic_tilt):
    """The oracle's verdict on all 2^20 orthants of the density the rule admits here: a
    handful flagged, barely, though the plane scan proves that density normalisable (false
    alarms: the oracle drops the orthant's sign constraints). The size of a flag says nothing
    either way: see docs/design.md, "The relu tilt rule"."""
    projected = project(exemplar, analytic_tilt)
    check = relu_orthant_check(exemplar.Sigma_inv, projected, exemplar.prob.rate_tilt)
    print(f"\noracle at the projected analytic tilt: {check}")
    assert check.fraction_violated < 1e-4
    assert check.worst_min_eig > -0.01


def test_the_rule_cuts_a_normalisable_learning_tilt_that_is_not_positive_definite(
    exemplar, analytic_tilt
):
    """The rule over-cuts: 0.4 x the analytic tilt (leading eigenvalue -0.37, about where the
    review found the orthant floor keeping 33 %) is normalisable, but Sigma^-1 + M + rate I is
    indefinite along a direction of mixed signs, which no point of the all-active orthant
    reaches. The positive definiteness the rule asks for is then stricter than the strict
    copositivity that orthant needs: it keeps about 0.6 of the leading eigenvalue, not all of
    it (the floor kept about 0.3)."""
    M = 0.4 * analytic_tilt
    assert plane_scan(exemplar, M).min_exponent > 0
    T = M + exemplar.prob.rate_tilt * torch.eye(M.shape[0], dtype=F64)
    w, V = torch.linalg.eigh(exemplar.Sigma_inv + T)
    assert w[0] < 0 and (V[:, 0] > 0).any() and (V[:, 0] < 0).any()
    share = kept_share(project(exemplar, M), M)
    share_floor = kept_share(orthant_floor_projection(exemplar, M), M)
    print(f"\n0.4 x analytic tilt: rule keeps {share:.2f}, orthant floor kept {share_floor:.2f}")
    assert 0.5 < share < 0.7 and share_floor < 0.35


# ----------------------------------------------------------------------------- the solver's tilts
@pytest.fixture(scope="module")
def default_steps(exemplar):
    return relu_solve(exemplar, 12)


def test_at_the_default_steps_the_first_tilts_admitted_are_normalisable(exemplar, default_steps):
    """Not a property of the rule (see the eta_max = 0.1 test below): this run's first 12
    admitted tilts happen to be normalisable."""
    for snap in default_steps:
        assert margin(exemplar, snap.Phi_hat[0]) > 0
        assert plane_scan(exemplar, snap.Phi_hat[0]).min_exponent > 0, f"step {snap.outer}"


def test_the_oracle_passes_the_last_default_step_on_the_first_12_units(exemplar, default_steps):
    """On the orthants whose active units are among the first 12 (2^12 of them): for those the
    oracle's K_S needs only Sigma's 12 x 12 block, whose inverse is the precision of the
    marginal prior. Seconds instead of the 15 s of all 2^20."""
    block = slice(0, 12)
    Sigma_inv = torch.linalg.inv(exemplar.Sigma[block, block])
    Phi_hat = default_steps[-1].Phi_hat[0][block, block]
    check = relu_orthant_check(Sigma_inv, Phi_hat, exemplar.prob.rate_tilt)
    print(f"\noracle at the solver's Phi_hat, first 12 units: {check}")
    assert check.n_violated == 0


class OuterRun(NamedTuple):
    outer: int
    before: torch.Tensor  # the tilt the inner ascent left, its trailing step included
    after: torch.Tensor  # the tilt the outer-gradient run accepted
    n_retry: int  # times the outer-gradient run retook the trailing step


def test_the_guard_rejects_the_non_normalisable_tilt_the_rule_admits_at_step_5(
    exemplar, monkeypatch
):
    """The rule under-cuts: with solver.eta_max = 0.1, the inner ascent of outer step 5 ends
    with a trailing step to a tilt the rule admits (margin +0.19) whose density is not
    normalisable (plane minimum -0.007): along the scan's minimising ray the solver's own
    exponent -log p = 1/2 h^T Sigma^-1 h + U(h) falls without bound, as -r^2. The
    outer-gradient run validates that step: its chains escape the prior's scale, the step is
    reverted and retaken at half size, and the tilt the solve goes on with is normalisable
    (a run at the unrevised tilt samples h^2 / prior about 3e3)."""
    runs = []
    outer_run = mf._outer_run

    def spy(prob, st, l, inner):
        before = st.Phi_hat[l].clone()
        out = outer_run(prob, st, l, inner)
        runs.append(OuterRun(st.outer, before, st.Phi_hat[l].clone(), out.n_retry))
        return out

    monkeypatch.setattr(mf, "_outer_run", spy)
    # The incident's settings: health-only inner steps, the trace-scaled outer step and both
    # step caps at 0.1.
    snapshots = relu_solve(
        exemplar, 6, **{**LEGACY_STEP_CONTROL, "eta_max": 0.1, "eta_max_phihat": 0.1}
    )
    step5 = runs[5]
    assert [r.n_retry for r in runs[:5]] == [0] * 5 and step5.n_retry >= 1
    assert snapshots[5].history["n_retry_outer_run"][5][0] == step5.n_retry
    assert margin(exemplar, step5.before) > 0.1  # admitted by the rule
    scan = plane_scan(exemplar, step5.before)
    assert scan.min_exponent < 0  # and not normalisable
    radii = torch.tensor([1.0, 10.0, 100.0], dtype=F64)
    h = radii[:, None] * scan.h
    U, _ = tilt_and_grad(
        h, step5.before, exemplar.prob.rate_tilt, torch.relu, phi_prime(torch.relu), True
    )
    neg_log_p = 0.5 * ((h @ exemplar.Sigma_inv) * h).sum(1) + U
    torch.testing.assert_close(neg_log_p, 0.5 * scan.min_exponent * radii**2, rtol=1e-3, atol=0)
    # what the solve went on with, at every step: normalisable, chains at the prior's scale
    for snap in snapshots:
        assert plane_scan(exemplar, snap.Phi_hat[0]).min_exponent > 0, f"step {snap.outer}"
        torch.testing.assert_close(snap.Phi_hat[0], runs[snap.outer].after, rtol=0, atol=0)
    h2_ratio = np.array(snapshots[-1].history["hmc_h2_ratio"])
    print(f"\neta_max 0.1 under the guard, h2/prior per step: {np.round(h2_ratio[:, 0], 2)}")
    assert h2_ratio.max() < 10
