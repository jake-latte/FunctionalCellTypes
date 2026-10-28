"""The relu tilt rule on the exemplar's relu problem: the all-active orthant condition that
`fct.hmc.project_tilt` enforces, checked against the per-orthant oracle
`fct.hmc.relu_orthant_check` and against an exact test of normalisability.

The problem is the exemplar notebook's default (notebooks/build_exemplar.py) with PHI = "relu":
a linear teacher with m = 6 features and k = 2 latents, P = 20, beta = 10, prior coefficient
1e-2 (weight_lambda 0.2, prior_scale 5), rate_lambda 1e-3 (rate_tilt 0.02). Its inputs span
k = 2 dimensions, so the prior Sigma = prior_scale X X^T / m has rank 2 (plus the solver's
jitter): h lives on a plane, and scanning the exponent over the directions of that plane
decides normalisability exactly. The oracle cannot use that structure and only certifies.
"""

import math

import pytest
import torch

from fct.data import make_dataset
from fct.exemplar import exemplar_config, with_solver
from fct.hmc import jittered_prior, project_tilt, relu_orthant_check
from fct.meanfield import network_constants, solve

F64 = torch.float64
BETA, WEIGHT_LAMBDA, RATE_LAMBDA = 10.0, 0.2, 1e-3
DELTA = 0.05  # the default solver.tilt_floor
N_ANGLES = 200_001


def silent(_msg):
    pass


@pytest.fixture(scope="module")
def exemplar():
    """X, Y, the action's constants and the jittered prior (Sigma, its Cholesky factor and
    inverse) of the exemplar's relu problem; no training and no cache."""
    cfg = exemplar_config(
        P=20, m=6, k=2, N=128, L=1, phi="relu", g=1.0, data_seed=0, net_seed=0,
        weight_lambda=WEIGHT_LAMBDA, rate_lambda=RATE_LAMBDA, optimizer="sgld", beta=BETA,
        lr=3e-2, n_epochs=10_000, train_seed=0,
    )  # fmt: skip
    dataset = make_dataset(cfg.data, torch.device("cpu"), F64)
    X, Y = dataset.x, dataset.y
    consts = network_constants(BETA, WEIGHT_LAMBDA, RATE_LAMBDA, X.shape[0], Y.shape[1])
    prior = jittered_prior(consts["prior_scale"] * X @ X.T / X.shape[1], 1e-6)
    return {"cfg": cfg, "X": X, "Y": Y, "consts": consts, "prior": prior}


def plane_min_exponent(Sigma: torch.Tensor, M: torch.Tensor, rate_tilt: float) -> float:
    """min over unit directions a of the plane the rank-2 prior lives on of the exponent
    a^T Lam^-1 a + relu(U a)^T (M + rate_tilt I) relu(U a), Sigma ~ U Lam U^T. Positive iff the
    density restricted to that plane is normalisable (it is 2-homogeneous in the radius); the
    jitter's other directions carry a precision of order 1e6, which no tilt here overturns."""
    w, V = torch.linalg.eigh(Sigma)
    U, lam = V[:, -2:], w[-2:]
    theta = torch.linspace(0.0, 2.0 * math.pi, N_ANGLES, dtype=F64)
    a = torch.stack([torch.cos(theta), torch.sin(theta)], 1)
    r = torch.relu(a @ U.T)
    T = M + rate_tilt * torch.eye(M.shape[0], dtype=F64)
    return ((a * a / lam).sum(1) + ((r @ T) * r).sum(1)).min().item()


def learning_tilt(Phi: torch.Tensor, Y: torch.Tensor, consts: dict) -> torch.Tensor:
    """The read-out layer's stationary tilt -2c A Y Y^T A, A = (alpha I + Phi)^-1."""
    A = torch.linalg.inv(consts["alpha"] * torch.eye(Phi.shape[0], dtype=F64) + Phi)
    return -2.0 * consts["learning_prefactor"] * A @ Y @ Y.T @ A


def orthant_floor_projection(M, rate_tilt, Sigma_inv):
    """The relu projection this rule replaced, kept here as the reference it is compared with:
    lambda(M + rate_tilt I) floored at -(1 - delta) lambda_min(Sigma^-1)."""
    eye = torch.eye(M.shape[0], dtype=F64)
    w, V = torch.linalg.eigh(M + rate_tilt * eye)
    floor = -(1 - DELTA) * torch.linalg.eigvalsh(Sigma_inv).min()
    return (V * w.clamp(min=floor)) @ V.T - rate_tilt * eye


def kept_share(projected: torch.Tensor, M: torch.Tensor) -> float:
    """The share of the most negative eigenvalue of M that the projection keeps."""
    return (torch.linalg.eigvalsh(projected).min() / torch.linalg.eigvalsh(M).min()).item()


def project(M, exemplar):
    _, chol, Sigma_inv = exemplar["prior"]
    rate = exemplar["consts"]["rate_tilt"]
    return project_tilt(M, DELTA, rate, torch.relu, True, chol, Sigma_inv)


def margin(M, exemplar):
    _, _, Sigma_inv = exemplar["prior"]
    T = M + exemplar["consts"]["rate_tilt"] * torch.eye(M.shape[0], dtype=F64)
    return torch.linalg.eigvalsh(Sigma_inv + T).min().item()


def test_the_exemplar_prior_has_rank_two(exemplar):
    w = torch.linalg.eigvalsh(exemplar["prior"][0])
    assert (w > 1e-3 * w.max()).sum() == 2  # the premise of the exact plane scan


# ----------------------------------------------------------------------------- the analytic tilt
@pytest.fixture(scope="module")
def analytic_tilt(exemplar):
    """The learning tilt at the untilted prior's kernel, which for relu is the arc-cosine
    kernel <relu(h_i) relu(h_j)> = sqrt(S_ii S_jj) (sin t + (pi - t) cos t) / (2 pi)."""
    Sigma = exemplar["prior"][0]
    d = Sigma.diagonal().sqrt()
    cos = (Sigma / torch.outer(d, d)).clamp(-1.0, 1.0)
    t = torch.arccos(cos)
    Phi = torch.outer(d, d) * (torch.sin(t) + (math.pi - t) * cos) / (2.0 * math.pi)
    return learning_tilt(Phi, exemplar["Y"], exemplar["consts"])


def test_at_the_analytic_tilt_the_rule_cuts_a_non_normalisable_tilt_to_a_normalisable_one(
    exemplar, analytic_tilt
):
    Sigma, _, Sigma_inv = exemplar["prior"]
    rate = exemplar["consts"]["rate_tilt"]
    assert plane_min_exponent(Sigma, analytic_tilt, rate) < 0  # it has to be cut
    assert margin(analytic_tilt, exemplar) < 0  # and the rule does cut it
    projected = project(analytic_tilt, exemplar)
    assert plane_min_exponent(Sigma, projected, rate) > 0
    # it keeps twice what the orthant floor kept (0.25 against 0.12 of the leading eigenvalue)
    floored = orthant_floor_projection(analytic_tilt, rate, Sigma_inv)
    assert kept_share(projected, analytic_tilt) > 2 * kept_share(floored, analytic_tilt)


def test_the_oracle_flags_few_orthants_of_the_projected_analytic_tilt(exemplar, analytic_tilt):
    """The oracle's verdict on the density the rule admits: a handful of the 2^20 orthants
    flagged, barely (the plane scan above proves that density normalisable, so these are the
    oracle's false alarms: it drops the orthant's sign constraints)."""
    projected = project(analytic_tilt, exemplar)
    check = relu_orthant_check(exemplar["prior"][2], projected, exemplar["consts"]["rate_tilt"])
    print(f"\noracle at the projected analytic tilt: {check}")
    assert check.fraction_violated < 1e-4
    assert check.worst_min_eig > -0.01


def test_the_rule_cuts_a_normalisable_learning_tilt_that_is_not_positive_definite(
    exemplar, analytic_tilt
):
    """The rule's regime: 0.4 x the analytic tilt (leading eigenvalue -0.37, about where the
    review found the orthant floor keeping 33 %) is normalisable, but Sigma^-1 + M + rate I is
    indefinite along a direction of mixed signs, which no point of the all-active orthant
    reaches. The positive definiteness the rule asks for is then stricter than the
    copositivity normalisability needs: it keeps about 0.6 of the leading eigenvalue, not all
    of it (the floor kept about 0.3)."""
    Sigma, _, Sigma_inv = exemplar["prior"]
    rate = exemplar["consts"]["rate_tilt"]
    M = 0.4 * analytic_tilt
    assert plane_min_exponent(Sigma, M, rate) > 0
    T = M + rate * torch.eye(M.shape[0], dtype=F64)
    w, V = torch.linalg.eigh(Sigma_inv + T)
    assert w[0] < 0 and (V[:, 0] > 0).any() and (V[:, 0] < 0).any()
    share = kept_share(project(M, exemplar), M)
    share_floor = kept_share(orthant_floor_projection(M, rate, Sigma_inv), M)
    print(f"\n0.4 x analytic tilt: rule keeps {share:.2f}, orthant floor kept {share_floor:.2f}")
    assert 0.5 < share < 0.7 and share_floor < 0.35


# ----------------------------------------------------------------------------- the solver's tilt
@pytest.fixture(scope="module")
def short_solve(exemplar):
    """A few outer steps of the exemplar's relu solve from the lazy start, at the notebook's HMC
    sizes, with the Phi_hat of every step (the tripwire is off: the solve is not converged)."""
    cfg = with_solver(
        exemplar["cfg"], init="lazy", n_outer=12, n_chains=512, warmup_trans=50, n_trans=5,
        n_trans_outer=10, sample_dtype="float64", seed=0, tilt_tripwire_resid=1e9,
    )  # fmt: skip
    meanfield = cfg.meanfield.model_copy(update={"beta": BETA})
    snapshots = []
    result = solve(
        exemplar["X"], exemplar["Y"], 1, WEIGHT_LAMBDA, RATE_LAMBDA, torch.relu, 1.0, meanfield,
        log=silent, on_outer=snapshots.append,
    )  # fmt: skip
    return result, snapshots


def test_every_tilt_the_solver_admits_is_normalisable(exemplar, short_solve):
    _, snapshots = short_solve
    Sigma = exemplar["prior"][0]
    rate = exemplar["consts"]["rate_tilt"]
    for snap in snapshots:
        assert margin(snap.Phi_hat[0], exemplar) > 0
        assert plane_min_exponent(Sigma, snap.Phi_hat[0], rate) > 0, f"step {snap.outer}"


def test_the_oracle_agrees_at_the_solvers_tilt(exemplar, short_solve):
    result, _ = short_solve
    check = relu_orthant_check(
        exemplar["prior"][2], result.Phi_hat[0], exemplar["consts"]["rate_tilt"]
    )
    print(f"\noracle at the solver's Phi_hat: {check}")
    assert check.n_violated == 0
