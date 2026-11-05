"""Check the test-pattern Gaussian of Appendix A1 (the corrected form at A1:95-97).

Joint Gaussian over (h, h_star, hat phi) with exponent
    -1/2 [h; h_star]^T Afull^{-1} [h; h_star] - 1/2 hat phi^T B^{-1} hat phi - s1 hat phi^T h,
    Afull = [[A, A_s], [A_s^T, A_ss]].
Integrating out the training h, the precision of (h_star, hat phi) must be
    [[A_ss^{-1},               s1 A_ss^{-1} A_s^T],
     [s1 A_s A_ss^{-1},  B^{-1} - s1^2 (A - A_s A_ss^{-1} A_s^T)]].
The script compares this, and the form originally written in A1, with the exact
marginal precision obtained by inverting the joint covariance.

Run from the git root:  uv run python latex/notes/scripts/check_a1_test_gaussian.py
"""
import numpy as np

SEED = 1
P, P_STAR, S1 = 3, 2, 0.8          # training patterns, test patterns, sigma_(1)
A_RIDGE, B_SCALE, B_RIDGE = 0.5, 0.05, 0.05


def random_problem(rng):
    G = rng.normal(size=(P + P_STAR, P + P_STAR))
    Afull = G @ G.T + A_RIDGE * np.eye(P + P_STAR)
    Bm = rng.normal(size=(P, P))
    B = B_SCALE * Bm @ Bm.T + B_RIDGE * np.eye(P)
    return Afull, B


def exact_marginal_precision(Afull, B):
    """Precision of (h_star, hat phi) from the joint precision over (h, h_star, hat phi)."""
    n = 2 * P + P_STAR
    Q = np.zeros((n, n))
    Q[:P + P_STAR, :P + P_STAR] = np.linalg.inv(Afull)
    Q[P + P_STAR:, P + P_STAR:] = np.linalg.inv(B)
    Q[:P, P + P_STAR:] = Q[P + P_STAR:, :P] = S1 * np.eye(P)
    keep = np.r_[P:P + P_STAR, P + P_STAR:n]
    return np.linalg.inv(np.linalg.inv(Q)[np.ix_(keep, keep)])


def blocks_corrected(Afull, B):
    A, A_s, A_ss = Afull[:P, :P], Afull[:P, P:], Afull[P:, P:]
    A_ssi = np.linalg.inv(A_ss)
    return A_ssi, np.linalg.inv(B) - S1**2 * (A - A_s @ A_ssi @ A_s.T), S1 * A_s @ A_ssi


def blocks_original(Afull, B):
    """As originally written: exponent ... + s1 hat phi^T A (A^{-1})_star h_star."""
    A, A_s, A_ss = Afull[:P, :P], Afull[:P, P:], Afull[P:, P:]
    Lam = np.linalg.inv(Afull)
    return (np.linalg.inv(A_ss - A_s.T @ np.linalg.inv(A) @ A_s),
            np.linalg.inv(B) - S1**2 * A, -S1 * A @ Lam[:P, P:])


def compare(Qm, blocks):
    ss, pp, ps = blocks
    return [np.abs(Qm[:P_STAR, :P_STAR] - ss).max(),
            np.abs(Qm[P_STAR:, P_STAR:] - pp).max(),
            np.abs(Qm[P_STAR:, :P_STAR] - ps).max()]


def main():
    Afull, B = random_problem(np.random.default_rng(SEED))
    Qm = exact_marginal_precision(Afull, B)
    print("max |error| in blocks (h*h*, phihat phihat, phihat h*):")
    print("  corrected form:", np.array2string(np.array(compare(Qm, blocks_corrected(Afull, B))), precision=2))
    print("  original form :", np.array2string(np.array(compare(Qm, blocks_original(Afull, B))), precision=2))


if __name__ == "__main__":
    main()
