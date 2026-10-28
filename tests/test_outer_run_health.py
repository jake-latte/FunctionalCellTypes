"""The health guard on the outer-gradient run and the absolute bound on the sampled scale.

An inner ascent ends with a step it has not validated (steps are validated by the next run).
The outer-gradient run is that next run: it must pass `_sampler_sound` (`_healthy` but for the
Monte Carlo error, which no revert can cure), and if it does not, the trailing step is
reverted and retaken at half the step size (dropped, if that size is already the floor), up to
max_backtrack times, after which the solve raises. `_sampler_sound` also fails a run whose sample scale ||h||²/P
exceeds max_h2_prior_ratio times the prior's, whatever the growth since the point before.

The stubbed layer is that of tests/test_meanfield_step_control.py: Σ = I, Φ = ½ I, and a tilt
x I gives ⟨ψψᵀ⟩ = I / (1 + x), so the sample scale is h² = 1 / (1 + x) against a prior's 1.
"""

import math
import re

import pytest
import torch

import fct.meanfield as mf
from fct.config import MeanFieldConfig
from tests.test_meanfield_step_control import EYE, layer, tilt

NAN = float("nan")


def h2_at(x: float) -> float:
    """The stub's sample scale ||h||²/P at the tilt x I."""
    return 1.0 / (1.0 + x)


def end_of_ascent(x_to: float, **solver):
    """(problem, state, sampler, inner result) as an inner ascent leaves layer 0: validated at
    the tilt 0 (h² = 1), then a trailing step to the tilt x_to I with eta_phihat = 1, pending."""
    prob, st, sampler = layer(0.0, eta=1.0, **solver)
    inner = mf._new_inner_result(1)
    # the step's tilt change is η̂ grad / c² with c² = ¼ (Φ = ½ I)
    grad = 0.25 * x_to * EYE
    inner.pending[0] = mf._PendingStep(sampler.get_state(), 0.0 * EYE, grad, 1e-2)
    st.Phi_hat[0] = x_to * EYE
    sampler.set_tilt(st.Phi_hat[0])
    return prob, st, sampler, inner


# ----------------------------------------------------------------------------- the bound
def hmc_diag(h2: float) -> dict:
    return {
        "h2_mean": h2, "prior_h2": 1.0, "psi_psi_rel_mc_err": 0.01, "n_divergent": 0,
        "n_reset": 0, "eps": 1.0,
    }  # fmt: skip


@pytest.mark.parametrize(("h2", "healthy"), [(99.0, True), (101.0, False)])
@pytest.mark.parametrize("sampler", ["hmc", "importance"])
def test_a_run_whose_scale_exceeds_the_bound_times_the_prior_is_unhealthy(h2, healthy, sampler):
    """No growth since the point before (h2_before = h2): only the absolute bound can fail it."""
    cfg = MeanFieldConfig(sampler=sampler).solver
    assert cfg.max_h2_prior_ratio == 100.0
    diag = hmc_diag(h2) | {"ess_frac": 0.5}
    assert mf._healthy(diag, h2, 0.5, cfg.n_trans, 1.0, cfg) is healthy


def test_the_bound_is_relative_to_the_prior_scale():
    cfg = MeanFieldConfig(max_h2_prior_ratio=10.0).solver
    diag = hmc_diag(50.0) | {"prior_h2": 10.0}
    assert mf._healthy(diag, 50.0, NAN, cfg.n_trans, 1.0, cfg)
    assert not mf._healthy(diag | {"prior_h2": 4.0}, 50.0, NAN, cfg.n_trans, 1.0, cfg)


# ----------------------------------------------------------------------------- the outer run
def test_a_healthy_outer_run_keeps_the_trailing_step():
    prob, st, sampler, inner = end_of_ascent(0.2)
    og = mf._outer_gradients(prob, st, inner)
    assert inner.n_backtrack[0] == 0 and st.eta_hat[0] == 1.0
    assert tilt(st) == pytest.approx(0.2) and sampler.state == 1
    assert st.h2_last[0] == pytest.approx(h2_at(0.2))
    assert og.hmc_diag[0]["h2_mean"] == pytest.approx(h2_at(0.2))


def test_an_outer_run_above_the_bound_reverts_the_trailing_step_and_retries_at_half_size():
    """The step to x = -0.995 makes h² 200 times the prior's (growth is switched off, so the
    absolute bound is what rejects it); halved, the step reaches x = -0.4975, h² = 1.99."""
    prob, st, sampler, inner = end_of_ascent(-0.995, max_h2_growth=1e6)
    assert h2_at(-0.995) > prob.solver.max_h2_prior_ratio
    og = mf._outer_gradients(prob, st, inner)
    assert inner.n_backtrack[0] == 1
    assert st.eta_hat[0] == 0.5
    assert tilt(st) == pytest.approx(-0.4975, rel=1e-12)
    assert sampler.state == 1  # run once, reverted to the point before the step, run again
    assert og.hmc_diag[0]["h2_mean"] == pytest.approx(h2_at(-0.4975))
    assert st.h2_last[0] == pytest.approx(h2_at(-0.4975))
    # the gradient is taken at the retried tilt
    A, _, _, _ = mf._learning_pieces(prob, st.Phi[-1])
    target = mf._last_stein_target(prob, A)
    torch.testing.assert_close(og.grads[0], -0.5 * st.Phi_hat[0] + 0.5 * target)


def test_the_outer_run_is_checked_for_growth_too():
    """Below the absolute bound (h² 20 times the prior's) but a growth of 20 since the point
    before, above max_h2_growth = 10: the outer run is held to all of `_sampler_sound`."""
    prob, st, _, inner = end_of_ascent(-0.95)
    mf._outer_gradients(prob, st, inner)
    assert inner.n_backtrack[0] == 1 and tilt(st) == pytest.approx(-0.475)


def test_a_large_monte_carlo_error_alone_does_not_reject_an_outer_run():
    """The MC error is a property of the sample size, not of the step: a revert cannot cure
    it, so it fails an inner run (`_healthy`) but not the outer one."""
    prob, st, sampler, inner = end_of_ascent(0.2, max_mc_err=0.5)
    sampler.mc_err = 2.0
    _, diag = sampler.run(1)
    assert not mf._healthy(diag, 1.0, NAN, 1, 1.0, prob.solver)
    sampler.state = 0
    mf._outer_gradients(prob, st, inner)
    assert inner.n_backtrack[0] == 0 and tilt(st) == pytest.approx(0.2)


def test_an_outer_run_that_stays_above_the_bound_raises_with_the_ratio():
    """Every run is 1e3 times a (stubbed) prior scale of 1e-3: halving cannot cure it, and after
    max_backtrack retries the solve stops, saying by how much the bound was exceeded."""
    prob, st, sampler, inner = end_of_ascent(-0.995, max_h2_growth=1e6, max_backtrack=3)
    sampler.prior_h2 = 1e-3
    with pytest.raises(mf.UnhealthySolveError, match="max_h2_prior_ratio") as err:
        mf._outer_gradients(prob, st, inner)
    assert inner.n_backtrack[0] == 3
    x_last = -0.995 / 2**3
    ratio = float(re.search(r"h2/prior ([0-9.e+]+)", str(err.value)).group(1))
    assert ratio == pytest.approx(h2_at(x_last) / 1e-3, rel=1e-2)
    assert isinstance(err.value, RuntimeError)


def test_a_trailing_step_that_fails_at_the_step_size_floor_is_dropped():
    """Halving stops at eta_min_rel x eta_phihat (here 1, where the step already is): retaking
    the step there would only repeat it, so it is dropped, back to the validated tilt 0."""
    prob, st, sampler, inner = end_of_ascent(-0.995, max_h2_growth=1e6, eta_min_rel=1.0)
    assert mf._eta_hat_floor(prob.solver) == st.eta_hat[0] == 1.0
    og = mf._outer_gradients(prob, st, inner)
    assert inner.n_backtrack[0] == 1 and st.eta_hat[0] == 1.0
    assert tilt(st) == 0.0 and sampler.state == 1
    assert og.hmc_diag[0]["h2_mean"] == pytest.approx(1.0)


def test_a_run_that_fails_after_the_step_is_dropped_raises():
    prob, st, sampler, inner = end_of_ascent(-0.995, max_h2_growth=1e6, eta_min_rel=1.0)
    sampler.prior_h2 = 1e-3
    with pytest.raises(mf.UnhealthySolveError, match="after 1 retries"):
        mf._outer_gradients(prob, st, inner)
    assert inner.n_backtrack[0] == 1 and tilt(st) == 0.0


def test_an_unhealthy_outer_run_with_no_trailing_step_raises_at_once():
    """Nothing to revert (no inner step was taken): the only honest outcome is to stop."""
    prob, st, sampler, inner = end_of_ascent(0.0, max_h2_growth=1e6)
    inner.pending[0] = None
    sampler.prior_h2 = 1e-3
    with pytest.raises(mf.UnhealthySolveError, match="h2/prior"):
        mf._outer_gradients(prob, st, inner)
    assert inner.n_backtrack[0] == 0 and math.isclose(tilt(st), 0.0)
