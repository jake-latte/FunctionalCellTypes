"""The Monte Carlo error check of a step is relative, like the ESS-drop rule: a step is reverted
when its run's MC error exceeds max_mc_err_growth times that of the accepted point before it.
A steadily high MC error is a property of the sample size, which no revert cures, so it is not
held against a step. Runs of different length are compared per transition (MC error ×
√transitions), since at a fixed point the MC error falls as 1/√transitions. The check holds for
every inner step, the ascent's trailing step included (validated by the outer-gradient run).

Also: the inner ascent's give-up for a step that keeps breaking the sampler.

The stubbed layer is that of tests/test_meanfield_step_control.py: Σ = I, Φ = ½ I, and a tilt
x I gives ⟨ψψᵀ⟩ = I / (1 + x).
"""

import math

import pytest

import fct.meanfield as mf
from fct.config import MeanFieldConfig
from tests.pinned_runs import pinned_solve
from tests.test_meanfield_step_control import ExactGaussianSampler, layer, tilt
from tests.test_outer_run_health import end_of_ascent, hmc_diag

NAN = float("nan")


def mc_failures(mc_err: float, mc_before: float, n_tr: int = 1, **solver) -> list[str]:
    """The failure messages of `_health_failures` for a run sound in every other way, MC error
    `mc_err` over n_tr transitions, against an accepted point whose per-transition MC error is
    `mc_before`."""
    cfg = MeanFieldConfig(**solver).solver
    diag = hmc_diag(1.0) | {"psi_psi_rel_mc_err": mc_err}
    before = mf._RunReference(1.0, NAN, mc_before, 1.0)
    return [f.message for f in mf._health_failures(diag, before, n_tr, cfg)]


# ----------------------------------------------------------------------------- the rule
def test_the_default_growth_bound_is_ten():
    """Above the noise of the estimate: sound runs of the exemplar's relu problem differ from
    the accepted point before by up to 6.3x (docs/design.md); ten is a hundredfold ESS drop."""
    assert MeanFieldConfig().solver.max_mc_err_growth == 10.0


@pytest.mark.parametrize(("mc_err", "fails"), [(9.9, False), (10.1, True)])
def test_an_mc_error_that_grows_beyond_the_bound_fails(mc_err, fails):
    assert bool(mc_failures(mc_err, 1.0)) is fails


@pytest.mark.parametrize("level", [1e-3, 0.5, 4.3, 50.0])
def test_a_steadily_high_mc_error_does_not_fail(level):
    """The pinned 32-chain solves sit at 1.75 to 5.3 at every run: not the step's fault."""
    assert mc_failures(level, level) == []


def test_runs_of_different_length_are_compared_per_transition():
    """Four times the transitions halve the MC error at the same point: 0.5 over 4 transitions
    is the same as 1 over 1; 6 over 4 is a twelvefold rise."""
    assert mc_failures(0.5, 1.0, n_tr=4) == []
    assert mc_failures(4.9, 1.0, n_tr=4) == []
    assert mc_failures(6.0, 1.0, n_tr=4) == ["MC error 6 > max_mc_err_growth x before = 5"]


@pytest.mark.parametrize("mc_before", [NAN, 0.0, math.inf])
def test_without_a_usable_reference_there_is_nothing_to_compare(mc_before):
    """No reference after the warm-up (it collects no moments); none at an MC error of 0
    (⟨ψψᵀ⟩ = 0: a relu layer off in every chain), from which any growth is infinite."""
    assert mc_failures(40.0, mc_before) == []


def test_a_nan_mc_error_fails():
    assert mc_failures(NAN, 1.0) == ["MC error nan > max_mc_err_growth x before = 10"]


def test_a_run_without_moments_has_no_mc_error_to_check():
    """The outer step's probe runs collect no moments (`_probe_samplers`)."""
    cfg = MeanFieldConfig().solver
    diag = {k: v for k, v in hmc_diag(1.0).items() if k != "psi_psi_rel_mc_err"}
    assert mf._health_failures(diag, mf._RunReference(1.0, NAN, 1e-3, 1.0), 1, cfg) == []


def test_the_bound_is_configurable():
    assert mc_failures(1.9, 1.0, max_mc_err_growth=2.0) == []
    assert mc_failures(2.1, 1.0, max_mc_err_growth=2.0) != []


def test_there_is_no_absolute_mc_error_bound():
    assert "max_mc_err" not in type(MeanFieldConfig().solver).model_fields


# ----------------------------------------------------------------------------- the inner step
def test_an_accepted_point_records_its_mc_error_per_transition():
    prob, st, _ = layer(0.8, eta=0.1, mc_err=0.02, n_trans=4)
    pending, res = [None], mf._new_inner_result(1)
    mf._inner_layer_step(prob, st, 0, 0, pending, res)
    assert st.mc_ref[0] == pytest.approx(0.02 * math.sqrt(4))


@pytest.mark.parametrize("mc_err", [0.0, NAN])
def test_an_accepted_point_without_a_usable_mc_error_leaves_no_reference(mc_err):
    """A NaN MC error at an accepted point (only an unchecked first run can be accepted with one)
    is logged at the end of the outer step; an MC error of 0 is legitimate and is not."""
    prob, st, _ = layer(0.8, eta=0.1, mc_err=mc_err)
    pending, res = [None], mf._new_inner_result(1)
    mf._inner_layer_step(prob, st, 0, 0, pending, res)
    assert math.isnan(st.mc_ref[0])
    if math.isnan(mc_err):
        warning = (
            "WARNING: layer 0: the run accepted at outer step 0, inner step 0 has MC error nan; "
            "the next step's MC error is not checked"
        )
        assert st.notes == [warning]
    else:
        assert st.notes == []


def test_an_inner_step_whose_mc_error_jumps_is_reverted():
    prob, st, sampler = layer(0.8, eta=0.1, mc_err=0.01)
    pending, res = [None], mf._new_inner_result(1)
    mf._inner_layer_step(prob, st, 0, 0, pending, res)
    sampler.mc_err = 0.2  # the step left the chains twenty times as noisy
    mf._inner_layer_step(prob, st, 0, 1, pending, res)
    assert res.n_backtrack[0] == 1 and res.n_accepted[0] == 0
    assert st.eta_hat[0] == 0.05 and sampler.state == 1
    assert st.mc_ref[0] == pytest.approx(0.01 * math.sqrt(prob.solver.n_trans))  # unchanged


def test_an_inner_step_at_a_steadily_high_mc_error_is_accepted():
    """What the pinned 32-chain solves see: MC error 4 at every run, the step is not blamed."""
    prob, st, _ = layer(0.8, eta=0.1, mc_err=4.0)
    pending, res = [None], mf._new_inner_result(1)
    for inner in range(3):
        mf._inner_layer_step(prob, st, 0, inner, pending, res)
    assert res.n_backtrack[0] == 0 and res.n_accepted[0] == 2


# ----------------------------------------------------------------------------- the trailing step
class NoisyBeyond(ExactGaussianSampler):
    """Per transition, MC error `high` at tilts x I below `x_noisy` and `low` elsewhere: a
    step there leaves the chains far noisier."""

    def __init__(self, base: ExactGaussianSampler, x_noisy: float, low: float, high: float):
        self.__dict__.update(base.__dict__)
        self.x_noisy, self.low, self.high = x_noisy, low, high

    def run(self, n_trans, collect=True):
        per_transition = self.high if self.M[0, 0].item() < self.x_noisy else self.low
        self.mc_err = per_transition / math.sqrt(n_trans)
        return super().run(n_trans, collect)


def trailing_step(x_to: float, mc_before: float, **solver):
    """`end_of_ascent`, with an accepted point of MC error `mc_before` per transition."""
    prob, st, sampler, inner = end_of_ascent(x_to, **solver)
    st.mc_ref[0] = mc_before
    return prob, st, sampler, inner


def test_a_trailing_step_whose_mc_error_jumps_is_retaken_at_half_size():
    """From an accepted point at 0.01 per transition, the trailing step to x = -0.6 runs at 1
    per transition; retaken at half size (x = -0.3), at 0.01."""
    prob, st, sampler, inner = trailing_step(-0.6, 0.01, max_h2_growth=1e6)
    st.samplers[0] = NoisyBeyond(sampler, x_noisy=-0.5, low=0.01, high=1.0)
    og = mf._outer_gradients(prob, st, inner)
    assert og.n_retry == [1] and og.dropped == [False]
    assert tilt(st) == pytest.approx(-0.3)
    assert st.mc_ref[0] == pytest.approx(0.01)


def test_a_trailing_step_at_a_steadily_high_mc_error_is_kept():
    prob, st, sampler, inner = trailing_step(0.2, 4.0)
    sampler.mc_err = 4.0 / math.sqrt(prob.solver.n_trans_outer)
    og = mf._outer_gradients(prob, st, inner)
    assert og.n_retry == [0] and tilt(st) == pytest.approx(0.2)


# ----------------------------------------------------------------------------- the inner give-up
def test_an_inner_ascent_whose_step_keeps_breaking_the_sampler_raises():
    """Every run after the first is 1e3 times the prior's scale. The ascent calls each layer
    once per inner index, so a give-up counted against the inner index could never fire; it is
    counted per step: after max_backtrack reverts since the last accepted point, it raises."""
    prob, st, sampler = layer(0.0, eta=1.0, max_backtrack=3, max_h2_growth=1e6, n_inner_max=30)
    first_run = sampler.run

    def run(n_trans, collect=True):
        moments, diag = first_run(n_trans, collect)
        return moments, diag | ({"prior_h2": 1e-3} if sampler.state > 1 else {})

    sampler.run = run
    with pytest.raises(mf.UnhealthySolveError, match=r"keeps breaking") as err:
        mf._inner_ascent(prob, st)
    assert "inner step 4 (3 reverts since the last accepted point" in str(err.value)
    assert "max_backtrack=3" in str(err.value)
    assert "h2/prior" in str(err.value)


# ----------------------------------------------------------------------------- the real sampler
def test_a_relu_layer_off_in_every_chain_does_not_stop_the_solve():
    """Regression: from outer step 10 of this solve the last relu layer is often off in every
    chain, its MC error 0, and a reference of 0 made every later run fail (10 x 0 = 0) until
    the outer-run retries raised. The tripwire is off: with that layer off, the hidden layer's
    Stein target is 0 and its relative fixed-point residual about 1e299."""
    result = pinned_solve("gaussian_phi_f64", n_outer=15, n_inner_max=10, tilt_tripwire_resid=1e308)
    assert len(result.history["outer_step"]) == 15
    assert min(min(layers) for layers in result.history["hmc_mc_err"]) == 0.0


def test_the_inner_give_up_fires_on_the_real_sampler():
    """A bound no run can meet: every checked inner run fails, and after max_backtrack reverts
    since the last accepted point the solve stops."""
    with pytest.raises(mf.UnhealthySolveError, match=r"8 reverts since the last accepted point"):
        pinned_solve("gaussian_phi_f64", max_mc_err_growth=1e-9, n_inner_max=30)
