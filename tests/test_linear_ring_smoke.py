"""Pipeline test: `linear_ring_tiny` trained through the CLI on the CPU, its final-state spectra
against `fct.linear_theory.predicted_spectra` of the same (floored) configuration.

What it tests: the CLI, both spatial parametrisations under SGLD, the spectrum monitor and the
per-member estimator, end to end. What it does not test: the theory's data term. At beta = 0.25
the prediction is within 0.15 SE of the prior alone on every asserted statistic
(tests/test_linear_ring_estimator.py), and both settings start from the prior (`init` prior), so
a sampler that never moved would match the prediction too. Two checks guard against that: per
member, the final tuning maps must have decorrelated from those of the initial state, and the
initial state is rebuilt from the config and must be the one the first record measured.

The estimator is per member: each of the B members contributes one sample, its final state
(the last record, taken after the last step); the statistics are of the member-averaged
spectrum and their errors the bootstrap over members (tests/linear_ring_helpers.py, where the
tolerances are derived and checked on the predicted law itself). The q = 0 shares and q_1/2
are printed, not asserted. The records before the last serve only the drift check, which is
weak by construction here: chains started in their stationary law show no drift whether or not
they mix.

Both diagonal settings run, each at its own lr: setting 0 (penalty) and setting 3 (whitened).
The run takes about a minute on one thread and is marked slow.
"""

import time
from typing import NamedTuple

import pytest
import torch

from fct.cli import main
from fct.exp import expand, load_exp, setting_dir
from fct.monitor import spectrum_records
from fct.spectrum import half_power_q, mode_spectra, tuning_maps
from fct.train import MONITOR_EPOCH, load_trained, make_net_from_config
from tests.linear_ring_helpers import (
    Law,
    bounds,
    estimate,
    map_correlation,
    predicted_law,
    ring_statistics,
    statistic_names,
    time_drift,
)

PRESET = "linear_ring_tiny"
DIAGONAL = {0: "penalty", 3: "whitened"}
# Records before this epoch are left out of the drift check: the penalty's softest data-carrying
# mode relaxes in about 2000 epochs at its lr (the preset's description), and the preset's 10^4
# epochs are five such times, enough for the final state to forget the initial one.
BURN_IN_EPOCHS = 2000
DRIFT_Z = 4.0  # the drift of each statistic must be within this many of its SEs of zero
EQUIVALENCE_Z = 4.0  # the two parametrisations' deviations must agree within this many SEs
# Per member, the correlation of independent maps averages 0 with a spread of about
# 1 / sqrt(K x 3 frequencies) = 0.4, so the member mean is 0 +- 0.03 at B = 256; a frozen
# sampler gives 1. A mean below 0.1 needs at least 90% decorrelation.
MAX_MAP_CORRELATION = 0.1
# At sigma = 0, x lies in the column space of Z and so does h: the null row is round-off of
# float32 records relative to the latent power (about 1e-14 measured), far below 1e-10.
MAX_NULL_FRACTION = 1e-10
CPU, F64 = torch.device("cpu"), torch.float64

pytestmark = pytest.mark.slow


class Measurement(NamedTuple):
    names: list[str]
    predicted: torch.Tensor  # (3K + 1,) the statistics of the predicted spectrum
    values: torch.Tensor  # (3K + 1,) of the member-averaged final-state spectrum
    se: torch.Tensor  # (3K + 1,) bootstrap over members
    bounds: tuple[torch.Tensor, torch.Tensor]  # (SE_MAX, tolerance) at the run's N and B
    shares: torch.Tensor  # (K,) q = 0 shares of the member-averaged spectrum (reported)
    shares_predicted: torch.Tensor  # (K,)
    q_half: list[int]  # of the member-averaged spectrum (reported)
    q_half_predicted: list[int]
    drift: torch.Tensor  # (3K + 1,) second-half minus first-half time average, member mean
    drift_se: torch.Tensor  # (3K + 1,)
    map_correlation: float  # member mean of the initial-final tuning-map correlation
    initial_record_error: float  # rebuilt initial spectra against the first record, relative
    null_fraction: float  # the null row's power over the latent rows', final state
    final_epoch: int
    n_epochs: int


def _burn_in_index(epochs: list[int]) -> int:
    for index, epoch in enumerate(epochs):
        if epoch >= BURN_IN_EPOCHS:
            return index
    raise ValueError(f"no record at or after epoch {BURN_IN_EPOCHS}; the run has {epochs[-1]}")


def _maps(net, law: Law) -> torch.Tensor:
    """(B, K, N) tuning maps of every member of an ensemble on the setting's data."""
    with torch.no_grad():
        H = net(law.dataset.x)[0]  # (B, P, L, N)
    return tuning_maps(H[:, :, 0], law.U)


def _measure(root, index: int) -> Measurement:
    exp = load_exp(root)
    law = predicted_law(PRESET, index)
    K = law.D.shape[0]
    net, history = load_trained(setting_dir(root, expand(exp)[index]), law.cfg, CPU, F64)
    records = spectrum_records(history)[:, :, 0].to(F64)  # (R, B, K + 1, N), one hidden layer
    final = records[-1, :, :K]  # one sample per member: its final state
    values, se = estimate(final, law)
    epochs = history[MONITOR_EPOCH]
    drift, drift_se = time_drift(records[:-1, :, :K], _burn_in_index(epochs), law)
    x, y = law.dataset.x, law.dataset.y
    initial = make_net_from_config(law.cfg, x.shape[1], y.shape[1], CPU, F64)  # same seed
    first = mode_spectra(initial(x)[0][:, :, 0].detach(), law.U)[:, :K]
    mean = final.mean(0)
    return Measurement(
        names=statistic_names(K),
        predicted=ring_statistics(law.D, law.ell)(law.D),
        values=values,
        se=se,
        bounds=bounds(law),
        shares=mean[:, 0] / mean.sum(-1),
        shares_predicted=law.D[:, 0] / law.D.sum(-1),
        q_half=torch.as_tensor(half_power_q(mean)).tolist(),
        q_half_predicted=torch.as_tensor(half_power_q(law.D)).tolist(),
        drift=drift,
        drift_se=drift_se,
        map_correlation=float(map_correlation(_maps(initial, law), _maps(net, law)).mean()),
        initial_record_error=float(((first - records[0, :, :K]).abs() / first.abs().max()).max()),
        null_fraction=float(records[-1, :, K].sum() / records[-1, :, :K].sum()),
        final_epoch=epochs[-1],
        n_epochs=law.cfg.train.n_epochs,
    )


def _report(index: int, m: Measurement) -> str:
    shares = [round(x, 3) for x in m.shares.tolist()]
    shares_predicted = [round(x, 3) for x in m.shares_predicted.tolist()]
    header = (
        f"setting {index} ({DIAGONAL[index]}): map correlation {m.map_correlation:+.3f}; "
        f"q_1/2 {m.q_half} (predicted {m.q_half_predicted}); "
        f"q = 0 shares {shares} (predicted {shares_predicted})"
    )
    rows = [header]
    _, tolerance = m.bounds
    for i, name in enumerate(m.names):
        rows.append(
            f"  {name:7s} predicted {m.predicted[i]:.4f} measured {m.values[i]:.4f} "
            f"+- {m.se[i]:.4f} (tolerance {tolerance[i]:.4f}) "
            f"drift {m.drift[i]:+.4f} +- {m.drift_se[i]:.4f}"
        )
    return "\n".join(rows)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory) -> dict[int, Measurement]:
    """Run `exp init --preset linear_ring_tiny` and `exp run --index` on both diagonal
    settings, and measure each; the wall time is printed, not asserted."""
    root = tmp_path_factory.mktemp("linear_ring_tiny")
    threads = torch.get_num_threads()
    # The smoke ring's matrices are 32 x 16: on many cores torch's thread synchronisation costs
    # more than the arithmetic (the run takes 4x longer on 32 threads than on one).
    torch.set_num_threads(1)
    start = time.perf_counter()
    try:
        assert main(["exp", "init", str(root), "--preset", PRESET]) == 0
        for index in DIAGONAL:
            assert main(["exp", "run", str(root), "--index", str(index)]) == 0
    finally:
        torch.set_num_threads(threads)
    seconds = time.perf_counter() - start
    measurements = {index: _measure(root, index) for index in DIAGONAL}
    print(f"\nlinear_ring_tiny through the CLI in {seconds:.0f} s")
    print("\n".join(_report(index, m) for index, m in measurements.items()))
    return measurements


@pytest.mark.parametrize("index", DIAGONAL)
def test_the_estimate_is_the_final_state_of_every_member(smoke, index):
    assert smoke[index].final_epoch == smoke[index].n_epochs


@pytest.mark.parametrize("index", DIAGONAL)
def test_the_sampler_moved_every_member_away_from_its_initial_state(smoke, index):
    m = smoke[index]
    assert m.initial_record_error < 1e-5, "the rebuilt initial state is not the first record's"
    assert m.map_correlation < MAX_MAP_CORRELATION, _report(index, m)


@pytest.mark.parametrize("index", DIAGONAL)
def test_the_standard_errors_are_at_most_a_third_of_the_tolerances(smoke, index):
    m = smoke[index]
    se_max, _ = m.bounds
    assert (m.se <= se_max).all(), _report(index, m)


@pytest.mark.parametrize("index", DIAGONAL)
def test_the_final_state_spectra_agree_with_the_prediction(smoke, index):
    m = smoke[index]
    _, tolerance = m.bounds
    assert ((m.values - m.predicted).abs() <= tolerance).all(), _report(index, m)


@pytest.mark.parametrize("index", DIAGONAL)
def test_the_records_after_the_burn_in_do_not_drift(smoke, index):
    m = smoke[index]
    assert (m.drift.abs() <= DRIFT_Z * m.drift_se).all(), _report(index, m)


@pytest.mark.parametrize("index", DIAGONAL)
def test_at_sigma_zero_the_activity_has_no_null_power(smoke, index):
    assert smoke[index].null_fraction < MAX_NULL_FRACTION


def test_the_two_parametrisations_agree(smoke):
    """Each against its own prediction (the penalty's floor moves R(1)/R(0) by 3e-3), the
    deviations agree within EQUIVALENCE_Z combined standard errors."""
    penalty, whitened = (smoke[index] for index in DIAGONAL)
    gap = (penalty.values - penalty.predicted) - (whitened.values - whitened.predicted)
    assert (gap.abs() <= EQUIVALENCE_Z * (penalty.se**2 + whitened.se**2).sqrt()).all()
