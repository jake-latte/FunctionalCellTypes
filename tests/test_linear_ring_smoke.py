"""End-to-end smoke test: `linear_ring_tiny` trained through the CLI on the CPU, its final-state
spectra against `fct.linear_theory.predicted_spectra`.

The estimator is per member: each of the B = 256 members contributes one sample, its final
state (the last record of the spectrum monitor, taken after the last step); the statistics are
of the member-averaged spectrum and their standard errors are the bootstrap over members. The
records before the final one serve only the burn-in diagnostic: per member, the time average
of each statistic over the second half of the records after burn-in minus that over the first
half, averaged across members, must be zero within four standard errors. The statistics, their
standard errors at B = 256 and the tolerances (three times the largest accepted SE) are
derived in tests/linear_ring_helpers.py and checked on the predicted law itself in
tests/test_linear_ring_estimator.py before being applied here.

Both diagonal settings run, each at its own lr: setting 0 (penalty, lr 6e-4) and setting 3
(whitened, lr 0.005). At beta = 0.25 the prediction is within 0.11 SE of the prior alone, so
this test checks the sampler, the two parametrisations and the measurement end to end, not
the data term of the theory.
"""

import time
from typing import NamedTuple

import pytest
import torch

from fct.cli import main
from fct.data import make_dataset
from fct.exp import expand, load_exp, preset, setting_config, setting_dir
from fct.linear_theory import predicted_spectra
from fct.monitor import spectrum_records
from fct.spectrum import bootstrap_members, half_power_q
from fct.train import MONITOR_EPOCH, load_trained
from tests.linear_ring_helpers import (
    N_BOOT,
    SE_MAX,
    TOLERANCE,
    estimate,
    half_power,
    kind_of,
    ring_statistics,
    statistic_names,
    time_drift,
)

PRESET = "linear_ring_tiny"
DIAGONAL = {0: "penalty", 3: "whitened"}
BURN_IN_EPOCHS = 2000  # two relaxation times of the penalty's softest data-carrying mode
DRIFT_Z = 4.0
MAX_SECONDS = 120.0
CPU, F64 = torch.device("cpu"), torch.float64


class Measurement(NamedTuple):
    names: list[str]
    predicted: torch.Tensor  # (3K + 1,) the statistics of the predicted spectrum
    values: torch.Tensor  # (3K + 1,) of the member-averaged final-state spectrum
    se: torch.Tensor  # (3K + 1,) bootstrap over members
    q_half_predicted: torch.Tensor  # (K,)
    q_half: torch.Tensor  # (K,)
    q_half_se: torch.Tensor  # (K,) bootstrap over members
    drift: torch.Tensor  # (3K + 1,) second-half minus first-half time average, member mean
    drift_se: torch.Tensor  # (3K + 1,)
    null_fraction: float  # the null row's power over the latent rows', final state
    final_epoch: int


def _measure(root, index: int) -> Measurement:
    exp = load_exp(root)
    setting = expand(exp)[index]
    cfg = setting_config(exp, setting)
    dataset = make_dataset(cfg.data, CPU, F64)
    D = torch.from_numpy(predicted_spectra(cfg, dataset).d)
    K = D.shape[0]
    _, history = load_trained(setting_dir(root, setting), cfg, CPU, F64)
    records = spectrum_records(history)[:, :, 0].to(F64)  # (R, B, K + 1, N), one hidden layer
    final = records[-1, :, :K]  # one sample per member: its final state
    values, se = estimate(final, D)
    _, q_half_se = bootstrap_members(final, half_power_q, N_BOOT)
    epochs = history[MONITOR_EPOCH]
    burn_in = next(i for i, epoch in enumerate(epochs) if epoch >= BURN_IN_EPOCHS)
    drift, drift_se = time_drift(records[:-1, :, :K], burn_in, D)
    return Measurement(
        names=statistic_names(K),
        predicted=ring_statistics(D)(D),
        values=values,
        se=se,
        q_half_predicted=torch.as_tensor(half_power_q(D)),
        q_half=half_power(final),
        q_half_se=q_half_se,
        drift=drift,
        drift_se=drift_se,
        null_fraction=float(records[-1, :, K].sum() / records[-1, :, :K].sum()),
        final_epoch=epochs[-1],
    )


def _report(index: int, m: Measurement) -> str:
    q_half_se = [round(x, 3) for x in m.q_half_se.tolist()]
    header = (
        f"setting {index} ({DIAGONAL[index]}): q_1/2 {m.q_half.tolist()} "
        f"(predicted {m.q_half_predicted.tolist()}, bootstrap SE {q_half_se})"
    )
    rows = [header]
    for i, name in enumerate(m.names):
        rows.append(
            f"  {name:8s} predicted {m.predicted[i]:.4f} measured {m.values[i]:.4f} "
            f"+- {m.se[i]:.4f} (tolerance {TOLERANCE[kind_of(name)]:.4f}) "
            f"drift {m.drift[i]:+.4f} +- {m.drift_se[i]:.4f}"
        )
    return "\n".join(rows)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory) -> tuple[dict[int, Measurement], float]:
    """Run `exp init --preset linear_ring_tiny` and `exp run --index` on both diagonal settings,
    and measure each; returns the measurements and the wall time of the CLI calls."""
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
    return measurements, seconds


def test_the_smoke_runs_in_under_two_minutes_on_a_cpu(smoke):
    _, seconds = smoke
    assert seconds < MAX_SECONDS


@pytest.mark.parametrize("index", DIAGONAL)
def test_the_estimate_is_the_final_state_of_every_member(smoke, index):
    assert smoke[0][index].final_epoch == preset(PRESET).n_epochs


@pytest.mark.parametrize("index", DIAGONAL)
def test_the_standard_errors_are_at_most_a_third_of_the_tolerances(smoke, index):
    m = smoke[0][index]
    se_max = torch.tensor([SE_MAX[kind_of(n)] for n in m.names], dtype=F64)
    assert (m.se <= se_max).all(), _report(index, m)


@pytest.mark.parametrize("index", DIAGONAL)
def test_the_final_state_spectra_agree_with_the_theory(smoke, index):
    m = smoke[0][index]
    tolerance = torch.tensor([TOLERANCE[kind_of(n)] for n in m.names], dtype=F64)
    assert ((m.values - m.predicted).abs() <= tolerance).all(), _report(index, m)
    assert torch.equal(m.q_half, m.q_half_predicted), _report(index, m)


@pytest.mark.parametrize("index", DIAGONAL)
def test_the_chains_are_stationary_after_the_burn_in(smoke, index):
    m = smoke[0][index]
    assert (m.drift.abs() <= DRIFT_Z * m.drift_se).all(), _report(index, m)


@pytest.mark.parametrize("index", DIAGONAL)
def test_at_sigma_zero_the_activity_has_no_null_power(smoke, index):
    """x = Z W_X^T lies in the column space of Z, so h does too, and the null row is round-off
    (relative to the float32 records)."""
    assert smoke[0][index].null_fraction < 1e-10


def test_the_two_parametrisations_agree(smoke):
    """Each against its own prediction (the penalty's floor moves R(1)/R(0) by 3e-3), the
    deviations agree within four combined standard errors."""
    penalty, whitened = (smoke[0][index] for index in DIAGONAL)
    gap = (penalty.values - penalty.predicted) - (whitened.values - whitened.predicted)
    assert (gap.abs() <= DRIFT_Z * (penalty.se**2 + whitened.se**2).sqrt()).all()
