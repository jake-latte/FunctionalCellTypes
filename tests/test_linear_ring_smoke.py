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
spectrum and their errors the bootstrap over members (fct.linear_ring; the thresholds are
checked on the predicted law itself in tests/test_linear_ring_estimator.py). The thresholds are
the pipeline rule (`fct.linear_ring.pipeline_bounds`), deliberately not the experiments' rule
(`disagrees` with its follow-up at FOLLOW_UP_FACTOR x B): a pipeline test must not depend on a
follow-up run. The q = 0 shares and q_1/2 are printed, not asserted. The records before the
last serve only the relaxation diagnostic (`relaxation_fit` of the second half of the run, its
deficit consistent with zero), which is weak by construction here: chains started in their
stationary law show no relaxation whether or not they mix.

Both diagonal settings run, each at its own lr: setting 0 (penalty) and setting 3 (whitened).
The run takes about a minute on one thread and is marked slow.
"""

import time
from typing import NamedTuple

import pytest
import torch

from fct import linear_ring as lr
from fct.cli import main
from fct.exp import expand, load_exp, preset, setting_dir
from fct.monitor import spectrum_records
from fct.spectrum import half_power_q, mode_spectra, tuning_maps
from fct.train import MONITOR_EPOCH, load_trained, make_net_from_config

PRESET = "linear_ring_tiny"
DIAGONAL = {0: "penalty", 3: "whitened"}
# Per member, the correlation of independent maps averages 0. Its spread is about
# 1 / sqrt(K x n_eff): on the smoke ring (ell = 2 sites, N = 16) lambda_q / lambda_0 = 1, 0.73,
# 0.29 at |q| = 0, 1, 2, so about three frequencies (q = 0, +-1) carry most of each map's power
# and the spread is 1 / sqrt(2 x 3) = 0.4, the member mean 0 +- 0.03 at B = 256; a frozen
# sampler gives 1. A mean below 0.1 needs at least 90% decorrelation.
MAX_MAP_CORRELATION = 0.1
# The rebuilt initial spectra against the first record: the record is stored in float32
# (relative precision 6e-8), and summing over patterns and the FFT leave well under 1e-6.
MAX_INITIAL_RECORD_ERROR = 1e-5
# At sigma = 0, x lies in the column space of Z and so does h: the null row is round-off of
# float32 records relative to the latent power (about 1e-14 measured), far below 1e-10.
MAX_NULL_FRACTION = 1e-10
CPU, F64 = torch.device("cpu"), torch.float64

pytestmark = pytest.mark.slow


class Measurement(NamedTuple):
    stats: list[lr.Statistic]
    predicted: torch.Tensor  # (3K + 1,) the statistics of the predicted spectrum
    values: torch.Tensor  # (3K + 1,) of the member-averaged final-state spectrum
    se: torch.Tensor  # (3K + 1,) bootstrap over members
    bounds: tuple[torch.Tensor, torch.Tensor]  # pipeline (largest SE, tolerance), run's N, B
    shares: torch.Tensor  # (K,) q = 0 shares of the member-averaged spectrum (reported)
    shares_predicted: torch.Tensor  # (K,)
    q_half: list[int]  # of the member-averaged spectrum (reported)
    q_half_predicted: list[int]
    relaxation: lr.RelaxationFit  # of the records before the final one
    map_correlation: float  # member mean of the initial-final tuning-map correlation
    initial_record_error: float  # rebuilt initial spectra against the first record, relative
    null_fraction: float  # the null row's power over the latent rows', final state
    final_epoch: int
    n_epochs: int


def _maps(net, law: lr.Law) -> torch.Tensor:
    """(B, K, N) tuning maps of every member of an ensemble on the setting's data."""
    with torch.no_grad():
        H = net(law.dataset.x)[0]  # (B, P, L, N)
    return tuning_maps(H[:, :, 0], law.U)


def _measure(root, index: int) -> Measurement:
    exp = load_exp(root)
    law = lr.predicted_law(preset(PRESET), index)
    K = law.D.shape[0]
    net, history = load_trained(setting_dir(root, expand(exp)[index]), law.cfg, CPU, F64)
    records = spectrum_records(history)[:, :, 0].to(F64)  # (R, B, K + 1, N), one hidden layer
    final = records[-1, :, :K]  # one sample per member: its final state
    values, se = lr.estimate(final, law)
    epochs = history[MONITOR_EPOCH]
    # The diagnostic uses the records before the final one, so it does not reuse the sample
    # whose stationarity it vouches for.
    relaxation = lr.relaxation_fit(
        torch.tensor(epochs[:-1]),
        records[:-1, :, :K],
        lr.law_statistics(law),
        start=law.cfg.train.n_epochs / 2,
    )
    x, y = law.dataset.x, law.dataset.y
    initial = make_net_from_config(law.cfg, x.shape[1], y.shape[1], CPU, F64)  # same seed
    first = mode_spectra(initial(x)[0][:, :, 0].detach(), law.U)[:, :K]
    mean = final.mean(0)
    return Measurement(
        stats=lr.statistic_list(K),
        predicted=lr.law_statistics(law)(law.D),
        values=values,
        se=se,
        bounds=lr.pipeline_bounds(law),
        shares=mean[:, 0] / mean.sum(-1),
        shares_predicted=law.D[:, 0] / law.D.sum(-1),
        q_half=torch.as_tensor(half_power_q(mean)).tolist(),
        q_half_predicted=torch.as_tensor(half_power_q(law.D)).tolist(),
        relaxation=relaxation,
        map_correlation=float(lr.map_correlation(_maps(initial, law), _maps(net, law)).mean()),
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
    fit = m.relaxation
    for i, stat in enumerate(m.stats):
        name = stat.kind if stat.mode is None else f"{stat.kind}_{stat.mode}"
        rows.append(
            f"  {name:7s} predicted {m.predicted[i]:.4f} measured {m.values[i]:.4f} "
            f"+- {m.se[i]:.4f} (tolerance {tolerance[i]:.4f}) "
            f"remaining relaxation {fit.deficit[i]:+.4f} +- {fit.deficit_se[i]:.4f}"
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
    assert m.initial_record_error < MAX_INITIAL_RECORD_ERROR, "not the first record's state"
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
def test_the_second_half_of_the_run_shows_no_relaxation_beyond_its_noise(smoke, index):
    """The fitted deficit of the second half is zero within PIPELINE_N_SE of its own SEs. This
    is weaker than `is_stationary`: with 20 records in the window, the deficit's SE is 0.63 to
    0.87 of one SE at the run's B on the penalty path and 0.23 to 0.37 on the whitened path
    (where the window-mean floor binds), so the diagnostic cannot reliably show the deficit
    below that SE, which `is_stationary` demands of a real run (there with an SE of about 0.1
    of the budget)."""
    fit = smoke[index].relaxation
    within = fit.deficit.abs() <= lr.PIPELINE_N_SE * fit.deficit_se
    assert within.all(), _report(index, smoke[index])


@pytest.mark.parametrize("index", DIAGONAL)
def test_at_sigma_zero_the_activity_has_no_null_power(smoke, index):
    assert smoke[index].null_fraction < MAX_NULL_FRACTION


def test_the_two_parametrisations_agree(smoke):
    """The experiments' equivalence rule (`fct.linear_ring.paths_disagree`), each path against
    its own prediction (the floored frequencies are excluded from both)."""
    penalty, whitened = (smoke[index] for index in DIAGONAL)
    disagree = lr.paths_disagree(
        (penalty.values, penalty.se, penalty.predicted),
        (whitened.values, whitened.se, whitened.predicted),
    )
    assert not disagree.any()
