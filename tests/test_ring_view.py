"""fct.ring_view on synthetic inputs: the summary and equivalence tables of measurements drawn from
a known law, the relaxation table, the regime sentences of a description, and the member values
of a history's last record. Loading from a trained ROOT is exercised by the notebook's execution
(tests/test_build_linear_ring.py)."""

import re

import pytest
import torch

from fct import linear_ring as lr
from fct import ring_view as rv
from fct.exp import preset
from fct.spectrum import mode_spectra
from tests.spectrum_helpers import gaussian_field

PRESET = "linear_ring_tiny"
WHITENED = 3
B = 64


@pytest.fixture(scope="module")
def law() -> lr.Law:
    return lr.predicted_law(preset(PRESET), WHITENED)


def _final(law: lr.Law, seed: int, D: torch.Tensor | None = None) -> torch.Tensor:
    t = gaussian_field(law.D if D is None else D, B, torch.Generator().manual_seed(seed))
    return mode_spectra(law.U @ t, law.U)


@pytest.fixture(scope="module")
def measured(law) -> lr.FinalMeasurement:
    return lr.measure_final(law, _final(law, seed=0))


def _rows(table: str) -> list[list[str]]:
    """The body rows of a markdown table, split into cells at the unescaped "|"."""
    lines = [ln for ln in table.splitlines() if ln.startswith("|")]
    return [
        [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", ln[1:-1])]
        for ln in lines[2:]
    ]


def test_summary_table_has_a_row_per_statistic_with_the_measured_and_predicted_values(
    measured, law
):
    table = rv.summary_table(measured, law)
    rows = _rows(table)
    K = law.D.shape[0]
    assert len(rows) == len(measured.stats) + 2 * K  # primary, then q_1/2 and q = 0 shares
    primary = rows[: len(measured.stats)]
    assert all(r[2] == "primary" for r in primary)
    assert all(r[2] == "secondary" for r in rows[len(measured.stats) :])
    for row, value, se, predicted in zip(
        primary, measured.values, measured.se, measured.predicted, strict=True
    ):
        assert float(row[3]) == pytest.approx(float(predicted), abs=5e-5)
        assert float(row[4]) == pytest.approx(float(value), abs=5e-5)
        assert float(row[5]) == pytest.approx(float(se), rel=1e-3, abs=5e-6)
        assert row[7] == "no"  # a draw from the law itself agrees
    assert "not assessed" in table  # no exact statistics given
    assert f"R({law.ell})/R(0)" in table and "q_1/2" in table


def test_summary_table_flags_a_disagreement(law):
    wrong = lr.measure_final(law, _final(law, seed=1, D=law.D[[0, 0]]))
    rows = _rows(rv.summary_table(wrong, law))
    ratio = next(r for r in rows if r[0].startswith("band power ratio"))
    assert ratio[7] == "**yes**"


def test_summary_table_states_the_regime_from_exact_statistics(measured, law):
    predicted = measured.predicted
    exact = predicted.clone()
    exact[0] += 10 * lr.budget_se(law)[0]  # one statistic far outside the budget
    rows = _rows(rv.summary_table(measured, law, exact=exact.tolist()))
    assert rows[0][8] == "no" and all(r[8] == "yes" for r in rows[1 : len(measured.stats)])
    with pytest.raises(ValueError, match="exact"):
        rv.summary_table(measured, law, exact=[1.0, 2.0])


def test_summary_table_adds_the_hexatic_row_when_measured(law):
    m = lr.measure_final(law, _final(law, seed=2))
    moments = torch.tensor([[0.3, 0.0]] * B, dtype=torch.float64)
    m = m._replace(hexatic=lr.hexatic_estimate(moments))
    row = _rows(rv.summary_table(m, law))[-1]
    assert row[0].startswith("|mean m|") and float(row[4]) == pytest.approx(0.3)
    assert row[7] == "**yes**"  # 0.3 > HEXATIC_MAX


def test_equivalence_table_compares_the_deviations_of_two_paths(measured, law):
    other = lr.measure_final(law, _final(law, seed=5))
    table = rv.equivalence_table({"penalty": measured, "whitened": other})
    rows = _rows(table)
    assert len(rows) == len(measured.stats)
    disagree = lr.paths_disagree(
        (measured.values, measured.se, measured.predicted),
        (other.values, other.se, other.predicted),
    )
    assert [r[-1] for r in rows] == ["**yes**" if d else "no" for d in disagree.tolist()]
    assert f"{len(measured.stats)} statistics compared" in table


def test_equivalence_table_needs_two_measurements(measured):
    with pytest.raises(ValueError, match="two"):
        rv.equivalence_table({"only": measured})


def test_relaxation_table_reports_the_deficit_against_the_budget(law):
    R = 8
    records = torch.stack([_final(law, seed=r)[:, :-1] for r in range(R)])
    epochs = torch.linspace(0, law.cfg.train.n_epochs, R, dtype=torch.float64)
    view = lr.measure_relaxation(law, epochs, records)
    rows = _rows(rv.relaxation_table(view))
    assert [r[0] for r in rows] == view.labels
    for row, deficit, budget, stationary in zip(
        rows, view.fit.deficit, view.budget, view.stationary, strict=True
    ):
        assert float(row[1]) == pytest.approx(float(deficit), abs=5e-5)
        assert float(row[4]) == pytest.approx(float(budget), rel=1e-3)
        assert row[5] == ("yes" if stationary else "**no**")


def test_regime_sentences_pick_the_sentences_on_accuracy_and_bias():
    text = (
        "The base point. Measured, lr 0.96 shifts R(1)/R(0) by -1.2 SE. That bias alone exceeds "
        "the one-SE accuracy budget, so the regime is not accurate at this lr. Cost: 2 hours."
    )
    assert rv.regime_sentences(text) == [
        (
            "That bias alone exceeds the one-SE accuracy budget, so the regime is not accurate "
            "at this lr."
        )
    ]
    assert rv.regime_sentences(None) == []


def test_the_linear_ring_description_states_the_lr_bias_on_mode_1():
    """The notebook quotes this sentence instead of restating its numbers."""
    sentences = rv.regime_sentences(preset("linear_ring").description)
    assert any("not accurate" in s for s in sentences)


def test_final_member_values_stack_the_members_of_the_last_record():
    ensemble = {"monitor_hex": [[[torch.ones(2)], [2 * torch.ones(2)]]] * 3}
    torch.testing.assert_close(
        rv.final_member_values(ensemble, "monitor_hex"),
        torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
    )
    single = {"monitor_hex": [[torch.ones(2)]]}
    assert rv.final_member_values(single, "monitor_hex").shape == (1, 2)
    assert rv.final_member_values({}, "monitor_hex") is None
