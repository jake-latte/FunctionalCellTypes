"""fct.ring_view without training: the tables of measurements drawn from a known law, the base B
an experiment's budget is computed at, the loader and its record accessors on an untrained ROOT,
the grouping of the band-share sweeps, and the description `exp init` derives. Loading a
trained ROOT is exercised by the notebook's execution (tests/test_build_linear_ring.py)."""

import re
from pathlib import Path

import pytest
import torch

from fct import linear_ring as lr
from fct import ring_view as rv
from fct.cli import main
from fct.exp import (
    ExpConfig,
    derived_description,
    load_exp,
    parse_derived_description,
    preset,
    save_exp,
    with_axes,
)
from fct.spectrum import N_THETA_BINS, mode_spectra, theta_histogram
from tests.spectrum_helpers import gaussian_field

PRESET = "linear_ring_tiny"
WHITENED = 3
B = 64
F64 = torch.float64


@pytest.fixture(scope="module")
def law() -> lr.Law:
    return lr.predicted_law(preset(PRESET), WHITENED)


def _maps(law: lr.Law, n: int, seed: int, D: torch.Tensor | None = None) -> torch.Tensor:
    return gaussian_field(law.D if D is None else D, n, torch.Generator().manual_seed(seed))


def _final(law: lr.Law, seed: int, D: torch.Tensor | None = None, n: int = B) -> torch.Tensor:
    return mode_spectra(law.U @ _maps(law, n, seed, D), law.U)


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


# ------------------------------------------------------------------ tables
def test_summary_table_has_a_row_per_statistic_with_the_measured_and_predicted_values(
    measured, law
):
    table = rv.summary_table(measured, law, base_B=law.B)
    rows = _rows(table)
    p, K = measured.primary, law.D.shape[0]
    assert len(rows) == len(p.stats) + 2 * K  # primary, then q_1/2 and q = 0 shares
    primary = rows[: len(p.stats)]
    assert all(r[2] == "primary" for r in primary)
    assert all(r[2] == "secondary" for r in rows[len(p.stats) :])
    for row, stat, value, se, predicted in zip(
        primary, p.stats, p.values, p.se, p.predicted, strict=True
    ):
        assert row[0] == lr.statistic_name(stat, law.ell) and row[1] == lr.statistic_mode(stat)
        assert float(row[3]) == pytest.approx(float(predicted), abs=5e-5)
        assert float(row[4]) == pytest.approx(float(value), abs=5e-5)
        assert float(row[5]) == pytest.approx(float(se), rel=1e-3, abs=5e-6)
        assert row[7] == "no"  # a draw from the law itself agrees
    assert "not assessed" in table  # no exact statistics given


def test_summary_table_flags_a_disagreement(law):
    wrong = lr.measure_final(law, _final(law, seed=1, D=law.D[[0, 0]]))
    rows = _rows(rv.summary_table(wrong, law, law.B))
    ratio = next(r for r in rows if r[0] == "band power ratio")
    assert ratio[7] == "**yes**"


def test_summary_table_states_the_regime_at_the_base_b(measured, law):
    exact = measured.primary.predicted.clone()
    exact[0] += 1.5 * lr.budget_se(law, law.B)[0]  # outside the budget at base B ...
    rows = _rows(rv.summary_table(measured, law, law.B, exact=exact.tolist()))
    assert rows[0][8] == "no" and all(r[8] == "yes" for r in rows[1 : len(measured.primary.stats)])
    # ... but inside it at a base B a quarter the size (its SE is twice as large)
    rows = _rows(rv.summary_table(measured, law, law.B // 4, exact=exact.tolist()))
    assert rows[0][8] == "yes"
    with pytest.raises(ValueError, match="exact"):
        rv.summary_table(measured, law, law.B, exact=[1.0, 2.0])


def test_summary_table_reports_an_unavailable_q_half_with_its_reason(law):
    everything = law._replace(keep=torch.ones_like(law.keep))
    final = _final(law, seed=2, n=8)
    final[:, :-1, 0] = 1e-9  # every mode rises to Nyquist: no half-power frequency
    m = lr.measure_final(everything, final)
    row = next(r for r in _rows(rv.summary_table(m, everything, law.B)) if r[0] == "q_1/2")
    assert row[4] == "unavailable" and "half-power frequency undefined" in row[7]


@pytest.mark.parametrize(
    "members, verdict",
    [
        (16, f"below N·B threshold (256 < {lr.HEXATIC_MIN_SAMPLES})"),  # N = 16: 16 x 16
        (64, "**yes**"),  # 16 x 64 = 1024 neuron-samples: 0.3 > HEXATIC_MAX is flagged
    ],
)
def test_summary_table_adds_the_hexatic_row_at_k_3(members, verdict):
    tree = preset(PRESET).model_dump()
    tree["linear_teacher"]["n_latents"] = 3
    law3 = lr.predicted_law(ExpConfig.model_validate(tree), WHITENED)
    t = _maps(law3, members, seed=3)
    moments = torch.tensor([[0.3, 0.0]] * members, dtype=F64)
    counts = theta_histogram(t[:, 1], t[:, 2])
    assert counts.shape[-1] == N_THETA_BINS
    m = lr.measure_final(law3, mode_spectra(law3.U @ t, law3.U), moments, counts)
    row = _rows(rv.summary_table(m, law3, law3.B))[-1]
    assert row[0] == "|mean m| (hexatic)" and row[1] == "2, 3"
    assert float(row[4]) == pytest.approx(0.3)
    assert row[7] == verdict


def test_summary_table_has_no_hexatic_row_below_k_3(measured, law):
    assert "hexatic" not in rv.summary_table(measured, law, law.B)


def test_equivalence_table_compares_the_deviations_of_two_paths(measured, law):
    other = lr.measure_final(law, _final(law, seed=5))
    a, b = measured.primary, other.primary
    table = rv.equivalence_table({"penalty": a, "whitened": b}, law.ell)
    rows = _rows(table)
    assert [r[0] for r in rows] == [lr.statistic_name(s, law.ell) for s in a.stats]
    disagree = lr.paths_disagree((a.values, a.se, a.predicted), (b.values, b.se, b.predicted))
    assert [r[-1] for r in rows] == ["**yes**" if d else "no" for d in disagree.tolist()]
    assert f"{len(a.stats)} statistics compared" in table
    with pytest.raises(ValueError, match="two"):
        rv.equivalence_table({"only": a}, law.ell)


def test_relaxation_table_reports_the_deficit_against_the_budget(law):
    R = 8
    records = torch.stack([_final(law, seed=r)[:, :-1] for r in range(R)])
    epochs = torch.linspace(0, law.cfg.train.n_epochs, R, dtype=F64)
    view = lr.measure_relaxation(law, epochs, records, base_B=law.B)
    rows = _rows(rv.relaxation_table(view, law.ell))
    assert [r[0] for r in rows] == [lr.statistic_name(s, law.ell) for s in view.stats]
    for row, deficit, budget, stationary in zip(
        rows, view.fit.deficit, view.budget, view.stationary, strict=True
    ):
        assert float(row[2]) == pytest.approx(float(deficit), abs=5e-5)
        assert float(row[5]) == pytest.approx(float(budget), rel=1e-3)
        assert row[6] == ("yes" if stationary else "**no**")


# ------------------------------------------------------------------ descriptions and base B
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


def test_the_derived_description_round_trips_through_exp_init(tmp_path, capsys):
    """The writer exp init uses and the parser the notebook uses are one module's pair."""
    sets, axes = ["lr=0.2", "B=32"], ["train.beta=[1.0, 10.0]"]
    args = ["exp", "init", str(tmp_path), "--preset", "linear_ring"]
    args += [x for s in sets for x in ("--set", s)] + [x for a in axes for x in ("--axis", a)]
    assert main(args) == 0
    capsys.readouterr()
    written = load_exp(tmp_path).description
    assert written == derived_description("linear_ring", sets, axes)
    assert parse_derived_description(written) == ("linear_ring", sets, axes)
    assert parse_derived_description("my own words") is None
    with pytest.raises(ValueError, match="cannot read the override"):
        parse_derived_description("Preset 'tiny' with overrides: --sett x=1")


def test_an_experiment_with_overrides_quotes_its_presets_description():
    text = rv.regime_markdown(Path("p1"), derived_description("linear_ring", ["lr=0.2"], []))
    assert "made from the preset `linear_ring` with overrides (--set lr=0.2)" in text
    assert "not accurate" in text  # the preset's regime sentence, quoted
    assert preset("linear_ring").description in text


def test_an_experiment_with_its_own_description_quotes_that():
    text = rv.regime_markdown(Path("p1"), "my own words. Accuracy is fine.")
    assert "- Accuracy is fine." in text and "made from" not in text
    assert "(it states no regime)" in rv.regime_markdown(Path("p1"), None)


def _untrained_root(tmp_path: Path, description: str | None = "unchanged") -> Path:
    exp = preset(PRESET)
    if description != "unchanged":
        exp = exp.model_copy(update={"description": description})
    save_exp(exp, tmp_path)
    return tmp_path


def _init(root: Path, *sets: str) -> Path:
    args = ["exp", "init", str(root), "--preset", PRESET]
    assert main(args + [x for item in sets for x in ("--set", item)]) == 0
    return root


def test_an_unextended_experiments_base_b_is_its_b(tmp_path, capsys):
    run = rv.load_ring_settings(_init(tmp_path))[0]
    assert run.base_B == preset(PRESET).B == rv.base_B_of(run)


def test_the_base_b_is_the_b_before_the_first_extension(tmp_path, capsys):
    root = _init(tmp_path, "B=32")
    assert main(["exp", "extend", str(root), "--by", "96"]) == 0
    assert main(["exp", "extend", str(root), "--by", "128"]) == 0
    capsys.readouterr()
    run = rv.load_ring_settings(root)[0]
    assert run.law.B == 256 and rv.base_B_of(run) == 32


def test_the_last_of_repeated_b_overrides_is_the_base(tmp_path, capsys):
    run = rv.load_ring_settings(_init(tmp_path, "B=16", "B=32"))[0]
    assert rv.base_B_of(run) == 32


def test_a_b_axis_gives_each_setting_its_own_base(tmp_path):
    save_exp(with_axes(preset(PRESET), {"net.B": [16, 48]}), tmp_path)
    assert [rv.base_B_of(r) for r in rv.load_ring_settings(tmp_path)] == [16, 48]


def test_a_given_base_b_wins_and_must_be_a_positive_int(tmp_path, capsys):
    run = rv.load_ring_settings(_init(tmp_path, "B=32"))[0]
    assert rv.base_B_of(run, 8) == 8
    for bad in (8.5, 8.0, True, "8"):
        with pytest.raises(TypeError, match="must be an int"):
            rv.base_B_of(run, bad)
    with pytest.raises(ValueError, match="positive"):
        rv.base_B_of(run, 0)


def test_base_b_names_must_be_shown_settings():
    with pytest.raises(KeyError, match="'twn'"):
        rv.measure_runs({}, None, {"twn": 16})


# ------------------------------------------------------------------ the loader
def test_the_loader_reads_every_matching_setting_and_reports_untrained_ones(tmp_path):
    root = _untrained_root(tmp_path)
    runs = rv.load_ring_settings(root)
    assert [r.chosen.setting.index for r in runs] == [0, 1, 2, 3]
    assert not any(r.trained for r in runs) and all(r.records is None for r in runs)
    whitened = rv.load_ring_settings(root, {"spatial.parametrisation": "whitened"})
    assert [r.chosen.setting.index for r in whitened] == [2, 3]
    reused = rv.load_ring_settings(root, None, loaded=whitened)
    assert reused[2] is whitened[0]  # passed in, not read again


def test_the_loader_refuses_an_unknown_axis_and_an_ambiguous_setting(tmp_path):
    root = _untrained_root(tmp_path)
    with pytest.raises(LookupError, match="unknown axis"):
        rv.load_ring_settings(root, {"train.beta": 0.25})
    with pytest.raises(LookupError, match="2 settings"):
        rv.load_ring_setting(root, {"spatial.parametrisation": "whitened"})
    with pytest.raises(LookupError, match="0 settings"):
        rv.load_ring_setting(root, {"train.lr": 0.1})


def test_the_record_accessors_refuse_an_untrained_setting(tmp_path):
    run = rv.load_ring_settings(_untrained_root(tmp_path))[0]
    for accessor in ("final_spectra", "final_latent", "relaxation_records", "relaxation_epochs"):
        with pytest.raises(ValueError, match="not trained"):
            getattr(run, accessor)


def _as_trained(run: rv.RingRun, seed: int, R: int = 3) -> rv.RingRun:
    """`run` with R synthetic records drawn from its law, marked done."""
    records = torch.stack([_final(run.law, seed + r, n=16) for r in range(R)])
    return run._replace(
        chosen=run.chosen._replace(status="done"),
        records=records,
        epochs=torch.arange(R, dtype=F64),
    )


def test_the_record_accessors_split_the_final_state_from_the_relaxation_records(tmp_path):
    run = _as_trained(rv.load_ring_settings(_untrained_root(tmp_path))[3], seed=0, R=4)
    K = run.K
    torch.testing.assert_close(run.final_spectra, run.records[-1])
    torch.testing.assert_close(run.final_latent, run.records[-1, :, :K])
    torch.testing.assert_close(run.relaxation_records, run.records[:-1, :, :K])
    torch.testing.assert_close(run.relaxation_epochs, run.epochs[:-1])


def test_final_member_values_are_told_ensemble_or_single_network():
    ensemble = {"monitor_hex": [[[torch.ones(2)], [2 * torch.ones(2)]]] * 3}
    torch.testing.assert_close(
        rv.final_member_values(ensemble, "monitor_hex", ensemble=True),
        torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
    )
    single = {"monitor_hex": [[torch.ones(2)]]}
    assert rv.final_member_values(single, "monitor_hex", ensemble=False).shape == (1, 2)
    with pytest.raises(ValueError, match="no 'monitor_hex'"):
        rv.final_member_values({}, "monitor_hex", ensemble=True)


# ------------------------------------------------------------------ the sweeps
def test_two_shown_settings_of_one_root_get_one_sweep_each_along_an_axis(tmp_path):
    root = _untrained_root(tmp_path)
    runs = rv.load_ring_settings(root)
    shown = {"setting": runs[0], "twin": runs[3]}  # penalty and whitened, one ROOT
    groups = rv.sweep_groups(shown, "train.lr")
    assert [fixed for _, fixed in groups.values()] == [
        {"spatial.parametrisation": "penalty"},
        {"spatial.parametrisation": "whitened"},
    ]
    assert len(rv.sweep_groups(shown, None)) == 1  # every setting of the ROOT, by index
    with pytest.raises(LookupError, match="unknown sweep axis"):
        rv.sweep_groups(shown, "train.beta")


def test_the_band_share_sweep_is_each_settings_band_share_estimate(tmp_path):
    runs = [_as_trained(r, seed=10 * i) for i, r in enumerate(rv.load_ring_settings(
        _untrained_root(tmp_path), {"spatial.parametrisation": "whitened"}))]  # fmt: skip
    sweep = rv.band_share_sweep(runs, "train.lr")
    assert sweep.values.tolist() == [0.0006, 0.005] and sweep.xlabel == "train.lr"
    for i, run in enumerate(runs):
        values, se, predicted = lr.band_share_estimate(run.law, run.final_latent)
        torch.testing.assert_close(sweep.shares[i], values)
        torch.testing.assert_close(sweep.se[i], se)
        torch.testing.assert_close(sweep.theory[i], predicted)
    assert rv.band_share_sweep(runs, None).values.tolist() == [2.0, 3.0]
    with pytest.raises(TypeError, match="numeric"):
        rv.band_share_sweep(runs, "spatial.parametrisation")


def test_a_sweep_of_untrained_settings_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no trained setting"):
        rv.band_share_sweep(rv.load_ring_settings(_untrained_root(tmp_path)), None)


def test_settings_tables_list_each_distinct_root_once(tmp_path):
    root = _untrained_root(tmp_path)
    text = rv.settings_tables([root, root / ".", Path(str(root))])
    assert text.count("| index |") == 1 and "| 3 | whitened | 0.005 | missing |" in text
