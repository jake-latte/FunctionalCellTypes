"""Notebook report: final-MIR table/heatmap on known values, prose from numbers, and the
loader's handling of missing and partial settings on a real tiny sweep."""

import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from fct.config import Config, DataConfig, NetConfig, TrainConfig
from fct.data import make_dataset
from fct.draw import fig_mir_heatmaps, plot_mir_heatmap
from fct.monitor import make_mir_monitor
from fct.report import (
    SettingResult,
    final_mir,
    find_result,
    load_results,
    mir_table_markdown,
    overview_text,
    setting_items,
    setting_label,
    setting_text,
    summary_text,
    sweep_root_from_env,
)
from fct.sweep import (
    LAYERS,
    Setting,
    expand,
    preset,
    run_setting,
    save_sweep,
    setting_dir,
)

QUIET = {"log": lambda s: None}
TASKS = ("linear_teacher", "object_action", "what_where")


# ---------------------------------------------------------------- known final values
def known(setting: Setting, layer: int, kind: str) -> float:
    """Final MIR of `layer` (0-based) of `setting`: unique per (setting, layer, kind) and exact
    in binary, so the table can be compared with ==."""
    index = expand(preset("tiny")).index(setting)
    offset = {"raw": 0.5, "corrected": 0.25, "chance": 0.125}[kind]
    return offset + index / 128 + layer / 1024


def synthetic_history(setting: Setting) -> dict:
    """A history whose FIRST monitor call has different values, so only the last is final."""

    def column(kind: str) -> list[list[float]]:
        first = [-1.0] * setting.L
        last = [known(setting, layer, kind) for layer in range(setting.L)]
        return [first, last]

    return {
        "monitor_epoch": [0, 6],
        "monitor_mir": column("raw"),
        "monitor_mir_corrected": column("corrected"),
        "monitor_mir_chance": column("chance"),
        "monitor_n_dead": [[0] * setting.L, [1] * setting.L],
        "monitor_n_dead_corrected": [[0] * setting.L, [2] * setting.L],
    }


def synthetic_results(missing=()) -> list[SettingResult]:
    out = []
    for s in expand(preset("tiny")):
        if s in missing:
            out.append(SettingResult(s, "missing", Path("nowhere")))
        else:
            out.append(SettingResult(s, "done", Path("here"), history=synthetic_history(s)))
    return out


def test_final_mir_equals_known_values_for_every_setting():
    results = synthetic_results()
    for task in TASKS:
        fm = final_mir(results, task)
        assert fm.task == task
        assert len(fm.labels) == 12 and fm.raw.shape == (12, max(LAYERS))
        settings = [r.setting for r in results if r.setting.task == task]
        for row, s in enumerate(settings):
            for layer in range(max(LAYERS)):
                for kind, table in (
                    ("raw", fm.raw),
                    ("corrected", fm.corrected),
                    ("chance", fm.chance),
                ):
                    if layer < s.L:
                        assert table[row, layer] == known(s, layer, kind)
                    else:  # the setting has no such layer
                        assert np.isnan(table[row, layer])


def test_markdown_table_prints_the_known_numbers():
    results = synthetic_results()
    text = mir_table_markdown(final_mir(results, "linear_teacher"))
    first = results[0].setting
    assert setting_label(first) in text
    for kind in ("raw", "corrected", "chance"):
        assert f"{known(first, 0, kind):.3f}" in text
    assert text.count("\n") >= 12 + 1  # header, rule and one row per setting


def test_summary_text_names_the_extremes_from_the_numbers():
    results = synthetic_results()
    settings = [r.setting for r in results if r.setting.task == "object_action"]
    cells = [(known(s, layer, "corrected"), s, layer) for s in settings for layer in range(s.L)]
    best, worst = max(cells), min(cells)
    text = summary_text(final_mir(results, "object_action"))
    assert "12 of 12" in text
    assert f"{best[0]:.3f}" in text and setting_label(best[1]) in text
    assert f"{worst[0]:.3f}" in text and setting_label(worst[1]) in text


def test_missing_settings_are_nan_rows_and_named_in_text():
    gone = expand(preset("tiny"))[1]
    fm = final_mir(synthetic_results(missing=[gone]), gone.task)
    assert np.isnan(fm.raw[1]).all()
    assert fm.status[1] == "missing"
    assert setting_label(gone) + " (missing)" in mir_table_markdown(fm)
    text = summary_text(fm)
    assert "11 of 12" in text and setting_label(gone) in text


def test_summary_text_with_nothing_done_says_so_instead_of_crashing():
    every = [s for s in expand(preset("tiny")) if s.task == "what_where"]
    text = summary_text(final_mir(synthetic_results(missing=every), "what_where"))
    assert "0 of 12" in text and "no finished setting" in text


def test_final_mir_rejects_a_done_setting_without_monitor_history():
    s = expand(preset("tiny"))[0]
    bad = SettingResult(s, "done", Path("x"), history={"epoch": [0]})
    with pytest.raises(ValueError, match="monitor"):
        final_mir([bad], s.task)


def test_final_mir_rejects_a_task_without_settings():
    with pytest.raises(ValueError, match="no settings"):
        final_mir(synthetic_results(), "position")


def test_heatmap_cells_hold_the_known_values():
    results = synthetic_results(missing=[expand(preset("tiny"))[3]])
    fm = final_mir(results, "linear_teacher")
    fig, ax = plt.subplots()
    assert plot_mir_heatmap(fm.raw, fm.labels, fm.status, "raw", ax=ax) is ax
    shown = np.ma.filled(ax.images[0].get_array().astype(float), np.nan)
    np.testing.assert_array_equal(shown, fm.raw)
    assert [t.get_text() for t in ax.get_yticklabels()] == fm.labels
    assert "missing" in [t.get_text() for t in ax.texts]  # the unfinished row is labelled
    plt.close(fig)


def test_heatmap_refuses_to_clip_values_outside_the_colour_range():
    with pytest.raises(ValueError, match="clipped"):
        plot_mir_heatmap(np.array([[1.5]]), ["a"], ["done"])
    with pytest.raises(ValueError, match="do not match"):
        plot_mir_heatmap(np.zeros((2, 2)), ["a"], ["done"])


def test_fig_mir_heatmaps_scale_covers_values_above_one_and_below_zero():
    wide = np.array([[1.25, -0.5]])
    fig = fig_mir_heatmaps({"a": wide, "b": wide}, ["row"], ["done"])
    for ax in (a for a in fig.axes if a.images):
        assert ax.images[0].get_clim() == (-0.5, 1.25)
    plt.close(fig)


def test_fig_mir_heatmaps_has_a_panel_per_name_and_one_colourbar():
    results = synthetic_results()
    fm = final_mir(results, "linear_teacher")
    panels = {"raw": fm.raw, "corrected": fm.corrected, "chance baseline": fm.chance}
    fig = fig_mir_heatmaps(panels, fm.labels, fm.status)
    assert [ax.get_title() for ax in fig.axes if ax.images] == list(panels)
    assert len(fig.axes) == len(panels) + 1  # the panels and a single colourbar
    plt.close(fig)


# ---------------------------------------------------------------- hand-set modular network
class FixedRates:
    """A stand-in net returning fixed hidden pre-activations, as `net(x)` does."""

    def __init__(self, hidden):
        self.hidden = hidden

    def __call__(self, x):
        return self.hidden, None


def _modular_hidden(dataset, L: int, N: int) -> torch.Tensor:
    """(P, L, N) pre-activations where unit n is a monotone function of exactly one factor
    (alternating over the factors), so each neuron is informative about one group only."""
    factors = list(dataset.factors.values())
    cols = []
    for n in range(N):
        f = factors[n % len(factors)].double()
        cols.append(torch.atanh(0.9 * (f - f.mean()) / (f.max() - f.min())))
    return torch.stack(cols, dim=1).unsqueeze(1).repeat(1, L, 1)


def test_hand_set_modular_net_scores_high_through_the_real_monitor_to_the_summary():
    """Ground truth: a modular net gets corrected MIR near 1 and a random net near 0 from the
    real monitor, and the table and summary report the monitor's numbers."""
    L, N, P = 3, 8, 512
    data = DataConfig(task="what_where", P=P, D_in=81, n_motifs=4, seed=3)
    cfg = Config(
        data=data,
        net=NetConfig(N=N, L=L, phi="tanh", seed=1),
        train=TrainConfig(monitor_bins=10, monitor_shuffles=5, seed=5),
        dtype="float64",
        device="cpu",
    )
    dataset = make_dataset(data, torch.device("cpu"), torch.float64)
    monitor = make_mir_monitor(dataset, cfg)
    generator = torch.Generator().manual_seed(0)
    nets = {
        Setting("what_where", L, "tanh", 0.0): _modular_hidden(dataset, L, N),
        Setting("what_where", L, "tanh", 1e-3): torch.randn(P, L, N, generator=generator).double(),
    }

    def result(setting, hidden):
        out = monitor(FixedRates(hidden))
        history = {"monitor_epoch": [0], **{f"monitor_{k}": [v] for k, v in out.items()}}
        return SettingResult(setting, "done", Path("x"), history=history)

    modular = next(iter(nets))
    fm = final_mir([result(s, h) for s, h in nets.items()], "what_where")
    assert (fm.corrected[0, :L] > 0.8).all(), fm.corrected[0]
    assert (fm.corrected[1, :L] < 0.2).all(), fm.corrected[1]
    assert (fm.raw[0, :L] > fm.chance[0, :L]).all()  # beats its own shuffled baseline
    assert f"{fm.corrected[0, 0]:.3f}" in mir_table_markdown(fm)
    text = summary_text(fm)
    assert f"{fm.corrected[0, :L].max():.3f}" in text and setting_label(modular) in text


# ---------------------------------------------------------------- the root parameter
def test_sweep_root_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FCT_SWEEP_ROOT", str(tmp_path))
    assert sweep_root_from_env() == tmp_path
    monkeypatch.delenv("FCT_SWEEP_ROOT")
    with pytest.raises(RuntimeError, match="FCT_SWEEP_ROOT"):
        sweep_root_from_env()


# ---------------------------------------------------------------- a real tiny sweep, partly run
def _index(task: str, L: int, phi: str, rate: float) -> int:
    return expand(preset("tiny")).index((task, L, phi, rate))


@pytest.fixture(scope="module")
def partial_root(tmp_path_factory) -> Path:
    """A tiny sweep of which only two of the 36 settings were run."""
    root = tmp_path_factory.mktemp("sweep")
    sweep = preset("tiny")
    save_sweep(sweep, root)
    run_setting(sweep, _index("linear_teacher", 1, "tanh", 0.0), root, **QUIET)
    run_setting(sweep, _index("object_action", 3, "relu", 1e-3), root, **QUIET)
    return root


def test_load_results_marks_every_setting_missing_in_an_empty_sweep(tmp_path):
    save_sweep(preset("tiny"), tmp_path)
    results = load_results(tmp_path)
    assert len(results) == 36
    assert {r.status for r in results} == {"missing"}
    assert all(r.history is None for r in results)


def test_load_results_reads_the_finished_settings(partial_root):
    done = [r for r in load_results(partial_root) if r.status == "done"]
    assert [(r.setting.task, r.setting.L) for r in done] == [
        ("linear_teacher", 1),
        ("object_action", 3),
    ]
    for r in done:
        assert r.metrics and r.examples is not None and r.data_cfg.task == r.setting.task
        assert len(r.history["monitor_mir"][-1]) == r.setting.L


def test_setting_items_for_done_setting_are_text_then_three_figures(partial_root):
    r = next(r for r in load_results(partial_root) if r.status == "done")
    items = setting_items(r)
    texts = [i for i in items if isinstance(i, str)]
    figs = [i for i in items if not isinstance(i, str)]
    assert len(figs) == 3 and texts and setting_label(r.setting) in texts[0]
    for fig in figs:
        plt.close(fig)


def test_setting_items_for_a_missing_setting_is_one_visible_message(partial_root):
    r = next(r for r in load_results(partial_root) if r.status == "missing")
    (message,) = setting_items(r)
    assert "missing" in message and setting_label(r.setting) in message


def test_interrupted_setting_is_reported_not_loaded(tmp_path):
    sweep = preset("tiny")
    save_sweep(sweep, tmp_path)
    run_setting(sweep, 0, tmp_path, **QUIET)
    # an interrupted run keeps its checkpoints but never wrote final.pt
    (setting_dir(tmp_path, expand(sweep)[0]) / "train" / "final.pt").unlink()
    r = load_results(tmp_path)[0]
    assert r.status == "partial" and r.history is None
    (message,) = setting_items(r)
    assert "partial" in message and "no final.pt" in message


def test_summary_over_a_partial_sweep_counts_the_finished_settings(partial_root):
    fm = final_mir(load_results(partial_root), "linear_teacher")
    assert "1 of 12" in summary_text(fm)


def test_overview_counts_each_status():
    gone = [expand(preset("tiny"))[0], expand(preset("tiny"))[5]]
    text = overview_text(synthetic_results(missing=gone))
    assert "36 settings: 34 done, 0 partial" in text and "2 missing" in text


def test_setting_text_reports_the_final_numbers_of_a_done_setting():
    s = expand(preset("tiny"))[2]
    history = {**synthetic_history(s), "epoch": [0, 4], "loss": [3.0, 1.25], "mse": [1.0, 0.5]}
    result = SettingResult(s, "done", Path("here"), history=history, metrics={"r2_pooled": 0.5})
    text = setting_text(result)
    for kind in ("raw", "corrected", "chance"):
        assert f"{known(s, 0, kind):.3f}" in text
    assert "1.25" in text and "epoch (4)" in text and "epoch 6" in text
    assert "r2_pooled = 0.500" in text


def test_summary_text_survives_a_task_whose_corrected_mir_is_all_nan():
    fm = final_mir(synthetic_results(), "linear_teacher")
    fm = dataclasses.replace(fm, corrected=np.full_like(fm.corrected, np.nan))
    assert "No finite corrected MIR" in summary_text(fm)


def test_summary_text_states_a_tie_and_pools_only_cells_with_all_three_values():
    fm = final_mir(synthetic_results(), "linear_teacher")
    corrected = fm.corrected.copy()
    corrected[0, 0] = corrected[1, 0] = 0.9  # the same, highest value in two cells
    raw = fm.raw.copy()
    raw[2, 0] = np.nan  # one cell without a raw value is left out of every mean
    fm = dataclasses.replace(fm, corrected=corrected, raw=raw)
    text = summary_text(fm)
    assert "highest (0.900) tied across 2 layer-cells" in text
    n_cells = int((np.isfinite(raw) & np.isfinite(corrected) & np.isfinite(fm.chance)).sum())
    assert f"Mean over the {n_cells} layer-cells" in text
    pooled = np.isfinite(raw) & np.isfinite(corrected) & np.isfinite(fm.chance)
    assert f"corrected {corrected[pooled].mean():.3f}" in text


def test_setting_text_names_a_missing_history_key():
    s = expand(preset("tiny"))[0]
    history = synthetic_history(s)  # has the monitor keys but no loss curve
    with pytest.raises(ValueError, match=r"history lacks \['epoch', 'loss', 'mse'\]"):
        setting_text(SettingResult(s, "done", Path("x"), history=history, metrics={}))


def test_find_result_returns_the_named_setting_and_rejects_unknown_ones():
    results = synthetic_results()
    wanted = results[7].setting
    assert find_result(results, *wanted) is results[7]
    with pytest.raises(ValueError, match="no result"):
        find_result(results, "linear_teacher", 2, "tanh", 0.0)


def test_loading_does_not_write_into_root_and_reports_a_missing_dataset(tmp_path):
    sweep = preset("tiny")
    save_sweep(sweep, tmp_path)
    run_setting(sweep, 0, tmp_path, **QUIET)
    dataset_file = setting_dir(tmp_path, expand(sweep)[0]) / "data" / "dataset.pt"
    dataset_file.unlink()
    before = sorted(p for p in tmp_path.rglob("*"))
    r = load_results(tmp_path)[0]
    assert sorted(p for p in tmp_path.rglob("*")) == before and not dataset_file.exists()
    assert r.status == "partial" and "dataset.pt" in setting_items(r)[0]
    assert r.directory == Path("linear_teacher/L1_tanh_r0")  # relative to ROOT


# ----------------------------------------------------------------- the MIR explanation


def test_explanation_defines_every_metric_and_names_the_functions():
    from fct.report import MIR_EXPLANATION

    for term in (
        "raw",
        "chance baseline",
        "corrected",
        "dead neuron",
        "preference",
        "rank bin",
        "plug-in",
        "(B-1)(C-1)",
        "n_shuffles",
        "layer_mir_report",
        "shuffled_mir",
        "rank_bins",
        "mutual_information",
        "preference",
        "dead-neuron",
        "monitor_shuffles",
    ):
        assert term in MIR_EXPLANATION, term


def test_explanation_equations_are_balanced():
    import re

    from fct.report import MIR_EXPLANATION

    text = MIR_EXPLANATION
    assert text.count("$$") % 2 == 0
    display = re.findall(r"\$\$(.+?)\$\$", text, flags=re.DOTALL)
    assert len(display) >= 8
    prose = re.sub(r"\$\$.+?\$\$", "", text, flags=re.DOTALL)
    assert prose.count("$") % 2 == 0  # every inline equation is closed
    inline = re.findall(r"\$(.+?)\$", prose)
    assert inline
    for equation in display + inline:
        for opening, closing in ("{}", "[]"):
            assert equation.count(opening) == equation.count(closing), equation
        for size in ("\\big", "\\Big"):
            assert equation.count(f"{size}(") == equation.count(f"{size})"), equation
