"""Showing a setting that is still running: the loader reads its latest checkpoint read-only,
and the table, heatmap and summary keep provisional numbers apart from final ones."""

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

import fct.config
import fct.train as train_module
from fct.cache import cache_paths, write_checkpoint
from fct.config import resolve_dtype
from fct.data import get_dataset
from fct.draw import fig_mir_heatmaps
from fct.exp import (
    Setting,
    expand,
    load_exp,
    preset,
    run_setting,
    save_exp,
    setting_config,
    setting_dir,
)
from fct.model import make_net
from fct.report import (
    SettingResult,
    final_mir,
    load_results,
    load_setting_result,
    mir_table_markdown,
    overview_text,
    setting_items,
    setting_label,
    setting_text,
    summary_text,
    task_items,
)
from fct.train import _new_history, _payload
from tests.report_helpers import known, synthetic_history, synthetic_results

QUIET = {"log": lambda s: None}
FIRST = expand(preset("tiny"))[0]  # linear_teacher L1 tanh r0
CPU = torch.device("cpu")


def _train_dir(root: Path, setting: Setting = FIRST) -> Path:
    return cache_paths(setting_dir(root, setting)).train


@pytest.fixture
def root(tmp_path) -> Path:
    """Setting 0 as it stood at its last checkpoint (epoch 4 of 6): run it to the end, then
    drop final.pt. Every other setting is missing."""
    save_exp(preset("tiny"), tmp_path)
    run_setting(preset("tiny"), 0, tmp_path, **QUIET)
    (_train_dir(tmp_path) / "final.pt").unlink()
    assert [p.name for p in _train_dir(tmp_path).glob("ckpt_*.pt")] == ["ckpt_000004.pt"]
    return tmp_path


def snapshot(root: Path) -> dict[Path, int]:
    return {p: p.stat().st_mtime_ns for p in sorted(root.rglob("*"))}


def load_first(root: Path) -> SettingResult:
    return load_setting_result(root, load_exp(root), FIRST)


def cpu_config(root: Path):
    return setting_config(load_exp(root), FIRST).model_copy(update={"device": "cpu"})


# ---------------------------------------------------------------- the loader
def test_partial_setting_loads_the_checkpoints_history_and_progress(root):
    payload = torch.load(next(_train_dir(root).glob("ckpt_*.pt")), weights_only=True)
    r = load_first(root)
    assert r.status == "partial" and r.provisional and r.note is None
    assert r.history == payload["history"]  # exactly, not a recomputation
    assert (r.epoch, r.n_epochs) == (4, 6)
    assert r.metrics and r.data_cfg.task == FIRST.task


def test_examples_come_from_the_checkpointed_weights(root):
    payload = torch.load(next(_train_dir(root).glob("ckpt_*.pt")), weights_only=True)
    cfg = cpu_config(root)
    state = payload["net_state_dict"]
    net = make_net(cfg.net, state["U"].shape[1], state["V"].shape[1], CPU, resolve_dtype(cfg.dtype))
    net.load_state_dict(state)
    dataset = get_dataset(setting_dir(root, FIRST), cfg)
    r = load_first(root)
    with torch.no_grad():
        _, expected = net(dataset.x[r.examples.indices])
    torch.testing.assert_close(r.examples.y_pred, expected)


def test_loading_creates_and_modifies_nothing_under_root(root):
    before = snapshot(root)
    for r in load_results(root):
        setting_items(r)
        plt.close("all")
    assert snapshot(root) == before


def test_a_setting_without_checkpoint_stays_missing(root):
    r = load_results(root)[1]
    assert r.status == "missing" and r.history is None and not r.provisional


def test_corrupt_checkpoint_is_reported_with_a_note_and_no_data(root):
    ckpt = next(_train_dir(root).glob("ckpt_*.pt"))
    ckpt.write_bytes(b"not a checkpoint")
    r = load_first(root)
    assert r.status == "partial" and r.history is None and not r.provisional
    assert ckpt.name in r.note
    assert ckpt.name in setting_items(r)[0]


def test_checkpoint_with_another_config_hash_is_reported_not_shown(root):
    other = load_exp(root).model_copy(update={"lr": 0.5})
    r = load_setting_result(root, other, FIRST)
    assert r.status == "partial" and r.history is None
    assert "config hash" in r.note


def test_checkpoint_without_dataset_file_is_partial_with_a_note(root):
    (setting_dir(root, FIRST) / "data" / "dataset.pt").unlink()
    before = snapshot(root)
    r = load_first(root)
    assert r.history is None and "dataset.pt" in r.note
    assert snapshot(root) == before  # not generated


def test_checkpoint_replaced_while_reading_is_retried_once(root, monkeypatch):
    real = train_module.latest_checkpoint
    calls = []  # the first listing names a checkpoint that is gone

    def vanishing(ckpt_dir, prefix="ckpt_"):
        calls.append(1)
        return Path(ckpt_dir) / "ckpt_000002.pt" if len(calls) == 1 else real(ckpt_dir, prefix)

    monkeypatch.setattr(train_module, "latest_checkpoint", vanishing)
    r = load_first(root)
    assert r.provisional and r.epoch == 4


def test_checkpoint_vanishing_twice_is_a_note_not_a_crash(root, monkeypatch):
    monkeypatch.setattr(
        train_module, "latest_checkpoint", lambda d, prefix="ckpt_": Path(d) / "ckpt_000002.pt"
    )
    r = load_first(root)
    assert r.history is None and "ckpt_000002.pt" in r.note


def test_checkpoint_at_epoch_zero_without_monitor_entry_loads(root):
    shutil.rmtree(_train_dir(root))
    cfg = cpu_config(root)
    dataset = get_dataset(setting_dir(root, FIRST), cfg)
    net = make_net(cfg.net, dataset.x.shape[1], dataset.y.shape[1], CPU, torch.float32)
    payload = _payload(net, torch.optim.Adam(net.parameters()), _new_history(True, "adam"), 0)
    write_checkpoint(
        _train_dir(root), "ckpt_", 0, payload, cfg.section_hash("data", "net", "train")
    )
    r = load_first(root)
    assert r.provisional and (r.epoch, r.n_epochs) == (0, 6)
    fm = final_mir([r], FIRST.task)
    assert np.isnan(fm.raw).all() and fm.provisional == [True]
    text = setting_text(r)
    assert "epoch 0 of 6" in text and "no MIR" in text
    items = setting_items(r)
    assert len(items) == 2  # prose and the example outputs; no curves exist yet
    plt.close("all")


# ---------------------------------------------------------------- provisional vs final numbers
def provisional(setting: Setting, epoch: int = 4, **history_overrides) -> SettingResult:
    # the last monitor call (epoch 2) lags the checkpoint (epoch 4): a checkpoint never holds
    # the monitor call of its own epoch
    history = {**synthetic_history(setting), "monitor_epoch": [0, 2], **history_overrides}
    return SettingResult(setting, "partial", Path("here"), history=history, epoch=epoch, n_epochs=6)


def results_with_extreme_provisional() -> tuple[list[SettingResult], Setting]:
    """Every setting done except the first, whose provisional values beat all finished ones."""
    out = synthetic_results()
    s = out[0].setting
    high = [[0.99] * s.axes["net.L"], [0.99] * s.axes["net.L"]]
    out[0] = provisional(s, monitor_mir_corrected=high, monitor_mir=high)
    return out, s


def test_final_mir_keeps_provisional_values_but_flags_the_row():
    results, s = results_with_extreme_provisional()
    fm = final_mir(results, s.task)
    assert fm.provisional[0] and not any(fm.provisional[1:])
    assert fm.status[0] == "partial" and fm.epoch[0] == 4 and fm.n_epochs[0] == 6
    assert fm.corrected[0, 0] == 0.99


def test_table_marks_provisional_rows_and_explains_the_mark():
    results, s = results_with_extreme_provisional()
    text = mir_table_markdown(final_mir(results, s.task))
    rows = text.splitlines()
    marked = next(line for line in rows if setting_label(s) in line)
    plain = next(line for line in rows if setting_label(results[1].setting) in line)
    assert "*" in marked and "0.990" in marked and "*" not in plain
    assert "provisional" in text and "checkpoint at epoch 4 of 6" in text
    assert "MIR from epoch 2" in text  # the epoch of the MIR shown, not of the checkpoint
    assert "(MIR@2, ckpt 4/6)" in marked


def test_summary_statistics_use_finished_settings_only_and_list_the_provisional():
    results, s = results_with_extreme_provisional()
    text = summary_text(final_mir(results, s.task))
    assert "11 of 12 settings finished" in text and "1 provisional" in text
    assert "0.990" not in text  # the provisional best does not leak into the extremes
    finished = [r.setting for r in results[1:] if r.setting.task == s.task]
    cells = [known(f, layer, "corrected") for f in finished for layer in range(f.axes["net.L"])]
    assert f"{max(cells):.3f}" in text and f"{min(cells):.3f}" in text
    assert f"corrected {np.mean(cells):.3f}" in text  # the mean pools the finished ones only
    assert "Provisional" in text and setting_label(s) in text and "epoch 4 of 6" in text


def test_summary_with_only_provisional_settings_has_nothing_to_summarise():
    s = expand(preset("tiny"))[0]
    text = summary_text(final_mir([provisional(s)], s.task))
    assert "0 of 1 settings finished" in text and "no finished setting" in text
    assert "1 provisional" in text


def test_overview_counts_provisional_settings_separately():
    results = synthetic_results(missing=[expand(preset("tiny"))[5]])
    results[0] = provisional(results[0].setting)
    results[1] = SettingResult(results[1].setting, "partial", Path("x"), note="broken")
    text = overview_text(results)
    assert "33 done, 1 partial (provisional)" in text
    assert "1 partial without data" in text and "1 missing" in text


def test_heatmap_hatches_provisional_rows_and_names_them():
    results, s = results_with_extreme_provisional()
    fig = task_items(results, s.task)[-1]
    panels = [a for a in fig.axes if a.images]
    assert len(panels) == 3
    for ax in panels:
        hatched = [p for p in ax.patches if p.get_hatch()]
        assert len(hatched) == 1 and hatched[0].get_y() == pytest.approx(-0.5)
    labels = [t.get_text() for t in panels[0].get_yticklabels()]  # the panels share them
    assert "*" in labels[0] and "*" not in labels[1]
    assert "provisional" in fig.get_supxlabel()
    plt.close(fig)


def test_provisional_prose_and_figures_say_partial_epoch_x_of_n(root):
    text, *figs = setting_items(load_first(root))
    assert text.startswith("**linear_teacher L1 tanh r=0: partial, epoch 4 of 6**")
    assert "provisional" in text
    assert len(figs) == 3
    for fig in figs:
        shown = [fig.get_suptitle()]
        shown += [ax.get_title() for ax in fig.axes]
        assert any("partial, epoch 4 of 6" in t for t in shown)
    plt.close("all")


# ---------------------------------------------------------------- the colour scale and the worst case
def extreme_provisional(value: float) -> tuple[list[SettingResult], Setting]:
    """Finished settings with MIR in [0, 1] and a provisional first one at `value`."""
    out = synthetic_results()
    s = out[0].setting
    column = [[value] * s.axes["net.L"], [value] * s.axes["net.L"]]
    out[0] = provisional(
        s, monitor_mir=column, monitor_mir_corrected=column, monitor_mir_chance=column
    )
    return out, s


@pytest.mark.parametrize("value", [5.0, -3.0])
def test_a_provisional_value_outside_the_range_does_not_move_the_colour_scale(value):
    results, s = extreme_provisional(value)
    fig = task_items(results, s.task)[-1]
    panels = [a for a in fig.axes if a.images]
    for ax in panels:
        assert ax.images[0].get_clim() == (0.0, 1.0)
        shown = np.ma.filled(ax.images[0].get_array().astype(float), np.nan)
        assert np.nanmax(shown[0]) <= 1.0 and np.nanmin(shown[0]) >= 0.0  # clipped for colour
        assert f"{value:.2f}" in [t.get_text() for t in ax.texts]  # the printed number is exact
    colourbar = [a for a in fig.axes if not a.images and a.get_ylabel() == "MIR"]
    assert colourbar[0].get_ylim() == (0.0, 1.0)
    assert "clipped" in fig.get_supxlabel()
    plt.close(fig)


def test_a_finished_value_outside_the_range_still_moves_the_scale():
    fig = fig_mir_heatmaps({"a": np.array([[1.25], [9.0]])}, ["x", "y"], ["done", "provisional"])
    assert fig.axes[0].images[0].get_clim() == (0.0, 1.25)
    plt.close(fig)


def test_a_provisional_corrected_value_below_every_finished_one_is_not_the_worst():
    results, s = extreme_provisional(-3.0)
    text = summary_text(final_mir(results, s.task))
    finished = [r.setting for r in results[1:] if r.setting.task == s.task]
    lowest = min(known(f, layer, "corrected") for f in finished for layer in range(f.axes["net.L"]))
    assert "-3.000" not in text
    assert f"({lowest:.3f})" in text


def test_provisional_row_names_the_epoch_of_its_mir_not_only_the_checkpoints():
    results, s = results_with_extreme_provisional()
    fm = final_mir(results, s.task)
    assert (fm.mir_epoch[0], fm.epoch[0]) == (2, 4)  # the monitor call lags the checkpoint
    assert "MIR@2, ckpt 4/6" in fm.display_labels()[0]
    assert fm.mir_epoch[1:] == [None] * 11


def test_finished_only_blanks_exactly_the_provisional_rows():
    results, s = results_with_extreme_provisional()
    fm = final_mir(results, s.task)
    fin = fm.finished_only()
    assert np.isnan(fin.corrected[0]).all()
    np.testing.assert_array_equal(fin.corrected[1:], fm.corrected[1:])
    assert fm.corrected[0, 0] == 0.99  # the original is untouched


# ---------------------------------------------------------------- unusable files
def test_corrupt_dataset_is_reported_as_a_dataset_problem(root):
    (setting_dir(root, FIRST) / "data" / "dataset.pt").write_bytes(b"garbage")
    r = load_first(root)
    assert r.history is None and r.note.startswith("dataset not readable")


def test_a_stale_file_handle_on_the_checkpoint_is_retried(root, monkeypatch):
    real = train_module._load_payload
    calls = []

    def flaky(path, cfg, device):
        calls.append(path)
        if len(calls) == 1:
            raise OSError(116, "Stale file handle")
        return real(path, cfg, device)

    monkeypatch.setattr(train_module, "_load_payload", flaky)
    assert load_first(root).provisional


def test_a_checkpoint_whose_weights_do_not_fit_the_net_is_reported(root):
    ckpt = next(_train_dir(root).glob("ckpt_*.pt"))
    payload = torch.load(ckpt, weights_only=True)
    del payload["net_state_dict"]["U"]
    torch.save(payload, ckpt)  # keeps the config hash, so only the weights are wrong
    r = load_first(root)
    assert r.history is None and ckpt.name in r.note


# ---------------------------------------------------------------- stale finished settings
def test_a_finished_setting_from_the_old_loss_is_listed_with_a_note(tmp_path, monkeypatch):
    exp = preset("tiny")
    save_exp(exp, tmp_path)
    with monkeypatch.context() as old:
        old.setattr(fct.config, "LOSS_VERSION", fct.config.LOSS_VERSION - 1)
        run_setting(exp, 0, tmp_path, **QUIET)
    run_setting(exp, 1, tmp_path, **QUIET)

    results = load_results(tmp_path)
    stale, good = results[0], results[1]
    assert stale.status == "stale" and stale.history is None and not stale.provisional
    assert "LOSS_VERSION" in stale.note or "old loss" in stale.note
    assert good.status == "done" and good.history is not None
    assert "1 stale" in overview_text(results)
    assert "stale" in setting_text(stale) and "not shown" in setting_text(stale)
    assert len(setting_items(stale)) == 1
    items = task_items(results, FIRST.task)  # the table, heatmap and summary still build
    assert "(stale)" in items[0]
    plt.close("all")
