"""Wiring from a sweep ROOT to the phase-transition figures: snapping a requested setting to the
grid, assembling MIR records (holes, provisional settings), per-neuron tuning of one layer.

The pure parts run on hand-built results; the loaders run on a tiny phase-transition sweep
trained in the test.

Grid order of the phase-transition sweep (L slowest, then phi, then lambda_r, lambda_w fastest):
indices 0..3 are L=1 tanh at (r0,w0) (r0,w1) (r1,w0) (r1,w1); 4..7 L=1 relu; 8..11 L=2 tanh.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch
from matplotlib.figure import Figure

from fct.grid_report import mir_grid
from fct.grid_view import (
    DEPTH_AXIS,
    LAMBDA_R_AXIS,
    LAMBDA_W_AXIS,
    PHI_AXIS,
    axis_values,
    choose_setting,
    factor_matrix,
    grid_selection,
    layer_rates,
    load_setting_data,
    matrix_items,
    neuron_mir_values,
    ratio_items,
    snap_text,
    snap_to_grid,
    tuning_items,
    tuning_view,
)
from fct.modularity import neuron_factor_information
from fct.report import SettingResult, load_results
from fct.sweep import expand, run_setting, save_sweep, setting_dir
from fct.tuning import TuningCurves
from tests.draw_helpers import FakeDeviceTensor, on_fake_device
from tests.report_helpers import QUIET, tiny_phase_sweep

LAMBDAS = [1e-6, 1.0]  # phase_transition(2)
GLOBAL = {"depth": 1, "phi": "tanh", "kind": "corrected"}


def history(raw: list[float], corrected: list[float]) -> dict:
    """Two monitor calls; only the last is the setting's MIR."""
    first = [-1.0] * len(raw)
    return {
        "monitor_epoch": [0, 6],
        "monitor_mir": [first, raw],
        "monitor_mir_corrected": [first, corrected],
        "monitor_mir_chance": [first, [0.0] * len(raw)],
    }


def synthetic(status_by_index: dict[int, str]) -> list[SettingResult]:
    """One result per setting: `done`, `partial` (provisional) or, by default, `missing`.
    Layer l of setting i has raw MIR i + l/10 and corrected MIR that plus 0.5."""
    out = []
    for s in expand(tiny_phase_sweep()):
        status = status_by_index.get(s.index, "missing")
        if status == "missing":
            out.append(SettingResult(s, status, Path("none")))
            continue
        raw = [s.index + layer / 10 for layer in range(s.axes[DEPTH_AXIS])]
        hist = history(raw, [v + 0.5 for v in raw])
        out.append(SettingResult(s, status, Path("x"), history=hist, epoch=4, n_epochs=6))
    return out


# ------------------------------------------------------------------ snapping and lookup
def test_snap_picks_the_nearest_value_and_reports_it():
    grid = [1e-6, 1e-3, 1.0]
    exact = snap_to_grid(1e-3, grid, log=True)
    assert (exact.value, exact.exact) == (1e-3, True)
    # 0.1 is nearer 1.0 on the log axis the grid is spaced on, nearer 1e-3 on a linear one
    near = snap_to_grid(0.1, grid, log=True)
    assert (near.requested, near.value, near.exact) == (0.1, 1.0, False)
    assert snap_to_grid(0.1, grid, log=False).value == 1e-3


def test_snap_integer_axis_and_bad_input():
    assert snap_to_grid(2.4, [1, 2, 3]).value == 2
    with pytest.raises(ValueError, match="non-positive"):
        snap_to_grid(0.0, [1e-6, 1.0], log=True)
    with pytest.raises(ValueError, match="finite"):
        snap_to_grid(float("nan"), [1.0, 2.0])
    with pytest.raises(ValueError, match="empty"):
        snap_to_grid(1.0, [])


def test_axis_values_are_the_grid_values_numbers_sorted():
    values = axis_values(synthetic({}))
    assert values[DEPTH_AXIS] == [1, 2]
    assert values[PHI_AXIS] == ["tanh", "relu"]
    assert values[LAMBDA_R_AXIS] == values[LAMBDA_W_AXIS] == LAMBDAS


def test_choose_setting_snaps_every_axis_and_finds_the_one_setting():
    choice = choose_setting(synthetic({}), L=2.2, phi="tanh", lambda_r=0.9, lambda_w=2e-6)
    assert choice.snapped[DEPTH_AXIS].value == 2
    assert choice.snapped[LAMBDA_R_AXIS].value == 1.0
    assert choice.snapped[LAMBDA_W_AXIS].value == 1e-6
    assert choice.result.setting.index == 8 + 2  # L=2 tanh (r1, w0)
    text = snap_text(choice)
    assert "lambda_r" in text and "0.9" in text and "snapped" in text


def test_choose_setting_rejects_an_unknown_activation():
    with pytest.raises(ValueError, match="phi.*tanh"):
        choose_setting(synthetic({}), L=1, phi="gelu", lambda_r=1.0, lambda_w=1.0)


# ------------------------------------------------------------------ records and holes
def test_grid_with_a_hole_gives_a_nan_cell_and_the_known_values():
    results = synthetic({0: "done", 1: "done", 2: "done"})  # setting 3, (r1, w1), never ran
    sel = grid_selection(results, depth=1, phi="tanh", kind="raw")
    assert sel.n_layers == 1 and sel.lambda_r_values == sel.lambda_w_values == LAMBDAS
    grid = mir_grid(sel.records, sel.lambda_r_values, sel.lambda_w_values, sel.n_layers)
    np.testing.assert_array_equal(grid[0], [[0.0, 1.0], [2.0, np.nan]])
    assert sel.provisional == []


def test_kind_toggles_between_raw_and_corrected():
    results = synthetic({1: "done"})
    assert grid_selection(results, 1, "tanh", "raw").records[0].mir == 1.0
    assert grid_selection(results, 1, "tanh", "corrected").records[0].mir == 1.5
    with pytest.raises(ValueError, match="kind"):
        grid_selection(results, 1, "tanh", "chance")


def test_partial_setting_is_marked_provisional_and_uses_its_latest_monitor_call():
    sel = grid_selection(synthetic({0: "done", 1: "partial"}), 1, "tanh", "corrected")
    assert sel.provisional == [(1e-6, 1.0)]
    assert {(r.lambda_r, r.lambda_w): r.mir for r in sel.records}[(1e-6, 1.0)] == 1.5


def test_partial_setting_without_a_monitor_call_has_no_record_and_is_not_provisional():
    results = synthetic({0: "done"})
    results[1] = SettingResult(
        results[1].setting, "partial", Path("x"), history={"monitor_epoch": []}, epoch=1
    )
    sel = grid_selection(results, 1, "tanh", "raw")
    assert [(r.lambda_r, r.lambda_w) for r in sel.records] == [(1e-6, 1e-6)]
    assert sel.provisional == []


def test_two_layer_selection_has_a_record_per_layer():
    sel = grid_selection(synthetic({8: "done"}), depth=2, phi="tanh", kind="raw")
    assert sorted((r.layer, r.mir) for r in sel.records) == [(0, 8.0), (1, 8.1)]


def test_history_with_the_wrong_number_of_layers_fails_loudly():
    results = synthetic({0: "done"})
    results[0] = SettingResult(
        results[0].setting, "done", Path("x"), history=history([0.1, 0.2], [0.1, 0.2])
    )
    with pytest.raises(ValueError, match="layers"):
        grid_selection(results, 1, "tanh", "raw")


def test_global_items_say_so_when_no_setting_has_data():
    results = synthetic({})
    for build in (matrix_items, ratio_items):
        items = build(results, **GLOBAL)
        assert len(items) == 1 and isinstance(items[0], str) and "no MIR" in items[0]


def test_global_items_carry_a_provisional_label():
    results = synthetic({0: "done", 1: "partial", 2: "done"})
    for build in (matrix_items, ratio_items):
        text, fig = build(results, **GLOBAL)
        assert "provisional" in text
        assert isinstance(fig, Figure) and "provisional" in fig.get_suptitle()


def test_global_items_without_provisional_settings_do_not_say_provisional():
    for build in (matrix_items, ratio_items):
        text, fig = build(synthetic({0: "done", 1: "done"}), **GLOBAL)
        assert "provisional" not in text and "provisional" not in fig.get_suptitle()


# ------------------------------------------------------------------ tuning (pure)
def test_factor_matrix_stacks_latents_and_rejects_mixed_kinds():
    names, matrix = factor_matrix({"a": torch.zeros(5), "b": torch.ones(5)})
    assert names == ["a", "b"] and matrix.shape == (5, 2)
    with pytest.raises(ValueError, match="mix"):
        factor_matrix({"a": torch.zeros(5), "b": torch.ones(5, dtype=torch.long)})


def test_neuron_mir_is_high_for_neurons_each_driven_by_one_latent():
    gen = torch.Generator().manual_seed(0)
    factors = {"z0": torch.randn(400, generator=gen), "z1": torch.randn(400, generator=gen)}
    rates = torch.stack([factors["z0"], factors["z1"], factors["z0"] ** 3], dim=1)
    groups = {"z0": ["z0"], "z1": ["z1"]}
    values = neuron_mir_values(rates, factors, groups, n_bins=5, kind="raw", n_shuffles=2, seed=0)
    assert values.shape == (3,) and np.all(values > 0.9)


def test_tuning_view_member_mi_is_the_information_behind_its_mir(root, results):
    loaded = load_setting_data(root, choice_for(results, 0).result)
    tc = loaded.cfg.train
    view = tuning_view(
        loaded.net, loaded.dataset, 0, tc.monitor_bins, "corrected", tc.monitor_shuffles, tc.seed
    )
    rates = layer_rates(loaded.net, loaded.dataset.x, 0).cpu()
    factors = {k: f.cpu() for k, f in loaded.dataset.factors.items()}
    direct = neuron_factor_information(
        rates,
        factors,
        loaded.dataset.groups,
        tc.monitor_bins,
        "shuffle",
        tc.monitor_shuffles,
        tc.seed,
    )
    n_neurons, n_latents = view.member_mi.shape
    assert (n_neurons, n_latents) == (loaded.cfg.net.N, len(view.latent_names))
    torch.testing.assert_close(view.member_mi, direct.member_mi, rtol=0, atol=0)
    torch.testing.assert_close(view.raw_mi, direct.raw_mi, rtol=0, atol=0)
    torch.testing.assert_close(view.null_mi, direct.null_mi, rtol=0, atol=0)
    assert view.raw_mi.shape == view.null_mi.shape == view.member_mi.shape


def test_tuning_view_of_the_raw_kind_has_no_null(root, results):
    loaded = load_setting_data(root, choice_for(results, 0).result)
    tc = loaded.cfg.train
    view = tuning_view(
        loaded.net, loaded.dataset, 0, tc.monitor_bins, "raw", tc.monitor_shuffles, tc.seed
    )
    assert view.null_mi is None
    torch.testing.assert_close(view.raw_mi, view.member_mi, rtol=0, atol=0)


def test_neuron_mir_rejects_an_unknown_kind():
    factors = {"z0": torch.randn(50), "z1": torch.randn(50)}
    with pytest.raises(ValueError, match="kind"):
        neuron_mir_values(torch.randn(50, 2), factors, None, 4, "chance", 2, 0)


# ------------------------------------------------------------------ real tiny sweep
@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    """Settings 0 and 1 done, 2 stopped at its checkpoint (final.pt removed), 3 never run; 8 (L=2) done."""
    root = tmp_path_factory.mktemp("phase")
    sweep = tiny_phase_sweep()
    save_sweep(sweep, root)
    for index in (0, 1, 2, 8):
        run_setting(sweep, index, root, **QUIET)
    (setting_dir(root, expand(sweep)[2]) / "train" / "final.pt").unlink()
    return root


@pytest.fixture(scope="module")
def results(root) -> list[SettingResult]:
    return load_results(root)


def choice_for(results: list[SettingResult], index: int):
    axes = results[index].setting.axes
    return choose_setting(
        results,
        L=axes[DEPTH_AXIS],
        phi=axes[PHI_AXIS],
        lambda_r=axes[LAMBDA_R_AXIS],
        lambda_w=axes[LAMBDA_W_AXIS],
    )


def test_global_matrix_from_a_real_root_has_the_hole_and_a_provisional_cell(results):
    sel = grid_selection(results, 1, "tanh", "corrected")
    assert len(sel.records) == 3  # settings 0, 1 and the checkpoint of 2; setting 3 never ran
    assert len(sel.provisional) == 1
    grid = mir_grid(sel.records, sel.lambda_r_values, sel.lambda_w_values, sel.n_layers)
    assert np.isnan(grid[0, 1, 1]) and np.isfinite(grid[0]).sum() == 3
    text, fig = matrix_items(results, **GLOBAL)
    assert "provisional" in text and isinstance(fig, Figure)


def test_tuning_of_a_done_setting_matches_the_monitored_mir(root, results):
    choice = choice_for(results, 0)
    items = tuning_items(root, choice, 0, 3, "mir_descending", 0, "corrected")
    assert isinstance(items[-1], Figure) and "provisional" not in items[0]
    loaded = load_setting_data(root, choice.result)
    tc = loaded.cfg.train
    view = tuning_view(
        loaded.net, loaded.dataset, 0, tc.monitor_bins, "corrected", tc.monitor_shuffles, tc.seed
    )
    monitored = choice.result.history["monitor_mir_corrected"][-1][0]
    assert float(view.neuron_mir.mean()) == pytest.approx(monitored, abs=1e-6)
    assert view.curves.mean.shape[:2] == (loaded.cfg.net.N, len(view.latent_names))


def test_tuning_of_a_partial_setting_is_labelled_provisional(root, results):
    items = tuning_items(root, choice_for(results, 2), 0, 2, "random", 1, "raw")
    assert "provisional" in items[0]
    assert "provisional" in items[-1].get_suptitle()
    assert "epoch 4 of 6" in items[0] and "epoch 4 of 6" in items[-1].get_suptitle()


def test_tuning_of_a_missing_setting_is_a_message_and_no_figure(root, results):
    items = tuning_items(root, choice_for(results, 3), 0, 2, "random", 1, "raw")
    assert len(items) == 1 and isinstance(items[0], str) and "missing" in items[0]


def test_tuning_layer_outside_the_net_fails_loudly(root, results):
    with pytest.raises(ValueError, match="layer"):
        tuning_items(root, choice_for(results, 0), 3, 2, "random", 0, "raw")


@pytest.mark.parametrize("kind", ["raw", "corrected"])
def test_second_layer_tuning_matches_the_monitored_mir_of_layer_two(root, results, kind):
    """Layer index 1 of a 2-layer net is checked against the monitor's second entry."""
    result = choice_for(results, 8).result  # L=2 tanh
    loaded = load_setting_data(root, result)
    tc = loaded.cfg.train
    view = tuning_view(
        loaded.net, loaded.dataset, 1, tc.monitor_bins, kind, tc.monitor_shuffles, tc.seed
    )
    key = {"raw": "monitor_mir", "corrected": "monitor_mir_corrected"}[kind]
    monitored = result.history[key][-1]
    assert len(monitored) == 2
    assert float(view.neuron_mir.mean()) == pytest.approx(monitored[1], abs=1e-6)
    assert monitored[0] != pytest.approx(monitored[1], abs=1e-6)  # the layers do differ


def test_unusable_settings_are_named_apart_from_not_yet_run():
    results = synthetic({0: "done"})
    results[1] = SettingResult(results[1].setting, "stale", Path("x"), note="old loss version")
    results[2] = SettingResult(
        results[2].setting, "partial", Path("x"), note="checkpoint not shown: truncated"
    )
    for build in (matrix_items, ratio_items):
        text = build(results, **GLOBAL)[0]
        assert "1 of 4 networks" in text and "1 have none yet" in text
        assert "stale or unreadable" in text
        assert "old loss version" in text and "truncated" in text
        assert "lambda_r=1e-06, lambda_w=1" in text  # setting 1
        assert "lambda_r=1, lambda_w=1e-06" in text  # setting 2


def test_provisional_cells_are_listed_by_their_lambdas():
    results = synthetic({0: "done", 3: "partial"})
    text, fig = matrix_items(results, **GLOBAL)
    assert "(lambda_r=1, lambda_w=1)" in text
    assert "(lambda_r=1, lambda_w=1)" in fig.get_suptitle()


def test_depth_is_coerced_to_an_int_and_a_fraction_is_rejected():
    results = synthetic({0: "done"})
    assert grid_selection(results, 1.0, "tanh", "raw").n_layers == 1
    with pytest.raises(ValueError, match="whole number"):
        grid_selection(results, 1.5, "tanh", "raw")


# ------------------------------------------------------------------ tuning of device tensors
def test_tuning_view_of_device_tensors_returns_host_arrays_equal_to_the_cpu_ones(
    root, results, monkeypatch
):
    loaded = load_setting_data(root, choice_for(results, 0).result)
    tc = loaded.cfg.train
    args = (0, tc.monitor_bins, "corrected", tc.monitor_shuffles, tc.seed)
    expected = tuning_view(loaded.net, loaded.dataset, *args)

    real_layer_rates = layer_rates
    monkeypatch.setattr(
        "fct.grid_view.layer_rates", lambda *a: on_fake_device(real_layer_rates(*a))
    )
    d = loaded.dataset
    on_device = dataclasses.replace(d, factors={k: on_fake_device(v) for k, v in d.factors.items()})
    view = tuning_view(loaded.net, on_device, *args)

    assert type(view.neuron_mir) is np.ndarray
    assert not isinstance(view.member_mi, FakeDeviceTensor)
    torch.testing.assert_close(view.member_mi, expected.member_mi, rtol=0, atol=0)
    for field in ("raw_mi", "null_mi"):
        assert not isinstance(getattr(view, field), FakeDeviceTensor)
        torch.testing.assert_close(getattr(view, field), getattr(expected, field), rtol=0, atol=0)
        assert getattr(view, field).device.type == "cpu"
    assert not any(isinstance(a, FakeDeviceTensor) for a in view.curves)
    np.testing.assert_array_equal(view.neuron_mir, expected.neuron_mir)
    for field in TuningCurves._fields:
        np.testing.assert_array_equal(getattr(view.curves, field), getattr(expected.curves, field))
