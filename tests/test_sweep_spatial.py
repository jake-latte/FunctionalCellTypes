"""Spatial sweeps: the `spatial` section, spatial axes, the two spatial presets, short labels
for dict-valued axes, and that sweeps without a spatial section are unchanged (CPU only)."""

import pytest

from fct.cli import main
from fct.config import Config, NetConfig, TrainConfig
from fct.spatial import SpatialConfig
from fct.sweep import (
    SweepConfig,
    _label_part,
    data_config,
    expand,
    load_sweep,
    phase_transition,
    preset,
    run_setting,
    save_sweep,
    setting_config,
    setting_dir,
    setting_status,
    with_grid,
)

LENGTH_AXIS = "spatial.hidden.kernel.length_scale"


def _tiny_spatial(extra_axes: dict) -> SweepConfig:
    """spatial_tiny (axis net.N) with `extra_axes` added after it."""
    sweep = preset("spatial_tiny")
    return with_grid(sweep, {**sweep.grid, **extra_axes})


def _train_hash(sweep: SweepConfig, index: int) -> str:
    return setting_config(sweep, expand(sweep)[index]).section_hash("data", "net", "train")


# ---------------------------------------------------------------- unchanged non-spatial sweeps
def _previous_config(sweep: SweepConfig, axes: dict) -> Config:
    """The Config the sweep built before it had a spatial section: the base fields written out
    by hand, then the axis values over it."""
    base = Config(
        data=data_config(sweep, axes.get("data.task", sweep.task)),
        net=NetConfig(N=sweep.N, seed=sweep.seeds.net),
        train=TrainConfig(
            n_epochs=sweep.n_epochs,
            lr=sweep.lr,
            log_int=sweep.log_int,
            save_int=sweep.save_int,
            seed=sweep.seeds.train,
            monitor_int=sweep.monitor_int,
            monitor_bins=sweep.monitor_bins,
            monitor_shuffles=sweep.monitor_shuffles,
        ),
        meanfield=sweep.meanfield,
        dtype=sweep.dtype,
        device=sweep.device,
    )
    tree = base.model_dump()
    for axis, value in axes.items():
        if axis == "data.task":  # chosen above, together with the task's own D_in
            continue
        section, field = axis.split(".")
        tree[section][field] = value
    return Config.model_validate(tree)


@pytest.mark.parametrize(
    "grid",
    [
        {"net.L": [1, 3], "net.phi": ["tanh", "relu"], "train.rate_lambda": [0.0, 1e-3]},
        {"net.N": [4, 8], "train.weight_lambda": [1e-3, 1e-2]},
    ],
)
def test_sweeps_without_spatial_build_the_configs_they_built_before(grid):
    sweep = with_grid(preset("tiny"), grid)
    assert sweep.spatial is None
    for setting in expand(sweep):
        assert setting_config(sweep, setting) == _previous_config(sweep, setting.axes)


@pytest.mark.parametrize("name", ["tiny", "modularity"])
def test_the_modularity_grid_builds_the_configs_it_built_before(name):
    sweep = preset(name)
    assert sweep.grid is None
    settings = expand(sweep)
    assert len(settings) == 36
    for setting in settings:
        assert setting_config(sweep, setting) == _previous_config(sweep, setting.axes)


def test_the_phase_transition_preset_builds_the_configs_it_built_before():
    sweep = phase_transition(2)
    for setting in expand(sweep):
        assert setting_config(sweep, setting) == _previous_config(sweep, setting.axes)


def _only_setting_config(sweep: SweepConfig) -> Config:
    (setting,) = expand(sweep)
    return setting_config(sweep, setting)


def test_the_shared_fields_default_to_the_library_defaults():
    cfg = _only_setting_config(with_grid(preset("tiny"), {"net.N": [4]}))
    net, train = NetConfig(), TrainConfig()
    assert (cfg.net.L, cfg.net.g, cfg.net.phi) == (net.L, net.g, net.phi)
    assert (cfg.train.weight_lambda, cfg.train.rate_lambda) == (
        train.weight_lambda,
        train.rate_lambda,
    )
    assert (cfg.train.optimizer, cfg.train.beta) == (train.optimizer, train.beta)


@pytest.mark.parametrize(
    "field, value, read",
    [
        ("L", 2, lambda c: c.net.L),
        ("g", 0.7, lambda c: c.net.g),
        ("phi", "sigmoid", lambda c: c.net.phi),
        ("weight_lambda", 0.02, lambda c: c.train.weight_lambda),
        ("rate_lambda", 0.03, lambda c: c.train.rate_lambda),
        ("optimizer", "sgld", lambda c: c.train.optimizer),
        ("beta", 123.0, lambda c: c.train.beta),
    ],
)
def test_each_shared_field_reaches_the_config(field, value, read):
    sweep = with_grid(preset("tiny").model_copy(update={field: value}), {"net.N": [4]})
    assert read(_only_setting_config(sweep)) == value


@pytest.mark.parametrize(
    "shared, axis, values, read",
    [
        ({"L": 1}, "net.L", [2, 3], lambda c: c.net.L),
        ({"beta": 1e4}, "train.beta", [5.0], lambda c: c.train.beta),
        ({"phi": "tanh"}, "net.phi", ["relu"], lambda c: c.net.phi),
        ({"weight_lambda": 3e-3}, "train.weight_lambda", [0.5], lambda c: c.train.weight_lambda),
    ],
)
def test_an_axis_wins_over_a_non_default_shared_field(shared, axis, values, read):
    sweep = with_grid(preset("tiny").model_copy(update=shared), {axis: values})
    assert [read(setting_config(sweep, s)) for s in expand(sweep)] == values


# ---------------------------------------------------------------- the presets
def test_spatial_n_scan_expands_to_five_spatial_settings(tmp_path):
    sweep = preset("spatial_n_scan")
    settings = expand(sweep)
    configs = [setting_config(sweep, s) for s in settings]
    widths = [64, 128, 256, 512, 1024]
    assert [c.net.N for c in configs] == widths
    assert [setting_dir(tmp_path, s).name for s in settings] == [
        f"s000{i}_N{n}" for i, n in enumerate(widths)
    ]
    assert all(c.spatial == configs[0].spatial and c.spatial is not None for c in configs)
    c = configs[0]
    assert c.spatial.hidden.kernel.length_scale == 0.15 and c.spatial.min_rel_eig == 1e-2
    assert (c.data.P, c.data.D_in, c.data.n_latents) == (164, 64, 2)
    assert (c.net.L, c.net.phi, c.net.g, c.dtype) == (1, "tanh", 1.0, "float64")
    t = c.train
    assert (t.weight_lambda, t.rate_lambda, t.optimizer, t.beta) == (3e-3, 0.0, "sgld", 1e4)
    assert (t.lr, t.n_epochs) == (1e-2, 250_000)
    assert (t.monitor_int, t.monitor_bins, t.monitor_shuffles) == (5000, 8, 5)


def test_spatial_n_scan_takes_a_length_scale_axis_from_the_cli(tmp_path, capsys):
    args = ["sweep", "init", str(tmp_path), "--preset", "spatial_n_scan"]
    assert main([*args, "--axis", f"{LENGTH_AXIS}=[0.05,0.15,0.3]"]) == 0
    assert len(expand(load_sweep(tmp_path))) == 15
    capsys.readouterr()
    assert main(["sweep", "count", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == "15"


# ---------------------------------------------------------------- spatial axes
def test_a_spatial_axis_changes_the_kernel_and_the_training_hash():
    sweep = _tiny_spatial({LENGTH_AXIS: [0.1, 0.2]})
    configs = [setting_config(sweep, s) for s in expand(sweep)]
    assert [c.spatial.hidden.kernel.length_scale for c in configs] == [0.1, 0.2] * 2
    assert _train_hash(sweep, 0) != _train_hash(sweep, 1)


@pytest.mark.parametrize(
    "axis, values",
    [
        ("spatial.hidden.embedding.kind", ["grid", "random"]),
        ("spatial.min_rel_eig", [1e-2, 1e-3]),
        ("spatial.box", [1.0, 2.0]),
    ],
)
def test_any_field_of_the_spatial_section_may_be_an_axis(axis, values):
    sweep = _tiny_spatial({axis: values})
    assert len(expand(sweep)) == 4 and _train_hash(sweep, 0) != _train_hash(sweep, 1)


def test_a_spatial_axis_without_a_spatial_section_fails_naming_the_axis():
    tree = preset("tiny").model_dump()
    tree["grid"] = {LENGTH_AXIS: [0.1, 0.2]}
    with pytest.raises(ValueError, match=rf"{LENGTH_AXIS}.*spatial section first"):
        SweepConfig.model_validate(tree)


def test_the_same_failure_when_sweep_json_is_loaded(tmp_path):
    save_sweep(preset("tiny"), tmp_path)
    path = tmp_path / "sweep.json"
    path.write_text(path.read_text().replace('"grid": null', f'"grid": {{"{LENGTH_AXIS}": [0.1]}}'))
    with pytest.raises(ValueError, match=LENGTH_AXIS):
        load_sweep(tmp_path)


def test_an_invalid_spatial_value_fails_at_load_naming_the_axis():
    with pytest.raises(ValueError, match="spatial.min_rel_eig"):
        _tiny_spatial({"spatial.min_rel_eig": [2.0]})


def test_a_dict_valued_axis_gets_a_short_distinct_directory_name(tmp_path):
    # `spatial` itself is an axis only while the sweep has no section; its values are whole dumps.
    prior = SpatialConfig().model_dump()
    sweep = with_grid(preset("tiny"), {"spatial": [None, prior, {**prior, "min_rel_eig": 1e-3}]})
    names = [setting_dir(tmp_path, s).name for s in expand(sweep)]
    assert len(set(names)) == 3 and all(len(n) < 40 for n in names)
    assert names[0] == "s0000_spatialNone" and names[1].startswith("s0001_spatial-")


@pytest.mark.parametrize(
    "axis, values",
    [
        ("train.lr", [[0.001 * (i + 1) for i in range(60)], [0.002 * (i + 1) for i in range(60)]]),
        ("spatial.hidden.embedding.shape", [[4], [8]]),
    ],
)
def test_list_valued_axes_give_short_distinct_labels(axis, values):
    """A list of 60 floats would otherwise give a name beyond NAME_MAX."""
    labels = [_label_part(axis, v) for v in values]
    assert len(set(labels)) == 2 and all(len(label) < 40 for label in labels)


def test_a_long_string_value_is_hashed_in_the_label():
    assert len(_label_part("data.task", "x" * 300)) < 40


# ---------------------------------------------------------------- running
def test_spatial_tiny_runs_end_to_end_and_resumes(tmp_path, capsys):
    root = str(tmp_path)
    assert main(["sweep", "init", root, "--preset", "spatial_tiny"]) == 0
    assert main(["sweep", "run", root, "--index", "0"]) == 0
    sweep = load_sweep(tmp_path)
    states = [setting_status(tmp_path, sweep, s) for s in expand(sweep)]
    assert states == ["done", "missing"]
    assert main(["sweep", "run", root]) == 0  # resume: the done setting is skipped
    capsys.readouterr()
    assert main(["sweep", "status", root]) == 0
    assert "2 done, 0 partial, 0 missing, 0 stale" in capsys.readouterr().out


class _Interrupt(Exception):
    pass


def test_an_interrupted_spatial_setting_resumes_from_its_checkpoint(tmp_path):
    sweep = preset("spatial_tiny")
    setting = expand(sweep)[0]

    def stop_at_epoch_4(line: str) -> None:
        if line.startswith("[00004/"):
            raise _Interrupt

    with pytest.raises(_Interrupt):
        run_setting(sweep, 0, tmp_path, log=stop_at_epoch_4)
    assert setting_status(tmp_path, sweep, setting) == "partial"
    lines: list[str] = []
    run_setting(sweep, 0, tmp_path, log=lines.append)
    assert any(line.startswith("resumed from") and "at epoch" in line for line in lines)
    assert setting_status(tmp_path, sweep, setting) == "done"


def test_sweep_run_stage_solve_on_a_spatial_setting_is_a_one_line_error(tmp_path, capsys):
    root = str(tmp_path)
    assert main(["sweep", "init", root, "--preset", "spatial_tiny"]) == 0
    capsys.readouterr()
    assert main(["sweep", "run", root, "--index", "0", "--stage", "solve"]) == 1
    err = capsys.readouterr().err.strip()
    assert err.startswith("error:") and "no spatial prior" in err and "\n" not in err
