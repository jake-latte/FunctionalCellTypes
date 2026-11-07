"""Spatial experiments: the `spatial` section, spatial axes, the two spatial presets, short labels
for dict-valued axes, and that experiments without a spatial section are unchanged (CPU only)."""

import json

import numpy as np
import pytest
import torch

from fct.cli import main
from fct.config import Config, NetConfig, TrainConfig
from fct.data import make_dataset, task_dims
from fct.exp import (
    ExpConfig,
    _label_part,
    data_config,
    exp_text,
    expand,
    load_exp,
    preset,
    run_setting,
    save_exp,
    setting_config,
    setting_dir,
    setting_status,
    with_axes,
)
from fct.spatial import SpatialConfig, build_spaces
from fct.tasks import latent_lattice_dataset
from fct.train import load_trained

LENGTH_AXIS = "spatial.hidden.kernel.length_scale"


def _tiny_spatial(extra_axes: dict) -> ExpConfig:
    """spatial_tiny (axis net.N) with `extra_axes` added after it."""
    exp = preset("spatial_tiny")
    return with_axes(exp, {**exp.axes, **extra_axes})


def _train_hash(exp: ExpConfig, index: int) -> str:
    return setting_config(exp, expand(exp)[index]).section_hash("data", "net", "train")


# ---------------------------------------------------------------- unchanged non-spatial experiments
def _previous_config(exp: ExpConfig, axes: dict) -> Config:
    """The Config the experiment built before it had a spatial section: the base fields written out
    by hand, then the axis values over it."""
    base = Config(
        data=data_config(exp, axes.get("data.task", exp.task)),
        net=NetConfig(N=exp.N, seed=exp.seeds.net),
        train=TrainConfig(
            n_epochs=exp.n_epochs,
            lr=exp.lr,
            log_int=exp.log_int,
            save_int=exp.save_int,
            seed=exp.seeds.train,
            monitor_int=exp.monitor_int,
            monitor_bins=exp.monitor_bins,
            monitor_shuffles=exp.monitor_shuffles,
        ),
        meanfield=exp.meanfield,
        dtype=exp.dtype,
        device=exp.device,
    )
    tree = base.model_dump()
    for axis, value in axes.items():
        if axis == "data.task":  # chosen above, together with the task's own D_in
            continue
        section, field = axis.split(".")
        tree[section][field] = value
    return Config.model_validate(tree)


@pytest.mark.parametrize(
    "axes",
    [
        {"net.L": [1, 3], "net.phi": ["tanh", "relu"], "train.rate_lambda": [0.0, 1e-3]},
        {"net.N": [4, 8], "train.weight_lambda": [1e-3, 1e-2]},
    ],
)
def test_exps_without_spatial_build_the_configs_they_built_before(axes):
    exp = with_axes(preset("tiny"), axes)
    assert exp.spatial is None
    for setting in expand(exp):
        assert setting_config(exp, setting) == _previous_config(exp, setting.axes)


@pytest.mark.parametrize("name", ["tiny", "modularity"])
def test_the_modularity_experiment_builds_the_configs_it_built_before(name):
    exp = preset(name)
    assert exp.axes is None
    settings = expand(exp)
    assert len(settings) == 36
    for setting in settings:
        assert setting_config(exp, setting) == _previous_config(exp, setting.axes)


def test_the_lambda_exp_preset_builds_the_configs_it_built_before():
    exp = preset("lambda_exp")
    for setting in expand(exp):
        assert setting_config(exp, setting) == _previous_config(exp, setting.axes)


def _only_setting_config(exp: ExpConfig) -> Config:
    (setting,) = expand(exp)
    return setting_config(exp, setting)


def test_the_shared_fields_default_to_the_library_defaults():
    cfg = _only_setting_config(with_axes(preset("tiny"), {"net.N": [4]}))
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
    exp = with_axes(preset("tiny").model_copy(update={field: value}), {"net.N": [4]})
    assert read(_only_setting_config(exp)) == value


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
    exp = with_axes(preset("tiny").model_copy(update=shared), {axis: values})
    assert [read(setting_config(exp, s)) for s in expand(exp)] == values


# ---------------------------------------------------------------- the presets
def test_spatial_n_scan_expands_to_five_spatial_settings(tmp_path):
    exp = preset("spatial_n_scan")
    settings = expand(exp)
    configs = [setting_config(exp, s) for s in settings]
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
    assert (t.weight_lambda, t.rate_lambda, t.optimizer, t.beta) == (60.0, 0.0, "sgld", 1e4)
    assert (t.lr, t.n_epochs) == (1e-2, 250_000)
    assert (t.monitor_int, t.monitor_bins, t.monitor_shuffles) == (5000, 8, 5)


def test_spatial_n_scan_takes_a_length_scale_axis_from_the_cli(tmp_path, capsys):
    args = ["exp", "init", str(tmp_path), "--preset", "spatial_n_scan"]
    assert main([*args, "--axis", f"{LENGTH_AXIS}=[0.05,0.15,0.3]"]) == 0
    assert len(expand(load_exp(tmp_path))) == 15
    capsys.readouterr()
    assert main(["exp", "count", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == "15"


# ---------------------------------------------------------------- spatial axes
def test_a_spatial_axis_changes_the_kernel_and_the_training_hash():
    exp = _tiny_spatial({LENGTH_AXIS: [0.1, 0.2]})
    configs = [setting_config(exp, s) for s in expand(exp)]
    assert [c.spatial.hidden.kernel.length_scale for c in configs] == [0.1, 0.2] * 2
    assert _train_hash(exp, 0) != _train_hash(exp, 1)


@pytest.mark.parametrize(
    "axis, values",
    [
        ("spatial.hidden.embedding.kind", ["grid", "random"]),
        ("spatial.min_rel_eig", [1e-2, 1e-3]),
        ("spatial.box", [1.0, 2.0]),
    ],
)
def test_any_field_of_the_spatial_section_may_be_an_axis(axis, values):
    exp = _tiny_spatial({axis: values})
    assert len(expand(exp)) == 4 and _train_hash(exp, 0) != _train_hash(exp, 1)


def test_a_spatial_axis_without_a_spatial_section_fails_naming_the_axis():
    tree = preset("tiny").model_dump()
    tree["axes"] = {LENGTH_AXIS: [0.1, 0.2]}
    with pytest.raises(ValueError, match=rf"{LENGTH_AXIS}.*spatial section first"):
        ExpConfig.model_validate(tree)


def test_the_same_failure_when_exp_json_is_loaded(tmp_path):
    save_exp(preset("tiny"), tmp_path)
    path = tmp_path / "exp.json"
    path.write_text(path.read_text().replace('"axes": null', f'"axes": {{"{LENGTH_AXIS}": [0.1]}}'))
    with pytest.raises(ValueError, match=LENGTH_AXIS):
        load_exp(tmp_path)


def test_an_invalid_spatial_value_fails_at_load_naming_the_axis():
    with pytest.raises(ValueError, match="spatial.min_rel_eig"):
        _tiny_spatial({"spatial.min_rel_eig": [2.0]})


def test_a_dict_valued_axis_gets_a_short_distinct_directory_name(tmp_path):
    # `spatial` itself is an axis only while the experiment has no section; its values are whole
    # dumps.
    prior = SpatialConfig().model_dump()
    exp = with_axes(preset("tiny"), {"spatial": [None, prior, {**prior, "min_rel_eig": 1e-3}]})
    names = [setting_dir(tmp_path, s).name for s in expand(exp)]
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
    assert main(["exp", "init", root, "--preset", "spatial_tiny"]) == 0
    assert main(["exp", "run", root, "--index", "0"]) == 0
    exp = load_exp(tmp_path)
    states = [setting_status(tmp_path, exp, s) for s in expand(exp)]
    assert states == ["done", "missing"]
    assert main(["exp", "run", root]) == 0  # resume: the done setting is skipped
    capsys.readouterr()
    assert main(["exp", "status", root]) == 0
    assert "2 done, 0 partial, 0 missing, 0 stale" in capsys.readouterr().out


def test_exp_json_names_the_parametrisation_only_when_it_is_whitened():
    """An exp.json written before the fields existed stays byte-identical (the penalty and the
    standard initial law are left out), and a whitened experiment, which starts from the
    prior, writes both fields and loads back to itself."""
    penalty = preset("spatial_tiny")
    text = json.loads(exp_text(penalty))
    assert "parametrisation" not in text["spatial"] and "init" not in text
    assert ExpConfig.model_validate_json(exp_text(penalty)) == penalty
    spatial = penalty.spatial.model_copy(update={"parametrisation": "whitened"})
    whitened = penalty.model_copy(update={"spatial": spatial, "init": "prior"})
    text = json.loads(exp_text(whitened))
    assert text["spatial"]["parametrisation"] == "whitened" and text["init"] == "prior"
    assert ExpConfig.model_validate_json(exp_text(whitened)) == whitened
    with pytest.raises(ValueError, match=r'net\.init="prior"'):
        ExpConfig.model_validate_json(exp_text(penalty.model_copy(update={"spatial": spatial})))


def test_a_parametrisation_axis_trains_the_penalty_and_the_whitened_prior_side_by_side(
    tmp_path, capsys
):
    """`spatial.parametrisation` as an axis, both paths starting from their prior (`--set
    init=prior`): the settings differ in directory and hash, each penalty setting is the
    preset's but for `net.init`, every setting trains on spatial_tiny, and the two paths share
    their history keys; a whitened net is read back with its factors."""
    root = str(tmp_path)
    axis = 'spatial.parametrisation=["penalty","whitened"]'
    args = ["exp", "init", root, "--preset", "spatial_tiny", "--set", "init=prior"]
    assert main([*args, "--axis", axis]) == 0
    exp = load_exp(tmp_path)
    settings = expand(exp)
    configs = [setting_config(exp, s) for s in settings]
    assert [c.spatial.parametrisation for c in configs] == ["penalty", "whitened"] * 2
    assert all(c.net.init == "prior" for c in configs)
    dirs = [setting_dir(tmp_path, s) for s in settings]
    assert len(set(dirs)) == 4 and "whitened" in dirs[1].name and "penalty" in dirs[0].name
    hashes = [_train_hash(exp, i) for i in range(4)]
    assert len(set(hashes)) == 4
    plain = preset("spatial_tiny")
    for config, setting in zip(configs[::2], expand(plain), strict=True):
        standard = config.model_copy(
            update={"net": config.net.model_copy(update={"init": "standard"})}
        )
        assert standard == setting_config(plain, setting)
    assert main(["exp", "run", root]) == 0
    capsys.readouterr()
    assert main(["exp", "status", root]) == 0
    assert "4 done, 0 partial, 0 missing, 0 stale" in capsys.readouterr().out
    dtype = torch.float32
    penalty_net, penalty_history = load_trained(dirs[0], configs[0], torch.device("cpu"), dtype)
    whitened_net, whitened_history = load_trained(dirs[1], configs[1], torch.device("cpu"), dtype)
    assert penalty_net.whitening is None and whitened_net.whitening is not None
    assert set(penalty_history) == set(whitened_history)
    assert penalty_history["epoch"] == whitened_history["epoch"]
    assert set(penalty_net.state_dict()) == set(whitened_net.state_dict())


class _Interrupt(Exception):
    pass


def test_an_interrupted_spatial_setting_resumes_from_its_checkpoint(tmp_path):
    exp = preset("spatial_tiny")
    setting = expand(exp)[0]

    def stop_at_epoch_4(line: str) -> None:
        if line.startswith("[00004/"):
            raise _Interrupt

    with pytest.raises(_Interrupt):
        run_setting(exp, 0, tmp_path, log=stop_at_epoch_4)
    assert setting_status(tmp_path, exp, setting) == "partial"
    lines: list[str] = []
    run_setting(exp, 0, tmp_path, log=lines.append)
    assert any(line.startswith("resumed from") and "at epoch" in line for line in lines)
    assert setting_status(tmp_path, exp, setting) == "done"


def test_exp_run_stage_solve_on_a_spatial_setting_is_a_one_line_error(tmp_path, capsys):
    root = str(tmp_path)
    assert main(["exp", "init", root, "--preset", "spatial_tiny"]) == 0
    capsys.readouterr()
    assert main(["exp", "run", root, "--index", "0", "--stage", "solve"]) == 1
    err = capsys.readouterr().err.strip()
    assert err.startswith("error:") and "no spatial prior" in err and "\n" not in err


# ---------------------------------------------------------------- spatial_exp
PHIS = ["id", "tanh", "relu"]
LATENTS = [2, 4]
LENGTHS = [0.05, 0.1, 0.15, 0.2]


def test_spatial_exp_expands_to_every_phi_latent_length_combination_once(tmp_path):
    exp = preset("spatial_exp")
    settings = expand(exp)
    configs = [setting_config(exp, s) for s in settings]
    seen = [(c.net.phi, c.data.n_latents, c.spatial.hidden.kernel.length_scale) for c in configs]
    assert len(settings) == 24
    assert sorted(seen) == sorted((p, k, ell) for p in PHIS for k in LATENTS for ell in LENGTHS)
    names = [setting_dir(tmp_path, s).name for s in settings]
    assert len(set(names)) == 24 and all(len(n) < 60 for n in names)


def test_spatial_exp_fixed_values():
    exp = preset("spatial_exp")
    for c in (setting_config(exp, s) for s in expand(exp)):
        assert c.spatial.input.kernel.length_scale == 0.15
        assert c.spatial.output.kernel.length_scale == 0.15
        assert c.spatial.min_rel_eig == 1e-2
        assert (c.net.N, c.data.P, c.net.L, c.net.g, c.dtype) == (512, 128, 1, 1.0, "float64")
        assert (c.data.D_in, c.data.latent_min, c.data.latent_max) == (64, -1.0, 1.0)
        t = c.train
        assert (t.rate_lambda, t.weight_lambda, t.optimizer, t.beta) == (0.0, 20.0, "sgld", 1e4)
        assert (t.lr, t.n_epochs) == (1e-2, 250_000)
        assert (t.monitor_int, t.monitor_bins, t.monitor_shuffles) == (5000, 8, 5)


def test_spatial_exp_counts_24_from_the_cli(tmp_path, capsys):
    assert main(["exp", "init", str(tmp_path), "--preset", "spatial_exp"]) == 0
    capsys.readouterr()
    assert main(["exp", "count", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == "24"


def test_spatial_exp_trains_id_tanh_relu_at_the_smallest_length_scale(tmp_path):
    """The preset shrunk to seconds: finite losses for each activation."""
    shrink = {
        "N": 16,
        "P": 32,
        "n_epochs": 6,
        "log_int": 2,
        "save_int": 2,
        "monitor_int": 2,
        "monitor_bins": 4,
        "monitor_shuffles": 2,
        "linear_teacher": preset("spatial_tiny").linear_teacher,
        "device": "cpu",
    }
    small = preset("spatial_exp").model_copy(update=shrink)
    settings = expand(small)
    # axes in order: phi slowest, then n_latents, then length; index 0 of a block is (k 2, 0.05).
    # The last case is tanh with 4 latents, so the axis (not the base value 2) reaches the data.
    block = len(LATENTS) * len(LENGTHS)
    cases = [
        (0, "id", 2),
        (block, "tanh", 2),
        (2 * block, "relu", 2),
        (block + len(LENGTHS), "tanh", 4),
    ]
    for index, phi, k in cases:
        config = setting_config(small, settings[index])
        assert (config.net.phi, config.data.n_latents) == (phi, k)
        assert config.spatial.hidden.kernel.length_scale == 0.05
        run_setting(small, index, tmp_path, log=lambda line: None)
        cache = setting_dir(tmp_path, settings[index])
        _, history = load_trained(cache, config, torch.device("cpu"), torch.float64)
        assert np.isfinite(history["loss"]).all()


def _spatial_exp_config(phi: str, n_latents: int, length: float):
    exp = preset("spatial_exp")
    (setting,) = [
        s
        for s in expand(exp)
        if s.axes
        == {
            "net.phi": phi,
            "data.n_latents": n_latents,
            "spatial.hidden.kernel.length_scale": length,
        }
    ]
    return setting_config(exp, setting)


def test_spatial_exp_four_latents_gives_four_factors_and_a_3_to_the_4_test_set():
    cfg = _spatial_exp_config("tanh", 4, 0.1)
    dataset = make_dataset(cfg.data, torch.device("cpu"), torch.float64)
    assert sorted(dataset.factors) == ["z0", "z1", "z2", "z3"]
    assert latent_lattice_dataset(cfg.data, dataset).x.shape[0] == 3**4


@pytest.mark.parametrize("length", LENGTHS)
def test_spatial_exp_spaces_give_the_hidden_population_the_axis_length_scale(length):
    cfg = _spatial_exp_config("relu", 2, length)
    m, n_out = task_dims(cfg.data)
    (u_space, v_space) = build_spaces(cfg.spatial, m, cfg.net.N, cfg.net.L, n_out)
    assert u_space.post is not None and v_space.pre is not None
    assert u_space.post.kernel.length_scale == v_space.pre.kernel.length_scale == length
    assert u_space.pre.kernel.length_scale == v_space.post.kernel.length_scale == 0.15
