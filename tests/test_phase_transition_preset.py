"""The `phase_transition` preset: d x lambda_max, a linear one-hidden-layer ensemble under the
lambda_max prior with random embeddings; its settings, kernels, SGLD stability and a tiny run."""

import hashlib
import json

import numpy as np
import pytest
import torch

from fct.cli import main
from fct.config import Config
from fct.data import task_dims
from fct.exp import (
    ExpConfig,
    exp_hash,
    exp_text,
    expand,
    load_exp,
    preset,
    preset_names,
    run_setting,
    same_experiment,
    setting_config,
    setting_dir,
    with_axes,
)
from fct.spatial import build_spaces, parameter_factors, periodic_gaussian_kernel
from fct.train import _check_sgld_step_resolves_the_prior, load_trained

DS = [1, 2, 3, 4]
LAMBDAS = [0.1, 0.5, 0.75, 0.9, 0.99]
# sha256 of the exp.json text of each preset. OLD_EXP_JSON is the text the Python presets wrote
# (lambda_exp at 7 points per axis, which is what the file holds), with the linear teacher's
# latent interval spelled `latent_min` / `latent_max`; the file adds a description to it and
# nothing else, so the text with that key removed must still be the old one.
OLD_EXP_JSON = {
    "full": "0a38938025be6af607416f68131dc73afb8e8bc116495e973b14459facc4c044",
    "lambda_exp": "8e0c88343f0413f328a64e269bab370e16fb0e700db577b8e236a435db833ee3",
    "modularity": "0a38938025be6af607416f68131dc73afb8e8bc116495e973b14459facc4c044",
    "phase_transition": "e0e498cc65190b09e7c8b09389aaef54a05f067ecaa596cc00ef26d1082bdae2",
    "spatial_exp": "b6ab301de6953a037275e4c4a7fdab1672aac9d756c764c85be943e8e626adf6",
    "spatial_n_scan": "913e4c32aa4bc666f47703c9ef6ca5198686cf0d718fc3500b3d3a5f265ab6fd",
    "spatial_tiny": "d08077ea9dd2638ddb737b7509ff22c3287e68dc8f81888cf53eea141a7fcd28",
    "tiny": "49e6c5936f9854c106bc056637c6e07fd6ef49cf0e6a81dc23bb5d33accf8b02",
}
NEW_EXP_JSON = {
    "full": "ae24961b09720fb9439e140d34041a108c7f067982ad9d8442da42d798db810f",
    "lambda_exp": "841dcdc2317868d445b4ddec6a9dfd403438498d7733fece7c5aab827be3f52d",
    "modularity": "46f5fdbc004ffad557dcc63c9549963aeac2c3b556f19ed30194962fa451d51c",
    "phase_transition": "d856bc439c305fa21c4508ec52ce8df3366803ba885a63a69258745017dc25b2",
    "spatial_exp": "3a5126b0509804ecd716a675c1e12d47b564dc3b89aab283213cc1476040c15a",
    "spatial_n_scan": "dc50b8aec339769a35d9f146cda9e858f9354e84105f705d2923f8b67ea9a564",
    "spatial_tiny": "6afc7b586b1eaef6b72090785c10d24d6a6f08e1dcf32bcc8a4333174cfe71b7",
    "tiny": "5a5b65629334c478eb9a0c13f2401182a4856bd5e467e877bcef7afa84e65a08",
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _settings():
    exp = preset("phase_transition")
    return exp, expand(exp)


def _spaces(cfg: Config):
    """The real layer spaces of a setting: the task's own input and output sizes."""
    m, n_out = task_dims(cfg.data)
    return build_spaces(cfg.spatial, m=m, N=cfg.net.N, L=cfg.net.L, n_out=n_out)


def _top_eigenvalue(K: torch.Tensor, iterations: int = 20_000) -> float:
    """Power iteration, independent of `eigvalsh` (K is PSD, so it converges to the top one)."""
    v = torch.ones(K.shape[0], dtype=K.dtype)
    v = v + 0.01 * torch.arange(K.shape[0], dtype=K.dtype)  # not orthogonal to the top mode
    for _ in range(iterations):
        w = K @ v
        v = w / torch.linalg.norm(w)
    return (v @ K @ v).item()


def test_the_twenty_settings_are_d_slowest_in_a_stable_order():
    _, settings = _settings()
    assert [(s.axes["spatial.d"], s.axes["spatial.lambda_max"]) for s in settings] == [
        (d, lam) for d in DS for lam in LAMBDAS
    ]


def test_every_setting_is_a_linear_one_layer_ensemble_with_random_embeddings():
    exp, settings = _settings()
    for setting in settings:
        cfg = setting_config(exp, setting)
        assert (cfg.net.phi, cfg.net.L, cfg.net.B) == ("id", 1, 8)
        assert (cfg.net.N, cfg.data.P, cfg.train.n_epochs) == (512, 128, 250_000)
        assert cfg.train.weight_lambda == 1.0 and cfg.train.optimizer == "sgld"
        for pop in (cfg.spatial.input, cfg.spatial.hidden, cfg.spatial.output):
            assert pop.embedding.kind == "random" and pop.kernel.length_scale == 0.15


def test_the_exp_json_written_by_the_cli_loads_back_to_the_preset(tmp_path):
    assert main(["exp", "init", str(tmp_path), "--preset", "phase_transition"]) == 0
    loaded = load_exp(tmp_path)
    assert loaded == preset("phase_transition")
    assert exp_hash(loaded) == exp_hash(preset("phase_transition"))


def test_the_pins_cover_every_preset():
    assert set(NEW_EXP_JSON) == set(OLD_EXP_JSON) == set(preset_names())


@pytest.mark.parametrize("name", sorted(NEW_EXP_JSON))
def test_a_preset_writes_the_exp_json_it_wrote_before_plus_its_description(name):
    exp = preset(name)
    text = exp_text(exp)
    assert _sha(text) == NEW_EXP_JSON[name]
    lines = text.splitlines(keepends=True)
    assert lines[1].startswith('  "description": ')
    assert _sha("".join(lines[:1] + lines[2:])) == OLD_EXP_JSON[name]


@pytest.mark.parametrize("name", sorted(NEW_EXP_JSON))
def test_an_exp_json_written_before_B_existed_loads_with_the_default_B(name):
    text = exp_text(preset(name))
    assert _sha(text) == NEW_EXP_JSON[name]  # the pinned text, written by the code without B
    assert "B" not in json.loads(text)
    loaded = ExpConfig.model_validate_json(text)
    assert loaded.B == 1
    assert same_experiment(loaded, preset(name))


@pytest.mark.parametrize("d", DS)
def test_kernels_rebuilt_from_config_json_are_psd_with_top_eigenvalue_lambda_max(d):
    exp, settings = _settings()
    for setting in (s for s in settings if s.axes["spatial.d"] == d):
        saved = Config.model_validate(json.loads(setting_config(exp, setting).model_dump_json()))
        spaces = _spaces(saved)
        for pop in [spaces[0].pre] + [s.post for s in spaces]:
            lam = setting.axes["spatial.lambda_max"]
            assert torch.linalg.eigvalsh(pop.K)[0].item() > -1e-10 * lam
            assert _top_eigenvalue(pop.K) == pytest.approx(lam, rel=1e-6)
            # diag(K) = lambda_max / top(G) for the unit-amplitude kernel G (diag 1), with the
            # top eigenvalue of G found without the code under test's `eigvalsh`
            x = pop.embedding.positions
            G = periodic_gaussian_kernel(x, x, pop.kernel.length_scale, saved.spatial.box, 1.0)
            assert torch.diagonal(pop.K) == pytest.approx(lam / _top_eigenvalue(G), rel=1e-6)


def test_sgld_is_stable_at_the_preset_lr_for_all_settings():
    exp, settings = _settings()
    for setting in settings:
        cfg = setting_config(exp, setting)
        # raises ValueError if lr * weight_lambda / (2 beta) * stiffest curvature >= 1
        _check_sgld_step_resolves_the_prior(parameter_factors(_spaces(cfg)), cfg.train)


def _reduced_setting():
    """One setting of the preset (d = 3, lambda_max 0.5) at a few epochs and a small net."""
    base = preset("phase_transition").model_copy(
        update={
            "N": 6, "P": 16, "n_epochs": 4, "log_int": 2, "save_int": 2, "monitor_int": 2,
            "monitor_bins": 4, "monitor_shuffles": 2, "device": "cpu",
            "linear_teacher": preset("tiny").linear_teacher.model_copy(update={"n_features": 8}),
        }
    )  # fmt: skip
    return with_axes(base, {"spatial.d": [3], "spatial.lambda_max": [0.5], "net.B": [2]})


def test_a_reduced_setting_trains_end_to_end(tmp_path):
    exp = _reduced_setting()
    (setting,) = expand(exp)
    run_setting(exp, 0, tmp_path, log=lambda line: None)
    directory = setting_dir(tmp_path, setting)
    assert directory.name == "s0000_d3_lambda_max0.5_B2"
    record = json.loads((directory / "setting.json").read_text())
    assert record["axes"] == {"spatial.d": 3, "spatial.lambda_max": 0.5, "net.B": 2}
    cfg = setting_config(exp, setting)
    _, history = load_trained(directory, cfg, torch.device("cpu"), torch.float64)
    assert np.isfinite(history["loss"]).all()
    # monitor records are (monitor calls, ensemble members, hidden layers)
    n_calls = len(history["monitor_epoch"])
    assert np.shape(history["monitor_mir"]) == (n_calls, cfg.net.B, cfg.net.L)
    assert np.isfinite(history["monitor_mir"]).all()
