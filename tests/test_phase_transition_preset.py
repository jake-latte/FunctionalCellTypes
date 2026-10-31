"""The `phase_transition` preset: d x lambda_max, a linear one-hidden-layer ensemble under the
lambda_max prior with random embeddings; its settings, kernels, SGLD stability and a tiny run."""

import json

import numpy as np
import pytest
import torch

from fct.cli import main
from fct.config import Config
from fct.spatial import build_spaces, parameter_factors
from fct.sweep import (
    FIXED_GRID_PRESETS,
    expand,
    grid_hash,
    load_sweep,
    preset,
    run_setting,
    setting_config,
    setting_dir,
    with_grid,
)
from fct.train import _check_sgld_step_resolves_the_prior, load_trained

DS = [1, 2, 3, 4]
LAMBDAS = [0.1, 0.5, 0.75, 0.9, 0.99]
N_FEATURES = 64  # the preset's input size m


def _settings():
    sweep = preset("phase_transition")
    return sweep, expand(sweep)


def _spaces(cfg: Config):
    return build_spaces(cfg.spatial, m=N_FEATURES, N=cfg.net.N, L=1, n_out=2)


def test_the_preset_is_a_fixed_grid_preset_and_rejects_n_lambda():
    assert "phase_transition" in FIXED_GRID_PRESETS
    with pytest.raises(ValueError, match="n_lambda"):
        preset("phase_transition", n_lambda=3)


def test_the_twenty_settings_are_d_slowest_in_a_stable_order():
    _, settings = _settings()
    assert [(s.axes["spatial.d"], s.axes["spatial.lambda_max"]) for s in settings] == [
        (d, lam) for d in DS for lam in LAMBDAS
    ]


def test_every_setting_is_a_linear_one_layer_ensemble_with_random_embeddings():
    sweep, settings = _settings()
    for setting in settings:
        cfg = setting_config(sweep, setting)
        assert (cfg.net.phi, cfg.net.L, cfg.net.B) == ("id", 1, 8)
        assert (cfg.net.N, cfg.data.P, cfg.train.n_epochs) == (512, 128, 250_000)
        assert cfg.train.weight_lambda == 1.0 and cfg.train.optimizer == "sgld"
        for pop in (cfg.spatial.input, cfg.spatial.hidden, cfg.spatial.output):
            assert pop.embedding.kind == "random" and pop.kernel.length_scale == 0.15


def test_the_sweep_json_written_by_the_cli_loads_back_to_the_preset(tmp_path):
    assert main(["sweep", "init", str(tmp_path), "--preset", "phase_transition"]) == 0
    loaded = load_sweep(tmp_path)
    assert loaded == preset("phase_transition")
    assert grid_hash(loaded) == grid_hash(preset("phase_transition"))


def test_the_ensemble_size_does_not_enter_any_other_sweep_json():
    # B is an axis of this preset alone, so no other sweep.json gains a field or an axis.
    assert "B" not in preset("spatial_grid").model_dump()
    assert "net.B" not in preset("spatial_grid").grid


@pytest.mark.parametrize("d", DS)
def test_kernels_rebuilt_from_config_json_are_psd_with_top_eigenvalue_lambda_max(d):
    sweep, settings = _settings()
    for setting in (s for s in settings if s.axes["spatial.d"] == d):
        saved = Config.model_validate(json.loads(setting_config(sweep, setting).model_dump_json()))
        spaces = _spaces(saved)
        for pop in [spaces[0].pre] + [s.post for s in spaces]:
            eigs = torch.linalg.eigvalsh(pop.K)
            assert eigs[0].item() > -1e-10 * eigs[-1].item()
            assert eigs[-1].item() == pytest.approx(setting.axes["spatial.lambda_max"], rel=1e-10)


def test_sgld_is_stable_at_the_preset_lr_for_all_settings():
    sweep, settings = _settings()
    for setting in settings:
        cfg = setting_config(sweep, setting)
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
    return with_grid(base, {"spatial.d": [3], "spatial.lambda_max": [0.5], "net.B": [2]})


def test_a_reduced_setting_trains_end_to_end(tmp_path):
    sweep = _reduced_setting()
    (setting,) = expand(sweep)
    run_setting(sweep, 0, tmp_path, log=lambda line: None)
    cfg = setting_config(sweep, setting)
    _, history = load_trained(
        setting_dir(tmp_path, setting), cfg, torch.device("cpu"), torch.float64
    )
    assert np.isfinite(history["loss"]).all()
