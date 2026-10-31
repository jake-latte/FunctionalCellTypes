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
from fct.spatial import build_spaces, parameter_factors, periodic_gaussian_kernel
from fct.sweep import (
    FIXED_GRID_PRESETS,
    PRESET_NAMES,
    expand,
    grid_hash,
    lambda_grid,
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
# sha256 of the sweep.json text of each other preset, and grid_hash of lambda_grid(2), computed
# from the code before phase_transition was added (the old phase_transition(2) is lambda_grid(2)).
PINNED_SWEEP_JSON = {
    "tiny": "e531b5d560226c521f548deb601d4d1f0adf434ba2661ff1b44dc02dc0248336",
    "full": "3441235417fa7e10097cde58790e6d63666394019d85d17c8c767b4979d2ac8a",
    "modularity": "3441235417fa7e10097cde58790e6d63666394019d85d17c8c767b4979d2ac8a",
    "lambda_grid": "5c36b97ef862554e82410ee932e4006a810bfd7f54ee694cb0c751699a2512ae",
    "spatial_tiny": "8bab6f6b1d4248e58ee87ffe723616240b0836d08537623c4ca8d7d0a88872c7",
    "spatial_n_scan": "b40bce5dba2eb08bdfbfa518fcaa849bf071028f2848cde19328fb7902ac66d8",
    "spatial_grid": "47f643d0b2f515b02e553b00cd2e43c0fa606bb8e7074797c1ee9c8d71b4b897",
}
PINNED_LAMBDA_GRID_HASH = "781bf3c7d26c"


def _settings():
    sweep = preset("phase_transition")
    return sweep, expand(sweep)


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


@pytest.mark.parametrize("name", sorted(set(PRESET_NAMES) - {"phase_transition"}))
def test_every_other_preset_writes_the_sweep_json_it_wrote_before(name):
    sweep = preset(name, 2) if name == "lambda_grid" else preset(name)
    text = sweep.model_dump_json(indent=2) + "\n"  # what save_sweep writes
    assert hashlib.sha256(text.encode()).hexdigest() == PINNED_SWEEP_JSON[name]
    assert "B" not in sweep.model_dump()


def test_lambda_grid_keeps_the_grid_hash_the_old_preset_had():
    assert grid_hash(lambda_grid(2)) == PINNED_LAMBDA_GRID_HASH


@pytest.mark.parametrize("d", DS)
def test_kernels_rebuilt_from_config_json_are_psd_with_top_eigenvalue_lambda_max(d):
    sweep, settings = _settings()
    for setting in (s for s in settings if s.axes["spatial.d"] == d):
        saved = Config.model_validate(json.loads(setting_config(sweep, setting).model_dump_json()))
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
    directory = setting_dir(tmp_path, setting)
    assert directory.name == "s0000_d3_lambda_max0.5_B2"
    record = json.loads((directory / "setting.json").read_text())
    assert record["axes"] == {"spatial.d": 3, "spatial.lambda_max": 0.5, "net.B": 2}
    cfg = setting_config(sweep, setting)
    _, history = load_trained(directory, cfg, torch.device("cpu"), torch.float64)
    assert np.isfinite(history["loss"]).all()
    # monitor records are (monitor calls, ensemble members, hidden layers)
    n_calls = len(history["monitor_epoch"])
    assert np.shape(history["monitor_mir"]) == (n_calls, cfg.net.B, cfg.net.L)
    assert np.isfinite(history["monitor_mir"]).all()
