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
    FIXED_AXES_PRESETS,
    PRESET_NAMES,
    exp_hash,
    expand,
    lambda_exp,
    load_exp,
    preset,
    run_setting,
    setting_config,
    setting_dir,
    with_axes,
)
from fct.spatial import build_spaces, parameter_factors, periodic_gaussian_kernel
from fct.train import _check_sgld_step_resolves_the_prior, load_trained

DS = [1, 2, 3, 4]
LAMBDAS = [0.1, 0.5, 0.75, 0.9, 0.99]
# sha256 of the exp.json text of each other preset, and exp_hash of lambda_exp(2). The texts are
# those of the code before phase_transition was added, with the key `grid` renamed `axes` (the
# only change; the hashes of the old texts were checked to match once the key was swapped back).
PINNED_EXP_JSON = {
    "tiny": "712cdde2fac021c1e596ddf21d6d51c4f9f9ee73661b580ac9634f3465a1a14a",
    "full": "25b55b25a244b6337c2d33008c33dc862eb976644d62c847223fe4cee3bbcd73",
    "modularity": "25b55b25a244b6337c2d33008c33dc862eb976644d62c847223fe4cee3bbcd73",
    "lambda_exp": "1da65a4c0ff88918780e6025e32ff40061e612c466a14fa9386644b93c9efd32",
    "spatial_tiny": "a00715e7482b7163121feaa9ea303ac246fdd5ba01aad86876054d3124e06a83",
    "spatial_n_scan": "1abd77565bc39ada4188857a397a7ca1222518915399aa85159479bf21e08c1e",
    "spatial_exp": "2165b782d399d9ec44373dabc3b0fdc096ef8b87fb0ab5772f3a8cbc5c7def4d",
}
PINNED_LAMBDA_EXP_HASH = "781bf3c7d26c"


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


def test_the_preset_is_a_fixed_axes_preset_and_rejects_n_lambda():
    assert "phase_transition" in FIXED_AXES_PRESETS
    with pytest.raises(ValueError, match="n_lambda"):
        preset("phase_transition", n_lambda=3)


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


@pytest.mark.parametrize("name", sorted(set(PRESET_NAMES) - {"phase_transition"}))
def test_every_other_preset_writes_the_exp_json_it_wrote_before(name):
    exp = preset(name, 2) if name == "lambda_exp" else preset(name)
    text = exp.model_dump_json(indent=2) + "\n"  # what save_exp writes
    assert hashlib.sha256(text.encode()).hexdigest() == PINNED_EXP_JSON[name]
    assert "B" not in exp.model_dump()


def test_lambda_exp_keeps_the_exp_hash_the_old_preset_had():
    assert exp_hash(lambda_exp(2)) == PINNED_LAMBDA_EXP_HASH


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
