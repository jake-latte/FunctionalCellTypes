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
    exp_hash,
    exp_text,
    expand,
    load_exp,
    preset,
    preset_names,
    run_setting,
    setting_config,
    setting_dir,
    with_axes,
)
from fct.spatial import build_spaces, parameter_factors, periodic_gaussian_kernel
from fct.train import _check_sgld_step_resolves_the_prior, load_trained

DS = [1, 2, 3, 4]
LAMBDAS = [0.1, 0.5, 0.75, 0.9, 0.99]
# sha256 of the exp.json text of each preset. OLD_EXP_JSON is the text the Python presets wrote
# (lambda_exp at 7 points per axis, which is what the file holds); the file adds a description to
# it and nothing else, so the text with that key removed must still be the old one.
OLD_EXP_JSON = {
    "tiny": "712cdde2fac021c1e596ddf21d6d51c4f9f9ee73661b580ac9634f3465a1a14a",
    "full": "25b55b25a244b6337c2d33008c33dc862eb976644d62c847223fe4cee3bbcd73",
    "modularity": "25b55b25a244b6337c2d33008c33dc862eb976644d62c847223fe4cee3bbcd73",
    "lambda_exp": "cbdfb5c226b67f4867e7f47d05fed64b283afa35bb148aa49ba04489ba15e1be",
    "spatial_tiny": "a00715e7482b7163121feaa9ea303ac246fdd5ba01aad86876054d3124e06a83",
    "spatial_n_scan": "1abd77565bc39ada4188857a397a7ca1222518915399aa85159479bf21e08c1e",
    "spatial_exp": "2165b782d399d9ec44373dabc3b0fdc096ef8b87fb0ab5772f3a8cbc5c7def4d",
    "phase_transition": "391a7da6e4320bae43e8538d4af90a3ffca694ff22f5580a4657ecbd56c9b3e7",
}
NEW_EXP_JSON = {
    "tiny": "f42d0b55033e3b77cb42f9b289bb07f417fcd065146eaafc7cf6e42fde01e3ae",
    "full": "f910c5de65dee0f40ba9779f96c1c8754544fe9b8020904a30b44c0051b5b304",
    "modularity": "7f4c526e57140c010d9f7fa1f063f6756e12461132a32b82037c74561fb2da64",
    "lambda_exp": "1790f54686c064dd97cfac9daac4c62f64565475f1fbc150c10fce08288402c2",
    "spatial_tiny": "dc9d9a2e902d202592063b9172241e067141cd661426c9f72cac329b0d6b10fe",
    "spatial_n_scan": "13d8e5a817bedf0548182143f4b94b42044e63a7fd601b22a334fca4207eb47c",
    "spatial_exp": "10533d2be87cf11bf83d21bd3bc202d90fd1da0c7089b232abbde72a58512477",
    "phase_transition": "3c1684daacf2f562ee3240c543fd9d1625605fa7473c4be229d2a761f08917fa",
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
