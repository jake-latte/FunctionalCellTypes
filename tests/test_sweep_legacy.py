"""A sweep.json written before the grid became configurable keeps its meaning.

The fixtures were produced by the fixed-grid code: `legacy_sweep.json` is its `tiny` preset,
`legacy_settings.json` the 36 settings it expanded to (directory and swept values, in order).
"""

import json
from pathlib import Path

from fct.config import Config, NetConfig, TrainConfig
from fct.sweep import (
    SweepConfig,
    data_config,
    expand,
    load_sweep,
    preset,
    setting_config,
    setting_dir,
)

FIXTURES = Path(__file__).parent / "fixtures"
ROOT = Path("/sweep-root")


def _legacy_sweep(tmp_path) -> SweepConfig:
    (tmp_path / "sweep.json").write_text((FIXTURES / "legacy_sweep.json").read_text())
    return load_sweep(tmp_path)


def _legacy_settings() -> list[dict]:
    return json.loads((FIXTURES / "legacy_settings.json").read_text())


def _fixed_grid_config(sweep: SweepConfig, task, L, phi, rate_lambda) -> Config:
    """The config the fixed-grid code built for one setting."""
    return Config(
        data=data_config(sweep, task),
        net=NetConfig(N=sweep.N, L=L, phi=phi, seed=sweep.seeds.net),
        train=TrainConfig(
            n_epochs=sweep.n_epochs,
            lr=sweep.lr,
            rate_lambda=rate_lambda,
            log_int=sweep.log_int,
            save_int=sweep.save_int,
            seed=sweep.seeds.train,
            monitor_int=sweep.monitor_int,
            monitor_bins=sweep.monitor_bins,
            monitor_shuffles=sweep.monitor_shuffles,
        ),
        dtype=sweep.dtype,
        device=sweep.device,
    )


def test_a_sweep_json_without_a_grid_loads(tmp_path):
    assert _legacy_sweep(tmp_path).grid is None


def test_legacy_sweep_expands_to_the_same_settings_and_directories(tmp_path):
    settings = expand(_legacy_sweep(tmp_path))
    expected = _legacy_settings()
    assert len(settings) == len(expected) == 36
    for setting, row in zip(settings, expected, strict=True):
        assert str(setting_dir(ROOT, setting).relative_to(ROOT)) == row["dir"]
        got = (setting.task, setting.L, setting.phi, setting.rate_lambda)
        assert got == (row["task"], row["L"], row["phi"], row["rate_lambda"])


def test_legacy_sweep_builds_the_same_configs(tmp_path):
    sweep = _legacy_sweep(tmp_path)
    for setting, row in zip(expand(sweep), _legacy_settings(), strict=True):
        expected = _fixed_grid_config(sweep, row["task"], row["L"], row["phi"], row["rate_lambda"])
        assert setting_config(sweep, setting) == expected


def test_the_modularity_preset_is_the_legacy_grid():
    got = [str(setting_dir(ROOT, s).relative_to(ROOT)) for s in expand(preset("modularity"))]
    assert got == [row["dir"] for row in _legacy_settings()]
