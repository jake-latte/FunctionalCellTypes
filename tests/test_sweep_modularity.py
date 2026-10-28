"""A sweep.json without `grid` is the modularity grid, and keeps its meaning.

`modularity_sweep.json` is the `tiny` preset of commit e1492ca (the last before the grid became
configurable) as serialised, and `modularity_settings.json` holds, for each of the 36 settings it
expanded to (in order), the directory and the swept values that code produced (extracted with
`git archive e1492ca fct | tar -x -C tmp/old`, run with `PYTHONPATH=tmp/old`). Those, and
`test_it_builds_the_same_configs`, are the independent check against the pre-grid code.

Each row's `train_hash` is NOT from that code: it is the `section_hash("data", "net", "train")`
of the setting's config computed by the current code (LOSS_VERSION 4, MODEL_VERSION 2), a pin
that catches an unintended hash change. The hash test sets both versions, so a deliberate
version change does not make it fail.
"""

import json
from pathlib import Path

import pytest

import fct.config
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
FIXTURE_LOSS_VERSION, FIXTURE_MODEL_VERSION = 4, 2
ROOT = Path("root")


def _sweep(tmp_path) -> SweepConfig:
    (tmp_path / "sweep.json").write_text((FIXTURES / "modularity_sweep.json").read_text())
    return load_sweep(tmp_path)


def _rows() -> list[dict]:
    return json.loads((FIXTURES / "modularity_settings.json").read_text())


def _fixed_grid_config(sweep: SweepConfig, task, L, phi, rate_lambda) -> Config:
    """The config the pre-grid code built for one setting."""
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
    assert _sweep(tmp_path).grid is None


def test_it_expands_to_the_same_settings_and_directories(tmp_path):
    settings = expand(_sweep(tmp_path))
    rows = _rows()
    assert len(settings) == len(rows) == 36
    for setting, row in zip(settings, rows, strict=True):
        assert str(setting_dir(ROOT, setting).relative_to(ROOT)) == row["dir"]
        axes = setting.axes
        got = (setting.task, axes["net.L"], axes["net.phi"], axes["train.rate_lambda"])
        assert got == (row["task"], row["L"], row["phi"], row["rate_lambda"])


def test_it_builds_the_same_configs(tmp_path):
    sweep = _sweep(tmp_path)
    for setting, row in zip(expand(sweep), _rows(), strict=True):
        expected = _fixed_grid_config(sweep, row["task"], row["L"], row["phi"], row["rate_lambda"])
        assert setting_config(sweep, setting) == expected


def test_its_train_hashes_are_pinned(tmp_path, monkeypatch):
    monkeypatch.setattr(fct.config, "LOSS_VERSION", FIXTURE_LOSS_VERSION)
    monkeypatch.setattr(fct.config, "MODEL_VERSION", FIXTURE_MODEL_VERSION)
    sweep = _sweep(tmp_path)
    hashes = [setting_config(sweep, s).section_hash("data", "net", "train") for s in expand(sweep)]
    assert hashes == [row["train_hash"] for row in _rows()]


@pytest.mark.parametrize("name", ["full", "modularity"])
def test_the_modularity_presets_use_the_original_directories(name):
    got = [str(setting_dir(ROOT, s).relative_to(ROOT)) for s in expand(preset(name))]
    assert got == [row["dir"] for row in _rows()]
