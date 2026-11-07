"""An exp.json without `axes` is the modularity experiment, and keeps its meaning.

`modularity_exp.json` is the `tiny` preset of commit e1492ca (the last before the axes became
configurable) as serialised, and `modularity_settings.json` holds, for each of the 36 settings it
expanded to (in order), the directory and the varied values that code produced (extracted with
`git archive e1492ca fct | tar -x -C tmp/old`, run with `PYTHONPATH=tmp/old`), except that its
`latent_range: 1.0` is spelled `latent_min: -1.0, latent_max: 1.0`. Those, and
`test_it_builds_the_same_configs`, are the independent check against the code that had fixed axes.

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
from fct.exp import (
    ExpConfig,
    data_config,
    expand,
    load_exp,
    preset,
    setting_config,
    setting_dir,
)

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_LOSS_VERSION, FIXTURE_MODEL_VERSION = 4, 2
ROOT = Path("root")


def _exp(tmp_path) -> ExpConfig:
    (tmp_path / "exp.json").write_text((FIXTURES / "modularity_exp.json").read_text())
    return load_exp(tmp_path)


def _rows() -> list[dict]:
    return json.loads((FIXTURES / "modularity_settings.json").read_text())


def _fixed_axes_config(exp: ExpConfig, task, L, phi, rate_lambda) -> Config:
    """The config the fixed-axes code built for one setting."""
    return Config(
        data=data_config(exp, task),
        net=NetConfig(N=exp.N, L=L, phi=phi, seed=exp.seeds.net),
        train=TrainConfig(
            n_epochs=exp.n_epochs,
            lr=exp.lr,
            rate_lambda=rate_lambda,
            log_int=exp.log_int,
            save_int=exp.save_int,
            seed=exp.seeds.train,
            monitor_int=exp.monitor_int,
            monitor_bins=exp.monitor_bins,
            monitor_shuffles=exp.monitor_shuffles,
        ),
        dtype=exp.dtype,
        device=exp.device,
    )


def test_an_exp_json_without_axes_loads(tmp_path):
    assert _exp(tmp_path).axes is None


def test_it_expands_to_the_same_settings_and_directories(tmp_path):
    settings = expand(_exp(tmp_path))
    rows = _rows()
    assert len(settings) == len(rows) == 36
    for setting, row in zip(settings, rows, strict=True):
        assert str(setting_dir(ROOT, setting).relative_to(ROOT)) == row["dir"]
        axes = setting.axes
        got = (setting.task, axes["net.L"], axes["net.phi"], axes["train.rate_lambda"])
        assert got == (row["task"], row["L"], row["phi"], row["rate_lambda"])


def test_it_builds_the_same_configs(tmp_path):
    exp = _exp(tmp_path)
    for setting, row in zip(expand(exp), _rows(), strict=True):
        expected = _fixed_axes_config(exp, row["task"], row["L"], row["phi"], row["rate_lambda"])
        assert setting_config(exp, setting) == expected


def test_its_train_hashes_are_pinned(tmp_path, monkeypatch):
    monkeypatch.setattr(fct.config, "LOSS_VERSION", FIXTURE_LOSS_VERSION)
    monkeypatch.setattr(fct.config, "MODEL_VERSION", FIXTURE_MODEL_VERSION)
    exp = _exp(tmp_path)
    hashes = [setting_config(exp, s).section_hash("data", "net", "train") for s in expand(exp)]
    assert hashes == [row["train_hash"] for row in _rows()]


@pytest.mark.parametrize("name", ["full", "modularity"])
def test_the_modularity_presets_use_the_original_directories(name):
    got = [str(setting_dir(ROOT, s).relative_to(ROOT)) for s in expand(preset(name))]
    assert got == [row["dir"] for row in _rows()]
