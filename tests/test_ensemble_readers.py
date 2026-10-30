"""Readers of trained runs on an ensemble (net.B > 1): a sweep trains it end to end with the
MIR monitor, readers that take a member index select one, and the others refuse loudly
instead of reading member 0 (CPU only, on the tiny spatial preset)."""

import pytest
import torch

from fct.cli import CliError, _load_net
from fct.config import MeanFieldConfig, NetConfig, resolve_dtype
from fct.meanfield import solve
from fct.model import EnsembleMLP, make_net
from fct.monitor import MIR_MONITOR_KEYS
from fct.report import load_setting_result
from fct.spatial_grid import load_grid_run, select_setting
from fct.spatial_view import train_in
from fct.sweep import (
    SweepError,
    expand,
    load_sweep,
    preset,
    run_setting,
    save_sweep,
    setting_dir,
    with_grid,
)
from fct.train import load_trained

B = 3


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    """spatial_tiny with one setting, an ensemble of B = 3, trained."""
    root = tmp_path_factory.mktemp("ensemble_grid")
    sweep = with_grid(preset("spatial_tiny"), {"net.B": [B]})
    save_sweep(sweep, root)
    run_setting(sweep, 0, root, log=lambda _: None)
    return root


def _directory_and_cfg(root):
    chosen = select_setting(root, {"net.B": B})
    return chosen.directory, chosen.cfg


def test_a_sweep_trains_an_ensemble_with_a_member_axis_in_the_monitor(root):
    directory, cfg = _directory_and_cfg(root)
    net, history = load_trained(directory, cfg, torch.device("cpu"), resolve_dtype(cfg.dtype))
    assert isinstance(net, EnsembleMLP) and net.B == B
    for key in MIR_MONITOR_KEYS:
        calls = history[f"monitor_{key}"]
        assert all(len(call) == B and len(call[0]) == cfg.net.L for call in calls), key
    assert all(len(entry) == B for entry in history["loss"])


def test_load_grid_run_takes_a_member_and_refuses_without_one(root):
    chosen = select_setting(root, {"net.B": B})
    with pytest.raises(ValueError, match="ensemble of B=3"):
        load_grid_run(chosen)
    run, _ = load_grid_run(chosen, member=2)
    cfg = chosen.cfg
    net, history = load_trained(
        chosen.directory, cfg, torch.device("cpu"), resolve_dtype(cfg.dtype)
    )
    assert torch.equal(run.net.U, net.U[2])
    assert run.history["loss"] == [entry[2] for entry in history["loss"]]
    assert run.history["monitor_mir"] == [call[2] for call in history["monitor_mir"]]


def test_the_report_refuses_an_ensemble(root):
    sweep = load_sweep(root)
    (setting,) = expand(sweep)
    with pytest.raises(ValueError, match="ensemble of B=3"):
        load_setting_result(root, sweep, setting)


def test_the_cli_refuses_an_ensemble(root):
    directory, cfg = _directory_and_cfg(root)
    with pytest.raises(CliError, match="does not support ensembles"):
        _load_net(directory, cfg)


def test_the_sweep_solve_stage_refuses_a_trained_ensemble(root):
    sweep = load_sweep(root)
    with pytest.raises(SweepError, match="ensemble"):
        run_setting(sweep, 0, root, log=lambda _: None, stage="solve")
    assert setting_dir(root, expand(sweep)[0]).is_dir()


def test_the_sweep_solve_stage_refuses_an_ensemble_before_it_is_trained(tmp_path):
    """A solve job that runs before training finishes must not solve (its result would read
    as stale once the ensemble is trained): it refuses at once, writing nothing."""
    sweep = with_grid(preset("tiny"), {"net.B": [B]})
    save_sweep(sweep, tmp_path)
    with pytest.raises(SweepError, match="ensemble"):
        run_setting(sweep, 0, tmp_path, log=lambda _: None, stage="solve")
    assert not setting_dir(tmp_path, expand(sweep)[0]).exists()


def test_the_solver_refuses_an_ensemble():
    net = make_net(NetConfig(N=4, L=1, phi="relu", B=2), 3, 2, "cpu", torch.float64)
    X = torch.randn(4, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    Y = torch.randn(4, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    with pytest.raises(TypeError, match="ensemble of B=2"):
        solve(X, Y, 1, 1e-3, 1e-3, torch.relu, 1.0, MeanFieldConfig(), net=net, log=print)


def test_the_spatial_view_refuses_to_train_an_ensemble(root, tmp_path):
    _, cfg = _directory_and_cfg(root)
    with pytest.raises(ValueError, match="net.B=3"):
        train_in(cfg, tmp_path)
    assert not any(tmp_path.iterdir())  # refused before writing anything
