"""End-to-end tests of `python -m fct` on a tiny CPU config, calling `main` directly."""

import io
import json
import sys

import pytest
import torch

from fct.cli import main
from fct.monitor import MIR_MONITOR_KEYS
from fct.train import MONITOR_EPOCH, MONITOR_PREFIX

TINY = [
    "data.P=64", "data.D_in=8", "net.N=8", "net.L=2",
    "train.n_epochs=6", "train.save_int=3", "train.log_int=3",
    "meanfield.n_points=8", "meanfield.n_chains=64", "meanfield.n_outer=2",
    "meanfield.n_inner_max=3", "meanfield.warmup_trans=10", "meanfield.n_init_samples=256",
    "meanfield.save_int=1", "analysis.n_grid=5", "analysis.n_components=3", "device=cpu",
]  # fmt: skip

FIGURES = ["learning", "representation", "neurons", "training", "solver", "kernels", "example"]


def _init(cache, *extra):
    args = ["init", str(cache)]
    for kv in [*TINY, *extra]:
        args += ["--set", kv]
    return main(args)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Run the whole workflow once; the tests below inspect the result."""
    cache = tmp_path_factory.mktemp("cache")
    codes = {"init": _init(cache)}
    for step in ("train", "solve", "plot", "plot-example"):
        codes[step] = main([step, str(cache)])
    return cache, codes


def test_full_workflow_creates_all_artefacts(pipeline):
    cache, codes = pipeline
    assert codes == {"init": 0, "train": 0, "solve": 0, "plot": 0, "plot-example": 0}
    for rel in ["data/dataset.pt", "train/final.pt", "meanfield/final.pt"]:
        assert (cache / rel).exists(), rel
    for name in FIGURES:
        assert (cache / "figures" / f"{name}.png").stat().st_size > 0, name


def test_train_twice_leaves_final_untouched(pipeline):
    cache, _ = pipeline
    final = cache / "train" / "final.pt"
    before, mtime = final.read_bytes(), final.stat().st_mtime_ns
    assert main(["train", str(cache)]) == 0
    assert final.read_bytes() == before and final.stat().st_mtime_ns == mtime


def test_init_refuses_overwrite_unless_forced(tmp_path):
    assert _init(tmp_path) == 0
    assert _init(tmp_path) == 1
    assert _init(tmp_path, "net.L=3") == 1
    assert main(["init", str(tmp_path), "--force", "--set", "net.L=3"]) == 0
    assert json.loads((tmp_path / "config.json").read_text())["net"]["L"] == 3


def test_init_applies_set_with_json_and_string_values(tmp_path):
    args = ["init", str(tmp_path), "--set", "net.L=2", "--set", "net.phi=tanh"]
    args += ["--set", "data.cov=[[1,0],[0,1]]", "--set", "data.d=2"]
    assert main(args) == 0
    written = json.loads((tmp_path / "config.json").read_text())
    assert written["net"]["L"] == 2 and written["net"]["phi"] == "tanh"
    assert written["data"]["cov"] == [[1, 0], [0, 1]]


@pytest.mark.parametrize("key", ["net.bogus=1", "bogus.L=1", "net=3", "net.L.x=1"])
def test_init_rejects_bad_keys(tmp_path, capsys, key):
    assert main(["init", str(tmp_path), "--set", key]) == 1
    assert capsys.readouterr().err.strip()
    assert not (tmp_path / "config.json").exists()


def test_init_unknown_key_shows_pydantic_message(tmp_path, capsys):
    assert main(["init", str(tmp_path), "--set", "net.bogus=1"]) == 1
    assert "bogus" in capsys.readouterr().err


def test_set_without_equals_is_an_error(tmp_path, capsys):
    assert main(["init", str(tmp_path), "--set", "net.L"]) == 1
    assert "net.L" in capsys.readouterr().err


def test_missing_config_is_one_line_error(tmp_path, capsys):
    assert main(["train", str(tmp_path)]) == 1
    err = capsys.readouterr().err.strip()
    assert "config.json" in err and "\n" not in err


def test_plot_before_training_names_missing_file(tmp_path, capsys):
    _init(tmp_path)
    assert main(["plot", str(tmp_path)]) == 1
    assert "train/final.pt" in capsys.readouterr().err


def test_plot_solver_before_solving_names_missing_file(tmp_path, capsys):
    _init(tmp_path)
    assert main(["plot", str(tmp_path), "--which", "solver"]) == 1
    assert "meanfield/final.pt" in capsys.readouterr().err


def test_solve_before_training_fails_cleanly(tmp_path, capsys):
    _init(tmp_path)
    assert main(["solve", str(tmp_path)]) == 1
    assert "final.pt" in capsys.readouterr().err


def test_plot_all_skips_missing_meanfield(tmp_path, capsys):
    _init(tmp_path)
    assert main(["train", str(tmp_path)]) == 0
    capsys.readouterr()
    assert main(["plot", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "skipped" in out and "meanfield/final.pt" in out
    assert (tmp_path / "figures" / "learning.png").exists()
    assert not (tmp_path / "figures" / "solver.png").exists()


def test_changed_config_gives_one_line_stale_error(tmp_path, capsys):
    _init(tmp_path)
    assert main(["train", str(tmp_path)]) == 0
    path = tmp_path / "config.json"
    cfg = json.loads(path.read_text())
    cfg["train"]["lr"] = 0.5
    path.write_text(json.dumps(cfg))
    capsys.readouterr()
    assert main(["train", str(tmp_path)]) == 1
    assert capsys.readouterr().err.count("\n") == 1


def test_help_lists_subcommands(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for cmd in ("init", "train", "solve", "plot", "plot-example"):
        assert cmd in out


def test_main_line_buffers_stdout(monkeypatch):
    # Block-buffered stdout hides progress from `> log` and SLURM logs until the process ends.
    stream = io.TextIOWrapper(io.BytesIO())
    monkeypatch.setattr(sys, "stdout", stream)
    with pytest.raises(SystemExit):
        main(["--help"])
    assert stream.line_buffering


@pytest.mark.parametrize(
    "text, fragment",
    [("{not json", "config.json"), ('{"net": {"bogus": 1}}', "bogus")],
)
def test_bad_config_file_is_a_cli_error(tmp_path, capsys, text, fragment):
    (tmp_path / "config.json").write_text(text)
    assert main(["train", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and fragment in err
    assert "Traceback" not in err


def test_solve_with_too_many_points_is_a_cli_error(tmp_path, capsys):
    _init(tmp_path, "meanfield.n_points=100")
    assert main(["train", str(tmp_path)]) == 0
    capsys.readouterr()
    assert main(["solve", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "n_points" in err and err.count("\n") == 1


def test_train_on_finished_cache_says_it_reused_the_model(tmp_path, capsys):
    _init(tmp_path)
    assert main(["train", str(tmp_path)]) == 0
    capsys.readouterr()
    assert main(["train", str(tmp_path)]) == 0
    assert "train/final.pt" in capsys.readouterr().out


def test_force_init_warns_about_stale_artefacts(tmp_path, capsys):
    _init(tmp_path)
    assert main(["train", str(tmp_path)]) == 0
    capsys.readouterr()
    assert main(["init", str(tmp_path), "--force"]) == 0
    assert "stale" in capsys.readouterr().out


def _trained_history(cache):
    return torch.load(cache / "train" / "final.pt", weights_only=True)["history"]


def test_train_records_the_monitor_only_when_asked(tmp_path):
    with_monitor, without = tmp_path / "with", tmp_path / "without"
    assert _init(with_monitor, "train.monitor_int=3", "train.monitor_bins=4") == 0
    assert _init(without) == 0
    for cache in (with_monitor, without):
        assert main(["train", str(cache)]) == 0
    hist = _trained_history(with_monitor)
    assert hist[MONITOR_EPOCH] == [0, 3, 6]  # n_epochs = 6: the last call is after the final step
    for key in MIR_MONITOR_KEYS:
        assert len(hist[f"{MONITOR_PREFIX}{key}"]) == 3
        assert all(len(layers) == 2 for layers in hist[f"{MONITOR_PREFIX}{key}"])  # net.L = 2
    assert not [k for k in _trained_history(without) if k.startswith(MONITOR_PREFIX)]


def _no_factors(*args):
    raise ValueError("the MIR monitor needs dataset.factors")


def test_finished_run_loads_without_building_the_monitor(tmp_path, monkeypatch):
    assert _init(tmp_path, "train.monitor_int=3") == 0
    assert main(["train", str(tmp_path)]) == 0
    monkeypatch.setattr("fct.cli.make_mir_monitor", _no_factors)
    assert main(["train", str(tmp_path)]) == 0


def test_fresh_run_without_factors_fails_loudly(tmp_path, monkeypatch):
    assert _init(tmp_path, "train.monitor_int=3") == 0
    monkeypatch.setattr("fct.cli.make_mir_monitor", _no_factors)
    with pytest.raises(ValueError, match="factors"):
        main(["train", str(tmp_path)])
