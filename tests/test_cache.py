from pathlib import Path

import numpy as np
import pytest
import torch

from fct.cache import (
    StaleCacheError,
    cache_paths,
    latest_checkpoint,
    load_artefact,
    load_or_compute,
    read_config_hash,
    save_artefact,
    write_checkpoint,
)

CPU = torch.device("cpu")


def test_cache_paths_layout(tmp_path):
    p = cache_paths(tmp_path)
    assert p.config == tmp_path / "config.json"
    assert p.data == tmp_path / "data"
    assert p.train == tmp_path / "train"
    assert p.meanfield == tmp_path / "meanfield"
    assert p.analysis == tmp_path / "analysis"
    assert p.figures == tmp_path / "figures"


def test_artefact_roundtrip(tmp_path):
    path = tmp_path / "a" / "b" / "x.pt"
    payload = {"w": torch.randn(3, 2), "n": 5, "name": "hi", "hist": {"mse": [1.0, 0.5]}}
    save_artefact(path, payload, "abc")
    out = load_artefact(path, "abc", CPU)
    assert torch.equal(out["w"], payload["w"])
    assert out["n"] == 5 and out["name"] == "hi" and out["hist"] == {"mse": [1.0, 0.5]}


def test_save_does_not_mutate_payload(tmp_path):
    payload = {"n": 1}
    save_artefact(tmp_path / "x.pt", payload, "abc")
    assert payload == {"n": 1}


def test_stale_hash_raises_with_path_and_hashes(tmp_path):
    path = tmp_path / "x.pt"
    save_artefact(path, {"n": 1}, "old")
    with pytest.raises(StaleCacheError) as err:
        load_artefact(path, "new", CPU)
    msg = str(err.value)
    assert str(path) in msg and "old" in msg and "new" in msg


def test_load_or_compute_computes_once(tmp_path):
    path = tmp_path / "x.pt"
    calls = []

    def compute():
        calls.append(1)
        return {"v": torch.arange(3)}

    first = load_or_compute(path, "h", compute, CPU)
    second = load_or_compute(path, "h", compute, CPU)
    assert len(calls) == 1
    assert torch.equal(first["v"], second["v"])


def test_load_or_compute_stale_raises_instead_of_recomputing(tmp_path):
    path = tmp_path / "x.pt"
    load_or_compute(path, "h1", lambda: {"v": 1}, CPU)
    with pytest.raises(StaleCacheError):
        load_or_compute(path, "h2", lambda: {"v": 2}, CPU)


def test_latest_checkpoint_none_when_empty(tmp_path):
    assert latest_checkpoint(tmp_path) is None
    assert latest_checkpoint(tmp_path / "missing") is None


def test_write_checkpoint_keeps_only_newest(tmp_path):
    write_checkpoint(tmp_path, "ckpt_", 10, {"s": 10}, "h")
    newest = write_checkpoint(tmp_path, "ckpt_", 20, {"s": 20}, "h")
    assert sorted(tmp_path.glob("ckpt_*.pt")) == [newest]
    assert latest_checkpoint(tmp_path) == newest
    assert load_artefact(newest, "h", CPU)["s"] == 20


def test_write_checkpoint_leaves_other_files(tmp_path):
    save_artefact(tmp_path / "final.pt", {"s": 0}, "h")
    write_checkpoint(tmp_path, "ckpt_", 1, {"s": 1}, "h")
    write_checkpoint(tmp_path, "ckpt_", 2, {"s": 2}, "h")
    assert (tmp_path / "final.pt").exists()
    assert len(list(tmp_path.glob("ckpt_*.pt"))) == 1


def test_latest_checkpoint_picks_highest_step(tmp_path):
    for step in (5, 100, 20):
        save_artefact(tmp_path / f"ckpt_{step:06d}.pt", {"s": step}, "h")
    assert latest_checkpoint(tmp_path).name == "ckpt_000100.pt"


def test_non_loadable_payload_raises_naming_key_and_writes_nothing(tmp_path):
    path = tmp_path / "x.pt"
    with pytest.raises(TypeError, match=r"outer\.inner\[1\]"):
        save_artefact(path, {"outer": {"inner": [1, np.zeros(2)]}}, "h")
    assert not path.exists() and not list(tmp_path.iterdir())


@pytest.mark.parametrize("bad", [np.float64(1.0), Path("a"), {1: 2}, {"s": {1, 2}}])
def test_other_non_loadable_payloads_raise(tmp_path, bad):
    with pytest.raises(TypeError):
        save_artefact(tmp_path / "x.pt", {"k": bad}, "h")


def test_payload_with_config_hash_key_raises(tmp_path):
    with pytest.raises(ValueError):
        save_artefact(tmp_path / "x.pt", {"config_hash": "mine"}, "h")


def test_empty_expected_hash_raises(tmp_path):
    save_artefact(tmp_path / "x.pt", {"n": 1}, "h")
    for bad in ("", None):
        with pytest.raises(ValueError):
            load_artefact(tmp_path / "x.pt", bad, CPU)


def test_load_non_dict_raises(tmp_path):
    path = tmp_path / "x.pt"
    torch.save(torch.zeros(2), path)
    with pytest.raises(TypeError):
        load_artefact(path, "h", CPU)


def test_load_or_compute_unloadable_compute_leaves_no_file(tmp_path):
    path = tmp_path / "x.pt"
    with pytest.raises(TypeError):
        load_or_compute(path, "h", lambda: {"a": np.zeros(1)}, CPU)
    assert not path.exists()


def test_load_or_compute_unlinks_file_if_reload_fails(tmp_path, monkeypatch):
    path = tmp_path / "x.pt"
    real_load = torch.load
    calls = []

    def flaky_load(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("cannot load")

    monkeypatch.setattr(torch, "load", flaky_load)
    with pytest.raises(RuntimeError, match="cannot load"):
        load_or_compute(path, "h", lambda: {"v": 1}, CPU)
    monkeypatch.setattr(torch, "load", real_load)
    assert not path.exists()


def test_write_checkpoint_keeps_higher_step(tmp_path):
    write_checkpoint(tmp_path, "ckpt_", 20, {"s": 20}, "h")
    write_checkpoint(tmp_path, "ckpt_", 10, {"s": 10}, "h")
    assert sorted(p.name for p in tmp_path.glob("ckpt_*.pt")) == [
        "ckpt_000010.pt",
        "ckpt_000020.pt",
    ]


def test_write_checkpoint_width(tmp_path):
    new = write_checkpoint(tmp_path, "ckpt_", 7, {"s": 7}, "h", width=5)
    assert new.name == "ckpt_00007.pt"


def test_write_checkpoint_writes_before_deleting(tmp_path, monkeypatch):
    write_checkpoint(tmp_path, "ckpt_", 1, {"s": 1}, "h")

    def boom(self, *args, **kwargs):
        raise OSError("unlink failed")

    monkeypatch.setattr(Path, "unlink", boom)
    with pytest.raises(OSError):
        write_checkpoint(tmp_path, "ckpt_", 2, {"s": 2}, "h")
    assert (tmp_path / "ckpt_000002.pt").exists()


def test_read_config_hash_returns_the_stored_hash_without_checking_it(tmp_path):
    save_artefact(tmp_path / "a.pt", {"w": torch.zeros(3)}, "abc123")
    assert read_config_hash(tmp_path / "a.pt") == "abc123"
