import pytest
import torch

from fct.cache import (
    StaleCacheError,
    cache_paths,
    latest_checkpoint,
    load_artefact,
    load_or_compute,
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
