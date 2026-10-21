import json

import pytest
import torch
from pydantic import ValidationError

from fct.config import (
    Config,
    default_config,
    load_config,
    resolve_device,
    resolve_dtype,
    save_config,
)


def _saved_json(tmp_path):
    path = tmp_path / "config.json"
    save_config(default_config(), path)
    return path, json.loads(path.read_text())


def test_roundtrip_unchanged(tmp_path):
    path = tmp_path / "config.json"
    cfg = default_config()
    save_config(cfg, path)
    assert load_config(path) == cfg


def test_save_is_indented_with_sorted_keys(tmp_path):
    path, raw = _saved_json(tmp_path)
    assert "\n  " in path.read_text()
    assert list(raw) == sorted(raw)


def test_unknown_key_raises(tmp_path):
    path, raw = _saved_json(tmp_path)
    raw["net"]["not_a_field"] = 1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValidationError):
        load_config(path)


def test_wrong_type_raises(tmp_path):
    path, raw = _saved_json(tmp_path)
    raw["net"]["N"] = "many"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValidationError):
        load_config(path)


@pytest.mark.parametrize(
    "raw", [{"net": {"N": "5"}}, {"meanfield": {"project_tilt": "yes"}}, {"data": {"var": "1.0"}}]
)
def test_lax_coercion_is_rejected(tmp_path, raw):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValidationError):
        load_config(path)


def test_int_accepted_for_float_field(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"data": {"var": 3}}))
    assert load_config(path).data.var == 3.0


def test_hash_changes_with_dtype():
    a, b = default_config(), default_config()
    b.dtype = "float32"
    assert a.section_hash("data") != b.section_hash("data")


def test_hash_requires_a_section():
    with pytest.raises(ValueError):
        default_config().section_hash()


def test_hash_stable_across_json_key_order(tmp_path):
    path, raw = _saved_json(tmp_path)
    reversed_raw = {k: raw[k] for k in reversed(list(raw))}
    reversed_raw["net"] = {k: raw["net"][k] for k in reversed(list(raw["net"]))}
    path.write_text(json.dumps(reversed_raw))
    assert load_config(path).section_hash("data", "net") == default_config().section_hash(
        "data", "net"
    )


def test_hash_changes_with_own_section():
    a, b = default_config(), default_config()
    b.data.P = 7
    assert a.section_hash("data") != b.section_hash("data")


def test_hash_ignores_unrelated_section():
    a, b = default_config(), default_config()
    b.analysis.n_grid = 99
    assert a.section_hash("data") == b.section_hash("data")
    assert a.section_hash("data", "analysis") != b.section_hash("data", "analysis")


def test_hash_independent_of_section_order():
    cfg = default_config()
    assert cfg.section_hash("data", "net") == cfg.section_hash("net", "data")


def test_hash_is_12_hex_chars():
    h = default_config().section_hash("data")
    assert len(h) == 12
    int(h, 16)


def test_hash_unknown_section_raises():
    with pytest.raises(ValueError):
        default_config().section_hash("nope")


def test_resolve_device_cpu():
    assert resolve_device(Config(device="cpu")) == torch.device("cpu")


@pytest.mark.parametrize("available", [True, False])
def test_resolve_device_auto_follows_cuda(monkeypatch, available):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    assert resolve_device(Config(device="auto")).type == ("cuda" if available else "cpu")


def test_resolve_device_cuda_without_cuda_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device(Config(device="cuda"))


def test_resolve_dtype():
    assert resolve_dtype("float32") is torch.float32
    assert resolve_dtype("float64") is torch.float64
    with pytest.raises(ValueError):
        resolve_dtype("float16")


def test_loss_definition_enters_the_train_hash_only(monkeypatch):
    import fct.config

    cfg = default_config()
    sections = ("data", "net", "train", "meanfield")
    before = {s: cfg.section_hash(s) for s in sections}
    combined = cfg.section_hash("data", "net", "train", "meanfield")  # the meanfield cache key
    monkeypatch.setattr(fct.config, "LOSS_VERSION", fct.config.LOSS_VERSION + 1)
    after = {s: cfg.section_hash(s) for s in sections}
    assert after["train"] != before["train"]
    assert all(after[s] == before[s] for s in ("data", "net", "meanfield"))
    assert cfg.section_hash("data", "net", "train", "meanfield") != combined
