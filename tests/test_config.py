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


def test_resolve_device_auto_follows_cuda():
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device(Config(device="auto")).type == expected


def test_resolve_dtype():
    assert resolve_dtype("float32") is torch.float32
    assert resolve_dtype("float64") is torch.float64
    with pytest.raises(ValueError):
        resolve_dtype("float16")
