import json

import pytest
import torch
from pydantic import ValidationError

from fct.config import (
    HEALTH_BOUND_DEFAULTS,
    HEALTH_BOUND_FIELDS,
    IMPORTANCE_FIELDS,
    LEGACY_SOLVER_FIELDS,
    RETIRED_SOLVER_FIELDS,
    STEP_CONTROL_FIELDS,
    STEP_CONTROL_LEGACY,
    UNHASHED_SOLVER_FIELDS,
    Config,
    MeanFieldConfig,
    SolverConfig,
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


def test_a_config_with_a_readout_field_fails_to_load_naming_it(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"meanfield": {"readout": "h"}}))
    with pytest.raises(ValidationError, match="readout"):
        load_config(path)


def test_model_definition_enters_every_hash_with_the_net_section_only(monkeypatch):
    import fct.config

    cfg = default_config()
    sections = ("data", "net", "train", "meanfield")
    before = {s: cfg.section_hash(s) for s in sections}
    combined = cfg.section_hash("data", "net", "train")  # the training cache key
    monkeypatch.setattr(fct.config, "MODEL_VERSION", fct.config.MODEL_VERSION + 1)
    after = {s: cfg.section_hash(s) for s in sections}
    assert after["net"] != before["net"]
    assert all(after[s] == before[s] for s in ("data", "train", "meanfield"))
    assert cfg.section_hash("data", "net", "train") != combined


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


# ----------------------------------------------------------------------------- meanfield split
PROBLEM_FIELDS = {
    "n_points", "beta", "prior_scale", "alpha", "learning_prefactor", "rate_tilt",
}  # fmt: skip
RESULT_SECTIONS = ("data", "net", "train", "meanfield")


def _with_solver(cfg: Config, **solver) -> Config:
    mf = cfg.meanfield.model_copy(update={"solver": cfg.meanfield.solver.model_copy(update=solver)})
    return cfg.model_copy(update={"meanfield": mf})


def test_meanfield_holds_the_problem_and_a_solver_group():
    assert set(MeanFieldConfig.model_fields) == PROBLEM_FIELDS | {"solver"}
    assert not PROBLEM_FIELDS & set(SolverConfig.model_fields)


def test_flat_solver_keys_move_into_the_solver_group():
    mf = MeanFieldConfig(beta=1e3, n_chains=64, sampler="importance")
    assert mf.beta == 1e3
    assert mf.solver.n_chains == 64 and mf.solver.sampler == "importance"
    assert mf == MeanFieldConfig(beta=1e3, solver=SolverConfig(n_chains=64, sampler="importance"))


def test_saved_config_nests_the_solver_fields(tmp_path):
    _, raw = _saved_json(tmp_path)
    assert set(raw["meanfield"]) == PROBLEM_FIELDS | {"solver"}
    assert raw["meanfield"]["solver"]["sampler"] == "hmc"


def test_a_solver_key_given_twice_is_rejected():
    with pytest.raises(ValidationError, match="n_chains"):
        MeanFieldConfig(n_chains=64, solver={"n_chains": 128})


def test_unknown_meanfield_key_still_raises():
    with pytest.raises(ValidationError):
        MeanFieldConfig(not_a_field=1)


def test_every_solver_field_has_a_hash_rule():
    rules = (
        LEGACY_SOLVER_FIELDS, IMPORTANCE_FIELDS, UNHASHED_SOLVER_FIELDS, STEP_CONTROL_FIELDS,
        HEALTH_BOUND_FIELDS,
    )  # fmt: skip
    assert set(SolverConfig.model_fields) == set().union(*rules)
    assert sum(len(r) for r in rules) == len(SolverConfig.model_fields)  # disjoint


def test_the_final_residual_check_does_not_enter_the_hash():
    a, b = default_config(), default_config()
    b.meanfield.solver.tilt_tripwire_resid = 0.5
    assert a.section_hash("meanfield") == b.section_hash("meanfield")


@pytest.mark.parametrize(
    "setting",
    [
        {"inner_step_control": "health"},
        {"eta_max_phihat": 0.5},
        {"phi_step_scale": "trace"},
        {"inner_start": "previous"},
    ],
)
def test_step_control_makes_the_result_stale(setting):
    cfg = default_config()
    other = _with_solver(cfg, **setting)
    assert other.section_hash(*RESULT_SECTIONS) != cfg.section_hash(*RESULT_SECTIONS)


def test_step_control_at_its_plain_values_stays_out_of_the_hash():
    payload = MeanFieldConfig(**STEP_CONTROL_LEGACY).hash_payload()
    assert not set(STEP_CONTROL_FIELDS) & set(payload)
    assert set(STEP_CONTROL_FIELDS) <= set(MeanFieldConfig().hash_payload())


def test_the_h2_prior_bound_enters_the_hash_only_off_its_default():
    """At its default a solve the bound never trips is the solve it was before the bound
    existed, so the default hash stays; another bound is another solve."""
    cfg = default_config()
    assert cfg.meanfield.solver.max_h2_prior_ratio == HEALTH_BOUND_DEFAULTS["max_h2_prior_ratio"]
    assert not set(HEALTH_BOUND_FIELDS) & set(cfg.meanfield.hash_payload())
    other = _with_solver(cfg, max_h2_prior_ratio=10.0)
    assert other.meanfield.hash_payload()["max_h2_prior_ratio"] == 10.0
    assert other.section_hash(*RESULT_SECTIONS) != cfg.section_hash(*RESULT_SECTIONS)


def test_the_mc_error_growth_bound_enters_the_hash_only_off_its_default():
    cfg = default_config()
    assert cfg.meanfield.solver.max_mc_err_growth == HEALTH_BOUND_DEFAULTS["max_mc_err_growth"]
    assert "max_mc_err_growth" not in cfg.meanfield.hash_payload()
    other = _with_solver(cfg, max_mc_err_growth=3.0)
    assert other.meanfield.hash_payload()["max_mc_err_growth"] == 3.0
    assert other.section_hash(*RESULT_SECTIONS) != cfg.section_hash(*RESULT_SECTIONS)


def test_the_retired_mc_error_bound_is_hashed_at_its_last_value():
    """max_mc_err is no longer a setting, but every hash written while it was holds it at 1:
    it stays in the payload so those hashes do not move."""
    assert RETIRED_SOLVER_FIELDS == {"max_mc_err": 1.0}
    assert "max_mc_err" not in SolverConfig.model_fields
    assert MeanFieldConfig().hash_payload()["max_mc_err"] == 1.0


@pytest.mark.parametrize("layout", ["flat", "nested"])
def test_a_config_with_the_retired_bound_at_its_old_default_loads(layout):
    """Every config.json written while the field existed holds it (at 1 unless changed)."""
    raw = {"max_mc_err": 1.0} if layout == "flat" else {"solver": {"max_mc_err": 1.0}}
    assert MeanFieldConfig.model_validate(raw) == MeanFieldConfig()


@pytest.mark.parametrize("layout", ["flat", "nested"])
def test_a_config_that_changed_the_retired_bound_is_rejected(layout):
    """Its value would be silently ignored: say so, and what replaced it."""
    raw = {"max_mc_err": 2.0} if layout == "flat" else {"solver": {"max_mc_err": 2.0}}
    with pytest.raises(ValidationError, match="max_mc_err.*max_mc_err_growth"):
        MeanFieldConfig.model_validate(raw)


def test_the_h2_prior_bound_must_be_positive():
    with pytest.raises(ValidationError, match="max_h2_prior_ratio"):
        MeanFieldConfig(max_h2_prior_ratio=0.0)


def test_hmc_hash_ignores_the_importance_settings():
    cfg = default_config()
    other = _with_solver(cfg, n_importance_samples=17, importance_proposal_scale=2.0)
    assert other.section_hash(*RESULT_SECTIONS) == cfg.section_hash(*RESULT_SECTIONS)


def test_changing_the_sampler_makes_the_result_stale():
    cfg = default_config()
    importance = _with_solver(cfg, sampler="importance")
    assert importance.section_hash(*RESULT_SECTIONS) != cfg.section_hash(*RESULT_SECTIONS)
    wider = _with_solver(importance, importance_proposal_scale=2.0)
    assert wider.section_hash(*RESULT_SECTIONS) != importance.section_hash(*RESULT_SECTIONS)


def test_solver_settings_never_touch_the_training_hash():
    cfg = default_config()
    other = _with_solver(cfg, sampler="importance", n_chains=3, seed=9)
    assert other.section_hash("data", "net", "train") == cfg.section_hash("data", "net", "train")
