"""The ensemble size B as a shared experiment field, beside L, g and phi."""

import pytest
from pydantic import ValidationError

from fct.exp import ExpConfig, expand, preset, same_experiment, setting_config, with_axes


def _tiny(**updates) -> ExpConfig:
    return preset("tiny").model_copy(update=updates)


def test_shared_B_reaches_every_setting():
    exp = _tiny(B=8)
    configs = [setting_config(exp, s) for s in expand(exp)]
    assert {c.net.B for c in configs} == {8}


def test_B_defaults_to_one():
    exp = _tiny()
    assert {setting_config(exp, s).net.B for s in expand(exp)} == {1}


def test_json_without_B_loads_and_equals_the_same_experiment_now():
    now = _tiny()
    old = ExpConfig.model_validate_json(now.model_dump_json(exclude={"B"}))
    assert old.B == 1
    assert same_experiment(old, now)


def test_a_B_axis_wins_over_the_shared_field():
    exp = with_axes(_tiny(B=8), {"net.B": [2, 4]})
    assert [setting_config(exp, s).net.B for s in expand(exp)] == [2, 4]


def test_shared_B_and_a_one_value_axis_give_the_same_config_hash():
    shared = with_axes(_tiny(B=8), {"net.L": [1, 2]})
    axis = with_axes(_tiny(), {"net.B": [8], "net.L": [1, 2]})
    shared_hashes = {setting_config(shared, s).section_hash("data", "net", "train") for s in expand(shared)}
    axis_hashes = {setting_config(axis, s).section_hash("data", "net", "train") for s in expand(axis)}
    assert shared_hashes == axis_hashes


def test_B_zero_is_rejected():
    with pytest.raises(ValidationError):
        _tiny().model_validate({**_tiny().model_dump(), "B": 0})


def test_exp_json_omits_B_at_one_and_writes_it_otherwise():
    from fct.exp import exp_text

    assert '"B"' not in exp_text(_tiny())
    assert '"B": 8' in exp_text(_tiny(B=8))
