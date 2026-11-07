"""`exclude_at_default`: the one rule by which a field is left out of a hash or of exp.json while
it holds the value every artefact written before the field existed implies."""

import pytest

from fct.config import HASH_OMITTED_AT_DEFAULT, Config, NetConfig
from fct.schema import StrictModel, exclude_at_default
from fct.spatial import SpatialConfig


class Inner(StrictModel):
    a: int = 1
    b: str = "x"


class Outer(StrictModel):
    top: int | None = None
    inner: Inner = Inner()
    optional: Inner | None = None


TABLE = {"top": None, "inner.a": 1, "inner.b": "x", "optional.a": 1}


def test_fields_at_the_listed_value_are_excluded_by_nested_path():
    assert exclude_at_default(Outer(), TABLE) == {"top": True, "inner": {"a": True, "b": True}}


def test_a_field_off_the_listed_value_stays():
    model = Outer(top=3, inner=Inner(a=2))
    assert exclude_at_default(model, TABLE) == {"inner": {"b": True}}


def test_a_path_whose_parent_is_none_is_skipped_and_a_set_parent_is_followed():
    assert "optional" not in exclude_at_default(Outer(), TABLE)
    assert exclude_at_default(Outer(optional=Inner()), TABLE)["optional"] == {"a": True}


def test_the_result_is_a_pydantic_exclude():
    model = Outer(top=3, optional=Inner(a=2))
    dumped = model.model_dump(exclude=exclude_at_default(model, TABLE))
    assert dumped == {"top": 3, "inner": {}, "optional": {"a": 2, "b": "x"}}


def test_an_unknown_path_fails_loudly():
    with pytest.raises(AttributeError, match="nope"):
        exclude_at_default(Outer(), {"inner.nope": 1})


def test_the_config_hash_table_leaves_out_exactly_the_default_valued_fields():
    cfg = Config(spatial=SpatialConfig(), net=NetConfig(B=2))
    omitted = exclude_at_default(cfg, HASH_OMITTED_AT_DEFAULT)
    assert omitted == {
        "net": {"init": True},
        "data": {"teacher": True},
        "spatial": {"lambda_max": True, "parametrisation": True},
        "train": {"monitor_kind": True},
    }
    assert "spatial" not in exclude_at_default(Config(), HASH_OMITTED_AT_DEFAULT)
