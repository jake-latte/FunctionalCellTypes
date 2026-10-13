"""Impossible hyperparameters must fail at the config boundary, naming the field."""

import pytest
from pydantic import ValidationError

from fct.config import (
    AnalysisConfig,
    DataConfig,
    MeanFieldConfig,
    NetConfig,
    TrainConfig,
    default_config,
)

# Section hashes key cached artefacts, so the dumped JSON of the defaults must not change.
DEFAULT_HASHES = {
    ("data",): "9b4dc993c839",
    ("data", "net", "train"): "57bd5055a2d2",
    ("data", "net", "train", "meanfield"): "466eab81d8e4",
    ("data", "net", "train", "analysis"): "f6d07e782c0c",
}

_POSITIVE_INT = {
    DataConfig: ["P", "D_in", "d"],
    NetConfig: ["N", "L"],
    TrainConfig: ["log_int", "save_int"],
    MeanFieldConfig: [
        "n_points", "n_outer", "n_inner_max", "n_chains", "n_trans", "n_trans_outer",
        "n_leap", "warmup_trans", "n_init_samples", "log_int", "save_int", "max_backtrack",
    ],
    AnalysisConfig: ["n_grid", "n_components", "n_cols"],
}  # fmt: skip
_POSITIVE_FLOAT = {
    DataConfig: ["var"],
    NetConfig: ["g"],
    TrainConfig: ["lr"],
    MeanFieldConfig: [
        "beta", "eps0", "target_accept", "adapt_rate", "eta_phi", "eta_phihat", "eta_max",
        "eta_grow", "inner_tol", "inner_tol_mc", "max_rel_step", "pd_floor_rel",
        "inv_floor_rel", "jitter_rel", "learning_prefactor", "prior_scale", "alpha",
        "max_h2_growth", "max_mc_err",
    ],
}  # fmt: skip
_NONNEGATIVE_INT = {TrainConfig: ["n_epochs"], AnalysisConfig: ["plot_layer"]}
_NONNEGATIVE_FLOAT = {
    TrainConfig: ["weight_lambda", "rate_lambda"],
    MeanFieldConfig: ["max_div_frac", "eps_collapse", "eta_min_rel", "tilt_floor", "rate_tilt"],
}


def _cases(table, values):
    return [(cls, f, v) for cls, fields in table.items() for f in fields for v in values]


BAD = (
    _cases(_POSITIVE_INT, [0, -1])
    + _cases(_POSITIVE_FLOAT, [0.0, -1.0])
    + _cases(_NONNEGATIVE_INT, [-1])
    + _cases(_NONNEGATIVE_FLOAT, [-0.1])
    + [
        (MeanFieldConfig, "target_accept", 1.0),
        (MeanFieldConfig, "target_accept", 1.5),
        (MeanFieldConfig, "tilt_floor", 1.0),
    ]
)


@pytest.mark.parametrize(("cls", "field", "value"), BAD)
def test_offending_value_raises_naming_the_field(cls, field, value):
    with pytest.raises(ValidationError) as exc:
        cls(**{field: value})
    err = exc.value.errors()[0]
    assert err["loc"] == (field,)
    assert err["type"] in {"greater_than", "greater_than_equal", "less_than"}


@pytest.mark.parametrize(
    ("cls", "field", "value"),
    [
        (TrainConfig, "n_epochs", 0),
        (TrainConfig, "weight_lambda", 0.0),
        (AnalysisConfig, "plot_layer", 0),
        (MeanFieldConfig, "tilt_floor", 0.0),
        (MeanFieldConfig, "rate_tilt", 0.0),
        (MeanFieldConfig, "rate_tilt", None),
        (MeanFieldConfig, "alpha", None),
        (MeanFieldConfig, "prior_scale", 0.5),
    ],
)
def test_boundary_values_are_accepted(cls, field, value):
    assert getattr(cls(**{field: value}), field) == value


def test_fourier_embedding_needs_even_D_in():
    with pytest.raises(ValidationError, match="D_in must be even") as exc:
        DataConfig(D_in=63, embedding="fourier")
    assert exc.value.errors()[0]["type"] == "value_error"


def test_linear_embedding_allows_odd_D_in():
    assert DataConfig(D_in=63, embedding="linear").D_in == 63


@pytest.mark.parametrize(("sections", "expected"), DEFAULT_HASHES.items())
def test_default_section_hashes_unchanged(sections, expected):
    assert default_config().section_hash(*sections) == expected
