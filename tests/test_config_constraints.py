"""Impossible hyperparameters must fail at the config boundary, naming the field."""

import hashlib
import json

import pytest
from pydantic import ValidationError

import fct.config
from fct.config import (
    LOSS_VERSION,
    MODEL_VERSION,
    AnalysisConfig,
    DataConfig,
    MeanFieldConfig,
    NetConfig,
    SolverConfig,
    TrainConfig,
    default_config,
)

# Section hashes key cached artefacts, so the dumped JSON of the defaults must not change
# unintentionally (adding a config field changes every hash that includes its section, so old
# caches become stale). The data pin is a literal; the sections that include train are rebuilt
# from canonical JSON, which also holds the loss version. DATA_HASH moved once deliberately,
# when `latent_range` was renamed to `latent_min` / `latent_max`.
DATA_HASH = "292f74b412b0"
TRAIN_SECTIONS = [
    ("data", "net", "train"),
    ("data", "net", "train", "meanfield"),
    ("data", "net", "train", "analysis"),
]

_POSITIVE_INT = {
    DataConfig: ["P", "D_in", "d", "n_latents", "grid_size", "n_motifs"],
    NetConfig: ["N", "L"],
    TrainConfig: ["log_int", "save_int", "monitor_shuffles"],
    MeanFieldConfig: ["n_points"],
    SolverConfig: [
        "n_outer", "n_inner_max", "n_chains", "n_trans", "n_trans_outer", "n_leap",
        "warmup_trans", "n_init_samples", "log_int", "save_int", "max_backtrack",
        "n_importance_samples",
    ],
    AnalysisConfig: ["n_grid", "n_components", "n_cols"],
}  # fmt: skip
_POSITIVE_FLOAT = {
    DataConfig: ["var"],
    NetConfig: ["g"],
    TrainConfig: ["lr"],
    MeanFieldConfig: ["beta", "learning_prefactor", "prior_scale", "alpha"],
    SolverConfig: [
        "eps0", "target_accept", "adapt_rate", "eta_phi", "eta_phihat", "eta_max", "eta_grow",
        "inner_tol", "inner_tol_mc", "max_rel_step", "pd_floor_rel", "inv_floor_rel",
        "jitter_rel", "max_h2_growth", "max_mc_err_growth", "importance_proposal_scale",
        "min_ess_drop",
    ],
}  # fmt: skip
_NONNEGATIVE_INT = {
    DataConfig: ["max_objects"],
    TrainConfig: ["n_epochs", "monitor_int"],
    AnalysisConfig: ["plot_layer"],
}
_NONNEGATIVE_FLOAT = {
    TrainConfig: ["weight_lambda", "rate_lambda"],
    MeanFieldConfig: ["rate_tilt"],
    SolverConfig: ["max_div_frac", "eps_collapse", "eta_min_rel", "tilt_floor", "min_ess_frac"],
}


def _cases(table, values):
    return [(cls, f, v) for cls, fields in table.items() for f in fields for v in values]


BAD = (
    _cases(_POSITIVE_INT, [0, -1])
    + _cases(_POSITIVE_FLOAT, [0.0, -1.0])
    + _cases(_NONNEGATIVE_INT, [-1])
    + _cases(_NONNEGATIVE_FLOAT, [-0.1])
    + [
        (TrainConfig, "monitor_bins", 1),
        (TrainConfig, "monitor_bins", 0),
        (SolverConfig, "target_accept", 1.0),
        (SolverConfig, "target_accept", 1.5),
        (SolverConfig, "tilt_floor", 1.0),
        (SolverConfig, "min_ess_frac", 1.5),
        (SolverConfig, "min_ess_drop", 1.5),
    ]
)


@pytest.mark.parametrize(("cls", "field", "value"), BAD)
def test_offending_value_raises_naming_the_field(cls, field, value):
    with pytest.raises(ValidationError) as exc:
        cls(**{field: value})
    err = exc.value.errors()[0]
    assert err["loc"] == (field,)
    assert err["type"] in {"greater_than", "greater_than_equal", "less_than", "less_than_equal"}


@pytest.mark.parametrize(
    ("cls", "field", "value"),
    [
        (TrainConfig, "n_epochs", 0),
        (TrainConfig, "weight_lambda", 0.0),
        (TrainConfig, "monitor_bins", 2),
        (AnalysisConfig, "plot_layer", 0),
        (SolverConfig, "tilt_floor", 0.0),
        (SolverConfig, "min_ess_frac", 0.0),
        (SolverConfig, "min_ess_frac", 1.0),
        (SolverConfig, "min_ess_drop", 1.0),
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


def test_default_data_hash_unchanged():
    assert default_config().section_hash("data") == DATA_HASH


@pytest.mark.parametrize("sections", TRAIN_SECTIONS)
def test_train_section_hashes_hold_the_loss_version(sections):
    cfg = default_config()
    payload = {
        **cfg.model_dump(include={*sections, "dtype"}),
        "loss_version": LOSS_VERSION,
        "model_version": MODEL_VERSION,
    }
    del payload["train"]["optimizer"], payload["train"]["beta"]  # adam: SGLD fields not hashed
    del payload["net"]["B"]  # a single network (B = 1): the ensemble size is not hashed
    del payload["data"]["teacher"]  # Gaussian teacher maps: the map type is not hashed
    if "meanfield" in sections:  # hashed in its flat pre-split layout, see hash_payload
        payload["meanfield"] = cfg.meanfield.hash_payload()
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    assert cfg.section_hash(*sections) == expected


@pytest.mark.parametrize("sections", TRAIN_SECTIONS)
def test_train_section_hashes_change_with_the_loss_version(sections, monkeypatch):
    cfg = default_config()
    before = cfg.section_hash(*sections)
    monkeypatch.setattr(fct.config, "LOSS_VERSION", LOSS_VERSION + 1)
    assert cfg.section_hash(*sections) != before
