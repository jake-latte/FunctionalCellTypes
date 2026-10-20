"""Hyperparameter schema, JSON I/O and section hashes; the CLI's `init` writes it to
CACHE_DIR/config.json.

Every model forbids unknown keys and is strict about types, so a typo or a "5" where an int
is expected fails at load time instead of silently becoming a default or a coerced value.
Fields are range-constrained too, so impossible values fail at load time naming the field.
"""

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal, Self

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Constrained types: the config is the boundary, so impossible values fail at load time.
PosInt = Annotated[int, Field(ge=1)]
NonNegInt = Annotated[int, Field(ge=0)]
PosFloat = Annotated[float, Field(gt=0)]
NonNegFloat = Annotated[float, Field(ge=0)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class DataConfig(_Strict):
    # `task` selects the generator; the fields below it are grouped by the task that reads them.
    task: Literal["position", "linear_teacher", "object_action", "what_where"] = "position"
    P: PosInt = 4096
    # position: input dimension of the embedding. linear_teacher: the number of features m of
    # both x and y. object_action / what_where: fixed by the task (2*grid_size**2 / 81) and
    # checked by make_dataset.
    D_in: PosInt = 64
    # position
    d: PosInt = 2
    var: PosFloat = 10.0
    cov: list[list[float]] | None = None  # None -> identity
    embedding: Literal["linear", "fourier"] = "fourier"
    # linear_teacher
    n_latents: PosInt = 3
    latent_range: PosFloat = 1.0
    # object_action
    grid_size: PosInt = 5
    max_objects: NonNegInt = 3
    # what_where
    n_motifs: PosInt = 8
    seed: int = 0

    @model_validator(mode="after")
    def _fourier_needs_even_D_in(self) -> Self:
        # The fourier embedding stacks cos and sin features, so D_in must split into pairs.
        if self.task == "position" and self.embedding == "fourier" and self.D_in % 2:
            raise ValueError(
                f"D_in must be even for the fourier embedding (cos/sin pairs); got {self.D_in}"
            )
        return self

    @model_validator(mode="after")
    def _teacher_has_more_features_than_latents(self) -> Self:
        if self.task == "linear_teacher" and self.D_in <= self.n_latents:
            raise ValueError(
                f"D_in (the number of features m) must exceed n_latents; "
                f"got D_in={self.D_in}, n_latents={self.n_latents}"
            )
        return self


class NetConfig(_Strict):
    N: PosInt = 100
    L: PosInt = 3
    g: PosFloat = 1.0
    phi: Literal["relu", "tanh", "sigmoid"] = "relu"
    seed: int = 0


class TrainConfig(_Strict):
    n_epochs: NonNegInt = 15_000
    lr: PosFloat = 1e-3
    weight_lambda: NonNegFloat = 1e-3
    rate_lambda: NonNegFloat = 1e-3
    log_int: PosInt = 100
    save_int: PosInt = 100
    seed: int = 0


class MeanFieldConfig(_Strict):
    n_points: PosInt = 100  # solve on the first n_points training points
    beta: PosFloat = 1e6
    readout: Literal["h", "phi"] = "h"
    # Algorithm 1
    n_outer: PosInt = 300
    n_inner_max: PosInt = 30
    inner_tol: PosFloat = 1e-2
    inner_tol_mc: PosFloat = 2.0
    eta_phi: PosFloat = 0.05
    eta_phihat: PosFloat = 0.05
    eta_max: PosFloat = 1.0
    eta_grow: PosFloat = 1.1
    max_backtrack: PosInt = 8
    eta_min_rel: NonNegFloat = 1e-3
    max_div_frac: NonNegFloat = 0.02
    max_h2_growth: PosFloat = 10.0
    max_mc_err: PosFloat = 1.0
    eps_collapse: NonNegFloat = 1e-2
    project_tilt: bool = True
    tilt_floor: Annotated[float, Field(ge=0, lt=1)] = 0.05
    max_rel_step: PosFloat = 0.1
    pd_floor_rel: PosFloat = 1e-8
    inv_floor_rel: PosFloat = 1e-4
    prior_scale: PosFloat | None = None
    alpha: PosFloat | None = None
    learning_prefactor: PosFloat = 0.5
    rate_tilt: NonNegFloat | None = None
    # HMC
    n_chains: PosInt = 2048
    n_trans: PosInt = 2
    n_trans_outer: PosInt = 5
    n_leap: PosInt = 10
    eps0: PosFloat = 0.05
    target_accept: Annotated[float, Field(gt=0, lt=1)] = 0.75
    adapt_rate: PosFloat = 0.05
    warmup_trans: PosInt = 200
    sample_dtype: Literal["float32", "float64"] = "float32"
    jitter_rel: PosFloat = 1e-6
    precondition: bool = True
    init: Literal["lazy", "gaussian", "net"] = "lazy"
    n_init_samples: PosInt = 20_000
    log_int: PosInt = 5
    save_int: PosInt = 10
    seed: int = 0


class AnalysisConfig(_Strict):
    n_grid: PosInt = 25  # test grid is n_grid x n_grid
    n_components: PosInt = 10  # PCA components per layer
    plot_layer: NonNegInt = 0  # layer whose neuron maps are drawn
    n_cols: PosInt = 8  # columns in the neuron-map grid


class Config(_Strict):
    data: DataConfig = DataConfig()
    net: NetConfig = NetConfig()
    train: TrainConfig = TrainConfig()
    meanfield: MeanFieldConfig = MeanFieldConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    device: Literal["auto", "cpu", "cuda"] = "auto"
    dtype: Literal["float32", "float64"] = "float64"

    def section_hash(self, *sections: str) -> str:
        """Hash of the named sections, so an artefact can tell which config it came from.

        The top-level `dtype` is always folded in: it changes every artefact's numbers but
        belongs to no section. Canonical JSON (sorted keys) makes the hash independent of
        field and argument order.
        """
        if not sections:
            raise ValueError("section_hash needs at least one section name")
        unknown = set(sections) - set(type(self).model_fields)
        if unknown:
            raise ValueError(f"unknown config sections {sorted(unknown)}")
        hashed = self.model_dump(include={*sections, "dtype"})
        canonical = json.dumps(hashed, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def default_config() -> Config:
    return Config()


def load_config(path: str | Path) -> Config:
    """Read a config file; raises pydantic.ValidationError on unknown keys or bad types."""
    return Config.model_validate_json(Path(path).read_text())


def save_config(cfg: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg.model_dump(), indent=2, sort_keys=True) + "\n")


def resolve_device(cfg: Config) -> torch.device:
    cuda = torch.cuda.is_available()
    if cfg.device == "cuda" and not cuda:
        raise RuntimeError('config requests device="cuda" but CUDA is not available on this host')
    if cfg.device == "auto":
        return torch.device("cuda" if cuda else "cpu")
    return torch.device(cfg.device)


def resolve_dtype(name: str) -> torch.dtype:
    dtypes = {"float32": torch.float32, "float64": torch.float64}
    if name not in dtypes:
        raise ValueError(f"unknown dtype {name!r}; valid: {sorted(dtypes)}")
    return dtypes[name]
