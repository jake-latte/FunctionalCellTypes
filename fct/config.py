"""Hyperparameter schema, written by `init` to CACHE_DIR/config.json.

Every model forbids unknown keys and is strict about types, so a typo or a "5" where an int
is expected fails at load time instead of silently becoming a default or a coerced value.

Only this module (config, activations, cache) exists on this branch so far: the
`python -m fct ...` CLI is not implemented yet.
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class DataConfig(_Strict):
    P: int = 4096
    D_in: int = 64
    d: int = 2
    var: float = 10.0
    cov: list[list[float]] | None = None  # None -> identity
    embedding: Literal["linear", "fourier"] = "fourier"
    seed: int = 0


class NetConfig(_Strict):
    N: int = 100
    L: int = 3
    g: float = 1.0
    phi: Literal["relu", "tanh", "sigmoid"] = "relu"
    seed: int = 0


class TrainConfig(_Strict):
    n_epochs: int = 15_000
    lr: float = 1e-3
    weight_lambda: float = 1e-3
    rate_lambda: float = 1e-3
    log_int: int = 100
    save_int: int = 100
    seed: int = 0


class MeanFieldConfig(_Strict):
    n_points: int = 100  # solve on the first n_points training points
    beta: float = 1e6
    readout: Literal["h", "phi"] = "h"
    # Algorithm 1
    n_outer: int = 300
    n_inner_max: int = 30
    inner_tol: float = 1e-2
    inner_tol_mc: float = 2.0
    eta_phi: float = 0.05
    eta_phihat: float = 0.05
    eta_max: float = 1.0
    eta_grow: float = 1.1
    max_backtrack: int = 8
    eta_min_rel: float = 1e-3
    max_div_frac: float = 0.02
    max_h2_growth: float = 10.0
    max_mc_err: float = 1.0
    eps_collapse: float = 1e-2
    project_tilt: bool = True
    tilt_floor: float = 0.05
    max_rel_step: float = 0.1
    pd_floor_rel: float = 1e-8
    inv_floor_rel: float = 1e-4
    prior_scale: float | None = None
    alpha: float | None = None
    learning_prefactor: float = 0.5
    rate_tilt: float | None = None
    # HMC
    n_chains: int = 2048
    n_trans: int = 2
    n_trans_outer: int = 5
    n_leap: int = 10
    eps0: float = 0.05
    target_accept: float = 0.75
    adapt_rate: float = 0.05
    warmup_trans: int = 200
    sample_dtype: Literal["float32", "float64"] = "float32"
    jitter_rel: float = 1e-6
    precondition: bool = True
    init: Literal["lazy", "gaussian", "net"] = "lazy"
    n_init_samples: int = 20_000
    log_int: int = 5
    save_int: int = 10
    seed: int = 0


class AnalysisConfig(_Strict):
    n_grid: int = 25  # test grid is n_grid x n_grid
    n_components: int = 10  # PCA components per layer
    plot_layer: int = 0  # layer whose neuron maps are drawn
    n_cols: int = 8  # columns in the neuron-map grid


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
