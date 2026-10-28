"""Hyperparameter schema, JSON I/O and section hashes; the CLI's `init` writes it to
CACHE_DIR/config.json.

Every model forbids unknown keys and is strict about types, so a typo or a "5" where an int
is expected fails at load time instead of silently becoming a default or a coerced value.
Fields are range-constrained too, so impossible values fail at load time naming the field.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import torch
from pydantic import Field, model_validator

# The constrained types and StrictModel are re-exported: other modules import them from here.
from fct.schema import NonNegFloat, NonNegInt, PosFloat, PosInt, StrictModel
from fct.spatial import SpatialConfig
from fct.tasks import TASK_SPECS

# Version of the training-loss definition (`fct.train.compute_loss`). The loss belongs to the
# train section but is code, not a config field, so it is hashed separately: artefacts made
# under another definition are rejected as stale instead of mixed with new ones. Increase it
# whenever the loss changes in a way the TrainConfig fields do not express. History: 1 summed
# the rate penalty over the samples, 2 averaged it over them, 3 sums it again, 4 scales the
# weight prior by 1 / (2 beta) (beta = 1 for Adam).
LOSS_VERSION = 4

# Version of the network definition (`fct.model.MLP`), folded into every hash that holds the
# net section, for the same reason: a trained net or a mean-field result made with another
# forward pass is rejected as stale. History: 1 read the output from the pre-activation,
# V^T h^L / N; 2 reads it from the post-activation, V^T phi(h^L) / N.
MODEL_VERSION = 2


class DataConfig(StrictModel):
    """Data settings. `task` selects the generator and only that task's fields are read: the
    fields of other tasks are ignored, but they still enter the cache hash, so changing one
    makes an existing data cache stale. Sizes and counts are PosInt (>= 1) except
    `max_objects` (>= 0); `latent_range` is PosFloat. Constraints between fields (the task's
    fixed D_in, max_objects <= grid_size**2, n_motifs <= 511, D_in > n_latents) are checked
    here so a bad config fails at load time.
    """

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
    def _task_accepts_the_fields(self) -> Self:
        # Each non-position task checks its own constraints (sizes that must agree, limits).
        if self.task in TASK_SPECS:
            TASK_SPECS[self.task].validate(self)
        return self


Phi = Literal["relu", "tanh", "sigmoid", "id"]
Optimizer = Literal["adam", "sgld"]


class NetConfig(StrictModel):
    N: PosInt = 100
    L: PosInt = 3
    g: PosFloat = 1.0
    phi: Phi = "relu"
    seed: int = 0


class TrainConfig(StrictModel):
    n_epochs: NonNegInt = 15_000
    lr: PosFloat = 1e-3
    # The loss carries the weight prior as weight_lambda / (2 beta) * quadratic form, with beta
    # = `beta` for SGLD and 1 for Adam (`fct.train.prior_coefficient`).
    weight_lambda: NonNegFloat = 2e-4
    rate_lambda: NonNegFloat = 1e-3
    log_int: PosInt = 100
    save_int: PosInt = 100
    seed: int = 0
    # Modularity monitor: 0 disables it. The settings are hashed with the run so a cached
    # run always matches the monitor settings that produced its history.
    monitor_int: NonNegInt = 0
    monitor_bins: Annotated[int, Field(ge=2)] = 10
    monitor_shuffles: PosInt = 5
    # "sgld" adds Gaussian noise of variance 2 * lr / beta to every step (see fct.sgld). `beta`
    # is read only by SGLD; both fields enter the hash only for SGLD (`section_hash`), so
    # adam runs keep the hashes they had before these fields existed.
    optimizer: Optimizer = "adam"
    beta: PosFloat = 1.0


# Solver fields that no longer exist, at the value every hash written while they did holds
# (`MeanFieldConfig.hash_payload` keeps them there, so those hashes do not move). A config that
# sets one at that value loads (every saved config.json has the key); at any other value it is
# rejected, since the value would be ignored. The text says what took the field's place.
RETIRED_SOLVER_FIELDS: dict[str, Any] = {"max_mc_err": 1.0}
_RETIRED_BECAUSE = {
    "max_mc_err": "a step's Monte Carlo error is now bounded relative to that of the accepted "
    "point before it, by max_mc_err_growth (an absolute bound rejected every step of a solve "
    "with few chains)",
}


class SolverConfig(StrictModel):
    """How the saddle point is found: Algorithm 1, the single-site sampler, the initial guess,
    logging and checkpoints. `MeanFieldConfig` holds the equations being solved.

    `sampler` picks the single-site expectations: persistent HMC chains ("hmc") or
    self-normalised importance sampling from the Gaussian prior ("importance", see
    `fct.importance`). Only HMC reads n_chains, n_leap, eps0, target_accept, adapt_rate,
    precondition, max_div_frac and eps_collapse; only importance sampling reads the importance
    block; both read everything else (a run is n_trans / n_trans_outer transitions of either).
    """

    sampler: Literal["hmc", "importance"] = "hmc"
    # Algorithm 1
    n_outer: PosInt = 300
    n_inner_max: PosInt = 30
    inner_tol: PosFloat = 1e-2
    inner_tol_mc: PosFloat = 2.0
    eta_phi: PosFloat = 0.05
    eta_phihat: PosFloat = 0.05
    eta_max: PosFloat = 0.25
    eta_grow: PosFloat = 1.1
    max_backtrack: PosInt = 8
    eta_min_rel: NonNegFloat = 1e-3
    max_div_frac: NonNegFloat = 0.02
    max_h2_growth: PosFloat = 10.0
    # A run whose sample scale ||h||^2/P exceeds max_h2_prior_ratio times the prior's (mean
    # diagonal of Sigma) is unhealthy, however it got there: max_h2_growth bounds only the jump
    # from the point before, so a scale can climb by steps or start a run already escaped.
    # Legitimate solves sit at 1 to 6; escaped HMC chains on a non-normalisable tilt at 3e3.
    max_h2_prior_ratio: PosFloat = 100.0
    # A run whose Monte Carlo error of <psi psi^T>, per transition (MC error x sqrt(transitions),
    # so runs of different length compare), exceeds max_mc_err_growth times that of the
    # accepted point before it is unhealthy: the step made the sampler mix worse. The size of
    # the MC error is a property of the sample size, which no revert cures, so only its growth
    # is held against a step. A backstop: a ratio of 10 is about a hundredfold ESS drop under
    # HMC, where the h2, divergence and step-size checks normally fire first, and under
    # importance sampling it overlaps min_ess_drop. Not lower: the estimate is itself noisy
    # (fourth moments, the lag-1 ESS estimate), and sound relu runs differ by up to 6x from the
    # point before (tanh ones by 1.3x).
    max_mc_err_growth: PosFloat = 10.0
    eps_collapse: NonNegFloat = 1e-2
    project_tilt: bool = True
    tilt_floor: Annotated[float, Field(ge=0, lt=1)] = 0.05
    max_rel_step: PosFloat = 0.1
    # Step control (`fct.meanfield`). inner_step_control "progress" also rejects an inner step
    # after which a layer's inner residual exceeds the smallest one accepted in that ascent by
    # more than the convergence tolerance, whose Monte Carlo part it caps at 0.5; "health"
    # validates a step on sampler health alone. eta_max caps only the outer step size;
    # eta_max_phihat caps each layer's inner one. phi_step_scale "kernel" steps Phi_l along
    # Phi_l dS/dPhi_l Phi_l, the Newton direction of a Gaussian layer (eta = 2 is a full Newton
    # step there; a bounded phi responds several times more weakly, hence the cap of 0.25),
    # "trace" along c_l^2 dS/dPhi_l with c_l = tr(Phi_l)/P. inner_start "target" restarts every
    # inner ascent of the last layer at its Stein target -2c A Y Y^T A (exact, no sampling), so
    # directions of Phi_hat_L the kernel barely responds to cannot collect Monte Carlo noise
    # across outer steps; the last layer's outer step is then a damped stochastic Picard step,
    # whose convergence measure is history["picard_resid"]. "previous" continues from the last
    # tilt. STEP_CONTROL_LEGACY with eta_max = 1 is the plain Algorithm 1.
    inner_step_control: Literal["health", "progress"] = "progress"
    eta_max_phihat: PosFloat = 0.25
    phi_step_scale: Literal["trace", "kernel"] = "kernel"
    inner_start: Literal["previous", "target"] = "target"
    # A tripwire for a lost tilt, not a convergence test: `solve` raises UnconvergedSolveError
    # instead of returning (and writing final.pt) when, in some layer, the largest Stein
    # fixed-point residual of Phi_hat over the last tilt_tripwire_steps outer steps is >=
    # tilt_tripwire_resid. A tilt near 0 (e.g. projected away) has a residual near 1; an
    # unconverged solve with a residual below the tripwire passes, so judge convergence from the
    # history.
    tilt_tripwire_resid: PosFloat = 0.9
    tilt_tripwire_steps: PosInt = 3
    pd_floor_rel: PosFloat = 1e-8
    inv_floor_rel: PosFloat = 1e-4
    # Sampler runs (HMC-only: n_chains, n_leap, eps0, target_accept, adapt_rate, precondition)
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
    # Importance sampling: draws per transition (n_trans / n_trans_outer transitions per run,
    # as for HMC) and the proposal N(0, s^2 Sigma) with s = importance_proposal_scale.
    # min_ess_drop: a step is reverted (and its step size halved, as with max_h2_growth) when
    # the ESS fraction after it is below min_ess_drop times that of the accepted point before it.
    # min_ess_frac: the ESS fraction every accepted point must reach; below it the prior
    # proposal is inadequate for the problem and `solve` raises. Both enter the importance
    # hash (conservative: lowering the floor after a raise starts a fresh cache directory
    # rather than resuming the raised run's checkpoints).
    n_importance_samples: PosInt = 4096
    importance_proposal_scale: PosFloat = 1.0
    min_ess_drop: Annotated[float, Field(gt=0, le=1)] = 0.1
    min_ess_frac: Annotated[float, Field(ge=0, le=1)] = 0.01
    # Initial guess, logging, checkpoints, randomness
    init: Literal["lazy", "gaussian", "net"] = "lazy"
    n_init_samples: PosInt = 20_000
    log_int: PosInt = 5
    save_int: PosInt = 10
    seed: int = 0

    @model_validator(mode="before")
    @classmethod
    def _retired_fields_only_at_their_last_value(cls, data: Any) -> Any:
        if not isinstance(data, dict) or not data.keys() & RETIRED_SOLVER_FIELDS.keys():
            return data
        for key, value in RETIRED_SOLVER_FIELDS.items():
            if key in data and data[key] != value:
                raise ValueError(
                    f"{key}={data[key]!r} is no longer a setting ({_RETIRED_BECAUSE[key]}); "
                    "delete the key"
                )
        return {k: v for k, v in data.items() if k not in RETIRED_SOLVER_FIELDS}


# The solver fields that existed before the sampler choice. They enter the mean-field result
# hash exactly as they did then (flat, next to the problem fields), so results solved before
# the sampler choice existed stay valid. log_int and save_int change no number but are among
# them: dropping them would change every existing hash (RETIRED_SOLVER_FIELDS holds those
# that are no longer settings).
LEGACY_SOLVER_FIELDS = (
    "n_outer", "n_inner_max", "inner_tol", "inner_tol_mc", "eta_phi", "eta_phihat", "eta_max",
    "eta_grow", "max_backtrack", "eta_min_rel", "max_div_frac", "max_h2_growth",
    "eps_collapse", "project_tilt", "tilt_floor", "max_rel_step", "pd_floor_rel",
    "inv_floor_rel", "n_chains", "n_trans", "n_trans_outer", "n_leap", "eps0", "target_accept",
    "adapt_rate", "warmup_trans", "sample_dtype", "jitter_rel", "precondition", "init",
    "n_init_samples", "log_int", "save_int", "seed",
)  # fmt: skip
# Fields that enter the result hash only when sampler == "importance": with "hmc" they change
# nothing, and leaving them out keeps the hash of an HMC solve what it always was.
IMPORTANCE_FIELDS = (
    "sampler", "n_importance_samples", "importance_proposal_scale", "min_ess_drop",
    "min_ess_frac",
)  # fmt: skip


# Solver fields left out of every hash: they check a result and never change a number of it.
UNHASHED_SOLVER_FIELDS = ("tilt_tripwire_resid", "tilt_tripwire_steps")
# The step-control fields, at the values under which the iteration is the plain Algorithm 1
# (with eta_max = 1, which LEGACY_SOLVER_FIELDS hashes). A field enters the hash only when it
# differs from its value here, as IMPORTANCE_FIELDS enter only off "hmc": a config set to
# these values keeps the hash it had before the fields existed, and one that leaves them
# unset gets the defaults and a new hash.
STEP_CONTROL_LEGACY: dict[str, Any] = {
    "inner_step_control": "health", "eta_max_phihat": 1.0, "phi_step_scale": "trace",
    "inner_start": "previous",
}  # fmt: skip
STEP_CONTROL_FIELDS = tuple(STEP_CONTROL_LEGACY)
# Health bounds, each hashed only when it differs from its default here, so a default config
# keeps its hash. That hash does not cover the solver's health checks themselves: a cached
# solve keyed by it may have been written by code without the check of the outer-gradient run
# (meanfield._outer_run), without these bounds, or with the absolute MC-error bound
# max_mc_err = 1 in their place, and a solve those checks would have changed keeps its hash:
# any outer run with escaped chains, divergences, resets, a step-size collapse, an ESS drop or
# an MC-error jump, and any solve with an MC error above 1 (fewer than about a hundred HMC
# chains), whose inner steps max_mc_err rejected. Solves they never trip are unchanged by them.
HEALTH_BOUND_DEFAULTS: dict[str, Any] = {"max_h2_prior_ratio": 100.0, "max_mc_err_growth": 10.0}
HEALTH_BOUND_FIELDS = tuple(HEALTH_BOUND_DEFAULTS)


class MeanFieldConfig(StrictModel):
    """The mean-field problem: which points, and the constants of the action (`beta`, the
    prior scale, `alpha`, the learning prefactor, the rate tilt; None means "follow the
    network": derived from the train section, see `fct.meanfield.network_constants`). How it is solved is `solver`.

    A flat section in the layout used before the split (solver fields next to the problem
    fields) still loads: the solver keys are moved into `solver`. A key given both flat and in
    `solver` is an error.
    """

    n_points: PosInt = 100  # solve on the first n_points training points
    beta: PosFloat | None = None  # None: train.beta for SGLD, 1 for Adam
    prior_scale: PosFloat | None = None
    alpha: PosFloat | None = None
    learning_prefactor: PosFloat | None = None
    rate_tilt: NonNegFloat | None = None
    solver: SolverConfig = SolverConfig()

    @model_validator(mode="before")
    @classmethod
    def _flat_solver_keys_into_the_group(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        solver_keys = SolverConfig.model_fields.keys() | RETIRED_SOLVER_FIELDS.keys()
        flat = {k: v for k, v in data.items() if k in solver_keys}
        if not flat:
            return data
        group = data.get("solver", {})
        if isinstance(group, SolverConfig):
            group = group.model_dump(exclude_unset=True)
        if not isinstance(group, dict):
            return data  # let field validation report the bad `solver` value
        twice = sorted(flat.keys() & group.keys())
        if twice:
            raise ValueError(
                f"{twice} given both in meanfield and in meanfield.solver; "
                "set solver fields only in meanfield.solver"
            )
        rest = {k: v for k, v in data.items() if k not in flat}
        return {**rest, "solver": {**group, **flat}}

    def hash_payload(self) -> dict[str, Any]:
        """The section as it enters a cache hash: the problem fields and LEGACY_SOLVER_FIELDS
        in one flat dict (the pre-split layout), plus each of STEP_CONTROL_FIELDS that differs
        from STEP_CONTROL_LEGACY and each of HEALTH_BOUND_FIELDS that differs from
        HEALTH_BOUND_DEFAULTS, plus IMPORTANCE_FIELDS when the sampler is "importance", plus
        RETIRED_SOLVER_FIELDS at their last values."""
        solver = self.solver.model_dump()
        payload = self.model_dump(exclude={"solver"})
        payload.update({k: solver[k] for k in LEGACY_SOLVER_FIELDS} | RETIRED_SOLVER_FIELDS)
        payload.update(
            {k: solver[k] for k in STEP_CONTROL_FIELDS if solver[k] != STEP_CONTROL_LEGACY[k]}
        )
        payload.update(
            {k: solver[k] for k in HEALTH_BOUND_FIELDS if solver[k] != HEALTH_BOUND_DEFAULTS[k]}
        )
        if self.solver.sampler != "hmc":
            payload.update({k: solver[k] for k in IMPORTANCE_FIELDS})
        return payload


class AnalysisConfig(StrictModel):
    n_grid: PosInt = 25  # test grid is n_grid x n_grid
    n_components: PosInt = 10  # PCA components per layer
    plot_layer: NonNegInt = 0  # layer whose neuron maps are drawn
    n_cols: PosInt = 8  # columns in the neuron-map grid


class Config(StrictModel):
    data: DataConfig = DataConfig()
    net: NetConfig = NetConfig()
    train: TrainConfig = TrainConfig()
    meanfield: MeanFieldConfig = MeanFieldConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    # The spatial weight prior (fct.spatial); None trains with the plain weight L2 norm. It
    # changes the training loss, so `section_hash` folds it in with the train section.
    spatial: SpatialConfig | None = None
    device: Literal["auto", "cpu", "cuda"] = "auto"
    dtype: Literal["float32", "float64"] = "float64"

    def section_hash(self, *sections: str) -> str:
        """Hash of the named sections, so an artefact can tell which config it came from.

        The top-level `dtype` is always folded in: it changes every artefact's numbers but
        belongs to no section. So is `MODEL_VERSION` when "net" is among the sections,
        `LOSS_VERSION` when "train" is, and
        `train.optimizer` / `train.beta` are left out of the hash for the adam optimiser. The
        `spatial` section enters with "train" only when it is set, so runs without a spatial
        prior keep the hashes they had before the section existed. The meanfield section
        enters as `MeanFieldConfig.hash_payload()`, not as its dump.
        Canonical JSON (sorted keys) makes the hash independent of field and argument order.
        """
        if not sections:
            raise ValueError("section_hash needs at least one section name")
        unknown = set(sections) - set(type(self).model_fields)
        if unknown:
            raise ValueError(f"unknown config sections {sorted(unknown)}")
        hashed = self.model_dump(include={*sections, "dtype"})
        if "meanfield" in sections:
            hashed["meanfield"] = self.meanfield.hash_payload()
        if "net" in sections:  # the forward pass belongs to the net section
            hashed["model_version"] = MODEL_VERSION
        if "train" in sections:  # the loss definition lives in the train section
            hashed["loss_version"] = LOSS_VERSION
            if self.train.optimizer == "adam":  # SGLD-only fields must not move adam hashes
                del hashed["train"]["optimizer"], hashed["train"]["beta"]
            if self.spatial is not None:  # the weight prior is part of the training loss
                hashed["spatial"] = self.spatial.model_dump()
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
    # Write then rename: an interrupted save never leaves a truncated config.json.
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(cfg.model_dump(), indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


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
