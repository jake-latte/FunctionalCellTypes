"""The tiny solves and configs whose outputs are pinned in tests/fixtures/.

The fixtures are written with

    uv run python -c "from tests import pinned_runs; pinned_runs.write_fixtures()"

They pin behaviour that must not change: `sampler="hmc"` reproduces those solves bit for bit,
and a config.json in the flat layout still loads and hashes to the same cache keys. Regenerate
them only when that behaviour is meant to change, and say so in the commit.

Provenance: the solves `hmc_solve_*.pt` and `meanfield_hashes.json` are written by code whose
net reads its output as V^T phi(h^L) / N (MODEL_VERSION 2) and whose tilt projection follows
`fct.hmc.tilt_rule`: `hmc_solve_<problem>.pt` at the solver's default step control,
`hmc_solve_<problem>_legacy_steps.pt` at `LEGACY_STEP_CONTROL` (the plain Algorithm 1).
`write_fixtures` rewrites all of them. `legacy_config.json` is the flat layout of the code
before the solver sub-section (commit d1d3091), edited by hand: the `readout` key deleted,
the step-control fields added at their `LEGACY_STEP_CONTROL` values, and `latent_range: 1.0`
replaced by `latent_max: 1.0, latent_min: -1.0`, which moved its `result` hash.
`write_fixtures` would save it in the current nested layout, so restore it from git after
running it.

Only flat `MeanFieldConfig` keyword arguments are used here, which both the old flat schema and
the split one accept, so the same code produced the fixtures and checks them.
"""

import json
from pathlib import Path

import torch

from fct.config import STEP_CONTROL_LEGACY

FIXTURES = Path(__file__).parent / "fixtures"
F64 = {"device": "cpu", "dtype": torch.float64}

# Two small, fast problems that between them reach every HMC code path the solver has:
# lazy init with a net and float32 chains; gaussian init, float64 chains, no net.
_PROBLEMS = {
    "lazy_net_f32": {
        "with_net": True,
        "meanfield": {
            "beta": 1e3, "n_chains": 32, "n_outer": 3, "warmup_trans": 10, "n_inner_max": 3,
            "n_trans": 2, "n_trans_outer": 3, "n_init_samples": 256, "init": "lazy",
            "sample_dtype": "float32", "log_int": 1, "seed": 7,
        },
    },
    "gaussian_phi_f64": {
        "with_net": False,
        "meanfield": {
            "beta": 1e3, "n_chains": 32, "n_outer": 3, "warmup_trans": 10,
            "n_inner_max": 3, "n_trans": 2, "n_trans_outer": 3, "n_init_samples": 256,
            "init": "gaussian", "sample_dtype": "float64", "log_int": 1, "seed": 8,
        },
    },
}  # fmt: skip
# The plain Algorithm 1 (`fct.config.STEP_CONTROL_LEGACY` with eta_max = 1), which wrote the
# `*_legacy_steps` fixtures and reproduces them bit for bit.
LEGACY_STEP_CONTROL = {**STEP_CONTROL_LEGACY, "eta_max": 1.0}
PINNED_SOLVES = _PROBLEMS | {
    f"{name}_legacy_steps": spec | {"meanfield": spec["meanfield"] | LEGACY_STEP_CONTROL}
    for name, spec in _PROBLEMS.items()
}
LAMBDA_W, LAMBDA_R, P_PIN, D_IN_PIN = 1e-3, 1e-3, 6, 4

# A config.json in the flat pre-split layout, with non-default values in both the problem
# and the solver fields of the meanfield section. Its solver runs the plain Algorithm 1, so
# its hashes are those of the code before the step-control fields existed.
LEGACY_CONFIG_FILE = FIXTURES / "legacy_config.json"
LEGACY_OVERRIDES = {
    "data": {"P": 64, "D_in": 8},
    "net": {"N": 16, "L": 2},
    "train": {"rate_lambda": 1e-2, "weight_lambda": 1e-4},
    "meanfield": {
        "n_points": 20, "beta": 1e3, "rate_tilt": 1e-4, "alpha": 0.3, "learning_prefactor": 0.5,
        "n_chains": 64, "eps0": 0.1, "init": "gaussian", "log_int": 1, "save_int": 3,
        "precondition": False, "seed": 4, **LEGACY_STEP_CONTROL,
    },
    "device": "cpu",
}  # fmt: skip
MEANFIELD_RESULT_SECTIONS = ("data", "net", "train", "meanfield")
HASHES_FILE = FIXTURES / "meanfield_hashes.json"


def pinned_inputs(seed: int = 90) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    X = torch.randn(P_PIN, D_IN_PIN, generator=gen, **F64)
    Y = torch.randn(P_PIN, 2, generator=gen, **F64)
    return X, Y


def pinned_solve(name: str, **meanfield_overrides):
    """Run the pinned solve `name` (optionally with extra meanfield settings)."""
    from fct.config import MeanFieldConfig, NetConfig
    from fct.meanfield import solve
    from fct.model import make_net

    spec = PINNED_SOLVES[name]
    X, Y = pinned_inputs()
    net = None
    if spec["with_net"]:
        net = make_net(NetConfig(N=8, L=2, phi="relu", seed=3), D_IN_PIN, 2, "cpu", torch.float64)
    # Three outer steps are far from converged: the final-residual check, which changes no
    # number, is switched off.
    cfg = MeanFieldConfig(
        **{"tilt_tripwire_resid": 1e9, **spec["meanfield"], **meanfield_overrides}
    )
    return solve(X, Y, 2, LAMBDA_W, LAMBDA_R, torch.relu, 1.0, cfg, net=net, log=lambda _m: None)


def pinned_solve_payload(result) -> dict:
    """What the fixture stores: the kernels and every history column except wall time."""
    from fct.meanfield import _history_to_lists

    history = {k: v for k, v in _history_to_lists(result.history).items() if k != "time"}
    return {"Phi": result.Phi, "Phi_hat": result.Phi_hat, "history": history}


def legacy_config():
    from fct.config import Config

    return Config.model_validate(LEGACY_OVERRIDES)


def write_hash_fixture() -> None:
    """Write `meanfield_hashes.json`: the hashes change with LOSS_VERSION / MODEL_VERSION (the
    train and net sections enter them), the solves do not, so a version bump needs only this."""
    from fct.config import default_config

    hashes = {
        label: {
            "meanfield": cfg.section_hash("meanfield"),
            "result": cfg.section_hash(*MEANFIELD_RESULT_SECTIONS),
        }
        for label, cfg in (("default", default_config()), ("legacy", legacy_config()))
    }
    HASHES_FILE.write_text(json.dumps(hashes, indent=2) + "\n")


def write_fixtures() -> None:
    """Write every fixture; run only with the code whose behaviour is to be pinned."""
    from fct.config import save_config

    FIXTURES.mkdir(exist_ok=True)
    for name in PINNED_SOLVES:
        torch.save(pinned_solve_payload(pinned_solve(name)), FIXTURES / f"hmc_solve_{name}.pt")
    save_config(legacy_config(), LEGACY_CONFIG_FILE)
    write_hash_fixture()
