"""Tiny training runs whose outputs are pinned in tests/fixtures/train_<name>.pt.

The fixtures are written with

    uv run python -c "from tests import pinned_training; pinned_training.write_fixtures()"

They pin a single-network run (net.B = 1) bit for bit: its config hash, every checkpoint and
the final file (net and optimiser state, history, RNG state). Regenerate them only when that
behaviour is meant to change, and say so in the commit.

Provenance: written by the code of `main` before the ensemble dimension (net.B) existed, so
the pinned test proves B = 1 runs are unchanged by it. "adam" is a monitored L2 run, "sgld" an
SGLD run with a dense spatial prior.
"""

from pathlib import Path

import torch

from fct.cache import cache_paths
from fct.config import Config, DataConfig, NetConfig, TrainConfig
from fct.data import make_dataset
from fct.spatial import EmbeddingSpec, KernelSpec, PopulationSpec, SpatialConfig
from fct.train import train_from_config

FIXTURES = Path(__file__).parent / "fixtures"
CPU, F64 = torch.device("cpu"), torch.float64


def _spatial() -> SpatialConfig:
    random = EmbeddingSpec(kind="random", seed=3)
    return SpatialConfig(
        input=PopulationSpec(embedding=random, kernel=KernelSpec(length_scale=0.3)),
        hidden=PopulationSpec(embedding=random, kernel=KernelSpec(length_scale=0.2)),
        output=PopulationSpec(embedding=random, kernel=KernelSpec(length_scale=0.4)),
        min_rel_eig=1e-2,
    )


def pinned_config(name: str, **net_kw) -> Config:
    """The config of pinned run `name` ("adam" or "sgld"); `net_kw` overrides net fields."""
    train = {"n_epochs": 7, "lr": 1e-2, "log_int": 2, "save_int": 3, "seed": 4}
    if name == "adam":
        train |= {"monitor_int": 3}
        spatial = None
    elif name == "sgld":
        train |= {"optimizer": "sgld", "beta": 50.0, "weight_lambda": 1e-2, "lr": 1e-3}
        spatial = _spatial()
    else:
        raise ValueError(f"unknown pinned run {name!r}")
    return Config(
        data=DataConfig(P=12, D_in=4, d=2, embedding="linear", seed=1),
        net=NetConfig(N=6, L=2, g=1.0, phi="tanh", seed=2, **net_kw),
        train=TrainConfig(**train),
        spatial=spatial,
        dtype="float64",
        device="cpu",
    )


def usum_monitor(net) -> dict[str, list]:
    """A cheap deterministic monitor shaped like the MIR monitor (lists over two 'layers')."""
    total = net.U.detach().sum().item()
    return {"usum": [total, -total]}


def run_pinned(name: str, cache_dir: Path, **net_kw) -> dict:
    """Train pinned run `name` in `cache_dir`, interrupting nothing; return what is pinned:
    the config hash and the raw contents of every file under train/ (by file name)."""
    cfg = pinned_config(name, **net_kw)
    dataset = make_dataset(cfg.data, CPU, F64)
    monitor = usum_monitor if cfg.train.monitor_int > 0 else None
    files = {}

    def keep_checkpoints(_line: str) -> None:
        # Only the latest checkpoint survives a run, so each one is read while it exists.
        for path in cache_paths(cache_dir).train.glob("ckpt_*.pt"):
            files.setdefault(path.name, torch.load(path, weights_only=True))

    train_from_config(cfg, dataset, cache_dir, log=keep_checkpoints, monitor=monitor)
    final = cache_paths(cache_dir).train / "final.pt"
    files["final.pt"] = torch.load(final, weights_only=True)
    return {"config_hash": cfg.section_hash("data", "net", "train"), "files": files}


def write_fixtures(scratch: Path = Path("tmp/pinned_training")) -> None:
    """Write every fixture; run only with the code whose behaviour is to be pinned."""
    FIXTURES.mkdir(exist_ok=True)
    for name in ("adam", "sgld"):
        cache_dir = scratch / name
        if cache_dir.exists():
            raise FileExistsError(f"{cache_dir} exists; delete it to write fresh fixtures")
        torch.save(run_pinned(name, cache_dir), FIXTURES / f"train_{name}.pt")
