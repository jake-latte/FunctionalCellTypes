"""An independent reference for the exemplar's mean-field solves (L = 1), and how far each
finished solve under a cache root is from it.

For one hidden layer the saddle point is a fixed point of Phi = <psi psi^T> under the tilt
T(Phi) = -2c A Y Y^T A, A = (alpha I + Phi)^-1, so it is found here by a damped Picard
iteration Phi <- (Phi + <psi psi^T>_T(Phi)) / 2 from the untilted kernel, with the moments from
many long HMC runs: no inner ascent, no outer descent and no net in it. Run

    python notebooks/exemplar_reference.py CACHE_ROOT

for every exemplar run under CACHE_ROOT (the directories `fct.exemplar` writes, each with its
config.json; the net and data are loaded from its cache, never retrained if present): the
reference's largest eigenvalue and mean diagonal, the net's kernel and each finished solve's
relative Frobenius distance to it. `docs/design.md` (the exemplar section) quotes its output.
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import torch

from fct.activations import get_phi
from fct.cache import cache_paths
from fct.config import Config, load_config
from fct.exemplar import train_exemplar
from fct.hmc import SingleSiteHMC, relative_fro
from fct.meanfield import FINAL_NAME, net_kernels, network_constants, resolve_beta

F64 = torch.float64


def picard_reference(
    Sigma: torch.Tensor,
    target: Callable[[torch.Tensor], torch.Tensor],
    rate_tilt: float,
    phi: Callable[[torch.Tensor], torch.Tensor],
    *,
    n_chains: int = 8192,
    warmup_trans: int = 100,
    n_trans: int = 100,
    n_iter: int = 30,
    seed: int = 0,
) -> tuple[torch.Tensor, float]:
    """The fixed point of Phi = <psi psi^T> under the tilt target(Phi) (prior N(0, Sigma), rate
    tilt rate_tilt, psi = phi(h)) by Phi <- (Phi + <psi psi^T>_target(Phi)) / 2 from the
    untilted kernel, each moment from n_trans transitions of n_chains HMC chains after
    warmup_trans at the new tilt; returns it and the relative change of its last iteration."""
    gen = torch.Generator(device=Sigma.device).manual_seed(seed)
    sampler = SingleSiteHMC(Sigma, torch.zeros_like(Sigma), rate_tilt, phi, True,
                            n_chains=n_chains, sample_dtype=F64, generator=gen)  # fmt: skip

    def moments(M: torch.Tensor) -> torch.Tensor:
        sampler.set_tilt(M)
        sampler.run(warmup_trans, collect=False)
        return sampler.run(n_trans)[0]["psi_psi"]

    Phi, change = moments(torch.zeros_like(Sigma)), float("nan")
    for _ in range(n_iter):
        new = 0.5 * Phi + 0.5 * moments(target(Phi))
        change, Phi = relative_fro(new, Phi), new
    return Phi, change


def exemplar_reference(cfg: Config, X: torch.Tensor, Y: torch.Tensor, **kw) -> torch.Tensor:
    """`picard_reference` for the exemplar config `cfg` on its solver inputs (X, Y): the action's
    constants as `fct.meanfield.run_meanfield` takes them (explicit meanfield values, otherwise
    `network_constants`)."""
    if cfg.net.L != 1:
        raise ValueError(
            f"the Picard reference is for one hidden layer, the config has {cfg.net.L}"
        )
    X, Y = X.to(F64), Y.to(F64)
    derived = network_constants(resolve_beta(cfg), cfg.train.weight_lambda, cfg.train.rate_lambda,
                                X.shape[0], Y.shape[1])  # fmt: skip
    mf = cfg.meanfield
    const = {k: derived[k] if getattr(mf, k) is None else getattr(mf, k) for k in derived}
    eye = torch.eye(X.shape[0], dtype=F64)
    Sigma = const["prior_scale"] * X @ X.T / X.shape[1]

    def target(Phi: torch.Tensor) -> torch.Tensor:
        A = torch.linalg.inv(const["alpha"] * eye + Phi)
        return -2.0 * const["learning_prefactor"] * A @ Y @ Y.T @ A

    Phi, change = picard_reference(Sigma, target, const["rate_tilt"], get_phi(cfg.net.phi), **kw)
    print(f"  Picard reference: last relative change {change:.1e}")
    return Phi


def _scales(Phi: torch.Tensor) -> str:
    top = torch.linalg.eigvalsh(Phi)[-1]
    return f"top eigenvalue {top:.3f}, mean diagonal {Phi.diagonal().mean():.4f}"


def report(cache_root: Path) -> None:
    """Print the reference and the distances to it for every exemplar run under cache_root."""
    for run_dir in sorted(Path(cache_root).glob("exemplar-*")):
        cfg = load_config(cache_paths(run_dir).config)
        if cfg.net.L != 1:
            print(f"{run_dir.name}: L = {cfg.net.L}, skipped")
            continue
        run = train_exemplar(cfg, cache_root, log=lambda _m: None)
        X, Y = run.dataset.x[: cfg.meanfield.n_points], run.dataset.y[: cfg.meanfield.n_points]
        print(f"{run_dir.name}:")
        ref = exemplar_reference(cfg, X, Y)
        net_Phi = net_kernels(run.net, run.dataset, cfg)[0]
        print(f"  reference: {_scales(ref)}; net {relative_fro(net_Phi, ref):.4f} from it")
        for final in sorted(run_dir.glob(f"solve-*/meanfield/{FINAL_NAME}")):
            payload = torch.load(final, map_location="cpu", weights_only=True)
            solver = payload["meanfield_config"]["solver"]
            Phi = payload["Phi"][0]
            print(f"  {final.parent.parent.name} ({solver['sampler']}, init {solver['init']}, "
                  f"{solver['n_outer']} steps): {relative_fro(Phi, ref):.4f} from it, "
                  f"{_scales(Phi)}")  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cache_root", type=Path, help="the exemplar notebook's CACHE_ROOT")
    report(parser.parse_args(argv).cache_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
