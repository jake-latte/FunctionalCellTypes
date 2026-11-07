"""Time one training step of an ensemble (net.B = B) against B single networks stepped in a
Python loop, on the same data. Prints one markdown table row per configuration.

    uv run python jobs/bench_ensemble.py --device cpu
    CUDA_VISIBLE_DEVICES=<uuid> uv run python jobs/bench_ensemble.py --device cuda

Short and in memory only: nothing is written to disk.
"""

import argparse
import statistics
import time
from collections.abc import Callable

import torch

from fct.config import NetConfig
from fct.model import make_net
from fct.sgld import SGLD
from fct.train import LossCoefficients, compute_loss

# (label, N, L, phi, D_in, D_out, P): the phase-transition preset's sizes, a deeper tanh net,
# and a large P.
CASES = [
    ("preset: L1 id P128", 512, 1, "id", 64, 64, 128),
    ("L3 tanh P256", 512, 3, "tanh", 64, 64, 256),
    ("L1 id P2048", 512, 1, "id", 64, 64, 2048),
]


def _step_fn(net, x, y) -> Callable[[], None]:
    optim = SGLD(net.parameters(), lr=1e-6, beta=1e4, member_axis=net.B > 1)

    def step() -> None:
        optim.zero_grad()
        compute_loss(net, x, y, LossCoefficients(prior=0.5, rate=1e-3))[0].backward()
        optim.step()

    return step


def _median_seconds(fn: Callable[[], None], device: torch.device, reps: int) -> float:
    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(3):  # warm-up
        fn()
    sync()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def bench(case, B: int, device: torch.device, dtype: torch.dtype, reps: int) -> str:
    label, N, L, phi, D_in, D_out, P = case
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(P, D_in, generator=gen, dtype=dtype).to(device)
    y = torch.randn(P, D_out, generator=gen, dtype=dtype).to(device)
    ensemble = make_net(NetConfig(N=N, L=L, phi=phi, B=B), D_in, D_out, device, dtype)
    singles = [ensemble.member(b) for b in range(B)]
    ens_step = _step_fn(ensemble, x, y)
    single_steps = [_step_fn(net, x, y) for net in singles]

    def loop() -> None:
        for step in single_steps:
            step()

    t_ens = _median_seconds(ens_step, device, reps)
    t_loop = _median_seconds(loop, device, reps)
    one = _median_seconds(single_steps[0], device, reps)
    return (
        f"| {device.type} | {label} | {B} | {1e3 * t_ens:.2f} | {1e3 * t_loop:.2f} | "
        f"{t_loop / t_ens:.2f} | {1e3 * one:.2f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    args = parser.parse_args()
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    if device.type == "cuda":
        print(f"device: {torch.cuda.get_device_name(device)}")
    else:
        print(f"torch threads: {torch.get_num_threads()}")
    print(f"dtype {args.dtype}; one SGLD step (loss, backward, update); median of {args.reps}")
    print("| device | case | B | ensemble ms | loop of B ms | loop / ensemble | one net ms |")
    print("|---|---|---|---|---|---|---|")
    for case in CASES:
        print(bench(case, args.B, device, dtype, args.reps), flush=True)


if __name__ == "__main__":
    main()
