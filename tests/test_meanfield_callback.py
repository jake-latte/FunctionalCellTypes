"""`solve(on_outer=...)`: the per-outer-step hook that live plots hang on.

The hook must be a pure observer. Every test compares against a run without it, so a callback
that perturbed the generator, the state or the history would show up as a difference.
"""

import numpy as np
import pytest
import torch

from fct.meanfield import HISTORY_KEYS, OuterSnapshot, solve
from tests.test_meanfield import (
    Interrupted,
    importance_cfg,
    interrupt_on_call,
    relu_problem,
    resume_cfg,
    silent,
    small_net,
)

SAMPLERS = {"hmc": resume_cfg, "importance": importance_cfg}


def run(sampler, **kwargs):
    X, Y = relu_problem(8, 8, seed=50)
    return solve(
        X, Y, 2, 1e-3, 1e-3, torch.relu, 1.0, SAMPLERS[sampler](), net=small_net(8), log=silent,
        **kwargs,
    )  # fmt: skip


def assert_same_result(a, b):
    assert torch.equal(a.Phi, b.Phi) and torch.equal(a.Phi_hat, b.Phi_hat)
    assert set(a.history) == set(b.history)
    for key in set(HISTORY_KEYS) - {"time"}:  # wall-clock time differs between any two runs
        for x, y in zip(a.history[key], b.history[key], strict=True):
            assert np.array_equal(x, y, equal_nan=True), key


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_a_reading_callback_changes_nothing_and_sees_every_step(sampler):
    seen = []

    def read(snap: OuterSnapshot) -> None:
        seen.append((snap.outer, len(snap.history["outer_step"]), snap.Phi.norm().item()))

    assert_same_result(run(sampler, on_outer=read), run(sampler))
    n_outer = SAMPLERS[sampler]().solver.n_outer
    assert [(o, n) for o, n, _ in seen] == [(k, k + 1) for k in range(n_outer)]


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_a_callback_that_scribbles_on_its_snapshot_changes_nothing(sampler):
    """The kernels and the history lists are copies: writing to them cannot reach the solver."""

    def scribble(snap: OuterSnapshot) -> None:
        snap.Phi.zero_()
        snap.Phi_hat.fill_(float("nan"))
        for column in snap.history.values():
            column.clear()

    assert_same_result(run(sampler, on_outer=scribble), run(sampler))


def test_the_snapshot_holds_the_kernels_and_history_after_that_step():
    snaps = []
    result = run("hmc", on_outer=snaps.append)
    last = snaps[-1]
    assert torch.equal(last.Phi, result.Phi) and torch.equal(last.Phi_hat, result.Phi_hat)
    assert last.history["outer_step"] == result.history["outer_step"]
    assert not torch.equal(snaps[0].Phi, last.Phi)  # snapshots do not alias one buffer


def test_a_loaded_final_solve_runs_no_steps_so_the_callback_is_not_called(tmp_path):
    run("hmc", checkpoint_dir=tmp_path, config_hash="h0")
    calls = []
    loaded = run("hmc", checkpoint_dir=tmp_path, config_hash="h0", on_outer=calls.append)
    assert calls == [] and len(loaded.history["outer_step"]) == resume_cfg().solver.n_outer


def test_a_raising_callback_stops_the_solve():
    def boom(snap: OuterSnapshot) -> None:
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        run("hmc", on_outer=boom)


def test_steps_restored_from_a_checkpoint_do_not_call_the_callback(tmp_path, monkeypatch):
    """An interrupted solve resumes at the checkpoint's step; only the steps run in the
    resuming call are reported, each with the history including the restored entries."""
    restore = interrupt_on_call(monkeypatch, 3)  # steps 0 and 1 run; checkpoint after step 1
    with pytest.raises(Interrupted):
        run("hmc", checkpoint_dir=tmp_path, config_hash="h0")
    restore()
    seen = []
    run(
        "hmc",
        checkpoint_dir=tmp_path,
        config_hash="h0",
        on_outer=lambda snap: seen.append((snap.outer, len(snap.history["outer_step"]))),
    )
    assert seen == [(2, 3), (3, 4)]
