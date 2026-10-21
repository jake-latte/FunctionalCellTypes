"""Learning-rate tuning: result table logic on hand-built tables, and the held-out evaluation."""

import itertools
import math

import numpy as np
import pytest
import torch

from fct.cache import cache_paths
from fct.config import DataConfig
from fct.data import make_dataset
from fct.lrtune import (
    COLUMNS,
    average_ranks,
    collapse_seeds,
    evaluate_setting,
    global_verdict,
    heldout_dataset,
    loss_curve_stats,
    rank_learning_rates,
    read_rows_csv,
    select,
    select_by,
    write_rows_csv,
)
from fct.sweep import expand, preset, run_setting, save_sweep, setting_dir

LRS = (1e-4, 1e-3, 1e-2)
TASKS = ("linear_teacher", "object_action")
WORST_SETTING = ("object_action", 5, "relu", 0.0)
BASE = {"linear_teacher": 0.5, "object_action": 20.0}  # different loss scales per task


def _table(best_lr, lrs=LRS, seeds=(0,), penalty=0.3, nan_at=()):
    """One row per (setting, lr, seed). The held-out MSE is the task's base value times
    1 + penalty * (decades from the setting's best lr), so the best lr is known by construction.
    `best_lr(task, L)` gives it; `nan_at` lists (task, L, lr) triples that diverge."""
    rows = []
    for task, L, phi, rate in itertools.product(TASKS, (1, 5), ("tanh", "relu"), (0.0, 1e-3)):
        for lr, seed in itertools.product(lrs, seeds):
            decades = abs(math.log10(lr / best_lr(task, L)))
            mse = BASE[task] * (1 + penalty * decades)
            failed = (task, L, lr) in nan_at
            rows.append(
                {
                    "task": task,
                    "L": L,
                    "phi": phi,
                    "rate_lambda": rate,
                    "lr": lr,
                    "seed": seed,
                    "heldout_mse": float("nan") if failed else mse,
                    "diverged": failed,
                }
            )
    return rows


# ---------------------------------------------------------------------------- ranks
def test_average_ranks_share_ties_and_put_nonfinite_last():
    values = np.array([3.0, 1.0, 1.0, np.nan, np.inf])
    assert average_ranks(values).tolist() == [3.0, 1.5, 1.5, 4.5, 4.5]


# ---------------------------------------------------------------------------- selection
def test_single_best_lr_is_found_everywhere_and_globally():
    scores = rank_learning_rates(_table(lambda task, L: 1e-3))
    assert select(scores).best == 1e-3
    assert select(scores).tied == (1e-3,)
    assert {sel.best for sel in select_by(scores, ("task", "L")).values()} == {1e-3}


def test_ranks_are_within_setting_so_loss_scale_does_not_matter():
    scores = rank_learning_rates(_table(lambda task, L: 1e-3))
    best = [s for s in scores if s.lr == 1e-3]
    assert {s.rank for s in best} == {1.0}
    assert {s.gap for s in best} == {0.0}
    worse = [s for s in scores if s.lr == 1e-2 and s.task == "linear_teacher"]
    assert all(s.gap == pytest.approx(0.3) for s in worse)  # relative to that setting's best


def test_per_depth_best_differs_and_is_reported_per_group():
    best = lambda task, L: 1e-2 if L == 1 else 1e-3
    by_group = select_by(rank_learning_rates(_table(best)), ("task", "L"))
    assert by_group[("linear_teacher", 1)].best == 1e-2
    assert by_group[("object_action", 5)].best == 1e-3


def test_a_diverged_run_is_never_selected_and_counts_as_failure():
    # lr 1e-3 is best everywhere but diverges for L=5 of one task.
    table = _table(lambda task, L: 1e-3, nan_at={("linear_teacher", 5, 1e-3)})
    scores = rank_learning_rates(table)
    failed = [s for s in scores if s.failed]
    assert len(failed) == 4 and all(math.isinf(s.gap) for s in failed)
    assert all(s.rank == 3.0 for s in failed)  # worst of three lrs
    assert select_by(scores, ("task", "L"))[("linear_teacher", 5)].best != 1e-3
    chosen = select(scores)
    assert chosen.best != 1e-3
    summary = {s.lr: s for s in chosen.summaries}
    assert summary[1e-3].n_failed == 4 and summary[1e-4].n_failed == 0


def test_diverged_flag_alone_marks_failure_even_with_a_finite_mse():
    table = _table(lambda task, L: 1e-3)
    table[0]["diverged"] = True
    assert sum(s.failed for s in rank_learning_rates(table)) == 1


def test_ties_are_reported():
    table = _table(lambda task, L: 1e-3, penalty=0.0)  # every lr equally good
    sel = select(rank_learning_rates(table))
    assert sel.tied == LRS
    assert sel.best == min(LRS)  # reported best is the smallest tied value, but tied is listed


def test_global_verdict_true_when_one_lr_is_best_everywhere():
    verdict = global_verdict(rank_learning_rates(_table(lambda task, L: 1e-3)))
    assert verdict.single_suffices
    assert verdict.lr == 1e-3 and verdict.median_gap == 0.0 and verdict.worst_ratio == 1.0


def test_global_verdict_false_when_best_lr_differs_by_depth_by_a_factor_of_10():
    best = lambda task, L: 1e-2 if L == 1 else 1e-3
    verdict = global_verdict(rank_learning_rates(_table(best, penalty=0.5)))
    assert not verdict.single_suffices
    assert verdict.median_gap > 0.10


def test_global_verdict_false_if_any_setting_is_more_than_2x_worse():
    table = _table(lambda task, L: 1e-3, penalty=0.01)
    for row in table:  # in one setting lr 1e-3 is 2.5x worse than the others
        if (row["task"], row["L"], row["phi"], row["rate_lambda"]) == WORST_SETTING:
            row["heldout_mse"] = BASE["object_action"] * (2.5 if row["lr"] == 1e-3 else 1.0)
    verdict = global_verdict(rank_learning_rates(table))
    assert verdict.lr == 1e-3  # still the best lr overall
    assert verdict.worst_ratio == pytest.approx(2.5)
    assert not verdict.single_suffices


def test_global_verdict_false_if_the_chosen_lr_has_a_failure():
    clean = _table(lambda task, L: 1e-3, nan_at={("linear_teacher", 5, 1e-3)})
    assert global_verdict(rank_learning_rates(clean)).n_failed == 0  # picks a clean lr
    everywhere = {(t, 5, lr) for t in TASKS for lr in LRS}
    verdict = global_verdict(rank_learning_rates(_table(lambda t, L: 1e-3, nan_at=everywhere)))
    assert verdict.n_failed > 0 and not verdict.single_suffices


def test_selection_rejects_unequal_lr_sets_per_setting():
    table = _table(lambda task, L: 1e-3)
    del table[0]
    with pytest.raises(ValueError, match="learning rates"):
        rank_learning_rates(table)


def test_duplicate_rows_for_one_setting_lr_seed_are_rejected():
    table = _table(lambda task, L: 1e-3)
    with pytest.raises(ValueError, match="duplicate"):
        collapse_seeds(table + table[:1])


def test_collapse_seeds_takes_the_median_and_fails_if_any_seed_diverged():
    table = _table(lambda task, L: 1e-3, seeds=(0, 1, 2))
    first = [
        r
        for r in table
        if (r["task"], r["L"], r["phi"], r["rate_lambda"], r["lr"])
        == ("linear_teacher", 1, "tanh", 0.0, 1e-3)
    ]
    for row, value in zip(first, (1.0, 5.0, 2.0)):
        row["heldout_mse"] = value
    collapsed = collapse_seeds(table)
    assert len(collapsed) == len(table) // 3
    hit = [
        r
        for r in collapsed
        if r["lr"] == 1e-3
        and r["task"] == "linear_teacher"
        and r["L"] == 1
        and r["phi"] == "tanh"
        and r["rate_lambda"] == 0.0
    ]
    assert hit[0]["heldout_mse"] == 2.0 and not hit[0]["diverged"]
    first[0]["diverged"] = True
    again = collapse_seeds(table)
    hit = [
        r
        for r in again
        if r["lr"] == 1e-3
        and r["task"] == "linear_teacher"
        and r["L"] == 1
        and r["phi"] == "tanh"
        and r["rate_lambda"] == 0.0
    ]
    assert hit[0]["diverged"] and math.isnan(hit[0]["heldout_mse"])


# ---------------------------------------------------------------------------- loss curve
def _history(losses, n_epochs=None):
    epochs = list(range(len(losses)))
    return {"epoch": epochs, "loss": list(losses), "mse": list(losses)}


def test_loss_curve_stats_on_a_converged_and_an_unconverged_curve():
    converged = loss_curve_stats(_history([10, 5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    assert converged["max_spike"] == pytest.approx(1.0)
    assert converged["tail_drop"] == pytest.approx(0.0)
    assert not converged["diverged"]
    falling = loss_curve_stats(_history(np.linspace(10, 1, 21)))
    assert falling["tail_drop"] > 0.05


def test_loss_curve_stats_spike_after_the_first_quarter_only():
    early = loss_curve_stats(_history([1, 100, 50, 10, 5, 4, 3, 2, 1.5, 1.2, 1.1, 1.0]))
    late = loss_curve_stats(_history([10, 8, 6, 4, 3, 2, 1, 1, 6, 2, 1, 1]))
    assert early["max_spike"] == pytest.approx(1.0)  # the early blow-up is not counted
    assert late["max_spike"] == pytest.approx(6.0)  # 6 against the best so far (1)


def test_loss_curve_stats_flags_non_finite_loss():
    stats = loss_curve_stats(_history([1.0, 0.5, float("nan"), float("nan")]))
    assert stats["diverged"]
    assert math.isinf(stats["max_spike"])


# ---------------------------------------------------------------------------- csv
def test_csv_round_trip_keeps_types_and_nan(tmp_path):
    rows = _table(lambda task, L: 1e-3, nan_at={("linear_teacher", 5, 1e-3)})
    for r in rows:
        r |= {"train_mse": 0.5, "loss": 1.5, "max_spike": 1.0, "tail_drop": 0.0}
        r["mir_corrected"] = "0.1;0.2"
    path = tmp_path / "t.csv"
    write_rows_csv(rows, path)
    back = read_rows_csv(path)
    assert len(back) == len(rows)
    assert back[0]["L"] == 1 and isinstance(back[0]["lr"], float) and back[0]["diverged"] is False
    assert [b["diverged"] for b in back] == [r["diverged"] for r in rows]
    assert math.isnan(next(b["heldout_mse"] for b in back if b["diverged"]))
    assert back[0]["mir_corrected"] == "0.1;0.2"


# ---------------------------------------------------------------------------- held-out data
def _data_cfg(task, seed=0, P=24):
    fields = {
        "linear_teacher": {"D_in": 6, "n_latents": 2, "latent_range": 1.0},
        "object_action": {"D_in": 18, "grid_size": 3, "max_objects": 2},
        "what_where": {"D_in": 81, "n_motifs": 4},
    }[task]
    return DataConfig(task=task, P=P, seed=seed, **fields)


@pytest.mark.parametrize("task", ["linear_teacher", "object_action", "what_where"])
def test_heldout_dataset_has_the_training_shapes_and_is_deterministic(task):
    cfg = _data_cfg(task)
    train = make_dataset(cfg, torch.device("cpu"), torch.float32)
    a = heldout_dataset(cfg, train, 7, torch.device("cpu"), torch.float32)
    b = heldout_dataset(cfg, train, 7, torch.device("cpu"), torch.float32)
    c = heldout_dataset(cfg, train, 8, torch.device("cpu"), torch.float32)
    assert a.x.shape == train.x.shape and a.y.shape == train.y.shape
    assert torch.equal(a.x, b.x) and torch.equal(a.y, b.y)
    assert not torch.equal(a.x, train.x) and not torch.equal(a.x, c.x)


def test_heldout_linear_teacher_keeps_the_training_teacher():
    cfg = _data_cfg("linear_teacher")
    train = make_dataset(cfg, torch.device("cpu"), torch.float32)
    new = heldout_dataset(cfg, train, 7, torch.device("cpu"), torch.float32)
    z = torch.stack([new.factors[f"z{i}"] for i in range(2)], dim=1)
    assert torch.allclose(new.y, z @ train.extras["W_Y"].T)
    assert torch.allclose(new.x, z @ train.extras["W_X"].T)


def test_heldout_what_where_keeps_the_training_motifs():
    cfg = _data_cfg("what_where", P=200)
    train = make_dataset(cfg, torch.device("cpu"), torch.float32)
    new = heldout_dataset(cfg, train, 7, torch.device("cpu"), torch.float32)
    motifs = {tuple(m.tolist()) for m in train.extras["motifs"]}
    # every held-out image holds, in its macro-cell, one of the training motifs
    where, what = new.factors["where"], new.factors["what"]
    for i in range(20):
        r, c = divmod(int(where[i]), 3)
        patch = new.x[i].reshape(9, 9)[3 * r : 3 * r + 3, 3 * c : 3 * c + 3].reshape(-1)
        assert tuple(patch.tolist()) in motifs
        assert torch.equal(patch, train.extras["motifs"][what[i]])


def test_heldout_dataset_rejects_the_same_seed_and_unknown_tasks():
    cfg = _data_cfg("linear_teacher")
    train = make_dataset(cfg, torch.device("cpu"), torch.float32)
    with pytest.raises(ValueError, match="seed"):
        heldout_dataset(cfg, train, cfg.seed, torch.device("cpu"), torch.float32)
    pos = DataConfig(task="position", P=8, D_in=4)
    with pytest.raises(ValueError, match="task"):
        heldout_dataset(
            pos,
            make_dataset(pos, torch.device("cpu"), torch.float32),
            5,
            torch.device("cpu"),
            torch.float32,
        )


def _snapshot(root):
    return {
        p: (p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_evaluate_setting_scores_a_finished_setting_without_touching_its_cache(tmp_path):
    sweep = preset("tiny")
    save_sweep(sweep, tmp_path)
    index = next(i for i, s in enumerate(expand(sweep)) if s.task == "linear_teacher")
    run_setting(sweep, index, tmp_path, log=lambda s: None)
    before = _snapshot(tmp_path)
    row = evaluate_setting(tmp_path, sweep, expand(sweep)[index], held_out_seed=77)
    again = evaluate_setting(tmp_path, sweep, expand(sweep)[index], held_out_seed=77)
    assert _snapshot(tmp_path) == before
    assert row == again
    # plain Python types: a numpy float would be written to the CSV as 'np.float64(...)'
    assert {k: type(v) for k, v in row.items()} == COLUMNS
    assert math.isfinite(row["train_mse"]) and math.isfinite(row["heldout_mse"])
    assert not row["diverged"] and row["lr"] == sweep.lr and row["task"] == "linear_teacher"
    assert len(row["mir_corrected"].split(";")) == expand(sweep)[index].L
    assert cache_paths(setting_dir(tmp_path, expand(sweep)[index])).train.exists()


def test_evaluate_setting_fails_loudly_for_an_unfinished_setting(tmp_path):
    sweep = preset("tiny")
    save_sweep(sweep, tmp_path)
    with pytest.raises(FileNotFoundError):
        evaluate_setting(tmp_path, sweep, expand(sweep)[0], held_out_seed=77)


def test_an_overfitting_net_is_worse_on_held_out_data_than_on_its_training_data(tmp_path):
    # 8 training points of a discrete task, 128 units: the net memorises them, so the held-out
    # MSE is far above the training MSE.
    sweep = preset("tiny").model_copy(update={"P": 8, "N": 128, "n_epochs": 400, "lr": 1e-2})
    save_sweep(sweep, tmp_path)
    index = next(i for i, s in enumerate(expand(sweep)) if s.task == "object_action")
    run_setting(sweep, index, tmp_path, log=lambda s: None)
    row = evaluate_setting(tmp_path, sweep, expand(sweep)[index], held_out_seed=77)
    assert row["heldout_mse"] > 3 * row["train_mse"]


# ---------------------------------------------------------------------------- plotting
def test_figures_draw_from_a_table_and_mark_failures():
    from fct.lrplot import fig_gap_vs_lr, fig_heldout_vs_lr, plot_heldout_vs_lr

    rows = _table(lambda task, L: 1e-3, nan_at={("linear_teacher", 5, 1e-2)})
    fig = fig_heldout_vs_lr(rows, {("linear_teacher", 1): 1e-3})
    assert len(fig.axes) == len(TASKS) * 2
    ax = fig.axes[1]  # linear_teacher, L=5: one red cross per failed (phi, rate) at lr 1e-2
    crosses = [line for line in ax.lines if line.get_marker() == "x"]
    assert sum(len(line.get_xdata()) for line in crosses) == 4
    scores = rank_learning_rates(rows)
    assert len(fig_gap_vs_lr(scores, TASKS, (1, 5)).axes) == 2
    with pytest.raises(ValueError, match="no rows"):
        plot_heldout_vs_lr(fig.axes[0], rows, "what_where", 1)


def test_subset_indices_and_trial_names():
    from pathlib import Path

    from fct.lrtune import subset_indices, trial_root, trial_sweep

    sweep = preset("tiny")
    indices = subset_indices(sweep, ["what_where"], [1, 5])
    settings = [expand(sweep)[i] for i in indices]
    assert len(indices) == 8 and {s.L for s in settings} == {1, 5}
    assert {s.task for s in settings} == {"what_where"}
    with pytest.raises(ValueError, match="unknown"):
        subset_indices(sweep, ["nope"], [1])
    assert trial_root(Path("b"), 3e-4, 2) == Path("b/lr_0.0003_seed2")
    trial = trial_sweep(sweep, 3e-4, 2)
    assert trial.lr == 3e-4 and trial.seeds.data == trial.seeds.net == trial.seeds.train == 2


def test_report_markdown_names_the_best_lr_and_the_verdict():
    from fct.lrtune import report_markdown

    text = report_markdown(_table(lambda task, L: 1e-3))
    assert "best: 0.001" in text and "suffices: True" in text
    assert "task and depth ('linear_teacher', 1)" in text
    mixed = report_markdown(_table(lambda task, L: 1e-2 if L == 1 else 1e-3, penalty=0.5))
    assert "suffices: False" in mixed
    assert "depth (1,)" in mixed and "depth (5,)" in mixed
