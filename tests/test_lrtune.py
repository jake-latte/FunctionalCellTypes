"""Learning-rate tuning: result table logic on hand-built tables, the evaluation of finished
settings, the runner and the command line on a tiny CPU experiment."""

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import fct.config
from fct import lrtune
from fct.cache import cache_paths, latest_checkpoint
from fct.exp import expand, preset, run_setting, save_exp, setting_dir, setting_status
from fct.lrplot import fig_metric_vs_lr, plot_metric_vs_lr
from fct.lrtune import (
    COLUMNS,
    TIMINGS_FILE,
    average_ranks,
    collapse_seeds,
    evaluate_setting,
    filter_rows,
    global_verdict,
    group_table_markdown,
    learned_settings,
    loss_curve_stats,
    main,
    rank_learning_rates,
    read_rows_csv,
    report_markdown,
    run_trials,
    select,
    select_by,
    setting_key,
    subset_indices,
    trial_exp,
    trial_root,
    tuning_template,
    write_rows_csv,
)

LRS = (1e-4, 1e-3, 1e-2)
TASKS = ("linear_teacher", "object_action")
WORST_SETTING = ("object_action", 5, "relu", 0.0)
BASE = {"linear_teacher": 0.5, "object_action": 20.0}  # different loss scales per task


def _depth_dependent_best(task, L):
    return 1e-2 if L == 1 else 1e-3


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
                    "n_epochs": 6000,
                    "heldout_mse": float("nan") if failed else mse,
                    "null_mse": 100 * BASE[task],
                    "diverged": failed,
                }
            )
    return rows


# ---------------------------------------------------------------------------- ranks
def test_average_ranks_tie_values_within_the_relative_tolerance():
    values = np.array([1.0, 1.0005, 2.0, 1.0009, 2.0015])
    assert average_ranks(values, rtol=1e-3).tolist() == [2.0, 2.0, 4.5, 2.0, 4.5]
    assert average_ranks(values).tolist() == [1.0, 2.0, 4.0, 3.0, 5.0]


def test_ranking_ignores_differences_below_the_resolution():
    table = _table(lambda task, L: 1e-3, penalty=0.0)
    for k, row in enumerate(table):
        row["heldout_mse"] *= 1 + 1e-6 * (k % 3)  # numerical noise only
    assert select(rank_learning_rates(table)).tied == LRS


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
    best = _depth_dependent_best
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
    best = _depth_dependent_best
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


def _cell(rows, task="linear_teacher", L=1, phi="tanh", rate=0.0, lr=1e-3):
    return [r for r in rows if setting_key(r) == (task, L, phi, rate) and r["lr"] == lr]


def test_collapse_seeds_takes_the_median_and_fails_if_any_seed_diverged():
    table = _table(lambda task, L: 1e-3, seeds=(0, 1, 2))
    first = _cell(table)
    for row, value in zip(first, (1.0, 5.0, 2.0)):
        row["heldout_mse"] = value
    collapsed = collapse_seeds(table)
    assert len(collapsed) == len(table) // 3
    (hit,) = _cell(collapsed)
    assert hit["heldout_mse"] == 2.0 and not hit["diverged"] and hit["n_seeds"] == 3
    assert "seed" not in hit  # a collapsed row summarises seeds and is not a result row
    first[0]["diverged"] = True
    (hit,) = _cell(collapse_seeds(table))
    assert hit["diverged"] and math.isnan(hit["heldout_mse"])


def test_rows_of_two_run_lengths_need_a_filter_first():
    table = _table(lambda task, L: 1e-3)
    longer = [r | {"n_epochs": 15000} for r in table]
    with pytest.raises(ValueError, match="run length"):
        collapse_seeds(table + longer)
    assert len(filter_rows(table + longer, n_epochs=15000)) == len(table)
    assert filter_rows(table, seeds=[1]) == []
    assert {r["lr"] for r in filter_rows(table, lrs=[1e-3])} == {1e-3}
    assert len(filter_rows(table, seeds=[0])) == len(table)


def test_learned_settings_judges_the_longest_run_not_a_transient():
    short = _table(lambda task, L: 1e-3)  # learns everywhere at 6000 epochs ...
    longer = [
        r
        | {
            "n_epochs": 15000,
            "heldout_mse": r["null_mse"] if r["task"] == "object_action" else r["heldout_mse"],
        }
        for r in short
    ]  # ... but object_action has collapsed to the zero net by 15000
    kept = learned_settings(short + longer)
    assert {r["task"] for r in kept} == {"linear_teacher"}
    assert {r["n_epochs"] for r in kept} == {15000}  # the converged rows are returned
    assert len(kept) == len(longer) // 2


def test_a_setting_where_only_a_diverged_lr_beats_the_zero_net_has_not_learned():
    table = _table(lambda task, L: 1e-3)
    for r in table:
        r["heldout_mse"] = r["null_mse"]
    first = _cell(table, lr=1e-3)[0]
    first["heldout_mse"], first["diverged"] = 0.0, True
    assert all(setting_key(r) != setting_key(first) for r in learned_settings(table))


# ---------------------------------------------------------------------------- guards
def test_average_ranks_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        average_ranks(np.array([]))


def test_loss_curve_stats_rejects_too_short_and_non_positive_curves():
    with pytest.raises(ValueError, match="at least 2"):
        loss_curve_stats(_history([]))
    with pytest.raises(ValueError, match="at least 2"):
        loss_curve_stats(_history([1.0]))
    with pytest.raises(ValueError, match="positive"):
        loss_curve_stats(_history([1.0, 0.5, 0.0, 0.0]))


def test_selection_rejects_empty_scores():
    with pytest.raises(ValueError, match="no scores"):
        select([])


# ---------------------------------------------------------------------------- loss curve
def _history(losses):
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
def _full_rows(**kw):
    rows = _table(lambda task, L: 1e-3, **kw)
    for r in rows:
        r |= {"train_mse": 0.5, "loss": 1.5, "max_spike": 1.0, "tail_drop": 0.0}
        r["mir_corrected"] = "0.1;0.2"
    return rows


def test_csv_round_trip_keeps_types_and_nan(tmp_path):
    rows = _full_rows(nan_at={("linear_teacher", 5, 1e-3)})
    path = tmp_path / "t.csv"
    write_rows_csv(rows, path)
    back = read_rows_csv(path)
    assert len(back) == len(rows)
    assert back[0]["L"] == 1 and isinstance(back[0]["lr"], float) and back[0]["diverged"] is False
    assert [b["diverged"] for b in back] == [r["diverged"] for r in rows]
    assert math.isnan(next(b["heldout_mse"] for b in back if b["diverged"]))
    assert back[0]["mir_corrected"] == "0.1;0.2"


def test_csv_with_a_wrong_header_or_a_bad_boolean_fails_loudly(tmp_path):
    path = tmp_path / "t.csv"
    write_rows_csv(_full_rows(), path)
    text = path.read_text()
    (tmp_path / "header.csv").write_text(text.replace("heldout_mse", "heldout"))
    with pytest.raises(ValueError, match="columns"):
        read_rows_csv(tmp_path / "header.csv")
    (tmp_path / "bool.csv").write_text(text.replace("False", "no", 1))
    with pytest.raises(ValueError, match="boolean"):
        read_rows_csv(tmp_path / "bool.csv")


# ---------------------------------------------------------------------------- evaluation
def _finished_tiny(tmp_path, task="linear_teacher", **update):
    exp = preset("tiny").model_copy(update=update)
    save_exp(exp, tmp_path)
    index = next(i for i, s in enumerate(expand(exp)) if s.task == task)
    run_setting(exp, index, tmp_path, log=lambda s: None)
    return exp, expand(exp)[index]


def _snapshot(root):
    return {
        p: (p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_evaluate_setting_scores_a_finished_setting_without_touching_its_cache(tmp_path):
    exp, setting = _finished_tiny(tmp_path)
    before = _snapshot(tmp_path)
    row = evaluate_setting(tmp_path, exp, setting, held_out_seed=77)
    again = evaluate_setting(tmp_path, exp, setting, held_out_seed=77)
    assert _snapshot(tmp_path) == before
    assert row == again
    # plain Python types: a numpy float would be written to the CSV as 'np.float64(...)'
    assert {k: type(v) for k, v in row.items()} == COLUMNS
    assert math.isfinite(row["train_mse"]) and math.isfinite(row["heldout_mse"])
    assert not row["diverged"] and row["lr"] == exp.lr and row["task"] == "linear_teacher"
    assert row["n_epochs"] == exp.n_epochs and row["null_mse"] > 0
    assert len(row["mir_corrected"].split(";")) == setting.axes["net.L"]
    assert cache_paths(setting_dir(tmp_path, setting)).train.exists()


def test_evaluate_setting_fails_loudly_for_an_unfinished_setting(tmp_path):
    exp = preset("tiny")
    save_exp(exp, tmp_path)
    with pytest.raises(FileNotFoundError):
        evaluate_setting(tmp_path, exp, expand(exp)[0], held_out_seed=77)


def test_evaluate_setting_fails_loudly_without_a_dataset_cache(tmp_path):
    exp, setting = _finished_tiny(tmp_path)
    (cache_paths(setting_dir(tmp_path, setting)).data / "dataset.pt").unlink()
    with pytest.raises(FileNotFoundError, match="refusing to generate"):
        evaluate_setting(tmp_path, exp, setting, held_out_seed=77)


def test_evaluate_setting_fails_loudly_if_the_history_has_no_monitor_entries(tmp_path, monkeypatch):
    exp, setting = _finished_tiny(tmp_path)
    real = lrtune.load_trained

    def without_monitor(*args):
        net, history = real(*args)
        return net, {k: v for k, v in history.items() if not k.startswith("monitor_")}

    monkeypatch.setattr(lrtune, "load_trained", without_monitor)
    with pytest.raises(ValueError, match="monitor_mir_corrected"):
        evaluate_setting(tmp_path, exp, setting, held_out_seed=77)


def test_an_overfitting_net_is_worse_on_held_out_data_than_on_its_training_data(tmp_path):
    # 8 training points of a discrete task, 128 units: the net memorises them, so the held-out
    # MSE is far above the training MSE.
    exp, setting = _finished_tiny(tmp_path, "object_action", P=8, N=128, n_epochs=400, lr=1e-2)
    row = evaluate_setting(tmp_path, exp, setting, held_out_seed=77)
    assert row["heldout_mse"] > 3 * row["train_mse"]


# ---------------------------------------------------------------------------- runner
def _tiny_template():
    return preset("tiny")  # CPU, seconds per setting


def test_subset_indices_and_trial_names():
    exp = preset("tiny")
    indices = subset_indices(exp, ["what_where"], [1, 5])
    settings = [expand(exp)[i] for i in indices]
    assert len(indices) == 8 and {s.axes["net.L"] for s in settings} == {1, 5}
    assert {s.task for s in settings} == {"what_where"}
    with pytest.raises(ValueError, match="unknown"):
        subset_indices(exp, ["nope"], [1])
    assert trial_root(Path("b"), 3e-4, 2) == Path("b/lr_0.0003_seed2")
    trial = trial_exp(exp, 3e-4, 2)
    assert trial.lr == 3e-4 and trial.seeds.data == trial.seeds.net == trial.seeds.train == 2


def test_tuning_template_overrides_sizes_and_the_intervals_follow_them():
    t = tuning_template(N=32, P=64, n_epochs=1000)
    assert (t.N, t.P, t.n_epochs, t.device) == (32, 64, 1000, "cuda")
    assert t.log_int == 10 and t.save_int == 100 and t.monitor_int == 250
    tiny = tuning_template(N=8, P=32, n_epochs=3)  # intervals never drop below 1
    assert tiny.log_int == tiny.save_int == tiny.monitor_int == 1


def test_run_trials_writes_one_exp_per_lr_and_seed_and_times_new_work_only(tmp_path):
    log = []
    run_trials(_tiny_template(), tmp_path, [0.01, 0.001], [0, 1], [0], log=log.append)
    assert log[0] == "device: cpu"
    for lr, seed in itertools.product([0.01, 0.001], [0, 1]):
        root = trial_root(tmp_path, lr, seed)
        exp = trial_exp(_tiny_template(), lr, seed)
        assert setting_status(root, exp, expand(exp)[0]) == "done"
    lines = (tmp_path / TIMINGS_FILE).read_text().splitlines()
    assert lines[0] == "lr,seed,index,seconds,device" and len(lines) == 5
    run_trials(_tiny_template(), tmp_path, [0.01, 0.001], [0, 1], [0], log=log.append)
    assert len((tmp_path / TIMINGS_FILE).read_text().splitlines()) == 5  # nothing new ran


def test_run_trials_resumes_an_interrupted_setting_from_its_checkpoint(tmp_path):
    template = _tiny_template()
    run_trials(template, tmp_path, [0.01], [0], [0], log=lambda s: None)
    exp, root = trial_exp(template, 0.01, 0), trial_root(tmp_path, 0.01, 0)
    train_dir = cache_paths(setting_dir(root, expand(exp)[0])).train
    final_before = torch.load(train_dir / "final.pt", weights_only=True)["net_state_dict"]
    (train_dir / "final.pt").unlink()
    assert latest_checkpoint(train_dir) is not None
    assert setting_status(root, exp, expand(exp)[0]) == "partial"
    run_trials(template, tmp_path, [0.01], [0], [0], log=lambda s: None)
    final_after = torch.load(train_dir / "final.pt", weights_only=True)["net_state_dict"]
    assert all(torch.equal(final_before[k], final_after[k]) for k in final_before)
    assert len((tmp_path / TIMINGS_FILE).read_text().splitlines()) == 3  # header + two runs


def test_a_trial_root_with_a_different_exp_is_refused(tmp_path):
    run_trials(_tiny_template(), tmp_path, [0.01], [0], [0], log=lambda s: None)
    changed = _tiny_template().model_copy(update={"N": 9})
    with pytest.raises(ValueError, match="differs"):
        run_trials(changed, tmp_path, [0.01], [0], [0], log=lambda s: None)


# ---------------------------------------------------------------------------- command line
def test_cli_init_run_collect_report_on_a_tiny_cpu_exp(tmp_path, capsys):
    base, table = tmp_path / "base", tmp_path / "rows.csv"
    sizes = ["--N", "8", "--P", "32", "--epochs", "6"]
    assert main(["init", str(base), *sizes, "--device", "cpu"]) == 0
    where = ["--lrs", "0.01", "0.001", "--tasks", "linear_teacher", "--layers", "1"]
    assert main(["run", str(base), *where]) == 0
    assert main(["collect", str(base), *where, "--csv", str(table)]) == 0
    rows = read_rows_csv(table)
    assert len(rows) == 2 * 4 and {r["lr"] for r in rows} == {0.01, 0.001}
    assert all(r["n_epochs"] == 6 for r in rows)
    capsys.readouterr()
    figs = tmp_path / "figs"
    assert main(["report", str(table), "--fig-dir", str(figs)]) == 0
    out = capsys.readouterr().out
    assert "suffices (median gap <= 10%, worst <= 2x)" in out  # default key: the loss
    assert (figs / "loss_vs_lr.png").exists()
    assert main(["report", str(table), "--key", "heldout_mse", "--n-epochs", "6"]) == 0
    assert "no verdict" in capsys.readouterr().out
    assert main(["report", str(table), "--key", "heldout_mse", "--max-ratio", "1.0001"]) == 0
    assert "no verdict" in capsys.readouterr().out


def test_cli_run_without_init_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="init"):
        main(["run", str(tmp_path), "--lrs", "0.01"])


def test_report_verdict_only_for_the_loss_or_the_learned_settings():
    table = _table(lambda task, L: 1e-3)
    for r in table:
        r["loss"] = r["heldout_mse"]
    assert "suffices" in report_markdown(table)  # the default key is the loss
    assert "no verdict" in report_markdown(table, key="heldout_mse")
    assert "suffices" in report_markdown(table, key="heldout_mse", learned_only=True)
    for r in table:
        r["heldout_mse"] = r["null_mse"]
    with pytest.raises(ValueError, match="none learned"):
        report_markdown(table, key="heldout_mse", learned_only=True)


def test_report_thresholds_are_parameters():
    table = _table(_depth_dependent_best, penalty=0.1)
    for r in table:
        r["loss"] = r["heldout_mse"]
    strict = report_markdown(table, max_median_gap=0.01)
    lax = report_markdown(table, max_median_gap=0.5, max_ratio=5.0)
    assert "suffices (median gap <= 1%, worst <= 2x): False" in strict
    assert "suffices (median gap <= 50%, worst <= 5x): True" in lax


def test_report_markdown_names_the_best_lr_and_the_verdict():
    table = _table(lambda task, L: 1e-3)
    for r in table:
        r["loss"] = r["heldout_mse"]
    text = report_markdown(table)
    assert "best: 0.001" in text and "suffices (median gap <= 10%, worst <= 2x): True" in text
    assert "task and depth ('linear_teacher', 1)" in text
    mixed = _table(_depth_dependent_best, penalty=0.5)
    for r in mixed:
        r["loss"] = r["heldout_mse"]
    text = report_markdown(mixed)
    assert "): False" in text
    assert "depth (1,)" in text and "depth (5,)" in text


def test_ranking_on_another_column_and_the_learned_filter():
    table = _table(lambda task, L: 1e-3)
    for row in table:  # the loss prefers the largest lr everywhere
        row["loss"] = 1.0 / row["lr"]
    assert select(rank_learning_rates(table, key="loss")).best == max(LRS)
    # one setting collapses at every lr: its held-out MSE equals the zero net's
    for row in table:
        if setting_key(row) == WORST_SETTING:
            row["heldout_mse"] = row["null_mse"]
    kept = learned_settings(table)
    assert len(kept) == len(table) - len(LRS)
    assert all(setting_key(r) != WORST_SETTING for r in kept)


def test_group_table_has_a_row_per_task_and_depth_with_the_best_lr():
    lines = group_table_markdown(rank_learning_rates(_table(_depth_dependent_best))).splitlines()
    assert len(lines) == 2 + len(TASKS) * 2
    row = next(line for line in lines if line.startswith("| linear_teacher | 1 "))
    assert row.endswith("| 0.01 |") and "1.0 / 0%" in row


# ---------------------------------------------------------------------------- plotting
def test_figures_draw_from_a_table_and_mark_failures():
    rows = _table(lambda task, L: 1e-3, nan_at={("linear_teacher", 5, 1e-2)})
    fig = fig_metric_vs_lr(rows, best_by_group={("linear_teacher", 1): 1e-3})
    assert len(fig.axes) == len(TASKS) * 2
    ax = fig.axes[1]  # linear_teacher, L=5: one red cross per failed (phi, rate) at lr 1e-2
    crosses = [line for line in ax.lines if line.get_marker() == "x"]
    assert sum(len(line.get_xdata()) for line in crosses) == 4
    with pytest.raises(ValueError, match="no rows"):
        plot_metric_vs_lr(rows, "what_where", 1, ax=fig.axes[0])


def test_plot_creates_its_own_axes_when_none_is_given():
    rows = _table(lambda task, L: 1e-3)
    ax = plot_metric_vs_lr(rows, "linear_teacher", 1)
    assert len(ax.get_legend_handles_labels()[1]) == 4


def test_an_unexpected_rate_lambda_is_drawn_and_an_unknown_phi_is_refused():
    rows = _table(lambda task, L: 1e-3)
    odd = [r | {"rate_lambda": 0.5} for r in rows if r["rate_lambda"] == 1e-3]
    ax = plot_metric_vs_lr([r for r in rows if r["rate_lambda"] == 0.0] + odd, "linear_teacher", 1)
    assert {line.get_label() for line in ax.lines} >= {"tanh, r=0.5", "relu, r=0.5"}
    bad = [r | {"phi": "gelu"} for r in rows]
    with pytest.raises(ValueError, match="gelu"):
        plot_metric_vs_lr(bad, "linear_teacher", 1)


def test_all_zero_values_get_a_non_degenerate_y_range():
    rows = _table(lambda task, L: 1e-3)
    for r in rows:
        r["heldout_mse"] = 0.0
    low, high = plot_metric_vs_lr(rows, "linear_teacher", 1).get_ylim()
    assert high > low


def test_evaluate_setting_says_stale_for_a_final_from_another_loss_version(tmp_path, monkeypatch):
    exp, setting = _finished_tiny(tmp_path)
    monkeypatch.setattr(fct.config, "LOSS_VERSION", fct.config.LOSS_VERSION + 1)
    with pytest.raises(FileNotFoundError, match="stale"):
        evaluate_setting(tmp_path, exp, setting, held_out_seed=77)


def test_a_template_with_a_non_ascii_description_round_trips_as_utf_8(tmp_path):
    base = tmp_path / "base"
    assert (
        main(["init", str(base), "--N", "8", "--P", "16", "--epochs", "6", "--device", "cpu"]) == 0
    )
    path = base / "template.json"
    tree = json.loads(path.read_text(encoding="utf-8"))
    path.write_bytes(
        json.dumps({**tree, "description": "\u03bb scan"}, ensure_ascii=False).encode()
    )
    assert lrtune._load_template(base).description == "\u03bb scan"
