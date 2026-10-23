"""The exemplar notebook's library layer: config assembly, cache layout, live refresh and the
cached train-then-solve path, on a tiny problem."""

import matplotlib.pyplot as plt
import pytest
import torch
from pydantic import ValidationError

from fct import exemplar
from fct.meanfield import OuterSnapshot
from tests.test_draw_live import synthetic_history

SETTINGS = {
    "P": 8, "m": 4, "k": 2, "N": 16, "L": 1, "phi": "tanh", "g": 1.0, "data_seed": 1, "net_seed": 2,
    "weight_lambda": 1e-2, "rate_lambda": 1e-3, "optimizer": "sgld", "beta": 5.0, "lr": 1e-2,
    "n_epochs": 20, "train_seed": 3, "log_int": 10, "save_int": 10,
}  # fmt: skip
QUIET = {"log": lambda message: None}


def config(**overrides):
    return exemplar.exemplar_config(**{**SETTINGS, **overrides})


def test_the_config_carries_every_setting_and_ties_both_betas():
    cfg = config()
    assert (cfg.data.task, cfg.data.P, cfg.data.D_in, cfg.data.n_latents) == (
        "linear_teacher",
        8,
        4,
        2,
    )
    assert (cfg.net.N, cfg.net.L, cfg.net.phi, cfg.net.g) == (16, 1, "tanh", 1.0)
    assert (cfg.data.seed, cfg.net.seed, cfg.train.seed) == (1, 2, 3)
    assert (cfg.train.weight_lambda, cfg.train.rate_lambda) == (1e-2, 1e-3)
    assert (cfg.train.optimizer, cfg.train.beta, cfg.meanfield.beta) == ("sgld", 5.0, 5.0)
    assert cfg.meanfield.n_points == 8
    assert (cfg.device, cfg.dtype) == ("cpu", "float64")


def test_an_impossible_setting_fails_at_config_time():
    with pytest.raises(ValidationError, match="n_latents"):
        config(k=4)  # m must exceed k
    with pytest.raises(ValidationError, match="optimizer"):
        config(optimizer="sgd")


def test_with_solver_sets_fields_and_rejects_unknown_ones():
    cfg = exemplar.with_solver(config(), sampler="importance", n_outer=7)
    assert (cfg.meanfield.solver.sampler, cfg.meanfield.solver.n_outer) == ("importance", 7)
    again = exemplar.with_solver(cfg, n_chains=3)
    assert again.meanfield.solver.sampler == "importance" and again.meanfield.solver.n_chains == 3
    with pytest.raises(ValidationError):
        exemplar.with_solver(cfg, n_outre=7)


def test_solver_settings_move_the_solve_directory_but_not_the_training_one(tmp_path):
    base = config()
    a = exemplar.exemplar_dirs(tmp_path, base)
    b = exemplar.exemplar_dirs(tmp_path, exemplar.with_solver(base, n_outer=5))
    assert a.train == b.train and a.solve != b.solve
    assert a.solve.parent == a.train and a.train.parent == tmp_path
    assert exemplar.exemplar_dirs(tmp_path, config(lr=2e-2)).train != a.train
    assert exemplar.exemplar_dirs(tmp_path, base) == a


@pytest.mark.parametrize("every, expected", [(1, [0, 1, 2, 3, 4]), (2, [0, 2, 4]), (9, [0])])
def test_the_plotter_refreshes_on_every_nth_outer_step(every, expected, monkeypatch):
    shown = []
    monkeypatch.setattr(exemplar, "live_figures", lambda snapshot: [snapshot.outer])
    refresh = exemplar.live_plotter(every, lambda figs: shown.extend(figs))
    P = torch.zeros(1, 2, 2)
    for step in range(5):
        refresh(OuterSnapshot(step, {}, P, P))
    assert shown == expected


def test_the_plotter_needs_a_positive_interval():
    with pytest.raises(ValueError, match="every"):
        exemplar.live_plotter(0, lambda figs: None)


def test_live_figures_are_kernels_solver_health_and_sampler_health():
    history = synthetic_history("hmc")
    P = torch.rand(2, 5, 5)
    figs = exemplar.live_figures(OuterSnapshot(3, history, P, P))
    assert len(figs) == 3
    assert figs[0].get_suptitle() == "outer step 3"
    assert any("residual" in ax.get_title() for ax in figs[1].axes)
    assert any("HMC" in ax.get_title() for ax in figs[2].axes)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("exemplar")
    cfg = config()
    dataset, net, history, directory = exemplar.train_exemplar(cfg, root, **QUIET)
    return root, cfg, dataset, net, history, directory


def test_training_goes_through_the_cache(trained):
    root, cfg, _dataset, net, history, directory = trained
    assert directory == exemplar.exemplar_dirs(root, cfg).train
    assert (directory / "train" / "final.pt").exists() and (directory / "config.json").exists()
    assert "grad_norm" in history and len(history["epoch"]) == 2
    _, again, history_again, _ = exemplar.train_exemplar(cfg, root, **QUIET)
    for before, after in zip(net.parameters(), again.parameters(), strict=True):
        assert torch.equal(before, after)
    assert history_again["loss"] == history["loss"]


@pytest.mark.parametrize("sampler", ["hmc", "importance"])
def test_solving_shows_live_frames_then_the_final_ones_and_reloads_from_cache(trained, sampler):
    root, cfg, dataset, net, _, _ = trained
    cfg = exemplar.with_solver(
        cfg, sampler=sampler, n_outer=4, n_chains=16, warmup_trans=5, n_importance_samples=64,
        sample_dtype="float64", n_inner_max=3, log_int=1, save_int=2, min_ess_frac=0.0,
    )  # fmt: skip
    frames = []

    def show(figures):
        frames.append(len(figures))
        plt.close("all")

    result = exemplar.solve_exemplar(cfg, dataset, net, root, plot_every=2, show=show, **QUIET)
    assert frames == [3, 3, 3]  # steps 0 and 2, then the final figures
    assert len(result.history["outer_step"]) == 4
    assert (
        len([a for a in exemplar.kernel_comparison(result, net, dataset, cfg).axes if a.images])
        == 2
    )

    frames.clear()
    again = exemplar.solve_exemplar(cfg, dataset, net, root, plot_every=2, show=show, **QUIET)
    assert frames == [3]  # a loaded solve runs no steps; only the final figures are shown
    assert torch.equal(again.Phi, result.Phi)


def test_the_comparison_and_empirical_figures_have_a_panel_per_layer(trained):
    _root, cfg, dataset, net, _, _ = trained
    assert len(exemplar.empirical_kernel_figure(net, dataset, cfg).axes) >= cfg.net.L
    outputs = exemplar.example_outputs_of(net, dataset, cfg.data.P)
    assert len(outputs.indices) == exemplar.N_EXAMPLE_OUTPUTS


# --------------------------------------------------------------------------- notebook edge
class FakeScreen:
    """Records what a LiveView does to the notebook output: clears, displayed figures, text."""

    def __init__(self):
        self.events = []

    def clear(self, wait):
        self.events.append(("clear", wait))

    def display(self, figure):
        self.events.append(("figure", figure))

    def text(self, line):
        self.events.append(("text", line))


def make_view(tail):
    screen = FakeScreen()
    return screen, exemplar.LiveView(
        tail, clear=screen.clear, display=screen.display, out=screen.text
    )


def test_a_refresh_clears_waiting_then_draws_the_figures_and_the_recent_log():
    screen, view = make_view(tail=2)
    for line in ("a", "b", "c"):
        view.log(line)
    figures = [plt.figure(), plt.figure()]
    view.show(figures)
    kinds = [kind for kind, _ in screen.events]
    assert kinds == ["clear", "figure", "figure", "text", "text"]
    assert screen.events[0] == ("clear", True)
    assert [value for kind, value in screen.events if kind == "text"] == ["b", "c"]
    assert not plt.get_fignums()  # the view closes the figures it was given


def test_log_lines_are_kept_across_refreshes_and_the_buffer_is_bounded():
    _screen, view = make_view(tail=3)
    for i in range(10):
        view.log(f"line {i}")
    assert view.lines == ["line 7", "line 8", "line 9"]


def test_the_cache_root_comes_from_the_environment_or_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.delenv(exemplar.CACHE_ROOT_ENV, raising=False)
    with pytest.raises(RuntimeError, match=exemplar.CACHE_ROOT_ENV):
        exemplar.cache_root_from_env()
    monkeypatch.setenv(exemplar.CACHE_ROOT_ENV, str(tmp_path))
    assert exemplar.cache_root_from_env() == tmp_path


def test_the_solve_summary_reports_the_final_numbers():
    history = synthetic_history("importance", n_steps=3)
    history["train_mse_theory"] = [0.5, 0.25, 0.125]
    history["train_mse_net"] = [0.2, 0.2, 0.2]
    history["hmc_ess_frac"] = [[0.5, 0.4], [0.3, 0.9], [0.8, 0.6]]
    history["align_net"] = [[0.1, 0.2], [0.3, 0.4], [0.9, 0.8]]
    text = exemplar.solve_summary(history, "importance")
    assert "3 outer steps" in text and "importance" in text
    assert "0.125" in text and "0.2" in text  # final theory and net train MSE
    assert "0.9" in text and "0.8" in text  # final alignment per layer
    assert "0.3" in text  # smallest ESS fraction over the run


def test_the_solve_summary_without_a_net_omits_the_net_numbers():
    history = synthetic_history("hmc", n_steps=2, with_net=False)
    text = exemplar.solve_summary(history, "hmc")
    assert "net" not in text and "alignment" not in text
