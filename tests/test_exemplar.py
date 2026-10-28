"""The exemplar notebook's library layer: config assembly, cache layout, live refresh and the
cached train-then-solve path, on a tiny problem."""

import matplotlib.pyplot as plt
import pytest
import torch
from pydantic import ValidationError

from fct import exemplar
from fct.meanfield import OuterSnapshot, network_constants
from tests.test_draw_live import synthetic_history

SETTINGS = {
    "P": 8, "m": 4, "k": 2, "N": 16, "L": 1, "phi": "tanh", "g": 1.0, "data_seed": 1, "net_seed": 2,
    "weight_lambda": 1e-2, "rate_lambda": 1e-3, "optimizer": "sgld", "beta": 5.0, "lr": 1e-2,
    "n_epochs": 20, "train_seed": 3,
}  # fmt: skip
QUIET = {"log": lambda message: None}


def config(**overrides):
    return exemplar.exemplar_config(**{**SETTINGS, **overrides})


def test_the_config_carries_every_setting():
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
    assert (cfg.train.optimizer, cfg.train.beta) == ("sgld", 5.0)
    assert cfg.meanfield.beta is None  # follows train.beta
    assert cfg.meanfield.n_points == 8
    assert (cfg.device, cfg.dtype) == ("cpu", "float64")
    assert cfg.spatial is None


def test_network_constants_follow_the_stationary_law_of_the_loss():
    """The loss prior weight_lambda / (2 beta) ||theta||^2 makes exp(-beta L) put variance
    1 / weight_lambda on each weight; the likelihood precision beta N / (P d_out) gives the ridge
    weight_lambda P d_out / beta; the tilt is 2 beta lambda_r and the prefactor weight_lambda / 2."""
    c = network_constants(beta=5.0, weight_lambda=1e-2, rate_lambda=1e-3, n_points=8, d_out=4)
    assert c["prior_scale"] == pytest.approx(100.0)
    assert c["alpha"] == pytest.approx(0.064)
    assert c["rate_tilt"] == pytest.approx(1e-2)
    assert c["learning_prefactor"] == pytest.approx(0.005)
    assert c["learning_prefactor"] == pytest.approx(1 / (2 * c["prior_scale"]))
    assert c["alpha"] == pytest.approx(8 * 4 / (5.0 * c["prior_scale"]))  # P d_out / (beta s)


def _problem_run_meanfield_solves(cfg, tmp_path, monkeypatch):
    """The `_Problem` of the arguments `run_meanfield` hands to `solve` for `cfg`."""
    import fct.meanfield as mf

    seen = {}

    def spy(X, Y, L, lambda_w, lambda_r, phi, g, meanfield, **kwargs):
        seen["problem"] = mf._make_problem(X, Y, L, lambda_w, lambda_r, phi, g, meanfield)

    monkeypatch.setattr(mf, "solve", spy)
    run = exemplar.train_exemplar(cfg, tmp_path, **QUIET)
    mf.run_meanfield(cfg, run.dataset, run.net, tmp_path / "solve", log=lambda message: None)
    return seen["problem"]


@pytest.mark.parametrize("beta", [0.1, 1.0, 5.0, 1e3])
def test_run_meanfield_defaults_are_the_network_constants_under_sgld(beta, tmp_path, monkeypatch):
    cfg = config(n_epochs=1, beta=beta, weight_lambda=0.03)
    prob = _problem_run_meanfield_solves(cfg, tmp_path, monkeypatch)
    c = network_constants(beta, 0.03, 1e-3, n_points=8, d_out=4)
    assert (prob.prior_scale, prob.alpha) == pytest.approx((c["prior_scale"], c["alpha"]))
    assert (prob.rate_tilt, prob.lp) == pytest.approx((c["rate_tilt"], c["learning_prefactor"]))


def test_run_meanfield_defaults_use_beta_one_under_adam(tmp_path, monkeypatch):
    cfg = config(n_epochs=1, optimizer="adam", beta=5.0, weight_lambda=0.03)
    prob = _problem_run_meanfield_solves(cfg, tmp_path, monkeypatch)
    c = network_constants(1.0, 0.03, 1e-3, n_points=8, d_out=4)
    assert (prob.prior_scale, prob.alpha) == pytest.approx((c["prior_scale"], c["alpha"]))
    assert (prob.rate_tilt, prob.lp) == pytest.approx((c["rate_tilt"], c["learning_prefactor"]))


def test_explicit_meanfield_constants_win_over_the_network_ones(tmp_path, monkeypatch):
    cfg = config(n_epochs=1)
    meanfield = cfg.meanfield.model_copy(update={"prior_scale": 3.0, "rate_tilt": 0.25})
    cfg = cfg.model_copy(update={"meanfield": meanfield})
    prob = _problem_run_meanfield_solves(cfg, tmp_path, monkeypatch)
    assert (prob.prior_scale, prob.rate_tilt) == (3.0, 0.25)
    assert prob.lp == pytest.approx(0.005)  # unset constants still follow the network


# (beta, weight_lambda) with lr * weight_lambda / beta = 0.04, so the weights equilibrate in
# about a hundred steps per relaxation time and the 6000 epochs are many relaxation times.
@pytest.mark.parametrize("beta, weight_lambda", [(0.5, 0.4), (5.0, 4.0), (50.0, 40.0)])
def test_sgld_weights_in_data_free_directions_have_the_variance_prior_scale(
    tmp_path, beta, weight_lambda
):
    """The law the constants claim: x spans only k of the m features, so U v for v orthogonal to
    the data feels neither the data term nor the rate penalty, and under SGLD at beta its
    entries are N(0, 1 / weight_lambda) = N(0, prior_scale) for every beta. 64 units x 2 directions
    give about 12% sampling error; the tolerance is 30%."""
    cfg = config(
        N=64, P=12, m=4, k=2, beta=beta, weight_lambda=weight_lambda, lr=0.05, n_epochs=6000
    )
    run = exemplar.train_exemplar(cfg, tmp_path, **QUIET)
    _, _, directions = torch.linalg.svd(run.dataset.x.double())
    free = directions[cfg.data.n_latents :]  # right singular vectors outside the data's span
    variance = (run.net.U.detach().double() @ free.T).var(unbiased=False).item()
    assert variance == pytest.approx(1 / weight_lambda, rel=0.3)


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
    run = exemplar.train_exemplar(cfg, root, **QUIET)
    return root, cfg, run.dataset, run.net, run.history, run.cache_dir


def test_training_goes_through_the_cache(trained):
    root, cfg, _dataset, net, history, directory = trained
    assert directory == exemplar.exemplar_dirs(root, cfg).train
    assert (directory / "train" / "final.pt").exists() and (directory / "config.json").exists()
    assert "grad_norm" in history and len(history["epoch"]) == 20
    again = exemplar.train_exemplar(cfg, root, **QUIET)
    history_again = again.history
    for before, after in zip(net.parameters(), again.net.parameters(), strict=True):
        assert torch.equal(before, after)
    assert history_again["loss"] == history["loss"]


@pytest.mark.parametrize("sampler", ["hmc", "importance"])
def test_solving_shows_live_frames_then_the_final_ones_and_reloads_from_cache(trained, sampler):
    root, cfg, dataset, net, _, _ = trained
    cfg = exemplar.with_solver(
        cfg, sampler=sampler, n_outer=4, n_chains=16, warmup_trans=5, n_importance_samples=64,
        sample_dtype="float64", n_inner_max=3, log_int=1, save_int=2, min_ess_frac=0.0,
        tilt_tripwire_resid=1e9,
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
    lines = []
    again = exemplar.solve_exemplar(
        cfg, dataset, net, root, plot_every=2, show=show, log=lines.append
    )
    assert frames == [3]  # a loaded solve runs no steps; only the final figures are shown
    assert any("loaded finished solve" in line for line in lines)
    assert torch.equal(again.Phi, result.Phi)


def test_a_failing_solve_calls_on_failure_and_reraises(trained, monkeypatch):
    root, cfg, dataset, net, _, _ = trained

    def boom(*args, **kwargs):
        kwargs["log"]("warm-up line")
        raise RuntimeError("inadequate sampler")

    monkeypatch.setattr(exemplar, "run_meanfield", boom)
    screen, view = make_view(tail=3)
    with pytest.raises(RuntimeError, match="inadequate sampler"):
        exemplar.solve_exemplar(
            cfg, dataset, net, root, 2, view.show, log=view.log, on_failure=view.flush
        )
    # no frame was ever shown, yet the held log line reaches the screen
    assert screen.events == [("text", "warm-up line")]


def test_the_comparison_and_empirical_figures_have_a_panel_per_layer(trained):
    _root, cfg, dataset, net, _, _ = trained
    assert len(exemplar.empirical_kernel_figure(net, dataset, cfg).axes) >= cfg.net.L
    outputs = exemplar.example_outputs_of(net, dataset)
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


def summary_history():
    history = synthetic_history("importance", n_steps=3)
    history["phihat_fp_resid_stein"] = [[4.0, 2.0], [3.0, 1.0], [2.0, 1.0]]
    history["train_mse_theory"] = [0.5, 0.25, 0.125]
    history["train_mse_net"] = [0.2, 0.2, 0.2]
    history["hmc_ess_frac"] = [[0.5, 0.4], [0.3, 0.9], [0.8, 0.6]]
    history["align_net"] = [[0.1, 0.2], [0.3, 0.4], [0.9, 0.8]]
    history["picard_resid"] = [0.5, 0.25, 0.125]
    return history


def test_the_solve_summary_reports_what_was_measured():
    kernels = torch.stack([3 * torch.eye(5), 3 * torch.eye(5)])
    text = exemplar.solve_summary(summary_history(), "importance", kernels, torch.ones(2, 5, 5))
    assert "3 outer steps with the importance sampler" in text
    assert "4, 2 at the first step, 2, 1 at the last, so it fell" in text
    assert "ESS fraction 0.3 (outer step 1, layer 1); at the last step 0.8, 0.6" in text
    assert "solved kernels: largest eigenvalue 3, 3, mean diagonal 3, 3" in text
    assert "net kernels: largest eigenvalue 5, 5, mean diagonal 1, 1" in text
    assert "theory 0.125, trained net 0.2" in text
    assert "alignment with the net per layer: 0.900, 0.800" in text
    assert (
        "Stein target (its convergence measure): 0.5 at the first step, 0.125 at the last" in text
    )


def test_the_solve_summary_omits_the_target_residual_when_it_was_not_recorded():
    history = summary_history()
    history["picard_resid"] = [float("nan")] * 3  # inner_start="previous"
    assert "Stein target" not in exemplar.solve_summary(history, "hmc")


def test_the_solve_summary_states_a_rising_residual_as_rising():
    history = summary_history()
    history["phihat_fp_resid_stein"] = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    assert "so it rose" in exemplar.solve_summary(history, "hmc")


def test_the_solve_summary_without_a_net_omits_the_net_numbers():
    history = synthetic_history("hmc", n_steps=2, with_net=False)
    text = exemplar.solve_summary(history, "hmc")
    assert "net" not in text and "alignment" not in text and "kernels" not in text


def test_the_summary_ignores_nan_ess_entries_and_reports_the_step_number_not_the_index():
    history = summary_history()
    history["hmc_ess_frac"][0] = [float("nan"), 0.4]
    history["outer_step"] = [10, 11, 12]  # a resumed run need not start at 0
    text = exemplar.solve_summary(history, "hmc")
    assert "ESS fraction 0.3 (outer step 11, layer 1)" in text
