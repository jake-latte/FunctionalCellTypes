"""`fct.phase_view`: the computation layer behind the phase-transition notebook, on synthetic
inputs with known answers (hand-made embeddings and histories, small random ensembles) and on a
tiny `phase_transition` experiment trained here on the CPU."""

import ast
import dataclasses
import hashlib
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fct import phase_view
from fct.activations import get_phi
from fct.analyse import example_outputs
from fct.config import DataConfig, NetConfig
from fct.data import get_dataset, get_latent_lattice_dataset, make_dataset
from fct.exp import (
    expand,
    preset,
    run_setting,
    save_exp,
    setting_config,
    setting_dir,
    with_axes,
)
from fct.exp_view import tuning_view
from fct.meanfield import readout_activities
from fct.model import MLP, effective_weights, make_net
from fct.modularity import information_overlap, rate_overlap
from fct.phase_view import (
    ENSEMBLE_KEYS,
    ascending_spectrum,
    average_views,
    covariance_factors,
    descending_spectrum,
    effective_covariance_factors,
    ensemble_example_outputs,
    ensemble_statistics,
    ensemble_views,
    layer_priors,
    load_phase_run,
    member_view,
    penalty_factors,
    random_test_set,
    unit_order,
    weight_kernels,
)
from fct.spatial import (
    EmbeddingSpec,
    KernelSpec,
    PopulationSpec,
    SpatialConfig,
    build_spaces,
    data_kernel,
    weight_quadratic_form,
    whitening_factors,
)
from fct.spatial_view import data_kernels, empirical_kernels
from fct.tasks import Dataset, latent_lattice_dataset
from notebooks import build_spatial

CPU, F64 = torch.device("cpu"), torch.float64
TUNING = {"n_bins": 4, "kind": "corrected", "n_shuffles": 2, "seed": 0}


# --------------------------------------------------------------------------- shared fixtures


def _random_spatial(d: int) -> SpatialConfig:
    pop = PopulationSpec(
        embedding=EmbeddingSpec(kind="random"), kernel=KernelSpec(length_scale=0.2)
    )
    return SpatialConfig(d=d, input=pop, hidden=pop, output=pop, min_rel_eig=1e-2, lambda_max=0.5)


@pytest.fixture(scope="module")
def small():
    """A random 3-member ensemble (L = 2, N = 6, m = 8, k = 3) with its data and spaces."""
    data_cfg = DataConfig(task="linear_teacher", P=20, D_in=8, n_latents=3, seed=1)
    train = make_dataset(data_cfg, CPU, F64)
    net = make_net(NetConfig(N=6, L=2, g=1.0, phi="tanh", seed=3, B=3), 8, 8, CPU, F64)
    spaces = build_spaces(_random_spatial(2), m=8, N=6, L=2, n_out=8)
    return data_cfg, train, net, spaces


# --------------------------------------------------------------------------- position order


def test_unit_order_sorts_by_each_axis_of_a_hand_made_embedding():
    xy = torch.tensor([[0.5, 0.1], [0.0, 0.9], [0.2, 0.4], [0.9, 0.0]], dtype=F64)
    assert unit_order(xy, 0).tolist() == [1, 2, 0, 3]
    assert unit_order(xy, 1).tolist() == [3, 0, 2, 1]


def test_unit_order_breaks_ties_by_unit_index():
    xy = torch.tensor([[0.5, 0.0], [0.1, 0.0], [0.5, 0.0], [0.1, 0.0]], dtype=F64)
    assert unit_order(xy, 0).tolist() == [1, 3, 0, 2]
    assert unit_order(xy, 1).tolist() == [0, 1, 2, 3]  # all tied: the identity


@pytest.mark.parametrize("sort_dim", [2, 5, -1])
def test_unit_order_refuses_an_axis_outside_the_embedding(sort_dim):
    with pytest.raises(ValueError, match="sort_dim"):
        unit_order(torch.zeros(4, 2, dtype=F64), sort_dim)


# --------------------------------------------------------------------------- random test set


def test_random_test_set_is_new_samples_of_the_training_teacher(small):
    data_cfg, train, _, _ = small
    test = random_test_set(data_cfg, train, n_points=50, seed=11)
    assert test.seed == 11 and test.dataset.x.shape == (50, 8) and test.dataset.y.shape == (50, 8)
    assert test.dataset.x.dtype == train.x.dtype
    assert test.dataset.x.shape[0] != train.x.shape[0]
    assert not torch.equal(test.dataset.x[:20], train.x)
    # the training set's own x -> y map, fitted by least squares (x = Z W_X^T is rank k)
    fit = torch.linalg.lstsq(train.x, train.y, rcond=1e-10).solution
    assert torch.allclose(test.dataset.x @ fit, test.dataset.y, atol=1e-8)
    z = torch.stack([test.dataset.factors[f"z{i}"] for i in range(3)], dim=1)
    assert torch.allclose(test.dataset.y, z @ train.extras["W_Y"].T)


def test_random_test_set_same_seed_same_set_and_other_seed_other_set(small):
    data_cfg, train, _, _ = small
    a, b, c = (random_test_set(data_cfg, train, 30, seed=s) for s in (11, 11, 12))
    assert torch.equal(a.dataset.x, b.dataset.x)
    assert not torch.equal(a.dataset.x, c.dataset.x)


def test_random_test_set_default_seed_differs_from_the_data_seed_and_is_recorded(small):
    data_cfg, train, _, _ = small
    test = random_test_set(data_cfg, train, n_points=10)
    assert test.seed != data_cfg.seed


def test_random_test_set_refuses_the_training_seed_and_an_empty_set(small):
    data_cfg, train, _, _ = small
    with pytest.raises(ValueError, match="seed"):
        random_test_set(data_cfg, train, 10, seed=data_cfg.seed)
    with pytest.raises(ValueError, match="n_points"):
        random_test_set(data_cfg, train, 0, seed=5)


def test_random_test_set_leaves_the_training_cache_alone(tmp_path):
    exp = _tiny_exp()
    cfg = setting_config(exp, expand(exp)[0])
    train = get_dataset(tmp_path, cfg)
    path = tmp_path / "data" / "dataset.pt"
    before = hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns
    random_test_set(cfg.data, train, 40, seed=9)
    assert (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns) == before


# --------------------------------------------------------------------------- prior matrices

LAMBDA_W = 4.0


@pytest.mark.parametrize("layer", [0, 1, 2])
def test_covariance_factors_multiply_to_the_layer_covariance_over_lambda_w(small, layer):
    space = small[3][layer]
    f = covariance_factors(space, LAMBDA_W)
    expected = torch.kron(space.post.K, space.pre.K) / LAMBDA_W
    assert torch.allclose(torch.kron(f.post, f.pre), expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("layer", [0, 1, 2])
def test_penalty_factors_multiply_to_lambda_w_times_the_layer_penalty(small, layer):
    space = small[3][layer]
    f = penalty_factors(space, LAMBDA_W)
    expected = LAMBDA_W * torch.kron(space.post.M, space.pre.M)
    assert torch.allclose(torch.kron(f.post, f.pre), expected, rtol=1e-10, atol=1e-12)


def test_lambda_w_is_split_as_its_square_root_on_each_factor(small):
    space = small[3][0]
    assert torch.allclose(covariance_factors(space, LAMBDA_W).post, space.post.K / 2.0)
    assert torch.allclose(penalty_factors(space, LAMBDA_W).pre, 2.0 * space.pre.M)


def test_penalty_is_the_matrix_the_loss_weights_up_to_two_beta(small):
    """lambda_w * quadratic form == 2 beta times the penalty term of the loss."""
    space = small[3][0]
    W = torch.randn(6, 8, dtype=F64)
    f = penalty_factors(space, LAMBDA_W)
    plotted = weight_quadratic_form(W, f.post, f.pre)
    assert torch.allclose(plotted, LAMBDA_W * weight_quadratic_form(W, space.post.M, space.pre.M))


def test_descending_spectrum_is_sorted_and_exactly_truncated():
    matrix = torch.diag(torch.tensor([1.0, 5.0, 3.0, 2.0], dtype=F64))
    assert descending_spectrum(matrix, 3).tolist() == pytest.approx([5.0, 3.0, 2.0])
    assert descending_spectrum(matrix, 4).tolist() == pytest.approx([5.0, 3.0, 2.0, 1.0])


def test_descending_spectrum_handles_n_eig_beyond_the_size_and_refuses_zero():
    matrix = torch.eye(3, dtype=F64)
    assert len(descending_spectrum(matrix, 10)) == 3
    with pytest.raises(ValueError, match="n_eig"):
        descending_spectrum(matrix, 0)


def test_layer_priors_give_every_layer_both_factor_pairs_and_their_spectra(small):
    _, _, _, spaces = small
    priors = layer_priors(spaces, LAMBDA_W, n_eig=4, whitened=False)
    assert [p.layer for p in priors] == [0, 1, 2]
    for p, space in zip(priors, spaces, strict=True):
        assert p.covariance.post.shape == (space.post.K.shape[0],) * 2
        assert len(p.covariance_spectra.post) == 4
        expected = torch.linalg.eigvalsh(covariance_factors(space, LAMBDA_W).pre).flip(0)[:4]
        assert torch.allclose(p.covariance_spectra.pre, expected)
        assert torch.all(p.penalty_spectra.post[:-1] <= p.penalty_spectra.post[1:])  # ascending


# --------------------------------------------------------------------------- ensemble statistics


def _synthetic_history(n_epochs=5, B=4, seed=0):
    rng = np.random.default_rng(seed)
    history = {"epoch": list(range(0, 10 * n_epochs, 10))}
    for key in ENSEMBLE_KEYS:
        history[key] = [list(rng.normal(size=B)) for _ in range(n_epochs)]
    history["grad_norm"] = [list(rng.normal(size=B)) for _ in range(n_epochs)]
    return history


def test_ensemble_statistics_match_numpy_mean_and_sem():
    history = _synthetic_history()
    stats = ensemble_statistics(history)
    assert stats.epoch.tolist() == history["epoch"]
    for key in ENSEMBLE_KEYS:
        values = np.array(history[key])
        assert np.allclose(stats.mean[key], values.mean(axis=1))
        assert np.allclose(stats.sem[key], values.std(axis=1, ddof=1) / np.sqrt(values.shape[1]))
    assert set(stats.mean) == set(ENSEMBLE_KEYS)


def test_ensemble_statistics_sem_of_one_member_is_nan_without_a_warning():
    history = _synthetic_history(B=1)
    with np.errstate(all="raise"):
        stats = ensemble_statistics(history)
    assert np.isnan(stats.sem["loss"]).all()
    assert np.allclose(stats.mean["loss"], np.array(history["loss"])[:, 0])


def test_ensemble_statistics_refuses_a_history_without_a_member_axis():
    history = {"epoch": [0, 1], **{key: [1.0, 2.0] for key in ENSEMBLE_KEYS}}
    with pytest.raises(ValueError, match="member"):
        ensemble_statistics(history)
    with pytest.raises(ValueError, match="loss"):
        ensemble_statistics({"epoch": [0]})


# --------------------------------------------------------------------------- per-member matrices


def _lattice(data_cfg, train):
    return latent_lattice_dataset(data_cfg, train)


def _notebook_member(net, lattice, test, spaces, seed=TUNING["seed"]):
    """What the spatial notebook computes for one `MLP`, written out cell by cell."""
    with torch.no_grad():
        H, _ = net(lattice.x.to(net.dtype))
    psis = readout_activities(H, net.phi)
    Ks = [spaces[l + 1].pre.K for l in range(len(psis))]
    A = torch.stack([data_kernel(psi.detach().cpu().to(K.dtype), K) for psi, K in zip(psis, Ks)])
    views = [tuning_view(net, test, layer, **{**TUNING, "seed": seed}) for layer in range(net.L)]
    info = torch.stack([information_overlap(v.member_mi) for v in views])
    rate = torch.stack([rate_overlap(v.curves.mean) for v in views])
    weights = [net.U.detach(), *(J.detach() for J in net.J), net.V.detach().T]
    kernels = [empirical_kernels(W) for W in weights]
    return A, info, rate, kernels


@pytest.fixture(scope="module")
def sets(small):
    data_cfg, train, _, _ = small
    return _lattice(data_cfg, train), random_test_set(data_cfg, train, 60, seed=7).dataset


def test_a_single_member_view_equals_the_notebook_computation(small, sets):
    _, _, net, spaces = small
    lattice, test = sets
    A, info, rate, kernels = _notebook_member(net.member(1), lattice, test, spaces)
    view = member_view(net.member(1), lattice, test, spaces, **TUNING)
    assert torch.allclose(view.A, A) and view.A.shape == (2, 27, 27)
    assert torch.allclose(view.info_overlap, info) and view.info_overlap.shape == (2, 6, 6)
    assert torch.allclose(view.rate_overlap, rate)
    assert len(view.weight_kernels) == 3
    for got, (post, pre) in zip(view.weight_kernels, kernels, strict=True):
        assert torch.allclose(got.post, post) and torch.allclose(got.pre, pre)


def test_ensemble_average_equals_the_mean_of_members_taken_one_at_a_time(small, sets):
    _, _, net, spaces = small
    lattice, test = sets
    per_member = [
        _notebook_member(net.member(b), lattice, test, spaces, seed=b) for b in range(net.B)
    ]
    views = ensemble_views(net, lattice, test, spaces, **TUNING)
    assert len(views) == net.B
    mean = average_views(views)
    for i, field in enumerate((mean.A, mean.info_overlap, mean.rate_overlap)):
        expected = torch.stack([m[i] for m in per_member]).mean(0)
        assert torch.allclose(field, expected)
    for l, got in enumerate(mean.weight_kernels):
        post = torch.stack([m[3][l][0] for m in per_member]).mean(0)
        pre = torch.stack([m[3][l][1] for m in per_member]).mean(0)
        assert torch.allclose(got.post, post) and torch.allclose(got.pre, pre)
    assert not torch.allclose(views[0].A, views[1].A)  # the members really differ


def test_average_views_refuses_nothing_to_average():
    with pytest.raises(ValueError, match="no views"):
        average_views([])


# --------------------------------------------------------------------------- example outputs


def test_ensemble_example_outputs_are_every_members_own_outputs(small, sets):
    _, _, net, _ = small
    _, test = sets
    picks = [3, 0, 17]
    examples = ensemble_example_outputs(net, test, picks)
    assert examples.y_pred.shape == (3, 3, 8)
    assert examples.indices.tolist() == picks
    assert torch.equal(examples.y_true, test.y[picks])
    for b in range(net.B):
        single = example_outputs(net.member(b), test, picks)
        assert torch.allclose(examples.y_pred[b], single.y_pred)


def test_ensemble_example_outputs_refuse_a_row_outside_the_dataset(small, sets):
    _, _, net, _ = small
    with pytest.raises(ValueError, match="rows"):
        ensemble_example_outputs(net, sets[1], [0, 999])


# --------------------------------------------------------------------------- the trained experiment


def _tiny_exp():
    """phase_transition at a few epochs and a small net: d in {2, 3} x lambda_max 0.5, B = 2."""
    base = preset("phase_transition").model_copy(
        update={
            "N": 6, "P": 16, "n_epochs": 4, "log_int": 2, "save_int": 2, "monitor_int": 2,
            "monitor_bins": 4, "monitor_shuffles": 2, "device": "cpu",
            "linear_teacher": preset("tiny").linear_teacher.model_copy(update={"n_features": 8}),
        }
    )  # fmt: skip
    return with_axes(base, {"spatial.d": [2, 3], "spatial.lambda_max": [0.5], "net.B": [2]})


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    """Setting 0 (d = 2) trained; setting 1 (d = 3) never run."""
    root = tmp_path_factory.mktemp("phase")
    exp = _tiny_exp()
    save_exp(exp, root)
    run_setting(exp, 0, root, log=lambda _: None)
    return root


def test_loader_gives_the_ensemble_history_spaces_and_training_set(root):
    run = load_phase_run(root, {"spatial.d": 2})
    cfg = run.setting.cfg
    assert run.setting.status == "done" and run.epoch == cfg.train.n_epochs
    assert run.net.B == 2 and run.net.N == 6
    assert np.shape(run.history["loss"])[1] == 2  # (logged epochs, members)
    assert len(run.spaces) == cfg.net.L + 1
    assert run.spaces[0].pre.embedding.positions.shape == (8, 2)  # input population, d = 2
    assert run.spaces[0].post.embedding.positions.shape == (6, 2)
    assert len(run.dataset.x) == cfg.data.P


def test_loader_reports_a_setting_never_run_without_writing_anything(root):
    run = load_phase_run(root, {"spatial.d": 3})
    assert run.setting.status == "missing"
    assert run.net is None and run.history is None and run.epoch == 0
    assert len(run.spaces) == 2 and len(run.dataset.x) == run.setting.cfg.data.P
    assert not (run.setting.directory / "data").exists()


def test_loader_reads_the_latest_checkpoint_of_a_partly_trained_setting(root, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    directory = setting_dir(copy, expand(_tiny_exp())[0])
    (directory / "train" / "final.pt").unlink()
    run = load_phase_run(copy, {"spatial.d": 2})
    assert run.setting.status == "partial"
    assert run.net.B == 2 and 0 < run.epoch <= run.setting.cfg.train.n_epochs
    assert len(run.history["loss"]) > 0


def test_loader_refuses_a_setting_that_is_not_an_ensemble(tmp_path):
    exp = with_axes(_tiny_exp(), {"spatial.d": [2], "spatial.lambda_max": [0.5], "net.B": [1]})
    save_exp(exp, tmp_path)
    run_setting(exp, 0, tmp_path, log=lambda _: None)
    with pytest.raises(ValueError, match="ensemble"):
        load_phase_run(tmp_path, {"spatial.d": 2})


def test_every_function_runs_on_the_trained_setting(root):
    run = load_phase_run(root, {"spatial.d": 2})
    cfg = run.setting.cfg
    positions = run.spaces[0].post.embedding.positions
    assert sorted(unit_order(positions, 1).tolist()) == list(range(6))

    test = random_test_set(cfg.data, run.dataset, n_points=40).dataset
    lattice = get_latent_lattice_dataset(run.setting.directory, cfg)
    views = ensemble_views(run.net, lattice, test, run.spaces, **TUNING)
    mean = average_views(views)
    assert mean.A.shape == (cfg.net.L, 3**cfg.data.n_latents, 3**cfg.data.n_latents)

    priors = layer_priors(run.spaces, cfg.train.weight_lambda, n_eig=3, whitened=False)
    assert len(priors) == cfg.net.L + 1 and len(priors[0].penalty_spectra.post) == 3

    stats = ensemble_statistics(run.history)
    assert np.isfinite(stats.mean["loss"]).all() and stats.epoch.tolist() == run.history["epoch"]

    examples = ensemble_example_outputs(run.net, test, [0, 1])
    assert examples.y_pred.shape == (2, 2, 8)


# --------------------------------------------------------------------------- review additions


def _floored_space():
    """A layer on 12 random points whose penalty floor bites (min_rel_eig 0.1)."""
    spatial = _random_spatial(1).model_copy(update={"min_rel_eig": 0.1})
    return build_spaces(spatial, m=12, N=12, L=1, n_out=12)[0]


def test_penalty_spectrum_holds_the_smallest_eigenvalues_ascending_not_the_flat_floor():
    space = _floored_space()
    assert space.post.report.n_floored > 1  # the largest M eigenvalues are all the floor
    prior = layer_priors([space], 1.0, n_eig=4, whitened=False)[0]
    expected = torch.linalg.eigvalsh(space.post.M)[:4]
    assert torch.allclose(prior.penalty_spectra.post, expected)
    assert torch.all(prior.penalty_spectra.post[:-1] <= prior.penalty_spectra.post[1:])
    # the smallest penalty eigenvalues are the reciprocals of K's largest, floored
    floor = prior.floors.post.covariance_floor
    top_k = torch.linalg.eigvalsh(space.post.K).flip(0)[:4].clamp(min=floor)
    assert torch.allclose(prior.penalty_spectra.post, 1 / top_k, rtol=1e-8)


def test_ascending_spectrum_is_sorted_truncated_and_refuses_zero():
    matrix = torch.diag(torch.tensor([4.0, 1.0, 3.0, 2.0], dtype=F64))
    assert ascending_spectrum(matrix, 2).tolist() == pytest.approx([1.0, 2.0])
    assert len(ascending_spectrum(matrix, 9)) == 4
    with pytest.raises(ValueError, match="n_eig"):
        ascending_spectrum(matrix, 0)


def test_layer_priors_report_the_floor_of_each_population():
    space = _floored_space()
    prior = layer_priors([space], LAMBDA_W, n_eig=3, whitened=False)[0]
    floor_value = 1 / torch.linalg.eigvalsh(space.post.M)[-1].item()  # M's top = 1 / floor
    assert prior.floors.post.n_floored == space.post.report.n_floored
    assert prior.floors.post.fraction_floored == space.post.report.fraction_floored
    assert prior.floors.post.covariance_floor == pytest.approx(floor_value / LAMBDA_W**0.5)
    assert prior.floors.post.penalty_ceiling == pytest.approx(LAMBDA_W**0.5 / floor_value)
    assert prior.floors.pre.n_floored == space.pre.report.n_floored


def test_effective_covariance_is_the_floored_inverse_penalty_the_nets_trained_under():
    space = _floored_space()
    f = effective_covariance_factors(space, LAMBDA_W, whitened=False)
    expected = torch.kron(space.post.M.inverse(), space.pre.M.inverse()) / LAMBDA_W
    assert torch.allclose(torch.kron(f.post, f.pre), expected, rtol=1e-8, atol=1e-12)
    # unlike K / lambda_w, its spectrum never falls below the floor
    prior = layer_priors([space], LAMBDA_W, n_eig=12, whitened=False)[0]
    floor = prior.floors.post.covariance_floor
    assert prior.effective_covariance_spectra.post.min() >= floor * (1 - 1e-9)
    assert prior.covariance_spectra.post.min() < floor
    spectrum = prior.effective_covariance_spectra.post
    assert torch.all(spectrum[:-1] >= spectrum[1:])


def test_a_whitened_run_trains_under_the_exact_kernel_not_the_floored_one():
    """Under the whitened parametrisation W = S xi S with S = K^{1/2} and xi ~ N(0, I /
    lambda_w), so the covariance trained under is K itself (S^2, nothing floored)."""
    space = _floored_space()
    assert space.post.report.n_floored > 0
    whitened = effective_covariance_factors(space, LAMBDA_W, whitened=True)
    root = LAMBDA_W**0.5
    for got, pop in ((whitened.post, space.post), (whitened.pre, space.pre)):
        assert torch.allclose(got, pop.S @ pop.S / root, rtol=0, atol=1e-12)
        assert torch.allclose(got, pop.K / root, rtol=0, atol=1e-12)
    penalty = effective_covariance_factors(space, LAMBDA_W, whitened=False)
    assert torch.allclose(penalty.post, space.post.M.inverse() / root, rtol=1e-10, atol=0)
    assert not torch.allclose(penalty.post, whitened.post, rtol=0, atol=1e-6)
    prior = layer_priors([space], LAMBDA_W, n_eig=12, whitened=True)[0]
    assert prior.whitened and not layer_priors([space], LAMBDA_W, 12, whitened=False)[0].whitened
    torch.testing.assert_close(prior.effective_covariance_spectra, prior.covariance_spectra)


def test_the_parametrisation_must_be_named():
    space = _floored_space()
    with pytest.raises(TypeError, match="whitened"):
        effective_covariance_factors(space, LAMBDA_W)
    with pytest.raises(TypeError, match="whitened"):
        layer_priors([space], LAMBDA_W, n_eig=3)


def test_covariance_factors_refuse_a_zero_lambda_w(small):
    with pytest.raises(ValueError, match="weight_lambda"):
        covariance_factors(small[3][0], 0.0)


# fixed-value oracles, computed by hand


def _identity_net(U, V, phi="id"):
    return MLP.from_weights(U, [], V, 1.0, get_phi(phi))


def test_data_kernel_of_an_identity_net_is_psi_psi_transpose_over_n_by_hand():
    U = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=F64)  # N = 3, D_in = 2
    net = _identity_net(U, torch.ones(3, 2, dtype=F64))
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]], dtype=F64)
    spaces = build_spaces(_random_spatial(1), m=2, N=3, L=1, n_out=2)
    hidden = dataclasses.replace(spaces[1].pre, K=torch.eye(3, dtype=F64))
    spaces[1] = dataclasses.replace(spaces[1], pre=hidden)
    # psi = x U^T / sqrt(2): rows (1, 0, 1), (0, 2, 1), (2, 4, 4), all over sqrt(2)
    expected = torch.tensor([[2, 1, 6], [1, 5, 12], [6, 12, 36]], dtype=F64) / 2 / 3
    assert torch.allclose(data_kernels(net, x, spaces)[0], expected)


def test_weight_kernels_of_a_non_symmetric_square_weight_by_hand():
    W = torch.tensor([[1.0, 2.0], [0.0, 1.0]], dtype=F64)
    net = _identity_net(W, torch.eye(2, dtype=F64))
    first = weight_kernels(net)[0]
    assert first.post.tolist() == [[2.5, 1.0], [1.0, 0.5]]  # W W^T / 2: rows of W
    assert first.pre.tolist() == [[0.5, 1.0], [1.0, 2.5]]  # W^T W / 2: columns of W
    readout = weight_kernels(net)[1]  # V is stored (pre, post): the weight is V.T = I
    assert readout.post.tolist() == [[0.5, 0.0], [0.0, 0.5]]


def test_weight_kernels_are_float64_whatever_the_net_dtype():
    ones = torch.ones(2, 2, dtype=torch.float32)
    net = MLP.from_weights(ones, [], ones.clone(), 1.0, get_phi("id"))
    assert all(k.post.dtype == torch.float64 for k in weight_kernels(net))


def test_weight_kernels_of_a_whitened_net_are_those_of_its_effective_weights():
    """Under the whitened prior `net.U` is xi; the kernels are of W = S xi S."""
    spaces = build_spaces(_random_spatial(1), m=2, N=3, L=1, n_out=2)
    factors = whitening_factors(spaces)
    net = make_net(NetConfig(N=3, L=1, phi="id", seed=0), 2, 2, CPU, F64, whitening=factors)
    first, readout = weight_kernels(net)
    W = {name: w.detach() for name, w in effective_weights(net).items()}
    assert torch.equal(first.post, empirical_kernels(W["U"])[0])
    assert torch.equal(readout.pre, empirical_kernels(W["V"].T)[1])
    assert not torch.allclose(first.post, empirical_kernels(net.U.detach())[0])


def test_overlaps_of_two_neurons_each_tuned_to_its_own_latent():
    """x = z (two independent latents), U = sqrt(2) I: neuron i is z_i. Each neuron carries
    about log B nats about its own latent and none about the other, and their tuning curves
    are ramps of different latents, so the off-diagonal overlaps vanish."""
    z = torch.rand(4000, 2, dtype=F64, generator=torch.Generator().manual_seed(0)) * 2 - 1
    data = Dataset(
        x=z, y=z, Q=None, period=None, factors={"z0": z[:, 0], "z1": z[:, 1]},
        groups={"z0": ["z0"], "z1": ["z1"]},
    )  # fmt: skip
    net = _identity_net(torch.eye(2, dtype=F64) * 2**0.5, torch.ones(2, 2, dtype=F64))
    spaces = build_spaces(_random_spatial(1), m=2, N=2, L=1, n_out=2)
    view = member_view(net, data, data, spaces, n_bins=4, kind="raw", n_shuffles=1, seed=0)
    info, rate = view.info_overlap[0], view.rate_overlap[0]
    assert info[0, 0] == pytest.approx((np.log(4) ** 2) / 2, rel=0.02)  # (1/k) sum_f I_0f^2
    assert info[0, 0] == pytest.approx(info[1, 1], rel=0.02)
    assert info[0, 1] < 0.01 * info[0, 0]
    assert abs(rate[0, 1]) < 0.1 * rate[0, 0] and rate[0, 0] > 0


def _cell_namespace(source: str, names: set[str], namespace: dict) -> dict:
    """Execute only the top-level `def`s and assignments of a notebook cell named in `names`."""
    keep = [
        node for node in ast.parse(source).body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in names)
    ]  # fmt: skip
    exec(compile(ast.Module(keep, []), "<cell>", "exec"), namespace)  # noqa: S102
    return namespace


def test_the_library_matches_the_notebook_cells_it_replaces(small, sets):
    """The data kernels and overlaps are computed in `build_spatial.py` cells; pin the
    library to those cells' own code so a change in either is caught."""
    _, _, net, spaces = small
    lattice, test = sets
    mlp = net.member(0)
    ns = {"np": np, "torch": torch, "readout_activities": readout_activities,
          "data_kernel": data_kernel}  # fmt: skip
    cell = _cell_namespace(build_spatial.DATA_KERNEL, {"data_kernels"}, ns)
    H = mlp(lattice.x)[0].detach()
    assert torch.equal(
        cell["data_kernels"](H, mlp.phi, spaces), data_kernels(mlp, lattice.x, spaces)
    )

    ns = {"RUN": SimpleNamespace(net=mlp, dataset=test), "N_BINS": TUNING["n_bins"],
          "MIR_KIND": TUNING["kind"], "N_SHUFFLES": TUNING["n_shuffles"],
          "TRAIN_SEED": TUNING["seed"], "tuning_view": tuning_view,
          "information_overlap": information_overlap, "rate_overlap": rate_overlap}  # fmt: skip
    cell = _cell_namespace(build_spatial.OVERLAP, {"views", "info_overlaps", "rate_overlaps"}, ns)
    view = member_view(mlp, lattice, test, spaces, **TUNING)
    assert torch.equal(torch.stack(cell["info_overlaps"]), view.info_overlap)
    assert torch.equal(torch.stack(cell["rate_overlaps"]), view.rate_overlap)


def test_ensemble_views_give_each_member_its_own_shuffle_seed(small, sets, monkeypatch):
    _, _, net, spaces = small
    seen = []
    real = phase_view.member_view

    def spy(member, *args, **kwargs):
        seen.append(kwargs.get("seed", args[-1] if args else None))
        return real(member, *args, **kwargs)

    monkeypatch.setattr(phase_view, "member_view", spy)
    ensemble_views(net, sets[0], sets[1], spaces, n_bins=4, kind="raw", n_shuffles=1, seed=10)
    assert seen == [10, 11, 12]


def test_ensemble_views_refuse_a_single_network(small, sets):
    _, _, net, spaces = small
    with pytest.raises(TypeError, match="EnsembleMLP"):
        ensemble_views(net.member(0), sets[0], sets[1], spaces, 4, "raw", 1, 0)


@pytest.mark.parametrize(
    "update",
    [
        {"n_latents": 2},
        {"D_in": 10},
        {"P": 21},
        {"latent_min": -0.5},
        {"latent_max": 2.0},
        {"seed": 5},
        {"teacher": "orthonormal"},
    ],
)
def test_random_test_set_refuses_a_config_that_is_not_the_trainings(small, update):
    data_cfg, train, _, _ = small
    (field,) = update
    # The message names the field's value in the refused config, so the mismatch can be found.
    with pytest.raises(ValueError, match=f"does not match.*{field}="):
        random_test_set(data_cfg.model_copy(update=update), train, 10, seed=5)


def test_random_test_set_latents_are_uniform_on_the_latent_interval(small):
    data_cfg, train, _, _ = small
    test = random_test_set(data_cfg, train, 20000, seed=5).dataset
    z = torch.stack(list(test.factors.values()), dim=1)
    lo, hi = data_cfg.latent_min, data_cfg.latent_max
    assert z.min() >= lo and z.max() <= hi
    assert z.mean() == pytest.approx((lo + hi) / 2, abs=0.02)
    assert z.var(0).mean() == pytest.approx((hi - lo) ** 2 / 12, rel=0.03)


def test_ensemble_statistics_refuse_rows_that_do_not_match_the_epochs():
    history = _synthetic_history()
    history["loss"] = history["loss"][:-1]
    with pytest.raises(ValueError, match="epoch"):
        ensemble_statistics(history)


def test_ensemble_example_outputs_come_back_on_the_cpu(small, sets):
    _, _, net, _ = small
    examples = ensemble_example_outputs(net, sets[1], [0, 1])
    assert examples.y_pred.device.type == "cpu" and examples.x.device.type == "cpu"


def test_loader_generates_in_memory_when_a_done_setting_lacks_its_dataset_file(root, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    directory = setting_dir(copy, expand(_tiny_exp())[0])
    (directory / "data" / "dataset.pt").unlink()
    run = load_phase_run(copy, {"spatial.d": 2})
    assert run.setting.status == "done" and run.net is not None
    assert not (directory / "data" / "dataset.pt").exists()
    cached = load_phase_run(root, {"spatial.d": 2}).dataset
    assert torch.equal(run.dataset.x, cached.x)


def test_loader_reports_a_stale_setting_without_weights(root, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    directory = setting_dir(copy, expand(_tiny_exp())[0])
    (directory / "train" / "final.pt").write_bytes(b"not a checkpoint")
    run = load_phase_run(copy, {"spatial.d": 2})
    assert run.setting.status == "stale"
    assert run.net is None and run.history is None and run.epoch == 0


def test_floor_is_min_rel_eig_times_top_even_when_nothing_was_floored():
    """With a tiny floor nothing is raised to it, so M's top eigenvalue is 1 / lambda_min(K),
    not 1 / floor: the floor must come from min_rel_eig and K's top eigenvalue."""
    pop = PopulationSpec(
        embedding=EmbeddingSpec(kind="random"), kernel=KernelSpec(length_scale=0.05)
    )
    spatial = SpatialConfig(d=1, input=pop, hidden=pop, output=pop, min_rel_eig=1e-6)
    space = build_spaces(spatial, m=8, N=8, L=1, n_out=8)[0]
    assert space.post.report.n_floored == 0
    top = np.linalg.eigvalsh(space.post.K.numpy())[-1]  # numpy, independent of torch's path
    prior = layer_priors([space], LAMBDA_W, n_eig=3, whitened=False)[0]
    assert prior.floors.post.covariance_floor == pytest.approx(1e-6 * top / LAMBDA_W**0.5)
    assert prior.floors.post.penalty_ceiling == pytest.approx(LAMBDA_W**0.5 / (1e-6 * top))
