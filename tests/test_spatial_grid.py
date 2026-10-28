"""Selecting one setting of a spatial sweep and loading its trained net with the held-out
latent-lattice test set (CPU only, on the tiny spatial preset)."""

import pytest
import torch

from fct.spatial import latent_kernel
from fct.spatial_grid import (
    TEST_POINTS_BASE,
    load_grid_run,
    prior_coeff,
    select_setting,
    settings_table,
)
from fct.spatial_view import make_config
from fct.sweep import preset, run_setting, save_sweep


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    """spatial_tiny (net.N in {4, 8}) with only N = 8 trained."""
    root = tmp_path_factory.mktemp("grid")
    sweep = preset("spatial_tiny")
    save_sweep(sweep, root)
    run_setting(sweep, 1, root, log=lambda _: None)
    return root


def test_select_finds_the_trained_setting_by_its_axis_values(root):
    chosen = select_setting(root, {"net.N": 8})
    assert chosen.status == "done"
    assert chosen.cfg.net.N == 8 and chosen.cfg.device == "cpu"
    assert chosen.directory.is_dir()


def test_select_snaps_a_float_written_as_an_int(root):
    assert (
        select_setting(root, {"net.N": 8.0}).directory
        == select_setting(root, {"net.N": 8}).directory
    )


def test_select_reports_a_setting_of_the_grid_that_was_never_run_as_missing(root):
    chosen = select_setting(root, {"net.N": 4})
    assert chosen.status == "missing" and chosen.cfg.net.N == 4


def test_select_fails_loudly_on_a_value_that_is_not_in_the_grid(root):
    with pytest.raises(LookupError, match="net.N"):
        select_setting(root, {"net.N": 5})


def test_select_fails_loudly_on_an_axis_that_is_not_in_the_grid(root):
    with pytest.raises(LookupError, match="unknown axis"):
        select_setting(root, {"net.L": 1})


def test_settings_table_lists_every_setting_with_its_status(root):
    lines = settings_table(root).splitlines()
    assert "net.N" in lines[0] and "status" in lines[0]
    rows = lines[2:]
    assert len(rows) == 2
    assert "| 4 |" in rows[0] and "missing" in rows[0]
    assert "| 8 |" in rows[1] and "done" in rows[1]


def test_load_grid_run_gives_the_net_history_and_a_lattice_test_set(root):
    chosen = select_setting(root, {"net.N": 8})
    run, test = load_grid_run(chosen)
    k = chosen.cfg.data.n_latents
    assert len(test.x) == TEST_POINTS_BASE**k  # one point per sign pattern
    assert len(run.dataset.x) == chosen.cfg.data.P  # the training samples are kept apart
    assert run.net.U.shape[0] == 8
    assert len(run.history["loss"]) > 0
    assert run.cache_dir == chosen.directory
    # the test points are inputs and targets of the training set's own teacher
    z = torch.stack(list(test.factors.values()), dim=1)
    teacher = run.dataset.extras
    assert torch.allclose(test.x, z @ teacher["W_X"].T)
    assert torch.allclose(test.y, z @ teacher["W_Y"].T)


def test_load_grid_run_refuses_a_setting_that_is_not_trained(root):
    with pytest.raises(ValueError, match="not trained"):
        load_grid_run(select_setting(root, {"net.N": 4}))


@pytest.mark.parametrize("optimizer", ["sgld", "adam"])
def test_prior_coeff_inverts_the_spatial_notebooks_weight_lambda(optimizer):
    """The spatial notebook sets weight_lambda = 2 beta PRIOR_COEFF (beta = 1 for Adam)."""
    beta, coeff = 1e3, 3e-3
    cfg = make_config(
        P=8, m=4, k=2, N=4, L=1, phi="tanh", g=1.0, data_seed=0, net_seed=0,
        weight_lambda=coeff * 2 * (beta if optimizer == "sgld" else 1.0),
        rate_lambda=0.0, optimizer=optimizer, beta=beta, lr=1e-3, n_epochs=2, train_seed=0,
        spatial=None, device="cpu",
    )  # fmt: skip
    assert prior_coeff(cfg.train) == pytest.approx(coeff)


def test_latent_kernel_is_the_gram_matrix_of_the_latents_over_k():
    factors = {"z0": torch.tensor([1.0, 0.0, -1.0]), "z1": torch.tensor([1.0, 1.0, 0.0])}
    expected = torch.tensor([[2.0, 1.0, -1.0], [1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]) / 2
    assert torch.allclose(latent_kernel(factors), expected)
