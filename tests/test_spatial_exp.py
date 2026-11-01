"""Selecting one setting of a spatial experiment and loading its trained net with the held-out
latent-lattice test set (CPU only, on the tiny spatial preset)."""

import pytest
import torch

from fct.exp import preset, run_setting, save_exp, with_axes
from fct.spatial import latent_kernel
from fct.spatial_exp import (
    load_exp_run,
    select_setting,
    settings_table,
)


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    """spatial_tiny (net.N in {4, 8}) with only N = 8 trained."""
    root = tmp_path_factory.mktemp("spatial_exp")
    exp = preset("spatial_tiny")
    save_exp(exp, root)
    run_setting(exp, 1, root, log=lambda _: None)
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


def test_select_reports_a_setting_of_the_experiment_that_was_never_run_as_missing(root):
    chosen = select_setting(root, {"net.N": 4})
    assert chosen.status == "missing" and chosen.cfg.net.N == 4


def test_select_fails_loudly_on_a_value_that_is_not_in_the_experiment(root):
    with pytest.raises(LookupError, match="net.N"):
        select_setting(root, {"net.N": 5})


def test_select_fails_loudly_on_an_axis_that_is_not_in_the_experiment(root):
    with pytest.raises(LookupError, match="unknown axis"):
        select_setting(root, {"net.L": 1})


def test_settings_table_lists_every_setting_with_its_status(root):
    lines = settings_table(root).splitlines()
    assert "net.N" in lines[0] and "status" in lines[0]
    rows = lines[2:]
    assert len(rows) == 2
    assert "| 4 |" in rows[0] and "missing" in rows[0]
    assert "| 8 |" in rows[1] and "done" in rows[1]


def test_load_exp_run_gives_the_net_history_and_a_lattice_test_set(root):
    chosen = select_setting(root, {"net.N": 8})
    run, test = load_exp_run(chosen)
    k = chosen.cfg.data.n_latents
    assert len(test.x) == 3**k  # one point per sign pattern
    assert len(run.dataset.x) == chosen.cfg.data.P  # the training samples are kept apart
    assert run.net.U.shape[0] == 8
    assert len(run.history["loss"]) > 0
    assert run.cache_dir == chosen.directory
    # the test points are inputs and targets of the training set's own teacher
    z = torch.stack(list(test.factors.values()), dim=1)
    teacher = run.dataset.extras
    assert torch.allclose(test.x, z @ teacher["W_X"].T)
    assert torch.allclose(test.y, z @ teacher["W_Y"].T)


def test_load_exp_run_refuses_a_setting_that_is_not_trained(root):
    with pytest.raises(ValueError, match="not trained"):
        load_exp_run(select_setting(root, {"net.N": 4}))


def test_latent_kernel_is_the_gram_matrix_of_the_latents_over_k():
    factors = {"z0": torch.tensor([1.0, 0.0, -1.0]), "z1": torch.tensor([1.0, 1.0, 0.0])}
    expected = torch.tensor([[2.0, 1.0, -1.0], [1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]) / 2
    assert torch.allclose(latent_kernel(factors), expected)


def test_select_refuses_an_ambiguous_request_whatever_was_run(tmp_path):
    """net.N [8] x train.lr [1e-2, 5e-3]: with the first trained, {"net.N": 8} names both settings
    and must not resolve to either; naming the lr picks one, run or not."""
    exp = preset("spatial_tiny")
    exp = with_axes(exp, {"net.N": [8], "train.lr": [1e-2, 5e-3]})
    save_exp(exp, tmp_path)
    run_setting(exp, 0, tmp_path, log=lambda _: None)
    for values in ({"net.N": 8}, {}):
        with pytest.raises(LookupError, match="2 settings"):
            select_setting(tmp_path, values)
    assert select_setting(tmp_path, {"train.lr": 1e-2}).status == "done"
    assert select_setting(tmp_path, {"train.lr": 5e-3}).status == "missing"


def test_select_refuses_a_device_name_that_is_not_one(root):
    with pytest.raises(ValueError, match="device"):
        select_setting(root, {"net.N": 8}, device="tpu")
