"""The identity activation "id": registry, linear forward pass, a tiny spatial experiment trained
through the CLI, the spatial experiment notebook on an `id` setting, lrplot, and the (iid)
solve stage."""

import math

import pytest
import torch

from fct.activations import PHI, get_phi, identity, known_tilt_class, phi_prime
from fct.cli import main
from fct.config import NetConfig
from fct.exp import (
    expand,
    load_exp,
    preset,
    run_setting,
    setting_config,
    setting_dir,
    with_axes,
)
from fct.lrplot import PHI_COLOURS, plot_metric_vs_lr
from fct.meanfield import load_meanfield
from fct.model import make_net
from fct.spatial_exp import load_exp_run, select_setting
from notebooks.build_spatial import execute_notebook
from notebooks.build_spatial_exp import build_notebook
from tests.report_helpers import QUIET
from tests.test_build_spatial import N_FIGURES, _n_figures_in, _text_outputs
from tests.test_exp_solve import TINY_MEANFIELD
from tests.test_lrtune import _table

CPU64 = (torch.device("cpu"), torch.float64)
PHI_AXIS = 'net.phi=["tanh","id"]'


# ---------------------------------------------------------------- registry and forward pass
def test_id_is_the_existing_identity_with_unit_derivative_and_linear_tilt():
    assert get_phi("id") is identity and "id" in PHI
    h = torch.randn(10, dtype=torch.float64)
    assert torch.equal(phi_prime(get_phi("id"))(h), torch.ones_like(h))
    assert known_tilt_class(get_phi("id")) == "linear"


def test_config_accepts_id_and_rejects_other_names():
    assert NetConfig(phi="id").phi == "id"
    with pytest.raises(ValueError):
        NetConfig(phi="linear")


def test_two_hidden_layer_id_net_is_the_matrix_product_and_linear_in_the_input():
    net = make_net(NetConfig(N=7, L=2, g=1.3, phi="id", seed=3), 5, 3, *CPU64)
    gen = torch.Generator().manual_seed(0)
    x1, x2 = (torch.randn(4, 5, dtype=torch.float64, generator=gen) for _ in range(2))

    def explicit(x):
        w = (net.V.T / net.N) @ (1.3 / math.sqrt(7) * net.J[0]) @ (net.U / math.sqrt(5))
        return x @ w.T

    assert torch.allclose(net(x1)[1], explicit(x1), atol=1e-12)
    a, b = 0.7, -1.9
    assert torch.allclose(net(a * x1 + b * x2)[1], a * net(x1)[1] + b * net(x2)[1], atol=1e-12)


# ---------------------------------------------------------------- spatial experiment through the
# CLI
@pytest.fixture(scope="module")
def trained_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("id_exp")
    # Adam, not the preset's SGLD: injected noise makes a six-epoch loss non-monotone.
    init = ["exp", "init", str(root), "--preset", "spatial_tiny", "--axis", PHI_AXIS]
    assert main([*init, "--set", "optimizer=adam", "--set", "n_epochs=40"]) == 0
    assert main(["exp", "run", str(root)]) == 0
    return root


def test_spatial_tiny_with_phi_over_tanh_and_id_trains_every_setting(trained_root):
    settings = expand(load_exp(trained_root))
    assert len(settings) == 4
    assert {s.axes["net.phi"] for s in settings} == {"tanh", "id"}
    for s in settings:
        run, _ = load_exp_run(select_setting(trained_root, s.axes))
        loss = run.history["loss"]
        assert all(math.isfinite(v) for v in loss), s.axes
        assert loss[-1] < loss[0], s.axes


def test_spatial_exp_notebook_runs_on_an_id_setting(trained_root):
    setting = {"net.phi": "id", "net.N": 8}
    nb = execute_notebook(build_notebook(), trained_root, {"SETTING": setting}, root_name="ROOT")
    text = _text_outputs(nb)
    assert "ERROR" not in text and "net.phi = id" in text
    assert _n_figures_in(nb) == N_FIGURES


def test_spatial_tiny_id_trains_with_the_presets_own_sgld(tmp_path):
    """The PI's path: the preset's SGLD and epochs untouched; the loss stays finite."""
    init = ["exp", "init", str(tmp_path), "--preset", "spatial_tiny", "--axis", 'net.phi=["id"]']
    assert main(init) == 0
    assert main(["exp", "run", str(tmp_path)]) == 0
    for s in expand(load_exp(tmp_path)):
        run, _ = load_exp_run(select_setting(tmp_path, s.axes))
        assert all(math.isfinite(v) for v in run.history["loss"]), s.axes


# ---------------------------------------------------------------- lrplot
def test_lrplot_draws_id_in_its_own_colour():
    rows = _table(lambda task, L: 1e-3)
    rows += [r | {"phi": "id"} for r in rows if r["phi"] == "tanh"]
    ax = plot_metric_vs_lr(rows, "linear_teacher", 1)
    assert {"id, r=0", "id, r=0.001"} <= {line.get_label() for line in ax.lines}
    assert len(set(PHI_COLOURS.values())) == len(PHI_COLOURS) == 3


# ---------------------------------------------------------------- solver on an iid id setting
def test_iid_id_solve_stage_completes_with_a_finite_residual(tmp_path):
    """Only completion and finiteness are checked here; the closed-form saddle of the linear case
    is checked in test_meanfield (`analytic_saddle`)."""
    base = preset("tiny").model_copy(update={"meanfield": TINY_MEANFIELD})
    exp = with_axes(base, {"net.phi": ["id"]})
    run_setting(exp, 0, tmp_path, stage="solve", **QUIET)
    s = expand(exp)[0]
    history = load_meanfield(setting_dir(tmp_path, s), setting_config(exp, s), "cpu")[2]
    assert all(math.isfinite(float(v)) for v in history["train_mse_theory"])
