"""The phase_transition_exp notebook: its source, one test per change the PI asked for (on the
built cells, on the plotting functions the cells define, and on executed outputs), and end-to-end
executions on a tiny `phase_transition` experiment (CPU) with one trained and one never-run setting."""

import ast
import shutil
from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pytest
import torch
from nbclient.exceptions import CellExecutionError

from fct.config import DataConfig, NetConfig
from fct.data import make_dataset
from fct.exp import expand, run_setting, save_exp, setting_dir
from fct.exp_view import tuning_view
from fct.model import make_net
from fct.phase_view import (
    average_views,
    ensemble_example_outputs,
    ensemble_statistics,
    ensemble_views,
    layer_priors,
    load_phase_run,
    random_test_set,
    unit_order,
)
from fct.spatial import (
    EmbeddingSpec,
    KernelSpec,
    PopulationSpec,
    SpatialConfig,
    build_spaces,
)
from fct.tasks import latent_lattice_dataset
from notebooks.build_phase_transition_exp import build_notebook, function_source
from notebooks.build_spatial import execute_notebook, set_parameters
from tests.test_build_spatial import _code, _n_figures_in, _text_outputs
from tests.test_phase_view import _tiny_exp

CPU, F64 = torch.device("cpu"), torch.float64
# One figure per weight layer for the prior matrices, the spectra and the weight kernels (L = 1:
# two weight layers), then the loss curves, the example outputs, A twice, tuning of one member,
# the overlaps twice.
N_PRIOR_FIGURES = 2 + 2
N_FIGURES = N_PRIOR_FIGURES + 1 + 1 + 2 + 2 + 1 + 2
GUARDED_CELLS = 7  # trained-only code cells: loss, examples, kernels, A, tuning, overlaps, + loader
N_LATENTS = _tiny_exp().linear_teacher.n_latents
PARAMETERS = {"MEMBER": 1, "SORT_DIM": 1, "P_TEST": 40, "N_EIG": 3, "N_NEURONS": 4}


def _plotting_namespace() -> dict:
    """The functions the notebook's cells define, executed in a namespace with its imports."""
    ns: dict = {"np": np, "torch": torch, "plt": plt, "mpl": mpl}
    exec("from fct.analyse import r_squared", ns)  # noqa: S102 - the notebook's own import
    for cell in build_notebook().cells:
        if cell.cell_type != "code":
            continue
        for node in ast.parse(cell.source).body:
            if isinstance(node, ast.FunctionDef):
                exec(ast.unparse(node), ns)  # noqa: S102 - the notebook's own function
    return ns


@pytest.fixture(scope="module")
def draw():
    return _plotting_namespace()


def _random_spatial(d: int) -> SpatialConfig:
    pop = PopulationSpec(
        embedding=EmbeddingSpec(kind="random"), kernel=KernelSpec(length_scale=0.2)
    )
    return SpatialConfig(d=d, input=pop, hidden=pop, output=pop, min_rel_eig=1e-2, lambda_max=0.5)


@pytest.fixture(scope="module")
def spaces():
    return build_spaces(_random_spatial(2), m=8, N=6, L=1, n_out=8)


@pytest.fixture(scope="module")
def ensemble():
    data_cfg = DataConfig(task="linear_teacher", P=20, D_in=8, n_latents=3, seed=1)
    train = make_dataset(data_cfg, CPU, F64)
    net = make_net(NetConfig(N=6, L=1, g=1.0, phi="tanh", seed=3, B=3), 8, 8, CPU, F64)
    return net, train


def _visible(fig):
    return [ax for ax in fig.axes if ax.get_visible() and ax.get_label() != "<colorbar>"]


# ---------------------------------------------------------------- the source notebook


def test_source_notebook_validates_has_no_outputs_and_no_path():
    nb = build_notebook()
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert all(not c.outputs and c.execution_count is None for c in code)
    assert "/ceph" not in _code(nb) and "/nfs" not in _code(nb)
    (parameters,) = [c for c in nb.cells if "parameters" in c.metadata.get("tags", [])]
    for name in ("ROOT", "SETTING", "MEMBER", "SORT_DIM", "P_TEST", "N_EIG", "N_NEURONS", "DEVICE"):
        assert f"\n{name} = " in "\n" + parameters.source, name


def test_the_committed_notebook_is_what_the_builder_produces():
    path = Path(__file__).parents[1] / "notebooks" / "phase_transition_exp.ipynb"
    committed = nbformat.read(path, 4)
    assert [(c.cell_type, c.source) for c in committed.cells] == [
        (c.cell_type, c.source) for c in build_notebook().cells
    ]


def test_every_figure_cell_is_introduced_by_an_equation():
    nb = build_notebook()
    drawing = [
        i
        for i, c in enumerate(nb.cells)
        if c.cell_type == "code" and ("plt.subplots(" in c.source or "plt.figure(" in c.source)
    ]
    assert drawing
    for i in drawing:
        assert "$" in nb.cells[i - 1].source, nb.cells[i].source[:200]


# ---------------------------------------------------------------- change 3 and 4: dropped figures


def test_the_u_weight_plot_and_the_trial_and_pair_sections_are_gone():
    nb = build_notebook()
    code = _code(nb)
    text = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    for name in ("plot_weights_by_position", "plot_trial_latents", "plot_spatial_kernel_pairs"):
        assert name not in code
    for phrase in (
        "First-layer weights $U$, sorted by position",
        "trials whose kernels are drawn",
        "between pairs of trials",
    ):
        assert phrase not in text


# ---------------------------------------------------------------- change 1: the prior matrices


def test_prior_figure_shows_covariance_effective_covariance_and_penalty_side_by_side(draw, spaces):
    prior = layer_priors(spaces, 4.0, n_eig=3)[0]
    fig = draw["plot_prior_matrices"](prior, spaces[0], np.arange(6), np.arange(8), 1)
    axes = _visible(fig)
    assert len(axes) == 6  # pre and post of three matrices
    titles = [ax.get_title() for ax in axes]
    assert sum("covariance" in t and "effective" not in t for t in titles) == 2
    assert sum("effective covariance" in t for t in titles) == 2
    assert sum("penalty" in t for t in titles) == 2
    assert all("lambda_w" in t for t in titles)


def test_prior_matrices_are_drawn_in_the_given_unit_order(draw, spaces):
    prior = layer_priors(spaces, 1.0, n_eig=3)[0]
    order_post, order_pre = np.array([5, 4, 3, 2, 1, 0]), np.arange(8)[::-1].copy()
    fig = draw["plot_prior_matrices"](prior, spaces[0], order_post, order_pre, 0)
    first_row = next(ax for ax in _visible(fig) if "covariance" in ax.get_title())
    shown = first_row.images[0].get_array()
    expected = prior.covariance.pre.numpy()[np.ix_(order_pre, order_pre)]
    assert np.allclose(np.asarray(shown), expected)
    assert "coordinate 0" in first_row.get_xlabel()


# ---------------------------------------------------------------- change 2: spectra and the floor


@pytest.mark.parametrize("n_eig", [2, 4])
def test_spectra_are_truncated_to_n_eig_and_the_floor_is_drawn(draw, spaces, n_eig):
    prior = layer_priors(spaces, 4.0, n_eig=n_eig)[0]
    fig = draw["plot_prior_spectra"](prior, spaces[0])
    axes = _visible(fig)
    assert len(axes) == 4
    for ax in axes:
        # the spectrum curves have n_eig points; the floor line is a two-point axhline
        spectra = [line for line in ax.lines if len(line.get_xdata()) == n_eig]
        assert len(spectra) >= 1 and all(len(line.get_xdata()) in (2, n_eig) for line in ax.lines)
    # a horizontal line at the floor (covariance) and at the ceiling (penalty), per population
    covariance_axes, penalty_axes = axes[:2], axes[2:]
    floors = [prior.floors.pre.covariance_floor, prior.floors.post.covariance_floor]
    ceilings = [prior.floors.pre.penalty_ceiling, prior.floors.post.penalty_ceiling]
    for ax, floor in zip(covariance_axes, floors):
        assert any(np.allclose(np.unique(line.get_ydata()), floor) for line in ax.lines)
    for ax, ceiling in zip(penalty_axes, ceilings):
        assert any(np.allclose(np.unique(line.get_ydata()), ceiling) for line in ax.lines)


def test_the_penalty_spectrum_axis_says_it_starts_from_the_smallest_end(draw, spaces):
    prior = layer_priors(spaces, 1.0, n_eig=3)[0]
    axes = _visible(draw["plot_prior_spectra"](prior, spaces[0]))
    assert "smallest" in axes[2].get_xlabel() and "largest" in axes[0].get_xlabel()


# ---------------------------------------------------------------- change 7: loss mean and sem


def test_loss_curves_are_the_member_mean_with_a_sem_band(draw):
    epochs = np.array([0, 10, 20])
    members = np.array([[1.0, 3.0], [2.0, 4.0], [4.0, 8.0]])  # (epochs, B = 2)
    history = {"epoch": epochs, **{k: members * (i + 1) for i, k in enumerate(
        ("loss", "mse", "weight_l2", "rate_l2"))}}  # fmt: skip
    stats = ensemble_statistics(history)
    fig = draw["plot_ensemble_loss"](stats, "t")
    (ax,) = _visible(fig)
    assert len(ax.lines) == 4 and len(ax.collections) == 4  # a mean line and a band per term
    mean = members.mean(axis=1)
    assert np.allclose(ax.lines[0].get_ydata(), mean)
    band = ax.collections[0].get_paths()[0].vertices[:, 1]
    sem = members.std(axis=1, ddof=1) / np.sqrt(2)
    assert np.isclose(band.max(), (mean + sem).max()) and np.isclose(band.min(), (mean - sem).min())
    assert "mean" in ax.get_ylabel() and "sem" in ax.get_ylabel()


# ---------------------------------------------------------------- change 8: example outputs


def test_example_outputs_draw_every_member_in_its_own_palette_colour(draw, ensemble):
    net, train = ensemble
    picks = ensemble_example_outputs(net, train, [0, 3])
    everything = ensemble_example_outputs(net, train, list(range(len(train.x))))
    colours = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    fig = draw["plot_ensemble_outputs"](picks, everything, colours, "t", 4)
    drawn = {
        mpl.colors.to_hex(line.get_color())
        for ax in fig.axes
        for line in ax.lines
        if line.get_marker() == "D"
    }
    assert drawn == {mpl.colors.to_hex(colours[b]) for b in range(3)}
    scatter = {
        mpl.colors.to_hex(c.get_facecolor()[0])
        for ax in fig.axes
        for c in ax.collections
        if isinstance(c, mpl.collections.PathCollection)
    }
    assert {mpl.colors.to_hex(colours[b]) for b in range(3)} <= scatter


def test_more_members_than_palette_colours_raises(draw, ensemble):
    net, train = ensemble
    outputs = ensemble_example_outputs(net, train, [0, 1])
    with pytest.raises(ValueError, match="palette"):
        draw["plot_ensemble_outputs"](outputs, outputs, ["C0", "C1"], "t", 4)


# ---------------------------------------------------------------- change 9: weight kernels


def test_weight_kernels_show_the_prior_the_member_and_the_average(draw, spaces):
    prior = layer_priors(spaces, 1.0, n_eig=3)[0]
    pair = prior.covariance
    panels = [("prior", pair), ("member 1", pair), ("average", pair)]
    fig = draw["plot_weight_kernels"](panels, spaces[0], np.arange(6), np.arange(8), 0)
    axes = _visible(fig)
    assert len(axes) == 6
    titles = " ".join(ax.get_title() for ax in axes)
    assert "member 1" in titles and "average" in titles and "prior" in titles


# ---------------------------------------------------------------- change 10: tuning truncation


def _tuning_view(n_neurons, k=2, n_bins=4):
    rng = np.random.default_rng(0)
    from fct.exp_view import TuningView
    from fct.tuning import TuningCurves

    curves = TuningCurves(
        torch.tensor(rng.normal(size=(n_neurons, k, n_bins))),
        torch.tensor(np.abs(rng.normal(size=(n_neurons, k, n_bins)))),
        torch.tensor(rng.normal(size=(k, n_bins))),
        torch.full((k, n_bins), 10),
    )
    info = torch.tensor(np.abs(rng.normal(size=(n_neurons, k))))
    return TuningView(
        curves, rng.uniform(size=n_neurons), info, info, None, [f"z{f}" for f in range(k)]
    )


@pytest.mark.parametrize("shown", [3, 6])
def test_tuning_draws_only_the_neurons_it_is_given_in_that_order(draw, shown):
    view = _tuning_view(6)
    positions = torch.rand(6, 2, dtype=F64)
    order = unit_order(positions, 1)[:shown]
    fig = draw["plot_tuning_by_order"](view, positions, order, 1, 4, 0.8)
    titles = [ax.get_title() for ax in fig.axes if ax.get_visible() and ax.get_title()]
    assert len(titles) == shown
    assert [int(t.split()[1].rstrip(",")) for t in titles] == order.tolist()
    assert all("coordinate 1" in t for t in titles)


# ---------------------------------------------------------------- change 11: unit order


def test_overlaps_are_shown_in_the_given_order_and_labelled_with_it(draw):
    C = torch.arange(16.0).reshape(4, 4)
    C = C + C.T
    order = np.array([2, 0, 3, 1])
    coordinate = np.array([0.4, 0.1, 0.3, 0.2])
    top = float(C.max())
    fig = draw["plot_overlaps"](C, C, order, coordinate, top, top, "t", 1)
    left = _visible(fig)[0]
    assert np.allclose(np.asarray(left.images[0].get_array()), C.numpy()[np.ix_(order, order)])
    assert "coordinate 1" in left.get_xlabel()


def _markdown(nb) -> str:
    """All the markdown, lower-cased with line breaks collapsed, for phrase searches."""
    return " ".join(
        " ".join(c.source.split()) for c in nb.cells if c.cell_type == "markdown"
    ).lower()


def test_the_text_says_a_sorted_coordinate_is_not_a_ring_position():
    nb = build_notebook()
    text = _markdown(nb)
    assert "not a ring position" in text and "general distance measures come later" in text
    assert "position_order(" not in _code(nb)  # no view sorts by ring position
    assert "unit_order(" in _code(nb)


# ---------------------------------------------------------------- change 5 and 6 and 8: the text


def test_text_says_the_overlap_average_is_the_mean_of_the_members_overlaps():
    text = _markdown(build_notebook())
    assert "mean over members of each member's own overlap" in text


def test_text_names_the_palette_limit_of_the_example_outputs():
    text = _markdown(build_notebook())
    assert "at most as many members as the palette has colours" in text


# ---------------------------------------------------------------- execution


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    """phase_transition tiny (spatial.d in {2, 3}, B = 2) with only d = 2 trained."""
    root = tmp_path_factory.mktemp("phase_nb")
    exp = _tiny_exp()
    save_exp(exp, root)
    run_setting(exp, 0, root, log=lambda _: None)
    return root


def _execute(root: Path, setting: dict, **parameters):
    overrides = {"SETTING": setting, **PARAMETERS, **parameters}
    return execute_notebook(build_notebook(), root, overrides, root_name="ROOT")


@pytest.fixture(scope="module")
def trained(root):
    return _execute(root, {"spatial.d": 2})


def test_executes_on_a_trained_setting_and_draws_every_figure(trained):
    text = _text_outputs(trained)
    assert "ERROR" not in text
    assert _n_figures_in(trained) == N_FIGURES
    assert "| 0 | 2 |" in text and "| 1 | 3 |" in text  # the settings table
    assert "done" in text and "missing" in text
    for i, cell in enumerate(trained.cells):
        if any("image/png" in o.get("data", {}) for o in cell.get("outputs", [])):
            assert "$" in trained.cells[i - 1].source  # an equation introduces every figure


def test_tuning_and_mir_use_a_fresh_random_test_set_of_p_test_points(trained):
    text = _text_outputs(trained)
    assert "random test set: 40 points" in text
    assert f"lattice test set: {3**N_LATENTS} points" in text
    assert "P = 40," in text  # the plug-in bias line of the tuning section uses P_TEST


def test_tuning_shows_the_first_n_neurons_in_sort_dim_order(trained):
    text = _text_outputs(trained)
    assert "first 4 of 6 neurons" in text and "coordinate 1 of d = 2" in text


def test_spectra_report_n_eig_of_each_population(trained):
    assert "3 of 6 eigenvalues" in _text_outputs(trained)


def test_a_and_overlap_figures_come_for_the_member_and_for_the_average(trained):
    sources = [c.source for c in trained.cells if c.cell_type == "code"]
    overlap = [s for s in sources if "plot_overlaps(" in s and "MEAN_VIEW" in s]
    kernels = [s for s in sources if "plot_data_kernels(" in s and "MEAN_VIEW" in s]
    assert overlap and kernels
    assert all("MEMBER_VIEW" in s for s in overlap + kernels)
    # member + average: two figures per hidden layer for the overlaps, two for A
    per_cell = {
        s: sum("image/png" in o.get("data", {}) for o in c.get("outputs", []))
        for c in trained.cells
        if c.cell_type == "code"
        for s in [c.source]
    }
    assert per_cell[overlap[0]] == 2 and per_cell[kernels[0]] == 2


def test_a_setting_that_is_not_trained_is_reported_and_only_its_prior_is_drawn(root):
    nb = _execute(root, {"spatial.d": 3}, SORT_DIM=2)
    text = _text_outputs(nb)
    assert "ERROR" not in text
    assert "is not trained** (status: missing)" in text
    assert text.count("is not trained") == GUARDED_CELLS
    assert _n_figures_in(nb) == N_PRIOR_FIGURES  # prior matrices and spectra need no net
    assert "SGLD needs < 1" in text


def test_sort_dim_outside_the_embedding_fails_loudly_in_the_notebook(root):
    with pytest.raises(CellExecutionError, match="sort_dim"):
        _execute(root, {"spatial.d": 2}, SORT_DIM=2)
    with pytest.raises(CellExecutionError, match="sort_dim"):
        _execute(root, {"spatial.d": 3}, SORT_DIM=3)  # also for a setting never run


def test_a_member_outside_the_ensemble_fails_loudly(root):
    with pytest.raises(CellExecutionError, match="MEMBER"):
        _execute(root, {"spatial.d": 2}, MEMBER=2)


def test_a_setting_outside_the_experiment_fails_loudly(root):
    with pytest.raises(CellExecutionError, match="spatial.d"):
        _execute(root, {"spatial.d": 5})


def test_a_partly_trained_setting_is_drawn_and_labelled_provisional(root, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    (setting_dir(copy, expand(_tiny_exp())[0]) / "train" / "final.pt").unlink()
    nb = _execute(copy, {"spatial.d": 2})
    text = _text_outputs(nb)
    assert "ERROR" not in text and "provisional" in text.lower()
    assert _n_figures_in(nb) == N_FIGURES


# ---------------------------------------------------------------- the review's additions
# Wiring: the notebook's own cells are run in this process, and the arrays in the figures they
# draw are compared with `fct.phase_view` computed independently here.

# raw MIR: the corrected one is all zero on a net this small, which would make the checks vacuous
DEFAULTS = {"n_bins": 8, "kind": "raw", "n_shuffles": 5}
MEMBER, SORT = PARAMETERS["MEMBER"], PARAMETERS["SORT_DIM"]


class Session:
    """The namespace after running every code cell of the notebook, and the figures each drew."""

    def __init__(self, nb, root, setting, **parameters):
        values = {
            "ROOT": str(root), "SETTING": setting, "MIR_KIND": "raw", **PARAMETERS, **parameters
        }  # fmt: skip
        self.ns: dict = {}
        self.figures: list[tuple[str, mpl.figure.Figure]] = []
        plt.close("all")
        for cell in set_parameters(nb, values).cells:
            if cell.cell_type != "code":
                continue
            exec(compile(cell.source, f"<{cell.id}>", "exec"), self.ns)  # noqa: S102
            self.figures += [(cell.source, plt.figure(n)) for n in plt.get_fignums()]
            plt.close("all")

    def figs(self, *needles: str):
        return [f for src, f in self.figures if all(n in src for n in needles)]


def _titled(fig, prefix):
    """The visible, non-colorbar axis whose title starts with `prefix`."""
    (ax,) = [a for a in _visible(fig) if a.get_title().startswith(prefix)]
    return ax


def _image(ax):
    return np.asarray(ax.images[0].get_array())


@pytest.fixture(scope="module")
def session(root):
    return Session(build_notebook(), root, {"spatial.d": 2})


@pytest.fixture(scope="module")
def reference(root):
    """phase_view computed independently of the notebook's cells."""
    run = load_phase_run(root, {"spatial.d": 2})
    cfg = run.setting.cfg
    test = random_test_set(cfg.data, run.dataset, PARAMETERS["P_TEST"]).dataset
    lattice = latent_lattice_dataset(cfg.data, run.dataset)
    views = ensemble_views(run.net, lattice, test, run.spaces, seed=cfg.train.seed, **DEFAULTS)
    pops = [run.spaces[0].pre, *(s.post for s in run.spaces)]
    orders = [unit_order(p.embedding.positions, SORT) for p in pops]
    return SimpleNamespace(
        run=run, cfg=cfg, test=test, lattice=lattice, views=views, mean=average_views(views),
        pops=pops, orders=orders,
    )  # fmt: skip


def _sorted(matrix, order):
    return np.asarray(matrix)[np.ix_(order, order)]


def test_the_fixture_distinguishes_the_sort_axes_and_the_input_from_the_output(reference):
    for pop in reference.pops:
        assert not np.array_equal(
            unit_order(pop.embedding.positions, 0), unit_order(pop.embedding.positions, SORT)
        )
    assert not np.array_equal(reference.orders[0], reference.orders[2])  # input vs output units


def test_wiring_the_notebook_orders_every_population_by_sort_dim(session, reference):
    for got, want in zip(session.ns["ORDERS"], reference.orders, strict=True):
        assert np.array_equal(got, want)


def test_wiring_overlaps_are_the_member_then_the_average_in_sort_dim_order(session, reference):
    member_fig, average_fig = session.figs("plot_overlaps(", "MEAN_VIEW")
    order = reference.orders[1]
    for fig, view, tag in [
        (member_fig, reference.views[MEMBER], f"member {MEMBER}"),
        (average_fig, reference.mean, "ensemble average"),
    ]:
        assert tag in fig.get_suptitle()
        info, rate = _visible(fig)
        assert np.allclose(_image(info), _sorted(view.info_overlap[0], order))
        assert np.allclose(_image(rate), _sorted(view.rate_overlap[0], order))
    assert not np.allclose(_image(_visible(member_fig)[0]), _image(_visible(average_fig)[0]))


def test_wiring_data_kernels_are_the_member_then_the_average(session, reference):
    member_fig, average_fig = session.figs("plot_data_kernels(", "MEAN_VIEW")
    z0 = reference.lattice.factors["z0"].numpy()
    order = np.argsort(z0, kind="stable")
    for fig, view in [(member_fig, reference.views[MEMBER]), (average_fig, reference.mean)]:
        _, a_axis = _visible(fig)  # the latent kernel B, then A of hidden layer 1
        assert np.allclose(_image(a_axis), _sorted(view.A[0], order))
    assert f"member {MEMBER}" in member_fig.get_suptitle()
    assert "average" in average_fig.get_suptitle()


def test_wiring_weight_kernels_are_the_prior_the_member_and_the_average(session, reference):
    figs = session.figs("plot_weight_kernels(", "MEAN_VIEW")
    priors = layer_priors(reference.run.spaces, reference.cfg.train.weight_lambda, 3)
    assert len(figs) == len(priors)
    for layer, fig in enumerate(figs):
        order = {"post": reference.orders[layer + 1], "pre": reference.orders[layer]}
        for which in ("post", "pre"):
            for prefix, pair in [
                ("effective prior", priors[layer].effective_covariance),
                (f"member {MEMBER}", reference.views[MEMBER].weight_kernels[layer]),
                (f"average of {reference.run.net.B} members", reference.mean.weight_kernels[layer]),
            ]:
                ax = _titled(fig, f"{prefix}: {which}")
                assert np.allclose(_image(ax), _sorted(getattr(pair, which), order[which]))


def test_wiring_prior_matrices_use_the_order_of_their_own_population(session, reference):
    priors = layer_priors(reference.run.spaces, reference.cfg.train.weight_lambda, 3)
    figs = session.figs("def plot_prior_matrices")
    assert len(figs) == len(priors)
    for layer, fig in enumerate(figs):
        order = {"post": reference.orders[layer + 1], "pre": reference.orders[layer]}
        for which in ("post", "pre"):
            for name, pair in [
                ("covariance", priors[layer].covariance),
                ("effective covariance (floored, as trained)", priors[layer].effective_covariance),
                ("penalty", priors[layer].penalty),
            ]:
                ax = _titled(fig, f"{name}, {which}")
                assert np.allclose(_image(ax), _sorted(getattr(pair, which), order[which]))


def test_wiring_tuning_is_of_the_chosen_member_on_the_random_test_set(session, reference):
    (fig,) = session.figs("plot_tuning_by_order(", "NET.member")
    view = tuning_view(
        reference.run.net.member(MEMBER), reference.test, 0,
        seed=reference.cfg.train.seed + MEMBER, **DEFAULTS,
    )  # fmt: skip
    shown = [a for a in _visible(fig) if a.get_title()]
    assert len(shown) == PARAMETERS["N_NEURONS"]
    positions = reference.pops[1].embedding.positions
    for k, ax in enumerate(shown):
        neuron = int(ax.get_title().split()[1].rstrip(","))
        assert neuron == reference.orders[1][k]
        assert f"{float(positions[neuron, SORT]):.2f}" in ax.get_title()
        assert np.allclose(ax.containers[0].lines[0].get_ydata(), view.curves.mean[neuron, 0])


# ---------------------------------------------------------------- the review's other changes


def test_example_outputs_show_a_subset_of_the_outputs_with_a_member_legend(draw, ensemble):
    net, train = ensemble
    picks = ensemble_example_outputs(net, train, [0, 3])
    everything = ensemble_example_outputs(net, train, list(range(len(train.x))))
    colours = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    fig = draw["plot_ensemble_outputs"](picks, everything, colours, "t", 4)
    first = fig.axes[0]
    diamonds = [ln for ln in first.lines if ln.get_marker() == "D"]
    assert len(diamonds) == 3 and all(len(ln.get_xdata()) == 4 for ln in diamonds)
    assert len(first.get_xticks()) == 4
    labels = [t.get_text() for t in first.get_legend().get_texts()]
    assert [f"member {b}" for b in range(3)] == [t for t in labels if t.startswith("member")]
    every = draw["plot_ensemble_outputs"](picks, everything, colours, "t", 99)  # all 8 outputs
    assert len(every.axes[0].get_xticks()) == 8


def test_sorted_matrices_are_antialiased_and_ticked_with_the_coordinate(draw, spaces):
    prior = layer_priors(spaces, 1.0, n_eig=3)[0]
    order_post, order_pre = unit_order(spaces[0].post.embedding.positions, 1), np.arange(8)
    fig = draw["plot_prior_matrices"](prior, spaces[0], order_post, order_pre, 1)
    fig.canvas.draw()
    ax = _titled(fig, "covariance, post")
    assert ax.images[0].get_interpolation() != "none"
    coordinate = spaces[0].post.embedding.positions[:, 1].numpy()
    ticks = [int(t) for t in ax.get_xticks()]
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        f"{coordinate[order_post[r]]:.2f}" for r in ticks
    ]
    assert [t.get_text() for t in ax.get_yticklabels()] == [
        t.get_text() for t in ax.get_xticklabels()
    ]


def test_the_overlap_figure_uses_the_same_tick_convention(draw):
    C = torch.rand(6, 6, dtype=F64)
    C = C + C.T
    order, coordinate = np.array([3, 1, 5, 0, 2, 4]), np.linspace(0, 1, 6)[::-1]
    fig = draw["plot_overlaps"](C, C, order, coordinate, float(C.max()), 1.0, "t", 0)
    fig.canvas.draw()
    ax = _visible(fig)[0]
    ticks = [int(t) for t in ax.get_xticks()]
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        f"{coordinate[order[r]]:.2f}" for r in ticks
    ]


@pytest.mark.parametrize("name", ["plot_prior_matrices", "plot_weight_kernels", "plot_overlaps"])
def test_an_order_that_is_not_a_permutation_of_the_units_raises(draw, spaces, name):
    prior = layer_priors(spaces, 1.0, n_eig=3)[0]
    good_post, good_pre = np.arange(6), np.arange(8)
    bad_post = np.array([0, 1, 2, 3, 4, 4])
    with pytest.raises(ValueError, match="permutation"):
        if name == "plot_prior_matrices":
            draw[name](prior, spaces[0], bad_post, good_pre, 0)
        elif name == "plot_weight_kernels":
            panels = [("a", prior.covariance), ("b", prior.covariance)]
            draw[name](panels, spaces[0], good_post, np.arange(1, 9), 0)
        else:
            C = torch.eye(6, dtype=F64)
            draw[name](C, C, bad_post, np.zeros(6), 1.0, 1.0, "t", 0)


def test_member_and_average_of_the_weight_kernels_share_a_symmetric_scale(draw, spaces):
    prior = layer_priors(spaces, 1.0, n_eig=3)[0]
    small = prior.covariance
    big = type(small)(small.post * 3, small.pre * 3)
    panels = [("prior", small), ("member", small), ("average", big)]
    fig = draw["plot_weight_kernels"](panels, spaces[0], np.arange(6), np.arange(8), 0)
    for which in ("post", "pre"):
        member, average = (_titled(fig, f"{p}: {which}").images[0] for p in ("member", "average"))
        assert member.get_clim() == average.get_clim()
        low, high = member.get_clim()
        assert low == -high


def test_member_and_average_overlaps_share_their_colour_scales(session):
    member_fig, average_fig = session.figs("plot_overlaps(", "MEAN_VIEW")
    for i in (0, 1):
        member, average = (_visible(f)[i].images[0] for f in (member_fig, average_fig))
        assert member.get_clim() == average.get_clim()
    low, high = _visible(member_fig)[1].images[0].get_clim()
    assert low == -high  # the rate overlap is signed: centred at 0


def test_spectra_figure_has_integer_ticks_and_counts_each_population(draw, spaces):
    prior = layer_priors(spaces, 1.0, n_eig=10)[0]  # 8 input units, 6 hidden: 8 and 6 drawn
    fig = draw["plot_prior_spectra"](prior, spaces[0])
    for ax in _visible(fig):
        assert all(float(t).is_integer() for t in ax.get_xticks())
    title = fig.get_suptitle()
    assert "pre 8" in title and "post 6" in title


def test_the_figure_titles_carry_the_status_tag_when_given(draw, spaces):
    prior = layer_priors(spaces, 1.0, n_eig=3)[0]
    tag = " [PROVISIONAL]"
    figs = [
        draw["plot_prior_matrices"](prior, spaces[0], np.arange(6), np.arange(8), 0, tag),
        draw["plot_prior_spectra"](prior, spaces[0], tag),
        draw["plot_weight_kernels"](
            [("a", prior.covariance)], spaces[0], np.arange(6), np.arange(8), 0, tag
        ),
    ]
    assert all(tag in f.get_suptitle() for f in figs)


def test_every_figure_of_a_partly_trained_run_is_tagged_provisional(root, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    (setting_dir(copy, expand(_tiny_exp())[0]) / "train" / "final.pt").unlink()
    partial = Session(build_notebook(), copy, {"spatial.d": 2})
    assert len(partial.figures) == N_FIGURES
    for _, fig in partial.figures:
        text = fig.get_suptitle() + " ".join(a.get_title() for a in fig.axes)
        assert "PROVISIONAL" in text and "epoch" in text


def test_a_finished_run_has_no_provisional_tag(session):
    for _, fig in session.figures:
        assert "PROVISIONAL" not in fig.get_suptitle() + " ".join(a.get_title() for a in fig.axes)


def test_tuning_neurons_can_be_picked_first_or_spread_along_the_order(draw):
    order = np.arange(10)[::-1]
    assert draw["pick_neurons"](order, 3, "first").tolist() == [9, 8, 7]
    assert draw["pick_neurons"](order, 3, "spread").tolist() == [9, 5, 0]
    assert draw["pick_neurons"](order, 50, "first").tolist() == order.tolist()
    with pytest.raises(ValueError, match="NEURON_PICK"):
        draw["pick_neurons"](order, 3, "random")


def test_the_spread_pick_is_reported_in_the_executed_notebook(root):
    nb = _execute(root, {"spatial.d": 2}, NEURON_PICK="spread")
    assert "4 evenly spaced of 6 neurons" in _text_outputs(nb)


def test_the_tuning_line_names_the_rate_bins_n_bins_not_b(trained):
    text = _text_outputs(trained)
    assert "P = 40, n_bins = 8" in text and "B = 8" not in text


def test_the_text_fits_this_exp():
    text = _markdown(build_notebook())
    assert "tanh rates" not in text and "at the defaults" not in text
    assert "log y-axis" in text and "log axis" not in text.replace("log y-axis", "")
    assert "$\\omega_w = \\sum_l" in text and "$\\omega_r = \\sum_{l, i, \\mu}" in text
    assert "only appears once" in text  # the floor shows once N_EIG passes the unfloored count


def test_function_source_returns_the_functions_in_the_order_asked_and_raises_on_a_miss():
    source = "def a():\n    return 1\n\n\ndef b():\n    return 2\n"
    got = function_source(source, "b", "a")
    assert got.index("def b") < got.index("def a")
    with pytest.raises(ValueError, match="c"):
        function_source(source, "a", "c")


def test_the_sorted_matrix_drawing_is_one_shared_helper():
    code = _code(build_notebook())
    assert code.count("def draw_sorted_matrix(") == 1
    assert code.count("draw_sorted_matrix(") >= 3  # its definition and the two figures
