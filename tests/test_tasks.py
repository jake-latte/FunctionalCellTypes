"""The linear-teacher, object-action and what-where task generators."""

import itertools
import json

import pytest
import torch
from pydantic import ValidationError

import fct.data as data_module
from fct.config import Config, DataConfig, load_config, save_config
from fct.data import embed, get_dataset, make_dataset, task_dims
from fct.tasks import optimal_actions

CPU = torch.device("cpu")
F64 = torch.float64

ACTIONS = ["stay", "up", "down", "left", "right"]
MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

SMALL = {
    "position": {"P": 50, "D_in": 8, "d": 2},
    "linear_teacher": {"P": 50, "D_in": 6, "n_latents": 3, "latent_min": -2.0, "latent_max": 2.0},
    "object_action": {"P": 50, "D_in": 18, "grid_size": 3, "max_objects": 2},
    "what_where": {"P": 50, "D_in": 81, "n_motifs": 5},
}
NEW_TASKS = ["linear_teacher", "object_action", "what_where"]


def _cfg(task: str, **kw) -> DataConfig:
    return DataConfig(task=task, **{**SMALL[task], **kw})


def _full(task: str, **kw) -> Config:
    return Config(data=_cfg(task, **kw), device="cpu", dtype="float64")


# --- every task ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", SMALL)
def test_dataset_is_deterministic_per_seed(task):
    a = make_dataset(_cfg(task, seed=1), CPU, F64)
    b = make_dataset(_cfg(task, seed=1), CPU, F64)
    c = make_dataset(_cfg(task, seed=2), CPU, F64)
    assert torch.equal(a.x, b.x) and torch.equal(a.y, b.y)
    assert not torch.equal(a.x, c.x)
    for name in a.factors:
        assert torch.equal(a.factors[name], b.factors[name])


@pytest.mark.parametrize("task", SMALL)
def test_task_dims_matches_the_dataset(task):
    cfg = _cfg(task)
    ds = make_dataset(cfg, CPU, F64)
    assert task_dims(cfg) == (ds.x.shape[1], ds.y.shape[1])
    assert ds.x.shape[0] == ds.y.shape[0] == cfg.P
    assert ds.x.dtype == ds.y.dtype == F64


VARIED = [
    ("position", {"d": 3, "D_in": 10}),
    ("linear_teacher", {"D_in": 4, "n_latents": 1}),
    ("linear_teacher", {"D_in": 9, "n_latents": 5}),
    ("object_action", {"grid_size": 2, "max_objects": 4, "D_in": 8}),
    ("object_action", {"grid_size": 4, "max_objects": 0, "D_in": 32}),
    ("what_where", {"n_motifs": 1}),
    ("what_where", {"n_motifs": 40}),
]


@pytest.mark.parametrize(("task", "kw"), VARIED)
def test_task_dims_equals_generated_shapes_across_configs(task, kw):
    cfg = _cfg(task, **kw)
    ds = make_dataset(cfg, CPU, F64)
    assert task_dims(cfg) == (ds.x.shape[1], ds.y.shape[1])


@pytest.mark.parametrize("task", SMALL)
def test_factors_and_groups_are_consistent(task):
    ds = make_dataset(_cfg(task), CPU, F64)
    assert all(f.shape == (50,) for f in ds.factors.values())
    assert all(f.dtype in (torch.int64, F64) for f in ds.factors.values())
    grouped = [name for members in ds.groups.values() for name in members]
    assert sorted(grouped) == sorted(ds.factors)


@pytest.mark.parametrize("task", NEW_TASKS)
def test_get_dataset_round_trips_factors_groups_extras(tmp_path, monkeypatch, task):
    cfg = _full(task)
    first = get_dataset(tmp_path, cfg)

    def boom(*args, **kwargs):
        raise AssertionError("regenerated despite cache")

    monkeypatch.setattr(data_module, "make_dataset", boom)
    second = get_dataset(tmp_path, cfg)
    assert torch.equal(first.x, second.x) and torch.equal(first.y, second.y)
    assert second.groups == first.groups
    assert set(second.factors) == set(first.factors)
    for name, value in first.factors.items():
        assert torch.equal(second.factors[name], value)
        assert second.factors[name].dtype == value.dtype
    assert set(second.extras) == set(first.extras)
    for name, value in first.extras.items():
        assert torch.equal(second.extras[name], value)


@pytest.mark.parametrize("task", ["object_action", "what_where"])
def test_wrong_D_in_names_the_value_to_set(task):
    expected = SMALL[task]["D_in"]
    with pytest.raises(ValueError, match=f"D_in={expected}"):
        make_dataset(_cfg(task, D_in=expected + 1), CPU, F64)


def test_non_fourier_tasks_may_have_odd_D_in():
    # The even-D_in rule is about the position task's cos/sin pairs only.
    assert make_dataset(_cfg("what_where"), CPU, F64).x.shape[1] == 81


@pytest.mark.parametrize(
    ("task", "kw", "match"),
    [
        ("object_action", {"grid_size": 5, "D_in": 64}, r"data\.D_in=50"),
        ("object_action", {"max_objects": 10}, "max_objects"),
        ("what_where", {"D_in": 64}, r"data\.D_in=81"),
        ("what_where", {"n_motifs": 512}, "511"),
    ],
)
def test_data_config_rejects_impossible_task_settings(task, kw, match):
    with pytest.raises(ValidationError, match=match):
        _cfg(task, **kw)


def test_data_config_rejects_bad_task_from_json(tmp_path):
    cfg = Config(data=_cfg("what_where"))
    path = tmp_path / "config.json"
    save_config(cfg, path)
    path.write_text(path.read_text().replace('"n_motifs": 5', '"n_motifs": 600'))
    with pytest.raises(ValidationError, match="511"):
        load_config(path)


def test_other_tasks_fields_are_ignored():
    a = make_dataset(_cfg("what_where", grid_size=2, n_latents=7), CPU, F64)
    b = make_dataset(_cfg("what_where"), CPU, F64)
    assert torch.equal(a.x, b.x)


def test_embed_is_position_only():
    ds = make_dataset(_cfg("what_where"), CPU, F64)
    with pytest.raises(ValueError, match="position"):
        embed(ds.y, ds, _cfg("what_where"))


# --- linear teacher -----------------------------------------------------------------------


def test_linear_teacher_latents_are_uniform_on_the_range():
    cfg = _cfg("linear_teacher", P=20_000, latent_min=-2.0, latent_max=2.0)
    ds = make_dataset(cfg, CPU, F64)
    z = torch.stack([ds.factors[f"z{i}"] for i in range(3)], dim=1)
    assert z.abs().max() <= 2.0
    assert z.mean(dim=0).abs().max() < 0.05
    assert torch.allclose(z.var(dim=0), torch.full((3,), 4.0 / 3, dtype=F64), rtol=0.05)


def test_linear_teacher_default_draws_are_bitwise_the_symmetric_formula():
    # The default interval [-1, 1] must reproduce `2 * rand - 1` on the same stream, so the
    # latents and both teacher maps (drawn after them) are unchanged for existing configs.
    cfg = DataConfig(task="linear_teacher", P=50, D_in=6, n_latents=3)
    ds = make_dataset(cfg, CPU, F64)
    gen = torch.Generator().manual_seed(cfg.seed)
    z = 2 * torch.rand((50, 3), dtype=F64, generator=gen) - 1
    w_x = torch.randn((6, 3), dtype=F64, generator=gen) / 3**0.5
    w_y = torch.randn((6, 3), dtype=F64, generator=gen) / 3**0.5
    assert torch.equal(torch.stack([ds.factors[f"z{i}"] for i in range(3)], dim=1), z)
    assert torch.equal(ds.extras["W_X"], w_x) and torch.equal(ds.extras["W_Y"], w_y)


def test_linear_teacher_latents_on_zero_one_have_mean_one_half():
    cfg = _cfg("linear_teacher", P=20_000, latent_min=0.0, latent_max=1.0)
    ds = make_dataset(cfg, CPU, F64)
    z = torch.stack([ds.factors[f"z{i}"] for i in range(3)], dim=1)
    assert z.min() >= 0.0 and z.max() <= 1.0
    assert torch.allclose(z.mean(dim=0), torch.full((3,), 0.5, dtype=F64), atol=0.01)


@pytest.mark.parametrize("bounds", [(1.0, 1.0), (1.0, -1.0)])
def test_latent_min_must_be_below_latent_max(bounds):
    with pytest.raises(ValueError, match="latent_min.*latent_max"):
        _cfg("linear_teacher", latent_min=bounds[0], latent_max=bounds[1])


@pytest.mark.parametrize("field", ["latent_min", "latent_max"])
def test_latent_bounds_must_be_finite(field):
    with pytest.raises(ValidationError, match=field):
        _cfg("linear_teacher", **{field: float("inf") if field == "latent_max" else -float("inf")})


def test_latent_range_is_rejected_naming_its_replacements(tmp_path):
    with pytest.raises(ValueError, match="latent_range.*latent_min.*latent_max"):
        DataConfig(task="linear_teacher", latent_range=1.0)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"data": {"task": "linear_teacher", "latent_range": 1.0}}))
    with pytest.raises(ValueError, match="latent_range.*latent_min.*latent_max"):
        load_config(path)


def test_linear_teacher_x_and_y_are_linear_in_the_latents():
    ds = make_dataset(_cfg("linear_teacher"), CPU, F64)
    z = torch.stack([ds.factors[f"z{i}"] for i in range(3)], dim=1)
    assert torch.allclose(ds.x, z @ ds.extras["W_X"].T)
    assert torch.allclose(ds.y, z @ ds.extras["W_Y"].T)
    assert ds.extras["W_X"].shape == ds.extras["W_Y"].shape == (6, 3)
    assert ds.groups == {f"z{i}": [f"z{i}"] for i in range(3)}


def test_linear_teacher_latents_are_recoverable_from_x_and_x_to_y_has_rank_k():
    ds = make_dataset(_cfg("linear_teacher", P=200), CPU, F64)
    z = torch.stack([ds.factors[f"z{i}"] for i in range(3)], dim=1)
    coef = torch.linalg.lstsq(ds.x, z).solution
    assert (ds.x @ coef - z).abs().max() < 1e-10
    xy = torch.linalg.lstsq(ds.x, ds.y).solution
    assert torch.linalg.matrix_rank(xy, atol=1e-8) == 3


def test_linear_teacher_weights_have_variance_one_over_k():
    ds = make_dataset(_cfg("linear_teacher", D_in=400, n_latents=4), CPU, F64)
    assert ds.extras["W_X"].var().item() == pytest.approx(0.25, rel=0.15)


def test_the_default_teacher_is_gaussian():
    assert DataConfig().teacher == "gaussian"


@pytest.mark.parametrize("key", ["W_X", "W_Y"])
def test_orthonormal_teacher_maps_have_orthonormal_columns(key):
    ds = make_dataset(_cfg("linear_teacher", D_in=10, teacher="orthonormal"), CPU, F64)
    w = ds.extras[key]
    assert w.shape == (10, 3)
    assert (w.T @ w - torch.eye(3, dtype=F64)).abs().max() < 1e-12


def test_orthonormal_teacher_orthonormalises_the_gaussian_draws_and_keeps_the_latents():
    gaussian = make_dataset(_cfg("linear_teacher"), CPU, F64)
    ortho = make_dataset(_cfg("linear_teacher", teacher="orthonormal"), CPU, F64)
    for name, z in gaussian.factors.items():
        assert torch.equal(ortho.factors[name], z)
    for key in ("W_X", "W_Y"):
        w, q = gaussian.extras[key], ortho.extras[key]
        # q is the Gram-Schmidt basis of w's columns: w = q r with r upper triangular, r_jj > 0
        r = q.T @ w
        assert torch.allclose(q @ r, w, atol=1e-12)
        assert torch.allclose(r, r.triu(), atol=1e-12) and (r.diagonal() > 0).all()
    z = torch.stack([ortho.factors[f"z{i}"] for i in range(3)], dim=1)
    assert torch.allclose(ortho.x, z @ ortho.extras["W_X"].T)
    assert torch.allclose(ortho.y, z @ ortho.extras["W_Y"].T)


@pytest.mark.parametrize("dtype", [torch.float32, F64])
@pytest.mark.parametrize("key", ["W_X", "W_Y"])
def test_each_orthonormal_column_points_the_way_of_its_gaussian_column(dtype, key):
    cfg = _cfg("linear_teacher", D_in=10, n_latents=4)
    w = make_dataset(cfg, CPU, dtype).extras[key]
    q = make_dataset(cfg.model_copy(update={"teacher": "orthonormal"}), CPU, dtype).extras[key]
    assert ((w * q).sum(dim=0) > 0).all()


def test_an_unknown_teacher_is_rejected_naming_the_field():
    with pytest.raises(ValidationError, match="teacher"):
        _cfg("linear_teacher", teacher="uniform")


@pytest.mark.parametrize("m", [3, 2])
def test_linear_teacher_needs_more_features_than_latents(m):
    with pytest.raises(ValidationError, match="D_in"):
        _cfg("linear_teacher", D_in=m, n_latents=3)


# --- object-action ------------------------------------------------------------------------


def _oracle_actions(agent: tuple[int, int], objects: set[tuple[int, int]]) -> list[float]:
    """Brute force: a move is optimal iff it lowers the minimum Manhattan distance by one."""

    def nearest(r, c):
        return min(abs(r - orow) + abs(c - ocol) for orow, ocol in objects)

    if not objects or agent in objects:
        return [1.0, 0.0, 0.0, 0.0, 0.0]
    now = nearest(*agent)
    out = [0.0]
    for dr, dc in MOVES.values():
        r, c = agent[0] + dr, agent[1] + dc
        inside = 0 <= r < 3 and 0 <= c < 3
        out.append(float(inside and nearest(r, c) == now - 1))
    return out


def _all_cases_3x3():
    agents, maps, expected = [], [], []
    for agent_cell in range(9):
        agent = divmod(agent_cell, 3)
        for mask in range(512):
            cells = {divmod(i, 3) for i in range(9) if mask >> i & 1}
            grid = torch.zeros(3, 3, dtype=torch.bool)
            for r, c in cells:
                grid[r, c] = True
            agents.append(agent)
            maps.append(grid)
            expected.append(_oracle_actions(agent, cells))
    return torch.tensor(agents), torch.stack(maps), torch.tensor(expected)


def test_optimal_actions_matches_brute_force_on_every_3x3_case():
    agents, maps, expected = _all_cases_3x3()
    got = optimal_actions(agents, maps)
    assert got.shape == (9 * 512, 5) and got.is_floating_point()
    assert torch.equal(got.double(), expected.double())


def _actions(agent, cells, G=5) -> dict[str, float]:
    grid = torch.zeros(1, G, G, dtype=torch.bool)
    for r, c in cells:
        grid[0, r, c] = True
    row = optimal_actions(torch.tensor([agent]), grid)[0].tolist()
    return dict(zip(ACTIONS, row, strict=True))


def test_no_objects_means_stay():
    assert _actions((2, 2), []) == {"stay": 1, "up": 0, "down": 0, "left": 0, "right": 0}


def test_agent_on_an_object_means_stay():
    assert _actions((2, 2), [(2, 2), (0, 0)])["stay"] == 1
    assert sum(_actions((2, 2), [(2, 2), (0, 0)]).values()) == 1


def test_diagonal_object_gives_two_moves():
    assert _actions((2, 2), [(0, 4)]) == {"stay": 0, "up": 1, "down": 0, "left": 0, "right": 1}


def test_equally_near_objects_in_opposite_directions_give_both_moves():
    assert _actions((2, 2), [(2, 0), (2, 4)]) == {
        "stay": 0,
        "up": 0,
        "down": 0,
        "left": 1,
        "right": 1,
    }


def test_object_action_x_decodes_to_the_factors():
    cfg = _cfg("object_action", P=300)
    ds = make_dataset(cfg, CPU, F64)
    assert ds.x.shape == (300, 18) and ds.y.shape == (300, 5)
    agent_onehot, objects = ds.x[:, :9], ds.x[:, 9:]
    assert torch.equal(agent_onehot.argmax(dim=1), ds.factors["position"])
    assert agent_onehot.sum(dim=1).eq(1).all()
    for i in range(9):
        assert torch.equal(objects[:, i].long(), ds.factors[f"object_{i}"])
    for j, name in enumerate(ACTIONS):
        assert torch.equal(ds.y[:, j].long(), ds.factors[f"action_{name}"])
    assert ds.factors["position"].dtype == torch.int64
    assert ds.groups == {
        "position": ["position"],
        "objects": [f"object_{i}" for i in range(9)],
        "actions": [f"action_{a}" for a in ACTIONS],
    }


def test_object_action_targets_follow_the_decoded_scene():
    ds = make_dataset(_cfg("object_action", P=200), CPU, F64)
    for p in range(200):
        agent = divmod(int(ds.factors["position"][p]), 3)
        cells = {divmod(i, 3) for i in range(9) if ds.factors[f"object_{i}"][p]}
        assert ds.y[p].tolist() == _oracle_actions(agent, cells)


def test_object_counts_are_uniform_and_cells_distinct():
    cfg = _cfg("object_action", P=20_000, grid_size=5, max_objects=3, D_in=50)
    ds = make_dataset(cfg, CPU, F64)
    objects = ds.x[:, 25:]
    assert set(objects.unique().tolist()) == {0.0, 1.0}  # binary map: no cell counted twice
    counts = torch.bincount(objects.sum(dim=1).long(), minlength=4).double()
    assert counts.sum() == 20_000
    chi2 = ((counts - 5000) ** 2 / 5000).sum()
    assert chi2 < 20  # 3 degrees of freedom: p ~ 1e-4
    positions = torch.bincount(ds.factors["position"], minlength=25).double()
    assert positions.min() > 0.8 * 800 and positions.max() < 1.2 * 800


def test_objects_may_sit_under_the_agent():
    ds = make_dataset(_cfg("object_action", P=500, max_objects=9), CPU, F64)
    under = ds.x[:, :9] * ds.x[:, 9:]
    assert under.sum() > 0


def test_object_action_rejects_more_objects_than_cells():
    with pytest.raises(ValueError, match="max_objects"):
        make_dataset(_cfg("object_action", max_objects=10), CPU, F64)


# --- what-where ---------------------------------------------------------------------------


def _render(where: int, motif: torch.Tensor) -> torch.Tensor:
    img = torch.zeros(9, 9, dtype=motif.dtype)
    big_r, big_c = divmod(where, 3)
    for r, c in itertools.product(range(3), range(3)):
        img[3 * big_r + r, 3 * big_c + c] = motif[r * 3 + c]
    return img.reshape(81)


def test_what_where_x_is_rebuilt_from_where_what_and_motifs():
    ds = make_dataset(_cfg("what_where", P=100), CPU, F64)
    motifs = ds.extras["motifs"]
    assert motifs.shape == (5, 9)
    for p in range(100):
        want = _render(int(ds.factors["where"][p]), motifs[ds.factors["what"][p]])
        assert torch.equal(ds.x[p], want)


def test_what_where_targets_are_the_two_one_hots():
    ds = make_dataset(_cfg("what_where", P=100), CPU, F64)
    where, what = ds.factors["where"], ds.factors["what"]
    assert ds.y.shape == (100, 14)
    assert torch.equal(ds.y[:, :9].argmax(dim=1), where) and ds.y[:, :9].sum(dim=1).eq(1).all()
    assert torch.equal(ds.y[:, 9:].argmax(dim=1), what) and ds.y[:, 9:].sum(dim=1).eq(1).all()
    assert where.dtype == what.dtype == torch.int64
    assert ds.groups == {"where": ["where"], "what": ["what"]}


def test_what_where_motifs_are_distinct_binary_and_non_empty():
    ds = make_dataset(_cfg("what_where", n_motifs=100), CPU, F64)
    motifs = ds.extras["motifs"]
    assert set(motifs.unique().tolist()) <= {0.0, 1.0}
    assert (motifs.sum(dim=1) > 0).all()
    assert len({tuple(m.tolist()) for m in motifs}) == 100


def test_what_where_can_use_the_whole_library():
    ds = make_dataset(_cfg("what_where", n_motifs=511, P=10), CPU, F64)
    assert len({tuple(m.tolist()) for m in ds.extras["motifs"]}) == 511


def test_what_where_where_and_what_are_uniform_and_independent():
    ds = make_dataset(_cfg("what_where", P=20_000, n_motifs=4), CPU, F64)
    joint = torch.zeros(9, 4)
    joint.index_put_((ds.factors["where"], ds.factors["what"]), torch.ones(20_000), accumulate=True)
    expected = 20_000 / 36
    assert (joint - expected).abs().max() < 0.2 * expected


def test_what_where_rejects_more_motifs_than_the_library():
    with pytest.raises(ValueError, match="511"):
        make_dataset(_cfg("what_where", n_motifs=512), CPU, F64)
