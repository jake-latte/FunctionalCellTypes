"""Pure analysis helpers behind the example-output figures (CPU, tiny synthetic data)."""

import pytest
import torch

from fct.analyse import (
    ExampleOutputs,
    decode_object_action,
    decode_what_where,
    example_outputs,
    pick_examples,
    prediction_report,
)
from fct.config import DataConfig, NetConfig
from fct.data import make_dataset, task_dims
from fct.model import make_net

CPU, F64 = torch.device("cpu"), torch.float64
G = 3
OA = DataConfig(task="object_action", P=40, grid_size=G, max_objects=3, D_in=2 * G * G, seed=1)
WW = DataConfig(task="what_where", P=40, n_motifs=4, D_in=81, seed=2)
LT = DataConfig(task="linear_teacher", P=40, D_in=6, n_latents=2, seed=3)
POS = DataConfig(task="position", P=40, D_in=8, d=2, embedding="linear", seed=4)


def _net_and_data(cfg):
    ds = make_dataset(cfg, CPU, F64)
    d_in, d_out = task_dims(cfg)
    return make_net(NetConfig(N=5, L=2, phi="relu"), d_in, d_out, CPU, F64), ds


# ----------------------------------------------------------------- picking and evaluating


def test_pick_examples_is_seeded_distinct_and_in_range():
    a, b = pick_examples(20, 6, seed=3), pick_examples(20, 6, seed=3)
    assert torch.equal(a, b)
    assert len(set(a.tolist())) == 6 and 0 <= a.min() and a.max() < 20
    assert not torch.equal(a, pick_examples(20, 6, seed=4))
    assert a.device.type == "cpu"


def test_pick_examples_rejects_more_than_available():
    with pytest.raises(ValueError, match="only 5"):
        pick_examples(5, 6, seed=0)


@pytest.mark.parametrize("cfg", [OA, WW, LT, POS], ids=lambda c: c.task)
def test_example_outputs_matches_a_direct_forward_pass(cfg):
    net, ds = _net_and_data(cfg)
    idx = [5, 0, 17]
    out = example_outputs(net, ds, idx)
    with torch.no_grad():
        _, z = net(ds.x)
    assert isinstance(out, ExampleOutputs)
    assert torch.equal(out.indices, torch.tensor(idx))
    assert torch.equal(out.x, ds.x[idx])
    assert torch.equal(out.y_true, ds.y[idx])
    # Row-subset matmuls may differ from the full batch in the last bits.
    torch.testing.assert_close(out.y_pred, z[idx], rtol=1e-12, atol=1e-14)
    assert not out.y_pred.requires_grad


def test_example_outputs_rejects_out_of_range_indices():
    net, ds = _net_and_data(LT)
    with pytest.raises(ValueError, match="indices"):
        example_outputs(net, ds, [0, LT.P])


# ----------------------------------------------------------------- decoding


def test_decode_object_action_inverts_the_generator():
    ds = make_dataset(OA, CPU, F64)
    agent_rc, objects = decode_object_action(ds.x, G)
    cell = ds.factors["position"]
    assert torch.equal(agent_rc, torch.stack([cell // G, cell % G], dim=1))
    assert objects.dtype == torch.bool and objects.shape == (OA.P, G, G)
    expected = torch.stack([ds.factors[f"object_{i}"] for i in range(G * G)], dim=1)
    assert torch.equal(objects.reshape(OA.P, -1), expected.bool())


def test_decode_object_action_rejects_wrong_width():
    with pytest.raises(ValueError, match="2\\*G\\*G"):
        decode_object_action(torch.zeros(2, 10), G)


def test_decode_what_where_returns_the_motif_in_its_macro_cell():
    ds = make_dataset(WW, CPU, F64)
    image = decode_what_where(ds.x)
    assert image.shape == (WW.P, 9, 9)
    where, what = ds.factors["where"], ds.factors["what"]
    motifs = ds.extras["motifs"]
    for i in range(WW.P):
        r, c = divmod(int(where[i]), 3)
        block = image[i, 3 * r : 3 * r + 3, 3 * c : 3 * c + 3]
        assert torch.equal(block.reshape(-1), motifs[what[i]])
        assert image[i].sum() == block.sum()  # nothing outside the macro-cell


def test_decode_what_where_rejects_wrong_width():
    with pytest.raises(ValueError, match="81"):
        decode_what_where(torch.zeros(2, 80))


# ----------------------------------------------------------------- prediction_report


def test_report_regression_tasks_give_r2_one_for_perfect_prediction():
    y = torch.tensor([[1.0, -1.0], [2.0, 0.0], [0.0, 3.0]], dtype=F64)
    for cfg in (LT, POS):
        assert prediction_report(cfg.task, y, y.clone(), cfg) == {"r2": 1.0}


def test_report_r2_is_computed_against_the_grand_mean():
    y = torch.tensor([[0.0, 2.0], [4.0, 6.0]], dtype=F64)  # mean 3, SS_tot = 9+1+1+9 = 20
    pred = y + torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=F64)  # SS_res = 2
    assert prediction_report("linear_teacher", y, pred, LT)["r2"] == pytest.approx(0.9)


def test_report_r2_of_a_constant_target_raises():
    y = torch.ones(3, 2, dtype=F64)
    with pytest.raises(ValueError, match="variance"):
        prediction_report("position", y, y, POS)


def _actions(rows):
    return torch.tensor(rows, dtype=F64)


def test_report_object_action_exact_match_and_argmax_optimal():
    # Targets: stay | up and left optimal | right optimal | up optimal.
    y = _actions([[1, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 1], [0, 1, 0, 0, 0]])
    pred = _actions(
        [
            [0.9, 0.1, 0.0, 0.2, 0.1],  # exact; argmax stay is optimal
            [0.1, 0.9, 0.1, 0.2, 0.1],  # thresholds to {up}, not {up, left}: not exact;
            #                             argmax up is optimal
            [0.1, 0.9, 0.1, 0.2, 0.6],  # thresholds to {up, right}: not exact; argmax up wrong
            [0.0, 0.6, 0.0, 0.0, 0.0],  # exact; argmax up optimal
        ]
    )
    report = prediction_report("object_action", y, pred, OA)
    assert report == {"exact_match": 0.5, "argmax_optimal": 0.75}


def test_report_object_action_argmax_only_credit():
    y = _actions([[0, 1, 0, 1, 0]])
    pred = _actions([[0.1, 0.4, 0.0, 0.2, 0.0]])  # thresholds to {} but argmax = up (optimal)
    assert prediction_report("object_action", y, pred, OA) == {
        "exact_match": 0.0,
        "argmax_optimal": 1.0,
    }


def test_report_what_where_scores_where_and_what_separately():
    y = torch.zeros(3, 13, dtype=F64)  # m = 4 motifs, so 9 + 4 outputs
    y[0, [2, 9 + 1]] = 1
    y[1, [5, 9 + 0]] = 1
    y[2, [8, 9 + 3]] = 1
    pred = y.clone()
    pred[1, 5], pred[1, 0] = 0.0, 1.0  # where wrong in sample 1
    pred[2, 9 + 3], pred[2, 9 + 2] = 0.0, 1.0  # what wrong in sample 2
    report = prediction_report("what_where", y, pred, WW)
    assert report == {"where_accuracy": pytest.approx(2 / 3), "what_accuracy": pytest.approx(2 / 3)}
    assert prediction_report("what_where", y, y, WW) == {
        "where_accuracy": 1.0,
        "what_accuracy": 1.0,
    }


def test_report_rejects_unknown_task_and_shape_mismatch():
    y = torch.zeros(2, 5, dtype=F64)
    with pytest.raises(ValueError, match="unknown task"):
        prediction_report("nope", y, y, OA)
    with pytest.raises(ValueError, match="shape"):
        prediction_report("object_action", y, y[:, :4], OA)
    with pytest.raises(ValueError, match="outputs"):
        prediction_report("what_where", y, y, WW)  # 5 outputs, WW needs 13
