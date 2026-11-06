"""`fct.tasks.resample`: held-out samples of the same task, with the task definition kept."""

import pytest
import torch

from fct.config import DataConfig
from fct.data import make_dataset
from fct.tasks import ACTION_NAMES, resample

CPU, F32 = torch.device("cpu"), torch.float32


def _cfg(task, seed=0, P=24):
    fields = {
        "linear_teacher": {"D_in": 6, "n_latents": 2, "latent_min": -1.0, "latent_max": 1.0},
        "object_action": {"D_in": 18, "grid_size": 3, "max_objects": 2},
        "what_where": {"D_in": 81, "n_motifs": 4},
    }[task]
    return DataConfig(task=task, P=P, seed=seed, **fields)


@pytest.mark.parametrize("task", ["linear_teacher", "object_action", "what_where"])
def test_resample_has_the_reference_shapes_is_deterministic_and_new(task):
    cfg = _cfg(task)
    ref = make_dataset(cfg, CPU, F32)
    a, b, c = (resample(cfg, ref, seed, CPU, F32) for seed in (7, 7, 8))
    assert a.x.shape == ref.x.shape and a.y.shape == ref.y.shape and a.x.dtype == F32
    assert torch.equal(a.x, b.x) and torch.equal(a.y, b.y)
    assert not torch.equal(a.x, ref.x) and not torch.equal(a.x, c.x)


def test_resample_does_not_change_what_the_generators_produce():
    cfg = _cfg("what_where")
    before = make_dataset(cfg, CPU, F32)
    resample(cfg, before, 7, CPU, F32)
    assert torch.equal(make_dataset(cfg, CPU, F32).x, before.x)


def test_linear_teacher_keeps_the_reference_teacher():
    cfg = _cfg("linear_teacher")
    ref = make_dataset(cfg, CPU, F32)
    new = resample(cfg, ref, 7, CPU, F32)
    z = torch.stack([new.factors[f"z{i}"] for i in range(2)], dim=1)
    assert torch.allclose(new.y, z @ ref.extras["W_Y"].T)
    assert torch.allclose(new.x, z @ ref.extras["W_X"].T)
    assert new.extras is ref.extras


def test_what_where_keeps_the_reference_motifs():
    cfg = _cfg("what_where", P=200)
    ref = make_dataset(cfg, CPU, F32)
    new = resample(cfg, ref, 7, CPU, F32)
    where, what = new.factors["where"], new.factors["what"]
    for i in range(20):
        r, c = divmod(int(where[i]), 3)
        patch = new.x[i].reshape(9, 9)[3 * r : 3 * r + 3, 3 * c : 3 * c + 3].reshape(-1)
        assert torch.equal(patch, ref.extras["motifs"][what[i]])
    # a plain generator call with the new seed would have drawn other motifs
    other = make_dataset(cfg.model_copy(update={"seed": 7}), CPU, F32)
    assert not torch.equal(other.extras["motifs"], ref.extras["motifs"])


def test_object_action_samples_are_new_and_the_targets_follow_the_rule():
    cfg = _cfg("object_action", P=50)
    ref = make_dataset(cfg, CPU, F32)
    new = resample(cfg, ref, 7, CPU, F32)
    assert not torch.equal(new.factors["position"], ref.factors["position"])
    for j, name in enumerate(ACTION_NAMES):
        assert torch.equal(new.y[:, j].long(), new.factors[f"action_{name}"])
    assert (new.y.sum(dim=1) >= 1).all()  # every sample has at least one optimal action


def test_resample_rejects_the_reference_seed_and_other_tasks():
    cfg = _cfg("linear_teacher")
    ref = make_dataset(cfg, CPU, F32)
    with pytest.raises(ValueError, match="seed"):
        resample(cfg, ref, cfg.seed, CPU, F32)
    pos = DataConfig(task="position", P=8, D_in=4)
    with pytest.raises(ValueError, match="task"):
        resample(pos, make_dataset(pos, CPU, F32), 5, CPU, F32)
