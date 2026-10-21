import pytest
import torch

from fct.modularity import layer_mir, layer_mir_report, shuffled_mir


def _data(P=300, N=12, seed=0):
    gen = torch.Generator().manual_seed(seed)
    a = torch.rand(P, dtype=torch.float64, generator=gen)
    labels = torch.randint(0, 3, (P,), generator=gen)  # categorical int factor
    noise = torch.randn(P, N, dtype=torch.float64, generator=gen)
    rates = noise.clone()
    rates[:, :4] += 3 * a[:, None]  # neurons 0-3 read the continuous factor
    rates[:, 4:8] += 3 * labels[:, None]  # neurons 4-7 read the categorical one
    rates[:, 11] = 0.0  # a constant neuron
    return rates, {"a": a, "c": labels, "b": torch.rand(P, dtype=torch.float64, generator=gen)}


@pytest.mark.parametrize(
    "groups", [None, {"cont": ["a", "b"], "cat": ["c"]}], ids=["singletons", "grouped"]
)
@pytest.mark.parametrize("n_shuffles", [1, 4])
def test_report_equals_the_separate_functions_exactly(groups, n_shuffles):
    rates, factors = _data()
    args = (factors, groups, 5)
    report = layer_mir_report(rates, *args, n_shuffles=n_shuffles, seed=7)
    raw = layer_mir(rates, *args, bias_correction="none")
    corrected = layer_mir(rates, *args, "shuffle", n_shuffles, 7)
    assert report.mir == raw.mir
    assert report.n_dead == raw.n_dead
    assert report.mir_corrected == corrected.mir
    assert report.n_dead_corrected == corrected.n_dead
    assert report.mir_chance == shuffled_mir(rates, *args, seed=7, n_shuffles=n_shuffles)


def test_report_validates_arguments():
    rates, factors = _data()
    with pytest.raises(ValueError, match="n_shuffles"):
        layer_mir_report(rates, factors, n_shuffles=0)
    with pytest.raises(ValueError, match="rates"):
        layer_mir_report(rates[:, 0], factors)
