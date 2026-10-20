import math

import pytest
import torch

from fct.modularity import factor_labels, mutual_information, rank_bins

P_BIG = 20000


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _reference_rank_bins(values: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Straightforward per-column definition: midrank = (#(v<x) + #(v<=x)) / 2."""
    P, N = values.shape
    out = torch.zeros(P, N, dtype=torch.int64)
    for n in range(N):
        col = values[:, n].tolist()
        for p, x in enumerate(col):
            below = sum(v < x for v in col)
            at_most = sum(v <= x for v in col)
            out[p, n] = math.floor((below + at_most) / 2 / P * n_bins)
    return out


# ---------------------------------------------------------------- rank_bins


def test_rank_bins_matches_reference_on_random_data_with_ties():
    g = _gen(1)
    values = torch.randint(0, 7, (60, 4), generator=g).double()  # many ties
    values[:, 0] = torch.randn(60, generator=g, dtype=torch.float64)
    assert torch.equal(rank_bins(values, 5), _reference_rank_bins(values, 5))


def test_rank_bins_puts_all_ties_in_one_bin():
    g = _gen(2)
    col = torch.relu(torch.randn(1000, 1, generator=g))
    col[torch.rand(1000, 1, generator=g) < 0.6] = 0.0
    bins = rank_bins(col, 10)
    assert bins[col[:, 0] == 0, 0].unique().numel() == 1
    assert bins.unique().numel() <= 10


@pytest.mark.parametrize("transform", [torch.exp, lambda v: 7.0 * v + 3.0])
def test_rank_bins_is_invariant_to_monotone_transforms(transform):
    values = torch.randn(500, 3, generator=_gen(3), dtype=torch.float64)
    assert torch.equal(rank_bins(values, 8), rank_bins(transform(values), 8))


def test_rank_bins_is_equal_frequency_for_a_continuous_column():
    values = torch.randn(1000, 1, generator=_gen(4))
    counts = torch.bincount(rank_bins(values, 10)[:, 0], minlength=10)
    assert torch.equal(counts, torch.full((10,), 100))


def test_rank_bins_rejects_bad_input():
    with pytest.raises(ValueError, match="2-D"):
        rank_bins(torch.zeros(5), 4)
    with pytest.raises(ValueError, match="n_bins"):
        rank_bins(torch.zeros(5, 2), 1)


# ------------------------------------------------------------ factor_labels


def test_factor_labels_relabels_integers_densely():
    labels, C = factor_labels(torch.tensor([5, 9, 5, -2, 9]), n_bins=10)
    assert C == 3
    assert labels.dtype == torch.int64
    assert labels.tolist() == [1, 2, 1, 0, 2]


def test_factor_labels_bins_floats_into_n_bins():
    labels, C = factor_labels(torch.rand(1000, generator=_gen(5)), n_bins=4)
    assert C == 4
    assert torch.bincount(labels).tolist() == [250] * 4


def test_factor_labels_rejects_non_numeric_and_non_1d():
    with pytest.raises(ValueError, match="1-D"):
        factor_labels(torch.zeros(3, 2), 4)
    with pytest.raises(ValueError, match="integer or floating"):
        factor_labels(torch.tensor([True, False]), 4)


# ------------------------------------------------------- mutual_information


def _entropy(labels: torch.Tensor) -> float:
    p = torch.bincount(labels).double() / labels.numel()
    return float(-(p * p.log()).sum())


def test_mi_of_a_neuron_equal_to_the_label_is_the_label_entropy():
    labels = torch.arange(5).repeat_interleave(200)  # uniform over C = 5
    bins = rank_bins(labels[:, None].double(), 10)
    mi = mutual_information(bins, 10, labels, 5)
    assert mi.shape == (1,)
    assert float(mi[0]) == pytest.approx(math.log(5), abs=1e-9)


def test_mi_of_an_independent_neuron_is_only_the_plug_in_bias():
    g = _gen(6)
    labels = torch.randint(0, 5, (P_BIG,), generator=g)
    bins = torch.randint(0, 5, (P_BIG, 1), generator=g)
    mi = mutual_information(bins, 5, labels, 5)
    assert 0 <= float(mi[0]) < 0.01


def test_mi_is_symmetric_in_its_two_arguments():
    g = _gen(7)
    a = torch.randint(0, 4, (3000,), generator=g)
    b = (a + torch.randint(0, 2, (3000,), generator=g)) % 3
    forward = mutual_information(a[:, None], 4, b, 3)
    backward = mutual_information(b[:, None], 3, a, 4)
    assert float(forward[0]) == pytest.approx(float(backward[0]), abs=1e-12)


def test_mi_matches_the_analytic_value_of_a_2x2_table():
    # joint p(x, y): [[0.4, 0.1], [0.1, 0.4]] over 1000 samples
    x = torch.cat([torch.zeros(500), torch.ones(500)]).long()
    y = torch.cat([torch.zeros(400), torch.ones(100), torch.zeros(100), torch.ones(400)]).long()
    expected = 2 * 0.4 * math.log(0.4 / 0.25) + 2 * 0.1 * math.log(0.1 / 0.25)
    assert float(mutual_information(x[:, None], 2, y, 2)[0]) == pytest.approx(expected, abs=1e-12)


def test_mi_handles_many_neurons_at_once_and_a_constant_neuron_is_zero():
    labels = torch.arange(3).repeat(100)
    bins = torch.stack([labels, torch.zeros_like(labels)], dim=1)
    mi = mutual_information(bins, 3, labels, 3)
    assert float(mi[0]) == pytest.approx(math.log(3), abs=1e-12)
    assert float(mi[1]) == pytest.approx(0.0, abs=1e-12)


def test_mi_rejects_mismatched_lengths_and_out_of_range_codes():
    with pytest.raises(ValueError, match="samples"):
        mutual_information(torch.zeros(5, 1, dtype=torch.int64), 2, torch.zeros(4).long(), 2)
    with pytest.raises(ValueError, match="range"):
        mutual_information(torch.full((5, 1), 3), 2, torch.zeros(5).long(), 2)
