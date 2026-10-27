import math
import time

import pytest
import torch

from fct.modularity import (
    factor_labels,
    information_overlap,
    layer_mir,
    mir,
    mutual_information,
    network_mir,
    neuron_factor_information,
    neuron_mir,
    preference,
    rank_bins,
    shuffle_factors,
    shuffled_mir,
)

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


# ------------------------------------------------- synthetic ground truth

K3, C3 = 3, 3  # three factors with three levels each


def _factors3(P: int = P_BIG, seed: int = 10) -> dict[str, torch.Tensor]:
    g = _gen(seed)
    return {f"f{k}": torch.randint(0, C3, (P,), generator=g) for k in range(K3)}


def _coded(factors: dict[str, torch.Tensor], names: list[str], scale: float) -> torch.Tensor:
    """Rate that is a monotone code of the joint configuration of `names`."""
    code = torch.zeros_like(next(iter(factors.values())))
    for name in names:
        code = code * C3 + factors[name]
    return scale * code.double()


def _modular_layer(factors) -> torch.Tensor:
    """Neuron n carries factor n % 3 only, with its own monotone gain."""
    cols = [_coded(factors, [f"f{n % K3}"], 1.0 + n) for n in range(2 * K3)]
    return torch.stack(cols, dim=1)


def _expected_mir(r: float, K: int) -> float:
    return (r - 1 / K) / (1 - 1 / K)


def test_preference_is_max_over_sum_and_dead_neurons_sit_at_chance():
    group_mi = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 2.0], [0.0, 0.0, 0.0]]).double()
    r, dead = preference(group_mi)
    assert r.tolist() == pytest.approx([1.0, 0.5, 1 / 3])
    assert dead.tolist() == [False, False, True]


def test_mir_formula_endpoints_and_validation():
    assert mir(torch.ones(5), 3) == pytest.approx(1.0)
    assert mir(torch.full((5,), 1 / 3, dtype=torch.float64), 3) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="K"):
        mir(torch.ones(5), 1)


def test_single_factor_neurons_give_mir_near_one():
    factors = _factors3()
    out = layer_mir(_modular_layer(factors), factors, n_bins=10)
    assert out.r.min() > 0.97
    assert out.mir > 0.97
    assert out.n_dead == 0
    assert out.group_names == ["f0", "f1", "f2"]
    assert out.group_mi.shape == (2 * K3, K3)


def test_two_factor_mixtures_give_the_analytic_mir():
    factors = _factors3()
    pairs = [["f0", "f1"], ["f1", "f2"], ["f0", "f2"]]
    rates = torch.stack([_coded(factors, pair, 1.0) for pair in pairs * 2], dim=1)
    out = layer_mir(rates, factors, n_bins=10)
    assert out.r.tolist() == pytest.approx([0.5] * 6, abs=0.01)
    assert out.mir == pytest.approx(_expected_mir(0.5, K3), abs=0.02)


def test_fully_mixed_neurons_give_mir_near_zero():
    factors = _factors3()
    rates = torch.stack([_coded(factors, ["f0", "f1", "f2"], 1.0 + n) for n in range(4)], dim=1)
    out = layer_mir(rates, factors, n_bins=C3**K3)  # enough bins to resolve all 27 configurations
    assert abs(out.mir) < 0.03


def test_noise_layer_mir_is_low_and_matches_the_shuffled_baseline():
    factors = _factors3()
    modular = _modular_layer(factors)
    noise = torch.randn(P_BIG, 200, generator=_gen(11), dtype=torch.float64)
    noise_mir = layer_mir(noise, factors).mir
    baseline = shuffled_mir(noise, factors, n_shuffles=3)
    assert noise_mir < 0.5 * layer_mir(modular, factors).mir
    assert noise_mir == pytest.approx(baseline, abs=0.05)


def test_shuffled_mir_of_a_modular_layer_is_far_below_its_true_mir():
    factors = _factors3()
    layer = _modular_layer(factors)
    assert shuffled_mir(layer, factors, n_shuffles=2) < 0.5 * layer_mir(layer, factors).mir


def test_shuffled_mir_is_seeded_and_shuffle_factors_keeps_dependencies():
    factors = _factors3(P=500)
    layer = torch.randn(500, 5, generator=_gen(12))
    assert shuffled_mir(layer, factors, seed=3) == shuffled_mir(layer, factors, seed=3)
    assert shuffled_mir(layer, factors, seed=3) != shuffled_mir(layer, factors, seed=4)
    dependent = {"a": factors["f0"], "b": factors["f0"].clone(), "c": torch.rand(500)}
    shuffled = shuffle_factors(dependent, torch.Generator().manual_seed(0))
    assert torch.equal(shuffled["a"], shuffled["b"])  # one permutation for every factor
    assert not torch.equal(shuffled["a"], dependent["a"])
    assert torch.equal(shuffled["c"].sort().values, dependent["c"].sort().values)


def test_continuous_latents_with_monotone_neurons_give_mir_near_one():
    g = _gen(13)
    u, v = torch.rand(P_BIG, generator=g), torch.rand(P_BIG, generator=g)
    rates = torch.stack([torch.exp(3 * u), v**3, -u, torch.relu(v - 0.3)], dim=1)
    assert layer_mir(rates, {"u": u, "v": v}, n_bins=10).mir > 0.97


# ------------------------------------------------------------------ groups


def _binary_factors(P: int = P_BIG, seed: int = 14):
    g = _gen(seed)
    return {name: torch.randint(0, 2, (P,), generator=g) for name in "abcd"}


def test_group_mi_is_the_sum_of_member_mi():
    factors = _binary_factors()
    rates = factors["a"][:, None].double()  # copies member a of the group
    info = neuron_factor_information(rates, factors, {"g": ["a", "b", "c"], "h": ["d"]})
    assert info.group_names == ["g", "h"]
    assert info.member_mi.shape == (1, 4)
    assert float(info.member_mi[0, 0]) == pytest.approx(_entropy(factors["a"]), abs=1e-9)
    assert info.member_mi[0, 1:].abs().max() < 0.01
    assert float(info.group_mi[0, 0]) == pytest.approx(_entropy(factors["a"]), abs=0.02)
    assert float(info.group_mi[0, 1]) == pytest.approx(float(info.member_mi[0, 3]))


def test_mir_over_groups_differs_from_mir_over_members_as_constructed():
    f = _binary_factors()
    # neuron 2 mixes a and b, which share a group
    rates = torch.stack([f["a"].double(), f["c"].double(), (2 * f["a"] + f["b"]).double()], dim=1)
    by_group = layer_mir(rates, f, {"ab": ["a", "b"], "cd": ["c", "d"]})
    by_member = layer_mir(rates, f)
    assert by_group.mir == pytest.approx(1.0, abs=0.02)
    # members: r = [1, 1, 0.5], K = 4
    assert by_member.mir == pytest.approx(_expected_mir((1 + 1 + 0.5) / 3, 4), abs=0.02)


def test_no_groups_equals_singleton_groups():
    f = _binary_factors(P=2000)
    rates = torch.randn(2000, 5, generator=_gen(15))
    default = neuron_factor_information(rates, f)
    explicit = neuron_factor_information(rates, f, {k: [k] for k in f})
    assert default.group_names == explicit.group_names == list(f)
    assert torch.equal(default.group_mi, explicit.group_mi)
    assert torch.equal(default.group_mi, default.member_mi)


# ------------------------------------------------------------ dead neurons


def test_constant_neurons_are_dead_sit_at_chance_and_add_nothing():
    factors = _factors3()
    live = _modular_layer(factors)[:, :2]
    rates = torch.cat([live, torch.full((P_BIG, 2), 0.7, dtype=torch.float64)], dim=1)
    out = layer_mir(rates, factors)
    assert out.n_dead == 2
    assert out.r[2:].tolist() == [1 / K3, 1 / K3]  # exactly chance
    live_mean = float(out.r[:2].mean())
    # Two of four neurons contribute 0 to the numerator: MIR = half the live neurons' MIR.
    assert out.mir == pytest.approx(0.5 * _expected_mir(live_mean, K3), abs=1e-12)


# --------------------------------------------------------- network, dtype


def test_network_mir_returns_one_entry_per_layer():
    factors = _factors3()
    modular = _modular_layer(factors)
    noise = torch.randn(P_BIG, modular.shape[1], generator=_gen(16), dtype=torch.float64)
    out = network_mir(torch.stack([modular, noise], dim=1), factors)
    assert len(out) == 2
    assert out[0].mir > 0.97
    assert out[1].mir < 0.5
    assert out[0].mir == layer_mir(modular, factors).mir


def test_float32_and_float64_rates_agree():
    factors = _factors3(P=3000)
    rates32 = torch.randn(3000, 8, generator=_gen(17)).relu()
    a = layer_mir(rates32, factors)
    b = layer_mir(rates32.double(), factors)
    assert a.mir == pytest.approx(b.mir, abs=1e-9)
    assert torch.allclose(a.group_mi, b.group_mi, atol=1e-9)


def test_many_neurons_are_scored_quickly():
    factors = {k: v[:5000] for k, v in _factors3().items()}
    rates = torch.randn(5000, 2000, generator=_gen(18))
    start = time.perf_counter()
    out = layer_mir(rates, factors)
    assert time.perf_counter() - start < 10.0
    assert out.r.shape == (2000,)


# --------------------------------------------------------- failing loudly


def test_invalid_inputs_fail_loudly():
    factors = _factors3(P=100)
    rates = torch.randn(100, 3)
    with pytest.raises(ValueError, match="2-D"):
        layer_mir(rates[:, 0], factors)
    with pytest.raises(ValueError, match="3-D"):
        network_mir(rates, factors)
    with pytest.raises(ValueError, match="samples"):
        layer_mir(rates[:50], factors)
    with pytest.raises(ValueError, match="n_bins"):
        layer_mir(rates, factors, n_bins=1)
    with pytest.raises(ValueError, match="K"):
        layer_mir(rates, {"f0": factors["f0"]})
    with pytest.raises(ValueError, match="unknown"):
        layer_mir(rates, factors, {"g": ["f0", "nope"], "h": ["f1"]})
    with pytest.raises(ValueError, match="at least one factor"):
        layer_mir(rates, {})
    with pytest.raises(ValueError, match="n_shuffles"):
        shuffled_mir(rates, factors, n_shuffles=0)


# ------------------------------------------------------- bias correction

P_MID = 8192


def _factor_sets(P: int = P_MID, seed: int = 20) -> dict[str, dict[str, torch.Tensor]]:
    g = _gen(seed)
    return {
        "3x5": {f"f{i}": torch.randint(0, 5, (P,), generator=g) for i in range(3)},
        "25+binary": {
            "a": torch.randint(0, 25, (P,), generator=g),
            "b": torch.randint(0, 2, (P,), generator=g),
            "c": torch.randint(0, 2, (P,), generator=g),
        },
    }


@pytest.mark.parametrize("set_name", ["3x5", "25+binary"])
def test_corrected_mir_of_noise_neurons_is_near_zero_for_any_cardinality(set_name):
    factors = _factor_sets()[set_name]
    noise = torch.randn(P_MID, 200, generator=_gen(21), dtype=torch.float64)
    raw = layer_mir(noise, factors).mir
    corrected = layer_mir(noise, factors, bias_correction="shuffle")
    assert abs(corrected.mir) < 0.03
    assert corrected.n_dead > 150  # almost all noise neurons are indistinguishable from chance
    assert raw > 0.05  # the uncorrected floor is what the correction removes


def test_raw_noise_floor_depends_on_the_factor_set():
    sets = _factor_sets()
    noise = torch.randn(P_MID, 200, generator=_gen(21), dtype=torch.float64)
    floors = {name: layer_mir(noise, f).mir for name, f in sets.items()}
    assert floors["25+binary"] > floors["3x5"] + 0.3


def test_corrected_mir_keeps_the_modular_and_mixed_ground_truth():
    factors = _factors3()
    modular = layer_mir(_modular_layer(factors), factors, bias_correction="shuffle")
    assert modular.mir > 0.95
    pairs = [["f0", "f1"], ["f1", "f2"], ["f0", "f2"]]
    mixed = torch.stack([_coded(factors, pair, 1.0) for pair in pairs * 2], dim=1)
    out = layer_mir(mixed, factors, bias_correction="shuffle")
    assert out.mir == pytest.approx(_expected_mir(0.5, K3), abs=0.02)


def test_bias_correction_is_deterministic_for_a_seed_and_none_is_unchanged():
    factors = _factors3(P=2000)
    rates = torch.randn(2000, 20, generator=_gen(22)).relu()
    a = neuron_factor_information(rates, factors, bias_correction="shuffle", seed=4)
    b = neuron_factor_information(rates, factors, bias_correction="shuffle", seed=4)
    assert torch.equal(a.group_mi, b.group_mi)
    plain = neuron_factor_information(rates, factors)
    explicit = neuron_factor_information(rates, factors, bias_correction="none")
    assert torch.equal(plain.member_mi, explicit.member_mi)
    assert (a.member_mi <= plain.member_mi + 1e-12).all()  # correction only subtracts
    with pytest.raises(ValueError, match="bias_correction"):
        neuron_factor_information(rates, factors, bias_correction="bogus")  # type: ignore[arg-type]


def test_bias_correction_passes_through_layer_and_network():
    factors = _factors3(P=3000)
    rates = torch.randn(3000, 2, 10, generator=_gen(23))
    out = network_mir(rates, factors, bias_correction="shuffle", n_shuffles=2, seed=1)
    ref = layer_mir(rates[:, 1], factors, bias_correction="shuffle", n_shuffles=2, seed=1)
    assert out[1].mir == ref.mir


# ------------------------------------------------- conventions, validation


def test_ungrouped_factor_is_ignored_shared_factor_counts_twice_constant_is_accepted():
    f = _binary_factors(P=4000)
    f["const"] = torch.zeros(4000, dtype=torch.int64)
    f["cont"] = torch.full((4000,), 0.5)
    rates = f["a"][:, None].double()
    info = neuron_factor_information(rates, f, {"g1": ["a"], "g2": ["a", "b"], "g3": ["const"]})
    assert float(info.group_mi[0, 0]) == pytest.approx(float(info.member_mi[0, 0]))
    assert float(info.group_mi[0, 1]) >= float(info.member_mi[0, 0])  # a counted in g2 as well
    assert float(info.group_mi[0, 2]) == 0.0  # constant factor: zero information, still a group
    assert info.member_mi[0, 4:].abs().max() == 0.0  # the ungrouped constant float factor too
    assert len(info.group_names) == 3


def test_preference_validates_shape_and_K():
    with pytest.raises(ValueError, match="K"):
        preference(torch.ones(4, 1))
    with pytest.raises(ValueError, match="2-D"):
        preference(torch.ones(4))


def test_low_level_functions_validate_dimensions_and_dtypes():
    ints = torch.zeros(5, 1, dtype=torch.int64)
    with pytest.raises(ValueError, match="2-D"):
        mutual_information(torch.zeros(5, dtype=torch.int64), 2, torch.zeros(5).long(), 2)
    with pytest.raises(ValueError, match="P = 0"):
        mutual_information(torch.zeros(0, 1, dtype=torch.int64), 2, torch.zeros(0).long(), 2)
    with pytest.raises(ValueError, match="integer"):
        mutual_information(ints.double(), 2, torch.zeros(5).long(), 2)
    with pytest.raises(ValueError, match="integer"):
        mutual_information(ints, 2, torch.zeros(5), 2)
    with pytest.raises(ValueError, match="P = 0"):
        rank_bins(torch.zeros(0, 3), 4)
    with pytest.raises(ValueError, match="at least one factor"):
        shuffle_factors({}, _gen())


def test_nan_inputs_are_rejected_by_name():
    factors = _factors3(P=50)
    rates = torch.randn(50, 3)
    bad_rates = rates.clone()
    bad_rates[3, 1] = float("nan")
    with pytest.raises(ValueError, match="rates contains NaN"):
        layer_mir(bad_rates, factors)
    bad_factor = torch.rand(50)
    bad_factor[0] = float("nan")
    with pytest.raises(ValueError, match="'u' contains NaN"):
        layer_mir(rates, {"u": bad_factor, "v": torch.rand(50)})


@pytest.mark.parametrize("bias_correction", ["none", "shuffle"])
def test_neuron_mir_mean_is_the_layer_mir_with_a_dead_neuron(bias_correction):
    g = _gen(3)
    factors = {"a": torch.randint(0, 3, (500,), generator=g), "b": torch.rand(500, generator=g)}
    rates = torch.randn(500, 4, generator=g)
    rates[:, 0] += factors["a"]
    rates[:, 3] = 1.0  # constant: dead
    out = layer_mir(rates, factors, bias_correction=bias_correction)
    per_neuron = neuron_mir(out.r, len(out.group_names))
    assert per_neuron.shape == (4,)
    assert per_neuron.mean().item() == pytest.approx(out.mir)
    assert per_neuron[3].item() == pytest.approx(0.0)


def test_neuron_mir_needs_at_least_two_groups():
    with pytest.raises(ValueError, match="K >= 2"):
        neuron_mir(torch.tensor([1.0]), 1)


# ---------------------------------------------------------------- information overlap


def _copy_block_rates(n_copy=(4, 3), n_noise=5, P=4000, seed=0):
    """Rates of 12 neurons to 2 latents: the first 4 copy z0, the next 3 copy z1, 5 are noise."""
    g = _gen(seed)
    z = torch.randn(P, 2, generator=g)
    noise = torch.randn(P, sum(n_copy) + n_noise, generator=g)
    rates = noise.clone()
    a, b = n_copy[0], n_copy[0] + n_copy[1]
    rates[:, :a] = z[:, :1] + 0.1 * noise[:, :a]
    rates[:, a:b] = z[:, 1:] + 0.1 * noise[:, a:b]
    return rates, {"z0": z[:, 0], "z1": z[:, 1]}, a, b


def test_information_overlap_is_block_diagonal_on_neurons_that_copy_a_latent():
    rates, factors, a, b = _copy_block_rates()
    info = neuron_factor_information(rates, factors, n_bins=8, bias_correction="shuffle")
    C = information_overlap(info.member_mi)
    assert C.shape == (12, 12)
    assert torch.allclose(C, C.T)
    assert torch.allclose(C.diagonal(), (info.member_mi**2).sum(1) / 2)
    block = torch.zeros(12, 12, dtype=torch.bool)
    block[:a, :a] = block[a:b, a:b] = True
    assert C[block].min() > 1.0  # copies carry ~2 nats each about their latent
    assert C[~block].max() < 0.05  # across blocks and on the noise neurons: ~0


def test_information_overlap_is_the_scaled_gram_matrix_of_the_informations():
    mi = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    expected = torch.tensor([[1.0, 0.0, 1.0], [0.0, 4.0, 2.0], [1.0, 2.0, 2.0]]) / 2
    assert torch.equal(information_overlap(mi), expected)


@pytest.mark.parametrize("shape", [(5,), (2, 3, 4)])
def test_information_overlap_rejects_non_2d_input(shape):
    with pytest.raises(ValueError, match="2-D"):
        information_overlap(torch.zeros(shape))
