"""Tuning curves and neuron selection on synthetic layers with known ground truth."""

import numpy as np
import pytest
import torch

from fct.tuning import select_neurons, tuning_curves

P, N, K, BINS = 4000, 5, 3, 8
NOISE = 0.1


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


@pytest.fixture
def layer():
    """Neuron 0 is 3 * latent_1**2 plus noise; every other neuron is noise. Latents are
    float64 uniform draws, which are distinct, so the bins are equal-count by construction."""
    g = _gen()
    factors = torch.rand(P, K, generator=g, dtype=torch.float64)
    rates = NOISE * torch.randn(P, N, generator=g, dtype=torch.float64)
    rates[:, 0] += 3 * factors[:, 1] ** 2
    return rates, factors


def test_tuning_curves_recover_the_known_function_of_the_driving_latent(layer):
    curves = tuning_curves(*layer, n_bins=BINS)
    b = torch.arange(BINS, dtype=torch.float64)
    expected = ((b + 1) ** 3 - b**3) / BINS**2  # mean of 3 f^2 over the bin [b/B, (b+1)/B]
    assert curves.mean.shape == (N, K, BINS)
    assert curves.sem.shape == (N, K, BINS)
    assert curves.centres.shape == (K, BINS)
    assert (curves.mean[0, 1] - expected).abs().lt(6 * curves.sem[0, 1] + 0.02).all()
    assert (curves.centres - (b + 0.5) / BINS).abs().max() < 0.03  # sample quantiles: sd <= 0.008
    assert (curves.mean[0, 1, 1:] > curves.mean[0, 1, :-1]).all()


def test_tuning_curves_are_flat_in_the_undriven_latents(layer):
    curves = tuning_curves(*layer, n_bins=BINS)
    for latent in (0, 2):
        spread = curves.mean[0, latent].max() - curves.mean[0, latent].min()
        assert spread < 5 * curves.sem[0, latent].max()


def test_tuning_curves_bins_are_equal_count(layer):
    curves = tuning_curves(*layer, n_bins=BINS)
    assert (curves.count == P // BINS).all()
    assert (curves.centres[:, 1:] > curves.centres[:, :-1]).all()


def test_tuning_curves_standard_error_matches_direct_computation():
    factors = torch.arange(8.0)[:, None]
    rates = torch.tensor([[1.0, 3.0, 2.0, 2.0, 0.0, 4.0, 5.0, 5.0]]).T
    curves = tuning_curves(rates, factors, n_bins=2)
    first = rates[:4, 0]
    assert curves.mean[0, 0, 0] == pytest.approx(first.mean().item())
    assert curves.sem[0, 0, 0] == pytest.approx((first.std() / 2).item())
    assert curves.centres[0, 0] == pytest.approx(1.5)


def test_tuning_curves_standard_error_survives_a_large_offset():
    rates = 1e6 + 1e-3 * torch.randn(200, 1, generator=_gen(1), dtype=torch.float64)
    factors = torch.rand(200, 1, generator=_gen(2), dtype=torch.float64)
    sem = tuning_curves(rates, factors, n_bins=2).sem
    assert sem.flatten().tolist() == pytest.approx([1e-4, 1e-4], rel=0.3)  # 1e-3 / sqrt(100)


def test_tuning_curves_use_one_bin_per_level_for_an_integer_latent():
    factors = torch.tensor([0, 1, 2] * 10)[:, None]
    rates = factors.float() * 2 + 1
    curves = tuning_curves(rates, factors, n_bins=10)
    assert curves.mean.shape == (1, 1, 3)  # not 10 bins with 7 empty
    assert curves.mean[0, 0].tolist() == [1.0, 3.0, 5.0]
    assert curves.centres[0].tolist() == [0.0, 1.0, 2.0]


def test_tuning_curves_pad_a_latent_with_fewer_levels_with_nan():
    factors = torch.stack([torch.tensor([0, 1] * 6), torch.tensor([0, 1, 2] * 4)], dim=1)
    curves = tuning_curves(torch.ones(12, 1), factors, n_bins=4)
    assert curves.mean.shape == (1, 2, 3)
    assert curves.mean[0, 0].isnan().tolist() == [False, False, True]
    assert curves.count[0].tolist() == [6.0, 6.0, 0.0]


def test_tuning_curves_reject_bad_inputs():
    with pytest.raises(ValueError, match="samples"):
        tuning_curves(torch.zeros(10, 2), torch.zeros(9, 1), n_bins=2)
    with pytest.raises(ValueError, match="n_bins"):
        tuning_curves(torch.zeros(10, 2), torch.zeros(10, 1), n_bins=1)
    with pytest.raises(ValueError, match="NaN"):
        tuning_curves(torch.full((10, 2), torch.nan), torch.zeros(10, 1), n_bins=2)


MIR = np.array([0.5, 0.9, 0.1, 0.7])


def test_select_neurons_orders_by_mir():
    assert list(select_neurons(MIR, 2, "mir_ascending", seed=0)) == [2, 0]
    assert list(select_neurons(MIR, 2, "mir_descending", seed=0)) == [1, 3]


def test_select_neurons_random_is_reproducible_and_distinct():
    a = select_neurons(MIR, 3, "random", seed=4)
    assert list(a) == list(select_neurons(MIR, 3, "random", seed=4))
    assert len(set(a)) == 3
    assert set(a) <= set(range(4))


def test_select_neurons_accepts_a_tensor_that_requires_grad():
    mir = torch.tensor([0.5, 0.9], requires_grad=True)
    assert list(select_neurons(mir, 1, "mir_descending", seed=0)) == [1]


@pytest.mark.parametrize("n", [5, 0, -1])
def test_select_neurons_rejects_a_count_outside_one_to_n(n):
    with pytest.raises(ValueError, match="between 1 and the 4"):
        select_neurons(MIR, n, "random", seed=0)


def test_select_neurons_rejects_bad_order_and_nan():
    with pytest.raises(ValueError, match="order"):
        select_neurons(MIR, 2, "sideways", seed=0)
    with pytest.raises(ValueError, match="NaN"):
        select_neurons(np.array([0.1, np.nan]), 1, "mir_ascending", seed=0)
