"""Tuning curves and neuron selection on synthetic layers with known ground truth."""

import numpy as np
import pytest
import torch

from fct.tuning import select_neurons, tuning_curves

P, N, K, BINS = 4000, 5, 3, 8


@pytest.fixture
def layer():
    """Neuron 0 is an increasing function of latent 1 plus noise; every other neuron is noise."""
    g = torch.Generator().manual_seed(0)
    factors = torch.rand(P, K, generator=g)
    rates = 0.1 * torch.randn(P, N, generator=g)
    rates[:, 0] += 3 * factors[:, 1] ** 2
    return rates, factors


def test_monotone_in_driving_latent_and_flat_elsewhere(layer):
    curves = tuning_curves(*layer, n_bins=BINS)
    assert curves.mean.shape == (N, K, BINS)
    assert curves.sem.shape == (N, K, BINS)
    assert curves.centres.shape == (K, BINS)
    driven = curves.mean[0, 1]
    assert (driven[1:] > driven[:-1]).all()
    assert driven[-1] - driven[0] > 2
    for latent in (0, 2):  # a latent independent of the neuron: flat within noise
        spread = curves.mean[0, latent].max() - curves.mean[0, latent].min()
        assert spread < 5 * curves.sem[0, latent].max()


def test_bins_are_equal_count_and_centres_increase(layer):
    curves = tuning_curves(*layer, n_bins=BINS)
    assert (curves.count == P // BINS).all()
    assert (curves.centres[:, 1:] > curves.centres[:, :-1]).all()


def test_standard_error_matches_direct_computation():
    factors = torch.arange(8.0)[:, None]
    rates = torch.tensor([[1.0, 3.0, 2.0, 2.0, 0.0, 4.0, 5.0, 5.0]]).T
    curves = tuning_curves(rates, factors, n_bins=2)
    first = rates[:4, 0]
    assert curves.mean[0, 0, 0] == pytest.approx(first.mean().item())
    assert curves.sem[0, 0, 0] == pytest.approx((first.std() / 2).item())
    assert curves.centres[0, 0] == pytest.approx(1.5)


def test_tuning_curves_rejects_bad_shapes():
    with pytest.raises(ValueError, match="samples"):
        tuning_curves(torch.zeros(10, 2), torch.zeros(9, 1), n_bins=2)
    with pytest.raises(ValueError, match="n_bins"):
        tuning_curves(torch.zeros(10, 2), torch.zeros(10, 1), n_bins=1)
    with pytest.raises(ValueError, match="NaN"):
        tuning_curves(torch.full((10, 2), torch.nan), torch.zeros(10, 1), n_bins=2)


MIR = np.array([0.5, 0.9, 0.1, 0.7])


def test_select_by_mir_order():
    assert list(select_neurons(MIR, 2, "mir_ascending", seed=0)) == [2, 0]
    assert list(select_neurons(MIR, 2, "mir_descending", seed=0)) == [1, 3]


def test_random_selection_is_reproducible_and_distinct():
    a = select_neurons(MIR, 3, "random", seed=4)
    assert list(a) == list(select_neurons(MIR, 3, "random", seed=4))
    assert len(set(a)) == 3
    assert set(a) <= set(range(4))


def test_select_errors():
    with pytest.raises(ValueError, match="only 4 neurons"):
        select_neurons(MIR, 5, "random", seed=0)
    with pytest.raises(ValueError, match="order"):
        select_neurons(MIR, 2, "sideways", seed=0)
    with pytest.raises(ValueError, match="NaN"):
        select_neurons(np.array([0.1, np.nan]), 1, "mir_ascending", seed=0)
