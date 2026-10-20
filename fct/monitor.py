"""A training monitor that records the modularity index (MIR) of every hidden layer.

`make_mir_monitor` binds a dataset and a config into a function `net -> dict` that
`fct.train.train` calls during training. Each value is a list over hidden layers, so the
history holds, per key, a list over monitor calls of lists over layers.
"""

from collections.abc import Callable

import torch

from fct.activations import get_phi
from fct.config import Config
from fct.model import MLP
from fct.modularity import network_mir, shuffled_mir
from fct.tasks import Dataset


def make_mir_monitor(dataset: Dataset, cfg: Config) -> Callable[[MLP], dict[str, list]]:
    """Monitor returning, per hidden layer: the raw MIR (`mir`), the shuffle-corrected MIR
    (`mir_corrected`), the chance level of the raw MIR (`mir_chance`) and the dead-neuron
    counts of the raw and corrected estimates (`n_dead`, `n_dead_corrected`).

    Factors, groups, bin count, shuffle count and seed come from the dataset and
    `cfg.train`; the result is a deterministic function of the net (every shuffle uses a
    local seeded generator, so the global RNG is untouched).
    """
    if dataset.factors is None:
        raise ValueError("the MIR monitor needs dataset.factors; this task defines none")
    factors, groups = dataset.factors, dataset.groups
    phi = get_phi(cfg.net.phi)
    tc = cfg.train
    settings = (factors, groups, tc.monitor_bins)

    def monitor(net: MLP) -> dict[str, list]:
        with torch.no_grad():
            rates = phi(net(dataset.x)[0])  # (P, L, N)
            raw = network_mir(rates, *settings, "none", tc.monitor_shuffles, tc.seed)
            corrected = network_mir(rates, *settings, "shuffle", tc.monitor_shuffles, tc.seed)
            chance = [
                shuffled_mir(
                    rates[:, layer], *settings, seed=tc.seed, n_shuffles=tc.monitor_shuffles
                )
                for layer in range(rates.shape[1])
            ]
        return {
            "mir": [m.mir for m in raw],
            "mir_corrected": [m.mir for m in corrected],
            "mir_chance": chance,
            "n_dead": [m.n_dead for m in raw],
            "n_dead_corrected": [m.n_dead for m in corrected],
        }

    return monitor
