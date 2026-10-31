"""A training monitor that records the modularity index (MIR) of every hidden layer.

`make_mir_monitor` binds a dataset and a config into a function `net -> dict` that
`fct.train.train` calls during training. Each value is a list over hidden layers, so the
history holds, per key, a list over monitor calls of lists over layers. The monitor always
sees a single network: `train` calls it on every member of an ensemble (net.B > 1), which
puts a member axis between the calls and the layers.
"""

from collections.abc import Callable

import torch

from fct.activations import get_phi
from fct.config import Config
from fct.model import MLP
from fct.modularity import LayerMIRReport, layer_mir_report
from fct.tasks import Dataset

MIR_MONITOR_KEYS = LayerMIRReport._fields  # one source of truth for the history keys


def make_mir_monitor(dataset: Dataset, cfg: Config) -> Callable[[MLP], dict[str, list]]:
    """Monitor returning, for each key of `MIR_MONITOR_KEYS`, a list over hidden layers:

    - `mir`: raw plug-in MIR (floats);
    - `mir_corrected`: shuffle-bias-corrected MIR (floats);
    - `mir_chance`: the raw MIR's chance level, `shuffled_mir` of the layer (floats);
    - `n_dead`, `n_dead_corrected`: dead-neuron counts of the raw and corrected estimates (ints).

    In the training history these appear as `monitor_<key>` (list over calls of lists over
    layers) next to `monitor_epoch`; see `fct.train.train`. Rates are phi(H) of every layer
    with phi from `cfg.net`; factors and groups come from the dataset (ValueError if it has
    no factors) and bins, shuffles and seed from `cfg.train` (`monitor_bins`,
    `monitor_shuffles`, `seed`). A call is a deterministic function of the net: the shuffles
    use a local seeded generator, so the global RNG is untouched.
    """
    if dataset.factors is None:
        raise ValueError("the MIR monitor needs dataset.factors; this task defines none")
    phi = get_phi(cfg.net.phi)
    tc = cfg.train

    def monitor(net: MLP) -> dict[str, list]:
        with torch.no_grad():
            rates = phi(net(dataset.x)[0])  # (P, L, N)
            reports = [
                layer_mir_report(
                    rates[:, layer],
                    dataset.factors,
                    dataset.groups,
                    tc.monitor_bins,
                    tc.monitor_shuffles,
                    tc.seed,
                )
                for layer in range(rates.shape[1])
            ]
        return {key: [getattr(report, key) for report in reports] for key in MIR_MONITOR_KEYS}

    return monitor
