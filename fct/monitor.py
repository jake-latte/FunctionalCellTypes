"""Training monitors: the modularity index (MIR) or the tuning-map spectra of every hidden layer.

A monitor binds a dataset and a config into a function `net -> dict` that `fct.train.train`
calls during training; `make_monitor` picks one by `cfg.train.monitor_kind`. Each value is a
list over hidden layers, so the history holds, per key, a list over monitor calls of lists over
layers. A monitor always sees a single network: `train` calls it on every member of an ensemble
(net.B > 1), which puts a member axis between the calls and the layers.
"""

from collections.abc import Callable

import torch
from torch import Tensor

from fct import spectrum as sp
from fct.activations import get_phi
from fct.config import Config
from fct.model import MLP
from fct.modularity import LayerMIRReport, layer_mir_report
from fct.tasks import Dataset
from fct.train import MONITOR_PREFIX, History, Monitor

MIR_MONITOR_KEYS = LayerMIRReport._fields  # one source of truth for the history keys
# The keys of a record at K >= 3 latents. The angle keys "hex" and "theta_hist" use the
# contrasts u_2 and u_3, so for K < 3 they are absent and a record holds "spectrum" alone. They
# are recorded for every K >= 3 but mean something only at K = 3 (see
# `fct.spectrum.hexatic_moment`: for K >= 4, (u_2, u_3) is an arbitrary plane of the contrasts).
SPECTRUM_MONITOR_KEYS = ("spectrum", "hex", "theta_hist")
_MIN_ANGLE_LATENTS = 3
# Records are kept as float32: a full-size run holds thousands of (L, K + 1, N) records, and
# float32's relative precision on each power is far below that power's sampling noise.
_RECORD_DTYPE = torch.float32


def make_mir_monitor(dataset: Dataset, cfg: Config) -> Monitor:
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


def _per_layer(x: Tensor) -> list[Tensor]:
    """(L, ...) on any device -> a list over layers of CPU tensors, floats as float32."""
    dtype = _RECORD_DTYPE if x.is_floating_point() else x.dtype
    return list(x.to(device="cpu", dtype=dtype).unbind(0))


def spectrum_record(H: Tensor, U: Tensor) -> dict[str, list[Tensor]]:
    """One record of the pre-activations H (P, L, N) in the latent frame U (P, K).

    Every value is a list over the L layers:
    - `spectrum`: (K + 1, N) float32, `fct.spectrum.mode_spectra` (rows 0..K-1 the latent
      modes, row K the null mean), q in FFT order;
    - `hex`: (2,) float32, the real and imaginary parts of `hexatic_moment` of the contrasts
      (rows 1 and 2 of the tuning maps); only for K >= 3;
    - `theta_hist`: (36,) int64, `theta_histogram` of the same contrasts; only for K >= 3.
    The complex moment and the counts, unlike |moment| or a normalised histogram, average
    and sum across records and members. The angle keys are meaningful only at K = 3.
    """
    h = H.transpose(0, 1)  # (L, P, N)
    record = {"spectrum": _per_layer(sp.mode_spectra(h, U))}
    if U.shape[1] >= _MIN_ANGLE_LATENTS:
        t = sp.tuning_maps(h, U)  # (L, K, N)
        moment = sp.hexatic_moment(t[:, 1], t[:, 2])  # (L,) complex
        record["hex"] = _per_layer(torch.stack([moment.real, moment.imag], dim=-1))
        record["theta_hist"] = _per_layer(sp.theta_histogram(t[:, 1], t[:, 2]))
    return record


def make_spectrum_monitor(dataset: Dataset, cfg: Config) -> Monitor:
    """Monitor returning `spectrum_record` of the net's pre-activations on `dataset.x`.

    The keys are those of `SPECTRUM_MONITOR_KEYS` for K >= 3 latents and `spectrum` alone
    below; in the history, `monitor_spectrum` is a list over calls of lists over layers of
    (K + 1, N) tensors (with a member list in between for an ensemble). The latent frame is
    `fct.spectrum.latent_basis(dataset.factors)`, built here, so data without a mean
    direction (symmetric latents) or without latents z0..z{K-1} raise ValueError before
    training starts, not at the first record. There is no input noise: the net sees
    `dataset.x` as it is. A call draws nothing from any RNG. `cfg` is not read; the signature
    is that of `make_mir_monitor`, so `make_monitor` builds either the same way.
    """
    if dataset.factors is None:
        raise ValueError("the spectrum monitor needs dataset.factors; this task defines none")
    U = sp.latent_basis(dataset.factors).U

    def monitor(net: MLP) -> dict[str, list]:
        with torch.no_grad():
            return spectrum_record(net(dataset.x)[0], U)

    return monitor


def spectrum_records(history: History) -> Tensor:
    """The spectra a spectrum-monitored run recorded, as one (R, B, L, K + 1, N) tensor.

    R records in `monitor_epoch` order, B members (1 for a single network, whose history has
    no member axis), L hidden layers, rows 0..K-1 the latent modes and row K the null mean,
    q in FFT order; float32, as recorded. The last record is the final state (`fct.train.train`
    monitors once more after the last step).
    """
    key = f"{MONITOR_PREFIX}spectrum"
    if key not in history:
        raise ValueError(
            f"the history has no {key!r}: it was not recorded by the spectrum monitor "
            '(train.monitor_kind = "spectrum")'
        )
    entries = history[key]
    if not entries:
        raise ValueError(f"the history's {key!r} holds no records")
    if isinstance(entries[0][0], Tensor):  # a single network: a list over layers per record
        entries = [[layers] for layers in entries]
    return torch.stack(
        [torch.stack([torch.stack(layers) for layers in members]) for members in entries]
    )


_MONITOR_MAKERS: dict[str, Callable[[Dataset, Config], Monitor]] = {
    "mir": make_mir_monitor,
    "spectrum": make_spectrum_monitor,
}


def make_monitor(dataset: Dataset, cfg: Config) -> Monitor:
    """The monitor `cfg.train.monitor_kind` names, built on `dataset`."""
    return _MONITOR_MAKERS[cfg.train.monitor_kind](dataset, cfg)


_RECORDS = {"mir": "MIR", "spectrum": "spectra"}  # what each monitor kind's history holds


def require_mir_monitor(monitor_kind: str, reader: str) -> None:
    """ValueError unless `monitor_kind` is "mir": `reader` (a sentence subject, such as "the
    report") reads the MIR monitor's history keys, which no other monitor writes."""
    if monitor_kind != "mir":
        raise ValueError(
            f"{reader} reads the MIR monitor's history, but this experiment records "
            f"{_RECORDS[monitor_kind]}, not MIR (monitor_kind={monitor_kind!r})"
        )
