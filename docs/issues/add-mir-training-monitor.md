---
type: infrastructure
complexity: simple
status: in review
---

<!-- status: todo | in progress | in review | done | blocked | awaiting your confirmation -->

# Record the modularity index (MIR) of every hidden layer throughout training

The study needs, for each trained network, the MIR of each hidden layer as a function of training time. `train` gets an optional monitor hook; a ready-made monitor computes the MIR curves from the dataset's task factors using fct/modularity.py.

## Interfaces
- `config.TrainConfig` gains `monitor_int: NonNegInt = 0` (epochs between monitor calls; 0 disables), `monitor_bins: int >= 2 = 10`, `monitor_shuffles: PosInt = 5` (all enter the train hash, so a cached run always matches the monitor settings that produced it).
- `train.train(cfg, dataset, net, cache_dir, log=print, monitor=None)`: `monitor: Callable[[MLP], dict[str, list[float]]] | None`. It is called, under `torch.no_grad()` and without touching the global RNG or the net's gradients, at the START of epoch e (after the periodic checkpoint for e has been written, before the step) for every e with e % monitor_int == 0 (so e = 0 records the initial network), and once more after the last step with e = n_epochs (the trained net). Results go in the history as `monitor_epoch` (list[int]) and, for each key k the monitor returns, `monitor_<k>` (list, one entry per call; entries are lists over layers). Passing a monitor with monitor_int == 0, or monitor_int > 0 without a monitor, raises a ValueError (loud, no silent no-op). With monitor_int == 0 the history keys and values are exactly what they were before this item. Resuming from a checkpoint must reproduce the history of an uninterrupted run exactly, including the monitor entries (the checkpoint written at epoch e does not yet contain e's monitor entry; the resumed run records it).
- `fct/monitor.py` (new): `make_mir_monitor(dataset: Dataset, cfg: Config) -> Callable[[MLP], dict[str, list]]`. It requires `dataset.factors` (ValueError otherwise). Each call runs the net on `dataset.x`, takes the firing rates phi(H) of every layer (phi from cfg.net), and returns, per layer l = 0..L-1: `mir` (raw MIR, `network_mir` with bias_correction="none"), `mir_corrected` (bias_correction="shuffle"), `mir_chance` (`shuffled_mir` of the layer), `n_dead` and `n_dead_corrected` (ints), using `dataset.groups`, `cfg.train.monitor_bins`, `cfg.train.monitor_shuffles` and seed `cfg.train.seed` (monitor calls are deterministic functions of the net). So the history keys are `monitor_epoch`, `monitor_mir`, `monitor_mir_corrected`, `monitor_mir_chance`, `monitor_n_dead`, `monitor_n_dead_corrected`; each of the latter five is a list over calls of lists over layers (floats/ints), saveable as a weights_only artefact. Read the signatures in fct/modularity.py (layer_mir/network_mir/shuffled_mir and the bias_correction rule) before writing this.
- `python -m fct train CACHE_DIR` builds the monitor automatically when `train.monitor_int > 0`; with monitor_int == 0 nothing changes. `load_trained` returns the history unchanged (it contains the monitor keys).
- docs/design.md: document the config fields, the hook, the history keys and fct/monitor.py (and add monitor.py to the package layout list).

## Acceptance criteria (tests: CPU, tiny sizes, synthetic)
- Monitor values equal a direct `network_mir` / `shuffled_mir` call on the same net, dataset and settings, for a tiny what_where dataset and for a tiny linear_teacher dataset (hand-built factors), and the lengths/shapes are (n_calls, L).
- History: the call epochs are exactly {0, monitor_int, 2*monitor_int, ...} up to n_epochs plus the final one (also when n_epochs is not a multiple of monitor_int, and no duplicate when it is); monitor_epoch is increasing.
- Resume exactness: train n epochs uninterrupted vs. interrupted after a checkpoint and resumed (reuse the existing resume test machinery): identical history including the monitor entries, identical final weights.
- The monitor does not perturb training: final weights with monitor_int > 0 are bit-identical to monitor_int == 0 on the same config (CPU float64), and the global torch RNG state after `train` is the same either way.
- monitor_int == 0: history has exactly the previous keys; passing a monitor with 0 (or missing one with > 0) raises.
- The MIR of a network whose first-layer neurons are hand-set to read single factors is high and the corrected MIR exceeds the chance baseline (a tiny deterministic net built by assigning weights; document how it is constructed), while a freshly initialised net on shuffled factors sits near the chance level.
- CLI: `init` + `train` on a tiny config with `--set train.monitor_int=...` produces a final.pt whose history has the monitor keys; without it, none.
- Report in your final message the wall time of one monitor call for a CPU net with N = 256, L = 5, P = 4096 on the object_action task (G = 5), and for what_where (m = 8), so the PI can size the monitoring interval for the real runs.
- All existing tests pass; ruff check and format clean.
