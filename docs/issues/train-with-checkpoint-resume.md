---
type: infrastructure
complexity: simple
status: in review
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Port the training loop to `fct/train.py` with checkpointing and resume

Move the `train` function of notebook cell 5 into `fct/train.py`, split into a
pure loss function and a loop that checkpoints periodically into
`CACHE_DIR/train/` and resumes from the latest checkpoint if one exists.

Why: training runs for tens of thousands of epochs; an interrupted run must
continue where it stopped, and the final model plus its history must be
loadable from the cache without the notebook.

## Inputs / outputs / interfaces

Implement exactly the *train.py* section of `docs/design.md`:
`compute_loss`, `History`, `train(cfg, dataset, net, cache_dir, log=print)`,
`load_trained(cache_dir, cfg, device, dtype)`. Checkpoints are written with
`fct.cache.write_checkpoint` (prefix `ckpt_`, step = epoch) and `final.pt`
with `save_artefact`; both carry `cfg.section_hash("data","net","train")`.
Checkpoint payload: `epoch` (next epoch to run), `net_state_dict`,
`optim_state_dict`, `history`, `rng_state` (`torch.get_rng_state()` and the
CUDA state when on CUDA). The optimiser is Adam with `cfg.train.lr`; seed the
run with `cfg.train.seed`. `log` is called every `log_int` epochs with the
notebook's one-line format; history is appended at the same cadence and
includes the `epoch` key. On resume, `log` a line saying which checkpoint was
loaded. If `final.pt` exists, load it and return without training.

## Acceptance criteria

- `compute_loss` on a hand-built 1-layer net matches
  `0.5*N*mse + λ_w*weight_l2 + λ_r*rate_l2` computed by hand.
- Synthetic ground truth: a linear task reaches train MSE below 10 % of the
  initial MSE in 300 epochs.
- Resume after an interruption at epoch 10 gives the same history and
  bit-identical parameters as an uninterrupted 20-epoch run; `train/` holds
  exactly one `ckpt_*.pt` plus `final.pt`.
- `train` with `final.pt` present runs no epochs and returns the same parameters.
- `load_trained` reproduces outputs and history; raises `StaleCacheError` on a
  changed `lr` and `FileNotFoundError` without `final.pt`.

## Implementation notes
- A checkpoint with `epoch=s` is written before the step of epoch s; the step,
  history entry and log line follow. An interrupt inside `log` at epoch s
  therefore resumes at s.
- Optimiser state is keyed by int, so it is converted to str keys on save and
  back on load; `state_dict()` is copied to a plain dict (the cache rejects
  `OrderedDict`).
