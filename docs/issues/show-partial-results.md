---
type: data-science
complexity: simple
status: done
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Show in-progress settings in the results notebook

The 36-setting sweep runs on the cluster for a long time and the results notebook should be
buildable while jobs are still running. Until now `load_results` returned data only for finished
settings (final.pt); a running setting was reported "partial" with nothing to see.

Why: checkpoints (written atomically every `save_int` epochs) hold the history, including the
MIR monitor history up to that epoch, and the network weights, so partial progress can be shown,
as long as it can never be mistaken for a final result.

## Changes

- `fct/train.py`: `load_latest_checkpoint` (read-only; retries once if the checkpoint is replaced
  by the running job; errors name the file).
- `fct/report.py`: a partial setting with a readable checkpoint and an existing dataset file
  loads like a finished one plus `epoch` / `n_epochs` (`SettingResult.provisional`); an
  unreadable or config-mismatched checkpoint is reported in `note` with no data; `FinalMIR` flags
  provisional rows; `summary_text` computes every statistic over finished settings only and lists
  the provisional ones apart; table rows are marked `*` with a footnote; `overview_text` reads
  "12 done, 4 partial (provisional), 20 missing"; prose and figure titles say
  "partial, epoch X of N".
- `fct/draw.py`: provisional heatmap rows are hatched (and the figure says so);
  `fig_example_outputs(title=...)`.
- `notebooks/build_modularity.py`: intro text; cells unchanged.
- Tests: `tests/test_report_partial.py` (synthetic checkpoints with known contents),
  one end-to-end notebook execution on a mix of done / partial / missing settings.
