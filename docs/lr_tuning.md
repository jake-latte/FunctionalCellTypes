# Learning-rate tuning of the modularity sweep

Item: `docs/issues/tune-learning-rate.md`. Code: `fct/lrtune.py` (scoring, selection, runner, report),
`fct/lrplot.py` (figures), `tests/test_lrtune.py`. Results table: `docs/lr_tuning_results.csv`
(one row per setting x lr x seed x run length; 432 rows). Raw runs and logs: `tmp/lr_tune*/` (git-ignored).

## Result in one paragraph

**Recommended: one global learning rate, `lr = 1e-3`, kept at the `full` preset's 15000 epochs (no change
to the preset's `lr`).** Nothing diverged anywhere in 1e-5..3e-2 (Adam, full batch). The learning rate
only decides how fast a run reaches its converged solution: at 15000 epochs lr = 3e-4, 1e-3 and 3e-3 reach
the same regularised loss in every one of the 24 settings (worst gap 0%), and a 60000-epoch run at 1e-4
lands on the same numbers as 1e-3 does in 6000. So a single value is enough; per-task or per-depth values
are not warranted and no `SweepConfig` schema change is needed. **The larger finding is not about the
learning rate: with the library's `weight_lambda = 1e-3` and `rate_lambda = 1e-3`, 15 of the 24 tuned
settings converge to the all-zero network (no task learned at all).** See "What the nets converge to".
That needs a decision before the full run.

## Setup actually used

- Subset of the 36-setting grid: every task x L in {1, 5} x phi in {tanh, relu} x rate_lambda in {0, 1e-3}
  = 24 settings (L = 3 not run). Each trial is a real sweep: one ROOT per (lr, seed), `SweepConfig` copied
  from the `full` preset with N, P, n_epochs, log/save/monitor intervals overridden, trained by
  `fct.sweep.run_setting`, so tuning goes through the production code path. float32, Adam, full batch,
  `weight_lambda = 1e-3` (library default), task parameters of the `full` preset.
- N = 256, P = 4096 (the `full` preset's values). Round 1: 6000 epochs, lr in
  {1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2} (the planned 1e-4..3e-2 grid plus two steps down, because
  the held-out criterion put its best value at the lower edge), seed 0, 192 runs. Round 2: seeds 1 and 2
  (net, train and data seed all set to the seed) for lr in {3e-4, 1e-3, 3e-3} at 6000 epochs (144 runs); the
  same three lrs at 15000 epochs, seed 0 (72 runs); one 60000-epoch probe at 1e-4, seed 0 (24 runs).
  MIR monitor on, 4 calls per run (every n_epochs/4, plus the final one), 10 bins, 5 shuffles.
- Held-out score: each trained net is evaluated on a fresh dataset of the same task, same P, data seed
  = training data seed + 1000, built in memory (`heldout_dataset`; the training cache is not touched, and a
  test checks that). The generators draw the task definition (the linear teacher's W_X, W_Y; the what-where
  motif set) from the same stream as the samples, so a new seed alone would be a different task: the helper
  re-expresses the new samples with the training definition. `null_mse` is the held-out MSE of the zero
  network.
- Selection (`fct/lrtune.py`): within each setting rank the lrs on a column (lower is better; MSEs closer
  than 0.1% are tied, ties share the mean rank, a diverged or non-finite run ranks last and has an infinite
  gap); a group's lr is the one with the lowest mean rank (then median gap) among the lrs with the fewest
  failures, ties are listed. "Single lr suffices" = chosen lr has no failure, median relative gap to each
  setting's best lr <= 10% and no setting more than 2x worse. Seeds are collapsed to the median.
- GPUs: T1000 (`GPU-71887a86-...`) and RTX 4500 (`GPU-e058dce4-...`, total memory on the card stayed below 2.5 GB with the PI's
  idle process), pinned by UUID; every run logs `torch.cuda.get_device_name(0)`. 2-5 processes ran in
  parallel, so the timings below include contention.

### Timings (N = 256, P = 4096, one setting)

Measured alone first (tanh, rate 1e-3, L = 5, 3000 epochs): T1000 28-32 s for the three tasks, RTX 4500
6.5-8 s (about 10 ms and 2.3 ms per epoch). Medians in the actual rounds (several processes per GPU):

| run | RTX 4500, L=1 | RTX 4500, L=5 | T1000, L=1 | T1000, L=5 |
|---|---|---|---|---|
| 6000 epochs | 8 s | 28 s | 9 s | 58 s |
| 15000 epochs | 19 s | 49 s | 23 s | 142 s |
| 60000 epochs | 72 s | 267 s | - | - |

The first six lrs of round 1 (144 runs) took about 40 min wall clock on both GPUs; everything (all rounds and
the probe) about 3 h. The 15000-epoch runs are 2.5x the cost of a 6000-epoch one. For the cluster: the full 36-setting
sweep at 15000 epochs is about 36 x (19..49 s) of RTX 4500 time, minutes per setting, so the learning-rate
check was cheap relative to the run it protects.

## What the nets converge to (read this before the learning rate)

For each setting the held-out MSE is compared with `null_mse`. At lr = 1e-3, 15000 epochs, **9 of 24
settings learn** (held-out MSE below 0.9 x null): linear teacher with rate_lambda = 0 (4 settings),
object-action with rate_lambda = 0 (4) and object-action relu L=1 with rate 1e-3. **The other 15 end at
exactly the zero network** (held-out MSE = `null_mse` to 4 digits, loss = 0.5 N mean(y^2)): linear teacher
with rate 1e-3 (4), object-action with rate 1e-3 except relu L=1 (3), and all 8 what-where settings. The converged regularised loss is the same for
every lr >= 1e-3 (e.g. 15.06 for what-where, 36.26 for object-action with rate 1e-3, 42.55 for linear teacher
with rate 1e-3), and a 60000-epoch run at 1e-4 ends at the same values, so this is where the objective's
minimum (or its attractor) is, not a learning-rate effect. Likely cause: the weight penalty
`weight_lambda * sum(W^2)` (about 25000 x 1e-3 = 25 at initialisation) and the rate penalty, which sums over
all P x N entries of every layer, are of the order of the data term `0.5 N mse` (about 15-40) or larger.
I did not vary the regularisers (they are not sweep fields).

Consequences: (1) for the 15 collapsed settings the learning rate cannot be judged by what is learned;
(2) the corrected MIR of a collapsed net is still non-zero for what-where (up to 0.57),
read off activations that are tiny but still rank-structured, and exactly 0 in the dead layers
(`n_dead`); the notebook's MIR curves for these settings describe nets that output zero; (3) the
specified criterion (held-out MSE) is misleading here, see the next section.

## Choosing the learning rate

### Held-out MSE, as specified, is confounded by collapse

Round 1, all 24 settings, ranked on held-out MSE (mean rank over settings; 1e-5 .. 3e-2):
4.69, 3.92, 4.02, 4.38, 4.54, 4.69, 4.81, 4.96; best 3e-5. Nearly flat, because 15 of 24 settings end
at the same zero network for every lr that has converged, and the transient before the collapse
is better than the collapsed end state: for what-where the lowest held-out MSE sits at the *smallest* lrs
(0.112 at 1e-4, L = 1) simply because those runs have not collapsed yet (first figure; the grey line marks
the group's "best", which is an artefact there). A 60000-epoch run at 1e-4 reaches the zero network too.
Selecting on this criterion alone would pick an under-trained net.

### Total regularised training loss (the objective) and held-out MSE on settings that learn

Mean rank / median gap by task and depth, round 1 (6000 epochs, seed 0), **ranked on the final regularised
training loss**:

| task | L | 1e-5 | 3e-5 | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 | 3e-2 | best |
|---|---|---|---|---|---|---|---|---|---|---|
| linear_teacher | 1 | 8.0 / 490% | 7.0 / 326% | 6.0 / 120% | 5.0 / 21% | 3.2 / 1% | 2.2 / 0% | 2.2 / 0% | 2.2 / 0% | 3e-3/1e-2/3e-2 |
| linear_teacher | 5 | 8.0 / 1468% | 7.0 / 1094% | 6.0 / 538% | 5.0 / 81% | 3.1 / 2% | 1.9 / 0% | 2.5 / 0% | 2.5 / 0% | 3e-3 |
| object_action | 1 | 8.0 / 91% | 7.0 / 75% | 6.0 / 36% | 5.0 / 4% | 2.5 / 0% | 2.5 / 0% | 2.5 / 0% | 2.5 / 0% | 1e-3 .. 3e-2 tied |
| object_action | 5 | 8.0 / 866% | 7.0 / 703% | 6.0 / 348% | 5.0 / 40% | 2.5 / 0% | 2.5 / 0% | 2.5 / 0% | 2.5 / 0% | 1e-3 .. 3e-2 tied |
| what_where | 1 | 8.0 / 236% | 7.0 / 182% | 6.0 / 80% | 5.0 / 10% | 2.5 / 0% | 2.5 / 0% | 2.5 / 0% | 2.5 / 0% | 1e-3 .. 3e-2 tied |
| what_where | 5 | 8.0 / 1857% | 7.0 / 1514% | 6.0 / 749% | 5.0 / 90% | 2.9 / 0% | 2.4 / 0% | 2.4 / 0% | 2.4 / 0% | 3e-3/1e-2/3e-2 |

All settings: mean rank 8.0, 7.0, 6.0, 5.0, 2.79, 2.33, 2.44, 2.44 for the eight lrs; worst gap 5% at 1e-3, 0%
at 3e-3, but 49% at 1e-2 and 50% at 3e-2 (relu, L = 5, linear teacher, rate 0 falls into a worse solution:
held-out MSE 0.083 against 0.0016, a 50-fold loss; see the first figure's top-right panel). So the useful
range is 1e-3..3e-3: below it the runs are not converged after 6000 epochs, above it one setting
degrades. Figure: `docs/figures/lr_tuning_loss_6000.png` (loss vs lr, log y; all curves are flat from 1e-3
and rise towards small lr).

Held-out MSE restricted to the 9 settings that learn (`--learned-only`), same round:
mean rank 8.0, 7.0, 4.94, 3.83, **2.50**, 2.89, 3.22, 3.61 (best 1e-3, worst gap 21% from the relu-L5 setting
above at 1e-3 vs the 3e-4 optimum, about 0.0017 vs 0.0014 in absolute terms).

### Seeds (3 candidates x 3 seeds, 6000 epochs)

Ranking on the loss is identical in all three seeds: mean rank 3e-4 3.00, 1e-3 1.62, 3e-3 1.38 (median gap
30% / 0% / 0%). Held-out MSE over all 24 settings is noise (best lr differs by seed, all three within 0.04 in
mean rank); restricted to the 9 learning settings the best is 1e-3 (seed 0) and 3e-3 (seeds 1, 2) with
mean ranks 1.72-1.89 for the two, 2.3-2.4 for 3e-4. So 1e-3 vs 3e-3 is not resolved at 6000 epochs; 3e-4 is
clearly worse (unconverged). Median over seeds: single lr 3e-3, median gap 0%, worst gap 21%,
suffices = True.

### 15000 epochs (the `full` preset's length), seed 0, lr in {3e-4, 1e-3, 3e-3}

| criterion | 3e-4 | 1e-3 | 3e-3 | worst gap |
|---|---|---|---|---|
| loss: mean rank over 24 settings | 2.08 | 1.96 | 1.96 | 0% |
| held-out MSE: mean rank over 24 settings | 1.94 | 1.98 | 2.08 | 2% (3e-3) |
| held-out MSE, 9 learning settings | 1.83 | 1.94 | 2.22 | 2% (3e-3) |

All three are converged (final-10% loss drop <= 0.7% everywhere, 0.0% at 1e-3 and 3e-3), nothing failed, the
largest loss spike is 1.00. The only visible preference is a slightly lower held-out MSE for the smaller lrs in
linear-teacher L = 5 (at most 2%). Verdict "single lr suffices": True for every column. Figure:
`docs/figures/lr_tuning_loss_15000.png`.

### Stability and convergence flags

No run diverged (non-finite loss) in any round and no run had a loss spike (max ratio to the best loss since
the first quarter of training: 1.00; 1.01 at 3e-2). At 6000 epochs the loss was still falling (final-10% drop
> 1%) in 24/24 runs at 1e-4 and 3e-4, 4/24 at 1e-3, 0/24 at lr >= 3e-3; at 15000 epochs 0/24 at 3e-4 .. 3e-3.
The failure markers in the held-out figure are therefore never drawn; a failure shows up instead as
convergence to a worse solution at lr >= 1e-2 (relu L = 5 linear teacher), which the rank/gap logic does
penalise.

## Recommendation

- Keep **one global `lr = 1e-3`** at 15000 epochs (the `full` preset already has `lr = 1e-3`, so no
  preset change is needed for the learning rate). Rationale: converged in all 24 settings with the same loss
  as the best lr, 10x below the lr (1e-2) at which one setting goes to a worse solution, and no
  gap to the best lr above 2% on either the loss or the held-out MSE of the settings that learn.
- If the epochs are cut to about 6000, use 3e-3 instead: 1e-3 is then still falling in 4 of 24 runs.
  Do not exceed 3e-3.
- Per-task and per-depth values: not warranted (the loss ranking has 3e-3 and 1e-3 at or next to the top in every task and depth; the loss gap at 3e-3 is 0%
  in every group and at 1e-3 at most 5%). The `SweepConfig` schema stays single-`lr`.
- Not tested: L = 3; the missing 12 settings of the grid; other N, P; another optimiser.

## Open question for the PI (not for this item to decide)

The 15 collapsed settings are an objective-scale issue, not a learning-rate issue. Options: decide whether
the intended regime is one where the net learns (then the regulariser scales have to be re-chosen: for example
`weight_lambda` and `rate_lambda` smaller, or the rate penalty averaged over P instead of summed, since a
sum over P x N x L grows with the data set), or whether the collapsed nets are the intended point (then the
notebook should say so). Changing it means new fields in the sweep config and a re-check of the learning
rate afterwards (the speed of collapse and learning both depend on it). Suggest a separate issue.

## Details worth knowing

- At L = 1 the net's output is linear in the hidden pre-activation, so phi enters only through the rate
  penalty: tanh and relu coincide for rate_lambda = 0 (thick and thin lines overlap in the figures).
- `docs/lr_tuning_results.csv` columns: setting (task, L, phi, rate_lambda), lr, seed, n_epochs, diverged,
  final train MSE, held-out MSE, `null_mse`, final regularised loss, `max_spike`, `tail_drop` (relative loss
  drop over the last 10% of the logged curve; > 0.01 means still improving), final corrected MIR per layer
  (';'-joined; context only, not used to choose).
- Reproduce: `python -m fct.lrtune init BASE --N 256 --P 4096 --epochs 6000`;
  `python -m fct.lrtune run BASE --lrs ... [--seeds ...] [--tasks ...] [--layers ...]` (resumable);
  `python -m fct.lrtune collect BASE --lrs ... --csv out.csv`; `python -m fct.lrtune report out.csv
  [--key loss|heldout_mse|train_mse] [--learned-only] [--seeds ..] [--n-epochs ..] [--fig-dir DIR]`.
- One 3e-4 / 15000-epoch setting (linear teacher, L=5, tanh, rate 0) was interrupted when I re-split the
  jobs between the GPUs and resumed from its checkpoint; resuming reproduces the run, but its `timings.csv`
  entry covers only the resumed part.

## What I looked at in the figures, and what I fixed

Held-out MSE vs lr (`docs/figures/lr_tuning_heldout_6000.png`): the first version used log-log axes, which
gave no tick labels for the narrow what-where range and overlapping tick text; it now uses linear y, a
minimum 5% y-range (so that 1e-6 differences are not stretched to fill a panel, which I saw in the 15000-epoch
version), explicit lr ticks and a taller layout. Checked: tanh/relu overlap at L = 1 handled by line width;
the relu-L5 jump at lr >= 1e-2 in linear teacher; the non-monotone what-where curve and its flat tail at the
null level; the loss figure shows the plateau from 1e-3. The failure cross marker was tested on synthetic
tables only, because no real run failed.
