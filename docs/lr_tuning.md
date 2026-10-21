# Learning-rate tuning of the modularity sweep

Item: `docs/issues/tune-learning-rate.md`. Code: `fct/lrtune.py`, `fct/lrplot.py`, `fct.tasks.resample`
(interfaces in `docs/design.md`). Results table: `docs/lr_tuning_results.csv`. Raw runs and logs were
written to `tmp/lr_tune*/` (git-ignored).

## Recommendation

**Provisional.** All tuning was done in the collapsed regime: with the library regularisers
(`weight_lambda = 1e-3`, `rate_lambda in {0, 1e-3}`) 15 of the 24 tuned settings converge to the all-zero
network, so they carry no information on the learning rate needed to learn anything. L = 3 and the
other 12 grid settings were not run, and lr = 1e-4 was not run at 15000 epochs (only at 6000 and 60000).
The PI accepted lr = 1e-3 as good enough and asked for no further tuning. The recommendation stands only
for the regulariser scales that were tested; if they change, the learning rate should be re-checked.

- Use one global `lr = 1e-3` with 15000 epochs (the `full` preset already has it; no preset change).
- At 15000 epochs lr = 3e-4, 1e-3 and 3e-3 are tied: the regularised loss agrees within 0.2% in every
  setting and the held-out MSE within 2%. 1e-3 is chosen by the tie-break (the lowest lr among the tied
  ones on the loss) and by the PI's acceptance, not because it wins: on held-out MSE the 15000-epoch report
  ranks 3e-4 marginally first.
- If the epochs are cut to about 6000, use 3e-3: at 6000 epochs 1e-3 is still falling in 4 of 24 runs.
  Staying at or below 3e-3 rests on a single seed at 1e-2 (see Results).
- No per-task or per-depth value is warranted, so `SweepConfig` keeps its single `lr`.

## Setup

- Subset of the 36-setting grid: every task x L in {1, 5} x phi in {tanh, relu} x rate_lambda in {0, 1e-3}
  = 24 settings. Each trial is a real sweep: one ROOT per (lr, seed), `SweepConfig` copied from the `full`
  preset with N, P, n_epochs and the log/save/monitor intervals overridden, trained by
  `fct.sweep.run_setting`. float32, Adam, full batch, `weight_lambda = 1e-3`, task parameters of the `full`
  preset, N = 256, P = 4096. MIR monitor on (4 calls per run plus the final one).
- Runs: 6000 epochs at seed 0 for lr in {1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2} (192 runs; the
  planned grid started at 1e-4, 1e-5 and 3e-5 were added because the held-out criterion put its best lr at
  that edge); 6000 epochs at seeds 1 and 2 for {3e-4, 1e-3, 3e-3} (144 runs; net, train and data seed all set
  to the seed); 15000 epochs at seed 0 for the same three (72 runs); one 60000-epoch probe at 1e-4 (24 runs).
- Held-out score: a fresh dataset of the same task and P, data seed = training data seed + 1000, built in
  memory by `fct.tasks.resample` (the training cache is not touched). The generators draw the task definition
  (the linear teacher's W_X, W_Y; the what-where motif set) from the same stream as the samples, so
  `resample` re-expresses the new samples with the training definition. `null_mse` is the held-out MSE of
  the zero network.
- Selection (`fct/lrtune.py`): within each setting, rank the lrs on a column (lower is better). Values within
  0.1% (`TIE_RTOL`) are tied and share the mean rank; a diverged or non-finite run ranks last with an
  infinite gap. A group's lr is the one with the lowest mean rank, then the lowest median gap, among the lrs
  with the fewest failures; exact ties are listed and the reported "best" is the lowest tied lr. Seeds are
  collapsed to the median. "Single lr suffices": chosen lr has no failure, the median relative gap to each
  setting's best lr is at most 10%, and no setting is more than 2x worse (`--max-median-gap`, `--max-ratio`).
- "Learns": a setting has learned if, at its longest run length, some lr gives a held-out MSE below 0.9 x
  `null_mse` (`--learned-only`).
- Timings (N = 256, P = 4096; medians with 2-5 processes per GPU; T1000 and RTX 4500):

  | run | RTX 4500, L=1 | RTX 4500, L=5 | T1000, L=1 | T1000, L=5 |
  |---|---|---|---|---|
  | 6000 epochs | 8 s | 28 s | 9 s | 58 s |
  | 15000 epochs | 19 s | 49 s | 23 s | 142 s |
  | 60000 epochs | 72 s | 267 s | - | - |

  Measured alone (L = 5, 3000 epochs): T1000 28-32 s, RTX 4500 6.5-8 s for the three tasks. The first six
  lrs of the 6000-epoch round (144 runs) took about 40 min on both GPUs; everything took about 3 h.

## Results

How to read the tables. The shipped CSV has two run lengths (6000, 15000, plus the 60000 probe), so
`report` needs `--n-epochs` (otherwise it stops with "duplicate result row ... more than one run length").
Table cells are `mean rank / median gap` of an lr within a group; gaps are relative to each setting's best.
The default `--key` is the regularised loss.

### What the nets converge to

At lr = 1e-3, 15000 epochs, 9 of 24 settings learn (held-out MSE below 0.9 x `null_mse`): linear teacher
with `rate_lambda = 0` (4 settings), object-action with `rate_lambda = 0` (4) and object-action relu L=1
with `rate_lambda = 1e-3`. The other 15 end at the zero network (held-out MSE = `null_mse` to 4 digits,
loss = 0.5 N mean(y^2)): linear teacher with `rate_lambda = 1e-3` (4), object-action with
`rate_lambda = 1e-3` except relu L=1 (3), and all 8 what-where settings. Their converged losses are the same
for every lr >= 1e-3 (15.06 what-where, 36.26 object-action, 42.55 linear teacher with rate 1e-3), and the
60000-epoch run at 1e-4 ends at the same values as 1e-3 at 6000 epochs. Per-setting values:
`python -m fct.lrtune report docs/lr_tuning_results.csv --n-epochs 15000 --lrs 0.001` and the CSV.

The corrected MIR of a collapsed net is not zero: for what-where at lr = 1e-3, 15000 epochs it is up to
0.60 (L=5, relu, rate 1e-3, last layer; 0.67 at 3e-3), read off tiny but rank-structured activations;
8 of the 72 what-where layers (three lrs, 15000 epochs) have MIR exactly 0 (no information about any
factor). `n_dead` is not a CSV column.

### 6000 epochs, seed 0, 8 learning rates

Recipe for every table in this subsection: `python -m fct.lrtune report docs/lr_tuning_results.csv
--n-epochs 6000 --seeds 0` plus the flags named.

*Regularised loss* (`--key loss`), by task and depth:

| task | L | 1e-5 | 3e-5 | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 | 3e-2 | best |
|---|---|---|---|---|---|---|---|---|---|---|
| linear_teacher | 1 | 8.0 / 490% | 7.0 / 326% | 6.0 / 120% | 5.0 / 21% | 3.2 / 1% | 2.2 / 0% | 2.2 / 0% | 2.2 / 0% | 3e-3/1e-2/3e-2 |
| linear_teacher | 5 | 8.0 / 1468% | 7.0 / 1094% | 6.0 / 538% | 5.0 / 81% | 3.1 / 2% | 1.9 / 0% | 2.5 / 0% | 2.5 / 0% | 3e-3 |
| object_action | 1 | 8.0 / 91% | 7.0 / 75% | 6.0 / 36% | 5.0 / 4% | 2.5 / 0% | 2.5 / 0% | 2.5 / 0% | 2.5 / 0% | 1e-3 .. 3e-2 tied |
| object_action | 5 | 8.0 / 866% | 7.0 / 703% | 6.0 / 348% | 5.0 / 40% | 2.5 / 0% | 2.5 / 0% | 2.5 / 0% | 2.5 / 0% | 1e-3 .. 3e-2 tied |
| what_where | 1 | 8.0 / 236% | 7.0 / 182% | 6.0 / 80% | 5.0 / 10% | 2.5 / 0% | 2.5 / 0% | 2.5 / 0% | 2.5 / 0% | 1e-3 .. 3e-2 tied |
| what_where | 5 | 8.0 / 1857% | 7.0 / 1514% | 6.0 / 749% | 5.0 / 90% | 2.9 / 0% | 2.4 / 0% | 2.4 / 0% | 2.4 / 0% | 3e-3/1e-2/3e-2 |

All 24 settings: mean ranks 8.0, 7.0, 6.0, 5.0, 2.79, 2.33, 2.44, 2.44; worst gap 5% at 1e-3, 0% at 3e-3,
49% at 1e-2 and 50% at 3e-2. The 49-50% is relu, L=5, linear teacher, `rate_lambda = 0`, which ends in a
worse solution (held-out MSE 0.083 against 0.0016). Figure: `docs/figures/loss_vs_lr_6000.png`.

*Held-out MSE, all 24 settings* (`--key heldout_mse`): mean ranks 4.69, 3.92, 4.02, 4.38, 4.54, 4.69, 4.81,
4.96, best 3e-5; for what-where the lowest held-out MSE is at the smallest lrs (0.112 at 1e-4, L = 1).
Figure: `docs/figures/heldout_mse_vs_lr_6000.png` (the grey line marks the lowest tied "best" lr of the panel).
No verdict is printed for this ranking.

*Held-out MSE, the 9 settings that learn* (`--key heldout_mse --learned-only`): mean ranks 8.0, 7.0, 4.94,
3.83, 2.50, 2.89, 3.22, 3.61; best 1e-3; single-lr verdict True with worst gap 21% (relu L=5 linear teacher at
1e-3, about 0.0017 against 0.0014).

### Seeds, 6000 epochs, lr in {3e-4, 1e-3, 3e-3}

Recipe: `report docs/lr_tuning_results.csv --n-epochs 6000 --lrs 0.0003 0.001 0.003 [--seeds S]
[--key loss | --key heldout_mse --learned-only]`.

- Loss, every one of seeds 0, 1, 2 (and the median over the three): mean ranks 3e-4 3.00, 1e-3 1.62, 3e-3 1.38;
  median gaps 30%, 0%, 0%; worst gaps 154%, 4%, 0%.
- Held-out MSE over all 24 settings is noise: the best lr differs by seed and the three lrs are within 0.04
  mean rank of each other.
- Held-out MSE on the 9 learning settings: best 1e-3 (seed 0), 3e-3 (seeds 1, 2, and the median over three);
  mean ranks 1.72-1.89 for those two and 2.3-2.4 for 3e-4. So 1e-3 and 3e-3 are not resolved at 6000
  epochs; 3e-4 is clearly worse (not converged). Median over seeds: single lr 3e-3, worst gap 21%, True.

### 15000 epochs, seed 0, lr in {3e-4, 1e-3, 3e-3}

Recipe: `report docs/lr_tuning_results.csv --n-epochs 15000 --key loss` (or `--key heldout_mse`, with
`--learned-only` for the last row). Figure: `docs/figures/loss_vs_lr_15000.png`.

| criterion | 3e-4 | 1e-3 | 3e-3 | worst gap of any lr |
|---|---|---|---|---|
| loss, mean rank, 24 settings | 2.08 | 1.96 | 1.96 | 0.19% (3e-4) |
| held-out MSE, mean rank, 24 settings | 1.94 | 1.98 | 2.08 | 1.9% (3e-3) |
| held-out MSE, mean rank, 9 learning settings | 1.83 | 1.94 | 2.22 | 1.9% (3e-3) |

Worst gaps per lr: loss 0.19% / 0.0002% / 0.016%; held-out MSE 0.09% / 0.30% / 1.9%. Selected lr: loss,
24 settings: 1e-3 (tied with 3e-3); held-out MSE: 3e-4. Verdict "single lr suffices" is True for every row.
All three are converged (final-10% loss drop at most 0.7%, 0.0% at 1e-3 and 3e-3).

### Stability and convergence

No run diverged in any round and no loss spike occurred (maximum ratio of a logged loss to the best loss since
the first quarter of the run: 1.00; 1.01 at 3e-2). At 6000 epochs the loss was still falling (final-10% drop
above 1%) in 24/24 runs at 1e-4 and 3e-4, 4/24 at 1e-3 and 0/24 at lr >= 3e-3; at 15000 epochs in 0/24 runs at
3e-4 to 3e-3. A failure appears only as a worse solution at lr >= 1e-2 (relu, L=5, linear teacher, seed 0).
The failure markers in the held-out figure were exercised on synthetic tables only.

## Interpretation

- At convergence the learning rate only changes speed. Between 1e-3 and 3e-3 the converged regularised loss agrees
  within 0.02% in all 24 settings, and 1e-4 reaches it after 60000 epochs.
- Held-out MSE is not a usable primary criterion when most settings collapse: it ranks the lrs of a collapsed
  setting by how little they have trained, so the loss and the 9 learning settings were used as well.
- Why the settings collapse (arithmetic checked on CPU for `full`-preset data, N = 256, P = 4096):
  - **Weight penalty alone collapses what-where.** At L = 1 the net is a linear map W = U^T V / (N sqrt(D_in)),
    and the minimum of U, V norms for a given W is 2 N sqrt(D_in) times its nuclear norm. The zero network
    is therefore optimal at `rate_lambda = 0` iff sigma_max((N / (P D_out)) Y^T X) <= 2 weight_lambda N
    sqrt(D_in). Values: what-where 4.25 <= 4.61 (zero is optimal, even without the rate penalty),
    object-action 12.6 > 3.6 and linear teacher 35.9 > 4.1 (both learn). Averaging the rate penalty over P
    would not rescue what-where: its weight penalty is the scale problem. (The criterion is derived for L = 1;
    L = 5 was not analysed, though the L = 5 what-where nets also collapse.)
  - **The summed rate penalty collapses linear teacher and object-action at `rate_lambda = 1e-3`.** At
    initialisation (tanh, L = 1) the rate term is 211 for linear teacher against a data term of 46 (object-action
    47 against 36). It sums over P x N x L entries, so it grows with the data set.
  - **The object-action relu L=1 `rate_lambda = 1e-3` "learner" is not evidence that the rate penalty is
    benign.** At L = 1 the readout uses the pre-activation H, not phi(H); relu units can go negative, where
    phi(H) = 0, so the rate penalty can vanish at no cost to the output. Its loss (31.37) equals the
    `rate_lambda = 0` loss.
- 1e-3 versus 3e-3 versus 3e-4 cannot be separated at 15000 epochs (above). The only visible trend is a
  slightly lower held-out MSE for smaller lrs in linear teacher L = 5 (at most 2%).
- At L = 1 phi acts only through the rate penalty, so tanh and relu coincide for `rate_lambda = 0` (thick and
  thin lines overlap in the figures).

## Limits and open questions

- Only 9 of 24 tuned settings learn; L = 3 and the remaining 12 grid settings were not run; 1e-4 not run at
  15000 epochs; the lr ceiling (between 3e-3 and 1e-2) rests on one seed at 1e-2; other N and P and other
  optimisers were not tested.
- Whether the regulariser scales will change, and what that does to the learning rate, is open (see the issue
  file).
- Reproduce the runs: `python -m fct.lrtune init BASE --N 256 --P 4096 --epochs 6000`, then
  `python -m fct.lrtune run BASE --lrs ... [--seeds ...] [--tasks ...] [--layers ...]` (resumable),
  `python -m fct.lrtune collect BASE --lrs ... [--seeds ...] --csv out.csv`.
- CSV columns: setting (task, L, phi, rate_lambda), lr, seed, n_epochs, diverged, final train MSE, held-out
  MSE, `null_mse`, final regularised loss, `max_spike`, `tail_drop` (relative loss drop over the last 10% of the
  logged curve; above 0.01 means still improving), final corrected MIR per layer (';'-joined; context only).
