# FunctionalCellTypes: status history, 2026-10-20 to 2026-10-27 (archived 2026-10-28)

Sections moved out of `docs/status.md` once their programs were merged. Everything here is done;
the open decisions they raised are listed in `docs/status.md` under "Decisions to flag".

## Done and merged into main (reviewed by fresh correctness + style reviewers; fixes applied)
| item | what |
|---|---|
| add-task-generators | `fct/tasks.py`: TaskSpec registry, linear_teacher / object_action / what_where; `Dataset` gains `factors`, `groups`, `extras`; `data.task_dims`; position task bit-identical to before |
| add-modularity-metrics | `fct/modularity.py`: rank binning, plug-in MI (nats), preference r, MIR, dead neurons, shuffled-label baseline, shuffle-based bias correction |
| add-result-plots | `analyse.py` example outputs / decoders / `prediction_report` (both R^2 conventions); `draw.py` loss curves, MIR history (raw, corrected, chance, dead counts), per-task example figures |

## State (2026-10-24): both programs are complete and merged; everything now awaits the PI
The first program (task generators, MIR, sweep runner, SLURM script never submitted, notebook, lr tuning,
loss change to LOSS_VERSION 3 = rate penalty summed over P, `weight_lambda` 1e-4, mean-field tilt defaulting
to `rate_lambda`) is fully merged; its items are in `docs/issues/archive/`. Nothing was submitted. Still for
the PI: review the tiny results (`tmp/modularity.ipynb`), decide the `rate_lambda` values, set the `full`
preset sizes in `fct/sweep.py`, tune the SBATCH lines in `jobs/fct_sweep.sh`, `--dry-run`, submit.

Phase 2 (ten items, plans confirmed by the PI 2026-10-22; current state in `docs/issues/*.md`):
- merged 2026-10-23, each reviewed by a fresh reviewer with the record in its merge commit:
  `remove-rate-penalty-p-scaling`, `generic-grid-sweep` (arbitrary grids, `phase_transition` preset,
  `s{index}_label` directories), `spatial-kernels` (`fct/spatial.py`), `tuning-and-grid-plot-helpers`
  (`fct/tuning.py`, `fct/grid_report.py`, grid figures), `sgld-training` (`train.optimizer`, `train.beta`),
  `spatial-weight-prior-training` (`Config.spatial`, factorised prior in the loss, `fct/schema.py`),
  `grid-results-notebook` (`fct/grid_view.py`, `notebooks/build_phase_transition.py`),
  `importance-weighted-sampler` (`fct/importance.py`, `MeanFieldConfig` split into problem fields +
  `solver: SolverConfig`; reviewed twice: the first review found the ESS gate freezing the solver silently).
- merged 2026-10-23/24, likewise reviewed: `spatial-notebook` (`fct/spatial_view.py`, `fct/spatial_report.py`,
  `notebooks/build_spatial.py`), `meanfield-exemplar-notebook` (`solve(..., on_outer=...)` hook, live solver
  figures in `draw.py`, `fct/exemplar.py`, `notebooks/build_exemplar.py`; reviewed twice), `meanfield-grid-solve`
  (`SweepConfig.meanfield`, `run_setting(stage="solve")`, `sweep status/run --stage solve`, `jobs/fct_solve.sh`
  never submitted; reviewed twice). Main tip after the last merge: 871ccf7; 1321 tests pass, ruff clean.
No worker branches or worktrees remain. Nothing has been submitted to SLURM.
Still for the PI: the first-program checklist above, plus the phase-2 decisions below (the SGLD mapping one
changes numbers). Executed notebooks are not committed: build them with `notebooks/build_*.py` into `tmp/`.
Tooling: pytest's temp dir is pinned to `tmp/pytest` (tests/conftest.py); never write outside the project.

## 2026-10-27: PI feedback on the spatial notebook (all merged)
Four items, each reviewed by a fresh reviewer with the record in its merge commit:
`fix-tuning-view-cuda` (tuning_view moves rates and latents to the CPU once; fake-device regression test),
`spatial-notebook-inline-plots-and-equations` (every figure's plotting function written out in the
cell that draws it, a markdown cell with the equation before each; `fct/spatial_report.py` and the
spatial-only figures of `fct/draw.py` deleted; the kernel cells say K and the weight second moments are
over units, not the mean-field Phi over samples; the floor enters squared: eps = 1e-6 is 1e8 times
stiffer than 1e-2, not 1e4), `phi-kernels-in-spatial-notebook` (empirical Phi^(l) of the trained net
on all P samples, sorted by a latent, readout cases in the equation and titles), `mi-overlap-matrix`
(`fct.modularity.information_overlap`, `TuningView.member_mi`, C = I I^T / k per layer). The notebook
builds on CPU and on the T1000 with `DEVICE='cuda'` (9 figures). The PI's own parameter set (N 256,
P 64, k 4, m 16, no rate penalty, 200k epochs, CUDA) is saved uncommitted in
`tmp/pm/spatial_pi_edits.ipynb`; rerun it with
`uv run python notebooks/build_spatial.py tmp/spatial_cache --set N=256 --set P=64 --set K_LATENTS=4 --set M_IN=16 --set RATE_LAMBDA=0 --set N_EPOCHS=200_000 --set "DEVICE='cuda'"`.
Known and untouched: `ruff format --check notebooks` flags the generated `notebooks/modularity.ipynb`,
which the committed-notebook test requires verbatim (fix the builder's output or exclude the file).
Open for the PI: a theory Phi for a net under the spatial prior would need the mean-field solver to
take the prior (new item); whether the PI's parameter set should become the notebook defaults.

## 2026-10-27 (later): MI = 0.00 on the PI's P = 64 run, and the paper's spatial Phi (all merged)
The MI estimator was fine: at P = 64, B = 8 the plug-in bias (~0.38 nats per latent) made the
shuffle noise rule, applied to each neuron's TOTAL over the k latents, zero every neuron. Four items,
each reviewed by a fresh reviewer with the record in its merge commit:
`tuning-panels-with-information-bars` (each neuron: curves plus a bar chart of I_if, raw pale, the
MIR_KIND estimate full colour, the shuffled null as a black mark; `FactorInformation`/`TuningView`
carry `raw_mi` and `null_mi`; a printed line per layer gives the bias (B-1)^2/(2P)),
`overlap-of-informations-and-of-rates` (`fct.modularity.rate_overlap`; C^I and C^r side by side),
`spatial-phi-kernels` (Latex eqs kernel-def / phi-diagonalised: Phi^(l){mu nu}_{ij} = K^post_ij A_{mu nu},
A = psi^T K^pre psi / N; `fct.spatial.data_kernel`, `spatial_kernel_pairs`,
`fct.meanfield.readout_activities`; cells: A per hidden layer, the latents of PHI_TRIALS = 4 trials
at evenly spaced ranks, an S x S grid of n_post x n_post blocks per layer; the last hidden layer
drives the OUTPUT population; under readout "h" its kernel uses h where the paper uses phi(h^L);
Phi^(0) not drawn), `mi-noise-rule-per-latent` (final rule PER GROUP: a group's summed raw
information against NOISE_FACTOR x its summed null, all members zeroed when it fails; a member in
no group is tested alone; with one latent per group this is the per-latent rule the PI first
confirmed; the per-member rule was rejected after review because noise neurons scored 0.77 on
object-action). Measured corrected noise MIR under the group rule: 0.00 on object-action, what-where
and the linear teacher; binary singleton factors keep a floor of ~0.11. Cost of the group rule: a
neuron selective for one of 25 object bits survives only above ~112/P nats (fine at the sweeps'
P = 4096, zeroed by P <= 512). Cached sweeps predate the rule (caches key on the config only).
The spatial notebook has 11 figures and builds on CPU and on the T1000. The PI's run is
`tmp/spatial_cache/spatial_0624b6833433` (100k epochs); rebuild with
`uv run python notebooks/build_spatial.py tmp/spatial_cache --set N=256 --set P=64 --set K_LATENTS=4 --set M_IN=16 --set RATE_LAMBDA=0 --set N_EPOCHS=100_000 --set "DEVICE='cuda'"`.
Open for the PI: N_BINS should respect P (B = 8 at P = 64 is one sample per joint cell; the printed
bias line shows it; `N_BINS = min(8, sqrt(P)/2)` would be the mechanical default); small-P runs on
object-action have no usable corrected MIR under the group rule; the earlier open items above.

## 2026-10-27 (evening): jobs/fct_sweep.sh takes --index A:B (merged)
`bash jobs/fct_sweep.sh ROOT --submit --index 0:10` submits settings 0 .. 9 as one array
(`--array=0-9%FCT_MAX_CONCURRENT`); `--index N` unchanged; the solve wrapper inherits it. Written by
the PM test-first, reviewed by a fresh reviewer (record in the merge commit). Still never submitted.

