---
type: infrastructure
complexity: complex
status: awaiting your confirmation
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Grow an experiment's ensembles in place (`--increase-ensemble-size`)

PI decision (2026-11-06): one SGLD chain is not trusted to be ergodic enough
to yield several independent posterior samples, so **one network is taken
from each of the B members** of an ensemble, and B is the sample size. To
add samples without retraining, an experiment whose settings are done must
be extendable: `B += K` in `exp.json`, train the K new members only, append
them to each setting's cache.

## Proposed design (to confirm)
- CLI: `fct exp extend ROOT --by K` rewrites `exp.json` with `B + K`
  (refusing if any setting is `partial`), then `fct exp run ROOT --index i`
  (and the SLURM array) trains whatever members a setting is missing: a
  setting with a `final.pt` of $B_{\mathrm{old}}$ members under the new `B`
  trains members $B_{\mathrm{old}}, \dots, B-1$ as a fresh `EnsembleMLP` of
  $K$ members in `train/extend_<b0>-<b1>/` (own checkpoints, resumable),
  then concatenates state dict (leading member axis), per-member history
  entries and monitor records into `final.pt`, deletes the staging
  directory, and restamps the hash. This splits the PI's one flag into the
  exp.json edit and the per-setting work, so the array job stays the unit of
  training. If the PI prefers the single flag on `exp run`, it is the same
  machinery behind one option.
- Seeds: `member_seed(net.seed, b)` already makes member $b$ independent of
  $B$ (`tests/test_ensemble.py::test_a_member_does_not_depend_on_the_ensemble_size`),
  so the new members' initial weights equal those of a $B + K$ run from
  scratch. SGLD noise: use the per-member noise streams
  (`test_sgld_ensemble_with_member_noise_streams_equals_separate_members`)
  so that the extended ensemble equals a from-scratch $B + K$ ensemble bit
  for bit; under the single global stream it is equal in distribution only
  (document which applies).
- Hash rule (decision): `net.B` is in the train hash. **Recommended:** keep
  it there and restamp `final.pt` with the new config's hash on append (the
  artefact then is the $B + K$ artefact); `setting_status` additionally
  checks the member count against `B` (`done` needs both). The alternative,
  dropping `B` from the hash because members are $B$-independent, would make
  every existing `B > 1` cache stale (the PI's `phase_transition` sweeps have
  `B = 8`).
- `exp status` gains a `members` column; `select_member` and every reader
  already index members, so analysis code is unchanged.

## Inputs / outputs / interfaces
- Inputs: `fct/exp.py` (`run_setting`, `setting_status`, `save_exp`),
  `fct/train.py` (`train_from_config`, `_net_from_payload`, checkpoints),
  `fct/model.py` (`EnsembleMLP.from_members`), `fct/sgld.py`, `fct/cli.py`,
  `jobs/fct_exp.sh`, `docs/design.md` ("Ensembles", "exp.py").
- Outputs: the subcommand (or flag), the staged training, the append, the
  status rule, the design.md contract.

## Acceptance criteria
- Extend a tiny experiment from $B = 2$ to $B = 5$: `final.pt` holds 5 members;
  members 0-1 are bit for bit the old ones; members 2-4 equal members 2-4 of
  a from-scratch $B = 5$ run (bit for bit with member noise streams, else in
  distribution with a stated test); every history list has length 5; the
  monitor records have a member axis of 5; `exp status` says `done`.
- Interrupting the extension and re-running resumes from the staging
  checkpoint and gives the same result.
- Extending a `partial` setting, or running with an `exp.json` whose `B` is
  smaller than the cached member count, fails loudly.
- `jobs/fct_exp.sh` works unchanged on an extended experiment.
- Full suite and `ruff` pass.
