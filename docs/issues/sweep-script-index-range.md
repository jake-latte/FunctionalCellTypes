---
type: infrastructure
complexity: simple
status: done
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Let jobs/fct_sweep.sh submit a range of settings with --index A:B

**Why.** The PI wants to submit part of a sweep: `--index 0:10` should submit settings 0 to 9 as
one array instead of the full sweep or one setting at a time.

**What.** `jobs/fct_sweep.sh` (and so `jobs/fct_solve.sh`, which execs it) accepts `--index A:B`
with A and B non-negative integers (base 10, as for `--index N`), meaning the half-open range of
settings A .. B-1, like a Python slice. `--submit` and `--dry-run` then use
`--array=A-(B-1)%FCT_MAX_CONCURRENT` (the same concurrency cap as the full sweep); a range of one
setting is the plain `--array=A`. Errors: an empty range (A >= B), a range past the sweep size
(B > count), and any other form (`:3`, `1:`, `1:2:3`, `-1:3`, `x:y`) fail with a clear message.
`--index N` keeps its meaning. Usage header of the script, `docs/design.md` (the
`jobs/fct_sweep.sh` section) and the solve script's usage lines updated.

## Inputs / outputs / interfaces
- CLI of `jobs/fct_sweep.sh`: `--index N` or `--index A:B` (also `--index=A:B`).
- Output: the sbatch `--array` spec.

## Acceptance criteria
- `tests/test_slurm_script.py` (fake sbatch, stub python, STUB_COUNT = 5): `--dry-run --index 0:3`
  prints `--array=0-2%4`; `--index=1:5` prints `--array=1-4%4`; `--index 2:3` prints `--array=2`;
  `--index 08:010` prints `--array=8-9%4` is out of range for 5 settings, so use `--index 01:03`
  -> `--array=1-2%4`; `FCT_MAX_CONCURRENT=2` with a range gives `%2`; `--submit --index 1:4` calls
  sbatch with `--array=1-3%4`; `3:6` fails "out of range"; `2:2` and `3:1` fail "empty"; `:3`,
  `1:`, `1:2:3`, `-1:3` fail with the integer-or-range message; the solve wrapper passes a range
  through.
- `bash -n` syntax check, lint and the full suite pass.
