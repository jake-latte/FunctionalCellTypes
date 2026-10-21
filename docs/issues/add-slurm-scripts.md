---
type: infrastructure
complexity: simple
status: done
---

<!-- status: todo | in progress | in review | done | blocked | awaiting your confirmation -->

# SLURM array script for the sweep (written, never submitted)

`jobs/fct_sweep.sh` modelled on `/ceph/scratch/jlaherty/DMFT/jobs/dmft_sweep.sh` (read-only template). Depends on `add-sweep-runner`.

## Interfaces
- Actions `--submit [--index N]`, `--dry-run`, `--list`, `--status`; array over settings with a concurrency cap; `#SBATCH -p gpu --gres=gpu:1`; exclude `gpu-380-10,gpu-380-12,gpu-380-14`; logs under `tmp/logs/`; runs `.venv/bin/python -m fct sweep run ROOT --index $SLURM_ARRAY_TASK_ID`.

## Acceptance criteria
- `bash -n` passes; `--dry-run`, `--list` and `--status` print what they would do without submitting.
- A test with a PATH-poisoned fake `sbatch` that fails if called proves the non-submit actions never submit.
- Nobody (agent or test) runs `--submit`; the PI submits.
