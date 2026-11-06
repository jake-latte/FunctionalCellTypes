---
type: infrastructure
complexity: simple
status: in review
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Add an orthonormal linear teacher (`data.teacher`)

W2 of `latex/notes/fct_experiment_plan.tex`, confirmed by the PI 2026-11-06.
The notes' predictions assume $W_X$, $W_Y$ with orthonormal columns; the code
draws them i.i.d. $\mathcal N(0, 1/k)$ (`fct/tasks.py:91-92`).

## What to change
- `DataConfig`: `teacher: Literal["gaussian", "orthonormal"] = "gaussian"`.
  `"orthonormal"` takes the `Q` factor of `torch.linalg.qr(W, mode="reduced")`
  of the same Gaussian draws (as the position task does, `fct/data.py:108-109`),
  so the random stream and every other field are unchanged.
- `Config.section_hash` (`fct/config.py:391-426`): drop `teacher` from the
  hashed data dump when it is `"gaussian"` (the `net.B` rule at lines 413-414),
  so every existing hash is unchanged; only `"orthonormal"` enters the hash.
- `LinearTeacherParams.teacher` (`fct/exp.py:152-155`) with the same default,
  plumbed through `_task_fields` / `data_config` (`:252-272`).
- `docs/design.md` (`DataConfig`, `tasks.py`, `exp.py` rows).

## Inputs / outputs / interfaces
- Inputs: `fct/tasks.py:81-103`, `fct/config.py`, `fct/exp.py`.
- Outputs: `data.teacher`; `LinearTeacherParams.teacher`.

## Acceptance criteria
- Default: every data/train/meanfield/analysis hash of every preset and
  fixture is unchanged (test with `section_hash` before/after setting the
  field explicitly to `"gaussian"`); the generated data is bit for bit the old.
- `"orthonormal"`: $W_X^\top W_X = I_k$ and $W_Y^\top W_Y = I_k$ to $10^{-12}$
  (float64); `Z` identical to the Gaussian teacher's; the hash differs.
- An experiment preset with `"teacher": "orthonormal"` round-trips through
  `exp init` / `load_exp` / `setting_config`.
- Full suite and `ruff` pass.
