---
type: data-science
complexity: simple
status: todo
---

<!-- status: todo | in progress | in review | done | blocked | awaiting your confirmation -->

# Run the tiny preset end to end, then stop for review

Run the `tiny` preset for all 36 settings locally (CPU, plus one tiny GPU smoke run pinned by UUID, T1000 preferred), execute the notebook on it, commit the executed notebook. Depends on the three items above.

## Acceptance criteria
- All 36 settings `done` in `status`; notebook executed without errors.
- The PI reviews the code and small results and submits the full jobs. No SLURM submission by the agent.
