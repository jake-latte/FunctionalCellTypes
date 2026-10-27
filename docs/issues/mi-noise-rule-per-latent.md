---
type: data-science
complexity: simple
status: todo
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Apply the shuffle noise rule per latent, not to the neuron's total information

**Why.** `fct/modularity.py: _subtract_null` zeroes a neuron's whole row when its raw total
information is not above `NOISE_FACTOR` = 2 times its shuffled-null total. The totals are sums over
all k latents, so a neuron selective for ONE latent is compared against k times the plug-in bias.
At P = 64, B = 8, k = 4 (the PI's run) the bias is about 0.38 nats per latent: a neuron carrying
0.85 nats about z2 has raw total 2.4 against a null total 1.9 and is declared noise, as is every
other neuron (256/256), so the corrected MIR is 0.00 for the whole layer while B = 4 gives 0.64.
Reproduction: `tmp/pm/mi_repro.py`.

**Change (confirmed by the PI 2026-10-27).** Compare each member separately: corrected I_if =
max(raw_if - null_if, 0), and set I_if = 0 where raw_if <= NOISE_FACTOR x null_if. A neuron is dead
when every member is zeroed (as before, through `preference`). The module docstring, the
`TUNING_TEXT` in `notebooks/build_spatial.py`, `docs/design.md`, and `fct/explanations.py`
(the `@NOISE@` sentence) are updated; the existing tests of the row rule are rewritten for the
member rule and a test adds the P = 64, B = 8 case: a neuron that is a monotone function of one
latent keeps that latent's information and loses the others.

**Consequences to accept.** The corrected MIR of every future run changes (the monitor's corrected
MIR during training, the grid sweeps' corrected numbers, the notebooks). Cached sweep results
computed with the old rule stay as they are; rerunning them gives different corrected values.

**Also recommended (not part of the rule).** The notebooks' `N_BINS` should respect P: the joint
table has B^2 cells, so B = 8 at P = 64 is one sample per cell. A guard line in the tuning cell
(printed bias, item `tuning-panels-with-information-bars`) shows this; a default of
`N_BINS = min(8, int(sqrt(P) / 2))` would be the mechanical alternative.

## Acceptance criteria
- `tests/test_modularity.py`: the member rule as described; the P = 64, B = 8 synthetic case; the
  existing row-rule tests replaced.
- Lint and the full suite pass.
