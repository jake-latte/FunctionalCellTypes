"""The notebook's explanation of the modularity metrics, as markdown with LaTeX.

The text restates `fct/modularity.py`. Every number that has a source in the code (the dead
threshold, the noise factor of the shuffle correction, the monitor's default bins and shuffles)
is filled in from that source, so it cannot drift; tests/test_explanations.py also recomputes the
quoted floor examples.
"""

import math

from fct.config import TrainConfig
from fct.modularity import DEAD_TOL, NOISE_FACTOR

_TEMPLATE = r"""## What the MIR numbers mean

All quantities are computed by `fct/modularity.py`, for one hidden layer at a time, at each
monitor call during training (`fct.monitor.make_mir_monitor`, which calls `layer_mir_report`).
$P$ is the number of samples the monitor sees: the whole training set. The task descriptions
below cover the three tasks of the study (`linear_teacher`, `object_action`, `what_where`).

### Setup: rates, bins, factors

For a layer of $N$ neurons and the $P$ training inputs $x_p$, the **rate** of neuron $i$ is its
post-activation value $r_{ip} = \phi(h_{i}(x_p))$. The task provides ground-truth **factors**
$F_m$ (one integer or real label per input, e.g. the latents $z_j$ of `linear_teacher`;
`where` and `what` for `what_where`; the agent position, one occupancy bit per grid cell and
one bit per action for `object_action`), collected into $K$ **factor groups** $g_1,\dots,g_K$
(`linear_teacher`: one group per latent; `what_where`: `where`, `what`; `object_action`:
`position`, `objects`, `actions`).

Rates are discretised into $B$ equal-frequency **rank bins** (`rank_bins`, $B$ =
`monitor_bins`, @BINS@ by default). With the midrank
$m_{ip} = \tfrac12\big(\#\{q: r_{iq} < r_{ip}\} + \#\{q: r_{iq} \le r_{ip}\}\big)$ the bin is

$$b_{ip} = \Big\lfloor \frac{B\, m_{ip}}{P} \Big\rfloor \in \{0,\dots,B-1\},$$

so tied rates (a ReLU's mass at zero) share one bin and the bins are invariant to monotone
transforms of the rate. Integer factors are categorical with $C_m$ distinct labels; real-valued
factors are rank-binned the same way into $C_m = B$ labels (`factor_labels`).

### Mutual information, preference, MIR

The **plug-in mutual information** (`mutual_information`, in nats) between the binned rate of
neuron $i$ and factor $F_m$ uses the empirical joint frequencies $\hat p$ over the $P$ samples:

$$I_{im} = \sum_{b,c} \hat p(b,c)\,\log\frac{\hat p(b,c)}{\hat p(b)\,\hat p(c)} \;\ge 0 .$$

The information about a group is the **sum** over its members (the information about their
joint configuration cannot be estimated by a plug-in estimator):

$$I_{ik} = \sum_{m \in g_k} I_{im}.$$

The **preference** of neuron $i$ (`preference`; called `r` in the code) is the share of its
information that sits in its best group:

$$\rho_i = \frac{\max_k I_{ik}}{\sum_{k} I_{ik}} \in \Big[\tfrac1K, 1\Big].$$

A **dead neuron** is one whose total information $\sum_k I_{ik}$ is at most $@DEAD@$ (constant
firing, or a rate unrelated to every factor at floating-point level). Its preference is
undefined; it is set to exactly $\rho_i = 1/K$, so it adds nothing to the numerator below instead
of producing NaN. The layer's **modularity index** (`mir`, normalised so that chance preference
$1/K$ maps to 0 and perfect preference 1 maps to 1) is

$$\mathrm{MIR} = \frac{\bar\rho - 1/K}{1 - 1/K}, \qquad \bar\rho = \frac1N \sum_{i=1}^{N} \rho_i .$$

It is 1 when every neuron carries information about one group only and 0 when information is
spread evenly over the $K$ groups (or neurons are dead).

### Raw (plug-in) MIR and why it has a floor

The **raw MIR** is the formula above applied to the plain plug-in informations
(`layer_mir`, `bias_correction="none"`; the `monitor_mir` curve). The plug-in estimate is biased
upward: for a rate independent of a factor, with $B$ occupied rate bins (fewer than the
nominal number when many rates tie, e.g. a ReLU's mass at zero) and $C$ factor labels,

$$\mathbb{E}[\hat I] \approx \frac{(B-1)(C-1)}{2P}\ \text{nats}.$$

A group's floor is the sum over its members. The bias grows with the factor's cardinality $C$, so even completely unrelated neurons have
positive information, concentrated in the group with the most labels, and $\max_k / \sum_k$ of
positive numbers is never exactly $1/K$. The raw MIR of noise neurons therefore sits above 0 by
an amount that depends on the cardinalities of the factors, not on the network. For example at
$P = 8192$ and $B = 10$, noise neurons score a raw MIR of about 0.89 with one 25-way and two
binary factors, but about 0.10 with three five-way factors. A raw MIR is only interpretable
next to its chance baseline.

### Chance baseline

The **chance baseline** (`shuffled_mir`; `monitor_mir_chance`) is the raw MIR of the same layer
when the link between rates and factors is destroyed: all factors are permuted across the $P$
samples with **one shared random permutation** $\pi$ (so dependencies between the factors are
kept, rates are untouched), the plug-in informations are recomputed as $I_{im}^{\pi}$ and the
raw MIR formula is applied. The baseline is the mean of this raw MIR over $S$ permutations
($S$ = `monitor_shuffles`, @SHUFFLES@ by default; a seeded generator, so it is deterministic):

$$\mathrm{MIR}_{\text{chance}} = \frac1S \sum_{s=1}^{S} \mathrm{MIR}\big[I^{\pi_s}\big].$$

A raw MIR is only evidence of modularity to the extent that it exceeds this baseline.

### Corrected MIR

The **corrected MIR** (`layer_mir`, `bias_correction="shuffle"`; `monitor_mir_corrected`)
removes the bias before the preferences are formed. `layer_mir_report` uses the same $S$
permutations as the baseline, so no extra shuffles are drawn (`n_shuffles` = $S$). For every
neuron and factor (member, not group) the mean shuffled information is the null
$\bar I^{0}_{im} = \frac1S\sum_s I^{\pi_s}_{im}$ and the corrected information is

$$\tilde I_{im} = \max\big(I_{im} - \bar I^{0}_{im},\, 0\big) .$$

A group $k$ whose raw information is not clearly above its null, summed over its members,
$\sum_{m \in k} I_{im} \le @NOISE@\,\sum_{m \in k} \bar I^{0}_{im}$, is indistinguishable from
noise and the $\tilde I_{im}$ of all its members are set to 0; a neuron with every member set
to 0 is dead ($\rho_i = 1/K$). (Without this step the leftover noise residuals, mostly in one
factor, would give noise neurons a spuriously high preference. The test is per group, the unit the
preference is formed on: not on the row total, which would compare a neuron selective for one group
with the bias of all $K$ groups, and not per member, which would let a group of many binary members
through whenever one of them passes by chance. Within a group the comparison is still with the sum
of its members' biases, so a neuron selective for one member of a large group is kept only if its
excess information exceeds about $|k|(B-1)(C-1)/(2P)$ nats: 25 object bits at 10 bins give
$112/P$, 0.03 nats at $P = 4096$.) The corrected informations are summed within groups, and
$\rho_i$ and the MIR are computed exactly as above:

$$\mathrm{MIR}_{\text{corrected}} = \frac{\bar{\tilde\rho} - 1/K}{1 - 1/K}.$$

Groups whose information is within @NOISE@ times their shuffled null are set to 0, so a neuron
with none left is at $1/K$; other noise neurons may leak a little preference by chance. The
corrected MIR of an unrelated layer is therefore about 0 for many-valued or grouped factors
(measured 0.00 on the object-action and what-where tasks and on continuous latents), and no
baseline has to be subtracted from it. A binary factor that is a group on its own passes the
test by chance a few percent of the time, so a set of such singletons keeps a floor of ~0.1.
Because $\rho_i \ge 1/K$, it cannot go below 0 (up to round-off); it is a modularity of the
neurons that carry detectable information, diluted by those that do not.

### Reading the plots and tables

- **raw MIR panel**: one solid line per layer (marker o) is the raw MIR; the dotted line of
  the same colour (marker x) is that layer's chance baseline. Read the gap between them.
- **corrected MIR panel**: one line per layer; 0 means indistinguishable from unrelated
  neurons, 1 means every informative neuron is selective for one group.
- **dead-neuron panel**: the number of dead neurons of each layer under the corrected estimate
  (`monitor_n_dead_corrected`): neurons with constant firing or no information beyond the
  shuffled null. At most $N$ per layer.
- **heatmaps and table**: one row per setting, one column per layer, the value at the last
  monitor call, in three panels or column groups: `raw`, `corrected` and `chance baseline`.
  Grey cells are layers the setting does not have or settings that have not finished.
"""


def _fill(template: str) -> str:
    train = TrainConfig.model_fields
    values = {
        "@DEAD@": f"10^{{{round(math.log10(DEAD_TOL))}}}",
        "@NOISE@": f"{NOISE_FACTOR:g}",
        "@BINS@": str(train["monitor_bins"].default),
        "@SHUFFLES@": str(train["monitor_shuffles"].default),
    }
    for key, value in values.items():
        template = template.replace(key, value)
    return template


MIR_EXPLANATION = _fill(_TEMPLATE)
