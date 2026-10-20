---
type: data-science
complexity: simple
status: in review
---

<!-- status: todo | in progress | in review | done | blocked | awaiting your confirmation -->

# Add the modularity metric: neuron-factor mutual information, preference and MIR

For a trained network we measure, per hidden layer, how "modular" its neurons are for the task factors. Let I[l, n, k] be the mutual information between the firing rate of neuron n in hidden layer l and task factor k. A neuron's preference is r[l, n] = max_k I[l, n, k] / sum_k I[l, n, k] and the layer's modularity index is MIR_l = (mean_n r[l, n] - 1/K) / (1 - 1/K), K the number of factors. This item adds the pure, vectorised computation (new module `fct/modularity.py`); a later item calls it during training and in a notebook.

## Interfaces (all pure; torch tensors on any device; no I/O, no plotting)
- Factors are a `dict[str, Tensor]` of (P,) tensors: int64 means categorical (labels), floating means continuous. `groups: dict[str, list[str]] | None` maps a group name to the factor names it contains; the scored "factors" k are the GROUPS, and a group's mutual information with a neuron is the SUM of the mutual informations of its member factors (documented: this is the total information the neuron carries about the group's members, robust to estimate, unlike the information about their joint configuration which a plug-in estimator cannot resolve). groups=None means every factor is its own group.
- `rank_bins(values: Tensor (P, N) float, n_bins: int) -> Tensor (P, N) int64`: equal-frequency binning per column from mid-ranks, bin = floor(midrank / P * n_bins) with midrank = (count(v < x) + count(v <= x) - 1) / 2 + 0.5 style so that ties (e.g. the mass of a ReLU at zero) all land in ONE bin and the result is invariant to monotone transforms of the column; at most n_bins distinct values per column. Vectorised (sort + batched searchsorted), no per-column Python loop.
- `factor_labels(factor: Tensor (P,), n_bins: int) -> (labels (P,) int64 in [0, C), C)`: integer factors are relabelled to 0..C-1 (dense), floating factors are binned with rank_bins into n_bins.
- `mutual_information(bins: Tensor (P, N) int64, n_bins: int, labels: Tensor (P,) int64, n_labels: int) -> Tensor (N,)`: plug-in mutual information in NATS from the joint counts of (bin, label) for every neuron at once (one flat bincount of n*B*C + b*C + c, no loop over neurons), clamped at >= 0 against round-off.
- `neuron_factor_information(rates: Tensor (P, N), factors, groups=None, n_bins=10) -> FactorInformation` NamedTuple: `member_mi (N, n_members)`, `group_mi (N, K)`, `group_names: list[str]`.
- `preference(group_mi: Tensor (N, K), tol: float = 1e-12) -> (r (N,), dead (N,) bool)`: r = max/sum; a neuron whose total information is <= tol (constant firing, or a rate unrelated to every factor at floating-point level) is DEAD and is assigned r = 1/K, i.e. exactly chance, so it contributes zero to the MIR numerator (deliberate, documented; the count of dead neurons is reported alongside).
- `mir(r: Tensor (N,), K: int) -> float` the formula above (assert K >= 2: with one factor the index is undefined).
- `layer_mir(rates, factors, groups=None, n_bins=10) -> LayerMIR` NamedTuple: `mir: float`, `r: (N,)`, `group_mi: (N, K)`, `n_dead: int`, `group_names`.
- `network_mir(rates: Tensor (P, L, N), factors, groups=None, n_bins=10) -> list[LayerMIR]` one entry per layer (rates are the firing rates phi(H); the caller applies phi).
- `shuffled_mir(rates, factors, groups=None, n_bins=10, seed=0, n_shuffles=1) -> float`: the chance level of the metric: the MIR of one layer when the factor labels are permuted jointly across samples (same permutation for all factors, so factor dependencies are kept), averaged over n_shuffles; seeded CPU generator. The plug-in estimator is biased upward and the max/sum ratio has a positive floor even for unrelated neurons, so the notebook plots this baseline next to every MIR curve.
- Add a short section to docs/design.md for the module.

## Acceptance criteria (tests: synthetic ground truth, CPU)
- MI against hand-computed values: a neuron equal to a categorical label has MI = the label's entropy (nats) exactly (to 1e-9, with n_bins >= C); a neuron independent of the label has MI close to its plug-in bias (< 0.01 nats at P = 20000, B = C = 5); MI of two labels is symmetric under swapping roles; a 2x2 table with known probabilities gives the analytic value.
- rank_bins: ties land in one bin (a ReLU-like column with 60% zeros gives the zeros one bin), invariance under monotone transforms (exp, scale), equal-frequency for a continuous column (each bin ~ P/B), and agreement with a straightforward per-column reference implementation on random data.
- Ground-truth recovery of modularity (P = 20000): with K = 3 independent categorical factors and N neurons of which each depends on exactly one factor (the neuron's rate is a distinct monotone-coded function of that factor's label), every r is ~1 and MIR ~ 1 (> 0.97); neurons that mix two factors equally (rate = code of the pair of labels) have r ~ 0.5 and MIR ~ ((0.5 - 1/K)/(1 - 1/K)), i.e. the analytic value; fully mixed (all three) gives MIR ~ 0; a layer of pure-noise neurons has MIR well below the modular layer and equal (within tolerance) to `shuffled_mir` of the same layer; `shuffled_mir` of the modular layer is far below its true MIR.
- Continuous factors: with two independent uniform latents and neurons that are monotone functions of one latent each, MIR ~ 1 after binning.
- Groups: a group of three binary members where the neuron copies one member gives group MI = that member's entropy (+ ~0 from the others); MIR over the groups differs from MIR over the members as expected on a constructed example; groups=None equals singleton groups.
- Dead neurons: constant neurons are flagged, get r = 1/K, are counted in n_dead and contribute 0 to the MIR numerator (test the exact value on a constructed layer).
- Device/dtype agnostic: the same numbers (to tolerance) for float32 and float64 rates; vectorised (a test with N = 2000 neurons, P = 5000 runs in a second or two on CPU); invalid inputs fail loudly (K < 2, mismatched lengths, n_bins < 2, rates that are not 2-D).
- All existing tests still pass; ruff check and ruff format --check clean.
