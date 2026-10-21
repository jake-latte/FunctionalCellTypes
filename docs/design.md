# `fct` library design — the contract between modules

This document fixes the interfaces that the work items implement, so that
pieces built in parallel fit together. It is the *present* design; it is not a
changelog. Everything in `playground.ipynb` is the numerical reference: ports
must reproduce its maths exactly (same formulas, same defaults), only the
*packaging* changes.

## Package layout

```
fct/
  __init__.py
  config.py        # pydantic Config (data/net/train/meanfield/analysis/device), JSON I/O, section hashes
  activations.py   # PHI registry (name -> callable), get_phi, phi_prime
  cache.py         # cache-directory layout + load_or_compute helpers (I/O layer)
  data.py          # task / data generation (Dataset, fourier_embedding, embed, grid_targets, task_dims)
  tasks.py         # pure generators of the linear-teacher, object-action and what-where tasks
  model.py         # MLP architecture, make_net
  train.py         # training loop with checkpoint / resume
  hmc.py           # SingleSiteHMC sampler + linear-algebra helpers
  meanfield.py     # saddle-point solver (Algorithm 1) with checkpoint / resume
  analyse.py       # pure analysis functions + cached wrappers
  modularity.py    # neuron-factor mutual information, preference, MIR (pure)
  monitor.py       # make_mir_monitor: the training hook that records per-layer MIR
  draw.py          # plotting: plot_* take an ax, fig_* assemble figures; never plt.show()
  sweep.py         # the 36-setting grid: SweepConfig, expand, run_setting, status
  cli.py           # argparse CLI: init / train / solve / plot / plot-example / sweep
  __main__.py      # `python -m fct ...` -> cli.main()
tests/             # pytest, synthetic fixtures only, CPU, tiny sizes
docs/design.md     # this file
docs/issues/*.md   # work items
pyproject.toml, uv.lock, .gitignore, README.md
```

Environment: `uv` + `pyproject.toml` + committed `uv.lock`. Tests run with
`uv run pytest`. Pytest config: `[tool.pytest.ini_options] testpaths=["tests"]`,
`pythonpath=["."]`. Dependencies: torch>=2.11, numpy, scikit-learn, matplotlib,
pydantic>=2; dev: pytest, ruff. `ruff` config: line length 100.

## Cache directory layout

The CLI takes one positional argument `CACHE_DIR`. Everything lives under it:

```
CACHE_DIR/config.json                # written by `init`; the single source of hyperparameters
CACHE_DIR/data/dataset.pt            # generated task data
CACHE_DIR/train/ckpt_<epoch:06d>.pt  # periodic checkpoint (only the latest kept)
CACHE_DIR/train/final.pt             # final trained model (+ history)
CACHE_DIR/meanfield/ckpt_<outer:05d>.pt, CACHE_DIR/meanfield/final.pt
CACHE_DIR/analysis/<name>.pt         # intermediate analysis results
CACHE_DIR/figures/<name>.png
```

**Cache-or-compute everywhere.** Every `get_*` / `run_*` function first looks
for an existing artefact under `CACHE_DIR` and only computes on a miss. Train
and solve resume from the latest checkpoint; if `final.pt` exists they return it
without recomputing. Caches are disposable: deleting a subdirectory only costs
time (principle 3). `CACHE_DIR` is git-ignored.

**Staleness is an error, not a silent recompute.** Each artefact stores the
hash of the config sections it depends on (`Config.section_hash(...)`). On load,
a mismatch raises `StaleCacheError` naming the file and telling the user to use
a fresh cache directory or delete the stale artefact (principle 9).

| artefact   | depends on sections                        |
|------------|--------------------------------------------|
| data       | data                                       |
| train      | data, net, train                           |
| meanfield  | data, net, train, meanfield                |
| analysis   | data, net, train, analysis                 |

Every artefact additionally depends on `dtype`; `section_hash` always includes it.

## Saved files

Every `.pt` file must be loadable with `torch.load(path, weights_only=True)`:
only tensors, Python scalars/str/None, lists, dicts. No pickled modules,
functions or numpy arrays (convert numpy to lists / tensors before saving).
Models are stored as `state_dict` and rebuilt from the config with `make_net`.

## config.py

```python
class DataConfig(BaseModel):      # model_config = ConfigDict(extra="forbid") on every model
    task: Literal["position", "linear_teacher", "object_action", "what_where"] = "position"
    P: int = 4096; D_in: int = 64
    # position task
    d: int = 2; var: float = 10.0
    cov: list[list[float]] | None = None          # None -> identity
    embedding: Literal["linear", "fourier"] = "fourier"
    # linear_teacher
    n_latents: int = 3; latent_range: float = 1.0   # k latents, uniform on [-a, a]
    # object_action
    grid_size: int = 5; max_objects: int = 3        # G x G grid, 0..A objects
    # what_where
    n_motifs: int = 8                               # m motifs out of the 511 non-empty 3x3 patterns
    seed: int = 0
class NetConfig(BaseModel):
    N: int = 100; L: int = 3; g: float = 1.0
    phi: Literal["relu", "tanh", "sigmoid"] = "relu"
    seed: int = 0
class TrainConfig(BaseModel):
    n_epochs: int = 15_000; lr: float = 1e-3
    weight_lambda: float = 1e-3; rate_lambda: float = 1e-3
    log_int: int = 100; save_int: int = 100; seed: int = 0
    monitor_int: int = 0                # epochs between monitor calls (>= 0); 0 disables the monitor
    monitor_bins: int = 10              # rate bins of the MIR monitor (>= 2)
    monitor_shuffles: int = 5           # label shuffles for the corrected MIR and the chance level (>= 1)
class MeanFieldConfig(BaseModel):
    n_points: int = 100                 # solve on the first n_points training points (X=x[:n_points])
    beta: float = 1e6
    readout: Literal["h", "phi"] = "h"
    # Algorithm 1 (defaults = notebook cell 25 / solve() signature)
    n_outer: int = 300; n_inner_max: int = 30; inner_tol: float = 1e-2; inner_tol_mc: float = 2.0
    eta_phi: float = 0.05; eta_phihat: float = 0.05; eta_max: float = 1.0; eta_grow: float = 1.1
    max_backtrack: int = 8; eta_min_rel: float = 1e-3; max_div_frac: float = 0.02
    max_h2_growth: float = 10.0; max_mc_err: float = 1.0; eps_collapse: float = 1e-2
    project_tilt: bool = True; tilt_floor: float = 0.05; max_rel_step: float = 0.1
    pd_floor_rel: float = 1e-8; inv_floor_rel: float = 1e-4
    prior_scale: float | None = None; alpha: float | None = None
    learning_prefactor: float = 0.5; rate_tilt: float | None = None
    # HMC
    n_chains: int = 2048; n_trans: int = 2; n_trans_outer: int = 5; n_leap: int = 10
    eps0: float = 0.05; target_accept: float = 0.75; adapt_rate: float = 0.05
    warmup_trans: int = 200; sample_dtype: Literal["float32", "float64"] = "float32"
    jitter_rel: float = 1e-6; precondition: bool = True
    init: Literal["lazy", "gaussian", "net"] = "lazy"; n_init_samples: int = 20_000
    log_int: int = 5; save_int: int = 10; seed: int = 0
class AnalysisConfig(BaseModel):
    n_grid: int = 25            # p in the notebook: test grid is n_grid x n_grid
    n_components: int = 10      # PCA components per layer
    plot_layer: int = 0         # layer whose neuron maps are drawn
    n_cols: int = 8             # columns in the neuron-map grid
class Config(BaseModel):
    data: DataConfig = DataConfig(); net: NetConfig = NetConfig(); train: TrainConfig = TrainConfig()
    meanfield: MeanFieldConfig = MeanFieldConfig(); analysis: AnalysisConfig = AnalysisConfig()
    device: Literal["auto", "cpu", "cuda"] = "auto"
    dtype: Literal["float32", "float64"] = "float64"
    def section_hash(self, *sections: str) -> str   # sha256 of canonical JSON of those sections AND the top-level dtype, first 12 hex chars; >=1 section required

def default_config() -> Config
def load_config(path: Path) -> Config          # raises on unknown keys / bad types
def save_config(cfg: Config, path: Path) -> None   # indent=2, sorted keys
def resolve_device(cfg: Config) -> torch.device   # "auto" -> cuda if available else cpu
def resolve_dtype(name: str) -> torch.dtype
```

`task` selects the generator and only that task's fields are read; the fields of other tasks are ignored but still enter the cache hash (changing one makes an existing data cache stale). Fields are range-constrained (sizes and intervals >= 1, rates and scales > 0, `target_accept` in (0,1), `tilt_floor` in [0,1); `D_in` must be even for the position task with `embedding="fourier"`; for `linear_teacher`, `D_in` (the number of features m of both x and y) must exceed `n_latents`; `max_objects` >= 0 while `n_latents`, `grid_size`, `n_motifs` are >= 1 and `latent_range` > 0; `max_objects <= grid_size**2`, `n_motifs <= 511`, and for `object_action` / `what_where` `D_in` must equal the fixed 2*G*G / 81), and `load_config` raises `ValidationError` naming the field.

## activations.py

```python
PHI: dict[str, Callable]   # {"relu": torch.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}
def get_phi(name: str) -> Callable          # KeyError with the valid names
def phi_prime(phi: Callable) -> Callable    # the notebook's _phi_prime_factory: analytic for relu/tanh/sigmoid, autograd otherwise
```

## cache.py (I/O layer)

```python
class StaleCacheError(RuntimeError)
def cache_paths(cache_dir: Path) -> CachePaths     # dataclass with config, data, train, meanfield, analysis, figures sub-paths
def save_artefact(path: Path, payload: dict, config_hash: str) -> None    # adds "config_hash", makes parent dirs, torch.save
def load_artefact(path: Path, expected_hash: str, device) -> dict          # weights_only=True; raises StaleCacheError on mismatch
def load_or_compute(path, expected_hash, compute: Callable[[], dict], device) -> dict
def latest_checkpoint(dir: Path, prefix: str = "ckpt_") -> Path | None
def write_checkpoint(ckpt_dir: Path, prefix: str, step: int, payload: dict, config_hash: str, width: int = 6) -> Path
    # writes ckpt_dir/<prefix><step:0{width}d>.pt, then removes ckpt files with the same prefix and a LOWER step
    # (train uses the default width=6; meanfield passes width=5)
```

## data.py

```python
@dataclass(frozen=True)
class Dataset:      # defined in tasks.py, re-exported by data.py
    x: Tensor   # (P, D_in)
    y: Tensor   # (P, D_out)  (position task: (P, d))
    Q: Tensor | None        # (D_in, d) orthonormal map for embedding="linear", else None
    period: float | None    # Fourier period = y.max() - y.min() used at generation time, else None
    factors: dict[str, Tensor] | None = None       # name -> (P,) ground-truth factor: int64 = categorical labels 0..C-1, floating = continuous
    groups: dict[str, list[str]] | None = None     # group name -> factor names; the unit over which modularity is scored (a factor that is its own group is a singleton group)
    extras: dict[str, Tensor] | None = None        # tensors needed to draw further examples of the task

def fourier_embedding(y, D, period) -> tuple[Tensor, Tensor]      # verbatim from the notebook
TASKS: dict[str, TaskSpec]     # the one registry: "position" (defined in data.py) plus tasks.TASK_SPECS
def task_dims(cfg: DataConfig) -> tuple[int, int]                 # (TASKS[cfg.task].input_dim(cfg), .output_dim(cfg)); position: (cfg.D_in, cfg.d)
def make_dataset(cfg: DataConfig, device, dtype) -> Dataset       # TASKS[cfg.task].generate; position = notebook position_task, seeded by torch.Generator(cfg.seed)
def embed(y_new: Tensor, dataset: Dataset, cfg: DataConfig) -> Tensor   # same embedding as the dataset (linear: y@Q.T; fourier: dataset.period); position task only (ValueError for other tasks)
def grid_targets(half_width: float, n_grid: int, device, dtype) -> Tensor  # (n_grid**2, 2) meshgrid 'ij' flattened as in notebook cell 13
def get_dataset(cache_dir: Path, cfg: Config) -> Dataset          # cache-or-compute at CACHE_DIR/data/dataset.pt, hash = section_hash("data"); persists x, y, Q, period, factors, groups, extras
```

Every task draws from a CPU `torch.Generator` seeded with `cfg.seed` and moves the result to the device, so
the data never depends on the device. `D_in` of `object_action` and `what_where` is fixed by the task
(2*G*G and 81): `DataConfig` rejects a different `D_in`, with a message naming the `data.D_in` to set. The position task's factors are `y0`, `y1`, ... (the columns of
`y`, floating), each a singleton group.

## tasks.py

`TaskSpec` (frozen dataclass) bundles `input_dim(cfg)`, `output_dim(cfg)`, `validate(cfg)` (raises
`ValueError` for a config the task cannot generate) and `generate(cfg, device, dtype) -> Dataset`; the sizes
that the config validates, `task_dims` reports and the generator produces all come from it. `TASK_SPECS`
holds the three tasks below; `data.TASKS` adds the position task. `DataConfig` calls `validate` of its task
(`D_in > n_latents`; `max_objects <= grid_size**2` and `D_in = 2*G*G`; `n_motifs <= 511` and `D_in = 81`) and
the generators call it again to protect direct calls. tasks.py imports `fct.config` only for type checking,
because the config imports `TASK_SPECS`.

```python
def linear_teacher(cfg, device, dtype) -> Dataset   # Q = period = None
    # Z (P,k) iid U[-a,a]; W_X, W_Y (m,k) iid N(0,1/k), m = cfg.D_in; x = Z W_X^T, y = Z W_Y^T, both (P,m)
    # factors z0..z{k-1} (float); one singleton group per latent; extras W_X, W_Y
def object_action(cfg, device, dtype) -> Dataset   # Q = period = None
    # G x G grid, cell = row*G + col; agent cell uniform; #objects uniform on 0..A; distinct object cells drawn
    # uniformly from ALL cells (an object may sit under the agent); ValueError if A > G*G
    # x = [one-hot agent cell (G*G), binary object map (G*G)] (P, 2*G*G); y = optimal action (P, 5)
    # factors: position (int64 in [0,G*G)), object_{i} (int64 0/1), action_{stay,up,down,left,right} (int64 0/1)
    # groups: position, objects (all object_i), actions (all action_*); extras empty
def optimal_actions(agent_rc: Tensor (P,2) int, objects: Tensor (P,G,G) bool) -> Tensor (P,5) float32
    # columns [stay, up, down, left, right], up = row-1, down = row+1, left = col-1, right = col+1; Manhattan
    # distance. stay = 1 iff no objects or the agent is on one; otherwise stay = 0 and a move is 1 iff it stays on
    # the grid and lowers the minimum distance to the object set by exactly 1 (ties and diagonals give several 1s)
def what_where(cfg, device, dtype) -> Dataset   # Q = period = None
    # 9x9 micro-grid = 3x3 macro-grid of 3x3 cells; library = the 511 non-empty binary 3x3 patterns; the m
    # motifs are chosen uniformly without replacement (ValueError if m > 511); `where` uniform on 9 macro-cells
    # (row-major) and `what` uniform on m, independent; x (P,81) row-major micro-grid, pixel (3R+r, 3C+c) =
    # motif[r*3+c] inside macro-cell (R,C), zero elsewhere; y (P, 9+m) = [one-hot where, one-hot what]
    # factors where, what (int64), each a singleton group; extras motifs (m,9) in the dataset dtype
TASK_SPECS: dict[str, TaskSpec]   # linear_teacher, object_action, what_where
```

## model.py

```python
class MLP(nn.Module)   # verbatim from the notebook: __init__(D_in, D_out, N, L, g, phi, device, dtype, generator=None); forward(x) -> (H (P,L,N), Z (P,D_out)); weight_l2()
def make_net(cfg: NetConfig, D_in: int, D_out: int, device, dtype) -> MLP   # weights drawn on CPU from a local torch.Generator seeded with cfg.seed, then moved to device (no global RNG touched); phi = get_phi(cfg.phi)
```

## train.py

```python
def compute_loss(net, x, y, weight_lambda, rate_lambda) -> tuple[Tensor, dict[str, Tensor]]
    # loss = 0.5*N*mse + weight_lambda*weight_l2 + rate_lambda*rate_l2  (notebook cell 5); dict of detached 0-d tensors: mse, weight_l2, rate_l2, loss (`train` calls .item() only on log epochs)
History = dict[str, list]     # keys: epoch, mse, weight_l2, rate_l2, loss  (logged every log_int epochs); plus the monitor_* keys below when monitored
def train(cfg: Config, dataset: Dataset, net: MLP, cache_dir: Path, log=print,
          monitor: Callable[[MLP], dict[str, list]] | None = None) -> tuple[MLP, History]
    # resume from latest CACHE_DIR/train/ckpt_*.pt (net + optimiser + history + epoch + RNG state); return final.pt directly if present
    # checkpoint every save_int epochs and at the end (final.pt); hash = section_hash("data","net","train")
    # monitor: required exactly when cfg.train.monitor_int > 0 (ValueError otherwise, before anything runs)
def train_from_config(cfg: Config, dataset: Dataset, cache_dir: Path, log=print, monitor=None) -> tuple[MLP, History]
    # make_net from cfg.net / cfg.dtype / the dataset's x and y widths on resolve_device(cfg), then train(...); shared by `fct train` and the sweep
def load_trained(cache_dir: Path, cfg: Config, device, dtype) -> tuple[MLP, History]   # rebuilds net with make_net and loads final.pt (history incl. monitor_* keys); raises FileNotFoundError if absent
```

Monitor hook: `monitor(net)` is called under `torch.no_grad()` with the global RNG state restored afterwards,
at the START of every epoch e with `e % monitor_int == 0` (after e's periodic checkpoint, before its step; so e = 0
records the initial net) and once more after the last step with e = `n_epochs` (the trained net; never a duplicate).
Each call appends e to `history["monitor_epoch"]` (list[int]) and, for every key `k` of its result, the value to
`history["monitor_<k>"]` (one entry per call). Every call must return the same keys. A checkpoint written at epoch e
does not yet hold e's entry; the resumed run records it, so resuming reproduces the uninterrupted history exactly.
With `monitor_int == 0` the history keys and values are those of an unmonitored run.

## monitor.py

```python
def make_mir_monitor(dataset: Dataset, cfg: Config) -> Callable[[MLP], dict[str, list]]   # ValueError if dataset.factors is None
```

The docstring of `make_mir_monitor` owns the contract (the keys `MIR_MONITOR_KEYS`, their meaning and the settings
used); `MIR_MONITOR_KEYS` is exported from `monitor.py`. One call runs the net on `dataset.x`, takes the rates phi(H)
of every hidden layer and gets each layer's numbers from one `layer_mir_report` call. The history keys are
`monitor_epoch` plus `monitor_<key>` for each of those keys, each a list over calls of lists over layers.

Cost: per layer a call bins the rates once and makes 1 + `monitor_shuffles` mutual-information passes (the raw
estimate plus one per shuffle, shared by the bias correction and the chance level), so it grows linearly with L,
`monitor_shuffles`, P, N and the number of factors. Measured (CPU, float64, 16 torch threads, N=256, L=5, P=4096, relu,
`monitor_bins=10`, `monitor_shuffles=5`): about 1.5 s per call on `object_action` (G=5; 3 factor groups) and 0.22 s on
`what_where` (m=8).

## hmc.py

```python
def sym(A), project_pd(A, floor_rel), relative_fro(A, B), kernel_alignment(A, B)   # notebook helpers
class SingleSiteHMC      # verbatim port; plus
    def state_dict(self) -> dict      # h-coordinates chain state (float64), log_eps, prev_summary, counters -> all tensors / scalars
    def load_state_dict(self, d) -> None
```

## meanfield.py

```python
@torch.no_grad()
def empirical_kernels(net, X, readout) -> tuple[Tensor, Tensor]     # notebook
@dataclass
class SolveResult: Phi: Tensor; Phi_hat: Tensor; history: dict[str, list]; samplers: list[SingleSiteHMC]
def solve(X, Y, L, lambda_w, lambda_r, phi, g, cfg: MeanFieldConfig, net=None, log=print,
          checkpoint_dir: Path | None = None, config_hash: str = "") -> SolveResult
    # Algorithm 1 exactly as the notebook; if checkpoint_dir is given, return its final.pt if present, else
    # resume from its latest ckpt_*.pt (ValueError if it was written with a net and `net` is None, or vice
    # versa); write ckpt_<outer:05d>.pt every cfg.save_int outer steps and final.pt at the end, which then
    # replaces the checkpoints
def run_meanfield(cfg: Config, dataset: Dataset, net: MLP, cache_dir: Path, log=print) -> SolveResult
    # X = dataset.x[:n_points], Y = dataset.y[:n_points]; lambda_w/lambda_r from cfg.train; phi/g/L from cfg.net
    # hash = section_hash("data","net","train","meanfield"); returns final.pt if present
def net_kernels(net: MLP, dataset: Dataset, cfg: Config) -> Tensor   # (L, P, P) float64
    # the trained net's empirical_kernels on exactly the solver's inputs (same slicing code as run_meanfield)
    # and cfg.meanfield.readout, the net evaluated in its own dtype; for the theory-vs-net kernels figure
def load_meanfield(cache_dir: Path, cfg: Config, device) -> tuple[Tensor, Tensor, dict]   # Phi, Phi_hat, history
```

History values are stored per outer step; arrays become lists for saving. The
history keys are exactly those of the notebook's `keys` list (`meanfield.HISTORY_KEYS`),
except that `align_net` / `dist_net` are omitted when `solve` is given no net (instead of
lists of `None`).

## analyse.py

Pure functions (no I/O, no plotting) and thin cached wrappers.

```python
@dataclass
class GridPredictions: half_width: float; n_grid: int; y_grid (p²,2); h_grid (p²,L,N); z_grid (p²,d); test_error (p,p); train_error (P,)
def grid_predictions(net, dataset, data_cfg, n_grid) -> GridPredictions   # half_width = data_cfg.var/2 as in notebook cell 13
@dataclass
class LayerPCA: explained_variance (L,k); components (L,k,p²); similarity (L,k,L,k)
def layer_pca(h_grid, phi, n_components) -> LayerPCA                      # notebook cell 15
def neuron_maps(h_grid, phi, layer, n_grid) -> Tensor (N, p, p)           # notebook cell 16
def kernel_scales(Phi) -> Tensor (L,)                                     # tr(Phi_l)/P
def get_grid_predictions(cache_dir, cfg, net, dataset) -> GridPredictions # cached: analysis/grid.pt
def get_layer_pca(cache_dir, cfg, grid: GridPredictions, phi) -> LayerPCA # cached: analysis/pca.pt

@dataclass
class ExampleOutputs: indices (n,); x (n,D_in); y_true (n,D_out); y_pred (n,D_out)      # all CPU
def pick_examples(P, n, seed) -> Tensor (n,)                  # seeded CPU randperm prefix; ValueError if n > P
def example_outputs(net, dataset, indices) -> ExampleOutputs  # net under no_grad on dataset.x[indices]; ValueError for out-of-range or non-integer indices
def decode_object_action(x, grid_size) -> (agent_rc (n,2) int, objects (n,G,G) bool)   # inverts object_action's x
def decode_what_where(x) -> image (n,9,9)                     # row-major micro-grid, pixel (3R+r, 3C+c)
def r_squared(y_true, y_pred) -> float                        # pooled: 1 - SS_res/SS_tot over all entries, SS_tot about the grand mean; ValueError if the targets are constant
def r_squared_per_output(y_true, y_pred) -> float             # each output column about its own mean, averaged uniformly (sklearn's default); ValueError if a column is constant
def split_what_where(y, n_motifs) -> (where (n,9), what (n,n_motifs))   # the one definition of the column split; ValueError on a wrong width
def prediction_report(task, y_true, y_pred, cfg: DataConfig) -> dict[str, float]
    # linear_teacher, position: {"r2_pooled", "r2_per_output"} (the notebook shows both);
    # object_action: {"exact_match" (pred > 0.5 equals the target row),
    #   "argmax_optimal" (the argmax of pred is a 1 of the target row; a tied maximum counts as NOT optimal, so a constant
    #   prediction does not score through argmax picking 'stay' first),
    #   "stay_rate" (fraction of samples whose only optimal action is stay, from y_true: what a constant 'stay' would get)};
    # what_where: {"where_accuracy", "what_accuracy"} (top-1 over the first 9 / last cfg.n_motifs outputs)
    # ValueError for a task not in data.TASKS, mismatched shapes or a wrong output count
```

## modularity.py

Pure (no I/O, no plotting), vectorised over neurons. Scores how modular a layer is for the
task factors: `I[n, k]` is the plug-in mutual information (nats) between neuron n's rate and
factor group k, `r[n] = max_k I / sum_k I`, and `MIR = (mean_n r - 1/K) / (1 - 1/K)`
(1: every neuron informative about one group; ~0: information spread evenly).

```python
rank_bins(values (P,N), n_bins) -> (P,N) int64        # equal-frequency per column from mid-ranks; ties share a bin
factor_labels(factor (P,), n_bins) -> (labels, C)     # int factors relabelled 0..C-1; float factors rank-binned
mutual_information(bins, n_bins, labels, n_labels) -> (N,)   # one flat bincount for all neurons
neuron_factor_information(rates (P,N), factors, groups=None, n_bins=10) -> FactorInformation(member_mi (N,M), group_mi (N,K), group_names)
preference(group_mi, tol=1e-12) -> (r, dead)
mir(r, K) -> float;  layer_mir(...) -> LayerMIR(mir, r, group_mi, n_dead, group_names)
network_mir(rates (P,L,N), ...) -> list[LayerMIR]     # rates are phi(H); the caller applies phi
layer_mir_report(rates (P,N), factors, groups=None, n_bins=10, n_shuffles=5, seed=0) -> LayerMIRReport(mir, mir_corrected, mir_chance, n_dead, n_dead_corrected)
                                                      # exactly the numbers of layer_mir (none / shuffle) and shuffled_mir, in one pass: one binning, one shuffle loop
shuffle_factors(factors, generator); shuffled_mir(rates, factors, groups=None, n_bins=10, seed=0, n_shuffles=1) -> float
```

Conventions: integer factors are categorical, floating ones continuous. A group's information
is the sum of its members' (the total the neuron carries about them; the information about
their joint configuration cannot be estimated by a plug-in estimator). `groups=None` makes
each factor its own group. Dead neurons (total information <= tol, e.g. constant firing) get
r = 1/K, so they add zero to the MIR numerator; `n_dead` reports them. The plug-in
estimator is biased upward and max/sum has a positive floor for unrelated neurons, so plot
`shuffled_mir` (factors jointly permuted across samples) next to every MIR curve.

Bias correction: `neuron_factor_information` / `layer_mir` / `network_mir` take
`bias_correction="none"|"shuffle"`, `n_shuffles=5`, `seed=0`. With "shuffle", each member
information (before the group sum) has its mean over `n_shuffles` joint label shuffles
subtracted and is clamped at 0; neurons whose raw total is not above twice their null total
are zeroed, so they are dead (r = 1/K). The raw MIR has a cardinality-dependent floor: at
P = 8192, 10 rate bins, noise neurons score ~0.89 with one 25-way plus two binary factors
and ~0.10 with three five-way factors; corrected, both are ~0. The notebook plots the raw
MIR, the corrected MIR and the `shuffled_mir` baseline.

Group edge cases: a factor in no group is ignored; a factor in two groups counts in both; a
constant factor is accepted (zero information, but it still counts towards K, so datasets with
a constant action bit work). Inputs are validated loudly (NaN rates or factors, mismatched
lengths, bad dtypes, K < 2).

## draw.py

Every `plot_*` takes `ax: Axes | None = None`, creates its own figure only if
none is given, returns the `ax`. `fig_*` functions build multi-panel figures
from `plot_*` and return the `Figure`. Nothing calls `plt.show()`.

```python
plot_test_error(test_error, half_width, ax=None)      # notebook cell 13 left (LogNorm, jet)
plot_train_error(y, train_error, ax=None)             # notebook cell 13 right
fig_learning(grid: GridPredictions, y) -> Figure
plot_pca_variance(explained_variance_l, ax=None); plot_pca_similarity(sim_ij, vmax, ax=None)
fig_representation(pca: LayerPCA) -> Figure           # notebook cell 15 layout
plot_neuron_map(map2d, half_width, ax=None); fig_neuron_maps(maps, half_width, n_cols, title) -> Figure   # cell 16
plot_training_history(history, ax=None)               # mse / loss vs epoch, log y
fig_solver_history(history) -> Figure                 # the notebook's plot_history dashboard
plot_example_trial(dataset, ax=None); fig_example(dataset, cfg) -> Figure   # independent of results: targets y scatter + x embedding of a few points
plot_kernel(Phi_l, ax=None, title=""); fig_kernels(Phi_theory, Phi_net, shared_scale=True) -> Figure   # theory vs net kernels per layer; shared_scale=False gives each panel its own colour scale

# Result figures. `history` is the train.py history extended by the monitor with the keys below.
plot_loss_curves(history, ax=None)                    # total loss (solid) and the RAW logged mse (dashed), weight_l2 (dotted), rate_l2 (dash-dot) vs epoch, log y;
                                                      # the raw terms are not the weighted terms of the loss, so they do not add up to it; an all-zero series is labelled "(zero)"
plot_training_history(history, ax=None)               # = plot_loss_curves
plot_mir_history(history, ax=None, which="raw", legend=True)   # one solid line per layer (LAYER_COLORS[layer]) vs monitor_epoch;
                                                      # which="raw": monitor_mir plus monitor_mir_chance dotted in the layer colour, y fixed to [-0.05, 1.05];
                                                      # "corrected": monitor_mir_corrected, lower y limit extended to min(-0.05, data min - 0.05)
plot_dead_neurons(history, ax=None, legend=True)      # monitor_n_dead_corrected per layer vs monitor_epoch
fig_mir_history(history, title="") -> Figure          # three panels sharing epoch: raw (+ baseline, legend), corrected, dead neurons
plot_prediction_scatter(y_true, y_pred, ax=None, r2=None)   # all entries, identity line; r2 (computed by the caller) shown in the legend as "R² on these examples"
plot_paired_bars(y_true, y_pred, labels, ax=None, threshold=None)   # true / predicted value per output; artists labelled, the caller makes the (figure-level) legend
plot_object_grid(agent_rc, objects, ax=None)          # one scene: squares = objects, ring+dot = agent (visible on an object); row 0 at the top (up = toward row 0); ValueError unless objects is (G,G) and the agent is inside
plot_motif_image(image, ax=None, n_macro=3, motif_size=3)   # (n_macro*motif_size)² image with the macro grid at motif_size*k - 0.5
plot_example_outputs(task, outputs: ExampleOutputs, ax=None, r2=None) -> Axes   # NOT task specific: predicted vs true scatter of all entries (task only validated)
fig_example_outputs(task, outputs, data_cfg, n_show=None) -> Figure
    # linear_teacher (4 examples: true/pred stems + scatter), object_action (6: scene + paired action bars with a 0.5 line),
    # what_where (4: image + where / what bars, titled with the argmax decisions), position (scatter), each with one figure-level legend;
    # R² on the shown outputs is computed here via analyse.r_squared (omitted if the targets are constant).
    # ValueError: task not in data.TASKS, data_cfg.task != task, empty outputs, n_show < 1, or outputs inconsistent with data_cfg
    # (grid size, n_motifs)
```

History keys. `train.py` records `epoch`, `mse`, `weight_l2`, `rate_l2`, `loss` (lists of floats). The monitor adds
`monitor_epoch` (list[int]) and `monitor_mir`, `monitor_mir_corrected`, `monitor_mir_chance`, `monitor_n_dead`,
`monitor_n_dead_corrected`: lists over monitor calls of lists over layers. The result plot functions raise
`ValueError` for a missing key (the message names it), an empty history, columns of different length, or a
different number of layers in different monitor calls.

## sweep.py

The grid of the study (3 tasks x hidden layers {1,3,5} x phi {tanh,relu} x rate_lambda {0,1e-3} = 36 settings) in one
place, so SLURM array jobs and local checks share a code path. Layout under a sweep `ROOT`:

```
ROOT/sweep.json                      # SweepConfig, written by `sweep init`
ROOT/<task>/L{L}_{phi}_r{rate}/      # one normal fct cache dir per setting, e.g. what_where/L3_relu_r0.001, linear_teacher/L1_tanh_r0
```

```python
SWEEP_TASKS = ("linear_teacher", "object_action", "what_where"); LAYERS = (1, 3, 5); PHIS = ("tanh", "relu"); RATE_LAMBDAS = (0.0, 1e-3)
class Setting(NamedTuple): task: str; L: int; phi: str; rate_lambda: float
class SweepConfig(StrictModel)   # N, P, n_epochs, lr, log_int, save_int, monitor_int (>=1), monitor_bins, monitor_shuffles,
                             # linear_teacher: LinearTeacherParams(n_features, n_latents, latent_range), object_action: ObjectActionParams(grid_size, max_objects),
                             # what_where: WhatWhereParams(n_motifs), seeds: Seeds(data, net, train) = 0, dtype = "float32", device = "auto"
    # task parameters are validated at load time by building each task's DataConfig; weight_lambda is the TrainConfig default (1e-3)
def preset(name: str) -> SweepConfig                     # "tiny": N=8, P=32, 6 epochs, CPU, seconds per setting (pipeline checks only); "full": N=256, P=4096, 15000 epochs (placeholders to tune); ValueError otherwise
def save_sweep(sweep, root) / load_sweep(root) -> SweepConfig   # ROOT/sweep.json; load raises FileNotFoundError naming the path
def expand(sweep) -> list[Setting]                       # itertools.product(SWEEP_TASKS, LAYERS, PHIS, RATE_LAMBDAS): task slowest, rate_lambda fastest; index i is stable and independent of the preset
def get_setting(sweep, index) -> Setting                 # SweepError (a ValueError) "index i is out of range; valid indices are 0..35"
def setting_dir(root, setting) -> Path                   # ROOT/<task>/L{L}_{phi}_r{rate:g}
def data_config(sweep, task) -> DataConfig               # D_in from data.task_dims (2*G*G, 81, or n_features), never repeated in the sweep
def setting_config(sweep, setting) -> Config             # data, net (seed = seeds.net), train (rate_lambda = the swept value, monitor_* from the sweep), dtype, device
def setting_status(root, setting) -> "done" | "partial" | "missing"   # done = train/final.pt exists; partial = a ckpt_*.pt but no final.pt; missing otherwise (so also a run that has not yet reached its first checkpoint)
def run_setting(sweep, index, root, log=print) -> None
    # first validates/writes config.json (SweepError if an existing one differs: the sweep changed under existing artefacts; this
    # check comes BEFORE the done check, so a changed sweep raises even for a done setting), then returns at once if done,
    # else get_dataset + train_from_config(cfg, dataset, dir, log, monitor=make_mir_monitor(dataset, cfg)); resumes a partial run exactly (see train.py)
```

All settings share the seeds, so they differ only in the swept axes. The history of a finished setting has the `monitor_*` keys of
monitor.py. The monitor runs at epochs 0, k*monitor_int for every k with k*monitor_int < n_epochs, and n_epochs, so
when n_epochs is not a multiple of monitor_int the last two entries are closer than monitor_int (n_epochs=7, monitor_int=2: 0, 2, 4, 6, 7).

## cli.py

```
python -m fct init CACHE_DIR [--set data.P=512 --set net.L=2 ...]   # writes config.json from defaults; refuses to overwrite unless --force
python -m fct train CACHE_DIR                                        # builds the MIR monitor (lazily, at its first call) when train.monitor_int > 0
python -m fct solve CACHE_DIR
python -m fct plot CACHE_DIR [--which all|learning|representation|neurons|training|solver|kernels]   # result-dependent figures -> CACHE_DIR/figures/
python -m fct plot-example CACHE_DIR                                 # result-independent figure

python -m fct sweep init ROOT [--preset tiny|full] [--set n_epochs=100 ...] [--force]   # writes ROOT/sweep.json; --set takes dotted keys like `init`
python -m fct sweep list ROOT      # "<index> <task>/L{L}_{phi}_r{rate}" per setting
python -m fct sweep count ROOT     # 36
python -m fct sweep status ROOT    # "<index> done|partial|missing <dir>" per setting, then "N done, N partial, N missing"
python -m fct sweep run ROOT [--index i]   # train setting i (a SLURM array task); without --index, all settings in order
```

The sweep subcommands exit 1 with one stderr line for a missing or invalid sweep.json, an out-of-range index, or a setting whose
config.json differs from what sweep.json builds (`SweepError`); an error raised by training itself propagates with its traceback.
`run` without `--index` continues past a failing setting, prints `error: setting i failed: <Type>: <message>` (and the traceback for
anything but a `SweepError`), and exits 1 listing the failed indices.

`--which all` (default) draws each figure whose inputs exist and lists the skipped ones — learning|representation|neurons|training need train/final.pt, solver needs meanfield/final.pt, kernels needs both; a single `--which` with missing inputs exits 1.

`--set` takes dotted keys and JSON-ish values (`--set net.phi=tanh`,
`--set data.cov=[[1,0],[0,1]]`); unknown keys fail loudly via pydantic.

## jobs/fct_sweep.sh

SLURM array over the sweep settings; written for the PI to submit, never submitted by tests or agents.

```
bash jobs/fct_sweep.sh ROOT --submit [--index N]   # sbatch --array=0-35%4 (or =N), logs in <project>/tmp/logs/
bash jobs/fct_sweep.sh ROOT --dry-run [--index N]  # print that sbatch command, run nothing
bash jobs/fct_sweep.sh ROOT --list | --status      # `fct sweep list` / `fct sweep status`
```

`ROOT` is the first argument or `$FCT_SWEEP_ROOT`; it must exist and hold a `sweep.json`. Array size comes from `fct sweep count`.
`FCT_MAX_CONCURRENT` (default 4) caps running tasks; `FCT_PYTHON` overrides `<project>/.venv/bin/python`. Each task runs
`python -m fct sweep run ROOT --index $SLURM_ARRAY_TASK_ID` on one GPU of partition `gpu`, excluding `gpu-380-10,12,14`.
`tests/test_slurm_script.py` puts a fake `sbatch` on PATH: non-submit actions must never call it; `--submit` is only exercised against it.
