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
  explanations.py  # MIR_EXPLANATION: the notebook's markdown/LaTeX definition of raw, corrected and chance-baseline MIR
  report.py        # what the results notebook shows: read-only load of a sweep ROOT, final-MIR table, prose from numbers
  cli.py           # argparse CLI: init / train / solve / plot / plot-example / sweep
  __main__.py      # `python -m fct ...` -> cli.main()
notebooks/         # build_modularity.py builds + executes modularity.ipynb (cells are one-line fct.report calls)
tests/             # pytest, synthetic fixtures only, CPU, tiny sizes
docs/design.md     # this file
docs/issues/*.md   # current work items (finished ones move to docs/issues/archive/)
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

Every artefact additionally depends on `dtype`; `section_hash` always includes it. The hash of any set of sections that
includes `train` also folds in `config.LOSS_VERSION`, the version of the loss definition in `compute_loss`. It is code,
not a config field, and must be increased whenever the loss changes meaning. Version 2 averages the rate penalty over the
samples (version 1 summed it): every train cache (`train/ckpt_*.pt`, `train/final.pt`) and meanfield cache made before that
change is rejected with `StaleCacheError` the next time it is loaded; data caches are unaffected. A sweep ROOT whose
settings were finished before the change is not silently reused: `sweep status` lists them as `stale`, `run_setting` raises
`StaleCacheError` instead of calling them done, and the results notebook shows them with a note and no results. Use fresh cache directories or
delete the old ones. Changing a config value such as the `weight_lambda` default (now 1e-4) needs no version bump: it
enters the hash by value.

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
    weight_lambda: float = 1e-4; rate_lambda: float = 1e-3
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
    def section_hash(self, *sections: str) -> str   # sha256 of canonical JSON of those sections AND the top-level dtype (and `LOSS_VERSION` when `train` is a section), first 12 hex chars; >=1 section required

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
def resample(cfg, reference: Dataset, seed, device, dtype) -> Dataset
    # fresh samples of the task `reference` came from, data seed `seed` != cfg.seed (ValueError otherwise, and for
    # a task that is not one of the three). The generators draw the task definition (linear teacher W_X, W_Y;
    # what-where motif set) from the same stream as the samples, so the new samples are re-expressed with the
    # definition in reference.extras (object_action has none: the new seed is enough). Generators are unchanged.
```

## model.py

```python
class MLP(nn.Module)   # verbatim from the notebook: __init__(D_in, D_out, N, L, g, phi, device, dtype, generator=None); forward(x) -> (H (P,L,N), Z (P,D_out)); weight_l2()
def make_net(cfg: NetConfig, D_in: int, D_out: int, device, dtype) -> MLP   # weights drawn on CPU from a local torch.Generator seeded with cfg.seed, then moved to device (no global RNG touched); phi = get_phi(cfg.phi)
```

## train.py

```python
def compute_loss(net, x, y, weight_lambda, rate_lambda) -> tuple[Tensor, dict[str, Tensor]]
    # loss = 0.5*N*mse + weight_lambda*weight_l2 + rate_lambda*rate_l2
    # weight_l2 = sum of squares of every parameter, rate_l2 = phi(h)^2 summed over neurons and layers and AVERAGED over the P samples (the notebook sums over the samples; the library departs from it by design)
    # returns a dict of detached 0-d tensors: mse, weight_l2, rate_l2, loss (`train` calls .item() only on log epochs)
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
    # rate tilt: the solver's tilt is a per-sample coefficient of phi^T phi summed over the n_points solved points, whereas the
    # trained loss averages the penalty over its P_train samples (per-sample coefficient rate_lambda / P_train). Taking
    # rate_tilt = rate_lambda would therefore describe an objective P_train times too strong, so there is NO default: ValueError
    # if train.rate_lambda != 0 and meanfield.rate_tilt is None. rate_tilt = rate_lambda / P_train is the same objective when
    # n_points == P_train; with fewer points the solved subset is a different problem, and rate_lambda / n_points treats it as
    # the whole training set (as the mapping alpha = lambda_w n_points D_out / beta already does). The code forces an explicit choice until the PI decides the default mapping.
    # X = dataset.x[:n_points], Y = dataset.y[:n_points]; lambda_w/lambda_r from cfg.train; phi/g/L from cfg.net
    # hash = section_hash("data","net","train","meanfield"); returns final.pt if present
def net_kernels(net: MLP, dataset: Dataset, cfg: Config) -> Tensor   # (L, P, P) float64
    # the trained net's empirical_kernels on exactly the solver's inputs (same slicing code as run_meanfield)
    # and cfg.meanfield.readout, the net evaluated in its own dtype; for the theory-vs-net kernels figure
def load_meanfield(cache_dir: Path, cfg: Config, device) -> tuple[Tensor, Tensor, dict]   # Phi, Phi_hat, history
    # raises RateTiltError (a ValueError) like run_meanfield when train.rate_lambda != 0 and meanfield.rate_tilt is None (P taken from cfg.data.P), so a result made before the check cannot load under an unstated objective; `fct solve` and `fct plot` turn it into one `error:` line (--set meanfield.rate_tilt=...)
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
                                                      # the legend sits outside the axis, to the right, so it never covers a curve
fig_loss_curves(history, title="") -> Figure          # plot_loss_curves as a full-width (FIG_WIDTH) titled figure
plot_training_history(history, ax=None)               # = plot_loss_curves
plot_mir_history(history, ax=None, which="raw", legend=True)   # one solid line per layer (LAYER_COLORS[layer]) vs monitor_epoch;
                                                      # which="raw": monitor_mir plus monitor_mir_chance dotted in the layer colour, y fixed to [-0.05, 1.05];
                                                      # "corrected": monitor_mir_corrected, same range (the corrected MIR is >= 0 up to round-off; the lower limit extends only if a value goes below -0.05)
plot_dead_neurons(history, ax=None, legend=True)      # monitor_n_dead_corrected per layer vs monitor_epoch
fig_mir_history(history, title="") -> Figure          # three panels sharing epoch: raw (+ baseline, legend), corrected, dead neurons
plot_prediction_scatter(y_true, y_pred, ax=None, r2=None)   # all entries, identity line; r2 (computed by the caller) shown in a legend below the axis as "R² on these examples"
plot_paired_bars(y_true, y_pred, labels, ax=None, threshold=None)   # true / predicted value per output; artists labelled, the caller makes the (figure-level) legend
plot_object_grid(agent_rc, objects, ax=None)          # one scene: squares = objects, ring+dot = agent (visible on an object); row 0 at the top (up = toward row 0); ValueError unless objects is (G,G) and the agent is inside
plot_motif_image(image, ax=None, n_macro=3, motif_size=3)   # (n_macro*motif_size)² image with the macro grid at motif_size*k - 0.5
plot_example_outputs(task, outputs: ExampleOutputs, ax=None, r2=None) -> Axes   # NOT task specific: predicted vs true scatter of all entries (task only validated)
fig_example_outputs(task, outputs, data_cfg, n_show=None) -> Figure
    # linear_teacher (4 examples: ONE axis per example, shared x = output index and shared y, true as a hollow ring on a stem and
    # predicted as a filled diamond at the same x, one figure-level legend, + scatter of all entries), object_action (6: scene + paired action bars with a 0.5 line),
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
    # task parameters are validated at load time by building each task's DataConfig; weight_lambda is the TrainConfig default (1e-4)
def preset(name: str) -> SweepConfig                     # "tiny": N=8, P=32, 6 epochs, CPU, seconds per setting (pipeline checks only); "full": N=256, P=4096, 15000 epochs (placeholders to tune); ValueError otherwise
def save_sweep(sweep, root) / load_sweep(root) -> SweepConfig   # ROOT/sweep.json; load raises FileNotFoundError naming the path
def expand(sweep) -> list[Setting]                       # itertools.product(SWEEP_TASKS, LAYERS, PHIS, RATE_LAMBDAS): task slowest, rate_lambda fastest; index i is stable and independent of the preset
def get_setting(sweep, index) -> Setting                 # SweepError (a ValueError) "index i is out of range; valid indices are 0..35"
def setting_dir(root, setting) -> Path                   # ROOT/<task>/L{L}_{phi}_r{rate:g}
def data_config(sweep, task) -> DataConfig               # D_in from data.task_dims (2*G*G, 81, or n_features), never repeated in the sweep
def setting_config(sweep, setting) -> Config             # data, net (seed = seeds.net), train (rate_lambda = the swept value, monitor_* from the sweep), dtype, device
def setting_status(root, sweep, setting) -> "done" | "partial" | "missing" | "stale"   # done = train/final.pt exists with the config hash of this setting (data, net, train, loss version; only the hash is read); stale = a final.pt with another hash; partial = a ckpt_*.pt but no final.pt; missing otherwise (so also a run that has not yet reached its first checkpoint)
def run_setting(sweep, index, root, log=print) -> None
    # first validates/writes config.json (SweepError if an existing one differs: the sweep changed under existing artefacts; this
    # check comes BEFORE the done check, so a changed sweep raises even for a done setting), then returns at once if done,
    # else get_dataset + train_from_config(cfg, dataset, dir, log, monitor=make_mir_monitor(dataset, cfg)); resumes a partial run exactly (see train.py)
```

All settings share the seeds, so they differ only in the swept axes. The history of a finished setting has the `monitor_*` keys of
monitor.py. The monitor runs at epochs 0, k*monitor_int for every k with k*monitor_int < n_epochs, and n_epochs, so
when n_epochs is not a multiple of monitor_int the last two entries are closer than monitor_int (n_epochs=7, monitor_int=2: 0, 2, 4, 6, 7).

Size and style: the four figures read in the results notebook (`fig_loss_curves`, `fig_mir_history`, `fig_example_outputs`,
`fig_mir_heatmaps`) are built under `plt.rc_context(STYLE_RC)` (the `_styled` decorator), so they look the same wherever they
are called from (the CLI's `plot` figures are other functions and keep their own sizes). Fonts and line widths are centralised in
`draw.py`: `STYLE_RC` (font 14, axes title 16, labels 14, ticks 12, legend 13, figure title 18, line width 2.5, marker size 8,
axes line width 1.2), `LINE_WIDTH`, `MARKER_SIZE`, `ANNOTATION_SIZE` (heatmap cell numbers, 12) and the widths `FIG_WIDTH = 13` and
`HEATMAP_WIDTH = FIG_WIDTH + 1` inches. Function bodies take text sizes from the rc parameters rather than hard-coding them.
Heights stay in each `fig_*` and grow with the number of rows of panels. The legends of `plot_loss_curves` (right of the axis) and
`plot_prediction_scatter` (below it) assume such a constrained-layout figure and may clip in a caller-supplied `ax`. The notebook
sets `figure.dpi = 100` (about 1300 px wide; the executed notebook of the 36-setting sweep is ~14 MB and is not committed).

## report.py and notebooks/

The results notebook `notebooks/modularity.ipynb` is generated: `notebooks/build_modularity.py` holds its source
(`build_notebook()`: the intro, parameter and setup cells, the markdown cell `explanations.MIR_EXPLANATION` (equations defining
the rates, rank bins, plug-in MI, preference, MIR, dead neurons, the raw MIR and its cardinality floor, the shuffled chance
baseline and the corrected MIR, and how to read the plots and tables; it restates `fct/modularity.py`, and a test checks its
numbers that have a source in the code (dead threshold `modularity.DEAD_TOL`, noise factor `modularity.NOISE_FACTOR`, the
`TrainConfig` monitor defaults) are filled in from it, and tests recompute the quoted floor examples and check the `$`/`$$` equations are balanced), a summary cell per task, then a cell per setting, each selecting its setting by name with
`find_result(RESULTS, task, L, phi, rate)`, so a reordered grid fails loudly instead of mislabelling), and
`execute_notebook(nb, root)` runs a deep copy in a fresh kernel (no timing metadata, so equal runs give equal notebooks) after
`with_root` has put `ROOT = '<root>'` into the tagged parameter cell; the source `nb` stays output-free. In the source the cell
is `ROOT = None`, which falls back to `$FCT_SWEEP_ROOT`. `python notebooks/build_modularity.py ROOT` writes the executed
notebook to the git-ignored `tmp/modularity.ipynb` (it records the ROOT it was run on and its figures are large). Every code cell after setup is one `show(...)` call into `fct.report`,
which carries all logic and tests. Only the source notebook (`--no-execute`, no outputs) is committed.

```python
@dataclass SettingResult: setting; status ("done"|"partial"|"missing"|"stale"); directory (relative to ROOT); note; history, examples (ExampleOutputs), metrics, data_cfg, epoch, n_epochs
    # the data fields are None unless done, or partial with a readable checkpoint; epoch = epochs completed (n_epochs for done), n_epochs = the sweep's total
    # .provisional = status partial and history is not None (a running setting shown from its latest checkpoint); .progress = "epoch X of N" (ValueError if no data)
    # .mir_epoch = epoch of the last monitor call in the history: a checkpoint at epoch s never holds the monitor call of epoch s, so it lags `epoch`
    # note = why a partial setting has no data (dataset file missing, checkpoint unreadable or from another config)
@dataclass FinalMIR: task; labels; status; mir_epoch, epoch, n_epochs (per row; None unless provisional); raw, corrected, chance  (n_settings, 5) arrays, NaN for absent layers and settings with no data
    # .provisional (derived: partial with data); a provisional row holds the MIR of its latest monitor call (NaN if none yet), so the arrays
    # include provisional rows: use .finished_only() (provisional rows blanked) for any statistic. display_labels() add "* (MIR@m, ckpt e/N)"
    # to provisional rows (the MIR's epoch m is not the checkpoint's); row_status() gives draw.py's "provisional"

def sweep_root_from_env() -> Path               # $FCT_SWEEP_ROOT; RuntimeError if unset
def load_results(root) -> list[SettingResult]   # 36 entries in grid order; FileNotFoundError if no sweep.json
    # read-only: nothing is generated, cached or written under ROOT (a running job may be writing to it); settings are read on the CPU,
    # validated against the config rebuilt from sweep.json. done: from final.pt (StaleCacheError if written under another config).
    # partial: from the latest checkpoint via train.load_latest_checkpoint -> the same data as a done setting (history exactly as
    # checkpointed, examples and metrics from the checkpointed weights) plus epoch/n_epochs, status stays "partial". The checkpoint
    # being replaced while read is retried once. A checkpoint that cannot be read, is from another config hash, or is inconsistent
    # (corrupt file, StaleCacheError, epoch > n_epochs, weights that do not fit the net) gives status "partial", no data and a `note`; an unreadable
    # dataset file gives a note "dataset not readable". Exceptions that are not about a bad file (a bug in the model code) propagate.
    # A dataset file that is missing (done or partial) also gives "partial" with a note, no data. No checkpoint: "missing".
    # A checkpoint at epoch 0 or without a monitor entry loads (no loss curve / MIR yet; the prose says so).
def find_result(results, task, L, phi, rate_lambda) -> SettingResult    # ValueError if absent
def final_mir(results, task) -> FinalMIR        # last monitor call; ValueError for a done setting without monitor history
def mir_table_markdown(fm) / summary_text(fm) / overview_text(results) / setting_text(result)   # prose generated from the numbers
    # summary_text: "N of M settings finished, K provisional (still running)"; extremes of the corrected MIR over layer-cells with ties
    # listed; means pool the layer-cells where raw, corrected and chance are all finite and give their count. ALL statistics are over
    # FINISHED settings only (provisional rows are blanked first); provisional settings are listed apart with their epoch.
    # overview_text: "36 settings: 12 done, 4 partial (provisional), 20 missing" (+ ", N partial without data" for unusable checkpoints, ", N stale" for finished settings of another config or loss definition; a stale setting has a note, no data)
    # mir_table_markdown: provisional rows are named "label \* (MIR@m, ckpt e/N)" and a footnote (MIR from epoch m; checkpoint at epoch e of N) says the values are not final
    # setting_text raises ValueError naming any history key a DONE setting needs but lacks; a provisional one is headed
    # "partial, epoch X of N", says "so far", and states when there is no logged epoch or no MIR yet
def setting_items(result) -> list[str | Figure] # prose + loss curves + example outputs + MIR history; prose only (with the reason) if no data.
    # A provisional setting's figures are titled "<task> <label> — partial, epoch X of N"; a figure with no data yet (no logged epoch,
    # no monitor call) is omitted
def task_items(results, task) -> list[str | Figure]    # summary prose, table, draw.fig_mir_heatmaps
def show(items)                                 # the only IPython use
```

`draw.py` has the heatmaps: `plot_mir_heatmap(values, row_labels, row_status, title, ax, vmin, vmax, colorbar)` (NaN cells grey, unfinished
rows labelled with their status, ValueError instead of clipping a value outside [vmin, vmax]) and `fig_mir_heatmaps(panels, row_labels,
row_status, title)` (one shared scale [min(0, data), max(1, data)] and one colourbar; values outside [0, 1] are shown, not clipped).
A row status `draw.PROVISIONAL` ("provisional") keeps its values and gets an orange hatch (`PROVISIONAL_COLOR`, `PROVISIONAL_HATCH`) over the row;
the colour scale is computed from the finished rows only, so a provisional value cannot rescale the figure: its cell colour is clipped to the scale
while its printed number is exact (`fig_mir_heatmaps` says so in `PROVISIONAL_NOTE` under the panels). `fig_example_outputs(..., title="")` heads the figure; with a title the legend moves to the bottom
(constrained layout lets a top legend and a title overlap).
`train.load_latest_checkpoint(cache_dir, cfg, device, dtype) -> (net, history, epoch)` is the read-only counterpart of `load_trained` for the latest checkpoint
(OSError if none or if it fails twice; ValueError naming the file if unreadable, from another config, or its weights do not fit the net; the read is
repeated once after an OSError, e.g. a vanished file or a stale NFS handle during checkpoint rotation).

A missing setting, or a partial one without usable data, renders as one bold line naming it, the reason and its directory (relative to ROOT); its row in
the table and heatmap is labelled with the status. A running setting is drawn like a finished one but labelled "partial, epoch X of N" everywhere, its
table and heatmap rows are marked provisional, and it never enters a summary statistic, so a provisional number cannot be read as a final one. Prediction metrics are on the whole training set (the sweep has no held-out data).

## lrtune.py and lrplot.py (learning-rate tuning)

A separate tool with its own entry point, `python -m fct.lrtune` (not a subcommand of `python -m fct`; folding it
into `cli.py` is not planned for now). Every trial is a normal sweep: one ROOT per (lr, seed), `trial_root(base,
lr, seed)` = `BASE/lr_<lr:g>_seed<seed>`, written with `sweep.run_setting`, so tuning uses the production path.

```python
# result rows: dict with the keys of COLUMNS (also the CSV schema, docs/lr_tuning_results.csv)
COLUMNS = task, L, phi, rate_lambda, lr, seed, n_epochs, diverged, train_mse, heldout_mse, null_mse (held-out
          MSE of the zero net), loss (final regularised loss on the training data), max_spike, tail_drop,
          mir_corrected (final corrected MIR per layer, ';'-joined)
evaluate_setting(root, sweep, setting, held_out_seed) -> Row   # read-only; FileNotFoundError if not done or no
    # dataset cache; ValueError if the history lacks loss / monitor entries; held-out data from tasks.resample,
    # seed = training data seed + HELD_OUT_SEED_OFFSET
loss_curve_stats(history) -> {diverged, max_spike, tail_drop}   # >= 2 logged points, loss > 0 else ValueError
write_rows_csv / read_rows_csv(path)                            # typed round trip, ValueError on a wrong header
# pure table logic (no disk):
collapse_seeds(rows, keys)        # median over seeds per (setting, lr); collapsed rows have n_seeds, no seed
filter_rows(rows, seeds, n_epochs, lrs); learned_settings(rows, margin)   # learned = longest run beats 0.9 x null_mse
average_ranks(values, rtol)       # ties within rtol (TIE_RTOL = 1e-3) share the mean rank, non-finite last
rank_learning_rates(rows, tie_rtol, key) -> [Score]     # rank / gap to the setting's best, per (setting, lr)
select(scores) / select_by(scores, keys) -> Selection   # lowest mean rank among the fewest-failure lrs; ties listed
global_verdict(scores, max_median_gap=0.10, max_ratio=2.0) -> Verdict
report_markdown(rows, key="loss", learned_only, ...)    # a verdict only for key loss or learned_only
run_trials(template, base, lrs, seeds, indices) / collect_rows(...)     # I/O edge, resumable, BASE/timings.csv
# command line: init BASE --N --P --epochs [--device]; run / collect BASE --lrs [--seeds --tasks --layers];
#               report CSV... [--key loss|heldout_mse|train_mse --learned-only --seeds --n-epochs --lrs
#                              --max-median-gap --max-ratio --fig-dir]
```

lrplot.py draws one result column against lr per task and depth (`plot_metric_vs_lr(rows, task, L, key, best_lr,
logy, ax=None)`, `fig_metric_vs_lr(rows, key, ylabel, logy, best_by_group)`), in the shared `draw.STYLE_RC`
style; a diverged run is a red cross at the top of its panel.

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
python -m fct sweep status ROOT    # "<index> done|partial|missing|stale <dir>" per setting, then "N done, N partial, N missing, N stale"
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

SLURM array over the sweep settings; written for the PI to submit. Tests exercise `--submit` only against a recording fake `sbatch` on PATH; agents never run it.

```
bash jobs/fct_sweep.sh ROOT --submit [--index N]   # sbatch --array=0-35%4 (or =N), logs in <project>/tmp/logs/
bash jobs/fct_sweep.sh ROOT --dry-run [--index N]  # print that sbatch command, run nothing
bash jobs/fct_sweep.sh ROOT --list | --status      # `fct sweep list` / `fct sweep status`
```

`ROOT` is the first argument or `$FCT_SWEEP_ROOT`; it must exist and hold a `sweep.json`. Array size comes from `fct sweep count`.
`FCT_MAX_CONCURRENT` (default 4) caps running tasks; `FCT_PYTHON` overrides `<project>/.venv/bin/python`. Each task runs
`python -m fct sweep run ROOT --index $SLURM_ARRAY_TASK_ID` on one GPU of partition `gpu`, excluding `gpu-380-10,gpu-380-12,gpu-380-14`.
`--index` (digits only, read as base 10) applies to `--submit`/`--dry-run` only; a comma in the project or python path is rejected
because `sbatch --export` is comma-delimited; a relative `FCT_PYTHON` is resolved against the caller's directory.
`tests/test_slurm_script.py` puts a fake `sbatch` on PATH: non-submit actions must never call it; `--submit` is only exercised against it.

## spatial.py (spatial embeddings and weight kernels)

Every unit of the input, each hidden layer and the output gets a position on the torus `[0, box)^d` (default `d = 1`, a ring).
The weight prior of layer l is `K^l = K^post (x) K^pre`; the penalty is `M = K^{-1}`, built per factor so the
`(n_post n_pre)^2` matrix is never formed. Not yet part of `Config`; the penalty is applied in a later item.

```python
# config (strict pydantic, like fct.config)
EmbeddingSpec(kind="grid"|"random", shape=None, seed=0)   # grid shape must multiply to n; None -> n points for d = 1
KernelSpec(length_scale=0.1, amplitude=1.0)               # length scale in box units
PopulationSpec(embedding, kernel)
SpatialConfig(d=1, box=1.0, input, hidden, output, min_rel_eig=1e-6)   # all hidden layers share `hidden`

# pure computation (float64, CPU)
make_embedding(spec, n, d, box, seed_offset=0) -> Embedding(positions (n, d), box)
torus_distance(x, y, box) -> (n, m)                       # Euclidean norm of per-axis minimum-image differences
periodic_gaussian_kernel(x_post, x_pre, length_scale, box, amplitude) -> (n_post, n_pre)   # amp * exp(-d^2 / 2 ell^2)
penalty_matrix(K, min_rel_eig) -> (M, FloorReport(n_floored, n_negative, cond_before, cond_after))
build_spaces(cfg, m, N, L, n_out) -> [LayerSpace(layer, pre, post)]   # L + 1 layers; M_pre, M_post properties
draw.fig_kernel(space) -> Figure      # K_pre, K_post, K vs torus distance with the analytic curve, positions (d = 1 ring, d = 2 plane)
```

Populations are indexed 0 (input, m), 1..L (hidden, N), L + 1 (output, n_out); layer l maps population l to l + 1, matching
`MLP`: U is layer 0, J_1..J_{L-1} the hidden layers, the readout V layer L. The post `Population` of layer l is the same object
as the pre `Population` of layer l + 1 (embedding, K, M computed once). Random hidden embeddings use seed
`hidden.embedding.seed + k` for hidden layer k, so layers differ and runs repeat.

`M` comes from the symmetric eigendecomposition with eigenvalues floored at `min_rel_eig` times the largest; the floor is the only
approximation and `FloorReport` says how many eigenvalues it touched and the condition number before and after. Caveat: the
minimum-image Gaussian is positive semi-definite only for a length scale small against the box (about ell <= box / 10 in the
tests); a wider one has negative eigenvalues, which the floor also clamps and `n_negative` counts.
