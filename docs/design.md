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
  data.py          # task / data generation (Dataset, fourier_embedding, embed, grid_targets)
  model.py         # MLP architecture, make_net
  train.py         # training loop with checkpoint / resume
  hmc.py           # SingleSiteHMC sampler + linear-algebra helpers
  meanfield.py     # saddle-point solver (Algorithm 1) with checkpoint / resume
  analyse.py       # pure analysis functions + cached wrappers
  draw.py          # plotting: plot_* take an ax, fig_* assemble figures; never plt.show()
  cli.py           # argparse CLI: init / train / solve / plot / plot-example
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
    P: int = 4096; D_in: int = 64; d: int = 2; var: float = 10.0
    cov: list[list[float]] | None = None          # None -> identity
    embedding: Literal["linear", "fourier"] = "fourier"
    seed: int = 0
class NetConfig(BaseModel):
    N: int = 100; L: int = 3; g: float = 1.0
    phi: Literal["relu", "tanh", "sigmoid"] = "relu"
    seed: int = 0
class TrainConfig(BaseModel):
    n_epochs: int = 15_000; lr: float = 1e-3
    weight_lambda: float = 1e-3; rate_lambda: float = 1e-3
    log_int: int = 100; save_int: int = 100; seed: int = 0
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

Fields are range-constrained (sizes and intervals >= 1, rates and scales > 0, `target_accept` in (0,1), `tilt_floor` in [0,1); `D_in` must be even for `embedding="fourier"`), and `load_config` raises `ValidationError` naming the field.

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
@dataclass
class Dataset:
    x: Tensor   # (P, D_in)
    y: Tensor   # (P, d)
    Q: Tensor | None        # (D_in, d) orthonormal map for embedding="linear", else None
    period: float | None    # Fourier period = y.max() - y.min() used at generation time, else None

def fourier_embedding(y, D, period) -> tuple[Tensor, Tensor]      # verbatim from the notebook
def make_dataset(cfg: DataConfig, device, dtype) -> Dataset       # notebook position_task, seeded by torch.Generator(cfg.seed)
def embed(y_new: Tensor, dataset: Dataset, cfg: DataConfig) -> Tensor   # same embedding as the dataset (linear: y@Q.T; fourier: dataset.period)
def grid_targets(half_width: float, n_grid: int, device, dtype) -> Tensor  # (n_grid**2, 2) meshgrid 'ij' flattened as in notebook cell 13
def get_dataset(cache_dir: Path, cfg: Config) -> Dataset          # cache-or-compute at CACHE_DIR/data/dataset.pt, hash = section_hash("data")
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
History = dict[str, list[int | float]]     # keys: epoch, mse, weight_l2, rate_l2, loss  (logged every log_int epochs)
def train(cfg: Config, dataset: Dataset, net: MLP, cache_dir: Path, log=print) -> tuple[MLP, History]
    # resume from latest CACHE_DIR/train/ckpt_*.pt (net + optimiser + history + epoch + RNG state); return final.pt directly if present
    # checkpoint every save_int epochs and at the end (final.pt); hash = section_hash("data","net","train")
def load_trained(cache_dir: Path, cfg: Config, device, dtype) -> tuple[MLP, History]   # rebuilds net with make_net and loads final.pt; raises FileNotFoundError if absent
```

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
shuffle_factors(factors, generator); shuffled_mir(rates, factors, groups=None, n_bins=10, seed=0, n_shuffles=1) -> float
```

Conventions: integer factors are categorical, floating ones continuous. A group's information
is the sum of its members' (the total the neuron carries about them; the information about
their joint configuration cannot be estimated by a plug-in estimator). `groups=None` makes
each factor its own group. Dead neurons (total information <= tol, e.g. constant firing) get
r = 1/K, so they add zero to the MIR numerator; `n_dead` reports them. The plug-in
estimator is biased upward and max/sum has a positive floor for unrelated neurons, so plot
`shuffled_mir` (factors jointly permuted across samples) next to every MIR curve.

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
```

## cli.py

```
python -m fct init CACHE_DIR [--set data.P=512 --set net.L=2 ...]   # writes config.json from defaults; refuses to overwrite unless --force
python -m fct train CACHE_DIR
python -m fct solve CACHE_DIR
python -m fct plot CACHE_DIR [--which all|learning|representation|neurons|training|solver|kernels]   # result-dependent figures -> CACHE_DIR/figures/
python -m fct plot-example CACHE_DIR                                 # result-independent figure
```

`--which all` (default) draws each figure whose inputs exist and lists the skipped ones — learning|representation|neurons|training need train/final.pt, solver needs meanfield/final.pt, kernels needs both; a single `--which` with missing inputs exits 1.

`--set` takes dotted keys and JSON-ish values (`--set net.phi=tanh`,
`--set data.cov=[[1,0],[0,1]]`); unknown keys fail loudly via pydantic.
