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
  schema.py        # StrictModel and the constrained field types shared by every config model
  config.py        # pydantic Config (data/net/train/meanfield/analysis/spatial/device), JSON I/O, section hashes
  activations.py   # PHI registry (name -> callable), get_phi, phi_prime
  cache.py         # cache-directory layout + load_or_compute helpers (I/O layer)
  data.py          # task / data generation (Dataset, fourier_embedding, embed, grid_targets, task_dims)
  tasks.py         # pure generators of the linear-teacher, object-action and what-where tasks
  model.py         # MLP architecture, make_net
  train.py         # training loop with checkpoint / resume (Adam or SGLD)
  sgld.py          # SGLD torch optimiser
  hmc.py           # SingleSiteHMC sampler, the pure single-site density pieces, linear-algebra helpers
  importance.py    # ImportanceSampler: self-normalised IS of the single-site density from the prior
  meanfield.py     # saddle-point solver (Algorithm 1) with checkpoint / resume
  analyse.py       # pure analysis functions + cached wrappers
  modularity.py    # neuron-factor mutual information, preference, MIR (pure)
  tuning.py        # tuning_curves (MIR's bins) and select_neurons: pure helpers for the grid notebook
  grid_report.py   # MirRecord and mir_grid: sweep records -> (layer, lambda_r, lambda_w) MIR array
  grid_view.py     # sweep ROOT -> phase-transition figures: setting snapping, MIR records of one depth/activation, tuning of one layer
  monitor.py       # make_mir_monitor: the training hook that records per-layer MIR
  spatial.py       # torus embeddings, wrapped Gaussian weight kernels K = K_post (x) K_pre, penalty matrices M = K^{-1}, the factorised weight quadratic form (pure)
  spatial_grid.py  # spatial-grid notebook: pick one setting of a spatial sweep by axis values, load its trained net and the 3^k latent-lattice test set
  spatial_view.py  # spatial-prior notebook: config, resumable run, ring/lattice layout of neurons and weights, empirical kernels (pure + one I/O function)
  draw.py          # plotting: plot_* take an ax, fig_* assemble figures; never plt.show()
  sweep.py         # a grid of Config fields: SweepConfig, expand, run_setting, status, presets
  explanations.py  # MIR_EXPLANATION: the notebook's markdown/LaTeX definition of raw, corrected and chance-baseline MIR
  report.py        # what the results notebook shows: read-only load of a sweep ROOT, final-MIR table, prose from numbers
  exemplar.py      # the exemplar notebook's logic: config from flat settings, cache layout, cached train / solve, live refresh
  cli.py           # argparse CLI: init / train / solve / plot / plot-example / sweep
  __main__.py      # `python -m fct ...` -> cli.main()
notebooks/         # build_modularity.py, build_phase_transition.py, build_spatial.py, build_spatial_grid.py, build_exemplar.py build + execute the notebooks (cells are one-line library calls, except build_spatial.py's figure cells, which define their own plotting functions)
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
| meanfield  | data, net, train, meanfield (problem + hashed solver fields) |
| analysis   | data, net, train, analysis                 |

Every artefact additionally depends on `dtype`; `section_hash` always includes it. The hash of any set of sections that
includes `train` also folds in `config.LOSS_VERSION`, the version of the loss definition in `compute_loss`. It is code,
not a config field, and must be increased whenever the loss changes meaning. Version 3 sums the rate penalty over the
samples (version 1 summed it, version 2 averaged it): every train cache (`train/ckpt_*.pt`, `train/final.pt`) and meanfield cache made before that
change is rejected with `StaleCacheError` the next time it is loaded; data caches are unaffected. A sweep ROOT whose
settings were finished before the change is not silently reused: `sweep status` lists them as `stale`, `run_setting` raises
`StaleCacheError` instead of calling them done, and the results notebook shows them with a note and no results. Use fresh cache directories or
delete the old ones. Likewise the hash of any set of sections that includes `net` folds in `config.MODEL_VERSION`, the
version of the forward pass of `MLP`. Version 2 reads the output from the post-activations, `Z = V^T phi(h^L) / N`
(version 1 read it from the pre-activation `h^L`): every train and meanfield cache made under version 1 is rejected with
`StaleCacheError`, and a sweep ROOT of version-1 runs shows them as `stale`. `train.optimizer` and `train.beta` (SGLD, below) are hashed only when `optimizer == "sgld"`: for Adam both are dropped from the
hashed train section, so every pre-existing Adam hash and cache is unchanged. The hash of any set of sections that includes `train` folds in `config.LOSS_VERSION`; version 4 scales the weight prior by `1 / (2 beta)` (see "Prior scaling convention" below), so every cache made under version 3 is rejected with `StaleCacheError`. Changing a config value such as the `weight_lambda` default (now 2e-4) needs no version bump: it
enters the hash by value.

The meanfield section enters every hash as `MeanFieldConfig.hash_payload()`, not as its nested dump: the problem fields
and the solver fields that existed before the sampler choice (`config.LEGACY_SOLVER_FIELDS`, `log_int` and `save_int`
included) in one flat dict, which is how the section was hashed before it was split. The step-control fields
(`config.STEP_CONTROL_FIELDS`: `inner_step_control`, `eta_max_phihat`, `phi_step_scale`, `inner_start`) are added each
only when it differs from its value in `config.STEP_CONTROL_LEGACY` (`health`, 1, `trace`, `previous`: with
`eta_max = 1` the plain Algorithm 1): a config that sets those values explicitly keeps the hash it had before the
fields existed (`tests/fixtures/legacy_config.json` does), while one that leaves them unset gets the new defaults, a
different iteration, and a new hash, as does every config with the new default `eta_max` (0.25). `sampler` and the importance fields (`config.IMPORTANCE_FIELDS`) are added only when
`sampler="importance"`: switching sampler makes a result stale, importance settings never touch an HMC result. Every
solver field is in exactly one of the three lists or in `config.UNHASHED_SOLVER_FIELDS` (`tilt_tripwire_resid`, `tilt_tripwire_steps`, a check on
the result that changes no number of it; a test enforces it), so a new field has to be given a hash rule explicitly. The train and analysis hashes never include the meanfield section, so solver settings never make a training
cache stale.

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
    phi: Literal["relu", "tanh", "sigmoid", "id"] = "relu"   # "id": phi(h) = h, a linear net
    seed: int = 0
class TrainConfig(BaseModel):
    n_epochs: int = 15_000; lr: float = 1e-3
    weight_lambda: float = 2e-4; rate_lambda: float = 1e-3
    log_int: int = 100; save_int: int = 100; seed: int = 0
    monitor_int: int = 0                # epochs between monitor calls (>= 0); 0 disables the monitor
    monitor_bins: int = 10              # rate bins of the MIR monitor (>= 2)
    monitor_shuffles: int = 5           # label shuffles for the corrected MIR and the chance level (>= 1)
    optimizer: Literal["adam", "sgld"] = "adam"
    beta: float = 1.0                   # SGLD inverse temperature (> 0); ignored, and not hashed, for adam
class MeanFieldConfig(BaseModel):      # the problem: the equations being solved
    n_points: int = 100                 # solve on the first n_points training points (X=x[:n_points])
    beta: float | None = None   # None: train.beta for SGLD, 1 for Adam (meanfield.resolve_beta)
    prior_scale: float | None = None; alpha: float | None = None   # None: follow the network (meanfield.network_constants)
    learning_prefactor: float | None = None; rate_tilt: float | None = None
    solver: SolverConfig = SolverConfig()
    def hash_payload(self) -> dict      # what enters section hashes (see the cache section above)
    # before-validator: a flat section in the pre-split layout (solver keys next to the problem keys) loads, its solver
    # keys moved into `solver`; a key given both flat and in `solver` is a ValidationError naming it
class SolverConfig(BaseModel):        # how it is solved
    sampler: Literal["hmc", "importance"] = "hmc"
    # Algorithm 1 (defaults = notebook cell 25 / solve() signature)
    n_outer: int = 300; n_inner_max: int = 30; inner_tol: float = 1e-2; inner_tol_mc: float = 2.0
    eta_phi: float = 0.05; eta_phihat: float = 0.05; eta_max: float = 0.25; eta_grow: float = 1.1
    max_backtrack: int = 8; eta_min_rel: float = 1e-3; max_div_frac: float = 0.02
    max_h2_growth: float = 10.0; max_mc_err: float = 1.0; eps_collapse: float = 1e-2
    project_tilt: bool = True; tilt_floor: float = 0.05; max_rel_step: float = 0.1
    # step control (STEP_CONTROL_FIELDS, hashed when not at STEP_CONTROL_LEGACY; see meanfield.py "Step control").
    # eta_max caps only the outer step size; the inner one is capped by eta_max_phihat, so a smaller inner step is set
    # there, not through eta_max
    inner_step_control: Literal["health", "progress"] = "progress"; eta_max_phihat: float = 0.25
    phi_step_scale: Literal["trace", "kernel"] = "kernel"; inner_start: Literal["previous", "target"] = "target"
    # STEP_CONTROL_LEGACY ("health", 1, "trace", "previous") with eta_max = 1 is the plain Algorithm 1
    # (tests/pinned_runs.LEGACY_STEP_CONTROL reproduces its pinned solves bit for bit)
    tilt_tripwire_resid: float = 0.9; tilt_tripwire_steps: int = 3   # not hashed: the tilt tripwire (see solve), not a convergence test
    pd_floor_rel: float = 1e-8; inv_floor_rel: float = 1e-4
    # sampler runs (HMC-only: n_chains, n_leap, eps0, target_accept, adapt_rate, precondition; also max_div_frac, eps_collapse)
    n_chains: int = 2048; n_trans: int = 2; n_trans_outer: int = 5; n_leap: int = 10
    eps0: float = 0.05; target_accept: float = 0.75; adapt_rate: float = 0.05
    warmup_trans: int = 200; sample_dtype: Literal["float32", "float64"] = "float32"
    jitter_rel: float = 1e-6; precondition: bool = True
    # importance sampling (read only when sampler="importance")
    n_importance_samples: int = 4096    # fresh draws per transition (n_trans / n_trans_outer transitions per run)
    importance_proposal_scale: float = 1.0   # s > 0: proposal N(0, s^2 Sigma)
    min_ess_drop: float = 0.1           # in (0, 1]: a step after which ESS / draws < min_ess_drop x that of the point before is reverted
    min_ess_frac: float = 0.01          # in [0, 1]: ESS / draws every accepted point must reach, else InadequateSamplerError
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
    spatial: SpatialConfig | None = None   # the spatial weight prior (fct.spatial); None -> weight L2 norm
    device: Literal["auto", "cpu", "cuda"] = "auto"
    dtype: Literal["float32", "float64"] = "float64"
    def section_hash(self, *sections: str) -> str   # sha256 of canonical JSON of those sections AND the top-level dtype (and `MODEL_VERSION` when `net` is a section, `LOSS_VERSION` when `train` is), first 12 hex chars; >=1 section required
        # with `train`: optimizer/beta dropped for adam; the `spatial` section added only when it is set

def default_config() -> Config
def load_config(path: Path) -> Config          # raises on unknown keys / bad types
def save_config(cfg: Config, path: Path) -> None   # indent=2, sorted keys
def resolve_device(cfg: Config) -> torch.device   # "auto" -> cuda if available else cpu
def resolve_dtype(name: str) -> torch.dtype
```

`task` selects the generator and only that task's fields are read; the fields of other tasks are ignored but still enter the cache hash (changing one makes an existing data cache stale). Fields are range-constrained (sizes and intervals >= 1, rates and scales > 0, `target_accept` in (0,1), `tilt_floor` in [0,1); `D_in` must be even for the position task with `embedding="fourier"`; for `linear_teacher`, `D_in` (the number of features m of both x and y) must exceed `n_latents`; `max_objects` >= 0 while `n_latents`, `grid_size`, `n_motifs` are >= 1 and `latent_range` > 0; `max_objects <= grid_size**2`, `n_motifs <= 511`, and for `object_action` / `what_where` `D_in` must equal the fixed 2*G*G / 81), and `load_config` raises `ValidationError` naming the field.

## activations.py

```python
PHI: dict[str, Callable]   # {"relu": torch.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid, "id": identity}
def get_phi(name: str) -> Callable          # KeyError with the valid names
def phi_prime(phi: Callable) -> Callable    # the notebook's _phi_prime_factory: analytic for relu/tanh/sigmoid/id (phi' = 1), autograd otherwise
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
def get_latent_lattice_dataset(cache_dir: Path, cfg: Config) -> Dataset   # tasks.latent_lattice_dataset(cfg.data, get_dataset(cache_dir, cfg)); no cache file of its own
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
def latent_lattice_dataset(cfg, reference: Dataset) -> Dataset
    # linear_teacher only (ValueError naming the task otherwise; ValueError if reference.extras lacks W_X / W_Y): one
    # row per point of {-a, 0, a}^k, a = latent_range, 3^k rows, z0 slowest and the last latent fastest, values in
    # the order (-a, 0, a); x = z W_X^T, y = z W_Y^T with the reference's maps; factors z0..z{k-1}, singleton groups,
    # extras = reference.extras; device and dtype of reference.x
```

## model.py

```python
class MLP(nn.Module)   # the notebook's net: __init__(D_in, D_out, N, L, g, phi, device, dtype, generator=None); forward(x) -> (H (P,L,N), Z (P,D_out)) with Z = phi(H[:, -1]) @ V / N, a linear readout of the last layer's post-activations (MODEL_VERSION 2); weight_l2()
def make_net(cfg: NetConfig, D_in: int, D_out: int, device, dtype) -> MLP   # weights drawn on CPU from a local torch.Generator seeded with cfg.seed, then moved to device (no global RNG touched); phi = get_phi(cfg.phi)
```

## train.py

```python
def compute_loss(net, x, y, prior_coefficient, rate_lambda, weight_penalty=MLP.weight_l2) -> tuple[Tensor, dict[str, Tensor]]
    # loss = 0.5*N*mse + prior_coefficient*weight_l2 + rate_lambda*rate_l2; callers pass the function
    #   prior_coefficient(cfg.train) = train.weight_lambda / (2 beta), never the raw train.weight_lambda
    # weight_l2 = weight_penalty(net): the sum of squares of every parameter by default, the spatial prior
    #   sum_l tr(M_post W_l M_pre W_l^T) from make_weight_penalty when cfg.spatial is set (see "Spatial weight prior in training")
    # rate_l2 = phi(h)^2 summed over samples, neurons and layers (as in the notebook), so it grows with P
WeightPenalty = Callable[[MLP], Tensor]
def make_weight_penalty(cfg: Config, net: MLP) -> WeightPenalty   # MLP.weight_l2 if cfg.spatial is None, else the spatial prior for net's sizes, dtype and device
def spatial_weight_penalty(net, factors: dict[str, tuple[Tensor, Tensor]]) -> Tensor   # sum over parameters of tr(M_rows P M_cols P^T)
    # returns a dict of detached 0-d tensors: mse, weight_l2, rate_l2, loss (`train` calls .item() only on log epochs)
History = dict[str, list]     # keys: epoch, mse, weight_l2, rate_l2, loss  (logged every log_int epochs); plus grad_norm, param_msd for SGLD; plus the monitor_* keys below when monitored
def train(cfg: Config, dataset: Dataset, net: MLP, cache_dir: Path, log=print,
          monitor: Callable[[MLP], dict[str, list]] | None = None) -> tuple[MLP, History]
    # resume from latest CACHE_DIR/train/ckpt_*.pt (net + optimiser + history + epoch + RNG state); return final.pt directly if present
    # checkpoint every save_int epochs and at the end (final.pt); hash = section_hash("data","net","train")
    # monitor: required exactly when cfg.train.monitor_int > 0 (ValueError otherwise, before anything runs)
def train_from_config(cfg: Config, dataset: Dataset, cache_dir: Path, log=print, monitor=None) -> tuple[MLP, History]
    # make_net from cfg.net / cfg.dtype / the dataset's x and y widths on resolve_device(cfg), then train(...); shared by `fct train` and the sweep
def load_trained(cache_dir: Path, cfg: Config, device, dtype) -> tuple[MLP, History]   # rebuilds net with make_net and loads final.pt (history incl. monitor_* keys); raises FileNotFoundError if absent
```

SGLD (`train.optimizer = "sgld"`, `fct/sgld.py`). The `SGLD(params, lr, beta)` optimiser takes the full-batch
Euler-Maruyama step `theta <- theta - lr * grad L + sqrt(2 * lr / beta) * N(0, I)` with `dt = lr`, where `L` is the
unchanged `compute_loss`. Its stationary law is `p(theta) ~ exp(-beta L)` as `lr -> 0`; at finite `lr` the stationary
variance on a quadratic of curvature `c` is `1 / (beta c (1 - lr c / 2))`. The noise is drawn from the global torch
generator, whose state is checkpointed, so a resumed SGLD run equals an uninterrupted one exactly. `optimizer="adam"`
(default) is unchanged. SGLD histories carry two extra per-logged-epoch keys: `grad_norm` (L2 norm of the full gradient of
the step) and `param_msd` (squared displacement of the step, averaged over parameters; `param_msd / (2 lr)` is an
effective temperature that tends to `1/beta` when the noise dominates the drift).

Mapping to `Latex/sections/01_setup.tex`. The Latex trains `d(theta) = -grad L_tex dt + sqrt(2/beta) eta` with
`L_tex = N/(2Pn) sum (f - y)^2 + (1/beta) P_weight + (1/beta) P_rates`, i.e. `p ~ exp(-beta L_tex)`. The repo loss is
`L = 0.5 N mse + weight_lambda / (2 beta) weight_l2 + rate_lambda rate_l2` (`weight_l2` is the spatial prior when one is set; `beta = 1` for Adam) and `mse` is the mean over the P*n entries, so the data
terms agree and `beta L_tex = beta (0.5 N mse) + P_weight + P_rates`. SGLD on `L` at inverse temperature `beta` samples
`exp(-beta L) = exp(-beta 0.5 N mse - weight_lambda/2 weight_l2 - beta rate_lambda rate_l2)`. The Latex regulariser
strengths are therefore `P_weight = weight_lambda/2 weight_l2` (independent of `beta`) and `lambda_tex = beta * rate_lambda` for the rates. `beta` sets only the noise; `lr` is the step
size `dt`.

#### Prior scaling convention

The weight prior enters the loss as `weight_lambda / (2 beta) * tr(M_post W M_pre W^T)` per layer (the iid prior `K = M = I` included: the plain L2 is
`weight_lambda / (2 beta) * sum W^2`), where `beta` is `train.beta` under SGLD and 1 under Adam (`fct.train.prior_coefficient`). The `1/beta` cancels the `beta` of the
stationary law, so `vec(W) ~ N(0, K / weight_lambda)` for every `beta` (`K = M^-1`); the data and rate terms carry the `beta`. The recorded `weight_l2` history is the raw quadratic form
(no coefficient), so it does not depend on `beta`. Consequences for the mean-field constants (`fct.meanfield.network_constants`, for the standard-normal-parametrised
weights and the 1/N readout), which are the solver's defaults whenever `MeanFieldConfig` leaves them unset (`beta` is `train.beta` under SGLD, 1 under Adam; an explicit value of any field wins):

| constant | value for `exp(-beta L)` |
|---|---|
| `prior_scale` | `1 / weight_lambda` |
| `alpha` | `weight_lambda P d_out / beta` |
| `rate_tilt` | `2 beta rate_lambda` (from `exp(-beta rate_lambda sum phi^2)`) |
| `learning_prefactor` | `1 / (2 prior_scale) = weight_lambda / 2` (the kernels are unscaled) |

All four are exact for SGLD. Under Adam there is no Gibbs distribution; the Adam loss is the `beta = 1` case, so its prior is `weight_lambda/2 weight_l2` and the solver uses `beta = 1`.

All `weight_lambda` values in this document, the presets and the notebooks are stated under this convention. A value chosen under the previous convention (loss prior `weight_lambda * form`) maps to the same effective loss by multiplying by `2 beta` for SGLD and by 2 for Adam (`beta = 1`); the library default is therefore `2e-4` (was `1e-4`), the spatial notebook's `60` (was `3e-3`, `beta = 1e4`) and the exemplar's `0.2` (was `1e-2`, `beta = 10`).

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
# pure pieces of the single-site density, shared with importance.py (SingleSiteHMC's methods call them):
def jittered_prior(Sigma, jitter_rel) -> (Sigma_j, chol, inv); def tilt_and_grad(h, M, lambda_r, phi, phi_prime, psi_is_phi) -> (U, u)
# psi_is_phi: psi = phi(h), which the solver uses on every layer; False gives psi = h, a Gaussian tilt kept for closed-form sampler checks
def psi_psi_rel_mc_err(psi_psi, psi2_psi2, n_eff); def stein_mismatch(h_h, h_u, Sigma)
def tilt_rule(phi, psi_is_phi) -> "gaussian" | "bounded" | "relu"   # by activations.tilt_class; ValueError for an activation without a rule
def project_tilt(M, delta, lambda_r, phi, psi_is_phi, Sigma_chol, Sigma_inv); def normalisability_margin(M64, Sigma_inv, lambda_r, phi, psi_is_phi)
# by tilt_rule: "gaussian" (psi = h, or a linear phi such as activations.identity): Sigma^-1 + Q > 0, exact, Q = M (+ lambda_r I
# for a linear phi), projected as Sigma^1/2 Q Sigma^1/2 >= -(1 - delta) I; "relu": the same condition and projection with
# Q = M + lambda_r I, the all-active orthant's (see "The relu tilt rule" below); "bounded" (tanh, sigmoid): no projection, the
# margin is lambda_min(Sigma^-1). The margin is lambda_min(Sigma^-1 + Q). An admissible M is returned unchanged.
@dataclass(frozen=True) class OrthantCheck: n_orthants; n_violated; worst_min_eig; fraction_violated (property)
def relu_orthant_check(Sigma_inv, M, lambda_r, max_P=20, chunk=2**14) -> OrthantCheck   # diagnostic / test oracle, never called by the solver
```

**The relu tilt rule.** With psi = phi = relu the exponent on the orthant whose active set is S is
h^T Sigma^-1 h + h_S^T T_SS h_S (T = M + lambda_r I). `project_tilt` enforces positive definiteness of Sigma^-1 + T, like
the Gaussian rule and delta-scaled as Sigma^1/2 T Sigma^1/2 >= -(1 - delta) I. Its exact content: the Gaussian density that
equals the relu one on the all-active orthant is normalisable. For the relu density it is neither necessary nor sufficient:
- it over-cuts: on the all-active orthant normalisability needs only strict copositivity of Sigma^-1 + T, so a tilt whose
  offending direction has mixed signs is normalisable and still cut. Learning tilts -2c A Y Y^T A with centred targets are
  of this kind;
- it under-cuts: the other orthants are not checked, so non-normalisable tilts are admitted (hand-built example in
  `tests/test_hmc.py`: Sigma^-1 = [[1, .8], [.8, 1]], M = diag(-.5, 10) passes the rule and the exponent is negative along
  h = (1, -.8); and on the exemplar, below). The samplers' Stein mismatch and ESS checks, the step validation (h^2 blow-up,
  divergences) and the tilt tripwire are meant to catch those, and on the exemplar they did not.
Measured on the exemplar's relu problem (`tests/test_relu_tilt.py`; P = 20, prior_scale 5, rate_tilt 0.02; the prior has rank
2, so a scan over the plane it spans decides normalisability, given a bound on the off-plane jitter that the test asserts):
- the learning tilt at the untilted prior's arc-cosine kernel (leading eigenvalue -0.93) is not normalisable; the rule cuts
  it to a normalisable one keeping 25 % of the leading eigenvalue (the orthant floor it replaced,
  lambda(T) >= -(1 - delta) lambda_min(Sigma^-1), kept 12 %). 0.4 x that tilt (-0.37, where the floor kept 31 %, the
  review's 33 %) is normalisable, and the rule keeps 59 % of it;
- with solver.eta_max = 0.1 (lazy start, HMC at the notebook's sizes; measured under the legacy step control, which
  `tests/pinned_runs.LEGACY_STEP_CONTROL` with eta_max = eta_max_phihat = 0.1 replays exactly) the rule admits
  non-normalisable tilts. At outer step
  5 the margin is +0.19 and the plane minimum -0.007; the chains escape (h^2/prior 3e3), the next step backtracks 21 times,
  the Stein mismatch jumps to 1.14 and the residual goes 1.3, 4.5, 16, 63, 92 over the following steps. From step 55 to 79
  the margin rides the delta floor (+0.005), the projection fires every step, and 17 of those 25 admitted densities are not
  normalisable while the Stein mismatch (0.05 to 0.6), h^2/prior (2 to 3.4) and the step validation flag nothing. A tilt on
  the floor leaves the all-active direction a precision of about 0.005 (standard deviation about 14 against the prior's
  about 3), on which HMC diverges. This runaway is driven by the rule;
- the default-step solves (lazy and net start) admitted only normalisable tilts over the 40 outer steps scanned, and so did
  the eta_phi 0.01 (lazy) and eta_max 0.1 (net start) runs at their ends; those runaways have the step-control signature
  (Stein mismatch near 1 with a normalisable density, residual climbing to 30 to 200), though the projection still fires on
  about half the steps from 18 to 60 of the default net-start solve. They are `solver-inner-step-control`'s.
Regime: positive definiteness on the active units; expect it to cut part of a strong, mixed-sign relu tilt, and to admit,
once the iteration pushes a tilt to the delta floor, densities that another orthant makes non-normalisable.

`relu_orthant_check` is the per-orthant oracle: K_S = (Sigma_SS)^-1 + T_SS > 0 for every non-empty S (the minimum of the
exponent over the inactive coordinates is h_S^T K_S h_S). A pass proves normalisability; a violated orthant only flags,
because neither the inactive coordinates' minimiser nor h_S is held to the orthant's signs (an indefinite copositive K_S is
flagged too). The size of a flag says nothing either way: on the exemplar it flags 20 of the 2^20 orthants of the projected
analytic tilt (worst eigenvalue -0.003) and 2 % (worst -0.38) of a late net-start solve's Phi_hat, both normalisable by the
plane scan, and 619367 of 2^20 (worst -1.9) of the eta_max 0.1 solve's final Phi_hat, which is not. Cost: 2^P
eigen-decompositions of P x P matrices, batched by padding each K_S to P x P; about 15 s at P = 20 on a CPU, doubling per
unit, so P > max_P (20) raises ValueError. It is a diagnostic, not part of the solver loop.

## importance.py

```python
class ImportanceSampler(Sigma, M, lambda_r, phi, psi_is_phi, n_samples=4096, proposal_scale=1.0, jitter_rel=1e-6,
                        sample_dtype=float32, generator=None)
    # the solver-facing interface of SingleSiteHMC: run(n_trans, collect) -> (moments, diag), set_prior, set_tilt,
    # get_state / set_state (no state: None), state_dict / load_state_dict ({"sampler": "importance", counter};
    # an HMC state is a ValueError), tilt_and_grad, project_tilt, normalisability_margin
```

Each transition draws `n_samples` fresh, independent h = s L z from the proposal N(0, s² Σ) (s = `proposal_scale`,
s = 1 is the prior) and weights them by log w = ½(1 - s²)|z|² - U(h), U the tilt; a run self-normalises all its
transitions' weights together (log-sum-exp per batch, combined across batches) and accumulates the moments in float64.
A non-finite log weight raises `FloatingPointError`. `run(n_trans, collect)` always draws `n_trans` batches, as
`SingleSiteHMC.run` makes `n_trans` transitions; the solver warms an importance sampler up with one transition, since
independent draws have nothing to equilibrate. Diagnostics: `ess_frac` (Kish ESS (Σw)²/Σw² over the draws), `ess`, `max_weight` (largest normalised
weight), `h2_mean` (weighted), `prior_h2`, and with collect `psi_psi_rel_mc_err` (n_eff = ESS) and `stein_mismatch`;
the HMC-only keys (`accept`, `eps`, `n_divergent`, `n_reset`, `max_abs_dH`, `mean_abs_dH`, `rho1`, `h2_max`) are NaN.
When one draw dominates, the variance-based MC error reads ~0 (one point has no spread): weight collapse is detected by
`ess_frac` / `max_weight`, which is why the solver gates on the ESS (see meanfield.py).

**Limitation: the prior proposal is for small P and mild tilts.** Its ESS fraction is a property of the point being
sampled, not of the solver's step, and it falls roughly exponentially with P and with the tilt. Measured at the start of
a relu, L=2, beta=1e3 problem, layer 1: 0.08 at P=8, 0.03 at P=16, 0.0085 at P=48 (seed-dependent: 0.013 at seed 0), 2e-4 to 2e-3 at P=100 (the default
`n_points`). More draws do not raise the fraction. At realistic P use `sampler="hmc"`.

## meanfield.py

```python
def readout_activities(H (P, L, N), phi) -> list[Tensor (P, N)]   # psi = phi(h^l) per hidden layer; the last is what the output reads
@torch.no_grad()
def empirical_kernels(net, X) -> tuple[Tensor, Tensor]     # Phi_l = phi(h^l) phi(h^l)^T / N (L, P, P), and H (P, L, N)
@dataclass
class SolveResult: Phi: Tensor; Phi_hat: Tensor; history: dict[str, list]; samplers: list[SingleSiteHMC | ImportanceSampler]
class OuterSnapshot(NamedTuple): outer: int; history: dict[str, list]; Phi: Tensor; Phi_hat: Tensor
def solve(X, Y, L, lambda_w, lambda_r, phi, g, cfg: MeanFieldConfig, net=None, log=print,
          checkpoint_dir: Path | None = None, config_hash: str = "",
          on_outer: Callable[[OuterSnapshot], Any] | None = None) -> SolveResult
    # on_outer (default None: nothing changes): called after every outer step run in this call with an OuterSnapshot
    # (the step index, the history so far, Phi, Phi_hat), for live plots. The kernels and the history's lists are copies
    # (the history's numpy entries are shared, read-only); it draws nothing from the generator and runs after the
    # history entry and before the checkpoint, so results are bit-for-bit those without it (tests/test_meanfield_callback.py,
    # with the pins of tests/test_meanfield_pins.py). Not called for steps restored from a checkpoint, nor for a loaded
    # final.pt (no steps run). An exception it raises stops the solve.
    # Algorithm 1 exactly as the notebook, settings from cfg.solver; single-site expectations from one sampler per
    # layer, SingleSiteHMC (solver.sampler="hmc", bit-for-bit the pre-split solver) or ImportanceSampler ("importance").
    # A sampler run validates the step before it (backtracking): HMC on divergences / resets / step-size collapse,
    # importance on an ESS drop (ess_frac < min_ess_drop x the ess_frac of the accepted point before the step), both on
    # ||h||^2/P blow-up and MC error > max_mc_err. Under importance sampling every accepted point (the initial point after
    # warm-up, each validated inner step, each outer-gradient run) must have ess_frac >= min_ess_frac, else
    # InadequateSamplerError (a RuntimeError) is raised before any final.pt is written: the ESS is a property of the
    # point, so backtracking cannot cure it, and carrying on would freeze the iteration at its start.
    # Tilt tripwire: when, in some layer, the largest Stein fixed-point residual of Phi_hat over the last
    # solver.tilt_tripwire_steps outer steps is >= solver.tilt_tripwire_resid, solve raises UnconvergedSolveError (a
    # RuntimeError) instead of returning, so no final.pt is written (the checkpoints stay). It catches a lost tilt (Phi_hat
    # near 0 has residual near 1), not non-convergence: an unconverged solve below it passes. The CLI (`solve`, `sweep run`)
    # reports it as a one-line error. history["phihat_change_rel"] is NaN for a step from a zero tilt; the log prints
    # margin=n/a for a bounded phi.
    # With project_tilt, phi needs a tilt_rule (ValueError up front otherwise).
    # If checkpoint_dir is given, return its final.pt if present, else
    # resume from its latest ckpt_*.pt (ValueError if it was written with a net and `net` is None, or vice
    # versa); write ckpt_<outer:05d>.pt every cfg.save_int outer steps and final.pt at the end, which then
    # replaces the checkpoints; every checkpoint stores cfg.model_dump() as "meanfield_config" for provenance
def run_meanfield(cfg: Config, dataset: Dataset, net: MLP, cache_dir: Path, log=print, on_outer=None) -> SolveResult
    # raises SpatialPriorUnsupportedError (a ValueError) when cfg.spatial is set, before touching the cache: the solver knows only the iid prior;
    # `python -m fct solve` and `python -m fct sweep run` (with or without --index) report it as a one-line error
    # constants of the action: prior_scale, alpha, rate_tilt (2 beta rate_lambda) and learning_prefactor follow the network
    # (`network_constants`, beta = meanfield.beta or train.beta for SGLD / 1 for Adam); an explicit meanfield field overrides it.
    # X = dataset.x[:n_points], Y = dataset.y[:n_points]; lambda_w/lambda_r from cfg.train; phi/g/L from cfg.net
    # hash = section_hash("data","net","train","meanfield") (meanfield as hash_payload); returns final.pt if present
def net_kernels(net: MLP, dataset: Dataset, cfg: Config) -> Tensor   # (L, P, P) float64
    # the trained net's empirical_kernels on exactly the solver's inputs (same slicing code as run_meanfield)
    # the net evaluated in its own dtype; for the theory-vs-net kernels figure
def load_meanfield(cache_dir: Path, cfg: Config, device) -> tuple[Tensor, Tensor, dict]   # Phi, Phi_hat, history
    # FileNotFoundError if there is no result, StaleCacheError if the config (or MODEL_VERSION / LOSS_VERSION) changed
```

Step control (`SolverConfig` defaults; each part can be switched back, see the config section):
- Inner step (`inner_step_control="progress"`): a sampler run validates the inner step before it on sampler health
  and, in addition, on progress: a step after which the layer's inner residual ||<psi psi^T> - Phi|| / ||Phi|| exceeds
  the smallest residual accepted so far in that ascent by more than the convergence tolerance of the point before the
  step, max(inner_tol, min(inner_tol_mc x MC error, 0.5)), is reverted (tilt and chain state) and retried with half
  the step size, not below eta_min_rel x eta_phihat (a step at the floor is accepted, so Monte Carlo noise cannot
  stall the ascent; the floor also bounds halvings after a health revert, so a step that still breaks the sampler at
  the floor is retried there until max_backtrack raises; under "health" halving has no floor; the log says when a
  layer sits at it). The reference is the smallest accepted residual, not the last, so a run of small accepted rises
  cannot let the residual creep up; the margin is the tolerance because the residuals are noisy estimates (the
  smallest a lucky low one), and a strict "did not rise" ratchets eta_phihat down to its floor. The Monte Carlo part
  of the tolerance is capped at 0.5 (`meanfield.INNER_TOL_MC_CAP`): a kernel off by half its size is not converged
  however noisy its estimate, and uncapped, a pure-noise layer (MC error 0.98, residual 0.999) was reported converged.
  The step size grows by eta_grow only after an accepted step, up to `eta_max_phihat` (0.25) per layer. Without the
  progress test a step that took the residual from 0.04 to 0.42 was accepted, the chains lagged the tilt and the
  exemplar diverged.
- Outer step (`phi_step_scale="kernel"`): Phi_l -= eta Phi_l dS/dPhi_l Phi_l instead of eta c_l^2 dS/dPhi_l
  (c_l = tr Phi_l / P). For a Gaussian layer the tilt follows the kernel as dPhi_hat = -Phi^-1 dPhi Phi^-1, so this is
  the Newton direction for Phi_hat(Phi) = T at fixed T (eta = 2 a full Newton step; a test checks it), and the two scales
  coincide exactly when Phi_l = c_l I (a test checks that too). The trace scale was far too small along the top
  eigenvectors of a low-rank kernel (the exemplar's tanh kernel: eigenvalues 4.4 and 2.1, the rest below 0.07,
  c^2 = 0.1). `eta_max`, which caps only the outer step size (`eta_max_phihat` caps the inner one), defaults to 0.25: a
  bounded phi responds to the tilt several times more weakly than a Gaussian (0.3 of it at the exemplar's kernel,
  measured), so its Newton step is near eta = 0.6, and at eta = 1 the exemplar's kernel wandered by up to 13 %.
- Last-layer start (`inner_start="target"`): every inner ascent of the last layer restarts at T = Pi(-2c A Y Y^T A),
  its Stein target at the current Phi_L (exact, a function of Phi_L only; Pi the tilt projection, whose change enters
  `tilt_projected_rel`). The inner residual pins Phi_hat_L only along directions the kernel responds to; along the
  others every inner step adds Monte Carlo noise that nothing removes, and continued across outer steps it grew the
  exemplar's ||Phi_hat_L|| from 0.44 to 0.8-1.4 against a fixed point of 0.41 (Frobenius fixed-point residual 1 to 3)
  while Phi itself sat within 2 to 4 % of an independent reference. Restarted at T it cannot pile up. What the iteration
  then is: steps are validated at the next run, so the ascent always ends with at least one step past its last
  measurement, and the outer gradient run sees Phi_hat_L = T + D, D the inner steps from T; on most outer steps the
  residual at T is already below tolerance and D is the single step eta_hat (<psi psi^T>_T - Phi_L) / (2 c^2). The outer
  step (eta / 2) Phi_L D Phi_L is then a damped stochastic Picard step of Phi_L towards <psi psi^T> under T (for a
  Gaussian layer, Phi <- Phi + (eta eta_hat / 4) (<psi psi^T>_T - Phi) up to the c^2 factor). So the recorded
  fixed-point residual ||Phi_hat_L - T|| / ||T|| of the last layer measures D and scales with eta_hat: it is not that
  layer's convergence measure. `history["picard_resid"]` is: ||<psi psi^T>_T - Phi_L|| / ||Phi_L||, the inner
  residual of the ascent's first run (the log prints it as `picard=`, the solver-health figure and the exemplar's
  summary show it). Near convergence it matches the exact value to its Monte Carlo error; after a large move of T it
  reads low for a step or two, because the chains, equilibrated under the previous tilt, lag the restart (the
  weak-response test measures both). Hidden layers continue from their previous tilt (their target is itself a Monte
  Carlo estimate). Not solved by any of this: a bounded phi at L >= 2. The exemplar at L = 2 ends in
  `UnconvergedSolveError` under the legacy and the new control alike; the hidden layer's Stein target
  sym(<u h^T>_{l+1} Phi_l^-1) inverts a nearly rank-two kernel (eigenvalues floored only at inv_floor_rel x lambda_max),
  and its fixed-point residual is about 1200 already at the net's kernels.
- What it costs a Gaussian layer: with eta capped at 0.25 the L = 1 linear test problem approaches its analytic saddle
  more slowly than with the legacy control (distance 0.054 after 10 outer steps against 0.019) and ends as close (0.015
  against 0.013 after 40; `tests/test_meanfield_step_control.py` pins a lag below 0.1 at every step and the end points
  within 0.01). A weak-response tanh layer with a quadrature reference (same file) ends 0.042 to 0.046 from it after 30
  outer steps (five seeds), where the legacy control stays 0.110 to 0.119 away.

History values are stored per outer step; arrays become lists for saving. The
history keys are those of the notebook's `keys` list plus `is_max_weight`, `picard_resid` (a scalar: the last
layer's residual at its Stein target, NaN unless `inner_start="target"`; see Step control), `n_accepted_inner` / `n_rejected_inner` (per
layer, the inner steps of that outer step accepted, and reverted because the residual rose; `n_backtrack_inner` counts
those reverted for sampler health) and `tilt_projected_rel` (per layer, the largest
relative change ||Pi(x) - x|| / ||x|| the tilt projection made to an inner step of that outer step, or to the last layer's
start; 0 without projection)
(`meanfield.HISTORY_KEYS`),
except that `align_net` / `dist_net` are omitted when `solve` is given no net (instead of
lists of `None`). Both samplers fill every key: under importance sampling `hmc_ess_frac` is the Kish ESS fraction,
`hmc_mc_err`, `hmc_stein_mismatch`, `hmc_h2_ratio` keep their meaning, and `hmc_accept`, `hmc_eps`, `hmc_div`,
`hmc_reset`, `hmc_max_abs_dH`, `hmc_rho1` are NaN; under HMC `is_max_weight` (largest importance weight share) is NaN.

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
neuron_factor_information(rates (P,N), factors, groups=None, n_bins=10) -> FactorInformation(member_mi (N,M), group_mi (N,K), group_names, raw_mi (N,M), null_mi (N,M) | None)
preference(group_mi, tol=1e-12) -> (r, dead)
neuron_mir(r, K) -> (N,)   # per-neuron (r - 1/K)/(1 - 1/K); K = len(layer_mir(...).group_names); dead neuron -> 0
mir(r, K) -> float         # = neuron_mir(r, K).mean()
layer_mir(...) -> LayerMIR(mir, r, group_mi, n_dead, group_names)
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
subtracted and is clamped at 0; a group whose raw information, summed over its members, is not
above `NOISE_FACTOR` (2) times its summed null has all its members zeroed, and a neuron with
every member zeroed is dead (r = 1/K). The test is per group, the unit the preference is formed
on: a row total compares a neuron selective for one group with the bias of all K groups (at P = 64
with k = 4 continuous latents and 8 bins that zeroes every neuron), and a per-member test lets a
group of many binary members through whenever one passes by chance (noise neurons scored a
corrected MIR of 0.77 on object-action). Its cost: within a group the comparison is with the sum of
the members' biases, so a neuron selective for one of the 25 object bits survives only if its
excess information exceeds ~112/P nats (0.03 at the sweeps' P = 4096; at P <= 512 every
object-selective neuron is zeroed, so small-P runs on object-action have no usable corrected MIR).
The raw MIR has a cardinality-dependent
floor: at P = 8192, 10 rate bins, noise neurons score ~0.89 with one 25-way plus two binary
factors and ~0.10 with three five-way factors; corrected, the five-way set and the grouped
object-action set are ~0, while binary singletons leave a floor of ~0.1 (one of them passes the
test by chance a few percent of the time). The caches key on the config, not on this rule, so a
cached corrected MIR may come from the earlier row rule. The notebook plots the raw MIR, the
corrected MIR and the `shuffled_mir` baseline.

Group edge cases: a factor in no group is ignored by the group sums and the MIR (its member
information is still reported, corrected against its own null); a factor in two groups counts in both; a
constant factor is accepted (zero information, but it still counts towards K, so datasets with
a constant action bit work). Inputs are validated loudly (NaN rates or factors, mismatched
lengths, bad dtypes, K < 2).

## draw.py

Every `plot_*` takes `ax: Axes | None = None`, creates its own figure only if
none is given, returns the `ax`. `fig_*` functions build multi-panel figures
from `plot_*` (a figure that is one plot, like `fig_mir_vs_ratio`, just wraps its `plot_*`) and return the `Figure`. Nothing calls `plt.show()`.

```python
plot_test_error(test_error, half_width, ax=None)      # notebook cell 13 left (LogNorm, jet)
plot_train_error(y, train_error, ax=None)             # notebook cell 13 right
fig_learning(grid: GridPredictions, y) -> Figure
plot_pca_variance(explained_variance_l, ax=None); plot_pca_similarity(sim_ij, vmax, ax=None)
fig_representation(pca: LayerPCA) -> Figure           # notebook cell 15 layout
plot_neuron_map(map2d, half_width, ax=None); fig_neuron_maps(maps, half_width, n_cols, title) -> Figure   # cell 16
plot_training_history(history, ax=None)               # mse / loss vs epoch, log y
fig_solver_history(history) -> Figure                 # the notebook's plot_history dashboard, plus ESS / weight-share panels;
                                                      # absent, all-None and all-NaN columns (the other sampler's diagnostics) get no panel; panels "backtracking" and
                                                      # "eigenvalues raised to the PD floor" (n_backtrack_inner/outer, n_pd_clamped) joined the dashboard with the live view; layers are labelled from 0 here
plot_example_trial(dataset, ax=None); fig_example(dataset, cfg) -> Figure   # independent of results: targets y scatter + x embedding of a few points
fig_solver_health(history); fig_sampler_health(history)   # the live view's two groups of solver_history panels, layers labelled from 1 like fig_layer_kernels (SOLVER_HEALTH_KEYS: residuals, outer step size,
                                                      # fixed-point residual, last layer at its Stein target (picard_resid), train MSE theory vs net, step sizes,
                                                      # backtracking, rejected inner steps (n_rejected_inner),
                                                      # PD clamping, normalisability margin, alignment; SAMPLER_HEALTH_KEYS: HMC acceptance /
                                                      # eps / divergences, ESS, importance max weight, MC error); panels the sampler does not record are omitted; ValueError if none are
fig_layer_kernels({name: (L,P,P) stack}, title="")    # one row per name (e.g. Φ, Φ̂), one column per layer, each panel its own colour scale
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
plot_neuron_tuning(curves, neuron, mir, latent_names, ax=None); fig_tuning(curves, neuron_indices, mir_per_neuron, latent_names) -> Figure
                                                       # one panel per neuron, one curve per latent (LATENT_COLORS, not the layer colours), common rate axis
plot_mir_matrix(values (n_r,n_w), lambda_r_values, lambda_w_values, title, ax, vmin, vmax)   # phase diagram: origin="lower", log tick labels, NaN grey
fig_mir_matrix(lambda_r_values, lambda_w_values, mir_by_layer (L,n_r,n_w)) -> Figure         # one per layer, shared _mir_colour_range + colourbar
plot_mir_vs_ratio(records: Iterable[MirRecord], ax=None, legend=True); fig_mir_vs_ratio(records) -> Figure   # MIR vs lambda_r/lambda_w, log x, colour by layer
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

A sweep is a grid over any `Config` fields, in one place so SLURM array jobs and local checks share a code path. The shared
settings (sizes, optimiser, monitor, seeds, task parameters) are the base `Config`; `grid` lists the axes that vary. Layout under a
sweep `ROOT`:

```
ROOT/sweep.json                          # SweepConfig, written by `sweep init`
ROOT/<task>/s{index:04d}_<label>/        # grid sweeps: one normal fct cache dir per setting, e.g. linear_teacher/s0001_L1_phitanh_rate_lambda1e-06_weight_lambda0.001
ROOT/<task>/s.../setting.json            # {"index": 7, "grid_hash": "...", "axes": {"net.L": 2, ...}}, written by run_setting beside config.json
ROOT/<task>/L{L}_{phi}_r{rate}/          # sweep.json without `grid` (the modularity grid): the original names, e.g. what_where/L3_relu_r0.001
```

The index makes a directory name unique (`:g` would print two close floats alike); the label (the last part of each axis path
and its value, characters outside letters, digits and `._+-` replaced by `-`) is for people. The index is zero-padded to
`INDEX_WIDTH = 4` digits so names sort in index order; `expand` refuses a grid of more than 10**4 settings. `setting.json` lets a
notebook find a setting from its axis values (`find_setting_dir`) without recomputing the grid.

```python
SWEEP_TASKS = ("linear_teacher", "object_action", "what_where"); LAYERS = (1, 3, 5); PHIS = ("tanh", "relu"); RATE_LAMBDAS = (0.0, 1e-3)
MODULARITY_GRID = {"data.task": SWEEP_TASKS, "net.L": LAYERS, "net.phi": PHIS, "train.rate_lambda": RATE_LAMBDAS}   # used when sweep.json has no grid
@dataclass(frozen=True) class Setting: index: int; axes: dict[axis, value] (grid order); task: str; relpath: Path
    # consumers read axes explicitly, e.g. setting.axes["net.L"]
MODULARITY_AXES = ("net.L", "net.phi", "train.rate_lambda"); SWEEP_OWNED_AXES: dict[axis, the sweep field to change instead]
class SweepConfig(StrictModel)   # N, P, n_epochs, lr, log_int, save_int, monitor_int (>=1), monitor_bins, monitor_shuffles,
                             # linear_teacher: LinearTeacherParams(n_features, n_latents, latent_range), object_action: ObjectActionParams(grid_size, max_objects),
                             # what_where: WhatWhereParams(n_motifs), seeds: Seeds(data, net, train) = 0, dtype = "float32", device = "auto",
                             # task = "linear_teacher" (the base task, used unless data.task is an axis), meanfield: MeanFieldConfig = MeanFieldConfig(), grid: dict[str, list] | None = None
    # Shared network and loss fields (library defaults, so a sweep without them builds the configs it always did): L, g, phi, weight_lambda,
    # rate_lambda, optimizer, beta; an axis of the same name wins. spatial: SpatialConfig | None = None, the spatial weight prior of every setting
    # (written to Config.spatial; training hashes change with it, grid_hash does not). A `spatial.<path>` axis varies a field of the section and
    # is rejected at load ("set the sweep's spatial section first") when the sweep has none; `spatial` itself is a valid axis only while the
    # section is None (its values are whole dumps, or None). A dict- or list-valued axis value (or one whose text exceeds MAX_LABEL_VALUE = 24 characters) is labelled in directory names as `<axis>-<6 hex of the
    # sha256 of its sorted JSON>`, so names stay short; the solve stage raises SpatialPriorUnsupportedError per spatial setting
    # meanfield: the problem fields plus the `solver` group, library defaults (constants None: follow the network). It reaches every setting's
    # Config.meanfield but never enters a training hash or a directory name, so a sweep.json without it (legacy) loads with the defaults and keeps both;
    # its n_points must not exceed P, checked when a setting is solved (SweepError), not at load, so a train-only sweep with a small P still loads;
    # it is read by the solve stage only (a meanfield.* path is also valid as a grid axis, as any Config field)
    # grid: axis = dotted Config path (net.L, net.phi, train.rate_lambda, train.weight_lambda, data.task, ...), values = explicit lists; an axis
    # overrides the base value (net.N over N, train.lr over lr, ...), except the fields the sweep owns, which are rejected: data.P, data.D_in,
    # data.seed, net.seed, train.seed, train.log_int, train.save_int, train.monitor_int, train.monitor_bins, train.monitor_shuffles (change the
    # sweep field instead). At load every axis is checked alone (a Config field, not a section; non-empty; each value valid; no two values
    # build the same Config, so [0, 0.0] is a repeat), then every combination is built into a Config, so a bad axis name or value fails
    # naming the axis. data.task picks that task's parameters and D_in; weight_lambda is the sweep's `weight_lambda` field (the TrainConfig default 2e-4 unless set) unless it is an axis
def preset(name: str, n_lambda: int | None = None) -> SweepConfig   # "tiny": N=8, P=32, 6 epochs, CPU, seconds per setting (pipeline checks only); "full" and "modularity": N=256, P=4096, 15000 epochs (placeholders to tune), the modularity grid;
                                                                    # "phase_transition": the sizes of "full" + phase_transition(n_lambda); n_lambda is required there and rejected elsewhere; ValueError otherwise
def spatial_tiny() -> SweepConfig                    # presets without lambda axes (FIXED_GRID_PRESETS; n_lambda is rejected): grid net.N [4, 8], P 16, linear teacher m 6 k 2, L 1, tanh, SGLD (beta 1e3, weight_lambda 0.2 = 2 beta x 1e-4), 6 epochs, CPU, spatial prior with grid embeddings, length scale 0.15, min_rel_eig 1e-2, meanfield.n_points 16
def spatial_n_scan() -> SweepConfig                   # grid net.N [64, 128, 256, 512, 1024] (directories s000i_N<N>): P 164, m 64, k 2, L 1, tanh, g 1, weight_lambda 60 (under the lambda_w / (2 beta) convention; 3e-3 under the old lambda_w x form), rate_lambda 0, SGLD beta 1e4, lr 1e-2, 250000 epochs, float64, monitor every 5000 (8 bins, 5 shuffles), same spatial prior
def phase_transition(n_lambda) -> SweepConfig             # linear teacher; grid net.L [1,2], net.phi [tanh, relu], train.rate_lambda and train.weight_lambda both np.logspace(-6, 0, n_lambda): 4 * n_lambda**2 settings
def save_sweep(sweep, root) / load_sweep(root) -> SweepConfig   # ROOT/sweep.json; load raises FileNotFoundError naming the path
def expand(sweep) -> list[Setting]                       # itertools.product over the axes in the order written (first axis slowest); index i is stable for a given sweep.json and independent of the sizes; adding a value to any axis but the first renumbers later settings and orphans their directories (grow a grid in a fresh ROOT)
def get_setting(sweep, index) -> Setting                 # SweepError (a ValueError) "index i is out of range; valid indices are 0..N-1"
def setting_dir(root, setting) -> Path                   # ROOT / setting.relpath (see Layout)
def data_config(sweep, task) -> DataConfig               # D_in from data.task_dims (2*G*G, 81, or n_features), never repeated in the sweep
def setting_config(sweep, setting) -> Config             # base config for the setting's task (net seed = seeds.net, monitor_* from the sweep) with each axis value written over it
def setting_status(root, sweep, setting, stage="train") -> "done" | "partial" | "missing" | "stale"   # stage "train": done = train/final.pt exists with the config hash of this setting (data, net, train, loss version; only the hash is read); stage "solve": the same for meanfield/final.pt and the hash of data, net, train, meanfield (what run_meanfield writes); stale = a final.pt with another hash, or an unreadable one; partial = a ckpt_*.pt but no final.pt; missing otherwise (so also a run that has not yet reached its first checkpoint). The stages are independent: a solve is stale when the solver or the training config changed, a trained net never when only the solver did
def grid_hash(sweep) -> str                               # short hash of the built values of every axis (so [0, 1e-3] == [0.0, 1e-3]) plus the base task unless data.task is an axis; stored in setting.json
def read_setting(directory, sweep) -> {"index", "grid_hash", "axes"}   # a setting directory's setting.json; SweepError naming the directory if it was written under another grid or has no grid hash
def built_values(sweep) -> {axis: [values as the built Config holds them]}   # used by grid_hash and the directory labels, so a value has one spelling in both
def find_setting_dir(root, sweep, values) -> Path         # the one run setting of `sweep` whose axes equal `values` (same_value: numbers to 1e-9 relative, also in lists; a bool equals only a bool;
                                                          # a subset of the axes is enough if unambiguous); only the sweep's own directories are considered; LookupError if none, several, or an unknown axis
def has_settings(root) -> bool                            # ROOT already holds a setting directory; `sweep init` refuses to write a new sweep.json there without --force
def run_setting(sweep, index, root, log=print, stage="train") -> None
    # first validates/writes config.json and setting.json (SweepError if an existing one differs in data, net, train or the like: the sweep changed
    # under existing artefacts; this check comes BEFORE the done check, so a changed sweep raises even for a done setting; only the meanfield section
    # may differ, and config.json is then updated to the sweep's, so `python -m fct solve DIR` solves what the sweep does),
    # StaleCacheError if the stage's final.pt is stale, returns at once if done (log "already done (<stage>)"), else get_dataset and
    #   stage "train": train_from_config(cfg, dataset, dir, log, monitor=make_mir_monitor(dataset, cfg)); resumes a partial run exactly (see train.py)
    #   stage "solve": run_meanfield(cfg, dataset, net, dir, log) into DIR/meanfield/, net = the trained net if train/final.pt exists (history gets the
    #     net-comparison keys), else None (keys absent; "no trained net" is logged). Resumes from a solver checkpoint; the net-or-not choice is fixed
    #     once a solve has started: a checkpoint written with the other choice raises SweepError (one line, naming the file; `sweep run --index` prints it
    #     and exits 1), so train before solving, or delete meanfield/. A FINISHED solve that ran without the net is `stale` in the status once
    #     train/final.pt exists (its hash is unchanged; the stored net_Phi is None), and run_setting then deletes that final.pt and solves again
    #     with the net, so a grid solved first and trained later ends up with the net comparison everywhere.
    #     config.json is rewritten (atomically, save_config writes a .tmp sibling and renames it) BEFORE the stale check, so even a stale solve's
    #     config.json already holds the sweep's current meanfield
    #     An InadequateSamplerError (importance sampling below the ESS floor) propagates and leaves no final.pt
```

CLI: `python -m fct sweep init ROOT --preset phase_transition --n-lambda 8 [--axis net.L=[1,2,3]] [--set N=64 --set n_epochs=2000 ...]`.
`--set` overrides a shared entry by dotted key; `--axis PATH=[values]` replaces or adds a grid axis (JSON list; the modularity presets have no editable axes and reject it); `--set grid...` is rejected (use `--axis`). `sweep count|list|status|run --index i` all read the same `expand`, and `jobs/fct_sweep.sh` takes the array size
from `sweep count`. `sweep status` and `sweep run` take `--stage train|solve` (default train): train the nets or solve the
saddle points of the same initialised grid, locally or as an array job (`jobs/fct_solve.sh`); `list` and `count` do not depend on the stage.
Solver settings are `meanfield` in sweep.json, set with `--set meanfield.n_points=50 --set meanfield.solver.n_outer=100 ...` at `sweep init`. `lrtune.py` and `notebooks/build_modularity.py` are written for the modularity grid (they read
`setting.axes["net.L"]`, `["net.phi"]`, `["train.rate_lambda"]`); `report.py` needs those three axes too (see below).

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
`find_result(RESULTS, task, {"net.L": L, "net.phi": phi, "train.rate_lambda": rate})`, so a reordered grid fails loudly instead of mislabelling), and
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
def load_results(root) -> list[SettingResult]   # one entry per setting in grid order (36 for the modularity grid); FileNotFoundError if no sweep.json; ValueError if the grid lacks net.L, net.phi or train.rate_lambda
    # (the report supports the modularity grid; a grid with those axes plus others, like phase_transition, loads, and `setting_label` shows every axis so labels stay distinct)
    # read-only: nothing is generated, cached or written under ROOT (a running job may be writing to it); settings are read on the CPU,
    # validated against the config rebuilt from sweep.json. done: from final.pt (StaleCacheError if written under another config).
    # partial: from the latest checkpoint via train.load_latest_checkpoint -> the same data as a done setting (history exactly as
    # checkpointed, examples and metrics from the checkpointed weights) plus epoch/n_epochs, status stays "partial". The checkpoint
    # being replaced while read is retried once. A checkpoint that cannot be read, is from another config hash, or is inconsistent
    # (corrupt file, StaleCacheError, epoch > n_epochs, weights that do not fit the net) gives status "partial", no data and a `note`; an unreadable
    # dataset file gives a note "dataset not readable". Exceptions that are not about a bad file (a bug in the model code) propagate.
    # A dataset file that is missing (done or partial) also gives "partial" with a note, no data. No checkpoint: "missing".
    # A checkpoint at epoch 0 or without a monitor entry loads (no loss curve / MIR yet; the prose says so).
def find_result(results, task, axes) -> SettingResult    # axes = {dotted path: value} of every axis but the task; LookupError if none or several match
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
# result rows: dict with the keys of COLUMNS (also the CSV schema, docs/archive/lr_tuning_results.csv)
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
python -m fct sweep status ROOT [--stage train|solve]   # "<index> done|partial|missing|stale <dir>" per setting, then "N done, N partial, N missing, N stale"
python -m fct sweep run ROOT [--index i] [--stage train|solve]   # train (default) or solve setting i (a SLURM array task); without --index, all settings in order
```

The sweep subcommands exit 1 with one stderr line for a missing or invalid sweep.json, an out-of-range index, or a setting whose
config.json differs from what sweep.json builds (`SweepError`); an error raised by training itself propagates with its traceback.
`run` without `--index` continues past a failing setting, prints `error: setting i failed: <Type>: <message>` (and the traceback for
anything but a `SweepError`), and exits 1 listing the failed indices. With `--index`, an `InadequateSamplerError` from a solve
(the importance sampler cannot represent the problem; see `meanfield.py`) is one stderr line and exit 1, with no final.pt written; in
a run of all settings it is reported like any other failure and the remaining settings still run.

`--which all` (default) draws each figure whose inputs exist and lists the skipped ones — learning|representation|neurons|training need train/final.pt, solver needs meanfield/final.pt, kernels needs both; a single `--which` with missing inputs exits 1.

`--set` takes dotted keys and JSON-ish values (`--set net.phi=tanh`,
`--set data.cov=[[1,0],[0,1]]`); unknown keys fail loudly via pydantic.

## jobs/fct_sweep.sh

SLURM array over the sweep settings; written for the PI to submit. Tests exercise `--submit` only against a recording fake `sbatch` on PATH; agents never run it.

```
bash jobs/fct_sweep.sh ROOT --submit [--index N|A:B]   # sbatch --array=0-35%4 (or =N, or =A-(B-1)%4), logs in <project>/tmp/logs/
bash jobs/fct_sweep.sh ROOT --dry-run [--index N|A:B]  # print that sbatch command, run nothing
bash jobs/fct_sweep.sh ROOT --list | --status      # `fct sweep list` / `fct sweep status`
```

`ROOT` is the first argument or `$FCT_SWEEP_ROOT`; it must exist and hold a `sweep.json`. Array size comes from `fct sweep count`.
`FCT_MAX_CONCURRENT` (default 4) caps running tasks; `FCT_PYTHON` overrides `<project>/.venv/bin/python`. Each task runs
`python -m fct sweep run ROOT --index $SLURM_ARRAY_TASK_ID --stage train` on one GPU of partition `gpu`, excluding `gpu-380-10,gpu-380-12,gpu-380-14`.
`--index` applies to `--submit`/`--dry-run` only: `N` is one setting, `A:B` the half-open range of settings A .. B-1 (like a Python
slice; `0:10` is 0 to 9), both read as base 10; a range is capped by `FCT_MAX_CONCURRENT` like the full array; an empty range
(A >= B) or a range past the sweep size (B > count) fails. A comma in the project or python path is rejected
because `sbatch --export` is comma-delimited; a relative `FCT_PYTHON` is resolved against the caller's directory.
`tests/test_slurm_script.py` puts a fake `sbatch` on PATH: non-submit actions must never call it; `--submit` is only exercised against it.

`jobs/fct_solve.sh` is the same array job at the solve stage: a short script that sets `FCT_STAGE=solve`, `FCT_LOG_NAME=fct_solve`
and `FCT_ENTRY` (the script `--submit` hands to sbatch) and execs `fct_sweep.sh`, so the two share every option, the
fake-sbatch tests. Its `#SBATCH` header is a copy of `fct_sweep.sh`'s (job name `fct-solve`, one GPU per task), because sbatch reads
the header from the submitted file; a test checks that the two agree apart from `-J`. Each task runs
`python -m fct sweep run ROOT --index $SLURM_ARRAY_TASK_ID --stage solve`; `--status` shows the solve stage; logs are
`tmp/logs/fct_solve_%A_%a.{out,err}`. Resubmitting is safe: a finished solve is skipped, a partial one resumes. Written and dry-run
only, like the train script.

## spatial.py (spatial embeddings and weight kernels)

Every unit of the input, each hidden layer and the output gets a position on the torus `[0, box)^d` (default `d = 1`, a ring).
The weight prior of layer l is `K^l = K^post (x) K^pre` and the penalty is `M = K^{-1}`, built per factor. `Config.spatial`
turns it on in training (see "Spatial weight prior in training" below). The maths (wrapped Gaussian, truncation, floor, orientation of the readout)
is stated once, in the module docstring of `fct/spatial.py`.

```python
# config (strict pydantic, like fct.config)
EmbeddingSpec(kind="grid"|"random", shape=None, seed=0)   # grid shape must multiply to n, None -> n points for d = 1; random takes no shape
KernelSpec(length_scale=0.1, amplitude=1.0)               # length_scale in the same units as box
PopulationSpec(embedding, kernel)
SpatialConfig(d=1, box=1.0, input, hidden, output, min_rel_eig=1e-6)   # hidden 1..L share `hidden`; grid in d > 1 needs a shape

# pure computation (float64, CPU)
make_embedding(spec, n, d, box, index=0) -> Embedding(positions (n, d), box)   # local generator: global RNG untouched
torus_distance(x, y, box) -> (n, m)                       # Euclidean norm of per-axis minimum-image differences (figure only)
periodic_gaussian_kernel(x_post, x_pre, length_scale, box, amplitude) -> (n_post, n_pre)   # wrapped Gaussian, diagonal = amplitude
wrapped_gaussian_profile(delta, length_scale, box, amplitude)   # the kernel along one axis (the analytic curve in the figure)
penalty_matrix(K, min_rel_eig) -> (M, FloorReport(n_floored, fraction_floored, n_negative, cond_before, cond_after))
build_spaces(cfg, m, N, L, n_out) -> [LayerSpace(layer, pre, post)]   # L + 1 layers; M_pre, M_post properties
weight_quadratic_form(W, M_rows, M_cols) -> Tensor        # tr(M_rows W M_cols W^T), any dtype/device; identity factors give W.square().sum() exactly
parameter_factors(spaces) -> {"U" | "J.k" | "V": (M_rows, M_cols)}   # factors of each MLP weight in its STORED orientation (V: (M_pre, M_post))
data_kernel(psi (P, n_pre), K_pre (n_pre, n_pre)) -> (P, P)           # A = psi K_pre psi^T / n_pre: the data factor of the paper's Phi^{mu nu}_ij = K^post_ij A_{mu nu} (eq. phi-diagonalised); K_pre = I gives psi psi^T / n_pre
spatial_kernel_pairs(A (P, P), K_post (n, n), trials) -> (S, S, n, n)   # block [s, t] = A[trials[s], trials[t]] * K_post; ValueError on non-square inputs and on trials empty or outside 0..P-1
```

Populations are indexed 0 (input, m), 1..L (hidden 1..hidden L, N), L + 1 (output, n_out); layer l maps population l to l + 1,
matching `MLP`: U is layer 0, J_1..J_{L-1} the hidden layers, the readout layer L. The post `Population` of layer l is the same
object as the pre `Population` of layer l + 1 (embedding, K, M computed once per population object). A random
embedding draws from a generator seeded `embedding.seed * 1000 + population index` (at most 1000 populations),
so populations differ, runs repeat and no two (seed, index) pairs collide. `MLP.V` is stored (pre, post): for
layer L the weight in (post, pre) orientation is `V.T`.

The floor. `M` is the inverse of `K` with eigenvalues floored at `min_rel_eig` times the largest. A smooth kernel has many
eigenvalues below that: at the defaults (`length_scale = 0.1`, `min_rel_eig = 1e-6`) 83% of a 100-unit ring's eigenvalues are
floored. So the floor is where the prior's near-zero-variance directions are capped, and it defines `M` on that subspace;
`FloorReport` gives the floored fraction and `n_negative` (eigenvalues below `-1e-10` times the largest, which is an error:
`penalty_matrix` raises). Whether to replace the floor by a nugget, `M = (K + sigma^2 I)^{-1}`, is pending a PI decision.

## Spatial weight prior in training

`Config.spatial = SpatialConfig(...)` replaces the weight L2 norm of the training loss by the spatial prior; `None` (the
default, and what every `config.json` written before the field existed loads as) keeps the L2 norm, bit for bit.

- The penalty. `weight_l2 = sum_l tr(M^post_l W_l M^pre_l W_l^T)` over the L + 1 weights with `W_l` in (post, pre)
  orientation (`U`, `J_k`, `V.T`), which is `sum_l w_l^T (M^post (x) M^pre) w_l` with `w_l` the row-major `vec(W_l)`
  (equivalently `M^pre (x) M^post` with the column-major vec); the Kronecker matrix is never formed. The populations are
  those of `build_spaces(cfg.spatial, m=D_in, N, L, n_out=D_out)`: the first layer's pre is the input embedding, the
  readout's post the output embedding. `MLP` has no biases; `make_weight_penalty` raises if the net has a parameter the
  prior does not cover, so a parameter added later cannot silently drop out of the loss.
- Convention: the penalty function carries NO coefficient, so identity factors give exactly the plain sum of squares; the loss
  multiplies it by `prior_coefficient = weight_lambda / (2 beta)` (see "Prior scaling convention"), the same coefficient as for the L2 norm. The history and log key stays `weight_l2` (it holds the
  raw value) so plots and logs keep one schema. Mapping to the Latex `P_weight = 1/2 sum_l W M^l_tex W` under SGLD at
  inverse temperature `beta`: `beta * weight_lambda / (2 beta) * tr(...) = P_weight` gives
  `M^l_tex = weight_lambda (M^post (x) M^pre)`, so the prior covariance is
  `K^l_tex = (K^post (x) K^pre) / weight_lambda`, for every `beta`. For identity kernels this is the L2 case, variance `1 / weight_lambda`.
- The rate penalty is unchanged (`rate_lambda * sum phi^2`, identity `G`).
- The matrices `M` are built once per `train` call (float64 on the CPU, then cast to the net's dtype and device). They are
  not parameters and not checkpointed: a resumed run rebuilds them from the config, so resume is exact.
- Hash: the `spatial` section enters `section_hash` with `train` only when it is set, so the caches of runs without it keep
  their hashes (pinned in `tests/test_spatial_prior.py`), and a run trained without the prior is stale for a config with it.
- SGLD stability. The prior's stiffest curvature is `(weight_lambda / beta) lambda_max(M^post) lambda_max(M^pre)`, which a wide
  kernel makes large (up to `1 / (min_rel_eig * lambda_max(K))` per factor). An Euler step diverges once
  `lr * weight_lambda / (2 beta) * lambda_max(M^post) lambda_max(M^pre) >= 1`, so `make_weight_penalty` raises for SGLD before the
  first step. The check is necessary, not sufficient (the data and rate terms add curvature). Adam rescales its steps and
  is not checked.
- Not covered: the initial weights are still iid `N(0, 1)` rather than drawn from the prior, so the prior term starts
  large for a wide kernel; the mean-field solver still assumes the L2 prior.

## tuning.py, grid_report.py and the grid figures

Sweep-independent helpers for the phase-transition notebook; all pure functions on plain inputs.

- `tuning.tuning_curves(rates (P,N), factors (P,K), n_bins) -> TuningCurves(mean, sem, centres, count)`: mean rate per neuron in the bins of each latent, other latents marginalised. The bins are exactly the MIR's (`modularity.factor_labels` per latent): a floating latent is rank-binned into `n_bins` (equal-count when its values are distinct; ties share a bin), an integer latent is categorical with one bin per level and its `centres` are the level values. Arrays are as wide as the largest latent's bin count; an empty or padded bin is NaN, as is the SEM of a bin with fewer than 2 samples. `factors` is one tensor, so a call is all-categorical (integer dtype) or all-float; integer factors cast to float are rank-binned with interior empty bins, so pass them as integers. `mean`/`sem` are `(N,K,B)`, `centres`/`count` `(K,B)`. The SEM uses sums of squares about each neuron's global mean so a large rate offset does not cancel it to 0.
- `tuning.select_neurons(mir_per_neuron, n, order, seed)`: `order` in `random` (seeded), `mir_ascending`, `mir_descending`; `n` outside `1..N`, unknown order or NaN is an error.
- `modularity.neuron_mir(r, K)`: per-neuron `(r - 1/K)/(1 - 1/K)` from `LayerMIR.r`; `mir` is its mean.
- `grid_report.mir_grid(records: Iterable[MirRecord], lambda_r_values, lambda_w_values, n_layers) -> (n_layers, n_r, n_w)`, `MirRecord(lambda_r, lambda_w, layer, mir)` with 0-based integer layers; NaN where missing (a layer without records is a NaN plane). Duplicate cells, off-grid lambdas (matched to rtol 1e-9), layers outside `[0, n_layers)`, non-finite MIR, near-duplicate grid values and empty input raise.
- `draw` figures: see the draw.py block above. The lambda axes of `fig_mir_matrix` and `fig_mir_vs_ratio` must be positive (log scale).

## grid_view.py and the phase-transition notebook

`fct/grid_view.py` wires a sweep ROOT to the helpers above; `notebooks/build_phase_transition.py` builds `phase_transition.ipynb` (source only; executed copies go to the ignored `tmp/`). It needs a sweep with the four axes `net.L`, `net.phi`, `train.rate_lambda`, `train.weight_lambda` (the `phase_transition` preset); `report.load_results` supplies the results, so a partial setting is read from its latest checkpoint and is *provisional*.

- Pure: `snap_to_grid(value, grid, log=False) -> Snapped(requested, value, exact)` (nearest grid value, the first of two equally near; `log=True` measures distance in log10 and needs everything positive). `axis_values(results)` gives each axis's values (numbers ascending, names in grid order). `choose_setting(results, L, phi, lambda_r, lambda_w) -> Choice(result, snapped)`: L snapped linearly, lambdas on a log scale, `phi` must match exactly (ValueError listing the activations); LookupError if the grid has more axes than these four. `snap_text(choice)` says what was selected and which values moved.
- `grid_selection(results, depth, phi, kind) -> GridSelection(records, lambda_r_values, lambda_w_values, n_layers, provisional, n_settings)`: `MirRecord`s of the settings with that depth and activation, `kind` in `raw` / `corrected` (history keys `monitor_mir` / `monitor_mir_corrected`, last monitor call). The lambda values are the whole grid, so a setting with no MIR yet is a NaN cell in `mir_grid`; a partial setting with a monitor call is a record and listed in `provisional`; a done setting without monitor entries, or a history with the wrong number of layers, raises.
- `layer_rates(net, x, layer)`: phi(H) of one hidden layer (the rates the MIR is computed on, not H). `factor_matrix(factors)`: (P, K) latents, ValueError for a mix of integer and floating ones. `neuron_mir_values(rates, factors, groups, n_bins, kind, n_shuffles, seed)`: per-neuron MIR (`modularity.neuron_mir`), whose mean is the layer MIR the monitor records (with the same bins, shuffles and seed). `tuning_view(net, dataset, layer, n_bins, kind, n_shuffles, seed) -> TuningView(curves, neuron_mir, member_mi, raw_mi, null_mi, latent_names)`.
- I/O: `load_setting_data(root, result) -> SettingData(net, dataset, cfg, epoch)` (`epoch` of the checkpoint actually read, used in the provisional label): final.pt for a done setting, the latest checkpoint for a partial one, ValueError for one with no data; CPU, nothing written.
- Items for a notebook cell (a list of markdown strings and figures for `report.show`): `tuning_items(root, choice, layer, n_neurons, order, seed, kind)` (0-based `layer`; a setting without data gives one message and no figure), `matrix_items(results, depth, phi, kind)` (`draw.fig_mir_matrix`), `ratio_items(results, depth, phi, kind)` (`draw.fig_mir_vs_ratio`; lambda ratio from the grid values as they are). Figures from a provisional setting carry "provisional" in their title; with no MIR in any network yet the matrix and scatter items are one message.

Notebook cells, in order: parameter cell (`ROOT`), setup (loads `RESULTS`), selector (`L`, `PHI`, `LAMBDA_R`, `LAMBDA_W`, `MIR_KIND`), the one-setting plot suite (`report.setting_items`: loss curves, example outputs, MIR over training), tuning (`N_PLOT_NEURONS`, `PLOT_LAYER` counted from 1, `NEURON_ORDER`, `NEURON_SEED`), the matrix and the scatter (`GLOBAL_L`, `GLOBAL_PHI`, by default the chosen setting's). Tuning curves use the rates phi(H), not H.

Limits: the viewer needs positive lambdas. A grid containing lambda = 0 cannot be snapped on a log scale (`snap_to_grid` raises) and the figures use log axes. A request outside the grid snaps to the nearest endpoint, and the only signal is the "snapped to" line of `snap_text`. The global prose counts settings with no MIR yet apart from stale (other config) or unreadable ones, which it names with their lambdas and reason, and lists the provisional cells by their lambdas. `GridSelection.unusable` holds the latter.

## exemplar.py and the exemplar notebook

`notebooks/build_exemplar.py` builds `exemplar.ipynb` (source only; executed copies go to the ignored `tmp/`): one linear-teacher run, trained with SGLD (or Adam), then the mean-field solve of the same problem with live diagnostics. `fct/exemplar.py` holds all the logic; the cells set variables and call it and `fct.draw`.

- `exemplar_config(*, P, m, k, N, L, phi, g, data_seed, net_seed, weight_lambda, rate_lambda, optimizer, beta, lr, n_epochs, train_seed) -> Config`: a thin wrapper over `spatial_view.make_config` (no spatial prior, CPU, float64; log and checkpoint intervals as there) that adds the `meanfield` section: `n_points = P` (all points are solved); every other constant follows the network (`fct.meanfield.network_constants`). `with_solver(cfg, **solver_fields)` sets `SolverConfig` fields (unknown ones raise).
- Why the solver's defaults follow the network: SGLD at `beta` samples `exp(-beta L)`, and the loss prior `weight_lambda / (2 beta) ||theta||^2` makes each (standard-normal-parametrised) weight have variance `1 / weight_lambda` for every `beta` (pinned by a test per `beta`: weights in directions the data does not span have that variance), the readout's Gaussian likelihood gives the ridge `alpha = weight_lambda P d_out / beta`, the rate penalty is the tilt `2 beta rate_lambda`, and, the solver's kernels being unscaled (`Sigma = prior_scale Phi0`), the data-term prefactor is `learning_prefactor = 1 / (2 prior_scale) = weight_lambda / 2`. A test checks `run_meanfield`'s defaults against `network_constants` for several `beta` and for Adam. An exact Gibbs / closed-form check found the fixed point near the net's kernel scale only with this prefactor (at c = 0.5 the single-site precision is non-normalisable near it).
- Cache layout under CACHE_ROOT: `exemplar-<section_hash(data, net, train)>/{data, train, config.json}` and, inside it, `solve-<section_hash(data, net, train, meanfield)>/meanfield/` (`exemplar_dirs`). A solver setting therefore re-solves without retraining. `train_exemplar(cfg, cache_root, log) -> spatial_view.TrainRun` (via `spatial_view.train_in`, which `train_run` shares) and `solve_exemplar(cfg, dataset, net, cache_root, plot_every, show, log, on_failure) -> SolveResult` go through `train_from_config` / `run_meanfield`, so both resume from checkpoints and reload finished work. `on_failure` is called before an exception from the solve propagates (the notebook passes `LiveView.flush`).
- Live view: `live_figures(snapshot)` is `[fig_layer_kernels({Φ, Φ̂}), fig_solver_health, fig_sampler_health]` (layers counted from 1 in all three); `live_plotter(every, show)` is an `on_outer` callback that calls `show(figures)` at outer steps 0, every, 2 every, ...; `solve_exemplar` calls `show` once more with the finished solve's figures (a loaded final.pt runs no steps). Plotting time is charged to the history's `time`. `LiveView(tail)` is the notebook edge: `show` does `clear_output(wait=True)`, displays and closes the figures, then reprints its last `tail` log lines (`log` collects them, `flush` prints them), because clearing empties the whole cell output. IPython is imported lazily; the clear / display / print functions are injectable.
- `kernel_comparison` (solved Φ_l beside the net's, `draw.fig_kernels` with an own colour scale per panel), `empirical_kernel_figure`, `example_outputs_of(net, dataset)`, `solve_summary(history, sampler, Phi=None, net_Phi=None)` (markdown computed from the solve: first and last fixed-point residual of Φ̂ with "fell" / "rose", smallest ESS fraction and where, largest eigenvalue and mean diagonal of the solved and the net's kernels, train MSE theory and net, alignment). `cache_root_from_env` lives in `fct.cache` (`$FCT_CACHE_ROOT`), shared with the spatial notebook.

Notebook cells, in order: intro, task and net (the `parameters` cell: `CACHE_ROOT`, `P`, `M_IN`, `K_LATENTS`, `DATA_SEED`, `N`, `L`, `PHI`, `G`, `NET_SEED`, `PRIOR_COEFF`, `RATE_LAMBDA`), training settings (`OPTIMIZER`, `BETA`, `LR`, `N_EPOCHS`, `TRAIN_SEED`), setup, train, loss / outputs / empirical kernels, solver settings (`SAMPLER`, `INIT`, `N_OUTER_HMC`, `N_OUTER_IMPORTANCE`, `PLOT_EVERY`, `SOLVER_SEED`, `SOLVE_LOG_INT`; HMC: `N_CHAINS`, `WARMUP_TRANS`, `N_TRANS`, `N_TRANS_OUTER`; importance: `N_IMPORTANCE_SAMPLES`; only the chosen sampler's are passed), the live solve and its printed summary, the kernel comparison. The builder reuses `build_spatial.build_main`, `set_parameters` (rewrites one-line `NAME = value` assignments; a name not assigned exactly once is an error) and `execute_notebook`, which is how the tests run it tiny with each sampler.

Default sizes (P = 20, m = 6, k = 2, N = 128, L = 1, tanh, beta = 10, 10000 SGLD epochs; HMC 512 chains, 5 / 10 transitions per run, 80 outer steps; importance sampling 80) run the whole notebook in about 45 s on a CPU. Measured with the step-controlled solver at these defaults (seed 0, `init = "net"`): every solve finishes under the tilt tripwire with the inner loop converged at every outer step. HMC: the Stein fixed-point residual of Φ̂ is 0.33 at step 0 and 0.91 at step 1 (the inner loop takes 13 steps to fit the tilt to the net's kernel), then 0.05 to 0.31, and 0.06 to 0.20 over the last 30 steps; the largest over the last three is 0.196, its Monte Carlo floor at 512 chains (3 inner steps rejected for progress in 80 outer steps, no sampler-health backtracks, smallest ESS fraction 0.11; the last layer's Picard residual `picard_resid` 0.03 to 0.07 over the last three). Importance sampling: 0.61 at step 0, at most 0.77, then mostly 0.04 to 0.1 (median 0.07 over the last 30 steps, 3 of them above 0.2, up to 0.42; last three at most 0.092), smallest ESS fraction 0.73. From `init = "lazy"` (HMC): transients up to 1.2 in the first 20 steps, then 0.09 to 0.80 (median 0.20 over the last 30 steps; last three at most 0.392, 12 inner steps rejected). The reference `notebooks/exemplar_reference.py` (a damped Picard iteration Phi <- <psi psi^T> under the Stein target of Phi, 8192 chains, run as `python notebooks/exemplar_reference.py CACHE_ROOT`) puts the fixed point 6.4 % from the net's kernel; the solves end 2.1 % (HMC), 2.7 % (HMC from `lazy`) and 1.6 % (importance) from it, so the agreement is not inherited from the start. Top kernel eigenvalue 4.43 / 4.43 / 4.42 against 4.42 (reference), mean diagonal 0.318 to 0.319 against 0.317. The theory's train MSE (0.108) is that of the mean predictor; one sampled net's is 0.110. Importance sampling draws from the prior, a small-P, mild-tilt tool; here, and at P = 24 (smallest ESS fraction 0.91), it stays far above its floor for 80 outer steps. Under `STEP_CONTROL_LEGACY` with eta_max = 1 the default HMC solve diverges (measured: residual 0.33, 0.49, then 32 at step 2 and 20 to 140 after) and ends in `UnconvergedSolveError` (`docs/issues/solver-inner-step-control.md`).

## spatial_view.py and the spatial-prior notebook

`notebooks/build_spatial.py` builds `spatial.ipynb` (source only; an executed copy goes to the ignored `tmp/`): one linear-teacher run under the spatial weight prior, trained through the normal cache, with the tuning of every neuron laid out by position. The library computes and the notebook draws: `fct/spatial_view.py` has the config, the run and the layout helpers (no plotting), `fct.spatial` the kernels and penalty matrices, `fct.grid_view.tuning_view` the tuning curves and per-neuron MIR, `fct.analyse` the example outputs. Every figure is drawn by a plotting function written out in full (plain matplotlib) in the code cell that shows it, and each such cell is preceded by a markdown cell defining the plotted quantity with an equation; the notebook does not import `fct.draw`. So the figure code exists once, in the notebook, where the reader sees it; its tests (`tests/test_build_spatial.py`) run those functions on synthetic inputs by executing the imports and function definitions of the generated cells. A new figure is one more markdown + code pair in `build_notebook`: settings, then the plotting function, then the call and `plt.show()`.

- Config and run: `population_spec(kind, length_scale, amplitude=1, shape=None, seed=0) -> PopulationSpec`; `make_config(*, P, m, k, N, L, phi, g, data_seed, net_seed, weight_lambda, rate_lambda, optimizer, beta, lr, n_epochs, train_seed, spatial, device) -> Config` (task `linear_teacher`, `D_in = m`, `n_latents = k`; `log_int = n_epochs // 50`, `save_int = n_epochs // 4`); `layer_spaces(cfg)` (the L + 1 `LayerSpace`s, `n_out = D_in`); `cache_dir_for(cfg, root) = root / "spatial_<section_hash(data, net, train)>"` (the hash includes the spatial section, so any change of setting is a new directory); `train_run(cfg, root, log) -> TrainRun(cache_dir, dataset, net, history)` writes `config.json`, loads or generates the dataset and calls `train_from_config`, so a finished run is loaded and an interrupted one resumes; `cache_root_from_env()` reads `$FCT_CACHE_ROOT`.
- Layout (pure): `position_order(positions)` argsort around a ring (d = 1 only); `order_weights(W, pos_post, pos_pre) -> OrderedWeights(W, post_order, pre_order)` (rows by post position, columns by pre position; W is (post, pre), so `MLP.U` is (hidden, input)); `panel_grid(positions, n_cols) -> PanelGrid(n_rows, n_cols, cells)` with `cells[u] = (row, col)`: ring order along wrapped rows of `n_cols` for d = 1, the unit's lattice cell for d = 2 (column from x, row from y with y up; positions that do not form a full lattice, such as a random 2-D embedding, raise), d > 2 raises. `empirical_kernels(W) = (W W^T / n_pre, W^T W / n_post)`; `ordered_kernel_comparison(space, W) -> KernelComparison` has the prior and empirical kernels of one layer indexed in the same ring order.
- Figures, each with its plotting function defined in its cell: `plot_layer_prior(space)` per weight layer (K_pre, K_post, every K_ij against torus distance with the profile `wrapped_gaussian_profile`, unit positions; d <= 2), `plot_loss_curves(history, title)` (loss and raw terms, log y), `plot_example_outputs(outputs, title)` (true and predicted output vectors and the pooled R^2 scatter), `plot_kernel_comparison(comparison, space)` (prior K_post / K_pre beside U U^T / m and U^T U / N, ring-sorted; d = 1), `plot_weights_by_position(W, space)` (U sorted by position; d = 1), `plot_tuning_by_position(view, positions, n_cols, bar_width)` (`view` a `TuningView`; bars drawn by the unlisted helper `_draw_information_bars`) (a curve panel and an information bar panel per neuron, placed by `panel_grid`, one curve and one bar per latent, from `grid_view.tuning_view`, so tuning and MIR are computed as in the phase-transition notebook). The kernels shown are weight-prior and weight second-moment kernels over units, not the mean-field Phi over samples of `meanfield.py` (the spatially indexed Phi is the next bullet); the markdown says so.
- Spatially indexed kernels (the paper's eq. kernel-def under the factorised prior): hidden layer l (0-based) is the pre population of `SPACES[l + 1]`, so K^pre = `SPACES[l + 1].pre.K` and K^post = `SPACES[l + 1].post.K` (the next hidden layer, or the output for the last one); psi is `meanfield.readout_activities` (phi(h) on every layer, the paper's eq. kernel-def), computed on the training device and moved to the CPU in float64 where the K live. Helpers defined in the cells and tested: `data_kernels(H, phi, spaces) -> (L, P, P)` and `layer_kernel_pairs(A, spaces, layer, trials) -> (post population, (S, S, n, n) blocks)`. Figures: `plot_data_kernels(A, order, sorted_by)` (one panel per hidden layer, A^(l) (P, P) sorted by `sample_order` on the latent `PHI_SORT_LATENT`, own symmetric scale, title with psi and tr A / P), `plot_trial_latents(factors, trials)` (bars of z_{f mu} of the `PHI_TRIALS` samples `spread_trials(order, S)` picks at evenly spaced ranks of the sorted order), and per hidden layer `plot_spatial_kernel_pairs(pairs, trials, unit_order, title)` (the S x S grid of (n_post, n_post) blocks Phi^{mu nu}_ij, units in `position_order` for d = 1 and index order otherwise, one symmetric colour scale for the grid, an all-zero grid shown on -1..1). The activities psi are those of the held-out test set (`fct.data.get_latent_lattice_dataset`, the 3^k lattice points, so A^(l) is (3^k, 3^k)): the cell's `TEST_DATASET` is made in the train cell, and `sample_order`, the trial picks and the pair grids all read it; the example outputs are drawn from it too: `plot_example_outputs(outputs, title, everything=None)` shows `N_EXAMPLES` test points as panels and, in the scatter, every entry of all 3^k test points with their pooled R^2 (`everything`; the shown examples only if omitted). `plot_data_kernels(A, order, sorted_by, B=None)` draws `fct.spatial.latent_kernel(factors)` = B = Z Z^T / k (P, P) first when given, in the same sample order, so each A^(l) is read against the kernel of a layer linear in the latents. The input kernel Phi^(0) is not drawn.
- Notebook cells: task and network (the `parameters`-tagged cell: `CACHE_ROOT`, `P`, `M_IN`, `K_LATENTS`, `N`, `L`, `PHI`, `G`, seeds, `PRIOR_COEFF`, `RATE_LAMBDA`, `DEVICE`), training (`OPTIMIZER`, `BETA`, `LR`, `N_EPOCHS`, `TRAIN_SEED`), setup (imports), spatial (`D`, `BOX`, `MIN_REL_EIG`, `INPUT`, `HIDDEN`, `OUTPUT`), the eigenvalue-floor and step-size tables, the prior kernels, train, loss curves, example outputs (`N_EXAMPLES`), first-layer kernels, first-layer weights, the data kernels A (`PHI_SORT_LATENT`), the chosen trials (`PHI_TRIALS`), the Phi grids, tuning (`N_BINS`, `N_SHUFFLES`, `MIR_KIND`, `N_COLS`). `build_spatial.set_parameters(nb, values)` rewrites the one-line `NAME = value` assignments (a name not assigned exactly once raises); `execute_notebook(nb, cache_root, overrides)` runs a copy with them set, and the CLI takes them as `--set NAME=literal`.
- Default sizes: 16 hidden units on a ring, 12 inputs, kernel length scale 0.15, `MIN_REL_EIG = 1e-2`, `LR = 1e-2`, `PRIOR_COEFF = 3e-3`, the coefficient of the penalty in the loss; the config cell derives `weight_lambda = 2 * PRIOR_COEFF * beta (beta = 1 for Adam)`, so both optimisers train the same loss). SGLD is rejected unless `lr * weight_lambda / (2 beta) * lambda_max(M_l) < 1` for every layer, with `lambda_max(M_l) = lambda_max(M_post) lambda_max(M_pre)` and `lambda_max(M^p) = 1 / (min_rel_eig * lambda_max(K^p))` (here about 368, so the product is about 0.011), and the library default floor of 1e-6 starts even Adam with a prior term of order 1e9; both optimisers run 2000 epochs in a few seconds on a CPU. SGLD fits the teacher less closely than Adam in the same number of epochs (executed: final train mse 0.198 under SGLD, 0.124 under Adam).
- Limits: kernels and tuning panels cover d <= 2; the weight image and the ring-sorted empirical kernels need d = 1. A hidden-layer panel per neuron gets large for N in the hundreds (the figure height grows with the rows).

## spatial_grid.py and the spatial-grid notebook

`notebooks/build_spatial_grid.py` builds `spatial_grid.ipynb` (source only; an executed copy goes to the ignored `tmp/`): every figure of the spatial notebook for ONE setting of an already-run spatial sweep, instead of a net the notebook trains itself. Its cells are `build_spatial.py`'s, imported as module constants so the two notebooks cannot drift; only the parameters, the setup, the grid listing and the loader are its own.

- Parameters (the `parameters`-tagged cell): `ROOT` (the sweep ROOT, None reads `$FCT_SWEEP_ROOT`), `SETTING` (axis values of the setting, e.g. `{"net.N": 256}`, a subset of the grid's axes that picks one setting), `DEVICE`. The other `--set` parameters are the ones the shared cells assign (`PHI_SORT_LATENT`, `PHI_TRIALS`, `N_EXAMPLES`, `N_BINS`, `N_SHUFFLES`, `MIR_KIND`, `N_COLS`). The sweep must have a spatial section on a linear teacher (`layer_spaces` raises otherwise).
- `fct.spatial_grid` (I/O + wiring, no plotting): `select_setting(root, values, device="cpu") -> GridSetting(setting, directory, cfg, status)` finds the directory through `fct.sweep.find_setting_dir` (values compared as `same_value`, so `8.0` finds `8`); the match is computed over the whole grid, so values or axes outside it, or matching several settings (run or not), raise `LookupError`, and a setting of the grid that was never run comes back with status `"missing"`; a run setting's directory is confirmed by `find_setting_dir`. `settings_table` shares its (index, setting, status) rows with `sweep status` (`fct.sweep.setting_statuses`). The device override is validated by `Config`. `settings_table(root)` is the markdown grid (index, every axis, `setting_status`); `load_grid_run(chosen) -> (TrainRun, test Dataset)` loads the trained net, history and training set and builds the 3^k test set, and raises `ValueError` unless the status is `"done"`; the loader sets `PRIOR_COEFF = fct.train.prior_coefficient(cfg.train)` (and `OPTIMIZER`, `BETA`, `LR`, `TRAIN_SEED`) from the setting's config for the shared cells.
- What is drawn from what: the Phi cells (data kernels A^(l) and the latent kernel B, the trial latents, the pair grids) and the example outputs (with R^2 over all 3^k test points) from the test set; the loss curves from the training history; weights, first-layer kernels, tuning and overlaps from the net and the training samples, as in the spatial notebook. The prior kernels and the floor tables need only the config, so they are drawn for any setting of the grid.
- A setting that is not `"done"` (missing, partial or stale) is reported in the loader cell (`NOT_TRAINED`), and every cell that needs the run keeps its plotting function at the top level and runs the rest under `if TRAINED:` (else it repeats the note), via `build_spatial_grid.when_trained`. So a typo fails and an untrained setting reports.
- The net is loaded on `DEVICE` (default CPU); cells bring device tensors to the host through `.detach().cpu()`.
