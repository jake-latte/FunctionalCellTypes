---
type: infrastructure
complexity: simple
status: done
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Scaffold the `fct` package: environment, config schema, cache helpers

Create the package skeleton that every other work item builds on: `uv` project
with pinned dependencies, the pydantic hyperparameter schema that `init` writes
to `CACHE_DIR/config.json`, the activation registry, and the cache-directory
I/O helpers (artefact save/load with staleness check, checkpoint rotation).
`docs/design.md` (provided with this issue) is the interface contract: copy it
into the repo unchanged.

Why: the library is assembled from parallel work items; the schema and the
cache helpers are the shared seams, so they must exist and be tested first.

## Inputs / outputs / interfaces

- `pyproject.toml` (+ `uv.lock`): project `fct`, `requires-python >=3.12`,
  deps `torch>=2.11`, `numpy`, `scikit-learn`, `matplotlib`, `pydantic>=2`;
  dev dep `pytest`; `[tool.pytest.ini_options] testpaths=["tests"]`,
  `pythonpath=["."]`; `[tool.ruff] line-length = 100`.
- `.gitignore`: add `cache/`, `.cache/`, `.venv/`, `.uv-cache/`,
  `__pycache__/`, `*.pt`, `.claude/worktrees/`, `.pytest_cache/`, `.pm/`.
- `fct/__init__.py`, `fct/config.py`, `fct/activations.py`, `fct/cache.py`
  exactly as specified in `docs/design.md` sections *config.py*,
  *activations.py*, *cache.py*.
- `tests/test_config.py`, `tests/test_activations.py`, `tests/test_cache.py`.
- Update `README.md`: keep the LaTeX section, add a "Python library" section
  with the install (`uv sync`) and test (`uv run pytest`) commands and the
  intended CLI workflow (init -> train / solve -> plot / plot-example).

## Acceptance criteria

- `uv sync` creates a working env; `uv run pytest` passes.
- `default_config()` round-trips through `save_config` / `load_config`
  unchanged; `load_config` raises on an unknown key (`extra="forbid"`) and on a
  wrong type.
- `section_hash("data")` changes when a data field changes and does not change
  when an unrelated section (e.g. `analysis.n_grid`) changes; it is stable
  across key order.
- `get_phi("relu")` is `torch.relu`; unknown name raises `KeyError` listing the
  valid names; `phi_prime(get_phi(n))` matches autograd on random inputs for
  all three names (within 1e-12 in float64).
- `save_artefact` / `load_artefact` round-trip a dict of tensors and scalars
  with `weights_only=True`; `load_artefact` with a different expected hash
  raises `StaleCacheError` whose message contains the file path.
- `write_checkpoint` leaves exactly one `ckpt_*.pt` in the directory and
  `latest_checkpoint` returns it; with no checkpoints it returns `None`.
- `resolve_device(Config(device="cpu"))` is CPU; `"auto"` picks CUDA iff
  available.

## Notes for the implementation
- Pin `torch` to the PyPI wheel (`torch>=2.11`); do not add index-url tricks. If `uv lock` cannot resolve torch, report the exact error and fall back to the loosest pin that resolves.
- `phi_prime` is the notebook's `_phi_prime_factory`: analytic derivative for `torch.relu` / `torch.nn.functional.relu`, `torch.tanh`, `torch.sigmoid`; autograd fallback for anything else. For relu use `(h > 0).to(h.dtype)`.
- `section_hash` hashes `json.dumps(cfg.model_dump(include=set(sections)), sort_keys=True)` (sha256, first 12 hex chars) so key order never matters.
- `load_artefact` must use `torch.load(path, map_location=device, weights_only=True)` and compare `payload["config_hash"]` to `expected_hash` before returning; the error message includes the path and both hashes and tells the user to use a fresh cache directory or delete the file.
- `write_checkpoint` writes the new file first and only then removes older `ckpt_*.pt` files, so an interruption never leaves zero checkpoints.
- Keep functions small and pure where possible; tests use `tmp_path` fixtures, CPU only, no real data.
