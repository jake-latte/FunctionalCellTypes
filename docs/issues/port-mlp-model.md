---
type: infrastructure
complexity: simple
status: in review
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Port the MLP architecture to `fct/model.py` with a seeded constructor

Move the `MLP` class from notebook cell 3 into `fct/model.py` unchanged in
its maths, and add `make_net` that builds it from `NetConfig` with the
activation resolved by name and the parameter initialisation seeded.

Why: the trained model must be reconstructible from `config.json` plus a
`state_dict`, so checkpoints never pickle the module or its activation
function (which `torch.load(weights_only=True)` cannot read).

## Inputs / outputs / interfaces

Implement exactly the *model.py* section of `docs/design.md`:
`MLP(D_in, D_out, N, L, g, phi, device, dtype)` with `forward(x) -> (H, Z)`
and `weight_l2()`, and `make_net(cfg: NetConfig, D_in, D_out, device, dtype)`.
Seed via `torch.random.fork_rng` + `torch.manual_seed(cfg.seed)` so the global
RNG is untouched. Uses `fct.activations.get_phi`.

## Acceptance criteria

- `forward` shapes: `H (P, L, N)`, `Z (P, D_out)`; `L=1` gives `H (P,1,N)`
  with no `J` parameters; wrong `D_in` raises.
- `H[:, 0]` equals `x @ U.T / sqrt(D_in)` and `Z` equals
  `H[:, -1] @ V / N` (readout from the last *pre*-activation, as the
  notebook).
- `weight_l2()` equals the sum of squares of all parameters.
- `make_net` with the same seed gives identical parameters; a different seed
  gives different ones; the global RNG state is unchanged after the call.
- `make_net(...).state_dict()` round-trips through `torch.save` /
  `torch.load(weights_only=True)` into a fresh `make_net` and gives identical
  outputs.

## Implementation notes
- `fork_rng(devices=...)`: pass the CUDA device index list only when the device is CUDA, otherwise `devices=[]`, so the CPU-only test path does not touch CUDA.
- Tests: CPU, float64, tiny sizes (P=5, D_in=3, N=4, L=1 and L=3), `tmp_path` for the save/load round trip.
