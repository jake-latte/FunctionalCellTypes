---
type: infrastructure
complexity: simple
status: in review
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Make the ensemble size B a shared experiment field

PI request (2026-11-02): the phase_transition preset should set B as an ordinary (shared) parameter instead of a one-value `net.B` axis. `ExpConfig` (`fct/exp.py`) has no B field: its shared net fields are `L`, `g`, `phi` (each defaulting to `NetConfig()`'s value) and `_base_config` builds `NetConfig(N=exp.N, L=exp.L, g=exp.g, phi=exp.phi, seed=exp.seeds.net)`. That is why the preset had to use a `net.B` axis. This item adds the field to the library only. Do NOT edit `fct/presets/phase_transition.json`: the PI has uncommitted edits to it, and the PM will change it in their working copy after this merges.

## Inputs / outputs / interfaces
- `fct/exp.py`: add `B: PosInt = NetConfig().B` to `ExpConfig` beside `L`, `g`, `phi`, and pass `B=exp.B` in `_base_config`. A `net.B` axis must still win over the shared field, as `net.L` does over `L`; check how that override works for L and make sure B follows the same path.
- Update the `ExpConfig` docstring, and any docs listing the shared fields (grep `docs/`, `README*`, `fct/exp.py` comments, and the CLI help in `fct/cli.py`), to include B.
- Check whether any preset or template (`fct/presets/*.json`, any `template.json` writer) lists every shared field explicitly and must gain `"B": 1`; if one does, add it there, but not to phase_transition.json.

## Acceptance criteria
- An `ExpConfig` with `"B": 8` and no axes builds settings whose `Config.net.B == 8`.
- Without `B`, every setting has `net.B == 1`. An exp.json written before this change, i.e. JSON without a `B` key, still loads, and `same_experiment` says it equals the same experiment written now.
- With `"B": 8` and a `net.B` axis of [2, 4], the axis wins in each setting.
- Every setting's Config hash, and so its directory and cache, is the same for shared `B = 8` as for a one-value `net.B: [8]` axis with the same other fields. If the setting's relative path differs (axis values may enter directory names), say so in your report rather than changing it.
- `B: 0` is rejected.
- The full test suite passes.

## Results
- Shared `B = 8` and a one-value `net.B: [8]` axis (same other fields) give identical `Config`s, train hashes and solve hashes, so the cache is shared.
- The relative path differs: the axis spelling puts `_B8` in the directory name and in `exp_hash`, so the shared spelling's directories lack that suffix and `same_experiment` is False. Not changed, as instructed; `tests/test_exp_hashes.py` pins this.
- exp.json omits `B` at the library default, so every exp.json written before `B` existed, and the exp.json text pins in `tests/test_phase_transition_preset.py`, stay byte-identical.
