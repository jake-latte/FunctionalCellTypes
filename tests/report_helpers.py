"""Hand-built results shared by the report tests: final MIR values that are unique per
(setting, layer, kind) and exact in binary, so tables can be compared with ==."""

from pathlib import Path

from fct.report import SettingResult
from fct.sweep import Setting, expand, preset

QUIET = {"log": lambda s: None}  # keyword arguments for run_setting: no progress output


def modularity_setting(task: str, L: int, phi: str, rate_lambda: float) -> Setting:
    """The setting of the fixed modularity grid with these swept values."""
    wanted = (task, L, phi, rate_lambda)
    return next(
        s
        for s in expand(preset("tiny"))
        if (s.task, s.axes["net.L"], s.axes["net.phi"], s.axes["train.rate_lambda"]) == wanted
    )


def known(setting: Setting, layer: int, kind: str) -> float:
    """Final MIR of `layer` (0-based) of `setting`: unique per (setting, layer, kind) and exact
    in binary, so the table can be compared with ==."""
    index = expand(preset("tiny")).index(setting)
    offset = {"raw": 0.5, "corrected": 0.25, "chance": 0.125}[kind]
    return offset + index / 128 + layer / 1024


def synthetic_history(setting: Setting) -> dict:
    """A history whose FIRST monitor call has different values, so only the last is final."""

    def column(kind: str) -> list[list[float]]:
        first = [-1.0] * setting.axes["net.L"]
        last = [known(setting, layer, kind) for layer in range(setting.axes["net.L"])]
        return [first, last]

    return {
        "monitor_epoch": [0, 6],
        "monitor_mir": column("raw"),
        "monitor_mir_corrected": column("corrected"),
        "monitor_mir_chance": column("chance"),
        "monitor_n_dead": [[0] * setting.axes["net.L"], [1] * setting.axes["net.L"]],
        "monitor_n_dead_corrected": [[0] * setting.axes["net.L"], [2] * setting.axes["net.L"]],
    }


def synthetic_results(missing=()) -> list[SettingResult]:
    out = []
    for s in expand(preset("tiny")):
        if s in missing:
            out.append(SettingResult(s, "missing", Path("nowhere")))
        else:
            out.append(SettingResult(s, "done", Path("here"), history=synthetic_history(s)))
    return out
