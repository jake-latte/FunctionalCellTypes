"""The preset files ship in the wheel and load from inside it (the zip), not only from a checkout.

The wheel is built with `uv build` into a temporary folder. The test is skipped only when the
build cannot run for want of `uv` or of a network (an offline resolution failure); any other
build failure is a failure of the test.
"""

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
# What uv prints when it cannot reach an index or resolve the build requirements offline.
OFFLINE_MARKERS = ("no solution found", "failed to fetch", "offline")

# Run in a child with `-I` (no current directory, no PYTHONPATH, no user site) so only the
# wheel put first on the path can supply `fct`.
CHILD = """
import sys
sys.path.insert(0, sys.argv[1])
import fct
from fct.exp import preset, preset_names
assert fct.__file__.startswith(sys.argv[1]), fct.__file__
print(",".join(preset_names()))
exp = preset("tiny")
print(exp.N, bool(exp.description))
"""


@pytest.fixture(scope="module")
def wheel(tmp_path_factory) -> Path:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not installed")
    out = tmp_path_factory.mktemp("dist")
    result = subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(out), str(REPO)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        if any(marker in result.stderr.lower() for marker in OFFLINE_MARKERS):
            pytest.skip(f"cannot build the wheel offline: {result.stderr.strip()[-300:]}")
        pytest.fail(f"uv build failed ({result.returncode}):\n{result.stderr}")
    (built,) = out.glob("fct-*.whl")
    return built


def test_the_wheel_contains_every_preset_file(wheel):
    with zipfile.ZipFile(wheel) as archive:
        names = {n for n in archive.namelist() if n.startswith("fct/presets/")}
    packaged = {f"fct/presets/{p.name}" for p in (REPO / "fct" / "presets").glob("*.json")}
    assert packaged and names == packaged


def test_the_presets_load_from_the_wheel(wheel):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, "-I", "-c", CHILD, str(wheel)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    names, tiny = result.stdout.strip().splitlines()
    assert names.split(",") == sorted(p.stem for p in (REPO / "fct" / "presets").glob("*.json"))
    assert tiny == "8 True"
