"""Host staging must preserve every frame and keep CUDA off simulator threads."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_simulator_floor_has_no_torch(tmp_path: Path) -> None:
    source = Path(os.environ.get("HAL_MSL_SOURCE", "~/src/melee-sim-light")).expanduser()
    assert (source / "data/raw/manifest.json").is_file(), "required MSL data fixture missing"
    output = tmp_path / "floor.json"
    env = dict(
        os.environ, PYTHONPATH=f".:{source}", MSL_CORE_LIBRARY=str(source / "build/melee_core/python/libmelee_core.so")
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.benchmark_o65_sim",
            "--data-dir",
            str(source / "data"),
            "--output",
            str(output),
        ],
        env=env,
        check=True,
    )
    result = json.loads(output.read_text())
    assert result["matches"] == 64 and result["frames"] == 256
    assert 0 < result["simulator_seconds"] <= result["wall_seconds"]
    assert result["settings"][0]["max_frame"] == 28800
