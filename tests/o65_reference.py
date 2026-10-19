"""Load the recorded control from Git without keeping a second trainer in source."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
CONTROL_COMMIT = "aacdfee0"


def load_control() -> ModuleType:
    override = os.environ.get("HAL_O65_CAPTURE_SOURCE")
    if override is None:
        directory = Path(os.environ["TMPDIR"]) / "o65-control"
        directory.mkdir(parents=True, exist_ok=True)
        source = directory / f"trainer-{CONTROL_COMMIT}.py"
        payload = subprocess.check_output(
            ["git", "show", f"{CONTROL_COMMIT}:experiments/065_selfplay_ppo.py"], cwd=ROOT
        )
        if not source.exists() or source.read_bytes() != payload:
            source.write_bytes(payload)
    else:
        source = Path(override)
    # The frozen trainer imports its original sampler only on CUDA. Load that
    # recorded dependency too; the maintained trainer has no import of it.
    name = "hal.inference.action_rng"
    if name not in sys.modules:
        dependency = source.parent / f"action-rng-{CONTROL_COMMIT}.py"
        dependency.write_bytes(
            subprocess.check_output(["git", "show", f"{CONTROL_COMMIT}:hal/inference/action_rng.py"], cwd=ROOT)
        )
        dependency_spec = importlib.util.spec_from_file_location(name, dependency)
        if dependency_spec is None or dependency_spec.loader is None:
            raise ValueError("control sampler is not a Python module")
        sampler = importlib.util.module_from_spec(dependency_spec)
        sys.modules[name] = sampler
        dependency_spec.loader.exec_module(sampler)
    spec = importlib.util.spec_from_file_location("hal_o65_frozen_reference", source)
    if spec is None or spec.loader is None:
        raise ValueError("control trainer is not a Python module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
