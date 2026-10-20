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
    directory = Path(os.environ["TMPDIR"]) / "o65-control"
    directory.mkdir(parents=True, exist_ok=True)
    override = os.environ.get("HAL_O65_CAPTURE_SOURCE")
    if override is None:
        source = directory / f"trainer-{CONTROL_COMMIT}.py"
        payload = subprocess.check_output(
            ["git", "show", f"{CONTROL_COMMIT}:experiments/065_selfplay_ppo.py"], cwd=ROOT
        )
        if not source.exists() or source.read_bytes() != payload:
            source.write_bytes(payload)
    else:
        source = Path(override)
    for name in ("hal.inference.action_rng", "hal.eval.periodic_self_play"):
        if name in sys.modules:
            continue
        dependency = directory / f"{name.rsplit('.', 1)[-1]}-{CONTROL_COMMIT}.py"
        dependency.write_bytes(
            subprocess.check_output(["git", "show", f"{CONTROL_COMMIT}:{name.replace('.', '/')}.py"], cwd=ROOT)
        )
        dependency_spec = importlib.util.spec_from_file_location(name, dependency)
        if dependency_spec is None or dependency_spec.loader is None:
            raise ValueError("control dependency is not a Python module")
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
