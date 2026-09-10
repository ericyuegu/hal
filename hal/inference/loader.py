"""Static dispatch for versioned HAL policy bundles."""

from pathlib import Path

from hal.inference.api import Policy
from hal.inference.bundle import read_policy_manifest
from hal.inference.o50_model import O50_BACKEND
from hal.inference.o50_model import O50_BACKEND_VERSION


def load_policy(
    path: str | Path,
    *,
    device: str = "cuda",
    seed: int | None = None,
    compiled: bool = False,
) -> Policy:
    """Load a known backend without importing an experiment module."""
    manifest = read_policy_manifest(path)
    identity = (manifest.backend, manifest.backend_version)
    if identity == (O50_BACKEND, O50_BACKEND_VERSION):
        from hal.inference.o50 import load_o50_policy

        return load_o50_policy(path, device=device, seed=seed, compiled=compiled)
    raise ValueError(f"unsupported policy backend {manifest.backend!r} version {manifest.backend_version}")
