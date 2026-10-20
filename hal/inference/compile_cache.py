"""Persist compiler artifacts behind a checked source and hardware identity."""

import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import torch


@dataclasses.dataclass(frozen=True, slots=True)
class CacheIdentity:
    torch: str
    triton: str
    cuda: str
    gpu: str
    capability: tuple[int, int]
    source_sha256: str
    format: str = "hal.o65.compile-cache.v1"

    def encoded(self) -> bytes:
        return json.dumps(dataclasses.asdict(self), sort_keys=True).encode()

    @property
    def key(self) -> str:
        return hashlib.sha256(self.encoded()).hexdigest()


def source_hash(root: Path) -> str:
    paths = [
        *root.joinpath("hal").rglob("*.py"),
        root / "experiments/065_selfplay_ppo.py",
        root / "pyproject.toml",
        root / "uv.lock",
    ]
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def current_identity(root: Path, device: torch.device) -> CacheIdentity:
    if str(torch.__version__).split("+")[0] != "2.11.0":
        raise RuntimeError("compiler cache is qualified only for Torch 2.11.0")
    if device.type != "cuda":
        raise ValueError("O65 compiler cache requires CUDA")
    return CacheIdentity(
        str(torch.__version__),
        importlib.metadata.version("triton"),
        str(torch.version.cuda),
        torch.cuda.get_device_name(device),
        torch.cuda.get_device_capability(device),
        source_hash(root),
    )


def _publish(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class CompileCache:
    """Select isolated modular caches before compilation; save portable artifacts.

    CUDA graphs contain process-local addresses and must be captured again.
    A manifest is published after the artifact bytes and validated before load.
    """

    def __init__(self, root: Path, identity: CacheIdentity) -> None:
        self.identity = identity
        self.directory = root.resolve() / identity.key
        self.directory.mkdir(parents=True, exist_ok=True)
        metadata = self.directory / "identity.json"
        if metadata.exists():
            if metadata.read_bytes() != identity.encoded():
                raise ValueError("compiler cache identity mismatch")
        elif any(self.directory.iterdir()):
            raise ValueError("compiler cache has no identity")
        else:
            _publish(metadata, identity.encoded())
        self.loaded = False
        self.loaded_counts: dict[str, int] = {}
        payload: bytes | None = None
        manifest = self.directory / "artifacts.json"
        if manifest.exists():
            entry = json.loads(manifest.read_text())
            if (
                not isinstance(entry, dict)
                or set(entry) != {"identity", "sha256", "counts"}
                or entry["identity"] != identity.key
                or not isinstance(entry["sha256"], str)
                or len(entry["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in entry["sha256"])
                or not isinstance(entry["counts"], dict)
                or any(
                    not isinstance(key, str) or type(value) is not int or value < 0
                    for key, value in entry["counts"].items()
                )
            ):
                raise ValueError("compiler artifact manifest mismatch")
            payload = (self.directory / f"{entry['sha256']}.bin").read_bytes()
            if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                raise ValueError("compiler artifact hash mismatch")
            self.loaded_counts = entry["counts"]
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(self.directory / "inductor")
        os.environ["TRITON_CACHE_DIR"] = str(self.directory / "triton")
        if payload is not None:
            if torch.compiler.load_cache_artifacts(payload) is None:
                raise ValueError("Torch rejected compiler cache artifacts")
            self.loaded = True

    def save(self) -> bool:
        artifacts = torch.compiler.save_cache_artifacts()
        if artifacts is None:
            return False
        payload, info = artifacts
        counts = {kind: len(keys) for kind, keys in info.artifacts.items()}
        digest = hashlib.sha256(payload).hexdigest()
        _publish(self.directory / f"{digest}.bin", payload)
        _publish(
            self.directory / "artifacts.json",
            json.dumps({"identity": self.identity.key, "sha256": digest, "counts": counts}, sort_keys=True).encode(),
        )
        return True
