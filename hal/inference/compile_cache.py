"""Persist compiler artifacts behind a checked source and hardware identity."""

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import torch


@dataclasses.dataclass(frozen=True, slots=True)
class CacheIdentity:
    torch: str
    triton: str
    cuda: str
    gpu: str
    capability: tuple[int, int]
    source_sha256: str
    format: str = "hal.o65.compile-cache.v2"

    def encoded(self) -> bytes:
        return json.dumps(dataclasses.asdict(self), sort_keys=True).encode()

    @property
    def key(self) -> str:
        return hashlib.sha256(self.encoded()).hexdigest()


def source_hash(root: Path, paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def current_identity(source_sha256: str, device: torch.device) -> CacheIdentity:
    if str(torch.__version__).split("+")[0] != "2.11.0":
        raise RuntimeError("compiler cache is qualified only for Torch 2.11.0")
    if device.type != "cuda":
        raise ValueError("compiler cache requires CUDA")
    return CacheIdentity(
        str(torch.__version__),
        importlib.metadata.version("triton"),
        str(torch.version.cuda),
        torch.cuda.get_device_name(device),
        torch.cuda.get_device_capability(device),
        source_sha256,
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


def _read_manifest(path: Path, identity: CacheIdentity) -> list[dict[str, object]]:
    entry = json.loads(path.read_text())
    if not isinstance(entry, dict) or set(entry) != {"identity", "entries"} or entry["identity"] != identity.key:
        raise ValueError("compiler artifact manifest mismatch")
    entries = entry["entries"]
    if not isinstance(entries, list):
        raise ValueError("compiler artifact manifest mismatch")
    for item in entries:
        if (
            not isinstance(item, dict)
            or set(item) != {"sha256", "keys"}
            or not isinstance(item["sha256"], str)
            or len(item["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in item["sha256"])
            or not isinstance(item["keys"], dict)
            or any(
                not isinstance(kind, str) or not isinstance(keys, list) or not all(isinstance(k, str) for k in keys)
                for kind, keys in item["keys"].items()
            )
        ):
            raise ValueError("compiler artifact manifest mismatch")
    return entries


def _key_set(keys: dict[str, list[str]]) -> set[tuple[str, str]]:
    return {(kind, key) for kind, values in keys.items() for key in values}


class CompileCache:
    """Select isolated modular caches before compilation; save portable artifacts.

    CUDA graphs contain process-local addresses and must be captured again.
    Each save records the artifacts its process used. Runs with different
    shapes use different artifacts, so the manifest keeps every entry that no
    later entry covers, and a load restores their union.
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
        manifest = self.directory / "artifacts.json"
        self.entries = _read_manifest(manifest, identity) if manifest.exists() else []
        payloads: list[bytes] = []
        loaded_keys: set[tuple[str, str]] = set()
        for item in self.entries:
            payload = (self.directory / f"{item['sha256']}.bin").read_bytes()
            if hashlib.sha256(payload).hexdigest() != item["sha256"]:
                raise ValueError("compiler artifact hash mismatch")
            payloads.append(payload)
            loaded_keys |= _key_set(cast(dict[str, list[str]], item["keys"]))
        for kind, _ in loaded_keys:
            self.loaded_counts[kind] = self.loaded_counts.get(kind, 0) + 1
        # Loose Inductor and Triton files stay on local disk; the packed artifact
        # is the persistent cache, and network volumes cap their file counts.
        scratch = Path(tempfile.mkdtemp(prefix=f"hal-compile-{identity.key[:12]}-"))
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(scratch / "inductor")
        os.environ["TRITON_CACHE_DIR"] = str(scratch / "triton")
        for payload in payloads:
            if torch.compiler.load_cache_artifacts(payload) is None:
                raise ValueError("Torch rejected compiler cache artifacts")
            self.loaded = True

    def save(self) -> bool:
        artifacts = torch.compiler.save_cache_artifacts()
        if artifacts is None:
            return False
        payload, info = artifacts
        keys = {kind: sorted(str(key) for key in values) for kind, values in info.artifacts.items()}
        used = _key_set(keys)
        manifest = self.directory / "artifacts.json"
        entries = _read_manifest(manifest, self.identity) if manifest.exists() else []
        if any(used <= _key_set(cast(dict[str, list[str]], item["keys"])) for item in entries):
            return False
        digest = hashlib.sha256(payload).hexdigest()
        _publish(self.directory / f"{digest}.bin", payload)
        retained = [item for item in entries if not _key_set(cast(dict[str, list[str]], item["keys"])) <= used]
        retained.append({"sha256": digest, "keys": keys})
        _publish(
            manifest,
            json.dumps({"identity": self.identity.key, "entries": retained}, sort_keys=True).encode(),
        )
        self.entries = retained
        return True
