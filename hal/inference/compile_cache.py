"""Persist compiler artifacts behind a checked source and hardware identity."""

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import pickle
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import torch
from torch._dynamo.utils import counters
from torch._inductor import config as inductor_config
from torch._inductor.output_code import CompiledFxGraph
from torch._inductor.triton_bundler import TritonBundle
from torch._inductor.triton_bundler import TritonBundler
from torch.compiler import config as compiler_config


def compilation_statistics() -> dict[str, int]:
    """Read the cache counters qualified against Torch 2.11.0."""
    if str(torch.__version__).split("+")[0] != "2.11.0":
        raise RuntimeError("compiler cache counters require tested Torch 2.11.0")
    return {
        name: counters["inductor"][name]
        for name in ("fxgraph_cache_hit", "fxgraph_cache_miss", "fxgraph_cache_bypass")
    }


def _restore_kernels(directory: Path) -> int:
    # Torch 2.11 restores kernels lazily on an FX hit. A different FX wrapper
    # can use the same kernels, so restore every bundle before compilation.
    # Remove this adapter when load_cache_artifacts restores Triton eagerly.
    if str(torch.__version__).split("+")[0] != "2.11.0":
        raise RuntimeError("Triton bundle restore requires tested Torch 2.11.0")
    seen: set[tuple[int, str]] = set()
    for path in (directory / "fxgraph").rglob("*"):
        if not path.is_file():
            continue
        graph = pickle.loads(path.read_bytes())
        if not isinstance(graph, CompiledFxGraph):
            raise ValueError("invalid FX compiler artifact")
        bundle = graph._triton_bundle
        if bundle is None:
            continue
        artifacts = []
        for entry in bundle.kernel_artifacts:
            if entry.device != 0:
                raise ValueError("compiler kernels must target the rank's cuda:0")
            key = entry.device, entry.kernel_hash
            if key not in seen:
                artifacts.append(entry)
                seen.add(key)
        if artifacts and TritonBundler.read_and_emit(TritonBundle(artifacts, [])) is None:
            raise RuntimeError("Torch did not restore bundled Triton kernels")
    return len(seen)


@dataclasses.dataclass(frozen=True, slots=True)
class CacheIdentity:
    torch: str
    triton: str
    cuda: str
    gpu: str
    capability: tuple[int, int]
    source_sha256: str
    enable_pdl: bool
    cache_key_tag: str = ""
    format: str = "hal.o65.compile-cache.v4"

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
        inductor_config.triton.enable_pdl,
        compiler_config.cache_key_tag,
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

    def __init__(
        self,
        root: Path,
        identity: CacheIdentity,
        *,
        seed_root: Path | None = None,
        seed_identity: CacheIdentity | None = None,
        require_loaded: bool = False,
    ) -> None:
        if seed_identity is not None and (
            seed_root is None or dataclasses.replace(seed_identity, source_sha256=identity.source_sha256) != identity
        ):
            raise ValueError("compiler seed must have the same compiler, configuration and hardware identity")
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
        sources = [(self.directory, item) for item in self.entries]
        if seed_root is not None:
            expected = identity if seed_identity is None else seed_identity
            seed = seed_root.resolve() / expected.key
            if seed != self.directory:
                if not (seed / "identity.json").is_file():
                    raise ValueError("compiler cache seed identity is missing")
                if (seed / "identity.json").read_bytes() != expected.encoded():
                    raise ValueError("compiler cache seed identity mismatch")
                if not (seed / "artifacts.json").is_file():
                    raise ValueError("compiler cache seed manifest is missing")
                sources.extend((seed, item) for item in _read_manifest(seed / "artifacts.json", expected))
        payloads: list[bytes] = []
        loaded_keys: set[tuple[str, str]] = set()
        for directory, item in sources:
            payload = (directory / f"{item['sha256']}.bin").read_bytes()
            if hashlib.sha256(payload).hexdigest() != item["sha256"]:
                raise ValueError("compiler artifact hash mismatch")
            payloads.append(payload)
            loaded_keys |= _key_set(cast(dict[str, list[str]], item["keys"]))
        for kind, _ in loaded_keys:
            self.loaded_counts[kind] = self.loaded_counts.get(kind, 0) + 1
        if require_loaded and not self.loaded_counts.get("inductor", 0):
            raise ValueError("warm compiler cache with Inductor graphs is required")
        # Loose Inductor and Triton files stay on local disk; the packed artifact
        # is the persistent cache, and network volumes cap their file counts.
        scratch = Path(tempfile.mkdtemp(prefix=f"hal-compile-{identity.key[:12]}-"))
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(scratch / "inductor")
        os.environ["TRITON_CACHE_DIR"] = str(scratch / "triton")
        os.environ["TRITON_CACHE_MANAGER"] = "triton.runtime.cache:FileCacheManager"
        os.environ.pop("HAL_TRITON_CACHE_SOURCE", None)
        for payload in payloads:
            if torch.compiler.load_cache_artifacts(payload) is None:
                raise ValueError("Torch rejected compiler cache artifacts")
            self.loaded = True
        self.restored_kernels = _restore_kernels(scratch / "inductor") if self.loaded else 0
        if require_loaded:
            os.environ["TRITON_CACHE_MANAGER"] = "hal.inference.triton_cache:WarmCacheManager"

    def kernel_misses(self) -> int:
        path = Path(os.environ["TRITON_CACHE_DIR"]) / "kernel-misses.json"
        return len(json.loads(path.read_text())) if path.exists() else 0

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
