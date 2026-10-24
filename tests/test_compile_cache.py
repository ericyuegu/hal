import dataclasses
import hashlib
import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch._inductor.output_code import CompiledFxGraph
from torch._inductor.triton_bundler import TritonBundle
from torch._inductor.triton_bundler import TritonKernelArtifact
from torch._inductor.triton_bundler import TritonKernelArtifacts

from hal.inference import compile_cache
from hal.inference.triton_cache import CacheOnlyManager


@pytest.fixture(autouse=True)
def restore_cache_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TRITON_CACHE_MANAGER", "HAL_TRITON_CACHE_SOURCE"):
        if name in os.environ:
            monkeypatch.setenv(name, os.environ[name])
        else:
            monkeypatch.delenv(name, raising=False)


def identity() -> compile_cache.CacheIdentity:
    return compile_cache.CacheIdentity("2.11.0+cu130", "3.6.0", "13.0", "RTX 3060", (8, 6), "a" * 64)


@pytest.mark.parametrize(
    "field,value",
    [
        ("torch", "2.11.0+cu128"),
        ("triton", "3.7.0"),
        ("cuda", "13.1"),
        ("gpu", "other"),
        ("capability", (12, 0)),
        ("source_sha256", "b" * 64),
    ],
)
def test_cache_identity_isolates_versions_hardware_and_source(field: str, value: object) -> None:
    assert identity().key != dataclasses.replace(identity(), **{field: value}).key


def test_cache_roundtrip_and_corruption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        torch.compiler,
        "save_cache_artifacts",
        lambda: (b"qualified graphs", SimpleNamespace(artifacts={"inductor": ["graph"]})),
    )
    loaded = []
    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", lambda data: loaded.append(data) or object())
    cache = compile_cache.CompileCache(tmp_path, identity())
    assert not cache.loaded
    assert cache.save()
    restored = compile_cache.CompileCache(tmp_path, identity())
    assert restored.loaded and loaded == [b"qualified graphs"]
    manifest = json.loads((cache.directory / "artifacts.json").read_text())
    digest = manifest["entries"][0]["sha256"]
    assert digest == hashlib.sha256(loaded[0]).hexdigest()
    (cache.directory / f"{digest}.bin").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="hash mismatch"):
        compile_cache.CompileCache(tmp_path, identity())
    assert len(loaded) == 1


def test_cache_rejects_wrong_identity_before_setting_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "unchanged")
    directory = tmp_path / identity().key
    directory.mkdir()
    (directory / "identity.json").write_bytes(dataclasses.replace(identity(), gpu="other").encoded())
    with pytest.raises(ValueError, match="identity mismatch"):
        compile_cache.CompileCache(tmp_path, identity())
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == "unchanged"


def test_cache_rejects_orphaned_modular_cache(tmp_path: Path) -> None:
    directory = tmp_path / identity().key
    (directory / "inductor").mkdir(parents=True)
    with pytest.raises(ValueError, match="no identity"):
        compile_cache.CompileCache(tmp_path, identity())


def test_cache_rejects_unreadable_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        torch.compiler, "save_cache_artifacts", lambda: (b"graphs", SimpleNamespace(artifacts={"inductor": ["graph"]}))
    )
    cache = compile_cache.CompileCache(tmp_path, identity())
    cache.save()
    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", lambda data: None)
    with pytest.raises(ValueError, match="Torch rejected"):
        compile_cache.CompileCache(tmp_path, identity())


def test_cache_without_compilations_does_not_publish_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", lambda: None)
    cache = compile_cache.CompileCache(tmp_path, identity())
    assert not cache.save()
    assert not (cache.directory / "artifacts.json").exists()


def test_source_hash_changes_with_model_code(tmp_path: Path) -> None:
    (tmp_path / "hal/models").mkdir(parents=True)
    (tmp_path / "experiments").mkdir()
    for name in ("experiments/065_selfplay_ppo.py", "pyproject.toml", "uv.lock", "hal/models/policy.py"):
        (tmp_path / name).write_text(name)
    before = compile_cache.source_hash(tmp_path, tuple(tmp_path.rglob("*.*")))
    (tmp_path / "hal/models/policy.py").write_text("changed")
    assert compile_cache.source_hash(tmp_path, tuple(tmp_path.rglob("*.*"))) != before


@pytest.mark.parametrize(
    "changes,entry_changes",
    [
        ({"identity": "b" * 64}, {}),
        ({"unknown": 1}, {}),
        ({}, {"sha256": "../bad"}),
        ({}, {"keys": {"inductor": [1]}}),
        ({}, {"unknown": 1}),
    ],
)
def test_cache_rejects_malformed_manifest_before_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changes: dict, entry_changes: dict
) -> None:
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", lambda: (b"graphs", SimpleNamespace(artifacts={})))
    cache = compile_cache.CompileCache(tmp_path, identity())
    cache.save()
    path = cache.directory / "artifacts.json"
    manifest = json.loads(path.read_text())
    manifest.update(changes)
    manifest["entries"][0].update(entry_changes)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest mismatch"):
        compile_cache.CompileCache(tmp_path, identity())


def test_loose_compiler_files_stay_off_the_cache_root(tmp_path: Path) -> None:
    cache = compile_cache.CompileCache(tmp_path / "volume", identity())
    for name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        assert not Path(os.environ[name]).is_relative_to(tmp_path / "volume")
    assert sorted(path.name for path in cache.directory.iterdir()) == ["identity.json"]


def test_rank_cache_uses_shared_seed_and_isolated_local_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        torch.compiler, "save_cache_artifacts", lambda: (b"graphs", SimpleNamespace(artifacts={"inductor": ["tick"]}))
    )
    loaded: list[bytes] = []
    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", lambda data: loaded.append(data) or object())
    seed_root = tmp_path / "rank0"
    seed = compile_cache.CompileCache(seed_root, identity())
    assert seed.save()
    original = {p.name: p.read_bytes() for p in seed.directory.iterdir()}
    local_paths: set[str] = set()
    for rank in range(1, 8):
        monkeypatch.setenv("HAL_TRITON_CACHE_SOURCE", "/network/loose-cache")
        cache = compile_cache.CompileCache(
            tmp_path / f"rank{rank}", identity(), seed_root=seed_root, require_loaded=True
        )
        assert cache.loaded_counts == {"inductor": 1}
        local_paths.add(os.environ["TRITON_CACHE_DIR"])
        assert os.environ["TRITON_CACHE_MANAGER"] == "hal.inference.triton_cache:CacheOnlyManager"
        assert "HAL_TRITON_CACHE_SOURCE" not in os.environ
        assert cache.directory != seed.directory
        assert cache.save()
    assert len(local_paths) == 7
    assert loaded == [b"graphs"] * 7
    assert original == {p.name: p.read_bytes() for p in seed.directory.iterdir()}


@pytest.mark.parametrize("failure", ["missing", "identity", "manifest", "hash"])
def test_invalid_seed_fails_before_import_or_environment_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    monkeypatch.setattr(
        torch.compiler, "save_cache_artifacts", lambda: (b"graphs", SimpleNamespace(artifacts={"inductor": ["tick"]}))
    )
    seed = compile_cache.CompileCache(tmp_path / "rank0", identity())
    seed.save()
    if failure == "missing":
        (seed.directory / "identity.json").unlink()
    elif failure == "identity":
        (seed.directory / "identity.json").write_bytes(b"wrong")
    elif failure == "manifest":
        (seed.directory / "artifacts.json").unlink()
    else:
        next(seed.directory.glob("*.bin")).write_bytes(b"corrupt")
    monkeypatch.setenv("TRITON_CACHE_DIR", "unchanged")
    with pytest.raises(ValueError):
        compile_cache.CompileCache(tmp_path / "rank7", identity(), seed_root=tmp_path / "rank0", require_loaded=True)
    assert os.environ["TRITON_CACHE_DIR"] == "unchanged"


def test_required_cache_rejects_empty_artifacts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Inductor graphs is required"):
        compile_cache.CompileCache(tmp_path, identity(), require_loaded=True)


def test_all_bundled_kernels_are_local_before_any_fx_lookup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    group = {"child_paths": {name: f"[REPLACE]/{name}" for name in ("kernel.json", "kernel.cubin")}}
    bundle = TritonBundle(
        [
            TritonKernelArtifacts(
                "kernel",
                0,
                [
                    TritonKernelArtifact("__grp__kernel.json", json.dumps(group).encode()),
                    TritonKernelArtifact("kernel.json", b"{}"),
                    TritonKernelArtifact("kernel.cubin", b"binary"),
                ],
            )
        ],
        [],
    )
    # Only the bundle field participates in restoration; no CUDA object is needed.
    graph = CompiledFxGraph.__new__(CompiledFxGraph)
    graph.compiled_fn_runner = None
    graph._triton_bundle = bundle
    directory = tmp_path / "inductor/fxgraph/key"
    directory.mkdir(parents=True)
    (directory / "first").write_bytes(pickle.dumps(graph))
    (directory / "second").write_bytes(pickle.dumps(graph))
    assert compile_cache._restore_kernels(tmp_path / "inductor") == 1
    restored = CacheOnlyManager("kernel").get_group("kernel.json")
    assert Path(restored["kernel.cubin"]).read_bytes() == b"binary"
    assert all(Path(path).parent == tmp_path / "triton/kernel" for path in restored.values())


def test_measurement_can_import_an_explicit_same_hardware_source_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = identity()
    new = dataclasses.replace(old, source_sha256="b" * 64)
    monkeypatch.setattr(
        torch.compiler, "save_cache_artifacts", lambda: (b"graphs", SimpleNamespace(artifacts={"inductor": ["tick"]}))
    )
    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", lambda payload: object())
    seed = compile_cache.CompileCache(tmp_path / "seed", old)
    assert seed.save()
    consumer = compile_cache.CompileCache(tmp_path / "new", new, seed_root=tmp_path / "seed", seed_identity=old)
    assert consumer.loaded and consumer.identity == new
    with pytest.raises(ValueError, match="compiler and hardware"):
        compile_cache.CompileCache(
            tmp_path / "wrong", new, seed_root=tmp_path / "seed", seed_identity=dataclasses.replace(old, gpu="other")
        )


def test_cache_keeps_artifacts_from_runs_with_different_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saves = iter(
        [
            (b"1024 matches", {"inductor": ["tick-1024"], "autotune": ["a"]}),
            (b"2048 matches", {"inductor": ["tick-2048"]}),
            (b"1024 again", {"inductor": ["tick-1024"]}),
            (b"both", {"inductor": ["tick-1024", "tick-2048"], "autotune": ["a"]}),
        ]
    )

    def save() -> tuple[bytes, SimpleNamespace]:
        payload, artifacts = next(saves)
        return payload, SimpleNamespace(artifacts=artifacts)

    loaded: list[bytes] = []
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", save)
    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", lambda data: loaded.append(data) or object())
    assert compile_cache.CompileCache(tmp_path, identity()).save()
    assert compile_cache.CompileCache(tmp_path, identity()).save()
    assert loaded == [b"1024 matches"]
    loaded.clear()
    # A run whose artifacts an existing entry covers adds nothing.
    assert not compile_cache.CompileCache(tmp_path, identity()).save()
    assert loaded == [b"1024 matches", b"2048 matches"]
    loaded.clear()
    # A run that used both shapes replaces the entries it covers.
    assert compile_cache.CompileCache(tmp_path, identity()).save()
    assert loaded == [b"1024 matches", b"2048 matches"]
    loaded.clear()
    restored = compile_cache.CompileCache(tmp_path, identity())
    assert loaded == [b"both"]
    assert restored.loaded_counts == {"inductor": 2, "autotune": 1}
