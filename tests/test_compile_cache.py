import dataclasses
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hal.inference import compile_cache


@pytest.fixture(autouse=True)
def restore_cache_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
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
    assert manifest["sha256"] == hashlib.sha256(loaded[0]).hexdigest()
    (cache.directory / f"{manifest['sha256']}.bin").write_bytes(b"damaged")
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
    "changes",
    [
        {"identity": "b" * 64},
        {"sha256": "../bad"},
        {"counts": {"inductor": True}},
        {"unknown": 1},
    ],
)
def test_cache_rejects_malformed_manifest_before_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changes: dict
) -> None:
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", lambda: (b"graphs", SimpleNamespace(artifacts={})))
    cache = compile_cache.CompileCache(tmp_path, identity())
    cache.save()
    path = cache.directory / "artifacts.json"
    manifest = json.loads(path.read_text())
    manifest.update(changes)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest mismatch"):
        compile_cache.CompileCache(tmp_path, identity())


def test_loose_compiler_files_stay_off_the_cache_root(tmp_path: Path) -> None:
    cache = compile_cache.CompileCache(tmp_path / "volume", identity())
    for name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        assert not Path(os.environ[name]).is_relative_to(tmp_path / "volume")
    assert sorted(path.name for path in cache.directory.iterdir()) == ["identity.json"]
