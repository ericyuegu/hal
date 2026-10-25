import json
import shutil
from pathlib import Path

import pytest
import triton
from triton.runtime.cache import get_cache_manager

from hal.inference.triton_cache import LocalCacheManager
from hal.inference.triton_cache import WarmCacheManager


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LocalCacheManager:
    monkeypatch.setenv("HAL_TRITON_CACHE_SOURCE", str(tmp_path / "warm"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "local"))
    (tmp_path / "warm/kernel").mkdir(parents=True)
    return LocalCacheManager("kernel")


def test_warm_group_is_relocated_and_survives_source_removal(cache: LocalCacheManager) -> None:
    assert cache.source is not None
    children = {}
    for name in ("kernel.json", "kernel.cubin", "kernel.ptx"):
        source = cache.source / name
        source.write_bytes(name.encode())
        children[name] = str(source)
    group_file = cache.source / "__grp__kernel.json"
    group_file.write_text(json.dumps({"child_paths": children}))
    original = group_file.read_bytes()
    group = cache.get_group("kernel.json")
    assert group is not None and set(group) == set(children)
    for name, path in group.items():
        assert Path(path).parent == Path(cache.cache_dir)
        assert Path(path).read_bytes() == name.encode()
    assert group_file.read_bytes() == original
    shutil.rmtree(cache.source)
    assert cache.get_group("kernel.json") == group
    assert all(Path(path).read_bytes() == name.encode() for name, path in group.items())


def test_driver_file_is_copied_once_and_cache_miss_is_preserved(cache: LocalCacheManager) -> None:
    assert cache.source is not None
    assert cache.get_file("missing.so") is None
    assert cache.get_group("missing.json") is None
    source = cache.source / "driver.so"
    source.write_bytes(b"driver")
    local = cache.get_file("driver.so")
    assert local is not None and local != str(source)
    source.unlink()
    assert cache.get_file("driver.so") == local
    assert Path(local).read_bytes() == b"driver"


@pytest.mark.parametrize("entry", [{}, {"child_paths": {}}, {"child_paths": {"x": "/elsewhere/x"}}])
def test_invalid_warm_group_fails(cache: LocalCacheManager, entry: dict[str, object]) -> None:
    assert cache.source is not None
    (cache.source / "__grp__kernel.json").write_text(json.dumps(entry))
    with pytest.raises(ValueError, match="cache group"):
        cache.get_group("kernel.json")


def test_missing_group_member_fails(cache: LocalCacheManager) -> None:
    assert cache.source is not None
    entry = {"child_paths": {"kernel.cubin": str(cache.source / "kernel.cubin")}}
    (cache.source / "__grp__kernel.json").write_text(json.dumps(entry))
    with pytest.raises(ValueError, match="missing kernel.cubin"):
        cache.get_group("kernel.json")


def test_untested_triton_version_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(triton, "__version__", "3.7.0")
    with pytest.raises(RuntimeError, match="tested Triton 3.6.0"):
        LocalCacheManager("kernel")
    with pytest.raises(RuntimeError, match="tested Triton 3.6.0"):
        WarmCacheManager("kernel")


def test_warm_plugin_uses_local_kernels_and_counts_new_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRITON_CACHE_MANAGER", "triton.runtime.cache:FileCacheManager")
    key = "0" * 64
    writer = get_cache_manager(key)
    metadata = writer.put(b"{}", "kernel.json")
    binary = writer.put(b"binary", "kernel.cubin")
    writer.put_group("kernel.json", {"kernel.json": metadata, "kernel.cubin": binary})
    monkeypatch.setenv("TRITON_CACHE_MANAGER", "hal.inference.triton_cache:WarmCacheManager")
    reader = get_cache_manager(key)
    assert isinstance(reader, WarmCacheManager)
    assert reader.get_group("kernel.json") == {"kernel.json": metadata, "kernel.cubin": binary}
    assert reader.get_file("driver.so") is None
    for index in range(128):
        assert reader.get_group(f"cold-kernel-{index}.json") is None
    assert reader.get_group("cold-kernel-0.json") is None
    assert get_cache_manager(key).get_group("cold-kernel-127.json") is None
    assert len(json.loads((tmp_path / "kernel-misses.json").read_text())) == 128
    Path(binary).unlink()
    with pytest.raises(ValueError, match="incomplete"):
        reader.get_group("kernel.json")
