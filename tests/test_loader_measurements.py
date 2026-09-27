import hashlib
import importlib.util
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest


def _load_tool(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_FIXTURES = Path(__file__).parent / "fixtures" / "o59"
loader_pairs = _load_tool("hal_loader_pairs", _FIXTURES / "run_loader_pairs.py")
training_measurement = _load_tool("hal_training_measurement", _FIXTURES / "measure_training_updates.py")
replay_benchmark = _load_tool(
    "hal_replay_benchmark", Path(__file__).resolve().parents[1] / "scripts/benchmark_replay_loader.py"
)


@pytest.mark.parametrize(
    "measure", [replay_benchmark._process_tree_high_water_rss_bytes, training_measurement._peak_process_tree_rss]
)
@pytest.mark.parametrize("child_peak", ["VmHWM:\t4096 kB\n", "", "VmHWM:\t0 kB\n"])
def test_peak_memory_requires_high_water_counters_for_every_process(
    tmp_path: Path, measure: Callable[[Path], int | None], child_peak: str
) -> None:
    parent = tmp_path / str(os.getpid())
    child = tmp_path / str(os.getpid() + 1)
    parent.mkdir()
    child.mkdir()
    (parent / "status").write_text("PPid:\t0\nVmHWM:\t2048 kB\nVmRSS:\t1024 kB\n")
    (child / "status").write_text(f"PPid:\t{os.getpid()}\nVmRSS:\t1024 kB\n{child_peak}")
    assert measure(tmp_path) == (6144 * 1024 if "4096" in child_peak else None)


def _corpus_index(root: Path) -> tuple[Path, str]:
    root.mkdir()
    data = b"retained replay data"
    (root / "shard.00000.mds").write_bytes(data)
    payload = json.dumps({"shards": [{"raw_data": {"basename": "shard.00000.mds", "bytes": len(data)}}]}).encode()
    index = root / "index.json"
    index.write_bytes(payload)
    return index, hashlib.sha256(payload).hexdigest()


def test_loader_comparison_prepares_only_shared_raw_shards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index, digest = _corpus_index(tmp_path / "corpus")
    control = tmp_path / "control"
    control.symlink_to(index.parent, target_is_directory=True)
    advice: list[tuple[bytes, int, int, int]] = []

    def advise(fd: int, offset: int, length: int, mode: int) -> None:
        advice.append((os.pread(fd, 100, 0), offset, length, mode))

    monkeypatch.setattr(os, "posix_fadvise", advise)
    result = loader_pairs._prepare_page_cache(index, control / "index.json", expected_sha256=digest)
    assert result.raw_shard_count == 1
    assert result.raw_bytes == 20
    assert advice == [(b"retained replay data", 0, 0, os.POSIX_FADV_DONTNEED)]
    assert (index.parent / "shard.00000.mds").read_bytes() == b"retained replay data"


def test_loader_comparison_rejects_separate_cache_directories(tmp_path: Path) -> None:
    index, digest = _corpus_index(tmp_path / "candidate")
    other, _ = _corpus_index(tmp_path / "control")
    with pytest.raises(ValueError, match="same local corpus"):
        loader_pairs._prepare_page_cache(index, other, expected_sha256=digest)


def test_loader_comparison_rejects_changed_manifest(tmp_path: Path) -> None:
    index, _ = _corpus_index(tmp_path / "corpus")
    with pytest.raises(ValueError, match="manifest differs"):
        loader_pairs._prepare_page_cache(index, index, expected_sha256="0" * 64)


@pytest.mark.parametrize("is_missing", [True, False])
def test_loader_comparison_requires_complete_raw_cache(tmp_path: Path, is_missing: bool) -> None:
    index, digest = _corpus_index(tmp_path / "corpus")
    raw = index.parent / "shard.00000.mds"
    if is_missing:
        raw.unlink()
    else:
        raw.write_bytes(b"truncated")
    with pytest.raises(FileNotFoundError if is_missing else ValueError):
        loader_pairs._prepare_page_cache(index, index, expected_sha256=digest)
