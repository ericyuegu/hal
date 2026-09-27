import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "hal_loader_pairs", Path(__file__).parent / "fixtures" / "o59" / "run_loader_pairs.py"
)
assert _SPEC is not None and _SPEC.loader is not None
loader_pairs = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = loader_pairs
_SPEC.loader.exec_module(loader_pairs)


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
