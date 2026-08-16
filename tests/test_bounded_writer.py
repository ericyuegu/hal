import threading
import time
from pathlib import Path

import numpy as np
import pytest

from hal.data import bounded_writer as module
from hal.data.bounded_writer import RcloneMDSWriter


def test_rclone_writer_bounds_uploads_and_publishes_index_last(tmp_path: Path, monkeypatch) -> None:
    lock = threading.Lock()
    active = 0
    high_water = 0
    completed: list[str] = []

    def upload(local: Path, destination: str) -> None:
        nonlocal active, high_water
        assert local.is_file()
        with lock:
            active += 1
            high_water = max(high_water, active)
        time.sleep(0.01)
        with lock:
            active -= 1
            completed.append(destination.rsplit("/", 1)[-1])

    monkeypatch.setattr(module, "rclone_copyto", upload)
    writer = RcloneMDSWriter(
        local=tmp_path / "staging",
        remote="r2:test/prefix",
        columns={"x": "ndarray:uint8"},
        compression="zstd",
        size_limit=64,
        max_workers=2,
        max_pending_uploads=2,
    )
    for value in range(20):
        writer.write({"x": np.full(32, value, dtype=np.uint8)})
    writer.finish()

    assert high_water <= 2
    assert completed[-1] == "index.json"
    assert all(name.endswith(".mds.zstd") for name in completed[:-1])
    assert not (tmp_path / "staging").exists()


def test_rclone_writer_surfaces_upload_failure(tmp_path: Path, monkeypatch) -> None:
    def fail(_local: Path, _destination: str) -> None:
        raise RuntimeError("injected upload failure")

    monkeypatch.setattr(module, "rclone_copyto", fail)
    writer = RcloneMDSWriter(
        local=tmp_path / "staging",
        remote="r2:test/prefix",
        columns={"x": "ndarray:uint8"},
        compression="zstd",
        size_limit=64,
        max_workers=1,
        max_pending_uploads=1,
    )
    with pytest.raises(RuntimeError, match="injected upload failure"):
        for _ in range(10):
            writer.write({"x": np.zeros(40, dtype=np.uint8)})
