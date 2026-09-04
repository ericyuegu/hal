"""Regression tests for solid-7z stream-factory backpressure."""

import queue
import threading
from pathlib import Path

import py7zr
import pytest
from py7zr.io import NullIO

from hal.data.archive import _require_supported_py7zr
from hal.data.archive import _StreamFactory
from hal.data.archive import archive_member_path
from hal.data.archive import iter_archive_members


def test_py7zr_private_api_version_is_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hal.data.archive.version", lambda _distribution: "1.2.0")
    with pytest.raises(RuntimeError, match="py7zr==1.1.0"):
        _require_supported_py7zr()


def test_finalize_thread_releases_last_writer_at_folder_boundary(tmp_path: Path) -> None:
    """A folder's final member reaches the consumer before extract returns."""
    output: queue.Queue = queue.Queue()
    slots = threading.Semaphore(1)
    factory = _StreamFactory(tmp_path, output, slots, filter_paths=None)

    writer = factory.create("first.slp")
    writer.write(b"first")
    factory.finalize_thread()

    member, path = output.get_nowait()
    assert member == "first.slp"
    assert Path(path).read_bytes() == b"first"
    Path(path).unlink()
    slots.release()

    second = factory.create("second.slp")
    second.write(b"second")
    factory.abort_all()
    assert not list(tmp_path.iterdir())


def test_abort_all_unblocks_writer_waiting_for_queue_slot(tmp_path: Path) -> None:
    """Early consumer exit releases an extraction thread blocked in create."""
    output: queue.Queue = queue.Queue()
    slots = threading.Semaphore(1)
    factory = _StreamFactory(tmp_path, output, slots, filter_paths=None)
    first = factory.create("first.slp")
    first.write(b"first")

    started = threading.Event()
    finished = threading.Event()
    result = []

    def create_second_writer() -> None:
        started.set()
        result.append(factory.create("second.slp"))
        finished.set()

    thread = threading.Thread(target=create_second_writer)
    thread.start()
    assert started.wait(timeout=1.0)
    assert not finished.wait(timeout=0.05)

    factory.abort_all()

    assert finished.wait(timeout=1.0)
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert len(result) == 1
    assert isinstance(result[0], NullIO)
    assert not list(tmp_path.iterdir())
    assert slots.acquire(blocking=False)


def test_synthetic_7z_streams_all_members_and_rejects_a_missing_filter(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    payloads = {"first.slp": b"first", "nested/second.slp": b"second"}
    archive = tmp_path / "replays.7z"
    with py7zr.SevenZipFile(archive, "w") as writer:
        for name, payload in payloads.items():
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            writer.write(path, arcname=name)

    extracted_root = tmp_path / "extracted"
    seen: dict[str, bytes] = {}
    for synthetic, extracted in iter_archive_members(archive, tmpfs_root=extracted_root, queue_size=1):
        seen[synthetic] = extracted.read_bytes()
        extracted.unlink()
    assert seen == {archive_member_path(archive, name): payload for name, payload in payloads.items()}

    with pytest.raises(FileNotFoundError, match="missing.slp"):
        list(iter_archive_members(archive, tmpfs_root=extracted_root, filter_paths={"missing.slp"}))
