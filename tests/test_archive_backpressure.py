"""Regression tests for solid-7z stream-factory backpressure."""

import queue
import threading
from pathlib import Path

from py7zr.io import NullIO

from hal.data.archive import _StreamFactory


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
