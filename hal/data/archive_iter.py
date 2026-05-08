"""Stream .slp members directly out of a solid .7z archive into /dev/shm.

The downloaded ranked-anonymized archives are 70+ GiB compressed and don't
fit on local disk uncompressed, so build_index/process_replays need to read
.slp content without extracting. peppi-py only accepts a filesystem path, so
we materialize each member to a tmpfs file just long enough for the consumer
to parse it, then unlink.

py7zr's `WriterFactory` hook lets us substitute our own per-file writer;
`parallel=True` (its default for file paths) spawns one worker thread per
solid block, so multi-block decompression overlaps "for free" — we don't
have to drive concurrency ourselves at this layer.

Two non-obvious things about py7zr's WriterFactory protocol that this module
works around:

1. py7zr never calls a ``close()`` method on the writer it receives from
   ``factory.create()`` — its internal ``MemIO`` wrapper has a no-op
   ``__exit__``. The signal "writer N is done" is implicit: it arrives when
   ``factory.create()`` is called for the *next* file in the same thread
   (folders extract files sequentially within a thread). We finalize the
   previous per-thread writer at that point, plus a sweep after extract
   returns to flush the last writer per thread.

2. Backpressure has to happen *before* a tmpfs file is opened, not after,
   or a slow consumer lets the producer fill /dev/shm. We acquire a bounded
   semaphore in ``TmpfsWriter.__init__`` and release it on the consumer
   side after the consumer is done with the path.
"""

import os
import queue
import threading
from collections.abc import Iterator
from pathlib import Path

import py7zr
from loguru import logger
from py7zr.io import NullIO
from py7zr.io import Py7zIO
from py7zr.io import WriterFactory

_SENTINEL: object = object()


def archive_member_path(archive: Path, member: str) -> str:
    """Synthetic path stored in ReplayIndexEntry.path for archive members.

    Resolves the archive to an absolute path so the index is portable across
    runs that cd around. Round-trippable via ``parse_archive_member_path``.
    """
    return f"archive://{archive.resolve()}!{member}"


def parse_archive_member_path(path: str) -> tuple[Path, str] | None:
    """Inverse of ``archive_member_path``; returns None for plain filesystem paths."""
    if not path.startswith("archive://"):
        return None
    rest = path[len("archive://") :]
    archive_str, _, member = rest.partition("!")
    if not member:
        raise ValueError(f"malformed archive path (missing '!member'): {path!r}")
    return Path(archive_str), member


class _TmpfsWriter(Py7zIO):
    """Writes one decompressed member to a unique tmpfs file."""

    def __init__(self, member: str, tmpfs_root: Path, out_q: queue.Queue, sem: threading.Semaphore) -> None:
        # Acquire BEFORE opening the file: bounds /dev/shm usage to queue_size
        # files even if the consumer is much slower than the producer.
        sem.acquire()
        self.member: str = member
        self._out_q: queue.Queue = out_q
        # Filename uniqueness: pid+tid+monotonic ns. Multiple producer threads
        # (one per solid block) write concurrently; basenames must not collide.
        self.path: Path = tmpfs_root / f"{os.getpid()}_{threading.get_ident()}_{os.urandom(4).hex()}.slp"
        self._fp = self.path.open("wb")
        self._size: int = 0
        self._finalized: bool = False

    def write(self, s: bytes | bytearray) -> int:
        n = self._fp.write(s)
        self._size += n
        return n

    def read(self, size: int | None = None) -> bytes:
        raise NotImplementedError("TmpfsWriter is write-only")

    def seek(self, offset: int, whence: int = 0) -> int:
        return 0

    def flush(self) -> None:
        self._fp.flush()

    def size(self) -> int:
        return self._size

    def finalize(self) -> None:
        """Close the file and hand it off to the consumer queue."""
        if self._finalized:
            return
        self._finalized = True
        self._fp.close()
        self._out_q.put((self.member, self.path))


class _StreamFactory(WriterFactory):
    """Per-extract factory that serializes finalize calls within each thread."""

    def __init__(
        self,
        tmpfs_root: Path,
        out_q: queue.Queue,
        sem: threading.Semaphore,
        filter_paths: set[str] | None,
    ) -> None:
        self._tmpfs_root: Path = tmpfs_root
        self._out_q: queue.Queue = out_q
        self._sem: threading.Semaphore = sem
        self._filter_paths: set[str] | None = filter_paths
        self._open_per_thread: dict[int, Py7zIO] = {}
        self._lock: threading.Lock = threading.Lock()

    def create(self, filename: str) -> Py7zIO:
        tid = threading.get_ident()
        with self._lock:
            prev = self._open_per_thread.get(tid)
        if isinstance(prev, _TmpfsWriter):
            prev.finalize()

        if self._filter_paths is not None and filename not in self._filter_paths:
            new: Py7zIO = NullIO()
        else:
            new = _TmpfsWriter(filename, self._tmpfs_root, self._out_q, self._sem)

        with self._lock:
            self._open_per_thread[tid] = new
        return new

    def finalize_all(self) -> None:
        with self._lock:
            writers = list(self._open_per_thread.values())
            self._open_per_thread.clear()
        for w in writers:
            if isinstance(w, _TmpfsWriter):
                w.finalize()


def iter_archive_members(
    archive: Path,
    *,
    tmpfs_root: Path,
    filter_paths: set[str] | None = None,
    queue_size: int = 64,
) -> Iterator[tuple[str, Path]]:
    """Yield ``(synthetic_path, tmpfs_path)`` for each archive member.

    The tmpfs file is owned by the consumer once yielded: the consumer MUST
    unlink it (success or failure) to release a slot in the bounded queue.
    Slots refill via ``sem.release()`` after the consumer's iteration body
    runs, so slow consumers backpressure the producer instead of filling
    /dev/shm.

    ``filter_paths`` is a set of *member* names (e.g. "dev/Game_X.slp"), not
    synthetic paths — it's matched against py7zr's filename, which is the
    raw archive member. Excluded files are still decompressed (unavoidable
    in a solid block) but discarded into a NullIO instead of materializing.

    Iteration order is "as files complete decompression"; for a solid archive
    that's roughly archive order interleaved across blocks. Do not rely on
    a strict order.
    """
    if not archive.exists():
        raise FileNotFoundError(f"archive not found: {archive}")
    tmpfs_root.mkdir(parents=True, exist_ok=True)

    out_q: queue.Queue = queue.Queue()
    sem = threading.Semaphore(queue_size)
    factory = _StreamFactory(tmpfs_root, out_q, sem, filter_paths)
    producer_exc: list[BaseException] = []

    def _producer() -> None:
        try:
            with py7zr.SevenZipFile(str(archive), "r") as z:
                z.extract(factory=factory)
        except BaseException as e:
            logger.error(f"archive producer crashed on {archive}: {e!r}")
            producer_exc.append(e)
        finally:
            factory.finalize_all()
            out_q.put(_SENTINEL)

    producer = threading.Thread(target=_producer, name=f"py7zr-producer-{archive.name}", daemon=True)
    producer.start()

    try:
        while True:
            item = out_q.get()
            if item is _SENTINEL:
                break
            member, tmpfs_path = item
            synthetic = archive_member_path(archive, member)
            try:
                yield synthetic, tmpfs_path
            finally:
                # Release one queue slot whether the consumer succeeded or not.
                # Caller is responsible for unlinking tmpfs_path.
                sem.release()
    finally:
        producer.join()

    if producer_exc:
        raise producer_exc[0]
