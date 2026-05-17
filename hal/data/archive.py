"""Stream .slp members directly out of a solid .7z archive into /dev/shm.

The ranked-anonymized raw slp archives are >100 GB and ~10x compressed.
We use peppi for batch decoding but peppi only accepts a filesystem path,
so we materialize each member to a tmpfs file in /dev/shm.

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
   or a slow consumer lets the producer fill /dev/shm. The factory acquires
   a bounded semaphore *before* constructing each per-file writer, and the
   consumer releases the slot after it's done with the path.
"""

import concurrent.futures
import contextlib
import os
import queue
import threading
import types
from collections import Counter
from collections.abc import Generator
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import py7zr
from loguru import logger
from py7zr.exceptions import InternalError
from py7zr.io import NullIO
from py7zr.io import Py7zIO
from py7zr.io import WriterFactory

from hal.paths import repo_relative

_SENTINEL: object = object()


def archive_member_path(archive: Path, member: str) -> str:
    """Synthetic path stored in ReplayIndexEntry.path for archive members.
    Archive is repo-relative when in-repo (portable), else absolute.
    Round-trips via ``parse_archive_member_path``.
    """
    return f"archive://{repo_relative(archive)}!{member}"


def parse_archive_member_path(path: str) -> tuple[Path, str] | None:
    """Inverse of ``archive_member_path``; returns None for plain filesystem paths."""
    if not path.startswith("archive://"):
        return None
    rest = path[len("archive://") :]
    archive_str, _, member = rest.partition("!")
    if not member:
        raise ValueError(f"malformed archive path (missing '!member'): {path!r}")
    return Path(archive_str), member


@dataclass(frozen=True, slots=True)
class ReplayWork:
    """One unit of replay-processing work shared across stage 1 and stage 3.

    Workers must unlink ``open_path`` in a finally-block when ``unlink_after``
    is True (the file is a tmpfs copy streamed from a .7z archive).
    """

    open_path: Path
    manifest_key: str
    unlink_after: bool


def iter_replay_work(
    *,
    fs_paths: Iterable[tuple[Path, str]] = (),
    archive_members: Mapping[Path, Iterable[str]] | None = None,
    tmpfs_root: Path,
    queue_size: int = 64,
) -> Generator[ReplayWork]:
    """Emit ``ReplayWork`` for every fs path then every archive member.

    ``fs_paths`` is a list of ``(open_path, manifest_key)`` pairs; they yield
    ``unlink_after=False``. ``archive_members`` maps each archive Path to the
    set of member names to extract; those stream through ``iter_archive_members``
    one archive at a time and yield ``unlink_after=True``.
    """
    for open_path, manifest_key in fs_paths:
        yield ReplayWork(open_path=open_path, manifest_key=manifest_key, unlink_after=False)
    if archive_members is None:
        return
    for archive, members in archive_members.items():
        for synthetic, tmpfs_path in iter_archive_members(
            archive,
            tmpfs_root=tmpfs_root,
            filter_paths=set(members),
            queue_size=queue_size,
        ):
            yield ReplayWork(open_path=tmpfs_path, manifest_key=synthetic, unlink_after=True)


class _TmpfsWriter(Py7zIO):
    """Writes one decompressed member to a unique tmpfs file."""

    def __init__(self, member: str, path: Path, out_q: queue.Queue) -> None:
        self.member: str = member
        self._out_q: queue.Queue = out_q
        self.path: Path = path
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
    """Per-extract factory that serializes finalize calls within each thread.

    Acquires a slot on the bounded semaphore *before* opening each per-file
    writer so backpressure happens before /dev/shm fills. The slot is
    released by the consumer after iteration (success or failure), or by
    this factory if writer construction fails.
    """

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
        self._stopped: bool = False
        # Monotonic per-process counter, incremented under _lock so the
        # filename suffix is unique across producer threads regardless of
        # whether the GIL is enabled (py3.14 free-threading safe).
        self._counter: int = 0

    def request_stop(self) -> None:
        """Tell create() to refuse all further real writers (NullIO instead).

        Used when the consumer aborts early: still drain the producer thread
        cleanly, but don't materialize any more files.
        """
        with self._lock:
            self._stopped = True

    def _next_path(self) -> Path:
        with self._lock:
            seq = self._counter
            self._counter += 1
        return self._tmpfs_root / f"{os.getpid()}_{threading.get_ident()}_{seq}.slp"

    def create(self, filename: str) -> Py7zIO:
        tid = threading.get_ident()
        with self._lock:
            prev = self._open_per_thread.get(tid)
            stopped = self._stopped
        if isinstance(prev, _TmpfsWriter):
            prev.finalize()

        skip = stopped or (self._filter_paths is not None and filename not in self._filter_paths)
        if skip:
            new: Py7zIO = NullIO()
        else:
            self._sem.acquire()
            try:
                new = _TmpfsWriter(filename, self._next_path(), self._out_q)
            except BaseException:
                self._sem.release()
                raise

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


def _bounded_pool_extract(
    self: Any,
    fp: Any,
    path: Any,
    parallel: bool,  # noqa: ARG001 (kept for signature compatibility)
    skip_notarget: bool = True,
    q: queue.Queue | None = None,
) -> None:
    """Drop-in replacement for py7zr.Worker.extract with bounded concurrency.

    The shipped implementation (py7zr 1.1.0, py7zr.py:1316-1342) spawns one
    Thread per folder simultaneously, each calling open(filename, "rb").
    For an archive with tens of thousands of folders this leaks fds linearly
    (the per-thread fp is never closed) and exhausts RLIMIT_NOFILE.

    Here we cap concurrency via a ThreadPoolExecutor and reuse one archive
    fd per worker thread — total extra fds = max_workers, constant in member
    count.
    """
    if not (hasattr(self.header, "main_streams") and self.header.main_streams is not None):
        empty = [f for f in self.files if f.emptystream]
        self.extract_single(fp, empty, path, 0, 0, q)
        return

    src_end = self.src_start + self.header.main_streams.packinfo.packpositions[-1]
    numfolders = self.header.main_streams.unpackinfo.numfolders
    if numfolders == 1:
        self.extract_single(fp, self.files, path, self.src_start, src_end, q, skip_notarget=skip_notarget)
        return

    folders = self.header.main_streams.unpackinfo.folders
    positions = self.header.main_streams.packinfo.packpositions
    empty = [f for f in self.files if f.emptystream]
    self.extract_single(fp, empty, path, 0, 0, q)

    targeted = [
        i
        for i in range(numfolders)
        if not skip_notarget or any(self.target_filepath.get(f.id, None) for f in folders[i].files)
    ]
    if not targeted:
        return

    filename = getattr(fp, "name", None)
    if filename is None:
        raise InternalError("bounded extract requires fp with a .name (path)")

    max_workers = min(len(targeted), _BOUNDED_EXTRACT_THREADS)
    if max_workers <= 1:
        for i in targeted:
            self.extract_single(
                fp,
                folders[i].files,
                path,
                self.src_start + positions[i],
                self.src_start + positions[i + 1],
                q,
                skip_notarget=skip_notarget,
            )
        return

    local = threading.local()
    open_fps: list = []
    open_fps_lock = threading.Lock()

    def _worker_fp() -> Any:
        wfp = getattr(local, "fp", None)
        if wfp is None:
            wfp = open(filename, "rb")  # noqa: SIM115 — fp is per-thread and reused across folders; closed in finally
            local.fp = wfp
            with open_fps_lock:
                open_fps.append(wfp)
        return wfp

    def _do_folder(i: int) -> None:
        self.extract_single(
            _worker_fp(),
            folders[i].files,
            path,
            self.src_start + positions[i],
            self.src_start + positions[i + 1],
            q,
            skip_notarget=skip_notarget,
        )

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_do_folder, i) for i in targeted]
            for f in concurrent.futures.as_completed(futures):
                f.result()
    finally:
        with open_fps_lock:
            for wfp in open_fps:
                with contextlib.suppress(OSError):
                    wfp.close()


_BOUNDED_EXTRACT_THREADS: int = max(2, min(8, (os.cpu_count() or 4)))


def _maybe_start_fd_watcher() -> tuple[threading.Event | None, threading.Thread | None]:
    """Start a per-2s fd-count logger when HAL_PROFILE_FDS=1, else no-op.

    Diagnostic for fd leaks in archive extraction; left in tree because
    py7zr's threading model is fragile and any future regression here
    would otherwise be opaque.
    """
    if os.environ.get("HAL_PROFILE_FDS") != "1":
        return None, None

    stop = threading.Event()
    fd_dir = Path(f"/proc/{os.getpid()}/fd")

    def _run() -> None:
        while not stop.wait(2.0):
            try:
                entries = list(fd_dir.iterdir())
            except OSError as e:
                logger.warning(f"fd watcher: cannot list {fd_dir}: {e!r}")
                continue
            buckets: Counter[str] = Counter()
            for e in entries:
                try:
                    target = os.readlink(e)
                except OSError:
                    target = "<gone>"
                if target.startswith("/dev/shm"):
                    bucket = "/dev/shm/*"
                elif target.startswith("/proc"):
                    bucket = "/proc/*"
                elif "pipe:" in target:
                    bucket = "pipe:*"
                elif "socket:" in target:
                    bucket = "socket:*"
                elif "anon_inode:" in target:
                    bucket = f"anon_inode:{target.split(':', 1)[1].split('[')[0]}"
                elif target.endswith(".7z"):
                    bucket = "*.7z"
                else:
                    bucket = target
                buckets[bucket] += 1
            logger.debug(f"fd watcher pid={os.getpid()}: total={len(entries)} top={buckets.most_common(8)}")

    t = threading.Thread(target=_run, name="fd-watcher", daemon=True)
    t.start()
    return stop, t


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

    Early consumer abort (``break``, ``GeneratorExit``, exception) drains
    the producer cleanly: the factory stops materializing new files, the
    queue is flushed, and any already-yielded tmpfs files are unlinked.
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
                # Replace py7zr's broken parallel extract (one Thread per
                # folder, each opening a fresh fd that is never closed) with
                # a bounded thread pool that reuses fds. See _bounded_pool_extract.
                z.worker.extract = types.MethodType(_bounded_pool_extract, z.worker)
                z.extract(factory=factory)
        except BaseException as e:
            logger.error(f"archive producer crashed on {archive}: {e!r}")
            producer_exc.append(e)
        finally:
            factory.finalize_all()
            out_q.put(_SENTINEL)

    producer = threading.Thread(target=_producer, name=f"py7zr-producer-{archive.name}", daemon=True)
    producer.start()

    fd_watcher_stop, watcher = _maybe_start_fd_watcher()

    seen_members: set[str] = set()
    drained = False
    try:
        while True:
            item = out_q.get()
            if item is _SENTINEL:
                drained = True
                break
            member, tmpfs_path = item
            seen_members.add(member)
            synthetic = archive_member_path(archive, member)
            try:
                yield synthetic, tmpfs_path
            finally:
                # Release one queue slot whether the consumer succeeded or not.
                # Caller is responsible for unlinking tmpfs_path.
                sem.release()
    finally:
        # If we didn't reach the sentinel, the consumer aborted early and the
        # producer is still extracting (potentially blocked on sem.acquire()).
        # Tell it to NullIO the rest, drain the queue releasing slots, and
        # unlink any leftover tmpfs files — otherwise producer.join() deadlocks.
        if not drained:
            factory.request_stop()
            while True:
                item = out_q.get()
                if item is _SENTINEL:
                    break
                _, leftover = item
                Path(leftover).unlink(missing_ok=True)
                sem.release()
        producer.join()
        if watcher is not None:
            assert fd_watcher_stop is not None
            fd_watcher_stop.set()
            watcher.join(timeout=3.0)

    if producer_exc:
        raise producer_exc[0]

    # Drained cleanly. If the caller filtered to a specific member set, surface
    # any entries that the archive did not contain — without this they're a
    # silent absence (caller asked for {A, B}, got just {A}, never knew).
    if drained and filter_paths is not None:
        missing = filter_paths - seen_members
        if missing:
            preview = sorted(missing)[:5]
            logger.warning(
                f"{archive.name}: {len(missing)}/{len(filter_paths)} requested members not in archive "
                f"(first few: {preview})"
            )
