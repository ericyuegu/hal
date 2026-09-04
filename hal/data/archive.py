"""Stream .slp members directly out of a public-dump replay archive into /dev/shm.

The dump ships in two mutually-incompatible on-disk layouts and both must work:

1. **Solid 7z of raw ``.slp``** — ``dev.7z`` and ranked-anonymized chunks 1-2.
   ~10x compressed (LZMA2), >100 GB per chunk. Decompression is folder-parallel
   under py7zr; we drive it via a custom ``WriterFactory`` so members stream
   to per-file tmpfs writers, with two non-obvious py7zr workarounds:

   a. py7zr never calls ``close()`` on the writer it receives from
      ``factory.create()`` — its internal ``MemIO`` wrapper has a no-op
      ``__exit__``. The signal "writer N is done" is implicit: it arrives
      when ``factory.create()`` is called for the *next* file in the same
      thread (folders extract files sequentially within a thread). We
      finalize the previous per-thread writer at that point and explicitly
      finalize the last writer at each folder boundary. The latter matters
      when a thread finishes its final folder while the bounded queue is full.

   b. Backpressure has to happen *before* a tmpfs file is opened, not after,
      or a slow consumer lets the producer fill /dev/shm. The factory
      acquires a bounded semaphore *before* constructing each per-file
      writer, and the consumer releases the slot after it's done with the
      path.

2. **ZIP-of-``.slp.gz``** — ranked-anonymized chunks 3+ (despite the ``.7z``
   extension upstream chose). Per-file gzip inside a zip aggregator container;
   ~25% less dense than solid 7z but trivially random-access. We stream one
   member at a time via stdlib ``zipfile`` + ``gzip``, no producer thread —
   the consumer's hold on the generator is sufficient backpressure.

``iter_archive_members`` sniffs by magic bytes (extension is unreliable) and
dispatches. Both paths yield ``(synthetic_path, tmpfs_path)`` with the same
ownership contract; the materialized tmpfs file is always raw ``.slp``
regardless of source — the ``.gz`` layer is stripped on the zip path.

The 7z path is validated against py7zr 1.1.0 and replaces its private worker
API. Synthetic archive and backpressure tests reproduce the resource failures.
Remove these workarounds only when those tests pass against a newer py7zr.
The ``.slpz`` path accepts only the 1.3.0 CLI used to validate decoding.
"""

import atexit
import concurrent.futures
import contextlib
import functools
import gzip
import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import types
import zipfile
from collections import Counter
from collections import deque
from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import py7zr
from loguru import logger
from py7zr.exceptions import InternalError
from py7zr.io import NullIO
from py7zr.io import Py7zIO
from py7zr.io import WriterFactory

from hal.paths import REPO_DIR
from hal.paths import repo_relative

_ZIP_MAGIC: bytes = b"PK\x03\x04"
_7Z_MAGIC: bytes = b"7z\xbc\xaf\x27\x1c"
_SLPZ_BIN_ENV: str = "HAL_SLPZ_BIN"
_PINNED_SLPZ_BIN: Path = Path(REPO_DIR) / "data" / "tools" / "slpz-1.3.0" / "bin" / "slpz"
_SUPPORTED_SLPZ_VERSION = "1.3.0"
_SUPPORTED_PY7ZR_VERSION = "1.1.0"


def _require_supported_py7zr() -> None:
    """Reject a py7zr version that has not passed HAL's private-API tests."""
    installed = version("py7zr")
    if installed != _SUPPORTED_PY7ZR_VERSION:
        raise RuntimeError(f"HAL's archive workarounds require py7zr=={_SUPPORTED_PY7ZR_VERSION}; found {installed}")


def _sniff_archive_format(archive: Path) -> str:
    """Return ``"7z"`` or ``"zip"`` based on magic bytes; raise on anything else.

    Cannot trust the extension: the upstream Slippi public dump labels its
    ZIP chunks as ``.7z`` (see chunks 3+ of ``ranked-anonymized-*``).
    """
    with archive.open("rb") as f:
        head = f.read(6)
    if head.startswith(_7Z_MAGIC):
        return "7z"
    if head.startswith(_ZIP_MAGIC):
        return "zip"
    raise ValueError(f"unrecognized archive magic for {archive}: {head!r}")


def list_archive_slps(archive: Path) -> list[str]:
    """Cheap (header-only) list of slp member names, exactly as stored.

    Names end in ``.slp`` for solid-7z chunks and ``.slp.gz`` for
    zip-of-gzipped chunks. The name is what ``iter_archive_members`` and
    ``read_archive_member_to_file`` expect as the archive-internal member key.
    """
    fmt = _sniff_archive_format(archive)
    if fmt == "7z":
        _require_supported_py7zr()
        with py7zr.SevenZipFile(str(archive), "r") as z:
            return [n for n in z.getnames() if n.endswith(".slp")]
    with zipfile.ZipFile(archive) as z:
        return [n for n in z.namelist() if n.endswith(".slp") or n.endswith(".slp.gz") or n.endswith(".slpz")]


@functools.cache
def _slpz_binary() -> Path:
    configured = os.environ.get(_SLPZ_BIN_ENV)
    if configured:
        return _validated_slpz_binary(Path(configured))
    if _PINNED_SLPZ_BIN.is_file():
        return _validated_slpz_binary(_PINNED_SLPZ_BIN)
    on_path = shutil.which("slpz")
    if on_path:
        return _validated_slpz_binary(Path(on_path))
    raise FileNotFoundError(
        "a .slpz member requires slpz 1.3.0; install it with "
        f"`cargo install slpz --version 1.3.0 --locked --root {_PINNED_SLPZ_BIN.parents[1]}` "
        f"or set {_SLPZ_BIN_ENV}"
    )


def _validated_slpz_binary(binary: Path) -> Path:
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"{_SLPZ_BIN_ENV} does not name an executable file: {binary}")
    installed = _cargo_binary_version(binary, "slpz")
    if installed != _SUPPORTED_SLPZ_VERSION:
        raise RuntimeError(
            f"slpz must be version {_SUPPORTED_SLPZ_VERSION}; {binary} has Cargo metadata version {installed!r}"
        )
    return binary


def _cargo_binary_version(binary: Path, package: str) -> str | None:
    """Read the version recorded by ``cargo install --root``.

    slpz 1.3.0 reports its CLI version as ``0``, so its own ``--version``
    output cannot validate the package version.
    """
    metadata_path = binary.parent.parent / ".crates2.json"
    try:
        installs = json.loads(metadata_path.read_text())["installs"]
    except FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError:
        return None
    if not isinstance(installs, dict):
        return None
    prefix = f"{package} "
    for identity, details in installs.items():
        if not isinstance(identity, str) or not identity.startswith(prefix) or not isinstance(details, dict):
            continue
        if binary.name in details.get("bins", ()):
            return identity.removeprefix(prefix).split(" ", 1)[0]
    return None


def _decode_slpz_member(raw: Any, out_path: Path) -> None:
    """Decode one ZIP member through the pinned CLI without an input copy."""
    with out_path.open("wb") as out, tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            [str(_slpz_binary()), "-d", "-q", "-o", "-", "-"],
            stdin=subprocess.PIPE,
            stdout=out,
            stderr=errors,
        )
        assert process.stdin is not None
        try:
            shutil.copyfileobj(raw, process.stdin, length=1 << 20)
        except BrokenPipeError:
            pass
        finally:
            process.stdin.close()
        returncode = process.wait()
        errors.seek(0)
        stderr = errors.read().decode("utf-8", errors="replace").strip()
    if returncode or not out_path.stat().st_size:
        out_path.unlink(missing_ok=True)
        raise RuntimeError(f"slpz decode failed (exit={returncode}): {stderr or 'empty output'}")


def _extract_zip_member_to(z: zipfile.ZipFile, member: str, out_path: Path) -> None:
    """Stream one zip member into ``out_path``, decompressing the .gz wrapper if present."""
    with z.open(member) as raw:
        if member.endswith(".slpz"):
            _decode_slpz_member(raw, out_path)
            return
        src = gzip.GzipFile(fileobj=raw, mode="rb") if member.endswith(".gz") else raw
        try:
            with out_path.open("wb") as out:
                shutil.copyfileobj(src, out, length=1 << 20)
        finally:
            if src is not raw:
                src.close()


def read_archive_member_to_file(archive: Path, member: str, dest_dir: Path) -> Path:
    """Materialize one archive member to ``dest_dir`` and return the written path.

    Strips the ``.gz`` suffix from zip-of-gzipped members so the consumer
    always sees a raw ``.slp`` file regardless of source format.
    """
    fmt = _sniff_archive_format(archive)
    if fmt == "7z":
        with py7zr.SevenZipFile(str(archive), "r") as z:
            z.extract(path=str(dest_dir), targets=[member])
        extracted = dest_dir / member
        if not extracted.is_file():
            raise FileNotFoundError(f"member {member!r} not in {archive}")
        return extracted
    name = Path(member).name
    out_name = f"{name[:-5]}.slp" if name.endswith(".slpz") else name.removesuffix(".gz")
    out_path = dest_dir / out_name
    with zipfile.ZipFile(archive) as z:
        try:
            _extract_zip_member_to(z, member, out_path)
        except KeyError as e:
            raise FileNotFoundError(f"member {member!r} not in {archive}") from e
    return out_path


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
    open_error: str | None = None


def _record_archive_error(
    errors: deque[ReplayWork],
    tmpfs_root: Path,
    manifest_key: str,
    error: BaseException,
) -> None:
    errors.append(
        ReplayWork(
            open_path=tmpfs_root / "archive-read-error.slp",
            manifest_key=manifest_key,
            unlink_after=False,
            open_error=repr(error),
        )
    )


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
        errors: deque[ReplayWork] = deque()
        record_error = functools.partial(_record_archive_error, errors, tmpfs_root)

        for synthetic, tmpfs_path in iter_archive_members(
            archive,
            tmpfs_root=tmpfs_root,
            filter_paths=set(members),
            queue_size=queue_size,
            on_error=record_error,
        ):
            while errors:
                yield errors.popleft()
            yield ReplayWork(open_path=tmpfs_path, manifest_key=synthetic, unlink_after=True)
        while errors:
            yield errors.popleft()


class _TmpfsWriter(Py7zIO):
    """Writes one decompressed member to a unique tmpfs file."""

    def __init__(self, member: str, path: Path, out_q: queue.Queue) -> None:
        self.member: str = member
        self._out_q: queue.Queue = out_q
        self.path: Path = path
        self._fp = self.path.open("wb")
        self._size: int = 0
        self._finalized: bool = False
        self._lock = threading.Lock()

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
        with self._lock:
            if self._finalized:
                return
            self._finalized = True
            self._fp.close()
        self._out_q.put((self.member, self.path))

    def abort(self) -> bool:
        """Close and remove this writer without handing it to the consumer."""
        with self._lock:
            if self._finalized:
                return False
            self._finalized = True
            self._fp.close()
            self.path.unlink(missing_ok=True)
        return True


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

    def abort_all(self) -> None:
        """Stop new writers and release slots held by unfinished writers.

        Closing an active writer makes its extraction folder fail promptly.
        The producer catches that failure and emits its sentinel, which lets
        an interrupted consumer exit instead of waiting on the semaphore.
        """
        with self._lock:
            self._stopped = True
            writers = list(self._open_per_thread.values())
            self._open_per_thread.clear()
        for writer in writers:
            if isinstance(writer, _TmpfsWriter) and writer.abort():
                self._sem.release()

    def finalize_thread(self) -> None:
        """Finalize the current extraction thread's last open writer."""
        tid = threading.get_ident()
        with self._lock:
            writer = self._open_per_thread.pop(tid, None)
        if isinstance(writer, _TmpfsWriter):
            writer.finalize()

    def create(self, filename: str) -> Py7zIO:
        tid = threading.get_ident()
        with self._lock:
            prev = self._open_per_thread.pop(tid, None)
            stopped = self._stopped
        if isinstance(prev, _TmpfsWriter):
            prev.finalize()

        skip = stopped or (self._filter_paths is not None and filename not in self._filter_paths)
        if skip:
            return NullIO()

        self._sem.acquire()
        with self._lock:
            if self._stopped:
                self._sem.release()
                return NullIO()
            seq = self._counter
            self._counter += 1
            path = self._tmpfs_root / f"{os.getpid()}_{tid}_{seq}.slp"
            try:
                new = _TmpfsWriter(filename, path, self._out_q)
            except BaseException:
                self._sem.release()
                raise
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
    *,
    factory: _StreamFactory,
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
        try:
            self.extract_single(fp, empty, path, 0, 0, q)
        finally:
            factory.finalize_thread()
        return

    src_end = self.src_start + self.header.main_streams.packinfo.packpositions[-1]
    numfolders = self.header.main_streams.unpackinfo.numfolders
    if numfolders == 1:
        try:
            self.extract_single(fp, self.files, path, self.src_start, src_end, q, skip_notarget=skip_notarget)
        finally:
            factory.finalize_thread()
        return

    folders = self.header.main_streams.unpackinfo.folders
    positions = self.header.main_streams.packinfo.packpositions
    empty = [f for f in self.files if f.emptystream]
    try:
        self.extract_single(fp, empty, path, 0, 0, q)
    finally:
        factory.finalize_thread()

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
            try:
                self.extract_single(
                    fp,
                    folders[i].files,
                    path,
                    self.src_start + positions[i],
                    self.src_start + positions[i + 1],
                    q,
                    skip_notarget=skip_notarget,
                )
            finally:
                factory.finalize_thread()
                folders[i].decompressor = None
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
        # py7zr caches each folder's LZMA decompressor on the Folder object
        # (archiveinfo.Folder.get_decompressor) and never frees it. Across
        # 10k+ folders that's tens of GB of dict buffers — RSS climbs until
        # the process swaps and throughput collapses. Drop the reference
        # here so the decompressor is collectable as soon as the folder
        # finishes.
        try:
            self.extract_single(
                _worker_fp(),
                folders[i].files,
                path,
                self.src_start + positions[i],
                self.src_start + positions[i + 1],
                q,
                skip_notarget=skip_notarget,
            )
        finally:
            factory.finalize_thread()
            folders[i].decompressor = None

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


def _is_dead_pid(pid: int) -> bool:
    """True iff the kernel has no process with this pid. EPERM (live, foreign user) is treated as alive."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


def _run_tmpfs_dir(tmpfs_root: Path) -> Path:
    """Create and return ``tmpfs_root/<my-pid>/``, reaping dead-PID siblings first.

    Three layers of defense against leaked tmpfs materializations:
    workers ``unlink()`` files as peppi finishes with them; an ``atexit`` hook
    removes our run_dir on normal shutdown (catches files held by SIGKILL'd
    workers in the caller's Pool); and this startup sweep removes sibling
    subdirs whose owning PID is gone (catches leaks from prior parent
    SIGKILL/OOM — the only path that bypasses atexit).
    """
    tmpfs_root.mkdir(parents=True, exist_ok=True)
    reaped = 0
    reclaimed = 0
    for entry in tmpfs_root.iterdir():
        if not entry.is_dir():
            continue
        try:
            owner_pid = int(entry.name)
        except ValueError:
            continue
        if not _is_dead_pid(owner_pid):
            continue
        for f in entry.iterdir():
            try:
                reclaimed += f.stat().st_size
            except OSError:
                continue
        shutil.rmtree(entry, ignore_errors=True)
        reaped += 1
    if reaped:
        logger.info(f"swept {reaped} stranded tmpfs dir(s) under {tmpfs_root}, reclaimed {reclaimed / 1e9:.2f} GB")
    run_dir = tmpfs_root / str(os.getpid())
    run_dir.mkdir(exist_ok=True)
    atexit.register(shutil.rmtree, run_dir, ignore_errors=True)
    return run_dir


def _archive_tmpfs_dir(tmpfs_root: Path) -> Path:
    """Return a unique directory for one archive extraction.

    ``iter_replay_work`` may advance to the next archive while process-pool
    workers still own files yielded from the previous one.  A per-archive
    namespace prevents each extractor's local sequence counter from
    overwriting or unlinking those in-flight files.
    """
    return Path(tempfile.mkdtemp(dir=_run_tmpfs_dir(tmpfs_root), prefix="archive-"))


def iter_archive_members(
    archive: Path,
    *,
    tmpfs_root: Path,
    filter_paths: set[str] | None = None,
    queue_size: int = 64,
    on_error: Callable[[str, BaseException], None] | None = None,
) -> Iterator[tuple[str, Path]]:
    """Yield ``(synthetic_path, tmpfs_path)`` for each slp member of ``archive``.

    Dispatches by magic bytes: solid 7z chunks go through the threaded
    py7zr producer (``_iter_7z_members``), zip-of-gzipped chunks through
    the stdlib serial reader (``_iter_zip_members``). The materialized
    tmpfs file is always raw ``.slp`` regardless of source — the ``.gz``
    layer is stripped on the zip path. Synthetic paths preserve the
    archive-internal member name (so zip members appear as ``.slp.gz`` in
    the synthetic path, matching ``list_archive_slps``).

    The tmpfs file is owned by the consumer once yielded: the consumer MUST
    unlink it (success or failure) so the producer can proceed.

    ``filter_paths`` is a set of *member* names — for 7z that's the
    ``.slp`` name as stored; for zip-of-gz that's the ``.slp.gz`` name.

    Iteration order: for 7z, "as files complete decompression" (roughly
    archive order interleaved across solid blocks — do not rely on strict
    order). For zip, archive order, member-by-member.

    Early consumer abort (``break``, ``GeneratorExit``, exception) drains
    the producer cleanly on both paths.
    """
    if not archive.exists():
        raise FileNotFoundError(f"archive not found: {archive}")
    fmt = _sniff_archive_format(archive)
    if fmt == "7z":
        yield from _iter_7z_members(archive, tmpfs_root=tmpfs_root, filter_paths=filter_paths, queue_size=queue_size)
    else:
        yield from _iter_zip_members(
            archive,
            tmpfs_root=tmpfs_root,
            filter_paths=filter_paths,
            on_error=on_error,
        )


def _iter_7z_members(
    archive: Path,
    *,
    tmpfs_root: Path,
    filter_paths: set[str] | None,
    queue_size: int,
) -> Iterator[tuple[str, Path]]:
    """Threaded py7zr producer for solid-7z archives. See ``iter_archive_members``.

    Slots in the bounded semaphore refill via ``sem.release()`` after the
    consumer's iteration body runs, so slow consumers backpressure the
    producer instead of filling /dev/shm. Excluded files (not in
    ``filter_paths``) are still decompressed (unavoidable in a solid block)
    but discarded into a ``NullIO`` instead of materializing.
    """
    _require_supported_py7zr()
    run_dir = _archive_tmpfs_dir(tmpfs_root)

    out_q: queue.Queue = queue.Queue()
    sem = threading.Semaphore(queue_size)
    factory = _StreamFactory(run_dir, out_q, sem, filter_paths)
    producer_exc: list[BaseException] = []

    def _producer() -> None:
        try:
            with py7zr.SevenZipFile(str(archive), "r") as z:
                # Replace py7zr's broken parallel extract (one Thread per
                # folder, each opening a fresh fd that is never closed) with
                # a bounded thread pool that reuses fds. See _bounded_pool_extract.
                extract = functools.partial(_bounded_pool_extract, factory=factory)
                z.worker.extract = types.MethodType(extract, z.worker)
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
            factory.abort_all()
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                try:
                    item = out_q.get(timeout=1.0)
                except queue.Empty:
                    if not producer.is_alive():
                        break
                    continue
                if item is _SENTINEL:
                    break
                _, leftover = item
                Path(leftover).unlink(missing_ok=True)
                sem.release()
            producer.join(timeout=max(0.0, deadline - time.monotonic()))
            if producer.is_alive():
                logger.warning(f"archive producer did not stop within 30 seconds: {archive}")
        else:
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
            raise FileNotFoundError(
                f"{archive.name}: {len(missing)}/{len(filter_paths)} requested members are not in the archive "
                f"(first few: {preview})"
            )


def _iter_zip_members(
    archive: Path,
    *,
    tmpfs_root: Path,
    filter_paths: set[str] | None,
    on_error: Callable[[str, BaseException], None] | None,
) -> Iterator[tuple[str, Path]]:
    """Serial reader for ZIP members in `.slp`, `.slp.gz`, or `.slpz` form.

    No producer thread: per-member compression makes random-access cheap and
    the consumer's hold on the generator keeps at most one materialized
    file in tmpfs at a time, so no semaphore is needed. The ``.gz`` layer
    is decompressed inline; output tmpfs files are always raw ``.slp``.
    """
    run_dir = _archive_tmpfs_dir(tmpfs_root)
    with zipfile.ZipFile(archive) as z:
        members = [n for n in z.namelist() if n.endswith(".slp") or n.endswith(".slp.gz") or n.endswith(".slpz")]
        if filter_paths is not None:
            wanted = set(filter_paths)
            members = [m for m in members if m in wanted]
        seen: set[str] = set()
        for seq, member in enumerate(members):
            tmpfs_path = run_dir / f"{os.getpid()}_zip_{seq}.slp"
            seen.add(member)
            synthetic = archive_member_path(archive, member)
            try:
                _extract_zip_member_to(z, member, tmpfs_path)
            except KeyboardInterrupt, SystemExit:
                raise
            except BaseException as error:
                tmpfs_path.unlink(missing_ok=True)
                if on_error is None:
                    raise
                logger.warning(f"cannot read {synthetic}: {error!r}")
                on_error(synthetic, error)
                continue
            yield synthetic, tmpfs_path
    if filter_paths is not None:
        missing = set(filter_paths) - seen
        if missing:
            preview = sorted(missing)[:5]
            raise FileNotFoundError(
                f"{archive.name}: {len(missing)}/{len(filter_paths)} requested members are not in the archive "
                f"(first few: {preview})"
            )
