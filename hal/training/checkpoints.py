"""Background checkpoint sync to R2.

Training writes checkpoints to a local run dir and hands each file to a
``BackgroundUploader`` that PUTs it to ``r2://<bucket>/<prefix>/<run>/<file>``
off the hot path (a daemon worker drains a queue), so a slow upload never
stalls a training step. ``download_latest`` pulls the newest checkpoint back to
resume after a preemption.

Checkpoints are mutable run *outputs* — unlike the immutable, sha-pinned
*inputs* in ``hal.fixtures`` — so they deliberately bypass the ``Fixture`` /
``fetch`` machinery. Both share one R2 client (``hal.r2``).
"""

import queue
import threading
from pathlib import Path
from typing import Final

from botocore.exceptions import BotoCoreError
from botocore.exceptions import ClientError
from loguru import logger

from hal import r2

_SENTINEL: Final[object] = object()
_NOT_FOUND: Final[frozenset[str]] = frozenset({"404", "NoSuchKey"})


class BackgroundUploader:
    """Async R2 uploader. A single daemon thread drains a queue of local paths,
    PUTting each under ``<prefix>/<run_name>/``. ``close()`` blocks until the
    queue is drained. Credentials are validated eagerly at construction so a
    misconfigured run fails loud before training starts, not silently mid-run.
    """

    def __init__(self, run_name: str, *, prefix: str = "runs") -> None:
        self._run_name = run_name
        self._prefix = prefix
        self._bucket = r2.bucket()
        self._client = r2.client()
        self._queue: queue.Queue = queue.Queue()
        self._failures = 0
        self._thread = threading.Thread(target=self._drain, name=f"r2-upload-{run_name}", daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _SENTINEL:
                    return
                local = Path(item)
                key = f"{self._prefix}/{self._run_name}/{local.name}"
                try:
                    self._client.upload_file(str(local), self._bucket, key)
                    logger.info(f"[ckpt] uploaded {local.name} -> r2://{self._bucket}/{key}")
                except (OSError, BotoCoreError, ClientError) as e:
                    self._failures += 1
                    logger.error(f"[ckpt] upload failed for {local.name}: {e}")
            finally:
                self._queue.task_done()

    def upload(self, path: Path) -> None:
        """Enqueue ``path`` for upload. Returns immediately (non-blocking)."""
        self._queue.put(str(path))

    def close(self) -> None:
        """Drain the queue and join the worker. Warns if any upload failed."""
        self._queue.put(_SENTINEL)
        self._thread.join()
        if self._failures:
            logger.warning(f"[ckpt] {self._failures} checkpoint upload(s) failed this run")


def download_latest(run_name: str, dest_dir: Path, *, name: str = "latest.pt", prefix: str = "runs") -> Path | None:
    """Pull ``<prefix>/<run_name>/<name>`` from R2 into ``dest_dir``.

    Returns the local path, or ``None`` if the object doesn't exist (fresh run).
    """
    client = r2.client()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    try:
        client.download_file(r2.bucket(), f"{prefix}/{run_name}/{name}", str(dest))
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in _NOT_FOUND:
            return None
        raise
    return dest
