"""Upload checkpoints and evaluation files to R2."""

import queue
import threading
from pathlib import Path
from typing import Any
from typing import Final

import torch
from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import BotoCoreError
from botocore.exceptions import ClientError
from loguru import logger

from hal import r2

_NOT_FOUND: Final[frozenset[str]] = frozenset({"404", "NoSuchKey"})
_FileVersion = tuple[str, int, int, int, int, int]
_UploadItem = tuple[str, str | None]


class _Stop:
    """Private queue marker that tells the upload thread to exit."""


_SENTINEL: Final[_Stop] = _Stop()


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
        self._queue: queue.Queue[_UploadItem | _Stop] = queue.Queue()
        self._queued_versions: set[_FileVersion] = set()
        self._queue_lock = threading.Lock()
        self._failures = 0
        self._thread = threading.Thread(target=self._drain, name=f"r2-upload-{run_name}", daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if isinstance(item, _Stop):
                    return
                local_str, rel_key = item
                local = Path(local_str)
                key = f"{self._prefix}/{self._run_name}/{rel_key or local.name}"
                try:
                    self._client.upload_file(str(local), self._bucket, key)
                    logger.info(f"[ckpt] uploaded {rel_key or local.name} -> r2://{self._bucket}/{key}")
                except (OSError, BotoCoreError, ClientError, S3UploadFailedError) as e:
                    self._failures += 1
                    logger.error(f"[ckpt] upload failed for {local.name}: {e}")
            finally:
                self._queue.task_done()

    def upload(self, path: Path, *, key: str | None = None) -> bool:
        """Queue a file unless the same version is already queued."""
        stat = path.stat()
        rel_key = key or path.name
        version: _FileVersion = (rel_key, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        with self._queue_lock:
            if version in self._queued_versions:
                return False
            self._queued_versions.add(version)
            self._queue.put((str(path), key))
        return True

    def upload_tree(self, root: Path, *, base: Path, pattern: str = "*") -> int:
        """Queue matching files and return the number of new file versions."""
        files = (path for path in sorted(root.rglob(pattern)) if path.is_file())
        return sum(1 for path in files if self.upload(path, key=str(path.relative_to(base))))

    def close(self) -> None:
        """Drain the queue and fail if any upload failed."""
        self._queue.put(_SENTINEL)
        self._thread.join()
        if self._failures:
            raise RuntimeError(f"{self._failures} R2 upload(s) failed")


def save_checkpoint(
    path: Path,
    *,
    step: int,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler,
    cfg: dict,
    wandb_id: str | None,
    uploader: BackgroundUploader | None = None,
    extra_state: dict[str, Any] | None = None,
) -> None:
    """Write a resumable checkpoint (model + optimizer + scheduler + config +
    wandb id) and, if an uploader is given, enqueue it for R2 sync."""
    state = {
        "step": step,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "cfg": cfg,
        "wandb_id": wandb_id,
    }
    if extra_state is not None:
        overlap = state.keys() & extra_state.keys()
        if overlap:
            raise ValueError(f"extra checkpoint state replaces reserved keys: {sorted(overlap)}")
        state.update(extra_state)
    torch.save(state, path)
    print(f"[ckpt] saved {path}", flush=True)
    if uploader is not None:
        uploader.upload(path)


def load_for_resume(run_name: str, ckpt_dir: Path, *, device: str, name: str = "latest.pt") -> dict[str, Any] | None:
    """Load the resume checkpoint for ``run_name``: prefer the local copy, else
    pull it from R2. Returns the deserialized state dict, or ``None`` if no
    checkpoint exists in either place (fresh run)."""
    local = ckpt_dir / name
    path = local if local.is_file() else download_latest(run_name, ckpt_dir, name=name)
    if path is None:
        return None
    return torch.load(path, map_location=device, weights_only=False)


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
