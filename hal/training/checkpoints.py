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
from typing import Any
from typing import Final

import torch
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
                local_str, rel_key = item
                local = Path(local_str)
                key = f"{self._prefix}/{self._run_name}/{rel_key or local.name}"
                try:
                    self._client.upload_file(str(local), self._bucket, key)
                    logger.info(f"[ckpt] uploaded {rel_key or local.name} -> r2://{self._bucket}/{key}")
                except (OSError, BotoCoreError, ClientError) as e:
                    self._failures += 1
                    logger.error(f"[ckpt] upload failed for {local.name}: {e}")
            finally:
                self._queue.task_done()

    def upload(self, path: Path, *, key: str | None = None) -> None:
        """Enqueue ``path`` for upload. Returns immediately (non-blocking). ``key`` is
        the object path under ``<prefix>/<run>/`` — defaults to the basename; pass a
        relative path (e.g. ``replays/step_000050/match_000/g.slp``) to mirror a tree
        instead of flattening to basenames that could collide."""
        self._queue.put((str(path), key))

    def upload_tree(self, root: Path, *, base: Path, pattern: str = "*") -> int:
        """Enqueue every file under ``root`` matching ``pattern``, keyed by its path
        relative to ``base`` (mirroring the tree under ``<prefix>/<run>/``). Returns
        the count enqueued. Used to ship eval ``.slp`` recordings to R2."""
        files = [p for p in sorted(root.rglob(pattern)) if p.is_file()]
        for p in files:
            self.upload(p, key=str(p.relative_to(base)))
        return len(files)

    def close(self) -> None:
        """Drain the queue and join the worker. Warns if any upload failed."""
        self._queue.put(_SENTINEL)
        self._thread.join()
        if self._failures:
            logger.warning(f"[ckpt] {self._failures} checkpoint upload(s) failed this run")


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
) -> None:
    """Write a resumable checkpoint (model + optimizer + scheduler + config +
    wandb id) and, if an uploader is given, enqueue it for R2 sync."""
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "cfg": cfg,
            "wandb_id": wandb_id,
        },
        path,
    )
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
