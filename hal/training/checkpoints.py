"""Upload checkpoints and evaluation files to R2."""

import hashlib
import json
import os
import queue
import re
import threading
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any
from typing import Final
from typing import cast

import torch
from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import BotoCoreError
from botocore.exceptions import ClientError
from loguru import logger

from hal import r2

_NOT_FOUND: Final[frozenset[str]] = frozenset({"404", "NoSuchKey"})
_FileVersion = tuple[str, int, int, int, int, int]
_UploadItem = tuple[str, str | None]


@dataclass(frozen=True, slots=True)
class ResumeLineage:
    parent_checkpoint_sha256: str
    old_source_sha: str
    new_source_sha: str
    parity_report_sha256: str

    def __post_init__(self) -> None:
        for name, value, length in (
            ("parent checkpoint", self.parent_checkpoint_sha256, 64),
            ("old source", self.old_source_sha, 40),
            ("new source", self.new_source_sha, 40),
            ("parity report", self.parity_report_sha256, 64),
        ):
            if not isinstance(value, str) or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
                raise ValueError(f"invalid resume lineage {name} identity")
        if self.old_source_sha == self.new_source_sha:
            raise ValueError("resume lineage must declare a source change")

    @classmethod
    def from_record(cls, record: object) -> ResumeLineage:
        expected = {"parent_checkpoint_sha256", "old_source_sha", "new_source_sha", "parity_report_sha256"}
        if not isinstance(record, dict) or set(record) != expected:
            raise ValueError("resume lineage fields changed")
        values = cast(dict[str, str], record)
        return cls(
            values["parent_checkpoint_sha256"],
            values["old_source_sha"],
            values["new_source_sha"],
            values["parity_report_sha256"],
        )

    def to_record(self) -> dict[str, str]:
        return asdict(self)


def read_resume_lineage(path: Path) -> ResumeLineage:
    return ResumeLineage.from_record(json.loads(path.read_text()))


def checkpoint_resume_lineage(record: object) -> tuple[ResumeLineage, ...]:
    if record is None:
        return ()
    if not isinstance(record, list):
        raise ValueError("checkpoint resume lineage must be an ordered list")
    lineage = tuple(ResumeLineage.from_record(item) for item in record)
    for previous, current in pairwise(lineage):
        if previous.new_source_sha != current.old_source_sha:
            raise ValueError("checkpoint resume lineage source transitions are not contiguous")
    return lineage


def validate_resume_provenance(
    stored: object,
    current: Mapping[str, object],
    *,
    transition: ResumeLineage | None = None,
    parent_checkpoint_sha256: str | None = None,
) -> None:
    if not isinstance(stored, Mapping):
        raise ValueError("resume checkpoint has no provenance record")
    previous = cast(Mapping[str, object], stored)
    if set(previous) != set(current):
        raise ValueError("resume provenance fields changed")
    changed = {name for name, value in current.items() if previous[name] != value}
    if transition is not None:
        if (
            transition.parent_checkpoint_sha256 != parent_checkpoint_sha256
            or transition.old_source_sha != previous.get("git_sha")
            or transition.new_source_sha != current.get("git_sha")
        ):
            raise ValueError("resume lineage does not identify this checkpoint and source transition")
        changed.discard("git_sha")
    if changed:
        raise ValueError(f"resume provenance changed: {sorted(changed)}")


class _Stop:
    """Private queue marker that tells the upload thread to exit."""


_SENTINEL: Final[_Stop] = _Stop()


def checkpoint_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a checkpoint file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def advance_checkpoint_link(source: Path, destination: Path) -> None:
    """Atomically point ``destination`` at an immutable checkpoint inode."""
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    os.replace(temporary, destination)


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

    def wait(self) -> None:
        """Wait for all queued uploads and fail if any upload failed."""
        self._queue.join()
        if self._failures:
            raise RuntimeError(f"{self._failures} R2 upload(s) failed")

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


def load_for_resume(
    run_name: str,
    ckpt_dir: Path,
    *,
    device: str,
    name: str = "latest.pt",
) -> dict[str, Any] | None:
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
    dest = dest_dir / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.download_file(r2.bucket(), f"{prefix}/{run_name}/{name}", str(dest))
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in _NOT_FOUND:
            return None
        raise
    return dest
