"""MDS writer with bounded asynchronous cloud-upload backlog."""

import shutil
import subprocess
from collections import deque
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from streaming import MDSWriter


class BoundedMDSWriter(MDSWriter):
    """Keep at most ``max_pending_uploads`` completed shard files locally.

    Mosaic's writer uses a bounded upload thread pool but an unbounded task
    queue.  A fast encoder can therefore fill local storage while R2 is slow.
    This subclass copies the small upstream ``flush_shard`` seam and waits for
    the oldest upload before admitting another queued shard.
    """

    def __init__(self, *, max_pending_uploads: int = 2, **kwargs: Any) -> None:
        if max_pending_uploads < 1:
            raise ValueError(f"max_pending_uploads must be positive, got {max_pending_uploads}")
        self._max_pending_uploads = max_pending_uploads
        self._pending_uploads: deque[Future[None]] = deque()
        super().__init__(**kwargs)

    def _reap_one(self) -> None:
        future = self._pending_uploads.popleft()
        future.result()
        if self.event.is_set():
            raise RuntimeError("an MDS shard upload failed")

    def flush_shard(self) -> None:
        raw_basename, zip_basename = self._name_next_shard()
        raw_data = self.encode_joint_shard()
        raw_info, zip_info = self._process_file(raw_data, raw_basename, zip_basename)
        shard = {"samples": len(self.new_samples), "raw_data": raw_info, "zip_data": zip_info}
        shard.update(self.get_config())
        self.shards.append(shard)

        future = self.executor.submit(self.cloud_writer.upload_file, zip_basename or raw_basename)
        future.add_done_callback(self.exception_callback)
        self._pending_uploads.append(future)
        if len(self._pending_uploads) >= self._max_pending_uploads:
            self._reap_one()

    def finish(self) -> None:
        if self.event.is_set():
            self.cancel_future_jobs()
        else:
            if self.new_samples:
                self.flush_shard()
                self._reset_cache()
            while self._pending_uploads:
                self._reap_one()
            # Publish index.json only after every shard upload has completed.
            self._write_index()
            self.executor.shutdown(wait=True)
        if self.remote and not self.keep_local:
            shutil.rmtree(self.local, ignore_errors=True)
        if self.event.is_set():
            raise RuntimeError("one of the MDS uploads failed")

    @property
    def local_staging_dir(self) -> Path:
        return Path(self.local)


def rclone_copyto(local: Path, destination: str) -> None:
    result = subprocess.run(
        ["rclone", "copyto", str(local), destination, "--retries", "5", "--low-level-retries", "10"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"rclone upload failed for {destination}: {detail}")


class _RcloneUploader:
    def __init__(self, local: Path, remote: str) -> None:
        self.local = str(local)
        self.remote = remote.rstrip("/")

    def upload_file(self, filename: str) -> None:
        local = Path(self.local) / filename
        destination = f"{self.remote}/{filename}"
        rclone_copyto(local, destination)
        local.unlink()


class RcloneMDSWriter(BoundedMDSWriter):
    """Bounded writer whose files are uploaded through a configured rclone remote."""

    def __init__(self, *, local: Path, remote: str, **kwargs: Any) -> None:
        if not remote.startswith("r2:"):
            raise ValueError(f"RcloneMDSWriter requires an r2: destination, got {remote!r}")
        super().__init__(out=str(local), keep_local=False, **kwargs)
        self.cloud_writer = _RcloneUploader(local, remote)
        self.local = str(local)
        self.remote = remote
