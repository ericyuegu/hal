"""Compatibility fixes for MosaicML Streaming 0.13.0.

Mosaic 0.13.0 mishandles Python 3.14 shared-memory registration, retains file
descriptors for non-owning prefix probes, and leaves failed shard downloads in
``PREPARING``. The focused tests in ``tests/test_dataloader.py`` reproduce all
three failures. Remove a patch only after those tests pass against an upgraded
Mosaic release.

This module replaces private APIs. ``patch_streaming`` therefore rejects every
version except the one validated here and applies the replacements once.
"""

import os
import tempfile
from importlib.metadata import version
from multiprocessing import resource_tracker
from typing import Any

import streaming.base.constant as streaming_constants
import streaming.base.dataset as streaming_dataset
import streaming.base.shared.memory as streaming_memory
import streaming.base.shared.prefix as streaming_prefix

_ORIGINAL_PREPARE_SHARD_ATTR = "_hal_original_prepare_shard"
if not hasattr(streaming_dataset.StreamingDataset, _ORIGINAL_PREPARE_SHARD_ATTR):
    setattr(
        streaming_dataset.StreamingDataset,
        _ORIGINAL_PREPARE_SHARD_ATTR,
        streaming_dataset.StreamingDataset.prepare_shard,
    )
_SUPPORTED_STREAMING_VERSION = "0.13.0"
_PATCHED = False


def require_supported_streaming() -> None:
    """Reject a Mosaic version that has not passed HAL's private-API tests."""
    installed = version("mosaicml-streaming")
    if installed != _SUPPORTED_STREAMING_VERSION:
        raise RuntimeError(
            "HAL's Mosaic compatibility patches require "
            f"mosaicml-streaming=={_SUPPORTED_STREAMING_VERSION}; found {installed}"
        )


def _register(self: streaming_memory.SharedMemory, name: str, resource_type: str) -> Any:
    if resource_type == "shared_memory":
        return None
    return resource_tracker._resource_tracker.register(name, resource_type)


def _unregister(self: streaming_memory.SharedMemory, name: str, resource_type: str) -> Any:
    if resource_type == "shared_memory":
        return None
    return resource_tracker._resource_tracker.unregister(name, resource_type)


def _check_and_find_without_fd_leak(
    streams_local: list[str],
    streams_remote: list[str | None],
    shm_name: str,
) -> int:
    """Find Streaming's next shared-memory prefix without retaining probes."""
    prefix_int = 0
    for prefix_int in streaming_prefix._each_prefix_int():
        name = streaming_prefix._get_path(prefix_int, shm_name)
        try:
            filelock_exists = any(
                os.path.exists(
                    os.path.join(
                        tempfile.gettempdir(),
                        streaming_prefix._get_path(prefix_int, filelock_name),
                    )
                )
                for filelock_name in (
                    streaming_constants.BARRIER_FILELOCK,
                    streaming_constants.CACHE_FILELOCK,
                )
            )
            if filelock_exists:
                continue
        except PermissionError:
            continue

        try:
            shared_memory = streaming_memory.SharedMemory(name, False, auto_cleanup=False)
        except PermissionError:
            continue
        except FileNotFoundError:
            break

        try:
            if shm_name != streaming_constants.LOCALS:
                continue
            their_locals, _ = streaming_prefix._unpack_locals(bytes(shared_memory.buf))
            if any(streams_remote):
                for index, local_dir in enumerate(streams_local):
                    if streams_remote[index] is not None and local_dir in their_locals:
                        raise ValueError(
                            f"Reused local directory: {streams_local} vs {their_locals}. "
                            "Provide a different one. If using a unique local directory, "
                            "try deleting the local directory and call "
                            "`streaming.base.util.clean_stale_shared_memory()` only once "
                            "in your script to clean up the stale shared memory before "
                            "instantiation of `StreamingDataset`."
                        )
        finally:
            # Streaming 0.13 registers every probe with atexit, retaining two
            # descriptors until process exit. Probe attachments have no ownership.
            shared_memory.cleanup()
    return prefix_int


def _prepare_shard_without_poisoned_state(
    self: streaming_dataset.StreamingDataset,
    shard_id: int,
    blocking: bool = True,
) -> None:
    """Return a failed download to REMOTE instead of stranding waiters."""
    original_prepare_shard = getattr(streaming_dataset.StreamingDataset, _ORIGINAL_PREPARE_SHARD_ATTR)
    try:
        original_prepare_shard(self, shard_id, blocking)
    except BaseException:
        if self._shard_states[shard_id] == streaming_dataset._ShardState.PREPARING:
            self._shard_states[shard_id] = streaming_dataset._ShardState.REMOTE
        raise


def patch_streaming() -> None:
    """Apply HAL's Python 3.14 and shared-memory compatibility fixes once."""
    global _PATCHED
    require_supported_streaming()
    if _PATCHED:
        return
    streaming_memory.SharedMemory.fix_register = _register  # ty: ignore[invalid-assignment]
    streaming_memory.SharedMemory.fix_unregister = _unregister  # ty: ignore[invalid-assignment]
    streaming_prefix._check_and_find = _check_and_find_without_fd_leak  # ty: ignore[invalid-assignment]
    streaming_dataset.StreamingDataset.prepare_shard = (  # ty: ignore[invalid-assignment]
        _prepare_shard_without_poisoned_state
    )
    _PATCHED = True
