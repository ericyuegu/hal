"""Compatibility fixes for MosaicML Streaming."""

import os
import tempfile
from multiprocessing import resource_tracker
from typing import Any

import streaming.base.constant as streaming_constants
import streaming.base.dataset as streaming_dataset
import streaming.base.shared.memory as streaming_memory
import streaming.base.shared.prefix as streaming_prefix

_ORIGINAL_PREPARE_SHARD = streaming_dataset.StreamingDataset.prepare_shard


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
    try:
        _ORIGINAL_PREPARE_SHARD(self, shard_id, blocking)
    except BaseException:
        if self._shard_states[shard_id] == streaming_dataset._ShardState.PREPARING:
            self._shard_states[shard_id] = streaming_dataset._ShardState.REMOTE
        raise


def patch_streaming() -> None:
    """Apply HAL's Python 3.14 and shared-memory compatibility fixes."""
    streaming_memory.SharedMemory.fix_register = _register  # ty: ignore[invalid-assignment]
    streaming_memory.SharedMemory.fix_unregister = _unregister  # ty: ignore[invalid-assignment]
    streaming_prefix._check_and_find = _check_and_find_without_fd_leak  # ty: ignore[invalid-assignment]
    streaming_dataset.StreamingDataset.prepare_shard = (  # ty: ignore[invalid-assignment]
        _prepare_shard_without_poisoned_state
    )
