"""Mosaic Streaming 0.13.0 compatibility boundary."""

import importlib
import json
import subprocess
import sys
import textwrap
from multiprocessing import resource_tracker
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from streaming import MDSWriter
from streaming.base.shared.memory import SharedMemory

import hal.data.streaming_compat as streaming_compat


def _write_scalar_mds(root: Path) -> None:
    with MDSWriter(out=str(root), columns={"value": "int"}, compression="zstd") as writer:
        writer.write({"value": 1})


def test_resource_tracker_forwards_without_extra_self() -> None:
    streaming_compat.patch_streaming()
    memory = object.__new__(SharedMemory)
    with patch.object(resource_tracker._resource_tracker, "register") as register:
        memory.fix_register("/semaphore", "semaphore")
    register.assert_called_once_with("/semaphore", "semaphore")
    with patch.object(resource_tracker._resource_tracker, "unregister") as unregister:
        memory.fix_unregister("/semaphore", "semaphore")
    unregister.assert_called_once_with("/semaphore", "semaphore")


def test_patches_are_idempotent_and_version_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    streaming_compat.patch_streaming()
    patched = streaming_compat.streaming_prefix._check_and_find
    streaming_compat.patch_streaming()
    assert streaming_compat.streaming_prefix._check_and_find is patched

    monkeypatch.setattr(streaming_compat, "version", lambda _distribution: "0.14.0")
    with pytest.raises(RuntimeError, match="mosaicml-streaming==0.13.0"):
        streaming_compat.patch_streaming()


def test_patches_survive_module_reload() -> None:
    dataset_type = streaming_compat.streaming_dataset.StreamingDataset
    original = getattr(dataset_type, streaming_compat._ORIGINAL_PREPARE_SHARD_ATTR)

    reloaded = importlib.reload(streaming_compat)
    reloaded.patch_streaming()

    assert getattr(dataset_type, reloaded._ORIGINAL_PREPARE_SHARD_ATTR) is original
    assert dataset_type.prepare_shard is reloaded._prepare_shard_without_poisoned_state
    assert original is not dataset_type.prepare_shard


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="requires Linux procfs")
def test_prefix_probes_do_not_leak_file_descriptors(tmp_path: Path) -> None:
    _write_scalar_mds(tmp_path / "train")
    program = textwrap.dedent(
        """\
        import gc
        import json
        import os
        import sys

        from hal.data.streaming_compat import patch_streaming
        from streaming import StreamingDataset

        patch_streaming()

        def count_shared_memory_descriptors():
            count = 0
            for name in os.listdir("/proc/self/fd"):
                try:
                    target = os.readlink(f"/proc/self/fd/{name}")
                except FileNotFoundError:
                    continue
                count += target.startswith("/dev/shm/")
            return count

        counts = [count_shared_memory_descriptors()]
        for _ in range(5):
            dataset = StreamingDataset(local=sys.argv[1], batch_size=1, shuffle=False)
            del dataset
            gc.collect()
            counts.append(count_shared_memory_descriptors())
        print(json.dumps(counts))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path / "train")],
        check=True,
        capture_output=True,
        text=True,
    )
    counts = json.loads(completed.stdout.splitlines()[-1])
    increments = [after - before for before, after in zip(counts[:-1], counts[1:], strict=True)]
    assert increments == [increments[0]] * len(increments), counts


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="requires Linux procfs")
def test_roundtrip_reader_applies_prefix_probe_fix(tmp_path: Path) -> None:
    with MDSWriter(
        out=str(tmp_path / "train"),
        columns={"schema_version": "int", "value": "int"},
        compression="zstd",
    ) as writer:
        writer.write({"schema_version": 7, "value": 1})
    program = textwrap.dedent(
        """\
        import json
        import os
        import sys
        from pathlib import Path

        from hal.scripts.roundtrip import _read_mds_row

        def count_shared_memory_descriptors():
            count = 0
            for name in os.listdir("/proc/self/fd"):
                try:
                    target = os.readlink(f"/proc/self/fd/{name}")
                except FileNotFoundError:
                    continue
                count += target.startswith("/dev/shm/")
            return count

        counts = [count_shared_memory_descriptors()]
        for _ in range(5):
            assert _read_mds_row(Path(sys.argv[1]), "train", 0) == {"value": 1}
            counts.append(count_shared_memory_descriptors())
        print(json.dumps(counts))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    counts = json.loads(completed.stdout.splitlines()[-1])
    assert counts[-1] - counts[0] <= 100, counts


def test_failed_download_does_not_poison_shared_shard_state(monkeypatch: pytest.MonkeyPatch) -> None:
    remote = streaming_compat.streaming_dataset._ShardState.REMOTE
    preparing = streaming_compat.streaming_dataset._ShardState.PREPARING

    class Dataset:
        _shard_states = np.array([remote], dtype=np.uint8)

    def fail(dataset: Dataset, shard_id: int, blocking: bool) -> None:
        del blocking
        dataset._shard_states[shard_id] = preparing
        raise RuntimeError("transient object-store failure")

    monkeypatch.setattr(
        streaming_compat.streaming_dataset.StreamingDataset,
        streaming_compat._ORIGINAL_PREPARE_SHARD_ATTR,
        fail,
    )
    with pytest.raises(RuntimeError, match="object-store"):
        streaming_compat._prepare_shard_without_poisoned_state(Dataset(), 0)  # type: ignore[arg-type]
    assert Dataset._shard_states[0] == remote
