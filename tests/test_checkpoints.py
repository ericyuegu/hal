import hashlib
import pickle
import zipfile
from pathlib import Path

import pytest
import torch

from hal.training import checkpoints
from hal.training.physical_shard_loader import GenerationDescriptor
from hal.training.physical_shard_loader import PhysicalRow


def _legacy_physical_checkpoint(tmp_path: Path, *, unknown_class: bool = False) -> Path:
    current = tmp_path / "current.pt"
    legacy = tmp_path / "legacy.pt"
    row = PhysicalRow("source", 2, 3)
    torch.save({"row": row, "generation": GenerationDescriptor(1, row, 4, 5, 6)}, current)
    with zipfile.ZipFile(current) as source, zipfile.ZipFile(legacy, "w") as destination:
        for member in source.infolist():
            content = source.read(member.filename)
            if member.filename.endswith("data.pkl"):
                content = content.replace(b"hal.training.physical_shard_loader", b"hal.training.o51_replay_loader")
                if unknown_class:
                    content = content.replace(b"PhysicalRow", b"UnlistedRow")
            destination.writestr(member, content)
    return legacy


def test_legacy_physical_shard_checkpoint_loads_without_module_shim(tmp_path: Path) -> None:
    path = _legacy_physical_checkpoint(tmp_path)

    loaded = checkpoints.load_legacy_physical_shard_checkpoint(path, device="cpu")
    resumed = checkpoints.load_for_resume("unused", tmp_path, device="cpu", name=path.name, legacy_physical_rows=True)

    assert loaded == resumed
    assert loaded["row"] == PhysicalRow("source", 2, 3)
    assert loaded["generation"] == GenerationDescriptor(1, loaded["row"], 4, 5, 6)


def test_legacy_physical_shard_checkpoint_rejects_unknown_class(tmp_path: Path) -> None:
    path = _legacy_physical_checkpoint(tmp_path, unknown_class=True)

    with pytest.raises(pickle.UnpicklingError, match="unsupported legacy physical-shard class"):
        checkpoints.load_legacy_physical_shard_checkpoint(path, device="cpu")


class _Client:
    def __init__(self, *, fail_name: str | None = None) -> None:
        self.fail_name = fail_name
        self.uploaded: list[tuple[str, str, str]] = []

    def upload_file(self, local: str, bucket: str, key: str) -> None:
        self.uploaded.append((local, bucket, key))
        if Path(local).name == self.fail_name:
            raise OSError("upload failed")


def _uploader(monkeypatch: pytest.MonkeyPatch, client: _Client) -> checkpoints.BackgroundUploader:
    monkeypatch.setattr(checkpoints.r2, "bucket", lambda: "test-bucket")
    monkeypatch.setattr(checkpoints.r2, "client", lambda: client)
    return checkpoints.BackgroundUploader("test-run")


def test_uploader_close_drains_all_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    uploader = _uploader(monkeypatch, client)
    paths = [tmp_path / "a.pt", tmp_path / "b.pt"]
    for path in paths:
        path.write_bytes(b"data")
        uploader.upload(path)

    uploader.close()

    assert [Path(local).name for local, _, _ in client.uploaded] == ["a.pt", "b.pt"]


def test_uploader_wait_confirms_queued_checkpoint_upload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    uploader = _uploader(monkeypatch, client)
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"data")
    uploader.upload(path)

    uploader.wait()

    assert [Path(local).name for local, _, _ in client.uploaded] == ["checkpoint.pt"]
    uploader.close()


def test_uploader_close_fails_after_draining_queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client(fail_name="a.pt")
    uploader = _uploader(monkeypatch, client)
    paths = [tmp_path / "a.pt", tmp_path / "b.pt"]
    for path in paths:
        path.write_bytes(b"data")
        uploader.upload(path)

    with pytest.raises(RuntimeError, match="1 R2 upload"):
        uploader.close()

    assert [Path(local).name for local, _, _ in client.uploaded] == ["a.pt", "b.pt"]


def test_uploader_skips_unchanged_file_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    uploader = _uploader(monkeypatch, client)
    path = tmp_path / "match.slp"
    path.write_bytes(b"replay")

    assert uploader.upload(path, key="replays/match.slp")
    assert not uploader.upload(path, key="replays/match.slp")
    uploader.close()

    assert len(client.uploaded) == 1


def test_uploader_queues_changed_file_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    uploader = _uploader(monkeypatch, client)
    path = tmp_path / "matches.jsonl"
    path.write_bytes(b"first")
    assert uploader.upload(path)

    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"second")
    replacement.replace(path)
    assert uploader.upload(path)
    uploader.close()

    assert len(client.uploaded) == 2


def test_uploader_treats_remote_keys_as_distinct(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    uploader = _uploader(monkeypatch, client)
    path = tmp_path / "match.slp"
    path.write_bytes(b"replay")

    assert uploader.upload(path, key="orientation_0/match.slp")
    assert uploader.upload(path, key="orientation_1/match.slp")
    uploader.close()

    assert len(client.uploaded) == 2


def test_download_latest_creates_nested_checkpoint_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Client:
        def download_file(self, bucket: str, key: str, destination: str) -> None:
            assert bucket == "test-bucket"
            assert key == "runs/run/checkpoints/step-0008192.pt"
            Path(destination).write_bytes(b"checkpoint")

    monkeypatch.setattr(checkpoints.r2, "bucket", lambda: "test-bucket")
    monkeypatch.setattr(checkpoints.r2, "client", Client)

    path = checkpoints.download_latest(
        "run",
        tmp_path / "run",
        name="checkpoints/step-0008192.pt",
    )

    assert path is not None
    assert path == tmp_path / "run/checkpoints/step-0008192.pt"
    assert path.read_bytes() == b"checkpoint"


def test_upload_tree_only_queues_new_versions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    uploader = _uploader(monkeypatch, client)
    root = tmp_path / "h2h"
    first = root / "orientation_0" / "match.slp"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"first")

    assert uploader.upload_tree(root, base=tmp_path) == 1
    assert uploader.upload_tree(root, base=tmp_path) == 0

    second = root / "orientation_1" / "match.slp"
    second.parent.mkdir(parents=True)
    second.write_bytes(b"second")
    assert uploader.upload_tree(root, base=tmp_path) == 1
    assert uploader.upload_tree(root, base=tmp_path) == 0
    uploader.close()

    assert len(client.uploaded) == 2


def test_checkpoint_sha256_reads_the_complete_file(tmp_path: Path) -> None:
    content = b"a" * (1024 * 1024) + b"tail"
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(content)

    assert checkpoints.checkpoint_sha256(path) == hashlib.sha256(content).hexdigest()


def test_advance_checkpoint_link_atomically_replaces_the_destination(tmp_path: Path) -> None:
    source = tmp_path / "boundary-step-0000002.pt"
    source.write_bytes(b"new checkpoint")
    destination = tmp_path / "latest.pt"
    destination.write_bytes(b"old checkpoint")

    checkpoints.advance_checkpoint_link(source, destination)

    assert destination.read_bytes() == b"new checkpoint"
    assert destination.stat().st_ino == source.stat().st_ino
    assert not (tmp_path / "latest.pt.tmp").exists()
