import hashlib
from pathlib import Path

import pytest
import torch

from hal.training import checkpoints
from hal.training.physical_shard_loader import PhysicalRow
from hal.training.physical_shard_loader import RingSlotDescriptor


def test_current_059_records_load_without_compatibility_unpickler(tmp_path: Path) -> None:
    row = PhysicalRow("source", 2, 3)
    slot = RingSlotDescriptor(1, row, 4, "a" * 64)
    path = tmp_path / "checkpoint.pt"
    torch.save({"row": row, "slot": slot}, path)

    loaded = checkpoints.load_for_resume("unused", tmp_path, device="cpu", name=path.name)

    assert loaded == {"row": row, "slot": slot}


def test_resume_lineage_permits_only_the_declared_source_transition() -> None:
    stored = {"git_sha": "1" * 40, "corpus": "unchanged", "optimizer": {"lr": 0.01}}
    current = {**stored, "git_sha": "2" * 40}
    lineage = checkpoints.ResumeLineage("a" * 64, "1" * 40, "2" * 40, "b" * 64)

    with pytest.raises(ValueError, match="git_sha"):
        checkpoints.validate_resume_provenance(stored, current)
    checkpoints.validate_resume_provenance(stored, current, transition=lineage, parent_checkpoint_sha256="a" * 64)
    assert checkpoints.ResumeLineage.from_record(lineage.to_record()) == lineage

    for changed in ({**current, "corpus": "different"}, {**current, "optimizer": {"lr": 0.02}}):
        with pytest.raises(ValueError, match="provenance changed"):
            checkpoints.validate_resume_provenance(
                stored, changed, transition=lineage, parent_checkpoint_sha256="a" * 64
            )
    with pytest.raises(ValueError, match="does not identify"):
        checkpoints.validate_resume_provenance(stored, current, transition=lineage, parent_checkpoint_sha256="c" * 64)
    with pytest.raises(ValueError, match="does not identify"):
        checkpoints.validate_resume_provenance(
            stored, {**current, "git_sha": "3" * 40}, transition=lineage, parent_checkpoint_sha256="a" * 64
        )


def test_resume_lineage_rejects_invalid_or_disconnected_records() -> None:
    with pytest.raises(ValueError, match="invalid resume lineage"):
        checkpoints.ResumeLineage("not-a-hash", "1" * 40, "2" * 40, "b" * 64)
    with pytest.raises(ValueError, match="declare a source change"):
        checkpoints.ResumeLineage("a" * 64, "1" * 40, "1" * 40, "b" * 64)
    first = checkpoints.ResumeLineage("a" * 64, "1" * 40, "2" * 40, "b" * 64)
    disconnected = checkpoints.ResumeLineage("c" * 64, "3" * 40, "4" * 40, "d" * 64)
    with pytest.raises(ValueError, match="not contiguous"):
        checkpoints.checkpoint_resume_lineage([first.to_record(), disconnected.to_record()])
    with pytest.raises(ValueError, match="fields changed"):
        checkpoints.ResumeLineage.from_record({**first.to_record(), "ignore_environment": True})


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
