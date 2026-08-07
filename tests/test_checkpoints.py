from pathlib import Path

import pytest

from hal.training import checkpoints


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
