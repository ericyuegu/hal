from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from hal import streams


class _DownloadClient:
    def __init__(self, error_code: str | None) -> None:
        self.error_code = error_code

    def download_file(self, _bucket: str, _key: str, destination: str) -> None:
        if self.error_code is not None:
            raise ClientError({"Error": {"Code": self.error_code}}, "HeadObject")
        Path(destination).write_text("{}")


def test_pull_stats_if_available_downloads_existing_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = streams.StreamSource("test", "s3://bucket/root", tmp_path / "dataset")
    monkeypatch.setattr(streams.r2, "client", lambda: _DownloadClient(None))

    result = streams.pull_stats_if_available(source)

    assert result == tmp_path / "dataset" / "stats.json"
    assert result is not None
    assert result.read_text() == "{}"


def test_pull_stats_if_available_skips_only_missing_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = streams.StreamSource("missing", "s3://bucket/root", tmp_path / "missing")
    monkeypatch.setattr(streams.r2, "client", lambda: _DownloadClient("404"))

    assert streams.pull_stats_if_available(source) is None

    monkeypatch.setattr(streams.r2, "client", lambda: _DownloadClient("AccessDenied"))
    with pytest.raises(ClientError, match="AccessDenied"):
        streams.pull_stats_if_available(source)
