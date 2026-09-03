"""Operational contracts for the O51 corpus materializer."""

from pathlib import Path

import pytest

from hal.scripts import materialize_o51


def test_remote_writer_uses_one_training_prefix_per_band(tmp_path: Path) -> None:
    remote = "s3://hal/processed/_staging/corpus"

    assert materialize_o51._writer_output(tmp_path / "band-4" / "train", None, 4) == str(tmp_path / "band-4" / "train")
    assert materialize_o51._writer_output(tmp_path / "band-4" / "train", remote, 4) == (
        str(tmp_path / "band-4" / "train"),
        f"{remote}/band-4/train",
    )


def test_remote_output_must_be_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    class Client:
        def list_objects_v2(self, **_kwargs: object) -> dict[str, int]:
            return {"KeyCount": 1}

    monkeypatch.setattr(materialize_o51.r2, "client", lambda: Client())

    with pytest.raises(FileExistsError, match="already exists"):
        materialize_o51._ensure_empty_remote("s3://hal/processed/existing")


def test_remote_artifacts_keep_paths_relative_to_corpus_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploaded: list[tuple[str, str, str]] = []

    class Client:
        def upload_file(self, local: str, bucket: str, key: str) -> None:
            uploaded.append((local, bucket, key))

    root = tmp_path / "corpus"
    manifest = root / "band-1" / "manifest.o51.jsonl.gz"
    report = root / "materialization.json"
    manifest.parent.mkdir(parents=True)
    manifest.touch()
    report.touch()
    monkeypatch.setattr(materialize_o51.r2, "client", lambda: Client())

    materialize_o51._upload_artifacts(
        root,
        "s3://hal/processed/_staging/corpus",
        [manifest, report],
    )

    assert uploaded == [
        (str(manifest), "hal", "processed/_staging/corpus/band-1/manifest.o51.jsonl.gz"),
        (str(report), "hal", "processed/_staging/corpus/materialization.json"),
    ]


def test_streaming_endpoint_uses_standard_aws_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://r2.example")
    monkeypatch.delenv("S3_ENDPOINT_URL", raising=False)

    materialize_o51._bridge_streaming_endpoint()

    assert materialize_o51.os.environ["S3_ENDPOINT_URL"] == "https://r2.example"
