import base64
import hashlib
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError

from hal import r2


class _Client:
    def __init__(self) -> None:
        self.put: dict[str, Any] | None = None
        self.deleted: tuple[str, str] | None = None

    def put_object(self, **kwargs: Any) -> dict[str, str]:
        body = kwargs.pop("Body")
        payload = body.read()
        self.put = {**kwargs, "payload": payload}
        return {"ETag": hashlib.md5(payload, usedforsecurity=False).hexdigest()}

    def delete_object(self, *, Bucket: str, Key: str) -> None:
        self.deleted = (Bucket, Key)


class _Paginator:
    def paginate(self, *, Bucket: str, Prefix: str) -> list[dict[str, object]]:
        assert Bucket == "hal"
        assert Prefix == "staging/"
        return [
            {
                "Contents": [
                    {"Key": "staging/b", "Size": 2, "ETag": '"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"'},
                    {"Key": "staging/a", "Size": 1, "ETag": '"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"'},
                ]
            }
        ]


class _ListingClient:
    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        assert Bucket == "hal"
        assert Key == "staging"
        raise ClientError(
            {"Error": {"Code": "404"}, "ResponseMetadata": {"HTTPStatusCode": 404}},
            "HeadObject",
        )

    def get_paginator(self, name: str) -> _Paginator:
        assert name == "list_objects_v2"
        return _Paginator()


def test_copy_file_uploads_with_boto3_and_checks_content_md5(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = tmp_path / "object"
    local.write_bytes(b"payload")
    fake = _Client()
    monkeypatch.setattr(r2, "client", lambda: fake)

    r2.copy_file(local, "r2:hal/staging/object")

    assert fake.put == {
        "Bucket": "hal",
        "Key": "staging/object",
        "ContentLength": 7,
        "ContentMD5": base64.b64encode(hashlib.md5(b"payload", usedforsecurity=False).digest()).decode("ascii"),
        "payload": b"payload",
    }
    assert fake.deleted is None


def test_list_objects_returns_sorted_relative_paths_and_etag_md5(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(r2, "client", _ListingClient)

    assert r2.list_objects("r2:hal/staging") == [
        r2.R2File(path="a", size=1, md5="a" * 32),
        r2.R2File(path="b", size=2, md5="b" * 32),
    ]


def test_copy_file_removes_an_upload_with_the_wrong_etag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = tmp_path / "object"
    local.write_bytes(b"payload")
    fake = _Client()
    monkeypatch.setattr(fake, "put_object", lambda **_kwargs: {"ETag": "0" * 32})
    monkeypatch.setattr(r2, "client", lambda: fake)

    with pytest.raises(r2.R2Error, match="ETag differs"):
        r2.copy_file(local, "r2:hal/staging/object")

    assert fake.deleted == ("hal", "staging/object")
