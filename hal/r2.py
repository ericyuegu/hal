"""Cloudflare R2 client and object helpers."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Final

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

_CRED_VARS: Final[tuple[str, ...]] = ("AWS_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
_MAX_SINGLE_COPY_BYTES: Final[int] = 5 * 2**30


class R2Error(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class R2File:
    path: str
    size: int
    md5: str | None


def missing_credentials() -> list[str]:
    """Required R2 env vars that are unset (empty list means ready to connect)."""
    return [name for name in _CRED_VARS if not os.environ.get(name)]


def bucket() -> str:
    name = os.environ.get("AWS_BUCKET")
    if not name:
        raise R2Error("AWS_BUCKET env var not set. See .env.example.")
    return name


def client():  # type: ignore[no-untyped-def]
    """Construct an S3 client for R2."""
    missing = missing_credentials()
    if missing:
        raise R2Error(f"missing env vars for R2: {missing}. See .env.example.")
    return boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def _split_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("r2:"):
        raise ValueError(f"expected an r2: URI, got {uri!r}")
    remote = uri.removeprefix("r2:")
    name, separator, key = remote.partition("/")
    if not name:
        raise ValueError(f"R2 URI has no bucket: {uri!r}")
    if not separator:
        key = ""
    return name, key


def _not_found(error: ClientError) -> bool:
    code = str(error.response.get("Error", {}).get("Code", ""))
    status = int(error.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
    return code in {"404", "NoSuchKey", "NotFound"} or status == 404


def _md5_from_etag(etag: object) -> str | None:
    value = str(etag or "").strip('"').lower()
    if len(value) == 32 and "-" not in value and all(character in "0123456789abcdef" for character in value):
        return value
    return None


def _object_rows(prefix: str) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
    name, key = _split_uri(prefix)
    r2_client = client()
    if key and not key.endswith("/"):
        try:
            head = r2_client.head_object(Bucket=name, Key=key)
        except ClientError as error:
            if not _not_found(error):
                raise
        else:
            row = {"Key": key, "Size": head["ContentLength"], "ETag": head["ETag"]}
            return name, [(Path(key).name, row)]

    base = key.rstrip("/")
    object_prefix = f"{base}/" if base else ""
    rows: list[tuple[str, dict[str, Any]]] = []
    paginator = r2_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=name, Prefix=object_prefix):
        for row in page.get("Contents", ()):
            object_key = str(row["Key"])
            relative = object_key.removeprefix(object_prefix)
            if relative:
                rows.append((relative, row))
    rows.sort(key=lambda item: item[0])
    return name, rows


def list_objects(prefix: str) -> list[R2File]:
    """List file metadata below one R2 prefix."""
    _, rows = _object_rows(prefix)
    return [
        R2File(path=relative, size=int(row["Size"]), md5=_md5_from_etag(row.get("ETag"))) for relative, row in rows
    ]


def list_files(prefix: str) -> list[str]:
    """List file paths below one R2 prefix."""
    return [item.path for item in list_objects(prefix)]


def object_size(uri: str) -> int:
    """Return the byte size of one R2 object."""
    name, key = _split_uri(uri)
    if not key:
        raise ValueError(f"R2 object URI has no key: {uri!r}")
    return int(client().head_object(Bucket=name, Key=key)["ContentLength"])


def read_bytes(uri: str) -> bytes:
    """Read one R2 object."""
    name, key = _split_uri(uri)
    if not key:
        raise ValueError(f"R2 object URI has no key: {uri!r}")
    response = client().get_object(Bucket=name, Key=key)
    with response["Body"] as body:
        return bytes(body.read())


def read_json(uri: str) -> Any:
    """Read one JSON object from R2."""
    return json.loads(read_bytes(uri))


def _file_md5(path: Path) -> tuple[str, str]:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest(), base64.b64encode(digest.digest()).decode("ascii")


def copy_file(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
    """Copy one local or R2 object and verify local-to-R2 content identity."""
    source_name = os.fspath(source)
    destination_name = os.fspath(destination)
    source_remote = source_name.startswith("r2:")
    destination_remote = destination_name.startswith("r2:")
    if source_remote and destination_remote:
        source_bucket, source_key = _split_uri(source_name)
        destination_bucket, destination_key = _split_uri(destination_name)
        source_objects = list_objects(source_name)
        if len(source_objects) != 1:
            raise FileNotFoundError(f"expected one R2 source object: {source_name}")
        source_object = source_objects[0]
        if source_object.size > _MAX_SINGLE_COPY_BYTES:
            raise ValueError(f"R2 object is too large for one server-side copy: {source_name}")
        r2_client = client()
        r2_client.copy_object(
            Bucket=destination_bucket,
            Key=destination_key,
            CopySource={"Bucket": source_bucket, "Key": source_key},
        )
        destination_objects = list_objects(destination_name)
        if len(destination_objects) != 1 or destination_objects[0].size != source_object.size:
            r2_client.delete_object(Bucket=destination_bucket, Key=destination_key)
            raise R2Error(f"R2 object differs after copying {source_name} to {destination_name}")
        if source_object.md5 is not None and destination_objects[0].md5 != source_object.md5:
            r2_client.delete_object(Bucket=destination_bucket, Key=destination_key)
            raise R2Error(f"R2 MD5 differs after copying {source_name} to {destination_name}")
        return
    if destination_remote:
        local = Path(source_name)
        size = local.stat().st_size
        if size > _MAX_SINGLE_COPY_BYTES:
            raise ValueError(f"local file is too large for one R2 upload: {local}")
        destination_bucket, destination_key = _split_uri(destination_name)
        md5_hex, md5_base64 = _file_md5(local)
        r2_client = client()
        with local.open("rb") as body:
            response = r2_client.put_object(
                Bucket=destination_bucket,
                Key=destination_key,
                Body=body,
                ContentLength=size,
                ContentMD5=md5_base64,
            )
        if _md5_from_etag(response.get("ETag")) != md5_hex:
            r2_client.delete_object(Bucket=destination_bucket, Key=destination_key)
            raise R2Error(f"R2 ETag differs after uploading {destination_name}")
        return
    if source_remote:
        source_bucket, source_key = _split_uri(source_name)
        local = Path(destination_name)
        local.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=local.parent, prefix=f".{local.name}.", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            client().download_file(source_bucket, source_key, str(temporary))
            temporary.replace(local)
        finally:
            temporary.unlink(missing_ok=True)
        return
    shutil.copy2(source_name, destination_name)


def copy_prefix(source: str, destination: str, *, immutable: bool = False) -> None:
    """Copy an R2 prefix with server-side object copies."""
    if immutable and list_files(destination):
        raise FileExistsError(f"destination prefix is not empty: {destination}")
    source_bucket, source_key = _split_uri(source)
    destination_bucket, destination_key = _split_uri(destination)
    source_base = source_key.rstrip("/")
    destination_base = destination_key.rstrip("/")
    _, rows = _object_rows(source)
    if not rows:
        raise FileNotFoundError(f"R2 prefix is empty: {source}")
    oversized = [relative for relative, row in rows if int(row["Size"]) > _MAX_SINGLE_COPY_BYTES]
    if oversized:
        raise ValueError(f"R2 objects are too large for one server-side copy: {oversized}")
    r2_client = client()
    for relative, _ in rows:
        source_object = f"{source_base}/{relative}" if source_base else relative
        destination_object = f"{destination_base}/{relative}" if destination_base else relative
        r2_client.copy_object(
            Bucket=destination_bucket,
            Key=destination_object,
            CopySource={"Bucket": source_bucket, "Key": source_object},
        )


def delete_prefix(prefix: str) -> list[str]:
    """Delete every object at one exact R2 prefix and return the removed paths."""
    name, rows = _object_rows(prefix)
    if not rows:
        return []
    r2_client = client()
    removed: list[str] = []
    for offset in range(0, len(rows), 1_000):
        batch = rows[offset : offset + 1_000]
        response = r2_client.delete_objects(
            Bucket=name,
            Delete={"Objects": [{"Key": str(row["Key"])} for _, row in batch], "Quiet": True},
        )
        if errors := response.get("Errors"):
            raise R2Error(f"R2 failed to delete objects below {prefix}: {errors}")
        removed.extend(relative for relative, _ in batch)
    if remaining := list_files(prefix):
        raise R2Error(f"R2 objects remain below {prefix}: {remaining}")
    return removed
