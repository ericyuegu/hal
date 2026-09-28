"""Pinned runner assets: a manifest, a hash-checked cache, and content-addressed uploads."""

import hashlib
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Final
from typing import Protocol

from botocore.exceptions import ClientError
from loguru import logger

_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")


class AssetError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def policy_bundle_key(sha256: str) -> str:
    return f"netplay/policies/{sha256}.halpolicy"


def account_key(sha256: str) -> str:
    return f"netplay/accounts/{sha256}.json"


@dataclass(frozen=True, slots=True)
class PinnedAsset:
    key: str
    sha256: str
    executable: bool = False

    def __post_init__(self) -> None:
        parts = self.key.split("/")
        if not self.key or self.key.startswith("/") or any(part in ("", ".", "..") for part in parts):
            raise ValueError(f"asset key must be a relative R2 key: {self.key!r}")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError(f"asset {self.key} SHA-256 must be lowercase hex")

    @property
    def name(self) -> str:
        return self.key.rsplit("/", 1)[-1]


class AssetSource(Protocol):
    def fetch(self, key: str, destination: Path) -> None: ...


class R2Source:
    def __init__(self, client: Any, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    def fetch(self, key: str, destination: Path) -> None:
        try:
            self._client.download_file(self._bucket, key, str(destination))
        except ClientError as error:
            raise AssetError(f"cannot download s3://{self._bucket}/{key}: {error}") from error


class LocalSource:
    """Serve R2 keys from a local mirror of the bucket layout, for development."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def fetch(self, key: str, destination: Path) -> None:
        source = self._root / key
        if not source.is_file():
            raise AssetError(f"local asset is missing: {source}")
        shutil.copyfile(source, destination)


class AssetCache:
    """Keep verified files under <root>/<sha256>/<name>; verify again on every use."""

    def __init__(self, root: Path, source: AssetSource) -> None:
        self._root = root
        self._source = source

    def get(self, asset: PinnedAsset) -> Path:
        directory = self._root / asset.sha256
        path = directory / asset.name
        mode = 0o755 if asset.executable else 0o644
        log = logger.bind(event="asset")
        if path.is_file():
            if sha256_file(path) == asset.sha256:
                path.chmod(mode)
                log.info("asset {} sha256={} verified in cache", asset.key, asset.sha256)
                return path
            # Another slot may be replacing this file; the atomic replace below overwrites it.
            log.warning("cached asset {} failed verification; fetching it again", path)
        directory.mkdir(parents=True, exist_ok=True)
        partial = directory / f".{asset.name}.{uuid.uuid4().hex}.partial"
        try:
            self._source.fetch(asset.key, partial)
            actual = sha256_file(partial)
            if actual != asset.sha256:
                raise AssetError(f"{asset.key} has SHA-256 {actual}; expected {asset.sha256}")
            partial.chmod(mode)
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
        log.info("asset {} sha256={} downloaded", asset.key, asset.sha256)
        return path


def ensure_uploaded(remote: Any, bucket: str, path: Path, key: str, sha256: str) -> bool:
    """Upload a content-addressed object once; an existing object must carry the same hash."""
    try:
        head = remote.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey", "NotFound"):
            raise
    else:
        if head.get("Metadata", {}).get("sha256") != sha256:
            raise AssetError(f"s3://{bucket}/{key} exists with a different hash")
        # Matching metadata on a truncated object would fail every runner download.
        if head.get("ContentLength") != path.stat().st_size:
            raise AssetError(
                f"s3://{bucket}/{key} has {head.get('ContentLength')} bytes; expected {path.stat().st_size}"
            )
        return False
    with path.open("rb") as body:
        remote.put_object(
            Bucket=bucket, Key=key, Body=body, ContentType="application/octet-stream", Metadata={"sha256": sha256}
        )
    head = remote.head_object(Bucket=bucket, Key=key)
    if head.get("ContentLength") != path.stat().st_size or head.get("Metadata", {}).get("sha256") != sha256:
        raise AssetError(f"s3://{bucket}/{key} differs from {path} after upload")
    return True
