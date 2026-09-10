"""Private R2 storage for completed production netplay replays."""

import hashlib
import json
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Final

from botocore.exceptions import ClientError

from hal import r2
from hal.netplay_service.domain import validate_player_code

REPLAY_PREFIX: Final[str] = "netplay/v1/replays/"
REPLAY_LIFECYCLE_ID: Final[str] = "hal-netplay-v1-replays-30d"


@dataclass(frozen=True, slots=True)
class ReplayMetadata:
    reservation_id: str
    player_code: str
    game_number: int
    actual_stage: str
    result: str
    policy_sha256: str
    git_sha: str
    started_at: datetime
    ended_at: datetime

    def __post_init__(self) -> None:
        validate_player_code(self.player_code)
        if not self.reservation_id or "/" in self.reservation_id:
            raise ValueError("reservation_id must be a non-empty R2 path segment")
        if not 1 <= self.game_number <= 5:
            raise ValueError("game_number must be in [1, 5]")
        if self.started_at.tzinfo is None or self.ended_at.tzinfo is None:
            raise ValueError("replay timestamps must include a timezone")
        if self.ended_at < self.started_at:
            raise ValueError("replay ended before it started")
        for name, value in (("policy_sha256", self.policy_sha256), ("git_sha", self.git_sha)):
            if not value or "/" in value:
                raise ValueError(f"{name} must be a non-empty identifier")


@dataclass(frozen=True, slots=True)
class UploadedReplay:
    key: str
    metadata_key: str
    sha256: str
    size: int
    etag: str


def replay_object_key(metadata: ReplayMetadata) -> str:
    date = metadata.started_at.astimezone(UTC).strftime("%Y/%m/%d")
    return (
        f"{REPLAY_PREFIX}{date}/{metadata.player_code}/{metadata.reservation_id}/game-{metadata.game_number:02d}.slp"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _etag(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError("R2 response has no ETag")
    return value.strip('"')


@contextmanager
def _r2_client(client: Any | None) -> Iterator[Any]:
    if client is not None:
        yield client
        return
    remote = r2.client()
    try:
        yield remote
    finally:
        remote.close()


def _metadata_json(metadata: ReplayMetadata, replay: UploadedReplay) -> bytes:
    values = asdict(metadata)
    values["started_at"] = metadata.started_at.astimezone(UTC).isoformat()
    values["ended_at"] = metadata.ended_at.astimezone(UTC).isoformat()
    values.update(
        {
            "schema_version": 1,
            "replay_key": replay.key,
            "replay_sha256": replay.sha256,
            "replay_size": replay.size,
            "replay_etag": replay.etag,
        }
    )
    return json.dumps(values, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()


def upload_replay(path: str | Path, metadata: ReplayMetadata, *, client: Any | None = None) -> UploadedReplay:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"replay does not exist: {source}")
    size = source.stat().st_size
    if size <= 0:
        raise ValueError(f"replay is empty: {source}")
    digest = _sha256(source)
    key = replay_object_key(metadata)
    bucket = r2.bucket()
    with _r2_client(client) as remote:
        with source.open("rb") as body:
            response = remote.put_object(
                Bucket=bucket,
                Key=key,
                Body=body,
                ContentType="application/octet-stream",
                Metadata={"sha256": digest},
            )
        etag = _etag(response.get("ETag"))
        head = remote.head_object(Bucket=bucket, Key=key)
        if head.get("ContentLength") != size:
            raise RuntimeError(
                f"R2 replay size differs after upload: expected {size}, got {head.get('ContentLength')}"
            )
        if head.get("Metadata", {}).get("sha256") != digest:
            raise RuntimeError("R2 replay SHA-256 metadata differs after upload")
        if _etag(head.get("ETag")) != etag:
            raise RuntimeError("R2 replay ETag differs after upload")

        metadata_key = key.removesuffix(".slp") + ".json"
        uploaded = UploadedReplay(key, metadata_key, digest, size, etag)
        encoded = _metadata_json(metadata, uploaded)
        metadata_sha256 = hashlib.sha256(encoded).hexdigest()
        metadata_response = remote.put_object(
            Bucket=bucket,
            Key=metadata_key,
            Body=encoded,
            ContentType="application/json",
            Metadata={"sha256": metadata_sha256},
        )
        metadata_head = remote.head_object(Bucket=bucket, Key=metadata_key)
        if metadata_head.get("ContentLength") != len(encoded):
            raise RuntimeError("R2 replay metadata size differs after upload")
        if metadata_head.get("Metadata", {}).get("sha256") != metadata_sha256:
            raise RuntimeError("R2 replay metadata SHA-256 differs after upload")
        if _etag(metadata_head.get("ETag")) != _etag(metadata_response.get("ETag")):
            raise RuntimeError("R2 replay metadata ETag differs after upload")
    return uploaded


def upload_and_delete(path: str | Path, metadata: ReplayMetadata, *, client: Any | None = None) -> UploadedReplay:
    source = Path(path)
    uploaded = upload_replay(source, metadata, client=client)
    source.unlink()
    return uploaded


def ensure_replay_lifecycle(*, client: Any | None = None) -> None:
    """Install the 30-day rule without replacing unrelated bucket rules."""
    bucket = r2.bucket()
    with _r2_client(client) as remote:
        try:
            current = remote.get_bucket_lifecycle_configuration(Bucket=bucket).get("Rules", [])
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code")
            if code not in ("NoSuchLifecycleConfiguration", "NoSuchLifecycle"):
                raise
            current = []
        rule = {
            "ID": REPLAY_LIFECYCLE_ID,
            "Status": "Enabled",
            "Filter": {"Prefix": REPLAY_PREFIX},
            "Expiration": {"Days": 30},
        }
        rules = [existing for existing in current if existing.get("ID") != REPLAY_LIFECYCLE_ID]
        rules.append(rule)
        remote.put_bucket_lifecycle_configuration(Bucket=bucket, LifecycleConfiguration={"Rules": rules})


@contextmanager
def soak_replay_directory() -> Iterator[Path]:
    """Provide a replay directory that cannot survive a soak test."""
    root = Path(tempfile.mkdtemp(prefix="hal-netplay-soak-"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)
