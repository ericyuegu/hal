"""Populate immutable HAL fixtures in a Modal image-build layer.

This helper intentionally has no dependency on the HAL package. The launcher
copies only this file before the repository source layer, then supplies the
fixture manifest through ``HAL_FIXTURE_MANIFEST``. Ordinary code changes can
therefore reuse the large fixture layer.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import urllib.request
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import cast

import boto3
from botocore.config import Config

_CHUNK_SIZE = 8 << 20
_SENTINEL = ".sha256"


@dataclass(frozen=True, slots=True)
class Fixture:
    """One validated fixture from the launcher-supplied manifest."""

    name: str
    sha256: str
    size_bytes: int
    dest: Path
    r2_key: str | None
    url: str | None
    extract: str | None

    @classmethod
    def from_json(cls, value: object) -> Fixture:
        """Parse one manifest object and reject unsafe destinations."""
        if not isinstance(value, dict):
            raise TypeError(f"fixture entry must be an object, got {type(value).__name__}")
        payload = cast(dict[str, Any], value)
        fixture = cls(
            name=str(payload["name"]),
            sha256=str(payload["sha256"]),
            size_bytes=int(payload["size_bytes"]),
            dest=Path(str(payload["dest"])),
            r2_key=None if payload["r2_key"] is None else str(payload["r2_key"]),
            url=None if payload["url"] is None else str(payload["url"]),
            extract=None if payload["extract"] is None else str(payload["extract"]),
        )
        if fixture.dest.is_absolute() or ".." in fixture.dest.parts:
            raise ValueError(f"{fixture.name}: destination must stay below the fixture root")
        if (fixture.r2_key is None) == (fixture.url is None):
            raise ValueError(f"{fixture.name}: exactly one download backend is required")
        if fixture.extract not in (None, "tar_zst", "appimage"):
            raise ValueError(f"{fixture.name}: unsupported extraction mode {fixture.extract!r}")
        return fixture


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _r2_client() -> Any:
    required = ("AWS_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_BUCKET")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"fixture build is missing secret variables: {missing}")
    return boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def _download(fixture: Fixture, destination: Path, r2_client: Any) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.download")
    temporary.unlink(missing_ok=True)
    if fixture.r2_key is not None:
        response = r2_client.get_object(Bucket=os.environ["AWS_BUCKET"], Key=fixture.r2_key)
        reader = response["Body"]
    else:
        assert fixture.url is not None
        reader = urllib.request.urlopen(fixture.url)
    digest = hashlib.sha256()
    downloaded = 0
    try:
        with closing(reader), temporary.open("wb") as output:
            while chunk := reader.read(_CHUNK_SIZE):
                output.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                print(f"[fixtures] {fixture.name}: {downloaded}/{fixture.size_bytes}", flush=True)
        observed = digest.hexdigest()
        if observed != fixture.sha256:
            raise RuntimeError(f"{fixture.name}: sha256 {observed} != expected {fixture.sha256}")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _extract(fixture: Fixture, archive: Path, destination: Path) -> None:
    staging = destination.with_name(f".{destination.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        if fixture.extract == "tar_zst":
            with tarfile.open(archive, "r:zst") as bundle:
                bundle.extractall(staging, filter="data")
        elif fixture.extract == "appimage":
            archive.chmod(0o755)
            subprocess.run([str(archive), "--appimage-extract"], cwd=staging, check=True)
        else:
            raise AssertionError(f"unexpected extraction mode {fixture.extract!r}")
        (staging / _SENTINEL).write_text(fixture.sha256 + "\n")
        if destination.exists():
            shutil.rmtree(destination)
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _ensure(fixture: Fixture, root: Path, r2_client: Any) -> None:
    destination = root / fixture.dest
    if fixture.extract is None:
        if destination.is_file() and _sha256(destination) == fixture.sha256:
            print(f"[fixtures] cached {fixture.name}", flush=True)
            return
        _download(fixture, destination, r2_client)
    else:
        sentinel = destination / _SENTINEL
        if sentinel.is_file() and sentinel.read_text().strip() == fixture.sha256:
            print(f"[fixtures] cached {fixture.name}", flush=True)
            return
        archive = root / "data" / ".fixture-build" / fixture.name
        _download(fixture, archive, r2_client)
        try:
            _extract(fixture, archive, destination)
        finally:
            archive.unlink(missing_ok=True)
    print(f"[fixtures] ready {fixture.name} -> {destination}", flush=True)


def main() -> None:
    raw_manifest = json.loads(os.environ["HAL_FIXTURE_MANIFEST"])
    if not isinstance(raw_manifest, list) or not raw_manifest:
        raise ValueError("HAL_FIXTURE_MANIFEST must be a non-empty JSON list")
    fixtures = [Fixture.from_json(value) for value in raw_manifest]
    root = Path(os.environ.get("HAL_FIXTURE_ROOT", "/opt/hal"))
    client = _r2_client() if any(fixture.r2_key is not None for fixture in fixtures) else None
    for fixture in fixtures:
        _ensure(fixture, root, client)


if __name__ == "__main__":
    main()
