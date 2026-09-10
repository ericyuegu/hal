"""Resolve local checkpoints or immutable Cloudflare R2 objects."""

import contextlib
import hashlib
import json
import os
import re
from pathlib import Path

from hal import r2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_r2_uri(uri: str) -> tuple[str, str]:
    """Split an exact ``r2://bucket/key`` object URI."""
    match = re.fullmatch(r"r2://([^/]+)/(.+)", uri)
    if match is None:
        raise ValueError(f"invalid R2 object URI: {uri!r}")
    bucket, key = match.groups()
    if key.endswith("/"):
        raise ValueError(f"R2 URI must name one object: {uri!r}")
    return bucket, key


def _valid_cache(path: Path, metadata_path: Path, expected: dict[str, object]) -> bool:
    try:
        metadata = json.loads(metadata_path.read_text())
    except OSError, json.JSONDecodeError:
        return False
    if not isinstance(metadata, dict) or any(metadata.get(name) != value for name, value in expected.items()):
        return False
    digest = metadata.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        return False
    try:
        return path.stat().st_size == expected["size"] and _sha256(path) == digest
    except OSError:
        return False


def resolve_checkpoint(source: str, *, cache_root: str | Path = "runs") -> Path:
    """Return a local file, downloading and validating an exact R2 object."""
    if not source.startswith("r2://"):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {path}")
        return path.resolve()

    bucket, key = parse_r2_uri(source)
    client = r2.client()
    remote = client.head_object(Bucket=bucket, Key=key)
    etag = remote.get("ETag")
    size = remote.get("ContentLength")
    if not isinstance(etag, str) or not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise RuntimeError(f"R2 returned invalid identity metadata for {source}")
    cache_key = hashlib.sha256(source.encode()).hexdigest()
    filename = Path(key).name
    cache_dir = Path(cache_root) / "r2-checkpoints" / cache_key
    path = cache_dir / filename
    metadata_path = cache_dir / f"{filename}.metadata.json"
    expected: dict[str, object] = {"uri": source, "etag": etag, "size": size}
    if _valid_cache(path, metadata_path, expected):
        return path.resolve()

    cache_dir.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    digest = hashlib.sha256()
    downloaded = 0
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        response_etag = response.get("ETag")
        if response_etag is not None and response_etag != etag:
            raise RuntimeError(f"R2 object changed while downloading {source}")
        body = response["Body"]
        with contextlib.closing(body), partial.open("wb") as output:
            while chunk := body.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
        if downloaded != size:
            raise RuntimeError(f"R2 object size mismatch for {source}: expected {size}, got {downloaded}")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    metadata = {**expected, "sha256": digest.hexdigest()}
    metadata_partial = metadata_path.with_suffix(metadata_path.suffix + ".partial")
    metadata_partial.write_text(json.dumps(metadata, separators=(",", ":"), sort_keys=True))
    os.replace(metadata_partial, metadata_path)
    return path.resolve()
