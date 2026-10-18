"""Publish a hash-checked result archive, including after a failed VM job.

The boot script also supplies hal.r2 when the HAL checkout fails. Input
payloads, scratch files and the upload staging directory stay on the VM.
"""

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path

from hal import r2


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def validate_prefix(prefix: str) -> None:
    if re.fullmatch(r"o65/pilot-results/[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*", prefix) is None:
        raise ValueError("results prefix must be a nonempty child of o65/pilot-results")


def archive_results(root: Path) -> tuple[Path, Path]:
    staging = root / ".upload"
    staging.mkdir(exist_ok=True)
    archive = staging / "results.tar"
    records = []
    with tarfile.open(archive, "w") as target:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if any(part in (".upload", "tmp", "delivery") for part in relative.parts):
                continue
            if path.is_symlink():
                raise ValueError(f"result contains a symlink: {relative}")
            if not path.is_file():
                continue
            records.append({"path": relative.as_posix(), "sha256": sha256(path), "size_bytes": path.stat().st_size})
            target.add(path, arcname=relative.as_posix(), recursive=False)
    manifest = staging / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "hal.o65.results.v1",
                "archive_sha256": sha256(archive),
                "archive_size_bytes": archive.stat().st_size,
                "files": records,
            },
            indent=2,
        )
        + "\n"
    )
    return archive, manifest


def upload(root: Path, prefix: str) -> None:
    validate_prefix(prefix)
    archive, manifest = archive_results(root)
    client = r2.client(timeout_seconds=30)
    client.upload_file(str(archive), r2.bucket(), f"{prefix}/results.tar")
    # Publication is last: an absent manifest identifies an incomplete upload.
    client.upload_file(str(manifest), r2.bucket(), f"{prefix}/manifest.json")
    print(f"Published R2 {prefix}/manifest.json", flush=True)


def fetch(root: Path, prefix: str) -> None:
    validate_prefix(prefix)
    root.mkdir(parents=True, exist_ok=False)
    client = r2.client(timeout_seconds=30)
    manifest = root / "manifest.json"
    archive = root / "results.tar"
    client.download_file(r2.bucket(), f"{prefix}/manifest.json", str(manifest))
    client.download_file(r2.bucket(), f"{prefix}/results.tar", str(archive))
    unpack_results(archive, manifest, root / "results")


def unpack_results(archive: Path, manifest: Path, output: Path) -> None:
    raw = json.loads(manifest.read_text())
    if (
        not isinstance(raw, dict)
        or set(raw) != {"format", "archive_sha256", "archive_size_bytes", "files"}
        or raw["format"] != "hal.o65.results.v1"
        or not isinstance(raw["files"], list)
    ):
        raise ValueError("unsupported result manifest")
    records: list[tuple[str, str, int]] = []
    for record in raw["files"]:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "size_bytes"}:
            raise ValueError("invalid result file record")
        name, digest, size = record["path"], record["sha256"], record["size_bytes"]
        if (
            not isinstance(name, str)
            or Path(name).is_absolute()
            or any(part in ("", ".", "..") for part in name.split("/"))
            or not isinstance(digest, str)
            or re.fullmatch("[0-9a-f]{64}", digest) is None
            or type(size) is not int
            or size < 0
        ):
            raise ValueError("invalid result file identity")
        records.append((name, digest, size))
    names = [name for name, _, _ in records]
    if len(names) != len(set(names)):
        raise ValueError("duplicate result file")
    verify_archive(archive, raw["archive_sha256"], raw["archive_size_bytes"])
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as source:
        members = source.getmembers()
        if (
            any(not member.isfile() for member in members)
            or len(members) != len(names)
            or {member.name for member in members} != set(names)
        ):
            raise ValueError("result archive differs from the file manifest")
        source.extractall(output, filter="data")
    for name, digest, size in records:
        path = output / name
        if sha256(path) != digest or path.stat().st_size != size:
            raise ValueError(f"result file mismatch: {path}")


def verify_archive(path: Path, expected_sha256: str, expected_size: int) -> None:
    if path.stat().st_size != expected_size or sha256(path) != expected_sha256:
        raise ValueError("result archive hash or size mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upload", "fetch"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    if args.action == "upload":
        upload(args.root, args.prefix)
    else:
        fetch(args.root, args.prefix)


if __name__ == "__main__":
    main()
