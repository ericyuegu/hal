"""Publish a stopped pilot's state to its Modal Volume and R2 backup."""

import dataclasses
import hashlib
import json
import os
import shutil
from pathlib import Path

import modal
import tyro

from hal import r2


@dataclasses.dataclass(frozen=True, slots=True)
class Args:
    source: Path
    destination: Path
    volume: str
    r2_prefix: str


def stage_state(source: Path, destination: Path) -> dict[str, str]:
    if not source.is_dir():
        raise ValueError("pilot publication source directory is missing")
    if (
        source.resolve() == destination.resolve()
        or source.resolve() in destination.resolve().parents
        or destination.resolve() in source.resolve().parents
    ):
        raise ValueError("publication source and destination must be separate trees")
    destination.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    files = [
        path
        for path in source.rglob("*")
        if path.is_file()
        and path.name != "publication.json"
        and not any(
            part in ("tmp", "delivery")
            or (part.startswith(".") and not (part.startswith(".o65-held-out-") and part.endswith(".json")))
            for part in path.relative_to(source).parts
        )
    ]
    # State manifests become visible only after every referenced shard is copied.
    files.sort(
        key=lambda path: (
            path.name != "pilot-identity.json",
            path.name.startswith("update-") and path.suffix == ".json",
            str(path),
        )
    )
    for path in files:
        relative = str(path.relative_to(source))
        target = destination / relative
        with path.open("rb") as stream:
            hashes[relative] = hashlib.file_digest(stream, "sha256").hexdigest()
        if target.is_file():
            with target.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() == hashes[relative]:
                    continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.partial")
        shutil.copyfile(path, temporary)
        temporary.replace(target)
    # Retired manifests and actor files stay in source. Remove compacted resume shards.
    for path in destination.rglob("*"):
        if path.is_file() and str(path.relative_to(destination)) not in hashes and path.name != "publication.json":
            path.unlink()
    return hashes


def main(args: Args) -> None:
    if not args.volume or not args.r2_prefix.startswith("o65/pilot-results/") or ".." in Path(args.r2_prefix).parts:
        raise ValueError("pilot publication needs an explicit Volume and an O65 R2 prefix")
    hashes = stage_state(args.source, args.destination)
    previous_path = args.destination / "publication.json"
    previous = json.loads(previous_path.read_text())["files"] if previous_path.is_file() else {}
    # Commit the complete resume state even if the subsequent R2 backup fails.
    volume = modal.Volume.from_name(args.volume)
    volume.commit()
    client, bucket = r2.client(timeout_seconds=60), r2.bucket()
    for relative, digest in hashes.items():
        if previous.get(relative) != digest:
            client.upload_file(str(args.destination / relative), bucket, f"{args.r2_prefix}/state/{relative}")
    record = {"format": "hal.o65.pilot-publication.v1", "files": hashes}
    temporary = args.destination / ".publication.partial"
    temporary.write_text(json.dumps(record, sort_keys=True))
    temporary.replace(previous_path)
    client.upload_file(str(previous_path), bucket, f"{args.r2_prefix}/publication.json")
    volume.commit()


if __name__ == "__main__":
    main(tyro.cli(Args))
