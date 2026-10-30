"""Publish complete pilot snapshots to the Modal Volume and R2 backup."""

import dataclasses
import hashlib
import json
import os
import shutil
import tempfile
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

from hal import r2
from hal.eval.self_play_checkpoint import read_training_manifest


@dataclasses.dataclass(frozen=True, slots=True)
class Args:
    source: Path
    destination: Path
    volume: str
    r2_prefix: str
    compact: bool = True


def compact_checkpoints(directory: Path, latest: int) -> None:
    """Keep complete resume state for the latest update.

    Retain every actor and move retired manifests into a separate history
    directory before removing their unreferenced training, simulator and league files.
    Call only at a checkpoint boundary, before the next save starts.
    """
    retained: set[str] = set()
    retired = directory.parent / "retired-checkpoints"
    parsed = [(path, read_training_manifest(path)) for path in sorted(directory.glob("update-*.json"))]
    if latest not in {manifest["update"] for _, manifest in parsed}:
        raise ValueError("latest checkpoint is missing")
    for path, manifest in parsed:
        if path.name != f"update-{manifest['update']:08d}.json":
            raise ValueError("checkpoint filename differs from its update")
        if (retired / path.name).exists():
            raise FileExistsError(retired / path.name)
    for path, manifest in parsed:
        if manifest["update"] == latest:
            retained.update(shard["name"] for shard in manifest["shards"])
        else:
            retired.mkdir(exist_ok=True)
            target = retired / path.name
            path.rename(target)
    for pattern in ("*.rank0.pt", "*.sim.zlib", "*.league.pt"):
        for path in directory.glob(pattern):
            if path.name not in retained:
                path.unlink()


def state_files(source: Path) -> list[Path]:
    return [
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
    files = state_files(source)
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


class CheckpointPublisher:
    """Keep at most one active publication and one queued snapshot.

    The trainer freezes local metadata at an update boundary. Immutable shards
    use hard links so publication and local compaction cannot race a later save.
    Network copies and hashing run on the worker, without touching trainer state.
    """

    def __init__(self, args: Args) -> None:
        validate_args(args)
        self.args = args
        self.pending: list[Future[None]] = []
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpoint-publication")

    def check(self) -> None:
        pending = []
        for future in self.pending:
            if future.done():
                future.result()
            else:
                pending.append(future)
        self.pending = pending

    def capture_metadata(self) -> dict[Path, bytes]:
        """Freeze mutable logs and settings while the trainer is at a boundary."""
        return {
            path.relative_to(self.args.source): path.read_bytes()
            for path in state_files(self.args.source)
            if path.suffix != ".pt"
            and not path.name.endswith(".sim.zlib")
            and "checkpoints" not in path.relative_to(self.args.source).parts
            and "retired-checkpoints" not in path.relative_to(self.args.source).parts
        }

    def submit(self, checkpoint: Path, *, metadata: dict[Path, bytes] | None = None) -> None:
        self.check()
        if len(self.pending) >= 2:
            raise RuntimeError("checkpoint publication queue is full")
        if checkpoint.parent.resolve() != (self.args.source / "train/checkpoints").resolve():
            raise ValueError("publication checkpoint must belong to the pilot training directory")
        update = read_training_manifest(checkpoint)["update"]
        if self.args.compact:
            compact_checkpoints(checkpoint.parent, update)
        temporary = self.args.source / "tmp"
        temporary.mkdir(exist_ok=True)
        snapshot = tempfile.TemporaryDirectory(prefix=f"publication-{update}-", dir=temporary)
        submitted = False
        try:
            root = Path(snapshot.name)
            for source in state_files(self.args.source):
                target = root / source.relative_to(self.args.source)
                target.parent.mkdir(parents=True, exist_ok=True)
                if source.suffix == ".pt" or source.name.endswith(".sim.zlib"):
                    os.link(source, target)
                elif (
                    metadata is None
                    or "checkpoints" in source.relative_to(self.args.source).parts
                    or "retired-checkpoints" in source.relative_to(self.args.source).parts
                ):
                    shutil.copyfile(source, target)
                elif source.relative_to(self.args.source) in metadata:
                    target.write_bytes(metadata[source.relative_to(self.args.source)])
            self.pending.append(self.executor.submit(publish_snapshot, self.args, snapshot))
            submitted = True
        finally:
            if not submitted:
                snapshot.cleanup()

    def wait(self) -> None:
        for future in self.pending:
            future.result()
        self.pending.clear()

    def close(self) -> None:
        try:
            self.wait()
        finally:
            self.executor.shutdown(wait=True)


def publish_snapshot(args: Args, snapshot: tempfile.TemporaryDirectory[str]) -> None:
    with snapshot as directory:
        publish(dataclasses.replace(args, source=Path(directory)))


def validate_args(args: Args) -> None:
    if not args.volume or not args.r2_prefix.startswith("o65/pilot-results/") or ".." in Path(args.r2_prefix).parts:
        raise ValueError("pilot publication needs an explicit Volume and an O65 R2 prefix")
    if (
        args.source.resolve() == args.destination.resolve()
        or args.source.resolve() in args.destination.resolve().parents
        or args.destination.resolve() in args.source.resolve().parents
    ):
        raise ValueError("publication source and destination must be separate trees")


def publish(args: Args) -> None:
    validate_args(args)
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
