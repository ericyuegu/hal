"""Prepare fixed O65 input archives, or download and verify them for the pilot.

Read expected hashes from HAL's committed manifest, verify each downloaded
file, then extract regular files and directories. Prepare writes local files.
"""

import dataclasses
import hashlib
import json
import re
import subprocess
import tarfile
from pathlib import Path
from typing import Literal

import tyro

from hal import r2

MSL_COMMIT = "2dbca5e3214d88859ac80333cc19caa328d4be4a"
MSL_REPOSITORY = "https://github.com/ericyuegu/melee-sim-light.git"
PROXY_KEY = (
    "runs/261017-081127_063_percentile_conditioning_o63v1-bootstrap30-d256-L12-h4-c256-t256x6-ff1024-"
    "dual-nonlinear-head-rc1e128-drop0.2-muon-wsd-b150-wmax10-g0.99855__o63-ranked1-bootstrap30/final.pt"
)
PROXY_SHA256 = "65e7e43e5cbb18a365a005922ec1cfd026ad807c045eae13b331f487b4ec82e6"
STATS_SHA256 = "6870bda6c0970826f6467407647cf9bb5901ef246b6947d87518db67252cdde3"
STATS_PATH = Path("data/processed/ranked-anonymized-1/mds-policy-world-v8/stats.json")
SAMPLER_PATH = Path("experiments/o65/pilot-sampler.json")
SAMPLER_SOURCE_PATH = Path("experiments/o65/pilot-sampler-source.json")


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _digest(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("invalid SHA-256")


@dataclasses.dataclass(frozen=True, slots=True)
class Blob:
    key: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _digest(self.sha256)
        if not isinstance(self.key, str) or not self.key or any(p in ("", ".", "..") for p in self.key.split("/")):
            raise ValueError("invalid R2 key")
        if type(self.size_bytes) is not int or self.size_bytes < 1:
            raise ValueError("invalid artifact size")


@dataclasses.dataclass(frozen=True, slots=True)
class Manifest:
    format: str
    msl_commit: str
    msl_repository: str
    source_sha256: str
    inputs: Blob
    checkpoint: Blob
    game_manifest_sha256: str
    statistics_sha256: str
    sampler_sha256: str
    sampler_source_sha256: str

    def __post_init__(self) -> None:
        if self.format != "hal.o65.delivery.v2":
            raise ValueError("unsupported O65 delivery format")
        for value in (
            self.source_sha256,
            self.game_manifest_sha256,
            self.statistics_sha256,
            self.sampler_sha256,
            self.sampler_source_sha256,
        ):
            _digest(value)


def read_manifest(path: Path) -> Manifest:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or set(raw) != {f.name for f in dataclasses.fields(Manifest)}:
        raise ValueError("invalid O65 delivery manifest fields")
    for name in ("inputs", "checkpoint"):
        value = raw[name]
        if not isinstance(value, dict) or set(value) != {f.name for f in dataclasses.fields(Blob)}:
            raise ValueError("invalid O65 blob fields")
        raw[name] = Blob(**value)
    return Manifest(**raw)


def verify_blob(path: Path, blob: Blob) -> None:
    if path.stat().st_size != blob.size_bytes or sha256(path) != blob.sha256:
        raise ValueError(f"artifact hash or size mismatch: {path}")


def extract(archive: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    with tarfile.open(archive) as source:
        members = source.getmembers()
        if any(not (m.isfile() or m.isdir()) for m in members):
            raise ValueError("archive contains a link or special file")
        source.extractall(destination, members=members, filter="data")


def prepare(msl: Path, root: Path, checkpoint: Path, output: Path, manifest_path: Path) -> None:
    if sha256(checkpoint) != PROXY_SHA256 or sha256(root / STATS_PATH) != STATS_SHA256:
        raise ValueError("local proxy or statistics hash mismatch")
    # A commit archive deliberately excludes build outputs and uncommitted source.
    commit = subprocess.check_output(
        ["git", "-C", str(msl), "rev-parse", "o65-hal-parity^{commit}"], text=True
    ).strip()
    if commit != MSL_COMMIT:
        raise ValueError("MSL o65-hal-parity no longer names the qualified commit")
    provenance = json.loads((root / SAMPLER_SOURCE_PATH).read_text())
    if provenance.get("format") != "hal.o65.sampler-source.v2":
        raise ValueError("unsupported sampler count provenance")
    for source in provenance["sources"].values():
        if sha256(root / source["manifest"]) != source["sha256"]:
            raise ValueError("sampler count source hash mismatch")
    output.mkdir(parents=True, exist_ok=False)
    source = output / "msl-source.tar"
    subprocess.run(
        ["git", "-C", str(msl), "archive", "--format=tar", f"--output={source.resolve()}", commit], check=True
    )
    inputs = output / "inputs.tar"
    with tarfile.open(inputs, "w") as archive:
        files = [(path, Path("msl/data/raw") / path.name) for path in sorted((msl / "data/raw").iterdir())]
        files.append((root / STATS_PATH, Path("hal") / STATS_PATH))
        for path, name in files:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"expected regular input file: {path}")
            info = tarfile.TarInfo(name.as_posix())
            info.size, info.mode, info.mtime = path.stat().st_size, 0o644, 0
            with path.open("rb") as data:
                archive.addfile(info, data)
    source_hash, inputs_hash = sha256(source), sha256(inputs)
    manifest = Manifest(
        "hal.o65.delivery.v2",
        commit,
        MSL_REPOSITORY,
        source_hash,
        Blob(f"o65/pilot/v1/inputs-{inputs_hash}.tar", inputs_hash, inputs.stat().st_size),
        Blob(PROXY_KEY, PROXY_SHA256, checkpoint.stat().st_size),
        sha256(msl / "data/raw/manifest.json"),
        STATS_SHA256,
        sha256(root / SAMPLER_PATH),
        sha256(root / SAMPLER_SOURCE_PATH),
    )
    manifest_path.write_text(json.dumps(dataclasses.asdict(manifest), indent=2) + "\n")
    (output / "SHA256SUMS").write_text(f"{source_hash}  msl-source.tar\n{inputs_hash}  inputs.tar\n")


def verify_source(msl: Path, expected_commit: str, expected_sha256: str, archive: Path) -> None:
    commit = subprocess.check_output(["git", "-C", str(msl), "rev-parse", "HEAD"], text=True).strip()
    if commit != expected_commit:
        raise ValueError("cloned MSL commit mismatch")
    if subprocess.check_output(["git", "-C", str(msl), "status", "--porcelain", "--untracked-files=no"]):
        raise ValueError("MSL tracked source is dirty")
    subprocess.run(
        ["git", "-C", str(msl), "archive", "--format=tar", f"--output={archive.resolve()}", commit], check=True
    )
    if sha256(archive) != expected_sha256:
        raise ValueError("cloned MSL source archive mismatch")


def install(manifest_path: Path, root: Path, output: Path, local_payloads: Path | None = None) -> None:
    manifest = read_manifest(manifest_path)
    if (
        sha256(root / SAMPLER_PATH) != manifest.sampler_sha256
        or sha256(root / SAMPLER_SOURCE_PATH) != manifest.sampler_source_sha256
    ):
        raise ValueError("sampler identity mismatch")
    output.mkdir(parents=True, exist_ok=False)
    cache = output / "payloads"
    cache.mkdir()
    for name, blob in (
        ("inputs.tar", manifest.inputs),
        ("final.pt", manifest.checkpoint),
    ):
        path = cache / name
        if local_payloads is None:
            r2.client().download_file(r2.bucket(), blob.key, str(path))
        else:
            # A hard link keeps local validation read-only and avoids another checkpoint copy.
            path.hardlink_to(local_payloads / name)
        verify_blob(path, blob)
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--no-checkout",
            "--single-branch",
            "--branch",
            "o65-hal-parity",
            manifest.msl_repository,
            str(output / "msl"),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(output / "msl"), "checkout", "--quiet", "--detach", manifest.msl_commit], check=True
    )
    verify_source(output / "msl", manifest.msl_commit, manifest.source_sha256, cache / "msl-source.tar")
    extract(cache / "inputs.tar", output / "inputs")
    stats = output / "inputs/hal" / STATS_PATH
    data = output / "inputs/msl/data"
    if (
        sha256(stats) != manifest.statistics_sha256
        or sha256(data / "raw/manifest.json") != manifest.game_manifest_sha256
    ):
        raise ValueError("extracted input identity mismatch")
    # Preserve the source-relative location used by the strict proxy loader.
    target = root / STATS_PATH
    if target.exists():
        if sha256(target) != manifest.statistics_sha256:
            raise ValueError("existing feature statistics mismatch")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(stats.read_bytes())
    (output / "delivery.json").write_bytes(manifest_path.read_bytes())


def verify_installation(manifest_path: Path, root: Path, output: Path) -> None:
    manifest = read_manifest(manifest_path)
    if (output / "delivery.json").read_bytes() != manifest_path.read_bytes():
        raise ValueError("installed delivery differs from the committed manifest")
    for name, blob in (("inputs.tar", manifest.inputs), ("final.pt", manifest.checkpoint)):
        verify_blob(output / "payloads" / name, blob)
    for path, expected in (
        (root / SAMPLER_PATH, manifest.sampler_sha256),
        (root / SAMPLER_SOURCE_PATH, manifest.sampler_source_sha256),
        (root / STATS_PATH, manifest.statistics_sha256),
        (output / "inputs/hal" / STATS_PATH, manifest.statistics_sha256),
        (output / "inputs/msl/data/raw/manifest.json", manifest.game_manifest_sha256),
    ):
        if sha256(path) != expected:
            raise ValueError(f"installed input identity mismatch: {path}")
    verify_source(output / "msl", manifest.msl_commit, manifest.source_sha256, output / "payloads/msl-source.tar")


def check_remote(manifest_path: Path) -> None:
    """Check availability before allocation; install still verifies the downloaded bytes."""
    manifest = read_manifest(manifest_path)
    client = r2.client(timeout_seconds=30)
    for blob in (manifest.inputs, manifest.checkpoint):
        if client.head_object(Bucket=r2.bucket(), Key=blob.key)["ContentLength"] != blob.size_bytes:
            raise ValueError(f"remote artifact size mismatch: {blob.key}")


@dataclasses.dataclass(frozen=True, slots=True)
class Args:
    action: Literal["prepare", "install", "verify", "check-remote"]
    output: Path | None = None
    manifest: Path = Path("experiments/o65/pilot-inputs.json")
    root: Path = Path(".")
    msl: Path = Path.home() / "src/melee-sim-light"
    checkpoint: Path = Path("runs/proxy/final.pt")
    local_payloads: Path | None = None


def main(args: Args) -> None:
    if args.action == "check-remote":
        check_remote(args.manifest)
        return
    if args.output is None:
        raise ValueError("--output is required for prepare, install and verify")
    if args.action == "prepare":
        prepare(args.msl, args.root, args.checkpoint, args.output, args.manifest)
    elif args.action == "install":
        install(args.manifest, args.root, args.output, args.local_payloads)
    else:
        verify_installation(args.manifest, args.root, args.output)


if __name__ == "__main__":
    main(tyro.cli(Args))
