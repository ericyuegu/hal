"""Run the bounded O59 suite without rebuilding dependencies or fetching Dolphin.

Mount an exact source snapshot onto the existing CUDA image. Persist source,
logs, data identity, and reports in a dedicated Volume, outside training runs.
"""

import hashlib
import json
import os
import signal
import subprocess
import tarfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import modal
import tyro

ROOT = Path(__file__).resolve().parents[1]
VOLUME = "hal-o59-benchmarks"


@dataclass(frozen=True, slots=True)
class Args:
    budget_seconds: int = 6900
    image: str = "ghcr.io/ericyuegu/hal:cuda13"
    output: Path = Path("results/o59-throughput-launch.json")
    compiler_cache_from: str | None = None


def save_compiler_cache(cache: Path, destination: Path) -> None:
    temporary = destination / "compiler-cache.tar.gz.tmp"
    with tarfile.open(temporary, "w:gz", compresslevel=1) as archive:
        archive.add(cache, arcname=".")
    target = destination / "compiler-cache.tar.gz"
    temporary.replace(target)
    with target.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    (destination / "compiler-cache.json").write_text(
        json.dumps({"format": "o59-compiler-cache-v1", "sha256": digest}) + "\n"
    )


def restore_compiler_cache(source: Path, destination: Path) -> None:
    metadata = json.loads((source / "compiler-cache.json").read_text())
    if metadata.get("format") != "o59-compiler-cache-v1":
        raise ValueError("unsupported compiler cache archive format")
    path = source / "compiler-cache.tar.gz"
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != metadata["sha256"]:
        raise ValueError("compiler cache archive SHA-256 mismatch")
    with tarfile.open(path) as archive:
        archive.extractall(destination, filter="data")


def run_suite(
    launch_id: str, git_sha: str, source_sha256: str, budget_seconds: int, compiler_cache_from: str | None
) -> int:
    """Own one GPU allocation, including preparation, compilation, and cleanup."""
    started = time.monotonic()
    Path("/opt/hal").mkdir(parents=True, exist_ok=True)
    os.chdir("/opt/hal")
    destination = Path("/benchmark-results") / launch_id
    destination.mkdir()
    if compiler_cache_from is not None:
        restore_compiler_cache(Path("/benchmark-results") / compiler_cache_from, Path("/opt/hal-cache"))
    source = Path("/opt/o59-source.tar.gz")
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_sha256:
        raise ValueError("mounted benchmark source archive has changed")
    with tarfile.open(source) as archive:
        archive.extractall("/opt/hal", filter="data")
    (destination / source.name).write_bytes(source.read_bytes())
    metadata: dict[str, object] = {
        "launch_id": launch_id,
        "git_sha": git_sha,
        "source_sha256": source_sha256,
        "budget_seconds": budget_seconds,
        "container_id": os.environ.get("MODAL_TASK_ID"),
        "compiler_cache_from": compiler_cache_from,
    }
    (destination / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
    env = os.environ.copy()
    env.update(
        {
            "HAL_GIT_SHA": git_sha,
            "PYTHONPATH": "/opt/hal",
            "PYTHONUNBUFFERED": "1",
            "TORCHINDUCTOR_CACHE_DIR": "/opt/hal-cache/inductor",
            "TRITON_CACHE_DIR": "/opt/hal-cache/triton",
            "OMP_NUM_THREADS": "8",
        }
    )
    command = [
        "/opt/venv/bin/python",
        "experiments/059_muon_history_decoder.py",
        "benchmark",
        "--output",
        str(destination / "suite"),
        "--budget-seconds",
        str(budget_seconds),
    ]
    volume = modal.Volume.from_name(VOLUME)
    with subprocess.Popen(command, env=env, start_new_session=True) as process:
        try:
            while process.poll() is None:
                remaining = budget_seconds + 60 - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("O59 GPU budget exhausted")
                try:
                    process.wait(timeout=min(30, remaining))
                except subprocess.TimeoutExpired:
                    volume.commit()
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if Path("/opt/hal-cache").exists():
                save_compiler_cache(Path("/opt/hal-cache"), destination)
            metadata["elapsed_seconds"] = time.monotonic() - started
            metadata["exit_code"] = process.returncode
            (destination / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
            volume.commit()
    return process.returncode


def main(args: Args) -> None:
    if not 1 <= args.budget_seconds <= 6900:
        raise ValueError("budget_seconds must be in [1, 6900]")
    if args.compiler_cache_from is not None and uuid.UUID(args.compiler_cache_from).hex != args.compiler_cache_from:
        raise ValueError("compiler_cache_from must be a normalized launch UUID")
    launch_id = uuid.uuid4().hex
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    snapshot = ROOT / "results" / f"o59-source-{launch_id}.tar.gz"
    snapshot.parent.mkdir(exist_ok=True)
    paths = [
        *sorted((ROOT / "hal").rglob("*.py")),
        ROOT / "experiments/059_muon_history_decoder.py",
        ROOT / "scripts/benchmark_o59_modal.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]
    with tarfile.open(snapshot, "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    image = modal.Image.from_registry(args.image).add_local_file(snapshot, "/opt/o59-source.tar.gz", copy=False)
    app = modal.App("hal-o59-throughput", tags={"git_sha": git_sha, "launch_id": launch_id})
    function = app.function(
        image=image,
        gpu="B200",
        cpu=32,
        memory=128 * 1024,
        ephemeral_disk=512 * 1024,
        timeout=7100,
        retries=0,
        max_containers=1,
        single_use_containers=True,
        secrets=[modal.Secret.from_name("hal")],
        volumes={"/benchmark-results": modal.Volume.from_name(VOLUME, create_if_missing=True)},
        serialized=True,
        include_source=False,
    )(run_suite)
    with modal.enable_output(), app.run(detach=True):
        call = function.spawn(launch_id, git_sha, digest, args.budget_seconds, args.compiler_cache_from)
        record = {
            "app_id": app.app_id,
            "call_id": call.object_id,
            "launch_id": launch_id,
            "git_sha": git_sha,
            "source_sha256": digest,
            "volume": VOLUME,
            "dashboard": f"https://modal.com/apps/{app.app_id}",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
