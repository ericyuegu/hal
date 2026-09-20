"""Queue immutable O59 benchmark jobs on one bounded B200 worker."""

import contextlib
import hashlib
import json
import os
import queue
import re
import signal
import subprocess
import tarfile
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated
from typing import cast

import modal
import tyro

ROOT = Path(__file__).resolve().parents[1]
VOLUME = "hal-o59-benchmarks"
MAX_WORK_SECONDS = 86_100
CASES = (
    "baseline",
    "reuse-history",
    "reuse-embeddings",
    "groupwise-loss",
    "combined",
    "compiled-muon",
    "max-autotune",
    "cpu-validation",
    "no-diagnostics",
)


@dataclass(frozen=True, slots=True)
class StartArgs:
    budget_seconds: int = MAX_WORK_SECONDS
    image: str = "ghcr.io/ericyuegu/hal:cuda13"
    output: Path = Path("results/o59-throughput-launch.json")
    compiler_cache_from: str | None = None


@dataclass(frozen=True, slots=True)
class SubmitArgs:
    launch: Path
    case: str = "baseline"
    bank: str | None = None
    bank_sha256: str | None = None
    reference: str | None = None
    reference_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class FinishArgs:
    launch: Path


@dataclass(frozen=True, slots=True)
class Job:
    format: str
    job_id: str
    git_sha: str
    source_sha256: str
    case: str
    bank: str | None
    bank_sha256: str | None
    reference: str | None
    reference_sha256: str | None


def validate_job(job: Job) -> None:
    if job.format != "o59-benchmark-job-v2" or job.case not in CASES:
        raise ValueError("unsupported benchmark job format or case")
    if uuid.UUID(job.job_id).hex != job.job_id or re.fullmatch(r"[0-9a-f]{40}", job.git_sha) is None:
        raise ValueError("invalid job or Git identity")
    if re.fullmatch(r"[0-9a-f]{64}", job.source_sha256) is None:
        raise ValueError("invalid source hash")
    for name, path, digest in (
        ("bank", job.bank, job.bank_sha256),
        ("reference", job.reference, job.reference_sha256),
    ):
        if (path is None) != (digest is None):
            raise ValueError(f"{name} requires both path and hash")
        if path is not None:
            if Path(path).is_absolute() or ".." in Path(path).parts or Path(path).suffix != ".pt":
                raise ValueError(f"invalid {name} path")
            if digest is None or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError(f"invalid {name} hash")
    if job.case != "baseline" and (job.bank is None or job.reference is None):
        raise ValueError("candidates require a bank and reference identity")
    if (job.bank is None) != (job.reference is None):
        raise ValueError("existing bank and reference must be supplied together")


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def checked_artifact(root: Path, relative: str, digest: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or file_sha256(path) != digest:
        raise ValueError(f"artifact SHA-256 mismatch: {relative}")
    return path


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


def run_child(
    command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path, deadline: float, persist: Callable[[], None]
) -> int:
    """Bound the whole process group and retain reports on failure or cancellation."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("O59 working deadline exhausted")
    with (
        log_path.open("w") as log,
        subprocess.Popen(
            command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        ) as process,
    ):
        try:
            while process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("O59 working deadline exhausted")
                try:
                    process.wait(timeout=min(30, remaining))
                except subprocess.TimeoutExpired:
                    persist()
            return cast(int, process.returncode)
        finally:
            # A failed parent can leave compiler subprocesses alive after exiting.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            persist()


def execute_job(
    job: Job, *, root: Path, launch_id: str, work: Path, cache: Path, deadline: float, persist: Callable[[], None]
) -> int:
    validate_job(job)
    destination = root / launch_id / "jobs" / job.job_id
    source = checked_artifact(destination, "source.tar.gz", job.source_sha256)
    artifacts = {}
    for name, relative, digest in (
        ("bank", job.bank, job.bank_sha256),
        ("reference", job.reference, job.reference_sha256),
    ):
        if relative is not None and digest is not None:
            artifacts[name] = checked_artifact(root, relative, digest)
    source_dir = work / job.job_id
    source_dir.mkdir(parents=True, exist_ok=False)
    with tarfile.open(source) as archive:
        archive.extractall(source_dir, filter="data")
    shared_data = work / "data"
    shared_data.mkdir(exist_ok=True)
    (source_dir / "data").symlink_to(shared_data, target_is_directory=True)
    reports = destination / "reports"
    reports.mkdir(exist_ok=False)
    env = os.environ.copy()
    env.update(
        {
            "HAL_GIT_SHA": job.git_sha,
            "PYTHONPATH": str(source_dir),
            "PYTHONUNBUFFERED": "1",
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
            "TRITON_CACHE_DIR": str(cache / "triton"),
            "OMP_NUM_THREADS": "8",
        }
    )
    command = [
        "/opt/venv/bin/python",
        "experiments/059_muon_history_decoder.py",
        "benchmark",
        "--case",
        job.case,
        "--batch-size",
        "512",
        "--output",
        str(reports),
    ]
    if artifacts:
        command.extend(
            [
                "--existing-bank",
                str(artifacts["bank"]),
                "--bank-sha256",
                cast(str, job.bank_sha256),
                "--reference-state",
                str(artifacts["reference"]),
                "--reference-sha256",
                cast(str, job.reference_sha256),
            ]
        )
    return run_child(
        command, cwd=source_dir, env=env, log_path=destination / "child.log", deadline=deadline, persist=persist
    )


def serve_jobs(
    *,
    receive: Callable[[float], object],
    execute: Callable[[Job], int],
    record: Callable[[dict[str, object]], None],
    deadline: float,
) -> None:
    """Consume FIFO jobs until the finish marker or the working deadline."""
    while time.monotonic() < deadline:
        try:
            message = receive(min(30, deadline - time.monotonic()))
        except queue.Empty:
            continue
        if message == {"finish": True}:
            return
        if not isinstance(message, dict):
            raise ValueError("benchmark queue message must be a manifest")
        job = Job(**cast(dict, message))
        result: dict[str, object] = {"job_id": job.job_id, "started_at": time.time()}
        try:
            validate_job(job)
            result["exit_code"] = execute(job)
            result["status"] = "complete" if result["exit_code"] == 0 else "failed"
        except InterruptedError:
            result["status"] = "cancelled"
            raise
        except (OSError, ValueError, TimeoutError, tarfile.TarError) as error:
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
        finally:
            result["finished_at"] = time.time()
            record(result)
    raise TimeoutError("O59 working deadline exhausted")


def run_worker(launch_id: str, budget_seconds: int, compiler_cache_from: str | None) -> None:
    started = time.monotonic()
    deadline = started + budget_seconds
    root = Path("/benchmark-results")
    destination = root / launch_id
    destination.mkdir(parents=True, exist_ok=True)
    work = Path("/opt/o59-jobs")
    work.mkdir(parents=True, exist_ok=True)
    cache = Path("/opt/hal-cache")
    cache.mkdir(parents=True, exist_ok=True)
    volume = modal.Volume.from_name(VOLUME)
    jobs = modal.Queue.from_name(f"hal-o59-{launch_id}")
    if compiler_cache_from is not None:
        restore_compiler_cache(root / compiler_cache_from, cache)

    def execute(job: Job) -> int:
        volume.reload()
        return execute_job(
            job, root=root, launch_id=launch_id, work=work, cache=cache, deadline=deadline, persist=volume.commit
        )

    def record(result: dict[str, object]) -> None:
        with (destination / "worker-results.jsonl").open("a") as handle:
            handle.write(json.dumps(result) + "\n")
        volume.commit()

    def interrupted(signum: int, _frame: object) -> None:
        raise InterruptedError(f"benchmark worker received signal {signum}")

    previous_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        serve_jobs(
            receive=lambda timeout: jobs.get(timeout=timeout), execute=execute, record=record, deadline=deadline
        )
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        save_compiler_cache(cache, destination)
        (destination / "worker-finished.json").write_text(
            json.dumps({"elapsed_seconds": time.monotonic() - started}) + "\n"
        )
        volume.commit()


def start(args: StartArgs) -> None:
    if not 1 <= args.budget_seconds <= MAX_WORK_SECONDS:
        raise ValueError("budget_seconds must be in [1, 86100]")
    if args.compiler_cache_from is not None and uuid.UUID(args.compiler_cache_from).hex != args.compiler_cache_from:
        raise ValueError("compiler_cache_from must be a normalized launch UUID")
    if args.output.exists():
        raise FileExistsError(args.output)
    launch_id = uuid.uuid4().hex
    jobs = modal.Queue.from_name(f"hal-o59-{launch_id}", create_if_missing=True)
    jobs.hydrate()
    app = modal.App("hal-o59-throughput", tags={"launch_id": launch_id})
    function = app.function(
        image=modal.Image.from_registry(args.image),
        gpu="B200",
        cpu=32,
        memory=128 * 1024,
        ephemeral_disk=512 * 1024,
        timeout=86_400,
        retries=0,
        max_containers=1,
        single_use_containers=True,
        secrets=[modal.Secret.from_name("hal")],
        volumes={"/benchmark-results": modal.Volume.from_name(VOLUME, create_if_missing=True)},
        serialized=True,
        include_source=False,
    )(run_worker)
    with modal.enable_output(), app.run(detach=True):
        call = function.spawn(launch_id, args.budget_seconds, args.compiler_cache_from)
        record = {
            "format": "o59-benchmark-launch-v2",
            "app_id": app.app_id,
            "call_id": call.object_id,
            "launch_id": launch_id,
            "queue": f"hal-o59-{launch_id}",
            "volume": VOLUME,
            "dashboard": f"https://modal.com/apps/{app.app_id}",
            "closed": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2), flush=True)


def launch_record(path: Path) -> dict:
    record = json.loads(path.read_text())
    if record.get("format") != "o59-benchmark-launch-v2" or record.get("closed") is not False:
        raise ValueError("launch record is incompatible or closed")
    if (
        uuid.UUID(record["launch_id"]).hex != record["launch_id"]
        or record["queue"] != f"hal-o59-{record['launch_id']}"
    ):
        raise ValueError("launch queue identity mismatch")
    return record


def submit(args: SubmitArgs) -> None:
    record = launch_record(args.launch)
    job_id = uuid.uuid4().hex
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    local = ROOT / "results" / "o59-submissions" / job_id
    local.mkdir(parents=True)
    source = local / "source.tar.gz"
    paths = [
        *sorted((ROOT / "hal").rglob("*.py")),
        ROOT / "experiments/059_muon_history_decoder.py",
        ROOT / "scripts/benchmark_o59_modal.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]
    with tarfile.open(source, "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    job = Job(
        "o59-benchmark-job-v2",
        job_id,
        git_sha,
        file_sha256(source),
        args.case,
        args.bank,
        args.bank_sha256,
        args.reference,
        args.reference_sha256,
    )
    validate_job(job)
    manifest = local / "manifest.json"
    manifest.write_text(json.dumps(asdict(job), indent=2) + "\n")
    remote = f"/{record['launch_id']}/jobs/{job_id}"
    volume = modal.Volume.from_name(VOLUME)
    with volume.batch_upload() as upload:
        upload.put_file(source, f"{remote}/source.tar.gz")
        upload.put_file(manifest, f"{remote}/manifest.json")
    modal.Queue.from_name(record["queue"]).put(asdict(job))
    print(json.dumps({"job": asdict(job), "reports": f"{remote}/reports"}, indent=2), flush=True)


def finish(args: FinishArgs) -> None:
    record = launch_record(args.launch)
    modal.Queue.from_name(record["queue"]).put({"finish": True})
    record["closed"] = True
    args.launch.write_text(json.dumps(record, indent=2) + "\n")
    print("Finish marker queued; the worker will persist reports and caches after preceding jobs.", flush=True)


type Command = (
    Annotated[StartArgs, tyro.conf.subcommand(name="start")]
    | Annotated[SubmitArgs, tyro.conf.subcommand(name="submit")]
    | Annotated[FinishArgs, tyro.conf.subcommand(name="finish")]
)


def main(args: Command) -> None:
    if isinstance(args, StartArgs):
        start(args)
    elif isinstance(args, SubmitArgs):
        submit(args)
    else:
        finish(args)


if __name__ == "__main__":
    main(tyro.cli(cast(type[Command], Command)))
