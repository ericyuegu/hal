"""Run bounded O65 smoke and profiling jobs on one Modal RTX PRO 6000."""

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import subprocess
import time
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Literal

import modal
import tyro
from modal.types import FileEntryType

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/opt/o65-hal")
RESULTS = Path("/o65-results")
VOLUME = "hal-o65-profile"
IMAGE = "ghcr.io/ericyuegu/hal@sha256:c9160a7bb2dc0aa3eb6dc583b1c4753ff98d9a40f802512ca402e2b93a3a7413"
AWS_KEYS = ("AWS_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_BUCKET")
CPU_CORES = 17
MEMORY_MIB = 65536
# Published Modal rates checked 2026-10-19; reserve more for startup and CPU builds.
GPU_USD_SECOND = 0.000842
CPU_USD_SECOND = 0.0000131
MEMORY_USD_GIB_SECOND = 0.00000222
MAX_GPU_SECONDS = 10800
MAX_USD = 30.0
BUILD_RESERVE_USD = 3.0


@dataclasses.dataclass(frozen=True, slots=True)
class Request:
    name: str
    git_sha: str
    mode: Literal["smoke", "profile"] = "profile"
    matches: int = 1024
    windows: int = 16
    decisions: int = 128
    timeout_seconds: int = 1800

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", self.name) is None:
            raise ValueError("invalid Modal run name")
        if re.fullmatch(r"[0-9a-f]{40}", self.git_sha) is None:
            raise ValueError("expected a full published HAL Git SHA")
        if self.mode not in ("smoke", "profile"):
            raise ValueError("unsupported run mode")
        allowed = (16,) if self.mode == "smoke" else (256, 1024, 4096)
        if type(self.matches) is not int or self.matches not in allowed:
            raise ValueError(f"{self.mode} matches must be one of {allowed}")
        if any(type(value) is not int or value < 1 for value in (self.windows, self.decisions)):
            raise ValueError("chunk sizes must be positive integers")
        if type(self.timeout_seconds) is not int or not 60 <= self.timeout_seconds <= 3600:
            raise ValueError("timeout must be 60 to 3600 seconds")


def command(request: Request, cpus: tuple[int, ...]) -> list[str]:
    if len(cpus) != 16 or len(set(cpus)) != 16:
        raise ValueError("O65 requires 16 distinct shard CPUs")
    delivery = REMOTE_ROOT / "delivery"
    argv = [
        "/opt/venv/bin/python",
        "experiments/065_selfplay_ppo.py",
        "train" if request.mode == "smoke" else "profile",
        "--checkpoint",
        str(delivery / "payloads/final.pt"),
        "--data-dir",
        str(delivery / "inputs/msl/data"),
        "--sampler",
        "experiments/o65/pilot-sampler.json",
        "--output",
        str(RESULTS / request.name / "profile"),
        "--cpus",
        *(str(cpu) for cpu in cpus),
        "--matches-per-shard",
        str(request.matches // 16),
        "--groups",
        "1" if request.mode == "smoke" else "4",
        "--replay-chunks.windows",
        str(request.windows),
        "--replay-chunks.decisions",
        str(request.decisions),
    ]
    if request.mode == "smoke":
        argv.extend(
            (
                "--updates",
                "1",
                "--compiled",
                "--training.critic-warmup-updates",
                "0",
                "--training.windows-per-step",
                "32",
                "--evaluation-every",
                "0",
            )
        )
    return argv


def select_cpus(allowed: set[int], sysfs: Path = Path("/sys/devices/system/cpu")) -> tuple[int, ...]:
    physical: dict[tuple[int, int], int] = {}
    for cpu in sorted(allowed):
        topology = sysfs / f"cpu{cpu}/topology"
        key = (int((topology / "physical_package_id").read_text()), int((topology / "core_id").read_text()))
        physical.setdefault(key, cpu)
    if len(physical) < CPU_CORES:
        raise ValueError("Modal job needs at least 17 available physical cores")
    return tuple(physical.values())[:CPU_CORES]


def run_remote(request: Request) -> int:
    output = RESULTS / request.name
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    cpus = select_cpus(set(os.sched_getaffinity(0)))
    os.sched_setaffinity(0, set(cpus))
    argv = command(request, cpus[:16])
    env = dict(os.environ)
    # The existing Secret also holds W&B credentials. This path has no W&B sink.
    env.pop("WANDB_API_KEY", None)
    env.update(
        WANDB_MODE="disabled",
        HAL_NATIVE_LIBRARY=str(REMOTE_ROOT / "delivery/libhal_native.so"),
        TMPDIR=str(output / "tmp"),
        TORCHINDUCTOR_CACHE_DIR=str(output / "inductor"),
        TRITON_CACHE_DIR=str(output / "triton"),
        TORCHINDUCTOR_COMPILE_THREADS="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    Path(env["TMPDIR"]).mkdir()
    identity = {
        "request": dataclasses.asdict(request),
        "command": argv,
        "image": IMAGE,
        "modal_sdk": importlib.metadata.version("modal"),
        "container_id": os.environ.get("MODAL_TASK_ID"),
        "cpus": cpus,
        "started_unix": started,
        "treatment": "active match count or no-grad chunks; see request",
        "control": "1024 active matches, 16 windows and 128 decisions per no-grad chunk",
        "invariants": "seed 65, proxy, sampler, simulator, D/R, reward, optimizer, 16 shards, four groups",
    }
    (output / "launch.json").write_text(json.dumps(identity, indent=2))
    with (output / "topology.txt").open("w") as log:
        subprocess.run(["lscpu", "-e"], stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(["nvidia-smi", "-q"], stdout=log, stderr=subprocess.STDOUT, check=True)
    code = -1
    try:
        with (output / "run.log").open("w") as log:
            result = subprocess.run(
                argv,
                cwd=REMOTE_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=request.timeout_seconds - 30,
            )
        code = result.returncode
        return code
    finally:
        (output / "result.json").write_text(
            json.dumps({"exit_code": code, "started_unix": started, "finished_unix": time.time()}, indent=2)
        )
        modal.Volume.from_name(VOLUME).commit()


def image_commands(sha: str) -> list[str]:
    root = str(REMOTE_ROOT)
    return [
        f"git init {root}",
        f"git -C {root} remote add origin https://github.com/ericyuegu/hal.git",
        f"git -C {root} fetch --depth=1 origin {sha}",
        f"git -C {root} checkout --detach FETCH_HEAD",
        "UV_INDEX_URL=https://pypi.org/simple uv sync --locked",
        "/opt/venv/bin/python -m scripts.o65_delivery --action install --output delivery",
        "bash docker/o65-build-msl.sh /opt/o65-hal/delivery",
        "/opt/venv/bin/python -m scripts.o65_delivery --action verify --output delivery",
    ]


def build_image(sha: str, secret: modal.Secret) -> modal.Image:
    root = str(REMOTE_ROOT)
    return (
        modal.Image.from_registry(IMAGE)
        .entrypoint([])
        .apt_install("build-essential", "binutils")
        .env({"PYTHONPATH": root, "UV_PROJECT_ENVIRONMENT": "/opt/venv"})
        .workdir(root)
        .run_commands(image_commands(sha), secrets=[secret])
        .env({"PYTHONPATH": f"{root}:{root}/delivery/msl"})
    )


def download(name: str, local: Path) -> None:
    volume = modal.Volume.from_name(VOLUME)
    for entry in volume.iterdir(name, recursive=True):
        if entry.type != FileEntryType.FILE:
            continue
        relative = Path(entry.path).relative_to(name)
        # Compile caches are not measurement evidence and can be large.
        if relative.parts[0] in ("inductor", "triton", "tmp"):
            continue
        target = local / name / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as out:
            for chunk in volume.read_file(entry.path):
                out.write(chunk)


@dataclasses.dataclass(frozen=True, slots=True)
class Args:
    mode: Literal["smoke", "profile", "download"] = "profile"
    name: str | None = None
    matches: int = 1024
    windows: int = 16
    decisions: int = 128
    timeout_seconds: int = 1800
    secret: str = "hal"
    output: Path = Path("runs/modal")


def main(args: Args) -> None:
    if args.mode == "download":
        if args.name is None:
            raise ValueError("download requires a run name")
        download(args.name, args.output)
        return
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise ValueError("commit changes before launching published source")
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() != "o65-modal":
        raise ValueError("only branch o65-modal can launch this job")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    published = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/o65-modal"], cwd=ROOT, text=True)
    if published.split()[0] != sha:
        raise ValueError("push o65-modal HEAD before launching")
    name = args.name or f"{args.mode}-{args.matches}-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    request = Request(name, sha, args.mode, args.matches, args.windows, args.decisions, args.timeout_seconds)
    args.output.mkdir(parents=True, exist_ok=True)
    budget = args.output / "budget.jsonl"
    prior = [json.loads(line) for line in budget.read_text().splitlines()] if budget.exists() else []
    # Reserve the full timeout for each call, including failures and client loss.
    allowance = request.timeout_seconds + 360
    reserved = sum(row["reserved_gpu_seconds"] for row in prior) + allowance
    rate = GPU_USD_SECOND + CPU_CORES * CPU_USD_SECOND + (MEMORY_MIB / 1024) * MEMORY_USD_GIB_SECOND
    if reserved > MAX_GPU_SECONDS or reserved * rate + BUILD_RESERVE_USD > MAX_USD:
        raise ValueError("O65 Modal time or cost budget exhausted")
    secret = modal.Secret.from_name(args.secret, required_keys=list(AWS_KEYS))
    secret.hydrate()
    app = modal.App(f"o65-{name}")
    volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
    worker = app.function(
        image=build_image(sha, secret),
        gpu="RTX-PRO-6000",
        cpu=(17.0, 17.0),
        memory=(MEMORY_MIB, MEMORY_MIB),
        ephemeral_disk=131072,
        volumes={str(RESULTS): volume},
        timeout=request.timeout_seconds,
        startup_timeout=300,
        retries=0,
        max_containers=1,
        single_use_containers=True,
        include_source=False,
        serialized=True,
    )(run_remote)
    with budget.open("a") as log:
        log.write(json.dumps({"request": dataclasses.asdict(request), "reserved_gpu_seconds": allowance}) + "\n")
    with modal.enable_output(), app.run():
        print(f"App: {app.app_id}; name: {name}; SDK: {importlib.metadata.version('modal')}", flush=True)
        call = worker.spawn(request)
        (args.output / f"{name}-call.json").write_text(
            json.dumps(
                {"app_id": app.app_id, "call_id": call.object_id, "request": dataclasses.asdict(request)}, indent=2
            )
        )
        print(f"Call: {call.object_id}; argv: {shlex.join(command(request, tuple(range(16))))}", flush=True)
        code = call.get()
    download(name, args.output)
    result = json.loads((args.output / name / "result.json").read_text())
    charged = min(allowance, result["finished_unix"] - result["started_unix"] + 360)
    with budget.open("a") as log:
        log.write(json.dumps({"name": name, "reserved_gpu_seconds": charged - allowance}) + "\n")
    if code:
        raise SystemExit(f"Modal job exited {code}; see {args.output / name / 'run.log'}")
    hashes = {}
    for path in sorted((args.output / name).rglob("*")):
        if path.is_file():
            with path.open("rb") as source:
                hashes[str(path.relative_to(args.output / name))] = hashlib.file_digest(source, "sha256").hexdigest()
    (args.output / name / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2))
    print(f"Downloaded {name} to {args.output / name}")


if __name__ == "__main__":
    main(tyro.cli(Args))
