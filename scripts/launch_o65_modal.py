"""Run bounded O65 smoke and profiling jobs on one Modal RTX PRO 6000."""

import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Literal
from typing import TextIO

import modal
import tyro
from modal.types import FileEntryType

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/opt/o65-hal")
RESULTS = Path("/o65-results")
WORK = Path("/opt/o65-work")
VOLUME = "hal-o65-profile"
IMAGE = "ghcr.io/ericyuegu/hal@sha256:c9160a7bb2dc0aa3eb6dc583b1c4753ff98d9a40f802512ca402e2b93a3a7413"
AWS_KEYS = ("AWS_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_BUCKET")
CPU_CORES = 17
MEMORY_MIB = 65536
# Published Modal rates checked 2026-10-19; reserve more for startup and CPU builds.
GPU_USD_SECOND = 0.000842
CPU_USD_SECOND = 0.0000131
MEMORY_USD_GIB_SECOND = 0.00000222
MAX_GPU_SECONDS = 7200
MAX_USD = 20.0
BUILD_RESERVE_USD = 3.0
BUDGET_FORMAT = "hal.o65.modal-budget.v1"
RESULT_FORMAT = "hal.o65.modal-result.v1"
MODAL_SDK = "1.5.3"
NSYS_VERSION_CHECK = 'test "$(nsys --version)" = "NVIDIA Nsight Systems version 2026.5.1.161-265138896106v0"'


def require_modal_version(version: str) -> None:
    if version != MODAL_SDK:
        raise ValueError(f"Modal {version} has not passed this launcher's compatibility tests; expected {MODAL_SDK}")


def validate_name(name: str) -> None:
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", name) is None:
        raise ValueError("invalid Modal run name")


def reserved_seconds(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0.0
    for line in path.read_text().splitlines():
        raw = json.loads(line)
        if (
            not isinstance(raw, dict)
            or set(raw) != {"format", "name", "reserved_gpu_seconds", "reason"}
            or raw["format"] != BUDGET_FORMAT
            or not isinstance(raw["name"], str)
            or not isinstance(raw["reason"], str)
            or not raw["reason"]
        ):
            raise ValueError("unsupported Modal budget record")
        validate_name(raw["name"])
        seconds = raw["reserved_gpu_seconds"]
        if type(seconds) not in (int, float) or not math.isfinite(seconds):
            raise ValueError("invalid Modal budget duration")
        total += seconds
        if total < 0:
            raise ValueError("Modal budget credits exceed reservations")
    return total


def record_budget(path: Path, name: str, seconds: float, reason: str) -> None:
    with path.open("a") as log:
        log.write(
            json.dumps({"format": BUDGET_FORMAT, "name": name, "reserved_gpu_seconds": seconds, "reason": reason})
            + "\n"
        )


def result_seconds(path: Path, exit_code: int) -> float:
    raw = json.loads(path.read_text())
    if (
        not isinstance(raw, dict)
        or set(raw) != {"format", "exit_code", "started_unix", "finished_unix", "elapsed_seconds"}
        or raw["format"] != RESULT_FORMAT
        or type(raw["exit_code"]) is not int
        or raw["exit_code"] != exit_code
        or any(
            type(raw[key]) not in (int, float) or not math.isfinite(raw[key]) or raw[key] < 0
            for key in ("started_unix", "finished_unix", "elapsed_seconds")
        )
        or raw["finished_unix"] < raw["started_unix"]
    ):
        raise ValueError("invalid Modal result identity or duration")
    return float(raw["elapsed_seconds"])


@dataclasses.dataclass(frozen=True, slots=True)
class Request:
    name: str
    git_sha: str
    mode: Literal["smoke", "profile", "floor", "floor-extra", "pilot"] = "profile"
    matches: int = 1024
    timeout_seconds: int = 1800
    created_unix: float = dataclasses.field(default_factory=time.time)
    source_name: str | None = None
    source_inputs_sha256: str | None = None
    source_layout_sha256: str | None = None

    def __post_init__(self) -> None:
        validate_name(self.name)
        if re.fullmatch(r"[0-9a-f]{40}", self.git_sha) is None:
            raise ValueError("expected a full published HAL Git SHA")
        if self.mode not in ("smoke", "profile", "floor", "floor-extra", "pilot"):
            raise ValueError("unsupported run mode")
        allowed = (16,) if self.mode == "smoke" else (256, 1024, 2048, 4096)
        if self.mode in ("floor", "floor-extra", "pilot"):
            allowed = (1024,)
        if type(self.matches) is not int or self.matches not in allowed:
            raise ValueError(f"{self.mode} matches must be one of {allowed}")
        if type(self.timeout_seconds) is not int or not 60 <= self.timeout_seconds <= (
            85500 if self.mode == "pilot" else 6600
        ):
            raise ValueError("timeout must be 60..6600 seconds (pilot: at most 85500)")
        if (
            type(self.created_unix) not in (int, float)
            or not math.isfinite(self.created_unix)
            or self.created_unix <= 0
        ):
            raise ValueError("launch time must be a finite positive timestamp")
        if self.mode == "floor-extra":
            if self.source_name is None:
                raise ValueError("extra model probes require a completed source run")
            validate_name(self.source_name)
            for digest in (self.source_inputs_sha256, self.source_layout_sha256):
                if digest is None or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                    raise ValueError("extra model probes require pinned source input hashes")
        elif any(
            value is not None for value in (self.source_name, self.source_inputs_sha256, self.source_layout_sha256)
        ):
            raise ValueError("source inputs apply only to extra model probes")


def remaining_seconds(request: Request, now: float) -> float:
    # A rescheduled container must share the original reservation's deadline.
    remaining = min(request.timeout_seconds, request.created_unix + request.timeout_seconds + 360 - now)
    if remaining <= 60:
        raise ValueError("O65 Modal reservation has expired")
    return remaining


def command(request: Request, cpus: tuple[int, ...]) -> list[str]:
    if len(cpus) != 16 or len(set(cpus)) != 16:
        raise ValueError("O65 requires 16 distinct shard CPUs")
    delivery = REMOTE_ROOT / "delivery"
    if request.mode == "pilot":
        return [
            "/opt/venv/bin/python",
            "experiments/o65/pilot.py",
            "--output",
            str(WORK / request.name / "pilot"),
            "--local-delivery",
            str(delivery),
            "--cpus",
            *(str(cpu) for cpu in cpus),
            "--compile-cache",
            str(RESULTS / "compile-cache"),
            "--resume",
            "--checkpoint-retention",
            "compact",
            "--publish-directory",
            str(RESULTS / request.name / "pilot"),
            "--publish-volume",
            VOLUME,
            "--r2-prefix",
            f"o65/pilot-results/{request.name}",
        ]
    if request.mode == "floor-extra":
        source = RESULTS / str(request.source_name) / "profile"
        return [
            "/opt/venv/bin/python",
            "-m",
            "scripts.profile_o65_floor",
            "--checkpoint",
            str(delivery / "payloads/final.pt"),
            "--inputs",
            str(source / "floor-inputs.pt"),
            "--layout",
            str(source / "floor-layout.json"),
            "--output",
            str(WORK / request.name / "profile"),
            "--extra-only",
            "--expected-inputs-sha256",
            str(request.source_inputs_sha256),
            "--expected-layout-sha256",
            str(request.source_layout_sha256),
        ]
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
        str(WORK / request.name / "profile"),
        "--cpus",
        *(str(cpu) for cpu in cpus),
        "--matches-per-shard",
        str(request.matches // 16),
        "--groups",
        "1",
        "--compile-cache",
        str(RESULTS / "compile-cache"),
    ]
    if request.mode in ("profile", "floor"):
        argv.extend(("--warm-updates", "1"))
    if request.mode == "floor":
        argv.append("--nsys")
    if request.mode == "smoke":
        argv.extend(
            (
                "--updates",
                "1",
                "--training.critic-warmup-updates",
                "0",
                "--training.windows-per-step",
                "32",
                "--evaluation-every",
                "0",
            )
        )
    return argv


def select_cpus(allowed: set[int]) -> tuple[int, ...]:
    """Modal exposes virtual CPU IDs and hides the host's sysfs topology."""
    if len(allowed) < CPU_CORES:
        raise ValueError("Modal job needs at least 17 available CPU IDs")
    return tuple(sorted(allowed))[:CPU_CORES]


def run_remote(request: Request) -> int:
    output = RESULTS / request.name
    output.mkdir(parents=True, exist_ok=request.mode == "pilot")
    local = WORK / request.name
    local.mkdir(parents=True, exist_ok=False)
    if request.mode == "pilot" and (output / "pilot").is_dir():
        stage_results(output / "pilot", local / "pilot")
    started = time.time()
    started_clock = time.perf_counter()
    code = -1
    try:
        code = run_child(request, local, started)
        return code
    finally:
        stage_results(local, output)
        (output / "result.json").write_text(
            json.dumps(
                {
                    "format": RESULT_FORMAT,
                    "exit_code": code,
                    "started_unix": started,
                    "finished_unix": time.time(),
                    "elapsed_seconds": time.perf_counter() - started_clock,
                },
                indent=2,
            )
        )
        modal.Volume.from_name(VOLUME).commit()


def copy_result(source: str, destination: str) -> str:
    target = Path(destination)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.partial")
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def stage_results(source: Path, destination: Path) -> None:
    # Modal Volumes do not support the hard links used for atomic checkpoints.
    # Finish on local POSIX storage, then copy bytes into the durable Volume.
    files = [
        path
        for path in source.rglob("*")
        if path.is_file()
        and not path.name.endswith(".partial")
        and not set(path.relative_to(source).parts).intersection(("tmp", "inductor", "triton"))
    ]
    files.sort(key=lambda path: (path.name.startswith("update-") and path.suffix == ".json", str(path)))
    for path in files:
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        copy_result(str(path), str(target))


def run_child(request: Request, output: Path, started: float) -> int:
    require_modal_version(importlib.metadata.version("modal"))
    cpus = select_cpus(set(os.sched_getaffinity(0)))
    os.sched_setaffinity(0, set(cpus))
    argv = command(request, cpus[:16])
    env = dict(os.environ)
    if request.mode != "pilot":
        env.pop("WANDB_API_KEY", None)
    elif not env.get("WANDB_API_KEY"):
        raise ValueError("the pilot requires W&B credentials in its Modal Secret")
    env.update(
        WANDB_MODE="online" if request.mode == "pilot" else "disabled",
        HAL_NATIVE_LIBRARY=str(REMOTE_ROOT / "delivery/libhal_native.so"),
        TMPDIR=str(output / "tmp"),
        TORCHINDUCTOR_COMPILE_THREADS="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    Path(env["TMPDIR"]).mkdir()
    identity = {
        "format": "hal.o65.modal-launch.v2",
        "request": dataclasses.asdict(request),
        "command": argv,
        "image": IMAGE,
        "modal_sdk": importlib.metadata.version("modal"),
        "container_id": os.environ.get("MODAL_TASK_ID"),
        "cpus": cpus,
        "cpu_topology": "Modal virtual CPU IDs; physical core and NUMA topology are not exposed",
        "started_unix": started,
        "treatment": f"mode={request.mode}, active_matches={request.matches}; pilot: 512 windows, Beta u and pretraining identity prior",
        "control": "hash-pinned O63 with matched episode sampling"
        if request.mode == "pilot"
        else "unchanged o65-dense trainer at 1024 active matches",
        "invariants": (
            "seed 65, proxy, simulator, D/R, reward, optimizer, 16 shards, 16 decisions per window; "
            + (
                "four groups, 512 windows per step"
                if request.mode == "pilot"
                else "one group, 32 windows per step, 64 reference slots"
            )
        ),
    }
    (output / "launch.json").write_text(json.dumps(identity, indent=2))
    with (output / "topology.txt").open("w") as log:
        log.write("Modal virtual CPU IDs; physical core and NUMA topology are not exposed.\n")
        log.write(Path("/proc/cpuinfo").read_text())
        log.flush()
        subprocess.run(["nvidia-smi", "-q"], stdout=log, stderr=subprocess.STDOUT, check=True)
    with (output / "run.log").open("a") as log:
        if request.mode == "floor":
            return run_floor(request, output, env, cpus[:16], log)
        if request.mode == "pilot":
            return run_pilot_child(argv, env, log, remaining_seconds(request, time.time()) - 90)
        result = subprocess.run(
            argv,
            cwd=REMOTE_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=remaining_seconds(request, time.time()) - 30,
        )
    return result.returncode


def run_pilot_child(argv: list[str], env: dict[str, str], log: TextIO, timeout: float) -> int:
    process = subprocess.Popen(
        argv, cwd=REMOTE_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    try:
        return process.wait(timeout=timeout)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def pilot_estimate() -> dict[str, object]:
    rate = GPU_USD_SECOND + CPU_CORES * CPU_USD_SECOND + (MEMORY_MIB / 1024) * MEMORY_USD_GIB_SECOND
    # Preserve model work: larger steps only save repeated step/optimizer overhead.
    seconds = (11.4 - (6.703817 - 4.246) - 1.2, 11.4 - (6.703817 - 4.500) - 0.6)
    return {
        "label": "estimates, not PRO 6000 measurements of this treatment",
        "baseline_seconds_per_update": 11.4,
        "projected_seconds_per_update": seconds,
        "training_gpu_hours": [2000 * value / 3600 for value in seconds],
        "training_compute_usd": [2000 * value * rate for value in seconds],
        "full_protocol_gpu_hours_estimate": [8, 12],
        "full_protocol_compute_usd_estimate": [
            8 * 3600 * rate + BUILD_RESERVE_USD,
            12 * 3600 * rate + BUILD_RESERVE_USD,
        ],
        "assumptions": "async projection from performance/async.md; 0.6..1.2 s saved in step overhead, unchanged FLOPs; full protocol allows compilation, 32-update restarts, evaluation and publication",
        "hard_timeout_seconds": 85500,
        "maximum_compute_usd": (85500 + 360) * rate + BUILD_RESERVE_USD,
        "excludes": "storage/egress beyond included allowances; build variation",
    }


def nsys_command(argv: list[str], output: Path) -> list[str]:
    return [
        "nsys",
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--sample=none",
        "--cpuctxsw=none",
        "--cuda-graph-trace=node",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--force-overwrite=true",
        f"--output={output}",
        *argv,
    ]


def run_floor(request: Request, output: Path, env: dict[str, str], cpus: tuple[int, ...], log: TextIO) -> int:
    deadline = time.monotonic() + remaining_seconds(request, time.time()) - 60

    def run(argv: list[str]) -> subprocess.CompletedProcess[bytes]:
        log.write(shlex.join(argv) + "\n")
        log.flush()
        return subprocess.run(
            argv,
            cwd=REMOTE_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=max(1, deadline - time.monotonic()),
        )

    result = run(nsys_command(command(request, cpus), output / "update"))
    if result.returncode:
        return result.returncode
    result = run(
        [
            "nsys",
            "export",
            "--type=sqlite",
            "--force-overwrite=true",
            f"--output={output / 'update.sqlite'}",
            str(output / "update.nsys-rep"),
        ]
    )
    if result.returncode:
        return result.returncode
    result = run(
        [
            "/opt/venv/bin/python",
            "-m",
            "scripts.profile_o65_floor",
            "--checkpoint",
            str(REMOTE_ROOT / "delivery/payloads/final.pt"),
            "--inputs",
            str(output / "profile/floor-inputs.pt"),
            "--layout",
            str(output / "profile/floor-layout.json"),
            "--output",
            str(output / "bare"),
        ]
    )
    if result.returncode:
        return result.returncode
    for matches in (256, 2048, 4096):
        arm = dataclasses.replace(request, mode="profile", matches=matches)
        argv = command(arm, cpus)
        argv[argv.index("--output") + 1] = str(output / f"sweep-{matches}")
        result = run(argv)
        (output / f"sweep-{matches}-exit.json").write_text(
            json.dumps({"matches": matches, "exit_code": result.returncode})
        )
        if result.returncode:
            # Preserve a failed larger arm; do not substitute another configuration.
            return result.returncode
    return 0


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
        .run_commands(
            "if ! command -v nsys >/dev/null; then "
            "curl -fL https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/"
            "NsightSystems-linux-cli-public-2026.5.1.161-3889610.deb -o /var/tmp/nsys.deb && "
            "echo '61829db6392e5c293ada1319df86a97c3356bc810335a1975cb551fcfe3eca08  /var/tmp/nsys.deb' | sha256sum -c - && "
            "dpkg -i /var/tmp/nsys.deb && rm /var/tmp/nsys.deb; fi",
            NSYS_VERSION_CHECK,
        )
        .env({"PYTHONPATH": root, "UV_PROJECT_ENVIRONMENT": "/opt/venv"})
        .workdir(root)
        .run_commands(image_commands(sha), secrets=[secret])
        .env({"PYTHONPATH": f"{root}:{root}/delivery/msl"})
    )


def relative_result_path(name: str, path: str) -> Path:
    validate_name(name)
    relative = Path(path.lstrip("/")).relative_to(name)
    if relative == Path(".") or ".." in relative.parts:
        raise ValueError("invalid Modal artifact path")
    return relative


def download(name: str, local: Path) -> None:
    validate_name(name)
    volume = modal.Volume.from_name(VOLUME)
    for entry in volume.iterdir(name, recursive=True):
        if entry.type != FileEntryType.FILE:
            continue
        relative = relative_result_path(name, entry.path)
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
    mode: Literal["smoke", "profile", "floor", "floor-extra", "pilot", "download"] = "profile"
    name: str | None = None
    matches: int = 1024
    timeout_seconds: int | None = None
    dry_run: bool = False
    secret: str = "hal"
    output: Path = Path("runs/modal")
    source_name: str | None = None


def main(args: Args) -> None:
    require_modal_version(importlib.metadata.version("modal"))
    if args.mode == "download":
        if args.name is None:
            raise ValueError("download requires a run name")
        download(args.name, args.output)
        return
    if not args.dry_run and subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise ValueError("commit changes before launching published source")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    expected_branch = "o65-bigstep" if args.mode == "pilot" else "o65-floor"
    if branch != expected_branch:
        raise ValueError(f"only branch {expected_branch} can launch this job")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if not args.dry_run:
        published = subprocess.check_output(
            ["git", "ls-remote", "origin", f"refs/heads/{branch}"], cwd=ROOT, text=True
        )
        if published.split()[:1] != [sha]:
            raise ValueError(f"push {branch} HEAD before launching")
    name = args.name or f"{args.mode}-{args.matches}-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    inputs_digest = layout_digest = None
    if args.mode == "floor-extra":
        if args.source_name is None:
            raise ValueError("extra model probes require --source-name")
        validate_name(args.source_name)
        source = ROOT / "runs/modal-floor" / args.source_name / "profile"
        with (source / "floor-inputs.pt").open("rb") as file:
            inputs_digest = hashlib.file_digest(file, "sha256").hexdigest()
        with (source / "floor-layout.json").open("rb") as file:
            layout_digest = hashlib.file_digest(file, "sha256").hexdigest()
    request = Request(
        name,
        sha,
        args.mode,
        args.matches,
        args.timeout_seconds if args.timeout_seconds is not None else (85500 if args.mode == "pilot" else 1800),
        source_name=args.source_name,
        source_inputs_sha256=inputs_digest,
        source_layout_sha256=layout_digest,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    budget = ROOT / ("runs/modal-pilot/budget.jsonl" if args.mode == "pilot" else "runs/modal-floor/budget.jsonl")
    budget.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the full timeout for each call, including failures and client loss.
    allowance = request.timeout_seconds + 360
    reserved = reserved_seconds(budget) + allowance
    rate = GPU_USD_SECOND + CPU_CORES * CPU_USD_SECOND + (MEMORY_MIB / 1024) * MEMORY_USD_GIB_SECOND
    max_seconds, max_usd = (85860, 110.0) if args.mode == "pilot" else (MAX_GPU_SECONDS, MAX_USD)
    if reserved > max_seconds or reserved * rate + BUILD_RESERVE_USD > max_usd:
        raise ValueError("O65 Modal time or cost budget exhausted")
    keys = [*AWS_KEYS, *(["WANDB_API_KEY"] if args.mode == "pilot" else [])]
    secret = modal.Secret.from_name(args.secret, required_keys=keys)
    if not args.dry_run:
        secret.hydrate()
    app = modal.App(f"o65-{name}")
    volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
    worker = app.function(
        image=build_image(sha, secret),
        secrets=[secret],
        gpu="RTX-PRO-6000",
        cpu=(17.0, 17.0),
        memory=(MEMORY_MIB, MEMORY_MIB),
        volumes={str(RESULTS): volume},
        timeout=request.timeout_seconds,
        startup_timeout=300,
        retries=0,
        max_containers=1,
        single_use_containers=True,
        include_source=False,
        serialized=True,
    )(run_remote)
    launch_command = [
        "uv",
        "run",
        "python",
        "-m",
        "scripts.launch_o65_modal",
        "--mode",
        args.mode,
        "--name",
        name,
        "--timeout-seconds",
        str(request.timeout_seconds),
        "--secret",
        args.secret,
        "--output",
        str(args.output),
    ]
    print(f"Launch: {shlex.join(launch_command)}", flush=True)
    if args.mode == "pilot":
        print(json.dumps(pilot_estimate(), indent=2), flush=True)
    if args.dry_run:
        plan = {
            "request": dataclasses.asdict(request),
            "command": command(request, tuple(range(16))),
            "volume": VOLUME,
            "image": IMAGE,
            "function_built": worker is not None,
            "launch_command": launch_command,
        }
        (args.output / f"{name}-dry-run.json").write_text(json.dumps(plan, indent=2))
        print("Dry run: job built locally; no Modal requests or GPU allocation.", flush=True)
        return
    record_budget(budget, name, allowance, "full timeout plus startup and teardown allowance")
    with modal.enable_output(), app.run():
        print(f"App: {app.app_id}; name: {name}; SDK: {importlib.metadata.version('modal')}", flush=True)
        call = worker.spawn(request)
        (args.output / f"{name}-call.json").write_text(
            json.dumps(
                {
                    "format": "hal.o65.modal-call.v1",
                    "app_id": app.app_id,
                    "call_id": call.object_id,
                    "request": dataclasses.asdict(request),
                },
                indent=2,
            )
        )
        print(f"Call: {call.object_id}; argv: {shlex.join(command(request, tuple(range(16))))}", flush=True)
        code = call.get()
    download(name, args.output)
    elapsed = result_seconds(args.output / name / "result.json", code)
    if args.mode == "pilot":
        # Include previous preempted attempts and builds in the conservative ledger.
        elapsed = max(elapsed, time.time() - request.created_unix)
    charged = min(allowance, elapsed + 360)
    record_budget(budget, name, charged - allowance, "completed call; retain startup and teardown allowance")
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
