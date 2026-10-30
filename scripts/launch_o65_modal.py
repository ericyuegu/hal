"""Run bounded O65 smoke and profiling jobs on one Modal RTX PRO 6000."""

import dataclasses
import fcntl
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
import sys
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
# Experiments stay source-only; also support running this launcher as a file.
if not __package__:
    sys.path.insert(0, str(ROOT))

from experiments.o65 import pilot  # noqa: E402

REMOTE_ROOT = Path("/opt/o65-hal")
RESULTS = Path("/o65-results")
WORK = Path("/opt/o65-work")
VOLUME = "hal-o65-profile"
IMAGE = "ghcr.io/ericyuegu/hal@sha256:c9160a7bb2dc0aa3eb6dc583b1c4753ff98d9a40f802512ca402e2b93a3a7413"
AWS_KEYS = ("AWS_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_BUCKET")
CPU_CORES = 17
MEMORY_MIB = 65536
# Published Modal rates checked 2026-10-20; reserve more for startup and CPU builds.
GPU_USD_SECOND = 0.000842
CPU_USD_SECOND = 0.0000131
MEMORY_USD_GIB_SECOND = 0.00000222
MAX_GPU_SECONDS = 7200
MAX_USD = 20.0
BUILD_RESERVE_USD = 3.0
PILOT_TIMEOUT_SECONDS = 36000
PILOT_MAX_USD = 50.0
BUDGET_FORMAT = "hal.o65.modal-budget.v1"
RESULT_FORMAT = "hal.o65.modal-result.v1"
MODAL_SDK = "1.5.3"
ABLATION_TIMEOUT_SECONDS = 9000
ABLATION_MAX_USD = 65.0
ABLATION_BUDGET_FORMAT = "hal.o65.modal-ablation-budget.v1"


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
    mode: Literal["smoke", "profile", "pilot", "ablation"] = "profile"
    matches: int = 1024
    timeout_seconds: int = 1800
    created_unix: float = dataclasses.field(default_factory=time.time)
    resume_learner: str | None = None
    arm: pilot.ArmName | None = None
    sweep: str | None = None

    def __post_init__(self) -> None:
        validate_name(self.name)
        if self.mode == "ablation":
            if self.arm is None or self.sweep is None:
                raise ValueError("ablation requires an arm and sweep")
            pilot.get_arm(self.arm)
            validate_name(self.sweep)
            if self.name != f"{self.sweep}-{self.arm}":
                raise ValueError("ablation run name must be <sweep>-<arm>")
        elif self.arm is not None or self.sweep is not None:
            raise ValueError("arm and sweep are only supported for ablation")
        if self.resume_learner is not None:
            path = Path(self.resume_learner)
            if self.mode != "pilot" or path.is_absolute() or ".." in path.parts or path.name != "update-00000256.json":
                raise ValueError("learner resume needs a relative update-256 path on the pilot volume")
            if path.parts[0] == self.name:
                raise ValueError("learner resume must use a new run name")
        if re.fullmatch(r"[0-9a-f]{40}", self.git_sha) is None:
            raise ValueError("expected a full published HAL Git SHA")
        if self.mode not in ("smoke", "profile", "pilot", "ablation"):
            raise ValueError("unsupported run mode")
        allowed = (16,) if self.mode == "smoke" else (256, 1024, 2048, 4096)
        if self.mode in ("pilot", "ablation"):
            allowed = (1024,)
        if type(self.matches) is not int or self.matches not in allowed:
            raise ValueError(f"{self.mode} matches must be one of {allowed}")
        if type(self.timeout_seconds) is not int or not 60 <= self.timeout_seconds <= (
            PILOT_TIMEOUT_SECONDS
            if self.mode == "pilot"
            else (ABLATION_TIMEOUT_SECONDS if self.mode == "ablation" else 6600)
        ):
            raise ValueError(f"timeout must be 60..6600 seconds (pilot: at most {PILOT_TIMEOUT_SECONDS})")
        if (
            type(self.created_unix) not in (int, float)
            or not math.isfinite(self.created_unix)
            or self.created_unix <= 0
        ):
            raise ValueError("launch time must be a finite positive timestamp")


def remaining_seconds(request: Request, now: float, *, first_started: float | None = None) -> float:
    # A rescheduled container must share the original reservation's deadline.
    if request.mode == "ablation":
        if first_started is None:
            raise ValueError("ablation deadline requires the container's first start")
        remaining = first_started + request.timeout_seconds - now
    else:
        remaining = min(request.timeout_seconds, request.created_unix + request.timeout_seconds + 360 - now)
    if remaining <= 60:
        raise ValueError("O65 Modal reservation has expired")
    return remaining


def first_start(request: Request, output: Path, now: float) -> float:
    path = output / "first-start.json"
    if not path.exists():
        with path.open("x") as target:
            json.dump(
                {"format": "hal.o65.modal-first-start.v1", "request": dataclasses.asdict(request), "unix": now}, target
            )
        return now
    raw = json.loads(path.read_text())
    if (
        set(raw) != {"format", "request", "unix"}
        or raw["format"] != "hal.o65.modal-first-start.v1"
        or raw["request"] != dataclasses.asdict(request)
        or type(raw["unix"]) not in (int, float)
        or not math.isfinite(raw["unix"])
        or not 0 < raw["unix"] <= now
    ):
        raise ValueError("invalid first-start identity or timestamp")
    return float(raw["unix"])


def command(request: Request, cpus: tuple[int, ...]) -> list[str]:
    if len(cpus) != 16 or len(set(cpus)) != 16:
        raise ValueError("O65 requires 16 distinct shard CPUs")
    delivery = REMOTE_ROOT / "delivery"
    if request.mode in ("pilot", "ablation"):
        argv = [
            "/opt/venv/bin/python",
            "experiments/o65/pilot.py",
            "--output",
            str(WORK / request.name / "pilot"),
            "--local-delivery",
            str(delivery),
            "--cpus",
            *(str(cpu) for cpu in cpus),
            "--compile-cache",
            str(RESULTS / request.name / "compile-cache" if request.mode == "ablation" else RESULTS / "compile-cache"),
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
        if request.arm is not None:
            argv.extend(("--arm", request.arm, "--sweep", str(request.sweep)))
        if request.resume_learner is not None:
            argv.extend(("--resume-learner", str(RESULTS / request.resume_learner)))
        return argv
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
        *(str(cpu) for cpu in (cpus[:4] if request.mode == "smoke" else cpus)),
        "--matches-per-shard",
        str(request.matches // (4 if request.mode == "smoke" else 16)),
        "--groups",
        "1",
        "--compile-cache",
        str(RESULTS / "compile-cache"),
    ]
    if request.mode == "profile":
        argv.extend(("--warm-updates", "1"))
    if request.mode == "smoke":
        argv.extend(
            (
                "--updates",
                "1",
                "--training.critic-warmup-updates",
                "0",
                "--training.advantage-scale",
                "batch",
                "--training.windows-per-step",
                "32",
            )
        )
    return argv


def select_cpus(allowed: set[int]) -> tuple[int, ...]:
    """Modal exposes virtual CPU IDs and hides the host's sysfs topology."""
    if len(allowed) < CPU_CORES:
        raise ValueError("Modal job needs at least 17 available CPU IDs")
    return tuple(sorted(allowed))[:CPU_CORES]


def run_remote(request: Request) -> int:
    started = time.time()
    started_clock = time.perf_counter()
    output = RESULTS / request.name
    output.mkdir(parents=True, exist_ok=request.mode in ("pilot", "ablation"))
    deadline_start = started
    if request.mode == "ablation":
        deadline_start = first_start(request, output, started)
        # Persist before builds or training so rescheduling cannot reset the deadline.
        modal.Volume.from_name(VOLUME).commit()
    local = WORK / request.name
    local.mkdir(parents=True, exist_ok=False)
    code = -1
    try:
        if request.mode in ("pilot", "ablation") and (output / "pilot").is_dir():
            stage_results(output / "pilot", local / "pilot")
        if request.mode == "ablation":
            seed_compile_cache(RESULTS / "compile-cache", output / "compile-cache")
        code = run_child(request, local, deadline_start)
        return code
    finally:
        # Only the publisher may replace the pilot resume tree. A failed trainer
        # can have newer local checkpoints which were never published.
        stage_results(local, output / "last-attempt" if request.mode in ("pilot", "ablation") else output)
        (output / "result.json").write_text(
            json.dumps(
                {
                    "format": RESULT_FORMAT,
                    "exit_code": code,
                    "started_unix": deadline_start,
                    "finished_unix": time.time(),
                    "elapsed_seconds": time.time() - deadline_start
                    if request.mode == "ablation"
                    else time.perf_counter() - started_clock,
                },
                indent=2,
            )
        )
        modal.Volume.from_name(VOLUME).commit()


def seed_compile_cache(shared: Path, own: Path) -> None:
    if own.is_dir():
        return
    if own.exists() or (shared.exists() and not shared.is_dir()):
        raise ValueError("compile cache must be a directory")
    if not shared.exists():
        own.mkdir()
        return
    temporary = own.with_name(f".{own.name}.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        shutil.copytree(shared, temporary)
        temporary.replace(own)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


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
    files.sort(
        key=lambda path: (
            path.name != "pilot-identity.json",
            path.name.startswith("update-") and path.suffix == ".json",
            str(path),
        )
    )
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
    is_pilot = request.mode in ("pilot", "ablation")
    if not is_pilot:
        env.pop("WANDB_API_KEY", None)
    elif not env.get("WANDB_API_KEY"):
        raise ValueError("the pilot requires W&B credentials in its Modal Secret")
    env.update(
        WANDB_MODE="online" if is_pilot else "disabled",
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
        "treatment": dataclasses.asdict(pilot.get_arm(request.arm))
        if request.arm is not None
        else f"mode={request.mode}, active_matches={request.matches}; pilot: 512 windows, Beta u and pretraining identity prior",
        "control": "hash-pinned O63 with matched episode sampling"
        if is_pilot
        else "unchanged o65-dense trainer at 1024 active matches",
        "invariants": (
            "seed 65, proxy, simulator, D/R, optimizer, 16 shards, 16 decisions per window; "
            + (
                "four groups, 512 windows per step"
                if is_pilot
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
        if is_pilot:
            return run_pilot_child(argv, env, log, remaining_seconds(request, time.time(), first_started=started) - 90)
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
        "full_protocol_gpu_hours_estimate": [6, 8],
        "full_protocol_compute_usd_estimate": [
            6 * 3600 * rate + BUILD_RESERVE_USD,
            8 * 3600 * rate + BUILD_RESERVE_USD,
        ],
        "assumptions": "async projection from performance.md; 0.6..1.2 s saved in step overhead, unchanged FLOPs; full protocol allows compilation, background publication every 32 updates and a final 384-match MSL evaluation with four stocks and an eight-minute timer",
        "hard_timeout_seconds": PILOT_TIMEOUT_SECONDS,
        "maximum_compute_usd": (PILOT_TIMEOUT_SECONDS + 360) * rate + BUILD_RESERVE_USD,
        "excludes": "storage/egress beyond included allowances; build variation",
    }


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
        if relative.parts[0] in ("compile-cache", "inductor", "triton", "tmp"):
            continue
        target = local / name / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as out:
            for chunk in volume.read_file(entry.path):
                out.write(chunk)


@dataclasses.dataclass(frozen=True, slots=True)
class AblationBalance:
    reserved: float
    charged: float | None = None


def compute_rate() -> float:
    return GPU_USD_SECOND + CPU_CORES * CPU_USD_SECOND + (MEMORY_MIB / 1024) * MEMORY_USD_GIB_SECOND


def read_ablation_budget(log: TextIO) -> dict[str, AblationBalance]:
    log.seek(0)
    balances: dict[str, AblationBalance] = {}
    for line in log:
        raw = json.loads(line)
        if (
            not isinstance(raw, dict)
            or set(raw) != {"format", "name", "seconds", "action"}
            or raw["format"] != ABLATION_BUDGET_FORMAT
            or not isinstance(raw["name"], str)
            or type(raw["seconds"]) not in (int, float)
            or not math.isfinite(raw["seconds"])
            or raw["seconds"] < 0
        ):
            raise ValueError("invalid ablation budget record")
        name, seconds = raw["name"], float(raw["seconds"])
        validate_name(name)
        if raw["action"] == "reserve" and name not in balances and seconds > 0:
            balances[name] = AblationBalance(seconds)
        elif (
            raw["action"] == "settle"
            and name in balances
            and balances[name].charged is None
            and seconds <= balances[name].reserved
        ):
            balances[name] = dataclasses.replace(balances[name], charged=seconds)
        else:
            raise ValueError("duplicate or unmatched ablation budget entry")
    return balances


def budget_line(name: str, seconds: float, action: Literal["reserve", "settle"]) -> str:
    return json.dumps({"format": ABLATION_BUDGET_FORMAT, "name": name, "seconds": seconds, "action": action}) + "\n"


def reserve_ablation(path: Path, requests: tuple[Request, ...]) -> None:
    if not requests or any(request.mode != "ablation" for request in requests):
        raise ValueError("reserve requires ablation arms")
    names = [request.name for request in requests]
    if len(set(names)) != len(names):
        raise ValueError("duplicate ablation reservations")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as log:
        fcntl.flock(log, fcntl.LOCK_EX)
        balances = read_ablation_budget(log)
        if set(names).intersection(balances):
            raise ValueError("ablation run already reserved; collect it or use a new sweep")
        seconds = sum(
            balance.reserved if balance.charged is None else balance.charged for balance in balances.values()
        )
        seconds += sum(request.timeout_seconds + 360 for request in requests)
        if seconds * compute_rate() + BUILD_RESERVE_USD > ABLATION_MAX_USD:
            raise ValueError("O65 ablation $65 budget exhausted")
        # A single locked write reserves every arm before any call can spawn.
        log.write("".join(budget_line(request.name, request.timeout_seconds + 360, "reserve") for request in requests))
        log.flush()
        os.fsync(log.fileno())


def settle_ablation(path: Path, name: str, elapsed: float) -> float:
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("invalid ablation elapsed time")
    with path.open("r+") as log:
        fcntl.flock(log, fcntl.LOCK_EX)
        balances = read_ablation_budget(log)
        balance = balances[name]
        if balance.charged is not None:
            return balance.charged
        charged = min(balance.reserved, elapsed + 360)
        log.seek(0, os.SEEK_END)
        log.write(budget_line(name, charged, "settle"))
        log.flush()
        os.fsync(log.fileno())
        return charged


def write_checksums(directory: Path) -> None:
    hashes: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name not in ("SHA256SUMS", "SHA256SUMS.json"):
            with path.open("rb") as source:
                hashes[str(path.relative_to(directory))] = hashlib.file_digest(source, "sha256").hexdigest()
    (directory / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in hashes.items()))
    (directory / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2))


def read_ablation_plan(path: Path, sweep: str) -> tuple[Request, ...]:
    raw = json.loads(path.read_text())
    if raw["format"] != "hal.o65.modal-ablation-plan.v1" or raw["sweep"] != sweep:
        raise ValueError("invalid ablation plan identity")
    requests = []
    for entry in raw["arms"]:
        row = entry["request"]
        request = Request(
            name=row["name"],
            git_sha=row["git_sha"],
            mode=row["mode"],
            matches=row["matches"],
            timeout_seconds=row["timeout_seconds"],
            created_unix=row["created_unix"],
            resume_learner=row["resume_learner"],
            arm=row["arm"],
            sweep=row["sweep"],
        )
        if request.mode != "ablation" or request.sweep != sweep or row != dataclasses.asdict(request):
            raise ValueError("invalid ablation request identity")
        requests.append(request)
    if not requests or len({request.name for request in requests}) != len(requests):
        raise ValueError("empty or duplicate ablation plan")
    return tuple(requests)


def collect_ablation(sweep: str, output: Path, budget: Path) -> None:
    validate_name(sweep)
    requests = read_ablation_plan(output / f"{sweep}-plan.json", sweep)
    failures: list[str] = []
    for request in requests:
        call_path = output / f"{request.name}-call.json"
        if not call_path.exists():
            failures.append(f"{request.name}: no recorded call; retaining reservation")
            continue
        record = json.loads(call_path.read_text())
        if (
            set(record) != {"format", "app_id", "call_id", "request"}
            or record["format"] != "hal.o65.modal-call.v1"
            or record["request"] != dataclasses.asdict(request)
            or not isinstance(record["call_id"], str)
            or not record["call_id"]
        ):
            raise ValueError("invalid ablation call identity")
        call = modal.FunctionCall.from_id(record["call_id"])
        code: int | None = None
        try:
            code = call.get()
        except (modal.exception.Error, OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            # One failed remote arm must not prevent collection of the other arms.
            failures.append(f"{request.name}: {error}")
        download(request.name, output)
        directory = output / request.name
        directory.mkdir(parents=True, exist_ok=True)
        result = directory / "result.json"
        if result.is_file():
            if code is None:
                code = json.loads(result.read_text())["exit_code"]
            elapsed = result_seconds(result, code)
        elif code is None:
            # Without a duration record, retain the full failed-call reservation.
            elapsed = float(request.timeout_seconds + 360)
        else:
            raise ValueError(f"missing result duration for {request.name}")
        write_checksums(directory)
        settle_ablation(budget, request.name, elapsed)
        if code:
            failures.append(f"{request.name}: exited {code}")
        print(f"Collected {request.name}: {directory}", flush=True)
    if failures:
        raise SystemExit("\n".join(failures))


@dataclasses.dataclass(frozen=True, slots=True)
class Args:
    mode: Literal["smoke", "profile", "pilot", "download", "ablation", "collect"] = "profile"
    name: str | None = None
    matches: int = 1024
    timeout_seconds: int | None = None
    dry_run: bool = False
    resume_learner: str | None = None
    secret: str = "hal"
    output: Path = Path("runs/modal")
    sweep: str | None = None
    arms: tuple[pilot.ArmName, ...] = ("old", "reward", "critic", "scale", "matchup")


def launch_ablation(args: Args, budget: Path) -> None:
    if args.sweep is None:
        raise ValueError("ablation requires --sweep")
    validate_name(args.sweep)
    if not args.arms or len(set(args.arms)) != len(args.arms):
        raise ValueError("ablation requires distinct arms")
    for name in args.arms:
        pilot.get_arm(name)
    if args.name is not None or args.resume_learner is not None:
        raise ValueError("ablation uses sweep-arm names and the fixed bootstrap")
    if not args.dry_run and subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise ValueError("commit changes before launching published source")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if not args.dry_run:
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
        published = subprocess.check_output(
            ["git", "ls-remote", "origin", f"refs/heads/{branch}"], cwd=ROOT, text=True
        )
        if published.split()[:1] != [sha]:
            raise ValueError(f"push {branch} HEAD before launching")
    requests = tuple(
        Request(
            f"{args.sweep}-{arm}",
            sha,
            mode="ablation",
            matches=args.matches,
            timeout_seconds=args.timeout_seconds if args.timeout_seconds is not None else ABLATION_TIMEOUT_SECONDS,
            arm=arm,
            sweep=args.sweep,
        )
        for arm in args.arms
    )
    plans = []
    for request in requests:
        pilot_argv = command(request, tuple(range(16)))
        pilot_args = tyro.cli(pilot.Args, args=pilot_argv[2:])
        trainer_argv = pilot.training_command(
            REMOTE_ROOT / "delivery",
            WORK / request.name / "pilot/train",
            tuple(range(16)),
            matches=64,
            groups=4,
            updates=672,
            checkpoint_every=32,
            warmup=32,
            compile_cache=RESULTS / request.name / "compile-cache",
            arm=request.arm,
        )
        trainer_argv[0:2] = ["/opt/venv/bin/python", str(REMOTE_ROOT / "experiments/065_selfplay_ppo.py")]
        trainer_argv[trainer_argv.index("--sampler") + 1] = str(REMOTE_ROOT / "experiments/o65/pilot-sampler.json")
        trainer_argv.extend(("--wandb-project", "hal-o65"))
        trainer_argv.extend(pilot.publication_arguments(pilot_args, pilot_args.output))
        plans.append({"request": dataclasses.asdict(request), "command": pilot_argv, "trainer_argv": trainer_argv})
        print(f"{request.name}: {shlex.join(trainer_argv)}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / f"{args.sweep}-{'dry-run' if args.dry_run else 'plan'}.json"
    with plan_path.open("w" if args.dry_run else "x") as target:
        json.dump({"format": "hal.o65.modal-ablation-plan.v1", "sweep": args.sweep, "arms": plans}, target, indent=2)
    if args.dry_run:
        print(f"Dry run: wrote {len(plans)} arm argvs to {plan_path}; no Modal requests.", flush=True)
        return
    reserve_ablation(budget, requests)
    secret = modal.Secret.from_name(args.secret, required_keys=[*AWS_KEYS, "WANDB_API_KEY"])
    secret.hydrate()
    image = build_image(sha, secret)
    app = modal.App(f"o65-{args.sweep}")
    volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
    worker = app.function(
        image=image,
        secrets=[secret],
        gpu="RTX-PRO-6000",
        cpu=(17.0, 17.0),
        memory=(MEMORY_MIB, MEMORY_MIB),
        volumes={str(RESULTS): volume},
        timeout=requests[0].timeout_seconds,
        startup_timeout=300,
        retries=0,
        max_containers=len(requests),
        single_use_containers=True,
        include_source=False,
        serialized=True,
    )(run_remote)
    with modal.enable_output(), app.run(detach=True):
        for request in requests:
            call = worker.spawn(request)
            with (args.output / f"{request.name}-call.json").open("x") as target:
                json.dump(
                    {
                        "format": "hal.o65.modal-call.v1",
                        "app_id": app.app_id,
                        "call_id": call.object_id,
                        "request": dataclasses.asdict(request),
                    },
                    target,
                    indent=2,
                )
                target.flush()
                os.fsync(target.fileno())
            print(f"{request.name}: call {call.object_id}; app {app.app_id}", flush=True)


def main(args: Args) -> None:
    require_modal_version(importlib.metadata.version("modal"))
    if args.mode in ("ablation", "collect"):
        budget = ROOT / "runs/modal-ablation/budget.jsonl"
        if args.mode == "ablation":
            launch_ablation(args, budget)
        else:
            if args.sweep is None:
                raise ValueError("collect requires --sweep")
            if args.dry_run:
                raise ValueError("collect does not support --dry-run")
            collect_ablation(args.sweep, args.output, budget)
        return
    if args.mode == "download":
        if args.name is None:
            raise ValueError("download requires a run name")
        download(args.name, args.output)
        return
    if not args.dry_run and subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise ValueError("commit changes before launching published source")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if not args.dry_run:
        published = subprocess.check_output(
            ["git", "ls-remote", "origin", f"refs/heads/{branch}"], cwd=ROOT, text=True
        )
        if published.split()[:1] != [sha]:
            raise ValueError(f"push {branch} HEAD before launching")
    name = args.name or f"{args.mode}-{args.matches}-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    request = Request(
        name,
        sha,
        args.mode,
        args.matches,
        args.timeout_seconds
        if args.timeout_seconds is not None
        else (PILOT_TIMEOUT_SECONDS if args.mode == "pilot" else 1800),
        resume_learner=args.resume_learner,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    budget = ROOT / ("runs/modal-pilot/budget.jsonl" if args.mode == "pilot" else "runs/modal-floor/budget.jsonl")
    budget.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the full timeout for each call, including failures and client loss.
    allowance = request.timeout_seconds + 360
    reserved = reserved_seconds(budget) + allowance
    rate = GPU_USD_SECOND + CPU_CORES * CPU_USD_SECOND + (MEMORY_MIB / 1024) * MEMORY_USD_GIB_SECOND
    max_seconds, max_usd = (
        (PILOT_TIMEOUT_SECONDS + 360, PILOT_MAX_USD) if args.mode == "pilot" else (MAX_GPU_SECONDS, MAX_USD)
    )
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
    if args.resume_learner is not None:
        launch_command.extend(("--resume-learner", args.resume_learner))
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
