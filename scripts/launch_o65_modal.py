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

from hal.sim.native import RewardWeights

ROOT = Path(__file__).resolve().parents[1]
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
    mode: Literal["smoke", "profile", "pilot"] = "profile"
    matches: int = 1024
    timeout_seconds: int = 1800
    created_unix: float = dataclasses.field(default_factory=time.time)
    resume_learner: str | None = None
    damage_weight: float = 1.0
    stock_weight: float = 120.0
    game_win_weight: float = 50.0

    def __post_init__(self) -> None:
        validate_name(self.name)
        weights = RewardWeights(self.damage_weight, self.stock_weight, self.game_win_weight)
        if self.mode != "pilot" and weights != RewardWeights():
            raise ValueError("reward treatment flags require pilot mode")
        if self.resume_learner is not None:
            path = Path(self.resume_learner)
            if self.mode != "pilot" or path.is_absolute() or ".." in path.parts or path.name != "update-00000256.json":
                raise ValueError("learner resume needs a relative update-256 path on the pilot volume")
            if path.parts[0] == self.name:
                raise ValueError("learner resume must use a new run name")
        if re.fullmatch(r"[0-9a-f]{40}", self.git_sha) is None:
            raise ValueError("expected a full published HAL Git SHA")
        if self.mode not in ("smoke", "profile", "pilot"):
            raise ValueError("unsupported run mode")
        allowed = (16,) if self.mode == "smoke" else (256, 1024, 2048, 4096)
        if self.mode == "pilot":
            allowed = (1024,)
        if type(self.matches) is not int or self.matches not in allowed:
            raise ValueError(f"{self.mode} matches must be one of {allowed}")
        if type(self.timeout_seconds) is not int or not 60 <= self.timeout_seconds <= (
            PILOT_TIMEOUT_SECONDS if self.mode == "pilot" else 6600
        ):
            raise ValueError(f"timeout must be 60..6600 seconds (pilot: at most {PILOT_TIMEOUT_SECONDS})")
        if (
            type(self.created_unix) not in (int, float)
            or not math.isfinite(self.created_unix)
            or self.created_unix <= 0
        ):
            raise ValueError("launch time must be a finite positive timestamp")


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
            "--damage-weight",
            str(request.damage_weight),
            "--stock-weight",
            str(request.stock_weight),
            "--game-win-weight",
            str(request.game_win_weight),
        ]
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
        # Only the publisher may replace the pilot resume tree. A failed trainer
        # can have newer local checkpoints which were never published.
        stage_results(local, output / "last-attempt" if request.mode == "pilot" else output)
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
        if relative.parts[0] in ("inductor", "triton", "tmp"):
            continue
        target = local / name / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as out:
            for chunk in volume.read_file(entry.path):
                out.write(chunk)


@dataclasses.dataclass(frozen=True, slots=True)
class Args:
    mode: Literal["smoke", "profile", "pilot", "download"] = "profile"
    name: str | None = None
    matches: int = 1024
    timeout_seconds: int | None = None
    dry_run: bool = False
    resume_learner: str | None = None
    damage_weight: float = 1.0
    stock_weight: float = 120.0
    game_win_weight: float = 50.0
    secret: str = "hal"
    output: Path = Path("runs/modal")


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
        damage_weight=args.damage_weight,
        stock_weight=args.stock_weight,
        game_win_weight=args.game_win_weight,
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
    launch_command.extend(
        (
            "--damage-weight",
            str(args.damage_weight),
            "--stock-weight",
            str(args.stock_weight),
            "--game-win-weight",
            str(args.game_win_weight),
        )
    )
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
