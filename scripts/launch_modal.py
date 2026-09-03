"""Run one retry-safe HAL training experiment on a Modal GPU.

The launcher ships the clean, pushed working tree in a layered image, then
starts one detached Modal Function call. Modal may preempt or time out a Function
attempt. A small Modal Volume records the W&B/checkpoint run name so the next
attempt can add ``--resume <run>`` after the checkpoint reaches R2.

    uv run scripts/launch_modal.py --dry-run -- uv run experiments/028_onehot_controller.py
    uv run scripts/launch_modal.py --wait -- uv run experiments/028_onehot_controller.py

The default ``hal`` Modal Secret must contain the R2 ``AWS_*`` variables and
``WANDB_API_KEY``. Immutable integration fixtures are fetched into a cached
image layer during the build. Dataset shards and compiler scratch use ephemeral
SSD. Modal memory snapshots accelerate Python container restoration and opt in
to GPU-state capture; training programs still compile into ephemeral SSD once
per Function attempt because model setup happens after input dispatch.
"""

import json
import os
import queue
import re
import resource
import shlex
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Final
from typing import Literal
from typing import TypedDict

import loguru
import modal
import tyro
from botocore.exceptions import ClientError

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT: Final[Path] = ROOT / "experiments"
REMOTE_ROOT: Final[Path] = Path("/opt/hal")
STATE_ROOT: Final[Path] = Path("/modal-state")
LOCAL_CACHE_ROOT: Final[Path] = Path("/opt/hal-cache")
IMAGE: Final[str] = "ghcr.io/ericyuegu/hal:cuda13"
PYPI_INDEX: Final[str] = "https://pypi.org/simple"
REQUIRED_SECRET_KEYS: Final[tuple[str, ...]] = (
    "AWS_ENDPOINT_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_BUCKET",
    "WANDB_API_KEY",
)
RUN_LINE: Final[re.Pattern[str]] = re.compile(r"^\[ckpt\] writing checkpoints to runs/([^\r\n/]+)$")
RUN_NAME: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._ -]*[A-Za-z0-9._-])?$")
SENSITIVE_FLAGS: Final[frozenset[str]] = frozenset(
    {
        "--api-key",
        "--key",
        "--password",
        "--secret",
        "--token",
    }
)
FORK_SOURCE_FLAGS: Final[frozenset[str]] = frozenset(
    {
        "--prefix-fork-checkpoint",
        "--prefix-fork-from-run",
        "--resume",
        "--resume-as",
        "--resume-checkpoint",
        "--target-positions",
    }
)
STATE_SCHEMA: Final[int] = 1
INTERRUPT_GRACE_S: Final[int] = 20


@dataclass(frozen=True, slots=True)
class Args:
    cmd: tyro.conf.Positional[list[str]] = field(default_factory=list)
    """Command after ``--``. Automatic recovery requires an ``experiments/*.py`` command."""
    gpu: str = "B200"
    """Modal GPU type. Use 'none' for CPU-only work; comma-separated fallbacks are accepted."""
    cpu: float = 32.0
    """Requested physical CPU cores."""
    cpu_limit: float = 48.0
    """CPU hard limit. Modal can burst above the request when capacity is available."""
    memory_gib: int = 128
    """Requested system memory in GiB."""
    memory_limit_gib: int | None = 384
    """Hard system-memory limit in GiB; None leaves the Modal default."""
    disk_gib: int = 2048
    """Ephemeral SSD in GiB. O51 is raised to its required 3 TiB minimum automatically."""
    image: str = IMAGE
    """Dependency image imported from a registry. The clean local source is copied on top."""
    cloud: str | None = None
    """Optional Modal cloud constraint: aws, gcp, oci, or auto."""
    region: str | None = None
    """Optional Modal region constraint. The default lets Modal choose."""
    timeout_hours: int = 24
    """Maximum duration of each attempt. Modal currently permits at most 24 hours."""
    startup_timeout_minutes: int = 30
    """Maximum container/image startup duration."""
    max_retries: int = 10
    """Retries for timeouts, preemptions, and infrastructure failures."""
    secret: str = "hal"
    """Modal Secret containing R2 AWS_* values and WANDB_API_KEY."""
    state_volume: str = "hal-modal-state"
    """Small Modal Volume used only for retry state."""
    gpu_memory_snapshot: bool = True
    """Enable Modal CPU snapshots and alpha GPU-state capture."""
    app_name: str | None = None
    """Modal App name. The default includes the UTC time and Git SHA."""
    stall_minutes: int = 60
    """Fail the run if the training log stays silent this long."""
    auto_resume: bool = True
    """Resume a preempted experiment from R2. Disable for arbitrary commands."""
    skip_sm120_probe: bool = False
    """Skip the compile smoke probe when a selected GPU has compute capability sm_120."""
    wait: bool = False
    """Stream output and wait for completion. The default detaches after submission."""
    dry_run: bool = False
    """Verify Git, Modal authentication, and the Secret, then print the request without launching."""


class FunctionResources(TypedDict):
    gpu: str | list[str] | None
    cpu: tuple[float, float]
    memory: int | tuple[int, int]
    ephemeral_disk: int
    timeout: int
    startup_timeout: int
    cloud: str | None
    region: str | None


@dataclass(frozen=True, slots=True)
class LaunchSpec:
    argv: tuple[str, ...]
    launch_id: str
    git_sha: str
    state_volume: str
    auto_resume: bool
    stall_s: int
    skip_sm120_probe: bool
    require_cuda: bool = True
    modal_app_url: str | None = None


@dataclass(frozen=True, slots=True)
class RunState:
    status: Literal["running", "failed", "succeeded"]
    run_name: str | None = None
    exit_code: int | None = None
    schema: int = STATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STATE_SCHEMA:
            raise ValueError(f"unsupported Modal run-state schema {self.schema}; expected {STATE_SCHEMA}")
        if self.run_name is not None and not RUN_NAME.fullmatch(self.run_name):
            raise ValueError(f"invalid run name in Modal state: {self.run_name!r}")
        if self.status == "failed" and self.exit_code is None:
            raise ValueError("failed Modal state must include an exit code")
        if self.status != "failed" and self.exit_code is not None:
            raise ValueError(f"{self.status} Modal state cannot include an exit code")


@dataclass(frozen=True, slots=True)
class RecoveryContext:
    explicit_resume: str | None
    run_name: str | None
    retrying_fork: bool


@dataclass(frozen=True, slots=True)
class AttemptPlan:
    action: Literal["run", "fail", "complete"]
    argv: tuple[str, ...]
    state: RunState


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def preflight_git() -> str:
    """Return the clean Git SHA and push it when it is not yet on a remote."""
    if _git("status", "--porcelain"):
        raise SystemExit("working tree is dirty — commit before launching (Modal runs the committed source).")
    sha = _git("rev-parse", "HEAD")
    if not _git("branch", "-r", "--contains", sha):
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        if branch == "HEAD":
            raise SystemExit(f"Git SHA {sha[:10]} is not pushed and the working tree is detached.")
        loguru.logger.info(f"{sha[:10]} is not on a remote; pushing {branch}")
        subprocess.run(["git", "push", "origin", branch], cwd=ROOT, check=True)
    return sha


def preflight_modal(secret_name: str) -> tuple[modal.Client, modal.Secret]:
    """Verify the active Modal profile and hydrate the required named Secret."""
    client = modal.Client.from_env()
    client.hello()
    secret = modal.Secret.from_name(secret_name, required_keys=list(REQUIRED_SECRET_KEYS))
    secret.hydrate(client=client)
    return client, secret


def validate_args(args: Args) -> None:
    if not args.cmd:
        raise SystemExit("missing command — pass it after `--`, for example `-- uv run experiments/028_...py`.")
    if args.cpu <= 0 or args.cpu_limit < args.cpu:
        raise SystemExit("--cpu must be positive and --cpu-limit must be at least --cpu.")
    if args.memory_gib <= 0 or args.disk_gib <= 0:
        raise SystemExit("--memory-gib and --disk-gib must be positive.")
    if args.memory_limit_gib is not None and args.memory_limit_gib < args.memory_gib:
        raise SystemExit("--memory-limit-gib must be at least --memory-gib.")
    if not 1 <= args.timeout_hours <= 24:
        raise SystemExit("--timeout-hours must be between 1 and Modal's 24-hour Function limit.")
    if args.startup_timeout_minutes <= 0 or args.max_retries < 0 or args.stall_minutes <= 0:
        raise SystemExit("startup timeout and stall duration must be positive; retries cannot be negative.")
    if args.auto_resume and experiment_script(args.cmd) is None:
        raise SystemExit(
            "automatic recovery only supports commands containing an existing experiments/*.py script; "
            "pass --no-auto-resume for an arbitrary command."
        )
    _recovery_context(args.cmd, None)


def experiment_script(argv: list[str] | tuple[str, ...]) -> Path | None:
    """Return the local experiment path embedded in an argv, if there is one."""
    for token in argv:
        candidate = (ROOT / token).resolve()
        if candidate.suffix == ".py" and candidate.is_relative_to(EXPERIMENTS_ROOT) and candidate.is_file():
            return candidate
    return None


def _command_option(argv: list[str] | tuple[str, ...], flag: str) -> str | None:
    """Read one named value without interpreting the rest of the experiment CLI."""
    found: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == flag:
            if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
                raise SystemExit(f"{flag} requires a run name.")
            found.append(argv[i + 1])
            i += 2
            continue
        name, separator, value = token.partition("=")
        if name == flag and separator:
            found.append(value)
        i += 1
    if len(found) > 1:
        raise SystemExit(f"the training command contains {flag} more than once.")
    if found and not RUN_NAME.fullmatch(found[0]):
        raise SystemExit(f"invalid {flag} run name: {found[0]!r}")
    return found[0] if found else None


def _recovery_context(
    argv: list[str] | tuple[str, ...],
    state: RunState | None,
) -> RecoveryContext:
    """Resolve the recovery-relevant command flags and durable run name."""
    resume = _command_option(argv, "--resume")
    resume_as = _command_option(argv, "--resume-as")
    prefix_fork = any(token.partition("=")[0] == "--prefix-fork-from-run" for token in argv)
    retrying_fork = state is not None and state.status == "running" and (resume_as is not None or prefix_fork)
    return RecoveryContext(
        explicit_resume=resume,
        run_name=resume_as or resume or (state.run_name if state is not None else None),
        retrying_fork=retrying_fork,
    )


def _resume_fork_from_destination(argv: tuple[str, ...], run_name: str) -> tuple[str, ...]:
    """Replace source-fork flags with one resume from the fork destination."""
    command: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        flag = token.partition("=")[0]
        if flag not in FORK_SOURCE_FLAGS:
            command.append(token)
            i += 1
            continue
        if token == flag:
            i += 2
        else:
            i += 1
    return (*command, "--resume", run_name)


def plan_attempt(
    state: RunState | None,
    argv: tuple[str, ...],
    *,
    auto_resume: bool,
    checkpoint_found: bool,
) -> AttemptPlan:
    """Convert durable state into the command for one Modal retry attempt."""
    recovery = _recovery_context(argv, state)
    if state is not None and state.status == "failed":
        return AttemptPlan(action="fail", argv=argv, state=state)
    if state is not None and state.status == "succeeded":
        return AttemptPlan(action="complete", argv=argv, state=state)

    run_name = recovery.run_name
    command = argv
    if auto_resume and recovery.retrying_fork and checkpoint_found and run_name is not None:
        command = _resume_fork_from_destination(argv, run_name)
    elif auto_resume and not recovery.retrying_fork and recovery.explicit_resume is None and run_name is not None:
        if checkpoint_found:
            command = (*argv, "--resume", run_name)
        else:
            run_name = None
    return AttemptPlan(action="run", argv=command, state=RunState(status="running", run_name=run_name))


def redact_argv(argv: list[str] | tuple[str, ...]) -> str:
    """Format a command while hiding values of conventional credential flags."""
    redacted: list[str] = []
    hide_next = False
    for token in argv:
        if hide_next:
            redacted.append("***")
            hide_next = False
            continue
        flag, separator, _ = token.partition("=")
        if flag.lower() in SENSITIVE_FLAGS:
            if separator:
                redacted.append(f"{flag}=***")
            else:
                redacted.append(token)
                hide_next = True
            continue
        redacted.append(token)
    return shlex.join(redacted)


def gpu_request(gpu: str) -> str | list[str] | None:
    choices = [item.strip() for item in gpu.split(",") if item.strip()]
    if not choices:
        raise SystemExit("--gpu must contain at least one GPU type.")
    if len(choices) == 1 and choices[0].lower() == "none":
        return None
    if any(item.lower() == "none" for item in choices):
        raise SystemExit("--gpu none cannot be combined with GPU fallbacks.")
    return choices[0] if len(choices) == 1 else choices


def requested_disk_gib(args: Args) -> int:
    """Keep legacy launch defaults while enforcing O51's full-tier disk floor."""
    script = experiment_script(args.cmd)
    minimum = 3072 if script is not None and script.name == "051_correct_parameterization.py" else 0
    return max(args.disk_gib, minimum)


def function_resources(args: Args) -> FunctionResources:
    """Modal resource values in the units expected by ``App.function``."""
    memory_request = args.memory_gib * 1024
    memory = memory_request if args.memory_limit_gib is None else (memory_request, args.memory_limit_gib * 1024)
    return {
        "gpu": gpu_request(args.gpu),
        "cpu": (args.cpu, args.cpu_limit),
        "memory": memory,
        "ephemeral_disk": requested_disk_gib(args) * 1024,
        "timeout": args.timeout_hours * 60 * 60,
        "startup_timeout": args.startup_timeout_minutes * 60,
        "cloud": args.cloud,
        "region": args.region,
    }


def retry_policy(max_retries: int) -> modal.Retries:
    return modal.Retries(max_retries=max_retries, initial_delay=0.0)


def _state_path(launch_id: str) -> Path:
    try:
        parsed = uuid.UUID(launch_id)
    except ValueError as e:
        raise ValueError(f"invalid Modal launch id: {launch_id!r}") from e
    if parsed.hex != launch_id:
        raise ValueError(f"Modal launch id must be normalized UUID hex: {launch_id!r}")
    return STATE_ROOT / f"{launch_id}.json"


def read_state(path: Path) -> RunState | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Modal state at {path} is not a JSON object")
    return RunState(**raw)


def write_state(path: Path, state: RunState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(asdict(state), sort_keys=True) + "\n")
    tmp.replace(path)


def _commit_state(path: Path, state: RunState, volume_name: str) -> None:
    write_state(path, state)
    modal.Volume.from_name(volume_name).commit()


def _checkpoint_exists(run_name: str) -> bool:
    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    try:
        client.head_object(Bucket=os.environ["AWS_BUCKET"], Key=f"runs/{run_name}/latest.pt")
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
    return True


def _run_checked(
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    capture: bool = False,
) -> str:
    loguru.logger.info(f"running: {shlex.join(argv)}")
    result = subprocess.run(
        argv,
        cwd=REMOTE_ROOT,
        env=env,
        timeout=timeout,
        check=True,
        capture_output=capture,
        text=capture,
    )
    if capture and result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
    if capture and result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", flush=True)
    return result.stdout.strip() if capture else ""


def _configure_compiler_cache(env: dict[str, str]) -> None:
    """Keep compiler caches and scratch on the attempt's ephemeral SSD."""
    cache_paths = {
        "TORCHINDUCTOR_CACHE_DIR": LOCAL_CACHE_ROOT / "torchinductor",
        "TRITON_CACHE_DIR": LOCAL_CACHE_ROOT / "triton",
        "CUDA_CACHE_PATH": LOCAL_CACHE_ROOT / "cuda",
        "TMPDIR": LOCAL_CACHE_ROOT / "tmp",
    }
    for key, path in cache_paths.items():
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.touch()
        probe.unlink()
        env[key] = str(path)
    env["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    env["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"


def _configure_tracking_context(env: dict[str, str], modal_app_url: str | None) -> None:
    """Expose the Modal dashboard in every W&B run launched by this process."""
    if modal_app_url is None:
        return
    env["HAL_MODAL_APP_URL"] = modal_app_url
    modal_note = f"Modal: {modal_app_url}"
    existing_notes = env.get("WANDB_NOTES", "").strip()
    if modal_app_url not in existing_notes:
        env["WANDB_NOTES"] = f"{existing_notes}\n\n{modal_note}" if existing_notes else modal_note


def _prepare_remote(*, skip_sm120_probe: bool, require_cuda: bool = True) -> dict[str, str]:
    """Prepare one remote job and check CUDA resources when it requested a GPU."""
    compute_capability: str | None = None
    if require_cuda:
        subprocess.run(["mount", "-o", "remount,size=16g", "/dev/shm"], check=False, capture_output=True)
        shm_gib = shutil.disk_usage("/dev/shm").total / 2**30
        loguru.logger.info(f"/dev/shm = {shm_gib:.1f} GiB")
        if shm_gib < 1:
            raise RuntimeError(f"/dev/shm is {shm_gib:.2f} GiB; training requires at least 1 GiB")

        if not os.environ.get("DISPLAY"):
            os.environ["DISPLAY"] = ":99"
        if subprocess.run(["pgrep", "-x", "Xvfb"], check=False, capture_output=True).returncode != 0:
            with Path("/tmp/xvfb.log").open("ab") as xvfb_log:
                subprocess.Popen(
                    ["Xvfb", ":99", "-screen", "0", "1280x720x24"],
                    stdout=xvfb_log,
                    stderr=subprocess.STDOUT,
                )

        compute_capability = _run_checked(
            [
                "uv",
                "run",
                "python",
                "-c",
                "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; "
                "print(''.join(map(str, torch.cuda.get_device_capability())))",
            ],
            capture=True,
        )
        loguru.logger.info(f"CUDA compute capability = sm_{compute_capability}")
    else:
        loguru.logger.info("CPU-only job; skipping CUDA and Xvfb checks")

    env = os.environ.copy()
    # Modal injects its internal PyPI mirror through UV_INDEX_URL. Keep uv's
    # runtime project check aligned with the portable pypi.org URLs in uv.lock.
    env["UV_INDEX_URL"] = PYPI_INDEX
    _configure_compiler_cache(env)
    if compute_capability == "120":
        env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    free_gib = shutil.disk_usage(LOCAL_CACHE_ROOT).free / 2**30
    loguru.logger.info(f"ephemeral compile cache = {LOCAL_CACHE_ROOT}; free = {free_gib:.1f} GiB")
    if free_gib < 10:
        raise RuntimeError(f"compiler scratch has {free_gib:.1f} GiB free; training requires at least 10 GiB")
    _run_checked(
        [
            "uv",
            "run",
            "python",
            "-c",
            "import os, tempfile; from torch._inductor.codecache import cache_dir; "
            "assert cache_dir() == os.environ['TORCHINDUCTOR_CACHE_DIR']; "
            "assert tempfile.gettempdir() == os.environ['TMPDIR']",
        ],
        env=env,
    )
    if compute_capability == "120" and not skip_sm120_probe:
        _run_checked(["uv", "run", "docker/probe_sm120.py"], env=env, timeout=600)

    _run_checked(["uv", "run", "fetch"], env=env)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    loguru.logger.info(f"open-file limit {soft} -> {hard}")
    return env


def _follow_log(path: Path, stopped: threading.Event, run_names: queue.SimpleQueue[str]) -> None:
    with path.open(errors="replace") as stream:
        while True:
            line = stream.readline()
            if line:
                print(line, end="", flush=True)
                match = RUN_LINE.fullmatch(line.rstrip("\n"))
                if match is not None:
                    run_names.put(match.group(1).rstrip())
                continue
            if stopped.is_set():
                return
            time.sleep(0.2)


def _kill_group(pid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        return


def _drain_run_names(
    run_names: queue.SimpleQueue[str],
    state: RunState,
    *,
    state_path: Path,
    state_volume_name: str,
) -> tuple[RunState, str | None]:
    while True:
        try:
            found = run_names.get_nowait()
        except queue.Empty:
            return state, None
        if not RUN_NAME.fullmatch(found):
            return state, f"training emitted invalid run name {found!r}"
        if state.run_name is not None and state.run_name != found:
            return state, f"training changed run name from {state.run_name!r} to {found!r}"
        if state.run_name is None:
            state = RunState(status="running", run_name=found)
            _commit_state(state_path, state, state_volume_name)
            loguru.logger.info(f"saved retry run name {found!r}")


def _run_training(
    argv: tuple[str, ...],
    state: RunState,
    *,
    env: dict[str, str],
    state_path: Path,
    state_volume_name: str,
    stall_s: int,
) -> int:
    log_path = REMOTE_ROOT / "train.log"
    log_path.write_text("")
    run_names: queue.SimpleQueue[str] = queue.SimpleQueue()
    follower_stopped = threading.Event()
    log_follower = threading.Thread(
        target=_follow_log,
        args=(log_path, follower_stopped, run_names),
        name="train-log",
        daemon=True,
    )
    log_follower.start()

    interrupted = threading.Event()
    process: subprocess.Popen[bytes] | None = None
    failure: str | None = None
    code: int | None = None

    def interrupt(signum: int, _frame: object) -> None:
        interrupted.set()
        loguru.logger.warning(f"received signal {signum}; forwarding SIGINT to the training process group")
        if process is not None:
            _kill_group(process.pid, signal.SIGINT)

    old_handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with log_path.open("ab", buffering=0) as train_log:
            try:
                process = subprocess.Popen(
                    list(argv),
                    cwd=REMOTE_ROOT,
                    env=env,
                    stdout=train_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError as e:
                failure = f"could not start training command: {e}"
                code = 127
            else:
                loguru.logger.info(f"training pid={process.pid}: {redact_argv(argv)}")
                interrupted_at: float | None = None
                while process.poll() is None:
                    state, failure = _drain_run_names(
                        run_names,
                        state,
                        state_path=state_path,
                        state_volume_name=state_volume_name,
                    )
                    if failure is not None:
                        _kill_group(process.pid, signal.SIGKILL)
                        break
                    quiet_s = time.time() - log_path.stat().st_mtime
                    if quiet_s >= stall_s:
                        failure = f"no training log output for {int(quiet_s)} seconds"
                        loguru.logger.error(f"watchdog: {failure}")
                        _kill_group(process.pid, signal.SIGKILL)
                        break
                    if interrupted.is_set():
                        interrupted_at = interrupted_at or time.monotonic()
                        if time.monotonic() - interrupted_at >= INTERRUPT_GRACE_S:
                            loguru.logger.warning(
                                "training did not exit during interrupt grace period; sending SIGKILL"
                            )
                            _kill_group(process.pid, signal.SIGKILL)
                    time.sleep(0.5)
                code = process.wait()
    finally:
        if process is not None:
            _kill_group(process.pid, signal.SIGKILL)
            if process.poll() is None:
                process.wait()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        follower_stopped.set()
        log_follower.join(timeout=5)

    if failure is None:
        state, failure = _drain_run_names(
            run_names,
            state,
            state_path=state_path,
            state_volume_name=state_volume_name,
        )
    if code is None:
        raise RuntimeError("training process ended without an exit status")

    if interrupted.is_set():
        raise RuntimeError("training was interrupted; leaving retry state running")
    if code != 0 or failure is not None:
        failed = RunState(status="failed", run_name=state.run_name, exit_code=code)
        _commit_state(state_path, failed, state_volume_name)
        detail = failure or f"training exited with status {code}"
        raise RuntimeError(f"{detail}; saved terminal failure state so retries do not rerun it")
    _commit_state(state_path, RunState(status="succeeded", run_name=state.run_name), state_volume_name)
    return code


def _run_remote(spec: LaunchSpec) -> int:
    """One Modal Function attempt. Modal serializes this function into the configured image."""
    os.chdir(REMOTE_ROOT)
    loguru.logger.info(f"starting launch {spec.launch_id} from Git {spec.git_sha[:10]}")
    state_path = _state_path(spec.launch_id)
    state = read_state(state_path)
    recovery = _recovery_context(spec.argv, state)
    can_resume = recovery.explicit_resume is None or recovery.retrying_fork
    checkpoint_found = bool(
        spec.auto_resume and can_resume and recovery.run_name and _checkpoint_exists(recovery.run_name)
    )
    attempt = plan_attempt(state, spec.argv, auto_resume=spec.auto_resume, checkpoint_found=checkpoint_found)
    if attempt.action == "complete":
        loguru.logger.info(f"launch {spec.launch_id} already succeeded; nothing to run")
        return 0
    if attempt.action == "fail":
        raise RuntimeError(
            f"launch {spec.launch_id} already failed with exit code {attempt.state.exit_code}; refusing to rerun"
        )
    _commit_state(state_path, attempt.state, spec.state_volume)
    env = _prepare_remote(skip_sm120_probe=spec.skip_sm120_probe, require_cuda=spec.require_cuda)
    _configure_tracking_context(env, spec.modal_app_url)
    return _run_training(
        attempt.argv,
        attempt.state,
        env=env,
        state_path=state_path,
        state_volume_name=spec.state_volume,
        stall_s=spec.stall_s,
    )


def _ignored_python_commands(
    argv: list[str] | tuple[str, ...], ignore: modal.FilePatternMatcher
) -> tuple[tuple[Path, Path], ...]:
    """Return command-referenced Python files omitted by the build context."""
    selected: dict[Path, Path] = {}
    for token in argv:
        candidate = (ROOT / token).resolve()
        if candidate.suffix != ".py" or not candidate.is_file() or not candidate.is_relative_to(ROOT):
            continue
        relative = candidate.relative_to(ROOT)
        if ignore(relative):
            selected[candidate] = relative
    return tuple(selected.items())


def _fixture_manifest_json() -> str:
    """Serialize the canonical fixture manifest without coupling its image layer to source."""
    from hal.fixtures import ALL

    return json.dumps(
        [
            {
                "name": fixture.name,
                "sha256": fixture.sha256,
                "size_bytes": fixture.size_bytes,
                "dest": str(fixture.dest),
                "r2_key": fixture.r2_key,
                "url": fixture.url,
                "extract": fixture.extract,
            }
            for fixture in ALL
        ],
        sort_keys=True,
    )


def _image(
    tag: str,
    argv: list[str] | tuple[str, ...],
    secret: modal.Secret,
) -> modal.Image:
    ignore = modal.FilePatternMatcher.from_file(ROOT / ".dockerignore")
    fixture_helper = ROOT / "scripts" / "cache_modal_fixtures.py"
    dependency_image = (
        modal.Image.from_registry(tag)
        .add_local_file(ROOT / "pyproject.toml", str(REMOTE_ROOT / "pyproject.toml"), copy=True)
        .add_local_file(ROOT / "uv.lock", str(REMOTE_ROOT / "uv.lock"), copy=True)
        .workdir(str(REMOTE_ROOT))
        # Modal's build-time UV_INDEX_URL points at its internal mirror. If it
        # reaches uv, the resolver wants to rewrite every registry URL in the
        # portable lockfile and --locked fails before a Function is submitted.
        .run_commands(f"UV_INDEX_URL={PYPI_INDEX} uv sync --locked --no-install-project")
    )
    fixture_image = dependency_image.add_local_file(
        fixture_helper,
        "/opt/cache_modal_fixtures.py",
        copy=True,
    ).run_commands(
        "/opt/venv/bin/python /opt/cache_modal_fixtures.py",
        env={
            "HAL_FIXTURE_MANIFEST": _fixture_manifest_json(),
            "HAL_FIXTURE_ROOT": str(REMOTE_ROOT),
        },
        secrets=[secret],
    )
    source_image = fixture_image.add_local_dir(ROOT, str(REMOTE_ROOT), copy=True, ignore=ignore)
    for source, relative in _ignored_python_commands(argv, ignore):
        source_image = source_image.add_local_file(source, str(REMOTE_ROOT / relative), copy=True)
    # Dependencies and fixtures are complete in the parent layers. Install
    # only the local project offline. Ordinary source commits therefore reuse
    # both the dependency environment and the large immutable fixture layer.
    return source_image.run_commands(f"UV_INDEX_URL={PYPI_INDEX} uv sync --locked --offline --no-build-isolation")


def _app_name(args: Args, sha: str) -> str:
    if args.app_name is not None:
        return args.app_name
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"hal-{stamp}-{sha[:7]}"


def _print_request(
    args: Args,
    resources: FunctionResources,
    *,
    sha: str,
    name: str,
    launch_id: str,
) -> None:
    loguru.logger.info(f"app={name} launch={launch_id} git={sha[:10]} image={args.image}")
    loguru.logger.info(
        f"gpu={resources['gpu']} cpu={resources['cpu']} "
        f"memory=({args.memory_gib},{args.memory_limit_gib})GiB "
        f"ephemeral_ssd={resources['ephemeral_disk'] / 1024:g}GiB "
        f"cloud={args.cloud or 'auto'} region={args.region or 'auto'}"
    )
    loguru.logger.info(
        f"attempt_timeout={args.timeout_hours}h retries={args.max_retries} secret={args.secret!r} "
        f"state_volume={args.state_volume!r} gpu_memory_snapshot={args.gpu_memory_snapshot} "
        f"auto_resume={args.auto_resume}"
    )
    loguru.logger.info(f"command: {redact_argv(args.cmd)}")


def main(args: Args) -> None:
    validate_args(args)
    sha = preflight_git()
    client, secret = preflight_modal(args.secret)
    name = _app_name(args, sha)
    launch_id = uuid.uuid4().hex
    resources = function_resources(args)
    _print_request(args, resources, sha=sha, name=name, launch_id=launch_id)
    if args.dry_run:
        loguru.logger.success("dry run passed; no Modal App, image build, Volume, or GPU Function was started")
        return

    state_volume = modal.Volume.from_name(args.state_volume, create_if_missing=True)
    app = modal.App(name=name, tags={"provider": "modal", "git_sha": sha, "launch_id": launch_id})
    function = app.function(
        image=_image(args.image, args.cmd, secret),
        secrets=[secret],
        volumes={
            str(STATE_ROOT): state_volume,
        },
        retries=retry_policy(args.max_retries),
        max_containers=1,
        single_use_containers=True,
        serialized=True,
        include_source=False,
        enable_memory_snapshot=args.gpu_memory_snapshot,
        experimental_options=(
            {"enable_gpu_snapshot": True} if args.gpu_memory_snapshot and resources["gpu"] is not None else None
        ),
        name="train",
        **resources,
    )(_run_remote)
    with modal.enable_output(), app.run(name=name, client=client, detach=not args.wait):
        spec = LaunchSpec(
            argv=tuple(args.cmd),
            launch_id=launch_id,
            git_sha=sha,
            state_volume=args.state_volume,
            auto_resume=args.auto_resume,
            stall_s=args.stall_minutes * 60,
            skip_sm120_probe=args.skip_sm120_probe,
            require_cuda=resources["gpu"] is not None,
            modal_app_url=f"https://modal.com/apps/{app.app_id}",
        )
        call = function.spawn(spec)
        loguru.logger.success(f"submitted Modal App {app.app_id}, Function call {call.object_id}, Git {sha[:10]}")
        loguru.logger.info(f"dashboard: {spec.modal_app_url}")
        loguru.logger.info(f"logs: uv run modal app logs {app.app_id} -f")
        loguru.logger.info(f"stop: uv run modal app stop {app.app_id}")
        if args.wait:
            call.get()


if __name__ == "__main__":
    main(tyro.cli(Args))
