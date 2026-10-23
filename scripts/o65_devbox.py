"""Keep a B200 Modal sandbox alive for fast O65 rollout benchmarks.

The sandbox image holds the repository at a base commit. Each benchmark first
copies the local files that differ from that commit into the sandbox, so code
changes reach the GPU without an image build. Inductor and Triton caches live
on a volume, so a rerun compiles only graphs whose code or shapes changed.
"""

import dataclasses
import io
import json
import shlex
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Literal

import modal
import tyro

from scripts import launch_o65_modal as launch

APP = "o65-devbox"
CACHE_VOLUME = "hal-o65-devbox"
CACHE = Path("/cache")
LOCAL_RESULTS = Path.home() / "data/scratch/o65-devbox"
SYNCED = ("hal", "experiments", "scripts", "tests")
SHARD_CPUS = 16


@dataclasses.dataclass(frozen=True, slots=True)
class Args:
    action: tyro.conf.Positional[Literal["up", "sync", "bench", "exec", "down"]]
    # Each name keeps its own sandbox, so parallel workstreams measure on separate GPUs.
    name: str = "main"
    # up: the commit the image is built from; defaults to HEAD, which must be pushed to origin.
    sha: str | None = None
    # bench: matches per simulator shard; 16 shards run per GPU.
    matches_per_shard: tuple[int, ...] = (64,)
    # bench: frames per rollout, in whole 128-frame blocks.
    rollout_frames: int = 128
    # bench: profile whole updates (rollout, preparation, actor, critic) instead of rollout ticks.
    full: bool = False
    # bench: training windows per optimizer step; the default of 512 runs out of memory at 1024 matches.
    windows_per_step: int = 64
    # bench: use another sandbox directory if the shared volume exhausts its file limit.
    cache_root: Path = CACHE
    # exec: a shell command to run in the repository root.
    command: str = ""
    # Sandbox lifetime; idle sandboxes stop after idle_minutes without an exec.
    hours: float = 6.0
    idle_minutes: int = 30


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _state(name: str) -> Path:
    return LOCAL_RESULTS / "state" / f"{name}.json"


def _sandbox(name: str) -> tuple[modal.Sandbox, str]:
    state = json.loads(_state(name).read_text())
    return modal.Sandbox.from_id(state["sandbox"]), state["sha"]


def _run(sandbox: modal.Sandbox, command: str) -> int:
    process = sandbox.exec("bash", "-c", f"{command} 2>&1", workdir=str(launch.REMOTE_ROOT))
    for line in process.stdout:
        print(line, end="", flush=True)
    process.wait()
    return process.returncode


def _capture(sandbox: modal.Sandbox, command: str) -> str:
    process = sandbox.exec("bash", "-c", command, workdir=str(launch.REMOTE_ROOT))
    output = process.stdout.read()
    process.wait()
    if process.returncode:
        raise RuntimeError(f"{command} failed: {process.stderr.read()}")
    return output


def up(name: str, sha: str | None, hours: float, idle_minutes: int) -> None:
    # A shallow fetch of an abbreviated SHA fails, so the image gets the full SHA.
    sha = _git("rev-parse", sha or "HEAD")
    if _git("branch", "-r", "--contains", sha) == "":
        raise ValueError(f"{sha} is not on origin; push it before building the image")
    secret = modal.Secret.from_name("hal", required_keys=list(launch.AWS_KEYS))
    app = modal.App.lookup(APP, create_if_missing=True)
    sandbox = modal.Sandbox.create(
        "sleep",
        "infinity",
        app=app,
        image=launch.build_image(sha, secret),
        secrets=[secret],
        gpu="B200",
        cpu=(20.0, 20.0),
        memory=(launch.MEMORY_MIB, launch.MEMORY_MIB),
        volumes={
            str(launch.RESULTS): modal.Volume.from_name(launch.VOLUME),
            str(CACHE): modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True),
        },
        timeout=int(hours * 3600),
        idle_timeout=idle_minutes * 60,
    )
    state = _state(name)
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"sandbox": sandbox.object_id, "sha": sha}))
    print(f"sandbox {name} {sandbox.object_id} at {sha}", flush=True)


def sync(sandbox: modal.Sandbox, sha: str) -> None:
    changed = _git("diff", "--name-only", sha, "--", *SYNCED).splitlines()
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *SYNCED).splitlines()
    paths = sorted({path for path in (*changed, *untracked) if Path(path).is_file()})
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path in paths:
            archive.add(path)
    process = sandbox.exec("tar", "-x", "-C", str(launch.REMOTE_ROOT))
    process.stdin.write(buffer.getvalue())
    process.stdin.write_eof()
    process.stdin.drain()
    process.wait()
    if process.returncode:
        raise RuntimeError(f"sync failed: {process.stderr.read()}")
    print(f"synced {len(paths)} files over {sha[:12]}", flush=True)


def bench(
    sandbox: modal.Sandbox,
    matches_per_shard: int,
    rollout_frames: int,
    *,
    full: bool,
    windows_per_step: int,
    cache_root: Path,
) -> int:
    name = f"bench-{time.strftime('%Y%m%d-%H%M%S')}-m{matches_per_shard}{'-full' if full else ''}"
    output = cache_root / "runs" / name
    root = launch.REMOTE_ROOT
    env = {
        "WANDB_MODE": "disabled",
        "HAL_NATIVE_LIBRARY": f"{root}/delivery/libhal_native.so",
        "TMPDIR": str(cache_root / "tmp"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache_root / "inductor"),
        "TRITON_CACHE_DIR": str(cache_root / "triton"),
        "TORCHINDUCTOR_COMPILE_THREADS": "16",
        "TORCHINDUCTOR_ENABLE_PDL": "1",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    argv = [
        "/opt/venv/bin/python",
        f"{root}/experiments/065_selfplay_ppo.py",
        "train",
        "--checkpoint",
        str(launch.RESULTS / "o64-inputs/step-0098304.pt"),
        "--sampler",
        f"{root}/experiments/o65/production-sampler.json",
        "--data-dir",
        f"{root}/delivery/inputs/msl/data",
        "--output",
        str(output),
        "--cpus",
        "$CPUS",
        "--matches-per-shard",
        str(matches_per_shard),
        "--cadence",
        "production",
        "--rollout-frames",
        str(rollout_frames),
        "--groups",
        "1",
        "--updates",
        "3",
        "--training.windows-per-step",
        str(windows_per_step),
        "--training.seed",
        "65",
        "--training.critic-warmup-updates",
        "0",
        "--training.head-start-frames",
        "0",
        "--training.reward-weights.damage",
        "0.1",
        "--training.reward-weights.stock",
        "120",
        "--training.reward-weights.game-win",
        "120",
        "profiling:profile-config",
        "--profiling.warm-updates",
        "1",
        *(() if full else ("--profiling.rollout-only",)),
    ]
    exports = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    # Sandbox CPU numbering varies; take the first available cores for simulator shards.
    cpus = f"import os; print(*sorted(os.sched_getaffinity(0))[:{SHARD_CPUS}])"
    arguments = " ".join(argument if argument == "$CPUS" else shlex.quote(argument) for argument in argv)
    command = f"mkdir -p {shlex.quote(str(cache_root / 'tmp'))} && CPUS=$(/opt/venv/bin/python -c '{cpus}') && env {exports} {arguments}"
    code = _run(sandbox, command)
    if _run(sandbox, f"test -d {shlex.quote(str(output))}"):
        print("no report written", flush=True)
        return code
    # Traces and reports go to one local directory per run for offline analysis.
    local = LOCAL_RESULTS / "runs" / name
    local.mkdir(parents=True)
    for path in _capture(sandbox, f"find {shlex.quote(str(output))} -type f").splitlines():
        target = local / Path(path).relative_to(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(sandbox.filesystem.read_bytes(path))
    print(f"report: {local}", flush=True)
    return code


def main(args: Args) -> None:
    if args.action == "up":
        up(args.name, args.sha, args.hours, args.idle_minutes)
        return
    sandbox, sha = _sandbox(args.name)
    if args.action == "down":
        sandbox.terminate()
        _state(args.name).unlink()
        return
    sync(sandbox, sha)
    if args.action == "exec":
        raise SystemExit(_run(sandbox, args.command))
    if args.action == "bench":
        codes = [
            bench(
                sandbox,
                matches,
                args.rollout_frames,
                full=args.full,
                windows_per_step=args.windows_per_step,
                cache_root=args.cache_root,
            )
            for matches in args.matches_per_shard
        ]
        raise SystemExit(max(codes))


if __name__ == "__main__":
    main(tyro.cli(Args))
