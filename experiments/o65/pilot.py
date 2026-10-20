"""Run the bounded O63 debug pilot with fixed inputs.

Build MSL, measure candidate pool sizes, kill training after checkpoint
publication, then resume the same configuration. Use --tiny for a local check.
The training CLI performs PPO and critic updates.
"""

import dataclasses
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

import torch
import tyro
import wandb

from hal.eval.self_play_checkpoint import read_training_manifest

ROOT = Path(__file__).resolve().parents[2]


@dataclasses.dataclass(frozen=True, slots=True)
class Args:
    output: Path
    local_delivery: Path | None = None
    tiny: bool = False
    wandb: bool = True
    cpus: tuple[int, ...] = ()
    checkpoint_retention: Literal["all", "compact"] = "compact"
    resume: bool = False
    compile_cache: Path | None = None
    publish_directory: Path | None = None
    publish_volume: str | None = None
    r2_prefix: str | None = None
    qualify_pool_sizes: bool = False


def compact_checkpoints(directory: Path, latest: int) -> None:
    """Keep resume state for the latest update and the kill/resume checks.

    Retain every actor and move retired manifests into a separate history
    directory before removing their unreferenced training and simulator files.
    Call only while the trainer and evaluation are stopped.
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
        if manifest["update"] in (latest, 256, 288):
            retained.update(shard["name"] for shard in manifest["shards"])
        else:
            retired.mkdir(exist_ok=True)
            target = retired / path.name
            path.rename(target)
    for pattern in ("*.rank0.pt", "*.sim.zlib"):
        for path in directory.glob(pattern):
            if path.name not in retained:
                path.unlink()


def execute(command: list[str], log: Path, environment: dict[str, str]) -> None:
    print(f"O65: {log.name}", flush=True)
    with log.open("ab") as target:
        subprocess.run(command, stdout=target, stderr=subprocess.STDOUT, env=environment, check=True, cwd=ROOT)


def kill_at_checkpoint(command: list[str], manifest: Path, log: Path, environment: dict[str, str]) -> None:
    """Kill the process group only after atomic checkpoint publication."""
    if manifest.exists():
        raise FileExistsError(manifest)
    with log.open("ab") as target:
        process = subprocess.Popen(
            command, stdout=target, stderr=subprocess.STDOUT, env=environment, cwd=ROOT, start_new_session=True
        )
        try:
            while not manifest.is_file():
                if process.poll() is not None:
                    raise RuntimeError(f"training exited {process.returncode} before the kill boundary; see {log}")
                time.sleep(0.25)
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            if process.returncode != -signal.SIGKILL:
                raise RuntimeError("pilot process did not reach the forced-kill check")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def training_command(
    delivery: Path,
    output: Path,
    cpus: tuple[int, ...],
    *,
    matches: int,
    groups: int,
    updates: int,
    checkpoint_every: int,
    warmup: int,
    verify: bool,
    deterministic: bool = False,
    compile_cache: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "experiments/065_selfplay_ppo.py"),
        "train",
        "--checkpoint",
        str(delivery / "payloads/final.pt"),
        "--sampler",
        str(ROOT / "experiments/o65/pilot-sampler.json"),
        "--data-dir",
        str(delivery / "inputs/msl/data"),
        "--output",
        str(output),
        "--cpus",
        *(str(cpu) for cpu in cpus),
        "--groups",
        str(groups),
        "--matches-per-shard",
        str(matches),
        "--updates",
        str(updates),
        "--training.windows-per-step",
        "512",
        "--training.seed",
        "65",
        "--training.critic-warmup-updates",
        str(warmup),
        "--training.checkpoint-interval",
        str(checkpoint_every),
        "--evaluation-every",
        "0",
    ]
    if compile_cache is not None:
        command.extend(("--compile-cache", str(compile_cache)))
    if deterministic:
        command.append("--training.deterministic")
    if verify:
        command.append("--verify-resume")
    return command


def qualify_pool(rows: list[list[dict[str, float]]]) -> None:
    if len(rows) != 3 or any(not run for run in rows):
        raise ValueError("pool qualification requires three nonempty candidates")
    rates = []
    for run in rows:
        samples = [row["systems/match_frames_per_second"] for row in run]
        if any(not math.isfinite(rate) or rate <= 0 for rate in samples):
            raise ValueError("pool qualification requires finite positive throughput")
        memory = [row["systems/peak_reserved_bytes"] for row in run]
        if any(not math.isfinite(value) or value < 0 for value in memory):
            raise ValueError("pool qualification requires finite nonnegative memory")
        if any(value >= 80 * 2**30 for value in memory):
            raise ValueError("pool qualification exceeded the 80 GiB memory limit")
        rates.append(sum(samples) / len(samples))
    if rates[-1] < 0.9 * max(rates):
        raise ValueError(
            "the planned 1024-match pool is over 10% slower; review throughput.json before changing the pool"
        )


def evaluation_command(
    delivery: Path,
    checkpoint: Path,
    protocol: Path,
    output: Path,
    *,
    held_out: bool,
    simulators: tuple[Literal["msl", "dolphin"], ...],
    anchors: tuple[Path, ...] = (),
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/eval_o65.py"),
        "--checkpoint",
        str(checkpoint),
        "--anchors",
        *(str(path) for path in (delivery / "payloads/final.pt", *anchors)),
        "--data-dir",
        str(delivery / "inputs/msl/data"),
        "--protocol",
        str(protocol),
        "--output",
        str(output),
        "--split",
        "held_out" if held_out else "selection",
        "--cadence-grid",
        "8/4",
        "10/2",
        "--lambda-grid",
        "1",
        "--device",
        "cuda",
        "--simulators",
        *simulators,
    ]


def log_evaluation(path: Path, update: int, environment: dict[str, str], *, online: bool) -> None:
    result = json.loads((path / "results.json").read_text())
    if result["format"] not in ("hal.o65.evaluation-results.v1", "hal.o65.evaluation-results.v2") or result[
        "split"
    ] not in ("selection", "held_out"):
        raise ValueError("unexpected pilot evaluation result")
    metrics: dict[str, float] = {"global_step": float(update)}
    for index, group in enumerate(result["groups"]):
        for simulator, values in group["metrics"].items():
            prefix = f"evaluation/{result['split']}_{simulator}_cell{index}"
            for name in ("net_stocks_per_min", "win_rate"):
                for statistic in ("mean", "low", "high"):
                    value = values[name][statistic]
                    if value is not None:
                        metrics[f"{prefix}_{name}_{statistic}"] = float(value)
            for name, interval in values.get("events", {}).items():
                if interval["mean"] is not None:
                    metrics[f"{prefix}_{name}"] = float(interval["mean"])
            for name in ("matches", "completed_pairs", "net_stock_lcb"):
                if values[name] is not None:
                    metrics[f"{prefix}_{name}"] = float(values[name])
    (path / "wandb-metrics.json").write_text(json.dumps(metrics, indent=2))
    if online:
        with wandb.init(project="hal-o65", id=environment["WANDB_RUN_ID"], resume="must"):
            wandb.define_metric("global_step")
            wandb.define_metric("*", step_metric="global_step")
            wandb.log(metrics)


def latest_checkpoint(directory: Path) -> Path | None:
    manifests = sorted(directory.glob("update-*.json"))
    if not manifests:
        return None
    latest = manifests[-1]
    manifest = read_training_manifest(latest)
    if latest.name != f"update-{manifest['update']:08d}.json":
        raise ValueError("latest checkpoint filename differs from its update")
    for shard in manifest["shards"]:
        path = directory / shard["name"]
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != shard["sha256"]:
            raise ValueError(f"latest checkpoint has a corrupt shard: {path}")
    return latest


def snapshot_anchors(directory: Path, update: int) -> tuple[Path, ...]:
    # Evaluation uses the same recent snapshot cadence as the training pool.
    updates = [value for value in range(256, update + 1, 256)][-3:]
    snapshots: list[Path] = []
    for value in updates:
        path = directory / f"update-{value:08d}.json"
        if not path.is_file():
            path = directory.parent / "retired-checkpoints" / path.name
        manifest = read_training_manifest(path)
        actors = [shard for shard in manifest["shards"] if shard["role"] == "actor"]
        if len(actors) != 1:
            raise ValueError("snapshot anchor requires one published actor")
        snapshots.append(directory / actors[0]["name"])
    return tuple(snapshots)


def publish(args: Args, output: Path) -> None:
    values = (args.publish_directory, args.publish_volume, args.r2_prefix)
    if all(value is None for value in values):
        return
    if any(value is None for value in values):
        raise ValueError("pilot publication requires directory, volume and R2 prefix")
    execute(
        [
            sys.executable,
            "-m",
            "scripts.o65_pilot_state",
            "--source",
            str(output),
            "--destination",
            str(args.publish_directory),
            "--volume",
            str(args.publish_volume),
            "--r2-prefix",
            str(args.r2_prefix),
        ],
        output.parent / "publish.log",
        dict(os.environ),
    )


def trim_interrupted_metrics(path: Path, update: int) -> None:
    if not path.is_file():
        return
    lines = path.read_text().splitlines()
    keep = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            break
        if row["global_step"] <= update:
            keep.append(line)
    previous = path.with_name(f"metrics-interrupted-{time.time_ns()}.jsonl")
    path.rename(previous)
    path.write_text("".join(line + "\n" for line in keep))


def terminate(_signal: int, _frame: object) -> None:
    raise SystemExit("pilot interrupted; resume from the latest published checkpoint")


def resume_verified(path: Path) -> bool:
    if not path.is_file():
        return False
    report = json.loads(path.read_text())
    if report.get("format") != "hal.o65.resume-check.v5" or report.get("passed") is not True:
        raise RuntimeError("resume verification failed or has an unsupported report; do not continue training")
    return True


def consume_held_out(output: Path, update: int, protocol: Path) -> Path:
    consumed = output / "held-out-consumed.json"
    if consumed.exists():
        raise RuntimeError("held-out evaluation was already consumed; preserve partial results without retry")
    with consumed.open("x") as target:
        json.dump(
            {"update": update, "protocol": str(protocol), "reason": "consume before starting held-out play"}, target
        )
    return consumed


def main(args: Args) -> None:
    signal.signal(signal.SIGTERM, terminate)
    os.chdir(ROOT)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    available = tuple(sorted(os.sched_getaffinity(0)))
    cpus = args.cpus or available[: 2 if args.tiny else 16]
    if len(cpus) != (2 if args.tiny else 16) or len(set(cpus)) != len(cpus) or set(cpus) - set(available):
        raise ValueError("pilot needs 2 (tiny) or 16 distinct available CPUs; select physical cores on the GPU socket")
    if not args.tiny:
        if not args.wandb:
            raise ValueError("the hardware pilot requires W&B logging")
        if not os.environ.get("WANDB_API_KEY"):
            raise ValueError("the hardware pilot requires WANDB_API_KEY")
        if len(available) < 17 or torch.cuda.device_count() != 1 or "RTX PRO 6000" not in torch.cuda.get_device_name():
            raise ValueError("the hardware pilot requires one RTX PRO 6000 and at least 17 available vCPUs")
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TORCHINDUCTOR_COMPILE_THREADS="1")
    delivery = args.local_delivery.resolve() if args.local_delivery is not None else output / "delivery"
    if args.local_delivery is None:
        execute(
            [sys.executable, str(ROOT / "scripts/o65_delivery.py"), "--action", "install", "--output", str(delivery)],
            output / "delivery.log",
            environment,
        )
    execute(
        [sys.executable, str(ROOT / "scripts/o65_delivery.py"), "--action", "verify", "--output", str(delivery)],
        output / "verify-delivery.log",
        environment,
    )
    environment["PYTHONPATH"] = os.pathsep.join((str(delivery / "msl"), str(ROOT)))
    environment["MSL_CORE_LIBRARY"] = str(delivery / "msl/build/melee_core/python/libmelee_core.so")
    environment["HAL_NATIVE_LIBRARY"] = str(delivery / "libhal_native.so")
    execute(["bash", str(ROOT / "docker/o65-build-msl.sh"), str(delivery)], output / "build.log", environment)
    if not args.tiny:
        for fixture in ("ssbm.ciso", "dolphin-exiai"):
            execute(
                [sys.executable, "-m", "hal.scripts.fetch", "--name", fixture],
                output / f"fetch-{fixture}.log",
                environment,
            )
    tiny_protocol = output / "tiny-evaluation.json"
    if args.tiny:
        raw = json.loads((ROOT / "experiments/o65/pilot-evaluation.json").read_text())
        raw["selection"]["seeds"] = [17]
        raw["max_frames"] = 132
        tiny_protocol.write_text(json.dumps(raw))
    protocol = tiny_protocol if args.tiny else ROOT / "experiments/o65/pilot-evaluation.json"
    sizes = ((1, 2, 4) if args.tiny else (16, 32, 64)) if args.qualify_pool_sizes else ()
    qualification = []
    for size in sizes:
        run = output / f"qualify-{size * len(cpus)}"
        execute(
            training_command(
                delivery,
                run,
                cpus,
                matches=size,
                groups=1,
                updates=3,
                checkpoint_every=3,
                warmup=0,
                verify=False,
            ),
            output / f"qualify-{size * len(cpus)}.log",
            environment,
        )
        metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        qualification.append({"matches": size * len(cpus), "steady_updates": metrics[1:]})
    (output / "throughput.json").write_text(json.dumps(qualification, indent=2))
    if qualification and not args.tiny:
        qualify_pool([row["steady_updates"] for row in qualification])
    if not args.tiny and not (output / "control/results.json").is_file():
        control = output / "control"
        if control.exists():
            control.rename(control.with_name(f"control-interrupted-{time.time_ns()}"))
        execute(
            evaluation_command(
                delivery,
                delivery / "payloads/final.pt",
                protocol,
                output / "control",
                held_out=False,
                simulators=("msl", "dolphin"),
            ),
            output / "control.log",
            environment,
        )
    # No automatic pool or optimizer sweep: each changes the experiment.
    run = output / "train"
    kill_update, updates = (2, 4) if args.tiny else (256, 2000)
    checkpoint_every = 1 if args.tiny else 32
    command = training_command(
        delivery,
        run,
        cpus,
        matches=1 if args.tiny else 64,
        groups=2 if args.tiny else 4,
        updates=kill_update + checkpoint_every,
        checkpoint_every=checkpoint_every,
        warmup=0 if args.tiny else 172,
        verify=False,
        compile_cache=args.compile_cache,
    )
    if args.wandb:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        identity = hashlib.sha256(str(output).encode()).hexdigest()[:10]
        environment.update(WANDB_RUN_ID=f"o65-debug-{sha[:8]}-{identity}", WANDB_RESUME="allow")
        command.extend(("--wandb-project", "hal-o65"))
    manifest = run / f"checkpoints/update-{kill_update:08d}.json"
    # Stop at each evaluation boundary, so one GPU never trains and evaluates concurrently.
    boundaries = list(range(1, updates + 1)) if args.tiny else [*range(32, updates + 1, 32), updates]
    boundaries = sorted(set(boundaries))
    previous = latest_checkpoint(run / "checkpoints") if args.resume else None
    restored_update = read_training_manifest(previous)["update"] if previous is not None else 0
    if args.resume and previous is None and run.exists():
        run.rename(run.with_name(f"train-interrupted-{time.time_ns()}"))
    if previous is not None:
        trim_interrupted_metrics(run / "metrics.jsonl", restored_update)
    if restored_update >= kill_update and not (output / "kill.json").is_file():
        raise RuntimeError(
            "preemption interrupted the forced-kill qualification; inspect its preserved boundary before continuing"
        )
    verify_boundary = kill_update + checkpoint_every
    if restored_update >= verify_boundary and not resume_verified(run / "resume-check.json"):
        check = command.copy()
        check[check.index("--updates") + 1] = str(verify_boundary + 1)
        check.extend(
            (
                "--resume",
                str(run / f"checkpoints/update-{verify_boundary:08d}.json"),
                "--verify-resume",
                "--verify-only",
            )
        )
        execute(check, output / "resume-verification.log", environment)
    publish(args, output)
    for boundary in boundaries:
        checkpoint = run / f"checkpoints/update-{boundary:08d}.json"
        result = output / "selection" / f"update-{boundary:08d}"
        if boundary < restored_update:
            continue
        if boundary == restored_update and (result / "results.json").is_file():
            continue
        current = command.copy()
        current[current.index("--updates") + 1] = str(boundary)
        if previous is not None:
            current.extend(("--resume", str(previous)))
        if boundary == restored_update:
            pass
        elif boundary == kill_update:
            # Give the child another update to do after publication, so the kill cannot race normal exit.
            current[current.index("--updates") + 1] = str(boundary + checkpoint_every)
            kill_at_checkpoint(current, manifest, output / "before-kill.log", environment)
            (output / "kill.json").write_text(
                json.dumps({"signal": "SIGKILL", "manifest": str(manifest), "update": kill_update})
            )
        else:
            if boundary == kill_update + checkpoint_every:
                current.append("--verify-resume")
            execute(current, output / f"train-{boundary:08d}.log", environment)
        previous = checkpoint
        publish(args, output)
        simulators: tuple[Literal["msl", "dolphin"], ...] = (
            ("msl", "dolphin") if not args.tiny and boundary % 128 == 0 else ("msl",)
        )
        if (result / "results.json").is_file():
            continue
        if result.exists():
            result.rename(result.with_name(f"{result.name}-interrupted-{time.time_ns()}"))
        anchors = snapshot_anchors(run / "checkpoints", boundary)
        execute(
            evaluation_command(
                delivery, previous, protocol, result, held_out=False, simulators=simulators, anchors=anchors
            ),
            output / f"evaluation-{boundary:08d}.log",
            environment,
        )
        log_evaluation(result, boundary, environment, online=args.wandb)
        if args.checkpoint_retention == "compact":
            compact_checkpoints(run / "checkpoints", boundary)
        publish(args, output)
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    last = next(row for row in reversed(metrics) if "train/warming" in row)
    if not args.tiny and (last.get("train/warming", 1) or last.get("train/actor_steps", 0) == 0):
        raise RuntimeError("pilot ended without calibrated PPO training; inspect warmup before extending")
    if not args.tiny and not (output / "held-out/results.json").is_file():
        consume_held_out(output, updates, protocol)
        publish(args, output)
        final_protocol = output / "final-evaluation.json"
        raw = json.loads(protocol.read_text())
        raw["max_frames"] = 28924
        final_protocol.write_text(json.dumps(raw))
        execute(
            evaluation_command(
                delivery,
                run / f"checkpoints/update-{updates:08d}.json",
                final_protocol,
                output / "held-out",
                held_out=True,
                simulators=("msl", "dolphin"),
            ),
            output / "held-out.log",
            environment,
        )
        log_evaluation(output / "held-out", updates, environment, online=args.wandb)
    publish(args, output)
    print(f"O65 complete: {output}; inspect gameplay intervals before claiming a learning signal", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
