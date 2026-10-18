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
    checkpoint_retention: Literal["all", "compact"] = "all"


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
    with log.open("xb") as target:
        subprocess.run(command, stdout=target, stderr=subprocess.STDOUT, env=environment, check=True, cwd=ROOT)


def kill_at_checkpoint(command: list[str], manifest: Path, log: Path, environment: dict[str, str]) -> None:
    """Kill the process group only after atomic checkpoint publication."""
    if manifest.exists():
        raise FileExistsError(manifest)
    with log.open("xb") as target:
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
        "--compiled",
        "--training.deterministic",
        "--training.windows-per-step",
        "32",
        "--training.seed",
        "65",
        "--training.critic-warmup-updates",
        str(warmup),
        "--training.checkpoint-interval",
        str(checkpoint_every),
        "--evaluation-every",
        "0",
    ]
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
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/eval_o65.py"),
        "--checkpoint",
        str(checkpoint),
        "--anchors",
        str(delivery / "payloads/final.pt"),
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
            for name in ("matches", "completed_pairs", "net_stock_lcb"):
                if values[name] is not None:
                    metrics[f"{prefix}_{name}"] = float(values[name])
    (path / "wandb-metrics.json").write_text(json.dumps(metrics, indent=2))
    if online:
        with wandb.init(project="hal-o65", id=environment["WANDB_RUN_ID"], resume="must"):
            wandb.define_metric("global_step")
            wandb.define_metric("*", step_metric="global_step")
            wandb.log(metrics)


def main(args: Args) -> None:
    os.chdir(ROOT)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    available = tuple(sorted(os.sched_getaffinity(0)))
    cpus = args.cpus or available[: 2 if args.tiny else 16]
    if len(cpus) != (2 if args.tiny else 16) or len(set(cpus)) != len(cpus) or set(cpus) - set(available):
        raise ValueError("pilot needs 2 (tiny) or 16 distinct available CPUs; select physical cores on the GPU socket")
    if not args.tiny:
        if not args.wandb:
            raise ValueError("the hardware pilot requires W&B logging")
        if not os.environ.get("WANDB_API_KEY"):
            raise ValueError("the hardware pilot requires WANDB_API_KEY")
        if (
            len(available) != 48
            or torch.cuda.device_count() != 1
            or "RTX PRO 6000" not in torch.cuda.get_device_name()
        ):
            raise ValueError("the hardware pilot requires one RTX PRO 6000 and 48 available vCPUs")
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
    sizes = (1, 2, 4) if args.tiny else (16, 32, 64)
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
    if not args.tiny:
        qualify_pool([row["steady_updates"] for row in qualification])
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
    kill_update, updates = (2, 4) if args.tiny else (256, 512)
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
    )
    if args.wandb:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        identity = hashlib.sha256(str(output).encode()).hexdigest()[:10]
        environment.update(WANDB_RUN_ID=f"o65-debug-{sha[:8]}-{identity}", WANDB_RESUME="allow")
        command.extend(("--wandb-project", "hal-o65"))
    manifest = run / f"checkpoints/update-{kill_update:08d}.json"
    # Stop at each evaluation boundary, so one GPU never trains and evaluates concurrently.
    boundaries = range(1, updates + 1) if args.tiny else range(32, updates + 1, 32)
    previous: Path | None = None
    for boundary in boundaries:
        current = command.copy()
        current[current.index("--updates") + 1] = str(boundary)
        if previous is not None:
            current.extend(("--resume", str(previous)))
        if boundary == kill_update:
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
        previous = run / f"checkpoints/update-{boundary:08d}.json"
        simulators: tuple[Literal["msl", "dolphin"], ...] = (
            ("msl", "dolphin") if not args.tiny and boundary % 128 == 0 else ("msl",)
        )
        result = output / "selection" / f"update-{boundary:08d}"
        execute(
            evaluation_command(delivery, previous, protocol, result, held_out=False, simulators=simulators),
            output / f"evaluation-{boundary:08d}.log",
            environment,
        )
        log_evaluation(result, boundary, environment, online=args.wandb)
        if args.checkpoint_retention == "compact":
            compact_checkpoints(run / "checkpoints", boundary)
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    last = next(row for row in reversed(metrics) if "train/warming" in row)
    if not args.tiny and (last.get("train/warming", 1) or last.get("train/actor_steps", 0) == 0):
        raise RuntimeError("pilot ended without calibrated PPO training; inspect warmup before extending")
    if not args.tiny:
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
    print(f"O65 complete: {output}; inspect gameplay intervals before claiming a learning signal", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
