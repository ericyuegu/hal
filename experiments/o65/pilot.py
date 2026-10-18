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

import torch
import tyro

ROOT = Path(__file__).resolve().parents[2]


@dataclasses.dataclass(frozen=True, slots=True)
class Args:
    output: Path
    local_delivery: Path | None = None
    tiny: bool = False
    wandb: bool = True
    cpus: tuple[int, ...] = ()


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
    protocol: Path | None,
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
        "0" if protocol is None else str(checkpoint_every * 4),
    ]
    if protocol is not None:
        command.extend(("--evaluation-protocol", str(protocol), "--evaluation-device", "cuda"))
    if verify:
        command.append("--verify-resume")
    return command


def qualify_pool(rows: list[list[dict[str, float]]]) -> None:
    rates = []
    for run in rows:
        rate = sum(row["systems/match_frames_per_second"] for row in run) / len(run)
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("pool qualification requires finite positive throughput")
        if any(row["systems/peak_reserved_bytes"] >= 80 * 2**30 for row in run):
            raise ValueError("pool qualification exceeded the 80 GiB memory limit")
        rates.append(rate)
    if rates[-1] < 0.9 * max(rates):
        raise ValueError(
            "the planned 1024-match pool is over 10% slower; review throughput.json before changing the pool"
        )


def evaluation_command(delivery: Path, checkpoint: Path, protocol: Path, output: Path, *, held_out: bool) -> list[str]:
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
    ]


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
    # A local installation must pass the same committed manifest identity.
    if (delivery / "delivery.json").read_bytes() != (ROOT / "experiments/o65/pilot-inputs.json").read_bytes():
        raise ValueError("installed delivery differs from the committed manifest")
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
                protocol=None,
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
            evaluation_command(delivery, delivery / "payloads/final.pt", protocol, output / "control", held_out=False),
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
        updates=updates,
        checkpoint_every=checkpoint_every,
        warmup=0 if args.tiny else 172,
        protocol=protocol,
        verify=True,
    )
    if args.wandb:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        identity = hashlib.sha256(str(output).encode()).hexdigest()[:10]
        environment.update(WANDB_RUN_ID=f"o65-debug-{sha[:8]}-{identity}", WANDB_RESUME="allow")
        command.extend(("--wandb-project", "hal-o65"))
    manifest = run / f"checkpoints/update-{kill_update:08d}.json"
    kill_at_checkpoint(command, manifest, output / "before-kill.log", environment)
    (output / "kill.json").write_text(
        json.dumps({"signal": "SIGKILL", "manifest": str(manifest), "update": kill_update})
    )
    execute([*command, "--resume", str(manifest)], output / "resumed.log", environment)
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
            ),
            output / "held-out.log",
            environment,
        )
    print(f"O65 complete: {output}; inspect gameplay intervals before claiming a learning signal", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
