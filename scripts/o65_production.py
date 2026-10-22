"""Construct O64 torchrun training, measurement and final evaluation commands."""

import dataclasses
import json
import os
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import torch
import tyro

from hal import r2
from hal.eval.self_play_checkpoint import read_training_manifest
from hal.eval.self_play_checkpoint import sha256
from hal.inference.action_sequence_artifact import source_stats_paths
from hal.training import self_play_publication

ROOT = Path(__file__).resolve().parents[1]
O64_KEY = (
    "runs/261017-184242_064_production_percentile_o64v1-bootstrap30-d1024-L12-h16-c256-t1024x6-ff4096-"
    "dual-nonlinear-head-rc1e128-drop0.2-muon-wsd-b150-wmax10-g0.99855__bootstrap30-u95-dense30/"
    "checkpoints/step-0098304.pt"
)
O64_SHA256 = "e6d6303fbf25993c768b1e86fecbb350433a2fb229235b556334e9eb6ae868cf"


def stage_checkpoint(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "step-0098304.pt"
    if not checkpoint.exists():
        temporary = checkpoint.with_suffix(".partial")
        try:
            with closing(r2.client()) as client:
                client.download_file(r2.bucket(), O64_KEY, str(temporary))
            if sha256(temporary) != O64_SHA256:
                raise ValueError("O64 checkpoint SHA-256 mismatch")
            temporary.replace(checkpoint)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256(checkpoint) != O64_SHA256:
        raise ValueError("O64 checkpoint SHA-256 mismatch")
    # Fetch once before torchrun; ranks must not race on statistics downloads.
    saved = torch.load(checkpoint, weights_only=False, mmap=True, map_location="cpu")
    names = tuple(saved["cfg"]["source_names"])
    paths = source_stats_paths(names, None)
    hashes = {name: sha256(path) for name, path in zip(names, paths, strict=True)}
    if hashes != saved["provenance"]["source_statistics_sha256"]:
        raise ValueError("O64 feature statistics SHA-256 mismatch")
    (directory / "identity.json").write_text(
        json.dumps({"key": O64_KEY, "sha256": O64_SHA256, "source_statistics_sha256": hashes}, indent=2)
    )
    return checkpoint


def shards_per_gpu(matches: int) -> int:
    if type(matches) is not int or matches < 16 or matches % 16:
        raise ValueError("matches per GPU must be a positive multiple of 16")
    shards = min(16, matches // 16)
    if matches % shards:
        raise ValueError("matches must divide evenly across at most 16 shards per GPU")
    return shards


@dataclasses.dataclass(frozen=True, slots=True)
class Args:
    output: Path
    delivery: Path
    checkpoint: Path
    cpus: tuple[int, ...]
    gpus: int
    matches: int
    windows_per_step: int
    updates: int = 256
    actor_microbatches: int = 1
    measurement: bool = False
    compile_cache: Path | None = None
    publish_directory: Path | None = None
    publish_volume: str | None = None
    r2_prefix: str | None = None

    def __post_init__(self) -> None:
        shards = shards_per_gpu(self.matches)
        if type(self.gpus) is not int or not 2 <= self.gpus <= 8:
            raise ValueError("O64 torchrun requires 2..8 GPUs")
        if len(self.cpus) != shards * self.gpus or len(set(self.cpus)) != len(self.cpus):
            raise ValueError("O64 requires distinct shard CPUs for every rank")
        if any(
            type(value) is not int or value < 1
            for value in (self.windows_per_step, self.actor_microbatches, self.updates)
        ):
            raise ValueError("windows, microbatches and updates must be positive integers")
        if self.windows_per_step % self.actor_microbatches:
            raise ValueError("actor microbatches must divide windows per step")
        publication = (self.publish_directory, self.publish_volume, self.r2_prefix)
        if self.measurement:
            if any(value is not None for value in publication):
                raise ValueError("measurement does not publish")
        elif self.updates != 256 or any(value is None for value in publication):
            raise ValueError("O64 training requires 256 updates and complete publication settings")


def training_command(args: Args) -> list[str]:
    argv = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node",
        str(args.gpus),
        str(ROOT / "experiments/065_selfplay_ppo.py"),
        "train",
        "--checkpoint",
        str(args.checkpoint),
        "--sampler",
        str(ROOT / "experiments/o65/production-sampler.json"),
        "--data-dir",
        str(args.delivery / "inputs/msl/data"),
        "--output",
        str(args.output / "train"),
        "--cpus",
        *(str(cpu) for cpu in args.cpus),
        "--matches-per-shard",
        str(args.matches // shards_per_gpu(args.matches)),
        "--cadence",
        "production",
        "--rollout-frames",
        "512",
        "--groups",
        "1",
        "--updates",
        str(args.updates),
        "--training.windows-per-step",
        str(args.windows_per_step),
        "--training.actor-microbatches",
        str(args.actor_microbatches),
        "--training.decisions-per-window",
        "16",
        "--training.seed",
        "65",
        "--training.critic-warmup-updates",
        "0" if args.measurement else "32",
        "--training.head-start-frames",
        "0" if args.measurement else "1200",
        "--training.checkpoint-interval",
        "32",
        "--training.snapshot-interval",
        "64",
        "--training.reward-weights.damage",
        "0.1",
        "--training.reward-weights.stock",
        "120",
        "--training.reward-weights.game-win",
        "120",
    ]
    if args.compile_cache is not None:
        argv.extend(("--compile-cache", str(args.compile_cache)))
    if args.measurement:
        argv.append("--measurement")
    else:
        argv.extend(
            (
                "--wandb-project",
                "hal-o65",
                "publication:args",
                "--publication.source",
                str(args.output),
                "--publication.destination",
                str(args.publish_directory),
                "--publication.volume",
                str(args.publish_volume),
                "--publication.r2-prefix",
                str(args.r2_prefix),
                "--publication.compact",
            )
        )
    return argv


def evaluation_command(args: Args) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/eval_o65.py"),
        "--checkpoint",
        str(args.output / f"train/checkpoints/update-{args.updates:08d}.json"),
        "--anchors",
        str(args.checkpoint),
        "--data-dir",
        str(args.delivery / "inputs/msl/data"),
        "--protocol",
        str(ROOT / "experiments/o65/production-evaluation.json"),
        "--output",
        str(args.output / "held-out"),
        "--split",
        "held_out",
        "--cadence-grid",
        "18/2",
        "--lambda-grid",
        "1",
        "--simulators",
        "msl",
        "--device",
        "cuda",
    ]


def main(args: Args) -> None:
    if sha256(args.checkpoint) != O64_SHA256:
        raise ValueError("O64 checkpoint SHA-256 mismatch")
    args.output.mkdir(parents=True, exist_ok=not args.measurement)
    identity = {
        "format": "hal.o65.production-identity.v1",
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "checkpoint_sha256": O64_SHA256,
        "settings": json.loads(json.dumps(dataclasses.asdict(args), default=str)),
        "inputs": {
            name: sha256(ROOT / "experiments/o65" / name)
            for name in (
                "production-sampler.json",
                "production-sampler-source.json",
                "production-evaluation.json",
                "production-256.json",
                "production-protocol.md",
            )
        },
    }
    path = args.output / "production-identity.json"
    if not path.exists() and any(args.output.iterdir()):
        raise ValueError("existing O64 state has no source/input identity")
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("O64 source, settings or input identity differs on resume")
    path.write_text(json.dumps(identity, indent=2))
    command = training_command(args)
    manifests = sorted((args.output / "train/checkpoints").glob("update-*.json"))
    previous = manifests[-1] if manifests else None
    update = read_training_manifest(previous)["update"] if previous is not None else 0
    if previous is not None:
        # The union argument must remain last for Tyro's nested CLI parser.
        position = command.index("publication:args")
        command[position:position] = ["--resume", str(previous)]
        metrics = args.output / "train/metrics.jsonl"
        if metrics.is_file():
            retained = [line for line in metrics.read_text().splitlines() if json.loads(line)["global_step"] <= update]
            metrics.write_text("".join(line + "\n" for line in retained))
    environment = dict(os.environ)
    if not args.measurement:
        if not environment.get("WANDB_API_KEY"):
            raise ValueError("O64 training requires WANDB_API_KEY")
        environment.update(WANDB_RUN_ID=args.output.parent.name, WANDB_RESUME="allow")
    if update < args.updates:
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    if args.measurement:
        return
    assert args.publish_directory is not None and args.publish_volume is not None and args.r2_prefix is not None
    publication = self_play_publication.Args(args.output, args.publish_directory, args.publish_volume, args.r2_prefix)
    if not (args.output / "held-out/results.json").is_file():
        with (args.output / "held-out-consumed.json").open("x") as claim:
            json.dump({"update": args.updates, "cadence": [18, 2], "u": 0.95, "lambda": 1, "games": 384}, claim)
        self_play_publication.publish(publication)
        subprocess.run(evaluation_command(args), cwd=ROOT, env=environment, check=True)
    self_play_publication.publish(publication)


if __name__ == "__main__":
    main(tyro.cli(Args))
