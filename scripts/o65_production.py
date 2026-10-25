"""Construct O64 torchrun training, measurement and final evaluation commands."""

import dataclasses
import json
import os
import subprocess
import sys
import uuid
from contextlib import closing
from pathlib import Path
from typing import TypedDict

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


# Modal caps a function at 64 CPUs, so eight ranks get eight simulator shards each.
SHARDS_PER_GPU = 8


def shards_per_gpu(matches: int) -> int:
    if type(matches) is not int or matches < 16 or matches % 16:
        raise ValueError("matches per GPU must be a positive multiple of 16")
    shards = min(SHARDS_PER_GPU, matches // 16)
    if matches % shards:
        raise ValueError(f"matches must divide evenly across at most {SHARDS_PER_GPU} shards per GPU")
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
    require_compile_cache: bool = False
    publish_directory: Path | None = None
    publish_volume: str | None = None
    r2_prefix: str | None = None
    resume_source: str | None = None
    resume_manifest_sha256: str | None = None
    verify_resume: bool = False

    def __post_init__(self) -> None:
        if self.require_compile_cache and self.compile_cache is None:
            raise ValueError("required compiler artifacts need a compile cache directory")
        if self.resume_source is not None and (
            self.measurement
            or len(self.resume_source) != 40
            or any(c not in "0123456789abcdef" for c in self.resume_source)
        ):
            raise ValueError("resume source requires training and a full Git SHA")
        if self.resume_manifest_sha256 is not None and (
            self.measurement
            or len(self.resume_manifest_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.resume_manifest_sha256)
        ):
            raise ValueError("resume manifest requires training and a full SHA-256")
        if self.resume_source is not None and self.resume_manifest_sha256 is None:
            raise ValueError("source-changing resume requires a checkpoint manifest SHA-256")
        shards = shards_per_gpu(self.matches)
        if type(self.gpus) is not int or not 1 <= self.gpus <= 8 or (self.gpus == 1 and not self.measurement):
            raise ValueError("training requires 2..8 GPUs; single-GPU measurement is supported")
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
        elif any(value is None for value in publication):
            raise ValueError("O64 training requires complete publication settings")


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
        "4",
        "--updates",
        str(args.updates),
        "--training.windows-per-step",
        str(args.windows_per_step),
        "--training.actor-microbatches",
        str(args.actor_microbatches),
        "--training.decisions-per-window",
        "all",
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
        directory = args.compile_cache / "rank0" if args.gpus == 1 else args.compile_cache
        argv.extend(("--compile-cache", str(directory)))
    if args.require_compile_cache:
        argv.append("--require-compile-cache")
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


@dataclasses.dataclass(frozen=True, slots=True)
class EvaluationArgs:
    output: Path
    delivery: Path
    checkpoint: Path
    checkpoint_sha256: str
    reference_checkpoint: Path
    publish_directory: Path
    publish_volume: str
    r2_prefix: str


def evaluation_command(args: EvaluationArgs) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/eval_o65.py"),
        "--checkpoint",
        str(args.checkpoint),
        "--anchors",
        str(args.reference_checkpoint),
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


def resume_settings(args: Args) -> dict[str, object]:
    # A replacement container may expose different virtual CPU IDs; they do not change the run.
    settings = json.loads(json.dumps(dataclasses.asdict(args), default=str))
    del settings["cpus"]
    del settings["resume_source"]
    del settings["resume_manifest_sha256"]
    del settings["require_compile_cache"]
    # The learning-rate schedule uses optimizer steps, not the requested endpoint.
    del settings["updates"]
    del settings["verify_resume"]
    # Compiler storage changes startup cost, not the training configuration.
    settings["compile_cache"] = None
    return settings


class ProductionIdentity(TypedDict):
    format: str
    git_sha: str
    checkpoint_sha256: str
    settings: dict[str, object]
    inputs: dict[str, str]


def validate_resume_identity(previous: ProductionIdentity, current: ProductionIdentity, source: str | None) -> None:
    if not isinstance(previous, dict) or set(previous) != set(ProductionIdentity.__annotations__):
        raise ValueError("invalid production identity schema")
    expected = current | {"git_sha": source} if source is not None else current
    normalized = dict(previous)
    if normalized.get("format") == "hal.o65.production-identity.v1":
        settings = previous["settings"]
        if not isinstance(settings, dict):
            raise ValueError("invalid legacy production settings")
        updates = settings.get("updates")
        if type(updates) is not int or updates < 1:
            raise ValueError("invalid legacy production endpoint")
        normalized["settings"] = {key: value for key, value in settings.items() if key != "updates"}
        normalized["format"] = "hal.o65.production-identity.v2"
    if normalized not in (current, expected):
        raise ValueError("O64 source, settings or input identity differs on resume")


def evaluate(args: EvaluationArgs) -> None:
    if sha256(args.reference_checkpoint) != O64_SHA256:
        raise ValueError("O64 reference checkpoint SHA-256 mismatch")
    if sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("final checkpoint SHA-256 mismatch")
    if args.checkpoint.parent != args.output / "train/checkpoints":
        raise ValueError("final checkpoint must belong to the published training run")
    manifest = read_training_manifest(args.checkpoint)
    claim = {
        "format": "hal.o65.final-evaluation.v2",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "reference_sha256": O64_SHA256,
        "protocol_sha256": sha256(ROOT / "experiments/o65/production-evaluation.json"),
        "update": manifest["update"],
        "cadence": [18, 2],
        "u": 0.95,
        "lambda": 1,
        "games": 384,
    }
    publication = self_play_publication.Args(args.output, args.publish_directory, args.publish_volume, args.r2_prefix)
    if (args.output / "held-out").exists():
        raise ValueError("held-out results already exist; preserve them instead of replaying evaluation")
    with (args.output / "held-out-consumed.json").open("x") as file:
        json.dump(claim, file, indent=2)
    self_play_publication.publish(publication)
    print("final_evaluation_started", flush=True)
    try:
        subprocess.run(evaluation_command(args), cwd=ROOT, check=True)
    finally:
        self_play_publication.publish(publication)


@dataclasses.dataclass(frozen=True, slots=True)
class RunAttempt:
    run_id: str
    group: str
    source: str
    previous_attempt_run_id: str | None
    checkpoint: str | None
    checkpoint_sha256: str | None
    checkpoint_source: str | None
    resume_update: int
    prior_metrics_sha256: str | None


def begin_attempt(output: Path, source: str, checkpoint: Path | None) -> RunAttempt:
    """Preserve rolled-back history and assign a distinct W&B trajectory to this attempt."""
    group = output.parent.name
    previous = output / "run-attempt.json"
    previous_attempt_run_id = group if checkpoint is not None else None
    if previous.exists():
        recorded = json.loads(previous.read_text())
        if (
            not isinstance(recorded, dict)
            or set(recorded) != {"format", *RunAttempt.__dataclass_fields__}
            or recorded["format"] != "hal.o65.run-attempt.v1"
            or recorded["group"] != group
            or not isinstance(recorded["run_id"], str)
            or not recorded["run_id"]
        ):
            raise ValueError("production attempt identity mismatch")
        previous_attempt_run_id = recorded["run_id"] if checkpoint is not None else None
    manifest = read_training_manifest(checkpoint) if checkpoint is not None else None
    update = manifest["update"] if manifest is not None else 0
    run_id = f"{group}-{uuid.uuid4().hex[:12]}"
    directory = output / "attempts" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    metrics = output / "train/metrics.jsonl"
    prior_metrics_sha256 = None
    if metrics.exists():
        payload = metrics.read_bytes()
        rows = [json.loads(line) for line in payload.splitlines()]
        if any(type(row.get("global_step")) not in (int, float) for row in rows):
            raise ValueError("production metrics need a global_step")
        archived = directory / "metrics-before-resume.jsonl"
        archived.write_bytes(payload)
        prior_metrics_sha256 = sha256(archived)
        retained = [
            line
            for line, row in zip(payload.splitlines(keepends=True), rows, strict=True)
            if row["global_step"] <= update
        ]
        metrics.write_bytes(b"".join(retained))
    attempt = RunAttempt(
        run_id,
        group,
        source,
        previous_attempt_run_id,
        str(checkpoint) if checkpoint is not None else None,
        sha256(checkpoint) if checkpoint is not None else None,
        manifest["provenance"]["git_sha"] if manifest is not None else None,
        update,
        prior_metrics_sha256,
    )
    encoded = json.dumps({"format": "hal.o65.run-attempt.v1", **dataclasses.asdict(attempt)}, indent=2)
    (directory / "lineage.json").write_text(encoded)
    previous.write_text(encoded)
    return attempt


def main(args: Args) -> None:
    if sha256(args.checkpoint) != O64_SHA256:
        raise ValueError("O64 checkpoint SHA-256 mismatch")
    args.output.mkdir(parents=True, exist_ok=not args.measurement)
    identity: ProductionIdentity = {
        "format": "hal.o65.production-identity.v2",
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "checkpoint_sha256": O64_SHA256,
        "settings": resume_settings(args),
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
    prior_identity = json.loads(path.read_text()) if path.exists() else identity
    validate_resume_identity(prior_identity, identity, args.resume_source)
    command = training_command(args)
    manifests = sorted((args.output / "train/checkpoints").glob("update-*.json"))
    previous = manifests[-1] if manifests else None
    if args.resume_manifest_sha256 is not None:
        if previous is None:
            raise ValueError("resume manifest requires an existing checkpoint")
        if sha256(previous) != args.resume_manifest_sha256:
            raise ValueError("checkpoint manifest differs from the requested SHA-256")
    update = read_training_manifest(previous)["update"] if previous is not None else 0
    if not args.measurement and args.updates <= update:
        raise ValueError("training endpoint must exceed the restored update")
    if args.verify_resume and (args.measurement or previous is None):
        raise ValueError("resume verification requires an existing training checkpoint")
    if not args.measurement and (args.output / "held-out-consumed.json").exists():
        raise ValueError("cannot extend a run after its final held-out evaluation was consumed")
    if previous is not None:
        # The union argument must remain last for Tyro's nested CLI parser.
        position = command.index("publication:args")
        command[position:position] = ["--resume", str(previous)]
        source = read_training_manifest(previous)["provenance"]["git_sha"]
        if source != identity["git_sha"]:
            if source != args.resume_source:
                raise ValueError("checkpoint source differs from the explicit resume source")
            position = command.index("publication:args")
            command[position:position] = ["--resume-source", source]
    elif args.resume_source is not None:
        raise ValueError("resume source requires an existing checkpoint")
    if prior_identity != identity:
        assert previous is not None
        transition = {
            "format": "hal.o65.source-transition.v1",
            "previous": prior_identity,
            "current": identity,
            "checkpoint": str(previous),
            "checkpoint_sha256": sha256(previous),
            "update": update,
        }
        with (args.output / "source-transitions.jsonl").open("a") as history:
            history.write(json.dumps(transition, sort_keys=True) + "\n")
    path.write_text(json.dumps(identity, indent=2))
    (args.output / "execution.json").write_text(
        json.dumps({"resume_update": update, "updates": args.updates}, indent=2)
    )
    environment = dict(os.environ)
    if not args.measurement:
        if not environment.get("WANDB_API_KEY"):
            raise ValueError("O64 training requires WANDB_API_KEY")
        if update < args.updates:
            attempt = begin_attempt(args.output, identity["git_sha"], previous)
            environment.update(
                WANDB_RUN_ID=attempt.run_id,
                WANDB_RUN_GROUP=attempt.group,
                WANDB_NAME=f"{attempt.group} from {attempt.resume_update} {attempt.source[:8]}",
                WANDB_RESUME="never",
                WANDB_NOTES=json.dumps(dataclasses.asdict(attempt), sort_keys=True),
            )
            if args.verify_resume:
                directory = args.output / "attempts" / attempt.run_id / "resume-check"
                position = command.index("publication:args")
                command[position:position] = ["--resume-check", str(directory)]
                preparation = command[: command.index("publication:args")]
                position = preparation.index("--wandb-project")
                del preparation[position : position + 2]
                preparation.append("--prepare-resume-check")
                subprocess.run(preparation, cwd=ROOT, env=environment, check=True)
    if update < args.updates:
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    if args.measurement:
        return
    assert args.publish_directory is not None and args.publish_volume is not None and args.r2_prefix is not None
    publication = self_play_publication.Args(args.output, args.publish_directory, args.publish_volume, args.r2_prefix)
    self_play_publication.publish(publication)


if __name__ == "__main__":
    main(tyro.cli(Args))
