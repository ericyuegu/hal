"""Publish self-play actors for evaluation and load the recorded O63 proxy."""

import hashlib
import json
import math
import os
import re
import uuid
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import fields
from pathlib import Path
from typing import TypedDict
from typing import cast

import torch

from hal.data.feature_stats import FeatureStats
from hal.data.feature_stats import load_consolidated_mixture_stats
from hal.data.policy_world_schema import POLICY_WORLD_SCHEMA_VERSION
from hal.data.schema import SCHEMA_VERSION
from hal.inference.action_sequence_artifact import source_stats_paths
from hal.models.action_sequence import ActionSequenceConfig
from hal.models.conditioning import ConditioningRanges
from hal.models.self_play import DecoderPrecision
from hal.models.self_play import SelfPlayTransformer
from hal.models.self_play import validate_decoder_precision
from hal.representation.player_identity import vocabulary_from_checkpoint_buffer

PROXY_SHA256 = "65e7e43e5cbb18a365a005922ec1cfd026ad807c045eae13b331f487b4ec82e6"
EVALUATION_FORMAT = "hal.o65.evaluation.v3"
TRAINING_FORMAT = "hal.o65.training.v8"
TRAINING_VERSION = 8


class CheckpointShard(TypedDict):
    name: str
    sha256: str
    role: str


class TrainingManifest(TypedDict):
    format: str
    version: int
    experiment: str
    update: int
    configuration_sha256: str
    environment: dict[str, str]
    provenance: dict[str, str]
    shards: list[CheckpointShard]


def read_training_manifest(manifest_path: Path) -> TrainingManifest:
    """Validate the shared resume/evaluation envelope before loading any shard."""
    manifest = json.loads(manifest_path.read_text())
    fields = {
        "format",
        "version",
        "experiment",
        "update",
        "configuration_sha256",
        "environment",
        "provenance",
        "shards",
    }
    if not isinstance(manifest, dict) or set(manifest) != fields:
        raise ValueError("checkpoint manifest schema changed")
    if (manifest["format"], manifest["version"], manifest["experiment"]) != (
        TRAINING_FORMAT,
        TRAINING_VERSION,
        "065_selfplay_ppo",
    ):
        raise ValueError("unsupported checkpoint identity")
    if type(manifest["version"]) is not int or type(manifest["update"]) is not int or manifest["update"] < 0:
        raise ValueError("invalid checkpoint update")
    provenance = manifest["provenance"]
    expected_provenance = {
        "git_sha",
        "checkpoint_sha256",
        "data_manifest_sha256",
        "simulator_sha256",
        "adapter_sha256",
    }
    if not isinstance(provenance, dict) or set(provenance) != expected_provenance:
        raise ValueError("invalid checkpoint provenance fields")
    for name, value in provenance.items():
        if (
            not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{40}" if name == "git_sha" else r"[0-9a-f]{64}", value) is None
        ):
            raise ValueError("invalid checkpoint provenance hash")
    if (
        not isinstance(manifest["configuration_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", manifest["configuration_sha256"]) is None
    ):
        raise ValueError("invalid checkpoint configuration hash")
    environment = manifest["environment"]
    if (
        not isinstance(environment, dict)
        or set(environment)
        != {"python", "torch", "numpy", "cuda", "device", "world_size", "tf32_matmul", "tf32_cudnn"}
        or any(not isinstance(value, str) or not value for value in environment.values())
    ):
        raise ValueError("invalid checkpoint environment")
    if not isinstance(manifest["shards"], list) or not manifest["shards"]:
        raise ValueError("checkpoint shards must be a nonempty list")
    roles: set[str] = set()
    for shard in manifest["shards"]:
        if not isinstance(shard, dict) or set(shard) != {"name", "sha256", "role"}:
            raise ValueError("invalid checkpoint shard entry")
        if (
            not all(isinstance(shard[key], str) for key in shard)
            or Path(shard["name"]).name != shard["name"]
            or shard["role"] in roles
            or re.fullmatch(r"[0-9a-f]{64}", shard["sha256"]) is None
            or not shard["name"].startswith(shard["sha256"] + ".")
        ):
            raise ValueError("invalid checkpoint shard identity")
        role = shard["role"]
        suffix = {"training": "rank0.pt", "actor": "actor.pt"}.get(role)
        if suffix is None and re.fullmatch(r"simulator_[0-9]+_[0-9]+", role):
            suffix = "sim.zlib"
        if suffix is None or shard["name"] != f"{shard['sha256']}.{suffix}":
            raise ValueError("invalid checkpoint shard role or filename")
        roles.add(shard["role"])
    if "training" not in roles:
        raise ValueError("checkpoint has no training state")
    return cast(TrainingManifest, manifest)


def evaluation_actor_path(manifest_path: Path) -> Path:
    """Resolve the actor in a published update-boundary training checkpoint."""
    manifest = read_training_manifest(manifest_path)
    actor: Path | None = None
    for shard in manifest["shards"]:
        if shard["role"] == "actor":
            actor = manifest_path.parent / shard["name"]
            if sha256(actor) != shard["sha256"]:
                raise ValueError("checkpoint actor hash mismatch")
    if actor is None:
        raise ValueError("checkpoint has no published evaluation actor")
    raw = torch.load(actor, map_location="cpu", weights_only=False)
    provenance = raw.get("provenance") if isinstance(raw, dict) else None
    expected = manifest["provenance"]
    if not isinstance(provenance, dict) or (
        provenance.get("update"),
        provenance.get("git_sha"),
        provenance.get("reference_sha256"),
        provenance.get("data_manifests"),
    ) != (
        manifest["update"],
        expected["git_sha"],
        expected["checkpoint_sha256"],
        {"training": expected["data_manifest_sha256"]},
    ):
        raise ValueError("checkpoint actor provenance mismatch")
    return actor


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


@dataclass(frozen=True, slots=True)
class EvaluationCheckpoint:
    model: SelfPlayTransformer
    statistics: dict[str, FeatureStats]
    checkpoint_sha256: str
    policy_sha256: str
    provenance: dict[str, object]


def save_evaluation_checkpoint(
    path: Path,
    model: SelfPlayTransformer,
    statistics: dict[str, FeatureStats],
    *,
    provenance: dict[str, object],
) -> None:
    """Publish an immutable actor snapshot, independent of training-resume shards.

    At an update boundary, record the reference hash, Git SHA, update, and data
    manifest hashes before scheduling evaluation.
    """
    metadata = evaluation_checkpoint_metadata(model, statistics, provenance=provenance)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("xb") as target:
            torch.save(
                {
                    **metadata,
                    "model": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
                },
                target,
            )
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def evaluation_checkpoint_metadata(
    model: SelfPlayTransformer,
    statistics: dict[str, FeatureStats],
    *,
    provenance: dict[str, object],
) -> dict[str, object]:
    """Describe the evaluation envelope without copying the model tensors."""
    _validate_provenance(provenance)
    return {
        "format": EVALUATION_FORMAT,
        "architecture": asdict(model.cfg),
        "ranges": asdict(model.ranges),
        "decoder_precision": model.decoder_precision,
        "max_opponent_queue": model.critic_residual.max_queue,
        "critic_hidden_dim": model.critic_residual.residual.up.out_features // 2,
        "statistics": {name: asdict(value) for name, value in statistics.items()},
        "provenance": provenance,
    }


def _validate_provenance(value: dict[str, object]) -> None:
    if not {"reference_sha256", "git_sha", "update", "data_manifests"}.issubset(value):
        raise ValueError("evaluation provenance needs reference, source, update and data manifests")
    for name, length in (("reference_sha256", 64), ("git_sha", 40)):
        digest = value[name]
        if not isinstance(digest, str) or len(digest) != length or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"invalid {name}")
    if type(value["update"]) is not int or value["update"] < 0:
        raise ValueError("update must be a nonnegative integer")
    manifests = value["data_manifests"]
    if not isinstance(manifests, dict) or not manifests:
        raise ValueError("data manifests must be a nonempty mapping")
    if any(
        not isinstance(k, str) or not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
        for k, v in manifests.items()
    ):
        raise ValueError("invalid data manifest hash")
    json.dumps(value, allow_nan=False)


def load_evaluation_checkpoint(
    path: Path, *, device: torch.device, decoder_precision: DecoderPrecision | None = None
) -> EvaluationCheckpoint:
    """Read saved O65 precision, or select precision when converting BC weights."""
    if decoder_precision is not None:
        validate_decoder_precision(decoder_precision)
    if path.suffix == ".json":
        path = evaluation_actor_path(path)
    digest = sha256(path)
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError("evaluation checkpoint must be a mapping")
    is_bc = isinstance(raw.get("cfg"), dict) and raw["cfg"].get("experiment_id") == "063_percentile_conditioning_v1"
    if is_bc:
        precision = "bf16" if decoder_precision is None else decoder_precision
        saved = raw["cfg"]
        if saved["checkpoint_format_version"] != 5:
            raise ValueError("unsupported O63 checkpoint version")
        if (
            saved["mds_schema_version"] != SCHEMA_VERSION
            or saved["policy_world_schema_version"] != POLICY_WORLD_SCHEMA_VERSION
            or saved["return_conditioning"] is not True
        ):
            raise ValueError("O63 representation or return conditioning differs")
        cfg = ActionSequenceConfig(
            d_model=256,
            n_layers=12,
            n_heads=4,
            temporal_d_model=256,
            temporal_layers=6,
            temporal_heads=4,
            temporal_ff_dim=1024,
            group_head_dim=256,
            value_hidden_dim=128,
        )
        if saved["architecture"] != {name: getattr(cfg, name) for name in saved["architecture"]}:
            raise ValueError("O63 proxy architecture differs")
        ranges = ConditioningRanges((math.log(0.01), math.log(100)), (8, 10), (2, 4))
        paths = source_stats_paths(tuple(saved["source_names"]), None)
        if len(paths) != 1:
            raise ValueError("O63 development requires a single statistics source")
        statistics_digest = sha256(paths[0])
        stats = load_consolidated_mixture_stats(paths, (1.0,), expected_mds_schema_version=SCHEMA_VERSION)
        provenance = {
            "reference_sha256": digest,
            "git_sha": raw["provenance"]["git_sha"],
            "update": raw["step"] + 1,
            "data_manifests": raw["provenance"]["source_manifest_sha256"],
            "statistics_sha256": statistics_digest,
        }
        queue, critic = 14, 128
    else:
        if (
            set(raw)
            != {
                "format",
                "architecture",
                "ranges",
                "decoder_precision",
                "max_opponent_queue",
                "critic_hidden_dim",
                "model",
                "statistics",
                "provenance",
            }
            or raw["format"] != EVALUATION_FORMAT
        ):
            raise ValueError("unsupported checkpoint; export an O65 evaluation snapshot at an update boundary")
        architecture = raw["architecture"]
        if not isinstance(architecture, dict) or set(architecture) != {
            field.name for field in fields(ActionSequenceConfig)
        }:
            raise ValueError("evaluation architecture fields changed")
        cfg = ActionSequenceConfig(**architecture)
        range_values = raw["ranges"]
        if not isinstance(range_values, dict) or set(range_values) != {"log_lambda", "delay", "replan"}:
            raise ValueError("evaluation conditioning fields changed")
        ranges = ConditioningRanges(range_values["log_lambda"], range_values["delay"], range_values["replan"])
        precision = validate_decoder_precision(raw["decoder_precision"])
        if decoder_precision is not None and decoder_precision != precision:
            raise ValueError("decoder precision differs from the evaluation checkpoint")
        queue, critic = raw["max_opponent_queue"], raw["critic_hidden_dim"]
        stats = {name: FeatureStats(**value) for name, value in raw["statistics"].items()}
        provenance = raw["provenance"]
        _validate_provenance(provenance)
    for value in stats.values():
        if not all(math.isfinite(x) for x in asdict(value).values()) or value.std < 0 or value.min > value.max:
            raise ValueError("invalid evaluation feature statistics")
    if not stats:
        raise ValueError("evaluation statistics must be nonempty")
    state = raw["model"]
    vocabulary = vocabulary_from_checkpoint_buffer(state["player_code_bytes"].numpy())
    model = SelfPlayTransformer(
        cfg, ranges, vocabulary, max_opponent_queue=queue, critic_hidden_dim=critic, decoder_precision=precision
    )
    expected = model.state_dict()
    for name, tensor in state.items():
        if name not in expected or not isinstance(tensor, torch.Tensor) or tensor.dtype != expected[name].dtype:
            raise ValueError("evaluation tensor dtype or identity changed")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError("evaluation tensors must be finite")
    if is_bc:
        model.load_bc_state_dict(state)
    else:
        model.load_state_dict(state, strict=True)
    policy_digest = hashlib.sha256(
        json.dumps(
            {
                "architecture": {name: value for name, value in asdict(cfg).items() if name != "value_hidden_dim"},
                "ranges": asdict(ranges),
                "decoder_precision": precision,
                "statistics": {name: asdict(value) for name, value in stats.items()},
            },
            sort_keys=True,
            allow_nan=False,
        ).encode()
    )
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith(("value_head.", "critic_residual.", "percentile_head.")):
            continue
        policy_digest.update(name.encode())
        policy_digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return EvaluationCheckpoint(
        model.eval().requires_grad_(False).to(device), stats, digest, policy_digest.hexdigest(), provenance
    )
