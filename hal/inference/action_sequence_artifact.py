"""Validate current 059 artifacts and construct the canonical model."""

import hashlib
import json
import math
import re
import tempfile
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Final
from typing import Literal
from typing import cast

import torch
from torch import Tensor

from hal import streams
from hal.data.feature_stats import FeatureStats
from hal.data.feature_stats import load_consolidated_mixture_stats
from hal.inference.api import PolicySpec
from hal.inference.bundle import BundleDescription
from hal.inference.bundle import PolicyBundleManifest
from hal.inference.bundle import extract_policy_bundle
from hal.inference.bundle import write_policy_bundle
from hal.inference.checkpoints import resolve_checkpoint
from hal.models.action_sequence import ActionSequenceConfig
from hal.models.action_sequence import ActionSequenceTransformer
from hal.models.attention import Rotary
from hal.models.controller_codec import DiscreteControllerCodec
from hal.representation.features import BASE_ITEMS_PROJECTION
from hal.representation.features import FLOAT_FEATURES
from hal.representation.features import ITEM_FLOATS
from hal.representation.player_identity import PlayerVocabulary
from hal.representation.player_identity import decode_player_codes
from hal.representation.player_identity import encode_player_codes
from hal.training.checkpoints import checkpoint_resume_lineage
from hal.training.checkpoints import checkpoint_sha256
from hal.training.returns import ReturnCalibration
from hal.wire import ACTION_CHANNELS

if TYPE_CHECKING:
    from hal.inference.action_sequence_policy import ActionSequencePolicy

# These identifiers are persisted contracts, independent of Python module names.
ACTION_SEQUENCE_BACKEND: Final[str] = "o59-history-decoder"
ACTION_SEQUENCE_BACKEND_VERSION: Final[int] = 2
_CONFIG_MEMBER: Final[str] = "backend.json"
_CHECKPOINT_MEMBER: Final[str] = "checkpoint.pt"
_STATS_MEMBER: Final[str] = "stats.json"
_ACTION_FIELDS: Final[frozenset[str]] = frozenset(f"ego_{name}" for name in ACTION_CHANNELS)
_MODEL_FIELDS: Final[frozenset[str]] = BASE_ITEMS_PROJECTION.columns - _ACTION_FIELDS
_STATISTIC_FIELDS: Final[frozenset[str]] = frozenset(
    (*FLOAT_FEATURES, *(f"nana_{name}" for name in FLOAT_FEATURES), *(f"item_{name}" for name in ITEM_FLOATS))
)


def _canonical_field(name: str) -> str:
    if name.startswith("ego_"):
        return f"p1_{name[4:]}"
    if name.startswith("opp_"):
        return f"p2_{name[4:]}"
    return name


REQUIRED_OBSERVATION_FIELDS: Final[tuple[str, ...]] = tuple(sorted(map(_canonical_field, _MODEL_FIELDS)))


@dataclass(frozen=True, slots=True)
class CheckpointContract:
    model: ActionSequenceConfig
    source_names: tuple[str, ...]
    mds_schema_version: int
    return_dropout: float
    player_sidecar_sha256: str


@dataclass(frozen=True, slots=True)
class ActionSequenceArtifact:
    model_config: ActionSequenceConfig
    model_tensors: dict[str, Tensor]
    statistics: dict[str, FeatureStats]
    vocabulary: PlayerVocabulary
    checkpoint_sha256: str
    return_p90: float
    spec: PolicySpec
    capability_version: int


_CHECKPOINT_CONFIG_FIELDS: Final[frozenset[str]] = frozenset(
    (
        "adam_beta1",
        "adam_beta2",
        "adam_eps",
        "adam_lr",
        "adam_weight_decay",
        "allow_tf32",
        "amp_dtype",
        "architecture",
        "awr_calibration",
        "batch_size",
        "cache_metrics_interval_s",
        "checkpoint_format_version",
        "ckpt_every",
        "compile_temporal",
        "compile_trunk",
        "compiled_inference_bucket",
        "decay_duration",
        "decay_start_update",
        "delay_frames",
        "depth_alpha",
        "download_retry",
        "eval_every",
        "eval_max_frames",
        "eval_max_parallel",
        "eval_n_matchups",
        "eval_seed",
        "experiment_id",
        "final_eval_n_matchups",
        "grad_clip",
        "hidden_std_multiplier",
        "identity_dropout",
        "inference_mode",
        "lr_floor_ratio",
        "max_steps",
        "mds_schema_version",
        "muon_lr",
        "muon_lr_multiplier",
        "muon_weight_decay",
        "num_workers",
        "optimizer",
        "parent_checkpoint_name",
        "parent_checkpoint_sha256",
        "parent_run_name",
        "parent_wandb_id",
        "player_sidecar_local",
        "player_sidecar_sha256",
        "player_vocab_sha256",
        "player_vocab_size",
        "policy_world_schema_version",
        "prediction_frames",
        "process_metrics_interval_s",
        "push_to_r2",
        "readout_init",
        "replan_interval_frames",
        "return_conditioning",
        "return_dropout",
        "seed",
        "source_names",
        "stable_updates",
        "system_metrics_every",
        "system_metrics_interval_s",
        "target_positions",
        "train_compile_mode",
        "val_batch_size",
        "val_every",
        "val_n_samples",
        "val_split",
        "wandb_log_code",
        "warmup_steps",
    )
)
_ARCHITECTURE_FIELDS: Final[tuple[str, ...]] = (
    "d_model",
    "n_layers",
    "n_heads",
    "attn_window",
    "L_ctx",
    "sample_chunk_length",
    "head_offsets",
    "temporal_d_model",
    "temporal_layers",
    "temporal_heads",
    "temporal_ff_dim",
    "group_head_dim",
    "return_embed_dim",
    "action_embed_dim",
    "offset_embed_dim",
    "action_vocab",
    "action_state_embed_dim",
    "char_vocab",
    "char_dim",
    "stage_vocab",
    "stage_dim",
    "item_type_dim",
    "item_state_dim",
    "item_hidden_dim",
    "item_dim",
    "value_hidden_dim",
)


def checkpoint_contract(values: object) -> CheckpointContract:
    if not isinstance(values, dict) or set(values) != _CHECKPOINT_CONFIG_FIELDS:
        raise ValueError("059 checkpoint config fields changed")
    values = cast(dict[str, object], values)
    if (values["experiment_id"], values["checkpoint_format_version"]) != ("059_muon_history_decoder_v5", 4):
        raise ValueError("059 checkpoint identity changed")
    defaults = ActionSequenceConfig()
    architecture = {name: getattr(defaults, name) for name in _ARCHITECTURE_FIELDS}
    if values["architecture"] != architecture:
        raise ValueError("059 checkpoint architecture is unsupported")
    if values["awr_calibration"] != {
        "beta": 150.0,
        "weight_max": 10.0,
        "gamma": 0.99855,
        "stock_value": 120.0,
        "damage_shaping": 1.0,
        "win_reward": 50.0,
        "value_loss_weight": 1.0,
        "auxiliary_loss_weight": 1.0,
    }:
        raise ValueError("059 checkpoint AWR calibration changed")
    if (values["prediction_frames"], values["delay_frames"], values["replan_interval_frames"]) != (4, 2, 2):
        raise ValueError("059 checkpoint official evaluation timing changed")
    sources = tuple(source.name for source in streams.POLICY_WORLD_V8_SOURCES)
    if values["source_names"] != sources:
        raise ValueError("059 checkpoint source list changed")
    if values["mds_schema_version"] != 7 or values["return_conditioning"] is not True:
        raise ValueError("059 checkpoint data or return conditioning changed")
    for name in ("batch_size", "target_positions", "player_vocab_size"):
        value = values[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"059 checkpoint {name} must be a positive integer")
    for name in ("depth_alpha", "hidden_std_multiplier", "return_dropout"):
        value = values[name]
        if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"059 checkpoint {name} must be finite")
    updates, remainder = divmod(
        cast(int, values["target_positions"]), cast(int, values["batch_size"]) * (defaults.L_ctx // 2)
    )
    if remainder or values["max_steps"] != updates or values["warmup_steps"] != 4096:
        raise ValueError("059 checkpoint schedule changed")
    if (
        not isinstance(values["player_vocab_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", values["player_vocab_sha256"]) is None
    ):
        raise ValueError("059 checkpoint vocabulary identity is invalid")
    model = replace(
        defaults,
        player_vocab_size=cast(int, values["player_vocab_size"]),
        player_vocab_sha256=values["player_vocab_sha256"],
        depth_alpha=float(cast(float, values["depth_alpha"])),
        hidden_std_multiplier=float(cast(float, values["hidden_std_multiplier"])),
    )
    sidecar_hash = values["player_sidecar_sha256"]
    if not isinstance(sidecar_hash, str) or re.fullmatch(r"[0-9a-f]{64}", sidecar_hash) is None:
        raise ValueError("059 checkpoint sidecar identity is invalid")
    return CheckpointContract(model, sources, 7, float(cast(float, values["return_dropout"])), sidecar_hash)


def validate_checkpoint_provenance(record: object, contract: CheckpointContract) -> None:
    if not isinstance(record, dict):
        raise ValueError("059 checkpoint provenance is missing")
    provenance = cast(dict[str, object], record)
    required = {
        "schema_version": 1,
        "experiment_id": "059_muon_history_decoder_v5",
        "source_selection_sha256": "2593361352b92e705be3fbeae1b4e9bb1a3c9f1787cd713014a7a95b7df62477",
        "source_manifest_sha256": streams.POLICY_WORLD_V8_TRAIN_MANIFEST_SHA256,
        "player_sidecar_sha256": contract.player_sidecar_sha256,
        "player_vocab_sha256": contract.model.player_vocab_sha256,
    }
    changed = [name for name, expected in required.items() if provenance.get(name) != expected]
    if changed:
        raise ValueError(f"059 checkpoint provenance changed: {changed}")
    source = provenance.get("git_sha")
    if not isinstance(source, str) or re.fullmatch(r"[0-9a-f]{40}", source) is None:
        raise ValueError("059 checkpoint source identity is invalid")


def read_checkpoint(path: Path) -> tuple[dict[str, object], CheckpointContract, tuple[str, ...], float]:
    raw = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError("O59 checkpoint root must be an object")
    cfg = checkpoint_contract(raw.get("cfg"))
    validate_checkpoint_provenance(raw.get("provenance"), cfg)
    lineage = checkpoint_resume_lineage(raw.get("resume_lineage"))
    if lineage and lineage[-1].new_source_sha != raw["provenance"]["git_sha"]:
        raise ValueError("059 checkpoint lineage does not end at its recorded source")
    model_state = raw.get("model")
    if not isinstance(model_state, dict):
        raise ValueError("O59 checkpoint has no model state")
    encoded = model_state.get("player_code_bytes")
    if not isinstance(encoded, Tensor) or encoded.dtype != torch.uint8 or encoded.ndim != 1:
        raise ValueError("O59 checkpoint player codes are invalid")
    codes = decode_player_codes(encoded.numpy().tobytes())
    if encode_player_codes(codes) != encoded.numpy().tobytes():
        raise ValueError("O59 player codes are not canonical")
    vocabulary = PlayerVocabulary(codes)
    if vocabulary.size != cfg.model.player_vocab_size or vocabulary.sha256 != cfg.model.player_vocab_sha256:
        raise ValueError("O59 player vocabulary does not match its checkpoint")
    protocol = {
        "version": 1,
        "horizon": 60,
        "gamma": 0.99855,
        "reward": "damage_opp-damage_ego+120*(stock_loss_opp-stock_loss_ego)+50*(last_stock_opp-last_stock_ego)",
        "scale": 120.0,
        "alignment": "sum(k=1..60, gamma**(k-1)*r[t+k])",
        "availability": "full observed horizon or known terminal with zero rewards thereafter",
        "evaluation": "positive p90 at every replan; separate unconditioned comparison",
        "modulation": "per-block RMSNorm affine, 128-wide SiLU, biased zero-initialized projections",
        "dropout": "independent CPU generator, per context position",
        "calibration_windows": 65_536,
    }
    if raw.get("conditioning_protocol") != protocol:
        raise ValueError("O59 checkpoint conditioning protocol changed")
    calibration = raw.get("return_calibration")
    if not isinstance(calibration, dict):
        raise ValueError("O59 checkpoint return calibration is missing")
    calibrator = ReturnCalibration(window_count=65_536)
    calibrator.load_state_dict(cast(dict[str, object], calibration))
    targets = calibrator.targets()
    if not isinstance(targets, tuple) or len(targets) != 3:
        raise ValueError("O59 checkpoint return targets are missing")
    p90 = targets[2]
    if not isinstance(p90, float) or not math.isfinite(p90):
        raise ValueError("O59 checkpoint return p90 is invalid")
    masker = raw.get("return_masker")
    if not isinstance(masker, dict) or set(masker) != {"version", "probability", "enabled", "generator"}:
        raise ValueError("O59 checkpoint return masker is missing")
    if (masker["version"], masker["probability"], masker["enabled"]) != (1, cfg.return_dropout, True):
        raise ValueError("O59 checkpoint return masker changed")
    generator_state = masker["generator"]
    if not isinstance(generator_state, Tensor):
        raise ValueError("O59 checkpoint return masker RNG is invalid")
    torch.Generator(device="cpu").set_state(generator_state.cpu())
    with torch.device("meta"):
        model = ActionSequenceTransformer(cfg.model, vocabulary)
    reference = model.state_dict()
    if set(model_state) != set(reference):
        raise ValueError("059 checkpoint model tensor names changed")
    for name, expected in reference.items():
        value = model_state[name]
        if not isinstance(value, Tensor) or value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError(f"059 checkpoint tensor {name!r} has an invalid shape or dtype")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"059 checkpoint tensor {name!r} is nonfinite")
    # These buffers define the representation; they are not learned weights.
    # Constructing a codec must not consume the caller's initialization RNG.
    with torch.random.fork_rng(devices=[]):
        codec_buffers = dict(DiscreteControllerCodec(cfg.model.action_embed_dim).named_buffers())
    for name, expected in model.named_buffers():
        if name.startswith("codec."):
            expected = codec_buffers[name.removeprefix("codec.")]
        elif name.endswith("rotary.inv_freq"):
            expected = Rotary(expected.numel() * 2).inv_freq
        if not torch.equal(model_state[name], expected):
            raise ValueError(f"059 checkpoint representation buffer {name!r} changed")
    model.load_state_dict(cast(dict[str, Tensor], model_state), strict=True, assign=True)
    return raw, cfg, codes, p90


def export_action_sequence_policy(checkpoint_source: str | Path, destination: str | Path) -> PolicyBundleManifest:
    checkpoint = resolve_checkpoint(str(checkpoint_source))
    raw, cfg, _codes, p90 = read_checkpoint(checkpoint)
    source_names = cfg.source_names
    sources = tuple(streams.BY_NAME[name] for name in source_names)
    paths = tuple(streams.ensure_stats(source.local_root / "stats.json") for source in sources)
    stats = load_consolidated_mixture_stats(
        paths,
        tuple(float(streams.POLICY_WORLD_V8_TRAIN_REPLAYS[name]) for name in source_names),
        expected_mds_schema_version=cfg.mds_schema_version,
    )
    stats_payload = {
        name: {"mean": item.mean, "std": item.std, "min": item.min, "max": item.max}
        for name, item in sorted(stats.items())
    }
    stats_bytes = json.dumps(stats_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    config = {
        "schema_version": 2,
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "wandb_id": raw["wandb_id"],
        "step": raw["step"],
        "return_p90": p90,
        "stats_sha256": hashlib.sha256(stats_bytes).hexdigest(),
        "source_stats_sha256": {name: checkpoint_sha256(path) for name, path in zip(source_names, paths, strict=True)},
    }
    with tempfile.TemporaryDirectory(prefix="hal-o59-export-") as temporary:
        root = Path(temporary)
        config_path = root / _CONFIG_MEMBER
        stats_path = root / _STATS_MEMBER
        config_path.write_text(json.dumps(config, sort_keys=True, separators=(",", ":")))
        stats_path.write_bytes(stats_bytes)
        return write_policy_bundle(
            destination,
            BundleDescription(
                policy_name=f"059 update {int(cast(int, raw['step'])) + 1}",
                backend=ACTION_SEQUENCE_BACKEND,
                backend_version=ACTION_SEQUENCE_BACKEND_VERSION,
                required_observation_fields=REQUIRED_OBSERVATION_FIELDS,
                supported_transport_delays=(0, 2, 3),
                requires_player_code=False,
                source_sha256=config["checkpoint_sha256"],
            ),
            {_CONFIG_MEMBER: config_path, _STATS_MEMBER: stats_path, _CHECKPOINT_MEMBER: checkpoint},
        )


def read_action_sequence_artifact(path: str | Path) -> ActionSequenceArtifact:
    with extract_policy_bundle(path) as (manifest, root):
        expected_delays = {1: (2,), 2: (0, 2, 3)}
        if (
            manifest.backend != ACTION_SEQUENCE_BACKEND
            or manifest.backend_version not in expected_delays
            or manifest.required_observation_fields != REQUIRED_OBSERVATION_FIELDS
            or manifest.supported_transport_delays != expected_delays[manifest.backend_version]
            or manifest.requires_player_code
            or manifest.action_schema != "hal.controller.v1"
            or manifest.observation_schema != "hal.flat.numeric.v1"
            or set(member.name for member in manifest.members) != {_CONFIG_MEMBER, _STATS_MEMBER, _CHECKPOINT_MEMBER}
        ):
            raise ValueError("059 bundle manifest is incompatible")
        config = json.loads((root / _CONFIG_MEMBER).read_text())
        if not isinstance(config, dict) or set(config) != {
            "schema_version",
            "checkpoint_sha256",
            "wandb_id",
            "step",
            "return_p90",
            "stats_sha256",
            "source_stats_sha256",
        }:
            raise ValueError("059 bundle config fields changed")
        if (
            config["schema_version"] != manifest.backend_version
            or config["checkpoint_sha256"] != manifest.source_sha256
        ):
            raise ValueError("059 bundle config identity changed")
        checkpoint = root / _CHECKPOINT_MEMBER
        if checkpoint_sha256(checkpoint) != config["checkpoint_sha256"]:
            raise ValueError("059 checkpoint hash differs from bundle config")
        stats_bytes = (root / _STATS_MEMBER).read_bytes()
        if hashlib.sha256(stats_bytes).hexdigest() != config["stats_sha256"]:
            raise ValueError("059 stats hash differs from bundle config")
        raw_stats = json.loads(stats_bytes)
        if not isinstance(raw_stats, dict) or set(raw_stats) != _STATISTIC_FIELDS:
            raise ValueError("059 statistics feature names changed")
        statistics: dict[str, FeatureStats] = {}
        for name, values in raw_stats.items():
            if (
                not isinstance(name, str)
                or not isinstance(values, dict)
                or set(values) != {"mean", "std", "min", "max"}
            ):
                raise ValueError("059 statistics fields changed")
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in values.values()):
                raise ValueError("059 statistics must be finite")
            if values["std"] < 0 or not values["min"] <= values["mean"] <= values["max"]:
                raise ValueError("059 statistics have invalid spread or bounds")
            statistics[name] = FeatureStats(
                mean=values["mean"], std=values["std"], min=values["min"], max=values["max"]
            )
        raw, contract, codes, p90 = read_checkpoint(checkpoint)
        if (raw["step"], raw["wandb_id"], p90) != (config["step"], config["wandb_id"], config["return_p90"]):
            raise ValueError("059 checkpoint state differs from bundle config")
        hashes = config["source_stats_sha256"]
        if not isinstance(hashes, dict) or set(hashes) != set(contract.source_names):
            raise ValueError("059 source statistics identity changed")
        if any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes.values()
        ):
            raise ValueError("059 source statistics hashes are invalid")
        provenance = raw.get("provenance")
        if (
            not isinstance(provenance, dict)
            or cast(dict[str, object], provenance).get("source_statistics_sha256") != hashes
        ):
            raise ValueError("059 source statistics differ from checkpoint provenance")
        return ActionSequenceArtifact(
            contract.model,
            cast(dict[str, Tensor], raw["model"]),
            statistics,
            PlayerVocabulary(codes),
            manifest.source_sha256,
            p90,
            PolicySpec(
                manifest.policy_name,
                manifest.backend,
                manifest.required_observation_fields,
                manifest.supported_transport_delays,
            ),
            manifest.backend_version,
        )


def build_action_sequence_model(
    artifact: ActionSequenceArtifact,
    *,
    device: str | torch.device,
    inference_dtype: torch.dtype | None = None,
) -> ActionSequenceTransformer:
    target = torch.device(device)
    expected_dtype = torch.bfloat16 if target.type == "cuda" else torch.float32
    dtype = expected_dtype if inference_dtype is None else inference_dtype
    if target.type not in ("cpu", "cuda") or dtype != expected_dtype:
        raise ValueError("action sequence inference supports CPU FP32 and CUDA BF16")
    with torch.device("meta"):
        model = ActionSequenceTransformer(artifact.model_config, artifact.vocabulary)
    model.load_state_dict(artifact.model_tensors, strict=True, assign=True)
    model.to(target).eval()
    # Keep embeddings and normalization in checkpoint precision. Static linear
    # conversion avoids recasting all weights in every autocast/graph execution.
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            layer.to(dtype=dtype)
    return model


def load_action_sequence_policy(
    path: str | Path,
    *,
    device: str = "cuda",
    seed: int | None = None,
    compiled: bool = False,
    history_mode: Literal["window", "kv_cache"] = "kv_cache",
    kv_update_frames: int = 2,
    kv_cuda_graphs: bool = True,
) -> ActionSequencePolicy:
    from hal.inference.action_sequence_policy import ActionSequencePolicy

    artifact = read_action_sequence_artifact(path)
    model = build_action_sequence_model(artifact, device=device)
    return ActionSequencePolicy(
        model,
        artifact.statistics,
        artifact.vocabulary.codes,
        spec=artifact.spec,
        checkpoint_sha256=artifact.checkpoint_sha256,
        return_p90=artifact.return_p90,
        capability_version=artifact.capability_version,
        device=torch.device(device),
        seed=seed,
        compiled=compiled,
        history_mode=history_mode,
        kv_update_frames=kv_update_frames,
        kv_cuda_graphs=kv_cuda_graphs,
    )
