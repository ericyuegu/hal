"""Intervene on the O50 Muon/AdamW checkpoint pair without training.

The pure state transforms in this module operate on model-only ``state_dict``
mappings. They use the optimizer-role contract recorded at Git commit
``cdf84f904d322d2a8c342fe3ea75d926e10346ba``. Evaluation is deliberately
optional: the transform tests do not need a checkpoint, replay corpus, or GPU.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from collections.abc import Collection
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from hal.inference import o50_model
from hal.training import features
from hal.training import trunk

type OptimizerPartition = Literal["adamw", "muon"]
type RestorePartition = Literal["readouts", "adamw", "muon"]
type State = Mapping[str, Tensor]

ROLE_COMMIT = "cdf84f904d322d2a8c342fe3ea75d926e10346ba"
INTERPOLATION_ALPHAS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
BOOTSTRAP_SEED = 20_260_911
BOOTSTRAP_RESAMPLES = 10_000
CONTROLLER_GROUP_NAMES = ("buttons", "main_stick", "c_stick", "triggers")
_REPRESENTATION_ARCHITECTURE_FIELDS = (
    "L_ctx",
    "head_offsets",
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
)

_EMBEDDING_PREFIXES = (
    "codec.class_embeddings.",
    "cat_embeds.",
    "char_emb.",
    "stage_emb.",
    "item_type_emb.",
    "item_state_emb.",
    "player_embedding.",
    "temporal.offset_embedding.",
)
_FINITE_PREFIXES = (
    "codec.semantic_projections.",
    "item_encoder.",
    "observation_encoder.",
    "player_projection.",
    "temporal.group_condition.",
)


@dataclass(frozen=True, slots=True)
class InterventionSpec:
    """One named state transform and its control metadata."""

    name: str
    kind: str
    alpha: float | None = None


def _is_final_readout(name: str) -> bool:
    return (
        (name.startswith("temporal.outputs.") and ".down." in name)
        or name.startswith("temporal.trunk_outputs.")
        or name.startswith("value_head.down.")
    )


def _canonical_parameter_name(name: str) -> str:
    if name.startswith("temporal.codec."):
        return "codec." + name.removeprefix("temporal.codec.")
    return name


def optimizer_partition(name: str) -> OptimizerPartition:
    """Return O50's optimizer partition for one named parameter.

    This is a direct transcription of ``optimizer_roles`` at ``ROLE_COMMIT``.
    Buffers are not passed to this function.
    """
    name = _canonical_parameter_name(name)
    if _is_final_readout(name):
        return "adamw"
    if name.startswith("trunk.blocks."):
        return "muon"
    if name.startswith("temporal.blocks."):
        return "muon"
    if name == "temporal.token_projection.weight":
        return "muon"
    if name.startswith("temporal.outputs.") and name.endswith("up.weight"):
        return "muon"
    if name == "value_head.up.weight":
        return "muon"
    if name.startswith(_EMBEDDING_PREFIXES):
        return "adamw"
    if name.startswith(_FINITE_PREFIXES):
        return "adamw"
    if name == "temporal.token_projection.bias":
        return "adamw"
    raise ValueError(f"no O50 optimizer role for parameter {name!r}")


def logical_splits(name: str) -> int:
    """Return the optimizer's logical row split count for one matrix."""
    if name.startswith("trunk.blocks.") and name.endswith("attn.c_attn.weight"):
        return 3
    if name.startswith("temporal.blocks.") and name.endswith("qkv.weight"):
        return 3
    if name == "value_head.up.weight":
        return 2
    return 1


def parameter_roles(parameter_keys: Collection[str]) -> dict[str, OptimizerPartition]:
    """Classify a model's named parameters and reject unknown names."""
    return {name: optimizer_partition(name) for name in parameter_keys}


def _validate_pair(early: State, slump: State, parameter_keys: Collection[str]) -> frozenset[str]:
    early_keys = set(early)
    slump_keys = set(slump)
    if early_keys != slump_keys:
        raise ValueError(
            f"checkpoint state keys differ: early-only={sorted(early_keys - slump_keys)}, "
            f"slump-only={sorted(slump_keys - early_keys)}"
        )
    keys = frozenset(parameter_keys)
    missing = keys - early_keys
    if missing:
        raise ValueError(f"parameter keys are absent from checkpoint state: {sorted(missing)}")
    for name in sorted(early_keys):
        left, right = early[name], slump[name]
        if not isinstance(left, Tensor) or not isinstance(right, Tensor):
            raise TypeError(f"state values must be tensors: {name!r}")
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(
                f"checkpoint tensor mismatch for {name!r}: "
                f"{tuple(left.shape)}/{left.dtype} versus {tuple(right.shape)}/{right.dtype}"
            )
        if name not in keys and not torch.equal(left, right):
            raise ValueError(f"non-parameter buffer differs between checkpoints: {name!r}")
    parameter_roles(keys)
    return keys


def _clone_state(state: State) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in state.items()}


def identity_state(state: State) -> dict[str, Tensor]:
    """Return an exact, detached copy of one state."""
    return _clone_state(state)


def interpolate_state(
    early: State,
    slump: State,
    alpha: float,
    *,
    parameter_keys: Collection[str],
) -> dict[str, Tensor]:
    """Linearly interpolate parameters while preserving every buffer.

    Both checkpoints must have the same state schema. Non-parameter tensors are
    required to be exactly equal and are copied from ``early``. This prevents
    interpolation from changing a vocabulary, mask, or rotary buffer.
    """
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be finite and in [0, 1], got {alpha!r}")
    keys = _validate_pair(early, slump, parameter_keys)
    if alpha == 0.0:
        return identity_state(early)
    if alpha == 1.0:
        return identity_state(slump)
    result: dict[str, Tensor] = {}
    for name, early_value in early.items():
        slump_value = slump[name]
        if name not in keys:
            if not torch.equal(early_value, slump_value):
                raise ValueError(f"non-parameter buffer differs between checkpoints: {name!r}")
            result[name] = early_value.detach().clone()
        elif not early_value.is_floating_point() or not slump_value.is_floating_point():
            if not torch.equal(early_value, slump_value):
                raise ValueError(f"non-floating parameter differs between checkpoints: {name!r}")
            result[name] = early_value.detach().clone()
        else:
            result[name] = torch.lerp(early_value, slump_value, alpha)
    return result


def reverse_interpolate_state(
    early: State,
    slump: State,
    alpha: float,
    *,
    parameter_keys: Collection[str],
) -> dict[str, Tensor]:
    """Interpolate from the slump checkpoint back toward the early checkpoint."""
    return interpolate_state(slump, early, alpha, parameter_keys=parameter_keys)


def restore_partition(
    slump: State,
    early: State,
    partition: RestorePartition,
    *,
    parameter_keys: Collection[str],
) -> dict[str, Tensor]:
    """Restore one early parameter partition into the slump state."""
    if partition not in ("readouts", "adamw", "muon"):
        raise ValueError(f"unknown restore partition {partition!r}")
    keys = _validate_pair(early, slump, parameter_keys)
    roles = parameter_roles(keys)
    result = _clone_state(slump)
    for name in keys:
        selected = _is_final_readout(name) if partition == "readouts" else roles[name] == partition
        if selected:
            result[name] = early[name].detach().clone()
    for name in set(result) - keys:
        if not torch.equal(early[name], slump[name]):
            raise ValueError(f"non-parameter buffer differs between checkpoints: {name!r}")
    return result


def _logical_chunks(value: Tensor, splits: int, name: str) -> tuple[Tensor, ...]:
    if value.ndim < 2:
        raise ValueError(f"Muon norm restoration requires a matrix for {name!r}, got {value.ndim}D")
    if value.shape[0] % splits:
        raise ValueError(f"{name!r} first dimension {value.shape[0]} is not divisible by {splits}")
    return tuple(value.chunk(splits, dim=0))


def restore_muon_norms(
    slump: State,
    early: State,
    *,
    parameter_keys: Collection[str],
) -> dict[str, Tensor]:
    """Restore each early Muon logical matrix norm without changing direction."""
    keys = _validate_pair(early, slump, parameter_keys)
    roles = parameter_roles(keys)
    result = _clone_state(slump)
    for name in sorted(keys):
        if roles[name] != "muon":
            continue
        early_value, slump_value = early[name], slump[name]
        splits = logical_splits(name)
        early_chunks = _logical_chunks(early_value, splits, name)
        slump_chunks = _logical_chunks(slump_value, splits, name)
        restored_chunks: list[Tensor] = []
        for early_chunk, slump_chunk in zip(early_chunks, slump_chunks, strict=True):
            early_norm = torch.linalg.vector_norm(early_chunk.float())
            slump_norm = torch.linalg.vector_norm(slump_chunk.float())
            if slump_norm == 0:
                if early_norm != 0:
                    raise ValueError(f"cannot preserve direction for zero slump matrix {name!r}")
                restored_chunks.append(slump_chunk.detach().clone())
            elif early_norm == 0:
                restored_chunks.append(torch.zeros_like(slump_chunk))
            else:
                restored_chunks.append(slump_chunk * (early_norm / slump_norm).to(slump_chunk.dtype))
        result[name] = torch.cat(restored_chunks, dim=0)
    for name in set(result) - keys:
        if not torch.equal(early[name], slump[name]):
            raise ValueError(f"non-parameter buffer differs between checkpoints: {name!r}")
    return result


def _attention_qk_name(name: str) -> bool:
    return (name.startswith("trunk.blocks.") and name.endswith("attn.c_attn.weight")) or (
        name.startswith("temporal.blocks.") and name.endswith("qkv.weight")
    )


def restore_attention_qk_norms(
    slump: State,
    early: State,
    *,
    parameter_keys: Collection[str],
) -> dict[str, Tensor]:
    """Restore only fused attention Q/K norms, leaving V and other Muon matrices unchanged."""
    keys = _validate_pair(early, slump, parameter_keys)
    roles = parameter_roles(keys)
    result = _clone_state(slump)
    for name in sorted(keys):
        if roles[name] != "muon" or not _attention_qk_name(name):
            continue
        early_chunks = _logical_chunks(early[name], 3, name)
        slump_chunks = _logical_chunks(slump[name], 3, name)
        restored_chunks = list(slump_chunks)
        for index in (0, 1):
            early_norm = torch.linalg.vector_norm(early_chunks[index].float())
            slump_norm = torch.linalg.vector_norm(slump_chunks[index].float())
            if slump_norm == 0:
                if early_norm != 0:
                    raise ValueError(f"cannot restore a zero-norm attention matrix {name!r}")
            elif early_norm == 0:
                restored_chunks[index] = torch.zeros_like(slump_chunks[index])
            else:
                restored_chunks[index] = slump_chunks[index] * (early_norm / slump_norm).to(slump_chunks[index].dtype)
        result[name] = torch.cat(restored_chunks, dim=0)
    return result


def qk_gauge_control(slump: State, *, parameter_keys: Collection[str]) -> dict[str, Tensor]:
    """Scale fused attention Q by two and K by one half on the slump state."""
    keys = frozenset(parameter_keys)
    if not keys <= set(slump):
        raise ValueError("parameter keys are absent from the slump state")
    parameter_roles(keys)
    result = _clone_state(slump)
    for name in sorted(keys):
        if not _attention_qk_name(name):
            continue
        chunks = _logical_chunks(slump[name], 3, name)
        result[name] = torch.cat((chunks[0] * 2, chunks[1] * 0.5, chunks[2]), dim=0)
    return result


def intervention_specs() -> tuple[InterventionSpec, ...]:
    """Return the small intervention set, including identity and reversal controls."""
    specs = [
        InterventionSpec("early_identity", "identity_early"),
        InterventionSpec("slump_identity", "identity_slump"),
    ]
    specs.extend(InterventionSpec(f"interp_a{alpha:g}", "interpolation", alpha) for alpha in INTERPOLATION_ALPHAS)
    specs.extend(
        InterventionSpec(f"reverse_a{alpha:g}", "reverse_interpolation", alpha) for alpha in INTERPOLATION_ALPHAS
    )
    specs.extend(
        (
            InterventionSpec("restore_readouts", "restore_readouts"),
            InterventionSpec("restore_adamw", "restore_adamw"),
            InterventionSpec("restore_muon", "restore_muon"),
            InterventionSpec("restore_muon_norms", "restore_muon_norms"),
            InterventionSpec("restore_attention_qk_norms", "restore_attention_qk_norms"),
            InterventionSpec("qk_gauge_q2_khalf", "qk_gauge"),
        )
    )
    return tuple(specs)


def apply_intervention(
    spec: InterventionSpec,
    early: State,
    slump: State,
    *,
    parameter_keys: Collection[str],
) -> dict[str, Tensor]:
    """Apply one named intervention to the paired states."""
    if spec.kind == "identity_early":
        return identity_state(early)
    if spec.kind == "identity_slump":
        return identity_state(slump)
    if spec.kind == "interpolation":
        if spec.alpha is None:
            raise ValueError(f"interpolation spec {spec.name!r} has no alpha")
        return interpolate_state(early, slump, spec.alpha, parameter_keys=parameter_keys)
    if spec.kind == "reverse_interpolation":
        if spec.alpha is None:
            raise ValueError(f"reverse spec {spec.name!r} has no alpha")
        return reverse_interpolate_state(early, slump, spec.alpha, parameter_keys=parameter_keys)
    if spec.kind == "restore_readouts":
        return restore_partition(slump, early, "readouts", parameter_keys=parameter_keys)
    if spec.kind == "restore_adamw":
        return restore_partition(slump, early, "adamw", parameter_keys=parameter_keys)
    if spec.kind == "restore_muon":
        return restore_partition(slump, early, "muon", parameter_keys=parameter_keys)
    if spec.kind == "restore_muon_norms":
        return restore_muon_norms(slump, early, parameter_keys=parameter_keys)
    if spec.kind == "restore_attention_qk_norms":
        return restore_attention_qk_norms(slump, early, parameter_keys=parameter_keys)
    if spec.kind == "qk_gauge":
        return qk_gauge_control(slump, parameter_keys=parameter_keys)
    raise ValueError(f"unknown intervention kind {spec.kind!r}")


def _bootstrap_mean(values: Tensor, *, seed: int = BOOTSTRAP_SEED) -> dict[str, object]:
    """Summarize one paired per-replay vector with a fixed bootstrap."""
    array = values.detach().float().cpu().numpy()
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ValueError("paired bootstrap values must be a finite vector with at least two replays")
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, array.size, size=(BOOTSTRAP_RESAMPLES, array.size))
    means = array[draws].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "positive_replays": int((array > 0).sum()),
        "bootstrap_95": [float(value) for value in np.quantile(means, (0.025, 0.975))],
    }


def _conditional_means(loss: Tensor, targets: Tensor, observed: Tensor) -> dict[str, object]:
    previous = torch.cat((observed[:, None], targets[:, :-1]), dim=1)
    changed = targets.ne(previous)

    def summarize(mask: Tensor) -> dict[str, object]:
        counts = mask.sum(dim=0)
        means = (loss * mask).sum(dim=0) / counts.clamp_min(1)
        values = means.detach().cpu().tolist()
        for horizon, group in zip(*torch.where(counts == 0), strict=True):
            values[int(horizon)][int(group)] = None
        selected = loss[mask]
        return {
            "count": counts.detach().cpu().tolist(),
            "mean_nll_nats": values,
            "overall_mean_nll_nats": float(selected.mean()) if selected.numel() else None,
        }

    return {"change": summarize(changed), "hold": summarize(~changed)}


def paired_output_metrics(
    baseline_logits: Mapping[str, Tensor],
    treatment_logits: Mapping[str, Tensor],
    baseline_loss: Tensor,
    treatment_loss: Tensor,
    targets: Tensor,
    observed: Tensor,
) -> dict[str, object]:
    """Return hold/change NLL and replay-paired output changes."""
    names = CONTROLLER_GROUP_NAMES
    js_parts: list[Tensor] = []
    argmax_parts: list[Tensor] = []
    for name in names:
        reference = baseline_logits[name].float()
        treatment = treatment_logits[name].float()
        log_p = torch.log_softmax(reference, dim=-1)
        log_q = torch.log_softmax(treatment, dim=-1)
        p, q = log_p.exp(), log_q.exp()
        log_m = torch.log(0.5 * (p + q))
        left = torch.where(p > 0, p * (log_p - log_m), torch.zeros_like(p)).sum(-1)
        right = torch.where(q > 0, q * (log_q - log_m), torch.zeros_like(q)).sum(-1)
        js_parts.append((0.5 * (left + right)).mean(1))
        argmax_parts.append(treatment.argmax(dim=-1).ne(reference.argmax(dim=-1)).float().mean(dim=1))
    js_per_replay = torch.stack(js_parts, dim=-1).mean(dim=-1)
    argmax_per_replay = torch.stack(argmax_parts, dim=-1).mean(dim=-1)
    nll_delta = treatment_loss.mean(dim=(1, 2)) - baseline_loss.mean(dim=(1, 2))
    max_abs_logit_delta = []
    for name in names:
        reference = baseline_logits[name].float()
        treatment = treatment_logits[name].float()
        finite = torch.isfinite(reference) & torch.isfinite(treatment)
        difference = (treatment - reference).abs()
        max_abs_logit_delta.append(float(difference[finite].max()) if finite.any() else 0.0)
    return {
        "baseline_hold_change": _conditional_means(baseline_loss, targets, observed),
        "treatment_hold_change": _conditional_means(treatment_loss, targets, observed),
        "paired_delta": {
            "nll_nats": _bootstrap_mean(nll_delta, seed=BOOTSTRAP_SEED + 1),
            "output_js_nats": _bootstrap_mean(js_per_replay, seed=BOOTSTRAP_SEED + 2),
            "argmax_change_rate": _bootstrap_mean(argmax_per_replay, seed=BOOTSTRAP_SEED + 3),
        },
        "output_js_nats_by_group": torch.stack(js_parts, dim=-1).mean(dim=0).detach().cpu().tolist(),
        "argmax_change_rate_by_group": torch.stack(argmax_parts, dim=-1).mean(dim=0).detach().cpu().tolist(),
        "max_abs_logit_delta_by_group": max_abs_logit_delta,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _string_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a mapping with string keys")
    return cast(Mapping[str, object], value)


def _trace_error(attention: Mapping[str, object]) -> float:
    """Return the validated hidden-state parity error from one trace record."""
    value = attention.get("traced_trunk_max_abs_error")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("attention trace parity error must be numeric")
    return float(value)


def _tensor_bytes(value: object, label: str) -> bytes:
    if not isinstance(value, Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
        raise TypeError(f"{label} must be a one-dimensional uint8 tensor")
    return value.detach().cpu().contiguous().numpy().tobytes()


def _json_normalize(value: object) -> object:
    """Normalize tuple/list distinctions at the JSON sidecar boundary."""
    return json.loads(json.dumps(value, sort_keys=True))


def _validate_representation(
    reference: o50_model.O50Config,
    checkpoint: o50_model.O50Config,
) -> None:
    """Check cohort-facing fields while allowing the historical model width to differ."""
    if reference.prediction_frames != checkpoint.prediction_frames:
        raise ValueError("checkpoint prediction frame count differs from the fixed cohort")
    if reference.mds_schema_version != checkpoint.mds_schema_version:
        raise ValueError("checkpoint MDS schema differs from the fixed cohort")
    if (
        reference.player_vocab_size != checkpoint.player_vocab_size
        or reference.player_vocab_sha256 != checkpoint.player_vocab_sha256
    ):
        raise ValueError("checkpoint player vocabulary differs from the fixed cohort")
    if reference.stats_sha256 != checkpoint.stats_sha256:
        raise ValueError("checkpoint statistics identity differs from the fixed cohort")
    for field_name in _REPRESENTATION_ARCHITECTURE_FIELDS:
        reference_value = getattr(reference.architecture, field_name)
        checkpoint_value = getattr(checkpoint.architecture, field_name)
        if reference_value != checkpoint_value:
            raise ValueError(f"checkpoint representation field {field_name!r} differs from the fixed cohort")


def _load_cohort(
    path: Path,
    *,
    expected_windows: int,
) -> tuple[dict[str, Tensor], Tensor, o50_model.O50Config, dict[str, object], bytes | None, dict[str, object]]:
    """Load the fixed normalized cohort prepared by the O50 probe."""
    package = _string_mapping(torch.load(path, map_location="cpu", weights_only=True), "cohort package")
    expected = {
        "bundle_reference_json",
        "context_features",
        "data_json",
        "player_codes",
        "reference_config",
        "stats_payload",
        "target",
    }
    if set(package) != expected:
        raise ValueError(f"cohort package fields differ: {sorted(set(package) ^ expected)}")
    config = o50_model.O50Config.from_json(_tensor_bytes(package["reference_config"], "reference_config"))
    stats = _string_mapping(json.loads(_tensor_bytes(package["stats_payload"], "stats_payload")), "cohort statistics")
    raw_features = _string_mapping(package["context_features"], "context_features")
    if any(not isinstance(value, Tensor) for value in raw_features.values()):
        raise TypeError("context_features must contain only tensors")
    context_features = {name: cast(Tensor, value) for name, value in raw_features.items()}
    target = package["target"]
    if not isinstance(target, Tensor):
        raise TypeError("cohort target must be a tensor")
    if target.shape[0] != expected_windows:
        raise ValueError(f"cohort has {target.shape[0]} windows, expected {expected_windows}")
    if target.ndim != 3 or target.shape[1] != config.prediction_frames:
        raise ValueError(f"cohort target shape {tuple(target.shape)} does not match O50 config")
    for name, value in context_features.items():
        if value.shape[0] != expected_windows or value.ndim < 2 or value.shape[1] != config.architecture.L_ctx:
            raise ValueError(f"context feature {name!r} has {value.shape[0]} windows")
    data = _string_mapping(json.loads(cast(str, package["data_json"])), "cohort data")
    windows = data.get("windows")
    if not isinstance(windows, list) or len(windows) != expected_windows:
        raise ValueError("cohort data does not record exactly the requested windows")
    replay_ids = [
        _string_mapping(record, "cohort window").get("replay_id") for record in windows if isinstance(record, Mapping)
    ]
    if len(replay_ids) != expected_windows or len(set(replay_ids)) != expected_windows:
        raise ValueError("cohort windows must identify distinct replays")
    bundle_reference = json.loads(cast(str, package["bundle_reference_json"]))
    _string_mapping(bundle_reference, "bundle reference")
    encoded_codes = _tensor_bytes(package["player_codes"], "player_codes")
    player_codes = encoded_codes or None
    provenance = {
        "path": str(path),
        "sha256": _sha256(path),
        "windows": expected_windows,
        "data": data,
        "bundle_reference": bundle_reference,
        "stats_sha256": config.stats_sha256,
        "player_codes_sha256": None if player_codes is None else hashlib.sha256(player_codes).hexdigest(),
    }
    return context_features, target, config, dict(stats), player_codes, provenance


def _load_export(
    path: Path,
    *,
    metadata_path: Path | None,
    reference_config: o50_model.O50Config,
    stats_payload: Mapping[str, object],
    player_codes: bytes | None,
) -> tuple[dict[str, Tensor], o50_model.O50Config, dict[str, object]]:
    """Restricted-load one verified model-only export into CPU memory."""
    raw = _string_mapping(torch.load(path, map_location="cpu", weights_only=True), "model export")
    allowed = {
        "cfg",
        "checkpoint_step",
        "data_protocol",
        "model",
        "source_checkpoint_sha256",
        "source_manifest_sha256",
        "source_r2_etag",
        "source_selection_sha256",
        "source_uri",
        "wandb_id",
    }
    if set(raw) != allowed:
        raise ValueError(
            f"model export fields differ: missing={sorted(allowed - set(raw))}, unexpected={sorted(set(raw) - allowed)}"
        )
    raw_config = _string_mapping(raw["cfg"], "checkpoint config")
    source_names = stats_payload.get("source_names")
    if not isinstance(source_names, list) or raw_config.get("source_names") not in (source_names, tuple(source_names)):
        raise ValueError("checkpoint source_names differ from the fixed cohort statistics")
    resolved_metadata = metadata_path or path.with_suffix(".metadata.json")
    if not resolved_metadata.is_file():
        raise FileNotFoundError(f"verified metadata sidecar is required: {resolved_metadata}")
    metadata = _string_mapping(json.loads(resolved_metadata.read_bytes()), "model export metadata")
    output_sha256 = _sha256(path)
    if metadata.get("output_sha256") != output_sha256 or metadata.get("bytes") != path.stat().st_size:
        raise ValueError("model export bytes differ from its verified metadata")
    if metadata.get("raw_cfg") != _json_normalize(raw_config):
        raise ValueError("model export config differs from its verified metadata")
    manifests = _string_mapping(stats_payload.get("source_manifest_sha256"), "cohort source manifests")
    export_manifests = _string_mapping(raw["source_manifest_sha256"], "checkpoint source manifests")
    if dict(export_manifests) != dict(manifests):
        raise ValueError("checkpoint source manifests differ from the fixed cohort")
    if raw["data_protocol"] != "o50-replay-ring-v8":
        raise ValueError("checkpoint data protocol is not the frozen O50 replay-ring protocol")
    if raw_config.get("mds_schema_version") != reference_config.mds_schema_version:
        raise ValueError("checkpoint MDS schema differs from the fixed cohort")
    if (
        raw_config.get("player_vocab_sha256") != reference_config.player_vocab_sha256
        or raw_config.get("player_vocab_size") != reference_config.player_vocab_size
    ):
        raise ValueError("checkpoint player vocabulary differs from the fixed cohort")
    state_mapping = _string_mapping(raw["model"], "checkpoint model state")
    if any(not isinstance(value, Tensor) for value in state_mapping.values()):
        raise TypeError("checkpoint model state contains a non-tensor value")
    state = {name: cast(Tensor, value).detach().cpu() for name, value in state_mapping.items()}
    encoded_codes = state.get("player_code_bytes")
    checkpoint_codes = _tensor_bytes(encoded_codes, "checkpoint player_code_bytes")
    expected_codes = player_codes or checkpoint_codes
    if checkpoint_codes != expected_codes:
        raise ValueError("checkpoint player vocabulary bytes differ from the fixed cohort")
    if (
        len(expected_codes) != reference_config.player_code_bytes
        or hashlib.sha256(expected_codes).hexdigest() != reference_config.player_vocab_sha256
    ):
        raise ValueError("checkpoint player bytes fail the fixed cohort identity check")
    config = o50_model.O50Config.from_checkpoint(
        raw_config,
        player_code_bytes=len(expected_codes),
        stats_sha256=reference_config.stats_sha256,
    )
    _validate_representation(reference_config, config)
    checkpoint_step = raw["checkpoint_step"]
    if not isinstance(checkpoint_step, int) or isinstance(checkpoint_step, bool) or checkpoint_step < 0:
        raise ValueError("checkpoint_step must be a non-negative integer")
    source_sha256 = raw["source_checkpoint_sha256"]
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise ValueError("source_checkpoint_sha256 is invalid")
    if metadata.get("checkpoint_step") != checkpoint_step or metadata.get("source_checkpoint_sha256") != source_sha256:
        raise ValueError("checkpoint step or source hash differs from its verified metadata")
    provenance = {
        "checkpoint_step": checkpoint_step,
        "model_export": str(path),
        "model_export_sha256": output_sha256,
        "model_export_metadata": str(resolved_metadata),
        "model_export_metadata_sha256": _sha256(resolved_metadata),
        "source_checkpoint_sha256": source_sha256,
        "source_r2_etag": raw["source_r2_etag"],
        "source_selection_sha256": raw["source_selection_sha256"],
        "source_uri": raw["source_uri"],
        "wandb_id": raw["wandb_id"],
    }
    return state, config, provenance


def teacher_forced_logits(
    model: o50_model.O50Model,
    hidden: Tensor,
    observed: Tensor,
    forced: Tensor,
) -> dict[str, Tensor]:
    """Reconstruct O50's teacher-forced temporal logits for one batch."""
    if observed.shape != forced.shape[:1] + (o50_model.CONTROLLER_GROUP_COUNT,):
        raise ValueError("observed action shape does not match forced actions")
    horizon = forced.shape[1]
    temporal = model.temporal
    raw_trunk = hidden[:, -1]
    normalized_trunk = F.rms_norm(raw_trunk, (raw_trunk.shape[-1],), eps=1e-6)
    weight = temporal.token_projection.weight
    trunk_width = temporal.trunk_width
    action_stop = trunk_width + temporal.controller_width
    state_bias = F.linear(normalized_trunk, weight[:, :trunk_width], temporal.token_projection.bias)
    previous = torch.cat((observed[:, None], forced[:, :-1]), dim=1)
    offsets = torch.as_tensor(temporal.head_offsets[:horizon], device=hidden.device)
    action = F.linear(model.codec.embed_frame(previous), weight[:, trunk_width:action_stop])
    offset = F.linear(temporal.offset_embedding(offsets), weight[:, action_stop:])
    state = F.rms_norm(state_bias[:, None] + action + offset[None], (temporal.d_model,), eps=1e-6)
    for raw_block in temporal.blocks:
        qkv = cast(nn.Linear, raw_block._modules["qkv"])
        proj = cast(nn.Linear, raw_block._modules["proj"])
        rotary = cast(trunk.Rotary, raw_block._modules["rotary"])
        up = cast(nn.Linear, raw_block._modules["up"])
        down = cast(nn.Linear, raw_block._modules["down"])
        n_heads = cast(int, raw_block.__dict__["n_heads"])
        head_dim = cast(int, raw_block.__dict__["head_dim"])
        scale = cast(float, raw_block.__dict__["scale"])
        mlp_scale = cast(float, raw_block.__dict__["mlp_scale"])
        normalized = F.rms_norm(state, (temporal.d_model,), eps=1e-6)
        query, key, value = qkv(normalized).split(temporal.d_model, dim=-1)
        shape = (state.shape[0], horizon, n_heads, head_dim)
        query = query.view(shape)
        key = key.view(shape)
        value = value.view(shape).transpose(1, 2)
        cosine, sine = rotary(query)
        query = trunk.apply_rotary_emb(query, cosine, sine).transpose(1, 2)
        key = trunk.apply_rotary_emb(key, cosine, sine).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view_as(state)
        state = state + scale * proj(attended)
        normalized = F.rms_norm(state, (temporal.d_model,), eps=1e-6)
        state = state + mlp_scale * down(F.silu(up(normalized)))
    state = F.rms_norm(state, (temporal.d_model,), eps=1e-6)
    embedded: dict[str, Tensor] = {}
    logits: dict[str, Tensor] = {}
    for position, name in enumerate(o50_model.CONTROLLER_DECODE_ORDER):
        if position:
            prefix = torch.cat([embedded[group] for group in o50_model.CONTROLLER_DECODE_ORDER[:position]], dim=-1)
            raw_scale, shift = temporal.group_condition[name](prefix).chunk(2, dim=-1)
            group_state = state * (1.0 + torch.tanh(raw_scale)) + shift
        else:
            group_state = state
        values = temporal.outputs[name](group_state) + temporal.trunk_outputs[name](raw_trunk)[:, None]
        values = values - values.mean(dim=-1, keepdim=True)
        if name == "buttons":
            trigger_group = o50_model.CONTROLLER_GROUP_INDEX["triggers"]
            values = values.masked_fill(model.codec.button_mask(forced[..., trigger_group]), float("-inf"))
        logits[name] = values
        group = o50_model.CONTROLLER_GROUP_INDEX[name]
        embedded[name] = model.codec.group_embedding(name, forced[..., group])
    return logits


def horizon_group_nll(logits: Mapping[str, Tensor], targets: Tensor) -> Tensor:
    """Return unreduced NLL as [replay, horizon, controller group]."""
    losses = []
    for group, name in enumerate(CONTROLLER_GROUP_NAMES):
        values = logits[name]
        losses.append(
            F.cross_entropy(
                values.float().reshape(-1, values.shape[-1]),
                targets[..., group].reshape(-1),
                reduction="none",
            ).view(targets.shape[:2])
        )
    return torch.stack(losses, dim=-1)


def trace_last_attention(
    model: o50_model.O50Model,
    tokens: Tensor,
    context_padding: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the final trunk state and every head's last-query attention."""
    state = tokens
    length = state.shape[1]
    mask = trunk.dense_mask(context_padding, length, model.trunk.attn_window)
    rows = []
    for raw_block in model.trunk.blocks:
        block = cast(trunk.Block, raw_block)
        normalized = trunk.rmsnorm(state)
        query, key, _ = block.attn.c_attn(normalized).split(block.attn.d_model, dim=-1)
        shape = (state.shape[0], length, block.attn.n_heads, block.attn.head_dim)
        query = query.view(shape)
        key = key.view(shape)
        cosine, sine = block.attn.rotary(query)
        query = trunk.apply_rotary_emb(query, cosine, sine)
        key = trunk.apply_rotary_emb(key, cosine, sine)
        with torch.autocast(device_type=query.device.type, enabled=False):
            scores = torch.einsum("bhd,bkhd->bhk", query[:, -1].float(), key.float())
            scores = scores / math.sqrt(block.attn.head_dim)
            scores = scores.masked_fill(~mask[:, 0, -1, None, :], float("-inf"))
            weights = F.softmax(scores, dim=-1)
            if weights.dtype != torch.float32:
                raise RuntimeError(f"attention weights must be float32, got {weights.dtype}")
            rows.append(weights)
        state = block(state, mask, context_padding)
    return trunk.rmsnorm(state), torch.stack(rows, dim=1)


def attention_metrics(weights: Tensor) -> dict[str, object]:
    """Summarize [replay, layer, head, context] last-query weights."""
    length = weights.shape[-1]
    lags = torch.arange(length - 1, -1, -1, device=weights.device, dtype=weights.dtype)
    entropy = -(weights * weights.clamp_min(1e-30).log()).sum(dim=-1)
    mean_lag = (weights * lags).sum(dim=-1)

    def aggregate(values: Tensor) -> list[list[float]]:
        return values.mean(dim=0).detach().cpu().tolist()

    return {
        "mean_entropy_nats": float(entropy.mean()),
        "mean_effective_context": float(entropy.exp().mean()),
        "entropy_nats": aggregate(entropy),
        "effective_context": aggregate(entropy.exp()),
        "mean_lag": aggregate(mean_lag),
        "last_4_mass": aggregate(weights[..., -4:].sum(dim=-1)),
        "last_16_mass": aggregate(weights[..., -16:].sum(dim=-1)),
        "distant_64_plus_mass": aggregate(weights[..., : max(0, length - 64)].sum(dim=-1)),
    }


def _evaluate_state(
    model: o50_model.O50Model,
    state: State,
    *,
    context_features: Mapping[str, Tensor],
    context_actions: Tensor,
    targets: Tensor,
    config: o50_model.O50Config,
    device: torch.device,
    batch_size: int,
    trace_attention: bool,
) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, int, dict[str, object] | None]:
    """Evaluate one CPU state on the fixed normalized cohort."""
    model.load_state_dict(state, strict=True)
    logit_chunks: dict[str, list[Tensor]] = {name: [] for name in CONTROLLER_GROUP_NAMES}
    loss_chunks: list[Tensor] = []
    greedy_chunks: list[Tensor] = []
    attention_chunks: list[Tensor] = []
    attention_dtypes: set[str] = set()
    public_mismatch_actions = 0
    trace_max_error = 0.0
    for start in range(0, targets.shape[0], batch_size):
        stop = min(targets.shape[0], start + batch_size)
        batch_features = {name: value[start:stop].to(device) for name, value in context_features.items()}
        batch_actions = context_actions[start:stop].to(device)
        batch_targets = targets[start:stop].to(device)
        observed = batch_actions[:, -1]
        padding = torch.zeros(stop - start, dtype=torch.long, device=device)
        with torch.inference_mode(), o50_model.amp_context(config, device):
            if trace_attention:
                tokens = model.context_tokens(batch_features, batch_actions)
                traced_hidden, attention = trace_last_attention(model, tokens, padding)
                attention_dtypes.add(str(attention.dtype))
                attention_chunks.append(attention.detach().float().cpu())
            hidden = model.forward_dense(batch_features, padding, batch_actions)
            if trace_attention:
                trace_max_error = max(
                    trace_max_error,
                    float((traced_hidden.float() - hidden.float()).abs().max().item()),
                )
            logits = teacher_forced_logits(model, hidden, observed, batch_targets)
            losses = horizon_group_nll(logits, batch_targets)
            empty_forced = torch.zeros_like(batch_targets)
            empty_mask = torch.zeros(batch_targets.shape[:2], dtype=torch.bool, device=device)
            greedy = model.temporal.sample_conditioned(
                hidden,
                observed,
                empty_forced,
                empty_mask,
                torch.full(
                    (config.prediction_frames, o50_model.CONTROLLER_GROUP_COUNT, stop - start),
                    0.5,
                    device=device,
                ),
                argmax=True,
            )
            greedy_logits = teacher_forced_logits(model, hidden, observed, greedy)
            reconstructed = torch.stack(
                [greedy_logits[name].argmax(dim=-1) for name in CONTROLLER_GROUP_NAMES],
                dim=-1,
            )
            public_mismatch_actions += int((greedy != reconstructed).sum().item())
        for name in CONTROLLER_GROUP_NAMES:
            logit_chunks[name].append(logits[name].detach().float().cpu())
        loss_chunks.append(losses.detach().float().cpu())
        greedy_chunks.append(greedy.detach().cpu())
    return (
        {name: torch.cat(chunks) for name, chunks in logit_chunks.items()},
        torch.cat(loss_chunks),
        context_actions[:, -1],
        torch.cat(greedy_chunks),
        public_mismatch_actions,
        None
        if not trace_attention
        else {
            "metrics": attention_metrics(torch.cat(attention_chunks)),
            "traced_trunk_max_abs_error": trace_max_error,
            "attention_dtype": sorted(attention_dtypes),
        },
    )


def _canonical_state_key(spec: InterventionSpec) -> str:
    if spec.kind == "identity_early":
        return "early"
    if spec.kind == "identity_slump":
        return "slump"
    if spec.kind == "interpolation":
        if spec.alpha is None:
            raise ValueError(f"interpolation spec {spec.name!r} has no alpha")
        return f"interp_a{spec.alpha:g}"
    if spec.kind == "reverse_interpolation":
        if spec.alpha is None:
            raise ValueError(f"reverse spec {spec.name!r} has no alpha")
        return f"interp_a{1.0 - spec.alpha:g}"
    return spec.name


def _role_summary(parameter_keys: Collection[str]) -> dict[str, object]:
    unique_keys = frozenset(_canonical_parameter_name(name) for name in parameter_keys)
    roles = parameter_roles(unique_keys)
    counts = {role: sum(value == role for value in roles.values()) for role in ("adamw", "muon")}
    parameter_counts = {
        role: sum(1 for name, assigned in roles.items() if assigned == role) for role in ("adamw", "muon")
    }
    return {
        "parameter_count": len(unique_keys),
        "state_parameter_slot_count": len(parameter_keys),
        "by_optimizer": counts,
        "muon_logical_split_matrices": sum(logical_splits(name) > 1 for name, role in roles.items() if role == "muon"),
        "final_readout_count": sum(_is_final_readout(name) for name in parameter_keys),
        "role_contract_commit": ROLE_COMMIT,
        "parameter_count_by_optimizer": parameter_counts,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--early-checkpoint", type=Path, required=True)
    parser.add_argument("--slump-checkpoint", type=Path, required=True)
    parser.add_argument("--early-metadata", type=Path)
    parser.add_argument("--slump-metadata", type=Path)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--windows", type=int, default=256)
    return parser.parse_args()


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    """Run all small state interventions on one fixed normalized replay cohort."""
    started = time.monotonic()
    if args.windows != 256:
        raise ValueError("the intervention protocol requires exactly 256 replay windows")
    if not 1 <= args.batch_size <= 32:
        raise ValueError("batch size must be in [1, 32]")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    context_features, target, reference_config, stats_payload, player_codes, cohort_provenance = _load_cohort(
        args.cohort,
        expected_windows=args.windows,
    )
    early_state, early_config, early_provenance = _load_export(
        args.early_checkpoint,
        metadata_path=args.early_metadata,
        reference_config=reference_config,
        stats_payload=stats_payload,
        player_codes=player_codes,
    )
    slump_state, slump_config, slump_provenance = _load_export(
        args.slump_checkpoint,
        metadata_path=args.slump_metadata,
        reference_config=reference_config,
        stats_payload=stats_payload,
        player_codes=player_codes,
    )
    if early_config.to_json() != slump_config.to_json():
        raise ValueError("early and slump checkpoint configs differ")

    _validate_representation(reference_config, early_config)
    model = o50_model.O50Model(early_config)
    parameter_ids = {id(parameter) for _, parameter in model.named_parameters()}
    parameter_keys = frozenset(
        name for name, value in model.state_dict(keep_vars=True).items() if id(value) in parameter_ids
    )
    state_keys = set(model.state_dict())
    if set(early_state) != state_keys or set(slump_state) != state_keys:
        raise ValueError("checkpoint model state keys differ from the public O50 model schema")
    _validate_pair(early_state, slump_state, parameter_keys)
    model.load_state_dict(slump_state, strict=True)
    context_actions = model.codec.quantize(features.stack_actions(context_features))
    targets = model.codec.quantize(target)
    model.to(device).eval()

    baseline_logits, baseline_loss, observed, baseline_greedy, baseline_public_mismatches, baseline_attention = (
        _evaluate_state(
            model,
            slump_state,
            context_features=context_features,
            context_actions=context_actions,
            targets=targets,
            config=early_config,
            device=device,
            batch_size=args.batch_size,
            trace_attention=True,
        )
    )
    if baseline_attention is None or _trace_error(baseline_attention) > 2e-2:
        raise RuntimeError("attention trace changed the public dense trunk output")
    cached: dict[str, tuple[dict[str, Tensor], Tensor, Tensor, dict[str, object] | None, int]] = {
        "slump": (baseline_logits, baseline_loss, baseline_greedy, baseline_attention, baseline_public_mismatches)
    }
    records: dict[str, object] = {}
    for spec in intervention_specs():
        key = _canonical_state_key(spec)
        if key not in cached:
            if key == "early":
                state = early_state
            elif key == "slump":
                state = slump_state
            else:
                state = apply_intervention(spec, early_state, slump_state, parameter_keys=parameter_keys)
            trace_this_state = key in {"early", "restore_attention_qk_norms", "qk_gauge_q2_khalf"}
            (
                treatment_logits,
                treatment_loss,
                treatment_observed,
                treatment_greedy,
                treatment_public_mismatches,
                treatment_attention,
            ) = _evaluate_state(
                model,
                state,
                context_features=context_features,
                context_actions=context_actions,
                targets=targets,
                config=early_config,
                device=device,
                batch_size=args.batch_size,
                trace_attention=trace_this_state,
            )
            if not torch.equal(observed, treatment_observed):
                raise RuntimeError(f"intervention {spec.name!r} changed the observed action cohort")
            if treatment_attention is not None and _trace_error(treatment_attention) > 2e-2:
                raise RuntimeError(f"intervention {spec.name!r} changed the public dense trunk output")
            cached[key] = (
                treatment_logits,
                treatment_loss,
                treatment_greedy,
                treatment_attention,
                treatment_public_mismatches,
            )
            if key not in ("early", "slump"):
                del state
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        treatment_logits, treatment_loss, treatment_greedy, treatment_attention, treatment_public_mismatches = cached[
            key
        ]
        metrics = paired_output_metrics(
            baseline_logits,
            treatment_logits,
            baseline_loss,
            treatment_loss,
            targets,
            observed,
        )
        record: dict[str, object] = {
            "canonical_state": key,
            "kind": spec.kind,
            "alpha": spec.alpha,
            "metrics": metrics,
            "public_greedy_control": {
                "mismatch_actions": treatment_public_mismatches,
                "total_actions": int(targets.numel()),
                "note": "CUDA bfloat16 sequential/full-attention argmax differences are recorded, not used to alter NLL.",
            },
        }
        if treatment_attention is not None:
            record["attention"] = treatment_attention
        if spec.kind == "qk_gauge":
            record["public_output_parity_control"] = {
                "greedy_actions_equal": bool(torch.equal(baseline_greedy, treatment_greedy)),
                "max_abs_logit_delta": max(cast(list[float], metrics["max_abs_logit_delta_by_group"])),
                "expected": "QK scores and public outputs should be unchanged up to floating-point noise",
            }
        records[spec.name] = record

    role_summary = _role_summary(parameter_keys)
    result: dict[str, object] = {
        "schema_version": 1,
        "protocol": {
            "checkpoint_pair": "p1fyyp1z step 24576 versus step 57344",
            "replays": args.windows,
            "teacher_forcing": True,
            "no_training": True,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "controller_group_order": list(CONTROLLER_GROUP_NAMES),
            "nll_units": "natural-log nats per controller group and predicted frame",
            "role_contract_commit": ROLE_COMMIT,
        },
        "cohort": cohort_provenance,
        "checkpoints": {"early": early_provenance, "slump": slump_provenance},
        "model": {
            "representation_config": json.loads(reference_config.to_json()),
            "checkpoint_config": json.loads(early_config.to_json()),
            "parameter_roles": role_summary,
        },
        "baseline_slump": {
            "mean_nll_nats": float(baseline_loss.mean()),
            "hold_change": _conditional_means(baseline_loss, targets, observed),
            "attention": baseline_attention,
            "public_greedy_control": {
                "mismatch_actions": baseline_public_mismatches,
                "total_actions": int(targets.numel()),
                "note": "CUDA bfloat16 sequential/full-attention argmax differences are recorded, not used to alter NLL.",
            },
        },
        "interventions": records,
        "device": {
            "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "batch_size": args.batch_size,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"intervention probe refuses to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main() -> None:
    args = _parse_args()
    run_probe(args)


if __name__ == "__main__":
    main()
