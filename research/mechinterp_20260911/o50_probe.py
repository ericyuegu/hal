"""Run a small, reproducible history probe on a portable O50 policy."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import sys
import time
from collections.abc import Iterator
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import torch as safetensors_torch
from torch import Tensor

from hal.data import feature_stats
from hal.data import mds
from hal.data import policy_schema
from hal.inference import bundle
from hal.inference import o50
from hal.inference import o50_model
from hal.training import dataloader
from hal.training import features
from hal.training import player_identity
from hal.training import trunk

SEED = 20_260_911
WINDOW_COUNT = 32
ANALYSIS_BASE_GIT_SHA = "52da5b2de1d402b86404fc4d24e382409649e525"
HISTORY_LENGTHS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
SOURCE_ROOT = Path("/home/ericgu/src/hal/data/processed/ranked-anonymized-1/mds-v7")
SOURCE_MANIFEST = SOURCE_ROOT / "manifest.jsonl"
IDENTITY_SIDECAR = Path("/home/ericgu/src/hal/data/processed/player-identity-v1/professional-code-v1.jsonl.gz")
BUNDLE_PATH = Path("/home/ericgu/src/hal/runs/policies/step-0016384.halpolicy")
OUTPUT_ROOT = Path("research/mechinterp_20260911/artifacts/o50")
HISTORICAL_MODELS = (
    "p1fyyp1z-step-0024576",
    "p1fyyp1z-step-0057344",
    "p1fyyp1z-step-0098304",
    "p1fyyp1z-step-0131072",
    "tgpuo1be-step-0057344",
    "tgpuo1be-step-0131072",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_arrays(values: Mapping[str, object], start: int, stop: int) -> str:
    """Hash names, dtypes, shapes, and exact bytes for one source window."""
    digest = hashlib.sha256()
    for name in sorted(values):
        value = np.asarray(values[name])
        if value.ndim:
            value = np.ascontiguousarray(value[start:stop])
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(json.dumps(value.shape, separators=(",", ":")).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _hash_tensors(values: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(tuple(value.shape), separators=(",", ":")).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _bytes_tensor(value: bytes) -> Tensor:
    return torch.from_numpy(np.frombuffer(value, dtype=np.uint8).copy())


def _tensor_bytes(value: Tensor) -> bytes:
    if value.dtype != torch.uint8 or value.ndim != 1:
        raise TypeError("encoded byte payload must be a one-dimensional uint8 tensor")
    return value.detach().cpu().contiguous().numpy().tobytes()


def _manifest_rows(path: Path) -> list[dict[str, object]]:
    rows: dict[int, dict[str, object]] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            raw = json.loads(line)
            annotation = raw.get("annotation")
            if not isinstance(annotation, dict) or annotation.get("split") != "val":
                continue
            index = int(annotation["mds_row_idx"])
            source_path = str(raw["path"])
            rows[index] = {
                "frame_count": int(annotation["frame_count_actual"]),
                "path": source_path,
                "replay_id": policy_schema.policy_replay_identity(source_path),
                "sha1": str(raw["sha1"]),
            }
    if sorted(rows) != list(range(len(rows))):
        raise ValueError("validation manifest row indices are not contiguous")
    return [rows[index] for index in range(len(rows))]


def _finalized_stats(path: Path) -> dict[str, feature_stats.FeatureStats]:
    raw = json.loads(path.read_bytes())
    finalized = raw.get("finalized")
    if not isinstance(finalized, dict):
        raise ValueError("validated O50 statistics have no finalized block")
    return {
        str(name): feature_stats.FeatureStats(
            mean=float(block["mean"]),
            std=float(block["std"]),
            min=float(block["min"]),
            max=float(block["max"]),
        )
        for name, block in finalized.items()
    }


def _selected_rows(total: int, count: int) -> np.ndarray:
    if count > total:
        raise ValueError(f"requested {count} windows from only {total} replays")
    return np.sort(np.random.default_rng(SEED).choice(total, size=count, replace=False))


def _window_rng(replay_id: str) -> np.random.Generator:
    identity = int.from_bytes(hashlib.blake2b(replay_id.encode(), digest_size=8).digest(), "little")
    return np.random.default_rng((SEED, identity & 0xFFFFFFFF, identity >> 32))


def load_windows(
    *,
    source_root: Path,
    source_manifest: Path,
    identity_sidecar: Path | None,
    context_length: int,
    horizon: int,
    count: int,
    scratch: Path,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], dict[str, str | None]]:
    """Select and load one full-context window from each distinct replay."""
    manifest = _manifest_rows(source_manifest)
    shard_infos = mds.read_shard_index(source_root, "val")
    if sum(int(info["samples"]) for info in shard_infos) != len(manifest):
        raise ValueError("validation MDS rows do not match the source manifest")
    selected = set(int(index) for index in _selected_rows(len(manifest), count))
    sidecar = None if identity_sidecar is None else player_identity.load_player_identity_sidecar(identity_sidecar)
    professional = {} if sidecar is None else sidecar.by_replay
    lookup = player_identity.ReplayPlayerLookup(professional)
    windows: list[dataloader.Window] = []
    records: list[dict[str, object]] = []
    global_row = 0
    length = context_length + horizon
    for shard_index, info in enumerate(shard_infos):
        shard_rows = int(info["samples"])
        local_rows = [
            index - global_row for index in sorted(selected) if global_row <= index < global_row + shard_rows
        ]
        if local_rows:
            with mds.open_shard(source_root, "val", info, scratch) as reader:
                for local_row in local_rows:
                    source = dict(reader[local_row])
                    row_index = global_row + local_row
                    metadata = manifest[row_index]
                    replay_id = str(metadata["replay_id"])
                    frame = np.asarray(source["frame"])
                    if int(source["schema_version"]) != 7:
                        raise ValueError(f"row {row_index} is not canonical schema 7")
                    if len(frame) < length:
                        raise ValueError(f"replay {replay_id} has only {len(frame)} frames; need {length}")
                    if len(frame) != cast(int, metadata["frame_count"]):
                        raise ValueError(f"replay {replay_id} frame count differs from its source manifest")
                    rng = _window_rng(replay_id)
                    start = int(rng.integers(0, len(frame) - length + 1))
                    ego_prefix = "p1" if rng.random() < 0.5 else "p2"
                    raw_window_sha256 = _hash_arrays(source, start, start + length)
                    p1_id, p2_id = lookup.ids(
                        {
                            "replay_id": replay_id,
                            "p1_rank": int(np.asarray(source["p1_rank"])[0]),
                            "p2_rank": int(np.asarray(source["p2_rank"])[0]),
                        }
                    )
                    source["p1_player_id"] = np.full(len(frame), p1_id, dtype=np.int32)
                    source["p2_player_id"] = np.full(len(frame), p2_id, dtype=np.int32)
                    window = dataloader.make_window(
                        source,
                        ego_prefix=ego_prefix,
                        start=start,
                        pad=0,
                        length=length,
                        projection=features.ITEM_PLAYER_PROJECTION,
                    )
                    window["ctx_pad"] = np.asarray(0, dtype=np.int64)
                    windows.append(window)
                    identity_id = p1_id if ego_prefix == "p1" else p2_id
                    records.append(
                        {
                            "ego": ego_prefix,
                            "ego_player_id": identity_id,
                            "frame_start": int(frame[start]),
                            "global_row": row_index,
                            "identity_source": "professional_code" if replay_id in professional else "rank",
                            "length": length,
                            "raw_schema7_window_sha256": raw_window_sha256,
                            "replay_id": replay_id,
                            "replay_sha1": metadata["sha1"],
                            "shard": shard_index,
                            "shard_row": local_row,
                            "start": start,
                        }
                    )
        global_row += shard_rows
    if len(windows) != count or len({str(record["replay_id"]) for record in records}) != count:
        raise RuntimeError("window selection did not produce the requested distinct replays")
    columns = dataloader.collate_windows(windows)
    provenance = {
        "identity_sidecar_sha256": None if sidecar is None else sidecar.sha256,
        "source_index_sha256": _sha256_file(source_root / "val" / "index.json"),
        "source_manifest_sha256": _sha256_file(source_manifest),
    }
    return columns, records, provenance


def teacher_forced_logits(
    model: o50_model.O50Model,
    hidden: Tensor,
    observed: Tensor,
    forced: Tensor,
    *,
    mask_invalid_buttons: bool = True,
) -> dict[str, Tensor]:
    """Reconstruct O50's four-step teacher-forced logits from public modules."""
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
    for block in temporal.blocks:
        qkv = cast(torch.nn.Linear, block.qkv)
        rotary = cast(trunk.Rotary, block.rotary)
        projection = cast(torch.nn.Linear, block.proj)
        up = cast(torch.nn.Linear, block.up)
        down = cast(torch.nn.Linear, block.down)
        d_model = cast(int, block.d_model)
        n_heads = cast(int, block.n_heads)
        head_dim = cast(int, block.head_dim)
        normalized = F.rms_norm(state, (temporal.d_model,), eps=1e-6)
        query, key, value = qkv(normalized).split(d_model, dim=-1)
        shape = (state.shape[0], horizon, n_heads, head_dim)
        query = query.view(shape)
        key = key.view(shape)
        value = value.view(shape).transpose(1, 2)
        cosine, sine = rotary(query)
        query = trunk.apply_rotary_emb(query, cosine, sine).transpose(1, 2)
        key = trunk.apply_rotary_emb(key, cosine, sine).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view_as(state)
        state = state + cast(float, block.scale) * projection(attended)
        normalized = F.rms_norm(state, (temporal.d_model,), eps=1e-6)
        state = state + cast(float, block.mlp_scale) * down(F.silu(up(normalized)))
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
        if name == "buttons" and mask_invalid_buttons:
            trigger_group = o50_model.CONTROLLER_GROUP_INDEX["triggers"]
            values = values.masked_fill(model.codec.button_mask(forced[..., trigger_group]), float("-inf"))
        logits[name] = values
        group = o50_model.CONTROLLER_GROUP_INDEX[name]
        embedded[name] = model.codec.group_embedding(name, forced[..., group])
    return logits


def group_nll(logits: Mapping[str, Tensor], targets: Tensor) -> Tensor:
    """Return per-row, per-controller-group NLL averaged over horizon."""
    losses = []
    for group, name in enumerate(o50_model.CONTROLLER_GROUP_NAMES):
        values = logits[name]
        losses.append(
            F.cross_entropy(
                values.float().reshape(-1, values.shape[-1]),
                targets[..., group].reshape(-1),
                reduction="none",
            ).view(targets.shape[:2])
        )
    return torch.stack(losses, dim=-1).mean(dim=1)


def horizon_group_nll(logits: Mapping[str, Tensor], targets: Tensor) -> Tensor:
    """Return unreduced NLL as [replay, horizon, controller group]."""
    losses = []
    for group, name in enumerate(o50_model.CONTROLLER_GROUP_NAMES):
        values = logits[name]
        losses.append(
            F.cross_entropy(
                values.float().reshape(-1, values.shape[-1]),
                targets[..., group].reshape(-1),
                reduction="none",
            ).view(targets.shape[:2])
        )
    return torch.stack(losses, dim=-1)


def group_js_divergence(reference: Mapping[str, Tensor], treatment: Mapping[str, Tensor]) -> Tensor:
    """Return per-row, per-group Jensen-Shannon divergence over the horizon."""
    divergences = []
    for name in o50_model.CONTROLLER_GROUP_NAMES:
        log_p = F.log_softmax(reference[name].float(), dim=-1)
        log_q = F.log_softmax(treatment[name].float(), dim=-1)
        p = log_p.exp()
        q = log_q.exp()
        mixture = 0.5 * (p + q)
        log_mixture = mixture.log()
        left = torch.where(p > 0, p * (log_p - log_mixture), 0.0).sum(dim=-1)
        right = torch.where(q > 0, q * (log_q - log_mixture), 0.0).sum(dim=-1)
        divergences.append((0.5 * (left + right)).mean(dim=1))
    return torch.stack(divergences, dim=-1)


def trace_last_attention(
    model: o50_model.O50Model,
    tokens: Tensor,
    context_padding: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return final hidden states and every trunk head's last-query weights."""
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
            scores = torch.einsum("bhd,bkhd->bhk", query[:, -1].float(), key.float()) / math.sqrt(block.attn.head_dim)
            scores = scores.masked_fill(~mask[:, 0, -1, None, :], float("-inf"))
            rows.append(F.softmax(scores, dim=-1))
        state = block(state, mask, context_padding)
    return trunk.rmsnorm(state), torch.stack(rows, dim=1)


def attention_metrics(weights: Tensor) -> dict[str, object]:
    """Summarize [batch, layer, head, key] last-query attention."""
    length = weights.shape[-1]
    lags = torch.arange(length - 1, -1, -1, device=weights.device, dtype=weights.dtype)
    entropy = -(weights * weights.clamp_min(1e-30).log()).sum(dim=-1)
    mean_lag = (weights * lags).sum(dim=-1)

    def aggregate(values: Tensor) -> list[list[float]]:
        return values.mean(dim=0).detach().cpu().tolist()

    return {
        "distant_64_plus_mass": aggregate(weights[..., : max(0, length - 64)].sum(dim=-1)),
        "effective_context": aggregate(entropy.exp()),
        "entropy_nats": aggregate(entropy),
        "last_16_mass": aggregate(weights[..., -16:].sum(dim=-1)),
        "last_4_mass": aggregate(weights[..., -4:].sum(dim=-1)),
        "mean_lag": aggregate(mean_lag),
        "current_mass": aggregate(weights[..., -1]),
    }


@contextlib.contextmanager
def zero_attention_head(block: trunk.Block, head: int) -> Iterator[None]:
    """Zero one head's concatenated value before the attention output projection."""
    if not 0 <= head < block.attn.n_heads:
        raise ValueError(f"head {head} is outside [0, {block.attn.n_heads})")
    start = head * block.attn.head_dim
    stop = start + block.attn.head_dim

    def zero_slice(module: torch.nn.Module, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        del module
        values = inputs[0].clone()
        values[..., start:stop] = 0
        return (values, *inputs[1:])

    handle = block.attn.c_proj.register_forward_pre_hook(zero_slice)
    try:
        yield
    finally:
        handle.remove()


def _tensor_rows(values: Tensor) -> list[list[float]]:
    return values.detach().cpu().tolist()


def _paired_summary(delta: Tensor) -> dict[str, object]:
    """Summarize the replay-paired horizon/group mean with a fixed bootstrap."""
    replay_means = delta.float().mean(dim=-1).detach().cpu().numpy()
    generator = np.random.default_rng(SEED)
    draws = generator.integers(0, len(replay_means), size=(20_000, len(replay_means)))
    bootstrap = replay_means[draws].mean(axis=1)
    return {
        "mean": float(replay_means.mean()),
        "median": float(np.median(replay_means)),
        "positive_replays": int((replay_means > 0).sum()),
        "bootstrap_95": [float(value) for value in np.quantile(bootstrap, (0.025, 0.975))],
    }


def _treatment_tensors(
    baseline_logits: Mapping[str, Tensor],
    baseline_nll: Tensor,
    logits: Mapping[str, Tensor],
    targets: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    nll = group_nll(logits, targets)
    delta = nll - baseline_nll
    js = group_js_divergence(baseline_logits, logits)
    changes = torch.stack(
        [
            (logits[name].argmax(dim=-1) != baseline_logits[name].argmax(dim=-1)).float().mean(1)
            for name in o50_model.CONTROLLER_GROUP_NAMES
        ],
        dim=-1,
    )
    return nll, delta, js, changes


def _treatment_metrics_from_tensors(nll: Tensor, delta: Tensor, js: Tensor, changes: Tensor) -> dict[str, object]:
    return {
        "group_js_mean": js.mean(dim=0).detach().cpu().tolist(),
        "group_nll_delta_mean": delta.mean(dim=0).detach().cpu().tolist(),
        "group_nll_mean": nll.mean(dim=0).detach().cpu().tolist(),
        "group_top1_change_rate": changes.mean(dim=0).detach().cpu().tolist(),
        "per_replay_group_js": _tensor_rows(js),
        "per_replay_group_nll_delta": _tensor_rows(delta),
        "replay_paired_horizon_group_mean_nll_delta": _paired_summary(delta),
    }


def treatment_metrics(
    baseline_logits: Mapping[str, Tensor],
    baseline_nll: Tensor,
    logits: Mapping[str, Tensor],
    targets: Tensor,
) -> dict[str, object]:
    return _treatment_metrics_from_tensors(*_treatment_tensors(baseline_logits, baseline_nll, logits, targets))


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_ranks = np.argsort(np.argsort(left))
    right_ranks = np.argsort(np.argsort(right))
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _string_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a mapping with string keys")
    return cast(Mapping[str, object], value)


def load_checkpoint_model(
    path: Path,
    *,
    reference_config: o50_model.O50Config,
    stats_payload: Mapping[str, object],
    player_codes: bytes | None,
    device: torch.device,
) -> tuple[o50_model.O50Model, o50_model.O50Config, dict[str, object]]:
    """Restricted-load and validate one model-only historical export."""
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
    required = allowed
    if set(raw) - allowed or not required <= raw.keys():
        raise ValueError(
            f"model export fields differ: missing={sorted(required - raw.keys())}, "
            f"unexpected={sorted(raw.keys() - allowed)}"
        )
    raw_config = _string_mapping(raw["cfg"], "checkpoint config")
    source_names = stats_payload.get("source_names")
    if not isinstance(source_names, list) or raw_config.get("source_names") not in (source_names, tuple(source_names)):
        raise ValueError("checkpoint source_names differ from the validated bundle statistics")
    metadata_path = path.with_suffix(".metadata.json")
    metadata = _string_mapping(json.loads(metadata_path.read_bytes()), "model export metadata")
    output_sha256 = _sha256_file(path)
    if metadata.get("output_sha256") != output_sha256 or metadata.get("bytes") != path.stat().st_size:
        raise ValueError("model export bytes differ from the adjacent verified metadata")
    canonical_raw_config = json.loads(json.dumps(raw_config, sort_keys=True))
    if metadata.get("raw_cfg") != canonical_raw_config:
        raise ValueError("model export config differs from the adjacent verified metadata")
    manifests = _string_mapping(stats_payload.get("source_manifest_sha256"), "bundle source manifests")
    export_manifests = _string_mapping(
        raw.get("source_manifest_sha256", metadata.get("source_manifest_sha256")),
        "checkpoint source manifests",
    )
    if dict(export_manifests) != dict(manifests):
        raise ValueError("checkpoint source manifests differ from the validated bundle statistics")
    if raw["data_protocol"] != "o50-replay-ring-v8":
        raise ValueError("checkpoint data protocol is not the frozen O50 replay-ring protocol")
    if raw_config.get("mds_schema_version") != reference_config.mds_schema_version:
        raise ValueError("checkpoint MDS schema differs from the validated bundle")
    if (
        raw_config.get("player_vocab_sha256") != reference_config.player_vocab_sha256
        or raw_config.get("player_vocab_size") != reference_config.player_vocab_size
    ):
        raise ValueError("checkpoint player vocabulary differs from the validated bundle")
    state = _string_mapping(raw["model"], "checkpoint model state")
    if any(not isinstance(value, Tensor) for value in state.values()):
        raise TypeError("checkpoint model state contains a non-tensor value")
    if player_codes is None:
        encoded_codes = state.get("player_code_bytes")
        if not isinstance(encoded_codes, Tensor):
            raise ValueError("checkpoint model has no player_code_bytes tensor")
        player_codes = _tensor_bytes(encoded_codes)
        if (
            len(player_codes) != reference_config.player_code_bytes
            or hashlib.sha256(player_codes).hexdigest() != reference_config.player_vocab_sha256
        ):
            raise ValueError("checkpoint model player bytes differ from the validated bundle identity")
        player_identity.decode_player_codes(player_codes)
    config = o50_model.O50Config.from_checkpoint(
        raw_config,
        player_code_bytes=len(player_codes),
        stats_sha256=reference_config.stats_sha256,
    )
    model = o50_model.O50Model(config)
    model.load_state_dict(cast(Mapping[str, Tensor], state), strict=True)
    embedded_codes = model.player_code_bytes.detach().cpu().contiguous().numpy().tobytes()
    if embedded_codes != player_codes:
        raise ValueError("checkpoint model and validated bundle player vocabulary differ")
    model.to(device).eval()
    step = raw["checkpoint_step"]
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("checkpoint_step must be a non-negative integer")
    source_sha256 = raw["source_checkpoint_sha256"]
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
    ):
        raise ValueError("source_checkpoint_sha256 is invalid")
    if metadata.get("checkpoint_step") != step or metadata.get("source_checkpoint_sha256") != source_sha256:
        raise ValueError("model export step or source hash differs from the adjacent verified metadata")
    provenance = {
        "checkpoint_step": step,
        "model_export": str(path),
        "model_export_metadata": str(metadata_path),
        "model_export_metadata_sha256": _sha256_file(metadata_path),
        "model_export_sha256": output_sha256,
        "source_checkpoint_sha256": source_sha256,
        "source_r2_etag": raw["source_r2_etag"],
        "source_selection_sha256": raw["source_selection_sha256"],
        "source_uri": raw.get("source_uri", metadata.get("source_uri")),
        "wandb_id": raw.get("wandb_id", metadata.get("run_id")),
    }
    return model, config, provenance


def _conditional_nll_report(loss: Tensor, targets: Tensor, observed: Tensor) -> dict[str, object]:
    previous = torch.cat((observed[:, None], targets[:, :-1]), dim=1)
    changed = targets != previous

    def summarize(mask: Tensor) -> dict[str, object]:
        counts = mask.sum(dim=0)
        totals = (loss * mask).sum(dim=0)
        means = totals / counts.clamp_min(1)
        serial_means = means.tolist()
        for horizon, group in zip(*torch.where(counts == 0), strict=True):
            serial_means[int(horizon)][int(group)] = None
        return {"count": counts.tolist(), "mean_nll": serial_means}

    return {
        "change": summarize(changed),
        "definition": "change iff the expert target group differs from the previous expert group; horizon 0 uses the observed action",
        "hold": summarize(~changed),
        "unconditional_mean_nll": loss.mean(dim=0).tolist(),
    }


def _change_report(actions: Tensor, targets: Tensor, observed: Tensor) -> dict[str, object]:
    predicted_previous = torch.cat((observed[:, None], actions[:, :-1]), dim=1)
    expert_previous = torch.cat((observed[:, None], targets[:, :-1]), dim=1)
    predicted_change = actions != predicted_previous
    expert_change = targets != expert_previous
    true_positive = (predicted_change & expert_change).sum(dim=0)
    false_positive = (predicted_change & ~expert_change).sum(dim=0)
    false_negative = (~predicted_change & expert_change).sum(dim=0)
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = 2 * true_positive / denominator.clamp_min(1)
    serial_f1 = f1.tolist()
    for horizon, group in zip(*torch.where(denominator == 0), strict=True):
        serial_f1[int(horizon)][int(group)] = None
    return {
        "change_f1": serial_f1,
        "definition": "a predicted switch differs from the model's previous action; expert change uses the previous expert action; horizon 0 shares the observed action",
        "exact_action_rate": (actions == targets).float().mean(dim=0).tolist(),
        "false_negative": false_negative.tolist(),
        "false_positive": false_positive.tolist(),
        "switch_rate": predicted_change.float().mean(dim=0).tolist(),
        "true_positive": true_positive.tolist(),
    }


def _logit_report(logits: Mapping[str, Tensor]) -> dict[str, object]:
    """Summarize centered logits before the trigger-conditioned button mask."""
    report: dict[str, object] = {}
    for name in o50_model.CONTROLLER_GROUP_NAMES:
        values = logits[name].float()
        probabilities = F.softmax(values, dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(dim=-1)
        top_two = values.topk(2, dim=-1).values
        margin = top_two[..., 0] - top_two[..., 1]
        absolute = values.abs().flatten().numpy()
        report[name] = {
            "absolute_logit_quantiles": {
                label: float(value)
                for label, value in zip(
                    ("p50", "p95", "p99", "p99.9", "max"),
                    np.quantile(absolute, (0.5, 0.95, 0.99, 0.999, 1.0)),
                    strict=True,
                )
            },
            "entropy_nats_by_horizon": entropy.mean(dim=0).tolist(),
            "entropy_nats_mean": float(entropy.mean()),
            "logit_rms": float(values.square().mean().sqrt()),
            "top1_margin_by_horizon": margin.mean(dim=0).tolist(),
            "top1_margin_mean": float(margin.mean()),
        }
    return {
        "definition": "teacher-forced class logits after per-group centering and before trigger-conditioned invalid-button masking",
        "groups": report,
    }


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    """Validate assets, run the fixed probe, and write machine-readable results."""
    started = time.monotonic()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    # This public loader validates the complete bundle, including statistics and vocabulary.
    validated_policy = o50.load_o50_policy(args.bundle, device="cpu", seed=SEED, compiled=False)
    del validated_policy
    gc.collect()

    with bundle.extract_policy_bundle(args.bundle) as (manifest, extracted):
        config = o50_model.O50Config.from_json((extracted / manifest.backend_config).read_bytes())
        stats = _finalized_stats(extracted / config.stats_member)
        model = o50_model.O50Model(config).to(device)
        safetensors_torch.load_model(model, extracted / config.weights_member, strict=True, device=str(device))
        model.eval()
        columns, records, data_provenance = load_windows(
            source_root=args.data_root,
            source_manifest=args.source_manifest,
            identity_sidecar=args.identity_sidecar,
            context_length=config.architecture.L_ctx,
            horizon=config.prediction_frames,
            count=args.windows,
            scratch=args.scratch,
        )
        batch = dataloader.train_batch_from_columns(
            columns,
            stats=stats,
            L_ctx=config.architecture.L_ctx,
            extra=features.ITEM_PLAYER_COLUMNS,
            projection=features.ITEM_PLAYER_PROJECTION,
        )
        context_features = {name: value.to(device) for name, value in batch.context.features.items()}
        native_context_actions = features.stack_actions(context_features)
        context_actions = model.codec.quantize(native_context_actions)
        targets = model.codec.quantize(batch.target.to(device))
        observed = context_actions[:, -1]
        full_padding = torch.zeros(args.windows, dtype=torch.long, device=device)
        with torch.inference_mode(), o50_model.amp_context(config, device):
            tokens = model.context_tokens(context_features, context_actions)
            traced_hidden, attention = trace_last_attention(model, tokens, full_padding)
            reference_hidden = model.forward_dense(context_features, full_padding, context_actions)
            baseline_logits = teacher_forced_logits(model, reference_hidden, observed, targets)
            baseline_nll = group_nll(baseline_logits, targets)
            parity_error = float((traced_hidden.float() - reference_hidden.float()).abs().max().item())

            history: dict[str, object] = {}
            for retained in HISTORY_LENGTHS:
                padding = torch.full(
                    (args.windows,),
                    config.architecture.L_ctx - retained,
                    dtype=torch.long,
                    device=device,
                )
                hidden = model.forward_dense(context_features, padding, context_actions)
                treatment = teacher_forced_logits(model, hidden, observed, targets)
                history[str(retained)] = treatment_metrics(baseline_logits, baseline_nll, treatment, targets)

            masked_features = dict(context_features)
            masked_features["ego_player_id"] = torch.zeros_like(masked_features["ego_player_id"])
            masked_hidden = model.forward_dense(masked_features, full_padding, context_actions)
            masked_logits = teacher_forced_logits(model, masked_hidden, observed, targets)

            shuffled_actions = context_actions.roll(1, dims=0)
            shuffled_actions[:, -1] = context_actions[:, -1]
            shuffled_hidden = model.forward_dense(context_features, full_padding, shuffled_actions)
            shuffled_logits = teacher_forced_logits(model, shuffled_hidden, observed, targets)

            head_ablations: list[dict[str, object]] = []
            for layer, raw_block in enumerate(model.trunk.blocks):
                block = cast(trunk.Block, raw_block)
                for head in range(block.attn.n_heads):
                    with zero_attention_head(block, head):
                        ablated_hidden = model.forward_dense(context_features, full_padding, context_actions)
                    ablated_logits = teacher_forced_logits(model, ablated_hidden, observed, targets)
                    head_ablations.append(
                        {
                            "head": head,
                            "layer": layer,
                            **treatment_metrics(baseline_logits, baseline_nll, ablated_logits, targets),
                        }
                    )
            restored_hidden = model.forward_dense(context_features, full_padding, context_actions)
            restored_error = float((restored_hidden.float() - reference_hidden.float()).abs().max().item())

            public_greedy = model.temporal.sample_conditioned(
                reference_hidden,
                observed,
                torch.zeros_like(targets),
                torch.zeros(targets.shape[:2], dtype=torch.bool, device=device),
                torch.full(
                    (config.prediction_frames, o50_model.CONTROLLER_GROUP_COUNT, args.windows),
                    0.5,
                    device=device,
                ),
                argmax=True,
            )
            greedy_logits = teacher_forced_logits(model, reference_hidden, observed, public_greedy)
            reconstructed_greedy = torch.stack(
                [greedy_logits[name].argmax(dim=-1) for name in o50_model.CONTROLLER_GROUP_NAMES], dim=-1
            )
            temporal_parity = bool(torch.equal(public_greedy, reconstructed_greedy))

        if not temporal_parity:
            raise RuntimeError("manual teacher-forced logits disagree with public temporal greedy decode")
        if parity_error > 2e-2:
            raise RuntimeError(f"attention trace changed the trunk result: max error {parity_error}")
        if restored_error != 0.0:
            raise RuntimeError(f"head-ablation hook was not removed cleanly: max error {restored_error}")
        attention_result = attention_metrics(attention)
        distant = np.asarray(attention_result["distant_64_plus_mass"]).reshape(-1)
        head_effect = np.asarray(
            [
                cast(
                    Mapping[str, object],
                    record["replay_paired_horizon_group_mean_nll_delta"],
                )["mean"]
                for record in head_ablations
            ]
        )
        head_js = np.asarray([np.mean(cast(list[float], record["group_js_mean"])) for record in head_ablations])
        records.sort(key=lambda record: int(record["global_row"]))
        for record, feature_row, target_row in zip(
            records,
            ({name: value[index] for name, value in batch.context.features.items()} for index in range(args.windows)),
            batch.target,
            strict=True,
        ):
            record["model_context_sha256"] = _hash_tensors(feature_row)
            record["target_sha256"] = _hash_tensors({"target": target_row})
        result: dict[str, object] = {
            "attention": attention_result,
            "baseline": {
                "group_nll_mean": baseline_nll.mean(dim=0).detach().cpu().tolist(),
                "per_replay_group_nll": _tensor_rows(baseline_nll),
            },
            "bundle": {
                "backend": manifest.backend,
                "backend_config": json.loads(config.to_json()),
                "bundle_sha256": _sha256_file(args.bundle),
                "experiment_id": config.experiment_id,
                "path": str(args.bundle),
                "source_checkpoint_sha256": manifest.source_sha256,
                "stats_content_sha256": config.stats_sha256,
                "weights_sha256": next(
                    member.sha256 for member in manifest.members if member.name == config.weights_member
                ),
            },
            "controls": {
                "cross_replay_old_action_corruption": treatment_metrics(
                    baseline_logits, baseline_nll, shuffled_logits, targets
                ),
                "masked_ego_identity": treatment_metrics(baseline_logits, baseline_nll, masked_logits, targets),
                "manual_temporal_argmax_matches_public": temporal_parity,
                "restored_after_head_ablation_max_abs_error": restored_error,
                "traced_trunk_max_abs_error": parity_error,
            },
            "data": {
                **data_provenance,
                "identity_sidecar": str(args.identity_sidecar),
                "projection": "ITEM_PLAYER_PROJECTION (schema-7 flags are present and intentionally excluded)",
                "seed": SEED,
                "source_manifest": str(args.source_manifest),
                "source_root": str(args.data_root),
                "windows": records,
            },
            "device": {
                "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                "numpy": np.__version__,
                "python": sys.version,
                "torch": torch.__version__,
            },
            "elapsed_seconds": time.monotonic() - started,
            "group_order": list(o50_model.CONTROLLER_GROUP_NAMES),
            "head_ablation": {
                "distant_mass_vs_abs_nll_effect_spearman": _rank_correlation(distant, np.abs(head_effect)),
                "distant_mass_vs_js_spearman": _rank_correlation(distant, head_js),
                "heads": head_ablations,
                "intervention": "zero one head slice before its trunk attention c_proj",
            },
            "history_retained_frames": history,
            "probe": {
                "analysis_base_git_sha": ANALYSIS_BASE_GIT_SHA,
                "sha256": _sha256_file(Path(__file__)),
            },
            "schema_version": 1,
        }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def run_historical_probe(args: argparse.Namespace) -> dict[str, object]:
    """Run the fixed 256-replay diagnostic on one model-only checkpoint."""
    started = time.monotonic()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    validated_policy = o50.load_o50_policy(args.bundle, device="cpu", seed=SEED, compiled=False)
    del validated_policy
    gc.collect()
    if args.checkpoint is None:
        raise ValueError("historical probe requires --checkpoint")

    with bundle.extract_policy_bundle(args.bundle) as (manifest, extracted):
        reference_config = o50_model.O50Config.from_json((extracted / manifest.backend_config).read_bytes())
        stats_path = extracted / reference_config.stats_member
        stats_payload = _string_mapping(json.loads(stats_path.read_bytes()), "validated bundle statistics")
        stats = _finalized_stats(stats_path)
        player_codes = (extracted / reference_config.player_codes_member).read_bytes()
        model, config, checkpoint_provenance = load_checkpoint_model(
            args.checkpoint,
            reference_config=reference_config,
            stats_payload=stats_payload,
            player_codes=player_codes,
            device=device,
        )
        columns, records, data_provenance = load_windows(
            source_root=args.data_root,
            source_manifest=args.source_manifest,
            identity_sidecar=args.identity_sidecar,
            context_length=config.architecture.L_ctx,
            horizon=config.prediction_frames,
            count=args.windows,
            scratch=args.scratch,
        )
        batch = dataloader.train_batch_from_columns(
            columns,
            stats=stats,
            L_ctx=config.architecture.L_ctx,
            extra=features.ITEM_PLAYER_COLUMNS,
            projection=features.ITEM_PLAYER_PROJECTION,
        )
        records.sort(key=lambda record: int(record["global_row"]))
        for record, feature_row, target_row in zip(
            records,
            ({name: value[index] for name, value in batch.context.features.items()} for index in range(args.windows)),
            batch.target,
            strict=True,
        ):
            record["model_context_sha256"] = _hash_tensors(feature_row)
            record["target_sha256"] = _hash_tensors({"target": target_row})

        random_generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
        uniforms = torch.rand(
            config.prediction_frames,
            o50_model.CONTROLLER_GROUP_COUNT,
            args.windows,
            generator=random_generator,
        )
        attention_chunks: list[Tensor] = []
        baseline_nll_chunks: list[Tensor] = []
        teacher_loss_chunks: list[Tensor] = []
        target_chunks: list[Tensor] = []
        observed_chunks: list[Tensor] = []
        greedy_chunks: list[Tensor] = []
        sampled_chunks: list[Tensor] = []
        pre_mask_logit_chunks: dict[str, list[Tensor]] = {name: [] for name in o50_model.CONTROLLER_GROUP_NAMES}
        history_chunks: dict[int, tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]] = {
            retained: ([], [], [], []) for retained in HISTORY_LENGTHS
        }
        trace_max_error = 0.0
        temporal_matches = 0
        temporal_elements = 0
        for batch_start in range(0, args.windows, args.batch_size):
            batch_stop = min(args.windows, batch_start + args.batch_size)
            context_features = {
                name: value[batch_start:batch_stop].to(device) for name, value in batch.context.features.items()
            }
            native_context_actions = features.stack_actions(context_features)
            context_actions = model.codec.quantize(native_context_actions)
            targets = model.codec.quantize(batch.target[batch_start:batch_stop].to(device))
            observed = context_actions[:, -1]
            rows = batch_stop - batch_start
            full_padding = torch.zeros(rows, dtype=torch.long, device=device)
            with torch.inference_mode(), o50_model.amp_context(config, device):
                tokens = model.context_tokens(context_features, context_actions)
                traced_hidden, attention = trace_last_attention(model, tokens, full_padding)
                reference_hidden = model.forward_dense(context_features, full_padding, context_actions)
                baseline_logits = teacher_forced_logits(model, reference_hidden, observed, targets)
                pre_mask_logits = teacher_forced_logits(
                    model,
                    reference_hidden,
                    observed,
                    targets,
                    mask_invalid_buttons=False,
                )
                baseline_nll = group_nll(baseline_logits, targets)
                teacher_loss = horizon_group_nll(baseline_logits, targets)
                trace_max_error = max(
                    trace_max_error,
                    float((traced_hidden.float() - reference_hidden.float()).abs().max().item()),
                )
                for retained in HISTORY_LENGTHS:
                    padding = torch.full(
                        (rows,),
                        config.architecture.L_ctx - retained,
                        dtype=torch.long,
                        device=device,
                    )
                    hidden = model.forward_dense(context_features, padding, context_actions)
                    treatment_logits = teacher_forced_logits(model, hidden, observed, targets)
                    values = _treatment_tensors(baseline_logits, baseline_nll, treatment_logits, targets)
                    for destination, value in zip(history_chunks[retained], values, strict=True):
                        destination.append(value.detach().cpu())

                empty_forced = torch.zeros_like(targets)
                empty_mask = torch.zeros(targets.shape[:2], dtype=torch.bool, device=device)
                greedy = model.temporal.sample_conditioned(
                    reference_hidden,
                    observed,
                    empty_forced,
                    empty_mask,
                    torch.full(
                        (config.prediction_frames, o50_model.CONTROLLER_GROUP_COUNT, rows),
                        0.5,
                        device=device,
                    ),
                    argmax=True,
                )
                sampled = model.temporal.sample_conditioned(
                    reference_hidden,
                    observed,
                    empty_forced,
                    empty_mask,
                    uniforms[:, :, batch_start:batch_stop].to(device),
                    argmax=False,
                )
                greedy_logits = teacher_forced_logits(model, reference_hidden, observed, greedy)
                reconstructed_greedy = torch.stack(
                    [greedy_logits[name].argmax(dim=-1) for name in o50_model.CONTROLLER_GROUP_NAMES], dim=-1
                )
                temporal_matches += int((greedy == reconstructed_greedy).sum().item())
                temporal_elements += greedy.numel()

            attention_chunks.append(attention.cpu())
            baseline_nll_chunks.append(baseline_nll.cpu())
            teacher_loss_chunks.append(teacher_loss.cpu())
            target_chunks.append(targets.cpu())
            observed_chunks.append(observed.cpu())
            greedy_chunks.append(greedy.cpu())
            sampled_chunks.append(sampled.cpu())
            for name in o50_model.CONTROLLER_GROUP_NAMES:
                pre_mask_logit_chunks[name].append(pre_mask_logits[name].float().cpu())

        temporal_match_rate = temporal_matches / temporal_elements
        if temporal_match_rate < 0.99:
            raise RuntimeError(
                "manual teacher-forced logits materially disagree with public temporal greedy decode: "
                f"match rate {temporal_match_rate:.6f}"
            )
        if trace_max_error > 2e-2:
            raise RuntimeError(f"attention trace changed the trunk result: max error {trace_max_error}")
        baseline_nll = torch.cat(baseline_nll_chunks)
        teacher_loss = torch.cat(teacher_loss_chunks)
        targets = torch.cat(target_chunks)
        observed = torch.cat(observed_chunks)
        greedy = torch.cat(greedy_chunks)
        sampled = torch.cat(sampled_chunks)
        pre_mask_logits = {name: torch.cat(chunks) for name, chunks in pre_mask_logit_chunks.items()}
        history = {
            str(retained): _treatment_metrics_from_tensors(*(torch.cat(chunks) for chunks in history_chunks[retained]))
            for retained in HISTORY_LENGTHS
        }
        result: dict[str, object] = {
            "attention": attention_metrics(torch.cat(attention_chunks)),
            "baseline": {
                "group_nll_mean": baseline_nll.mean(dim=0).tolist(),
                "per_replay_group_nll": _tensor_rows(baseline_nll),
                "teacher_forced_nll": _conditional_nll_report(teacher_loss, targets, observed),
            },
            "bundle_reference": {
                "backend": manifest.backend,
                "bundle_sha256": _sha256_file(args.bundle),
                "path": str(args.bundle),
                "stats_content_sha256": reference_config.stats_sha256,
            },
            "checkpoint": {
                **checkpoint_provenance,
                "backend_config": json.loads(config.to_json()),
            },
            "controls": {
                "parallel_teacher_argmax_match_rate_vs_public_cached_decode": temporal_match_rate,
                "temporal_numerical_note": (
                    "Parallel teacher scoring changes BF16 operation order; this action-match fraction is approximate "
                    "parity, not exact logit equivalence."
                ),
                "traced_trunk_max_abs_error": trace_max_error,
            },
            "data": {
                **data_provenance,
                "identity_sidecar": str(args.identity_sidecar),
                "projection": "ITEM_PLAYER_PROJECTION (schema-7 flags are present and intentionally excluded)",
                "seed": SEED,
                "source_manifest": str(args.source_manifest),
                "source_root": str(args.data_root),
                "windows": records,
            },
            "decoding": {
                "greedy": _change_report(greedy, targets, observed),
                "sampled": {
                    **_change_report(sampled, targets, observed),
                    "uniform_seed": SEED + 1,
                    "uniform_table_shape": list(uniforms.shape),
                },
            },
            "device": {
                "batch_size": args.batch_size,
                "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                "numpy": np.__version__,
                "python": sys.version,
                "torch": torch.__version__,
            },
            "elapsed_seconds": time.monotonic() - started,
            "group_order": list(o50_model.CONTROLLER_GROUP_NAMES),
            "history_retained_frames": history,
            "logits": _logit_report(pre_mask_logits),
            "nll_units": {
                "group_nll": "natural-log nats per controller group per predicted frame, then averaged over four frames",
                "mean_joint_bits_per_frame": float(baseline_nll.sum(dim=-1).mean() / math.log(2)),
                "note": "not the training dashboard's logged joint bits unless that dashboard uses the same cohort and frame weighting",
            },
            "probe": {
                "analysis_base_git_sha": ANALYSIS_BASE_GIT_SHA,
                "sha256": _sha256_file(Path(__file__)),
            },
            "schema_version": 2,
        }
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "results.json"
    if output.exists():
        raise FileExistsError(f"historical probe refuses to overwrite {output}")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def prepare_cohort(args: argparse.Namespace) -> dict[str, object]:
    """Prepare the fixed normalized cohort for a remote checkpoint-only run."""
    if args.prepare_cohort is None:
        raise ValueError("cohort preparation requires --prepare-cohort")
    validated_policy = o50.load_o50_policy(args.bundle, device="cpu", seed=SEED, compiled=False)
    del validated_policy
    gc.collect()
    with bundle.extract_policy_bundle(args.bundle) as (manifest, extracted):
        config = o50_model.O50Config.from_json((extracted / manifest.backend_config).read_bytes())
        stats_path = extracted / config.stats_member
        stats_payload = stats_path.read_bytes()
        stats = _finalized_stats(stats_path)
        player_codes = (extracted / config.player_codes_member).read_bytes()
        columns, records, data_provenance = load_windows(
            source_root=args.data_root,
            source_manifest=args.source_manifest,
            identity_sidecar=args.identity_sidecar,
            context_length=config.architecture.L_ctx,
            horizon=config.prediction_frames,
            count=args.windows,
            scratch=args.scratch,
        )
        batch = dataloader.train_batch_from_columns(
            columns,
            stats=stats,
            L_ctx=config.architecture.L_ctx,
            extra=features.ITEM_PLAYER_COLUMNS,
            projection=features.ITEM_PLAYER_PROJECTION,
        )
        records.sort(key=lambda record: int(record["global_row"]))
        for record, feature_row, target_row in zip(
            records,
            ({name: value[index] for name, value in batch.context.features.items()} for index in range(args.windows)),
            batch.target,
            strict=True,
        ):
            record["model_context_sha256"] = _hash_tensors(feature_row)
            record["target_sha256"] = _hash_tensors({"target": target_row})
        package = {
            "bundle_reference_json": json.dumps(
                {
                    "backend": manifest.backend,
                    "bundle_sha256": _sha256_file(args.bundle),
                    "path": str(args.bundle),
                    "stats_content_sha256": config.stats_sha256,
                },
                sort_keys=True,
            ),
            "context_features": batch.context.features,
            "data_json": json.dumps(
                {
                    **data_provenance,
                    "identity_sidecar": str(args.identity_sidecar),
                    "projection": "ITEM_PLAYER_PROJECTION (schema-7 flags are present and intentionally excluded)",
                    "seed": SEED,
                    "source_manifest": str(args.source_manifest),
                    "source_root": str(args.data_root),
                    "windows": records,
                },
                sort_keys=True,
            ),
            "player_codes": _bytes_tensor(player_codes),
            "reference_config": _bytes_tensor(config.to_json()),
            "stats_payload": _bytes_tensor(stats_payload),
            "target": batch.target,
        }
    args.prepare_cohort.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, args.prepare_cohort)
    result = {
        "bytes": args.prepare_cohort.stat().st_size,
        "path": str(args.prepare_cohort),
        "sha256": _sha256_file(args.prepare_cohort),
        "windows": args.windows,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def evaluate_packaged_checkpoint(
    package_path: Path,
    checkpoint_path: Path,
    *,
    device_name: str,
    batch_size: int,
) -> dict[str, object]:
    """Evaluate one checkpoint against a preprocessed, hash-recorded cohort."""
    started = time.monotonic()
    package = _string_mapping(torch.load(package_path, map_location="cpu", weights_only=True), "cohort package")
    expected_package = {
        "bundle_reference_json",
        "context_features",
        "data_json",
        "player_codes",
        "reference_config",
        "stats_payload",
        "target",
    }
    if set(package) != expected_package:
        raise ValueError("cohort package fields differ from the fixed protocol")
    reference_config = o50_model.O50Config.from_json(_tensor_bytes(cast(Tensor, package["reference_config"])))
    stats_payload = _string_mapping(
        json.loads(_tensor_bytes(cast(Tensor, package["stats_payload"]))), "packaged bundle statistics"
    )
    encoded_player_codes = _tensor_bytes(cast(Tensor, package["player_codes"]))
    player_codes = encoded_player_codes or None
    raw_features = _string_mapping(package["context_features"], "packaged context features")
    if any(not isinstance(value, Tensor) for value in raw_features.values()):
        raise TypeError("packaged context features contain a non-tensor value")
    context_features_cpu = cast(Mapping[str, Tensor], raw_features)
    target_cpu = package["target"]
    if not isinstance(target_cpu, Tensor):
        raise TypeError("packaged target is not a tensor")
    window_count = target_cpu.shape[0]
    device = torch.device(device_name)
    model, config, checkpoint_provenance = load_checkpoint_model(
        checkpoint_path,
        reference_config=reference_config,
        stats_payload=stats_payload,
        player_codes=player_codes,
        device=device,
    )
    gc.collect()
    random_generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    uniforms = torch.rand(
        config.prediction_frames,
        o50_model.CONTROLLER_GROUP_COUNT,
        window_count,
        generator=random_generator,
    )
    attention_chunks: list[Tensor] = []
    baseline_nll_chunks: list[Tensor] = []
    teacher_loss_chunks: list[Tensor] = []
    target_chunks: list[Tensor] = []
    observed_chunks: list[Tensor] = []
    greedy_chunks: list[Tensor] = []
    sampled_chunks: list[Tensor] = []
    pre_mask_logit_chunks: dict[str, list[Tensor]] = {name: [] for name in o50_model.CONTROLLER_GROUP_NAMES}
    history_chunks: dict[int, tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]] = {
        retained: ([], [], [], []) for retained in HISTORY_LENGTHS
    }
    trace_max_error = 0.0
    temporal_matches = 0
    temporal_elements = 0
    for batch_start in range(0, window_count, batch_size):
        batch_stop = min(window_count, batch_start + batch_size)
        context_features = {
            name: value[batch_start:batch_stop].to(device) for name, value in context_features_cpu.items()
        }
        native_context_actions = features.stack_actions(context_features)
        context_actions = model.codec.quantize(native_context_actions)
        targets = model.codec.quantize(target_cpu[batch_start:batch_stop].to(device))
        observed = context_actions[:, -1]
        rows = batch_stop - batch_start
        full_padding = torch.zeros(rows, dtype=torch.long, device=device)
        with torch.inference_mode(), o50_model.amp_context(config, device):
            tokens = model.context_tokens(context_features, context_actions)
            traced_hidden, attention = trace_last_attention(model, tokens, full_padding)
            reference_hidden = model.forward_dense(context_features, full_padding, context_actions)
            baseline_logits = teacher_forced_logits(model, reference_hidden, observed, targets)
            pre_mask_logits = teacher_forced_logits(
                model,
                reference_hidden,
                observed,
                targets,
                mask_invalid_buttons=False,
            )
            baseline_nll = group_nll(baseline_logits, targets)
            teacher_loss = horizon_group_nll(baseline_logits, targets)
            trace_max_error = max(
                trace_max_error,
                float((traced_hidden.float() - reference_hidden.float()).abs().max().item()),
            )
            for retained in HISTORY_LENGTHS:
                padding = torch.full(
                    (rows,),
                    config.architecture.L_ctx - retained,
                    dtype=torch.long,
                    device=device,
                )
                hidden = model.forward_dense(context_features, padding, context_actions)
                treatment_logits = teacher_forced_logits(model, hidden, observed, targets)
                values = _treatment_tensors(baseline_logits, baseline_nll, treatment_logits, targets)
                for destination, value in zip(history_chunks[retained], values, strict=True):
                    destination.append(value.detach().cpu())
            empty_forced = torch.zeros_like(targets)
            empty_mask = torch.zeros(targets.shape[:2], dtype=torch.bool, device=device)
            greedy = model.temporal.sample_conditioned(
                reference_hidden,
                observed,
                empty_forced,
                empty_mask,
                torch.full(
                    (config.prediction_frames, o50_model.CONTROLLER_GROUP_COUNT, rows),
                    0.5,
                    device=device,
                ),
                argmax=True,
            )
            sampled = model.temporal.sample_conditioned(
                reference_hidden,
                observed,
                empty_forced,
                empty_mask,
                uniforms[:, :, batch_start:batch_stop].to(device),
                argmax=False,
            )
            greedy_logits = teacher_forced_logits(model, reference_hidden, observed, greedy)
            reconstructed_greedy = torch.stack(
                [greedy_logits[name].argmax(dim=-1) for name in o50_model.CONTROLLER_GROUP_NAMES], dim=-1
            )
            temporal_matches += int((greedy == reconstructed_greedy).sum().item())
            temporal_elements += greedy.numel()
        attention_chunks.append(attention.cpu())
        baseline_nll_chunks.append(baseline_nll.cpu())
        teacher_loss_chunks.append(teacher_loss.cpu())
        target_chunks.append(targets.cpu())
        observed_chunks.append(observed.cpu())
        greedy_chunks.append(greedy.cpu())
        sampled_chunks.append(sampled.cpu())
        for name in o50_model.CONTROLLER_GROUP_NAMES:
            pre_mask_logit_chunks[name].append(pre_mask_logits[name].float().cpu())
    temporal_match_rate = temporal_matches / temporal_elements
    if temporal_match_rate < 0.99:
        raise RuntimeError(
            "manual teacher-forced logits materially disagree with public temporal greedy decode: "
            f"match rate {temporal_match_rate:.6f}"
        )
    if trace_max_error > 2e-2:
        raise RuntimeError(f"attention trace changed the trunk result: max error {trace_max_error}")
    baseline_nll = torch.cat(baseline_nll_chunks)
    teacher_loss = torch.cat(teacher_loss_chunks)
    targets = torch.cat(target_chunks)
    observed = torch.cat(observed_chunks)
    greedy = torch.cat(greedy_chunks)
    sampled = torch.cat(sampled_chunks)
    pre_mask_logits = {name: torch.cat(chunks) for name, chunks in pre_mask_logit_chunks.items()}
    history = {
        str(retained): _treatment_metrics_from_tensors(*(torch.cat(chunks) for chunks in history_chunks[retained]))
        for retained in HISTORY_LENGTHS
    }
    return {
        "attention": attention_metrics(torch.cat(attention_chunks)),
        "baseline": {
            "group_nll_mean": baseline_nll.mean(dim=0).tolist(),
            "per_replay_group_nll": _tensor_rows(baseline_nll),
            "teacher_forced_nll": _conditional_nll_report(teacher_loss, targets, observed),
        },
        "bundle_reference": json.loads(cast(str, package["bundle_reference_json"])),
        "checkpoint": {**checkpoint_provenance, "backend_config": json.loads(config.to_json())},
        "cohort_package": {
            "path": str(package_path),
            "sha256": _sha256_file(package_path),
            "windows": window_count,
        },
        "controls": {
            "parallel_teacher_argmax_match_rate_vs_public_cached_decode": temporal_match_rate,
            "temporal_numerical_note": (
                "Parallel teacher scoring changes BF16 operation order; this action-match fraction is approximate "
                "parity, not exact logit equivalence."
            ),
            "traced_trunk_max_abs_error": trace_max_error,
        },
        "data": json.loads(cast(str, package["data_json"])),
        "decoding": {
            "greedy": _change_report(greedy, targets, observed),
            "sampled": {
                **_change_report(sampled, targets, observed),
                "uniform_seed": SEED + 1,
                "uniform_table_shape": list(uniforms.shape),
            },
        },
        "device": {
            "batch_size": batch_size,
            "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "numpy": np.__version__,
            "python": sys.version,
            "torch": torch.__version__,
        },
        "elapsed_seconds": time.monotonic() - started,
        "group_order": list(o50_model.CONTROLLER_GROUP_NAMES),
        "history_retained_frames": history,
        "logits": _logit_report(pre_mask_logits),
        "nll_units": {
            "group_nll": "natural-log nats per controller group per predicted frame, then averaged over four frames",
            "mean_joint_bits_per_frame": float(baseline_nll.sum(dim=-1).mean() / math.log(2)),
            "note": "not the training dashboard's logged joint bits unless that dashboard uses the same cohort and frame weighting",
        },
        "probe": {"analysis_base_git_sha": ANALYSIS_BASE_GIT_SHA, "sha256": _sha256_file(Path(__file__))},
        "schema_version": 2,
    }


def _remote_cohort(
    *,
    stats_payload: bytes,
    reference_config_payload: bytes,
    bundle_reference: Mapping[str, object],
) -> Path:
    """Download canonical source rows from HAL R2 and build the cohort in-cloud."""
    from hal import r2

    source_root = Path("/tmp/o50-source/mds-v7")
    split_root = source_root / "val"
    split_root.mkdir(parents=True, exist_ok=True)
    client = r2.client()
    bucket_name = r2.bucket()
    key_root = "processed/ranked-anonymized-1/mds-v7"
    client.download_file(bucket_name, f"{key_root}/val/index.json", str(split_root / "index.json"))
    for info in mds.read_shard_index(source_root, "val"):
        zipped = info.get("zip_data")
        if not isinstance(zipped, dict) or not isinstance(zipped.get("basename"), str):
            raise ValueError("remote canonical validation shard has no compressed member")
        basename = zipped["basename"]
        client.download_file(bucket_name, f"{key_root}/val/{basename}", str(split_root / basename))
    source_manifest = source_root / "manifest.jsonl"
    client.download_file(bucket_name, f"{key_root}/manifest.jsonl", str(source_manifest))
    stats_path = Path("/tmp/o50-reference-stats.json")
    stats_path.write_bytes(stats_payload)
    config = o50_model.O50Config.from_json(reference_config_payload)
    stats = _finalized_stats(stats_path)
    columns, records, data_provenance = load_windows(
        source_root=source_root,
        source_manifest=source_manifest,
        identity_sidecar=None,
        context_length=config.architecture.L_ctx,
        horizon=config.prediction_frames,
        count=256,
        scratch=Path("/tmp"),
    )
    batch = dataloader.train_batch_from_columns(
        columns,
        stats=stats,
        L_ctx=config.architecture.L_ctx,
        extra=features.ITEM_PLAYER_COLUMNS,
        projection=features.ITEM_PLAYER_PROJECTION,
    )
    records.sort(key=lambda record: int(record["global_row"]))
    for record, feature_row, target_row in zip(
        records,
        ({name: value[index] for name, value in batch.context.features.items()} for index in range(256)),
        batch.target,
        strict=True,
    ):
        record["model_context_sha256"] = _hash_tensors(feature_row)
        record["target_sha256"] = _hash_tensors({"target": target_row})
    package_path = Path("/tmp/o50-cohort-256.pt")
    torch.save(
        {
            "bundle_reference_json": json.dumps(dict(bundle_reference), sort_keys=True),
            "context_features": batch.context.features,
            "data_json": json.dumps(
                {
                    **data_provenance,
                    "identity_sidecar": None,
                    "projection": "ITEM_PLAYER_PROJECTION (schema-7 flags are present and intentionally excluded)",
                    "seed": SEED,
                    "source_manifest": "HAL R2 canonical ranked-anonymized-1 mds-v7 manifest",
                    "source_root": "s3://hal/processed/ranked-anonymized-1/mds-v7",
                    "windows": records,
                },
                sort_keys=True,
            ),
            "player_codes": torch.empty(0, dtype=torch.uint8),
            "reference_config": _bytes_tensor(reference_config_payload),
            "stats_payload": _bytes_tensor(stats_payload),
            "target": batch.target,
        },
        package_path,
    )
    return package_path


def _remote_six(
    stats_payload: bytes,
    reference_config_payload: bytes,
    bundle_reference: dict[str, object],
) -> list[dict[str, object]]:
    package_path = _remote_cohort(
        stats_payload=stats_payload,
        reference_config_payload=reference_config_payload,
        bundle_reference=bundle_reference,
    )
    outputs = []
    for name in HISTORICAL_MODELS:
        checkpoint = Path("/mechinterp") / f"{name}.model.pt"
        outputs.append(
            evaluate_packaged_checkpoint(
                package_path,
                checkpoint,
                device_name="cuda",
                batch_size=32,
            )
        )
        gc.collect()
        torch.cuda.empty_cache()
    return outputs


def launch_modal_six(args: argparse.Namespace) -> None:
    """Run all historical checkpoints in one H100 function and save returned JSON."""
    import modal

    validated_policy = o50.load_o50_policy(args.bundle, device="cpu", seed=SEED, compiled=False)
    del validated_policy
    gc.collect()
    with bundle.extract_policy_bundle(args.bundle) as (manifest, extracted):
        config = o50_model.O50Config.from_json((extracted / manifest.backend_config).read_bytes())
        stats_payload = (extracted / config.stats_member).read_bytes()
        reference_config_payload = config.to_json()
        bundle_reference = {
            "backend": manifest.backend,
            "bundle_sha256": _sha256_file(args.bundle),
            "path": str(args.bundle),
            "stats_content_sha256": config.stats_sha256,
        }
    local_root = Path(__file__).resolve().parents[2]
    ignore = modal.FilePatternMatcher.from_file(local_root / ".dockerignore")
    image = (
        modal.Image.from_registry(args.modal_image)
        .add_local_dir(local_root, "/opt/hal", copy=True, ignore=ignore)
        .workdir("/opt/hal")
    )
    app = modal.App("hal-o50-historical-probe-20260911")
    volume = modal.Volume.from_name(args.modal_volume, create_if_missing=False)
    secret = modal.Secret.from_name("hal")
    function = app.function(
        image=image,
        secrets=[secret],
        volumes={"/mechinterp": volume},
        gpu="H100",
        cpu=16.0,
        memory=64 * 1024,
        ephemeral_disk=512 * 1024,
        timeout=2 * 60 * 60,
        max_containers=1,
        single_use_containers=True,
        serialized=True,
        include_source=False,
        name="six-checkpoint-probe",
    )(_remote_six)
    with modal.enable_output(), app.run():
        results = function.remote(stats_payload, reference_config_payload, bundle_reference)
    args.output.mkdir(parents=True, exist_ok=True)
    for result in results:
        checkpoint = cast(Mapping[str, object], result["checkpoint"])
        name = Path(cast(str, checkpoint["model_export"])).name.removesuffix(".model.pt")
        output = args.output / name / "results.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"historical probe refuses to overwrite {output}")
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=BUNDLE_PATH)
    parser.add_argument("--data-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--identity-sidecar", type=Path, default=IDENTITY_SIDECAR)
    parser.add_argument("--scratch", type=Path, default=Path("/tmp"))
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--windows", type=int, default=WINDOW_COUNT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cohort-package", type=Path)
    parser.add_argument("--prepare-cohort", type=Path)
    parser.add_argument("--skip-head-ablations", action="store_true")
    parser.add_argument("--modal-six", action="store_true")
    parser.add_argument("--modal-image", default="ghcr.io/ericyuegu/hal:cuda13")
    parser.add_argument("--modal-volume", default="hal-mechinterp-20260911")
    args = parser.parse_args()
    if args.modal_six:
        if (
            args.checkpoint is not None
            or args.cohort_package is not None
            or args.prepare_cohort is not None
            or args.windows != 256
            or not args.skip_head_ablations
            or args.output == OUTPUT_ROOT
        ):
            raise ValueError("--modal-six requires 256 windows, skipped heads, a distinct output, and no local input")
    elif args.prepare_cohort is not None:
        if (
            args.checkpoint is not None
            or args.cohort_package is not None
            or args.windows != 256
            or not args.skip_head_ablations
        ):
            raise ValueError("cohort preparation requires --windows 256 --skip-head-ablations and no checkpoint")
    elif args.checkpoint is None:
        if args.cohort_package is not None or args.windows != WINDOW_COUNT or args.skip_head_ablations:
            raise ValueError("the preserved bundle pilot requires 32 windows and its fixed head screen")
    else:
        if args.windows != 256 or not args.skip_head_ablations:
            raise ValueError("historical probes require --windows 256 --skip-head-ablations")
        if args.output == OUTPUT_ROOT:
            raise ValueError("historical probes require a distinct --output directory to preserve the pilot")
        if not 1 <= args.batch_size <= 64:
            raise ValueError("historical --batch-size must be in [1, 64]")
    return args


def main() -> None:
    args = parse_args()
    if args.modal_six:
        launch_modal_six(args)
        return
    if args.prepare_cohort is not None:
        prepare_cohort(args)
        return
    if args.checkpoint is None:
        result = run_probe(args)
    elif args.cohort_package is None:
        result = run_historical_probe(args)
    else:
        result = evaluate_packaged_checkpoint(
            args.cohort_package,
            args.checkpoint,
            device_name=args.device,
            batch_size=args.batch_size,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        output = args.output / "results.json"
        if output.exists():
            raise FileExistsError(f"historical probe refuses to overwrite {output}")
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"elapsed_seconds": result["elapsed_seconds"], "output": "results.json"}, sort_keys=True))


if __name__ == "__main__":
    main()
