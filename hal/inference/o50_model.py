"""Inference-only O50 model definition for portable policy bundles."""

import contextlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import fields
from pathlib import Path
from typing import Any
from typing import Final
from typing import Literal
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from hal.training import scoring
from hal.training.trunk import Rotary
from hal.training.trunk import Trunk
from hal.training.trunk import TrunkConfig
from hal.training.trunk import apply_rotary_emb

O50_BACKEND: Final[str] = "hal.o50.temporal-awr"
O50_BACKEND_VERSION: Final[int] = 1
O50_CONFIG_SCHEMA_VERSION: Final[int] = 1
O50_EXPERIMENT_IDS: Final[tuple[str, ...]] = (
    "050_scaled_temporal_awr_v4",
    "050_scaled_temporal_awr_v5",
    "050_scaled_temporal_awr_v6",
)
_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")

# This module is a frozen inference model. Keep its tensor geometry independent
# of the emulator-facing ``hal.wire`` and feature modules, which import melee.
_ACTION_DIM: Final[int] = 14
_CONTINUOUS_CHANNEL_COUNT: Final[int] = 6
_TRIGGER_LEFT_CHANNEL: Final[int] = 4
_TRIGGER_RIGHT_CHANNEL: Final[int] = 5
_BUTTON_LEFT_CHANNEL: Final[int] = 12
_BUTTON_RIGHT_CHANNEL: Final[int] = 11
CONTROLLER_GROUP_NAMES: Final[tuple[str, ...]] = ("buttons", "main_stick", "c_stick", "triggers")
CONTROLLER_GROUP_COUNT: Final[int] = len(CONTROLLER_GROUP_NAMES)
_BUTTONS_GROUP, _MAIN_STICK_GROUP, _C_STICK_GROUP, _TRIGGERS_GROUP = range(CONTROLLER_GROUP_COUNT)
CONTROLLER_GROUP_VOCABS: Final[tuple[int, ...]] = (
    scoring.N_BUTTON_COMBOS,
    scoring.STICK_CLUSTER_CENTERS_MAIN.shape[0],
    scoring.STICK_CLUSTER_CENTERS_C.shape[0],
    scoring.TRIGGER_CENTERS.shape[0] ** 2,
)
CONTROLLER_GROUP_INDEX: Final[dict[str, int]] = {name: index for index, name in enumerate(CONTROLLER_GROUP_NAMES)}
CONTROLLER_DECODE_ORDER: Final[tuple[str, ...]] = ("c_stick", "main_stick", "triggers", "buttons")
BASE_PLAYER_PREFIXES: Final[tuple[str, ...]] = ("ego", "ego_nana", "opp_nana", "opp")
FLOAT_FEATURES: Final[tuple[str, ...]] = (
    "position_x",
    "position_y",
    "percent",
    "shield",
    "direction",
    "hitlag_left",
)
CAT_FEATURES: Final[dict[str, tuple[int, int]]] = {
    "action": (512, 64),
    "stock": (5, 2),
    "jumps_used": (9, 2),
    "hurtbox_state": (4, 2),
    "airborne": (2, 1),
}
ITEM_SLOTS: Final[int] = 4
ITEM_FLOATS: Final[tuple[str, ...]] = ("pos_x", "pos_y", "vel_x", "vel_y")
ITEM_CAT_VOCABS: Final[dict[str, int]] = {"type": 256, "state": 256}
ITEM_PRESENCE_SUFFIX: Final[str] = "pos_x"
ITEM_PROBE_COLUMN: Final[str] = "item0_pos_x"
MASKED_PLAYER_ID: Final[int] = 0


def item_column(slot: int, suffix: str) -> str:
    return f"item{slot}_{suffix}"


def sample_categorical(logits: Tensor, *, argmax: bool, uniform: Tensor) -> Tensor:
    values = logits.float()
    if argmax:
        return values.argmax(dim=-1)
    probabilities = F.softmax(values, dim=-1)
    if uniform.shape != probabilities.shape[:-1]:
        raise ValueError(f"uniform shape {tuple(uniform.shape)} != batch shape {tuple(probabilities.shape[:-1])}")
    draw = uniform.to(device=probabilities.device, dtype=probabilities.dtype)
    return (probabilities.cumsum(-1) < draw[..., None]).sum(-1).clamp_max(probabilities.shape[-1] - 1)


def _codec_rmsnorm(tensor: Tensor) -> Tensor:
    return F.rms_norm(tensor, (tensor.shape[-1],), eps=1e-6)


class DiscreteControllerCodec(nn.Module):
    """Checkpoint-compatible controller codec without emulator dependencies."""

    main_centers: Tensor
    c_centers: Tensor
    trigger_centers: Tensor
    button_valid_for_trigger: Tensor

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.class_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(CONTROLLER_GROUP_VOCABS[CONTROLLER_GROUP_INDEX[name]], embed_dim)
                for name in CONTROLLER_GROUP_NAMES
            }
        )
        semantic_dims = {"buttons": 8, "main_stick": 2, "c_stick": 2, "triggers": 2}
        self.semantic_projections = nn.ModuleDict(
            {name: nn.Linear(width, embed_dim, bias=False) for name, width in semantic_dims.items()}
        )
        self.register_buffer("main_centers", scoring.STICK_CLUSTER_CENTERS_MAIN.clone())
        self.register_buffer("c_centers", scoring.STICK_CLUSTER_CENTERS_C.clone())
        self.register_buffer("trigger_centers", scoring.TRIGGER_CENTERS.clone())
        button_bits = scoring.combo_to_buttons(torch.arange(CONTROLLER_GROUP_VOCABS[_BUTTONS_GROUP]))
        trigger_pairs = torch.arange(CONTROLLER_GROUP_VOCABS[_TRIGGERS_GROUP])
        trigger_count = len(self.trigger_centers)
        left_full = trigger_pairs.div(trigger_count, rounding_mode="floor") == trigger_count - 1
        right_full = trigger_pairs.remainder(trigger_count) == trigger_count - 1
        left_click = button_bits[:, _BUTTON_LEFT_CHANNEL - _CONTINUOUS_CHANNEL_COUNT].bool()
        right_click = button_bits[:, _BUTTON_RIGHT_CHANNEL - _CONTINUOUS_CHANNEL_COUNT].bool()
        valid = (~left_click[None, :] | left_full[:, None]) & (~right_click[None, :] | right_full[:, None])
        self.register_buffer("button_valid_for_trigger", valid)

    def _class_embedding(self, name: str) -> nn.Embedding:
        return cast(nn.Embedding, self.class_embeddings[name])

    def _semantic_projection(self, name: str) -> nn.Linear:
        return cast(nn.Linear, self.semantic_projections[name])

    @staticmethod
    def canonicalize(actions: Tensor) -> Tensor:
        if actions.shape[-1] != _ACTION_DIM:
            raise ValueError(f"controller actions must end in {_ACTION_DIM} channels, got {tuple(actions.shape)}")
        output = actions.clone()
        output[..., _TRIGGER_LEFT_CHANNEL] = torch.where(
            output[..., _BUTTON_LEFT_CHANNEL] > 0.5,
            torch.ones_like(output[..., _TRIGGER_LEFT_CHANNEL]),
            output[..., _TRIGGER_LEFT_CHANNEL],
        )
        output[..., _TRIGGER_RIGHT_CHANNEL] = torch.where(
            output[..., _BUTTON_RIGHT_CHANNEL] > 0.5,
            torch.ones_like(output[..., _TRIGGER_RIGHT_CHANNEL]),
            output[..., _TRIGGER_RIGHT_CHANNEL],
        )
        return output

    def quantize(self, actions: Tensor) -> Tensor:
        actions = self.canonicalize(actions)
        continuous = actions[..., :_CONTINUOUS_CHANNEL_COUNT]
        buttons = scoring.buttons_to_combo(actions[..., _CONTINUOUS_CHANNEL_COUNT:])
        main = scoring.nearest_cluster(continuous[..., 0:2], self.main_centers)
        c_stick = scoring.nearest_cluster(continuous[..., 2:4], self.c_centers)
        trigger_pair = scoring.nearest_center(continuous[..., 4:6], self.trigger_centers)
        triggers = trigger_pair[..., 0] * self.trigger_centers.shape[0] + trigger_pair[..., 1]
        return torch.stack((buttons, main, c_stick, triggers), dim=-1)

    def dequantize(self, indices: Tensor) -> Tensor:
        trigger_count = self.trigger_centers.shape[0]
        buttons = scoring.combo_to_buttons(indices[..., _BUTTONS_GROUP])
        main = scoring.cluster_to_xy(indices[..., _MAIN_STICK_GROUP], self.main_centers)
        c_stick = scoring.cluster_to_xy(indices[..., _C_STICK_GROUP], self.c_centers)
        trigger_left = scoring.center_to_value(indices[..., _TRIGGERS_GROUP] // trigger_count, self.trigger_centers)
        trigger_right = scoring.center_to_value(indices[..., _TRIGGERS_GROUP] % trigger_count, self.trigger_centers)
        return torch.cat((main, c_stick, torch.stack((trigger_left, trigger_right), dim=-1), buttons), dim=-1)

    def semantic_values(self, name: str, indices: Tensor) -> Tensor:
        if name == "buttons":
            return scoring.combo_to_buttons(indices).to(self._class_embedding(name).weight.dtype)
        if name == "main_stick":
            return scoring.cluster_to_xy(indices, self.main_centers)
        if name == "c_stick":
            return scoring.cluster_to_xy(indices, self.c_centers)
        if name == "triggers":
            trigger_count = self.trigger_centers.shape[0]
            return torch.stack(
                (
                    scoring.center_to_value(indices // trigger_count, self.trigger_centers),
                    scoring.center_to_value(indices % trigger_count, self.trigger_centers),
                ),
                dim=-1,
            )
        raise ValueError(f"unknown controller group {name!r}")

    def group_embedding(self, name: str, indices: Tensor) -> Tensor:
        class_embedding = self._class_embedding(name)
        semantic = self.semantic_values(name, indices).to(class_embedding.weight.dtype)
        return _codec_rmsnorm(class_embedding(indices) + self._semantic_projection(name)(semantic))

    def embed_groups(self, indices: Tensor) -> dict[str, Tensor]:
        return {
            name: self.group_embedding(name, indices[..., CONTROLLER_GROUP_INDEX[name]])
            for name in CONTROLLER_GROUP_NAMES
        }

    def embed_frame(self, indices: Tensor, embedded: dict[str, Tensor] | None = None) -> Tensor:
        values = self.embed_groups(indices) if embedded is None else embedded
        return torch.cat([values[name] for name in CONTROLLER_GROUP_NAMES], dim=-1)

    def button_mask(self, trigger_indices: Tensor) -> Tensor:
        return ~self.button_valid_for_trigger[trigger_indices]


def _integer(mapping: Mapping[str, object], name: str, *, minimum: int = 1) -> int:
    value = mapping.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"O50 {name} must be an integer >= {minimum}, got {value!r}")
    return value


def _string(mapping: Mapping[str, object], name: str) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"O50 {name} must be a non-empty string")
    return value


def _sha256(mapping: Mapping[str, object], name: str) -> str:
    value = _string(mapping, name)
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"O50 {name} must be a lowercase SHA-256")
    return value


def _member(mapping: Mapping[str, object], name: str) -> str:
    value = _string(mapping, name)
    if Path(value).name != value:
        raise ValueError(f"O50 {name} must be a plain filename")
    return value


@dataclass(frozen=True, slots=True)
class O50Architecture:
    """Persisted O50 model geometry used during inference."""

    d_model: int
    n_layers: int
    n_heads: int
    attn_window: int
    L_ctx: int
    sample_chunk_length: int
    head_offsets: tuple[int, ...]
    temporal_d_model: int
    temporal_layers: int
    temporal_heads: int
    temporal_ff_dim: int
    group_head_dim: int
    action_embed_dim: int
    offset_embed_dim: int
    action_vocab: int
    action_state_embed_dim: int
    char_vocab: int
    char_dim: int
    stage_vocab: int
    stage_dim: int
    item_type_dim: int
    item_state_dim: int
    item_hidden_dim: int
    item_dim: int
    value_hidden_dim: int

    @classmethod
    def from_mapping(cls, raw: object) -> O50Architecture:
        if not isinstance(raw, Mapping):
            raise ValueError("O50 architecture must be an object")
        values = cast(Mapping[str, object], raw)
        expected = {item.name for item in fields(cls)}
        if set(values) != expected:
            raise ValueError(
                "O50 architecture fields differ: "
                f"missing={sorted(expected - values.keys())}, unexpected={sorted(values.keys() - expected)}"
            )
        parsed: dict[str, object] = {}
        for name in expected - {"head_offsets", "attn_window"}:
            parsed[name] = _integer(values, name)
        parsed["attn_window"] = _integer(values, "attn_window", minimum=0)
        offsets = values.get("head_offsets")
        if not isinstance(offsets, (list, tuple)) or any(
            not isinstance(value, int) or isinstance(value, bool) for value in offsets
        ):
            raise ValueError("O50 head_offsets must be a sequence of integers")
        parsed["head_offsets"] = tuple(offsets)
        return cls(**cast(Any, parsed))

    def __post_init__(self) -> None:
        positive = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name not in {"attn_window", "head_offsets"}
        }
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in positive.values()):
            raise ValueError(f"O50 architecture needs positive integer dimensions, got {positive}")
        if not isinstance(self.attn_window, int) or isinstance(self.attn_window, bool) or self.attn_window < 0:
            raise ValueError("O50 attn_window must be a non-negative integer")
        if self.d_model % self.n_heads or self.temporal_d_model % self.temporal_heads:
            raise ValueError("O50 model dimensions must be divisible by their head counts")
        if (self.temporal_d_model // self.temporal_heads) % 2:
            raise ValueError("O50 temporal attention head width must be even")
        if tuple(sorted(set(self.head_offsets))) != self.head_offsets or not self.head_offsets:
            raise ValueError("O50 head offsets must be sorted and unique")
        if self.head_offsets[:4] != (1, 2, 3, 4):
            raise ValueError("O50 deployment requires dense +1,+2,+3,+4 heads")
        if self.head_offsets[-1] > self.sample_chunk_length:
            raise ValueError("O50 head offsets extend past the sampled action chunk")


@dataclass(frozen=True, slots=True)
class O50Config:
    """Typed portable subset of the O50 training configuration."""

    experiment_id: str
    architecture: O50Architecture
    prediction_frames: int
    player_vocab_size: int
    player_vocab_sha256: str
    player_code_bytes: int
    amp_dtype: Literal["bfloat16", "float32"]
    depth_alpha: float
    mds_schema_version: int
    stats_sha256: str
    weights_member: str = "weights.safetensors"
    stats_member: str = "stats.json"
    player_codes_member: str = "players.json"
    schema_version: int = O50_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != O50_CONFIG_SCHEMA_VERSION:
            raise ValueError(f"unsupported O50 bundle config schema {self.schema_version}")
        if self.experiment_id not in O50_EXPERIMENT_IDS:
            raise ValueError(f"unsupported O50 checkpoint experiment {self.experiment_id!r}")
        if self.prediction_frames != 4:
            raise ValueError(f"O50 live inference requires four prediction frames, got {self.prediction_frames}")
        if self.player_vocab_size < 1:
            raise ValueError("O50 player vocabulary must be non-empty")
        if self.player_code_bytes < 1:
            raise ValueError("O50 bundle must contain its encoded player vocabulary")
        if _SHA256_PATTERN.fullmatch(self.player_vocab_sha256) is None:
            raise ValueError("O50 player vocabulary SHA-256 is invalid")
        if self.depth_alpha != 0.5:
            raise ValueError("O50 inference requires the recorded depth_alpha=0.5")
        if self.amp_dtype not in ("bfloat16", "float32"):
            raise ValueError(f"unsupported O50 amp dtype {self.amp_dtype!r}")
        if self.mds_schema_version != 7:
            raise ValueError(f"O50 needs MDS schema 7 statistics, got {self.mds_schema_version}")
        if _SHA256_PATTERN.fullmatch(self.stats_sha256) is None:
            raise ValueError("O50 statistics SHA-256 is invalid")
        members = (self.weights_member, self.stats_member, self.player_codes_member)
        if any(not member or Path(member).name != member for member in members):
            raise ValueError("O50 asset members must be plain filenames")
        if len(set(members)) != len(members):
            raise ValueError("O50 asset member names must be distinct")

    @classmethod
    def from_checkpoint(
        cls,
        raw: object,
        *,
        player_code_bytes: int,
        stats_sha256: str,
    ) -> O50Config:
        """Read only the inference contract from a trusted raw checkpoint."""
        if not isinstance(raw, Mapping):
            raise ValueError("O50 checkpoint config must be an object")
        values = cast(Mapping[str, object], raw)
        architecture = O50Architecture.from_mapping(values.get("architecture"))
        required = (
            "experiment_id",
            "prediction_frames",
            "player_vocab_size",
            "player_vocab_sha256",
            "amp_dtype",
            "depth_alpha",
            "mds_schema_version",
        )
        missing = [name for name in required if name not in values]
        if missing:
            raise ValueError(f"O50 checkpoint config is missing {missing}")
        amp_dtype = _string(values, "amp_dtype")
        if amp_dtype not in ("bfloat16", "float32"):
            raise ValueError(f"unsupported O50 amp dtype {amp_dtype!r}")
        depth_alpha = values.get("depth_alpha")
        if (
            not isinstance(depth_alpha, (int, float))
            or isinstance(depth_alpha, bool)
            or not math.isfinite(depth_alpha)
        ):
            raise ValueError(f"O50 depth_alpha must be finite, got {depth_alpha!r}")
        return cls(
            experiment_id=_string(values, "experiment_id"),
            architecture=architecture,
            prediction_frames=_integer(values, "prediction_frames"),
            player_vocab_size=_integer(values, "player_vocab_size"),
            player_vocab_sha256=_sha256(values, "player_vocab_sha256"),
            player_code_bytes=player_code_bytes,
            amp_dtype=cast(Literal["bfloat16", "float32"], amp_dtype),
            depth_alpha=float(depth_alpha),
            mds_schema_version=_integer(values, "mds_schema_version"),
            stats_sha256=stats_sha256,
        )

    def to_json(self) -> bytes:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()

    @classmethod
    def from_json(cls, encoded: bytes) -> O50Config:
        try:
            raw = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("O50 backend config is not valid JSON") from error
        if not isinstance(raw, Mapping):
            raise ValueError("O50 backend config must be an object")
        values = cast(Mapping[str, object], raw)
        expected = {item.name for item in fields(cls)}
        if set(values) != expected:
            raise ValueError(
                f"O50 backend config fields differ: missing={sorted(expected - values.keys())}, "
                f"unexpected={sorted(values.keys() - expected)}"
            )
        amp_dtype = _string(values, "amp_dtype")
        if amp_dtype not in ("bfloat16", "float32"):
            raise ValueError(f"unsupported O50 amp dtype {amp_dtype!r}")
        depth_alpha = values.get("depth_alpha")
        if (
            not isinstance(depth_alpha, (int, float))
            or isinstance(depth_alpha, bool)
            or not math.isfinite(depth_alpha)
        ):
            raise ValueError(f"O50 depth_alpha must be finite, got {depth_alpha!r}")
        return cls(
            experiment_id=_string(values, "experiment_id"),
            architecture=O50Architecture.from_mapping(values.get("architecture")),
            prediction_frames=_integer(values, "prediction_frames"),
            player_vocab_size=_integer(values, "player_vocab_size"),
            player_vocab_sha256=_sha256(values, "player_vocab_sha256"),
            player_code_bytes=_integer(values, "player_code_bytes"),
            amp_dtype=cast(Literal["bfloat16", "float32"], amp_dtype),
            depth_alpha=float(depth_alpha),
            mds_schema_version=_integer(values, "mds_schema_version"),
            stats_sha256=_sha256(values, "stats_sha256"),
            weights_member=_member(values, "weights_member"),
            stats_member=_member(values, "stats_member"),
            player_codes_member=_member(values, "player_codes_member"),
            schema_version=_integer(values, "schema_version"),
        )


def _decoder_rmsnorm(tensor: Tensor) -> Tensor:
    return F.rms_norm(tensor, (tensor.shape[-1],), eps=1e-6)


def _action_rmsnorm(tensor: Tensor) -> Tensor:
    return F.rms_norm(tensor, (tensor.shape[-1],), eps=1e-5)


def _center_class_logits(logits: Tensor) -> Tensor:
    return logits - logits.mean(dim=-1, keepdim=True)


class _SwiGLU(nn.Module):
    def __init__(self, input_width: int, hidden_width: int, output_width: int, *, output_bias: bool = False) -> None:
        super().__init__()
        self.up = nn.Linear(input_width, 2 * hidden_width, bias=False)
        self.down = nn.Linear(hidden_width, output_width, bias=output_bias)

    def forward(self, tensor: Tensor) -> Tensor:
        gate, value = self.up(tensor).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class _NonlinearActionHead(nn.Module):
    def __init__(self, model_width: int, hidden_width: int, vocabulary_size: int) -> None:
        super().__init__()
        self.up = nn.Linear(model_width, hidden_width, bias=False)
        self.down = nn.Linear(hidden_width, vocabulary_size)

    def forward(self, tensor: Tensor) -> Tensor:
        return self.down(F.silu(self.up(_decoder_rmsnorm(tensor))))


@dataclass(frozen=True, slots=True)
class _DepthRule:
    attention: float
    mlp: float


def _depth_rule(stack: Literal["trunk", "temporal"], layers: int, alpha: float) -> _DepthRule:
    if alpha != 0.5:
        raise ValueError("O50 fixes depth_alpha to 0.5")
    if stack == "trunk":
        base_layers, base_attention = 8, 0.25
    elif stack == "temporal":
        base_layers, base_attention = 2, 0.5
    else:
        raise ValueError(f"unknown O50 stack {stack!r}")
    branch = (layers / base_layers) ** -alpha
    return _DepthRule(attention=base_attention * branch, mlp=branch)


class _TemporalBlock(nn.Module):
    def __init__(self, config: O50Config) -> None:
        super().__init__()
        architecture = config.architecture
        self.n_heads = architecture.temporal_heads
        self.d_model = architecture.temporal_d_model
        self.head_dim = self.d_model // self.n_heads
        rule = _depth_rule("temporal", architecture.temporal_layers, config.depth_alpha)
        self.scale = rule.attention
        self.mlp_scale = rule.mlp
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.rotary = Rotary(self.head_dim)
        self.up = nn.Linear(self.d_model, architecture.temporal_ff_dim, bias=False)
        self.down = nn.Linear(architecture.temporal_ff_dim, self.d_model, bias=False)

    def _qkv(self, tensor: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = tensor.shape
        query, key, value = self.qkv(_decoder_rmsnorm(tensor)).split(self.d_model, dim=-1)
        shape = (batch, length, self.n_heads, self.head_dim)
        return query.view(shape), key.view(shape), value.view(shape)

    def forward_step(self, tensor: Tensor, past: tuple[Tensor, Tensor] | None) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        query, key, value = self._qkv(tensor[:, None])
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if past is not None:
            key = torch.cat((past[0], key), dim=2)
            value = torch.cat((past[1], value), dim=2)
        cosine, sine = self.rotary.at(key.shape[2], tensor.device)
        query = apply_rotary_emb(query, cosine[:, -1:], sine[:, -1:]).transpose(1, 2)
        rotated_key = apply_rotary_emb(key.transpose(1, 2), cosine, sine).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, rotated_key, value)
        attended = attended.transpose(1, 2).contiguous().view_as(tensor)
        tensor = tensor + self.scale * self.proj(attended)
        return (
            tensor + self.mlp_scale * self.down(F.silu(self.up(_decoder_rmsnorm(tensor)))),
            (key, value),
        )


class _CausalTemporalDecoder(nn.Module):
    def __init__(self, config: O50Config, codec: DiscreteControllerCodec) -> None:
        super().__init__()
        architecture = config.architecture
        self.codec = codec
        self.head_offsets = tuple(architecture.head_offsets)
        self.d_model = architecture.temporal_d_model
        controller_width = CONTROLLER_GROUP_COUNT * architecture.action_embed_dim
        self.offset_embedding = nn.Embedding(architecture.sample_chunk_length + 1, architecture.offset_embed_dim)
        self.token_projection = nn.Linear(
            architecture.d_model + controller_width + architecture.offset_embed_dim,
            self.d_model,
        )
        self.blocks = nn.ModuleList([_TemporalBlock(config) for _ in range(architecture.temporal_layers)])
        self.group_condition = nn.ModuleDict(
            {
                name: nn.Linear(position * architecture.action_embed_dim, 2 * self.d_model)
                for position, name in enumerate(CONTROLLER_DECODE_ORDER)
                if position
            }
        )
        self.outputs = nn.ModuleDict(
            {
                name: _NonlinearActionHead(
                    self.d_model,
                    architecture.group_head_dim,
                    CONTROLLER_GROUP_VOCABS[CONTROLLER_GROUP_INDEX[name]],
                )
                for name in CONTROLLER_GROUP_NAMES
            }
        )
        self.trunk_outputs = nn.ModuleDict(
            {
                name: nn.Linear(
                    architecture.d_model,
                    CONTROLLER_GROUP_VOCABS[CONTROLLER_GROUP_INDEX[name]],
                    bias=False,
                )
                for name in CONTROLLER_GROUP_NAMES
            }
        )
        self.trunk_width = architecture.d_model
        self.controller_width = controller_width

    def _state_bias(self, trunk: Tensor) -> Tensor:
        weight = self.token_projection.weight
        return F.linear(trunk, weight[:, : self.trunk_width], self.token_projection.bias)

    def _step_features(self, previous: Tensor, offsets: Tensor) -> Tensor:
        weight = self.token_projection.weight
        action_stop = self.trunk_width + self.controller_width
        action = F.linear(self.codec.embed_frame(previous), weight[:, self.trunk_width : action_stop])
        return action + F.linear(self.offset_embedding(offsets), weight[:, action_stop:])

    def _decode_step(
        self,
        previous: Tensor,
        offset: int,
        state_bias: Tensor,
        caches: list[tuple[Tensor, Tensor] | None],
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor] | None]]:
        offsets = torch.full((previous.shape[0],), offset, device=previous.device, dtype=torch.long)
        state = _decoder_rmsnorm(state_bias + self._step_features(previous, offsets))
        next_caches = []
        for module, past in zip(self.blocks, caches, strict=True):
            block = cast(_TemporalBlock, module)
            state, present = block.forward_step(state, past)
            next_caches.append(present)
        return _decoder_rmsnorm(state), next_caches

    def _group_features(self, state: Tensor, name: str, embedded: dict[str, Tensor]) -> Tensor:
        position = CONTROLLER_DECODE_ORDER.index(name)
        if not position:
            return state
        prefix = torch.cat([embedded[group] for group in CONTROLLER_DECODE_ORDER[:position]], dim=-1)
        raw_scale, shift = self.group_condition[name](prefix).chunk(2, dim=-1)
        return state * (1.0 + torch.tanh(raw_scale)) + shift

    def sample_conditioned(
        self,
        hidden: Tensor,
        observed: Tensor,
        forced: Tensor,
        uniforms: Tensor,
        *,
        argmax: bool = False,
    ) -> Tensor:
        """Force the committed prefix and sample only the uncommitted tail."""
        delay = forced.shape[1]
        horizon = 4
        if forced.shape != (hidden.shape[0], delay, CONTROLLER_GROUP_COUNT):
            raise ValueError("O50 forced actions have the wrong shape")
        if not 0 <= delay < horizon:
            raise ValueError(f"O50 transport delay must be in [0, {horizon - 1}], got {delay}")
        if uniforms.shape != (horizon - delay, CONTROLLER_GROUP_COUNT, hidden.shape[0]):
            raise ValueError("O50 uniform table must cover only sampled tail actions")
        raw_trunk = hidden[:, -1]
        state_bias = self._state_bias(_decoder_rmsnorm(raw_trunk))
        previous = observed
        caches: list[tuple[Tensor, Tensor] | None] = [None] * len(self.blocks)
        sampled = []
        for depth, offset in enumerate(self.head_offsets[:horizon]):
            state, caches = self._decode_step(previous, offset, state_bias, caches)
            if depth < delay:
                previous = forced[:, depth]
                continue
            embedded: dict[str, Tensor] = {}
            picks: dict[str, Tensor] = {}
            for name in CONTROLLER_DECODE_ORDER:
                logits = _center_class_logits(
                    self.outputs[name](self._group_features(state, name, embedded))
                    + self.trunk_outputs[name](raw_trunk)
                )
                if name == "buttons":
                    logits = logits.masked_fill(self.codec.button_mask(picks["triggers"]), float("-inf"))
                group = CONTROLLER_GROUP_INDEX[name]
                pick = sample_categorical(
                    logits,
                    argmax=argmax,
                    uniform=uniforms[depth - delay, group],
                )
                picks[name] = pick
                embedded[name] = self.codec.group_embedding(name, pick)
            previous = torch.stack([picks[name] for name in CONTROLLER_GROUP_NAMES], dim=-1)
            sampled.append(previous)
        return torch.stack(sampled, dim=1)


class O50Model(nn.Module):
    """O50 module tree with checkpoint-compatible parameter names."""

    player_code_bytes: Tensor

    def __init__(self, config: O50Config) -> None:
        super().__init__()
        architecture = config.architecture
        self.config = config
        self.codec = DiscreteControllerCodec(architecture.action_embed_dim)
        self.cat_specs = {**CAT_FEATURES, "action": (architecture.action_vocab, architecture.action_state_embed_dim)}
        self.cat_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocabulary, width) for name, (vocabulary, width) in self.cat_specs.items()}
        )
        self.char_emb = nn.Embedding(architecture.char_vocab, architecture.char_dim)
        self.stage_emb = nn.Embedding(architecture.stage_vocab, architecture.stage_dim)
        per_player = len(FLOAT_FEATURES) * 2 + sum(width for _, width in self.cat_specs.values())
        input_width = (
            len(BASE_PLAYER_PREFIXES) * per_player
            + CONTROLLER_GROUP_COUNT * architecture.action_embed_dim
            + 2 * architecture.char_dim
            + architecture.stage_dim
        )
        self.item_type_emb = nn.Embedding(ITEM_CAT_VOCABS["type"], architecture.item_type_dim)
        self.item_state_emb = nn.Embedding(ITEM_CAT_VOCABS["state"], architecture.item_state_dim)
        item_width = architecture.item_type_dim + architecture.item_state_dim + 2 * len(ITEM_FLOATS) + 1
        self.item_encoder = _SwiGLU(item_width, architecture.item_hidden_dim, architecture.item_dim)
        input_width += architecture.item_dim
        self.observation_encoder = nn.Linear(input_width, architecture.d_model)
        self.player_embedding = nn.Embedding(
            config.player_vocab_size,
            32,
            padding_idx=MASKED_PLAYER_ID,
        )
        self.player_projection = nn.Linear(32, architecture.d_model, bias=False)
        self.register_buffer("player_code_bytes", torch.empty(config.player_code_bytes, dtype=torch.uint8))
        trunk_rule = _depth_rule("trunk", architecture.n_layers, config.depth_alpha)
        self.trunk = Trunk(
            TrunkConfig(
                d_model=architecture.d_model,
                n_layers=architecture.n_layers,
                n_heads=architecture.n_heads,
                L_ctx=architecture.L_ctx,
                attn_window=architecture.attn_window,
                attention_backend="varlen_flash",
                attention_scale=trunk_rule.attention,
                mlp_scale=trunk_rule.mlp,
            )
        )
        self.temporal = _CausalTemporalDecoder(config, self.codec)
        self.value_head = _SwiGLU(
            architecture.d_model,
            architecture.value_hidden_dim,
            1,
            output_bias=True,
        )

    def _per_player_features(self, features: dict[str, Tensor], prefix: str) -> Tensor:
        reference = features[f"{prefix}_position_x"]
        values = [features[f"{prefix}_{name}"][..., None] for name in FLOAT_FEATURES]
        masks = [
            features.get(f"{prefix}_{name}_mask", torch.zeros_like(reference))[..., None] for name in FLOAT_FEATURES
        ]
        parts = [*values, *masks]
        for name, (vocabulary, _) in self.cat_specs.items():
            parts.append(self.cat_embeds[name](features[f"{prefix}_{name}"].clamp(0, vocabulary - 1)))
        return torch.cat(parts, dim=-1)

    def _item_features(self, features: dict[str, Tensor]) -> Tensor:
        if ITEM_PROBE_COLUMN not in features:
            raise ValueError(f"O50 observation has no {ITEM_PROBE_COLUMN!r} projectile column")
        zeros = torch.zeros_like(features[ITEM_PROBE_COLUMN])
        slots = []
        presence = []
        for slot in range(ITEM_SLOTS):
            type_ids = features[item_column(slot, "type")].clamp(0, self.item_type_emb.num_embeddings - 1)
            state_ids = features[item_column(slot, "state")].clamp(0, self.item_state_emb.num_embeddings - 1)
            masks = {name: features.get(f"{item_column(slot, name)}_mask", zeros) for name in ITEM_FLOATS}
            live = 1.0 - masks[ITEM_PRESENCE_SUFFIX]
            parts = [self.item_type_emb(type_ids), self.item_state_emb(state_ids)]
            parts.extend(features[item_column(slot, name)][..., None] for name in ITEM_FLOATS)
            parts.extend(masks[name][..., None] for name in ITEM_FLOATS)
            parts.append(live[..., None])
            slots.append(torch.cat(parts, dim=-1))
            presence.append(live)
        encoded = self.item_encoder(torch.stack(slots, dim=-2))
        return (encoded * torch.stack(presence, dim=-1)[..., None]).sum(dim=-2)

    def context_tokens(self, features: dict[str, Tensor], action_indices: Tensor) -> Tensor:
        if "opp_player_id" in features:
            raise ValueError("opponent identity must not enter O50")
        if "ego_player_id" not in features:
            raise KeyError("O50 context is missing ego_player_id")
        parts = [self._per_player_features(features, prefix) for prefix in BASE_PLAYER_PREFIXES]
        parts.append(self.codec.embed_frame(action_indices))
        parts.append(self.char_emb(features["ego_character"].clamp(0, self.char_emb.num_embeddings - 1)))
        parts.append(self.char_emb(features["opp_character"].clamp(0, self.char_emb.num_embeddings - 1)))
        parts.append(self.stage_emb(features["stage"].clamp(0, self.stage_emb.num_embeddings - 1)))
        parts.append(self._item_features(features))
        observation = self.observation_encoder(torch.cat(parts, dim=-1))
        player_ids = features["ego_player_id"].clamp(0, self.player_embedding.num_embeddings - 1)
        return observation + self.player_projection(self.player_embedding(player_ids))

    def forward_dense(self, features: dict[str, Tensor], context_padding: Tensor, action_indices: Tensor) -> Tensor:
        return self.trunk.forward_dense(self.context_tokens(features, action_indices), context_padding)


def amp_context(config: O50Config, device: torch.device | str) -> contextlib.AbstractContextManager[object]:
    if config.amp_dtype == "bfloat16" and torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()
