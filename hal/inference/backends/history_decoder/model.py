"""O59 v5 inference model, copied from the frozen experiment.

The simulator and training loop are intentionally absent.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Annotated
from typing import ClassVar
from typing import Final
from typing import Literal
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention

from hal import streams
from hal.data.policy_world_schema import POLICY_WORLD_SCHEMA_VERSION
from hal.eval.policy_sampling import sample_categorical
from hal.eval.policy_sampling import validate_sampling_temperature
from hal.inference.backends.history_decoder.kv_cache import KVMemory
from hal.inference.backends.history_decoder.kv_cache import rotate_positions
from hal.training.controller_codec import BUTTONS_GROUP
from hal.training.controller_codec import CONTROLLER_DECODE_ORDER
from hal.training.controller_codec import CONTROLLER_GROUP_COUNT
from hal.training.controller_codec import CONTROLLER_GROUP_INDEX
from hal.training.controller_codec import CONTROLLER_GROUP_NAMES
from hal.training.controller_codec import CONTROLLER_GROUP_VOCABS
from hal.training.controller_codec import TRIGGERS_GROUP
from hal.training.controller_codec import DiscreteControllerCodec
from hal.training.features import BASE_PLAYER_PREFIXES
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import ITEM_CAT_VOCABS
from hal.training.features import ITEM_FLOATS
from hal.training.features import ITEM_PRESENCE_SUFFIX
from hal.training.features import ITEM_PROBE_COLUMN
from hal.training.features import stack_actions
from hal.training.player_identity import MASKED_PLAYER_ID
from hal.training.player_identity import PlayerVocabulary
from hal.training.player_identity import vocabulary_buffer
from hal.training.trunk import Rotary
from hal.training.trunk import Trunk
from hal.training.trunk import TrunkConfig
from hal.training.trunk import apply_rotary_emb
from hal.wire import ITEM_SLOTS
from hal.wire import item_column

RETURN_SCALE: Final[float] = 120.0
CALIBRATION_WINDOWS: Final[int] = 65_536


def sample_with_temperature(logits: Tensor, uniform: Tensor, temperature: Tensor) -> Tensor:
    """Use a runtime scalar without specializing the compiled decoder."""
    probabilities = F.softmax(logits.float() / temperature, dim=-1)
    return (probabilities.cumsum(-1) < uniform[..., None]).sum(-1).clamp_max(probabilities.shape[-1] - 1)


class ReturnCalibration:
    def __init__(self) -> None:
        self.values: list[float] = []
        self.valid: list[bool] = []
        self.replay_ids: list[str] = []

    def targets(self) -> tuple[float, float, float]:
        if len(self.values) != CALIBRATION_WINDOWS:
            raise ValueError("return calibration is incomplete")
        values = np.asarray(self.values)
        positive = values[np.asarray(self.valid) & (values > 0)]
        if not positive.size:
            raise ValueError("calibration has no positive valid returns")
        median, p90 = np.quantile(positive, [0.5, 0.9], method="linear")
        return 0.0, float(median), float(p90)

    def state_dict(self) -> dict[str, object]:
        data = {"values": self.values.copy(), "valid": self.valid.copy(), "replay_ids": self.replay_ids.copy()}
        identity = hashlib.sha256(json.dumps(data, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        return {
            "version": 1,
            **data,
            "sha256": identity,
            "targets": self.targets() if len(self.values) == CALIBRATION_WINDOWS else None,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if set(state) != {"version", "values", "valid", "replay_ids", "sha256", "targets"} or state["version"] != 1:
            raise ValueError("incompatible return calibration")
        values = state["values"]
        valid = state["valid"]
        replay_ids = state["replay_ids"]
        if not isinstance(values, list) or not isinstance(valid, list) or not isinstance(replay_ids, list):
            raise ValueError("calibration samples must be ordered lists")
        if (
            any(type(value) is not float or not math.isfinite(value) for value in values)
            or any(type(value) is not bool for value in valid)
            or any(not isinstance(value, str) or not value for value in replay_ids)
        ):
            raise ValueError("invalid calibration sample types")
        if len(values) != len(valid) or len(values) != len(replay_ids) or len(values) > CALIBRATION_WINDOWS:
            raise ValueError("invalid calibration lengths")
        self.values = cast(list[float], values).copy()
        self.valid = cast(list[bool], valid).copy()
        self.replay_ids = cast(list[str], replay_ids).copy()
        if self.state_dict() != state:
            raise ValueError("return calibration identity or targets changed")


@dataclass(frozen=True)
class Architecture:
    player_embed_dim: ClassVar[int] = 32
    activation_percentile_sample_size: ClassVar[int] = 65_536
    trunk_attention_backend: ClassVar[str] = "varlen_flash"
    trunk_reference_layers: ClassVar[int] = 8
    temporal_reference_layers: ClassVar[int] = 2
    trunk_reference_attention_scale: ClassVar[float] = 0.25
    temporal_reference_attention_scale: ClassVar[float] = 0.5

    d_model: int = 1024
    n_layers: int = 12
    n_heads: int = 16
    attn_window: int = 0
    L_ctx: int = 256

    sample_chunk_length: int = 28
    head_offsets: tuple[int, ...] = (*range(1, 13), 16, 20, 24, 28)
    temporal_d_model: int = 1024
    temporal_layers: int = 6
    temporal_heads: int = 16
    temporal_ff_dim: int = 4096
    group_head_dim: int = 1024
    return_embed_dim: int = 128
    action_embed_dim: int = 32
    offset_embed_dim: int = 16
    action_vocab: int = 1024
    action_state_embed_dim: int = 48
    char_vocab: int = 32
    char_dim: int = 8
    stage_vocab: int = 32
    stage_dim: int = 4
    item_type_dim: int = 16
    item_state_dim: int = 4
    item_hidden_dim: int = 64
    item_dim: int = 32
    value_hidden_dim: int = 512

    @property
    def direct_loss_start(self) -> int:
        if self.L_ctx % 2:
            raise ValueError("context length must be even for suffix supervision")
        return self.L_ctx // 2

    @property
    def parameter_count_contract(self) -> dict[str, int]:
        if self == Architecture():
            return {
                "trunk": 150_994_944,
                "temporal_decoder": 81_300_464,
                "group_heads": 4_558_179,
                "trunk_skip_heads": 4_558_179,
                "value_head": 1_049_089,
                "return_conditioner": 3_170_560,
                "other": 1_230_790,
                "total": 246_862_205,
            }
        proxy = Architecture(
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
        if self == proxy:
            return {
                "trunk": 9_437_184,
                "temporal_decoder": 5_195_504,
                "group_heads": 353_379,
                "trunk_skip_heads": 353_379,
                "value_head": 65_665,
                "return_conditioner": 792_832,
                "other": 861_382,
                "total": 17_059_325,
            }
        raise ValueError(f"no parameter contract for architecture {self}")


@dataclass(frozen=True)
class AWRCalibration:
    return_suffix: ClassVar[str] = "awr_return"
    near_offsets: ClassVar[int] = 6
    start_update: ClassVar[int] = 4097

    beta: float = 150.0
    weight_max: float = 10.0
    gamma: float = 0.99855
    stock_value: float = 120.0
    damage_shaping: float = 1.0
    win_reward: float = 50.0
    # Regressing the value error in beta units keeps the critic loss O(1)
    # despite the reward's roughly hundred-point scale.
    value_loss_weight: float = 1.0
    # Retained at 1.0 for checkpoint-format compatibility; offsets own the objective.
    auxiliary_loss_weight: float = 1.0

    @property
    def ego_return_column(self) -> str:
        return f"ego_{self.return_suffix}"

    @property
    def ego_return_valid_column(self) -> str:
        return f"{self.ego_return_column}_valid"


@dataclass(frozen=True)
class TrainConfig:
    reference_batch_size: ClassVar[int] = 512
    reference_positions: ClassVar[int] = 2**30
    base_adam_betas: ClassVar[tuple[float, float]] = (0.9, 0.95)
    base_adam_eps: ClassVar[float] = 1e-12
    inference_buckets: ClassVar[tuple[int, ...]] = (1, 2, 4, 8, 16, 32, 64)
    train_metrics_every: ClassVar[int] = 25
    train_prefetch_factor: ClassVar[int] = 4
    raw_shard_materialization_threads: ClassVar[int] = 64
    materialization_threads_env: ClassVar[str] = "HAL_O59_MATERIALIZATION_THREADS"
    data_protocol: ClassVar[str] = "o59-replay-ring-v2"
    selection_sha256: ClassVar[str] = "2593361352b92e705be3fbeae1b4e9bb1a3c9f1787cd713014a7a95b7df62477"
    mds_index_version: ClassVar[int] = 2
    mds_manifest_schema_sha256: ClassVar[str] = "405199de9494fe01350506734f0b2ec392fe79b0122d69cbcb5cae2afabc0d49"
    replay_slots: ClassVar[int] = 131_072
    windows_per_generation: ClassVar[int] = 8
    replay_phase_block_batches: ClassVar[int] = 25
    minimum_replay_gap_batches: ClassVar[int] = 200
    reserved_disk_bytes: ClassVar[int] = 256 * 2**30
    player_sidecar_remote: ClassVar[str] = "s3://hal/processed/player-identity-v1/professional-code-v1.jsonl.gz"

    arch: Annotated[Architecture, tyro.conf.Suppress] = Architecture()
    awr: Annotated[AWRCalibration, tyro.conf.Suppress] = AWRCalibration()

    prediction_frames: int = 4
    delay_frames: int = 2
    replan_interval_frames: int = 2
    inference_mode: str = "compiled"  # explicit "eager" is for debugging
    # Hardware-derived by default. An explicit power of two is a reproducibility
    # or memory-pressure override, not an architecture parameter.
    compiled_inference_bucket: int | None = None

    seed: int = 0
    eval_seed: int = 0
    batch_size: int = 512
    optimizer: Literal["muon", "adamw"] = "muon"
    muon_lr: float = 0.007
    muon_lr_multiplier: Annotated[float, tyro.conf.Suppress] = 1.0
    muon_weight_decay: float = 1e-4
    adam_lr: float = 4.25e-4
    adam_weight_decay: float = 1e-4
    grad_clip: float = 1.0
    lr_floor_ratio: float = 1 / 170
    amp_dtype: str = "bfloat16"
    allow_tf32: bool = True
    compile_trunk: bool = True
    compile_temporal: bool = True
    train_compile_mode: Literal["reduce-overhead", "max-autotune"] = "reduce-overhead"

    wandb_log_code: bool = True
    val_every: int = 4096
    val_n_samples: int = 2048
    val_batch_size: int = 128
    ckpt_every: int = 2048
    eval_every: int = 8192
    eval_max_frames: int = 7200
    eval_n_matchups: int = 96
    final_eval_n_matchups: int = 96
    eval_max_parallel: int | None = 32

    source_names: tuple[str, ...] = tuple(source.name for source in streams.POLICY_WORLD_V8_SOURCES)
    mds_schema_version: int = 7
    policy_world_schema_version: int = POLICY_WORLD_SCHEMA_VERSION
    download_retry: int = 8
    val_split: str = "val"
    num_workers: int = 24
    push_to_r2: bool = True
    system_metrics_every: int = 25
    system_metrics_interval_s: float = 5.0
    process_metrics_interval_s: float = 30.0
    cache_metrics_interval_s: float = 30.0
    identity_dropout: float = 0.10
    return_conditioning: bool = True
    return_dropout: float = 0.2
    parent_run_name: Annotated[str | None, tyro.conf.Suppress] = None
    parent_checkpoint_name: Annotated[str | None, tyro.conf.Suppress] = None
    parent_checkpoint_sha256: Annotated[str | None, tyro.conf.Suppress] = None
    parent_wandb_id: Annotated[str | None, tyro.conf.Suppress] = None
    player_sidecar_local: str = "data/processed/player-identity-v1/professional-code-v1.jsonl.gz"
    player_sidecar_sha256: str = "54ccf8a2497fe240313117297ca2ea31158e08db2cc53c67e7aa46853a8dac1c"
    player_vocab_sha256: str = "c67c97c995ad033ea7f5b2223efce5b061394566439f091ff6e7aaa6a9d1cfd6"
    player_vocab_size: int = 21_181
    target_positions: int = 8 * 2**30
    stable_updates: int = 98_304
    decay_start_update: int | None = None
    decay_duration: int | None = None
    depth_alpha: float = 0.5
    hidden_std_multiplier: float = 0.5
    readout_init: Literal["mup-normal"] = "mup-normal"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-12

    @property
    def max_steps(self) -> int:
        positions_per_update = self.batch_size * (self.arch.L_ctx - self.arch.L_ctx // 2)
        updates, remainder = divmod(self.target_positions, positions_per_update)
        if remainder:
            raise ValueError("target_positions must end on an optimizer boundary")
        return updates

    @property
    def warmup_steps(self) -> int:
        return 4096

    @property
    def policy_prefixes_per_update(self) -> int:
        return self.batch_size * 32

    @property
    def value_prefixes_per_update(self) -> int:
        return self.batch_size * (self.arch.L_ctx - self.arch.direct_loss_start)

    @property
    def minimum_replay_frames(self) -> int:
        return self.arch.L_ctx + self.arch.sample_chunk_length + self.windows_per_generation - 1

    @property
    def source_list_sha256(self) -> str:
        encoded = json.dumps(self.source_names, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def train_replays(self) -> int:
        return sum(streams.POLICY_WORLD_V8_TRAIN_REPLAYS[name] for name in self.source_names)

    @property
    def train_frames(self) -> int:
        return sum(streams.POLICY_WORLD_V8_TRAIN_FRAMES[name] for name in self.source_names)

    def materialization_threads(self) -> int:
        value = os.environ.get(self.materialization_threads_env)
        if value is None:
            return self.raw_shard_materialization_threads
        try:
            threads = int(value)
        except ValueError as error:
            raise ValueError(f"{self.materialization_threads_env} must be an integer") from error
        if threads not in (32, 64, 96):
            raise ValueError(f"{self.materialization_threads_env} must be 32, 64, or 96")
        return threads


def amp_context(cfg: TrainConfig, device: torch.device | str):
    if cfg.amp_dtype == "bfloat16" and torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def decoder_rmsnorm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.shape[-1],), eps=1e-6)


class SwiGLU(nn.Module):
    """Gated MLP used by every nonlinear projection in the policy."""

    def __init__(self, d_input: int, d_hidden: int, d_output: int, *, output_bias: bool = False) -> None:
        super().__init__()
        self.up = nn.Linear(d_input, 2 * d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, d_output, bias=output_bias)

    def activations(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return the activated gate, value branch, and their product."""
        gate_projection, value = self.up(x).chunk(2, dim=-1)
        gate = F.silu(gate_projection)
        return gate, value, gate * value

    def forward(self, x: Tensor) -> Tensor:
        _, _, product = self.activations(x)
        return self.down(product)


class NonlinearActionHead(nn.Module):
    """O26 RMSNorm-SiLU controller readout."""

    def __init__(self, d_model: int, d_hidden: int, vocab: int, *, norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.norm_eps = norm_eps
        self.up = nn.Linear(d_model, d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, vocab)

    def normalize(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), eps=self.norm_eps)

    def project(self, x: Tensor) -> Tensor:
        """Apply the head MLP to an input prepared by its caller."""
        return self.down(F.silu(self.up(x)))

    def forward(self, x: Tensor) -> Tensor:
        return self.project(self.normalize(x))

    def forward_with_input(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return logits and the normalized tensor read by the hidden layer."""
        logits, normalized, _up_output, _down_input = self.projection_activations(x)
        return logits, normalized

    def projection_activations(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return the output and the tensors at both projection boundaries."""
        normalized = self.normalize(x)
        output, up_output, down_input = self.project_activations(normalized)
        return output, normalized, up_output, down_input

    def project_activations(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return the output and activations for an input prepared by its caller."""
        up_output = self.up(x)
        down_input = F.silu(up_output)
        return self.down(down_input), up_output, down_input


def _sampled_quantile(tensor: Tensor, percentile: float, *, absolute: bool = False) -> Tensor:
    """Estimate a quantile from a deterministic bounded-size flat sample."""
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    values = tensor.detach().flatten()
    sample_size = Architecture.activation_percentile_sample_size
    stride = max((values.numel() + sample_size - 1) // sample_size, 1)
    if stride > 1:
        stride += 1
    sample = values[::stride].float()
    if absolute:
        sample = sample.abs()
    rank = min(max(math.ceil(percentile * sample.numel() / 100.0), 1), sample.numel())
    return torch.kthvalue(sample, rank).values


def _bounded_flat_sample(tensor: Tensor) -> Tensor:
    """Return a deterministic activation sample without retaining its source."""
    values = tensor.detach().flatten()
    sample_size = Architecture.activation_percentile_sample_size
    stride = max((values.numel() + sample_size - 1) // sample_size, 1)
    return values[::stride][:sample_size].float()


def _activation_input_metrics(prefix: str, tensor: Tensor) -> dict[str, Tensor]:
    values = _bounded_flat_sample(tensor)
    return {
        f"{prefix}/input_rms": values.square().mean().sqrt(),
        f"{prefix}/input_abs_max": values.abs().amax(),
    }


def _activation_output_metrics(prefix: str, tensor: Tensor) -> dict[str, Tensor]:
    """Measure a bounded output sample, ignoring masked infinite logits."""
    values = _bounded_flat_sample(tensor)
    legal = torch.isfinite(values)
    count = legal.sum().clamp_min(1)
    finite = values.masked_fill(~legal, 0)
    ordered = values.abs().masked_fill(~legal, torch.inf).sort().values

    def percentile(numerator: int, denominator: int) -> Tensor:
        rank = (count * numerator + denominator - 1).div(denominator, rounding_mode="floor").clamp_min(1)
        return ordered.gather(0, (rank - 1).reshape(1)).squeeze(0)

    return {
        f"{prefix}/output_rms": (finite.square().sum() / count).sqrt(),
        f"{prefix}/output_abs_p99": percentile(99, 100),
        f"{prefix}/output_abs_p999": percentile(999, 1000),
    }


def short_causal_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> Tensor:
    """Use explicit causal attention for the 16-token training sequence.

    On a B200, cuDNN flash SDPA used 149 ms per step for forward and
    backward. At this length, its launch and layout costs were more than the
    cost to materialize 121 scores per head. This implementation reduced the
    full train step by approximately 100 ms.
    """
    scores = query @ key.transpose(-2, -1)
    scores = scores.float() * (query.shape[-1] ** -0.5)
    causal = torch.ones(scores.shape[-2:], dtype=torch.bool, device=scores.device).tril()
    weights = F.softmax(scores.masked_fill(~causal, -torch.inf), dim=-1).to(query.dtype)
    return weights @ value


TEMPORAL_ATTENTION_BATCH = 16_384


@dataclass(frozen=True, slots=True)
class DepthRule:
    attention: float
    mlp: float


def depth_rule(stack: Literal["trunk", "temporal"], layers: int, alpha: float) -> DepthRule:
    """Return O51's residual multipliers for one stack."""
    if alpha != 0.5:
        raise ValueError("O59 fixes depth_alpha to 0.5")
    if layers < 1:
        raise ValueError("stack depth must be positive")
    if stack == "trunk":
        base_layers = Architecture.trunk_reference_layers
        base_attention = Architecture.trunk_reference_attention_scale
    elif stack == "temporal":
        base_layers = Architecture.temporal_reference_layers
        base_attention = Architecture.temporal_reference_attention_scale
    else:
        raise ValueError(f"unknown stack {stack!r}")
    branch = (layers / base_layers) ** -alpha
    return DepthRule(attention=base_attention * branch, mlp=branch)


class ReturnConditioner(nn.Module):
    """Modulate decoder norms once per prefix; absence is exactly neutral."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.width = cfg.arch.temporal_d_model
        self.enabled = cfg.return_conditioning
        self.embedding = nn.Linear(1, cfg.arch.return_embed_dim, bias=True)
        self.projections = nn.ModuleList(
            nn.Linear(cfg.arch.return_embed_dim, 4 * self.width, bias=True) for _ in range(cfg.arch.temporal_layers)
        )
        nn.init.normal_(self.embedding.weight, std=cfg.hidden_std_multiplier)
        nn.init.zeros_(self.embedding.bias)
        for module in self.projections:
            projection = cast(nn.Linear, module)
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, return_value: Tensor, condition_present: Tensor) -> tuple[Tensor, ...]:
        if return_value.shape != condition_present.shape or condition_present.dtype != torch.bool:
            raise ValueError("return values and boolean presence must have the same shape")
        present = condition_present & self.enabled
        safe = torch.where(present, return_value, 0.0)
        embedded = F.silu(self.embedding((safe / RETURN_SCALE)[..., None]))
        return tuple(
            torch.where(
                present[..., None, None],
                projection(embedded).reshape(*return_value.shape, 4, self.width),
                0.0,
            )
            for projection in self.projections
        )


class TemporalBlock(nn.Module):
    """RoPE causal block with O51 depth scaling on both branches."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.n_heads = cfg.arch.temporal_heads
        self.d_model = cfg.arch.temporal_d_model
        self.head_dim = self.d_model // self.n_heads
        rule = depth_rule("temporal", cfg.arch.temporal_layers, cfg.depth_alpha)
        self.scale = rule.attention
        self.mlp_scale = rule.mlp
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.rotary = Rotary(self.head_dim)
        self.up = nn.Linear(self.d_model, cfg.arch.temporal_ff_dim, bias=False)
        self.down = nn.Linear(cfg.arch.temporal_ff_dim, self.d_model, bias=False)

    def _qkv(self, x: Tensor, scale: Tensor, shift: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = x.shape
        q, k, v = self.qkv((1 + scale) * decoder_rmsnorm(x) + shift).split(self.d_model, dim=-1)
        shape = (batch, length, self.n_heads, self.head_dim)
        return q.view(shape), k.view(shape), v.view(shape)

    def forward(self, x: Tensor, modulation: Tensor) -> Tensor:
        scale_attn, shift_attn, scale_mlp, shift_mlp = modulation.unbind(-2)
        q, k, v = self._qkv(x, scale_attn, shift_attn)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin).transpose(1, 2)
        k = apply_rotary_emb(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)
        # Equal attention chunks keep the tiny BMMs efficient. The large linear
        # and SwiGLU operations use the full batch.
        attended = torch.cat(
            [
                short_causal_attention(query, key, values)
                for query, key, values in zip(
                    q.split(TEMPORAL_ATTENTION_BATCH),
                    k.split(TEMPORAL_ATTENTION_BATCH),
                    v.split(TEMPORAL_ATTENTION_BATCH),
                    strict=True,
                )
            ],
            dim=0,
        )
        attended = attended.transpose(1, 2).contiguous().view_as(x)
        x = x + self.scale * self.proj(attended)
        return x + self.mlp_scale * self.down(F.silu(self.up((1 + scale_mlp) * decoder_rmsnorm(x) + shift_mlp)))

    def forward_step(
        self, x: Tensor, past: tuple[Tensor, Tensor] | None, modulation: Tensor
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        scale_attn, shift_attn, scale_mlp, shift_mlp = modulation.unbind(-2)
        q, k, v = self._qkv(x[:, None], scale_attn[:, None], shift_attn[:, None])
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        cos, sin = self.rotary.at(k.shape[2], x.device)
        q = apply_rotary_emb(q, cos[:, -1:], sin[:, -1:]).transpose(1, 2)
        rotated_k = apply_rotary_emb(k.transpose(1, 2), cos, sin).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, rotated_k, v)
        attended = attended.transpose(1, 2).contiguous().view_as(x)
        x = x + self.scale * self.proj(attended)
        x = x + self.mlp_scale * self.down(F.silu(self.up((1 + scale_mlp) * decoder_rmsnorm(x) + shift_mlp)))
        return x, (k, v)


class HistoryCrossAttention(nn.Module):
    """Attend packed prefix/offset queries to one shared trunk memory."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.n_heads = cfg.arch.temporal_heads
        self.d_model = cfg.arch.temporal_d_model
        self.head_dim = self.d_model // self.n_heads
        self.scale = depth_rule("temporal", cfg.arch.temporal_layers, cfg.depth_alpha).attention
        self.query = nn.Linear(self.d_model, self.d_model, bias=False)
        self.key_value = nn.Linear(cfg.arch.d_model, 2 * self.d_model, bias=False)
        self.output = nn.Linear(self.d_model, self.d_model, bias=False)
        self.rotary = Rotary(self.head_dim)

    def project_memory(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        if hidden.ndim != 3:
            raise ValueError(f"history memory must be [B, L, d], got {tuple(hidden.shape)}")
        batch, length, _ = hidden.shape
        key, value = self.key_value(decoder_rmsnorm(hidden)).chunk(2, dim=-1)
        shape = (batch, length, self.n_heads, self.head_dim)
        key = key.view(shape)
        value = value.view(shape).transpose(1, 2)
        cos, sin = self.rotary.at(length, key.device, key.dtype)
        return apply_rotary_emb(key, cos, sin).transpose(1, 2), value

    def forward_with_kv_cache(self, x: Tensor, memory: KVMemory) -> Tensor:
        query = self.query(decoder_rmsnorm(x)).view(1, 1, self.n_heads, self.head_dim)
        query = rotate_positions(query, memory.query_position, self.rotary).transpose(1, 2)
        valid = (
            (memory.positions >= 0)
            & (memory.positions <= memory.query_position)
            & (memory.positions > memory.query_position - memory.window)
        )
        attended = F.scaled_dot_product_attention(query, memory.kv[0], memory.kv[1], attn_mask=valid[None, None, None])
        return x + self.scale * self.output(attended.transpose(1, 2).reshape_as(x))

    def forward_projected(
        self,
        x: Tensor,
        key: Tensor,
        value: Tensor,
        ctx_pad: Tensor,
        prefix_positions: Tensor,
        *,
        offsets_per_prefix: int,
    ) -> Tensor:
        """Apply causal history attention without replicating K/V by prefix."""
        batch, queries, _ = x.shape
        length = key.shape[2]
        if key.shape != (batch, self.n_heads, length, self.head_dim) or value.shape != key.shape:
            raise ValueError("history K/V shape mismatch")
        if ctx_pad.shape != (batch,) or prefix_positions.ndim != 2:
            raise ValueError("history padding and prefix positions have the wrong shape")
        if queries != prefix_positions.shape[1] * offsets_per_prefix:
            raise ValueError("history query packing does not match prefix positions")

        query = self.query(decoder_rmsnorm(x)).view(batch, queries, self.n_heads, self.head_dim)
        query_positions = prefix_positions.repeat_interleave(offsets_per_prefix, dim=1)
        cos, sin = self.rotary.at(length, query.device, query.dtype)
        query_cos = cos[0, :, 0][query_positions][:, :, None, :]
        query_sin = sin[0, :, 0][query_positions][:, :, None, :]
        query = apply_rotary_emb(query, query_cos, query_sin).transpose(1, 2)

        if query.device.type == "cuda":

            def score_mod(score: Tensor, b: Tensor, h: Tensor, q: Tensor, kv: Tensor) -> Tensor:
                del h
                allowed = (kv >= ctx_pad[b]) & (kv <= query_positions[b, q])
                return torch.where(allowed, score, -torch.inf)

            attended = cast(Tensor, flex_attention(query, key, value, score_mod=score_mod))
        else:
            memory = torch.arange(length, device=x.device)
            valid = (memory[None, None, :] >= ctx_pad[:, None, None]) & (
                memory[None, None, :] <= query_positions[:, :, None]
            )
            attended = F.scaled_dot_product_attention(query, key, value, attn_mask=valid[:, None])
        attended = attended.transpose(1, 2).contiguous().view_as(x)
        return x + self.scale * self.output(attended)


class CausalTemporalDecoder(nn.Module):
    """Temporal action chain conditioned by concatenation."""

    def __init__(self, cfg: TrainConfig, codec: DiscreteControllerCodec) -> None:
        super().__init__()
        self.codec = codec
        self.head_offsets = tuple(cfg.arch.head_offsets)
        self.live_horizons = (cfg.prediction_frames,)
        self.d_model = cfg.arch.temporal_d_model
        self.training_diagnostics = cfg.optimizer == "adamw"
        controller_width = CONTROLLER_GROUP_COUNT * cfg.arch.action_embed_dim
        self.offset_embedding = nn.Embedding(cfg.arch.sample_chunk_length + 1, cfg.arch.offset_embed_dim)
        self.token_projection = nn.Linear(
            cfg.arch.d_model + controller_width + cfg.arch.offset_embed_dim, self.d_model
        )
        self.blocks = nn.ModuleList([TemporalBlock(cfg) for _ in range(cfg.arch.temporal_layers)])
        self.history_attention = HistoryCrossAttention(cfg)
        self.group_condition = nn.ModuleDict(
            {
                name: nn.Linear(position * cfg.arch.action_embed_dim, 2 * self.d_model)
                for position, name in enumerate(CONTROLLER_DECODE_ORDER)
                if position
            }
        )
        self.outputs = nn.ModuleDict(
            {
                name: NonlinearActionHead(
                    self.d_model,
                    cfg.arch.group_head_dim,
                    CONTROLLER_GROUP_VOCABS[CONTROLLER_GROUP_INDEX[name]],
                    # Keep the near-zero button Jacobian bounded in every path.
                    norm_eps=1e-5 if name == "buttons" else 1e-6,
                )
                for name in CONTROLLER_GROUP_NAMES
            }
        )
        self.trunk_outputs = nn.ModuleDict(
            {
                name: NonlinearActionHead(
                    cfg.arch.d_model,
                    cfg.arch.group_head_dim,
                    CONTROLLER_GROUP_VOCABS[CONTROLLER_GROUP_INDEX[name]],
                )
                for name in CONTROLLER_GROUP_NAMES
            }
        )
        self.trunk_width = cfg.arch.d_model
        self.controller_width = controller_width

    return_conditioner: ReturnConditioner

    def _state_bias(self, trunk: Tensor) -> Tensor:
        """The trunk share of the token projection, computed once per position.

        A linear layer over a concatenation decomposes into a sum of per-part
        linears: ``W [h | a | o] + b = W_h h + W_a a + W_o o + b``. The trunk
        part is constant across the chain's steps, so it never needs the
        per-step copy the concatenation implied. The single ``token_projection``
        parameter is kept (same shape, same initialization as the concatenating
        form); only the compute schedule changes.
        """
        weight = self.token_projection.weight
        return F.linear(trunk, weight[:, : self.trunk_width], self.token_projection.bias)

    def _step_features(self, previous: Tensor, offsets: Tensor, embedded: Tensor | None = None) -> Tensor:
        """The per-step share of the token projection: previous action and offset."""
        weight = self.token_projection.weight
        action_weight = weight[:, self.trunk_width : self.trunk_width + self.controller_width]
        offset_weight = weight[:, self.trunk_width + self.controller_width :]
        action = F.linear(self.codec.embed_frame(previous) if embedded is None else embedded, action_weight)
        return action + F.linear(self.offset_embedding(offsets), offset_weight)

    def _trunk_skip_logits(self, hidden: Tensor) -> dict[str, Tensor]:
        """Compute context-only logits once for all decoded offsets."""
        return {name: self.trunk_outputs[name](hidden) for name in CONTROLLER_GROUP_NAMES}

    def _decode_step(
        self,
        previous: Tensor,
        offset: int,
        state_bias: Tensor,
        caches: list[tuple[Tensor, Tensor] | None],
        history: tuple[Tensor, Tensor, Tensor, Tensor] | KVMemory,
        conditioning: tuple[Tensor, ...],
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor] | None]]:
        """Advance the temporal chain by one selected frame offset."""
        offsets = torch.full((previous.shape[0],), offset, device=previous.device, dtype=torch.long)
        state = decoder_rmsnorm(state_bias + self._step_features(previous, offsets))
        next_caches: list[tuple[Tensor, Tensor] | None] = []
        for index, (module, past) in enumerate(zip(self.blocks, caches, strict=True)):
            block = cast(TemporalBlock, module)
            state, present = block.forward_step(state, past, conditioning[index])
            if index == 0:
                if isinstance(history, KVMemory):
                    state = self.history_attention.forward_with_kv_cache(state[:, None], history)[:, 0]
                else:
                    key, value, ctx_pad, prefix_positions = history
                    state = self.history_attention.forward_projected(
                        state[:, None],
                        key,
                        value,
                        ctx_pad,
                        prefix_positions,
                        offsets_per_prefix=1,
                    )[:, 0]
            next_caches.append(present)
        return decoder_rmsnorm(state), next_caches

    def teacher_forced_states(
        self,
        hidden: Tensor,
        ctx_pad: Tensor,
        prefix_positions: Tensor,
        observed: Tensor,
        targets: Tensor,
        return_value: Tensor,
        condition_present: Tensor,
    ) -> Tensor:
        prefix_shape = tuple(prefix_positions.shape)
        expected = (*prefix_shape, len(self.head_offsets), CONTROLLER_GROUP_COUNT)
        if observed.shape != (*prefix_shape, CONTROLLER_GROUP_COUNT) or targets.shape != expected:
            raise ValueError(
                f"expected observed {(*prefix_shape, CONTROLLER_GROUP_COUNT)} and targets {expected}, got "
                f"{tuple(observed.shape)} and {tuple(targets.shape)}"
            )
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
        selected_hidden = hidden[batch_indices, prefix_positions]
        previous = torch.cat((observed[:, :, None], targets[..., :-1, :]), dim=2)
        trunk = decoder_rmsnorm(selected_hidden)
        offsets = torch.tensor(self.head_offsets, device=hidden.device)
        x = self._state_bias(trunk)[:, :, None] + self._step_features(previous, offsets)
        x = decoder_rmsnorm(x)
        x = x.reshape(hidden.shape[0] * prefix_positions.shape[1], len(self.head_offsets), self.d_model)
        conditioning = self.return_conditioner(return_value, condition_present)
        for index, block in enumerate(self.blocks):
            x = block(x, conditioning[index].reshape(-1, 1, 4, self.d_model))
            if index == 0:
                packed = x.view(hidden.shape[0], -1, self.d_model)
                key, value = self.history_attention.project_memory(hidden)
                packed = self.history_attention.forward_projected(
                    packed,
                    key,
                    value,
                    ctx_pad,
                    prefix_positions,
                    offsets_per_prefix=len(self.head_offsets),
                )
                x = packed.view_as(x)
        return decoder_rmsnorm(x.view(*prefix_shape, len(self.head_offsets), self.d_model))

    def group_features(self, states: Tensor, name: str, embedded: dict[str, Tensor]) -> Tensor:
        head = cast(NonlinearActionHead, self.outputs[name])
        normalized = head.normalize(states)
        position = CONTROLLER_DECODE_ORDER.index(name)
        if position == 0:
            return normalized
        prefix = torch.cat([embedded[group] for group in CONTROLLER_DECODE_ORDER[:position]], dim=-1)
        raw_scale, raw_shift = self.group_condition[name](prefix).chunk(2, dim=-1)
        scale = torch.tanh(raw_scale)
        shift = raw_shift
        return normalized * (1.0 + scale) + shift

    def _teacher_forced_outputs(
        self,
        hidden: Tensor,
        ctx_pad: Tensor,
        prefix_positions: Tensor,
        observed: Tensor,
        targets: Tensor,
        return_value: Tensor,
        condition_present: Tensor,
    ) -> tuple[
        dict[str, Tensor],
        tuple[Tensor, Tensor, Tensor],
        dict[str, tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]],
    ]:
        """Return group logits and tensors at the action projection boundaries."""
        states = self.teacher_forced_states(
            hidden, ctx_pad, prefix_positions, observed, targets, return_value, condition_present
        )
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
        selected_hidden = hidden[batch_indices, prefix_positions]
        embedded = self.codec.embed_groups(targets)
        logits: dict[str, Tensor] = {}
        projection_values: dict[
            str,
            tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        ] = {}
        button_values: tuple[Tensor, Tensor] | None = None
        for name in CONTROLLER_GROUP_NAMES:
            features = self.group_features(states, name, embedded)
            head = cast(NonlinearActionHead, self.outputs[name])
            head_output, up_output, down_input = head.project_activations(features)
            head_input = features
            trunk_head = cast(NonlinearActionHead, self.trunk_outputs[name])
            trunk_output, trunk_input, trunk_up_output, trunk_down_input = trunk_head.projection_activations(
                selected_hidden
            )
            combined_logits = head_output + trunk_output[..., None, :]
            projection_values[name] = (
                head_input,
                up_output,
                down_input,
                head_output,
                trunk_input,
                trunk_up_output,
                trunk_down_input,
                trunk_output,
            )
            if name == "buttons":
                button_values = (head_input, combined_logits)
            logits[name] = self._center(combined_logits)
        if button_values is None:
            raise RuntimeError("button head was not evaluated")
        button_mask = self.codec.button_mask(targets[..., TRIGGERS_GROUP])
        logits["buttons"] = logits["buttons"].masked_fill(button_mask, float("-inf"))
        return logits, (*button_values, button_mask), projection_values

    def teacher_forced_logits_by_group(
        self,
        hidden: Tensor,
        ctx_pad: Tensor,
        prefix_positions: Tensor,
        observed: Tensor,
        targets: Tensor,
        return_value: Tensor,
        condition_present: Tensor,
    ) -> dict[str, Tensor]:
        logits, _button_values, _projection_values = self._teacher_forced_outputs(
            hidden, ctx_pad, prefix_positions, observed, targets, return_value, condition_present
        )
        return logits

    @staticmethod
    def nll_from_logits(logits: dict[str, Tensor], targets: Tensor) -> Tensor:
        losses = [
            F.cross_entropy(
                logits[name].float().reshape(-1, CONTROLLER_GROUP_VOCABS[group]),
                targets[..., group].reshape(-1),
                reduction="none",
            ).view(*targets.shape[:-1])
            for group, name in enumerate(CONTROLLER_GROUP_NAMES)
        ]
        return torch.stack(losses, dim=-1)

    def teacher_forced_nll(
        self,
        hidden: Tensor,
        ctx_pad: Tensor,
        prefix_positions: Tensor,
        observed: Tensor,
        targets: Tensor,
        return_value: Tensor,
        condition_present: Tensor,
    ) -> Tensor:
        logits = self.teacher_forced_logits_by_group(
            hidden, ctx_pad, prefix_positions, observed, targets, return_value, condition_present
        )
        return self.nll_from_logits(logits, targets)

    def teacher_forced_nll_with_diagnostics(
        self,
        hidden: Tensor,
        ctx_pad: Tensor,
        prefix_positions: Tensor,
        observed: Tensor,
        targets: Tensor,
        return_value: Tensor,
        condition_present: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return NLL plus a compact button-boundary stability signal."""
        logits, button_values, projection_values = self._teacher_forced_outputs(
            hidden, ctx_pad, prefix_positions, observed, targets, return_value, condition_present
        )
        metrics = self._button_metrics(*button_values, targets)
        if self.training_diagnostics:
            for name, values in projection_values.items():
                metrics.update(self._group_metrics(name, values, logits[name]))
        return self.nll_from_logits(logits, targets), metrics

    @staticmethod
    def _button_metrics(
        head_input: Tensor, raw_logits: Tensor, button_mask: Tensor, targets: Tensor
    ) -> dict[str, Tensor]:
        input_values = head_input.detach()
        raw_logits_values = raw_logits.detach()
        button_targets = targets[..., BUTTONS_GROUP, None]
        target_logits = raw_logits_values.gather(-1, button_targets).squeeze(-1).float()
        legal_logits = raw_logits_values.masked_fill(button_mask, float("-inf"))
        competing_logits = legal_logits.scatter(-1, button_targets, float("-inf")).amax(dim=-1).float()
        margin = target_logits - competing_logits
        return {
            "stability/button_input_abs_p999": _sampled_quantile(input_values, 99.9, absolute=True),
            "stability/button_logit_abs_p999": _sampled_quantile(raw_logits_values, 99.9, absolute=True),
            "stability/button_margin_mean": margin.mean(),
        }

    @staticmethod
    def _group_metrics(
        name: str, values: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor], logits: Tensor
    ) -> dict[str, Tensor]:
        (
            head_input,
            up_output,
            down_input,
            down_output,
            trunk_input,
            trunk_up_output,
            trunk_down_input,
            trunk_output,
        ) = values
        metrics: dict[str, Tensor] = {}
        prefix = f"diagnostics/activations/action/{name}"
        metrics.update(_activation_input_metrics(f"{prefix}/up", head_input))
        metrics.update(_activation_output_metrics(f"{prefix}/up", up_output))
        metrics.update(_activation_input_metrics(f"{prefix}/down", down_input))
        metrics.update(_activation_output_metrics(f"{prefix}/down", down_output))
        metrics.update(_activation_input_metrics(f"{prefix}/trunk_skip/up", trunk_input))
        metrics.update(_activation_output_metrics(f"{prefix}/trunk_skip/up", trunk_up_output))
        metrics.update(_activation_input_metrics(f"{prefix}/trunk_skip/down", trunk_down_input))
        metrics.update(_activation_output_metrics(f"{prefix}/trunk_skip/down", trunk_output))
        metrics.update(_activation_output_metrics(f"{prefix}/combined_centered_logits", logits))
        return metrics

    def teacher_forced_logits(
        self,
        hidden: Tensor,
        ctx_pad: Tensor,
        prefix_positions: Tensor,
        observed: Tensor,
        targets: Tensor,
        return_value: Tensor,
        condition_present: Tensor,
    ) -> list[dict[str, Tensor]]:
        values = self.teacher_forced_logits_by_group(
            hidden, ctx_pad, prefix_positions, observed, targets, return_value, condition_present
        )
        return [
            {name: logits[..., depth, :] for name, logits in values.items()} for depth in range(len(self.head_offsets))
        ]

    def _live_history(self, hidden: Tensor, ctx_pad: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if ctx_pad is None:
            ctx_pad = torch.zeros(hidden.shape[0], dtype=torch.long, device=hidden.device)
        prefix_positions = torch.full(
            (hidden.shape[0], 1), hidden.shape[1] - 1, dtype=torch.long, device=hidden.device
        )
        return (*self.history_attention.project_memory(hidden), ctx_pad, prefix_positions)

    def forced_stepwise_logits(
        self,
        hidden: Tensor,
        observed: Tensor,
        targets: Tensor,
        return_value: Tensor,
        condition_present: Tensor,
        *,
        ctx_pad: Tensor | None = None,
    ) -> list[dict[str, Tensor]]:
        if targets.shape != (hidden.shape[0], len(self.head_offsets), CONTROLLER_GROUP_COUNT):
            raise ValueError("stepwise targets have the wrong shape")
        raw_trunk = hidden[:, -1]
        trunk = decoder_rmsnorm(raw_trunk)
        state_bias = self._state_bias(trunk)
        trunk_logits = self._trunk_skip_logits(raw_trunk)
        previous = observed
        caches: list[tuple[Tensor, Tensor] | None] = [None] * len(self.blocks)
        history = self._live_history(hidden, ctx_pad)
        conditioning = self.return_conditioner(return_value, condition_present)
        out: list[dict[str, Tensor]] = []
        for depth, offset in enumerate(self.head_offsets):
            state, caches = self._decode_step(previous, offset, state_bias, caches, history, conditioning)
            target = targets[:, depth]
            embedded = self.codec.embed_groups(target)
            group_logits = {
                name: self._center(
                    cast(NonlinearActionHead, self.outputs[name]).project(self.group_features(state, name, embedded))
                    + trunk_logits[name]
                )
                for name in CONTROLLER_GROUP_NAMES
            }
            group_logits["buttons"] = group_logits["buttons"].masked_fill(
                self.codec.button_mask(target[:, TRIGGERS_GROUP]), float("-inf")
            )
            out.append(group_logits)
            previous = target
        return out

    def sample_indices(
        self,
        hidden: Tensor,
        observed: Tensor,
        offsets: tuple[int, ...],
        return_value: Tensor,
        condition_present: Tensor,
        *,
        argmax: bool,
        uniforms: Tensor | None = None,
        gen: torch.Generator | None = None,
        temperature: float | Tensor = 1.0,
        ctx_pad: Tensor | None = None,
        forced_prefix: Tensor | None = None,
        history: KVMemory | None = None,
    ) -> Tensor:
        indices, _ = self._sample_indices_and_logits(
            hidden,
            observed,
            offsets,
            return_value,
            condition_present,
            argmax=argmax,
            uniforms=uniforms,
            gen=gen,
            temperature=temperature,
            capture_logits=False,
            ctx_pad=ctx_pad,
            forced_prefix=forced_prefix,
            history=history,
        )
        return indices

    def sample_indices_with_logits(
        self,
        hidden: Tensor,
        observed: Tensor,
        offsets: tuple[int, ...],
        return_value: Tensor,
        condition_present: Tensor,
        *,
        argmax: bool,
        uniforms: Tensor | None = None,
        gen: torch.Generator | None = None,
        temperature: float | Tensor = 1.0,
        ctx_pad: Tensor | None = None,
        forced_prefix: Tensor | None = None,
        history: KVMemory | None = None,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        """Sample once and return the exact conditional logits used by each draw."""
        return self._sample_indices_and_logits(
            hidden,
            observed,
            offsets,
            return_value,
            condition_present,
            argmax=argmax,
            uniforms=uniforms,
            gen=gen,
            temperature=temperature,
            capture_logits=True,
            ctx_pad=ctx_pad,
            forced_prefix=forced_prefix,
            history=history,
        )

    def configure_live_horizons(self, horizons: tuple[int, ...]) -> None:
        """Select runtime decode shapes without changing the saved training config."""
        if not horizons or tuple(sorted(set(horizons))) != horizons:
            raise ValueError("live horizons must be sorted and unique")
        for horizon in horizons:
            if horizon < 1 or self.head_offsets[:horizon] != tuple(range(1, horizon + 1)):
                raise ValueError("live horizons require contiguous trained prediction heads")
        self.live_horizons = horizons

    def _sample_indices_and_logits(
        self,
        hidden: Tensor,
        observed: Tensor,
        offsets: tuple[int, ...],
        return_value: Tensor,
        condition_present: Tensor,
        *,
        argmax: bool,
        uniforms: Tensor | None,
        gen: torch.Generator | None,
        temperature: float | Tensor,
        capture_logits: bool,
        ctx_pad: Tensor | None,
        forced_prefix: Tensor | None,
        history: KVMemory | None = None,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        if not isinstance(temperature, Tensor):
            temperature = validate_sampling_temperature(temperature)
        allowed = tuple(self.head_offsets[:horizon] for horizon in self.live_horizons)
        if offsets not in allowed:
            raise ValueError(f"live decode offsets must select one of the dense prefixes {allowed}")
        if uniforms is not None and uniforms.shape != (len(offsets), CONTROLLER_GROUP_COUNT, hidden.shape[0]):
            raise ValueError("uniform table must be [frames, groups, batch]")
        if forced_prefix is not None and (
            forced_prefix.ndim != 3
            or forced_prefix.shape[0] != hidden.shape[0]
            or forced_prefix.shape[2] != CONTROLLER_GROUP_COUNT
            or forced_prefix.shape[1] > len(offsets)
        ):
            raise ValueError("forced prefix must be [B, K, groups] with K no larger than the horizon")
        raw_trunk = hidden[:, -1]
        trunk = decoder_rmsnorm(raw_trunk)
        state_bias = self._state_bias(trunk)
        trunk_logits = self._trunk_skip_logits(raw_trunk)
        previous = observed
        caches: list[tuple[Tensor, Tensor] | None] = [None] * len(self.blocks)
        memory = self._live_history(hidden, ctx_pad) if history is None else history
        conditioning = self.return_conditioner(return_value, condition_present)
        frames: list[Tensor] = []
        captured: dict[str, list[Tensor]] = {name: [] for name in CONTROLLER_GROUP_NAMES}
        for depth, offset in enumerate(offsets):
            state, caches = self._decode_step(previous, offset, state_bias, caches, memory, conditioning)
            embedded: dict[str, Tensor] = {}
            picks: dict[str, Tensor] = {}
            for name in CONTROLLER_DECODE_ORDER:
                logits = self._center(
                    cast(NonlinearActionHead, self.outputs[name]).project(self.group_features(state, name, embedded))
                    + trunk_logits[name]
                )
                if name == "buttons":
                    logits = logits.masked_fill(self.codec.button_mask(picks["triggers"]), float("-inf"))
                if capture_logits:
                    captured[name].append(logits)
                group = CONTROLLER_GROUP_INDEX[name]
                uniform = None if uniforms is None else uniforms[depth, group]
                if forced_prefix is not None and depth < forced_prefix.shape[1]:
                    pick = forced_prefix[:, depth, group]
                else:
                    if isinstance(temperature, Tensor):
                        if uniform is None or argmax:
                            raise ValueError("tensor temperature requires stochastic uniforms")
                        pick = sample_with_temperature(logits, uniform, temperature)
                    else:
                        pick = sample_categorical(
                            logits,
                            argmax=argmax,
                            uniform=uniform,
                            generator=gen,
                            temperature=temperature,
                        )
                picks[name] = pick
                embedded[name] = self.codec.group_embedding(name, pick)
            indices = torch.stack([picks[name] for name in CONTROLLER_GROUP_NAMES], dim=-1)
            frames.append(indices)
            previous = indices
        logits_by_group = (
            tuple(torch.stack(captured[name], dim=1) for name in CONTROLLER_GROUP_NAMES) if capture_logits else ()
        )
        return torch.stack(frames, dim=1), logits_by_group

    def rollout_conditioned_logits(
        self,
        hidden: Tensor,
        observed: Tensor,
        return_value: Tensor,
        condition_present: Tensor,
        *,
        ctx_pad: Tensor | None = None,
    ) -> tuple[list[dict[str, Tensor]], Tensor]:
        """Offline ancestral diagnostic across every selected offset.

        Unlike :meth:`sample_indices`, this intentionally includes the sparse
        tail.  It is used only by validation to measure exposure gaps and is not
        reachable from the closed-loop inference wrapper.
        """
        raw_trunk = hidden[:, -1]
        trunk = decoder_rmsnorm(raw_trunk)
        state_bias = self._state_bias(trunk)
        trunk_logits = self._trunk_skip_logits(raw_trunk)
        previous = observed
        caches: list[tuple[Tensor, Tensor] | None] = [None] * len(self.blocks)
        history = self._live_history(hidden, ctx_pad)
        conditioning = self.return_conditioner(return_value, condition_present)
        frames: list[Tensor] = []
        all_logits: list[dict[str, Tensor]] = []
        for offset in self.head_offsets:
            state, caches = self._decode_step(previous, offset, state_bias, caches, history, conditioning)
            embedded: dict[str, Tensor] = {}
            picks: dict[str, Tensor] = {}
            frame_logits: dict[str, Tensor] = {}
            for name in CONTROLLER_DECODE_ORDER:
                logits = self._center(
                    cast(NonlinearActionHead, self.outputs[name]).project(self.group_features(state, name, embedded))
                    + trunk_logits[name]
                )
                if name == "buttons":
                    logits = logits.masked_fill(self.codec.button_mask(picks["triggers"]), float("-inf"))
                pick = logits.argmax(dim=-1)
                frame_logits[name] = logits
                picks[name] = pick
                embedded[name] = self.codec.group_embedding(name, pick)
            previous = torch.stack([picks[name] for name in CONTROLLER_GROUP_NAMES], dim=-1)
            frames.append(previous)
            all_logits.append(frame_logits)
        return all_logits, torch.stack(frames, dim=1)

    @staticmethod
    def _center(logits: Tensor) -> Tensor:
        return center_class_logits(logits)


class GPT(nn.Module):
    def __init__(self, cfg: TrainConfig, vocabulary: PlayerVocabulary | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.L_chunk = cfg.arch.sample_chunk_length
        self.head_offsets = tuple(cfg.arch.head_offsets)
        self.codec = DiscreteControllerCodec(cfg.arch.action_embed_dim)
        self.cat_specs = {**CAT_FEATURES, "action": (cfg.arch.action_vocab, cfg.arch.action_state_embed_dim)}
        self.cat_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocab, dim) for name, (vocab, dim) in self.cat_specs.items()}
        )
        self.char_emb = nn.Embedding(cfg.arch.char_vocab, cfg.arch.char_dim)
        self.stage_emb = nn.Embedding(cfg.arch.stage_vocab, cfg.arch.stage_dim)
        per_player = len(FLOAT_FEATURES) * 2 + sum(dim for _, dim in self.cat_specs.values())
        d_in = (
            len(BASE_PLAYER_PREFIXES) * per_player
            + CONTROLLER_GROUP_COUNT * cfg.arch.action_embed_dim
            + 2 * cfg.arch.char_dim
            + cfg.arch.stage_dim
        )
        self.item_type_emb = nn.Embedding(ITEM_CAT_VOCABS["type"], cfg.arch.item_type_dim)
        self.item_state_emb = nn.Embedding(ITEM_CAT_VOCABS["state"], cfg.arch.item_state_dim)
        slot_width = cfg.arch.item_type_dim + cfg.arch.item_state_dim + 2 * len(ITEM_FLOATS) + 1
        self.item_encoder = SwiGLU(slot_width, cfg.arch.item_hidden_dim, cfg.arch.item_dim)
        d_in += cfg.arch.item_dim
        self.observation_encoder = nn.Linear(d_in, cfg.arch.d_model)
        self.player_embedding = nn.Embedding(
            cfg.player_vocab_size,
            cfg.arch.player_embed_dim,
            padding_idx=MASKED_PLAYER_ID,
        )
        self.player_projection = nn.Linear(cfg.arch.player_embed_dim, cfg.arch.d_model, bias=False)
        code_payload = b"" if vocabulary is None else vocabulary_buffer(vocabulary)
        if vocabulary is not None and (
            vocabulary.size != cfg.player_vocab_size or vocabulary.sha256 != cfg.player_vocab_sha256
        ):
            raise ValueError("identity vocabulary does not match the frozen O59 contract")
        self.register_buffer("player_code_bytes", torch.from_numpy(np.frombuffer(code_payload, dtype=np.uint8).copy()))
        trunk_rule = depth_rule("trunk", cfg.arch.n_layers, cfg.depth_alpha)
        self.trunk = Trunk(
            TrunkConfig(
                d_model=cfg.arch.d_model,
                n_layers=cfg.arch.n_layers,
                n_heads=cfg.arch.n_heads,
                L_ctx=cfg.arch.L_ctx,
                attn_window=cfg.arch.attn_window,
                attention_backend=cfg.arch.trunk_attention_backend,
                attention_scale=trunk_rule.attention,
                mlp_scale=trunk_rule.mlp,
            )
        )
        self.temporal = CausalTemporalDecoder(cfg, self.codec)
        # V(s_t) predicts G_{t+1}, the return aligned with the next action. Keep
        # it last so the same seed preserves every policy parameter's draw.
        # Closed-loop inference never reads it.
        self.value_head = SwiGLU(cfg.arch.d_model, cfg.arch.value_hidden_dim, 1, output_bias=True)
        initialize_o51_parameters(self, cfg)
        with torch.random.fork_rng(devices=[]):
            self.temporal.return_conditioner = ReturnConditioner(cfg)
        self.return_calibration = ReturnCalibration()

    def _per_player_features(self, features: dict[str, Tensor], prefix: str) -> Tensor:
        ref = features[f"{prefix}_position_x"]
        batch, length = ref.shape
        values: list[Tensor] = []
        masks: list[Tensor] = []
        for name in FLOAT_FEATURES:
            value = features[f"{prefix}_{name}"]
            mask = features.get(f"{prefix}_{name}_mask", torch.zeros_like(ref))
            values.append(value[..., None])
            masks.append(mask[..., None])
        parts: list[Tensor] = values + masks
        for name, (vocab, _) in self.cat_specs.items():
            parts.append(self.cat_embeds[name](features[f"{prefix}_{name}"].clamp(0, vocab - 1)))
        return torch.cat(parts, dim=-1)

    def _item_features(self, features: dict[str, Tensor]) -> Tensor:
        """Pool the four projectile slots into one permutation-invariant vector.

        One shared encoder reads each slot, the presence flag gates its output, and the
        gated outputs are summed. An empty slot therefore contributes the exact zero
        vector, the pooled value does not depend on WHICH slots the live items occupy,
        and the item count stays implicit in the sum.
        """
        if ITEM_PROBE_COLUMN not in features:
            raise ValueError(
                f"the observation carries no {ITEM_PROBE_COLUMN!r} column; training needs policy-world "
                "sources and closed-loop evaluation needs projectile routing"
            )
        zeros = torch.zeros_like(features[ITEM_PROBE_COLUMN])
        slots: list[Tensor] = []
        presence: list[Tensor] = []
        for slot in range(ITEM_SLOTS):
            # The stored type is peppi's raw u16 item id, so the clamp lands every id at
            # or above the last row on that row, which is the unknown projectile.
            type_ids = features[item_column(slot, "type")].clamp(0, self.item_type_emb.num_embeddings - 1)
            state_ids = features[item_column(slot, "state")].clamp(0, self.item_state_emb.num_embeddings - 1)
            masks = {name: features.get(f"{item_column(slot, name)}_mask", zeros) for name in ITEM_FLOATS}
            live = 1.0 - masks[ITEM_PRESENCE_SUFFIX]
            parts = [self.item_type_emb(type_ids), self.item_state_emb(state_ids)]
            parts += [features[item_column(slot, name)][..., None] for name in ITEM_FLOATS]
            parts += [masks[name][..., None] for name in ITEM_FLOATS]
            parts.append(live[..., None])
            slots.append(torch.cat(parts, dim=-1))
            presence.append(live)
        encoded = self.item_encoder(torch.stack(slots, dim=-2))
        return (encoded * torch.stack(presence, dim=-1)[..., None]).sum(dim=-2)

    def context_tokens(self, features: dict[str, Tensor], action_indices: Tensor | None = None) -> Tensor:
        if "opp_player_id" in features:
            raise ValueError("opponent identity must never enter the model")
        if "ego_player_id" not in features:
            raise KeyError("context is missing ego_player_id")
        if action_indices is None:
            action_indices = self.codec.quantize(stack_actions(features))
        parts = [self._per_player_features(features, prefix) for prefix in BASE_PLAYER_PREFIXES]
        parts.append(self.codec.embed_frame(action_indices))
        parts.append(self.char_emb(features["ego_character"].clamp(0, self.char_emb.num_embeddings - 1)))
        parts.append(self.char_emb(features["opp_character"].clamp(0, self.char_emb.num_embeddings - 1)))
        parts.append(self.stage_emb(features["stage"].clamp(0, self.stage_emb.num_embeddings - 1)))
        parts.append(self._item_features(features))
        observation = self.observation_encoder(torch.cat(parts, dim=-1))
        player_ids = features["ego_player_id"].clamp(0, self.player_embedding.num_embeddings - 1)
        return observation + self.player_projection(self.player_embedding(player_ids))

    def forward(self, features: dict[str, Tensor], ctx_pad: Tensor, action_indices: Tensor | None = None) -> Tensor:
        return self.trunk(self.context_tokens(features, action_indices), ctx_pad)

    def forward_dense(
        self,
        features: dict[str, Tensor],
        ctx_pad: Tensor,
        action_indices: Tensor | None = None,
    ) -> Tensor:
        """Run the shared model weights through the dense inference trunk."""
        return self.trunk.forward_dense(self.context_tokens(features, action_indices), ctx_pad)


def center_class_logits(logits: Tensor) -> Tensor:
    """Remove the softmax-invariant common mode from each class group."""
    return logits - logits.mean(dim=-1, keepdim=True)


def mup_readout_std(fan_in: int, base_fan_in: int) -> float:
    if fan_in < 1 or base_fan_in < 1:
        raise ValueError("readout fan-ins must be positive")
    return math.sqrt(base_fan_in) / fan_in


def _final_readouts(model: GPT) -> tuple[tuple[nn.Linear, int], ...]:
    action = tuple((cast(nn.Linear, model.temporal.outputs[name].down), 128) for name in CONTROLLER_GROUP_NAMES)
    trunk_skip = tuple(
        (cast(NonlinearActionHead, model.temporal.trunk_outputs[name]).down, 128) for name in CONTROLLER_GROUP_NAMES
    )
    return (*action, *trunk_skip, (model.value_head.down, 128))


def initialize_o51_parameters(model: GPT, cfg: TrainConfig) -> None:
    """Apply O51's selected hidden and μP-normal readout initialization."""
    final_modules = {id(module) for module, _ in _final_readouts(model)}
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if id(module) in final_modules:
                continue
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=cfg.hidden_std_multiplier / math.sqrt(module.weight.shape[1]),
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=cfg.hidden_std_multiplier / math.sqrt(module.embedding_dim),
            )
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
    for module, base_fan_in in _final_readouts(model):
        nn.init.normal_(module.weight, mean=0.0, std=mup_readout_std(module.in_features, base_fan_in))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    nn.init.zeros_(model.player_projection.weight)
