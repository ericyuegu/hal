"""Dense autoregressive action chunks with training-time RTC.

This experiment keeps O43's observation trunk, historical controller codec,
optimizer, data, and evaluation suites.  Its action decoder uses the same
parameters over a dense 20-frame chain.  During training, each valid context
prefix samples an inference delay and receives loss only on the corresponding
action postfix.  During evaluation, the decoder forces the unexecuted prefix
from the previous plan through its causal cache, then samples the postfix.

Run:
    uv run experiments/046_ar_rtc.py
    uv run experiments/046_ar_rtc.py --cfg.training-delay-frames 0
    uv run experiments/046_ar_rtc.py --eval runs/<run>/final.pt --eval-delay-frames 2
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import math
import sys
import time
from collections.abc import Callable
from collections.abc import Iterator
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any
from typing import Literal

import numpy as np
import torch
import torch._inductor.config
import torch.nn.functional as F
import tyro
from torch import Tensor

import wandb
from hal.data.feature_stats import FeatureStats
from hal.eval.self_play import benchmark_checkpoint as benchmark_self_play
from hal.training.checkpoints import download_latest
from hal.training.checkpoints import load_for_resume
from hal.training.ego_stats import load_consolidated_stats
from hal.training.features import A_DIM
from hal.training.features import Context
from hal.training.features import stack_actions


def _load_o43() -> ModuleType:
    path = Path(__file__).with_name("043_legacy_codec.py")
    name = "_hal_experiment_043_for_046"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_o43 = _load_o43()
DEVICE = _o43.DEVICE
_O43_TRAIN_CONFIG = _o43.TrainConfig

PREDICTION_HORIZON_FRAMES = 20
ACTION_OFFSETS_FRAMES = tuple(range(1, PREDICTION_HORIZON_FRAMES + 1))
RTC_OBJECTIVE_VERSION = 1


def _disable_triton_pointwise_autotuning() -> None:
    # The first B32 evaluation graph failed in this autotuning path after
    # compiled training on an L40S.
    torch._inductor.config.triton.autotune_pointwise = False


@dataclass
class TrainConfig:
    """O43 training configuration with explicit RTC timing."""

    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    attn_window: int = 0
    require_flex: bool = False
    L_ctx: int = 128

    decoder_arch_version: int = 4
    codec_version: int = 2
    temporal_d_model: int = 128
    temporal_layers: int = 2
    temporal_heads: int = 4
    temporal_ff_dim: int = 256
    group_head_dim: int = 256
    action_embed_dim: int = 16
    offset_embed_dim: int = 16
    aux_loss_weight: float = 1.0
    group_order: tuple[str, ...] = _o43.GROUP_ORDER

    action_vocab: int = 1024
    action_state_embed_dim: int = 48
    char_vocab: int = 32
    char_dim: int = 8
    stage_vocab: int = 32
    stage_dim: int = 4
    observation_bundle: str = "base"

    prediction_horizon_frames: int = PREDICTION_HORIZON_FRAMES
    training_delay_frames: tuple[int, ...] = (0, 1, 2, 3, 4)
    inference_delay_frames: int = 4
    execution_stride_frames: int = 4
    decode_temp: float = 1.0
    inference_mode: str = "compiled"
    inference_buckets: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    compiled_inference_bucket: int | None = None

    seed: int = 0
    eval_seed: int = 0
    batch_size: int = 512
    grad_accum_steps: int = 1
    muon_lr: float = 0.02
    adam_lr: float = 8.5e-4
    muon_weight_decay: float = 0.01
    adam_weight_decay: float = 0.01
    lr_floor_ratio: float = 1e-5 / 8.5e-4
    warmup_steps: int = 500
    max_steps: int = 16_384
    amp_dtype: str = "bfloat16"
    allow_tf32: bool = True
    compile_trunk: bool = True
    compile_temporal: bool = True

    wandb_log_code: bool = True
    wandb_grad_every: int = 1024
    val_every: int = 1024
    val_n_samples: int = 1192
    val_batch_size: int = 128
    ckpt_every: int = 1024
    eval_every: int = 4096
    eval_max_frames: int = 7200
    eval_n_matchups: int = 32
    final_eval_n_matchups: int = 96
    final_diag_n_matchups: int = 0
    eval_max_parallel: int | None = 32

    data_root: str = "data/processed/ranked-anonymized-1/mds-policy-v7"
    train_replay_paths: str | None = None
    replay_format: Literal["policy", "policy-world"] = "policy"
    val_data_root: str = "data/processed/ranked-anonymized-1/mds-policy-v7"
    val_replay_format: Literal["policy", "policy-world"] = "policy"
    compact_data: bool = True
    mds_schema_version: int = 7
    cache_limit_gb: int = 160
    shuffle_block_size: int = 2000
    predownload: int = 512
    windows_per_replay: int = 4
    reservoir_capacity: int = 4096
    val_split: str = "val"
    num_workers: int = 16
    prefetch_factor: int = 2
    prefetch_batches: int = 4
    push_to_r2: bool = True
    system_metrics_every: int = 25
    system_metrics_interval_s: float = 5.0
    process_metrics_interval_s: float = 30.0
    cache_metrics_interval_s: float = 30.0

    # These compatibility properties let O43's unchanged training loop consume
    # O46 without exposing O43's ambiguous timing names in saved configuration.
    @property
    def head_offsets(self) -> tuple[int, ...]:
        return ACTION_OFFSETS_FRAMES

    @property
    def sample_chunk_length(self) -> int:
        return self.prediction_horizon_frames

    @property
    def next_frame_loss_share(self) -> None:
        return None

    @property
    def exec_horizon(self) -> int:
        return self.execution_stride_frames

    @property
    def final_diag_exec_horizon(self) -> int:
        return self.execution_stride_frames


def _as_o43_config(cfg: TrainConfig) -> Any:
    values = {item.name: getattr(cfg, item.name) for item in fields(_O43_TRAIN_CONFIG)}
    # O46 changes decoder semantics but not parameter geometry.  Use values that
    # pass O43's frozen validation for fields O46 validates under clearer names.
    values.update(
        decoder_arch_version=3,
        exec_horizon=4,
        final_diag_exec_horizon=4,
    )
    return _O43_TRAIN_CONFIG(**values)


_O43_VALIDATE_CONFIG = _o43.validate_config


def validate_config(cfg: TrainConfig) -> None:
    if cfg.decoder_arch_version != 4:
        raise ValueError("experiment 046 requires decoder_arch_version=4")
    if cfg.prediction_horizon_frames != PREDICTION_HORIZON_FRAMES:
        raise ValueError(f"experiment 046 fixes prediction_horizon_frames at {PREDICTION_HORIZON_FRAMES}")
    delays = tuple(cfg.training_delay_frames)
    if any(not isinstance(delay, int) or isinstance(delay, bool) for delay in delays):
        raise ValueError("training_delay_frames must contain integers")
    if delays != tuple(sorted(set(delays))) or not delays or delays[0] != 0:
        raise ValueError("training_delay_frames must be sorted, unique, non-empty, and start at zero")
    if any(delay < 0 or delay > cfg.prediction_horizon_frames // 2 for delay in delays):
        raise ValueError("training delays must satisfy 0 <= d <= prediction_horizon_frames / 2")
    if not isinstance(cfg.inference_delay_frames, int) or isinstance(cfg.inference_delay_frames, bool):
        raise ValueError("inference_delay_frames must be an integer")
    if not isinstance(cfg.execution_stride_frames, int) or isinstance(cfg.execution_stride_frames, bool):
        raise ValueError("execution_stride_frames must be an integer")
    expected_stride = max(cfg.inference_delay_frames, 1)
    if cfg.execution_stride_frames != expected_stride:
        raise ValueError(
            "paper-style RTC requires execution_stride_frames=max(inference_delay_frames, 1), "
            f"got d={cfg.inference_delay_frames}, s={cfg.execution_stride_frames}"
        )
    if not 0 <= cfg.inference_delay_frames <= cfg.prediction_horizon_frames - cfg.execution_stride_frames:
        raise ValueError("inference delay must satisfy 0 <= d <= prediction_horizon_frames - execution_stride_frames")
    if cfg.aux_loss_weight != 1.0:
        raise ValueError("experiment 046 fixes aux_loss_weight at 1; the RTC objective does not use it")
    if cfg.final_diag_n_matchups != 0:
        raise ValueError("use --eval-delay-frames for the RTC delay sweep; final_diag_n_matchups must remain zero")
    _O43_VALIDATE_CONFIG(_as_o43_config(cfg))


class CausalTemporalDecoder(_o43.CausalTemporalDecoder):
    """O43's causal decoder with a hard, externally committed prefix."""

    def sample_indices(
        self,
        hidden: Tensor,
        observed: Tensor,
        offsets: tuple[int, ...],
        *,
        committed: Tensor | None = None,
        argmax: bool,
        uniforms: Tensor | None = None,
        gen: torch.Generator | None = None,
    ) -> Tensor:
        expected_offsets = tuple(range(1, len(offsets) + 1))
        if offsets != expected_offsets or not offsets or offsets[-1] > len(self.head_offsets):
            raise ValueError(f"live decode requires a dense prefix of {self.head_offsets}")
        batch = hidden.shape[0]
        delay = 0 if committed is None else committed.shape[1]
        if committed is not None and committed.shape != (batch, delay, _o43.N_GROUPS):
            raise ValueError("committed indices must be [batch, delay, groups]")
        if delay > len(offsets):
            raise ValueError("committed prefix is longer than the action chunk")
        if uniforms is not None and uniforms.shape != (len(offsets), _o43.N_GROUPS, batch):
            raise ValueError("uniform table must be [frames, groups, batch]")

        trunk = _o43.decoder_rmsnorm(hidden[:, -1])
        trunk_logits = {name: self.trunk_outputs[name](trunk) for name in _o43.GROUP_NAMES}
        previous = observed
        caches: list[tuple[Tensor, Tensor] | None] = [None] * len(self.blocks)
        frames: list[Tensor] = []
        for depth, offset in enumerate(offsets):
            offset_tensor = torch.full((batch,), offset, device=hidden.device, dtype=torch.long)
            state = self.token_projection(
                torch.cat((trunk, self.codec.embed_frame(previous), self.offset_embedding(offset_tensor)), dim=-1)
            )
            next_caches = []
            for block, past in zip(self.blocks, caches, strict=True):
                state, present = block.forward_step(state, past)
                next_caches.append(present)
            caches = next_caches
            state = _o43.decoder_rmsnorm(state)

            if depth < delay:
                indices = committed[:, depth]
            else:
                embedded: dict[str, Tensor] = {}
                picks: dict[str, Tensor] = {}
                for name in _o43.GROUP_ORDER:
                    logits = self.codec.mask_logits(
                        name,
                        self.outputs[name](self.group_features(state, name, embedded)) + trunk_logits[name],
                    )
                    if name == "buttons":
                        logits = logits.masked_fill(self.codec.button_mask(picks["triggers"]), float("-inf"))
                    group = _o43.GROUP_INDEX[name]
                    uniform = None if uniforms is None else uniforms[depth, group]
                    valid_logits = logits[..., : _o43.LEGACY_GROUP_VOCABS[group]]
                    pick = _o43.sample_categorical(valid_logits, argmax=argmax, uniform=uniform, gen=gen)
                    picks[name] = pick
                    embedded[name] = self.codec.group_embedding(name, pick)
                indices = torch.stack([picks[name] for name in _o43.GROUP_NAMES], dim=-1)
            frames.append(indices)
            previous = indices
        return torch.stack(frames, dim=1)

    def rollout_conditioned_logits(
        self,
        hidden: Tensor,
        observed: Tensor,
        committed: Tensor | None = None,
    ) -> tuple[list[dict[str, Tensor]], Tensor]:
        """Greedily roll out after a fixed prefix for validation diagnostics."""
        if committed is None:
            return super().rollout_conditioned_logits(hidden, observed)
        batch, delay = committed.shape[:2]
        if committed.shape != (batch, delay, _o43.N_GROUPS) or batch != hidden.shape[0]:
            raise ValueError("committed indices must be [batch, delay, groups]")
        if delay >= len(self.head_offsets):
            raise ValueError("validation needs at least one postfix action")

        trunk = _o43.decoder_rmsnorm(hidden[:, -1])
        trunk_logits = {name: self.trunk_outputs[name](trunk) for name in _o43.GROUP_NAMES}
        previous = observed
        caches: list[tuple[Tensor, Tensor] | None] = [None] * len(self.blocks)
        frames: list[Tensor] = []
        all_logits: list[dict[str, Tensor]] = []
        for depth, offset in enumerate(self.head_offsets):
            offset_tensor = torch.full((batch,), offset, device=hidden.device, dtype=torch.long)
            state = self.token_projection(
                torch.cat((trunk, self.codec.embed_frame(previous), self.offset_embedding(offset_tensor)), dim=-1)
            )
            next_caches = []
            for block, past in zip(self.blocks, caches, strict=True):
                state, present = block.forward_step(state, past)
                next_caches.append(present)
            caches = next_caches
            state = _o43.decoder_rmsnorm(state)

            embedded: dict[str, Tensor] = {}
            picks: dict[str, Tensor] = {}
            frame_logits: dict[str, Tensor] = {}
            for name in _o43.GROUP_ORDER:
                logits = self.codec.mask_logits(
                    name,
                    self.outputs[name](self.group_features(state, name, embedded)) + trunk_logits[name],
                )
                if name == "buttons":
                    logits = logits.masked_fill(self.codec.button_mask(picks["triggers"]), float("-inf"))
                group = _o43.GROUP_INDEX[name]
                pick = committed[:, depth, group] if depth < delay else logits.argmax(dim=-1)
                frame_logits[name] = logits
                picks[name] = pick
                embedded[name] = self.codec.group_embedding(name, pick)
            previous = torch.stack([picks[name] for name in _o43.GROUP_NAMES], dim=-1)
            frames.append(previous)
            all_logits.append(frame_logits)
        return all_logits, torch.stack(frames, dim=1)


class GPT(_o43.GPT):
    """O43 parameter geometry evaluated over all 20 consecutive offsets."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__(_as_o43_config(cfg))
        self.cfg = cfg


def _validate_objective_delays(delays: tuple[int, ...], horizon: int) -> None:
    if any(not isinstance(delay, int) or isinstance(delay, bool) for delay in delays):
        raise ValueError("training delays must contain integers")
    if not delays or delays != tuple(sorted(set(delays))) or any(delay < 0 or delay >= horizon for delay in delays):
        raise ValueError(f"training delays must be sorted, unique, non-empty, and lie in [0, {horizon})")


_ACTIVE_TRAINING_DELAYS: tuple[int, ...] | None = None


@contextlib.contextmanager
def _training_delays(delays: tuple[int, ...]) -> Iterator[None]:
    global _ACTIVE_TRAINING_DELAYS
    previous = _ACTIVE_TRAINING_DELAYS
    _ACTIVE_TRAINING_DELAYS = tuple(delays)
    try:
        yield
    finally:
        _ACTIVE_TRAINING_DELAYS = previous


def _joint_objective(
    joint_nll: Tensor,
    valid_prefixes: int,
    *,
    aux_loss_weight: float,
    next_frame_loss_share: float | None,
) -> Tensor:
    """Sample a delay per prefix and average that prefix's postfix NLL."""
    del aux_loss_weight, next_frame_loss_share
    if joint_nll.ndim != 2 or valid_prefixes <= 0:
        raise ValueError("joint_nll must be [valid prefixes, horizon]")
    delays = (0,) if _ACTIVE_TRAINING_DELAYS is None else _ACTIVE_TRAINING_DELAYS
    _validate_objective_delays(delays, joint_nll.shape[1])
    choices = torch.randint(len(delays), (joint_nll.shape[0],), device=joint_nll.device)
    delay_values = torch.tensor(delays, device=joint_nll.device)[choices]
    offsets = torch.arange(joint_nll.shape[1], device=joint_nll.device)
    postfix = offsets[None] >= delay_values[:, None]
    per_prefix = (joint_nll * postfix).sum(dim=1) / postfix.sum(dim=1)
    # O43 calls this once per micro-batch but supplies the valid-prefix count for
    # the whole gradient accumulation.  Summing here preserves that contract.
    return per_prefix.sum() / valid_prefixes


ActionLoss = _o43.ActionLoss


def objective(parts: ActionLoss, training_delay_frames: tuple[int, ...] = (0, 1, 2, 3, 4)) -> Tensor:
    """Return one sampled-delay RTC objective in nats."""
    with _training_delays(training_delay_frames):
        return _joint_objective(
            parts.nll.sum(dim=-1),
            parts.nll.shape[0],
            aux_loss_weight=1.0,
            next_frame_loss_share=None,
        )


def nll_mean_metrics(
    mean_nll: Tensor,
    offsets: tuple[int, ...],
    *,
    aux_loss_weight: float = 1.0,
    next_frame_loss_share: float | None = None,
) -> dict[str, float]:
    """Report dense and delay-conditional teacher-forced NLL in bits."""
    del aux_loss_weight, next_frame_loss_share
    if mean_nll.shape != (len(offsets), _o43.N_GROUPS):
        raise ValueError(f"mean NLL has shape {tuple(mean_nll.shape)}")
    delays = (0,) if _ACTIVE_TRAINING_DELAYS is None else _ACTIVE_TRAINING_DELAYS
    _validate_objective_delays(delays, len(offsets))
    joint = mean_nll.sum(dim=-1) / math.log(2.0)
    conditional = {delay: joint[delay:].mean() for delay in delays}
    out = {
        "loss": float(torch.stack(tuple(conditional.values())).mean()),
        "dense_nll": float(joint.mean()),
    }
    for delay, value in conditional.items():
        out[f"conditional_nll_d{delay:02d}"] = float(value)
        out[f"first_postfix_nll_d{delay:02d}"] = float(joint[delay])
    for depth, offset in enumerate(offsets):
        out[f"nll_o{offset:02d}"] = float(joint[depth])
        for group, name in enumerate(_o43.GROUP_NAMES):
            out[f"nll_o{offset:02d}_{name}"] = float(mean_nll[depth, group] / math.log(2.0))
    return out


def nll_metrics(
    nll: Tensor,
    offsets: tuple[int, ...],
    *,
    aux_loss_weight: float = 1.0,
    next_frame_loss_share: float | None = None,
) -> dict[str, float]:
    return nll_mean_metrics(
        nll.mean(dim=0),
        offsets,
        aux_loss_weight=aux_loss_weight,
        next_frame_loss_share=next_frame_loss_share,
    )


_O43_VAL_METRICS = _o43.val_metrics


@torch.no_grad()
def val_metrics(model: GPT, batches: list[Any], cfg: TrainConfig) -> dict[str, float]:
    with _training_delays(cfg.training_delay_frames):
        values = dict(_O43_VAL_METRICS(model, batches, cfg))

    delay = cfg.inference_delay_frames
    rollout_nll = 0.0
    rollout_frames = 0
    first_exact = transition_correct = transition_count = hold_correct = hold_count = 0
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    try:
        for cpu_batch in batches:
            batch = cpu_batch.to(device)
            history, targets, _ = _o43.prepared_targets(model, batch)
            with _o43.amp_context(cfg, device):
                hidden = model(batch.context.features, batch.context.ctx_pad, history)
                target = targets[:, -1]
                committed = target[:, :delay]
                logits, sampled = model.temporal.rollout_conditioned_logits(
                    hidden,
                    history[:, -1],
                    committed,
                )
            for depth in range(delay, cfg.prediction_horizon_frames):
                for group, name in enumerate(_o43.GROUP_NAMES):
                    rollout_nll += float(
                        F.cross_entropy(logits[depth][name].float(), target[:, depth, group], reduction="sum")
                    )
            rollout_frames += target.shape[0] * (cfg.prediction_horizon_frames - delay)
            first_matches = (sampled[:, delay] == target[:, delay]).all(dim=-1)
            first_exact += int(first_matches.sum())
            previous = history[:, -1] if delay == 0 else target[:, delay - 1]
            changed = (target[:, delay] != previous).any(dim=-1)
            transition_correct += int((first_matches & changed).sum())
            transition_count += int(changed.sum())
            hold_correct += int((first_matches & ~changed).sum())
            hold_count += int((~changed).sum())
    finally:
        model.train(was_training)

    suffix_bits = rollout_nll / max(rollout_frames, 1) / math.log(2.0)
    teacher_key = f"conditional_nll_d{delay:02d}"
    values[f"rtc_rollout_nll_d{delay:02d}"] = suffix_bits
    if teacher_key in values:
        values[f"rtc_exposure_gap_d{delay:02d}"] = suffix_bits - values[teacher_key]
    values[f"rtc_first_postfix_exact_d{delay:02d}"] = first_exact / max(sum(b.target.shape[0] for b in batches), 1)
    values[f"rtc_boundary_transition_acc_d{delay:02d}"] = transition_correct / max(transition_count, 1)
    values[f"rtc_boundary_hold_acc_d{delay:02d}"] = hold_correct / max(hold_count, 1)
    return values


class RTCDecodeTelemetry(_o43.DecodeTelemetry):
    def metrics(self) -> dict[str, float]:
        values = super().metrics()
        values["decode_predicted_frames_per_s"] = values.pop("decode_executed_frames_per_s")
        return values


class BF16Inference(_o43.BF16Inference):
    """O43's compiled trunk with a fixed-prefix, full-horizon decoder."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _disable_triton_pointwise_autotuning()
        super().__init__(*args, **kwargs)

    def _decoder(self, bucket: int, delay: int) -> Callable:
        key = (bucket, delay)
        if key not in self._decoders:
            offsets = self.model.head_offsets

            def fn(hidden: Tensor, observed: Tensor, committed: Tensor, uniforms: Tensor) -> Tensor:
                return self.model.temporal.sample_indices(
                    hidden,
                    observed,
                    offsets,
                    committed=committed,
                    argmax=False,
                    uniforms=uniforms,
                )

            self._decoders[key] = torch.compile(fn, dynamic=False, mode=self.compile_mode) if self.compiled else fn
        return self._decoders[key]

    @torch.no_grad()
    def decode(
        self,
        ctx: Context,
        committed: np.ndarray | Tensor | None = None,
        *,
        streams: Any | None = None,
        argmax: bool = False,
        gen: torch.Generator | None = None,
    ) -> Tensor:
        rows = ctx.ctx_pad.shape[0]
        delay = 0 if committed is None else int(committed.shape[1])
        if committed is not None and tuple(committed.shape) != (rows, delay, A_DIM):
            raise ValueError("committed actions must be [batch, delay, action_dim]")
        if committed is not None and delay != self.cfg.inference_delay_frames:
            raise ValueError(
                f"committed prefix has {delay} frames, expected configured delay {self.cfg.inference_delay_frames}"
            )
        bucket = self._bucket(rows)
        padded = _o43.canonical_context(_o43._pad_context(ctx, bucket), self.cfg.observation_bundle)
        observed = self.model.codec.quantize(stack_actions(padded.features))

        if committed is None:
            committed_native = torch.empty(rows, 0, A_DIM, device=ctx.ctx_pad.device)
        else:
            committed_native = torch.as_tensor(committed, device=ctx.ctx_pad.device, dtype=torch.float32)
        if rows < bucket:
            committed_padded = torch.cat(
                (
                    committed_native,
                    torch.zeros(bucket - rows, delay, A_DIM, device=committed_native.device),
                )
            )
        else:
            committed_padded = committed_native
        if delay:
            committed_indices = self.model.codec.quantize(committed_padded)
        else:
            committed_indices = torch.empty(
                bucket,
                0,
                _o43.N_GROUPS,
                dtype=torch.long,
                device=ctx.ctx_pad.device,
            )

        if streams is not None:
            streams.begin(ctx)
        uniform_parts: list[Tensor] = []
        for _ in self.model.head_offsets:
            groups = []
            for name in _o43.GROUP_NAMES:
                if streams is None:
                    real = torch.rand(rows, device=ctx.ctx_pad.device, generator=gen)
                else:
                    real = streams.uniforms(name)
                groups.append(F.pad(real, (0, bucket - rows), value=0.5))
            uniform_parts.append(torch.stack(groups))
        uniforms = torch.stack(uniform_parts)

        if self.uses_cuda_graphs:
            torch.compiler.cudagraph_mark_step_begin()
        with _o43.amp_context(self.cfg, ctx.ctx_pad.device):
            hidden = self._trunk(bucket)(padded.features, padded.ctx_pad, observed)
            if argmax:
                indices = self.model.temporal.sample_indices(
                    hidden,
                    observed[:, -1],
                    self.model.head_offsets,
                    committed=committed_indices,
                    argmax=True,
                )
            else:
                indices = self._decoder(bucket, delay)(hidden, observed[:, -1], committed_indices, uniforms)
        actions = self.model.codec.dequantize(indices[:rows])
        if delay:
            actions = torch.cat((committed_native.to(actions.dtype), actions[:, delay:]), dim=1)
        return actions


@torch.no_grad()
def decode_chunk(
    model: GPT,
    ctx: Context,
    committed: np.ndarray | Tensor | None = None,
    *,
    argmax: bool = False,
    gen: torch.Generator | None = None,
) -> Tensor:
    cfg = replace(model.cfg, inference_mode="eager")
    return BF16Inference(model, cfg, compiled=False).decode(ctx, committed, argmax=argmax, gen=gen)


def make_policy(
    model: GPT,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    exec_horizon: int | None = None,
    decode_seed: int | None = None,
    inference: BF16Inference | None = None,
    telemetry: RTCDecodeTelemetry | None = None,
    device: str = DEVICE,
) -> Any:
    stride = cfg.execution_stride_frames if exec_horizon is None else exec_horizon
    if stride != max(cfg.inference_delay_frames, 1):
        raise ValueError("execution stride must equal max(inference delay, 1)")
    engine = BF16Inference(model, cfg) if inference is None else inference
    if engine.model is not model:
        raise ValueError("the supplied inference engine must own the policy model")
    streams = None if decode_seed is None else _o43.SlotGroupRandom(decode_seed)
    generator = None if decode_seed is None else torch.Generator(device=device).manual_seed(decode_seed)

    @torch.no_grad()
    def predict(ctx: Context, committed: np.ndarray | None) -> np.ndarray:
        started = time.perf_counter()
        result = engine.decode(ctx, committed, streams=streams, gen=generator).cpu().numpy()
        if telemetry is not None:
            telemetry.record(
                rows=ctx.ctx_pad.shape[0],
                horizon=cfg.prediction_horizon_frames,
                seconds=time.perf_counter() - started,
            )
        return result

    v6 = cfg.observation_bundle == "v6_lean"
    return _o43.RecedingHorizon(
        predict_chunk=predict,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=cfg.prediction_horizon_frames,
        s=stride,
        d=cfg.inference_delay_frames,
        device=device,
        float_dtype=next(model.parameters()).dtype,
        extra=_o43.V6_PLAYER_COLUMNS if v6 else None,
        projection=None if v6 else _o43.BASE_ACTION_PROJECTION,
    )


@dataclass(frozen=True, slots=True)
class EvalProtocol:
    suite: str
    fixed_ego_character: int | None
    n_matchups: int
    allowed_cpus: int
    hardware_wave_bucket: int
    max_parallel: int
    max_frames: int
    seed: int
    cpu_level: int
    ego_port: int
    seed_stage: int
    matchup_schedule_sha256: str
    oriented_pairs: int
    ego_characters: int
    cpu_characters: int
    prediction_horizon_frames: int
    training_delay_frames: tuple[int, ...]
    inference_delay_frames: int
    execution_stride_frames: int
    action_offsets_frames: tuple[int, ...]
    dtype: str
    inference_mode: str
    inference_compile_mode: str
    compiled_inference_bucket: int
    checkpoint_sha256: str
    bootstrap_resamples: int = _o43.BOOTSTRAP_RESAMPLES
    start_retries: int = _o43.DEFAULT_START_RETRIES


_O43_EVAL_PROTOCOL = _o43._eval_protocol


def _eval_protocol(
    cfg: TrainConfig,
    model: GPT,
    *,
    n_matchups: int,
    exec_horizon: int,
    checkpoint_sha256: str,
    inference_compile_mode: str = "reduce-overhead",
    fixed_ego_character: Any | None = None,
) -> EvalProtocol:
    if exec_horizon != cfg.execution_stride_frames:
        raise ValueError("evaluation stride must match the configured RTC stride")
    legacy = _O43_EVAL_PROTOCOL(
        cfg,
        model,
        n_matchups=n_matchups,
        exec_horizon=exec_horizon,
        checkpoint_sha256=checkpoint_sha256,
        inference_compile_mode=inference_compile_mode,
        fixed_ego_character=fixed_ego_character,
    )
    values = asdict(legacy)
    values.pop("exec_horizon")
    values.update(
        prediction_horizon_frames=cfg.prediction_horizon_frames,
        training_delay_frames=cfg.training_delay_frames,
        inference_delay_frames=cfg.inference_delay_frames,
        execution_stride_frames=cfg.execution_stride_frames,
        action_offsets_frames=ACTION_OFFSETS_FRAMES,
    )
    return EvalProtocol(**values)


def _write_eval_evidence(replay_dir: Path, rows: list[Any], metrics: dict[str, float], protocol: EvalProtocol) -> None:
    metrics.pop("exec_horizon", None)
    metrics.update(
        prediction_horizon_frames=float(protocol.prediction_horizon_frames),
        inference_delay_frames=float(protocol.inference_delay_frames),
        execution_stride_frames=float(protocol.execution_stride_frames),
    )
    replay_dir.mkdir(parents=True, exist_ok=True)
    payloads = (
        (
            replay_dir / "match_rows.json",
            {"schema_version": 8, "protocol": asdict(protocol), "rows": [row.as_dict() for row in rows]},
        ),
        (replay_dir / "metrics.json", metrics),
    )
    for path, payload in payloads:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True))
        temporary.replace(path)


_O43_EVAL_VS_CPU = _o43.eval_vs_cpu


def eval_vs_cpu(
    model: GPT,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    n_matchups: int,
    replay_dir: Path,
    checkpoint_sha256: str = "unavailable",
    inference: BF16Inference | None = None,
    fixed_ego_character: Any | None = None,
) -> dict[str, float]:
    return _O43_EVAL_VS_CPU(
        model,
        stats,
        cfg,
        n_matchups=n_matchups,
        replay_dir=replay_dir,
        exec_horizon=cfg.execution_stride_frames,
        checkpoint_sha256=checkpoint_sha256,
        inference=inference,
        fixed_ego_character=fixed_ego_character,
    )


def eval_suites(
    model: GPT,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    n_matchups: int,
    replay_dir: Path,
    checkpoint_sha256: str,
    inference: BF16Inference,
    **legacy: Any,
) -> dict[str, dict[str, float]]:
    if legacy and legacy != {"exec_horizon": cfg.execution_stride_frames}:
        raise TypeError(f"unexpected evaluation arguments: {legacy}")
    return {
        name: eval_vs_cpu(
            model,
            stats,
            cfg,
            n_matchups=n_matchups,
            replay_dir=replay_dir / name,
            checkpoint_sha256=checkpoint_sha256,
            inference=inference,
            fixed_ego_character=character,
        )
        for name, character in _o43._EVAL_SUITES
    }


def model_tag(cfg: TrainConfig) -> str:
    delays = "-".join(map(str, cfg.training_delay_frames))
    return (
        f"ar046-rtc-v{cfg.codec_version}-d{cfg.d_model}-L{cfg.n_layers}-h{cfg.n_heads}-Lc{cfg.L_ctx}-"
        f"t{cfg.temporal_d_model}x{cfg.temporal_layers}-H{cfg.prediction_horizon_frames}-td{delays}-"
        f"d{cfg.inference_delay_frames}-s{cfg.execution_stride_frames}-{cfg.observation_bundle}"
    )


def _init_wandb(cfg: TrainConfig, run_name: str, resume_state: dict | None) -> None:
    wandb.init(
        project="hal",
        name=run_name,
        id=None if resume_state is None else resume_state.get("wandb_id"),
        resume="allow" if resume_state is not None else None,
        tags=["046", "gpt", "autoregressive", "training-time-rtc", "legacy-codec"],
        config=asdict(cfg),
        settings=wandb.Settings(x_stats_sampling_interval=5.0, x_stats_track_process_tree=True),
    )
    if wandb.run is None:
        return
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")
    wandb.run.summary["architecture/treatment"] = "O43 parameter geometry; dense 20-frame AR chain"
    wandb.run.summary["objective/semantics"] = "sample one delay per context prefix; loss on postfix only"
    wandb.run.summary["objective/training_delay_frames"] = list(cfg.training_delay_frames)
    wandb.run.summary["evaluation/inference_delay_frames"] = cfg.inference_delay_frames
    wandb.run.summary["evaluation/execution_stride_frames"] = cfg.execution_stride_frames
    wandb.run.summary["evaluation/suites"] = "char_matchup,fox"
    wandb.run.summary["compiler/triton_autotune_pointwise"] = False
    wandb.run.summary["training/updates"] = cfg.max_steps
    wandb.run.summary["data/nominal_samples"] = cfg.max_steps * cfg.batch_size
    wandb.run.summary["data/max_context_prefixes"] = cfg.max_steps * cfg.batch_size * cfg.L_ctx
    if cfg.wandb_log_code:
        _o43.log_wandb_code(wandb.run)


_CHECKPOINT_ARCH_FIELDS = {
    "action_offsets_frames",
    "codec_version",
    "decoder_arch_version",
    "group_order",
    "inference_delay_frames",
    "prediction_horizon_frames",
    "rtc_objective_version",
    "temporal_d_model",
    "temporal_heads",
    "temporal_layers",
    "training_delay_frames",
}


def config_from_state(values: dict[str, Any]) -> TrainConfig:
    missing = _CHECKPOINT_ARCH_FIELDS - values.keys()
    if missing:
        raise ValueError(f"checkpoint is not an experiment-046 architecture; missing {sorted(missing)}")
    if tuple(values["action_offsets_frames"]) != ACTION_OFFSETS_FRAMES:
        raise ValueError("checkpoint action offsets do not match experiment 046")
    if values["rtc_objective_version"] != RTC_OBJECTIVE_VERSION:
        raise ValueError("checkpoint RTC objective version does not match experiment 046")
    known = {item.name for item in fields(TrainConfig)}
    return TrainConfig(**{name: value for name, value in values.items() if name in known})


_O43_SAVE_CHECKPOINT = _o43.save_checkpoint


def save_checkpoint(path: Path, **kwargs: Any) -> None:
    checkpoint_config = dict(kwargs["cfg"])
    checkpoint_config["action_offsets_frames"] = ACTION_OFFSETS_FRAMES
    checkpoint_config["rtc_objective_version"] = RTC_OBJECTIVE_VERSION
    _O43_SAVE_CHECKPOINT(path, **{**kwargs, "cfg": checkpoint_config})


def load_checkpoint(path: str, *, device: str = DEVICE) -> tuple[GPT, TrainConfig, dict[str, FeatureStats], dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_state(state["cfg"])
    validate_config(cfg)
    model = GPT(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    return model, cfg, stats, state


def eval_checkpoint(
    path: str,
    *,
    inference_delay_frames: int | None = None,
    n_matchups: int | None = None,
    eager: bool = False,
    max_parallel: int | None = None,
    output_name: str | None = None,
    upload_run: str | None = None,
    backfill_wandb: bool = False,
) -> dict[str, dict[str, float]]:
    model, cfg, stats, state = load_checkpoint(path)
    delay = cfg.inference_delay_frames if inference_delay_frames is None else inference_delay_frames
    cfg = replace(
        cfg,
        inference_delay_frames=delay,
        execution_stride_frames=max(delay, 1),
        inference_mode="eager" if eager else cfg.inference_mode,
        eval_max_parallel=cfg.eval_max_parallel if max_parallel is None else max_parallel,
    )
    validate_config(cfg)
    model.cfg = cfg
    step = int(state["step"])
    matchups = cfg.final_eval_n_matchups if n_matchups is None else n_matchups
    label = f"d{delay:02d}_s{cfg.execution_stride_frames:02d}_H{cfg.prediction_horizon_frames:02d}"
    default_name = f"eval_backfill_step_{step:07d}_{label}" if upload_run else f"eval_replays_{label}"
    if output_name is not None and (Path(output_name).name != output_name or output_name in ("", ".", "..")):
        raise ValueError(f"evaluation output name must be one directory name, got {output_name!r}")
    replay_dir = Path(path).resolve().parent / (default_name if output_name is None else output_name)
    inference = BF16Inference(model, cfg)
    suites = eval_suites(
        model,
        stats,
        cfg,
        n_matchups=matchups,
        replay_dir=replay_dir,
        checkpoint_sha256=_o43._checkpoint_sha256(Path(path)),
        inference=inference,
    )
    for values in suites.values():
        _o43.require_complete_eval(values, matchups)
    if upload_run is not None:
        _o43._upload_eval_evidence(upload_run, replay_dir)
    if backfill_wandb:
        wandb_id = state.get("wandb_id")
        if not isinstance(wandb_id, str):
            raise RuntimeError("checkpoint has no W&B run id to backfill")
        _o43._backfill_eval_metrics(wandb_id, step, suites)
    print(f"[eval] step={step} {label}: {suites}", flush=True)
    return suites


_O43_TRAIN = _o43.train


def train(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    *,
    comment: str = "",
    resume_run: str | None = None,
    resume_state: dict | None = None,
) -> None:
    _disable_triton_pointwise_autotuning()
    with _training_delays(cfg.training_delay_frames):
        _O43_TRAIN(
            cfg,
            stats,
            comment=comment,
            resume_run=resume_run,
            resume_state=resume_state,
        )


def run_benchmark(cfg: TrainConfig, *, iterations: int = 20) -> dict[str, float]:
    validate_config(cfg)
    device = torch.device(DEVICE)
    model = GPT(cfg).to(device).eval()
    rows = min(32, _o43.micro_batch_size(cfg))
    ctx = _o43.synthetic_context(cfg, rows, device)
    committed = np.zeros((rows, cfg.inference_delay_frames, A_DIM), dtype=np.float32)
    eager = BF16Inference(model, replace(cfg, inference_mode="eager"), compiled=False)
    compiled = BF16Inference(model, cfg)

    def measure(engine: BF16Inference, prefix: np.ndarray | None) -> float:
        for _ in range(2):
            engine.decode(ctx, prefix)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            engine.decode(ctx, prefix)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - started) / iterations

    out: dict[str, float] = {}
    for name, prefix in (("bootstrap", None), (f"rtc_d{cfg.inference_delay_frames}", committed)):
        eager_s = measure(eager, prefix)
        compiled_s = measure(compiled, prefix)
        out[f"eager_{name}_ms"] = eager_s * 1000
        out[f"compiled_{name}_ms"] = compiled_s * 1000
        out[f"compiled_{name}_predicted_fps"] = rows * cfg.prediction_horizon_frames / compiled_s
    print(json.dumps(out, indent=2, sort_keys=True), flush=True)
    return out


# O43 is private to this module.  Replacing its globals makes its established
# training and evaluation loops call O46's variant pieces.
_o43.__file__ = __file__
_o43.CausalTemporalDecoder = CausalTemporalDecoder
_o43.TrainConfig = TrainConfig
_o43.GPT = GPT
_o43.validate_config = validate_config
_o43._joint_objective = _joint_objective
_o43.nll_mean_metrics = nll_mean_metrics
_o43.nll_metrics = nll_metrics
_o43.val_metrics = val_metrics
_o43.DecodeTelemetry = RTCDecodeTelemetry
_o43.BF16Inference = BF16Inference
_o43.decode_chunk = decode_chunk
_o43.make_policy = make_policy
_o43._eval_protocol = _eval_protocol
_o43._write_eval_evidence = _write_eval_evidence
_o43.eval_vs_cpu = eval_vs_cpu
_o43.eval_suites = eval_suites
_o43.model_tag = model_tag
_o43._init_wandb = _init_wandb
_o43.config_from_state = config_from_state
_o43.load_checkpoint = load_checkpoint
_o43.save_checkpoint = save_checkpoint

prepared_targets = _o43.prepared_targets
chunk_targets = _o43.chunk_targets
action_loss = _o43.action_loss
amp_context = _o43.amp_context
dequantize = _o43.dequantize
micro_batch_size = _o43.micro_batch_size
quantize = _o43.quantize
require_complete_eval = _o43.require_complete_eval
synthetic_context = _o43.synthetic_context

GROUP_NAMES = _o43.GROUP_NAMES
GROUP_ORDER = _o43.GROUP_ORDER
GROUP_INDEX = _o43.GROUP_INDEX
GROUP_VOCABS = _o43.GROUP_VOCABS
LEGACY_GROUP_VOCABS = _o43.LEGACY_GROUP_VOCABS
N_GROUPS = _o43.N_GROUPS
StructuredControllerCodec = _o43.StructuredControllerCodec


@dataclass
class Args:
    cfg: TrainConfig = field(default_factory=TrainConfig)
    comment: str = ""
    resume: str | None = None
    eval: str | None = None
    eval_run: str | None = None
    eval_delay_frames: int | None = None
    eval_n_matchups: int | None = None
    eval_eager: bool = False
    eval_max_parallel: int | None = None
    eval_output_name: str | None = None
    eval_backfill_wandb: bool = False
    self_play_eval: str | None = None
    self_play_matches: int = 12
    self_play_frames: int = 14_400
    self_play_eager: bool = False
    self_play_instant_match_restart: bool = False
    self_play_process_cohorts: int = 1
    self_play_cohort_sweep: bool = False
    benchmark: bool = False
    benchmark_iterations: int = 20


def _resolve_eval_checkpoint(checkpoint: str, run: str | None) -> Path:
    if run is None:
        return Path(checkpoint)
    path = download_latest(run, Path("runs") / run, name=checkpoint)
    if path is None:
        raise SystemExit(f"no {checkpoint!r} for run {run!r}")
    return path


def main(args: Args) -> None:
    if args.benchmark:
        if args.eval is not None or args.resume is not None or args.self_play_eval is not None:
            raise SystemExit("--benchmark cannot be combined with --eval, --self-play-eval, or --resume")
        run_benchmark(args.cfg, iterations=args.benchmark_iterations)
        return
    selected = sum(value is not None for value in (args.eval, args.self_play_eval, args.resume))
    if selected > 1:
        raise SystemExit("pass only one of --eval, --self-play-eval, or --resume")
    if args.eval is not None:
        checkpoint = _resolve_eval_checkpoint(args.eval, args.eval_run)
        eval_checkpoint(
            str(checkpoint),
            inference_delay_frames=args.eval_delay_frames,
            n_matchups=args.eval_n_matchups,
            eager=args.eval_eager,
            max_parallel=args.eval_max_parallel,
            output_name=args.eval_output_name,
            upload_run=args.eval_run,
            backfill_wandb=args.eval_backfill_wandb,
        )
        return
    if args.self_play_eval is not None:
        cohorts = (1, 2, 3, 4) if args.self_play_cohort_sweep else (args.self_play_process_cohorts,)
        for cohort_count in cohorts:
            benchmark_self_play(
                args.self_play_eval,
                load_checkpoint=load_checkpoint,
                make_inference=BF16Inference,
                make_policy=make_policy,
                n_matches=args.self_play_matches,
                max_frames=args.self_play_frames,
                eager=args.self_play_eager,
                instant_match_restart=args.self_play_instant_match_restart,
                process_cohorts=cohort_count,
            )
        return
    resume_run = resume_state = None
    cfg = args.cfg
    if args.resume is not None:
        resume_state = load_for_resume(args.resume, Path("runs") / args.resume, device=DEVICE)
        if resume_state is None:
            raise SystemExit(f"no latest.pt for run {args.resume!r}")
        resume_run = args.resume
        cfg = config_from_state(resume_state["cfg"])
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    train(cfg, stats, comment=args.comment, resume_run=resume_run, resume_state=resume_state)


if __name__ == "__main__":
    main(tyro.cli(Args))
