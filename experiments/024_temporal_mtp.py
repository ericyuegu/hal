"""Train a 20-frame sequential multi-token action policy.

The causal trunk produces one state for each observed frame. Twenty small MLP
modules then model the next controller frames as one temporal chain. During
training, depth ``k`` receives the true action from depth ``k - 1``. During
decode, it receives its own previous sample. Each frame is also factorized over
controller groups in the fixed order C-stick, triggers, buttons, main stick.

Run:
    uv run experiments/024_temporal_mtp.py
    uv run experiments/024_temporal_mtp.py --eval runs/<run>/final.pt
"""

# %%
import contextlib
import itertools
import math
import time
from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from dataclasses import replace
from pathlib import Path
from typing import cast

import melee
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

import wandb
from hal import streams
from hal.data.feature_stats import FeatureStats
from hal.eval.cross_stage import sweep_vs_cpu_prior
from hal.eval.cross_stage import vs_cpu_metrics
from hal.eval.harness import default_session_cfg
from hal.training import scoring
from hal.training.checkpoints import BackgroundUploader
from hal.training.checkpoints import load_for_resume
from hal.training.checkpoints import save_checkpoint
from hal.training.closed_loop import RecedingHorizon
from hal.training.dataloader import make_loader
from hal.training.dataloader import make_replay_reservoir_loader
from hal.training.ego_stats import load_consolidated_stats
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import BASE_ACTION_PROJECTION
from hal.training.features import BASE_PLAYER_PREFIXES
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import Context
from hal.training.features import TrainBatch
from hal.training.features import stack_actions
from hal.training.muon import SingleDeviceMuonWithAuxAdam
from hal.training.runs import make_run_name
from hal.training.runs import profile
from hal.training.runs import setup_run_dir
from hal.training.trunk import Trunk
from hal.training.trunk import TrunkConfig
from hal.training.trunk import rmsnorm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_LN2 = math.log(2.0)

_N_CONT = 6
_PLAYER_PREFIXES = BASE_PLAYER_PREFIXES
_INPUT_PROJECTION = BASE_ACTION_PROJECTION

GROUP_NAMES: tuple[str, ...] = ("buttons", "main_stick", "c_stick", "triggers")
GROUP_VOCABS: tuple[int, ...] = (
    scoring.N_BUTTON_COMBOS,
    scoring.STICK_CLUSTER_CENTERS_MAIN.shape[0],
    scoring.STICK_CLUSTER_CENTERS_C.shape[0],
    scoring.TRIGGER_CENTERS.shape[0] ** 2,
)
N_GROUPS = len(GROUP_NAMES)
BUTTONS_G, MAIN_G, C_G, TRIG_G = range(N_GROUPS)
GROUP_INDEX = {name: index for index, name in enumerate(GROUP_NAMES)}
GROUP_ORDER: tuple[str, ...] = ("c_stick", "triggers", "buttons", "main_stick")

TRIGGER_L_CH = ACTION_CHANNELS.index("trigger_l")
TRIGGER_R_CH = ACTION_CHANNELS.index("trigger_r")
BUTTON_L_CH = ACTION_CHANNELS.index("button_l")
BUTTON_R_CH = ACTION_CHANNELS.index("button_r")


# %%
@dataclass
class TrainConfig:
    # Trunk.
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 4
    attn_window: int = 0
    require_flex: bool = False
    L_ctx: int = 256

    # Sequential controller decoder.
    L_chunk: int = 20
    action_embed_dim: int = 64
    action_mlp_ratio: int = 2
    action_vocab: int = 1024
    char_vocab: int = 32
    char_dim: int = 12
    stage_vocab: int = 32
    stage_dim: int = 4

    # Decode four planned frames before observing and replanning.
    exec_horizon: int = 4
    decode_temp: float = 1.0
    decode_click_trigger_fix: bool = True

    seed: int = 0
    # Effective batch. Four micro-batches keep the 20 sequential heads practical
    # on a 24 GiB device while preserving 512 examples per optimizer update.
    batch_size: int = 512
    grad_accum_steps: int = 4
    muon_lr: float = 0.02
    adam_lr: float = 8.5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 16_384
    amp_dtype: str = "bfloat16"
    allow_tf32: bool = True
    compile_trunk: bool = True

    val_every: int = 1024
    val_n_samples: int = 1192
    ckpt_every: int = 2048
    eval_every: int = 4096
    eval_max_frames: int = 7200
    eval_n_matchups: int = 32
    final_eval_n_matchups: int = 96
    eval_max_parallel: int = 32

    data_root: str = "data/processed/ranked-anonymized-1/mds-policy-v7"
    compact_data: bool = True
    mds_schema_version: int = 7
    cache_limit_gb: int = 128
    shuffle_block_size: int = 2000
    predownload: int = 512
    windows_per_replay: int = 4
    reservoir_capacity: int = 4096
    val_split: str = "val"
    num_workers: int = 16
    prefetch_factor: int = 2
    train_batch_prefetch: bool = True
    push_to_r2: bool = True


def validate_config(cfg: TrainConfig) -> None:
    positive = {
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "L_ctx": cfg.L_ctx,
        "L_chunk": cfg.L_chunk,
        "action_embed_dim": cfg.action_embed_dim,
        "action_mlp_ratio": cfg.action_mlp_ratio,
        "batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
        "max_steps": cfg.max_steps,
        "exec_horizon": cfg.exec_horizon,
        "val_n_samples": cfg.val_n_samples,
    }
    for name, value in positive.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if cfg.L_chunk != 20:
        raise ValueError(f"this experiment requires L_chunk=20, got {cfg.L_chunk}")
    if cfg.d_model % cfg.n_heads:
        raise ValueError("d_model must be divisible by n_heads")
    if cfg.batch_size % cfg.grad_accum_steps:
        raise ValueError("batch_size must be divisible by grad_accum_steps")
    if not 1 <= cfg.exec_horizon <= cfg.L_chunk:
        raise ValueError("exec_horizon must be in [1, L_chunk]")
    if not math.isfinite(cfg.decode_temp) or cfg.decode_temp <= 0:
        raise ValueError("decode_temp must be finite and positive")
    if cfg.amp_dtype not in ("bfloat16", "float32"):
        raise ValueError("amp_dtype must be 'bfloat16' or 'float32'")
    if cfg.warmup_steps < 0 or cfg.warmup_steps > cfg.max_steps:
        raise ValueError("warmup_steps must be in [0, max_steps]")
    if cfg.reservoir_capacity < 2 * micro_batch_size(cfg):
        raise ValueError("reservoir_capacity must be at least twice the micro-batch size")


def micro_batch_size(cfg: TrainConfig) -> int:
    return cfg.batch_size // cfg.grad_accum_steps


def model_tag(cfg: TrainConfig) -> str:
    attention = "full" if cfg.attn_window == 0 else f"swa{cfg.attn_window}"
    return (
        f"mtp20-d{cfg.d_model}-L{cfg.n_layers}-h{cfg.n_heads}-Lc{cfg.L_ctx}-"
        f"{attention}-mlp{cfg.action_mlp_ratio}-s{cfg.exec_horizon}"
    )


# %%
def quantize_groups(
    main_centers: Tensor,
    c_centers: Tensor,
    trigger_centers: Tensor,
    actions: Tensor,
) -> Tensor:
    """Convert native controller vectors to four categorical group indices."""
    continuous, buttons_raw = actions[..., :_N_CONT], actions[..., _N_CONT:]
    buttons = scoring.buttons_to_combo(buttons_raw)
    main = scoring.nearest_cluster(continuous[..., 0:2], main_centers)
    c_stick = scoring.nearest_cluster(continuous[..., 2:4], c_centers)
    trigger_pair = scoring.nearest_center(continuous[..., 4:6], trigger_centers)
    triggers = trigger_pair[..., 0] * trigger_centers.shape[0] + trigger_pair[..., 1]
    return torch.stack((buttons, main, c_stick, triggers), dim=-1)


def dequantize_groups(
    main_centers: Tensor,
    c_centers: Tensor,
    trigger_centers: Tensor,
    indices: Tensor,
) -> Tensor:
    """Convert four categorical group indices to native controller vectors."""
    n_trigger = trigger_centers.shape[0]
    buttons = scoring.combo_to_buttons(indices[..., BUTTONS_G])
    main = scoring.cluster_to_xy(indices[..., MAIN_G], main_centers)
    c_stick = scoring.cluster_to_xy(indices[..., C_G], c_centers)
    trigger_l = scoring.center_to_value(indices[..., TRIG_G] // n_trigger, trigger_centers)
    trigger_r = scoring.center_to_value(indices[..., TRIG_G] % n_trigger, trigger_centers)
    return torch.cat((main, c_stick, torch.stack((trigger_l, trigger_r), dim=-1), buttons), dim=-1)


class TemporalHead(nn.Module):
    """One DeepSeek-style temporal fusion MLP and four conditional classifiers."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        d = cfg.d_model
        hidden = cfg.action_mlp_ratio * d
        self.fuse = nn.Linear(2 * d, d, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)
        self.condition = nn.ModuleDict(
            {
                name: nn.Linear(position * cfg.action_embed_dim, d, bias=False)
                for position, name in enumerate(GROUP_ORDER)
                if position > 0
            }
        )
        self.outputs = nn.ModuleDict({name: nn.Linear(d, GROUP_VOCABS[GROUP_INDEX[name]]) for name in GROUP_NAMES})

    def advance(self, previous_state: Tensor, previous_action: Tensor) -> Tensor:
        fused = self.fuse(torch.cat((rmsnorm(previous_state), rmsnorm(previous_action)), dim=-1))
        return previous_state + self.down(F.silu(self.up(rmsnorm(fused))))

    def group_logits(
        self,
        state: Tensor,
        name: str,
        prefix: dict[str, Tensor],
        embeddings: nn.ModuleDict,
    ) -> Tensor:
        position = GROUP_ORDER.index(name)
        earlier = GROUP_ORDER[:position]
        if tuple(prefix) != earlier:
            raise ValueError(f"{name} requires prefix {earlier}, got {tuple(prefix)}")
        features = state
        if earlier:
            condition = torch.cat([embeddings[group](prefix[group]) for group in earlier], dim=-1)
            features = features + self.condition[name](condition)
        return cast(nn.Linear, self.outputs[name])(rmsnorm(features))

    def teacher_forced(
        self,
        state: Tensor,
        target: Tensor,
        embeddings: nn.ModuleDict,
    ) -> dict[str, Tensor]:
        prefix: dict[str, Tensor] = {}
        logits: dict[str, Tensor] = {}
        for name in GROUP_ORDER:
            logits[name] = self.group_logits(state, name, prefix, embeddings)
            prefix[name] = target[..., GROUP_INDEX[name]]
        return logits


class GPT(nn.Module):
    """Causal game-state trunk followed by a 20-frame temporal action chain."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.L_chunk = cfg.L_chunk
        self.group_order = GROUP_ORDER
        self.cat_specs = {**CAT_FEATURES, "action": (cfg.action_vocab, CAT_FEATURES["action"][1])}
        self.cat_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocab, dim) for name, (vocab, dim) in self.cat_specs.items()}
        )
        self.char_emb = nn.Embedding(cfg.char_vocab, cfg.char_dim)
        self.stage_emb = nn.Embedding(cfg.stage_vocab, cfg.stage_dim)
        per_player = len(FLOAT_FEATURES) * 2 + sum(dim for _, dim in CAT_FEATURES.values())
        d_in = len(_PLAYER_PREFIXES) * per_player + A_DIM + 2 * cfg.char_dim + cfg.stage_dim
        self.ctx_proj = nn.Linear(d_in, cfg.d_model)
        self.trunk = Trunk(
            TrunkConfig(
                d_model=cfg.d_model,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                L_ctx=cfg.L_ctx,
                attn_window=cfg.attn_window,
                require_flex=cfg.require_flex,
            )
        )

        self.action_embeddings = nn.ModuleDict(
            {name: nn.Embedding(GROUP_VOCABS[GROUP_INDEX[name]], cfg.action_embed_dim) for name in GROUP_NAMES}
        )
        self.frame_projection = nn.Linear(N_GROUPS * cfg.action_embed_dim, cfg.d_model, bias=False)
        self.bos_action = nn.Parameter(torch.zeros(cfg.d_model))
        self.temporal_heads = nn.ModuleList([TemporalHead(cfg) for _ in range(cfg.L_chunk)])

        self.register_buffer("main_centers", scoring.STICK_CLUSTER_CENTERS_MAIN.clone())
        self.register_buffer("c_centers", scoring.STICK_CLUSTER_CENTERS_C.clone())
        self.register_buffer("trigger_centers", scoring.TRIGGER_CENTERS.clone())

    def _per_player_features(self, features: dict[str, Tensor], prefix: str) -> Tensor:
        ref = features[f"{prefix}_position_x"]
        batch, length = ref.shape
        parts: list[Tensor] = [features[f"{prefix}_{name}"][..., None] for name in FLOAT_FEATURES]
        for name in FLOAT_FEATURES:
            key = f"{prefix}_{name}_mask"
            mask = features.get(key)
            parts.append(
                mask[..., None]
                if mask is not None
                else torch.zeros(batch, length, 1, device=ref.device, dtype=ref.dtype)
            )
        for name, (vocab, _) in self.cat_specs.items():
            parts.append(self.cat_embeds[name](features[f"{prefix}_{name}"].clamp(0, vocab - 1)))
        return torch.cat(parts, dim=-1)

    def context_tokens(self, features: dict[str, Tensor]) -> Tensor:
        parts = [self._per_player_features(features, prefix) for prefix in _PLAYER_PREFIXES]
        parts.append(torch.stack([features[f"ego_{name}"] for name in ACTION_CHANNELS], dim=-1))
        parts.append(self.char_emb(features["ego_character"].clamp(0, self.char_emb.num_embeddings - 1)))
        parts.append(self.char_emb(features["opp_character"].clamp(0, self.char_emb.num_embeddings - 1)))
        parts.append(self.stage_emb(features["stage"].clamp(0, self.stage_emb.num_embeddings - 1)))
        return self.ctx_proj(torch.cat(parts, dim=-1))

    def forward(self, features: dict[str, Tensor], ctx_pad: Tensor) -> Tensor:
        return self.trunk(self.context_tokens(features), ctx_pad)

    def frame_embedding(self, indices: Tensor) -> Tensor:
        parts = [self.action_embeddings[name](indices[..., GROUP_INDEX[name]]) for name in GROUP_NAMES]
        return self.frame_projection(torch.cat(parts, dim=-1))

    def teacher_forced_logits(self, hidden: Tensor, targets: Tensor) -> list[dict[str, Tensor]]:
        """Return depth logits for targets shaped ``[..., H, N_GROUPS]``."""
        if targets.shape[-2:] != (self.L_chunk, N_GROUPS):
            raise ValueError(f"targets must end in ({self.L_chunk}, {N_GROUPS}), got {tuple(targets.shape)}")
        state = hidden
        previous = self.bos_action.expand_as(hidden)
        out: list[dict[str, Tensor]] = []
        for depth, module in enumerate(self.temporal_heads):
            state = module.advance(state, previous)
            target = targets[..., depth, :]
            out.append(module.teacher_forced(state, target, self.action_embeddings))
            previous = self.frame_embedding(target)
        return out


def quantize(model: GPT, actions: Tensor) -> Tensor:
    return quantize_groups(model.main_centers, model.c_centers, model.trigger_centers, actions)


def dequantize(model: GPT, indices: Tensor) -> Tensor:
    return dequantize_groups(model.main_centers, model.c_centers, model.trigger_centers, indices)


def chunk_targets(model: GPT, batch: TrainBatch) -> tuple[Tensor, Tensor]:
    """Return next-20 frame targets at every context position and their valid mask."""
    if batch.target.shape[1] < model.L_chunk:
        raise ValueError(f"target contains {batch.target.shape[1]} frames, expected at least {model.L_chunk}")
    history = stack_actions(batch.context.features)
    full = quantize(model, torch.cat((history, batch.target[:, : model.L_chunk]), dim=1))
    length = history.shape[1]
    targets = torch.stack([full[:, offset : offset + length] for offset in range(1, model.L_chunk + 1)], dim=2)
    positions = torch.arange(length, device=full.device)
    valid = positions[None, :] >= batch.context.ctx_pad[:, None]
    return targets, valid


@dataclass(frozen=True, slots=True)
class ActionLoss:
    nll: Tensor  # [N_valid, H, N_GROUPS], nats
    targets: Tensor  # [N_valid, H, N_GROUPS]


def action_loss(model: GPT, batch: TrainBatch, *, hidden: Tensor | None = None) -> ActionLoss:
    targets, valid = chunk_targets(model, batch)
    if hidden is None:
        hidden = model(batch.context.features, batch.context.ctx_pad)
    logits = model.teacher_forced_logits(hidden, targets)
    flat_valid = valid.reshape(-1)
    target_valid = targets.reshape(-1, model.L_chunk, N_GROUPS)[flat_valid]
    losses: list[Tensor] = []
    for depth, by_group in enumerate(logits):
        group_losses = []
        for name in GROUP_NAMES:
            group = GROUP_INDEX[name]
            values = by_group[name].reshape(-1, by_group[name].shape[-1])[flat_valid]
            group_losses.append(F.cross_entropy(values.float(), target_valid[:, depth, group], reduction="none"))
        losses.append(torch.stack(group_losses, dim=-1))
    return ActionLoss(nll=torch.stack(losses, dim=1), targets=target_valid)


def objective(parts: ActionLoss) -> Tensor:
    """Mean over frames of each frame's joint four-group NLL."""
    return parts.nll.sum(dim=-1).mean()


def sample_categorical(logits: Tensor, *, temperature: float, argmax: bool, gen: torch.Generator | None) -> Tensor:
    values = logits.float()
    if argmax:
        return values.argmax(dim=-1)
    return torch.multinomial(F.softmax(values / temperature, dim=-1), 1, generator=gen).squeeze(-1)


@torch.no_grad()
def sample_chunk_from_hidden(
    model: GPT,
    hidden: Tensor,
    n_frames: int,
    *,
    temperature: float = 1.0,
    argmax: bool = False,
    click_trigger_fix: bool = True,
    gen: torch.Generator | None = None,
) -> Tensor:
    if not 1 <= n_frames <= model.L_chunk:
        raise ValueError(f"n_frames must be in [1, {model.L_chunk}], got {n_frames}")
    if not argmax and (not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("temperature must be finite and positive")
    state = hidden
    previous = model.bos_action.expand_as(hidden)
    frames: list[Tensor] = []
    for module in model.temporal_heads[:n_frames]:
        state = module.advance(state, previous)
        prefix: dict[str, Tensor] = {}
        picks: dict[str, Tensor] = {}
        for name in GROUP_ORDER:
            logits = module.group_logits(state, name, prefix, model.action_embeddings)
            pick = sample_categorical(logits, temperature=temperature, argmax=argmax, gen=gen)
            prefix[name] = pick
            picks[name] = pick
        indices = torch.stack([picks[name] for name in GROUP_NAMES], dim=-1)
        frames.append(indices)
        previous = model.frame_embedding(indices)
    action = dequantize(model, torch.stack(frames, dim=1))
    if click_trigger_fix:
        action[..., TRIGGER_L_CH] = torch.where(
            action[..., BUTTON_L_CH] > 0.5,
            torch.ones_like(action[..., TRIGGER_L_CH]),
            action[..., TRIGGER_L_CH],
        )
        action[..., TRIGGER_R_CH] = torch.where(
            action[..., BUTTON_R_CH] > 0.5,
            torch.ones_like(action[..., TRIGGER_R_CH]),
            action[..., TRIGGER_R_CH],
        )
    return action


@torch.no_grad()
def decode_chunk(
    model: GPT,
    ctx: Context,
    n_frames: int,
    *,
    temperature: float = 1.0,
    argmax: bool = False,
    click_trigger_fix: bool = True,
    gen: torch.Generator | None = None,
) -> Tensor:
    hidden = model(ctx.features, ctx.ctx_pad)[:, -1]
    return sample_chunk_from_hidden(
        model,
        hidden,
        n_frames,
        temperature=temperature,
        argmax=argmax,
        click_trigger_fix=click_trigger_fix,
        gen=gen,
    )


def make_policy(
    model: GPT,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    exec_horizon: int | None = None,
    temperature: float | None = None,
    seed: int | None = None,
    device: str = DEVICE,
) -> RecedingHorizon:
    horizon = cfg.exec_horizon if exec_horizon is None else exec_horizon
    if not 1 <= horizon <= cfg.L_chunk:
        raise ValueError(f"execution horizon must be in [1, {cfg.L_chunk}]")
    temp = cfg.decode_temp if temperature is None else temperature
    generator = None if seed is None else torch.Generator(device=device).manual_seed(seed)

    @torch.no_grad()
    def predict(ctx: Context, committed: np.ndarray | None) -> np.ndarray:
        if committed is not None:
            raise ValueError("this policy does not use a committed RTC prefix")
        return (
            decode_chunk(
                model,
                ctx,
                horizon,
                temperature=temp,
                click_trigger_fix=cfg.decode_click_trigger_fix,
                gen=generator,
            )
            .cpu()
            .numpy()
        )

    return RecedingHorizon(
        predict_chunk=predict,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=horizon,
        s=horizon,
        d=0,
        device=device,
        float_dtype=next(model.parameters()).dtype,
        projection=_INPUT_PROJECTION,
    )


# %%
def nll_metrics(nll: Tensor) -> dict[str, float]:
    """Summarize an ``[N, H, G]`` teacher-forced NLL tensor in bits."""
    per_horizon = nll.sum(dim=-1).mean(dim=0) / _LN2
    metrics = {
        "loss": float(per_horizon.mean()),
        "nll_chunk": float(per_horizon.sum()),
        **{f"nll_h{depth + 1:02d}": float(value) for depth, value in enumerate(per_horizon)},
    }
    per_group = nll.mean(dim=(0, 1)) / _LN2
    metrics.update({f"nll_{name}": float(per_group[index]) for index, name in enumerate(GROUP_NAMES)})
    return metrics


@torch.no_grad()
def val_metrics(model: GPT, batches: list[TrainBatch]) -> dict[str, float]:
    was_training = model.training
    model.eval()
    all_nll: list[Tensor] = []
    exact_frames = correct_groups = total_groups = 0
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    try:
        for cpu_batch in batches:
            batch = cpu_batch.to(DEVICE)
            parts = action_loss(model, batch)
            all_nll.append(parts.nll.cpu())
            hidden = model(batch.context.features, batch.context.ctx_pad)[:, -1]
            sample = sample_chunk_from_hidden(model, hidden, model.L_chunk, argmax=True, gen=generator)
            predicted = quantize(model, sample)
            target = quantize(model, batch.target[:, : model.L_chunk])
            matches = predicted == target
            correct_groups += int(matches.sum())
            total_groups += matches.numel()
            exact_frames += int(matches.all(dim=-1).sum())
    finally:
        model.train(was_training)
    nll = torch.cat(all_nll)
    n_frames = sum(batch.target.shape[0] for batch in batches) * model.L_chunk
    return {
        **nll_metrics(nll),
        "ancestral_group_acc": correct_groups / max(total_groups, 1),
        "ancestral_frame_acc": exact_frames / max(n_frames, 1),
    }


def lr_schedule(cfg: TrainConfig):
    floor = 1e-5 / cfg.adam_lr

    def schedule(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(cfg.warmup_steps, 1)
        progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return floor + (1.0 - floor) * cosine

    return schedule


def make_optimizer(model: GPT, cfg: TrainConfig) -> SingleDeviceMuonWithAuxAdam:
    muon = [parameter for parameter in model.trunk.blocks.parameters() if parameter.ndim >= 2]
    muon_ids = {id(parameter) for parameter in muon}
    embedding_ids = {
        id(parameter)
        for module in (model.cat_embeds, model.char_emb, model.stage_emb, model.action_embeddings)
        for parameter in module.parameters()
    }
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in model.parameters():
        if id(parameter) in muon_ids:
            continue
        (no_decay if parameter.ndim < 2 or id(parameter) in embedding_ids else decay).append(parameter)
    adam = dict(betas=(0.9, 0.95), eps=1e-10, use_muon=False)
    return SingleDeviceMuonWithAuxAdam(
        [
            dict(params=muon, lr=cfg.muon_lr, momentum=0.95, weight_decay=cfg.weight_decay, use_muon=True),
            dict(params=decay, lr=cfg.adam_lr, weight_decay=cfg.weight_decay, **adam),
            dict(params=no_decay, lr=cfg.adam_lr, weight_decay=0.0, **adam),
        ]
    )


def loader_kwargs(cfg: TrainConfig, stats: dict[str, FeatureStats]) -> dict:
    return dict(
        data_root=cfg.data_root,
        remote=streams.remote_for_local(cfg.data_root),
        cache_limit=f"{cfg.cache_limit_gb}gb",
        shuffle_block_size=cfg.shuffle_block_size,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=cfg.L_chunk,
        batch_size=micro_batch_size(cfg),
        seed=cfg.seed,
        schema_version=cfg.mds_schema_version,
        projection=_INPUT_PROJECTION,
    )


def cache_validation(loader: Iterable[TrainBatch], n_samples: int) -> list[TrainBatch]:
    batches: list[TrainBatch] = []
    count = 0
    for batch in loader:
        remaining = n_samples - count
        if remaining <= 0:
            break
        if batch.target.shape[0] > remaining:
            batch = TrainBatch(
                context=Context(
                    features={name: value[:remaining] for name, value in batch.context.features.items()},
                    ctx_pad=batch.context.ctx_pad[:remaining],
                ),
                target=batch.target[:remaining],
                replay_ids=None if batch.replay_ids is None else batch.replay_ids[:remaining],
            )
        batches.append(batch)
        count += batch.target.shape[0]
    if count != n_samples:
        raise RuntimeError(f"validation yielded {count} samples, expected {n_samples}")
    return batches


def eval_vs_cpu(
    model: GPT,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    n_matchups: int,
    replay_dir: Path,
    eager_forward: Callable | None = None,
) -> dict[str, float]:
    was_training = model.training
    compiled_forward = model.forward
    if eager_forward is not None:
        model.forward = eager_forward
    model.eval()
    policies = itertools.count()
    try:
        result = sweep_vs_cpu_prior(
            lambda: make_policy(model, stats, cfg, seed=cfg.seed + next(policies)),
            session_cfg=default_session_cfg(replay_dir, instant_match_restart=True),
            n_matchups=n_matchups,
            max_parallel=min(n_matchups, cfg.eval_max_parallel),
            max_frames=cfg.eval_max_frames,
        )
    finally:
        model.forward = compiled_forward
        model.train(was_training)
    return vs_cpu_metrics(result)


def train(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    *,
    comment: str = "",
    resume_run: str | None = None,
    resume_state: dict | None = None,
) -> None:
    validate_config(cfg)
    run_name = resume_run or make_run_name(Path(__file__).stem, model_tag(cfg), cfg.data_root, comment)
    uploader = BackgroundUploader(run_name) if cfg.push_to_r2 else None
    wandb.init(
        project="hal",
        name=run_name,
        id=None if resume_state is None else resume_state.get("wandb_id"),
        resume="allow" if resume_state is not None else None,
        tags=["gpt", "temporal-mtp", "chunk20"],
        config=asdict(cfg),
    )
    run_dir, replay_dir = setup_run_dir(run_name)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    autocast = (
        torch.autocast(DEVICE, dtype=torch.bfloat16)
        if cfg.amp_dtype == "bfloat16" and DEVICE == "cuda"
        else contextlib.nullcontext()
    )
    model = GPT(cfg).to(DEVICE)
    eager_forward = model.forward
    if cfg.compile_trunk and DEVICE == "cuda":
        model.forward = torch.compile(model.forward, dynamic=False)
    optimizer = make_optimizer(model, cfg)
    scheduler = LambdaLR(optimizer, lr_schedule(cfg))
    start_step = 0
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["opt"])
        scheduler.load_state_dict(resume_state["sched"])
        start_step = int(resume_state["step"]) + 1

    kwargs = loader_kwargs(cfg, stats)
    if cfg.compact_data:
        train_loader = make_replay_reservoir_loader(
            split="train",
            num_workers=cfg.num_workers,
            prefetch_factor=cfg.prefetch_factor,
            predownload=cfg.predownload,
            windows_per_replay=cfg.windows_per_replay,
            reservoir_capacity=cfg.reservoir_capacity,
            batch_prefetch=cfg.train_batch_prefetch,
            **kwargs,
        )
    else:
        train_loader = make_loader(
            split="train",
            num_workers=cfg.num_workers,
            prefetch_factor=cfg.prefetch_factor,
            windows_per_replay=cfg.windows_per_replay,
            compact=False,
            **kwargs,
        )
    # These experiments intentionally use their own 20-frame validation geometry.
    val_loader = make_loader(split=cfg.val_split, num_workers=0, compact=cfg.compact_data, **kwargs)
    val_cache = cache_validation(val_loader, cfg.val_n_samples)
    iterator = iter(train_loader)
    run_started = time.monotonic()
    model.train()
    try:
        for step in range(start_step, cfg.max_steps):
            with profile("step") as stopwatch:
                optimizer.zero_grad()
                nll_parts: list[Tensor] = []
                for _ in range(cfg.grad_accum_steps):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(train_loader)
                        batch = next(iterator)
                    batch = batch.to(DEVICE)
                    with autocast:
                        parts = action_loss(model, batch)
                        loss = objective(parts) / cfg.grad_accum_steps
                    loss.backward()
                    nll_parts.append(parts.nll.detach().cpu())
                gradients = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                if not torch.isfinite(gradients):
                    raise FloatingPointError(f"step {step}: non-finite gradient norm {gradients}")
                optimizer.step()
                scheduler.step()
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
            metrics = nll_metrics(torch.cat(nll_parts))
            log = {
                "global_step": step,
                "samples": (step + 1) * cfg.batch_size,
                **{f"train/{name}": value for name, value in metrics.items()},
                "train/grad_norm": float(gradients),
                "throughput/step_s": stopwatch.elapsed,
                "throughput/samples_per_s": cfg.batch_size / stopwatch.elapsed,
            }
            wandb.log(log)
            if step < 10 or step % 50 == 0:
                print(
                    f"[t+{time.monotonic() - run_started:.0f}s] step {step}: "
                    f"nll {metrics['loss']:.3f} bits/frame  dt={stopwatch.elapsed:.3f}s",
                    flush=True,
                )
            if cfg.val_every > 0 and step > 0 and step % cfg.val_every == 0:
                compiled_forward = model.forward
                model.forward = eager_forward
                try:
                    values = val_metrics(model, val_cache)
                finally:
                    model.forward = compiled_forward
                wandb.log({"global_step": step, **{f"val/{name}": value for name, value in values.items()}})
                print(f"[val] step {step}: {values}", flush=True)
            if cfg.ckpt_every > 0 and step > 0 and step % cfg.ckpt_every == 0:
                save_checkpoint(
                    run_dir / "latest.pt",
                    step=step,
                    model=model,
                    opt=optimizer,
                    sched=scheduler,
                    cfg=asdict(cfg),
                    wandb_id=None if wandb.run is None else wandb.run.id,
                    uploader=uploader,
                )
            if cfg.eval_every > 0 and step > 0 and step % cfg.eval_every == 0:
                values = eval_vs_cpu(
                    model,
                    stats,
                    cfg,
                    n_matchups=cfg.eval_n_matchups,
                    replay_dir=replay_dir / f"step_{step:06d}",
                    eager_forward=eager_forward,
                )
                wandb.log({"global_step": step, **{f"eval/{name}": value for name, value in values.items()}})

        compiled_forward = model.forward
        model.forward = eager_forward
        try:
            final_val = val_metrics(model, val_cache)
        finally:
            model.forward = compiled_forward
        wandb.log({"global_step": cfg.max_steps, **{f"val/{name}": value for name, value in final_val.items()}})
        final_eval = eval_vs_cpu(
            model,
            stats,
            cfg,
            n_matchups=cfg.final_eval_n_matchups,
            replay_dir=replay_dir / "final",
            eager_forward=eager_forward,
        )
        wandb.log({"global_step": cfg.max_steps, **{f"eval/{name}": value for name, value in final_eval.items()}})
        save_checkpoint(
            run_dir / "final.pt",
            step=cfg.max_steps,
            model=model,
            opt=optimizer,
            sched=scheduler,
            cfg=asdict(cfg),
            wandb_id=None if wandb.run is None else wandb.run.id,
            uploader=uploader,
        )
    finally:
        if uploader is not None:
            uploader.upload_tree(replay_dir, base=run_dir)
            uploader.close()
        wandb.finish()


# %%
def config_from_state(values: dict) -> TrainConfig:
    known = {item.name for item in fields(TrainConfig)}
    return TrainConfig(**{name: value for name, value in values.items() if name in known})


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
    exec_horizon: int | None = None,
    temperature: float | None = None,
    n_matchups: int | None = None,
) -> dict[str, float]:
    model, cfg, stats, state = load_checkpoint(path)
    if exec_horizon is not None:
        cfg = replace(cfg, exec_horizon=exec_horizon)
    if temperature is not None:
        cfg = replace(cfg, decode_temp=temperature)
    validate_config(cfg)
    replay_dir = Path(path).resolve().parent / "eval_replays"
    values = eval_vs_cpu(
        model,
        stats,
        cfg,
        n_matchups=cfg.final_eval_n_matchups if n_matchups is None else n_matchups,
        replay_dir=replay_dir,
    )
    print(f"[eval] step={state['step']} horizon={cfg.exec_horizon}: {values}", flush=True)
    return values


@dataclass
class Args:
    cfg: TrainConfig = field(default_factory=TrainConfig)
    eval: str | None = None
    eval_exec_horizon: int | None = None
    eval_temperature: float | None = None
    eval_n_matchups: int | None = None
    resume: str | None = None
    comment: str = ""


def main(args: Args) -> None:
    if args.eval is not None and args.resume is not None:
        raise SystemExit("pass only one of --eval or --resume")
    if args.eval is not None:
        eval_checkpoint(
            args.eval,
            exec_horizon=args.eval_exec_horizon,
            temperature=args.eval_temperature,
            n_matchups=args.eval_n_matchups,
        )
        return
    if args.resume is not None:
        state = load_for_resume(args.resume, Path("runs") / args.resume, device=DEVICE)
        if state is None:
            raise SystemExit(f"no latest.pt for run {args.resume!r}")
        cfg = config_from_state(state["cfg"])
        defaults = TrainConfig()
        cfg = replace(cfg, num_workers=defaults.num_workers, prefetch_factor=defaults.prefetch_factor)
        stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
        train(cfg, stats, resume_run=args.resume, resume_state=state)
        return
    stats = load_consolidated_stats(Path(args.cfg.data_root) / "stats.json")
    train(args.cfg, stats, comment=args.comment or "sequential-mtp20")


if __name__ == "__main__":
    main(tyro.cli(Args))
