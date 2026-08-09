"""Warm-start experiment 024 and add a detached 20-frame flow action expert.

The causal game-state trunk and categorical temporal decoder keep training with
their original objective. A small bidirectional transformer independently
denoises a native controller trajectory, conditioned on the final (detached)
trunk state. Sticks stay in [-1, 1]; triggers and digital buttons are mapped
from [0, 1] to [-1, 1] only while they are inside the flow model.

Run:
    uv run experiments/025_mtp_flow.py --init-ar-checkpoint runs/<024>/final.pt
    uv run experiments/025_mtp_flow.py --eval runs/<025>/final.pt
"""

# %%
import hashlib
import importlib.util
import itertools
import math
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

import wandb
from hal.data.feature_stats import FeatureStats
from hal.eval.cross_stage import sweep_vs_cpu_prior
from hal.eval.cross_stage import vs_cpu_metrics
from hal.eval.harness import default_session_cfg
from hal.training.checkpoints import BackgroundUploader
from hal.training.checkpoints import load_for_resume
from hal.training.checkpoints import save_checkpoint
from hal.training.closed_loop import RecedingHorizon
from hal.training.dataloader import make_loader
from hal.training.dataloader import make_replay_reservoir_loader
from hal.training.ego_stats import load_consolidated_stats
from hal.training.features import A_DIM
from hal.training.features import Context
from hal.training.features import TrainBatch
from hal.training.muon import SingleDeviceMuonWithAuxAdam
from hal.training.runs import make_run_name
from hal.training.runs import profile
from hal.training.runs import setup_run_dir
from hal.training.trunk import rmsnorm


def _load_temporal_mtp() -> ModuleType:
    """Load the numbered sibling without making experiments a Python package."""
    name = "_hal_experiment_024_temporal_mtp"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name("024_temporal_mtp.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mtp = _load_temporal_mtp()
DEVICE = mtp.DEVICE


# %%
@dataclass
class TrainConfig(mtp.TrainConfig):
    # The action expert is deliberately much smaller than the 8-layer trunk.
    flow_d_model: int = 128
    flow_layers: int = 2
    flow_heads: int = 4
    flow_ff_dim: int = 512
    flow_time_dim: int = 128
    flow_steps: int = 10

    # pi0/pi0.5-style time sampling: t = scale * (1 - Beta(alpha, 1)).
    flow_time_alpha: float = 1.5
    flow_time_scale: float = 0.999
    flow_loss_weight: float = 1.0
    ar_loss_weight: float = 1.0
    flow_noise_scale: float = 1.0

    # Populated from --init-ar-checkpoint for provenance in fresh runs.
    init_ar_checkpoint: str = ""
    init_ar_sha256: str = ""


def validate_config(cfg: TrainConfig) -> None:
    mtp.validate_config(cfg)
    positive_ints = {
        "flow_d_model": cfg.flow_d_model,
        "flow_layers": cfg.flow_layers,
        "flow_heads": cfg.flow_heads,
        "flow_ff_dim": cfg.flow_ff_dim,
        "flow_time_dim": cfg.flow_time_dim,
        "flow_steps": cfg.flow_steps,
    }
    for name, value in positive_ints.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if cfg.flow_d_model % cfg.flow_heads:
        raise ValueError("flow_d_model must be divisible by flow_heads")
    if cfg.flow_time_dim % 2:
        raise ValueError("flow_time_dim must be even")
    for name in ("flow_time_alpha", "flow_time_scale", "flow_loss_weight", "ar_loss_weight"):
        value = getattr(cfg, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}")
    if cfg.flow_time_scale > 1:
        raise ValueError("flow_time_scale must be at most 1")
    if not math.isfinite(cfg.flow_noise_scale) or cfg.flow_noise_scale < 0:
        raise ValueError("flow_noise_scale must be finite and non-negative")


def model_tag(cfg: TrainConfig) -> str:
    return f"{mtp.model_tag(cfg)}-flow{cfg.flow_d_model}x{cfg.flow_layers}-n{cfg.flow_steps}"


class FlowBlock(nn.Module):
    """One fully bidirectional self-attention block for the action trajectory."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        d = cfg.flow_d_model
        self.attention = nn.MultiheadAttention(d, cfg.flow_heads, batch_first=True, bias=False)
        self.up = nn.Linear(d, cfg.flow_ff_dim, bias=False)
        self.down = nn.Linear(cfg.flow_ff_dim, d, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        normalized = rmsnorm(x)
        x = x + self.attention(normalized, normalized, normalized, need_weights=False)[0]
        return x + self.down(F.silu(self.up(rmsnorm(x))))


def sinusoidal_time_embedding(t: Tensor, dim: int) -> Tensor:
    frequencies = torch.exp(
        -math.log(10_000.0) * torch.arange(dim // 2, device=t.device, dtype=torch.float32) / max(dim // 2 - 1, 1)
    )
    angles = t.float()[:, None] * frequencies[None, :]
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


class FlowHead(nn.Module):
    """A compact action expert conditioned on one detached trunk state."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.L_chunk = cfg.L_chunk
        self.time_dim = cfg.flow_time_dim
        self.action_in = nn.Linear(A_DIM, cfg.flow_d_model)
        self.condition_in = nn.Linear(cfg.d_model, cfg.flow_d_model)
        self.position = nn.Parameter(torch.empty(cfg.L_chunk + 1, cfg.flow_d_model))
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.flow_time_dim, cfg.flow_d_model),
            nn.SiLU(),
            nn.Linear(cfg.flow_d_model, cfg.flow_d_model),
        )
        self.blocks = nn.ModuleList([FlowBlock(cfg) for _ in range(cfg.flow_layers)])
        self.action_out = nn.Linear(cfg.flow_d_model, A_DIM)
        nn.init.normal_(self.position, std=0.02)
        nn.init.zeros_(self.action_out.weight)
        nn.init.zeros_(self.action_out.bias)

    def forward(self, noisy_action: Tensor, t: Tensor, condition: Tensor) -> Tensor:
        if noisy_action.shape[1:] != (self.L_chunk, A_DIM):
            raise ValueError(
                f"noisy_action must have shape [B, {self.L_chunk}, {A_DIM}], got {tuple(noisy_action.shape)}"
            )
        time = self.time_mlp(sinusoidal_time_embedding(t, self.time_dim)).to(noisy_action.dtype)
        condition_token = self.condition_in(condition)[:, None, :]
        action_tokens = self.action_in(noisy_action)
        x = torch.cat((condition_token, action_tokens), dim=1)
        x = x + self.position[None, :, :] + time[:, None, :]
        for block in self.blocks:
            x = block(x)
        return self.action_out(rmsnorm(x[:, 1:, :]))


class GPT(nn.Module):
    """Experiment-024 policy plus a detached flow-matching action expert."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.ar = mtp.GPT(cfg)
        self.flow = FlowHead(cfg)

    def forward(self, features: dict[str, Tensor], ctx_pad: Tensor) -> Tensor:
        return self.ar(features, ctx_pad)


# %%
def flow_encode(actions: Tensor) -> Tensor:
    """Map native controls to the symmetric flow space without quantization."""
    encoded = actions.clone()
    encoded[..., 4:] = encoded[..., 4:] * 2.0 - 1.0
    return encoded


def flow_decode(actions: Tensor, *, click_trigger_fix: bool = True) -> Tensor:
    """Map flow space back to legal native controls; only buttons are discrete."""
    decoded = actions.clone()
    decoded[..., :4] = decoded[..., :4].clamp(-1.0, 1.0)
    decoded[..., 4:6] = ((decoded[..., 4:6] + 1.0) * 0.5).clamp(0.0, 1.0)
    decoded[..., 6:] = ((decoded[..., 6:] + 1.0) * 0.5 > 0.5).to(decoded.dtype)
    if click_trigger_fix:
        decoded[..., mtp.TRIGGER_L_CH] = torch.where(
            decoded[..., mtp.BUTTON_L_CH] > 0.5,
            torch.ones_like(decoded[..., mtp.TRIGGER_L_CH]),
            decoded[..., mtp.TRIGGER_L_CH],
        )
        decoded[..., mtp.TRIGGER_R_CH] = torch.where(
            decoded[..., mtp.BUTTON_R_CH] > 0.5,
            torch.ones_like(decoded[..., mtp.TRIGGER_R_CH]),
            decoded[..., mtp.TRIGGER_R_CH],
        )
    return decoded


def sample_flow_time(
    batch_size: int,
    cfg: TrainConfig,
    *,
    device: torch.device,
    gen: torch.Generator | None = None,
) -> Tensor:
    # If U is uniform, U**(1/alpha) is Beta(alpha, 1).
    uniform = torch.rand(batch_size, device=device, generator=gen)
    return cfg.flow_time_scale * (1.0 - uniform.pow(1.0 / cfg.flow_time_alpha))


@dataclass(frozen=True, slots=True)
class FlowLoss:
    squared_error: Tensor  # [B, H, A_DIM]


def flow_matching_loss(
    model: GPT,
    batch: TrainBatch,
    hidden: Tensor,
    cfg: TrainConfig,
    *,
    gen: torch.Generator | None = None,
) -> FlowLoss:
    target = flow_encode(batch.target[:, : cfg.L_chunk])
    noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=gen)
    t = sample_flow_time(target.shape[0], cfg, device=target.device, gen=gen)
    noisy = (1.0 - t[:, None, None]) * noise + t[:, None, None] * target
    velocity = target - noise
    prediction = model.flow(noisy, t, hidden[:, -1].detach())
    return FlowLoss((prediction.float() - velocity.float()).square())


@torch.no_grad()
def integrate_chunk(
    model: GPT,
    hidden: Tensor,
    cfg: TrainConfig,
    *,
    gen: torch.Generator | None = None,
    noise: Tensor | None = None,
) -> Tensor:
    batch_size = hidden.shape[0]
    if noise is None:
        noise = torch.randn(
            batch_size,
            cfg.L_chunk,
            A_DIM,
            device=hidden.device,
            dtype=hidden.dtype,
            generator=gen,
        )
    x = noise * cfg.flow_noise_scale
    dt = 1.0 / cfg.flow_steps
    condition = hidden[:, -1].detach()
    for index in range(cfg.flow_steps):
        t = torch.full((batch_size,), index * dt, device=x.device, dtype=torch.float32)
        x = x + dt * model.flow(x, t, condition)
    return flow_decode(x, click_trigger_fix=cfg.decode_click_trigger_fix)


def make_policy(
    model: GPT,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    exec_horizon: int | None = None,
    seed: int | None = None,
    device: str = DEVICE,
) -> RecedingHorizon:
    horizon = cfg.exec_horizon if exec_horizon is None else exec_horizon
    if not 1 <= horizon <= cfg.L_chunk:
        raise ValueError(f"execution horizon must be in [1, {cfg.L_chunk}]")
    generator = None if seed is None else torch.Generator(device=device).manual_seed(seed)

    @torch.no_grad()
    def predict(ctx: Context, committed: np.ndarray | None) -> np.ndarray:
        if committed is not None:
            raise ValueError("this policy does not use a committed RTC prefix")
        with mtp.amp_context(cfg, device):
            hidden = model(ctx.features, ctx.ctx_pad)
            action = integrate_chunk(model, hidden, cfg, gen=generator)[:, :horizon]
        return action.cpu().numpy()

    return RecedingHorizon(
        predict_chunk=predict,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=horizon,
        s=horizon,
        d=0,
        device=device,
        float_dtype=next(model.parameters()).dtype,
        projection=mtp._INPUT_PROJECTION,
    )


# %%
def flow_metrics(squared_error: Tensor) -> dict[str, float]:
    if squared_error.ndim != 3 or squared_error.shape[-1] != A_DIM:
        raise ValueError(f"flow squared error must be [B, H, {A_DIM}], got {tuple(squared_error.shape)}")
    return flow_mean_metrics(squared_error.mean(dim=(0, 2)))


def flow_mean_metrics(mse_horizon: Tensor) -> dict[str, float]:
    if mse_horizon.ndim != 1:
        raise ValueError(f"flow horizon MSE must be one-dimensional, got {tuple(mse_horizon.shape)}")
    return {
        "flow_mse": float(mse_horizon.mean()),
        **{f"flow_mse_h{index + 1:02d}": float(value) for index, value in enumerate(mse_horizon)},
    }


@torch.no_grad()
def val_metrics(model: GPT, batches: list[TrainBatch], cfg: TrainConfig) -> dict[str, float]:
    was_training = model.training
    model.eval()
    try:
        # The AR forward is compiled only for the fixed training micro-batch.
        # Keep it eager for every cached validation batch, including the final
        # sliced batch, in both the AR and flow passes.
        with mtp.evaluation_mode(model.ar):
            ar = mtp.val_metrics(model.ar, batches, cfg)
            device = next(model.parameters()).device
            generator = torch.Generator(device=device).manual_seed(0)
            errors: list[Tensor] = []
            stick_error = trigger_error = 0.0
            stick_count = trigger_count = 0
            button_correct = button_count = 0
            group_correct = group_count = exact_frames = 0
            for cpu_batch in batches:
                mtp.validate_batch_geometry(cpu_batch, cfg)
                batch = cpu_batch.to(device)
                with mtp.amp_context(cfg, device):
                    hidden = model(batch.context.features, batch.context.ctx_pad)
                    loss = flow_matching_loss(model, batch, hidden, cfg, gen=generator)
                    predicted = integrate_chunk(model, hidden, cfg, gen=generator)
                errors.append(loss.squared_error.cpu())
                target = batch.target[:, : cfg.L_chunk]
                stick_error += float((predicted[..., :4] - target[..., :4]).abs().sum())
                stick_count += target[..., :4].numel()
                trigger_error += float((predicted[..., 4:6] - target[..., 4:6]).abs().sum())
                trigger_count += target[..., 4:6].numel()
                button_matches = predicted[..., 6:] == (target[..., 6:] > 0.5)
                button_correct += int(button_matches.sum())
                button_count += button_matches.numel()
                predicted_groups = mtp.quantize(model.ar, predicted)
                target_groups = mtp.quantize(model.ar, target)
                matches = predicted_groups == target_groups
                group_correct += int(matches.sum())
                group_count += matches.numel()
                exact_frames += int(matches.all(dim=-1).sum())
    finally:
        model.train(was_training)
    return {
        **{f"ar_{name}": value for name, value in ar.items()},
        **flow_metrics(torch.cat(errors)),
        "flow_stick_mae": stick_error / max(stick_count, 1),
        "flow_trigger_mae": trigger_error / max(trigger_count, 1),
        "flow_button_acc": button_correct / max(button_count, 1),
        "flow_group_acc": group_correct / max(group_count, 1),
        "flow_frame_acc": exact_frames / max(group_count // mtp.N_GROUPS, 1),
    }


def make_optimizer(model: GPT, cfg: TrainConfig) -> SingleDeviceMuonWithAuxAdam:
    muon = [parameter for parameter in model.ar.trunk.blocks.parameters() if parameter.ndim >= 2]
    muon_ids = {id(parameter) for parameter in muon}
    embedding_ids = {
        id(parameter)
        for module in (
            model.ar.cat_embeds,
            model.ar.char_emb,
            model.ar.stage_emb,
            model.ar.temporal.controller_embeddings,
        )
        for parameter in module.parameters()
    }
    embedding_ids.add(id(model.flow.position))
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in model.parameters():
        if id(parameter) in muon_ids:
            continue
        (no_decay if parameter.ndim < 2 or id(parameter) in embedding_ids else decay).append(parameter)
    if len(muon) + len(decay) + len(no_decay) != sum(1 for _ in model.parameters()):
        raise RuntimeError("optimizer parameter partition is incomplete")
    adam = dict(betas=(0.9, 0.95), eps=1e-10, use_muon=False)
    return SingleDeviceMuonWithAuxAdam(
        [
            dict(params=muon, lr=cfg.muon_lr, momentum=0.95, weight_decay=cfg.weight_decay, use_muon=True),
            dict(params=decay, lr=cfg.adam_lr, weight_decay=cfg.weight_decay, **adam),
            dict(params=no_decay, lr=cfg.adam_lr, weight_decay=0.0, **adam),
        ]
    )


def cache_validation(loader: Iterable[TrainBatch], n_samples: int) -> list[TrainBatch]:
    return mtp.cache_validation(loader, n_samples)


def eval_vs_cpu(
    model: GPT,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    n_matchups: int,
    replay_dir: Path,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    policies = itertools.count()
    try:
        with mtp.evaluation_mode(model.ar):
            result = sweep_vs_cpu_prior(
                lambda: make_policy(model, stats, cfg, seed=cfg.seed + next(policies)),
                session_cfg=default_session_cfg(replay_dir, instant_match_restart=True),
                n_matchups=n_matchups,
                max_parallel=min(n_matchups, cfg.eval_max_parallel),
                max_frames=cfg.eval_max_frames,
            )
    finally:
        model.train(was_training)
    return vs_cpu_metrics(result)


_WARMSTART_FIELDS = (
    "d_model",
    "n_layers",
    "n_heads",
    "attn_window",
    "require_flex",
    "L_ctx",
    "decoder_arch_version",
    "L_chunk",
    "temporal_d_model",
    "temporal_layers",
    "temporal_heads",
    "temporal_ff_dim",
    "main_stick_embed_dim",
    "c_stick_embed_dim",
    "trigger_embed_dim",
    "action_vocab",
    "action_state_embed_dim",
    "char_vocab",
    "char_dim",
    "stage_vocab",
    "stage_dim",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def warm_start_ar(model: GPT, cfg: TrainConfig, path: str) -> None:
    checkpoint = Path(path).expanduser().resolve()
    state = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    source_cfg = mtp.config_from_state(state["cfg"])
    differences = {
        name: (getattr(source_cfg, name), getattr(cfg, name))
        for name in _WARMSTART_FIELDS
        if getattr(source_cfg, name) != getattr(cfg, name)
    }
    if differences:
        raise ValueError(f"experiment-024 checkpoint is structurally incompatible: {differences}")
    model.ar.load_state_dict(state["model"], strict=True)
    cfg.init_ar_checkpoint = str(checkpoint)
    cfg.init_ar_sha256 = sha256_file(checkpoint)


# %%
def train(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    *,
    init_ar_checkpoint: str | None = None,
    comment: str = "",
    resume_run: str | None = None,
    resume_state: dict | None = None,
) -> None:
    validate_config(cfg)
    if resume_state is None and init_ar_checkpoint is None:
        raise ValueError("a fresh run requires --init-ar-checkpoint from experiment 024")
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    model = GPT(cfg).to(DEVICE)
    if resume_state is None:
        warm_start_ar(model, cfg, init_ar_checkpoint or "")
    run_name = resume_run or make_run_name(Path(__file__).stem, model_tag(cfg), cfg.data_root, comment)
    uploader = BackgroundUploader(run_name) if cfg.push_to_r2 else None
    wandb.init(
        project="hal",
        name=run_name,
        id=None if resume_state is None else resume_state.get("wandb_id"),
        resume="allow" if resume_state is not None else None,
        tags=["gpt", "temporal-mtp", "flow-matching", "chunk20"],
        config=asdict(cfg),
    )
    run_dir, replay_dir = setup_run_dir(run_name)
    if cfg.compile_trunk and DEVICE == "cuda":
        model.ar.forward = torch.compile(model.ar.forward, dynamic=False)
    optimizer = make_optimizer(model, cfg)
    scheduler = LambdaLR(optimizer, mtp.lr_schedule(cfg))
    start_step = 0
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["opt"])
        scheduler.load_state_dict(resume_state["sched"])
        start_step = int(resume_state["step"]) + 1

    kwargs = mtp.loader_kwargs(cfg, stats)
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
    val_kwargs = {**kwargs, "batch_size": cfg.val_batch_size}
    val_loader = make_loader(split=cfg.val_split, num_workers=0, compact=cfg.compact_data, **val_kwargs)
    val_cache = cache_validation(val_loader, cfg.val_n_samples)
    iterator = iter(train_loader)
    run_started = time.monotonic()
    train_batches_seen = 0
    model.train()
    try:
        for step in range(start_step, cfg.max_steps):
            if DEVICE == "cuda":
                torch.cuda.reset_peak_memory_stats()
            with profile("step") as stopwatch:
                optimizer.zero_grad()
                cpu_batches: list[tuple[TrainBatch, tuple[int, ...]]] = []
                loader_started = time.monotonic()
                for _ in range(cfg.grad_accum_steps):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(train_loader)
                        batch = next(iterator)
                    mtp.validate_batch_geometry(batch, cfg, expected_batch_size=mtp.micro_batch_size(cfg))
                    cpu_batches.append((batch, mtp.padding_groups(batch)))
                    train_batches_seen += 1
                loader_wait = time.monotonic() - loader_started
                valid_prefixes = sum(int((cfg.L_ctx - batch.context.ctx_pad).sum()) for batch, _ in cpu_batches)
                flow_samples = sum(batch.target.shape[0] for batch, _ in cpu_batches)
                if valid_prefixes <= 0 or flow_samples != cfg.batch_size:
                    raise RuntimeError(
                        f"bad accumulated geometry: valid_prefixes={valid_prefixes}, "
                        f"flow_samples={flow_samples}, expected_samples={cfg.batch_size}"
                    )
                ar_normalizer = valid_prefixes * cfg.L_chunk
                flow_normalizer = flow_samples * cfg.L_chunk * A_DIM
                nll_sum = torch.zeros(cfg.L_chunk, mtp.N_GROUPS, device=DEVICE, dtype=torch.float32)
                flow_sum = torch.zeros(cfg.L_chunk, device=DEVICE, dtype=torch.float32)
                n_prefixes = 0
                for cpu_batch, pads in cpu_batches:
                    batch = cpu_batch.to(DEVICE)
                    with mtp.amp_context(cfg, DEVICE):
                        hidden = model(batch.context.features, batch.context.ctx_pad)
                        ar = mtp.action_loss(model.ar, batch, hidden=hidden, pad_values=pads)
                        flow = flow_matching_loss(model, batch, hidden, cfg)
                        loss = (
                            cfg.ar_loss_weight * ar.nll.sum() / ar_normalizer
                            + cfg.flow_loss_weight * flow.squared_error.sum() / flow_normalizer
                        )
                    loss.backward()
                    nll_sum += ar.nll.detach().sum(dim=0)
                    flow_sum += flow.squared_error.detach().sum(dim=(0, 2))
                    n_prefixes += ar.nll.shape[0]
                if n_prefixes != valid_prefixes:
                    raise RuntimeError(f"decoded {n_prefixes} prefixes, expected {valid_prefixes}")
                gradients = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                if not torch.isfinite(gradients):
                    raise FloatingPointError(f"step {step}: non-finite gradient norm {gradients}")
                optimizer.step()
                scheduler.step()
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
            ar_values = mtp.nll_mean_metrics((nll_sum / n_prefixes).cpu())
            flow_values = flow_mean_metrics((flow_sum / (flow_samples * A_DIM)).cpu())
            log = {
                "global_step": step,
                "samples": (step + 1) * cfg.batch_size,
                **{f"train/ar_{name}": value for name, value in ar_values.items()},
                **{f"train/{name}": value for name, value in flow_values.items()},
                "train/grad_norm": float(gradients),
                "lr/muon": next(group["lr"] for group in optimizer.param_groups if group["use_muon"]),
                "lr/adam": next(group["lr"] for group in optimizer.param_groups if not group["use_muon"]),
                "throughput/step_s": stopwatch.elapsed,
                "throughput/loader_wait_s": loader_wait,
                "throughput/samples_per_s": cfg.batch_size / stopwatch.elapsed,
                "throughput/prefixes_per_s": n_prefixes / stopwatch.elapsed,
                "data/train_batches_seen": train_batches_seen,
            }
            if DEVICE == "cuda":
                log["hardware/peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
                log["hardware/peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
            wandb.log(log)
            if step == start_step:
                print(
                    f"[model] attention path: {model.ar.trunk.attn_path}, window={cfg.attn_window}",
                    flush=True,
                )
                if wandb.run is not None:
                    wandb.run.summary["model/attn_path"] = model.ar.trunk.attn_path
                    wandb.run.summary["startup/compiled_step0_s"] = stopwatch.elapsed
            if step < 10 or step % 50 == 0:
                print(
                    f"[t+{time.monotonic() - run_started:.0f}s] step {step}: "
                    f"AR {ar_values['loss']:.3f} bits/frame  "
                    f"flow {flow_values['flow_mse']:.4f}  dt={stopwatch.elapsed:.3f}s",
                    flush=True,
                )
            val_due = cfg.val_every > 0 and step > 0 and step % cfg.val_every == 0
            eval_due = cfg.eval_every > 0 and step > 0 and step % cfg.eval_every == 0
            periodic_ckpt_due = cfg.ckpt_every > 0 and step > 0 and step % cfg.ckpt_every == 0
            if periodic_ckpt_due or val_due or eval_due:
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
            if val_due:
                values = val_metrics(model, val_cache, cfg)
                wandb.log({"global_step": step, **{f"val/{name}": value for name, value in values.items()}})
                print(f"[val] step {step}: {values}", flush=True)
            if eval_due:
                values = eval_vs_cpu(
                    model,
                    stats,
                    cfg,
                    n_matchups=cfg.eval_n_matchups,
                    replay_dir=replay_dir / f"step_{step:06d}",
                )
                wandb.log({"global_step": step, **{f"eval/{name}": value for name, value in values.items()}})

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
        final_val = val_metrics(model, val_cache, cfg)
        wandb.log({"global_step": cfg.max_steps, **{f"val/{name}": value for name, value in final_val.items()}})
        final_eval = eval_vs_cpu(
            model,
            stats,
            cfg,
            n_matchups=cfg.final_eval_n_matchups,
            replay_dir=replay_dir / "final",
        )
        wandb.log({"global_step": cfg.max_steps, **{f"eval/{name}": value for name, value in final_eval.items()}})
    finally:
        if uploader is not None:
            uploader.upload_tree(replay_dir, base=run_dir)
            uploader.close()
        wandb.finish()


# %%
_FLOW_CHECKPOINT_FIELDS = {
    "flow_d_model",
    "flow_layers",
    "flow_heads",
    "flow_ff_dim",
    "flow_time_dim",
    "flow_steps",
}


def config_from_state(values: dict) -> TrainConfig:
    missing = (mtp._CHECKPOINT_ARCH_FIELDS | _FLOW_CHECKPOINT_FIELDS) - values.keys()
    if missing:
        raise ValueError(f"checkpoint is structurally incompatible; missing config fields: {sorted(missing)}")
    known = {item.name for item in fields(TrainConfig)}
    return TrainConfig(**{name: value for name, value in values.items() if name in known})


def load_checkpoint(
    path: str,
    *,
    device: str = DEVICE,
) -> tuple[GPT, TrainConfig, dict[str, FeatureStats], dict]:
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
    n_matchups: int | None = None,
) -> dict[str, float]:
    model, cfg, stats, state = load_checkpoint(path)
    if exec_horizon is not None:
        cfg = replace(cfg, exec_horizon=exec_horizon)
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
    init_ar_checkpoint: str | None = None
    eval: str | None = None
    eval_exec_horizon: int | None = None
    eval_n_matchups: int | None = None
    resume: str | None = None
    comment: str = ""


def main(args: Args) -> None:
    selected = sum(value is not None for value in (args.eval, args.resume))
    if selected > 1:
        raise SystemExit("pass only one of --eval or --resume")
    if args.eval is not None:
        eval_checkpoint(args.eval, exec_horizon=args.eval_exec_horizon, n_matchups=args.eval_n_matchups)
        return
    if args.resume is not None:
        if args.init_ar_checkpoint is not None:
            raise SystemExit("--init-ar-checkpoint is only valid for a fresh run")
        state = load_for_resume(args.resume, Path("runs") / args.resume, device=DEVICE)
        if state is None:
            raise SystemExit(f"no latest.pt for run {args.resume!r}")
        cfg = config_from_state(state["cfg"])
        defaults = TrainConfig()
        cfg = replace(cfg, num_workers=defaults.num_workers, prefetch_factor=defaults.prefetch_factor)
        stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
        train(cfg, stats, resume_run=args.resume, resume_state=state)
        return
    if args.init_ar_checkpoint is None:
        raise SystemExit("fresh flow training requires --init-ar-checkpoint runs/<024>/final.pt")
    stats = load_consolidated_stats(Path(args.cfg.data_root) / "stats.json")
    train(
        args.cfg,
        stats,
        init_ar_checkpoint=args.init_ar_checkpoint,
        comment=args.comment or "warm-flow20",
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
