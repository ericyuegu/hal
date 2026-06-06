"""Multi-position flow-matching action-chunk policy.

Splits the model into a causal backbone (decoder-style Transformer over the L_ctx
context tokens) + a per-position flow head (a small velocity net that denoises one
L_chunk chunk from a single backbone hidden vector, attending only within its own
L_chunk frames). Training supervises an action chunk at EVERY context position, not
just the last — the context attention is already paid for, so we just attach more
targets; gain is sublinear in T since adjacent positions predict overlapping chunks.

Inference protocol is unchanged from the baseline: read the LAST position's hidden
vector, integrate one chunk for n_flow_steps, execute it open-loop.
"""

# %%
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import contextlib
import itertools
import math
import time
import warnings
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from pathlib import Path

import melee
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from beartype import beartype
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import jaxtyped
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import wandb
from hal import streams
from hal.data.stats import FeatureStats
from hal.eval.cross_stage import sweep_self_play
from hal.eval.cross_stage import sweep_vs_cpu
from hal.eval.cross_stage import vs_cpu_metrics
from hal.eval.harness import default_session_cfg
from hal.training import scoring
from hal.training.checkpoints import BackgroundUploader
from hal.training.checkpoints import load_for_resume
from hal.training.checkpoints import save_checkpoint
from hal.training.closed_loop import RecedingHorizon
from hal.training.dataloader import make_loader
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import Context
from hal.training.features import TrainBatch
from hal.training.features import stack_actions
from hal.training.runs import make_run_name
from hal.training.runs import profile
from hal.training.runs import setup_run_dir
from hal.training.stats import load_consolidated_stats

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# %%
@dataclass
class TrainConfig:
    # model
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    time_emb_dim: int = 128
    # per-position flow head (small velocity net): denoises one L_chunk chunk from a
    # single backbone hidden vector, attending only within its own L_chunk frames.
    d_head: int = 128
    n_head_layers: int = 2
    head_heads: int = 4
    head_ff: int = 512
    # Seeds model init, flow-matching t/z noise, and the dataloader's window +
    # ego-port sampling — so a run is reproducible and val windows are fixed.
    seed: int = 0
    # window / chunking
    L_ctx: int = 256
    L_chunk: int = 16
    # multi-position supervision: number of context positions to supervise per sequence
    # each step (random subset, redrawn per step). Adjacent positions predict overlapping
    # chunks, so the learning signal is sublinear in L_ctx — sampling K << L_ctx cuts the
    # per-position head's flattened batch (and thus its compute) ~L_ctx/K with little loss.
    # -1 supervises every valid position (the original all-positions objective).
    train_positions: int = 64
    # inference
    n_flow_steps: int = 8
    # optimization
    batch_size: int = 32  # micro-batch run on the GPU per forward
    grad_accum_steps: int = 1  # optimizer step sees batch_size * grad_accum_steps samples
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 2**15
    # precision. The step is GPU-bound; in pure FP32 the Ampere tensor cores sit idle
    # (an A6000 ran 100% SM at only ~105/275 W). bf16 autocast ~2.3x's the step and
    # needs no GradScaler (bf16 keeps FP32's exponent range); TF32 speeds the residual
    # FP32 matmuls. Set amp_dtype="float32" to fall back to the old behavior.
    amp_dtype: str = "bfloat16"  # "bfloat16" | "float32"
    allow_tf32: bool = True
    # eval cadence
    val_every: int = 1024
    val_n_batches: int = 16
    eval_every: int = 2048
    eval_max_frames: int = 7200
    # closed-loop eval parallelism: replicas per stage, run concurrently in
    # waves of eval_max_parallel emulators (one batched forward across all live).
    eval_replicas: int = 2
    eval_max_parallel: int = 8
    # checkpointing: write + background-upload latest.pt every N steps (preemption resilience)
    ckpt_every: int = 2048
    # push checkpoints to R2 as we train (needs AWS_*); --resume pulls them back
    push_to_r2: bool = True
    # data
    data_root: str = "data/processed/ranked-anonymized-1/mds"
    # cap the local shard cache (StreamingDataset evicts past this) as a disk-full guard.
    # Set above the ~380 GB decompressed prod MDS so the whole set caches once with no
    # eviction churn, but below container disk so it evicts before Errno 28. Ignored for
    # local datasets (remote=None).
    cache_limit_gb: int = 440
    # py1e shuffle unit (samples). The streaming default (~4M) exceeds the 112k-sample
    # dataset, so the loader buffers ~everything (~380 GB) before the first batch and
    # stalls. ~3 shards' worth starts after pulling ~17 GB while still mixing shard order.
    shuffle_block_size: int = 2000
    val_split: str = "val"  # tiny datasets may have an empty val split; point this at "test"/"train"
    num_workers: int = 16  # data-pipeline-bound: the per-batch numpy preprocess + shard
    # decompress is CPU-heavy, so feed the GPU from more cores (cloud boxes have 24-36).
    prefetch_factor: int = 4  # num_workers * this = in-flight batches (fd + shm pressure)


def _model_tag(cfg: TrainConfig) -> str:
    return f"fm-mp-d{cfg.d_model}-L{cfg.n_layers}-H{cfg.n_heads}-Lc{cfg.L_ctx}-Lk{cfg.L_chunk}-fs{cfg.n_flow_steps}"


# %%
@jaxtyped(typechecker=beartype)
def sinusoidal_time_embedding(t: Float[Tensor, " B"], dim: int) -> Float[Tensor, "B d_time"]:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t.unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FlowMatchingPolicy(nn.Module):
    """Causal backbone + per-position flow head.

    The **backbone** is a decoder-style Transformer over the L_ctx context tokens
    (observed ego+opp gamestate + ego controller history) under a causal mask, so
    ``hidden[i]`` depends only on positions ``<= i``. ``encode_context`` returns the
    full ``[B, L_ctx, d_model]`` stack.

    The **per-position flow head** (``velocity``) is a small velocity net that
    denoises one L_chunk-frame chunk from a *single* backbone hidden vector. It
    attends only within its own L_chunk frames (bidirectional within the chunk) and
    never across context positions — the leakage guarantee depends on that. All
    supervised positions are flattened into the batch dim so they denoise in one
    head forward.

    Training supervises a chunk at every context position (``flow_loss``); inference
    reads only the last position's hidden vector (``integrate_chunk`` / ``act``).
    """

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.L_ctx = cfg.L_ctx
        self.L_chunk = cfg.L_chunk
        self.n_heads = cfg.n_heads
        d = cfg.d_model
        dh = cfg.d_head
        self.time_emb_dim = cfg.time_emb_dim

        self.cat_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocab, dim) for name, (vocab, dim) in CAT_FEATURES.items()}
        )

        n_float = len(FLOAT_FEATURES)
        n_mask = len(FLOAT_FEATURES)
        n_cat = sum(dim for _, dim in CAT_FEATURES.values())
        per_player_dim = n_float + n_mask + n_cat
        per_frame_in_dim = 2 * per_player_dim + A_DIM  # ego + opp + ego controller history

        # --- causal backbone ---
        self.ctx_proj = nn.Linear(per_frame_in_dim, d)
        self.pos_emb = nn.Embedding(self.L_ctx, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        # enable_nested_tensor=False: norm_first=True forces use_nested_tensor
        # to False internally anyway; passing it explicitly silences the warning.
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers, enable_nested_tensor=False)

        # --- per-position flow head (operates on flattened B*T positions) ---
        self.chunk_proj = nn.Linear(A_DIM, dh)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_emb_dim, dh),
            nn.GELU(),
            nn.Linear(dh, dh),
        )
        self.cond_proj = nn.Linear(d, dh)
        self.chunk_pos_emb = nn.Embedding(self.L_chunk, dh)
        head_layer = nn.TransformerEncoderLayer(
            d_model=dh,
            nhead=cfg.head_heads,
            dim_feedforward=cfg.head_ff,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.head_encoder = nn.TransformerEncoder(head_layer, num_layers=cfg.n_head_layers, enable_nested_tensor=False)
        self.out_proj = nn.Linear(dh, A_DIM)

    def _per_player_features(self, features: dict[str, Tensor], prefix: str) -> Tensor:
        """[B, L_ctx, per_player_dim] — all observed features for one player.
        Mixed-dtype concat (float + int-embed lookup) so a single jaxtyping
        annotation doesn't fit cleanly; covered by the ``velocity`` annotation
        downstream. ``features`` are already sliced to L_ctx by the dataloader."""
        ref = features[f"{prefix}_position_x"]
        B, L = ref.shape
        device = ref.device
        parts: list[Tensor] = []
        for f in FLOAT_FEATURES:
            parts.append(features[f"{prefix}_{f}"][..., None])
        for f in FLOAT_FEATURES:
            mk = f"{prefix}_{f}_mask"
            parts.append(features[mk][..., None] if mk in features else torch.zeros(B, L, 1, device=device))
        for cat_name, (vocab, _) in CAT_FEATURES.items():
            ids = features[f"{prefix}_{cat_name}"].clamp(0, vocab - 1)
            parts.append(self.cat_embeds[cat_name](ids))
        return torch.cat(parts, dim=-1)

    def _ego_history_features(self, features: dict[str, Tensor]) -> Tensor:
        """Ego's real past controller inputs, one row per context frame. The
        not-yet-filled rolling-buffer prefix is hidden at the backbone attention
        level (``_backbone_mask`` via ``ctx_pad``), not by zeroing features here."""
        return torch.cat([features[f"ego_{ch}"][..., None] for ch in ACTION_CHANNELS], dim=-1)

    def _context_tokens(self, features: dict[str, Tensor]) -> Float[Tensor, "B L_ctx d_model"]:
        ego = self._per_player_features(features, "ego")
        opp = self._per_player_features(features, "opp")
        hist = self._ego_history_features(features)
        return self.ctx_proj(torch.cat([ego, opp, hist], dim=-1))

    def _backbone_mask(self, ctx_pad: Int[Tensor, " B"], T: int, device: torch.device) -> Tensor:
        """Combined causal + left-padding additive-bool attn mask for the backbone,
        shape ``[B*n_heads, T, T]`` (True = block). Query ``i`` may attend to key ``j``
        iff ``j <= i`` (causal) and (``j`` is real, i.e. ``j >= ctx_pad``, OR ``j == i``).
        The diagonal exception keeps the padded-prefix rows from being fully masked
        (a fully-masked softmax row → NaN → NaN grads); those rows carry no loss
        (``valid`` is 0), and real rows never attend to padded keys."""
        idx = torch.arange(T, device=device)
        causal = idx[:, None] >= idx[None, :]  # [T, T]: j <= i
        real_key = idx[None, :] >= ctx_pad[:, None]  # [B, T]: j >= ctx_pad
        diag = torch.eye(T, dtype=torch.bool, device=device)  # [T, T]
        block = ~(causal[None] & (real_key[:, None, :] | diag[None]))  # [B, T, T], True = block
        B = ctx_pad.shape[0]
        return block[:, None].expand(B, self.n_heads, T, T).reshape(B * self.n_heads, T, T)

    def encode_context(self, ctx: Context) -> Float[Tensor, "B L_ctx d_model"]:
        """Causal backbone over the L_ctx context tokens → one hidden vector per
        position; ``hidden[i]`` depends only on positions ``<= i``."""
        tok = self._context_tokens(ctx.features)
        T = tok.size(1)
        tok = tok + self.pos_emb.weight[None, :T, :]
        mask = self._backbone_mask(ctx.ctx_pad, T, tok.device)
        return self.encoder(tok, mask=mask)

    @jaxtyped(typechecker=beartype)
    def velocity(
        self,
        cond: Float[Tensor, "N d_model"],
        a_t: Float[Tensor, "N L_chunk d_action"],
        t: Float[Tensor, " N"],
    ) -> Float[Tensor, "N L_chunk d_action"]:
        """Per-position velocity: denoise the L_chunk chunk ``a_t`` (at flow time ``t``)
        conditioned on one backbone hidden vector ``cond``. ``N`` flattens whatever
        positions are being denoised (B*T valid positions at train, B at inference).
        Bidirectional within the chunk, no cross-position attention."""
        H = a_t.size(1)
        t_proj = self.time_mlp(sinusoidal_time_embedding(t, self.time_emb_dim))
        chunk = self.chunk_proj(a_t) + t_proj[:, None, :] + self.cond_proj(cond)[:, None, :]
        chunk = chunk + self.chunk_pos_emb.weight[None, :H, :]
        return self.out_proj(self.head_encoder(chunk))


# %%
def _position_targets(ctx: Context, target: Tensor, H: int) -> tuple[Tensor, Tensor]:
    """Build, for every context position ``i``, the next-H-action target chunk and a
    validity mask, from the reconstructed full action window.

    The full window action array ``A_full[:, :L_ctx] = stack_actions(ctx.features)``
    (the same normalized ego channels the loader sliced into context) and
    ``A_full[:, L_ctx:] = target``. With post_i ego alignment (``ego[i]`` is the action
    taken AT frame ``i``), the leak-free target for position ``i`` is the next H actions
    ``A_full[:, i+1 : i+1+H]`` — and the last position recovers exactly ``target``.

    Returns ``(tgt [B, T, H, A_DIM], valid [B, T] bool)``. ``valid`` is 0 on the
    leftmost ``ctx_pad`` padded positions (whose targets straddle the zero pad). Since
    the window is ``L_ctx + L_chunk`` long and ``H == L_chunk``, no tail is truncated."""
    a_full = torch.cat([stack_actions(ctx.features), target], dim=1)  # [B, T+H, A_DIM]
    T = a_full.size(1) - H
    tgt = a_full.unfold(1, H, 1)[:, 1:].permute(0, 1, 3, 2).contiguous()  # [B, T, H, A_DIM]
    pos = torch.arange(T, device=a_full.device)
    valid = pos[None, :] >= ctx.ctx_pad[:, None]  # [B, T]
    return tgt, valid


def flow_loss(
    model: FlowMatchingPolicy,
    batch: TrainBatch,
    *,
    gen: torch.Generator | None = None,
    multi: bool = True,
    max_positions: int = -1,
) -> Float[Tensor, "N L_chunk d_action"]:
    """Per-element conditional flow-matching squared error on the velocity (unreduced),
    over the supervised context positions flattened into ``N``.

    Default (``multi=True``): supervise an H-frame chunk at every valid context
    position. ``multi=False``: supervise only the last position (matches inference;
    used by val for cross-run comparability). Either way draws t ~ U[0,1] and z ~ N(0,I)
    independently per supervised position and returns the element-wise squared error vs
    the target velocity. ``.mean()`` it for the scalar (mean over valid elements → loss
    scale is T-invariant); feed it to ``velocity_mse_breakdown`` for the modality/horizon
    split."""
    ctx = batch.context
    H = model.L_chunk
    hidden = model.encode_context(ctx)  # [B, T, d_model]
    B, T, _ = hidden.shape
    tgt, valid = _position_targets(ctx, batch.target, H)
    if not multi:
        valid = valid & (torch.arange(T, device=hidden.device)[None, :] == T - 1)
    elif 0 < max_positions < T:
        # Supervise a random ``max_positions`` of each row's valid positions (the head's
        # flattened batch — and its compute — scales with the count kept). Rank valid
        # positions by uniform noise, take the top-K; invalid positions score below 0 so
        # they only enter the top-K in rows with fewer than K valid, where ``valid &`` drops
        # them. Redrawn each step via ``gen``, so every position is supervised in expectation.
        scores = torch.rand(B, T, device=hidden.device, generator=gen).masked_fill(~valid, -1.0)
        keep = torch.zeros_like(valid).scatter_(1, scores.topk(max_positions, dim=1).indices, True)
        valid = valid & keep
    sel = valid.reshape(B * T)
    cond = hidden.reshape(B * T, -1)[sel]
    tgt = tgt.reshape(B * T, H, A_DIM)[sel]
    N = cond.shape[0]
    t = torch.rand(N, device=hidden.device, generator=gen)
    z = torch.randn(tgt.shape, device=hidden.device, dtype=tgt.dtype, generator=gen)
    t_b = t.view(N, 1, 1)
    a_t = (1 - t_b) * z + t_b * tgt
    v_target = tgt - z
    return F.mse_loss(model.velocity(cond, a_t, t), v_target, reduction="none")


@torch.no_grad()
def integrate_chunk(
    model: FlowMatchingPolicy,
    ctx: Context,
    *,
    n_steps: int,
    gen: torch.Generator | None = None,
    device: str = DEVICE,
) -> Float[Tensor, "B L_chunk d_action"]:
    """Euler-integrate one action chunk from z ~ N(0, I) for ``n_steps``, conditioned on
    the LAST context position's backbone hidden vector.

    The single inference integrator: closed-loop play, the val reconstruction metric,
    and ``--diag`` all call this so there is one integration path. The backbone runs
    once; only the per-position head re-runs each Euler step. Pass ``gen`` for
    reproducible noise (fixed val metric / histograms); leave ``None`` for an
    independent draw (closed-loop, cross-sample spread)."""
    cond = model.encode_context(ctx)[:, -1, :]  # [B, d_model]
    B = cond.shape[0]
    a = torch.randn(B, model.L_chunk, A_DIM, device=device, generator=gen)
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((B,), k * dt, device=device)
        a = a + dt * model.velocity(cond, a, t)
    return a


@torch.no_grad()
def act(model: FlowMatchingPolicy, ctx: Context, *, n_steps: int, device: str = DEVICE) -> Float[Tensor, "B d_action"]:
    """Inference convenience: integrate one chunk from the last context position and
    return just its first frame ``[B, A_DIM]``. The closed-loop driver uses the full
    chunk via ``integrate_chunk``; this is the per-frame surface (and the inference
    invariant the tests pin)."""
    return integrate_chunk(model, ctx, n_steps=n_steps, device=device)[:, 0, :]


def make_policy(
    model: FlowMatchingPolicy,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    device: str = DEVICE,
    n_flow_steps: int | None = None,
) -> RecedingHorizon:
    """Fresh closed-loop policy for one eval wave (rolling state must not leak).

    Open-loop (no committed prefix): the driver replans every ``L_chunk``,
    integrating one chunk from the last context position. Pass ``n_flow_steps`` to
    override ``cfg.n_flow_steps`` at eval time (test-time compute sweep) without
    editing the config or retraining."""
    n_steps = n_flow_steps if n_flow_steps is not None else cfg.n_flow_steps

    @torch.no_grad()
    def predict_chunk(ctx: Context, committed: np.ndarray | None) -> np.ndarray:
        assert committed is None, "open-loop policy does not condition on a committed prefix"
        return integrate_chunk(model, ctx, n_steps=n_steps, device=device).cpu().numpy()

    return RecedingHorizon(
        predict_chunk=predict_chunk,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=cfg.L_chunk,
        s=cfg.L_chunk,
        d=0,
        device=device,
    )


# %%
def lr_schedule(cfg: TrainConfig):
    """Linear warmup → cosine to floor."""
    floor = 1e-5 / cfg.lr

    def fn(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        progress = min(1.0, progress)
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        return floor + (1 - floor) * cos

    return fn


@torch.no_grad()
def val_loss(model: FlowMatchingPolicy, val_cache: list[TrainBatch]) -> tuple[float, dict[str, float]]:
    """Sample-weighted velocity MSE over cached val batches with FIXED noise
    (re-seeded each call), plus its per-modality / per-horizon marginal breakdown
    (``velocity_mse_breakdown``), off the same forward pass. Toggles eval/train."""
    was_training = model.training
    model.eval()
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    total = 0.0
    count = 0
    breakdown_sums: dict[str, float] = {}
    for batch in val_cache:
        n = batch.target.shape[0]
        err2 = flow_loss(model, batch, gen=gen, multi=False)
        total += err2.mean().item() * n
        for k, v in velocity_mse_breakdown(err2).items():
            breakdown_sums[k] = breakdown_sums.get(k, 0.0) + v * n
        count += n
    if was_training:
        model.train()
    return total / count, {k: s / count for k, s in breakdown_sums.items()}


# Channel split inside the A_DIM=14 action vec: [0:6] sticks+triggers (continuous), [6:14] buttons {0,1}.
_N_CONT = 6

# Action-vector modality slices over ACTION_CHANNELS (hal.training.features).
ACTION_MODALITIES: dict[str, slice] = {
    "main_stick": slice(0, 2),
    "c_stick": slice(2, 4),
    "triggers": slice(4, 6),
    "buttons": slice(6, A_DIM),
}


@jaxtyped(typechecker=beartype)
def velocity_mse_breakdown(err2: Float[Tensor, "N L_chunk d_action"]) -> dict[str, float]:
    """Marginal velocity MSE by modality and by horizon (frame offset), from the
    per-element squared error of ``flow_loss`` (reduction='none'), over the flattened
    ``N`` supervised positions.

    ``modality/<name>``: mean over all positions, all frames, that modality's channels.
    ``horizon/frame_<k>``: mean over all positions, all channels, chunk position k (k
    frames into the future). One host sync (a single ``.tolist()``)."""
    names: list[str] = []
    vals: list[Tensor] = []
    for name, sl in ACTION_MODALITIES.items():
        names.append(f"modality/{name}")
        vals.append(err2[..., sl].mean())
    per_frame = err2.mean(dim=(0, 2))  # [L_chunk]
    for k in range(per_frame.shape[0]):
        names.append(f"horizon/frame_{k + 1:02d}")
        vals.append(per_frame[k])
    return dict(zip(names, torch.stack(vals).tolist()))


@torch.no_grad()
def recon_metrics(model: FlowMatchingPolicy, val_cache: list[TrainBatch], *, n_steps: int) -> dict[str, float]:
    """Sample-space reconstruction on cached val batches: integrate a chunk from
    noise (FIXED seed) and score it against the ground-truth chunk. Velocity MSE
    (``val_loss``) is a weak proxy for sample quality — this tracks what the
    closed-loop driver actually executes. Buttons → accuracy + F1 at the 0.5
    decode threshold; sticks/triggers → MAE."""
    was_training = model.training
    model.eval()
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    tp = fp = fn = btn_correct = btn_total = 0
    cont_abs_err = 0.0
    cont_count = 0
    for batch in val_cache:
        pred = integrate_chunk(model, batch.context, n_steps=n_steps, gen=gen)
        tgt = batch.target
        pb = pred[..., _N_CONT:] > 0.5
        tb = tgt[..., _N_CONT:] > 0.5
        tp += int((pb & tb).sum())
        fp += int((pb & ~tb).sum())
        fn += int((~pb & tb).sum())
        btn_correct += int((pb == tb).sum())
        btn_total += pb.numel()
        cont_abs_err += float((pred[..., :_N_CONT] - tgt[..., :_N_CONT]).abs().sum())
        cont_count += tgt[..., :_N_CONT].numel()
    if was_training:
        model.train()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "recon_button_acc": btn_correct / btn_total,
        "recon_button_f1": f1,
        "recon_cont_mae": cont_abs_err / cont_count,
    }


@torch.no_grad()
def likelihood_metrics(
    model: FlowMatchingPolicy,
    val_cache: list[TrainBatch],
    cfg: TrainConfig,
    *,
    n_ode_steps: int = 16,
    n_mc: int = 8,
    max_batches: int = 8,
) -> dict[str, float]:
    """Cross-family comparison metrics (shares ``hal.training.scoring`` with the
    classification experiment): continuous bits/dim via the PF-ODE change-of-variables, and
    a proper Bernoulli button score via a Monte-Carlo threshold bridge (the flow models
    buttons as continuous, so sampling ``n_mc`` chunks and thresholding gives a per-button
    probability). Both on the LAST context position (inference-matched), over a small fixed
    val subset (the PF-ODE is expensive).

    NOTE: ``bits_per_dim`` is the 14-dim *joint* continuous NLL — NOT the classifier's
    6-channel ``cont_density_bits_per_dim``; the buttons score is the apples-to-apples
    cross-family axis."""
    was_training = model.training
    model.eval()
    gen = torch.Generator(device=DEVICE).manual_seed(0)  # FIXED noise so curves track the model, not MC/probe noise
    bpd_sum = 0.0
    bpd_n = 0
    probs_list: list[Tensor] = []
    tgt_list: list[Tensor] = []
    for batch in val_cache[:max_batches]:
        ctx = batch.context
        hidden = model.encode_context(ctx)
        tgt, valid = _position_targets(ctx, batch.target, model.L_chunk)
        last = valid[:, -1]
        cond = hidden[:, -1, :][last]
        x1 = tgt[:, -1][last]
        if cond.shape[0] > 0:
            bpd = scoring.pf_ode_bits_per_dim(
                lambda a, t, cond=cond: model.velocity(cond, a, t), x1, n_steps=n_ode_steps, gen=gen
            )
            bpd_sum += bpd.sum().item()
            bpd_n += bpd.shape[0]
        acc = torch.zeros_like(batch.target[..., _N_CONT:])
        for _ in range(n_mc):
            acc += (integrate_chunk(model, ctx, n_steps=cfg.n_flow_steps, gen=gen)[..., _N_CONT:] > 0.5).float()
        probs_list.append(acc / n_mc)
        tgt_list.append(batch.target[..., _N_CONT:])
    logloss, brier = scoring.bernoulli_scores_from_probs(torch.cat(probs_list), torch.cat(tgt_list))
    if was_training:
        model.train()
    return {
        "bits_per_dim": bpd_sum / max(bpd_n, 1),
        "buttons/logloss_bits": logloss.item(),
        "buttons/brier": brier.item(),
    }


def eval_vs_cpu(
    model: FlowMatchingPolicy,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    max_frames: int,
    replay_dir: Path | None = None,
) -> dict[str, float]:
    """In-training closed-loop eval on FD vs lvl-9 CPU, reduced to a flat metric dict."""
    was_training = model.training
    model.eval()
    try:
        results = sweep_vs_cpu(
            lambda: make_policy(model, stats, cfg),
            session_cfg=default_session_cfg(replay_dir),
            stages=(melee.Stage.FINAL_DESTINATION,),
            replicas=cfg.eval_replicas,
            max_parallel=cfg.eval_max_parallel,
            max_frames=max_frames,
        )
    finally:
        if was_training:
            model.train()
    return vs_cpu_metrics(results)


# %%
def train(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    *,
    comment: str = "",
    resume_run: str | None = None,
    resume_state: dict | None = None,
) -> None:
    run_name = resume_run or make_run_name(_model_tag(cfg), cfg.data_root, comment)
    uploader = BackgroundUploader(run_name) if cfg.push_to_r2 else None
    wandb.init(
        project="hal",
        name=run_name,
        id=resume_state["wandb_id"] if resume_state else None,
        resume="allow" if resume_state else None,
        tags=["flow-matching", "multi-position", f"d{cfg.d_model}", f"L{cfg.n_layers}"],
        config=asdict(cfg),
    )
    ckpt_dir, replay_dir = setup_run_dir(run_name)

    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    if cfg.amp_dtype not in ("bfloat16", "float32"):
        raise ValueError(f"amp_dtype must be 'bfloat16' or 'float32', got {cfg.amp_dtype!r}")
    autocast = (
        torch.autocast(DEVICE, dtype=torch.bfloat16)
        if cfg.amp_dtype == "bfloat16" and DEVICE == "cuda"
        else contextlib.nullcontext()
    )
    model = FlowMatchingPolicy(cfg).to(DEVICE)
    loader_kwargs = dict(
        data_root=cfg.data_root,
        remote=streams.remote_for_local(cfg.data_root),
        cache_limit=f"{cfg.cache_limit_gb}gb",
        shuffle_block_size=cfg.shuffle_block_size,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=cfg.L_chunk,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )
    train_loader = make_loader(
        split="train", num_workers=cfg.num_workers, prefetch_factor=cfg.prefetch_factor, **loader_kwargs
    )
    val_loader = make_loader(split=cfg.val_split, num_workers=0, **loader_kwargs)

    opt = AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
    sched = LambdaLR(opt, lr_schedule(cfg))
    start_step = 0
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        opt.load_state_dict(resume_state["opt"])
        sched.load_state_dict(resume_state["sched"])
        start_step = resume_state["step"] + 1
        print(f"[resume] {run_name}: continuing from step {start_step}", flush=True)

    print("[val] building cached val set…", flush=True)
    val_t0 = time.monotonic()
    val_cache = [b.to(DEVICE) for b in itertools.islice(val_loader, cfg.val_n_batches)]
    if not val_cache:
        raise RuntimeError("val loader yielded zero batches")
    print(
        f"[val] cached {len(val_cache)} batches "
        f"({sum(b.target.shape[0] for b in val_cache)} samples) in {time.monotonic() - val_t0:.1f}s",
        flush=True,
    )

    def _wandb_id() -> str | None:
        return wandb.run.id if wandb.run is not None else None

    def _eval_and_upload(step_tag: str) -> dict[str, float]:
        # Per-step replay subdir so successive evals don't overwrite each other's
        # .slp, then ship the recordings to R2 (keyed under runs/<run>/replays/...).
        sub = replay_dir / step_tag
        metrics = eval_vs_cpu(model, stats, cfg, max_frames=cfg.eval_max_frames, replay_dir=sub)
        if uploader is not None:
            n = uploader.upload_tree(sub, base=ckpt_dir, pattern="*.slp")
            print(f"[eval] queued {n} .slp for R2 ({step_tag})", flush=True)
        return metrics

    model.train()
    it = iter(train_loader)
    run_t0 = time.monotonic()
    for step in range(start_step, cfg.max_steps):
        with profile("step") as sw:
            opt.zero_grad()
            loss_val = 0.0  # mean loss over the effective (accumulated) batch
            errs: list[Tensor] = []  # detached per-element squared error per micro-batch
            for _ in range(cfg.grad_accum_steps):
                try:
                    batch = next(it).to(DEVICE)
                except StopIteration:
                    it = iter(train_loader)
                    batch = next(it).to(DEVICE)
                with autocast:
                    err2 = flow_loss(model, batch, max_positions=cfg.train_positions)
                    loss = err2.mean() / cfg.grad_accum_steps
                loss.backward()
                loss_val += loss.item()
                errs.append(err2.detach())
            opt.step()
            sched.step()
            if DEVICE == "cuda":
                # CUDA kernels are async: without a sync, sw.elapsed times the launch,
                # not the work, and step_s/samples_per_s read ~10-30x too fast (a real
                # past footgun). Sync inside the timed block so throughput is honest.
                torch.cuda.synchronize()
        # Decompose the train loss over the full effective batch outside the timed
        # block, so its extra host sync doesn't inflate the throughput measurement.
        breakdown = velocity_mse_breakdown(torch.cat(errs))
        sps = cfg.batch_size * cfg.grad_accum_steps / sw.elapsed
        wandb.log(
            {
                "train/loss": loss_val,
                **{f"train/loss/{k}": v for k, v in breakdown.items()},
                "train/lr": opt.param_groups[0]["lr"],
                "throughput/step_s": sw.elapsed,
                "throughput/samples_per_s": sps,
            },
            step=step,
        )
        if step < 20 or step % 50 == 0:
            print(
                f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: loss {loss_val:.4f} "
                f"step_dt={sw.elapsed * 1000:.0f}ms ({sps:.1f} samples/s)",
                flush=True,
            )
        if cfg.ckpt_every > 0 and step > 0 and step % cfg.ckpt_every == 0:
            save_checkpoint(
                ckpt_dir / "latest.pt",
                step=step,
                model=model,
                opt=opt,
                sched=sched,
                cfg=asdict(cfg),
                wandb_id=_wandb_id(),
                uploader=uploader,
            )
        if cfg.val_every > 0 and step > 0 and step % cfg.val_every == 0:
            vl, vbreak = val_loss(model, val_cache)
            rm = recon_metrics(model, val_cache, n_steps=cfg.n_flow_steps)
            lm = likelihood_metrics(model, val_cache, cfg)
            wandb.log(
                {
                    "val/loss": vl,
                    **{f"val/loss/{k}": v for k, v in vbreak.items()},
                    **{f"val/{k}": v for k, v in rm.items()},
                    **{f"val/{k}": v for k, v in lm.items()},
                },
                step=step,
            )
            print(
                f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: val_loss {vl:.4f} "
                f"recon_btn_f1 {rm['recon_button_f1']:.3f} recon_cont_mae {rm['recon_cont_mae']:.4f}",
                flush=True,
            )
        if cfg.eval_every > 0 and step > 0 and step % cfg.eval_every == 0:
            save_checkpoint(
                ckpt_dir / f"step_{step:06d}.pt",
                step=step,
                model=model,
                opt=opt,
                sched=sched,
                cfg=asdict(cfg),
                wandb_id=_wandb_id(),
                uploader=uploader,
            )
            metrics = _eval_and_upload(f"step_{step:06d}")
            wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)
            print(f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: closed_loop {metrics}", flush=True)

    vl_final, vbreak_final = val_loss(model, val_cache)
    rm_final = recon_metrics(model, val_cache, n_steps=cfg.n_flow_steps)
    lm_final = likelihood_metrics(model, val_cache, cfg)
    wandb.log(
        {
            "val/loss": vl_final,
            **{f"val/loss/{k}": v for k, v in vbreak_final.items()},
            **{f"val/{k}": v for k, v in rm_final.items()},
            **{f"val/{k}": v for k, v in lm_final.items()},
        },
        step=cfg.max_steps,
    )
    print(f"[final] val_loss {vl_final:.4f} recon {rm_final}", flush=True)
    metrics_final = _eval_and_upload("final")
    wandb.log({f"eval/{k}": v for k, v in metrics_final.items()}, step=cfg.max_steps)
    print(f"[final] closed_loop {metrics_final}", flush=True)
    save_checkpoint(
        ckpt_dir / "final.pt",
        step=cfg.max_steps,
        model=model,
        opt=opt,
        sched=sched,
        cfg=asdict(cfg),
        wandb_id=_wandb_id(),
        uploader=uploader,
    )
    if uploader is not None:
        uploader.close()


# %%
def _load_ckpt(ckpt_path: str) -> tuple[FlowMatchingPolicy, TrainConfig, dict[str, FeatureStats], dict]:
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = TrainConfig(**state["cfg"])
    model = FlowMatchingPolicy(cfg).to(DEVICE)
    model.load_state_dict(state["model"])
    model.eval()
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    return model, cfg, stats, state


def diagnose(ckpt_path: str, *, n_batches: int = 16, k_samples: int = 8) -> None:
    """Offline (no Dolphin) flow diagnostics for an erratic/low-quality policy.

    Three probes, all on held-out val chunks:
      1. reconstruction sweep — ``recon_metrics`` over ``n_flow_steps ∈ {8,32,64}``
         (does more integration crisp up the decoded actions?);
      2. raw-output histograms — pre-threshold 15-dim outputs vs ground truth,
         saved as a PNG next to the checkpoint (are buttons piled at ~0.5?);
      3. cross-sample spread — per-channel std across ``k_samples`` independent
         integrations of the SAME contexts (sampling variance vs multimodality)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model, cfg, stats, state = _load_ckpt(ckpt_path)
    print(f"[diag] loaded {ckpt_path}  step={state['step']}  device={DEVICE}", flush=True)
    loader = make_loader(
        split=cfg.val_split,
        num_workers=0,
        data_root=cfg.data_root,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=cfg.L_chunk,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )
    val_cache = [b.to(DEVICE) for b in itertools.islice(loader, n_batches)]
    if not val_cache:
        raise RuntimeError("val loader yielded zero batches")
    out_dir = Path(ckpt_path).resolve().parent

    print("\n[diag] reconstruction sweep over n_flow_steps", flush=True)
    print(f"  {'steps':>6} {'btn_acc':>8} {'btn_f1':>8} {'cont_mae':>9}", flush=True)
    for n in (8, 32, 64):
        m = recon_metrics(model, val_cache, n_steps=n)
        print(
            f"  {n:>6} {m['recon_button_acc']:>8.3f} {m['recon_button_f1']:>8.3f} {m['recon_cont_mae']:>9.4f}",
            flush=True,
        )

    gen = torch.Generator(device=DEVICE).manual_seed(0)
    pred = torch.cat(
        [
            integrate_chunk(model, b.context, n_steps=cfg.n_flow_steps, gen=gen).reshape(-1, A_DIM).cpu()
            for b in val_cache
        ]
    ).numpy()
    tgt = torch.cat([b.target.reshape(-1, A_DIM).cpu() for b in val_cache]).numpy()
    fig, axes = plt.subplots(3, 5, figsize=(20, 10))
    for i, (ax, ch) in enumerate(zip(axes.ravel(), ACTION_CHANNELS)):
        ax.hist(tgt[:, i], bins=60, alpha=0.5, density=True, label="gt")
        ax.hist(pred[:, i], bins=60, alpha=0.5, density=True, label="pred")
        if ch.startswith("button_") or "trigger" in ch:
            ax.axvline(0.5, color="k", lw=0.6)
        ax.set_title(ch, fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(f"{Path(ckpt_path).name}  step={state['step']}  n_flow_steps={cfg.n_flow_steps}")
    fig.tight_layout()
    hist_path = out_dir / "diag_action_hist.png"
    fig.savefig(hist_path, dpi=100)
    plt.close(fig)
    print(f"\n[diag] wrote raw-output histograms → {hist_path}", flush=True)

    ctx0 = val_cache[0].context
    samples = torch.stack([integrate_chunk(model, ctx0, n_steps=cfg.n_flow_steps) for _ in range(k_samples)])
    spread = samples.std(dim=0).mean(dim=(0, 1)).cpu().numpy()  # [A_DIM]
    print(f"\n[diag] cross-sample std (K={k_samples}, n_steps={cfg.n_flow_steps}):", flush=True)
    for ch, s in zip(ACTION_CHANNELS, spread):
        print(f"  {ch:24s} {s:.4f}", flush=True)


# %%
def eval_ckpt(ckpt_path: str, *, n_flow_steps: int | None = None) -> None:
    """Load a checkpoint, sweep stages vs CPU + self-play, print summaries.

    ``n_flow_steps`` overrides the trained ``cfg.n_flow_steps`` for this eval only
    (test-time compute sweep)."""
    import melee

    from hal.policy import INCLUDED_STAGES

    model, cfg, stats, state = _load_ckpt(ckpt_path)
    print(f"[eval] loaded {ckpt_path}  step={state['step']}  device={DEVICE}", flush=True)

    replay_dir = Path(ckpt_path).resolve().parent / "eval_replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] writing replays to {replay_dir}  (n_flow_steps={n_flow_steps or cfg.n_flow_steps})", flush=True)
    session_cfg = default_session_cfg(replay_dir)
    stages = tuple(s for s in INCLUDED_STAGES if s is not melee.Stage.FOUNTAIN_OF_DREAMS)

    def policy_factory() -> RecedingHorizon:
        return make_policy(model, stats, cfg, n_flow_steps=n_flow_steps)

    print("\n[eval] ============== vs-cpu ==============", flush=True)
    vs_cpu = sweep_vs_cpu(
        policy_factory,
        session_cfg=session_cfg,
        stages=stages,
        replicas=cfg.eval_replicas,
        max_parallel=cfg.eval_max_parallel,
        max_frames=15_000,
    )
    for stage, r, s in vs_cpu:
        print(f"  {stage.name:18s} r{r} {s.as_dict() if s else 'CRASHED'}", flush=True)

    print("\n[eval] ============== self-play ==============", flush=True)
    sp = sweep_self_play(
        policy_factory,
        session_cfg=session_cfg,
        stages=stages,
        replicas=cfg.eval_replicas,
        max_parallel=cfg.eval_max_parallel,
        max_frames=15_000,
    )
    for stage, r, s in sp:
        print(f"  {stage.name:18s} r{r} {s.as_dict() if s else 'CRASHED'}", flush=True)


# %%
@dataclass
class Args:
    """Top-level CLI surface. Pass TrainConfig fields as kebab-case flags,
    e.g. ``--cfg.batch-size 128 --cfg.max-steps 100000``."""

    cfg: TrainConfig = field(default_factory=TrainConfig)
    eval: str | None = None  # ckpt path; if set, closed-loop eval instead of train
    diag: str | None = None  # ckpt path; offline flow diagnostics (no Dolphin), then exit
    eval_flow_steps: int | None = None  # override cfg.n_flow_steps for --eval (test-time compute sweep)
    resume: str | None = None  # run_name to resume; pulls latest.pt (local, else R2)
    comment: str = ""


def main(args: Args) -> None:
    if args.diag is not None:
        diagnose(args.diag)
        return
    if args.eval is not None:
        eval_ckpt(args.eval, n_flow_steps=args.eval_flow_steps)
        return
    if args.resume is not None:
        state = load_for_resume(args.resume, Path("runs") / args.resume, device=DEVICE)
        if state is None:
            raise SystemExit(f"no latest.pt for run {args.resume!r} (local or R2)")
        # Only pure host-scaling knobs (worker/prefetch counts) follow the current code so a
        # resume picks up dataloader fixes. Objective knobs like train_positions are part of
        # the experiment identity and MUST come from the checkpoint, not be silently reset.
        d = TrainConfig()
        cfg = replace(
            TrainConfig(**state["cfg"]),
            num_workers=d.num_workers,
            prefetch_factor=d.prefetch_factor,
        )
        stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
        train(cfg, stats, resume_run=args.resume, resume_state=state)
        return
    cfg = args.cfg
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    auto_comment = f"mp-{cfg.max_steps // 1000}k-b{cfg.batch_size}"
    train(cfg, stats, comment=args.comment or auto_comment)


if __name__ == "__main__":
    main(tyro.cli(Args))
