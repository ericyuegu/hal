"""Flow-matching policy with training-time real-time chunking (RTC).

Single-file experiment. Owns ONLY the variant pieces: the model architecture,
the flow-matching objective (``flow_loss``), the inference integrator
(``predict_chunk``), and the train loop. Everything invariant — the obs/action
codec + dataloader, the closed-loop rolling-buffer driver, run naming /
profiling, checkpointing, and the sim eval harness — is imported from
``hal.training`` / ``hal.eval``.

RTC (arXiv 2512.05964): the chunk of length ``H = L_chunk`` splits into a clean
action **prefix** of length ``d`` (the inference-delay frames already committed by
the previous chunk) and a noisy **postfix** of length ``H - d``. Training feeds the
prefix as ground-truth tokens with no loss and denoises only the postfix;
**per-token** flow-matching timesteps (prefix pinned at t=1, postfix noised) signal
the delay to the model. At inference the integrator pins the first ``d`` positions
to the committed prefix and integrates the rest. The non-RTC baseline lives in
``001_flow_matching_baseline.py``.

Lego-piece contract:
    * new architecture  → copy this file, rewrite ``FlowMatchingPolicy``.
    * new integrator     → edit ``predict_chunk`` (e.g. soft-mask guidance over the
                           H - s overlap, or step-size conditioning).
    * new objective      → edit ``flow_loss``.
The data, eval, and val windows stay fixed, so runs remain comparable.

Tensor-dim names (jaxtyping annotations + docstrings):
    B           = batch
    L_ctx       = context length             (cfg.L_ctx)
    L_chunk     = prediction horizon H        (cfg.L_chunk)
    d           = inference delay / prefix    (cfg.inference_delay)
    s           = execution horizon           (cfg.execution_horizon)
    P           = context prefix length       (= L_ctx)
    d_model     = hidden dim                 (cfg.d_model)
    d_action    = action vec dim (14)        (A_DIM)
    d_time      = time-embedding dim         (cfg.time_emb_dim)

Run:
    python experiments/002_flow_matching_rtc.py                   # train
    python experiments/002_flow_matching_rtc.py --eval <ckpt>     # eval a checkpoint
"""

# %%
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import itertools
import math
import time
import warnings
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
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
from hal.data.stats import FeatureStats
from hal.eval.cross_stage import sweep_self_play
from hal.eval.cross_stage import sweep_vs_cpu
from hal.eval.cross_stage import vs_cpu_metrics
from hal.eval.harness import default_session_cfg
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
    # Seeds model init, flow-matching t/z noise, and the dataloader's window +
    # ego-port sampling — so a run is reproducible and val windows are fixed.
    seed: int = 0
    # window / chunking
    L_ctx: int = 256
    L_chunk: int = 16  # prediction horizon H
    inference_delay: int = 4  # d: committed action prefix; require 0 <= d <= L_chunk - execution_horizon
    execution_horizon: int = 4  # s: actions executed per replan; require 0 < s <= L_chunk
    # inference
    n_flow_steps: int = 8
    # optimization
    batch_size: int = 32  # micro-batch run on the GPU per forward
    grad_accum_steps: int = 1  # optimizer step sees batch_size * grad_accum_steps samples
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 15_000
    # eval cadence
    val_every: int = 500
    val_n_batches: int = 16
    eval_every: int = 2500
    eval_max_frames: int = 3600
    # closed-loop eval parallelism: replicas per stage, run concurrently in
    # waves of eval_max_parallel emulators (one batched forward across all live).
    eval_replicas: int = 1
    eval_max_parallel: int = 4
    # checkpointing: write + background-upload latest.pt every N steps (preemption resilience)
    ckpt_every: int = 1000
    # push checkpoints to R2 as we train (needs AWS_*); --resume pulls them back
    push_to_r2: bool = True
    # data
    data_root: str = "data/processed/ranked-anonymized-1/mds"
    val_split: str = "val"  # tiny datasets may have an empty val split; point this at "test"/"train"
    num_workers: int = 8
    prefetch_factor: int = 8

    def __post_init__(self) -> None:
        if not 0 < self.execution_horizon <= self.L_chunk:
            raise ValueError(
                f"execution_horizon={self.execution_horizon} must satisfy 0 < s <= L_chunk={self.L_chunk}"
            )
        if not 0 <= self.inference_delay <= self.L_chunk - self.execution_horizon:
            raise ValueError(
                f"inference_delay={self.inference_delay} must satisfy 0 <= d <= L_chunk - s={self.L_chunk - self.execution_horizon}"
            )


def _model_tag(cfg: TrainConfig) -> str:
    return f"fm-d{cfg.d_model}-L{cfg.n_layers}-H{cfg.n_heads}-Lc{cfg.L_ctx}-Lk{cfg.L_chunk}-fs{cfg.n_flow_steps}"


# %%
@jaxtyped(typechecker=beartype)
def sinusoidal_time_embedding(t: Float[Tensor, "B L_chunk"], dim: int) -> Float[Tensor, "B L_chunk d_time"]:
    """Per-token flow-matching time embedding: one timestep per chunk position
    (RTC pins the prefix at t=1 and noises the postfix)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t[..., None] * freqs
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FlowMatchingPolicy(nn.Module):
    """Unified Transformer over [L_ctx ctx tokens | L_chunk action tokens].

    Context tokens carry observed ego+opp gamestate + ego controller history.
    Action (chunk) tokens carry the action a_t + a per-token time embedding + a
    learned chunk-type embedding. Under RTC the chunk's first ``d`` tokens are the
    clean committed prefix (their per-token timestep is pinned at t=1); the rest
    are noised. The head reads the chunk positions and predicts the flow-matching
    velocity v̂ ∈ R^{L_chunk × d_action} (loss / integration only on the postfix).

    Split into ``encode_context`` (the L_ctx prefix, fixed across an integration)
    + ``velocity`` (chunk-dependent) so the inference integrator encodes the
    context once and only re-runs the chunk path per Euler step.
    """

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.L_ctx = cfg.L_ctx
        self.L_chunk = cfg.L_chunk
        self.inference_delay = cfg.inference_delay
        d = cfg.d_model
        self.time_emb_dim = cfg.time_emb_dim

        self.cat_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocab, dim) for name, (vocab, dim) in CAT_FEATURES.items()}
        )

        n_float = len(FLOAT_FEATURES)
        n_mask = len(FLOAT_FEATURES)
        n_cat = sum(dim for _, dim in CAT_FEATURES.values())
        per_player_dim = n_float + n_mask + n_cat
        per_frame_in_dim = 2 * per_player_dim + A_DIM  # ego + opp + ego controller history

        self.ctx_proj = nn.Linear(per_frame_in_dim, d)
        self.chunk_proj = nn.Linear(A_DIM, d)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_emb_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.chunk_type_emb = nn.Parameter(torch.zeros(d))
        self.pos_emb = nn.Embedding(self.L_ctx + self.L_chunk, d)

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
        self.head = nn.Linear(d, A_DIM)

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
        not-yet-filled rolling-buffer prefix is handled by attention masking
        (``ctx_pad``), not by zeroing features here."""
        return torch.cat([features[f"ego_{ch}"][..., None] for ch in ACTION_CHANNELS], dim=-1)

    def _context_tokens(self, features: dict[str, Tensor]) -> Float[Tensor, "B L_ctx d_model"]:
        ego = self._per_player_features(features, "ego")
        opp = self._per_player_features(features, "opp")
        hist = self._ego_history_features(features)
        return self.ctx_proj(torch.cat([ego, opp, hist], dim=-1))

    def _key_padding_mask(self, ctx_pad: Int[Tensor, " B"], seq_len: int) -> Tensor:
        """[B, seq_len] bool, True = ignore. Hides each sample's leftmost
        ``ctx_pad`` context positions; chunk tokens (index >= L_ctx) are never
        masked, so no row is fully masked (softmax stays finite)."""
        positions = torch.arange(seq_len, device=ctx_pad.device)[None, :]
        return positions < ctx_pad[:, None]

    def encode_context(self, ctx: Context) -> tuple[Tensor, Tensor | None]:
        """Build the L_ctx context prefix tokens (with positional embeddings) and
        the key-padding mask. Fixed across an integration, so the inference
        integrator calls this once per replan."""
        prefix = self._context_tokens(ctx.features)
        n_prefix = prefix.size(1)
        prefix = prefix + self.pos_emb(torch.arange(n_prefix, device=prefix.device))[None, :, :]
        seq_len = n_prefix + self.L_chunk
        key_padding_mask = self._key_padding_mask(ctx.ctx_pad, seq_len) if ctx.ctx_pad is not None else None
        return prefix, key_padding_mask

    @jaxtyped(typechecker=beartype)
    def velocity(
        self,
        prefix: Float[Tensor, "B P d_model"],
        key_padding_mask: Tensor | None,
        a_t: Float[Tensor, "B L_chunk d_action"],
        t: Float[Tensor, "B L_chunk"],
    ) -> Float[Tensor, "B L_chunk d_action"]:
        n_prefix = prefix.size(1)
        t_proj = self.time_mlp(sinusoidal_time_embedding(t, self.time_emb_dim))
        chunk = self.chunk_proj(a_t) + t_proj + self.chunk_type_emb[None, None, :]
        pos = self.pos_emb(torch.arange(n_prefix, n_prefix + self.L_chunk, device=prefix.device))
        chunk = chunk + pos[None, :, :]
        seq = torch.cat([prefix, chunk], dim=1)
        out = self.encoder(seq, src_key_padding_mask=key_padding_mask)
        return self.head(out[:, n_prefix:, :])

    @jaxtyped(typechecker=beartype)
    def forward(
        self, ctx: Context, a_t: Float[Tensor, "B L_chunk d_action"], t: Float[Tensor, "B L_chunk"]
    ) -> Float[Tensor, "B L_chunk d_action"]:
        prefix, key_padding_mask = self.encode_context(ctx)
        return self.velocity(prefix, key_padding_mask, a_t, t)


# %%
def flow_loss(model: FlowMatchingPolicy, batch: TrainBatch, *, gen: torch.Generator | None = None) -> Tensor:
    """Training-time RTC conditional flow-matching MSE.

    The chunk's first ``d`` positions are the clean ground-truth prefix (per-token
    timestep pinned at t=1, so ``a_t`` there equals the target); the remaining
    postfix is noised at a per-sample t ~ U[0,1]. Loss is the velocity MSE over the
    postfix only — the prefix is pure conditioning."""
    target = batch.target
    B, H, _ = target.shape
    d = model.inference_delay
    t = torch.rand(B, 1, device=target.device, generator=gen).expand(B, H).contiguous()
    if d > 0:
        t[:, :d] = 1.0  # clean prefix tokens
    z = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=gen)
    t_e = t[..., None]
    a_t = (1 - t_e) * z + t_e * target  # prefix (t=1) collapses to the clean target
    v_target = target - z
    v_pred = model(batch.context, a_t, t)
    return F.mse_loss(v_pred[:, d:], v_target[:, d:])


def make_policy(
    model: FlowMatchingPolicy, stats: dict[str, FeatureStats], cfg: TrainConfig, *, device: str = DEVICE
) -> RecedingHorizon:
    """Fresh closed-loop policy for one eval wave (rolling state must not leak).

    ``predict_chunk`` is the inference integrator — Euler from z ~ N(0, I) for
    ``n_flow_steps`` steps. Under RTC the first ``d`` chunk positions are pinned to
    the committed prefix (clean, per-token t=1) every step and only the postfix is
    integrated. The context is encoded once per replan; only the chunk path re-runs
    each step. This is the seam to swap for adjustable test-time compute or
    inference-time soft-mask guidance over the H - s overlap."""

    @torch.no_grad()
    def predict_chunk(ctx: Context, committed: np.ndarray | None) -> np.ndarray:
        prefix, key_padding_mask = model.encode_context(ctx)
        B = ctx.batch
        d = committed.shape[1] if committed is not None else 0
        committed_t = torch.from_numpy(committed).to(device) if d > 0 else None
        a = torch.randn(B, cfg.L_chunk, A_DIM, device=device)
        dt = 1.0 / cfg.n_flow_steps
        for k in range(cfg.n_flow_steps):
            t = torch.full((B, cfg.L_chunk), k * dt, device=device)
            if d > 0:
                a[:, :d] = committed_t  # hold the prefix clean across the integration
                t[:, :d] = 1.0
            a = a + dt * model.velocity(prefix, key_padding_mask, a, t)
        if d > 0:
            a[:, :d] = committed_t
        return a.cpu().numpy()

    return RecedingHorizon(
        predict_chunk=predict_chunk,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=cfg.L_chunk,
        s=cfg.execution_horizon,
        d=cfg.inference_delay,
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
def val_loss(model: FlowMatchingPolicy, val_cache: list[TrainBatch]) -> float:
    """Sample-weighted MSE over cached val batches with FIXED noise (re-seeded
    each call). Toggles model.eval/train."""
    was_training = model.training
    model.eval()
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    total = 0.0
    count = 0
    for batch in val_cache:
        n = batch.target.shape[0]
        total += flow_loss(model, batch, gen=gen).item() * n
        count += n
    if was_training:
        model.train()
    return total / count


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
        tags=["flow-matching", "rtc", f"d{cfg.d_model}", f"L{cfg.n_layers}"],
        config=asdict(cfg),
    )
    ckpt_dir, replay_dir = setup_run_dir(run_name)

    torch.manual_seed(cfg.seed)
    model = FlowMatchingPolicy(cfg).to(DEVICE)
    loader_kwargs = dict(
        data_root=cfg.data_root,
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

    model.train()
    it = iter(train_loader)
    run_t0 = time.monotonic()
    for step in range(start_step, cfg.max_steps):
        with profile("step") as sw:
            opt.zero_grad()
            loss_val = 0.0  # mean loss over the effective (accumulated) batch
            for _ in range(cfg.grad_accum_steps):
                try:
                    batch = next(it).to(DEVICE)
                except StopIteration:
                    it = iter(train_loader)
                    batch = next(it).to(DEVICE)
                loss = flow_loss(model, batch) / cfg.grad_accum_steps
                loss.backward()
                loss_val += loss.item()
            opt.step()
            sched.step()
        sps = cfg.batch_size * cfg.grad_accum_steps / sw.elapsed
        wandb.log(
            {
                "train/loss": loss_val,
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
            vl = val_loss(model, val_cache)
            wandb.log({"val/loss": vl}, step=step)
            print(f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: val_loss {vl:.4f}", flush=True)
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
            metrics = eval_vs_cpu(model, stats, cfg, max_frames=cfg.eval_max_frames, replay_dir=replay_dir)
            wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)
            print(f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: closed_loop {metrics}", flush=True)

    vl_final = val_loss(model, val_cache)
    wandb.log({"val/loss": vl_final}, step=cfg.max_steps)
    print(f"[final] val_loss {vl_final:.4f}", flush=True)
    metrics_final = eval_vs_cpu(model, stats, cfg, max_frames=cfg.eval_max_frames, replay_dir=replay_dir)
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
def eval_ckpt(ckpt_path: str) -> None:
    """Load a checkpoint, sweep stages vs CPU + self-play, print summaries."""
    import melee

    from hal.policy import INCLUDED_STAGES

    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = TrainConfig(**state["cfg"])
    model = FlowMatchingPolicy(cfg).to(DEVICE)
    model.load_state_dict(state["model"])
    model.eval()
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    print(f"[eval] loaded {ckpt_path}  step={state['step']}  device={DEVICE}", flush=True)

    replay_dir = Path(ckpt_path).resolve().parent / "eval_replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] writing replays to {replay_dir}", flush=True)
    session_cfg = default_session_cfg(replay_dir)
    stages = tuple(s for s in INCLUDED_STAGES if s is not melee.Stage.FOUNTAIN_OF_DREAMS)

    def policy_factory() -> RecedingHorizon:
        return make_policy(model, stats, cfg)

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
    eval: str | None = None  # ckpt path; if set, eval instead of train
    resume: str | None = None  # run_name to resume; pulls latest.pt (local, else R2)
    comment: str = ""


def main(args: Args) -> None:
    if args.eval is not None:
        eval_ckpt(args.eval)
        return
    if args.resume is not None:
        state = load_for_resume(args.resume, Path("runs") / args.resume, device=DEVICE)
        if state is None:
            raise SystemExit(f"no latest.pt for run {args.resume!r} (local or R2)")
        cfg = TrainConfig(**state["cfg"])
        stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
        train(cfg, stats, resume_run=args.resume, resume_state=state)
        return
    cfg = args.cfg
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    auto_comment = f"rtc-d{cfg.inference_delay}-s{cfg.execution_horizon}-{cfg.max_steps // 1000}k-b{cfg.batch_size}"
    train(cfg, stats, comment=args.comment or auto_comment)


if __name__ == "__main__":
    main(tyro.cli(Args))
