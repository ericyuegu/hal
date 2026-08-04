"""Advantage-weighted regression (AWR), forked from experiment 016.

Observation assembly, backbone, action heads, optimizer, evaluation machinery and
decode hygiene are the DEPLOYED 016-base recipe (``spatial_features`` off, so the
token is 374 wide, and the four action groups stay INDEPENDENT so 020-vs-016
isolates one axis: 019 owns the head factorization). The studied axis here is WHICH
FRAMES the imitation loss listens to.

The idea. Plain behavior cloning fits every human frame equally, winning and losing
alike, so the policy's ceiling is the average of the data. AWR keeps the same
maximum-likelihood objective and reweights it by how well the demonstrated frame
turned out:

    r_t = -1 on an ego stock loss, +1 on an opponent stock loss   (+ optional damage shaping)
    G_t = sum_k gamma^k r_{t+k}                                    (reverse scan, FULL replay)
    A_t = G_t - V(s_t)                                             (V = a value head on the trunk)
    w_t = clip(exp(A_t / beta), w_max),  rescaled to mean 1 over the batch
    loss = sum over heads/groups of  weighted_mean(w_t, per-frame NLL)  +  value MSE

Frames on the way to taking a stock get upweighted; frames on the way to losing one
get downweighted. ``awr_enabled=False`` restores the exact 016-base objective (the
plain mean NLL), so the arm is a single flag.

The return is computed over the WHOLE replay before windowing. A train window is
~270 frames and gamma=0.999 has a ~700-frame half-life, so a return summed inside
the window would be truncation-biased toward zero exactly where it matters. The one
place a whole replay exists is the row the sampler reads, so the returns are labeled
onto that row as two ordinary per-frame columns (``p1_awr_return`` /
``p2_awr_return``) and then travel through hal's window/pad/ego-relabel path like any
other column. See :class:`ReturnLabeledReplays`.

What stays comparable. Only the BACKPROP objective is weighted: the logged
``train/loss`` and every validation metric are unweighted, so val NLL compares
directly with 016 and 019. The value head is ignored at decode — the deployed policy
is byte-identical to 016's sampler.

Beta. ``awr_beta`` sets how sharply the weights separate; the effective sample size
``train/awr_ess`` is the diagnostic (weights informative but not collapsed onto a few
frames). Pick it from the offline return audit before launching:

    uv run experiments/020_awr.py --audit-returns --cfg.data-root data/processed/dev/mds

Run:
    uv run experiments/020_awr.py
    uv run experiments/020_awr.py --cfg.no-awr-enabled          # the 016-base objective arm
    uv run experiments/020_awr.py --eval <ckpt> --eval-temp 0.7
"""

# %%
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import contextlib
import functools
import itertools
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from beartype import beartype
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import jaxtyped
from streaming import StreamingDataset
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import wandb
from hal import streams
from hal.data.feature_stats import FeatureStats
from hal.data.schema import check_schema_version
from hal.eval.cross_stage import MatchRow
from hal.eval.cross_stage import sweep_vs_cpu_prior_with_rows
from hal.eval.cross_stage import vs_cpu_metrics
from hal.eval.h2h import run_h2h
from hal.eval.harness import default_session_cfg
from hal.eval.harness import usable_cpus
from hal.eval.paired import summarize_paired
from hal.training import scoring
from hal.training.checkpoints import BackgroundUploader
from hal.training.checkpoints import download_latest
from hal.training.checkpoints import load_for_resume
from hal.training.checkpoints import save_checkpoint
from hal.training.closed_loop import RecedingHorizon
from hal.training.dataloader import VAL_L_CHUNK
from hal.training.dataloader import WindowDataset
from hal.training.dataloader import collate_train_batch
from hal.training.dataloader import make_loader
from hal.training.ego_stats import load_consolidated_stats
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import Context
from hal.training.features import TrainBatch
from hal.training.features import stack_actions
from hal.training.muon import SingleDeviceMuonWithAuxAdam
from hal.training.runs import make_run_name
from hal.training.runs import profile
from hal.training.runs import setup_run_dir
from hal.wire import mask_value

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_LN2 = math.log(2.0)

# Action-vector channel split (A_DIM=14): [0:6] sticks+triggers (continuous), [6:14] buttons {0,1}.
_N_CONT = 6
_N_BUTTONS = A_DIM - _N_CONT

# Per-frame input: all four players' gamestate concatenated in the feature dim.
_PLAYER_PREFIXES: tuple[str, ...] = ("ego", "ego_nana", "opp_nana", "opp")

# Output groups (fixed order) + their discrete vocab sizes from the scoring discretizers.
_GROUP_NAMES: tuple[str, ...] = ("buttons", "main_stick", "c_stick", "triggers")
_GROUP_VOCABS: tuple[int, ...] = (
    scoring.N_BUTTON_COMBOS,  # 256
    scoring.STICK_CLUSTER_CENTERS_MAIN.shape[0],  # 65
    scoring.STICK_CLUSTER_CENTERS_C.shape[0],  # 9
    scoring.TRIGGER_CENTERS.shape[0] ** 2,  # 25 (joint L*5 + R)
)
N_GROUPS = len(_GROUP_NAMES)
_BUTTONS_G, _MAIN_G, _C_G, _TRIG_G = range(N_GROUPS)
_GROUP_OFFSETS: tuple[int, ...] = tuple(itertools.accumulate((0,) + _GROUP_VOCABS))[:N_GROUPS]  # (0,256,321,330)
A_VOCAB = sum(_GROUP_VOCABS)  # 355

_BUTTON_COUNTS_VERSION = 1

# Per-frame return columns. They are named like every other per-player MDS column so that
# ``dataloader.relabel_ego`` renames them to ego/opp for free; ``EGO_RETURN_COLUMN`` is what the
# collate reads. ``features._classify`` does not recognize the suffix, so they never reach the model
# as an input feature — they are a TARGET for the value head.
_RETURN_SUFFIX = "awr_return"
_PORT_RETURN_COLUMNS: tuple[str, ...] = tuple(f"{port}_{_RETURN_SUFFIX}" for port in ("p1", "p2"))
EGO_RETURN_COLUMN = f"ego_{_RETURN_SUFFIX}"

# Action-vector channels for the click=>trigger hygiene fix (digital L/R click => analog trigger = 1.0).
_TRIGGER_L_CH = ACTION_CHANNELS.index("trigger_l")
_TRIGGER_R_CH = ACTION_CHANNELS.index("trigger_r")
_BUTTON_L_CH = ACTION_CHANNELS.index("button_l")
_BUTTON_R_CH = ACTION_CHANNELS.index("button_r")


# %%
@dataclass
class TrainConfig:
    # THE studied axis: weight each frame's imitation loss by how well that frame turned out.
    # False restores the 016-base objective EXACTLY (plain mean NLL, no value loss) — the arm is
    # this one flag. Everything below tunes the weighting, never the architecture.
    awr_enabled: bool = True
    # Return discount. 0.999 gives a ~693-frame (11.5 s) half-life: long enough to credit the
    # exchange that led to a stock, short enough that the whole match is not one flat number.
    awr_gamma: float = 0.999
    # Weight temperature: w = exp(A / beta). Small beta separates good frames sharply and collapses
    # the effective sample size; large beta flattens toward plain behavior cloning. 0.5 is read off
    # the offline audit (--audit-returns, dev MDS, gamma=0.999): G has std 0.49 (0.41 within a
    # replay), and beta 0.25 / 0.5 / 1.0 / 2.0 give ESS 0.29 / 0.44 / 0.79 / 0.94 against a global-mean
    # baseline. 0.5 is the one inside the 30-60% band — informative but not collapsed — and it puts
    # only 0.02% of frames on the w_max clip, so the ceiling stays a guard rather than a shaper.
    awr_beta: float = 0.5
    # Ceiling on a single frame's weight, before the batch is rescaled to mean 1. Bounds the
    # variance one lucky trajectory can inject.
    awr_weight_max: float = 20.0
    # Scalar on the value head's MSE inside the backprop objective. The policy loss never
    # backpropagates into the value head (the advantage is detached), so this trains V alone.
    awr_value_loss_weight: float = 1.0
    # Optional dense shaping term added to the sparse stock reward, per percent: the opponent's
    # damage taken minus the ego's, each clipped at >= 0 so a respawn's percent reset is not read
    # as healing. 0 = stock events only.
    awr_damage_shaping: float = 0.0
    # GPT backbone
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 4
    # Multi-token (multi-frame) auxiliary output heads: one independent head per future-frame offset;
    # head o predicts the action o frames ahead. MUST contain 1 — closed-loop decodes only the offset-1
    # head, so the far-horizon heads are a training-only signal. The spread-out default is inherited
    # from 012; the planned ablations compare it with next-only and contiguous alternatives.
    head_offsets: tuple[int, ...] = (1, 5, 9, 13)
    # PER-AUXILIARY-HEAD multiplier. Total auxiliary scalar weight is this times the number of aux heads;
    # use lambda_total / n_aux to implement primary + lambda_total * mean(auxiliary heads).
    aux_loss_weight: float = 1.0
    # Per-sample ego-history input dropout (train only): with probability p, zero a sample's ego
    # controller-history slice of the context token so the trunk cannot lean on copying its own recent
    # inputs. Lives purely in the model's input assembly (targets untouched). 0.0 = current 012 behavior.
    history_dropout_p: float = 0.0
    # Upweight transition-target positions (the predicted group id differs from the frame before it) in the
    # BACKPROP objective only; every logged/val NLL stays unweighted so arms compare. Per (offset, group) the
    # plain mean becomes sum(w·nll)/sum(w) with w=λ on transitions else 1. λ=1.0 reduces exactly to the mean.
    transition_loss_weight: float = 1.0
    # Matchup conditioning (schema v4). char/stage embeddings are indexed by the RAW libmelee id
    # (characters 0-26 dense; stages sparse in 0-26), so the vocab must exceed the max id, not the
    # number of included categories; out-of-range ids clamp to the last row.
    char_vocab: int = 32
    char_dim: int = 12
    stage_vocab: int = 32
    stage_dim: int = 4
    # closed-loop sampling temperature. Greedy argmax collapses the policy to a do-nothing fixed
    # point in closed loop, so deployed play always samples; argmax stays for the recon metric.
    decode_temp: float = 1.0
    # Decode-time hygiene (all default to current behavior). decode_temps overrides the single decode_temp
    # per group in _GROUP_NAMES order (buttons, main_stick, c_stick, triggers); None -> decode_temp for all.
    # decode_btn_support_min >= 1 masks button combos with fewer than that many train frames
    # according to the configured dataset-scoped count artifact to -inf before softmax/argmax.
    # decode_min_p > 0 keeps only classes with
    # p >= decode_min_p * p_max per group, then renormalizes. decode_click_trigger_fix forces trigger_l/r to
    # 1.0 wherever the sampled combo sets the digital L/R bit (the only train-supported click joint).
    decode_temps: tuple[float, float, float, float] | None = None
    decode_btn_support_min: int = 0
    decode_min_p: float = 0.0
    decode_click_trigger_fix: bool = False
    # Chunked execution (deploy-time only): replan every s frames, executing the contiguous heads 1..s from
    # ONE backbone forward (head_offsets must contain 1..s). s=1 = per-frame decode (current 012). Training
    # is unaffected — closed-loop deployment only; eval can override via --eval-exec-horizon.
    exec_horizon: int = 1
    # Reproducible training RNG and transformer context geometry.
    seed: int = 0
    L_ctx: int = 256
    # optimization. batch_size / max_steps below are the DEPLOYED 016-base recipe (read back from that
    # run's checkpoint cfg), not 016's file defaults, so 020-vs-016-base is one axis.
    batch_size: int = 512
    grad_accum_steps: int = 1
    # Two LRs: Muon for the blocks' hidden matrices, AdamW for the input proj / head / embeddings / biases.
    muon_lr: float = 0.02
    adam_lr: float = 8.5e-4
    weight_decay: float = 0.01
    # When False, keep the output-head (heads) weights out of weight decay (route them to the no-decay
    # AdamW group). Default True leaves the heads in the decayed group — exactly the current behavior.
    head_weight_decay: bool = True
    # Shared warmup/cosine schedule and training duration.
    warmup_steps: int = 500
    max_steps: int = 16384
    amp_dtype: str = "bfloat16"  # "bfloat16" | "float32"
    allow_tf32: bool = True
    # eval cadence
    val_every: int = 1024
    val_n_batches: int = 16
    # Exact per-head shared-trunk gradient Gram matrix on this many examples from the first frozen val
    # batch, computed only at validation cadence. Keeps the diagnostic observational and bounded-cost.
    gradient_diagnostic_batch_size: int = 64
    # Validation-only rarity threshold. The metric is emitted only when the checkpoint embeds validated
    # full-dataset button counts; it never falls back to the old reference-sample table.
    diagnostic_rare_button_count: int = 100
    # Closed-loop evaluation cadence and per-boot frame budget.
    eval_every: int = 4096
    eval_max_frames: int = 7200
    # Periodic eval is a cheap diagnostic; final eval uses enough fixed prior-sampled matchups for a useful estimate.
    # These are statistical sample sizes, independent of host concurrency.
    eval_n_matchups: int = 32
    final_eval_n_matchups: int = 96
    eval_seed: int = 0
    # Closed-loop eval parallelism scales with the box: max_parallel = round(this * cpu_count).
    # Each parallel slot is one Dolphin boot that (via instant-restart) plays many prior-sampled
    # matches back-to-back. Profile {0.5, 1, 2, 4} for the best eval throughput vs trainer impact.
    eval_parallel_per_cpu: float = 1.0
    # Closed-loop eval runs in a background subprocess. By default the trainer waits for that
    # subprocess before resuming, so evaluator and trainer never contend for the same CUDA device.
    # Set true only when the evaluator has an actually separate GPU (or intentionally accepts
    # same-device contention).
    eval_overlap_training: bool = False
    # Experimental incremental decoder. Disabled until a locally-trained attention window (or
    # periodic boundary recomputation) is selected; a naive sliding KV cache is not exactly
    # equivalent to recomputing the finite context because cached states contain dropped history.
    eval_incremental_kv: bool = False
    # If an eval is still running at the next boundary, the trainer waits up to this bound and
    # then kills the worker.
    eval_timeout_seconds: float = 2700.0
    # Final in-process mirrored h2h vs a reference run. The cloud box can self-destruct after
    # training, so the sweep runs inside train() and its records/replays upload before exit.
    # None disables the sweep.
    final_h2h_reference_run: str | None = None
    final_h2h_reference_experiment: str = "experiments/016_spatial_features.py"
    final_h2h_reference_label: str = "016-base"
    final_h2h_self_label: str = "020-awr"
    final_h2h_n_configs: int = 64
    # checkpointing
    ckpt_every: int = 2048
    # data (v4 MDS carries the stage + p{1,2}_character + nana columns)
    data_root: str = "data/processed/ranked-anonymized-1/mds"
    # MDS materialization this run reads. ranked-anonymized-1 is materialized at v5, so this run opts
    # DOWN from the code's current SCHEMA_VERSION explicitly and visibly; it flips to 6 once ra-1 is
    # re-materialized. The dataloader's per-row guard rejects any other version — never silent.
    mds_schema_version: int = 5
    # Optional versioned JSON artifact containing full-dataset button-combo counts. Required when
    # decode_btn_support_min > 0; the 012 614-replay reference sample is not authoritative support.
    button_combo_counts_path: str | None = None
    # Streaming dataset cache and shuffle geometry.
    cache_limit_gb: int = 440
    shuffle_block_size: int = 2000
    # Each replay deserialized off disk yields this many non-overlapping windows,
    # amortizing the whole-replay read (the disk bottleneck) over K samples. Train
    # only; val stays 1/replay so its loss stays comparable across runs. Small (4)
    # keeps same-replay windows per batch low.
    windows_per_replay: int = 4
    val_split: str = "val"
    num_workers: int = 16
    prefetch_factor: int = 4


def _model_tag(cfg: TrainConfig) -> str:
    offs = ".".join(str(o) for o in cfg.head_offsets)
    awr = f"-awr{cfg.awr_beta:g}" if cfg.awr_enabled else "-bc"
    return f"gpt-d{cfg.d_model}-L{cfg.n_layers}-h{cfg.n_heads}-Lc{cfg.L_ctx}-o{offs}{awr}"


def _eval_max_parallel(cfg: TrainConfig, n_matchups: int) -> int:
    """Concurrent Dolphin boots per wave; never changes the fixed statistical sample size."""
    return min(n_matchups, max(1, round(cfg.eval_parallel_per_cpu * usable_cpus())))


def _load_button_combo_counts(cfg: TrainConfig) -> Tensor | None:
    """Load and validate the dataset-scoped 256-way button support artifact, when configured."""
    if cfg.button_combo_counts_path is None:
        return None
    path = Path(cfg.button_combo_counts_path)
    data = json.loads(path.read_text())
    required = {"version", "data_root", "total_frames", "counts"}
    if set(data) != required:
        raise ValueError(f"{path}: expected exactly keys {sorted(required)}, got {sorted(data)}")
    if data["version"] != _BUTTON_COUNTS_VERSION:
        raise ValueError(
            f"{path}: button-count version {data['version']} != supported version {_BUTTON_COUNTS_VERSION}"
        )
    artifact_root = Path(data["data_root"]).resolve()
    cfg_root = Path(cfg.data_root).resolve()
    if artifact_root != cfg_root:
        raise ValueError(f"{path}: data_root {artifact_root} does not match configured data_root {cfg_root}")
    counts = data["counts"]
    if not isinstance(counts, list) or len(counts) != scoring.N_BUTTON_COMBOS:
        raise ValueError(f"{path}: counts must contain exactly {scoring.N_BUTTON_COMBOS} entries")
    if any(not isinstance(c, int) or c < 0 for c in counts):
        raise ValueError(f"{path}: every button count must be a non-negative integer")
    if not isinstance(data["total_frames"], int) or data["total_frames"] <= 0:
        raise ValueError(f"{path}: total_frames must be a positive integer")
    if sum(counts) != data["total_frames"]:
        raise ValueError(f"{path}: counts sum to {sum(counts)}, expected total_frames={data['total_frames']}")
    return torch.tensor(counts, dtype=torch.long)


# %%
@jaxtyped(typechecker=beartype)
def quantize_groups(
    main_centers: Float[Tensor, "n_main 2"],
    c_centers: Float[Tensor, "n_c 2"],
    trig_centers: Float[Tensor, " n_trig"],
    actions: Float[Tensor, "*batch d_action"],
) -> Int[Tensor, "*batch n_groups"]:
    """Raw ``A_DIM`` action vec → the four group class indices, in order
    ``(buttons, main_stick, c_stick, triggers)``. Inverse: ``dequantize_groups``."""
    cont, btn = actions[..., :_N_CONT], actions[..., _N_CONT:]
    buttons = scoring.buttons_to_combo(btn)
    main = scoring.nearest_cluster(cont[..., 0:2], main_centers)
    c = scoring.nearest_cluster(cont[..., 2:4], c_centers)
    trig = scoring.nearest_center(cont[..., 4:6], trig_centers)  # [*batch, 2]
    triggers = trig[..., 0] * trig_centers.shape[0] + trig[..., 1]
    return torch.stack([buttons, main, c, triggers], dim=-1)


@jaxtyped(typechecker=beartype)
def dequantize_groups(
    main_centers: Float[Tensor, "n_main 2"],
    c_centers: Float[Tensor, "n_c 2"],
    trig_centers: Float[Tensor, " n_trig"],
    idx: Int[Tensor, "*batch n_groups"],
) -> Float[Tensor, "*batch d_action"]:
    """Inverse of ``quantize_groups``: group class indices → raw ``A_DIM`` action vec
    (``[-1,1]`` sticks, ``[0,1]`` triggers, ``{0,1}`` buttons)."""
    n_trig = trig_centers.shape[0]
    btn = scoring.combo_to_buttons(idx[..., _BUTTONS_G])
    main = scoring.cluster_to_xy(idx[..., _MAIN_G], main_centers)
    c = scoring.cluster_to_xy(idx[..., _C_G], c_centers)
    tl = scoring.center_to_value(idx[..., _TRIG_G] // n_trig, trig_centers)
    tr = scoring.center_to_value(idx[..., _TRIG_G] % n_trig, trig_centers)
    trig = torch.stack([tl, tr], dim=-1)
    return torch.cat([main, c, trig, btn], dim=-1)


# %%
# --- GPT backbone (nanoGPT-style: rotary, RMSNorm, causal SDPA) ---------------
class Rotary(nn.Module):
    inv_freq: Tensor
    cache_key: tuple[int, torch.device, torch.dtype] | None
    cos_cached: Tensor | None
    sin_cached: Tensor | None

    def __init__(self, dim: int, base: int = 10000) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cache_key = None
        self.cos_cached = None
        self.sin_cached = None

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "B L n_heads head_dim"]
    ) -> tuple[
        Float[Tensor, "1 L 1 half_dim"],
        Float[Tensor, "1 L 1 half_dim"],
    ]:
        seq_len = x.shape[1]
        key = (seq_len, x.device, self.inv_freq.dtype)
        if key != self.cache_key:
            self.cache_key = key
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        assert self.cos_cached is not None and self.sin_cached is not None
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

    def at(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """RoPE factors for absolute positions used by incremental decoding."""
        freqs = torch.outer(positions.to(self.inv_freq), self.inv_freq)
        return freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]


@jaxtyped(typechecker=beartype)
def apply_rotary_emb(
    x: Float[Tensor, "B L n_heads head_dim"],
    cos: Float[Tensor, "1 L 1 half_dim"],
    sin: Float[Tensor, "1 L 1 half_dim"],
) -> Float[Tensor, "B L n_heads head_dim"]:
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)


@jaxtyped(typechecker=beartype)
def rmsnorm(x0: Float[Tensor, "... d"], eps: float = 1e-6) -> Float[Tensor, "... d"]:
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        if cfg.n_heads <= 0 or cfg.d_model % cfg.n_heads != 0:
            raise ValueError(f"d_model={cfg.d_model} must be divisible by positive n_heads={cfg.n_heads}")
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        self.head_dim = cfg.d_model // cfg.n_heads
        if self.head_dim % 2 != 0:
            raise ValueError(f"rotary attention head_dim must be even, got {self.head_dim}")
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rotary = Rotary(self.head_dim)

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "B L d_model"], mask: Bool[Tensor, "B 1 L L"]) -> Float[Tensor, "B L d_model"]:
        B, L, _ = x.shape
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, L, self.n_heads, self.head_dim)
        k = k.view(B, L, self.n_heads, self.head_dim)
        v = v.view(B, L, self.n_heads, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.c_proj(y)

    def forward_incremental(
        self, x: Tensor, past: tuple[Tensor, Tensor] | None, position: int, max_cache: int
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """One-token causal attention with a rolling KV cache."""
        B, L, _ = x.shape
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, L, self.n_heads, self.head_dim)
        k = k.view(B, L, self.n_heads, self.head_dim)
        v = v.view(B, L, self.n_heads, self.head_dim)
        # Keep K unrotated in the cache. Re-apply RoPE on the retained window with positions
        # 0..T-1, matching the existing full-context model when the left edge slides.
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        if k.size(2) > max_cache:
            k = k[:, :, -max_cache:]
            v = v[:, :, -max_cache:]
        positions = torch.arange(k.size(2), device=x.device)
        cos, sin = self.rotary.at(positions)
        q = apply_rotary_emb(q, cos[:, -L:], sin[:, -L:])
        k_rot = k.transpose(1, 2)
        k_rot = apply_rotary_emb(k_rot, cos, sin).transpose(1, 2)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k_rot, v)
        y = y.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.c_proj(y), (k, v)


class MLP(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.c_proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "B L d_model"]) -> Float[Tensor, "B L d_model"]:
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg)
        self.attn_scale = 1 / (2 * cfg.n_layers) ** 0.5

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "B L d_model"], mask: Bool[Tensor, "B 1 L L"]) -> Float[Tensor, "B L d_model"]:
        x = x + self.attn_scale * self.attn(rmsnorm(x), mask)
        x = x + self.mlp(rmsnorm(x))
        return x

    def forward_incremental(
        self, x: Tensor, past: tuple[Tensor, Tensor] | None, position: int, max_cache: int
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        attn, kv = self.attn.forward_incremental(rmsnorm(x), past, position, max_cache)
        x = x + self.attn_scale * attn
        x = x + self.mlp(rmsnorm(x))
        return x, kv


# %%
def stock_loss_events(stock: np.ndarray) -> np.ndarray:
    """1.0 on every frame whose stock count DROPS below the frame before it, else 0.0.

    A drop is the event; its size is not (Melee never takes two stocks in one frame). Increments
    are ignored — the counter only rises between games — and frame 0 has no predecessor, so it can
    never fire. A masked sentinel on either side of a step suppresses the event rather than reading
    the sentinel as a huge drop."""
    ids = np.asarray(stock).astype(np.int64)
    known = ids != mask_value(np.int32)
    out = np.zeros(ids.shape, dtype=np.float32)
    out[1:] = ((ids[1:] < ids[:-1]) & known[1:] & known[:-1]).astype(np.float32)
    return out


def damage_taken(percent: np.ndarray) -> np.ndarray:
    """Per-frame INCREASE in a player's percent, clipped at >= 0.

    The drop back to 0 on a respawn (and the masked NaN of an unavailable frame) is a reset, not
    healing, so only rises count."""
    values = np.asarray(percent, dtype=np.float32)
    out = np.zeros(values.shape, dtype=np.float32)
    out[1:] = np.maximum(values[1:] - values[:-1], 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def frame_reward(sample: dict[str, np.ndarray], *, ego: str, opp: str, damage_shaping: float) -> np.ndarray:
    """Per-frame reward for the player at port ``ego`` over one whole replay.

    ``+1`` when the opponent loses a stock, ``-1`` when the ego does (both on the frame the drop
    becomes visible), plus ``damage_shaping`` times the percent the opponent took minus the percent
    the ego took on that frame. The sparse stock term is the outcome the match is scored on; the
    shaping term is off by default and exists to densify the signal, not to redefine it."""
    reward = stock_loss_events(sample[f"{opp}_stock"]) - stock_loss_events(sample[f"{ego}_stock"])
    if damage_shaping:
        dealt = damage_taken(sample[f"{opp}_percent"]) - damage_taken(sample[f"{ego}_percent"])
        reward = reward + damage_shaping * dealt
    return reward


def discounted_returns(reward: np.ndarray, gamma: float) -> np.ndarray:
    """``G_t = sum_k gamma^k r_{t+k}`` to the END OF THE EPISODE, by one reverse scan.

    Run on the full replay before windowing: at gamma=0.999 the discount half-life (~693 frames)
    is longer than a whole train window, so a return summed inside the window would be truncated
    toward zero exactly where the credit lives."""
    tail = itertools.accumulate(
        np.asarray(reward, dtype=np.float32)[::-1].tolist(), lambda carry, r: r + gamma * carry
    )
    return np.fromiter(tail, dtype=np.float32, count=len(reward))[::-1].copy()


def replay_returns(sample: dict[str, np.ndarray], *, gamma: float, damage_shaping: float) -> dict[str, np.ndarray]:
    """Both ports' return columns for one replay row, keyed ``p{1,2}_awr_return``.

    Named per port rather than per role because the sampler picks the ego port AFTER windowing:
    ``dataloader.relabel_ego`` then renames the right one to ``ego_awr_return`` with no AWR-specific
    code, exactly as it does for every gamestate column."""
    return {
        f"{port}_{_RETURN_SUFFIX}": discounted_returns(
            frame_reward(sample, ego=port, opp=other, damage_shaping=damage_shaping), gamma
        )
        for port, other in (("p1", "p2"), ("p2", "p1"))
    }


class ReturnLabeledReplays:
    """The replay stream ``WindowDataset`` iterates, with the two return columns added.

    AWR needs ``G_t`` over the WHOLE episode, and the only place a whole replay exists in the
    pipeline is the MDS row the sampler reads before it slices windows. Labeling the row here means
    the returns are ordinary per-frame columns from that point on: the sampler windows them,
    left-pads them and relabels them ego/opp without knowing AWR exists, and the value target is
    aligned with the action targets by construction.

    REVIEW: the seam is ``WindowDataset._mds`` (see ``_attach_returns``), the one place an
    experiment can inject a per-replay transform without editing ``hal/training/dataloader.py``. A
    ``dataset``/``collate`` argument on ``make_loader`` would make the injection explicit; that is a
    shared-infra change, so it is flagged rather than taken here."""

    def __init__(self, replays: Iterable[dict], *, gamma: float, damage_shaping: float) -> None:
        self._replays = replays
        self._gamma = gamma
        self._damage_shaping = damage_shaping

    def __iter__(self) -> Iterator[dict]:
        for sample in self._replays:
            yield {**sample, **replay_returns(sample, gamma=self._gamma, damage_shaping=self._damage_shaping)}


@dataclass(frozen=True, slots=True)
class AWRBatch:
    """One ``TrainBatch`` plus the ego's discounted return at each CONTEXT position.

    ``TrainBatch`` is the shared, frozen train/eval contract, so the extra target composes with it
    instead of extending it. ``returns[b, t]`` is ``G_t`` at context frame ``t`` — the state whose
    hidden vector the value head reads and whose action every offset head predicts."""

    batch: TrainBatch
    returns: Tensor  # [B, L_ctx] float32

    def to(self, device: str | torch.device) -> AWRBatch:
        return AWRBatch(batch=self.batch.to(device), returns=self.returns.to(device, non_blocking=True))

    def pin_memory(self) -> AWRBatch:
        return AWRBatch(batch=self.batch.pin_memory(), returns=self.returns.pin_memory())

    def valid_returns(self, valid: Bool[Tensor, "B L_ctx"]) -> Float[Tensor, " n_valid"]:
        """``G_t`` flattened by the SAME validity mask the per-position NLLs use, so returns,
        value predictions, weights and NLLs are elementwise aligned."""
        return self.returns.reshape(-1)[valid.reshape(-1)]


def collate_awr_batch(windows: list[dict], *, stats: dict[str, FeatureStats], L_ctx: int) -> AWRBatch:
    """Worker-side collate: hal's ``collate_train_batch`` builds the observation/action batch and the
    ego return column is stacked beside it, sliced to the context positions. The return columns stay
    in the window dicts hal collates: ``features._classify`` does not recognize the suffix, so
    ``preprocess`` drops them and they can never reach the model as an input feature."""
    returns = np.stack([window[EGO_RETURN_COLUMN] for window in windows])[:, :L_ctx]
    batch = collate_train_batch(windows, stats=stats, L_ctx=L_ctx)
    return AWRBatch(batch=batch, returns=torch.from_numpy(np.ascontiguousarray(returns)))


def _attach_returns(loader: DataLoader, cfg: TrainConfig, stats: dict[str, FeatureStats]) -> DataLoader:
    """Re-point a loader hal built at the AWR sample stream and collate.

    hal keeps ownership of everything that is easy to get wrong — the StreamingDataset (shard
    prefetch depth, cache limit, shuffle geometry), the worker/pinning geometry and the window
    sampler itself; this replaces exactly two things: the replay stream (now return-labeled) and the
    collate (now emitting ``AWRBatch``). See ``ReturnLabeledReplays`` for the review flag."""
    sampler = loader.dataset
    if not isinstance(sampler, WindowDataset):
        raise TypeError(f"expected make_loader to yield a WindowDataset sampler, got {type(sampler).__name__}")
    sampler._mds = ReturnLabeledReplays(sampler._mds, gamma=cfg.awr_gamma, damage_shaping=cfg.awr_damage_shaping)
    loader.collate_fn = functools.partial(collate_awr_batch, stats=stats, L_ctx=cfg.L_ctx)
    return loader


def awr_weights(
    advantage: Float[Tensor, " n_valid"], *, beta: float, weight_max: float
) -> tuple[Float[Tensor, " n_valid"], dict[str, float]]:
    """``w = clip(exp(A / beta), w_max)``, rescaled so the batch mean is 1.

    The rescaling keeps the objective's overall scale (hence the effective learning rate) fixed
    whatever the advantage distribution does, so beta changes WHICH frames count, not how loudly.
    The weights are pure data — the advantage must arrive detached, so no gradient can flow into the
    value head through the policy loss. Reported alongside: the effective sample size fraction
    ``(sum w)^2 / (N * sum w^2)`` (1.0 = uniform weighting, small = a few frames own the batch) and
    the fraction of positions sitting at the clip."""
    if advantage.requires_grad:
        raise ValueError("AWR weights must be built from a DETACHED advantage; the policy loss never trains V")
    raw = torch.exp(advantage / beta).clamp(max=weight_max)
    weight = raw / raw.mean().clamp_min(torch.finfo(raw.dtype).tiny)
    n = weight.numel()
    if n == 0:
        return weight, {"ess": 0.0, "weight_max_frac": 0.0}
    ess = (weight.sum().pow(2) / (n * weight.pow(2).sum())).item()
    return weight, {"ess": ess, "weight_max_frac": (raw >= weight_max).float().mean().item()}


class GPT(nn.Module):
    """Causal GPT over per-frame tokens with multi-token auxiliary heads. ``hidden[i]`` (causal) feeds
    one independent head per offset in ``cfg.head_offsets``; head ``o`` predicts the action ``o`` frames
    ahead via a joint ``A_VOCAB`` logit vector (the concatenation of the four group vocabs). Closed-loop
    decode uses only the offset-1 head (``primary_head_idx``); the rest are an auxiliary training signal.

    ``value_head`` is 020's only architectural addition: one scalar per frame off the same trunk hidden
    state, trained to the discounted return so the AWR advantage has a baseline. It is built LAST so
    every other parameter draws the same initialization 016 does from the same seed, and decode never
    reads it — the deployed policy is 016's."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        if not cfg.decode_temp > 0:
            raise ValueError(f"decode_temp must be > 0, got {cfg.decode_temp}")
        if not 0.0 <= cfg.history_dropout_p <= 1.0:
            raise ValueError(f"history_dropout_p must be in [0, 1], got {cfg.history_dropout_p}")
        self.history_dropout_p = cfg.history_dropout_p
        offs = tuple(cfg.head_offsets)
        if not offs:
            raise ValueError("head_offsets must be non-empty")
        if any(o < 1 for o in offs):
            raise ValueError(f"head_offsets must all be >= 1, got {offs}")
        if len(set(offs)) != len(offs):
            raise ValueError(f"head_offsets must be unique, got {offs}")
        if 1 not in offs:
            raise ValueError(f"head_offsets must contain 1 (closed-loop next-frame decode), got {offs}")
        self.head_offsets = offs
        self.primary_head_idx = offs.index(1)
        self.L_ctx = cfg.L_ctx

        # Gamestate categoricals: one table per feature name, shared across the four players.
        self.cat_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocab, dim) for name, (vocab, dim) in CAT_FEATURES.items()}
        )
        self.char_emb = nn.Embedding(cfg.char_vocab, cfg.char_dim)
        self.stage_emb = nn.Embedding(cfg.stage_vocab, cfg.stage_dim)
        per_player = len(FLOAT_FEATURES) * 2 + sum(dim for _, dim in CAT_FEATURES.values())  # float+mask+cat
        d_in = len(_PLAYER_PREFIXES) * per_player + A_DIM + 2 * cfg.char_dim + cfg.stage_dim  # 374

        self.ctx_proj = nn.Linear(d_in, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        # One independent joint head per future-frame offset (order matches self.head_offsets).
        self.heads = nn.ModuleList([nn.Linear(cfg.d_model, A_VOCAB) for _ in offs])
        # V(s) for the AWR advantage. Last, so the trunk and the action heads keep 016's draws.
        self.value_head = nn.Linear(cfg.d_model, 1)

        # Stick/trigger center grids (registered so they move with .to() and serialize).
        self.register_buffer("main_centers", scoring.STICK_CLUSTER_CENTERS_MAIN.clone())
        self.register_buffer("c_centers", scoring.STICK_CLUSTER_CENTERS_C.clone())
        self.register_buffer("trig_centers", scoring.TRIGGER_CENTERS.clone())
        # -1 means unavailable. A validated full-dataset artifact populates this at train start, and the
        # buffer then travels with checkpoints so later eval never depends on a mutable sidecar file.
        self.register_buffer("button_combo_counts", torch.full((scoring.N_BUTTON_COMBOS,), -1, dtype=torch.long))
        self._btn_support_dead_cache: dict[tuple[int, torch.device], Tensor] = {}

    def _per_player_features(self, features: dict[str, Tensor], prefix: str) -> Tensor:
        ref = features[f"{prefix}_position_x"]
        B, L = ref.shape
        device = ref.device
        parts: list[Tensor] = [features[f"{prefix}_{feat}"][..., None] for feat in FLOAT_FEATURES]
        for feat in FLOAT_FEATURES:
            mk = f"{prefix}_{feat}_mask"
            parts.append(features[mk][..., None] if mk in features else torch.zeros(B, L, 1, device=device))
        for name, (vocab, _) in CAT_FEATURES.items():
            parts.append(self.cat_embeds[name](features[f"{prefix}_{name}"].clamp(0, vocab - 1)))
        return torch.cat(parts, dim=-1)

    def _context_tokens(self, features: dict[str, Tensor]) -> Float[Tensor, "B L_ctx d_model"]:
        parts = [self._per_player_features(features, p) for p in _PLAYER_PREFIXES]
        # Ego controller history slice, assembled into a FRESH tensor (never mutating `features`, so the
        # targets built from stack_actions(features) stay intact). Per-sample history dropout (train only):
        # draw a Bernoulli keep-mask via the module RNG (probability history_dropout_p of dropping) and zero
        # the whole slice for dropped samples, forcing the trunk off copying its own recent inputs.
        ego_hist = torch.cat([features[f"ego_{ch}"][..., None] for ch in ACTION_CHANNELS], dim=-1)
        if self.training and self.history_dropout_p > 0.0:
            keep = (torch.rand(ego_hist.shape[0], device=ego_hist.device) >= self.history_dropout_p).to(ego_hist.dtype)
            ego_hist = ego_hist * keep[:, None, None]
        parts.append(ego_hist)
        parts.append(self.char_emb(features["ego_character"].clamp(0, self.char_emb.num_embeddings - 1)))
        parts.append(self.char_emb(features["opp_character"].clamp(0, self.char_emb.num_embeddings - 1)))
        parts.append(self.stage_emb(features["stage"].clamp(0, self.stage_emb.num_embeddings - 1)))
        return self.ctx_proj(torch.cat(parts, dim=-1))

    def _attn_mask(self, ctx_pad: Int[Tensor, " B"], L: int, device: torch.device) -> Bool[Tensor, "B 1 L L"]:
        """Causal mask that also hides each sample's left-padded cold-start prefix (key < ctx_pad).
        A padded query keeps its diagonal so its row is never fully masked (SDPA would NaN)."""
        idx = torch.arange(L, device=device)
        causal = idx[:, None] >= idx[None, :]
        key_real = idx[None, :] >= ctx_pad[:, None]
        diag = torch.eye(L, dtype=torch.bool, device=device)
        return (causal[None] & (key_real[:, None, :] | diag[None]))[:, None]

    def forward(self, features: dict[str, Tensor], ctx_pad: Int[Tensor, " B"]) -> Float[Tensor, "B L_ctx d_model"]:
        """Backbone hidden (one rmsnorm'd vector per frame); callers apply the per-offset heads."""
        x = self._context_tokens(features)
        mask = self._attn_mask(ctx_pad, x.size(1), x.device)
        for block in self.blocks:
            x = block(x, mask)
        return rmsnorm(x)

    @torch.no_grad()
    def forward_incremental(
        self,
        features: dict[str, Tensor],
        past: list[tuple[Tensor, Tensor] | None],
        position: int,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """Encode one current token using per-layer rolling KV state."""
        x = self._context_tokens(features)
        if x.size(1) != 1:
            raise ValueError(f"incremental decode expects one token, got L={x.size(1)}")
        new_past: list[tuple[Tensor, Tensor]] = []
        for block, old in zip(self.blocks, past, strict=True):
            x, kv = block.forward_incremental(x, old, position, self.L_ctx)
            new_past.append(kv)
        return rmsnorm(x)[:, -1], new_past


# %%
def _quantize(model: GPT, actions: Tensor) -> Tensor:
    return quantize_groups(model.main_centers, model.c_centers, model.trig_centers, actions)


def _dequantize(model: GPT, idx: Tensor) -> Tensor:
    return dequantize_groups(model.main_centers, model.c_centers, model.trig_centers, idx)


def _multi_offset_targets(ctx: Context, target: Tensor, offsets: tuple[int, ...]) -> tuple[dict[int, Tensor], Tensor]:
    """Per context position ``i`` and offset ``o``, the action ``o`` frames ahead + a shared validity
    mask. The ego controller history already lives in ``ctx.features``, so ``a_full = [history | target]``
    spans frames ``0 .. L_ctx + max(offsets) - 1`` and offset ``o``'s target is ``a_full[:, o : o + L_ctx]``
    (offset 1 recovers 011's next-frame target). A position is valid iff it is a real (non-pad) context
    frame; ``i >= ctx_pad`` then guarantees the target frame ``i + o`` is real too (``o >= 1``)."""
    a_full = torch.cat([stack_actions(ctx.features), target], dim=1)  # [B, L_ctx + max_off, A_DIM]
    L_ctx = a_full.size(1) - target.size(1)
    pos = torch.arange(L_ctx, device=a_full.device)
    valid = pos[None, :] >= ctx.ctx_pad[:, None]  # [B, L_ctx], shared by all offsets
    targets = {o: a_full[:, o : o + L_ctx] for o in offsets}  # each [B, L_ctx, A_DIM]
    return targets, valid


def group_nll(logits: Tensor, tgt_idx: Tensor, valid: Tensor) -> dict[str, Tensor]:
    """Per-group categorical NLL (nats) over the VALID positions only. Returns ``{name: [n_valid]}``
    1D tensors (same ordering across groups) so callers reduce once for exact sample weighting."""
    flat_valid = valid.reshape(-1)
    out: dict[str, Tensor] = {}
    for g, name in enumerate(_GROUP_NAMES):
        lo = _GROUP_OFFSETS[g]
        lg = logits[..., lo : lo + _GROUP_VOCABS[g]].reshape(-1, _GROUP_VOCABS[g])[flat_valid]
        out[name] = F.cross_entropy(lg, tgt_idx[..., g].reshape(-1)[flat_valid], reduction="none")
    return out


@dataclass(frozen=True, slots=True)
class LossParts:
    """One backbone forward's per-position pieces, all flattened by the SAME validity mask.

    ``nll`` and ``transition`` are keyed ``(offset, group_name)`` → ``[n_valid]`` (nats; bool);
    ``value`` is ``V(s_t)`` at those same positions; ``valid`` is the ``[B, L_ctx]`` mask that
    produced the flattening, so a caller can align any other per-position quantity (the AWR
    returns) with them."""

    nll: dict[tuple[int, str], Tensor]
    transition: dict[tuple[int, str], Tensor]
    value: Float[Tensor, " n_valid"]
    valid: Bool[Tensor, "B L_ctx"]


def action_loss(model: GPT, batch: TrainBatch) -> LossParts:
    """Dense multi-token NLL + aligned transition flags. Every valid context position predicts the action at
    each head offset; one shared backbone forward, one head each. ``a_full = [history | target]`` is quantized
    ONCE ([B, L_ctx+max_off, n_groups]) and its per-frame boundary mask computed once, both sliced per offset:
    for head ``o`` position ``i``'s target frame is ``i+o`` and it is a transition iff ``q[i+o] != q[i+o-1]``.
    The value head reads the SAME hidden state, so the AWR baseline costs one 256->1 matmul and no second
    forward; when ``awr_enabled`` is false nothing consumes it and it takes no gradient."""
    ctx = batch.context
    max_offset = max(model.head_offsets)
    if batch.target.size(1) < max_offset:
        raise ValueError(
            f"target chunk has {batch.target.size(1)} frames, but max(head_offsets)={max_offset}; "
            "the target horizon must cover every prediction head"
        )
    h = model(ctx.features, ctx.ctx_pad)  # [B, L_ctx, d_model]
    a_full = torch.cat([stack_actions(ctx.features), batch.target], dim=1)  # [B, L_ctx + max_off, A_DIM]
    q_full = _quantize(model, a_full)  # [B, L_ctx + max_off, n_groups]
    L_ctx = a_full.size(1) - batch.target.size(1)
    pos = torch.arange(L_ctx, device=a_full.device)
    valid = pos[None, :] >= ctx.ctx_pad[:, None]  # [B, L_ctx], shared by all offsets
    flat_valid = valid.reshape(-1)
    bnd_full = scoring.transition_mask(q_full)  # [B, L_ctx + max_off - 1, n_groups]; pos t = (q[t+1] != q[t])
    nll: dict[tuple[int, str], Tensor] = {}
    trans: dict[tuple[int, str], Tensor] = {}
    for hi, o in enumerate(model.head_offsets):
        logits = model.heads[hi](h).float()  # [B, L_ctx, A_VOCAB]
        tgt_idx = q_full[:, o : o + L_ctx]  # target frame i+o
        bnd_o = bnd_full[:, o - 1 : o - 1 + L_ctx]  # transition at i iff q[i+o] != q[i+o-1]
        gnll = group_nll(logits, tgt_idx, valid)
        for g, name in enumerate(_GROUP_NAMES):
            nll[(o, name)] = gnll[name]
            trans[(o, name)] = bnd_o[..., g].reshape(-1)[flat_valid]
    value = model.value_head(h).float().reshape(-1)[flat_valid]  # [n_valid] V(s_i)
    return LossParts(nll=nll, transition=trans, value=value, valid=valid)


def _btn_support_dead(model: GPT, min_count: int, device: torch.device) -> Tensor:
    """``[N_BUTTON_COMBOS]`` bool mask, True on button combos with fewer than ``min_count`` train
    frames according to the checkpoint's dataset-scoped counts. Cached per ``(min_count, device)`` so
    closed-loop decode builds it once, never per frame."""
    counts = model.button_combo_counts
    if bool((counts < 0).any()):
        raise ValueError(
            "decode_btn_support_min > 0 requires a validated dataset-scoped button-count artifact; "
            "set TrainConfig.button_combo_counts_path when training a new checkpoint"
        )
    dead = model._btn_support_dead_cache.get((min_count, device))
    if dead is None:
        counts = counts.to(device)
        dead = counts < min_count
        if bool(dead.all()):
            raise ValueError(
                f"btn_support_min={min_count} masks every button combo (max train count is {int(counts.max())})"
            )
        model._btn_support_dead_cache[(min_count, device)] = dead
    return dead


def _resolve_decode_args(
    temp: float,
    temps: tuple[float, float, float, float] | None,
    btn_support_min: int,
    min_p: float,
    argmax: bool,
) -> tuple[float, ...]:
    """Validate the shared decode knobs and resolve the per-group temperatures (buttons, main_stick, c_stick,
    triggers): ``temps`` overrides the scalar ``temp`` when given. Fails loud on a wrong-length ``temps``, a
    negative support floor, a min-p outside [0, 1], or a non-positive sampling temperature."""
    if temps is not None and len(temps) != N_GROUPS:
        raise ValueError(f"decode temps must have {N_GROUPS} entries (one per group), got {len(temps)}")
    if btn_support_min < 0:
        raise ValueError(f"decode btn_support_min must be >= 0, got {btn_support_min}")
    if not math.isfinite(min_p) or not 0.0 <= min_p <= 1.0:
        raise ValueError(f"decode min_p must be in [0, 1], got {min_p}")
    group_temps = (temp,) * N_GROUPS if temps is None else temps
    if not argmax and any(not math.isfinite(t) or t <= 0 for t in group_temps):
        raise ValueError(f"decode temperatures must be > 0, got {group_temps}")
    return group_temps


def _sample_action_from_logits(
    model: GPT,
    logits: Tensor,
    *,
    group_temps: tuple[float, ...],
    btn_support_min: int,
    min_p: float,
    click_trigger_fix: bool,
    argmax: bool,
    gen: torch.Generator | None,
) -> Float[Tensor, "B d_action"]:
    """Sample one action vector from a single head's ``[B, A_VOCAB]`` joint logits, applying the decode
    hygiene in order support-mask -> per-group temperature -> min-p -> per-group sample (or ``argmax``),
    then dequantize + the click=>trigger fix. Shared by the offset-1 ``decode`` and the chunked
    ``decode_chunk`` so the per-group sampler never forks between the two deploy paths."""
    if btn_support_min >= 1:
        dead = _btn_support_dead(model, btn_support_min, logits.device)
        logits = logits.clone()
        logits[:, : scoring.N_BUTTON_COMBOS].masked_fill_(dead, float("-inf"))
    picks: list[Tensor] = []
    for g in range(N_GROUPS):
        lo = _GROUP_OFFSETS[g]
        lg = logits[:, lo : lo + _GROUP_VOCABS[g]]
        if argmax:
            picks.append(lg.argmax(-1))
            continue
        probs = F.softmax(lg / group_temps[g], dim=-1)
        if min_p > 0:
            probs = probs * (probs >= min_p * probs.amax(dim=-1, keepdim=True))
            probs = probs / probs.sum(dim=-1, keepdim=True)
        picks.append(torch.multinomial(probs, 1, generator=gen).squeeze(-1))
    idx = torch.stack(picks, dim=-1)  # [B, N_GROUPS]
    a = _dequantize(model, idx)  # [B, A_DIM]
    if click_trigger_fix:
        a[..., _TRIGGER_L_CH] = torch.where(a[..., _BUTTON_L_CH] > 0.5, 1.0, a[..., _TRIGGER_L_CH])
        a[..., _TRIGGER_R_CH] = torch.where(a[..., _BUTTON_R_CH] > 0.5, 1.0, a[..., _TRIGGER_R_CH])
    return a


@torch.no_grad()
def decode(
    model: GPT,
    ctx: Context,
    *,
    temp: float = 1.0,
    temps: tuple[float, float, float, float] | None = None,
    btn_support_min: int = 0,
    min_p: float = 0.0,
    click_trigger_fix: bool = False,
    argmax: bool = False,
    gen: torch.Generator | None = None,
) -> Float[Tensor, "B 1 d_action"]:
    """One next-frame action per sample from the LAST context position, in raw action ranges, via the
    offset-1 (primary) head only. Each group's logit slice is sampled (per-group ``temp``-scaled softmax,
    optional min-p nucleus) or taken greedily (``argmax``, for the recon metric) independently. Decode-time
    hygiene applied in order support-mask -> per-group temperature -> min-p -> sample: ``btn_support_min``
    >= 1 masks button combos with fewer than that many train frames to -inf; ``temps`` overrides ``temp``
    per group (buttons, main_stick, c_stick, triggers); ``min_p`` > 0 keeps only classes with
    ``p >= min_p * p_max`` then renormalizes; ``click_trigger_fix`` forces trigger_l/r to 1.0 wherever the
    sampled combo sets the digital L/R bit. ``argmax`` ignores ``temps``/``min_p`` but respects the mask.
    The value head takes no part: this is 016's sampler."""
    group_temps = _resolve_decode_args(temp, temps, btn_support_min, min_p, argmax)
    h = model(ctx.features, ctx.ctx_pad)[:, -1]  # [B, d_model]
    logits = model.heads[model.primary_head_idx](h).float()  # [B, A_VOCAB]
    a = _sample_action_from_logits(
        model,
        logits,
        group_temps=group_temps,
        btn_support_min=btn_support_min,
        min_p=min_p,
        click_trigger_fix=click_trigger_fix,
        argmax=argmax,
        gen=gen,
    )
    return a[:, None, :]  # [B, 1, A_DIM]


@torch.no_grad()
def decode_chunk(
    model: GPT,
    ctx: Context,
    offsets: tuple[int, ...],
    *,
    temp: float = 1.0,
    temps: tuple[float, float, float, float] | None = None,
    btn_support_min: int = 0,
    min_p: float = 0.0,
    click_trigger_fix: bool = False,
    argmax: bool = False,
    gen: torch.Generator | None = None,
) -> Float[Tensor, "B s d_action"]:
    """Chunked receding-horizon decode: from ONE backbone forward's last hidden state, sample the action at
    each offset in ``offsets`` (the contiguous execution horizon 1..s) via that offset's own head, stacked in
    offset order → ``[B, s, A_DIM]``. One forward per s frames — the deploy-time saving of chunked execution.
    Same per-group sampling + hygiene as ``decode``; ``offsets == (1,)`` is byte-identical to ``decode``."""
    group_temps = _resolve_decode_args(temp, temps, btn_support_min, min_p, argmax)
    h = model(ctx.features, ctx.ctx_pad)[:, -1]  # [B, d_model]
    actions: list[Tensor] = []
    for o in offsets:
        logits = model.heads[model.head_offsets.index(o)](h).float()  # [B, A_VOCAB]
        actions.append(
            _sample_action_from_logits(
                model,
                logits,
                group_temps=group_temps,
                btn_support_min=btn_support_min,
                min_p=min_p,
                click_trigger_fix=click_trigger_fix,
                argmax=argmax,
                gen=gen,
            )
        )
    return torch.stack(actions, dim=1)  # [B, s, A_DIM]


def _exec_horizon_offsets(head_offsets: tuple[int, ...], s: int) -> tuple[int, ...]:
    """The contiguous execution-horizon offsets ``(1, ..., s)`` decoded per chunk. Fails loud if ``s < 1`` or
    ``head_offsets`` is missing any of them (e.g. offsets (1,5,9,13) with s=2 has no offset-2 head)."""
    if s < 1:
        raise ValueError(f"exec_horizon must be >= 1, got {s}")
    required = tuple(range(1, s + 1))
    missing = [o for o in required if o not in head_offsets]
    if missing:
        raise ValueError(
            f"exec_horizon={s} needs head_offsets to contain the contiguous prefix {required}; "
            f"missing {missing} in {head_offsets}"
        )
    return required


def validate_config(cfg: TrainConfig, *, has_button_combo_counts: bool) -> None:
    """Fail before W&B, loader construction, or Dolphin startup on invalid experiment geometry."""
    positive_ints = {
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "L_ctx": cfg.L_ctx,
        "batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
        "max_steps": cfg.max_steps,
        "exec_horizon": cfg.exec_horizon,
        "char_vocab": cfg.char_vocab,
        "char_dim": cfg.char_dim,
        "stage_vocab": cfg.stage_vocab,
        "stage_dim": cfg.stage_dim,
        "val_n_batches": cfg.val_n_batches,
        "gradient_diagnostic_batch_size": cfg.gradient_diagnostic_batch_size,
        "diagnostic_rare_button_count": cfg.diagnostic_rare_button_count,
        "eval_n_matchups": cfg.eval_n_matchups,
        "final_eval_n_matchups": cfg.final_eval_n_matchups,
        "eval_max_frames": cfg.eval_max_frames,
        "windows_per_replay": cfg.windows_per_replay,
        "shuffle_block_size": cfg.shuffle_block_size,
        "prefetch_factor": cfg.prefetch_factor,
        "cache_limit_gb": cfg.cache_limit_gb,
        "mds_schema_version": cfg.mds_schema_version,
    }
    for name, value in positive_ints.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if cfg.final_h2h_reference_run is not None:
        if not cfg.final_h2h_reference_run:
            raise ValueError("final_h2h_reference_run must be a run name or None, not an empty string")
        if cfg.final_h2h_n_configs < 1:
            raise ValueError(f"final_h2h_n_configs must be >= 1, got {cfg.final_h2h_n_configs}")
        if cfg.final_h2h_self_label == cfg.final_h2h_reference_label:
            raise ValueError(f"h2h labels must differ, got {cfg.final_h2h_self_label!r} twice")
        exp = cfg.final_h2h_reference_experiment
        if exp.endswith(".py") and not Path(exp).exists():
            raise ValueError(f"final_h2h_reference_experiment does not exist: {exp}")
    if cfg.d_model % cfg.n_heads != 0:
        raise ValueError(f"d_model={cfg.d_model} must be divisible by n_heads={cfg.n_heads}")
    head_dim = cfg.d_model // cfg.n_heads
    if head_dim % 2:
        raise ValueError(f"rotary attention head_dim=d_model/n_heads={head_dim} must be even")
    offsets = tuple(cfg.head_offsets)
    if not offsets or any(not isinstance(o, int) or isinstance(o, bool) or o < 1 for o in offsets):
        raise ValueError(f"head_offsets must be non-empty positive integers, got {offsets}")
    if len(set(offsets)) != len(offsets):
        raise ValueError(f"head_offsets must be unique, got {offsets}")
    if 1 not in offsets:
        raise ValueError(f"head_offsets must contain 1 (the deployed next-frame head), got {offsets}")
    if max(offsets) > VAL_L_CHUNK:
        raise ValueError(f"max(head_offsets)={max(offsets)} exceeds frozen VAL_L_CHUNK={VAL_L_CHUNK}")
    _exec_horizon_offsets(offsets, cfg.exec_horizon)
    finite_nonnegative = {
        "aux_loss_weight": cfg.aux_loss_weight,
        "weight_decay": cfg.weight_decay,
    }
    for name, value in finite_nonnegative.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    finite_positive = {
        "muon_lr": cfg.muon_lr,
        "adam_lr": cfg.adam_lr,
        "eval_parallel_per_cpu": cfg.eval_parallel_per_cpu,
        "eval_timeout_seconds": cfg.eval_timeout_seconds,
    }
    for name, value in finite_positive.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    if not math.isfinite(cfg.transition_loss_weight) or cfg.transition_loss_weight <= 0:
        raise ValueError(f"transition_loss_weight must be finite and > 0, got {cfg.transition_loss_weight!r}")
    if not math.isfinite(cfg.history_dropout_p) or not 0.0 <= cfg.history_dropout_p <= 1.0:
        raise ValueError(f"history_dropout_p must be in [0, 1], got {cfg.history_dropout_p!r}")
    if not isinstance(cfg.awr_enabled, bool):
        raise ValueError(f"awr_enabled must be a bool, got {cfg.awr_enabled!r}")
    if not math.isfinite(cfg.awr_gamma) or not 0.0 < cfg.awr_gamma <= 1.0:
        raise ValueError(f"awr_gamma must be in (0, 1], got {cfg.awr_gamma!r}")
    if not math.isfinite(cfg.awr_beta) or cfg.awr_beta <= 0:
        raise ValueError(f"awr_beta must be finite and > 0, got {cfg.awr_beta!r}")
    if not math.isfinite(cfg.awr_weight_max) or cfg.awr_weight_max <= 0:
        raise ValueError(f"awr_weight_max must be finite and > 0, got {cfg.awr_weight_max!r}")
    for name in ("awr_value_loss_weight", "awr_damage_shaping"):
        value = getattr(cfg, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    if not isinstance(cfg.warmup_steps, int) or isinstance(cfg.warmup_steps, bool) or cfg.warmup_steps < 0:
        raise ValueError(f"warmup_steps must be a non-negative integer, got {cfg.warmup_steps!r}")
    if cfg.warmup_steps > cfg.max_steps:
        raise ValueError(f"warmup_steps={cfg.warmup_steps} exceeds max_steps={cfg.max_steps}")
    for name in ("val_every", "eval_every", "ckpt_every", "num_workers"):
        value = getattr(cfg, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    for name in ("seed", "eval_seed"):
        value = getattr(cfg, name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer, got {value!r}")
    if not isinstance(cfg.decode_btn_support_min, int) or isinstance(cfg.decode_btn_support_min, bool):
        raise ValueError(f"decode_btn_support_min must be a non-negative integer, got {cfg.decode_btn_support_min!r}")
    if not cfg.data_root or not cfg.val_split:
        raise ValueError("data_root and val_split must be non-empty")
    if cfg.amp_dtype not in ("bfloat16", "float32"):
        raise ValueError(f"amp_dtype must be 'bfloat16' or 'float32', got {cfg.amp_dtype!r}")
    _resolve_decode_args(
        cfg.decode_temp,
        cfg.decode_temps,
        cfg.decode_btn_support_min,
        cfg.decode_min_p,
        argmax=False,
    )
    if cfg.decode_btn_support_min > 0 and not has_button_combo_counts:
        raise ValueError(
            "decode_btn_support_min > 0 requires button_combo_counts_path for a fresh run or embedded "
            "dataset-scoped counts in a resumed checkpoint"
        )


def _load_model_state(model: GPT, state_dict: dict[str, Tensor]) -> None:
    """Load 020 state, tolerating only the count buffer when inspecting an older checkpoint. A 016
    checkpoint has no ``value_head`` and is rejected here as a missing key rather than silently
    loading with a randomly initialized baseline."""
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing - {"button_combo_counts"} or unexpected:
        raise RuntimeError(f"checkpoint/model mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")


@dataclass(frozen=True, slots=True)
class DecodeSettings:
    temp: float
    temps: tuple[float, float, float, float] | None
    btn_support_min: int
    min_p: float
    click_trigger_fix: bool


def _decode_settings(
    model: GPT,
    cfg: TrainConfig,
    *,
    temp: float | None = None,
    temps: tuple[float, float, float, float] | None = None,
    btn_support_min: int | None = None,
    min_p: float | None = None,
    click_trigger_fix: bool | None = None,
) -> DecodeSettings:
    settings = DecodeSettings(
        temp=cfg.decode_temp if temp is None else temp,
        temps=cfg.decode_temps if temps is None else temps,
        btn_support_min=cfg.decode_btn_support_min if btn_support_min is None else btn_support_min,
        min_p=cfg.decode_min_p if min_p is None else min_p,
        click_trigger_fix=cfg.decode_click_trigger_fix if click_trigger_fix is None else click_trigger_fix,
    )
    _resolve_decode_args(settings.temp, settings.temps, settings.btn_support_min, settings.min_p, argmax=False)
    if settings.btn_support_min > 0 and bool((model.button_combo_counts < 0).any()):
        raise ValueError("button-support masking requested, but this checkpoint has no dataset-scoped counts")
    return settings


def make_policy(
    model: GPT,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    device: str = DEVICE,
    exec_horizon: int | None = None,
    decode_temp: float | None = None,
    decode_temps: tuple[float, float, float, float] | None = None,
    decode_btn_support_min: int | None = None,
    decode_min_p: float | None = None,
    decode_click_trigger_fix: bool | None = None,
    decode_seed: int | None = None,
) -> RecedingHorizon:
    """Fresh closed-loop policy for one eval wave. Replan every ``s`` frames (``exec_horizon``, defaulting to
    ``cfg.exec_horizon``): s=1 decodes the offset-1 head per frame (byte-identical to prior 012); s>1 decodes
    the contiguous heads 1..s from one backbone forward and executes all s before replanning. Each decode-
    hygiene knob falls back to its ``cfg`` field when the override is ``None`` so an eval can A/B without a
    retrain."""
    s = cfg.exec_horizon if exec_horizon is None else exec_horizon
    offsets = _exec_horizon_offsets(model.head_offsets, s)
    settings = _decode_settings(
        model,
        cfg,
        temp=decode_temp,
        temps=decode_temps,
        btn_support_min=decode_btn_support_min,
        min_p=decode_min_p,
        click_trigger_fix=decode_click_trigger_fix,
    )
    model_device = next(model.parameters()).device
    gen = None if decode_seed is None else torch.Generator(device=model_device).manual_seed(decode_seed)

    # Per-slot incremental state. Entries are kept separately because instant-restart boundaries
    # occur at different frames; callbacks batch together slots with the same absolute position.
    kv_cache: dict[int, list[tuple[torch.Tensor, torch.Tensor] | None]] = {}
    kv_position: dict[int, int] = {}

    @torch.no_grad()
    def predict_chunk(ctx: Context, committed: np.ndarray | None) -> np.ndarray:
        assert committed is None, "receding-horizon policy does not condition on a committed prefix"
        chunk = (
            decode(
                model,
                ctx,
                temp=settings.temp,
                temps=settings.temps,
                btn_support_min=settings.btn_support_min,
                min_p=settings.min_p,
                click_trigger_fix=settings.click_trigger_fix,
                gen=gen,
            )
            if s == 1
            else decode_chunk(
                model,
                ctx,
                offsets,
                temp=settings.temp,
                temps=settings.temps,
                btn_support_min=settings.btn_support_min,
                min_p=settings.min_p,
                click_trigger_fix=settings.click_trigger_fix,
                gen=gen,
            )
        )
        return chunk.cpu().numpy()

    @torch.no_grad()
    def predict_incremental(ctx: Context, committed: np.ndarray | None) -> np.ndarray:
        assert committed is None, "receding-horizon policy does not condition on a committed prefix"
        if ctx.slot_ids is None or ctx.reset is None:
            raise ValueError("incremental closed-loop decode requires slot_ids and reset metadata")
        n = ctx.slot_ids.numel()
        out = torch.empty((n, s, A_DIM), device=model_device, dtype=torch.float32)
        ids = ctx.slot_ids.detach().cpu().tolist()
        resets = ctx.reset.detach().cpu().tolist()
        groups: dict[int, list[int]] = {}
        for i, (sid, reset) in enumerate(zip(ids, resets, strict=True)):
            sid = int(sid)
            if reset:
                kv_cache.pop(sid, None)
                kv_position.pop(sid, None)
            groups.setdefault(kv_position.get(sid, 0), []).append(i)
        for position, rows in groups.items():
            row_idx = torch.tensor(rows, device=model_device, dtype=torch.long)
            features = {k: v.index_select(0, row_idx) for k, v in ctx.features.items()}
            # All rows in a group have the same position and therefore the same cache length.
            past_batch: list[tuple[torch.Tensor, torch.Tensor] | None] = []
            for layer in range(len(model.blocks)):
                vals = [kv_cache.get(int(ids[r]), [None] * len(model.blocks))[layer] for r in rows]
                if vals[0] is None:
                    past_batch.append(None)
                else:
                    assert all(v is not None for v in vals)
                    past_batch.append(
                        (
                            torch.cat([v[0] for v in vals if v is not None], 0),
                            torch.cat([v[1] for v in vals if v is not None], 0),
                        )
                    )
            h, new = model.forward_incremental(features, past_batch, position)
            for j, r in enumerate(rows):
                sid = int(ids[r])
                kv_cache[sid] = [(k[j : j + 1], v[j : j + 1]) for k, v in new]
                kv_position[sid] = position + 1
            # Incremental mode currently supports the same offset-1 deployment contract as s=1.
            if s != 1:
                raise ValueError("incremental decode currently requires exec_horizon=1")
            logits = model.heads[model.primary_head_idx](h).float()
            out[row_idx, 0] = _sample_action_from_logits(
                model,
                logits,
                group_temps=settings.temps or (settings.temp,) * N_GROUPS,
                btn_support_min=settings.btn_support_min,
                min_p=settings.min_p,
                click_trigger_fix=settings.click_trigger_fix,
                gen=gen,
            )
        return out.cpu().numpy()

    return RecedingHorizon(
        predict_chunk=predict_chunk,
        predict_incremental=predict_incremental if cfg.eval_incremental_kv else None,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=s,
        s=s,
        d=0,
        device=device,
    )


# %%
def lr_schedule(cfg: TrainConfig):
    """Linear warmup → cosine to a small floor. The returned multiplier scales every param group's
    base lr uniformly, so the Muon and AdamW groups share one schedule shape."""
    floor = 0.01

    def fn(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return fn


def make_optimizer(model: GPT, cfg: TrainConfig) -> SingleDeviceMuonWithAuxAdam:
    """Muon for the transformer blocks' hidden weight matrices (attn + MLP); AdamW for everything
    else — input projection, output head, embeddings, biases — split by weight-decay eligibility.
    Exactly two LRs (``cfg.muon_lr`` / ``cfg.adam_lr``); the partition asserts full coverage so no
    parameter can silently escape an optimizer."""
    muon_params = [p for p in model.blocks.parameters() if p.ndim >= 2]
    muon_ids = {id(p) for p in muon_params}
    embed_ids = {id(p) for m in (model.cat_embeds, model.char_emb, model.stage_emb) for p in m.parameters()}
    # Optionally exclude the output-head weights from weight decay (cfg.head_weight_decay=False). The
    # value head is an output head too and follows the same rule, so one knob covers both.
    output_heads = (*model.heads.parameters(), *model.value_head.parameters())
    head_ids = set() if cfg.head_weight_decay else {id(p) for p in output_heads}

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for p in model.parameters():
        if id(p) in muon_ids:
            continue
        # AdamW: no weight decay on embeddings, 1D params (biases), or the head weights when disabled.
        (no_decay if id(p) in embed_ids or p.ndim < 2 or id(p) in head_ids else decay).append(p)

    n_assigned = len(muon_params) + len(decay) + len(no_decay)
    n_total = sum(1 for _ in model.parameters())
    if n_assigned != n_total:
        raise RuntimeError(f"optimizer param partition covers {n_assigned}/{n_total} params")

    adam = dict(betas=(0.9, 0.95), eps=1e-10, use_muon=False)
    param_groups = [
        dict(params=muon_params, lr=cfg.muon_lr, momentum=0.95, weight_decay=cfg.weight_decay, use_muon=True),
        dict(params=decay, lr=cfg.adam_lr, weight_decay=cfg.weight_decay, **adam),
        dict(params=no_decay, lr=cfg.adam_lr, weight_decay=0.0, **adam),
    ]
    return SingleDeviceMuonWithAuxAdam(param_groups)


def nll_breakdown(comps: dict[str, Tensor]) -> dict[str, float]:
    """Per-group NLL (bits) + ``total`` bits/frame, from one head's per-group ``[n_valid]`` nats. Flat
    keys (``buttons``/``main_stick``/``c_stick``/``triggers``/``total``) so callers land in one W&B section."""
    out = {name: (c.mean().item() / _LN2) for name, c in comps.items()}
    out["total"] = sum(c.mean() for c in comps.values()).item() / _LN2
    return out


def _weighted_mean(nll: Tensor, is_trans: Tensor, weight: float, sample_weight: Tensor | None = None) -> Tensor:
    """Mean per-position NLL (nats) under two independent per-position weights: transition positions
    scale by ``weight`` (1.0 = off), and ``sample_weight`` is the AWR weight (None = off). With both off
    this is exactly the plain mean (016); otherwise ``sum(w·nll)/sum(w)`` with ``w`` their product, so the
    reduction stays a weighted AVERAGE and the objective's scale does not move with the weights."""
    if weight == 1.0 and sample_weight is None:
        return nll.mean()
    w = torch.where(is_trans, weight, 1.0).to(nll.dtype)
    if sample_weight is not None:
        w = w * sample_weight.to(nll.dtype)
    return (w * nll).sum() / w.sum()


def _offset_objective(
    nll: dict[tuple[int, str], Tensor],
    trans: dict[tuple[int, str], Tensor],
    offset: int,
    transition_weight: float,
) -> Tensor:
    """One head's unscaled sum-over-groups objective in nats. Deliberately AWR-free: it feeds the
    gradient diagnostics, which ask how the horizons pull on the shared trunk."""
    return torch.stack(
        [_weighted_mean(nll[(offset, name)], trans[(offset, name)], transition_weight) for name in _GROUP_NAMES]
    ).sum()


def objective(
    nll: dict[tuple[int, str], Tensor],
    trans: dict[tuple[int, str], Tensor],
    aux_weight: float,
    transition_weight: float,
    sample_weight: Tensor | None = None,
) -> Tensor:
    """Weighted-sum multi-token training objective (nats): the offset-1 (primary) head's per-group NLL at
    weight 1, every auxiliary head (offset != 1) at ``aux_weight``; within each (offset, group) the per-group
    reduction upweights transition targets by ``transition_weight`` (1.0 = plain mean) and scales each
    position by the AWR weight ``sample_weight`` (None = the 016-base objective). The AWR weight applies to
    EVERY head and group: it says which FRAMES are worth imitating, not which parts of the controller."""
    terms = [
        (1.0 if o == 1 else aux_weight) * _weighted_mean(c, trans[(o, name)], transition_weight, sample_weight)
        for (o, name), c in nll.items()
    ]
    return torch.stack(terms).sum()


def _slice_batch(batch: AWRBatch, n: int) -> AWRBatch:
    """First ``n`` examples of a frozen batch, preserving Context structure."""
    inner = batch.batch
    return AWRBatch(
        batch=TrainBatch(
            context=Context(
                features={name: value[:n] for name, value in inner.context.features.items()},
                ctx_pad=inner.context.ctx_pad[:n],
            ),
            target=inner.target[:n],
        ),
        returns=batch.returns[:n],
    )


def _gradient_dot(a: tuple[Tensor, ...], b: tuple[Tensor, ...]) -> Tensor:
    return torch.stack([torch.dot(x.reshape(-1), y.reshape(-1)) for x, y in zip(a, b, strict=True)]).sum()


def _gradient_cosine(a: tuple[Tensor, ...], b: tuple[Tensor, ...], norm_a: Tensor, norm_b: Tensor) -> Tensor:
    denom = (norm_a * norm_b).clamp_min(torch.finfo(norm_a.dtype).tiny)
    return (_gradient_dot(a, b) / denom).clamp(-1.0, 1.0)


@contextlib.contextmanager
def _evaluation_mode(model: nn.Module) -> Iterator[None]:
    """Temporarily enter eval mode and restore the exact prior mode on every exit."""
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def gradient_diagnostics(model: GPT, batch: AWRBatch, cfg: TrainConfig) -> dict[str, float]:
    """Exact shared-trunk gradient norms/cosines for each horizon on a fixed val subset.

    Output-head parameters are deliberately excluded: the question is whether each
    task asks the representation shared with the deployed head to move in an aligned
    direction. ``autograd.grad`` leaves ``parameter.grad`` untouched, and eval mode
    disables history dropout, so this cannot perturb the optimizer or RNG stream.
    """
    with _evaluation_mode(model):
        return _gradient_diagnostics_eval(model, batch, cfg)


def _gradient_diagnostics_eval(model: GPT, batch: AWRBatch, cfg: TrainConfig) -> dict[str, float]:
    diagnostic_batch = _slice_batch(batch, min(cfg.gradient_diagnostic_batch_size, batch.batch.context.batch))
    parts = action_loss(model, diagnostic_batch.batch)
    losses = {
        offset: _offset_objective(parts.nll, parts.transition, offset, cfg.transition_loss_weight)
        for offset in model.head_offsets
    }
    # Both output heads are excluded: the value head takes no gradient from the action loss at all,
    # and the question is what each horizon asks of the SHARED trunk.
    trunk = tuple(
        parameter for name, parameter in model.named_parameters() if not name.startswith(("heads.", "value_head."))
    )
    gradients: dict[int, tuple[Tensor, ...]] = {}
    for i, offset in enumerate(model.head_offsets):
        gradients[offset] = tuple(
            gradient.detach()
            for gradient in torch.autograd.grad(
                losses[offset],
                trunk,
                retain_graph=i + 1 < len(model.head_offsets),
            )
        )

    norms = {offset: _gradient_dot(gradient, gradient).sqrt() for offset, gradient in gradients.items()}
    out = {f"grad/head_{offset}_norm": norm.item() for offset, norm in norms.items()}
    for i, left in enumerate(model.head_offsets):
        for right in model.head_offsets[i + 1 :]:
            out[f"grad/cos_{left}_{right}"] = _gradient_cosine(
                gradients[left], gradients[right], norms[left], norms[right]
            ).item()

    aux_offsets = tuple(offset for offset in model.head_offsets if offset != 1)
    if aux_offsets:
        # This is the direction the CURRENT objective applies, not a mean-head convention:
        # aux_loss_weight * sum_o g_o. Its norm ratio makes objective-scale domination visible.
        weighted_aux = tuple(
            cfg.aux_loss_weight * sum((gradients[offset][pi] for offset in aux_offsets), start=torch.zeros_like(p))
            for pi, p in enumerate(trunk)
        )
        aux_norm = _gradient_dot(weighted_aux, weighted_aux).sqrt()
        primary = gradients[1]
        out["grad/weighted_aux_norm"] = aux_norm.item()
        out["grad/weighted_aux_to_primary_norm"] = (aux_norm / norms[1].clamp_min(norms[1].new_tensor(1e-30))).item()
        out["grad/cos_1_weighted_aux"] = _gradient_cosine(primary, weighted_aux, norms[1], aux_norm).item()
        conflicting = sum(
            int(((p != 0) & (a != 0) & ((p > 0) != (a > 0))).sum()) for p, a in zip(primary, weighted_aux)
        )
        comparable = sum(int(((p != 0) & (a != 0)).sum()) for p, a in zip(primary, weighted_aux))
        out["grad/sign_conflict_frac_1_weighted_aux"] = conflicting / comparable if comparable else 0.0
    return out


def _offset_total_bits(comps: dict[tuple[int, str], Tensor], o: int) -> float:
    """Total bits/frame summed over the four groups for one head offset ``o``."""
    return sum(comps[(o, name)].mean() for name in _GROUP_NAMES).item() / _LN2


def _masked_mean_bits(nats: Tensor, mask: Tensor) -> float:
    """Mean of per-position NLL (nats) over the masked subset, in bits; 0.0 when the subset is empty."""
    return (nats[mask].mean().item() / _LN2) if bool(mask.any()) else 0.0


def _bool_mean(parts: list[Tensor], *, invert: bool = False) -> float:
    values = torch.cat(parts)
    if values.numel() == 0:
        return 0.0
    return ((~values) if invert else values).float().mean().item()


def _group_kl_bits(logits_p: Tensor, logits_q: Tensor) -> Tensor:
    """Summed-over-groups KL(p‖q) in bits per position: for each group's logit slice, p=softmax(p_slice),
    q=softmax(q_slice); ``[..., A_VOCAB]`` logits → ``[...]`` bits summed over the four group softmaxes."""
    total = torch.zeros(logits_p.shape[:-1], device=logits_p.device)
    for g in range(N_GROUPS):
        lo = _GROUP_OFFSETS[g]
        sl = slice(lo, lo + _GROUP_VOCABS[g])
        logp = F.log_softmax(logits_p[..., sl], dim=-1)
        logq = F.log_softmax(logits_q[..., sl], dim=-1)
        total = total + (logp.exp() * (logp - logq)).sum(-1)
    return total / _LN2


@torch.no_grad()
def val_metrics(model: GPT, val_cache: list[TrainBatch], cfg: TrainConfig) -> dict[str, float]:
    """Dense multi-token proper-scoring metrics over the cached val batches. Per-offset NLL (``nll_off{o}``)
    tracks how predictability decays with horizon. The offset-1 (deployed) head additionally drives button
    proper-scoring, per-group categorical Brier (a proper score on the shared discretizer grids — the
    objective-independent yardstick that stays comparable to model families without an exact likelihood),
    transition-vs-hold NLL splits, ±1-tolerant change-event F1, and a copycat history-ablation probe. Every
    number here is UNWEIGHTED — AWR touches the backprop objective only — so all of it compares directly
    with 016. Per-element tensors are concatenated then reduced once (exactly sample-weighted)."""
    with _evaluation_mode(model):
        return _val_metrics_eval(model, val_cache, cfg)


def _val_metrics_eval(model: GPT, val_cache: list[TrainBatch], cfg: TrainConfig) -> dict[str, float]:
    comps_cat: dict[tuple[int, str], list[Tensor]] = {}
    ablated_cat: dict[str, list[Tensor]] = {}
    brier_cat: dict[str, list[Tensor]] = {}
    trans_cat: dict[str, list[Tensor]] = {}
    pred_change_cat: dict[str, list[Tensor]] = {}
    pred_next_change_cat: dict[str, list[Tensor]] = {}
    pred_temporal_change_cat: dict[str, list[Tensor]] = {}
    pred_flipback_cat: dict[str, list[Tensor]] = {}
    true_change_cat: dict[str, list[Tensor]] = {}
    kl_bits: list[Tensor] = []
    btn_probs: list[Tensor] = []
    btn_tgts: list[Tensor] = []
    multipress: list[Tensor] = []
    rare_mass: list[Tensor] = []
    unseen_mass: list[Tensor] = []
    click_trigger_invalid_l: list[Tensor] = []
    click_trigger_invalid_r: list[Tensor] = []
    counts_available = bool((model.button_combo_counts >= 0).all())
    rare_mask = model.button_combo_counts < cfg.diagnostic_rare_button_count
    unseen_mask = model.button_combo_counts == 0
    combo_bits = scoring.combo_to_buttons(torch.arange(scoring.N_BUTTON_COMBOS, device=model.main_centers.device))
    for batch in val_cache:
        ctx = batch.context
        h = model(ctx.features, ctx.ctx_pad)
        # Copycat probe: a second backbone forward with the ego's own controller history zeroed.
        ablated_features = dict(ctx.features)
        for ch in ACTION_CHANNELS:
            ablated_features[f"ego_{ch}"] = torch.zeros_like(ablated_features[f"ego_{ch}"])
        h_ablated = model(ablated_features, ctx.ctx_pad)
        targets, valid = _multi_offset_targets(ctx, batch.target[:, : max(model.head_offsets)], model.head_offsets)
        flat_valid = valid.reshape(-1)
        adjacent_valid = valid[:, 1:] & valid[:, :-1]
        triple_valid = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
        cur_idx = _quantize(model, stack_actions(ctx.features))  # [B, L_ctx, n_groups] current frames
        for hi, o in enumerate(model.head_offsets):
            logits = model.heads[hi](h).float()
            tgt_idx = _quantize(model, targets[o])
            for name, c in group_nll(logits, tgt_idx, valid).items():
                comps_cat.setdefault((o, name), []).append(c)
            if o == 1:  # deployed head drives button / transition / ablation stats
                true_change = scoring.transition_mask(torch.cat([cur_idx, tgt_idx[:, -1:]], dim=1))  # [B,L_ctx,n_grp]
                ablated_logits = model.heads[model.primary_head_idx](h_ablated).float()
                kl_bits.append(_group_kl_bits(logits, ablated_logits).reshape(-1)[flat_valid])
                for name, c in group_nll(ablated_logits, tgt_idx, valid).items():
                    ablated_cat.setdefault(name, []).append(c)
                for g, name in enumerate(_GROUP_NAMES):
                    lo = _GROUP_OFFSETS[g]
                    pred_id = logits[..., lo : lo + _GROUP_VOCABS[g]].argmax(-1)  # [B, L_ctx] argmax next-frame id
                    probs_g = F.softmax(
                        logits[..., lo : lo + _GROUP_VOCABS[g]].reshape(-1, _GROUP_VOCABS[g])[flat_valid], dim=-1
                    )
                    onehot_g = F.one_hot(tgt_idx[..., g].reshape(-1)[flat_valid], _GROUP_VOCABS[g]).to(probs_g.dtype)
                    brier_cat.setdefault(name, []).append((probs_g - onehot_g).pow(2).sum(-1))
                    tc = true_change[..., g]
                    trans_cat.setdefault(name, []).append(tc.reshape(-1)[flat_valid])
                    pred_change = (pred_id != cur_idx[..., g]) & valid
                    pred_change_cat.setdefault(name, []).append(pred_change)
                    pred_next_change_cat.setdefault(name, []).append(pred_change.reshape(-1)[flat_valid])
                    pred_temporal_change_cat.setdefault(name, []).append(
                        (pred_id[:, 1:] != pred_id[:, :-1])[adjacent_valid]
                    )
                    pred_flipback_cat.setdefault(name, []).append(
                        ((pred_id[:, 2:] == pred_id[:, :-2]) & (pred_id[:, 1:-1] != pred_id[:, :-2]))[triple_valid]
                    )
                    true_change_cat.setdefault(name, []).append(tc & valid)
                btn_logits = logits[..., : scoring.N_BUTTON_COMBOS].reshape(-1, scoring.N_BUTTON_COMBOS)[flat_valid]
                combo_probs = F.softmax(btn_logits, dim=-1)
                marginal_btn_probs = combo_probs @ combo_bits.to(combo_probs.dtype)
                btn_probs.append(marginal_btn_probs)
                if counts_available:
                    rare_mass.append(combo_probs[:, rare_mask].sum(-1))
                    unseen_mass.append(combo_probs[:, unseen_mask].sum(-1))
                trig_lo = _GROUP_OFFSETS[_TRIG_G]
                n_trig = model.trig_centers.shape[0]
                trig_probs = F.softmax(
                    logits[..., trig_lo : trig_lo + _GROUP_VOCABS[_TRIG_G]].reshape(-1, _GROUP_VOCABS[_TRIG_G])[
                        flat_valid
                    ],
                    dim=-1,
                ).reshape(-1, n_trig, n_trig)
                trigger_l_full = trig_probs[:, -1, :].sum(-1)
                trigger_r_full = trig_probs[:, :, -1].sum(-1)
                click_trigger_invalid_l.append(marginal_btn_probs[:, _BUTTON_L_CH - _N_CONT] * (1.0 - trigger_l_full))
                click_trigger_invalid_r.append(marginal_btn_probs[:, _BUTTON_R_CH - _N_CONT] * (1.0 - trigger_r_full))
                tgt_btn = _dequantize(model, tgt_idx)[..., _N_CONT:].reshape(-1, _N_BUTTONS)[flat_valid]
                btn_tgts.append(tgt_btn)
                multipress.append((tgt_btn > 0.5).sum(-1) >= 2)
    comps = {k: torch.cat(v) for k, v in comps_cat.items()}
    ablated = {k: torch.cat(v) for k, v in ablated_cat.items()}
    primary = nll_breakdown({name: comps[(1, name)] for name in _GROUP_NAMES})
    logloss, brier = scoring.bernoulli_scores_from_probs(torch.cat(btn_probs), torch.cat(btn_tgts))
    out = {
        "loss": primary["total"],  # offset-1 total bits/frame (deployed policy); per-group below
        **{f"nll_{name}": primary[name] for name in _GROUP_NAMES},
        "cont_discrete_bits": (
            comps[(1, "main_stick")].mean() + comps[(1, "c_stick")].mean() + comps[(1, "triggers")].mean()
        ).item()
        / _LN2,
        "btn_logloss": logloss.item(),
        "btn_brier": brier.item(),
        "btn_multipress": torch.cat(multipress).float().mean().item(),
        "btn_counts_available": float(counts_available),
        "click_trigger_invalid_l_mass": torch.cat(click_trigger_invalid_l).mean().item(),
        "click_trigger_invalid_r_mass": torch.cat(click_trigger_invalid_r).mean().item(),
        "ablate_hist_kl": torch.cat(kl_bits).mean().item(),  # KL(full ‖ history-ablated), bits
        **{f"nll_off{o}": _offset_total_bits(comps, o) for o in model.head_offsets},
    }
    out["click_trigger_invalid_mass"] = 0.5 * (
        out["click_trigger_invalid_l_mass"] + out["click_trigger_invalid_r_mass"]
    )
    if counts_available:
        out["btn_rare_mass"] = torch.cat(rare_mass).mean().item()
        out["btn_unseen_mass"] = torch.cat(unseen_mass).mean().item()
        out["btn_rare_count_threshold"] = float(cfg.diagnostic_rare_button_count)
    ablate_total = 0.0
    for name in _GROUP_NAMES:
        trans = torch.cat(trans_cat[name])
        out[f"nll_{name}_trans"] = _masked_mean_bits(comps[(1, name)], trans)
        out[f"nll_{name}_hold"] = _masked_mean_bits(comps[(1, name)], ~trans)
        out[f"trans_rate_{name}"] = trans.float().mean().item()
        out[f"pred_change_rate_{name}"] = _bool_mean(pred_next_change_cat[name])
        out[f"pred_persistence_{name}"] = _bool_mean(pred_temporal_change_cat[name], invert=True)
        out[f"pred_flipback_rate_{name}"] = _bool_mean(pred_flipback_cat[name])
        out[f"changeF1_{name}"] = scoring.change_event_prf(
            torch.cat(pred_change_cat[name]), torch.cat(true_change_cat[name])
        )[2]
        out[f"brier_{name}"] = torch.cat(brier_cat[name]).mean().item()
        d = (ablated[name].mean() - comps[(1, name)].mean()).item() / _LN2  # positive ⇒ history helps
        out[f"ablate_hist_dnll_{name}"] = d
        ablate_total += d
    out["ablate_hist_dnll"] = ablate_total
    return out


@torch.no_grad()
def awr_val_metrics(model: GPT, val_cache: list[AWRBatch], cfg: TrainConfig) -> tuple[dict[str, float], Tensor]:
    """The AWR-side diagnostics on the FROZEN val batches: how well the value head fits the return, and
    what the weights it implies would look like. Returns the flat metric dict plus the pooled per-position
    weights, which the caller logs as a histogram — a collapsed weight distribution (a spike at the clip,
    everything else at zero) is the failure mode ``awr_beta`` guards against, and a mean/std alone hides it.
    The weights here are DIAGNOSTIC: val NLL itself stays unweighted."""
    with _evaluation_mode(model):
        values: list[Tensor] = []
        returns: list[Tensor] = []
        for batch in val_cache:
            ctx = batch.batch.context
            h = model(ctx.features, ctx.ctx_pad)
            valid = torch.arange(h.shape[1], device=h.device)[None, :] >= ctx.ctx_pad[:, None]
            values.append(model.value_head(h).float().reshape(-1)[valid.reshape(-1)])
            returns.append(batch.valid_returns(valid))
        value = torch.cat(values)
        target = torch.cat(returns)
        weight, stats = awr_weights(target - value, beta=cfg.awr_beta, weight_max=cfg.awr_weight_max)
        out = {
            "value_mse": F.mse_loss(value, target).item(),
            "value_mean": value.mean().item(),
            "return_mean": target.mean().item(),
            "return_std": target.std().item(),
            "awr_ess": stats["ess"],
            "awr_weight_max_frac": stats["weight_max_frac"],
        }
    return out, weight


@torch.no_grad()
def recon_metrics(
    model: GPT,
    val_cache: list[TrainBatch],
    *,
    argmax: bool,
    temp: float = 1.0,
    temps: tuple[float, float, float, float] | None = None,
    btn_support_min: int = 0,
    min_p: float = 0.0,
    click_trigger_fix: bool = False,
    gen: torch.Generator | None = None,
) -> dict[str, float]:
    """Sample-space reconstruction proxy: decode the next action and score it vs ground truth.
    Buttons → acc + F1 @ decode; continuous → MAE. ``argmax`` is the deterministic controller proxy."""
    with _evaluation_mode(model):
        return _recon_metrics_eval(
            model,
            val_cache,
            argmax=argmax,
            temp=temp,
            temps=temps,
            btn_support_min=btn_support_min,
            min_p=min_p,
            click_trigger_fix=click_trigger_fix,
            gen=gen,
        )


def _recon_metrics_eval(
    model: GPT,
    val_cache: list[TrainBatch],
    *,
    argmax: bool,
    temp: float,
    temps: tuple[float, float, float, float] | None,
    btn_support_min: int,
    min_p: float,
    click_trigger_fix: bool,
    gen: torch.Generator | None,
) -> dict[str, float]:
    tp = fp = fn = btn_correct = btn_total = 0
    cont_abs_err = 0.0
    cont_count = 0
    for batch in val_cache:
        pred = decode(
            model,
            batch.context,
            temp=temp,
            temps=temps,
            btn_support_min=btn_support_min,
            min_p=min_p,
            click_trigger_fix=click_trigger_fix,
            argmax=argmax,
            gen=gen,
        )
        # decode yields one frame ([B, 1, A_DIM]); score it against the next-frame target only. Comparing
        # against the full VAL_L_CHUNK-wide target would silently broadcast [B,1,·] vs [B,L_chunk,·].
        tgt = batch.target[:, :1]
        pb = pred[..., _N_CONT:] > 0.5
        tb = tgt[..., _N_CONT:] > 0.5
        tp += int((pb & tb).sum())
        fp += int((pb & ~tb).sum())
        fn += int((~pb & tb).sum())
        btn_correct += int((pb == tb).sum())
        btn_total += pb.numel()
        cont_abs_err += float((pred[..., :_N_CONT] - tgt[..., :_N_CONT]).abs().sum())
        cont_count += tgt[..., :_N_CONT].numel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "recon_button_acc": btn_correct / btn_total,
        "recon_button_f1": f1,
        "recon_cont_mae": cont_abs_err / cont_count,
    }


@dataclass(frozen=True, slots=True)
class EvalProtocol:
    n_matchups: int
    max_parallel: int
    max_frames: int
    seed: int
    exec_horizon: int
    decode_temp: float
    decode_temps: tuple[float, float, float, float] | None
    decode_btn_support_min: int
    decode_min_p: float
    decode_click_trigger_fix: bool


def _eval_protocol(
    cfg: TrainConfig,
    *,
    settings: DecodeSettings,
    exec_horizon: int,
    default_n_matchups: int,
    n_matchups: int | None = None,
    max_frames: int | None = None,
    seed: int | None = None,
) -> EvalProtocol:
    n = default_n_matchups if n_matchups is None else n_matchups
    frames = cfg.eval_max_frames if max_frames is None else max_frames
    resolved_seed = cfg.eval_seed if seed is None else seed
    if n <= 0:
        raise ValueError(f"n_matchups must be > 0, got {n}")
    if frames <= 0:
        raise ValueError(f"max_frames must be > 0, got {frames}")
    return EvalProtocol(
        n_matchups=n,
        max_parallel=_eval_max_parallel(cfg, n),
        max_frames=frames,
        seed=resolved_seed,
        exec_horizon=exec_horizon,
        decode_temp=settings.temp,
        decode_temps=settings.temps,
        decode_btn_support_min=settings.btn_support_min,
        decode_min_p=settings.min_p,
        decode_click_trigger_fix=settings.click_trigger_fix,
    )


def _write_match_rows(path: Path, rows: list[MatchRow], protocol: EvalProtocol) -> None:
    """Atomically persist exact trajectory-derived rows plus pairing protocol."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "protocol": asdict(protocol),
        "rows": [row.as_dict() for row in rows],
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True))
    tmp.replace(path)


def _run_eval_sweep(
    policy_factory: Callable[[], RecedingHorizon],
    *,
    protocol: EvalProtocol,
    replay_dir: Path | None,
    rows_path: Path | None,
) -> dict[str, float]:
    results, rows = sweep_vs_cpu_prior_with_rows(
        policy_factory,
        session_cfg=default_session_cfg(replay_dir, instant_match_restart=True),
        n_matchups=protocol.n_matchups,
        max_parallel=protocol.max_parallel,
        max_frames=protocol.max_frames,
    )
    if rows_path is not None:
        _write_match_rows(rows_path, rows, protocol)
    return vs_cpu_metrics(results, seed=protocol.seed)


def eval_vs_cpu(
    model: GPT,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    max_frames: int,
    replay_dir: Path | None = None,
    n_matchups: int | None = None,
    eval_seed: int | None = None,
    rows_path: Path | None = None,
) -> dict[str, float]:
    """In-training closed-loop eval vs lvl-9 CPU over prior-sampled char matchups.

    A fixed ``n_matchups`` controls statistical coverage; host CPU count controls only
    how many of those boots execute concurrently. Each policy wave gets an explicit,
    deterministic sampling seed. Reduced to a flat metric dict."""
    settings = _decode_settings(model, cfg)
    protocol = _eval_protocol(
        cfg,
        settings=settings,
        exec_horizon=cfg.exec_horizon,
        default_n_matchups=cfg.eval_n_matchups,
        n_matchups=n_matchups,
        max_frames=max_frames,
        seed=eval_seed,
    )
    policy_index = itertools.count()

    def policy_factory() -> RecedingHorizon:
        return make_policy(
            model,
            stats,
            cfg,
            exec_horizon=protocol.exec_horizon,
            decode_temp=settings.temp,
            decode_temps=settings.temps,
            decode_btn_support_min=settings.btn_support_min,
            decode_min_p=settings.min_p,
            decode_click_trigger_fix=settings.click_trigger_fix,
            decode_seed=protocol.seed + next(policy_index),
        )

    with _evaluation_mode(model):
        return _run_eval_sweep(
            policy_factory,
            protocol=protocol,
            replay_dir=replay_dir,
            rows_path=rows_path,
        )


# %%
def _loader_kwargs(cfg: TrainConfig, stats: dict[str, FeatureStats]) -> dict:
    """Loader arguments shared by the train and val splits. ``schema_version`` declares which MDS
    materialization this run reads, so a stale (or newer) dataset raises on the first row."""
    return dict(
        data_root=cfg.data_root,
        remote=streams.remote_for_local(cfg.data_root),
        cache_limit=f"{cfg.cache_limit_gb}gb",
        shuffle_block_size=cfg.shuffle_block_size,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=max(cfg.head_offsets),  # target horizon must cover the farthest auxiliary head
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        schema_version=cfg.mds_schema_version,
    )


def _fetch_h2h_reference(cfg: TrainConfig, run_dir: Path) -> Path:
    """Download the reference final.pt at train START, so a bad reference fails in
    minute one, not after the full training run."""
    assert cfg.final_h2h_reference_run is not None
    ref_ckpt = download_latest(cfg.final_h2h_reference_run, run_dir / "h2h_reference", name="final.pt")
    if ref_ckpt is None:
        raise RuntimeError(f"no final.pt on R2 for reference run {cfg.final_h2h_reference_run!r}")
    return ref_ckpt


def _final_h2h(
    cfg: TrainConfig,
    model: GPT,
    stats: dict[str, FeatureStats],
    run_dir: Path,
    uploader: BackgroundUploader,
    ref_ckpt: Path,
) -> None:
    """Mirrored h2h vs the reference run, in-process after training.

    Records and replays land in ``run_dir/h2h_final``. Evidence uploads after EACH
    orientation and again in the ``finally`` block, so a kill mid-sweep costs at most
    one orientation and a tripwire raise costs nothing."""
    from hal.scripts.h2h import ModelArgs
    from hal.scripts.h2h import load_policy_builder

    build_ref, ref_protocol = load_policy_builder(
        ModelArgs(
            name=cfg.final_h2h_reference_label,
            checkpoint=str(ref_ckpt),
            experiment=cfg.final_h2h_reference_experiment,
        )
    )

    def build_self(seed: int) -> RecedingHorizon:
        return make_policy(model, stats, cfg, decode_seed=seed)

    out_dir = run_dir / "h2h_final"
    try:
        with _evaluation_mode(model):
            records = run_h2h(
                build_self,
                build_ref,
                name_a=cfg.final_h2h_self_label,
                name_b=cfg.final_h2h_reference_label,
                n_configs=cfg.final_h2h_n_configs,
                out_dir=out_dir,
                max_frames=cfg.eval_max_frames,
                max_parallel=_eval_max_parallel(cfg, cfg.final_h2h_n_configs),
                seed=cfg.eval_seed,
                meta={"reference": ref_protocol},
                on_orientation_done=lambda _o: uploader.upload_tree(out_dir, base=run_dir),
            )
    finally:
        uploader.upload_tree(out_dir, base=run_dir)
    summary = summarize_paired(records, focal_model=cfg.final_h2h_self_label)
    if wandb.run is not None:
        wandb.run.summary["h2h_final"] = summary.as_dict()
    print(summary.format_table(), flush=True)


def train(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    *,
    comment: str = "",
    resume_run: str | None = None,
    resume_state: dict | None = None,
) -> None:
    resumed_counts = resume_state is not None and "button_combo_counts" in resume_state["model"]
    button_combo_counts = None if resumed_counts else _load_button_combo_counts(cfg)
    validate_config(cfg, has_button_combo_counts=button_combo_counts is not None or resumed_counts)
    run_name = resume_run or make_run_name(Path(__file__).stem, _model_tag(cfg), cfg.data_root, comment)
    uploader = BackgroundUploader(run_name)
    wandb.init(
        project="hal",
        name=run_name,
        id=resume_state["wandb_id"] if resume_state else None,
        resume="allow" if resume_state else None,
        tags=["gpt", f"d{cfg.d_model}", f"L{cfg.n_layers}"],
        config=asdict(cfg),
    )
    # W&B's own step is a free-running monotonic timestamp; we plot everything against the training
    # step logged as data (``global_step``). This lets an async eval that *finishes* late be logged
    # at its *origin* step without violating step monotonicity.
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")
    if wandb.run is not None:
        wandb.run.summary["nll_semantics"] = (
            "AWR weights the BACKPROP objective only; train/loss and every val metric are unweighted "
            "and compare directly with 016"
        )
    ckpt_dir, replay_dir = setup_run_dir(run_name)
    # Fetch the h2h reference NOW: a bad reference must fail in minute one, not at hour three.
    h2h_ref_ckpt = _fetch_h2h_reference(cfg, ckpt_dir) if cfg.final_h2h_reference_run else None

    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    autocast = (
        torch.autocast(DEVICE, dtype=torch.bfloat16)
        if cfg.amp_dtype == "bfloat16" and DEVICE == "cuda"
        else contextlib.nullcontext()
    )
    start_step = resume_state["step"] + 1 if resume_state else 0
    model = GPT(cfg).to(DEVICE)
    if button_combo_counts is not None:
        model.button_combo_counts.copy_(button_combo_counts.to(DEVICE))
    n_params = sum(p.numel() for p in model.parameters())
    if wandb.run is not None:
        wandb.run.summary["model/num_params"] = n_params
    print(f"[model] {_model_tag(cfg)}  num_params={n_params / 1e6:.2f}M", flush=True)
    loader_kwargs = _loader_kwargs(cfg, stats)
    # Both splits carry the return label (val needs it for the value-head metrics), so both go through
    # _attach_returns and both yield AWRBatch.
    train_loader = _attach_returns(
        make_loader(
            split="train",
            num_workers=cfg.num_workers,
            prefetch_factor=cfg.prefetch_factor,
            windows_per_replay=cfg.windows_per_replay,
            **loader_kwargs,
        ),
        cfg,
        stats,
    )
    # Val uses the FROZEN wider chunk (VAL_L_CHUNK) so its window geometry — hence its NLL — is
    # comparable across experiments regardless of the train-time L_chunk. This makes val loss NOT
    # comparable to pre-freeze 012 runs; that break is the intended freeze. The val path slices the
    # wider target back to max(head_offsets) frames.
    val_loader = _attach_returns(
        make_loader(split=cfg.val_split, num_workers=0, **{**loader_kwargs, "L_chunk": VAL_L_CHUNK}), cfg, stats
    )

    opt = make_optimizer(model, cfg)
    sched = LambdaLR(opt, lr_schedule(cfg))
    if resume_state is not None:
        _load_model_state(model, resume_state["model"])
        opt.load_state_dict(resume_state["opt"])
        sched.load_state_dict(resume_state["sched"])
        print(f"[resume] {run_name}: continuing from step {start_step}", flush=True)

    print("[val] building cached val set…", flush=True)
    val_t0 = time.monotonic()
    val_cache = [b.to(DEVICE) for b in itertools.islice(val_loader, cfg.val_n_batches)]
    if not val_cache:
        raise RuntimeError("val loader yielded zero batches")
    print(
        f"[val] cached {len(val_cache)} batches "
        f"({sum(b.batch.target.shape[0] for b in val_cache)} samples) in {time.monotonic() - val_t0:.1f}s",
        flush=True,
    )

    def _wandb_id() -> str | None:
        return wandb.run.id if wandb.run is not None else None

    def _eval_and_upload(step_tag: str, *, n_matchups: int) -> dict[str, float]:
        """Synchronous closed-loop eval on the live model + .slp upload (the final eval).
        Returns the flat metric dict."""
        sub = replay_dir / step_tag
        rows_path = sub / "match_rows.json"
        metrics = eval_vs_cpu(
            model,
            stats,
            cfg,
            max_frames=cfg.eval_max_frames,
            replay_dir=sub,
            n_matchups=n_matchups,
            rows_path=rows_path,
        )
        n = uploader.upload_tree(sub, base=ckpt_dir, pattern="*.slp")
        uploader.upload(rows_path, key=str(rows_path.relative_to(ckpt_dir)))
        print(f"[eval] queued {n} .slp + matchup rows for R2 ({step_tag})", flush=True)
        return metrics

    def _val_log_dict() -> dict[str, object]:
        """Flat ``val/*`` metric dict (one W&B section). Merged into the per-step log; no wandb.log here.
        Every NLL/recon number is computed on the plain ``TrainBatch`` inside each ``AWRBatch``, so the
        whole surface stays 016's; the AWR block adds value-head fit plus the weight histogram."""
        batches = [b.batch for b in val_cache]
        vm = val_metrics(model, batches, cfg)
        gen = torch.Generator(device=DEVICE).manual_seed(cfg.eval_seed)
        settings = _decode_settings(model, cfg)
        decode_kwargs = {
            "temp": settings.temp,
            "temps": settings.temps,
            "btn_support_min": settings.btn_support_min,
            "min_p": settings.min_p,
            "click_trigger_fix": settings.click_trigger_fix,
        }
        recon = {"argmax": recon_metrics(model, batches, argmax=True, **decode_kwargs)}
        recon["sample"] = recon_metrics(model, batches, argmax=False, gen=gen, **decode_kwargs)
        out: dict[str, object] = {f"val/{k}": v for k, v in vm.items()}
        for tag, rm in recon.items():
            out[f"val/recon_{tag}_acc"] = rm["recon_button_acc"]
            out[f"val/recon_{tag}_f1"] = rm["recon_button_f1"]
            out[f"val/recon_{tag}_mae"] = rm["recon_cont_mae"]
        awr_vm, weights = awr_val_metrics(model, val_cache, cfg)
        out.update({f"val/{k}": v for k, v in awr_vm.items()})
        out["val/awr_weights"] = wandb.Histogram(weights.detach().cpu().numpy())
        out.update(gradient_diagnostics(model, val_cache[0], cfg))
        return out

    def _log_eval(step: int, metrics: dict[str, float]) -> None:
        """Sole eval-logging site: plot ``eval/*`` at the eval's origin ``global_step``."""
        wandb.log({**{f"eval/{k}": v for k, v in metrics.items()}, "global_step": step})
        print(f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: closed_loop {metrics}", flush=True)

    def _save(name: str, step: int) -> None:
        save_checkpoint(
            ckpt_dir / name,
            step=step,
            model=model,
            opt=opt,
            sched=sched,
            cfg=asdict(cfg),
            wandb_id=_wandb_id(),
            uploader=uploader,
        )

    # At most one async eval in flight. The worker is a separate process (own GPU/CUDA + GIL) that
    # evals the just-saved checkpoint and writes a metrics JSON; the trainer drains it between steps.
    pending_eval: dict | None = None

    def _drain_eval(*, wait: bool) -> None:
        """Reap the in-flight eval. ``wait`` blocks (bounded by ``eval_timeout_seconds``) for the
        result; otherwise just polls. A worker over budget is killed. On success: log + upload .slp."""
        nonlocal pending_eval
        if pending_eval is None:
            return
        proc: subprocess.Popen = pending_eval["proc"]
        if wait:
            try:
                proc.wait(timeout=max(0.0, cfg.eval_timeout_seconds - (time.monotonic() - pending_eval["t0"])))
            except subprocess.TimeoutExpired:
                pass
        rc = proc.poll()
        if rc is None:
            if not wait and (time.monotonic() - pending_eval["t0"]) <= cfg.eval_timeout_seconds:
                return  # still running, within budget — re-check next iteration
            proc.kill()
            proc.wait()
            print(
                f"[eval] step {pending_eval['step']} timed out (>{cfg.eval_timeout_seconds:.0f}s); "
                f"killed. see {pending_eval['log']}",
                flush=True,
            )
        else:
            step, result = pending_eval["step"], pending_eval["result"]
            if rc == 0 and result.is_file():
                data = json.loads(result.read_text())
                _log_eval(data["step"], data["metrics"])
                n = uploader.upload_tree(pending_eval["replay"], base=ckpt_dir, pattern="*.slp")
                rows_path = pending_eval["replay"] / "match_rows.json"
                if rows_path.is_file():
                    uploader.upload(rows_path, key=str(rows_path.relative_to(ckpt_dir)))
                print(f"[eval] queued {n} .slp + matchup rows for R2 (step {step})", flush=True)
            else:
                print(f"[eval] worker for step {step} failed (rc={rc}); see {pending_eval['log']}", flush=True)
        pending_eval["log_f"].close()
        pending_eval = None

    def _launch_eval(step: int) -> None:
        """Save the checkpoint and spawn a background eval worker for it. Waits out any prior eval
        first (bounded), so only one runs at a time."""
        nonlocal pending_eval
        _drain_eval(wait=True)
        _save(f"step_{step:06d}.pt", step)
        result = ckpt_dir / "eval_results" / f"step_{step:06d}.json"
        log = ckpt_dir / "eval_logs" / f"step_{step:06d}.log"
        result.parent.mkdir(parents=True, exist_ok=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log, "w")  # noqa: SIM115 — spans the worker's lifetime; closed in _drain_eval
        proc = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--eval-worker",
                str(ckpt_dir / f"step_{step:06d}.pt"),
                "--eval-worker-step",
                str(step),
                "--eval-worker-result",
                str(result),
                "--eval-worker-replay",
                str(replay_dir / f"step_{step:06d}"),
            ],
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        pending_eval = {
            "step": step,
            "proc": proc,
            "result": result,
            "replay": replay_dir / f"step_{step:06d}",
            "log": log,
            "log_f": log_f,
            "t0": time.monotonic(),
        }
        print(f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: launched async eval (pid {proc.pid})", flush=True)
        if not cfg.eval_overlap_training:
            # The worker and trainer otherwise inherit the same CUDA device. Waiting here makes
            # the subprocess asynchronous with respect to process setup/logging, but exclusive
            # with respect to GPU execution; this avoids the measured ~6 FPS/boot contention mode.
            _drain_eval(wait=True)

    model.train()
    it = iter(train_loader)
    run_t0 = time.monotonic()
    for step in range(start_step, cfg.max_steps):
        with profile("step") as sw:
            opt.zero_grad()
            comps_acc: dict[tuple[int, str], list[Tensor]] = {}
            obj_acc: Tensor | None = None
            awr_acc: dict[str, float] = {"value_loss": 0.0, "ess": 0.0, "weight_max_frac": 0.0}
            for _ in range(cfg.grad_accum_steps):
                try:
                    batch = next(it).to(DEVICE)
                except StopIteration:
                    it = iter(train_loader)
                    batch = next(it).to(DEVICE)
                with autocast:
                    parts = action_loss(model, batch.batch)
                    # AWR: score the demonstrated frames by their outcome. The advantage is detached, so
                    # the policy loss trains the policy and the MSE trains V — never the reverse.
                    weight: Tensor | None = None
                    value_loss = torch.zeros((), device=parts.value.device)
                    if cfg.awr_enabled:
                        returns = batch.valid_returns(parts.valid)
                        value_loss = F.mse_loss(parts.value, returns)
                        weight, awr_stats = awr_weights(
                            returns - parts.value.detach(), beta=cfg.awr_beta, weight_max=cfg.awr_weight_max
                        )
                    obj = objective(
                        parts.nll, parts.transition, cfg.aux_loss_weight, cfg.transition_loss_weight, weight
                    )
                    total = obj + cfg.awr_value_loss_weight * value_loss if cfg.awr_enabled else obj
                    loss = total / cfg.grad_accum_steps
                loss.backward()
                obj_acc = obj.detach() if obj_acc is None else obj_acc + obj.detach()
                if cfg.awr_enabled:
                    awr_acc["value_loss"] += value_loss.item() / cfg.grad_accum_steps
                    awr_acc["ess"] += awr_stats["ess"] / cfg.grad_accum_steps
                    awr_acc["weight_max_frac"] += awr_stats["weight_max_frac"] / cfg.grad_accum_steps
                for k, v in parts.nll.items():
                    comps_acc.setdefault(k, []).append(v.detach())
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))  # measure only
            opt.step()
            sched.step()
            if DEVICE == "cuda":
                torch.cuda.synchronize()
        assert obj_acc is not None  # grad_accum_steps >= 1
        objective_bits = (obj_acc / cfg.grad_accum_steps).item() / _LN2  # the actual backprop objective, bits
        comps_cat = {k: torch.cat(v) for k, v in comps_acc.items()}
        primary = nll_breakdown({name: comps_cat[(1, name)] for name in _GROUP_NAMES})
        aux_offsets = [o for o in cfg.head_offsets if o != 1]
        aux_loss = (
            sum(_offset_total_bits(comps_cat, o) for o in aux_offsets) / len(aux_offsets) if aux_offsets else 0.0
        )
        sps = cfg.batch_size * cfg.grad_accum_steps / sw.elapsed
        samples = (step + 1) * cfg.batch_size * cfg.grad_accum_steps
        log: dict[str, object] = {
            "global_step": step,
            "samples": samples,
            "tokens": samples * cfg.L_ctx,
            "train/loss": primary["total"],  # offset-1 head (deployed), UNWEIGHTED; comparable to 016
            **{f"train/nll_{name}": primary[name] for name in _GROUP_NAMES},
            "train/aux_loss": aux_loss,  # mean total bits/frame over the auxiliary (offset != 1) heads
            "train/objective": objective_bits,  # the weighted policy objective actually backpropagated, bits
            "train/value_loss": awr_acc["value_loss"],  # V's MSE to the return (native units, not bits)
            "train/awr_ess": awr_acc["ess"],  # 1.0 = uniform weights; small = a few frames own the batch
            "train/awr_weight_max_frac": awr_acc["weight_max_frac"],
            "lr/muon": next(g["lr"] for g in opt.param_groups if g["use_muon"]),
            "lr/adam": next(g["lr"] for g in opt.param_groups if not g["use_muon"]),
            "train/gnorm": grad_norm.item(),
            "throughput/step_s": sw.elapsed,
            "throughput/samples_per_s": sps,
        }
        if step < 20 or step % 50 == 0:
            # ESS rides the console line: a weight distribution collapsing onto a few frames is the
            # failure mode of this experiment, and it must be visible in train.log without W&B.
            awr_note = (
                f" ess {awr_acc['ess']:.3f} v_mse {awr_acc['value_loss']:.4f}" if cfg.awr_enabled else " (awr off)"
            )
            print(
                f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: loss {primary['total']:.4f}{awr_note} "
                f"step_dt={sw.elapsed * 1000:.0f}ms ({sps:.1f} samples/s)",
                flush=True,
            )
        if cfg.ckpt_every > 0 and step > 0 and step % cfg.ckpt_every == 0:
            _save("latest.pt", step)
        if cfg.val_every > 0 and step > 0 and step % cfg.val_every == 0:
            vm = _val_log_dict()
            log.update(vm)
            print(
                f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: "
                f"action_nll {vm['val/loss']:.3f} btn_logloss {vm['val/btn_logloss']:.3f}",
                flush=True,
            )
        wandb.log(log)
        _drain_eval(wait=False)
        if cfg.eval_every > 0 and step > 0 and step % cfg.eval_every == 0:
            _launch_eval(step)

    _drain_eval(wait=True)  # finish the last async eval before the final pass
    vm_final = _val_log_dict()
    wandb.log({**vm_final, "global_step": cfg.max_steps})
    print(f"[final] action_nll {vm_final['val/loss']:.3f}", flush=True)
    _log_eval(cfg.max_steps, _eval_and_upload("final", n_matchups=cfg.final_eval_n_matchups))
    _save("final.pt", cfg.max_steps)
    try:
        if cfg.final_h2h_reference_run:
            _final_h2h(cfg, model, stats, ckpt_dir, uploader, ref_ckpt=h2h_ref_ckpt)
    finally:
        # Drain pending uploads even when the h2h raises: the box can self-destruct after exit,
        # so everything already written must reach R2.
        uploader.close()


# %%
def _cfg_from_state(saved: dict) -> TrainConfig:
    """Rebuild a ``TrainConfig`` from a checkpoint's saved cfg dict, tolerating
    schema drift in *eval/host* knobs across code versions: keys no longer on
    ``TrainConfig`` (e.g. the old ``eval_max_parallel``/``eval_replicas``, replaced
    by ``eval_parallel_per_cpu``) are dropped and new fields take their defaults, so
    past checkpoints still load. Model-identity fields (``d_model``, ``head_offsets``,
    …) are unaffected — they're always present and reconstruct exactly."""
    known = {f.name for f in fields(TrainConfig)}
    dropped = sorted(set(saved) - known)
    if dropped:
        print(f"[ckpt] dropping {len(dropped)} stale cfg key(s) not on current TrainConfig: {dropped}", flush=True)
    return TrainConfig(**{k: v for k, v in saved.items() if k in known})


def _load_ckpt(ckpt_path: str) -> tuple[GPT, TrainConfig, dict[str, FeatureStats], dict]:
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = _cfg_from_state(state["cfg"])
    embedded_counts = "button_combo_counts" in state["model"]
    button_combo_counts = None if embedded_counts else _load_button_combo_counts(cfg)
    validate_config(cfg, has_button_combo_counts=embedded_counts or button_combo_counts is not None)
    model = GPT(cfg).to(DEVICE)
    if button_combo_counts is not None:
        model.button_combo_counts.copy_(button_combo_counts.to(DEVICE))
    _load_model_state(model, state["model"])
    model.eval()
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    return model, cfg, stats, state


def eval_ckpt(
    ckpt_path: str,
    *,
    eval_exec_horizon: int | None = None,
    decode_temp: float | None = None,
    decode_temps: tuple[float, float, float, float] | None = None,
    decode_btn_support_min: int | None = None,
    decode_min_p: float | None = None,
    decode_click_trigger_fix: bool | None = None,
    eval_n_matchups: int | None = None,
    eval_max_frames: int | None = None,
    eval_seed: int | None = None,
    wandb_run_id: str | None = None,
    wandb_project: str = "hal",
    wandb_entity: str | None = None,
    wandb_label: str | None = None,
) -> None:
    """Load a checkpoint and run the prior-distribution vs-CPU sweep, printing the pooled metrics. Each
    override (execution horizon, temp, per-group temps, button-support floor, min-p, click=>trigger fix)
    replaces the trained cfg for this eval only (test-time sweep); ``None`` keeps the trained value."""
    model, cfg, stats, state = _load_ckpt(ckpt_path)
    exec_horizon = cfg.exec_horizon if eval_exec_horizon is None else eval_exec_horizon
    settings = _decode_settings(
        model,
        cfg,
        temp=decode_temp,
        temps=decode_temps,
        btn_support_min=decode_btn_support_min,
        min_p=decode_min_p,
        click_trigger_fix=decode_click_trigger_fix,
    )
    protocol = _eval_protocol(
        cfg,
        settings=settings,
        exec_horizon=exec_horizon,
        default_n_matchups=cfg.final_eval_n_matchups,
        n_matchups=eval_n_matchups,
        max_frames=eval_max_frames,
        seed=eval_seed,
    )
    _exec_horizon_offsets(model.head_offsets, exec_horizon)
    print(
        f"[eval] loaded {ckpt_path}  step={state['step']}  device={DEVICE}  exec_horizon={exec_horizon}  "
        f"temp={settings.temp}  temps={settings.temps}  btn_support_min={settings.btn_support_min}  "
        f"min_p={settings.min_p}  click_trigger_fix={settings.click_trigger_fix}",
        flush=True,
    )
    replay_dir = Path(ckpt_path).resolve().parent / "eval_replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    policy_index = itertools.count()

    def policy_factory() -> RecedingHorizon:
        return make_policy(
            model,
            stats,
            cfg,
            exec_horizon=protocol.exec_horizon,
            decode_temp=settings.temp,
            decode_temps=settings.temps,
            decode_btn_support_min=settings.btn_support_min,
            decode_min_p=settings.min_p,
            decode_click_trigger_fix=settings.click_trigger_fix,
            decode_seed=protocol.seed + next(policy_index),
        )

    print(
        f"\n[eval] ===== vs-cpu, {protocol.n_matchups} prior-sampled matchups, "
        f"max_parallel={protocol.max_parallel} (instant-restart, seed={protocol.seed}) =====",
        flush=True,
    )
    metrics = _run_eval_sweep(
        policy_factory,
        protocol=protocol,
        replay_dir=replay_dir,
        rows_path=replay_dir / "match_rows.json",
    )
    if wandb_run_id is not None:
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            id=wandb_run_id,
            resume="allow",
            name=wandb_label,
            config={
                "manual_eval_checkpoint": str(Path(ckpt_path).resolve()),
                "manual_eval_label": wandb_label or "",
            },
        )
        wandb.define_metric("global_step")
        wandb.define_metric("eval/*", step_metric="global_step")
        protocol_log = {
            f"eval_protocol/{key}": value
            for key, value in asdict(protocol).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        wandb.log(
            {
                **{f"eval/{key}": value for key, value in metrics.items()},
                **protocol_log,
                "global_step": state["step"],
            }
        )
        run.summary["eval/last_checkpoint"] = str(Path(ckpt_path).resolve())
        run.summary["eval/last_label"] = wandb_label or "manual"
        wandb.finish()
    print(f"  {metrics}", flush=True)


# %%
def run_eval_worker(ckpt_path: str, step: int, result_path: str, replay_dir: str) -> None:
    """One-shot closed-loop eval for the async path: load a checkpoint, sweep vs CPU, and write the
    flat metric dict to ``result_path`` (atomically) with the .slp recordings under ``replay_dir``.
    Touches neither W&B nor R2 — the launching trainer is the sole writer/uploader."""
    model, cfg, stats, _ = _load_ckpt(ckpt_path)
    replay_path = Path(replay_dir)
    metrics = eval_vs_cpu(
        model,
        stats,
        cfg,
        max_frames=cfg.eval_max_frames,
        replay_dir=replay_path,
        rows_path=replay_path / "match_rows.json",
    )
    out = Path(result_path)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"step": step, "metrics": metrics}))
    tmp.replace(out)  # atomic rename: the trainer never reads a partial file
    print(f"[eval-worker] step {step}: {metrics}", flush=True)


# %%
_AUDIT_BETAS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)


def return_audit(cfg: TrainConfig, *, split: str, betas: tuple[float, ...] = _AUDIT_BETAS) -> dict[str, object]:
    """Offline audit of the return distribution one split implies, and of the weights each ``beta``
    would produce. Run it BEFORE picking ``awr_beta``: beta is a temperature on an empirical
    distribution, so it cannot be chosen from first principles, and a beta that collapses the
    effective sample size silently turns 16k steps of training into a few thousand frames.

    Two baselines are reported because a trained ``V`` sits between them: a GLOBAL mean (V knows
    nothing about the state) and a PER-REPLAY mean (V has learned the match's overall trajectory).
    The real ESS lands between the two columns."""
    root = Path(cfg.data_root) / split
    mds = StreamingDataset(local=str(root), batch_size=1, shuffle=False)
    per_replay: list[np.ndarray] = []
    for sample in mds:
        check_schema_version(sample, expected=cfg.mds_schema_version)
        # One row per (replay, port): the ego is drawn per window at train time, so both are real targets.
        per_replay.extend(replay_returns(sample, gamma=cfg.awr_gamma, damage_shaping=cfg.awr_damage_shaping).values())
    if not per_replay:
        raise RuntimeError(f"no replays under {root}")
    returns = np.concatenate(per_replay)
    centered = {
        "global mean": returns - returns.mean(),
        "per-replay mean": np.concatenate([g - g.mean() for g in per_replay]),
    }
    rows = []
    for beta in betas:
        row: dict[str, object] = {"beta": beta}
        for label, advantage in centered.items():
            weight, stats = awr_weights(torch.from_numpy(advantage), beta=beta, weight_max=cfg.awr_weight_max)
            row[label] = (stats["ess"], stats["weight_max_frac"], float(weight.max()))
        rows.append(row)
    quantiles = (1, 5, 25, 50, 75, 95, 99)
    out: dict[str, object] = {
        "replays": len(per_replay) // 2,
        "frames": int(returns.size),
        "gamma": cfg.awr_gamma,
        "damage_shaping": cfg.awr_damage_shaping,
        "mean": float(returns.mean()),
        "std": float(returns.std()),
        "within_replay_std": float(np.mean([g.std() for g in per_replay])),
        "near_zero_frac": float(np.mean(np.abs(returns) < 0.01)),
        "quantiles": {f"p{q}": float(np.percentile(returns, q)) for q in quantiles},
        "beta_table": rows,
    }
    print(
        f"[audit] {out['replays']} replays  {out['frames']} port-frames  gamma={cfg.awr_gamma}  "
        f"damage_shaping={cfg.awr_damage_shaping}",
        flush=True,
    )
    print(f"[audit] G mean={out['mean']:+.4f} std={out['std']:.4f} within-replay std={out['within_replay_std']:.4f}")
    print(f"[audit] G quantiles {out['quantiles']}")
    print(f"[audit] |G| < 0.01 on {out['near_zero_frac']:.1%} of frames")
    print(f"[audit] {'beta':>6}  {'ESS (global V)':>22}  {'ESS (per-replay V)':>22}   (ESS, clip frac, w_max)")
    for row in rows:
        cells = "  ".join(
            f"({value[0]:.3f}, {value[1]:.4f}, {value[2]:6.2f})" for label, value in row.items() if label != "beta"
        )
        print(f"[audit] {row['beta']:>6}  {cells}")
    return out


# %%
@dataclass
class Args:
    """Top-level CLI surface. Pass TrainConfig fields as kebab-case flags, e.g. ``--cfg.d-model 512``."""

    cfg: TrainConfig = field(default_factory=TrainConfig)
    eval: str | None = None  # ckpt path; closed-loop eval instead of train
    eval_exec_horizon: int | None = None  # override execution horizon s for --eval (chunked decode; 1=per-frame)
    eval_temp: float | None = None  # override decode temperature for --eval
    eval_temps: tuple[float, float, float, float] | None = None  # per-group temps (buttons, main, c, triggers)
    eval_btn_support_min: int | None = None  # mask button combos with < this many train frames (0=off)
    eval_min_p: float | None = None  # min-p nucleus: keep classes with p >= min_p * p_max
    eval_click_trigger_fix: bool | None = None  # force trigger_l/r to 1.0 on a digital L/R click
    eval_n_matchups: int | None = None  # manual --eval override; default is cfg.final_eval_n_matchups (96)
    eval_max_frames: int | None = None  # manual --eval override; default is checkpoint cfg.eval_max_frames
    eval_seed: int | None = None  # manual --eval sampling/bootstrap seed override
    wandb_run_id: str | None = None  # resume an existing run and log this manual eval to it
    wandb_project: str = "hal"
    wandb_entity: str | None = None
    wandb_label: str | None = None
    resume: str | None = None  # run_name to resume; pulls latest.pt (local, else R2)
    comment: str = ""
    # Offline return/weight audit on cfg.data_root instead of training. Pick awr_beta from it.
    audit_returns: bool = False
    audit_split: str = "train"
    # internal: one-shot async-eval worker (the trainer spawns this; not for manual use).
    eval_worker: str | None = None  # ckpt path
    eval_worker_step: int = 0
    eval_worker_result: str | None = None
    eval_worker_replay: str | None = None


def main(args: Args) -> None:
    if args.audit_returns:
        return_audit(args.cfg, split=args.audit_split)
        return
    if args.eval_worker is not None:
        assert args.eval_worker_result is not None and args.eval_worker_replay is not None
        run_eval_worker(args.eval_worker, args.eval_worker_step, args.eval_worker_result, args.eval_worker_replay)
        return
    if args.eval is not None:
        eval_ckpt(
            args.eval,
            eval_exec_horizon=args.eval_exec_horizon,
            decode_temp=args.eval_temp,
            decode_temps=args.eval_temps,
            decode_btn_support_min=args.eval_btn_support_min,
            decode_min_p=args.eval_min_p,
            decode_click_trigger_fix=args.eval_click_trigger_fix,
            eval_n_matchups=args.eval_n_matchups,
            eval_max_frames=args.eval_max_frames,
            eval_seed=args.eval_seed,
            wandb_run_id=args.wandb_run_id,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_label=args.wandb_label,
        )
        return
    if args.resume is not None:
        state = load_for_resume(args.resume, Path("runs") / args.resume, device=DEVICE)
        if state is None:
            raise SystemExit(f"no latest.pt for run {args.resume!r} (local or R2)")
        # Only pure host-scaling knobs (worker/prefetch counts) follow the current code; the
        # model-identity knobs MUST come from the checkpoint so a resume can't silently change them.
        d = TrainConfig()
        cfg = replace(_cfg_from_state(state["cfg"]), num_workers=d.num_workers, prefetch_factor=d.prefetch_factor)
        stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
        train(cfg, stats, resume_run=args.resume, resume_state=state)
        return
    cfg = args.cfg
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    auto_comment = f"gpt-{cfg.max_steps // 1000}k-b{cfg.batch_size}"
    train(cfg, stats, comment=args.comment or auto_comment)


if __name__ == "__main__":
    main(tyro.cli(Args))
