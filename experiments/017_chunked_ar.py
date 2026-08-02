"""Chunked autoregressive action decoder: the exact joint over a 16-frame action chunk.

MOTIVATION
----------
013 factorizes the action distribution twice over: INDEPENDENT-PER-GROUP (buttons, main stick,
c-stick, triggers are conditionally independent given the trunk hidden) and INDEPENDENT-PER-FRAME
(each future offset gets its own head, conditioned only on the context). Both factorizations are
wrong in the same direction: a multimodal transition — "wavedash left OR wavedash right", "jump then
fastfall OR hold drift" — has a mode-averaged product-of-marginals fit whose per-coordinate argmax /
per-coordinate sample is the *previous* input. That is the measured failure: >90% ``pred_persistence``
(the model's frame-to-frame prediction almost never changes) and a closed-loop policy that mostly
holds whatever it last held.

017 models the JOINT distribution over an H=16-frame action chunk by the exact chain rule across
BOTH axes — time and modality. From each context position the chunk is a 64-token autoregressive
sequence, so a coherent multi-frame motion is a single high-probability path rather than a product of
mode-averaged marginals, and within a frame the c-stick can depend on the buttons that were just
chosen for that same frame.

TOKENIZATION
------------
Time-major, so the executed prefix is a token prefix: frame i+1's four groups in the fixed order
(buttons, main_stick, c_stick, triggers), then frame i+2's, ... 16 frames x 4 groups = 64 tokens.
The group quantizers/vocabs are 013's unchanged (``quantize_groups`` / ``dequantize_groups`` over
``scoring``: 256 button combos, 65 main-stick clusters, 9 c-stick clusters, 25 joint trigger cells),
so targets, decode and every discretization-sensitive metric are byte-identical to 013's.

CONDITIONING SCHEME
-------------------
The trunk hidden ``h_i`` is projected to ``d_dec`` and ADDED to every decoder position (with a learned
BOS embedding as the first input), rather than prepended as a separate prefix token. Reason: the
decoder is deliberately tiny (1-2 layers). Routing all conditioning through one prefix position makes
every one of the 64 positions spend attention capacity re-retrieving ``h_i`` through a single shared
value projection; broadcasting it into the residual stream makes the context unconditionally available
at every position and every layer for one add. It also keeps the decoder sequence length exactly equal
to the number of predicted tokens (no prefix off-by-one), which matters because sampling the execution
horizon ``s`` is then exactly the first ``4*s`` positions.

Input embeddings are one ``[A_VOCAB, d_dec]`` table partitioned by ``_GROUP_OFFSETS`` — four disjoint
per-group tables in one lookup, mirroring how the group logits concatenate. Positions are a factorized
learned ``time_emb[H] + group_emb[4]``, indexed by the token the slot PREDICTS (so the query slot knows
which group it must emit). Output is four separate ``Linear(d_dec, vocab_g)`` heads: the four vocabs
are disjoint and differ 28x in size, so there is nothing to tie; nothing is tied to the input table
either (the input table is a value representation of an already-chosen action, the head is a
classifier over the next one).

TRAINING COMPUTE — why chunks are decoded at M sampled positions, not densely
----------------------------------------------------------------------------
Forward FLOPs at the defaults (d_model 256, 8 layers, L_ctx 256 / d_dec 128, 2 layers, 64 tokens):

    trunk, per window   2*12*d_model^2*n_layers*L_ctx + attn  ~= 3.2 + 0.3  ~= 3.5 GFLOP
    decoder, per chunk  2*12*d_dec^2*n_dec_layers*64 + attn + one group head ~= 0.056 GFLOP

Decoding a chunk at every one of the 256 context positions is 256 * 0.056 = 14.3 GFLOP/window — 4.1x
the trunk it is supposed to be a head on, and (the binding constraint) 1024 * 256 * 64 = 16.8M decoder
tokens per batch, i.e. ~4 GB per stored bf16 activation tensor. So chunks are decoded at ``M_pos``
context positions per sample (default 32): 32 * 0.056 = 1.79 GFLOP/window, 0.51x the trunk, ~1.5x a
trunk-only step, ~0.5 GB per activation. Supervision is 32*64 = 2048 target tokens/sample vs 013's
256*4*4 = 4096 — half the targets for ~1.5x the compute, and each target is a strictly harder
(fully-conditioned) prediction.

The sampled positions are drawn uniformly WITHOUT per-sample stratification from the batch-flattened
valid set (``i >= ctx_pad``), so the pooled estimate is unbiased for the dense-over-all-valid-positions
mean — a sample with more valid positions contributes proportionally more chunks, exactly as dense
would. Target coverage needs no extra guard: with the chunk horizon H equal to the loader's
``L_chunk``, every valid position's frames ``i+1 .. i+H`` are inside ``[history | target]``, which is
why ``H <= VAL_L_CHUNK`` is a hard config check (the frozen val geometry must cover every target too).

Teacher forcing, exact NLL, plain unweighted mean over tokens. No transition upweighting, no auxiliary
head weights: with one decoder there is nothing to trade off.

METRIC COMPARABILITY WITH 013
-----------------------------
``val/nll_off1`` is the frame-1 JOINT NLL: the sum of the four chain-rule terms
``-log2 p(buttons | ctx) - log2 p(main | ctx, buttons) - log2 p(c | ctx, buttons, main)
- log2 p(trig | ctx, buttons, main, c)`` = ``-log2 p(a_{i+1} | ctx)``. 013's ``val/loss`` is the sum of
its four offset-1 group marginals = ``-log2 prod_g p_g(a_{g,i+1} | ctx)``. Same random variable, same
conditioning set, same discretization, same frozen ``VAL_L_CHUNK=16`` val windows: DIRECTLY COMPARABLE
in bits, and 013's independent family is a strict subfamily of this one, so a well-fit 017 cannot be
worse in population. (017 evaluates on a uniform random subset of the same valid positions — unbiased,
with extra variance; the val position draw is re-seeded identically every validation so the number is
stable across steps.) ``val/nll_buttons`` is comparable for the same reason (buttons is first in the
chain, hence conditioned on the context alone); the other three per-group terms are NOT — they are
conditionals where 013's are marginals.

``nll_off{2..16}`` are teacher-forced conditionals (frame i+o given the context AND the true actions of
frames i+1..i+o-1), so they are NOT comparable to 013's independent per-offset heads, which see only
the context. They measure the chunk's internal predictability, which is the quantity the joint model is
supposed to exploit.

DEPLOYMENT
----------
Same ``make_policy`` / ``RecedingHorizon`` contract: one trunk forward per replan, then AR-sample only
the first ``s`` frames' ``4*s`` tokens (time-major makes the executed horizon a token prefix; sampling
the unused tail would be pure waste), so the returned chunk is ``[B, s, A_DIM]`` and the harness runs
with ``L_chunk = s``, ``d = 0``. 013's decode hygiene applies per group inside the AR loop: per-group
temperatures, min-p, button-support masking, and the click => trigger fix after dequantization. The
decoder deliberately has NO KV cache — at 4*s <= 64 tokens over a 2-layer d=128 stack the AR loop is
kernel-launch-bound, not prefix-length-bound, so a cache would buy nothing: measured on an RTX 3060 at
the default geometry, one replan costs 21.8 / 25.4 / 31.0 / 54.7 / 101.7 ms at s = 1 / 2 / 4 / 8 / 16
(one slot), of which 11.9 ms is the trunk forward alone, and the per-token AR cost FALLS from 2.5 ms at
s=1 to 1.4 ms at s=16 as the fixed overhead amortizes. Against the 16.6ms * s budget that is 1.31x at
s=1 (over budget) and 0.76x / 0.47x / 0.41x / 0.38x from s=2 up — i.e. the joint decoder is real-time
from a two-frame execution horizon, and s=1 is trunk-bound rather than decoder-bound.
``--bench-decode`` re-measures this on the host at hand.

Dropped from 013 (subsumed or dead): the per-offset auxiliary heads and ``aux_loss_weight``;
``transition_loss_weight``; the shared-trunk gradient Gram/cosine diagnostics (they answered "do the
auxiliary heads pull the trunk apart" — there is one head now); and the experimental incremental-KV
trunk decoder, which was disabled by default and never validated.

Run:
    uv run experiments/017_chunked_ar.py
    uv run experiments/017_chunked_ar.py --cfg.d-dec 192 --cfg.n-dec-layers 2 --cfg.m-pos 32
    uv run experiments/017_chunked_ar.py --eval <ckpt> --eval-exec-horizon 4 --eval-temp 0.9
    uv run experiments/017_chunked_ar.py --bench-decode --cfg.batch-size 8
"""

# %%
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import contextlib
import itertools
import json
import math
import subprocess
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
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

import wandb
from hal import streams
from hal.data.feature_stats import FeatureStats
from hal.eval.cross_stage import MatchRow
from hal.eval.cross_stage import sweep_vs_cpu_prior_with_rows
from hal.eval.cross_stage import vs_cpu_metrics
from hal.eval.harness import default_session_cfg
from hal.training import scoring
from hal.training.checkpoints import BackgroundUploader
from hal.training.checkpoints import load_for_resume
from hal.training.checkpoints import save_checkpoint
from hal.training.closed_loop import RecedingHorizon
from hal.training.dataloader import VAL_L_CHUNK
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_LN2 = math.log(2.0)
# Melee runs at 60 fps: one frame of execution buys 1000/60 ms of decode budget.
_FRAME_MS = 1000.0 / 60.0

# Action-vector channel split (A_DIM=14): [0:6] sticks+triggers (continuous), [6:14] buttons {0,1}.
_N_CONT = 6
_N_BUTTONS = A_DIM - _N_CONT

# Per-frame input: all four players' gamestate concatenated in the feature dim.
_PLAYER_PREFIXES: tuple[str, ...] = ("ego", "ego_nana", "opp_nana", "opp")

# Output groups (fixed order) + their discrete vocab sizes from the scoring discretizers. This order is
# also the within-frame autoregressive order of the decoder's token sequence.
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

# Action-vector channels for the click=>trigger hygiene fix (digital L/R click => analog trigger = 1.0).
_TRIGGER_L_CH = ACTION_CHANNELS.index("trigger_l")
_TRIGGER_R_CH = ACTION_CHANNELS.index("trigger_r")
_BUTTON_L_CH = ACTION_CHANNELS.index("button_l")
_BUTTON_R_CH = ACTION_CHANNELS.index("button_r")


# %%
@dataclass
class TrainConfig:
    # GPT backbone (trunk over per-frame gamestate tokens)
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 4
    # Chunked AR action decoder. A small causal transformer over the L_chunk * N_GROUPS token sequence,
    # conditioned on the trunk hidden (added to every position). Kept deliberately small: it runs at
    # M_pos positions per sample in training and 4*s sequential steps per replan at deploy.
    d_dec: int = 128
    n_dec_layers: int = 2
    n_dec_heads: int = 4
    # Chunk horizon H (frames of action predicted from one context position). Must be <= VAL_L_CHUNK so
    # the FROZEN val window geometry covers every target frame; the train loader uses it as L_chunk.
    L_chunk: int = 16
    # Chunks decoded per sample: M_pos context positions drawn uniformly from that sample's valid
    # positions, pooled batch-wide (unbiased for the dense-over-all-valid mean). Dense (M_pos = L_ctx)
    # is 4x the trunk's FLOPs and does not fit in memory — see the module docstring's FLOP budget.
    M_pos: int = 32
    # Per-sample ego-history input dropout (train only): with probability p, zero a sample's ego
    # controller-history slice of the context token so the trunk cannot lean on copying its own recent
    # inputs. Lives purely in the model's input assembly (targets untouched).
    history_dropout_p: float = 0.0
    # Matchup conditioning (schema v5). char/stage embeddings are indexed by the RAW libmelee id
    # (characters 0-26 dense; stages sparse in 0-26), so the vocab must exceed the max id, not the
    # number of included categories; out-of-range ids clamp to the last row.
    char_vocab: int = 32
    char_dim: int = 12
    stage_vocab: int = 32
    stage_dim: int = 4
    # closed-loop sampling temperature. Greedy argmax collapses the policy to a do-nothing fixed
    # point in closed loop, so deployed play always samples; argmax stays for the recon metric.
    decode_temp: float = 1.0
    # Decode-time hygiene, applied per group INSIDE the AR loop. decode_temps overrides the single
    # decode_temp per group in _GROUP_NAMES order (buttons, main_stick, c_stick, triggers); None ->
    # decode_temp for all. decode_btn_support_min >= 1 masks button combos with fewer than that many
    # train frames (per the configured dataset-scoped count artifact) to -inf before softmax/argmax.
    # decode_min_p > 0 keeps only classes with p >= decode_min_p * p_max per group, then renormalizes.
    # decode_click_trigger_fix forces trigger_l/r to 1.0 wherever the sampled combo sets the digital
    # L/R bit (the only train-supported click joint).
    decode_temps: tuple[float, float, float, float] | None = None
    decode_btn_support_min: int = 0
    decode_min_p: float = 0.0
    decode_click_trigger_fix: bool = False
    # Chunked execution (deploy-time only): replan every s frames, AR-sampling exactly the first
    # 4*s tokens from ONE trunk forward and executing all s actions. 1 <= s <= L_chunk. Training is
    # unaffected — the decoder always learns the full L_chunk chain; eval can sweep s without retraining.
    exec_horizon: int = 1
    # Reproducible training RNG and transformer context geometry.
    seed: int = 0
    L_ctx: int = 256
    # optimization
    batch_size: int = 1024
    grad_accum_steps: int = 1
    # Two LRs: Muon for the trunk AND decoder blocks' hidden matrices, AdamW for the input/conditioning
    # projections, the vocab heads, the embeddings, and biases.
    muon_lr: float = 0.02
    adam_lr: float = 8.5e-4
    weight_decay: float = 0.01
    # When False, keep the decoder's output-head weights out of weight decay (route them to the
    # no-decay AdamW group).
    head_weight_decay: bool = True
    # Shared warmup/cosine schedule and training duration.
    warmup_steps: int = 500
    max_steps: int = 2**15
    amp_dtype: str = "bfloat16"  # "bfloat16" | "float32"
    allow_tf32: bool = True
    # eval cadence
    val_every: int = 1024
    val_n_batches: int = 16
    # Validation-only rarity threshold. The metric is emitted only when the checkpoint embeds validated
    # full-dataset button counts; it never falls back to a reference-sample table.
    diagnostic_rare_button_count: int = 100
    # Closed-loop evaluation cadence and per-boot frame budget. eval_every == 0 disables closed-loop
    # eval ENTIRELY (periodic and final) — the switch for smoke runs and boxes without an emulator.
    eval_every: int = 2048
    eval_max_frames: int = 7200
    # Periodic eval is a cheap diagnostic; final eval uses enough fixed prior-sampled matchups for a useful estimate.
    # These are statistical sample sizes, independent of host concurrency.
    eval_n_matchups: int = 16
    final_eval_n_matchups: int = 96
    eval_seed: int = 0
    # Closed-loop eval parallelism scales with the box: max_parallel = round(this * cpu_count).
    eval_parallel_per_cpu: float = 1.0
    # Closed-loop eval runs in a background subprocess. By default the trainer waits for that
    # subprocess before resuming, so evaluator and trainer never contend for the same CUDA device.
    eval_overlap_training: bool = False
    # If an eval is still running at the next boundary, the trainer waits up to this bound and
    # then kills the worker.
    eval_timeout_seconds: float = 900.0
    # checkpointing
    ckpt_every: int = 2048
    # data (v5 MDS carries the stage + p{1,2}_character + nana columns)
    data_root: str = "data/processed/ranked-anonymized-1/mds"
    # Optional versioned JSON artifact containing full-dataset button-combo counts. Required when
    # decode_btn_support_min > 0.
    button_combo_counts_path: str | None = None
    # Streaming dataset cache and shuffle geometry.
    cache_limit_gb: int = 440
    shuffle_block_size: int = 2000
    # Each replay deserialized off disk yields this many non-overlapping windows,
    # amortizing the whole-replay read (the disk bottleneck) over K samples. Train
    # only; val stays 1/replay so its loss stays comparable across runs.
    windows_per_replay: int = 4
    val_split: str = "val"
    num_workers: int = 16
    prefetch_factor: int = 4


def _model_tag(cfg: TrainConfig) -> str:
    return (
        f"car-d{cfg.d_model}-L{cfg.n_layers}-h{cfg.n_heads}-Lc{cfg.L_ctx}"
        f"-dec{cfg.d_dec}x{cfg.n_dec_layers}-H{cfg.L_chunk}-M{cfg.M_pos}"
    )


def _eval_max_parallel(cfg: TrainConfig, n_matchups: int) -> int:
    """Concurrent Dolphin boots per wave; never changes the fixed statistical sample size."""
    return min(n_matchups, max(1, round(cfg.eval_parallel_per_cpu * (os.cpu_count() or 1))))


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
# --- transformer primitives (nanoGPT-style: RMSNorm, causal SDPA; rotary on the trunk only) ----
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
    """Causal SDPA over ``d`` channels. ``rotary`` selects the trunk's relative positions; the chunk
    decoder passes ``rotary=False`` because its slots are addressed by learned (time, group) embeddings,
    not by a frame distance."""

    def __init__(self, d: int, n_heads: int, *, rotary: bool) -> None:
        super().__init__()
        if n_heads <= 0 or d % n_heads != 0:
            raise ValueError(f"d={d} must be divisible by positive n_heads={n_heads}")
        self.n_heads = n_heads
        self.d = d
        self.head_dim = d // n_heads
        if rotary and self.head_dim % 2 != 0:
            raise ValueError(f"rotary attention head_dim must be even, got {self.head_dim}")
        self.c_attn = nn.Linear(d, 3 * d, bias=False)
        self.c_proj = nn.Linear(d, d, bias=False)
        self.rotary = Rotary(self.head_dim) if rotary else None

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "B L d"], mask: Bool[Tensor, "B_mask 1 L L"]) -> Float[Tensor, "B L d"]:
        B, L, _ = x.shape
        q, k, v = self.c_attn(x).split(self.d, dim=2)
        q = q.view(B, L, self.n_heads, self.head_dim)
        k = k.view(B, L, self.n_heads, self.head_dim)
        v = v.view(B, L, self.n_heads, self.head_dim)
        if self.rotary is not None:
            cos, sin = self.rotary(q)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, L, self.d)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.c_fc = nn.Linear(d, 4 * d, bias=False)
        self.c_proj = nn.Linear(4 * d, d, bias=False)

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "B L d"]) -> Float[Tensor, "B L d"]:
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, n_layers: int, *, rotary: bool) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(d, n_heads, rotary=rotary)
        self.mlp = MLP(d)
        self.attn_scale = 1 / (2 * n_layers) ** 0.5

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "B L d"], mask: Bool[Tensor, "B_mask 1 L L"]) -> Float[Tensor, "B L d"]:
        x = x + self.attn_scale * self.attn(rmsnorm(x), mask)
        x = x + self.mlp(rmsnorm(x))
        return x


# %%
class ChunkDecoder(nn.Module):
    """Causal transformer over one chunk's ``L_chunk * N_GROUPS`` action tokens, time-major.

    Slot ``j`` predicts token ``j`` = (frame ``j // N_GROUPS + 1``, group ``j % N_GROUPS``); its INPUT
    is token ``j - 1`` (a learned BOS at ``j = 0``), plus the factorized learned position embedding for
    the token it predicts, plus the projected trunk hidden broadcast to every slot. Trained with teacher
    forcing over all ``n_dec_tok`` slots; sampled one token at a time over the first ``4*s`` slots.
    """

    causal: Tensor
    time_ids: Tensor
    group_ids: Tensor
    group_offsets: Tensor

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.L_chunk = cfg.L_chunk
        self.n_dec_tok = cfg.L_chunk * N_GROUPS
        self.cond_proj = nn.Linear(cfg.d_model, cfg.d_dec, bias=False)
        self.bos = nn.Parameter(torch.zeros(cfg.d_dec))
        # One [A_VOCAB, d_dec] table partitioned by _GROUP_OFFSETS: four disjoint per-group tables in a
        # single lookup, so an input token id is the group's class shifted into the concatenated space.
        self.tok_emb = nn.Embedding(A_VOCAB, cfg.d_dec)
        self.time_emb = nn.Embedding(cfg.L_chunk, cfg.d_dec)
        self.group_emb = nn.Embedding(N_GROUPS, cfg.d_dec)
        self.blocks = nn.ModuleList(
            [Block(cfg.d_dec, cfg.n_dec_heads, cfg.n_dec_layers, rotary=False) for _ in range(cfg.n_dec_layers)]
        )
        self.heads = nn.ModuleList([nn.Linear(cfg.d_dec, v) for v in _GROUP_VOCABS])
        slot = torch.arange(self.n_dec_tok)
        # Geometry, not learned state: kept out of the state dict so a checkpoint carries weights only.
        self.register_buffer("time_ids", slot // N_GROUPS, persistent=False)
        self.register_buffer("group_ids", slot % N_GROUPS, persistent=False)
        self.register_buffer("group_offsets", torch.tensor(_GROUP_OFFSETS), persistent=False)
        causal = torch.tril(torch.ones(self.n_dec_tok, self.n_dec_tok, dtype=torch.bool))
        self.register_buffer("causal", causal, persistent=False)

    @jaxtyped(typechecker=beartype)
    def project_cond(self, h: Float[Tensor, "n_chunk d_model"]) -> Float[Tensor, "n_chunk d_dec"]:
        """Trunk hidden → the vector added to every decoder slot. Hoisted out of ``hidden`` so the AR
        sampling loop projects once per replan rather than once per token."""
        return self.cond_proj(h)

    @jaxtyped(typechecker=beartype)
    def hidden(
        self, cond: Float[Tensor, "n_chunk d_dec"], prev: Int[Tensor, "n_chunk n_prev"]
    ) -> Float[Tensor, "n_chunk n_slot d_dec"]:
        """Slot states for the ``len(prev) + 1`` slots decodable from the already-known tokens ``prev``
        (flat ``A_VOCAB`` ids). Slot ``j``'s state feeds group ``j % N_GROUPS``'s head to score token
        ``j``; the last slot is the next token to sample."""
        n_slot = prev.shape[1] + 1
        if n_slot > self.n_dec_tok:
            raise ValueError(f"decoder has {self.n_dec_tok} slots, got {n_slot} inputs")
        x = torch.cat([self.bos.expand(prev.shape[0], 1, -1), self.tok_emb(prev)], dim=1)
        pos = self.time_emb(self.time_ids[:n_slot]) + self.group_emb(self.group_ids[:n_slot])
        x = x + pos[None] + cond[:, None, :]
        mask = self.causal[None, None, :n_slot, :n_slot]
        for block in self.blocks:
            x = block(x, mask)
        return rmsnorm(x)


# %%
class GPT(nn.Module):
    """Causal GPT over per-frame gamestate tokens + a chunked AR action decoder. ``forward`` returns the
    trunk hidden per frame; ``decoder`` turns any hidden into the exact joint over the next ``L_chunk``
    frames of actions, factorized time-major by the chain rule over (frame, group)."""

    main_centers: Tensor
    c_centers: Tensor
    trig_centers: Tensor
    button_combo_counts: Tensor

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        if not cfg.decode_temp > 0:
            raise ValueError(f"decode_temp must be > 0, got {cfg.decode_temp}")
        if not 0.0 <= cfg.history_dropout_p <= 1.0:
            raise ValueError(f"history_dropout_p must be in [0, 1], got {cfg.history_dropout_p}")
        self.history_dropout_p = cfg.history_dropout_p
        self.L_ctx = cfg.L_ctx
        self.L_chunk = cfg.L_chunk
        self.M_pos = cfg.M_pos

        # Gamestate categoricals: one table per feature name, shared across the four players.
        self.cat_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocab, dim) for name, (vocab, dim) in CAT_FEATURES.items()}
        )
        self.char_emb = nn.Embedding(cfg.char_vocab, cfg.char_dim)
        self.stage_emb = nn.Embedding(cfg.stage_vocab, cfg.stage_dim)
        per_player = len(FLOAT_FEATURES) * 2 + sum(dim for _, dim in CAT_FEATURES.values())  # float+mask+cat
        d_in = len(_PLAYER_PREFIXES) * per_player + A_DIM + 2 * cfg.char_dim + cfg.stage_dim

        self.ctx_proj = nn.Linear(d_in, cfg.d_model)
        self.blocks = nn.ModuleList(
            [Block(cfg.d_model, cfg.n_heads, cfg.n_layers, rotary=True) for _ in range(cfg.n_layers)]
        )
        self.decoder = ChunkDecoder(cfg)

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
        # draw a Bernoulli keep-mask (probability history_dropout_p of dropping) and zero the whole slice
        # for dropped samples, forcing the trunk off copying its own recent inputs.
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
        """Backbone hidden (one rmsnorm'd vector per frame); callers feed selected frames to ``decoder``."""
        x = self._context_tokens(features)
        mask = self._attn_mask(ctx_pad, x.size(1), x.device)
        for block in self.blocks:
            x = block(x, mask)
        return rmsnorm(x)


# %%
def _quantize(model: GPT, actions: Tensor) -> Tensor:
    return quantize_groups(model.main_centers, model.c_centers, model.trig_centers, actions)


def _dequantize(model: GPT, idx: Tensor) -> Tensor:
    return dequantize_groups(model.main_centers, model.c_centers, model.trig_centers, idx)


@jaxtyped(typechecker=beartype)
def _quantized_full(model: GPT, batch: TrainBatch) -> Int[Tensor, "B L_full n_groups"]:
    """``[history | target]`` quantized once: frame ``t < L_ctx`` is a context frame's own action, frame
    ``L_ctx + j`` is target frame ``j``. Position ``i``'s chunk targets are frames ``i+1 .. i+L_chunk``,
    which stay in range for every ``i < L_ctx`` because the loader's chunk covers ``L_chunk`` frames."""
    if batch.target.size(1) < model.L_chunk:
        raise ValueError(f"target chunk has {batch.target.size(1)} frames, but the decoder horizon is {model.L_chunk}")
    a_full = torch.cat([stack_actions(batch.context.features), batch.target[:, : model.L_chunk]], dim=1)
    return _quantize(model, a_full)


@jaxtyped(typechecker=beartype)
def _valid_positions(ctx: Context, L_ctx: int) -> Bool[Tensor, "B L_ctx"]:
    """Context positions that are real (non-pad) frames. ``i >= ctx_pad`` also guarantees every target
    frame ``i + o`` (``o >= 1``) is real, so no further mask is needed."""
    pos = torch.arange(L_ctx, device=ctx.ctx_pad.device)
    return pos[None, :] >= ctx.ctx_pad[:, None]


@jaxtyped(typechecker=beartype)
def sample_positions(
    valid: Bool[Tensor, "B L_ctx"], M_pos: int, gen: torch.Generator | None = None
) -> Int[Tensor, " n_chunk"]:
    """``B * M_pos`` flat indices into ``[B*L_ctx]``, drawn uniformly WITH REPLACEMENT from the pooled
    valid set. Pooling (rather than drawing M_pos per sample) is what makes the reduction over the
    selected chunks an unbiased estimator of the dense over-all-valid-positions mean."""
    flat = valid.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    if flat.numel() == 0:
        raise ValueError("batch has no valid context positions (every sample is fully padded)")
    n_chunk = valid.shape[0] * M_pos
    pick = torch.randint(flat.numel(), (n_chunk,), device=valid.device, generator=gen)
    return flat[pick]


@jaxtyped(typechecker=beartype)
def _chunk_targets(
    q_full: Int[Tensor, "B L_full n_groups"], sel: Int[Tensor, " n_chunk"], L_ctx: int, L_chunk: int
) -> tuple[Int[Tensor, "n_chunk L_chunk n_groups"], Int[Tensor, "n_chunk n_groups"]]:
    """Per selected flat position: the chunk's ``L_chunk`` target frames and the action of the context
    frame itself (the "previous input" every change/transition metric is measured against)."""
    b, i = sel // L_ctx, sel % L_ctx
    t = i[:, None] + 1 + torch.arange(L_chunk, device=sel.device)[None, :]
    return q_full[b[:, None], t], q_full[b, i]


@jaxtyped(typechecker=beartype)
def _cond_at(h: Float[Tensor, "B L_ctx d_model"], sel: Int[Tensor, " n_chunk"]) -> Float[Tensor, "n_chunk d_model"]:
    return h.reshape(-1, h.shape[-1])[sel]


def chunk_logits(model: GPT, cond: Tensor, tgt: Tensor) -> list[Tensor]:
    """Teacher-forced logits for every token of every selected chunk: one ``[n_chunk, L_chunk, vocab_g]``
    tensor per group. Slot ``(o, g)`` scores frame ``o``'s group ``g`` given the context and every
    earlier token — the exact chain-rule conditional."""
    n_chunk = tgt.shape[0]
    flat_tgt = (tgt + model.decoder.group_offsets).reshape(n_chunk, -1)
    x = model.decoder.hidden(model.decoder.project_cond(cond), flat_tgt[:, :-1])
    x = x.reshape(n_chunk, model.L_chunk, N_GROUPS, -1)
    return [head(x[:, :, g]).float() for g, head in enumerate(model.decoder.heads)]


@jaxtyped(typechecker=beartype)
def chunk_nll(
    logits: list[Tensor], tgt: Int[Tensor, "n_chunk L_chunk n_groups"]
) -> Float[Tensor, "n_chunk L_chunk n_groups"]:
    """Exact per-token NLL (nats) of the teacher-forced chain. Summing over the group axis at offset
    ``o`` gives ``-log p(a_{i+o} | ctx, a_{i+1..i+o-1})``; summing over both axes gives the chunk joint."""
    per_group = [
        F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt[..., g].reshape(-1), reduction="none").reshape(tgt.shape[:2])
        for g, lg in enumerate(logits)
    ]
    return torch.stack(per_group, dim=-1)


def action_loss(model: GPT, batch: TrainBatch, *, gen: torch.Generator | None = None) -> Tensor:
    """One trunk forward + one teacher-forced decoder forward at ``M_pos`` sampled positions per sample.
    Returns ``[n_chunk, L_chunk, n_groups]`` NLL in nats."""
    ctx = batch.context
    h = model(ctx.features, ctx.ctx_pad)
    q_full = _quantized_full(model, batch)
    sel = sample_positions(_valid_positions(ctx, model.L_ctx), model.M_pos, gen)
    tgt, _ = _chunk_targets(q_full, sel, model.L_ctx, model.L_chunk)
    return chunk_nll(chunk_logits(model, _cond_at(h, sel), tgt), tgt)


def objective(nll: Tensor) -> Tensor:
    """Plain unweighted mean per-token NLL (nats). Every one of the ``L_chunk * N_GROUPS`` chain terms
    counts once; there is no transition upweighting and no per-offset weighting to tune."""
    return nll.mean()


# %%
def _btn_support_dead(model: GPT, min_count: int, device: torch.device) -> Tensor:
    """``[N_BUTTON_COMBOS]`` bool mask, True on button combos with fewer than ``min_count`` train
    frames according to the checkpoint's dataset-scoped counts. Cached per ``(min_count, device)`` so
    closed-loop decode builds it once, never per token."""
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


@torch.no_grad()
def ar_sample(
    model: GPT,
    cond: Tensor,
    n_frames: int,
    *,
    group_temps: tuple[float, ...],
    btn_support_min: int,
    min_p: float,
    argmax: bool,
    gen: torch.Generator | None,
) -> Int[Tensor, "n_chunk n_frames n_groups"]:
    """Autoregressively sample the first ``n_frames * N_GROUPS`` tokens of the chunk (time-major, so the
    executed horizon is a token prefix). Decode hygiene is applied per group in the order support-mask ->
    temperature -> min-p -> sample (or ``argmax``). No KV cache: at <= 64 tokens over a 2-layer d_dec
    stack, recomputing the prefix is cheap and is exactly the training-time forward."""
    dec = model.decoder
    cond_d = dec.project_cond(cond)
    dead = _btn_support_dead(model, btn_support_min, cond.device) if btn_support_min >= 1 else None
    prev = torch.empty((cond.shape[0], 0), dtype=torch.long, device=cond.device)
    picks: list[Tensor] = []
    for slot in range(n_frames * N_GROUPS):
        g = slot % N_GROUPS
        logits = dec.heads[g](dec.hidden(cond_d, prev)[:, -1]).float()  # [n_chunk, vocab_g]
        if g == _BUTTONS_G and dead is not None:
            logits = logits.masked_fill(dead, float("-inf"))
        if argmax:
            pick = logits.argmax(-1)
        else:
            probs = F.softmax(logits / group_temps[g], dim=-1)
            if min_p > 0:
                probs = probs * (probs >= min_p * probs.amax(dim=-1, keepdim=True))
                probs = probs / probs.sum(dim=-1, keepdim=True)
            pick = torch.multinomial(probs, 1, generator=gen).squeeze(-1)
        picks.append(pick)
        prev = torch.cat([prev, (pick + _GROUP_OFFSETS[g])[:, None]], dim=1)
    return torch.stack(picks, dim=-1).reshape(cond.shape[0], n_frames, N_GROUPS)


@torch.no_grad()
def decode_chunk(
    model: GPT,
    ctx: Context,
    n_frames: int,
    *,
    temp: float = 1.0,
    temps: tuple[float, float, float, float] | None = None,
    btn_support_min: int = 0,
    min_p: float = 0.0,
    click_trigger_fix: bool = False,
    argmax: bool = False,
    gen: torch.Generator | None = None,
) -> Float[Tensor, "B n_frames d_action"]:
    """One trunk forward from the LAST context position, then AR-sample ``n_frames`` frames of actions in
    raw action ranges. ``click_trigger_fix`` forces trigger_l/r to 1.0 wherever the sampled combo sets the
    digital L/R bit; ``argmax`` (the recon metric's deterministic proxy) ignores temps/min-p but respects
    the support mask."""
    if not 1 <= n_frames <= model.L_chunk:
        raise ValueError(f"n_frames must satisfy 1 <= n <= L_chunk={model.L_chunk}, got {n_frames}")
    group_temps = _resolve_decode_args(temp, temps, btn_support_min, min_p, argmax)
    h = model(ctx.features, ctx.ctx_pad)[:, -1]  # [B, d_model]
    idx = ar_sample(
        model,
        h,
        n_frames,
        group_temps=group_temps,
        btn_support_min=btn_support_min,
        min_p=min_p,
        argmax=argmax,
        gen=gen,
    )
    a = _dequantize(model, idx)  # [B, n_frames, A_DIM]
    if click_trigger_fix:
        a[..., _TRIGGER_L_CH] = torch.where(a[..., _BUTTON_L_CH] > 0.5, 1.0, a[..., _TRIGGER_L_CH])
        a[..., _TRIGGER_R_CH] = torch.where(a[..., _BUTTON_R_CH] > 0.5, 1.0, a[..., _TRIGGER_R_CH])
    return a


# %%
def validate_config(cfg: TrainConfig, *, has_button_combo_counts: bool) -> None:
    """Fail before W&B, loader construction, or Dolphin startup on invalid experiment geometry."""
    positive_ints = {
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "d_dec": cfg.d_dec,
        "n_dec_layers": cfg.n_dec_layers,
        "n_dec_heads": cfg.n_dec_heads,
        "L_chunk": cfg.L_chunk,
        "M_pos": cfg.M_pos,
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
        "diagnostic_rare_button_count": cfg.diagnostic_rare_button_count,
        "eval_n_matchups": cfg.eval_n_matchups,
        "final_eval_n_matchups": cfg.final_eval_n_matchups,
        "eval_max_frames": cfg.eval_max_frames,
        "windows_per_replay": cfg.windows_per_replay,
        "shuffle_block_size": cfg.shuffle_block_size,
        "prefetch_factor": cfg.prefetch_factor,
        "cache_limit_gb": cfg.cache_limit_gb,
    }
    for name, value in positive_ints.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if cfg.d_model % cfg.n_heads != 0:
        raise ValueError(f"d_model={cfg.d_model} must be divisible by n_heads={cfg.n_heads}")
    head_dim = cfg.d_model // cfg.n_heads
    if head_dim % 2:
        raise ValueError(f"rotary attention head_dim=d_model/n_heads={head_dim} must be even")
    if cfg.d_dec % cfg.n_dec_heads != 0:
        raise ValueError(f"d_dec={cfg.d_dec} must be divisible by n_dec_heads={cfg.n_dec_heads}")
    # The chunk horizon is also the train loader's L_chunk, and the val loader is FROZEN at VAL_L_CHUNK:
    # a longer horizon would have target frames the frozen val windows simply do not contain.
    if cfg.L_chunk > VAL_L_CHUNK:
        raise ValueError(f"L_chunk={cfg.L_chunk} exceeds frozen VAL_L_CHUNK={VAL_L_CHUNK}")
    if cfg.M_pos > cfg.L_ctx:
        raise ValueError(f"M_pos={cfg.M_pos} exceeds L_ctx={cfg.L_ctx} (more chunks than context positions)")
    if cfg.exec_horizon > cfg.L_chunk:
        raise ValueError(f"exec_horizon={cfg.exec_horizon} exceeds the decoder horizon L_chunk={cfg.L_chunk}")
    if not math.isfinite(cfg.weight_decay) or cfg.weight_decay < 0:
        raise ValueError(f"weight_decay must be finite and >= 0, got {cfg.weight_decay!r}")
    finite_positive = {
        "muon_lr": cfg.muon_lr,
        "adam_lr": cfg.adam_lr,
        "eval_parallel_per_cpu": cfg.eval_parallel_per_cpu,
        "eval_timeout_seconds": cfg.eval_timeout_seconds,
    }
    for name, value in finite_positive.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    if not math.isfinite(cfg.history_dropout_p) or not 0.0 <= cfg.history_dropout_p <= 1.0:
        raise ValueError(f"history_dropout_p must be in [0, 1], got {cfg.history_dropout_p!r}")
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


# %%
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
    """Fresh closed-loop policy for one eval wave. Replan every ``s`` frames (``exec_horizon``, defaulting
    to ``cfg.exec_horizon``): one trunk forward, then AR-sample exactly the executed prefix's ``4*s``
    tokens. The harness's ``L_chunk`` is therefore ``s`` (we produce only what we execute) with ``d = 0``.
    Each decode-hygiene knob falls back to its ``cfg`` field when the override is ``None`` so an eval can
    A/B without a retrain."""
    s = cfg.exec_horizon if exec_horizon is None else exec_horizon
    if not 1 <= s <= model.L_chunk:
        raise ValueError(f"exec_horizon must satisfy 1 <= s <= L_chunk={model.L_chunk}, got {s}")
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

    @torch.no_grad()
    def predict_chunk(ctx: Context, committed: np.ndarray | None) -> np.ndarray:
        assert committed is None, "receding-horizon policy does not condition on a committed prefix"
        chunk = decode_chunk(
            model,
            ctx,
            s,
            temp=settings.temp,
            temps=settings.temps,
            btn_support_min=settings.btn_support_min,
            min_p=settings.min_p,
            click_trigger_fix=settings.click_trigger_fix,
            gen=gen,
        )
        return chunk.cpu().numpy()

    return RecedingHorizon(
        predict_chunk=predict_chunk,
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
    """Muon for the hidden weight matrices of BOTH transformer stacks (trunk blocks + chunk-decoder
    blocks); AdamW for everything else.

    The decoder's blocks get Muon for the same reason the trunk's do: they are hidden matrices with no
    privileged input or output basis, which is exactly the setting Muon's orthogonalized update assumes.
    Its embeddings (token / time / group), the ``cond_proj`` that reads the trunk's feature basis, the
    ``bos`` vector, and the four vocab heads keep AdamW — those all have a privileged basis on one side.
    Exactly two LRs (``cfg.muon_lr`` / ``cfg.adam_lr``); the partition asserts full coverage so no
    parameter can silently escape an optimizer."""
    muon_params = [p for m in (model.blocks, model.decoder.blocks) for p in m.parameters() if p.ndim >= 2]
    muon_ids = {id(p) for p in muon_params}
    embed_modules = (
        model.cat_embeds,
        model.char_emb,
        model.stage_emb,
        model.decoder.tok_emb,
        model.decoder.time_emb,
        model.decoder.group_emb,
    )
    embed_ids = {id(p) for m in embed_modules for p in m.parameters()}
    # Optionally exclude the decoder's vocab-head weights from weight decay (cfg.head_weight_decay=False).
    head_ids = set() if cfg.head_weight_decay else {id(p) for p in model.decoder.heads.parameters()}

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for p in model.parameters():
        if id(p) in muon_ids:
            continue
        # AdamW: no weight decay on embeddings, 1D params (biases, bos), or the head weights when disabled.
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


# %%
def _bits(x: Tensor) -> float:
    return x.mean().item() / _LN2


def nll_summary(nll: Tensor) -> dict[str, float]:
    """Flat bits/frame summary of a ``[n_chunk, L_chunk, n_groups]`` NLL tensor.

    ``loss`` / ``nll_off1`` is the frame-1 JOINT NLL (the chain rule's four terms summed) — the quantity
    that lines up bit-for-bit with 013's independent-groups offset-1 sum. ``nll_off{o}`` for o > 1 is
    teacher-forced (conditioned on the true intervening frames) and has no 013 counterpart.
    ``nll_chunk`` is the whole 16-frame joint."""
    off = {f"nll_off{o + 1}": _bits(nll[:, o].sum(-1)) for o in range(nll.shape[1])}
    out: dict[str, float] = {
        "loss": off["nll_off1"],
        "nll_chunk": _bits(nll.sum((1, 2))),
        "nll_token": _bits(nll),
        **off,
    }
    # Offset-1 per-group chain terms. Only `buttons` (first in the chain, conditioned on the context
    # alone) is comparable to 013's same-named marginal; the rest are conditionals.
    out.update({f"nll_{name}": _bits(nll[:, 0, g]) for g, name in enumerate(_GROUP_NAMES)})
    return out


@contextlib.contextmanager
def _evaluation_mode(model: nn.Module) -> Iterator[None]:
    """Temporarily enter eval mode and restore the exact prior mode on every exit."""
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def _masked_mean(values: Tensor, mask: Tensor) -> float:
    """Mean of ``values`` over the masked subset; 0.0 when the subset is empty."""
    return values[mask].mean().item() if bool(mask.any()) else 0.0


def _masked_mean_bits(nats: Tensor, mask: Tensor) -> float:
    """Mean of per-position NLL (nats) over the masked subset, in bits; 0.0 when the subset is empty."""
    return _masked_mean(nats, mask) / _LN2


def _group_kl_bits(logits_p: list[Tensor], logits_q: list[Tensor]) -> Tensor:
    """Summed-over-groups KL(p‖q) in bits per position, from two aligned per-group logit lists."""
    total = torch.zeros(logits_p[0].shape[:-1], device=logits_p[0].device)
    for lp, lq in zip(logits_p, logits_q, strict=True):
        logp = F.log_softmax(lp, dim=-1)
        logq = F.log_softmax(lq, dim=-1)
        total = total + (logp.exp() * (logp - logq)).sum(-1)
    return total / _LN2


@torch.no_grad()
def val_metrics(model: GPT, val_cache: list[TrainBatch], cfg: TrainConfig) -> dict[str, float]:
    """Teacher-forced proper-scoring metrics over the cached val batches at a FIXED position draw.

    The position sample is re-seeded from ``cfg.seed`` on every call, so the reported numbers move only
    with the model. ``nll_off{1..L_chunk}`` is the per-offset joint NLL curve; frame 1 additionally drives
    button proper scoring, the transition-vs-hold split, change-event F1, the click=>trigger consistency
    probe and the copycat history-ablation probe. Persistence / change rates are measured ALONG THE CHUNK
    (adjacent offsets), which is the axis the mode-averaging failure shows up on."""
    with _evaluation_mode(model):
        return _val_metrics_eval(model, val_cache, cfg)


def _val_metrics_eval(model: GPT, val_cache: list[TrainBatch], cfg: TrainConfig) -> dict[str, float]:
    nll_cat: list[Tensor] = []
    ablated_nll_cat: list[Tensor] = []
    kl_cat: list[Tensor] = []
    trans_cat: list[Tensor] = []
    pred_change_cat: list[Tensor] = []
    pred_hold_cat: list[Tensor] = []
    btn_probs: list[Tensor] = []
    btn_tgts: list[Tensor] = []
    multipress: list[Tensor] = []
    rare_mass: list[Tensor] = []
    unseen_mass: list[Tensor] = []
    click_l: list[tuple[Tensor, Tensor]] = []
    click_r: list[tuple[Tensor, Tensor]] = []
    counts_available = bool((model.button_combo_counts >= 0).all())
    rare_mask = model.button_combo_counts < cfg.diagnostic_rare_button_count
    unseen_mask = model.button_combo_counts == 0
    combo_bits = scoring.combo_to_buttons(torch.arange(scoring.N_BUTTON_COMBOS, device=model.main_centers.device))
    n_trig = model.trig_centers.shape[0]
    gen = torch.Generator(device=model.main_centers.device).manual_seed(cfg.seed)
    for batch in val_cache:
        ctx = batch.context
        h = model(ctx.features, ctx.ctx_pad)
        # Copycat probe: a second trunk forward with the ego's own controller history zeroed.
        ablated_features = dict(ctx.features)
        for ch in ACTION_CHANNELS:
            ablated_features[f"ego_{ch}"] = torch.zeros_like(ablated_features[f"ego_{ch}"])
        h_ablated = model(ablated_features, ctx.ctx_pad)
        q_full = _quantized_full(model, batch)
        sel = sample_positions(_valid_positions(ctx, model.L_ctx), model.M_pos, gen)
        tgt, prev = _chunk_targets(q_full, sel, model.L_ctx, model.L_chunk)
        logits = chunk_logits(model, _cond_at(h, sel), tgt)
        ablated_logits = chunk_logits(model, _cond_at(h_ablated, sel), tgt)
        nll_cat.append(chunk_nll(logits, tgt))
        ablated_nll_cat.append(chunk_nll(ablated_logits, tgt))
        kl_cat.append(_group_kl_bits([lg[:, 0] for lg in logits], [lg[:, 0] for lg in ablated_logits]))

        # The action each chunk frame is compared against: frame 1 against the context frame's own
        # action, frame o>1 against the (true, teacher-forced) frame o-1.
        prev_full = torch.cat([prev[:, None], tgt[:, :-1]], dim=1)  # [n_chunk, L_chunk, n_groups]
        trans_cat.append(tgt != prev_full)
        argmax_idx = torch.stack([lg.argmax(-1) for lg in logits], dim=-1)  # [n_chunk, L_chunk, n_groups]
        pred_change_cat.append(argmax_idx != prev_full)
        pred_hold_cat.append(argmax_idx[:, 1:] == argmax_idx[:, :-1])

        combo_probs = F.softmax(logits[_BUTTONS_G][:, 0], dim=-1)  # frame-1 button conditional == marginal
        btn_probs.append(combo_probs @ combo_bits.to(combo_probs.dtype))
        tgt_btn = _dequantize(model, tgt[:, 0])[..., _N_CONT:]
        btn_tgts.append(tgt_btn)
        multipress.append((tgt_btn > 0.5).sum(-1) >= 2)
        if counts_available:
            rare_mass.append(combo_probs[:, rare_mask].sum(-1))
            unseen_mass.append(combo_probs[:, unseen_mask].sum(-1))
        # Click => trigger consistency, as a TEACHER-FORCED CONDITIONAL: on frames whose true combo sets
        # the digital L (resp. R) bit, how much probability does the model still put on trigger != 1.0?
        # 013's same-named metric was a product of independent marginals; this one measures the thing the
        # AR factorization is supposed to fix, so the two numbers are not interchangeable.
        trig_probs = F.softmax(logits[_TRIG_G][:, 0], dim=-1).reshape(-1, n_trig, n_trig)
        true_btn = tgt_btn > 0.5
        click_l.append((1.0 - trig_probs[:, -1, :].sum(-1), true_btn[:, _BUTTON_L_CH - _N_CONT]))
        click_r.append((1.0 - trig_probs[:, :, -1].sum(-1), true_btn[:, _BUTTON_R_CH - _N_CONT]))

    nll = torch.cat(nll_cat)
    ablated = torch.cat(ablated_nll_cat)
    trans = torch.cat(trans_cat)
    pred_change = torch.cat(pred_change_cat)
    pred_hold = torch.cat(pred_hold_cat)
    logloss, brier = scoring.bernoulli_scores_from_probs(torch.cat(btn_probs), torch.cat(btn_tgts))
    out = nll_summary(nll)
    out.update(
        {
            "btn_logloss": logloss.item(),
            "btn_brier": brier.item(),
            "btn_multipress": torch.cat(multipress).float().mean().item(),
            "btn_counts_available": float(counts_available),
            "ablate_hist_kl": torch.cat(kl_cat).mean().item(),  # KL(full ‖ history-ablated) at frame 1, bits
        }
    )
    for tag, parts in (("l", click_l), ("r", click_r)):
        invalid = torch.cat([p for p, _ in parts])
        clicked = torch.cat([c for _, c in parts])
        out[f"click_trigger_cond_invalid_{tag}"] = _masked_mean(invalid, clicked)
    out["click_trigger_cond_invalid"] = 0.5 * (
        out["click_trigger_cond_invalid_l"] + out["click_trigger_cond_invalid_r"]
    )
    if counts_available:
        out["btn_rare_mass"] = torch.cat(rare_mass).mean().item()
        out["btn_unseen_mass"] = torch.cat(unseen_mass).mean().item()
        out["btn_rare_count_threshold"] = float(cfg.diagnostic_rare_button_count)
    ablate_total = 0.0
    for g, name in enumerate(_GROUP_NAMES):
        t1 = trans[:, 0, g]
        out[f"nll_{name}_trans"] = _masked_mean_bits(nll[:, 0, g], t1)
        out[f"nll_{name}_hold"] = _masked_mean_bits(nll[:, 0, g], ~t1)
        out[f"trans_rate_{name}"] = trans[..., g].float().mean().item()
        out[f"pred_change_rate_{name}"] = pred_change[..., g].float().mean().item()
        # 1 - (fraction of adjacent chunk frames whose argmax differs): the "holds the last input"
        # number the joint factorization is meant to break.
        out[f"pred_persistence_{name}"] = pred_hold[..., g].float().mean().item()
        # ±1-frame-tolerant change-event F1 ALONG THE CHUNK, and the strict frame-1-only F1. Neither is
        # numerically comparable to 013's (which scores independent per-context-position predictions).
        out[f"changeF1_{name}"] = scoring.change_event_prf(pred_change[..., g], trans[..., g])[2]
        out[f"changeF1_off1_{name}"] = scoring.change_event_prf(pred_change[:, :1, g], trans[:, :1, g])[2]
        d = (ablated[:, 0, g].mean() - nll[:, 0, g].mean()).item() / _LN2  # positive ⇒ history helps
        out[f"ablate_hist_dnll_{name}"] = d
        ablate_total += d
    out["ablate_hist_dnll"] = ablate_total
    return out


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
    """Sample-space reconstruction proxy: AR-decode the next action (s=1, i.e. the chunk's first four
    tokens) and score it vs ground truth. Buttons → acc + F1 @ decode; continuous → MAE."""
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
        pred = decode_chunk(
            model,
            batch.context,
            1,
            temp=temp,
            temps=temps,
            btn_support_min=btn_support_min,
            min_p=min_p,
            click_trigger_fix=click_trigger_fix,
            argmax=argmax,
            gen=gen,
        )
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


# %%
def _slice_context(ctx: Context, n: int) -> Context:
    """First ``n`` rows of a Context (closed-loop-style batch for the decode benchmark)."""
    return Context(features={name: value[:n] for name, value in ctx.features.items()}, ctx_pad=ctx.ctx_pad[:n])


@torch.no_grad()
def bench_decode(
    model: GPT,
    ctx: Context,
    cfg: TrainConfig,
    *,
    horizons: tuple[int, ...],
    n_iters: int,
) -> dict[int, float]:
    """Wall-clock per replan (one trunk forward + ``4*s`` sequential AR token steps) against the real-time
    budget of ``s`` frames at 60 fps. Returns ``{s: ms_per_replan}``."""
    settings = _decode_settings(model, cfg)
    out: dict[int, float] = {}
    with _evaluation_mode(model):
        for s in horizons:
            kwargs = dict(
                temp=settings.temp,
                temps=settings.temps,
                btn_support_min=settings.btn_support_min,
                min_p=settings.min_p,
                click_trigger_fix=settings.click_trigger_fix,
            )
            for _ in range(2):
                decode_chunk(model, ctx, s, **kwargs)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iters):
                decode_chunk(model, ctx, s, **kwargs)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / n_iters * 1000
            out[s] = ms
            budget = _FRAME_MS * s
            print(
                f"[bench] s={s:2d}  slots={ctx.batch:3d}  {ms:7.2f} ms/replan  "
                f"budget {budget:6.1f} ms  ({ms / budget:.2f}x)",
                flush=True,
            )
    return out


# %%
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
        tags=["gpt", "chunked-ar", f"d{cfg.d_model}", f"L{cfg.n_layers}", f"dec{cfg.d_dec}x{cfg.n_dec_layers}"],
        config=asdict(cfg),
    )
    # W&B's own step is a free-running monotonic timestamp; we plot everything against the training
    # step logged as data (``global_step``). This lets an async eval that *finishes* late be logged
    # at its *origin* step without violating step monotonicity.
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")
    ckpt_dir, replay_dir = setup_run_dir(run_name)

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
    n_dec_params = sum(p.numel() for p in model.decoder.parameters())
    if wandb.run is not None:
        wandb.run.summary["model/num_params"] = n_params
        wandb.run.summary["model/num_decoder_params"] = n_dec_params
    print(
        f"[model] {_model_tag(cfg)}  num_params={n_params / 1e6:.2f}M (decoder {n_dec_params / 1e6:.2f}M)",
        flush=True,
    )
    loader_kwargs = dict(
        data_root=cfg.data_root,
        remote=streams.remote_for_local(cfg.data_root),
        cache_limit=f"{cfg.cache_limit_gb}gb",
        shuffle_block_size=cfg.shuffle_block_size,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=cfg.L_chunk,  # the decoder's full horizon; every sampled position needs all H targets
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )
    train_loader = make_loader(
        split="train",
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        windows_per_replay=cfg.windows_per_replay,
        **loader_kwargs,
    )
    # Val uses the FROZEN wider chunk (VAL_L_CHUNK) so its window geometry — hence its NLL — is
    # comparable across experiments regardless of the train-time L_chunk. The val path slices the
    # target back to cfg.L_chunk frames.
    val_loader = make_loader(split=cfg.val_split, num_workers=0, **{**loader_kwargs, "L_chunk": VAL_L_CHUNK})

    opt = make_optimizer(model, cfg)
    sched = LambdaLR(opt, lr_schedule(cfg))
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
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
        f"({sum(b.target.shape[0] for b in val_cache)} samples) in {time.monotonic() - val_t0:.1f}s",
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

    def _val_log_dict() -> dict[str, float]:
        """Flat ``val/*`` metric dict (one W&B section). Merged into the per-step log; no wandb.log here."""
        vm = val_metrics(model, val_cache, cfg)
        gen = torch.Generator(device=DEVICE).manual_seed(cfg.eval_seed)
        settings = _decode_settings(model, cfg)
        decode_kwargs = {
            "temp": settings.temp,
            "temps": settings.temps,
            "btn_support_min": settings.btn_support_min,
            "min_p": settings.min_p,
            "click_trigger_fix": settings.click_trigger_fix,
        }
        recon = {"argmax": recon_metrics(model, val_cache, argmax=True, **decode_kwargs)}
        recon["sample"] = recon_metrics(model, val_cache, argmax=False, gen=gen, **decode_kwargs)
        out = {f"val/{k}": v for k, v in vm.items()}
        for tag, rm in recon.items():
            out[f"val/recon_{tag}_acc"] = rm["recon_button_acc"]
            out[f"val/recon_{tag}_f1"] = rm["recon_button_f1"]
            out[f"val/recon_{tag}_mae"] = rm["recon_cont_mae"]
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
            # with respect to GPU execution.
            _drain_eval(wait=True)

    model.train()
    it = iter(train_loader)
    run_t0 = time.monotonic()
    for step in range(start_step, cfg.max_steps):
        with profile("step") as sw:
            opt.zero_grad()
            nll_acc: list[Tensor] = []
            for _ in range(cfg.grad_accum_steps):
                try:
                    batch = next(it).to(DEVICE)
                except StopIteration:
                    it = iter(train_loader)
                    batch = next(it).to(DEVICE)
                with autocast:
                    nll = action_loss(model, batch)
                    loss = objective(nll) / cfg.grad_accum_steps
                loss.backward()
                nll_acc.append(nll.detach())
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))  # measure only
            opt.step()
            sched.step()
            if DEVICE == "cuda":
                torch.cuda.synchronize()
        summary = nll_summary(torch.cat(nll_acc))
        sps = cfg.batch_size * cfg.grad_accum_steps / sw.elapsed
        samples = (step + 1) * cfg.batch_size * cfg.grad_accum_steps
        log = {
            "global_step": step,
            "samples": samples,
            "tokens": samples * cfg.L_ctx,
            "train/loss": summary["loss"],  # frame-1 joint bits — the 013-comparable number
            "train/nll_chunk": summary["nll_chunk"],  # whole 16-frame joint, bits
            "train/objective": summary["nll_token"],  # the actual backprop objective (mean/token), bits
            **{f"train/nll_{name}": summary[f"nll_{name}"] for name in _GROUP_NAMES},
            "lr/muon": next(g["lr"] for g in opt.param_groups if g["use_muon"]),
            "lr/adam": next(g["lr"] for g in opt.param_groups if not g["use_muon"]),
            "train/gnorm": grad_norm.item(),
            "throughput/step_s": sw.elapsed,
            "throughput/samples_per_s": sps,
        }
        if step < 20 or step % 50 == 0:
            print(
                f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: loss {summary['loss']:.4f} "
                f"chunk {summary['nll_chunk']:.3f} step_dt={sw.elapsed * 1000:.0f}ms ({sps:.1f} samples/s)",
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
    if cfg.eval_every > 0:
        _log_eval(cfg.max_steps, _eval_and_upload("final", n_matchups=cfg.final_eval_n_matchups))
    _save("final.pt", cfg.max_steps)
    uploader.close()


# %%
def _cfg_from_state(saved: dict) -> TrainConfig:
    """Rebuild a ``TrainConfig`` from a checkpoint's saved cfg dict, tolerating schema drift in
    *eval/host* knobs across code versions: keys no longer on ``TrainConfig`` are dropped and new
    fields take their defaults, so past checkpoints still load. Model-identity fields (``d_model``,
    ``d_dec``, ``L_chunk``, …) are unaffected — they're always present and reconstruct exactly."""
    known = {f.name for f in fields(TrainConfig)}
    dropped = sorted(set(saved) - known)
    if dropped:
        print(f"[ckpt] dropping {len(dropped)} stale cfg key(s) not on current TrainConfig: {dropped}", flush=True)
    return TrainConfig(**{k: v for k, v in saved.items() if k in known})


def _load_ckpt(ckpt_path: str) -> tuple[GPT, TrainConfig, dict[str, FeatureStats], dict]:
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = _cfg_from_state(state["cfg"])
    embedded_counts = bool((state["model"]["button_combo_counts"] >= 0).all())
    button_combo_counts = None if embedded_counts else _load_button_combo_counts(cfg)
    validate_config(cfg, has_button_combo_counts=embedded_counts or button_combo_counts is not None)
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(state["model"])
    if button_combo_counts is not None:
        model.button_combo_counts.copy_(button_combo_counts.to(DEVICE))
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
    if not 1 <= exec_horizon <= model.L_chunk:
        raise ValueError(f"exec_horizon must satisfy 1 <= s <= L_chunk={model.L_chunk}, got {exec_horizon}")
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


def run_bench_decode(cfg: TrainConfig, ckpt_path: str | None, n_slots: int, n_iters: int) -> None:
    """Time one replan (trunk forward + AR token loop) against the real-time budget, using a real val
    Context so the trunk sees the production feature set. Weights are irrelevant to the timing, so a
    fresh model is fine when no checkpoint is given."""
    if ckpt_path is None:
        validate_config(cfg, has_button_combo_counts=False)
        stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
        model = GPT(cfg).to(DEVICE)
    else:
        model, cfg, stats, _ = _load_ckpt(ckpt_path)
    loader = make_loader(
        data_root=cfg.data_root,
        split=cfg.val_split,
        remote=streams.remote_for_local(cfg.data_root),
        cache_limit=f"{cfg.cache_limit_gb}gb",
        shuffle_block_size=cfg.shuffle_block_size,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=VAL_L_CHUNK,
        batch_size=max(n_slots, 1),
        seed=cfg.seed,
        num_workers=0,
    )
    batch = next(iter(loader)).to(DEVICE)
    ctx = _slice_context(batch.context, n_slots)
    horizons = tuple(s for s in (1, 2, 4, 8, 16) if s <= cfg.L_chunk)
    print(f"[bench] device={DEVICE} {_model_tag(cfg)} n_iters={n_iters}", flush=True)
    bench_decode(model, ctx, cfg, horizons=horizons, n_iters=n_iters)


# %%
@dataclass
class Args:
    """Top-level CLI surface. Pass TrainConfig fields as kebab-case flags, e.g. ``--cfg.d-model 512``."""

    cfg: TrainConfig = field(default_factory=TrainConfig)
    eval: str | None = None  # ckpt path; closed-loop eval instead of train
    eval_exec_horizon: int | None = None  # override execution horizon s for --eval (1 = per-frame replan)
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
    # decode wall-clock benchmark (no Dolphin): ms per replan vs the 16.6ms * s real-time budget.
    bench_decode: bool = False
    bench_slots: int = 8  # concurrent closed-loop slots batched into one replan
    bench_iters: int = 10
    # internal: one-shot async-eval worker (the trainer spawns this; not for manual use).
    eval_worker: str | None = None  # ckpt path
    eval_worker_step: int = 0
    eval_worker_result: str | None = None
    eval_worker_replay: str | None = None


def main(args: Args) -> None:
    if args.eval_worker is not None:
        assert args.eval_worker_result is not None and args.eval_worker_replay is not None
        run_eval_worker(args.eval_worker, args.eval_worker_step, args.eval_worker_result, args.eval_worker_replay)
        return
    if args.bench_decode:
        run_bench_decode(args.cfg, args.eval, args.bench_slots, args.bench_iters)
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
    auto_comment = f"car-{cfg.max_steps // 1000}k-b{cfg.batch_size}"
    train(cfg, stats, comment=args.comment or auto_comment)


if __name__ == "__main__":
    main(tyro.cli(Args))
