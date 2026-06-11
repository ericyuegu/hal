"""Run-length-event action-chunk policy (FAST-style "compress, then autoregress").

Controller signals in Melee are piecewise-constant: after quantization a 16-frame chunk
holds ~3.5 state changes, median button holds 5-7 frames. 008 tests the FAST hypothesis
("compress the chunk, then autoregress over the compressed stream") with the codec that
actually fits this signal family -- **run-length encoding of quantized joint controller
states (chords)** -- instead of DCT (which Gibbs-rings on step functions). A 16-frame chunk
becomes a short event stream: typically ~4-5 (chord, duration) events ~ 8-10 tokens vs 16
per-frame tokens; one long hold is one decision instead of 6+ repeated ones.

**Chord** = the per-frame joint quantized state, packed
``(((main*N_C + c)*N_TRIG + tl)*N_TRIG + tr)*N_COMBO + btn`` over the ``scoring`` discretizers
(65 main clusters x 9 c-stick x 5 trigger_l x 5 trigger_r x 256 button combos). The product
space is ~3.74M but only ~8.2k distinct chords occur in human data (~5.4k cover 99.9%), so we
learn a **data-derived chord vocab**: scan the (seeded, deterministic) train loader at startup,
harvest chords from full windows (ego controller history + target frames) up to a frame budget,
keep the top-``chord_vocab_size`` by descending frequency. The vocab persists in the run dir AND
inside the checkpoint (the ``vocab_chords`` buffer), so resume/eval load it, never rebuild.
Out-of-vocab chords map deterministically to the nearest in-vocab chord (same ``(btn,tl,tr)``
nearest main then c; else same ``btn`` nearest sticks; else the most-frequent chord).

**RLE codec (bijective, per chunk)**: quantize the ``L_chunk`` target frames to chord ids, run
them into maximal runs ``[(chord_1, d_1), ..., (chord_m, d_m)]`` with ``sum d_i = L_chunk`` and
``chord_i != chord_{i+1}`` (the first run's chord is encoded explicitly, no delta coding). The
token stream is ``chord_1, dur_1, chord_2, dur_2, ...`` of length ``2m <= 2*L_chunk``, padded to
``2*L_chunk`` with the loss masked on pads; durations are a ``L_chunk``-way categorical (1..L).

*Bijectivity*: given the horizon ``L_chunk``, (maximal runs) <-> (frame chord sequence) is a
bijection -- the durations sum to exactly ``L_chunk`` and consecutive chords differ, so the runs
are uniquely recoverable from the frames and vice versa. Hence the chunk NLL in bits is exactly
comparable to a per-frame chord AR model's chunk NLL: same distribution over frame sequences, a
different factorization, up to the shared quantizer/OOV mapping. The constraint
``chord_{i+1} != chord_i`` means the V-way softmax CAN place mass on a repeated chord; that is
modeling slack (it wastes probability), not a codec defect -- renormalizing the chord head over
the non-repeat support is left as future work and deliberately not implemented here.

**Backbone**: identical causal Transformer to 003/005 (``encode_context``), optionally
conditioned on per-player character + global stage (``cond_char_stage``). **Head**: a causal
Transformer decoder over the ``2*L_chunk`` token positions. Input at position ``j`` is the
embedding of the previous realized token (chord-vocab or duration table; learned BOS at j=0) plus
a token-type embedding (chord-slot vs dur-slot), an event-index embedding, and the projected
backbone cond (broadcast). Two output heads at their slots: V-way chord logits at chord slots,
``L_chunk``-way duration logits at dur slots.

Decode alternates sampling chord then duration (``cfg.decode`` / ``cfg.decode_temp``, default
``sample`` @ 1.0 -- argmax collapses an AR policy to a do-nothing fixed point in closed loop, as
in 005) until cumulative duration reaches ``L_chunk``, then expands the runs to ``L_chunk`` chord
frames and dequantizes to a ``[B, L_chunk, 14]`` action vec. Closed loop is the same
``RecedingHorizon`` driver as 005.

``cfg.opp_controller`` is a roofline cheat: concat the opponent's 14-channel controller history
onto the context tokens to measure headroom a human can't have (we can't see the opponent's
controller). Teacher-forced only -- the closed-loop driver does not inject opp controller history.

Run:
    python experiments/008_rle_events.py                                  # train (closed-loop eval samples)
    python experiments/008_rle_events.py --eval <ckpt>                    # eval at the trained decode
    python experiments/008_rle_events.py --eval <ckpt> --eval-temp 0.7    # test-time temperature sweep
    python experiments/008_rle_events.py --val-report <ckpt>              # teacher-forced val fit (no emulator)
"""

# %%
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import contextlib
import itertools
import math
import time
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
_LN2 = math.log(2.0)

# Channel split inside the A_DIM=14 action vec: [0:6] sticks+triggers (continuous), [6:14] buttons {0,1}.
_N_CONT = 6
_N_BUTTONS = A_DIM - _N_CONT

# Chord component cardinalities (the scoring discretizers are the single source of truth).
N_MAIN = scoring.STICK_CLUSTER_CENTERS_MAIN.shape[0]  # 65
N_C = scoring.STICK_CLUSTER_CENTERS_C.shape[0]  # 9
N_TRIG = scoring.TRIGGER_CENTERS.shape[0]  # 5
N_COMBO = scoring.N_BUTTON_COMBOS  # 256
N_CHORD = N_MAIN * N_C * N_TRIG * N_TRIG * N_COMBO  # ~3.74M product space


# %%
@dataclass
class TrainConfig:
    # model backbone (identical to 003/005)
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    # causal event-decoder head (small Transformer over the 2*L_chunk token positions)
    d_head: int = 128
    n_head_layers: int = 2
    head_heads: int = 4
    head_ff: int = 512
    # matchup conditioning (schema v4): per-player character + global stage embeddings (see 005).
    cond_char_stage: bool = True
    char_vocab: int = 64
    char_dim: int = 12
    stage_vocab: int = 64
    stage_dim: int = 8
    # roofline cheat: concat the opp's 14-channel controller history onto context tokens. A
    # deliberate information cheat to measure headroom (a human cannot see the opp controller).
    # Teacher-forced only -- the closed-loop driver does not provide opp controller history.
    opp_controller: bool = False
    # data-derived chord vocab
    chord_vocab_size: int = 8192  # cap on the kept chords (by descending train frequency)
    chord_vocab_frames: int = 5_000_000  # frame budget scanned at startup to harvest chord counts
    # decode (closed-loop controller + recon proxy). Sampling is the DEFAULT: argmax greedily
    # picks the mode, which for an AR policy collapses to a "do nothing" fixed point in closed
    # loop (see 005's collapse rationale). argmax stays for the deterministic recon metric.
    decode: str = "sample"  # "sample" (temperature) | "argmax" (greedy)
    decode_temp: float = 1.0
    norm_div: float = 1.0  # training-stability divisor on the joint-NLL scalar ONLY; never on reported bits
    seed: int = 0
    # window / chunking
    L_ctx: int = 256
    L_chunk: int = 16
    train_positions: int = 64
    # optimization
    batch_size: int = 32
    grad_accum_steps: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 2**15
    amp_dtype: str = "bfloat16"
    allow_tf32: bool = True
    # eval cadence
    val_every: int = 1024
    val_n_batches: int = 16
    eval_every: int = 2048
    eval_max_frames: int = 7200
    eval_replicas: int = 16
    eval_max_parallel: int = 8
    # checkpointing
    ckpt_every: int = 2048
    push_to_r2: bool = True
    # data (v4 MDS carries the stage + p{1,2}_character columns cond_char_stage needs)
    data_root: str = "data/processed/ranked-anonymized-1/mds"
    cache_limit_gb: int = 440
    shuffle_block_size: int = 2000
    val_split: str = "val"
    num_workers: int = 8
    prefetch_factor: int = 4


def _model_tag(cfg: TrainConfig, vocab_size: int) -> str:
    cs = "-cs" if cfg.cond_char_stage else ""
    oc = "-oppc" if cfg.opp_controller else ""
    return (
        f"rle-V{vocab_size}{cs}{oc}"
        f"-d{cfg.d_model}-L{cfg.n_layers}-Lc{cfg.L_ctx}-Lk{cfg.L_chunk}-tp{cfg.train_positions}"
    )


# %%
# --- chord <-> action quantizer (vocab-independent; the shared lossy step) -------------------
@jaxtyped(typechecker=beartype)
def quantize_to_chord(
    action: Float[Tensor, "*batch d_action"],
    main_centers: Float[Tensor, "n_main two"],
    c_centers: Float[Tensor, "n_c two"],
    trig_centers: Float[Tensor, " n_trig"],
) -> Int[Tensor, "*batch"]:
    """Nearest-cluster quantize a 14-dim action vec to a product chord id in ``[0, N_CHORD)``."""
    cont, btn = action[..., :_N_CONT], action[..., _N_CONT:]
    main = scoring.nearest_cluster(cont[..., 0:2], main_centers)
    c = scoring.nearest_cluster(cont[..., 2:4], c_centers)
    trig = scoring.nearest_center(cont[..., 4:6], trig_centers)  # [..., 2]
    tl, tr = trig[..., 0], trig[..., 1]
    combo = scoring.buttons_to_combo(btn)
    return (((main * N_C + c) * N_TRIG + tl) * N_TRIG + tr) * N_COMBO + combo


@jaxtyped(typechecker=beartype)
def chord_to_action(
    chord: Int[Tensor, "*batch"],
    main_centers: Float[Tensor, "n_main two"],
    c_centers: Float[Tensor, "n_c two"],
    trig_centers: Float[Tensor, " n_trig"],
) -> Float[Tensor, "*batch d_action"]:
    """Inverse of ``quantize_to_chord``: product chord id -> dequantized 14-dim action vec."""
    btn = chord % N_COMBO
    r = chord // N_COMBO
    tr = r % N_TRIG
    r = r // N_TRIG
    tl = r % N_TRIG
    r = r // N_TRIG
    c = r % N_C
    main = r // N_C
    main_xy = scoring.cluster_to_xy(main, main_centers)
    c_xy = scoring.cluster_to_xy(c, c_centers)
    tl_v = scoring.center_to_value(tl, trig_centers)[..., None]
    tr_v = scoring.center_to_value(tr, trig_centers)[..., None]
    btn_bits = scoring.combo_to_buttons(btn)
    return torch.cat([main_xy, c_xy, tl_v, tr_v, btn_bits], dim=-1)


def decompose_chords(chords: Tensor, main_centers: Tensor, c_centers: Tensor) -> dict[str, Tensor]:
    """Split product chord ids ``[V]`` into the component tensors the OOV nearest-neighbour and
    the first-event button marginal need."""
    btn = chords % N_COMBO
    r = chords // N_COMBO
    tr = r % N_TRIG
    r = r // N_TRIG
    tl = r % N_TRIG
    r = r // N_TRIG
    c = r % N_C
    main = r // N_C
    return {
        "btn": btn,
        "tl": tl,
        "tr": tr,
        "c": c,
        "main": main,
        "main_xy": main_centers[main],
        "c_xy": c_centers[c],
        "button_bits": scoring.combo_to_buttons(btn),
    }


# --- chord vocab build + OOV mapping ----------------------------------------------------------
def build_chord_vocab(counts: Int[Tensor, " n_chord"], max_size: int) -> Int[Tensor, " vocab"]:
    """Top-``max_size`` product chord ids by descending count (stable -> ties by ascending id,
    so the vocab is reproducible). Drops never-seen chords; fails loud on an empty harvest."""
    n_obs = int((counts > 0).sum())
    if n_obs == 0:
        raise ValueError("chord harvest saw zero chords")
    order = torch.argsort(counts, descending=True, stable=True)
    return order[: min(n_obs, max_size)].contiguous().to(torch.int64)


@torch.no_grad()
def harvest_chord_counts(
    loader, *, budget: int, main_centers: Tensor, c_centers: Tensor, trig_centers: Tensor, device: str
) -> Int[Tensor, " n_chord"]:
    """Scan ``loader`` (seeded -> deterministic) until ``budget`` frames are seen, counting product
    chords over full windows (ego controller history + target frames)."""
    counts = torch.zeros(N_CHORD, dtype=torch.int64, device=device)
    seen = 0
    for batch in loader:
        b = batch.to(device)
        frames = torch.cat([stack_actions(b.context.features), b.target], dim=1)  # [B, L_ctx+L_chunk, 14]
        chords = quantize_to_chord(frames, main_centers, c_centers, trig_centers).reshape(-1)
        counts += torch.bincount(chords, minlength=N_CHORD)
        seen += chords.numel()
        if seen >= budget:
            break
    if seen == 0:
        raise RuntimeError("chord harvest loader yielded zero frames")
    return counts


def _resolve_oov(
    query: Int[Tensor, " q"],
    vocab: dict[str, Tensor],
    main_centers: Tensor,
    c_centers: Tensor,
) -> Int[Tensor, " q"]:
    """Deterministic nearest in-vocab chord for OOV product ids. Tier 1: same ``(btn,tl,tr)``,
    nearest main (then c as a small tiebreak). Tier 2: same ``btn``, nearest sticks. Tier 3: the
    most-frequent chord (vocab index 0, since the vocab is freq-sorted)."""
    q = decompose_chords(query, main_centers, c_centers)
    main_dist = (q["main_xy"][:, None, :] - vocab["main_xy"][None]).pow(2).sum(-1)  # [Q, V]
    c_dist = (q["c_xy"][:, None, :] - vocab["c_xy"][None]).pow(2).sum(-1)  # [Q, V]
    same_btr = (
        (q["btn"][:, None] == vocab["btn"][None])
        & (q["tl"][:, None] == vocab["tl"][None])
        & (q["tr"][:, None] == vocab["tr"][None])
    )
    same_btn = q["btn"][:, None] == vocab["btn"][None]
    inf = torch.full_like(main_dist, float("inf"))
    cost1 = torch.where(same_btr, main_dist + 1e-4 * c_dist, inf)
    cost2 = torch.where(same_btn, main_dist + c_dist, inf)
    best1, has1 = cost1.argmin(1), same_btr.any(1)
    best2, has2 = cost2.argmin(1), same_btn.any(1)
    zero = torch.zeros_like(best1)
    return torch.where(has1, best1, torch.where(has2, best2, zero))


@jaxtyped(typechecker=beartype)
def chord_to_vocab_index(
    product: Int[Tensor, "*batch"],
    vocab_chords: Int[Tensor, " vocab"],
    vocab_sorted: Int[Tensor, " vocab"],
    vocab_sort_perm: Int[Tensor, " vocab"],
    vocab: dict[str, Tensor],
    main_centers: Tensor,
    c_centers: Tensor,
) -> tuple[Int[Tensor, "*batch"], Tensor]:
    """Map product chord ids to compact vocab indices. Exact match via searchsorted on the sorted
    vocab; OOV ids resolved by ``_resolve_oov``. Returns ``(vocab_idx, oov_mask)`` (both batch-shaped)."""
    shape = product.shape
    q = product.reshape(-1)
    pos = torch.searchsorted(vocab_sorted, q).clamp(max=vocab_sorted.shape[0] - 1)
    found = vocab_sorted[pos] == q
    idx = vocab_sort_perm[pos]
    oov = ~found
    if bool(oov.any()):
        idx = idx.clone()
        idx[oov] = _resolve_oov(q[oov], vocab, main_centers, c_centers)
    return idx.reshape(shape), oov.reshape(shape)


# --- bijective RLE codec over chord-id frame sequences ----------------------------------------
@jaxtyped(typechecker=beartype)
def rle_encode(
    chord_frames: Int[Tensor, "N L_chunk"], L_chunk: int
) -> tuple[Int[Tensor, "N L_chunk"], Int[Tensor, "N L_chunk"], Tensor]:
    """Maximal run-length encode each row of per-frame chord ids. Returns
    ``(chord_of_event, dur_index, event_valid)`` each ``[N, L_chunk]``: ``chord_of_event[:, e]``
    is event ``e``'s chord, ``dur_index[:, e]`` its (duration-1) in ``[0, L_chunk)``, and
    ``event_valid[:, e]`` marks the ``m`` real events (the rest are zero pad)."""
    N, L = chord_frames.shape
    change = torch.ones(N, L, dtype=torch.bool, device=chord_frames.device)
    change[:, 1:] = chord_frames[:, 1:] != chord_frames[:, :-1]
    event_of_frame = change.long().cumsum(1) - 1  # [N, L] in [0, m)
    m = change.sum(1)  # [N]
    chord_of_event = torch.zeros_like(chord_frames).scatter_(1, event_of_frame, chord_frames)
    duration = torch.zeros_like(chord_frames).scatter_add_(1, event_of_frame, torch.ones_like(chord_frames))
    dur_index = (duration - 1).clamp(min=0)  # pad events have duration 0 -> 0; masked out
    event_valid = torch.arange(L, device=chord_frames.device)[None, :] < m[:, None]
    return chord_of_event, dur_index, event_valid


@jaxtyped(typechecker=beartype)
def rle_decode(
    chord_of_event: Int[Tensor, "N L_chunk"], dur_index: Int[Tensor, "N L_chunk"], L_chunk: int
) -> Int[Tensor, "N L_chunk"]:
    """Expand runs back to ``L_chunk`` per-frame chord ids; the exact inverse of ``rle_encode``
    (durations sum to L over the real events, pad events get pushed past the horizon and are never
    referenced). Decode-time overshoot of the last sampled run is truncated by the horizon."""
    duration = dur_index + 1  # >= 1, so cum_end is strictly increasing -> searchsorted-able
    cum_end = duration.cumsum(1)
    frames = torch.arange(L_chunk, device=chord_of_event.device)[None, :].expand(chord_of_event.shape[0], L_chunk)
    event_of_frame = torch.searchsorted(cum_end, frames, right=True).clamp(max=L_chunk - 1)
    return torch.gather(chord_of_event, 1, event_of_frame)


# %%
class RLEHead(nn.Module):
    """Causal Transformer decoder over the ``2*L_chunk`` event-token positions.

    Token sequence per chunk is ``chord_0, dur_0, chord_1, dur_1, ...``. The input at position
    ``j`` is the embedding of the previously realized token (``chord_embed`` for the chord at an
    even predecessor, ``dur_embed`` for the duration at an odd predecessor; a learned BOS at j=0)
    plus a token-type embedding (chord-slot vs dur-slot), an event-index embedding, and the
    projected backbone cond broadcast over positions. Under the causal mask, the chord head reads
    even positions (V-way) and the duration head reads odd positions (``L_chunk``-way)."""

    def __init__(self, cfg: TrainConfig, vocab_size: int):
        super().__init__()
        self.L_chunk = cfg.L_chunk
        dh = cfg.d_head
        Lt = 2 * cfg.L_chunk
        self.cond_proj = nn.Linear(cfg.d_model, dh)
        self.chord_embed = nn.Embedding(vocab_size, dh)
        self.dur_embed = nn.Embedding(cfg.L_chunk, dh)
        self.bos = nn.Parameter(torch.zeros(dh))
        self.type_embed = nn.Embedding(2, dh)  # 0 = chord slot, 1 = dur slot
        self.event_embed = nn.Embedding(cfg.L_chunk, dh)
        layer = nn.TransformerEncoderLayer(
            d_model=dh,
            nhead=cfg.head_heads,
            dim_feedforward=cfg.head_ff,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=cfg.n_head_layers, enable_nested_tensor=False)
        self.chord_out = nn.Linear(dh, vocab_size)
        self.dur_out = nn.Linear(dh, cfg.L_chunk)
        self.register_buffer(
            "causal_mask", torch.triu(torch.ones(Lt, Lt, dtype=torch.bool), diagonal=1), persistent=False
        )
        self.register_buffer("type_idx", torch.arange(Lt) % 2, persistent=False)
        self.register_buffer("event_idx", torch.arange(Lt) // 2, persistent=False)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        cond: Float[Tensor, "N d_model"],
        chord_of_event: Int[Tensor, "N L_chunk"],
        dur_index: Int[Tensor, "N L_chunk"],
    ) -> tuple[Float[Tensor, "N L_chunk V_chord"], Float[Tensor, "N L_chunk L_dur"]]:
        N, L = chord_of_event.shape
        dh = self.bos.shape[0]
        chord_in = self.chord_embed(chord_of_event)  # [N, L, dh] -> the L odd input slots
        dur_in = self.dur_embed(dur_index)  # [N, L, dh] -> the L-1 even input slots (>= position 2)
        emb = torch.zeros(N, 2 * L, dh, dtype=chord_in.dtype, device=cond.device)
        emb[:, 0] = self.bos
        emb[:, 1::2] = chord_in
        emb[:, 2::2] = dur_in[:, :-1]
        emb = (
            emb + self.type_embed(self.type_idx) + self.event_embed(self.event_idx) + self.cond_proj(cond).unsqueeze(1)
        )
        h = self.decoder(emb, mask=self.causal_mask)
        return self.chord_out(h[:, 0::2]), self.dur_out(h[:, 1::2])


# %%
class RLEEventsPolicy(nn.Module):
    """Causal backbone (verbatim from 003/005) + the RLE event-decoder head, sharing one
    data-derived chord vocab. ``vocab_chords`` is the persisted product chord id list; the
    derived component buffers (sorted index, decomposed parts, cluster xy, button bits) are
    recomputed from it at construction so only the list itself rides in the checkpoint."""

    def __init__(self, cfg: TrainConfig, vocab_chords: Tensor):
        super().__init__()
        self.L_ctx = cfg.L_ctx
        self.L_chunk = cfg.L_chunk
        self.n_heads = cfg.n_heads
        self.cond_char_stage = cfg.cond_char_stage
        self.opp_controller = cfg.opp_controller

        if cfg.decode not in ("argmax", "sample"):
            raise ValueError(f"decode must be argmax|sample, got {cfg.decode!r}")
        if not cfg.decode_temp > 0:
            raise ValueError(f"decode_temp must be > 0, got {cfg.decode_temp}")
        if not cfg.norm_div > 0:
            raise ValueError(f"norm_div must be > 0, got {cfg.norm_div}")
        if vocab_chords.ndim != 1 or vocab_chords.shape[0] == 0:
            raise ValueError(f"vocab_chords must be a non-empty 1D tensor, got shape {tuple(vocab_chords.shape)}")

        self.cat_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocab, dim) for name, (vocab, dim) in CAT_FEATURES.items()}
        )
        n_float = len(FLOAT_FEATURES)
        n_mask = len(FLOAT_FEATURES)
        n_cat = sum(dim for _, dim in CAT_FEATURES.values())
        per_player_dim = n_float + n_mask + n_cat
        per_frame_in_dim = 2 * per_player_dim + A_DIM  # ego + opp gamestate + ego controller history
        if cfg.opp_controller:
            per_frame_in_dim += A_DIM
        if cfg.cond_char_stage:
            self.char_emb = nn.Embedding(cfg.char_vocab, cfg.char_dim)
            self.stage_emb = nn.Embedding(cfg.stage_vocab, cfg.stage_dim)
            per_frame_in_dim += 2 * cfg.char_dim + cfg.stage_dim

        # --- causal backbone ---
        self.ctx_proj = nn.Linear(per_frame_in_dim, cfg.d_model)
        self.pos_emb = nn.Embedding(self.L_ctx, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers, enable_nested_tensor=False)

        # --- chord discretizer + vocab buffers ---
        self.register_buffer("main_centers", scoring.STICK_CLUSTER_CENTERS_MAIN.clone(), persistent=False)
        self.register_buffer("c_centers", scoring.STICK_CLUSTER_CENTERS_C.clone(), persistent=False)
        self.register_buffer("trig_centers", scoring.TRIGGER_CENTERS.clone(), persistent=False)
        self.register_buffer("vocab_chords", vocab_chords.to(torch.int64))  # persisted in the checkpoint
        sorted_chords, sort_perm = torch.sort(self.vocab_chords)
        self.register_buffer("vocab_sorted", sorted_chords, persistent=False)
        self.register_buffer("vocab_sort_perm", sort_perm.to(torch.int64), persistent=False)
        parts = decompose_chords(self.vocab_chords, self.main_centers, self.c_centers)
        for k, v in parts.items():
            self.register_buffer(f"vocab_{k}", v, persistent=False)

        self.head = RLEHead(cfg, self.vocab_chords.shape[0])

    @property
    def vocab_size(self) -> int:
        return self.vocab_chords.shape[0]

    def _vocab_parts(self) -> dict[str, Tensor]:
        return {
            k: getattr(self, f"vocab_{k}") for k in ("btn", "tl", "tr", "c", "main", "main_xy", "c_xy", "button_bits")
        }

    def to_vocab_index(self, product: Tensor) -> tuple[Tensor, Tensor]:
        return chord_to_vocab_index(
            product,
            self.vocab_chords,
            self.vocab_sorted,
            self.vocab_sort_perm,
            self._vocab_parts(),
            self.main_centers,
            self.c_centers,
        )

    # --- backbone (verbatim from 003/005, plus the optional opp-controller cheat) ---
    def _per_player_features(self, features: dict[str, Tensor], prefix: str) -> Tensor:
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
        return torch.cat([features[f"ego_{ch}"][..., None] for ch in ACTION_CHANNELS], dim=-1)

    def _opp_history_features(self, features: dict[str, Tensor]) -> Tensor:
        return torch.cat([features[f"opp_{ch}"][..., None] for ch in ACTION_CHANNELS], dim=-1)

    def _context_tokens(self, features: dict[str, Tensor]) -> Float[Tensor, "B L_ctx d_model"]:
        ego = self._per_player_features(features, "ego")
        opp = self._per_player_features(features, "opp")
        parts = [ego, opp, self._ego_history_features(features)]
        if self.opp_controller:
            parts.append(self._opp_history_features(features))
        if self.cond_char_stage:
            ec = self.char_emb(features["ego_character"].clamp(0, self.char_emb.num_embeddings - 1))
            oc = self.char_emb(features["opp_character"].clamp(0, self.char_emb.num_embeddings - 1))
            st = self.stage_emb(features["stage"].clamp(0, self.stage_emb.num_embeddings - 1))
            parts += [ec, oc, st]
        return self.ctx_proj(torch.cat(parts, dim=-1))

    def _backbone_mask(self, ctx_pad: Int[Tensor, " B"], T: int, device: torch.device) -> Tensor:
        idx = torch.arange(T, device=device)
        causal = idx[:, None] >= idx[None, :]
        real_key = idx[None, :] >= ctx_pad[:, None]
        diag = torch.eye(T, dtype=torch.bool, device=device)
        block = ~(causal[None] & (real_key[:, None, :] | diag[None]))
        B = ctx_pad.shape[0]
        return block[:, None].expand(B, self.n_heads, T, T).reshape(B * self.n_heads, T, T)

    def encode_context(self, ctx: Context) -> Float[Tensor, "B L_ctx d_model"]:
        tok = self._context_tokens(ctx.features)
        T = tok.size(1)
        tok = tok + self.pos_emb.weight[None, :T, :]
        mask = self._backbone_mask(ctx.ctx_pad, T, tok.device)
        return self.encoder(tok, mask=mask)


# %%
def _position_targets(ctx: Context, target: Tensor, H: int) -> tuple[Tensor, Tensor]:
    """For every context position ``i``, the next-H-action target chunk + a validity mask
    (verbatim from 003/005)."""
    a_full = torch.cat([stack_actions(ctx.features), target], dim=1)
    T = a_full.size(1) - H
    tgt = a_full.unfold(1, H, 1)[:, 1:].permute(0, 1, 3, 2).contiguous()
    pos = torch.arange(T, device=a_full.device)
    valid = pos[None, :] >= ctx.ctx_pad[:, None]
    return tgt, valid


def _select_cond_tgt(
    model: RLEEventsPolicy,
    batch: TrainBatch,
    *,
    multi: bool,
    max_positions: int = -1,
    gen: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Pick the supervised context positions (verbatim selection from 003/005) and return
    ``(cond [N, d_model], tgt [N, L_chunk, A_DIM])``. ``multi=False`` keeps only the last
    position (inference-matched; used by val)."""
    ctx = batch.context
    H = model.L_chunk
    hidden = model.encode_context(ctx)
    B, T, _ = hidden.shape
    tgt, valid = _position_targets(ctx, batch.target, H)
    if not multi:
        valid = valid & (torch.arange(T, device=hidden.device)[None, :] == T - 1)
    elif 0 < max_positions < T:
        scores = torch.rand(B, T, device=hidden.device, generator=gen).masked_fill(~valid, -1.0)
        keep = torch.zeros_like(valid).scatter_(1, scores.topk(max_positions, dim=1).indices, True)
        valid = valid & keep
    sel = valid.reshape(B * T)
    cond = hidden.reshape(B * T, -1)[sel]
    tgt = tgt.reshape(B * T, H, A_DIM)[sel]
    return cond, tgt


def _forward_supervised(
    model: RLEEventsPolicy,
    batch: TrainBatch,
    *,
    multi: bool = True,
    max_positions: int = -1,
    gen: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """Teacher-forced forward over the supervised positions: select cond/tgt, RLE-encode each
    target chunk, and run the head once. Returns the logits + encoded targets + masks."""
    cond, tgt = _select_cond_tgt(model, batch, multi=multi, max_positions=max_positions, gen=gen)
    product = quantize_to_chord(tgt, model.main_centers, model.c_centers, model.trig_centers)  # [N, L]
    vocab_frames, oov_mask = model.to_vocab_index(product)
    chord_of_event, dur_index, event_valid = rle_encode(vocab_frames, model.L_chunk)
    chord_logits, dur_logits = model.head(cond, chord_of_event, dur_index)
    return {
        "chord_logits": chord_logits,
        "dur_logits": dur_logits,
        "chord_tgt": chord_of_event,
        "dur_tgt": dur_index,
        "event_valid": event_valid,
        "oov_mask": oov_mask,
        "tgt": tgt,
    }


def _component_nats(out: dict[str, Tensor]) -> dict[str, Tensor]:
    """Masked per-position chord/duration NLL (nats, ``[N]`` -- summed over a chunk's events).
    Same code at train and val so the two halves of the stream NLL partition cleanly."""
    valid = out["event_valid"].float()
    chord_ce = _ce_nats(out["chord_logits"], out["chord_tgt"]) * valid  # [N, L]
    dur_ce = _ce_nats(out["dur_logits"], out["dur_tgt"]) * valid  # [N, L]
    return {"chord": chord_ce.sum(1), "duration": dur_ce.sum(1)}


def _ce_nats(logits: Tensor, idx: Tensor) -> Tensor:
    """Per-element categorical cross-entropy in nats: ``logits [..., K]``, ``idx [...]`` -> ``[...]``."""
    flat = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), idx.reshape(-1), reduction="none")
    return flat.reshape(idx.shape)


def action_loss(
    model: RLEEventsPolicy,
    batch: TrainBatch,
    *,
    multi: bool = True,
    max_positions: int = -1,
    gen: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """Per-position ``{chord, duration}`` stream NLL (nats, ``[N]``). Sum and ``.mean()`` for the
    training scalar."""
    out = _forward_supervised(model, batch, multi=multi, max_positions=max_positions, gen=gen)
    return _component_nats(out)


@torch.no_grad()
def decode(
    model: RLEEventsPolicy,
    ctx: Context,
    *,
    mode: str = "argmax",
    temp: float = 1.0,
    gen: torch.Generator | None = None,
) -> Float[Tensor, "B L_chunk d_action"]:
    """One action chunk per sample from the LAST context position, in raw action ranges.

    Alternately samples chord then duration from the causal event decoder until every sample's
    cumulative duration reaches ``L_chunk`` (``sample`` draws the ``temp``-scaled distribution --
    the default; ``argmax`` is the deterministic recon mode, which collapses an AR policy to a
    do-nothing fixed point in closed loop, see 005). The runs expand to ``L_chunk`` chord frames
    (overshoot truncated by the horizon) and dequantize to the action vec."""
    if mode not in ("argmax", "sample"):
        raise ValueError(f"decode mode must be argmax|sample, got {mode!r}")
    cond = model.encode_context(ctx)[:, -1, :]  # [B, d_model]
    B, L = cond.shape[0], model.L_chunk
    device = cond.device
    chord_of_event = torch.zeros(B, L, dtype=torch.int64, device=device)
    dur_index = torch.zeros(B, L, dtype=torch.int64, device=device)
    cum = torch.zeros(B, dtype=torch.int64, device=device)

    def pick(logits: Tensor) -> Tensor:
        if mode == "argmax":
            return logits.argmax(-1)
        probs = F.softmax(logits / temp, dim=-1)
        return torch.multinomial(probs, 1, generator=gen).squeeze(-1)

    for e in range(L):
        chord_logits, _ = model.head(cond, chord_of_event, dur_index)
        chord_of_event[:, e] = pick(chord_logits[:, e])
        _, dur_logits = model.head(cond, chord_of_event, dur_index)
        d = pick(dur_logits[:, e])  # [B] in [0, L)
        dur_index[:, e] = d
        cum = cum + d + 1
        if bool((cum >= L).all()):
            break
    frames_vocab = rle_decode(chord_of_event, dur_index, L)  # [B, L]
    product = model.vocab_chords[frames_vocab]
    return chord_to_action(product, model.main_centers, model.c_centers, model.trig_centers)


def make_policy(
    model: RLEEventsPolicy,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    device: str = DEVICE,
    decode_mode: str | None = None,
    decode_temp: float | None = None,
    s: int | None = None,
) -> RecedingHorizon:
    """Fresh open-loop ``RecedingHorizon`` policy for one eval wave (see 005). ``decode_mode`` /
    ``decode_temp`` / ``s`` override ``cfg`` for test-time sweeps without retraining."""
    mode = decode_mode or cfg.decode
    temp = cfg.decode_temp if decode_temp is None else decode_temp

    @torch.no_grad()
    def predict_chunk(ctx: Context, committed: np.ndarray | None) -> np.ndarray:
        assert committed is None, "open-loop policy does not condition on a committed prefix"
        return decode(model, ctx, mode=mode, temp=temp).cpu().numpy()

    return RecedingHorizon(
        predict_chunk=predict_chunk,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=cfg.L_chunk,
        s=1 if s is None else s,
        d=0,
        device=device,
    )


# %%
def lr_schedule(cfg: TrainConfig):
    """Linear warmup -> cosine to floor."""
    floor = 1e-5 / cfg.lr

    def fn(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return fn


@torch.no_grad()
def val_metrics(model: RLEEventsPolicy, val_cache: list[TrainBatch], cfg: TrainConfig) -> dict[str, float]:
    """Last-position (inference-matched) stream metrics over the cached val batches.

    ``action_nll_bits_per_frame`` is the total masked stream NLL (chord + duration) in bits over
    ``L_chunk`` -- directly comparable to a per-frame chord AR model's chunk NLL per the bijectivity
    note. ``buttons/first_event_logloss_bits`` marginalizes the FIRST chord token's softmax over a
    ``[V, 8]`` vocab->button-bit matrix; it scores only the first event (NOT the per-frame button
    model 005's ``buttons/logloss_bits`` scores -- a different, non-equivalent quantity, hence the
    distinct name)."""
    was_training = model.training
    model.eval()
    chord_nats: list[Tensor] = []
    dur_nats: list[Tensor] = []
    events: list[Tensor] = []
    oov_sum = frame_sum = 0
    first_probs: list[Tensor] = []
    first_tgt: list[Tensor] = []
    for batch in val_cache:
        out = _forward_supervised(model, batch, multi=False)
        comps = _component_nats(out)
        chord_nats.append(comps["chord"])
        dur_nats.append(comps["duration"])
        events.append(out["event_valid"].sum(1))
        oov_sum += int(out["oov_mask"].sum())
        frame_sum += out["oov_mask"].numel()
        first_probs.append(F.softmax(out["chord_logits"][:, 0], dim=-1) @ model.vocab_button_bits)
        first_tgt.append(out["tgt"][:, 0, _N_CONT:])
    chord = torch.cat(chord_nats)
    dur = torch.cat(dur_nats)
    out: dict[str, float] = {
        "action_nll_bits_per_frame": (chord + dur).mean().item() / (cfg.L_chunk * _LN2),
        "loss/chord_bits": chord.mean().item() / (cfg.L_chunk * _LN2),
        "loss/duration_bits": dur.mean().item() / (cfg.L_chunk * _LN2),
        "events_per_chunk": torch.cat(events).float().mean().item(),
        "chord_oov_rate": oov_sum / frame_sum,
    }
    logloss, brier = scoring.bernoulli_scores_from_probs(torch.cat(first_probs), torch.cat(first_tgt))
    out["buttons/first_event_logloss_bits"] = logloss.item()
    out["buttons/brier"] = brier.item()
    if was_training:
        model.train()
    return out


@torch.no_grad()
def recon_metrics(
    model: RLEEventsPolicy,
    val_cache: list[TrainBatch],
    *,
    mode: str,
    temp: float = 1.0,
    gen: torch.Generator | None = None,
) -> dict[str, float]:
    """Sample-space reconstruction proxy over the expanded 16 frames: decode a chunk and score it
    vs ground truth (button acc + F1, continuous MAE), as in 005."""
    was_training = model.training
    model.eval()
    tp = fp = fn = btn_correct = btn_total = 0
    cont_abs_err = 0.0
    cont_count = 0
    for batch in val_cache:
        pred = decode(model, batch.context, mode=mode, temp=temp, gen=gen)
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


def eval_vs_cpu(
    model: RLEEventsPolicy,
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
def _vocab_loader(cfg: TrainConfig, stats: dict[str, FeatureStats], *, num_workers: int) -> object:
    return make_loader(
        split="train",
        num_workers=num_workers,
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


def build_vocab_from_loader(cfg: TrainConfig, stats: dict[str, FeatureStats], *, device: str) -> Tensor:
    """Harvest chord counts over the seeded train loader and keep the top-``chord_vocab_size``."""
    loader = _vocab_loader(cfg, stats, num_workers=cfg.num_workers)
    counts = harvest_chord_counts(
        loader,
        budget=cfg.chord_vocab_frames,
        main_centers=scoring.STICK_CLUSTER_CENTERS_MAIN.to(device),
        c_centers=scoring.STICK_CLUSTER_CENTERS_C.to(device),
        trig_centers=scoring.TRIGGER_CENTERS.to(device),
        device=device,
    )
    return build_chord_vocab(counts, cfg.chord_vocab_size).cpu()


def train(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    *,
    comment: str = "",
    resume_run: str | None = None,
    resume_state: dict | None = None,
) -> None:
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    if cfg.amp_dtype not in ("bfloat16", "float32"):
        raise ValueError(f"amp_dtype must be 'bfloat16' or 'float32', got {cfg.amp_dtype!r}")

    if resume_state is not None:
        vocab_chords = resume_state["model"]["vocab_chords"].cpu()
        print(f"[vocab] resumed {vocab_chords.shape[0]} chords from checkpoint", flush=True)
    else:
        print("[vocab] harvesting chord vocab from train loader…", flush=True)
        vocab_chords = build_vocab_from_loader(cfg, stats, device=DEVICE)
        print(f"[vocab] kept {vocab_chords.shape[0]} chords (cap {cfg.chord_vocab_size})", flush=True)

    run_name = resume_run or make_run_name(_model_tag(cfg, vocab_chords.shape[0]), cfg.data_root, comment)
    uploader = BackgroundUploader(run_name) if cfg.push_to_r2 else None
    wandb.init(
        project="hal",
        name=run_name,
        id=resume_state["wandb_id"] if resume_state else None,
        resume="allow" if resume_state else None,
        tags=["rle_events", f"V{vocab_chords.shape[0]}", f"d{cfg.d_model}", f"tp{cfg.train_positions}"],
        config=asdict(cfg),
    )
    ckpt_dir, replay_dir = setup_run_dir(run_name)
    torch.save(vocab_chords, ckpt_dir / "chord_vocab.pt")

    autocast = (
        torch.autocast(DEVICE, dtype=torch.bfloat16)
        if cfg.amp_dtype == "bfloat16" and DEVICE == "cuda"
        else contextlib.nullcontext()
    )
    start_step = resume_state["step"] + 1 if resume_state else 0
    model = RLEEventsPolicy(cfg, vocab_chords).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    if wandb.run is not None:
        wandb.run.summary["model/num_params"] = n_params
    print(f"[model] {_model_tag(cfg, vocab_chords.shape[0])}  num_params={n_params / 1e6:.2f}M", flush=True)

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

    def _eval_and_upload(step_tag: str) -> dict[str, float]:
        sub = replay_dir / step_tag
        metrics = eval_vs_cpu(model, stats, cfg, max_frames=cfg.eval_max_frames, replay_dir=sub)
        if uploader is not None:
            n = uploader.upload_tree(sub, base=ckpt_dir, pattern="*.slp")
            print(f"[eval] queued {n} .slp for R2 ({step_tag})", flush=True)
        return metrics

    def _log_val(step: int) -> dict[str, float]:
        vm = val_metrics(model, val_cache, cfg)
        gen = torch.Generator(device=DEVICE).manual_seed(0)
        rm_arg = recon_metrics(model, val_cache, mode="argmax")
        rm_smp = recon_metrics(model, val_cache, mode="sample", temp=cfg.decode_temp, gen=gen)
        wandb.log(
            {
                **{f"val/{k}": v for k, v in vm.items()},
                **{f"val/argmax/{k}": v for k, v in rm_arg.items()},
                **{f"val/sample/{k}": v for k, v in rm_smp.items()},
            },
            step=step,
        )
        return vm

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

    model.train()
    it = iter(train_loader)
    run_t0 = time.monotonic()
    for step in range(start_step, cfg.max_steps):
        with profile("step") as sw:
            opt.zero_grad()
            loss_val = 0.0
            comps_acc: dict[str, list[Tensor]] = {}
            for _ in range(cfg.grad_accum_steps):
                try:
                    batch = next(it).to(DEVICE)
                except StopIteration:
                    it = iter(train_loader)
                    batch = next(it).to(DEVICE)
                with autocast:
                    comps = action_loss(model, batch, max_positions=cfg.train_positions)
                    loss = sum(comps.values()).mean() / cfg.norm_div / cfg.grad_accum_steps
                loss.backward()
                loss_val += loss.item()
                for k, v in comps.items():
                    comps_acc.setdefault(k, []).append(v.detach())
            opt.step()
            sched.step()
            if DEVICE == "cuda":
                torch.cuda.synchronize()
        chord = torch.cat(comps_acc["chord"])
        dur = torch.cat(comps_acc["duration"])
        sps = cfg.batch_size * cfg.grad_accum_steps / sw.elapsed
        wandb.log(
            {
                "train/loss": loss_val,
                "train/loss/chord_bits": chord.mean().item() / (cfg.L_chunk * _LN2),
                "train/loss/duration_bits": dur.mean().item() / (cfg.L_chunk * _LN2),
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
            _save("latest.pt", step)
        if cfg.val_every > 0 and step > 0 and step % cfg.val_every == 0:
            vm = _log_val(step)
            print(
                f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: "
                f"action_nll {vm['action_nll_bits_per_frame']:.3f} "
                f"events/chunk {vm['events_per_chunk']:.2f} oov {vm['chord_oov_rate']:.4f}",
                flush=True,
            )
        if cfg.eval_every > 0 and step > 0 and step % cfg.eval_every == 0:
            _save(f"step_{step:06d}.pt", step)
            metrics = _eval_and_upload(f"step_{step:06d}")
            wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)
            print(f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: closed_loop {metrics}", flush=True)

    vm_final = _log_val(cfg.max_steps)
    print(f"[final] action_nll {vm_final['action_nll_bits_per_frame']:.3f}", flush=True)
    metrics_final = _eval_and_upload("final")
    wandb.log({f"eval/{k}": v for k, v in metrics_final.items()}, step=cfg.max_steps)
    print(f"[final] closed_loop {metrics_final}", flush=True)
    _save("final.pt", cfg.max_steps)
    if uploader is not None:
        uploader.close()


# %%
def _load_ckpt(ckpt_path: str) -> tuple[RLEEventsPolicy, TrainConfig, dict[str, FeatureStats], dict]:
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = TrainConfig(**state["cfg"])
    model = RLEEventsPolicy(cfg, state["model"]["vocab_chords"]).to(DEVICE)
    model.load_state_dict(state["model"])
    model.eval()
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    return model, cfg, stats, state


def val_report(ckpt_path: str, *, n_batches: int = 24) -> None:
    """D3 diagnostic: how well does a trained checkpoint FIT the human val data (teacher-forced,
    no emulator)? Prints the stream NLL plus sample-space reconstruction at ``sample`` and ``argmax``."""
    from streaming.base.util import clean_stale_shared_memory

    clean_stale_shared_memory()
    model, cfg, stats, state = _load_ckpt(ckpt_path)
    loader = make_loader(
        split=cfg.val_split,
        num_workers=4,
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
    val_cache = [b.to(DEVICE) for b in itertools.islice(loader, n_batches)]
    if not val_cache:
        raise RuntimeError("val loader yielded zero batches")
    vm = val_metrics(model, val_cache, cfg)
    rec_s = recon_metrics(model, val_cache, mode="sample", temp=cfg.decode_temp)
    rec_a = recon_metrics(model, val_cache, mode="argmax")
    frames = sum(b.target.shape[0] * b.target.shape[1] for b in val_cache)
    print(f"\n[d3] {ckpt_path}  step={state['step']}  {frames} val frames  decode_temp={cfg.decode_temp}", flush=True)
    print(f"[d3] action_nll_bits_per_frame = {vm['action_nll_bits_per_frame']:.3f}", flush=True)
    print(
        f"[d3]   chord_bits={vm['loss/chord_bits']:.3f}  duration_bits={vm['loss/duration_bits']:.3f}  "
        f"events/chunk={vm['events_per_chunk']:.2f}  oov_rate={vm['chord_oov_rate']:.4f}",
        flush=True,
    )
    print(
        f"[d3] buttons(first event): logloss_bits={vm['buttons/first_event_logloss_bits']:.4f}  "
        f"brier={vm['buttons/brier']:.4f}",
        flush=True,
    )
    print(
        f"[d3] recon(sample): button_acc={rec_s['recon_button_acc']:.4f} f1={rec_s['recon_button_f1']:.4f} "
        f"cont_mae={rec_s['recon_cont_mae']:.4f}",
        flush=True,
    )
    print(
        f"[d3] recon(argmax): button_acc={rec_a['recon_button_acc']:.4f} f1={rec_a['recon_button_f1']:.4f} "
        f"cont_mae={rec_a['recon_cont_mae']:.4f}",
        flush=True,
    )


def eval_control_freq(
    ckpt_path: str,
    *,
    s_values: tuple[int, ...] = (1,),
    decode_mode: str | None = None,
    decode_temp: float | None = None,
    replicas: int | None = None,
    max_frames: int = 7200,
) -> None:
    """D1 diagnostic: closed-loop control-frequency sweep on FD vs lvl-9 CPU, WITHOUT retraining
    (see 005). ``s`` is the execution horizon; ``s=1`` replans every frame (the deployed default), ``s=L_chunk`` is the
    full-chunk open-loop extreme."""
    model, cfg, stats, state = _load_ckpt(ckpt_path)
    mode = decode_mode or cfg.decode
    temp = cfg.decode_temp if decode_temp is None else decode_temp
    reps = replicas or cfg.eval_replicas
    replay_dir = Path(ckpt_path).resolve().parent / "eval_replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    horizons = sorted({s for s in (*s_values, cfg.L_chunk) if 0 < s <= cfg.L_chunk})
    print(
        f"[d1] {ckpt_path}  step={state['step']}  decode={mode} temp={temp}  "
        f"L_chunk={cfg.L_chunk}  horizons={horizons}  replicas={reps}",
        flush=True,
    )
    for s in horizons:
        results = sweep_vs_cpu(
            lambda s=s: make_policy(model, stats, cfg, decode_mode=decode_mode, decode_temp=decode_temp, s=s),
            session_cfg=default_session_cfg(replay_dir),
            stages=(melee.Stage.FINAL_DESTINATION,),
            replicas=reps,
            max_parallel=cfg.eval_max_parallel,
            max_frames=max_frames,
        )
        print(f"[d1] s={s:>2d}  {vs_cpu_metrics(results)}", flush=True)


def eval_ckpt(ckpt_path: str, *, decode_mode: str | None = None, decode_temp: float | None = None) -> None:
    """Load a checkpoint, sweep stages vs CPU + self-play, print summaries (see 005)."""
    from hal.policy import INCLUDED_STAGES

    model, cfg, stats, state = _load_ckpt(ckpt_path)
    mode = decode_mode or cfg.decode
    temp = cfg.decode_temp if decode_temp is None else decode_temp
    print(f"[eval] loaded {ckpt_path}  step={state['step']}  device={DEVICE}  decode={mode} temp={temp}", flush=True)
    replay_dir = Path(ckpt_path).resolve().parent / "eval_replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    session_cfg = default_session_cfg(replay_dir)
    stages = tuple(s for s in INCLUDED_STAGES if s is not melee.Stage.FOUNTAIN_OF_DREAMS)

    def policy_factory() -> RecedingHorizon:
        return make_policy(model, stats, cfg, decode_mode=decode_mode, decode_temp=decode_temp)

    print("\n[eval] ============== vs-cpu ==============", flush=True)
    for stage, r, s in sweep_vs_cpu(
        policy_factory,
        session_cfg=session_cfg,
        stages=stages,
        replicas=cfg.eval_replicas,
        max_parallel=cfg.eval_max_parallel,
        max_frames=15_000,
    ):
        print(f"  {stage.name:18s} r{r} {s.as_dict() if s else 'CRASHED'}", flush=True)
    print("\n[eval] ============== self-play ==============", flush=True)
    for stage, r, s in sweep_self_play(
        policy_factory,
        session_cfg=session_cfg,
        stages=stages,
        replicas=cfg.eval_replicas,
        max_parallel=cfg.eval_max_parallel,
        max_frames=15_000,
    ):
        print(f"  {stage.name:18s} r{r} {s.as_dict() if s else 'CRASHED'}", flush=True)


# %%
@dataclass
class Args:
    """Top-level CLI surface. Pass TrainConfig fields as kebab-case flags,
    e.g. ``--cfg.opp-controller --cfg.chord-vocab-size 4096``."""

    cfg: TrainConfig = field(default_factory=TrainConfig)
    eval: str | None = None  # ckpt path; closed-loop eval instead of train
    eval_control_sweep: str | None = None  # ckpt path; D1 control-frequency sweep (s=1 vs L_chunk) on FD
    eval_cs_replicas: int | None = None  # override replicas for --eval-control-sweep
    val_report: str | None = None  # ckpt path; D3 teacher-forced val fit report (no emulator)
    eval_decode: str | None = None  # override decode mode for --eval (sample|argmax)
    eval_temp: float | None = None  # override decode temperature for --eval
    resume: str | None = None  # run_name to resume; pulls latest.pt (local, else R2)
    comment: str = ""


def main(args: Args) -> None:
    if args.eval is not None:
        eval_ckpt(args.eval, decode_mode=args.eval_decode, decode_temp=args.eval_temp)
        return
    if args.val_report is not None:
        val_report(args.val_report)
        return
    if args.eval_control_sweep is not None:
        eval_control_freq(
            args.eval_control_sweep,
            decode_mode=args.eval_decode,
            decode_temp=args.eval_temp,
            replicas=args.eval_cs_replicas,
        )
        return
    if args.resume is not None:
        state = load_for_resume(args.resume, Path("runs") / args.resume, device=DEVICE)
        if state is None:
            raise SystemExit(f"no latest.pt for run {args.resume!r} (local or R2)")
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
    auto_comment = f"rle-{cfg.max_steps // 1000}k-b{cfg.batch_size}"
    train(cfg, stats, comment=args.comment or auto_comment)


if __name__ == "__main__":
    main(tyro.cli(Args))
