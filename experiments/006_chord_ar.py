"""Causal AR decoder over per-frame joint "chord" tokens (fixes 005's marginal-only head).

005's classification head is *not* autoregressive: given the context, all L_chunk frames ×
channel groups are conditionally independent (the head never sees action inputs — neither
teacher-forced targets nor sampled values). With ~65% frame-to-frame action persistence and
median button holds of 5-7 frames, independent per-frame sampling destroys temporal
coherence (the same marginal-only failure family that produced 005's argmax do-nothing
collapse). 006 keeps the backbone and multi-position supervision identical and replaces the
head with a small causal Transformer decoder over the L_chunk positions, where each token is
the *joint* quantized controller state of one frame — a **chord**.

A chord packs the per-frame quantized channels (main 65 × c 9 × trigL 5 × trigR 5 ×
buttons 256, the ``hal.training.scoring`` quantizers) into one id:
``(((main*9 + c)*5 + tl)*5 + tr)*256 + btn``. The full product space is ~3.74M — far too
big for a softmax — but only ~8.2k distinct chords occur in 3.2M human frames (~5.4k cover
99.9%), so the vocabulary is **data-derived**: at train start a seeded scan of the train
loader harvests chords from full windows (context ego history + target chunk) until
``cfg.vocab_frame_budget`` frames, and the vocab is the unique chords by descending
frequency. The vocab is part of the experiment identity (like ``train_positions``): it is
persisted in the run dir AND inside every checkpoint, and resume/eval load it from the
checkpoint — never rebuild. Val/late-train chords outside the vocab are projected
deterministically to the nearest in-vocab chord (same (btn, tl, tr) with nearest main- then
c-center; else same btn nearest sticks; else the most frequent chord) and the projection
rate is logged as ``val/chord_oov_rate`` (expect <1%).

``val/action_nll_bits_per_frame`` here is the FULL JOINT chord NLL (one categorical per
frame). It is comparable to 005's summed factorized NLL only up to the different
quantizers/OOV projection — same caveat discipline as the flow-vs-classification axis.

Decode is sequential over the L_chunk frames: sample (temperature; default ``sample`` @ 1.0
— greedy argmax collapses an autoregressive policy to a do-nothing fixed point in closed
loop, exactly as in 005) chord_k, feed it back, repeat. No KV cache — the head is 2 tiny
layers over <=16 tokens; clarity over cleverness. This is 16 sequential head passes per
replan, vs 64 for a per-group (4 groups × 16 frames) AR factorization — the chord's
inference-speed advantage.

``cfg.opp_controller`` is a DELIBERATE INFORMATION CHEAT for a roofline: it concatenates the
opponent's 14-channel controller history onto every context token. Humans cannot see the
opponent's controller (rollback netplay technically transmits it, but project policy is
human-information parity), so this flag exists only to measure the headroom such information
would buy, teacher-forced. Closed-loop eval is unsupported under the cheat — the sim's
observation stream (``flatten_canonical_frame``) carries post-frame gamestate only, no
opponent controller — and fails loud.

Run:
    python experiments/006_chord_ar.py                         # train (closed-loop eval samples)
    python experiments/006_chord_ar.py --eval <ckpt>                       # eval at the trained decode
    python experiments/006_chord_ar.py --eval <ckpt> --eval-temp 0.7       # test-time temperature sweep
    python experiments/006_chord_ar.py --eval <ckpt> --eval-decode argmax  # re-eval an old run greedily
    python experiments/006_chord_ar.py --val-report <ckpt>                 # teacher-forced val fit (no emulator)
"""

# %%
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import contextlib
import itertools
import json
import math
import time
from collections.abc import Iterable
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
from jaxtyping import Bool
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

# Quantized per-channel cardinalities (the hal.training.scoring quantizers) and the packed
# chord id space. The id space is the FULL product (~3.74M) — only the data-derived vocab
# gets softmax classes; the id space is just the packing/LUT domain.
N_MAIN = scoring.STICK_CLUSTER_CENTERS_MAIN.shape[0]  # 65
N_C = scoring.STICK_CLUSTER_CENTERS_C.shape[0]  # 9
N_TRIG = scoring.TRIGGER_CENTERS.shape[0]  # 5
N_BTN = scoring.N_BUTTON_COMBOS  # 256
N_CHORD_SPACE = N_MAIN * N_C * N_TRIG * N_TRIG * N_BTN  # 3,744,000


# %%
@dataclass
class TrainConfig:
    # model backbone (identical to 005/003)
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    # causal AR chord head (small Transformer decoder over the L_chunk chord tokens)
    d_head: int = 128
    n_head_layers: int = 2
    head_heads: int = 4
    head_ff: int = 512
    # matchup conditioning (schema v4), identical to 005
    cond_char_stage: bool = True
    char_vocab: int = 64  # slp/libmelee Character ids (0..~32), padded
    char_dim: int = 12
    stage_vocab: int = 64  # libmelee Stage values, padded
    stage_dim: int = 8
    # ROOFLINE CHEAT (see module docstring): concatenate the opponent's 14-channel controller
    # history onto every context token. Information humans don't have — headroom measurement
    # only; closed-loop eval is unsupported under it.
    opp_controller: bool = False
    # data-derived chord vocab: scan the (seeded) train loader for this many frames at train
    # start. Part of the experiment identity — persisted in every checkpoint, never rebuilt.
    vocab_frame_budget: int = 5_000_000
    # decode (closed-loop controller + recon proxy). Sampling is the DEFAULT: argmax greedily
    # picks the mode of P(action | recent inputs), which for an autoregressive policy collapses
    # to a "do nothing" fixed point in closed loop (it feeds neutral back to itself). argmax
    # stays available for the deterministic recon metric and as a test-time override.
    decode: str = "sample"  # "sample" (temperature) | "argmax" (greedy)
    decode_temp: float = 1.0  # softmax temperature for sample decode (higher = more random)
    # training-stability divisor on the NLL scalar ONLY; reported likelihood is never divided by it
    norm_div: float = 1.0
    # Seeds model init, the dataloader's window + ego-port sampling, and therefore the vocab scan.
    seed: int = 0
    # window / chunking
    L_ctx: int = 256
    L_chunk: int = 16
    # multi-position supervision: number of context positions supervised per sequence each step
    # (random subset, redrawn per step). -1 supervises every valid position.
    train_positions: int = 64
    # optimization
    batch_size: int = 32  # micro-batch run on the GPU per forward
    grad_accum_steps: int = 1  # optimizer step sees batch_size * grad_accum_steps samples
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 2**15
    amp_dtype: str = "bfloat16"  # "bfloat16" | "float32"
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


def _model_tag(cfg: TrainConfig) -> str:
    cs = "-cs" if cfg.cond_char_stage else ""
    oppc = "-oppc" if cfg.opp_controller else ""
    return (
        f"chord-ar{cs}{oppc}-d{cfg.d_model}-L{cfg.n_layers}-dh{cfg.d_head}"
        f"-Lc{cfg.L_ctx}-Lk{cfg.L_chunk}-tp{cfg.train_positions}"
    )


# %%
# --- chord packing (experiment-local: the joint quantized controller state of one frame) ---
@jaxtyped(typechecker=beartype)
def pack_chord(
    main: Int[Tensor, "*b"],
    c: Int[Tensor, "*b"],
    tl: Int[Tensor, "*b"],
    tr: Int[Tensor, "*b"],
    btn: Int[Tensor, "*b"],
) -> Int[Tensor, "*b"]:
    """Mixed-radix pack of the five quantized channels into one chord id in [0, N_CHORD_SPACE)."""
    return (((main * N_C + c) * N_TRIG + tl) * N_TRIG + tr) * N_BTN + btn


@jaxtyped(typechecker=beartype)
def unpack_chord(
    chord: Int[Tensor, "*b"],
) -> tuple[Int[Tensor, "*b"], Int[Tensor, "*b"], Int[Tensor, "*b"], Int[Tensor, "*b"], Int[Tensor, "*b"]]:
    """Inverse of ``pack_chord``: chord id → (main, c, tl, tr, btn) quantizer indices."""
    btn = chord % N_BTN
    rest = chord // N_BTN
    tr = rest % N_TRIG
    rest = rest // N_TRIG
    tl = rest % N_TRIG
    rest = rest // N_TRIG
    c = rest % N_C
    main = rest // N_C
    return main, c, tl, tr, btn


@jaxtyped(typechecker=beartype)
def quantize_actions(actions: Float[Tensor, "*b d_action"]) -> Int[Tensor, "*b"]:
    """Raw A_DIM action vectors → chord ids via the shared ``hal.training.scoring`` quantizers
    (the same code 005's stick_clusters targets use, so the two stay byte-comparable)."""
    cont, btn = actions[..., :_N_CONT], actions[..., _N_CONT:]
    main = scoring.nearest_cluster(cont[..., 0:2], scoring.STICK_CLUSTER_CENTERS_MAIN)
    c = scoring.nearest_cluster(cont[..., 2:4], scoring.STICK_CLUSTER_CENTERS_C)
    trig = scoring.nearest_center(cont[..., 4:6], scoring.TRIGGER_CENTERS)  # [*b, 2]
    return pack_chord(main, c, trig[..., 0], trig[..., 1], scoring.buttons_to_combo(btn))


@jaxtyped(typechecker=beartype)
def chord_to_actions(chord: Int[Tensor, "*b"]) -> Float[Tensor, "*b d_action"]:
    """Inverse of ``quantize_actions`` up to quantization: chord id → raw A_DIM action vector
    (``[-1,1]`` sticks, ``[0,1]`` triggers, ``{0,1}`` buttons) via the scoring inverses."""
    main, c, tl, tr, btn = unpack_chord(chord)
    return torch.cat(
        [
            scoring.cluster_to_xy(main, scoring.STICK_CLUSTER_CENTERS_MAIN),
            scoring.cluster_to_xy(c, scoring.STICK_CLUSTER_CENTERS_C),
            scoring.center_to_value(tl, scoring.TRIGGER_CENTERS)[..., None],
            scoring.center_to_value(tr, scoring.TRIGGER_CENTERS)[..., None],
            scoring.combo_to_buttons(btn),
        ],
        dim=-1,
    )


# %%
class ChordVocab:
    """Data-derived chord vocabulary: unique chord ids by descending train-scan frequency.

    Owns the chord↔class mapping the model trains against: a dense LUT over the full
    ~3.74M-id pack space (chord id → vocab index, -1 = OOV), the per-entry dequantized
    action table (decode = one gather), the per-entry button bits (marginal button probs =
    one matmul), and the deterministic OOV projection (cached per unique OOV id). Part of
    the experiment identity: serialize with ``to_state`` into every checkpoint and restore
    with ``from_state`` — never rebuild on resume/eval.
    """

    def __init__(self, chords: list[int], counts: list[int]) -> None:
        if not chords:
            raise ValueError("empty chord vocab")
        if len(chords) != len(counts):
            raise ValueError(f"chords/counts length mismatch: {len(chords)} vs {len(counts)}")
        self.chord_ids = torch.tensor(chords, dtype=torch.long)
        self.counts = torch.tensor(counts, dtype=torch.long)
        if not (self.counts[:-1] >= self.counts[1:]).all():
            raise ValueError("vocab must be sorted by descending count (index 0 = most frequent chord)")
        if self.chord_ids.min() < 0 or self.chord_ids.max() >= N_CHORD_SPACE:
            raise ValueError("chord id outside the pack space")
        self.size = len(chords)
        lut = torch.full((N_CHORD_SPACE,), -1, dtype=torch.int32)
        lut[self.chord_ids] = torch.arange(self.size, dtype=torch.int32)
        if int((lut >= 0).sum()) != self.size:
            raise ValueError("duplicate chord ids in vocab")
        self._lut = lut
        self._v_main, self._v_c, self._v_tl, self._v_tr, self._v_btn = unpack_chord(self.chord_ids)
        self.actions = chord_to_actions(self.chord_ids)  # [V_chord, A_DIM] float32
        self.button_bits = scoring.combo_to_buttons(self._v_btn)  # [V_chord, 8] float32
        # Pairwise squared center distances for the OOV projection's lexicographic argmin.
        self._main_d2 = torch.cdist(scoring.STICK_CLUSTER_CENTERS_MAIN, scoring.STICK_CLUSTER_CENTERS_MAIN) ** 2
        self._c_d2 = torch.cdist(scoring.STICK_CLUSTER_CENTERS_C, scoring.STICK_CLUSTER_CENTERS_C) ** 2
        self._oov_cache: dict[int, int] = {}

    def to_state(self) -> dict:
        return {"chords": self.chord_ids.tolist(), "counts": self.counts.tolist()}

    @classmethod
    def from_state(cls, state: dict) -> ChordVocab:
        return cls(state["chords"], state["counts"])

    def coverage_report(self) -> str:
        total = float(self.counts.sum())
        cum = self.counts.cumsum(0).float() / total
        k99 = int((cum < 0.99).sum()) + 1
        k999 = int((cum < 0.999).sum()) + 1
        return f"{self.size} chords / {total:.0f} scanned frames; 99% mass in {k99}, 99.9% in {k999}"

    def _map_oov(self, cid: int) -> int:
        """Deterministic projection of one out-of-vocab chord id onto a vocab index: prefer the
        same (btn, tl, tr) with lexicographically nearest (main-center, c-center) squared
        distance; fall back to same btn nearest sticks; final fallback the most frequent chord
        (index 0). Cached per unique OOV id."""
        hit = self._oov_cache.get(cid)
        if hit is not None:
            return hit
        main, c, tl, tr, btn = (int(x) for x in unpack_chord(torch.tensor(cid, dtype=torch.long)))
        tiers = (
            (self._v_btn == btn) & (self._v_tl == tl) & (self._v_tr == tr),
            self._v_btn == btn,
        )
        out = 0  # most frequent chord
        for tier in tiers:
            if bool(tier.any()):
                d_main = self._main_d2[main, self._v_main].masked_fill(~tier, math.inf)
                near = d_main <= d_main.min() + 1e-9
                d_c = self._c_d2[c, self._v_c].masked_fill(~near, math.inf)
                out = int(d_c.argmin())  # argmin's first-index tie-break = most frequent
                break
        self._oov_cache[cid] = out
        return out

    @jaxtyped(typechecker=beartype)
    def encode(self, chords: Int[Tensor, "*b"]) -> tuple[Int[Tensor, "*b"], Bool[Tensor, "*b"]]:
        """Chord ids → (vocab indices with OOV projected, OOV mask). One LUT gather on the
        ids' device; OOV ids (rare, <1%) resolve through the cached projection."""
        if self._lut.device != chords.device:
            self._lut = self._lut.to(chords.device)
        idx = self._lut[chords]
        oov = idx < 0
        if bool(oov.any()):
            for cid in torch.unique(chords[oov]).tolist():
                idx[chords == cid] = self._map_oov(cid)
        return idx.long(), oov


def build_chord_vocab(batches: Iterable[TrainBatch], *, frame_budget: int) -> ChordVocab:
    """Scan ``batches`` (the seeded train loader — deterministic given seed + worker layout)
    harvesting chords from FULL windows: the context's ego action history AND the target
    chunk, so one window yields up to L_ctx + L_chunk frames (left-pad positions, which are
    zero-filled filler, are excluded via ``ctx_pad``). Stops at ``frame_budget`` frames.
    Vocab = unique chords sorted by descending frequency (id-ascending tie-break)."""
    if frame_budget <= 0:
        raise ValueError(f"frame_budget must be > 0, got {frame_budget}")
    counts = torch.zeros(N_CHORD_SPACE, dtype=torch.long)
    seen = 0
    for batch in batches:
        ctx = batch.context
        hist = stack_actions(ctx.features)  # [B, L_ctx, A_DIM]
        valid = torch.arange(hist.shape[1])[None, :] >= ctx.ctx_pad[:, None]
        ids = torch.cat([quantize_actions(hist)[valid], quantize_actions(batch.target).reshape(-1)])
        counts += torch.bincount(ids, minlength=N_CHORD_SPACE)
        seen += ids.numel()
        if seen >= frame_budget:
            break
    if seen == 0:
        raise RuntimeError("vocab scan saw zero frames")
    order = torch.argsort(counts, descending=True, stable=True)  # count desc, chord id asc on ties
    keep = order[counts[order] > 0]
    return ChordVocab(keep.tolist(), counts[keep].tolist())


# %%
class ChordARPolicy(nn.Module):
    """Causal backbone (identical to 005/003) + causal AR decoder over per-frame chord tokens.

    The **backbone** is a decoder-style Transformer over the L_ctx context tokens under a
    causal mask, so ``hidden[i]`` depends only on positions ``<= i``. The **head** is itself
    autoregressive over the chunk: token k's input is ``chord_emb(chord_{k-1})`` (a learned
    BOS embedding at k=0) + ``chunk_pos_emb[k]`` + the projected backbone conditioning
    (broadcast), under a causal mask, emitting V_chord-way logits per position. Training is
    teacher-forced in ONE forward across all supervised positions (005's multi-position
    machinery); inference samples the 16 frames sequentially (``decode``)."""

    def __init__(self, cfg: TrainConfig, vocab: ChordVocab):
        super().__init__()
        self.L_ctx = cfg.L_ctx
        self.L_chunk = cfg.L_chunk
        self.n_heads = cfg.n_heads
        d = cfg.d_model
        dh = cfg.d_head

        if cfg.decode not in ("argmax", "sample"):
            raise ValueError(f"decode must be argmax|sample, got {cfg.decode!r}")
        if not cfg.decode_temp > 0:
            raise ValueError(f"decode_temp must be > 0, got {cfg.decode_temp}")
        if not cfg.norm_div > 0:
            raise ValueError(f"norm_div must be > 0, got {cfg.norm_div}")

        self.cat_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocab_size, dim) for name, (vocab_size, dim) in CAT_FEATURES.items()}
        )
        n_float = len(FLOAT_FEATURES)
        n_mask = len(FLOAT_FEATURES)
        n_cat = sum(dim for _, dim in CAT_FEATURES.values())
        per_player_dim = n_float + n_mask + n_cat
        per_frame_in_dim = 2 * per_player_dim + A_DIM  # ego + opp + ego controller history
        # Roofline cheat: the opponent's controller history as a second 14-channel block.
        self.opp_controller = cfg.opp_controller
        if cfg.opp_controller:
            per_frame_in_dim += A_DIM
        # Matchup conditioning: a shared per-player character embedding (ego + opp) + one
        # global stage embedding, concatenated onto every context token.
        self.cond_char_stage = cfg.cond_char_stage
        if cfg.cond_char_stage:
            self.char_emb = nn.Embedding(cfg.char_vocab, cfg.char_dim)
            self.stage_emb = nn.Embedding(cfg.stage_vocab, cfg.stage_dim)
            per_frame_in_dim += 2 * cfg.char_dim + cfg.stage_dim

        # --- causal backbone (verbatim from 005) ---
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
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers, enable_nested_tensor=False)

        # --- causal AR chord head ---
        self.vocab = vocab
        self.V_chord = vocab.size
        self.bos_idx = vocab.size  # chord_emb row V_chord is the learned BOS token
        self.cond_proj = nn.Linear(d, dh)
        self.chunk_pos_emb = nn.Embedding(self.L_chunk, dh)
        self.chord_emb = nn.Embedding(vocab.size + 1, dh)
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
        self.chord_out = nn.Linear(dh, vocab.size)
        # True = masked. Strict causality over the L_chunk chord positions.
        self.register_buffer("head_mask", torch.triu(torch.ones(self.L_chunk, self.L_chunk, dtype=torch.bool), 1))
        # Dequantized action + button-bit tables: decode is one gather, button marginals one
        # matmul. Buffers so they ride .to(device) and the checkpoint state_dict.
        self.register_buffer("vocab_actions", vocab.actions.clone())
        self.register_buffer("vocab_button_bits", vocab.button_bits.clone())

    # --- backbone (verbatim from 005, plus the opp-controller cheat block) ---
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

    def _history_features(self, features: dict[str, Tensor], prefix: str) -> Tensor:
        return torch.cat([features[f"{prefix}_{ch}"][..., None] for ch in ACTION_CHANNELS], dim=-1)

    def _context_tokens(self, features: dict[str, Tensor]) -> Float[Tensor, "B L_ctx d_model"]:
        ego = self._per_player_features(features, "ego")
        opp = self._per_player_features(features, "opp")
        hist = self._history_features(features, "ego")
        parts = [ego, opp, hist]
        if self.opp_controller:
            # fails loud (KeyError) if the obs stream has no opp controller — see make_policy
            parts.append(self._history_features(features, "opp"))
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
        """Causal backbone over the L_ctx context tokens → one hidden per position;
        ``hidden[i]`` depends only on positions ``<= i``."""
        tok = self._context_tokens(ctx.features)
        T = tok.size(1)
        tok = tok + self.pos_emb.weight[None, :T, :]
        mask = self._backbone_mask(ctx.ctx_pad, T, tok.device)
        return self.encoder(tok, mask=mask)

    # --- causal AR chord head ---
    @jaxtyped(typechecker=beartype)
    def chord_logits(
        self, cond: Float[Tensor, "N d_model"], tgt_idx: Int[Tensor, "N L_chunk"]
    ) -> Float[Tensor, "N L_chunk V_chord"]:
        """Teacher-forced V_chord-way logits at every chunk position in ONE forward. Token k's
        input embeds ``tgt_idx[k-1]`` (BOS at k=0), so under the causal mask the logits at
        position k depend only on the conditioning and chords ``< k``. ``N`` flattens whatever
        positions are decoded (B*train_positions at train, B at inference)."""
        H = self.L_chunk
        prev = F.pad(tgt_idx[:, : H - 1], (1, 0), value=self.bos_idx)  # [N, H]: BOS, chord_0, ..., chord_{H-2}
        tok = self.chord_emb(prev) + self.chunk_pos_emb.weight[None, :H, :] + self.cond_proj(cond)[:, None, :]
        h = self.head_encoder(tok, mask=self.head_mask)  # [N, H, d_head]
        return self.chord_out(h)


# %%
def _position_targets(ctx: Context, target: Tensor, H: int) -> tuple[Tensor, Tensor]:
    """For every context position ``i``, the next-H-action target chunk + a validity mask
    (verbatim from 005/003). ``A_full[:, :L_ctx] = stack_actions(ctx.features)``,
    ``A_full[:, L_ctx:] = target``; position ``i``'s leak-free target is ``A_full[i+1:i+1+H]``
    (last position recovers ``target``). Returns ``(tgt [B,T,H,A_DIM], valid [B,T])``."""
    a_full = torch.cat([stack_actions(ctx.features), target], dim=1)
    T = a_full.size(1) - H
    tgt = a_full.unfold(1, H, 1)[:, 1:].permute(0, 1, 3, 2).contiguous()
    pos = torch.arange(T, device=a_full.device)
    valid = pos[None, :] >= ctx.ctx_pad[:, None]
    return tgt, valid


def _ce_nats(logits: Tensor, idx: Tensor) -> Tensor:
    """Per-element categorical cross-entropy in nats, computed on a flattened VIEW of the
    logits (no extra [N,H,V_chord] copy — at defaults that tensor is already
    N=2048 × H=16 × V≈8k ≈ 1 GiB fp32, so one materialization is the budget)."""
    flat = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), idx.reshape(-1), reduction="none")
    return flat.reshape(idx.shape)


def _select(
    model: ChordARPolicy,
    batch: TrainBatch,
    *,
    multi: bool,
    max_positions: int = -1,
    gen: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Pick the supervised context positions (verbatim selection from 005) and return
    ``(cond [N, d_model], tgt [N, H, A_DIM])`` — the head is teacher-forced so it needs the
    targets as inputs, hence cond rather than logits comes back. ``multi=False`` supervises
    only the last position (matches inference; used by val)."""
    ctx = batch.context
    H = model.L_chunk
    hidden = model.encode_context(ctx)  # [B, T, d_model]
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


def action_loss(
    model: ChordARPolicy,
    batch: TrainBatch,
    *,
    multi: bool = True,
    max_positions: int = -1,
    gen: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Teacher-forced joint chord NLL over the supervised positions: ``(nll [N, H] nats,
    oov_rate scalar)``. Targets are quantized to chord ids and projected into the vocab
    (``oov_rate`` is the projected fraction); ``nll.mean()`` is the training scalar."""
    cond, tgt = _select(model, batch, multi=multi, max_positions=max_positions, gen=gen)
    idx, oov = model.vocab.encode(quantize_actions(tgt))
    logits = model.chord_logits(cond, idx)
    return _ce_nats(logits, idx), oov.float().mean()


@torch.no_grad()
def decode(
    model: ChordARPolicy,
    ctx: Context,
    *,
    mode: str = "argmax",
    temp: float = 1.0,
    gen: torch.Generator | None = None,
) -> Float[Tensor, "B L_chunk d_action"]:
    """One action chunk per sample from the LAST context position, in raw action ranges
    (``[-1,1]`` sticks, ``[0,1]`` triggers, ``{0,1}`` buttons).

    Sequential AR decode over the L_chunk frames: pick chord_k from the ``temp``-scaled
    softmax (``"sample"``, the default controller — argmax collapses an autoregressive policy
    to a do-nothing fixed point in closed loop, as in 005) or greedily (``"argmax"``, the
    deterministic recon metric), feed it back, repeat. No KV cache — the head is 2 tiny
    layers over <=16 tokens, so each step re-runs the full-width head under the causal mask
    (future slots hold BOS filler the mask hides); clarity over cleverness. 16 sequential
    head passes per replan, vs 64 for a per-group factorization. Dequantization is one gather
    from the vocab's precomputed action table. The single inference path for this policy:
    closed-loop play and the recon proxy both call this."""
    if mode not in ("argmax", "sample"):
        raise ValueError(f"decode mode must be argmax|sample, got {mode!r}")
    cond = model.encode_context(ctx)[:, -1, :]  # [B, d_model]
    H = model.L_chunk
    B = cond.shape[0]
    cond_tok = model.cond_proj(cond)[:, None, :] + model.chunk_pos_emb.weight[None, :H, :]  # [B, H, d_head]
    prev = torch.full((B, H), model.bos_idx, dtype=torch.long, device=cond.device)  # prev[k] = chord_{k-1}
    picked = torch.empty(B, H, dtype=torch.long, device=cond.device)
    for k in range(H):
        h = model.head_encoder(model.chord_emb(prev) + cond_tok, mask=model.head_mask)
        logits_k = model.chord_out(h[:, k])  # causal mask ⇒ position k ignores the BOS filler at > k
        if mode == "argmax":
            idx_k = logits_k.argmax(-1)
        else:
            idx_k = torch.multinomial(F.softmax(logits_k / temp, dim=-1), 1, generator=gen).squeeze(-1)
        picked[:, k] = idx_k
        if k + 1 < H:
            prev[:, k + 1] = idx_k
    return model.vocab_actions[picked]


def make_policy(
    model: ChordARPolicy,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    device: str = DEVICE,
    decode_mode: str | None = None,
    decode_temp: float | None = None,
    s: int | None = None,
) -> RecedingHorizon:
    """Fresh open-loop closed-loop policy for one eval wave (rolling state must not leak).
    Same surface as 005: ``decode_mode``/``decode_temp`` override ``cfg`` for a test-time
    decode sweep; ``s`` overrides the execution horizon to probe control frequency. Closed-
    loop sampling draws fresh randomness each replan (``gen=None``)."""
    if cfg.opp_controller:
        raise ValueError(
            "opp_controller is a teacher-forced roofline cheat: the closed-loop observation "
            "stream (flatten_canonical_frame) carries no opponent controller, so this model "
            "cannot play"
        )
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


def nll_breakdown(nll: Tensor) -> dict[str, float]:
    """Per-horizon-frame marginals (bits) of the joint chord NLL from the ``[N, H]`` nats.
    No modality split — the chord is the joint; 005's per-modality partition doesn't exist
    here by construction."""
    per_frame = nll.mean(dim=0)  # [H]
    return {f"horizon/frame_{k + 1:02d}": per_frame[k].item() / _LN2 for k in range(per_frame.shape[0])}


@torch.no_grad()
def val_metrics(model: ChordARPolicy, val_cache: list[TrainBatch], cfg: TrainConfig) -> dict[str, float]:
    """Last-position (``multi=False``, inference-matched) proper-scoring metrics over the
    cached val batches. ``action_nll_bits_per_frame`` is the full joint chord NLL in bits —
    comparable to 005's summed factorized NLL only up to the quantizers/OOV projection (the
    NLL scores the vocab-projected target; ``chord_oov_rate`` quantifies that projection).
    Button scores marginalize the chord softmax through the precomputed [V_chord, 8] bit
    matrix (one matmul), so the buttons are scored as a proper Bernoulli model against the
    TRUE (unprojected) button targets."""
    was_training = model.training
    model.eval()
    nll_cat: list[Tensor] = []
    oov_cat: list[Tensor] = []
    btn_probs: list[Tensor] = []
    btn_tgts: list[Tensor] = []
    multipress: list[Tensor] = []
    for batch in val_cache:
        cond, tgt = _select(model, batch, multi=False)
        idx, oov = model.vocab.encode(quantize_actions(tgt))
        logits = model.chord_logits(cond, idx)
        nll_cat.append(_ce_nats(logits, idx))
        oov_cat.append(oov.reshape(-1))
        btn_probs.append(F.softmax(logits, dim=-1) @ model.vocab_button_bits)
        btn_tgts.append(tgt[..., _N_CONT:])
        multipress.append(scoring.buttons_to_class(tgt[..., _N_CONT:])[1])
    nll = torch.cat(nll_cat)  # [M, H]
    out = {f"loss/{k}": v for k, v in nll_breakdown(nll).items()}
    out["action_nll_bits_per_frame"] = nll.mean().item() / _LN2
    out["chord_oov_rate"] = torch.cat(oov_cat).float().mean().item()
    logloss, brier = scoring.bernoulli_scores_from_probs(torch.cat(btn_probs), torch.cat(btn_tgts))
    out["buttons/logloss_bits"] = logloss.item()
    out["buttons/brier"] = brier.item()
    out["buttons/multipress_rate"] = torch.cat(multipress).float().mean().item()
    if was_training:
        model.train()
    return out


@torch.no_grad()
def recon_metrics(
    model: ChordARPolicy,
    val_cache: list[TrainBatch],
    *,
    mode: str,
    temp: float = 1.0,
    gen: torch.Generator | None = None,
) -> dict[str, float]:
    """Sample-space reconstruction proxy (verbatim semantics from 005): decode a chunk and
    score it vs ground truth. ``mode="argmax"`` is the deterministic controller proxy;
    ``"sample"`` the distributional one at ``temp`` — pass ``cfg.decode_temp`` so this matches
    the deployed policy. Buttons → acc + F1 @ decode; continuous → MAE."""
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
    model: ChordARPolicy,
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
def _save_checkpoint(
    path: Path,
    *,
    step: int,
    model: ChordARPolicy,
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler,
    cfg: dict,
    vocab: ChordVocab,
    wandb_id: str | None,
    uploader: BackgroundUploader | None = None,
) -> None:
    """Local mirror of ``hal.training.checkpoints.save_checkpoint`` that additionally carries
    the data-derived chord vocab — part of the experiment identity, so resume/eval load it
    from here and never rebuild (the shared helper has no extra-payload slot)."""
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "cfg": cfg,
            "vocab": vocab.to_state(),
            "wandb_id": wandb_id,
        },
        path,
    )
    print(f"[ckpt] saved {path}", flush=True)
    if uploader is not None:
        uploader.upload(path)


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
        tags=["chord_ar", f"d{cfg.d_model}", f"tp{cfg.train_positions}"] + (["oppc"] if cfg.opp_controller else []),
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
    start_step = resume_state["step"] + 1 if resume_state else 0
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

    # Vocab BEFORE model: V_chord sizes the head. Resume loads it from the checkpoint —
    # never rebuild (the scan's exact windows aren't replayable mid-run, and the class↔chord
    # mapping is the experiment identity). Fresh runs scan the train loader's first epoch(s);
    # training then re-iterates from a fresh epoch, which is fine — windows are redrawn per
    # epoch by design.
    if resume_state is not None:
        vocab = ChordVocab.from_state(resume_state["vocab"])
        print(f"[vocab] restored from checkpoint: {vocab.coverage_report()}", flush=True)
    else:
        print(f"[vocab] scanning train loader for {cfg.vocab_frame_budget} frames…", flush=True)
        v_t0 = time.monotonic()
        vocab = build_chord_vocab(train_loader, frame_budget=cfg.vocab_frame_budget)
        print(f"[vocab] {vocab.coverage_report()} in {time.monotonic() - v_t0:.1f}s", flush=True)
    (ckpt_dir / "chord_vocab.json").write_text(json.dumps(vocab.to_state()))
    if uploader is not None:
        uploader.upload(ckpt_dir / "chord_vocab.json")

    model = ChordARPolicy(cfg, vocab).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    if wandb.run is not None:
        wandb.run.summary["model/num_params"] = n_params
        wandb.run.summary["model/vocab_size"] = vocab.size
    print(f"[model] {_model_tag(cfg)}  V_chord={vocab.size}  num_params={n_params / 1e6:.2f}M", flush=True)

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
        if cfg.opp_controller:
            print("[eval] skipped: opp_controller cheat has no closed-loop observation stream", flush=True)
            return {}
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
        _save_checkpoint(
            ckpt_dir / name,
            step=step,
            model=model,
            opt=opt,
            sched=sched,
            cfg=asdict(cfg),
            vocab=vocab,
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
            nll_acc: list[Tensor] = []
            oov_acc: list[float] = []
            for _ in range(cfg.grad_accum_steps):
                try:
                    batch = next(it).to(DEVICE)
                except StopIteration:
                    it = iter(train_loader)
                    batch = next(it).to(DEVICE)
                with autocast:
                    nll, oov_rate = action_loss(model, batch, max_positions=cfg.train_positions)
                    loss = nll.mean() / cfg.norm_div / cfg.grad_accum_steps
                loss.backward()
                loss_val += loss.item()
                nll_acc.append(nll.detach())
                oov_acc.append(oov_rate.item())
            opt.step()
            sched.step()
            if DEVICE == "cuda":
                torch.cuda.synchronize()
        breakdown = nll_breakdown(torch.cat(nll_acc))
        sps = cfg.batch_size * cfg.grad_accum_steps / sw.elapsed
        wandb.log(
            {
                "train/loss": loss_val,
                **{f"train/loss/{k}": v for k, v in breakdown.items()},
                "train/chord_oov_rate": sum(oov_acc) / len(oov_acc),
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
                f"action_nll {vm['action_nll_bits_per_frame']:.3f} btn_logloss {vm['buttons/logloss_bits']:.3f} "
                f"oov {vm['chord_oov_rate']:.4f}",
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
def _model_from_state(state: dict, *, device: str = DEVICE) -> tuple[ChordARPolicy, TrainConfig, ChordVocab]:
    """Rebuild (model, cfg, vocab) from a checkpoint state dict. The vocab comes from the
    checkpoint — the experiment identity — never from a fresh scan."""
    cfg = TrainConfig(**state["cfg"])
    vocab = ChordVocab.from_state(state["vocab"])
    model = ChordARPolicy(cfg, vocab).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, cfg, vocab


def _load_ckpt(ckpt_path: str) -> tuple[ChordARPolicy, TrainConfig, dict[str, FeatureStats], dict]:
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model, cfg, _ = _model_from_state(state, device=DEVICE)
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    return model, cfg, stats, state


def val_report(ckpt_path: str, *, n_batches: int = 24) -> None:
    """D3 diagnostic: how well does a trained checkpoint FIT the human val data (teacher-forced,
    no emulator)? Prints the joint chord NLL, the OOV projection rate, marginalized button
    scores, and sample-space reconstruction at the deployed ``sample`` decode and at ``argmax``."""
    from streaming.base.util import clean_stale_shared_memory

    clean_stale_shared_memory()  # drop stale /dev/shm from prior (killed) StreamingDataset procs
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
    print(
        f"\n[d3] {ckpt_path}  step={state['step']}  {frames} val frames  "
        f"V_chord={model.V_chord}  decode_temp={cfg.decode_temp}",
        flush=True,
    )
    print(f"[d3] action_nll_bits_per_frame = {vm['action_nll_bits_per_frame']:.3f} (joint chord)", flush=True)
    print(f"[d3] chord_oov_rate = {vm['chord_oov_rate']:.4f}", flush=True)
    print(
        f"[d3] buttons: logloss_bits={vm['buttons/logloss_bits']:.4f}  brier={vm['buttons/brier']:.4f}  "
        f"multipress_rate={vm['buttons/multipress_rate']:.4f}",
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
    for k in sorted(k for k in vm if k.startswith("loss/horizon/")):
        print(f"[d3]   {k} = {vm[k]:.4f} bits", flush=True)


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
    (verbatim semantics from 005). ``s=L_chunk`` is the full-chunk open-loop extreme; ``s=1``
    replans every frame using only the next-frame prediction."""
    model, cfg, stats, state = _load_ckpt(ckpt_path)
    mode = decode_mode or cfg.decode
    temp = cfg.decode_temp if decode_temp is None else decode_temp
    reps = replicas or cfg.eval_replicas
    replay_dir = Path(ckpt_path).resolve().parent / "eval_replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    # de-dup and always include the full-chunk horizon L_chunk as the reference point
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
    """Load a checkpoint, sweep stages vs CPU + self-play, print summaries. ``decode_mode``/
    ``decode_temp`` override the trained cfg for this eval only (test-time decode sweep)."""
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
    e.g. ``--cfg.opp-controller --cfg.vocab-frame-budget 2000000``."""

    cfg: TrainConfig = field(default_factory=TrainConfig)
    eval: str | None = None  # ckpt path; closed-loop eval instead of train
    eval_control_sweep: str | None = None  # ckpt path; D1 control-frequency sweep (s=1 vs L_chunk) on FD
    eval_cs_replicas: int | None = None  # override replicas for --eval-control-sweep (default: cfg.eval_replicas)
    val_report: str | None = None  # ckpt path; D3 teacher-forced val fit report (no emulator)
    eval_decode: str | None = None  # override decode mode for --eval (sample|argmax); test-time sweep
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
        # Only pure host-scaling knobs (worker/prefetch counts) follow the current code; the
        # experiment-identity knobs (train_positions, opp_controller, vocab_frame_budget) MUST
        # come from the checkpoint — as must the vocab itself (train() reads resume_state["vocab"]).
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
    auto_comment = f"chord-{cfg.max_steps // 1000}k-b{cfg.batch_size}"
    train(cfg, stats, comment=args.comment or auto_comment)


if __name__ == "__main__":
    main(tyro.cli(Args))
