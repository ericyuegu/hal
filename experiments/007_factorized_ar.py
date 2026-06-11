"""Factorized autoregressive action-chunk policy (fixes 005's conditional independence).

005's classification head is **not** autoregressive: given the backbone context, every one
of the ``L_chunk`` frames and every channel group is predicted in ONE forward, so they are
conditionally independent. The head sees no action inputs — neither teacher-forced targets
nor sampled values — so chaining the heads on logits would add nothing (logits are
deterministic given context; they carry no stochastic coupling). With ~65% frame-to-frame
action persistence and median button holds of 5-7 frames, sampling each frame/group
independently shreds temporal and cross-channel coherence.

007 keeps 005's backbone verbatim and replaces the head with a TRUE autoregression over
**factorized channel-group sub-tokens** — AR across frames AND across the groups within a
frame, each sub-token conditioned on the *realized* values of all earlier ones (teacher-forced
targets at train, sampled outcomes at decode).

Sub-token order per frame (fixed): **buttons (256-way combo) → main stick (65) → c-stick (9)
→ triggers (joint L×R = 25)**. Buttons go first so the trigger analog can condition on the
digital L/R click — empirically ``P(trigger==1.0 | click) ≈ 0.75``, strong coupling — and
triggers go last so they see everything. The full product space stays reachable: no
data-derived vocabulary, so this is the **RL-safe** variant — policy-gradient exploration can
express button/stick combinations absent from human data, unlike a frozen observed-chord
vocab.

The head is a small causal Transformer decoder over ``L_chunk * 4`` sub-token positions
(group-specific embedding tables for the previous realized class + learned group embeddings +
chunk-frame position embeddings + projected backbone cond broadcast onto every token; learned
BOS at the very first position). Group-specific output linears emit 256/65/9/25 logits at
their own positions. Loss = sum of the four group CEs (nats), ``.mean()``, keeping 005's
``norm_div`` divisor. Decode is 64 sequential steps sampling (or argmax) each sub-token given
all previously realized ones — no KV cache (head is tiny: 2 layers, <=64 tokens), but the
backbone is encoded ONCE per replan, never per step.

Closed-loop decode is sampled by default (``cfg.decode`` / ``cfg.decode_temp``, default
``sample`` @ 1.0): greedy argmax collapses an autoregressive policy to a do-nothing fixed
point in closed loop, so sampling is the deployed controller; argmax stays for the
deterministic recon metric and as a test-time override.

Run:
    python experiments/007_factorized_ar.py                          # train (closed-loop eval samples)
    python experiments/007_factorized_ar.py --cfg.opp-controller True # roofline: cheat on opp controller
    python experiments/007_factorized_ar.py --eval <ckpt>                       # eval at the trained decode
    python experiments/007_factorized_ar.py --eval <ckpt> --eval-temp 0.7       # test-time temperature sweep
    python experiments/007_factorized_ar.py --eval <ckpt> --eval-decode argmax  # re-eval greedily
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

# --- factorized sub-token groups (fixed AR order within a frame) -------------
# Buttons first so the trigger analog can condition on the digital L/R click; triggers last so
# they see everything. Vocab sizes come from the scoring discretizers (no data-derived vocab —
# RL-safe: the full product space stays reachable).
_N_TRIG_CENTERS = scoring.TRIGGER_CENTERS.shape[0]  # 5 per shoulder
_GROUP_NAMES: tuple[str, ...] = ("buttons", "main_stick", "c_stick", "triggers")
_GROUP_VOCABS: tuple[int, ...] = (
    scoring.N_BUTTON_COMBOS,  # 256
    scoring.STICK_CLUSTER_CENTERS_MAIN.shape[0],  # 65
    scoring.STICK_CLUSTER_CENTERS_C.shape[0],  # 9
    _N_TRIG_CENTERS * _N_TRIG_CENTERS,  # 25 (joint L*5 + R)
)
N_GROUPS = len(_GROUP_NAMES)  # 4
_BUTTONS_G, _MAIN_G, _C_G, _TRIG_G = range(N_GROUPS)


# %%
@dataclass
class TrainConfig:
    # model backbone (identical to 005)
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    # autoregressive sub-token head (small causal Transformer over the L_chunk*4 sub-tokens)
    d_head: int = 128
    n_head_layers: int = 2
    head_heads: int = 4
    head_ff: int = 512
    # matchup conditioning (schema v4): per-player character + global stage embeddings — the
    # signal the old AR model had that v3 dropped. Needs a v4 MDS (stage + p{1,2}_character
    # columns) + the v4 closed-loop obs injection; off => identical to the v3-data model.
    cond_char_stage: bool = True
    char_vocab: int = 64  # slp/libmelee Character ids (0..~32), padded
    char_dim: int = 12
    stage_vocab: int = 64  # libmelee Stage values, padded
    stage_dim: int = 8
    # Roofline cheat: concat the OPPONENT's 14-channel controller history onto every context
    # token. A deliberate information cheat — a human cannot see the opponent's controller — to
    # measure headroom only, never a deployable model. Tagged ``-oppc`` so cheat runs are obvious.
    opp_controller: bool = False
    # decode (closed-loop controller + recon proxy). Sampling is the DEFAULT: argmax greedily
    # picks the mode of P(action | recent inputs), which for an autoregressive policy collapses
    # to a "do nothing" fixed point in closed loop (it feeds neutral back to itself; the flow
    # baselines escape this via their noise draw). argmax stays available for the deterministic
    # recon metric and as a test-time override.
    decode: str = "sample"  # "sample" (temperature) | "argmax" (greedy)
    decode_temp: float = 1.0  # softmax temperature for sample decode (higher = more random)
    # training-stability divisor on the joint-NLL scalar ONLY; reported likelihood is never divided by it
    norm_div: float = 1.0
    # Seeds model init and the dataloader's window + ego-port sampling.
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
        f"fact{cs}{oppc}"
        f"-d{cfg.d_model}-L{cfg.n_layers}-Lc{cfg.L_ctx}-Lk{cfg.L_chunk}"
        f"-dh{cfg.d_head}-hl{cfg.n_head_layers}-tp{cfg.train_positions}"
    )


# %%
@jaxtyped(typechecker=beartype)
def quantize_groups(
    main_centers: Float[Tensor, "n_main 2"],
    c_centers: Float[Tensor, "n_c 2"],
    trig_centers: Float[Tensor, " n_trig"],
    actions: Float[Tensor, "*batch d_action"],
) -> Int[Tensor, "*batch n_groups"]:
    """Quantize a raw ``A_DIM`` action vec to the four group class indices, in AR order
    ``(buttons, main_stick, c_stick, triggers)``. Sticks use the nearest joint-2D cluster, the
    triggers a joint ``tl * n_trig + tr`` of the two per-shoulder 1D centers, buttons the 256-way
    combo bitmask. The decode inverse is ``dequantize_groups``; values that already sit on the
    center grids round-trip exactly."""
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
    """Inverse of ``quantize_groups``: the four group class indices → a raw ``A_DIM`` action vec
    (``[-1,1]`` sticks, ``[0,1]`` triggers, ``{0,1}`` buttons). Trigger joint index unpacks to
    ``(tl, tr) = divmod(idx, n_trig)``; buttons unpack the combo bitmask back to the 8 bits."""
    n_trig = trig_centers.shape[0]
    btn = scoring.combo_to_buttons(idx[..., _BUTTONS_G])  # [*batch, 8]
    main = scoring.cluster_to_xy(idx[..., _MAIN_G], main_centers)  # [*batch, 2]
    c = scoring.cluster_to_xy(idx[..., _C_G], c_centers)  # [*batch, 2]
    tl = scoring.center_to_value(idx[..., _TRIG_G] // n_trig, trig_centers)
    tr = scoring.center_to_value(idx[..., _TRIG_G] % n_trig, trig_centers)
    trig = torch.stack([tl, tr], dim=-1)  # [*batch, 2]
    return torch.cat([main, c, trig, btn], dim=-1)


# %%
class FactorizedARPolicy(nn.Module):
    """Causal backbone (identical to 005) + autoregressive factorized sub-token head.

    The **backbone** is a decoder-style Transformer over the L_ctx context tokens under a causal
    mask, so ``hidden[i]`` depends only on positions ``<= i``. The **head** is a small causal
    Transformer over the ``L_chunk * N_GROUPS`` sub-token stream: each sub-token's input is the
    previous realized sub-token's class embedding (group-specific tables; learned BOS at the very
    first position) + a learned group embedding + a chunk-frame position embedding + the projected
    backbone cond (broadcast). Group-specific output linears emit per-group logits at their own
    positions. Training teacher-forces ONE head forward per supervised context position; decode
    walks the 64 positions sequentially, feeding each realized class back in."""

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.L_ctx = cfg.L_ctx
        self.L_chunk = cfg.L_chunk
        self.n_heads = cfg.n_heads
        self.opp_controller = cfg.opp_controller
        d = cfg.d_model
        dh = cfg.d_head

        if cfg.decode not in ("argmax", "sample"):
            raise ValueError(f"decode must be argmax|sample, got {cfg.decode!r}")
        if not cfg.decode_temp > 0:
            raise ValueError(f"decode_temp must be > 0, got {cfg.decode_temp}")
        if not cfg.norm_div > 0:
            raise ValueError(f"norm_div must be > 0, got {cfg.norm_div}")

        self.cat_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocab, dim) for name, (vocab, dim) in CAT_FEATURES.items()}
        )
        n_float = len(FLOAT_FEATURES)
        n_mask = len(FLOAT_FEATURES)
        n_cat = sum(dim for _, dim in CAT_FEATURES.values())
        per_player_dim = n_float + n_mask + n_cat
        per_frame_in_dim = 2 * per_player_dim + A_DIM  # ego + opp + ego controller history
        if cfg.opp_controller:
            per_frame_in_dim += A_DIM  # roofline cheat: opp controller history too
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

        # --- AR sub-token head ---
        self.cond_proj = nn.Linear(d, dh)
        self.group_emb = nn.Embedding(N_GROUPS, dh)  # which channel-group a sub-token position is
        self.chunk_pos_emb = nn.Embedding(self.L_chunk, dh)  # which chunk frame k
        # Previous realized class embedding, one table per group (the class spaces are disjoint).
        self.group_in_embeds = nn.ModuleList([nn.Embedding(v, dh) for v in _GROUP_VOCABS])
        self.bos = nn.Parameter(torch.zeros(dh))  # input at the very first sub-token (no predecessor)
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
        self.group_out = nn.ModuleList([nn.Linear(dh, v) for v in _GROUP_VOCABS])

        # Stick/trigger center grids (registered so they move with .to()/.cuda() and serialize).
        self.register_buffer("main_centers", scoring.STICK_CLUSTER_CENTERS_MAIN.clone())
        self.register_buffer("c_centers", scoring.STICK_CLUSTER_CENTERS_C.clone())
        self.register_buffer("trig_centers", scoring.TRIGGER_CENTERS.clone())
        # Causal mask over the L_chunk*N_GROUPS flattened sub-token stream (registered, never re-allocated).
        S = self.L_chunk * N_GROUPS
        self.register_buffer("_head_mask", torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1))
        # Per-position group ids (k repeats across the 4 groups → [S]) for the group/output gather.
        self.register_buffer("_pos_group", torch.arange(S) % N_GROUPS)
        self.register_buffer("_pos_frame", torch.arange(S) // N_GROUPS)

    # --- backbone (verbatim from 005, plus optional opp-controller cheat) ---
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

    def _controller_history(self, features: dict[str, Tensor], prefix: str) -> Tensor:
        return torch.cat([features[f"{prefix}_{ch}"][..., None] for ch in ACTION_CHANNELS], dim=-1)

    def _context_tokens(self, features: dict[str, Tensor]) -> Float[Tensor, "B L_ctx d_model"]:
        ego = self._per_player_features(features, "ego")
        opp = self._per_player_features(features, "opp")
        hist = self._controller_history(features, "ego")
        parts = [ego, opp, hist]
        if self.opp_controller:
            parts.append(self._controller_history(features, "opp"))  # deliberate cheat (see cfg.opp_controller)
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

    # --- AR head ---
    @jaxtyped(typechecker=beartype)
    def _input_stream(
        self, cond: Float[Tensor, "N d_model"], prev_idx: Int[Tensor, "N n_prev"], s_out: int
    ) -> Float[Tensor, "N s_out d_head"]:
        """Build the ``s_out`` head-input positions: position 0 is the learned BOS; position
        ``p`` (1..s_out-1) embeds the realized class of stream sub-token ``p-1`` via that
        sub-token's group table. ``prev_idx`` is the row-major ``(frame, group)`` realized-class
        prefix (``[N, n_prev]`` in stream order; only its first ``s_out-1`` entries are consumed as
        inputs — the last realized class is never an input since nothing follows it). A learned
        group embedding (which channel-group) + chunk-frame position embedding + the projected
        backbone cond are added to every position."""
        N = cond.shape[0]
        n_in = s_out - 1  # realized classes consumed as inputs (the rest are outputs only)
        emb = self.bos.to(cond.dtype).expand(N, 1, -1).clone()  # [N, 1, d_head] BOS at position 0
        if n_in > 0:
            classes = prev_idx[:, :n_in]
            groups = self._pos_group[:n_in]  # group of stream sub-token p (= p % N_GROUPS)
            per = torch.empty(N, n_in, emb.shape[-1], device=cond.device, dtype=cond.dtype)
            for g in range(N_GROUPS):
                sel = groups == g
                if sel.any():
                    per[:, sel] = self.group_in_embeds[g](classes[:, sel]).to(cond.dtype)
            emb = torch.cat([emb, per], dim=1)  # [N, s_out, d_head]
        emb = emb + self.group_emb(self._pos_group[:s_out]) + self.chunk_pos_emb(self._pos_frame[:s_out])
        return emb + self.cond_proj(cond)[:, None, :]

    @jaxtyped(typechecker=beartype)
    def teacher_forced_logits(
        self, cond: Float[Tensor, "N d_model"], tgt_idx: Int[Tensor, "N L_chunk n_groups"]
    ) -> Float[Tensor, "N L_chunk n_groups max_vocab"]:
        """ONE forward producing every sub-token's logits under teacher forcing: feed the realized
        target classes as the AR inputs, run the causal head over all ``L_chunk * N_GROUPS``
        positions, and apply each position's group output linear. Returned logits are right-padded
        to ``max(_GROUP_VOCABS)`` on the class axis (unused entries are ``-inf``) so the four groups
        stack into one tensor; the loss/decode index only each group's real ``_GROUP_VOCABS[g]``."""
        N = cond.shape[0]
        H = self.L_chunk
        S = H * N_GROUPS
        x = self._input_stream(cond, tgt_idx.reshape(N, S), S)  # [N, S, d_head], stream order
        h = self.head_encoder(x, mask=self._head_mask)  # [N, S, d_head], causal
        max_vocab = max(_GROUP_VOCABS)
        out = h.new_full((N, H, N_GROUPS, max_vocab), float("-inf"))
        h = h.reshape(N, H, N_GROUPS, -1)
        for g in range(N_GROUPS):
            out[:, :, g, : _GROUP_VOCABS[g]] = self.group_out[g](h[:, :, g])
        return out


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
    """Per-element categorical cross-entropy in nats. ``logits [..., K]``, ``idx [...]`` →
    ``[...]`` (the last logit dim is the class axis). The ``-inf`` padding on the unused class
    slots contributes nothing (its softmax mass is exactly 0)."""
    flat = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), idx.reshape(-1), reduction="none")
    return flat.reshape(idx.shape)


def _quantize(model: FactorizedARPolicy, actions: Tensor) -> Tensor:
    return quantize_groups(model.main_centers, model.c_centers, model.trig_centers, actions)


def _dequantize(model: FactorizedARPolicy, idx: Tensor) -> Tensor:
    return dequantize_groups(model.main_centers, model.c_centers, model.trig_centers, idx)


def _component_nll(model: FactorizedARPolicy, logits: Tensor, tgt_idx: Tensor) -> dict[str, Tensor]:
    """The factorized joint NLL split into the four groups, each ``[N, H]`` **nats**. ``logits``
    is ``[N, H, N_GROUPS, max_vocab]`` (from ``teacher_forced_logits``); ``tgt_idx`` the
    ``[N, H, N_GROUPS]`` realized classes. Same code at train and val so the marginals exactly
    partition the joint."""
    return {name: _ce_nats(logits[:, :, g], tgt_idx[:, :, g]) for g, name in enumerate(_GROUP_NAMES)}


def _select(
    model: FactorizedARPolicy,
    batch: TrainBatch,
    *,
    multi: bool,
    max_positions: int = -1,
    gen: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Pick the supervised context positions (verbatim selection from 005), run the teacher-forced
    head once over the flattened ``N`` positions, and return ``(logits [N,H,4,max_vocab],
    tgt_idx [N,H,4])``. ``multi=False`` supervises only the last position (matches inference; used
    by val)."""
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
    tgt_idx = _quantize(model, tgt)  # [N, H, 4]
    return model.teacher_forced_logits(cond, tgt_idx), tgt_idx


def action_loss(
    model: FactorizedARPolicy,
    batch: TrainBatch,
    *,
    multi: bool = True,
    max_positions: int = -1,
    gen: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """Per-group joint NLL (nats, ``[N, H]``) over the supervised positions. Sum the four
    components and ``.mean()`` for the training scalar; feed to ``nll_breakdown`` for the
    group/horizon split."""
    logits, tgt_idx = _select(model, batch, multi=multi, max_positions=max_positions, gen=gen)
    return _component_nll(model, logits, tgt_idx)


@torch.no_grad()
def decode(
    model: FactorizedARPolicy,
    ctx: Context,
    *,
    mode: str = "argmax",
    temp: float = 1.0,
    gen: torch.Generator | None = None,
) -> Float[Tensor, "B L_chunk d_action"]:
    """One action chunk per sample from the LAST context position, in raw action ranges
    (``[-1,1]`` sticks, ``[0,1]`` triggers, ``{0,1}`` buttons).

    This is the AUTOREGRESSIVE decoder: it walks the ``L_chunk * N_GROUPS`` sub-token stream
    sequentially, sampling (``"sample"`` — the default; draw from the ``temp``-scaled softmax) or
    taking the mode (``"argmax"`` — deterministic, for the recon metric) of each sub-token's logits
    GIVEN the realized classes of all earlier sub-tokens. The backbone is encoded ONCE (not per
    step); the tiny head is recomputed each step (no KV cache: <=64 tokens, clarity over
    cleverness). Deliberately separate from the flow policies' integrate-from-noise decoder."""
    if mode not in ("argmax", "sample"):
        raise ValueError(f"decode mode must be argmax|sample, got {mode!r}")
    cond = model.encode_context(ctx)[:, -1, :]  # [B, d_model]
    B = cond.shape[0]
    H = model.L_chunk
    S = H * N_GROUPS
    realized = torch.empty(B, S, dtype=torch.long, device=cond.device)  # flattened (frame, group) classes

    def pick(logits: Tensor) -> Tensor:
        """Categorical choice over the last logit dim at one stream position: greedy mode, or a
        ``temp``-scaled draw. ``logits [B, K]`` → ``[B]``."""
        if mode == "argmax":
            return logits.argmax(-1)
        probs = F.softmax(logits / temp, dim=-1)
        return torch.multinomial(probs, 1, generator=gen).squeeze(-1)

    for t in range(S):
        g = t % N_GROUPS
        x = model._input_stream(cond, realized[:, :t], t + 1)  # [B, t+1, d_head]: BOS + realized 0..t-1
        h = model.head_encoder(x, mask=model._head_mask[: t + 1, : t + 1])  # causal over the known prefix
        logits = model.group_out[g](h[:, t])  # [B, _GROUP_VOCABS[g]] — the next sub-token
        realized[:, t] = pick(logits)

    idx = realized.reshape(B, H, N_GROUPS)
    return _dequantize(model, idx)


def make_policy(
    model: FactorizedARPolicy,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    device: str = DEVICE,
    decode_mode: str | None = None,
    decode_temp: float | None = None,
    s: int | None = None,
) -> RecedingHorizon:
    """Fresh open-loop closed-loop policy for one eval wave (rolling state must not leak).
    The driver replans every ``s`` frames (default 1: per-frame replanning — 16 frames of
    open-loop execution is longer than human reaction time), decoding one chunk from the
    last context position and executing its first ``s`` actions. ``decode_mode``/``decode_temp``
    override ``cfg`` for a test-time decode sweep without retraining; ``s`` overrides the
    execution horizon to probe control frequency. Closed-loop sampling draws fresh randomness each
    replan (``gen=None``)."""
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


def nll_breakdown(comps: dict[str, Tensor]) -> dict[str, float]:
    """Group + horizon marginals (bits) of the joint NLL, from the per-group ``[N,H]`` nats
    components. ``modality/<name>``: mean over positions+frames. ``horizon/frame_kk``: mean over
    positions+groups at chunk position k."""
    out = {f"modality/{name}": (c.mean().item() / _LN2) for name, c in comps.items()}
    per_frame = sum(comps.values()).mean(dim=0)  # [H]
    for k in range(per_frame.shape[0]):
        out[f"horizon/frame_{k + 1:02d}"] = per_frame[k].item() / _LN2
    return out


@torch.no_grad()
def val_metrics(model: FactorizedARPolicy, val_cache: list[TrainBatch], cfg: TrainConfig) -> dict[str, float]:
    """Last-position (``multi=False``, inference-matched) proper-scoring metrics over the cached
    val batches. Concatenates per-element tensors then reduces once, so the means are exactly
    sample-weighted."""
    was_training = model.training
    model.eval()
    comps_cat: dict[str, list[Tensor]] = {}
    btn_probs: list[Tensor] = []
    btn_tgts: list[Tensor] = []
    multipress: list[Tensor] = []
    for batch in val_cache:
        logits, tgt_idx = _select(model, batch, multi=False)
        comps = _component_nll(model, logits, tgt_idx)
        for k, v in comps.items():
            comps_cat.setdefault(k, []).append(v)
        # Buttons scored as a proper Bernoulli model via the 256-way combo marginals.
        btn_probs.append(scoring.combo_marginal_probs(logits[:, :, _BUTTONS_G, : scoring.N_BUTTON_COMBOS]))
        tgt_btn = _dequantize(model, tgt_idx)[..., _N_CONT:]  # realized button bits from the combo class
        btn_tgts.append(tgt_btn)
        multipress.append((tgt_btn > 0.5).sum(-1) >= 2)
    comps = {k: torch.cat(v) for k, v in comps_cat.items()}
    out = nll_breakdown(comps)
    out = {f"loss/{k}": v for k, v in out.items()}
    out["action_nll_bits_per_frame"] = sum(comps.values()).mean().item() / _LN2
    out["cont_discrete_bits"] = (comps["main_stick"] + comps["c_stick"] + comps["triggers"]).mean().item() / _LN2

    logloss, brier = scoring.bernoulli_scores_from_probs(torch.cat(btn_probs), torch.cat(btn_tgts))
    out["buttons/logloss_bits"] = logloss.item()
    out["buttons/brier"] = brier.item()
    out["buttons/multipress_rate"] = torch.cat(multipress).float().mean().item()
    if was_training:
        model.train()
    return out


@torch.no_grad()
def recon_metrics(
    model: FactorizedARPolicy,
    val_cache: list[TrainBatch],
    *,
    mode: str,
    temp: float = 1.0,
    gen: torch.Generator | None = None,
) -> dict[str, float]:
    """Sample-space reconstruction proxy: decode a chunk and score it vs ground truth.
    ``mode="argmax"`` is the deterministic controller proxy; ``"sample"`` the distributional one
    at ``temp`` — pass ``cfg.decode_temp`` so this matches the deployed policy. Buttons → acc + F1
    @ decode; continuous → MAE."""
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
    model: FactorizedARPolicy,
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
        tags=["factorized_ar", f"d{cfg.d_model}", f"tp{cfg.train_positions}"]
        + (["oppc"] if cfg.opp_controller else []),
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
    model = FactorizedARPolicy(cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    if wandb.run is not None:
        wandb.run.summary["model/num_params"] = n_params
    print(f"[model] {_model_tag(cfg)}  num_params={n_params / 1e6:.2f}M", flush=True)
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
        breakdown = nll_breakdown({k: torch.cat(v) for k, v in comps_acc.items()})
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
            _save("latest.pt", step)
        if cfg.val_every > 0 and step > 0 and step % cfg.val_every == 0:
            vm = _log_val(step)
            print(
                f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: "
                f"action_nll {vm['action_nll_bits_per_frame']:.3f} btn_logloss {vm['buttons/logloss_bits']:.3f}",
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
def _load_ckpt(ckpt_path: str) -> tuple[FactorizedARPolicy, TrainConfig, dict[str, FeatureStats], dict]:
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = TrainConfig(**state["cfg"])
    model = FactorizedARPolicy(cfg).to(DEVICE)
    model.load_state_dict(state["model"])
    model.eval()
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    return model, cfg, stats, state


def val_report(ckpt_path: str, *, n_batches: int = 24) -> None:
    """D3 diagnostic: how well does a trained checkpoint FIT the human val data (teacher-forced,
    no emulator)? Strong fit + weak closed-loop winrate ⇒ the bottleneck is the data's skill
    ceiling / distribution shift, not optimization. Prints proper-scoring NLL plus sample-space
    reconstruction (button F1, continuous MAE) at the deployed ``sample`` decode and at ``argmax``."""
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
    print(f"\n[d3] {ckpt_path}  step={state['step']}  {frames} val frames  decode_temp={cfg.decode_temp}", flush=True)
    print(f"[d3] action_nll_bits_per_frame = {vm['action_nll_bits_per_frame']:.3f}", flush=True)
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
    for k in sorted(k for k in vm if k.startswith("loss/modality/")):
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
    """D1 diagnostic: closed-loop control-frequency sweep on FD vs lvl-9 CPU, WITHOUT retraining.

    A trained chunk model decodes ``L_chunk`` actions per replan; the execution horizon ``s``
    controls how many it commits before replanning. ``s=L_chunk`` is the full-chunk open-loop extreme;
    ``s=1`` replans every frame using only the next-frame prediction (the old AR every-frame
    regime). A large ``stocks_taken`` gap from ``s=1`` to ``s=L_chunk`` implicates control
    frequency rather than model quality. Same checkpoint, same decode."""
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
    e.g. ``--cfg.opp-controller True --cfg.decode argmax``."""

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
        # experiment-identity knobs (train_positions, opp_controller, dims) MUST come from the
        # checkpoint so a resume can't silently flip the ablation.
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
    auto_comment = f"fact-{cfg.max_steps // 1000}k-b{cfg.batch_size}"
    train(cfg, stats, comment=args.comment or auto_comment)


if __name__ == "__main__":
    main(tyro.cli(Args))
