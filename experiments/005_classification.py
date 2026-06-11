"""Classification action-chunk policy (ablates loss type vs. flow matching).

Identical to ``003_multi_position.py`` — same causal backbone, same per-position
multi-position supervision, same ``train_positions`` gradient-density knob — except the
per-position *flow* head is replaced by a per-position *classification* head. Holding
architecture and gradient density fixed isolates the BCE/CE-vs-MSE/flow learning dynamic:
sweep ``train_positions ∈ {1, 64}`` on both this file and 003 for the architecture-matched
2×2. (001's unified ``[ctx|chunk]`` Transformer is a different architecture, not the
sparse control.)

The head emits per-frame categorical logits in ONE forward — no noise, no flow time, no
Euler integration. Buttons are a 256-way joint categorical over the full 8-bit combo bitmask
(``combo``, primary — every co-press representable, conflicting presses impossible by
construction), 8 independent Bernoullis (``multi_label`` — marginals only, can emit incoherent
co-presses), or a 9-way "none-or-one" categorical (``single_label`` — diagnostic, dishonest on
the ``--audit`` multipress frames). Continuous channels are per-axis bins
(``naive_bins``) or joint-2D stick clusters + hand-tuned 1D trigger centers
(``stick_clusters``); the discretizers live in ``hal.training.scoring`` so targets, decode,
and metrics share one quantizer.

Metrics are proper scoring rules in bits (``hal.training.scoring``) named so nothing
implies invalid cross-family comparability: ``val/action_nll_bits_per_frame`` is the
classifier's own mixed objective; ``val/cont_density_bits_per_dim`` (naive only) is the
density-corrected continuous dim that lines up with a flow model's PF-ODE bits/dim;
``val/buttons/{logloss_bits,brier}`` score the buttons as a proper Bernoulli model.

Closed-loop decode is sampled with a temperature knob (``cfg.decode`` / ``cfg.decode_temp``,
default ``sample`` @ 1.0): greedy argmax collapses an autoregressive policy to a do-nothing
fixed point in closed loop, so sampling is the default controller; argmax stays for the
deterministic recon metric. This is the AR analog of the flow policies' integrate-from-noise
decode and is kept deliberately separate.

Run:
    python experiments/005_classification.py --audit                 # quantization report, no GPU train
    python experiments/005_classification.py                         # train (closed-loop eval samples)
    python experiments/005_classification.py --eval <ckpt>                       # eval at the trained decode
    python experiments/005_classification.py --eval <ckpt> --eval-temp 0.7       # test-time temperature sweep
    python experiments/005_classification.py --eval <ckpt> --eval-decode argmax  # re-eval an old run greedily
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


# %%
@dataclass
class TrainConfig:
    # model backbone (identical to 003)
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    # per-position classification head (small Transformer over the L_chunk query tokens)
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
    # output factorization (the variant axis)
    # "combo" (256-way joint bitmask, primary) | "multi_label" (8x BCE, marginals-only ablation)
    # | "single_label" (9-way none-or-one CE, diagnostic — dishonest on multipress frames)
    button_head: str = "combo"
    continuous_head: str = "naive_bins"  # "naive_bins" | "stick_clusters"
    n_bins: int = 21  # per continuous channel in naive_bins mode (unused in cluster mode); odd centers neutral
    # decode (closed-loop controller + recon proxy). Sampling is the DEFAULT: argmax greedily
    # picks the mode of P(action | recent inputs), which for an autoregressive policy collapses
    # to a "do nothing" fixed point in closed loop (it feeds neutral back to itself; the flow
    # baselines escape this via their noise draw). argmax stays available for the deterministic
    # recon metric and as a test-time override.
    decode: str = "sample"  # "sample" (temperature) | "argmax" (greedy)
    decode_temp: float = 1.0  # softmax/sigmoid temperature for sample decode (higher = more random)
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
    return (
        f"cls-{cfg.button_head}-{cfg.continuous_head}-b{cfg.n_bins}{cs}"
        f"-d{cfg.d_model}-L{cfg.n_layers}-Lc{cfg.L_ctx}-Lk{cfg.L_chunk}-tp{cfg.train_positions}"
    )


# %%
class ClassifierPolicy(nn.Module):
    """Causal backbone (identical to 003) + per-position classification head.

    The **backbone** is a decoder-style Transformer over the L_ctx context tokens under a
    causal mask, so ``hidden[i]`` depends only on positions ``<= i``. The **head** denoise-
    free: from a single backbone hidden vector it predicts the next-H-frame action chunk as
    per-frame categorical logit groups in one forward (no noise / flow-time / integration).
    Training supervises a chunk at every context position (``action_loss``); inference reads
    only the last position's hidden vector (``decode``)."""

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.L_ctx = cfg.L_ctx
        self.L_chunk = cfg.L_chunk
        self.n_heads = cfg.n_heads
        self.button_head_kind = cfg.button_head
        self.continuous_head_kind = cfg.continuous_head
        self.n_bins = cfg.n_bins
        d = cfg.d_model
        dh = cfg.d_head

        if cfg.button_head not in ("combo", "multi_label", "single_label"):
            raise ValueError(f"button_head must be combo|multi_label|single_label, got {cfg.button_head!r}")
        if cfg.continuous_head not in ("naive_bins", "stick_clusters"):
            raise ValueError(f"continuous_head must be naive_bins|stick_clusters, got {cfg.continuous_head!r}")
        if cfg.decode not in ("argmax", "sample"):
            raise ValueError(f"decode must be argmax|sample, got {cfg.decode!r}")
        if not cfg.n_bins > 0:
            raise ValueError(f"n_bins must be > 0, got {cfg.n_bins}")
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
        # Matchup conditioning: a shared per-player character embedding (ego + opp) + one
        # global stage embedding, concatenated onto every context token.
        self.cond_char_stage = cfg.cond_char_stage
        if cfg.cond_char_stage:
            self.char_emb = nn.Embedding(cfg.char_vocab, cfg.char_dim)
            self.stage_emb = nn.Embedding(cfg.stage_vocab, cfg.stage_dim)
            per_frame_in_dim += 2 * cfg.char_dim + cfg.stage_dim

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
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers, enable_nested_tensor=False)

        # --- per-position head (operates on flattened B*T positions) ---
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

        # --- output logit groups ---
        self.cont_binspecs = scoring.cont_binspecs(cfg.n_bins)
        if cfg.continuous_head == "naive_bins":
            self.cont_head = nn.Linear(dh, _N_CONT * cfg.n_bins)
        else:
            # Cluster mode: joint-2D heads for the sticks + a 1D hand-tuned trigger-center head
            # per shoulder (the uniform trigger bins waste mass on the empty sub-deadzone range).
            self.register_buffer("main_centers", scoring.STICK_CLUSTER_CENTERS_MAIN.clone())
            self.register_buffer("c_centers", scoring.STICK_CLUSTER_CENTERS_C.clone())
            self.register_buffer("trig_centers", scoring.TRIGGER_CENTERS.clone())
            self.main_head = nn.Linear(dh, self.main_centers.shape[0])
            self.c_head = nn.Linear(dh, self.c_centers.shape[0])
            self.trig_head = nn.Linear(dh, 2 * self.trig_centers.shape[0])
        button_out = {
            "combo": scoring.N_BUTTON_COMBOS,
            "multi_label": _N_BUTTONS,
            "single_label": _N_BUTTONS + 1,
        }[cfg.button_head]
        self.button_head = nn.Linear(dh, button_out)

    # --- backbone (verbatim from 003) ---
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

    def _context_tokens(self, features: dict[str, Tensor]) -> Float[Tensor, "B L_ctx d_model"]:
        ego = self._per_player_features(features, "ego")
        opp = self._per_player_features(features, "opp")
        hist = self._ego_history_features(features)
        parts = [ego, opp, hist]
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

    # --- classification head ---
    def logits(self, cond: Float[Tensor, "N d_model"]) -> dict[str, Tensor]:
        """Per-position chunk logits from one backbone hidden vector. ``N`` flattens whatever
        positions are decoded (B*T valid positions at train, B at inference). Returns the
        per-group logit tensors; loss/decode iterate them per ``button_head``/``continuous_head``."""
        H = self.L_chunk
        N = cond.shape[0]
        chunk = self.cond_proj(cond)[:, None, :] + self.chunk_pos_emb.weight[None, :H, :]
        h = self.head_encoder(chunk)  # [N, H, d_head]
        out: dict[str, Tensor] = {}
        if self.continuous_head_kind == "naive_bins":
            out["cont"] = self.cont_head(h).reshape(N, H, _N_CONT, self.n_bins)
        else:
            out["main"] = self.main_head(h)
            out["c"] = self.c_head(h)
            out["trig"] = self.trig_head(h).reshape(N, H, 2, self.trig_centers.shape[0])
        out["buttons"] = self.button_head(h)
        return out


# %%
def _position_targets(ctx: Context, target: Tensor, H: int) -> tuple[Tensor, Tensor]:
    """For every context position ``i``, the next-H-action target chunk + a validity mask
    (verbatim from 003). ``A_full[:, :L_ctx] = stack_actions(ctx.features)``,
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
    ``[...]`` (the last logit dim is the class axis)."""
    flat = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), idx.reshape(-1), reduction="none")
    return flat.reshape(idx.shape)


def _component_nll(model: ClassifierPolicy, logits: dict[str, Tensor], tgt: Tensor) -> dict[str, Tensor]:
    """The factorized joint NLL split into ``{main_stick, c_stick, triggers, buttons}``,
    each ``[N, H]`` **nats** (summed over that modality's channels). Same code at train and
    val so the marginals exactly partition the joint."""
    cont_tgt, btn_tgt = tgt[..., :_N_CONT], tgt[..., _N_CONT:]
    comps: dict[str, Tensor] = {}
    if model.continuous_head_kind == "naive_bins":
        ce = _ce_nats(logits["cont"], scoring.bins_to_idx(cont_tgt, model.cont_binspecs))  # [N,H,6]
        comps["main_stick"] = ce[..., 0] + ce[..., 1]
        comps["c_stick"] = ce[..., 2] + ce[..., 3]
        comps["triggers"] = ce[..., 4] + ce[..., 5]
    else:
        comps["main_stick"] = _ce_nats(logits["main"], scoring.nearest_cluster(cont_tgt[..., 0:2], model.main_centers))
        comps["c_stick"] = _ce_nats(logits["c"], scoring.nearest_cluster(cont_tgt[..., 2:4], model.c_centers))
        trig_idx = scoring.nearest_center(cont_tgt[..., 4:6], model.trig_centers)  # [N,H,2]
        tl = _ce_nats(logits["trig"][..., 0, :], trig_idx[..., 0])
        tr = _ce_nats(logits["trig"][..., 1, :], trig_idx[..., 1])
        comps["triggers"] = tl + tr
    if model.button_head_kind == "combo":
        comps["buttons"] = _ce_nats(logits["buttons"], scoring.buttons_to_combo(btn_tgt))
    elif model.button_head_kind == "multi_label":
        comps["buttons"] = F.binary_cross_entropy_with_logits(logits["buttons"], btn_tgt, reduction="none").sum(-1)
    else:
        cls, _ = scoring.buttons_to_class(btn_tgt)
        comps["buttons"] = _ce_nats(logits["buttons"], cls)
    return comps


def _select(
    model: ClassifierPolicy,
    batch: TrainBatch,
    *,
    multi: bool,
    max_positions: int = -1,
    gen: torch.Generator | None = None,
) -> tuple[dict[str, Tensor], Tensor]:
    """Pick the supervised context positions (verbatim selection from 003's ``flow_loss``),
    run the head once over the flattened ``N`` positions, and return ``(logits, tgt[N,H,A_DIM])``.
    ``multi=False`` supervises only the last position (matches inference; used by val)."""
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
    return model.logits(cond), tgt


def action_loss(
    model: ClassifierPolicy,
    batch: TrainBatch,
    *,
    multi: bool = True,
    max_positions: int = -1,
    gen: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """Per-modality joint NLL (nats, ``[N, H]``) over the supervised positions. Sum the four
    components and ``.mean()`` for the training scalar; feed to ``nll_breakdown`` for the
    modality/horizon split."""
    logits, tgt = _select(model, batch, multi=multi, max_positions=max_positions, gen=gen)
    return _component_nll(model, logits, tgt)


@torch.no_grad()
def decode(
    model: ClassifierPolicy,
    ctx: Context,
    *,
    mode: str = "argmax",
    temp: float = 1.0,
    gen: torch.Generator | None = None,
) -> Float[Tensor, "B L_chunk d_action"]:
    """One action chunk per sample from the LAST context position, in raw action ranges
    (``[-1,1]`` sticks, ``[0,1]`` triggers, ``{0,1}`` buttons).

    This is the AUTOREGRESSIVE decoder: per-frame categorical (sticks/triggers, single-label
    buttons) and Bernoulli (multi-label buttons) logits are turned into actions by either
    ``"sample"`` (draw from the ``temp``-scaled distribution — the default; higher ``temp`` =
    more random) or ``"argmax"`` (greedy mode — deterministic, for the recon metric). It is
    deliberately separate from the flow policies' decoder (which integrates from noise); the
    only shared piece is the discretizer inverse in ``hal.training.scoring``. The single
    inference path for this policy: closed-loop play and the recon proxy both call this."""
    if mode not in ("argmax", "sample"):
        raise ValueError(f"decode mode must be argmax|sample, got {mode!r}")
    cond = model.encode_context(ctx)[:, -1, :]  # [B, d_model]
    lg = model.logits(cond)

    def pick(logits: Tensor) -> Tensor:
        """Categorical choice over the last logit dim: greedy mode, or a ``temp``-scaled draw."""
        if mode == "argmax":
            return logits.argmax(-1)
        probs = F.softmax(logits / temp, dim=-1)
        flat = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1, generator=gen)
        return flat.reshape(logits.shape[:-1])

    if model.continuous_head_kind == "naive_bins":
        cont = scoring.idx_to_centers(pick(lg["cont"]), model.cont_binspecs)  # [B,H,6]
    else:
        main = scoring.cluster_to_xy(pick(lg["main"]), model.main_centers)
        c = scoring.cluster_to_xy(pick(lg["c"]), model.c_centers)
        trig = scoring.center_to_value(pick(lg["trig"]), model.trig_centers)  # [B,H,2]
        cont = torch.cat([main, c, trig], dim=-1)
    if model.button_head_kind == "combo":
        btn = scoring.combo_to_buttons(pick(lg["buttons"]))  # 256-way pick → 8 float bits
    elif model.button_head_kind == "multi_label":
        if mode == "argmax":
            btn = (lg["buttons"] > 0).float()  # sigmoid(x) > 0.5 ⇔ x > 0
        else:
            btn = torch.bernoulli(torch.sigmoid(lg["buttons"] / temp), generator=gen)
    else:
        btn = scoring.class_to_onehot(pick(lg["buttons"]), n_buttons=_N_BUTTONS)
    return torch.cat([cont, btn], dim=-1)


def make_policy(
    model: ClassifierPolicy,
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
    override ``cfg`` for a test-time decode sweep (the AR analog of the flow policies'
    ``n_flow_steps`` override) without retraining; ``s`` overrides the execution horizon to
    probe control frequency (``s=1`` ≈ every-frame control, the old AR regime). Closed-loop
    sampling draws fresh randomness each replan (``gen=None``)."""
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
    """Modality + horizon marginals (bits) of the joint NLL, from the per-modality ``[N,H]``
    nats components. ``modality/<name>``: mean over positions+frames. ``horizon/frame_kk``:
    mean over positions+modalities at chunk position k."""
    out = {f"modality/{name}": (c.mean().item() / _LN2) for name, c in comps.items()}
    per_frame = sum(comps.values()).mean(dim=0)  # [H]
    for k in range(per_frame.shape[0]):
        out[f"horizon/frame_{k + 1:02d}"] = per_frame[k].item() / _LN2
    return out


def _button_marginal_probs(model: ClassifierPolicy, button_logits: Tensor) -> Tensor:
    """P(button_k pressed) ∈ [0,1], ``[..., 8]`` — the marginalized per-button probability of
    each head: sigmoid (multi-label), the per-button softmax marginal over classes 1..8
    (single-label), or the bit-marginal of the 256-way softmax (combo). Lets all three heads be
    scored as a proper Bernoulli model of the buttons, so ``val/buttons/*`` stays comparable."""
    if model.button_head_kind == "combo":
        return scoring.combo_marginal_probs(button_logits)
    if model.button_head_kind == "multi_label":
        return torch.sigmoid(button_logits)
    return F.softmax(button_logits, dim=-1)[..., 1:]  # drop the "none" class


@torch.no_grad()
def val_metrics(model: ClassifierPolicy, val_cache: list[TrainBatch], cfg: TrainConfig) -> dict[str, float]:
    """Last-position (``multi=False``, inference-matched) proper-scoring metrics over the
    cached val batches. Concatenates per-element tensors then reduces once, so the means are
    exactly sample-weighted."""
    was_training = model.training
    model.eval()
    comps_cat: dict[str, list[Tensor]] = {}
    cont_ch_nats: list[Tensor] = []  # [M,H,6] per-channel continuous nats (naive only)
    btn_probs: list[Tensor] = []
    btn_tgts: list[Tensor] = []
    multipress: list[Tensor] = []
    for batch in val_cache:
        logits, tgt = _select(model, batch, multi=False)
        comps = _component_nll(model, logits, tgt)
        for k, v in comps.items():
            comps_cat.setdefault(k, []).append(v)
        if model.continuous_head_kind == "naive_bins":
            cont_ch_nats.append(_ce_nats(logits["cont"], scoring.bins_to_idx(tgt[..., :_N_CONT], model.cont_binspecs)))
        btn_probs.append(_button_marginal_probs(model, logits["buttons"]))
        btn_tgts.append(tgt[..., _N_CONT:])
        multipress.append(scoring.buttons_to_class(tgt[..., _N_CONT:])[1])
    comps = {k: torch.cat(v) for k, v in comps_cat.items()}
    out = nll_breakdown(comps)
    out = {f"loss/{k}": v for k, v in out.items()}
    out["action_nll_bits_per_frame"] = sum(comps.values()).mean().item() / _LN2

    if model.continuous_head_kind == "naive_bins":
        ch = torch.cat(cont_ch_nats).mean(dim=(0, 1)) / _LN2  # [6] discrete bits per channel
        density = sum(ch[i].item() + math.log2(model.cont_binspecs[i].width) for i in range(_N_CONT)) / _N_CONT
        out["cont_density_bits_per_dim"] = density
    else:
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
    model: ClassifierPolicy,
    val_cache: list[TrainBatch],
    *,
    mode: str,
    temp: float = 1.0,
    gen: torch.Generator | None = None,
) -> dict[str, float]:
    """Sample-space reconstruction proxy: decode a chunk and score it vs ground truth.
    ``mode="argmax"`` is the deterministic controller proxy; ``"sample"`` the distributional
    one (argmax collapses multimodality) at ``temp`` — pass ``cfg.decode_temp`` so this
    matches the policy actually deployed. Buttons → acc + F1 @ decode; continuous → MAE."""
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
    model: ClassifierPolicy,
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
        tags=["classification", cfg.button_head, cfg.continuous_head, f"d{cfg.d_model}", f"tp{cfg.train_positions}"],
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
    model = ClassifierPolicy(cfg).to(DEVICE)
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
def run_audit(cfg: TrainConfig, stats: dict[str, FeatureStats], *, n_batches: int = 16) -> None:
    """Quantization fidelity report over real val targets (no GPU train). Scores each
    candidate discretizer so bins/clusters are chosen on evidence, not assumed."""
    loader = make_loader(
        split=cfg.val_split,
        num_workers=0,
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
    batches = list(itertools.islice(loader, n_batches))
    if not batches:
        raise RuntimeError("val loader yielded zero batches")
    cont = torch.cat([b.target[..., :_N_CONT].reshape(-1, _N_CONT) for b in batches])  # [M,6]
    buttons = torch.cat([b.target[..., _N_CONT:].reshape(-1, _N_BUTTONS) for b in batches])  # [M,8]
    print(f"\n[audit] {cont.shape[0]} frames from {len(batches)} val batches\n", flush=True)

    def _mae(recon: Tensor, tgt: Tensor) -> float:
        return (recon - tgt).abs().mean().item()

    def _dead(idx: Tensor, n_centers: int, *, thresh: float = 0.001) -> int:
        """Number of centers holding < ``thresh`` of the mass (massively over-binned signal)."""
        occ = torch.bincount(idx, minlength=n_centers).float() / idx.numel()
        return int((occ < thresh).sum())

    print(f"  {'discretizer':<22} {'main_mae':>9} {'cstick_mae':>11} {'trig_mae':>9} {'>0.1 frac':>10} {'dead':>6}")
    for nb in (15, 21, 31):
        specs = scoring.cont_binspecs(nb)
        idx = scoring.bins_to_idx(cont, specs)
        recon = scoring.idx_to_centers(idx, specs)
        main_mae = _mae(recon[:, 0:2], cont[:, 0:2])
        c_mae = _mae(recon[:, 2:4], cont[:, 2:4])
        trig_mae = _mae(recon[:, 4:6], cont[:, 4:6])
        far = (recon[:, :4] - cont[:, :4]).abs().max(-1).values.gt(0.1).float().mean().item()
        dead = sum(_dead(idx[:, ch], nb) for ch in range(4, 6))  # both trigger channels
        print(
            f"  {'naive_bins(' + str(nb) + ')':<22} {main_mae:>9.4f} {c_mae:>11.4f} {trig_mae:>9.4f} "
            f"{far:>10.3f} {dead:>6}",
            flush=True,
        )

    # joint-2D stick clusters: their own main/c sets + the hand-tuned 1D trigger centers.
    main_c, c_c, trig_c = (
        scoring.STICK_CLUSTER_CENTERS_MAIN,
        scoring.STICK_CLUSTER_CENTERS_C,
        scoring.TRIGGER_CENTERS,
    )
    main_idx = scoring.nearest_cluster(cont[:, 0:2], main_c)
    c_idx = scoring.nearest_cluster(cont[:, 2:4], c_c)
    trig_idx = scoring.nearest_center(cont[:, 4:6], trig_c)  # [M,2]
    main_recon = scoring.cluster_to_xy(main_idx, main_c)
    c_recon = scoring.cluster_to_xy(c_idx, c_c)
    trig_recon = scoring.center_to_value(trig_idx, trig_c)
    far = (main_recon - cont[:, 0:2]).abs().max(-1).values.gt(0.1).float().mean().item()
    dead = (
        _dead(main_idx, main_c.shape[0])
        + _dead(c_idx, c_c.shape[0])
        + sum(_dead(trig_idx[:, ch], trig_c.shape[0]) for ch in range(2))
    )
    print(
        f"  {f'clusters(m{main_c.shape[0]}/c{c_c.shape[0]}/t{trig_c.shape[0]})':<22} "
        f"{_mae(main_recon, cont[:, 0:2]):>9.4f} {_mae(c_recon, cont[:, 2:4]):>11.4f} "
        f"{_mae(trig_recon, cont[:, 4:6]):>9.4f} {far:>10.3f} {dead:>6}",
        flush=True,
    )
    print(
        f"  (clusters dead@<0.1%: main {_dead(main_idx, main_c.shape[0])}/{main_c.shape[0]}  "
        f"c {_dead(c_idx, c_c.shape[0])}/{c_c.shape[0]}  "
        f"trig {sum(_dead(trig_idx[:, ch], trig_c.shape[0]) for ch in range(2))}/{2 * trig_c.shape[0]})",
        flush=True,
    )

    pressed = (buttons > 0.5).sum(-1)
    combo = scoring.buttons_to_combo(buttons)  # [M] in [0, 256)
    counts = torch.bincount(combo, minlength=scoring.N_BUTTON_COMBOS).float()
    occupied = int((counts > 0).sum())
    cov = counts.sort(descending=True).values.cumsum(0) / counts.sum()
    top16 = cov[15].item() if cov.numel() >= 16 else cov[-1].item()
    cover99 = int((cov < 0.99).sum()) + 1  # combos needed to reach 99% mass
    print(
        f"\n[audit] buttons: any-press {float((pressed >= 1).float().mean()):.3f}  "
        f"multipress(>=2) {float((pressed >= 2).float().mean()):.3f}  "
        f"(single_label is only honest if multipress is small)\n"
        f"[audit] combo (256-way joint): {occupied}/256 combos observed  "
        f"top-16 coverage {top16:.3f}  combos for 99% mass {cover99}  "
        f"(combo head is the full product space — empty classes are fine)\n",
        flush=True,
    )


# %%
def _load_ckpt(ckpt_path: str) -> tuple[ClassifierPolicy, TrainConfig, dict[str, FeatureStats], dict]:
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = TrainConfig(**state["cfg"])
    model = ClassifierPolicy(cfg).to(DEVICE)
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
    controls how many it commits before replanning. ``s=L_chunk`` is the full-chunk open-loop extreme
    (~267 ms of unreactive input); ``s=1`` replans every frame using only the next-frame prediction
    (the old AR every-frame regime). A large ``stocks_taken`` gap from ``s=1`` to ``s=L_chunk``
    implicates control frequency rather than model quality. Same checkpoint, same decode."""
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
    ``decode_temp`` override the trained cfg for this eval only (test-time decode sweep) — e.g.
    to re-evaluate an argmax-trained checkpoint with sampling."""
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
    e.g. ``--cfg.button-head single_label --cfg.continuous-head stick_clusters``."""

    cfg: TrainConfig = field(default_factory=TrainConfig)
    audit: bool = False  # quantization fidelity report (no GPU train), then exit
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
    if args.audit:
        cfg = args.cfg
        run_audit(cfg, load_consolidated_stats(Path(cfg.data_root) / "stats.json"))
        return
    if args.resume is not None:
        state = load_for_resume(args.resume, Path("runs") / args.resume, device=DEVICE)
        if state is None:
            raise SystemExit(f"no latest.pt for run {args.resume!r} (local or R2)")
        # Only pure host-scaling knobs (worker/prefetch counts) follow the current code; the
        # objective/ablation knobs (train_positions, head kinds, n_bins) are part of the
        # experiment identity and MUST come from the checkpoint — resetting train_positions
        # would silently flip a sparse (tp=1) run to the dense default.
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
    auto_comment = f"cls-{cfg.max_steps // 1000}k-b{cfg.batch_size}"
    train(cfg, stats, comment=args.comment or auto_comment)


if __name__ == "__main__":
    main(tyro.cli(Args))
