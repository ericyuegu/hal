"""Toy flow-matching policy with latency-aware K-frame bridge.

Single-file experiment. Owns: model architecture, feature schema,
preprocessing, ControllerSource impl (with rolling-history state machine),
training loop, eval mode. Shared infra (MDS dataloader, dataset stats,
gamestate→flat helper, sim eval harness) imported from ``hal.training`` /
``hal.eval``.

Tensor-dim name convention (used in jaxtyping annotations and docstrings):
    B           = batch
    L_ctx       = context length             (cfg.L_ctx)
    L_chunk     = predicted chunk length     (cfg.L_chunk)
    n_lat       = latency / bridge frames    (cfg.latency_frames)
    d_model     = hidden dim                 (cfg.d_model)
    d_action    = action vec dim (15)        (A_DIM)
    d_time      = time-embedding dim         (cfg.time_emb_dim)
    n_heads     = attention heads            (cfg.n_heads)
    d_ff        = feed-forward inner dim     (cfg.dim_feedforward)
    seq         = full encoder seq length    (L_ctx + n_lat + L_chunk)

Run:
    python experiments/toy_train.py                       # train
    python experiments/toy_train.py --eval <ckpt>         # eval a checkpoint
"""

# %%
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import math
import time
import warnings
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from beartype import beartype
from jaxtyping import Float
from jaxtyping import jaxtyped
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import wandb
from hal.data.stats import FeatureStats
from hal.eval.cross_stage import sweep_stages_self_play
from hal.eval.cross_stage import sweep_stages_vs_cpu
from hal.eval.harness import SessionConfig
from hal.fixtures import DOLPHIN_EXIAI
from hal.fixtures import ISO
from hal.fixtures import ensure
from hal.paths import EMULATOR_PATH
from hal.sim.inputs import ControllerInputsValue
from hal.training.canonical import flatten_canonical_frame
from hal.training.dataloader import make_loader
from hal.training.dataloader import relabel_ego
from hal.training.stats import consolidate_key
from hal.training.stats import load_consolidated_stats
from hal.wire import BUTTON_BITS
from hal.wire import mask_value

# NVML init noise on systems without nvidia-ml-py wired up. Surgical filter
# (not a blanket "ignore everything") so other torch warnings still surface.
warnings.filterwarnings("ignore", message="Can't initialize NVML")

torch.set_printoptions(linewidth=300)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
A_DIM = 15  # 4 sticks + 2 triggers + 9 buttons


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
    # window / chunking
    L_ctx: int = 256
    L_chunk: int = 16
    latency_frames: int = 4
    # P(zero ego history's [:k] for a random k ~ U{0..L_ctx}) per sample at train.
    # Trains the model to handle the closed-loop rolling-buffer transient where
    # ego_inputs_hist starts as L_ctx zeros and slowly fills with model outputs.
    ego_history_dropout_prob: float = 0.5
    # inference
    n_flow_steps: int = 8
    # optimization
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 15_000
    # eval cadence
    val_every: int = 500
    val_n_batches: int = 16
    eval_every: int = 2500
    eval_max_frames: int = 3600
    # data
    data_root: str = "data/processed/ranked-anonymized-1/mds"
    num_workers: int = 8
    prefetch_factor: int = 8


# %%
# Feature schema for this experiment. The classifier below routes every MDS
# column into exactly one of these buckets (or drops it).
FLOAT_FEATURES: tuple[str, ...] = (
    "position_x",
    "position_y",
    "percent",
    "shield",
    "direction",
    "hitlag_left",
)

# (vocab_size, embed_dim) per categorical feature. action vocab rounded up
# from libmelee's ~395 to a safe power-of-two-ish.
CAT_FEATURES: dict[str, tuple[int, int]] = {
    "action": (512, 32),
    "stock": (5, 8),
    "jumps_used": (9, 8),
    "hurtbox_state": (4, 8),
    "airborne": (2, 4),
}

# Canonical ordering of the 15-channel ego action vector. Matches
# ControllerInputsValue field order for sticks/triggers and BUTTON_BITS for
# buttons. Used both as model target and to construct ControllerInputsValue
# at inference — must stay in lockstep with action_vec_to_controller.
ACTION_CHANNELS: tuple[str, ...] = (
    "main_stick_x",
    "main_stick_y",
    "c_stick_x",
    "c_stick_y",
    "trigger_l_physical",
    "trigger_r_physical",
    "button_a",
    "button_b",
    "button_x",
    "button_y",
    "button_z",
    "button_r",
    "button_l",
    "button_start",
    "button_d_up",
)
assert len(ACTION_CHANNELS) == A_DIM

_BUTTON_ORDER = ("a", "b", "x", "y", "z", "r", "l", "start", "d_up")
# raw_* dropped (we use logical sticks); nana_* skipped for toy; trigger_logical
# redundant with physical l/r channels already consumed.
_DROP_PATTERNS = ("_raw_", "_nana_", "_trigger_logical")

_NEUTRAL_ACTION = np.zeros(A_DIM, dtype=np.float32)


# %%
def _classify(name: str) -> str:
    if name == "frame":
        return "drop"
    if any(p in name for p in _DROP_PATTERNS):
        return "drop"
    if name.endswith("_action_frame"):
        return "action_frame"
    if any(name.endswith(f"_{c}") for c in CAT_FEATURES):
        return "cat"
    if "_button_" in name:
        return "button"
    if any(
        name.endswith(f"_{s}")
        for s in (
            "main_stick_x",
            "main_stick_y",
            "c_stick_x",
            "c_stick_y",
            "trigger_l_physical",
            "trigger_r_physical",
        )
    ):
        return "stick_trigger"
    if any(name.endswith(f"_{f}") for f in FLOAT_FEATURES):
        return "float"
    return "drop"


def _is_masked(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind == "f":
        return np.isnan(arr)
    return arr == mask_value(arr.dtype)


def _normalize(arr: np.ndarray, s: FeatureStats) -> np.ndarray:
    return (2.0 * (arr - s.min) / (s.max - s.min) - 1.0).astype(np.float32)


def _standardize(arr: np.ndarray, s: FeatureStats) -> np.ndarray:
    return ((arr - s.mean) / s.std).astype(np.float32)


def preprocess_inputs(
    batch: dict[str, np.ndarray],
    feature_stats: dict[str, FeatureStats],
) -> dict[str, Tensor]:
    """Tokenizer-style per-feature sanitization + per-float mask sidecars.
    Operates on either single-sample [L] arrays or batched [B, L]; the numpy
    ops broadcast either way and torch.from_numpy preserves the shape.
    """
    out: dict[str, Tensor] = {}
    for name, arr in batch.items():
        kind = _classify(name)
        if kind == "drop":
            continue
        mask = _is_masked(arr)
        if kind == "button" or kind == "stick_trigger":
            x = np.where(mask, 0.0, arr).astype(np.float32)
        elif kind == "cat":
            x = np.where(mask, 0, arr).astype(np.int64)
        elif kind == "action_frame":
            # Vocab unbounded — treat as a float. /60 puts most values in [0, ~5].
            x = np.where(mask, 0, arr).astype(np.float32) / 60.0
        elif kind == "float":
            s = feature_stats[consolidate_key(name)]
            x = _standardize(arr, s) if "position" in name else _normalize(arr, s)
            x = np.where(mask, 0.0, x)
        else:
            raise AssertionError(f"unhandled kind {kind} for {name}")
        out[name] = torch.from_numpy(np.ascontiguousarray(x))
        if kind == "float" and mask.any():
            out[f"{name}_mask"] = torch.from_numpy(np.ascontiguousarray(mask.astype(np.float32)))
    return out


@jaxtyped(typechecker=beartype)
def stack_ego_actions(batch: dict[str, Tensor]) -> Float[Tensor, "B L d_action"]:
    """Stack ego action channels in canonical order. ``L`` is whatever
    sequence length the batch carries (full window at train; L_ctx at inference)."""
    return torch.stack([batch[f"ego_{ch}"] for ch in ACTION_CHANNELS], dim=-1)


def action_vec_to_controller(a: np.ndarray) -> ControllerInputsValue:
    """One 15-vector → one ControllerInputsValue. Buttons threshold at 0.5."""
    a = np.asarray(a).reshape(-1)
    buttons = 0
    for i, name in enumerate(_BUTTON_ORDER):
        if a[6 + i] > 0.5:
            buttons |= BUTTON_BITS[name]
    return ControllerInputsValue(
        main_x=float(np.clip(a[0], -1.0, 1.0)),
        main_y=float(np.clip(a[1], -1.0, 1.0)),
        c_x=float(np.clip(a[2], -1.0, 1.0)),
        c_y=float(np.clip(a[3], -1.0, 1.0)),
        trigger_l=float(np.clip(a[4], 0.0, 1.0)),
        trigger_r=float(np.clip(a[5], 0.0, 1.0)),
        buttons=int(buttons),
    )


# %%
@jaxtyped(typechecker=beartype)
def sinusoidal_time_embedding(t: Float[Tensor, " B"], dim: int) -> Float[Tensor, "B d_time"]:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t.unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FlowMatchingPolicy(nn.Module):
    """Unified Transformer over [L_ctx ctx tokens | n_lat bridge tokens | L_chunk noise tokens].

    Context tokens carry observed ego+opp gamestate + ego controller history.
    Bridge tokens (when ``latency_frames`` n_lat > 0) carry the n_lat already-
    committed actions about to execute while the new chunk is being computed at
    one inference period; they get their own type embedding and a separate
    projection. Chunk tokens carry the noised action a_t + time embedding + a
    learned chunk-type embedding. Output head reads the chunk positions and
    predicts the flow-matching velocity v̂ ∈ R^{L_chunk × d_action}.

    n_lat=0 reproduces the original open-loop architecture exactly (bridge_proj
    / bridge_type_emb are not constructed), keeping older checkpoints loadable.
    """

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.L_ctx = cfg.L_ctx
        self.L_chunk = cfg.L_chunk
        self.n_lat = cfg.latency_frames
        d = cfg.d_model
        self.time_emb_dim = cfg.time_emb_dim
        self.ego_history_dropout_prob = float(cfg.ego_history_dropout_prob)

        self.cat_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocab, dim) for name, (vocab, dim) in CAT_FEATURES.items()}
        )

        n_float = len(FLOAT_FEATURES)
        n_mask = len(FLOAT_FEATURES)
        n_action_frame = 1
        n_cat = sum(dim for _, dim in CAT_FEATURES.values())
        per_player_dim = n_float + n_mask + n_action_frame + n_cat
        per_frame_in_dim = 2 * per_player_dim + A_DIM  # ego + opp + ego controller history
        self.per_frame_in_dim = per_frame_in_dim

        self.ctx_proj = nn.Linear(per_frame_in_dim, d)
        self.chunk_proj = nn.Linear(A_DIM, d)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_emb_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.chunk_type_emb = nn.Parameter(torch.zeros(d))
        if self.n_lat > 0:
            self.bridge_proj = nn.Linear(A_DIM, d)
            self.bridge_type_emb = nn.Parameter(torch.zeros(d))
        self.pos_emb = nn.Embedding(self.L_ctx + self.n_lat + self.L_chunk, d)

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

    def _per_player_features(self, batch: dict[str, Tensor], prefix: str) -> Tensor:
        """[B, L_ctx, per_player_dim] — all observed features for one player.
        Mixed-dtype concat (float + int-embed lookup) so a single jaxtyping
        annotation on the dict input doesn't fit cleanly; covered by the
        forward() annotation downstream.
        """
        L_ctx = self.L_ctx
        ref = batch[f"{prefix}_position_x"]
        B = ref.shape[0]
        device = ref.device
        parts: list[Tensor] = []
        for f in FLOAT_FEATURES:
            parts.append(batch[f"{prefix}_{f}"][:, :L_ctx, None])
        for f in FLOAT_FEATURES:
            mk = f"{prefix}_{f}_mask"
            if mk in batch:
                parts.append(batch[mk][:, :L_ctx, None])
            else:
                parts.append(torch.zeros(B, L_ctx, 1, device=device))
        parts.append(batch[f"{prefix}_action_frame"][:, :L_ctx, None])
        for cat_name, (vocab, _) in CAT_FEATURES.items():
            ids = batch[f"{prefix}_{cat_name}"][:, :L_ctx].clamp(0, vocab - 1)
            parts.append(self.cat_embeds[cat_name](ids))
        return torch.cat(parts, dim=-1)

    def _ego_history_features(self, batch: dict[str, Tensor]) -> Float[Tensor, "B L_ctx d_action"]:
        """Ego's real past controller inputs.

        During training, with probability ``ego_history_dropout_prob`` per
        sample, zero a random *left* prefix ``[:k]`` (k ~ U{0..L_ctx}, inclusive
        of L_ctx so full-history zero is reachable). Matches the closed-loop
        rolling-buffer transient where ``ego_inputs_hist`` starts as L_ctx
        zeros and is gradually displaced by live predictions.
        """
        L_ctx = self.L_ctx
        hist = torch.cat([batch[f"ego_{ch}"][:, :L_ctx, None] for ch in ACTION_CHANNELS], dim=-1)
        if self.training and self.ego_history_dropout_prob > 0:
            B = hist.size(0)
            device = hist.device
            apply = torch.rand(B, device=device) < self.ego_history_dropout_prob
            ks = torch.randint(0, L_ctx + 1, (B,), device=device)
            ks = torch.where(apply, ks, torch.zeros_like(ks))
            positions = torch.arange(L_ctx, device=device)[None, :].expand(B, L_ctx)
            mask = positions < ks[:, None]
            hist = hist.masked_fill(mask[..., None], 0.0)
        return hist

    def build_context_tokens(self, batch: dict[str, Tensor]) -> Float[Tensor, "B L_ctx d_model"]:
        ego = self._per_player_features(batch, "ego")
        opp = self._per_player_features(batch, "opp")
        hist = self._ego_history_features(batch)
        return self.ctx_proj(torch.cat([ego, opp, hist], dim=-1))

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        batch_or_ctx: dict[str, Tensor] | Float[Tensor, "B L_ctx d_model"],
        a_t: Float[Tensor, "B L_chunk d_action"],
        t: Float[Tensor, " B"],
        bridge: Float[Tensor, "B n_lat d_action"] | None = None,
    ) -> Float[Tensor, "B L_chunk d_action"]:
        """
        batch_or_ctx — either a preprocessed batch dict OR a precomputed
                       context-token tensor (useful when integrating the flow
                       at inference, where context is fixed across all steps).
        bridge — required iff self.n_lat > 0; ignored otherwise.
        """
        if isinstance(batch_or_ctx, dict):
            ctx_tokens = self.build_context_tokens(batch_or_ctx)
        else:
            ctx_tokens = batch_or_ctx
        chunk_tokens = self.chunk_proj(a_t)
        t_emb = sinusoidal_time_embedding(t, self.time_emb_dim)
        t_proj = self.time_mlp(t_emb)
        chunk_tokens = chunk_tokens + t_proj[:, None, :] + self.chunk_type_emb[None, None, :]
        if self.n_lat > 0:
            if bridge is None:
                raise ValueError(
                    f"latency_frames={self.n_lat} requires bridge tensor of shape [B, {self.n_lat}, A_DIM]"
                )
            bridge_tokens = self.bridge_proj(bridge) + self.bridge_type_emb[None, None, :]
            seq = torch.cat([ctx_tokens, bridge_tokens, chunk_tokens], dim=1)
        else:
            seq = torch.cat([ctx_tokens, chunk_tokens], dim=1)
        pos_ids = torch.arange(seq.size(1), device=seq.device)
        seq = seq + self.pos_emb(pos_ids)[None, :, :]
        out = self.encoder(seq)
        chunk_start = self.L_ctx + self.n_lat
        return self.head(out[:, chunk_start:, :])


# %%
def make_run_name(cfg: TrainConfig, comment: str = "") -> str:
    """`YYMMDD-HHMMSS_fm-d256-L6-H8-Lc256-Lk16-fs8_ranked-anon-1[_comment]`."""
    stamp = datetime.now().strftime("%y%m%d-%H%M%S")
    model_tag = f"fm-d{cfg.d_model}-L{cfg.n_layers}-H{cfg.n_heads}-Lc{cfg.L_ctx}-Lk{cfg.L_chunk}-fs{cfg.n_flow_steps}"
    data_tag = Path(cfg.data_root).parent.name.replace("anonymized", "anon")
    parts = [stamp, model_tag, data_tag]
    if comment:
        parts.append(comment)
    return "_".join(parts)


def _to_device(batch: dict[str, Tensor], device: str) -> dict[str, Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def build_val_cache(
    val_loader,
    n_batches: int,
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    device: str,
) -> list[tuple]:
    """Materialize n_batches of val windows + fixed (t, z) noise on-device.
    Caching makes val loss comparable across evaluations — same windows,
    same noise — so a drop in val loss is a real model improvement, not
    sampling variance. CPU generator + .to(device) avoids cuda-rng pinning."""
    L_ctx, L_chunk, n_lat = cfg.L_ctx, cfg.L_chunk, cfg.latency_frames
    cache: list[tuple] = []
    g = torch.Generator(device="cpu").manual_seed(0)
    for raw in val_loader:
        batch = _to_device(preprocess_inputs(raw, stats), device)
        actions_all = stack_ego_actions(batch)
        bridge = actions_all[:, L_ctx : L_ctx + n_lat, :] if n_lat > 0 else None
        a_target = actions_all[:, L_ctx + n_lat :, :]
        B = a_target.shape[0]
        t = torch.rand(B, generator=g).to(device)
        z = torch.randn(B, L_chunk, A_DIM, generator=g).to(device)
        t_b = t.view(B, 1, 1)
        a_t = (1 - t_b) * z + t_b * a_target
        v_target = a_target - z
        cache.append((batch, a_t, t, v_target, bridge))
        if len(cache) >= n_batches:
            break
    if not cache:
        raise RuntimeError("val loader yielded zero batches")
    return cache


@torch.no_grad()
def val_loss(model: FlowMatchingPolicy, val_cache: list[tuple]) -> float:
    """Sample-weighted MSE across cached val batches. Toggles model.eval/train."""
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    for batch, a_t, t, v_target, bridge in val_cache:
        v_pred = model(batch, a_t, t, bridge=bridge)
        total += F.mse_loss(v_pred, v_target).item() * v_target.shape[0]
        count += v_target.shape[0]
    if was_training:
        model.train()
    return total / count


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


# %%
# Inference-time integrators. Live here (not in hal/eval/) because they're
# specific to flow-matching with a chunked-velocity head — i.e. THIS model.
@torch.no_grad()
@jaxtyped(typechecker=beartype)
def integrate_chunk_batched(
    model: FlowMatchingPolicy,
    batch: dict[str, Tensor],
    n_steps: int,
    device: str,
    bridge: Float[Tensor, "B n_lat d_action"] | None = None,
) -> Float[np.ndarray, "B L_chunk d_action"]:
    """Euler-integrate from z ~ N(0,I) for n_steps. Returns numpy for
    downstream rolling-buffer plumbing (lives in numpy land)."""
    model.eval()
    ctx = model.build_context_tokens(batch)
    B = next(iter(batch.values())).shape[0]
    a = torch.randn(B, model.L_chunk, A_DIM, device=device)
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t_val = torch.full((B,), k * dt, device=device)
        v = model(ctx, a, t_val, bridge=bridge)
        a = a + dt * v
    return a.cpu().numpy()


def integrate_chunk(
    model: FlowMatchingPolicy,
    batch: dict[str, Tensor],
    n_steps: int,
    device: str,
    bridge: Tensor | None = None,
) -> np.ndarray:
    """Single-sample shim. Returns [L_chunk, d_action]."""
    return integrate_chunk_batched(model, batch, n_steps, device, bridge=bridge)[0]


# %%
def _live_batch_from_rolling(
    flat_history: list[dict],
    ego_inputs_hist: list[np.ndarray],
    ego_prefix: str,
) -> dict[str, np.ndarray]:
    """[1, L_ctx] batch the model expects, built from rolling buffers."""
    out: dict[str, np.ndarray] = {}
    keys = flat_history[0].keys()
    for k in keys:
        sample = flat_history[0][k]
        dtype = np.int32 if isinstance(sample, int) else np.float32
        out[k] = np.array([h[k] for h in flat_history], dtype=dtype)
    # Ego controller history (intended actions, not whatever libmelee reads back).
    # Buttons stored as int 0/1 so the classifier routes via "button".
    hist_arr = np.stack(ego_inputs_hist)
    for i, ch in enumerate(ACTION_CHANNELS):
        col = hist_arr[:, i]
        if ch.startswith("button_"):
            out[f"{ego_prefix}_{ch}"] = (col > 0.5).astype(np.int32)
        else:
            out[f"{ego_prefix}_{ch}"] = col.astype(np.float32)
    out.pop("frame", None)
    relabeled = relabel_ego(out, ego_prefix)
    return {k: v[None, ...] for k, v in relabeled.items()}


@dataclass
class ToyModelSource:
    """ControllerSource for THIS experiment's flow-matching policy.

    Owns its rolling history (gamestate + intended-ego-action) and the
    receding-horizon replan state machine. ``n_lat=0`` → open-loop:
    predict L_chunk frames, play them, replan after L_chunk frames.
    ``n_lat>0`` → receding horizon: replan every n_lat frames; each chunk
    predicts L_chunk frames starting n_lat ahead of the replan time (models
    inference latency). Between replans, play the n_lat actions from the
    previous chunk's first n_lat predictions (the "bridge"); at bootstrap,
    bridge is zeros so the first n_lat frames are neutral.
    """

    model: FlowMatchingPolicy
    stats: dict[str, FeatureStats]
    ego_prefix: str
    L_ctx: int
    L_chunk: int
    n_lat: int
    n_flow_steps: int
    device: str = DEVICE
    _flat_hist: list = field(default_factory=list)
    _ego_inputs_hist: list = field(default_factory=list)
    _pending: np.ndarray | None = None
    _current_bridge: np.ndarray | None = None
    _offset: int = 0

    def __call__(self, frame_index: int, last_gamestate: dict | None):
        if last_gamestate is not None:
            self._flat_hist.append(flatten_canonical_frame(last_gamestate))
            if len(self._flat_hist) > self.L_ctx:
                self._flat_hist.pop(0)
        # Warm-up: not enough context yet → hold neutral.
        if len(self._flat_hist) < self.L_ctx:
            self._ego_inputs_hist.append(_NEUTRAL_ACTION.copy())
            if len(self._ego_inputs_hist) > self.L_ctx:
                self._ego_inputs_hist.pop(0)
            return action_vec_to_controller(_NEUTRAL_ACTION)
        # Transition: on the first inference call, _ego_inputs_hist is one
        # short of _flat_hist. Pad so both rolling buffers have L_ctx entries.
        if len(self._ego_inputs_hist) < self.L_ctx:
            self._ego_inputs_hist.append(_NEUTRAL_ACTION.copy())
        replan_period = self.n_lat if self.n_lat > 0 else self.L_chunk
        if self._pending is None or self._offset >= replan_period:
            raw = _live_batch_from_rolling(self._flat_hist, self._ego_inputs_hist, self.ego_prefix)
            batch = preprocess_inputs(raw, self.stats)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            if self.n_lat > 0:
                # Bootstrap: no prev chunk → zero bridge; bridge plays neutral
                # for the next n_lat frames. Steady state: bridge = prev_chunk[:n_lat].
                if self._pending is None:
                    new_bridge = np.zeros((self.n_lat, A_DIM), dtype=np.float32)
                else:
                    new_bridge = self._pending[: self.n_lat].astype(np.float32)
                bridge_t = torch.from_numpy(new_bridge).unsqueeze(0).to(self.device)
                self._pending = integrate_chunk(self.model, batch, self.n_flow_steps, self.device, bridge=bridge_t)
                self._current_bridge = new_bridge
            else:
                self._pending = integrate_chunk(self.model, batch, self.n_flow_steps, self.device)
            self._offset = 0
        # Played action: bridge if n_lat>0 (next n_lat to execute), else chunk directly.
        if self.n_lat > 0:
            a = self._current_bridge[self._offset]
        else:
            a = self._pending[self._offset]
        self._offset += 1
        self._ego_inputs_hist.append(a.astype(np.float32))
        if len(self._ego_inputs_hist) > self.L_ctx:
            self._ego_inputs_hist.pop(0)
        return action_vec_to_controller(a)


@dataclass
class ToySelfPlayCoord:
    """Drive both ports of a self-play match from one model with a single
    batched forward pass per chunk boundary. ``view("p1")`` / ``view("p2")``
    return ControllerSource adapters; calls per frame round-robin into a
    shared ``_advance`` that runs one batched [2, L_ctx, ...] forward at
    each replan boundary."""

    model: FlowMatchingPolicy
    stats: dict[str, FeatureStats]
    L_ctx: int
    L_chunk: int
    n_lat: int
    n_flow_steps: int
    device: str = DEVICE
    _ports: dict = field(
        default_factory=lambda: {
            "p1": {"flat_hist": [], "ego_inputs_hist": []},
            "p2": {"flat_hist": [], "ego_inputs_hist": []},
        }
    )
    _pending: dict = field(default_factory=lambda: {"p1": None, "p2": None})
    _current_bridge: dict = field(default_factory=lambda: {"p1": None, "p2": None})
    _offset: int = 0
    _last_frame_done: int = -1
    _last_actions: dict = field(default_factory=lambda: {"p1": _NEUTRAL_ACTION.copy(), "p2": _NEUTRAL_ACTION.copy()})

    def view(self, ego_prefix: Literal["p1", "p2"]) -> _PortView:
        return _PortView(coord=self, ego_prefix=ego_prefix)

    def _tick(self, ego_prefix: Literal["p1", "p2"], frame_index: int, last_gamestate: dict | None) -> np.ndarray:
        if frame_index != self._last_frame_done:
            self._advance(last_gamestate)
            self._last_frame_done = frame_index
        return self._last_actions[ego_prefix]

    def _advance(self, last_gamestate: dict | None) -> None:
        for ego in ("p1", "p2"):
            buf = self._ports[ego]
            if last_gamestate is not None:
                buf["flat_hist"].append(flatten_canonical_frame(last_gamestate))
                if len(buf["flat_hist"]) > self.L_ctx:
                    buf["flat_hist"].pop(0)
        # Warm-up: hold neutral until both buffers are full.
        if any(len(self._ports[e]["flat_hist"]) < self.L_ctx for e in ("p1", "p2")):
            for ego in ("p1", "p2"):
                buf = self._ports[ego]
                buf["ego_inputs_hist"].append(_NEUTRAL_ACTION.copy())
                if len(buf["ego_inputs_hist"]) > self.L_ctx:
                    buf["ego_inputs_hist"].pop(0)
            self._last_actions = {ego: _NEUTRAL_ACTION.copy() for ego in ("p1", "p2")}
            return
        for ego in ("p1", "p2"):
            buf = self._ports[ego]
            if len(buf["ego_inputs_hist"]) < self.L_ctx:
                buf["ego_inputs_hist"].append(_NEUTRAL_ACTION.copy())
        replan_period = self.n_lat if self.n_lat > 0 else self.L_chunk
        if self._pending["p1"] is None or self._offset >= replan_period:
            stacked = self._build_stacked_batch()
            batch = preprocess_inputs(stacked, self.stats)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            if self.n_lat > 0:
                if self._pending["p1"] is None:
                    new_bridges = {ego: np.zeros((self.n_lat, A_DIM), dtype=np.float32) for ego in ("p1", "p2")}
                else:
                    new_bridges = {ego: self._pending[ego][: self.n_lat].astype(np.float32) for ego in ("p1", "p2")}
                bridge_stack = np.stack([new_bridges["p1"], new_bridges["p2"]], axis=0)
                bridge_t = torch.from_numpy(bridge_stack).to(self.device)
                plans = integrate_chunk_batched(self.model, batch, self.n_flow_steps, self.device, bridge=bridge_t)
                self._pending = {"p1": plans[0], "p2": plans[1]}
                self._current_bridge = new_bridges
            else:
                plans = integrate_chunk_batched(self.model, batch, self.n_flow_steps, self.device)
                self._pending = {"p1": plans[0], "p2": plans[1]}
            self._offset = 0
        actions: dict[str, np.ndarray] = {}
        for ego in ("p1", "p2"):
            if self.n_lat > 0:
                a = self._current_bridge[ego][self._offset]
            else:
                a = self._pending[ego][self._offset]
            actions[ego] = a
            buf = self._ports[ego]
            buf["ego_inputs_hist"].append(a.astype(np.float32))
            if len(buf["ego_inputs_hist"]) > self.L_ctx:
                buf["ego_inputs_hist"].pop(0)
        self._offset += 1
        self._last_actions = actions

    def _build_stacked_batch(self) -> dict[str, np.ndarray]:
        per_ego: dict[str, dict[str, np.ndarray]] = {}
        for ego in ("p1", "p2"):
            buf = self._ports[ego]
            per_ego[ego] = _live_batch_from_rolling(buf["flat_hist"], buf["ego_inputs_hist"], ego_prefix=ego)
        return {k: np.concatenate([per_ego["p1"][k], per_ego["p2"][k]], axis=0) for k in per_ego["p1"]}


@dataclass(frozen=True)
class _PortView:
    coord: ToySelfPlayCoord
    ego_prefix: Literal["p1", "p2"]

    def __call__(self, frame_index: int, last_gamestate: dict | None):
        a = self.coord._tick(self.ego_prefix, frame_index, last_gamestate)
        return action_vec_to_controller(a)


# %%
def _default_session_cfg(replay_dir: Path | None = None) -> SessionConfig:
    """``replay_dir``: where Dolphin writes the match's .slp. With
    ``tmp_home_directory=True`` and ``replay_dir=None``, .slps land in
    ``<tmp_home>/Slippi/`` and disappear when the Session context exits;
    pass an explicit path to preserve them."""
    ensure(DOLPHIN_EXIAI)
    return SessionConfig(
        iso_path=ensure(ISO),
        dolphin_path=EMULATOR_PATH,
        use_exi_inputs=True,
        enable_ffw=True,
        emulation_speed=0.0,
        blocking_input=True,
        step_timeout_seconds=30.0,
        tmp_home_directory=True,
        replay_dir=str(replay_dir) if replay_dir is not None else None,
    )


def _vs_cpu_single_stage(
    model: FlowMatchingPolicy,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    max_frames: int,
    replay_dir: Path | None = None,
) -> dict:
    """In-training eval: one match on FD vs lvl-9 CPU. Aggregates into a
    flat metric dict for wandb. ``replay_dir`` (when not None) preserves the
    match .slp; otherwise the file is destroyed with the Session tmp home."""
    import melee

    was_training = model.training
    model.eval()
    try:
        results = sweep_stages_vs_cpu(
            source_factory=lambda ego: ToyModelSource(
                model=model,
                stats=stats,
                ego_prefix=ego,
                L_ctx=cfg.L_ctx,
                L_chunk=cfg.L_chunk,
                n_lat=cfg.latency_frames,
                n_flow_steps=cfg.n_flow_steps,
                device=DEVICE,
            ),
            session_cfg=_default_session_cfg(replay_dir=replay_dir),
            stages=(melee.Stage.FINAL_DESTINATION,),
            max_frames=max_frames,
        )
    finally:
        if was_training:
            model.train()
    _, summary = results[0]
    if summary is None:
        return dict(crashed=1.0)
    return dict(
        stocks_taken=4 - summary.p2_stocks_left,
        stocks_lost=4 - summary.p1_stocks_left,
        damage_dealt=summary.p2_max_pct,
        damage_taken=summary.p1_max_pct,
        frames=summary.frames,
    )


# %%
def train(cfg: TrainConfig, stats: dict[str, FeatureStats], comment: str = "") -> None:
    run_name = make_run_name(cfg, comment)
    wandb.init(
        project="hal",
        name=run_name,
        tags=["toy", "flow-matching", f"d{cfg.d_model}", f"L{cfg.n_layers}"],
        config=asdict(cfg),
    )
    ckpt_dir = Path("runs") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    replay_dir = ckpt_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ckpt] writing checkpoints to {ckpt_dir}", flush=True)
    print(f"[ckpt] writing eval replays to {replay_dir}", flush=True)

    model = FlowMatchingPolicy(cfg).to(DEVICE)
    train_loader = make_loader(
        data_root=cfg.data_root,
        split="train",
        L_ctx=cfg.L_ctx,
        L_chunk=cfg.L_chunk,
        n_lat=cfg.latency_frames,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
    )
    val_loader = make_loader(
        data_root=cfg.data_root,
        split="val",
        L_ctx=cfg.L_ctx,
        L_chunk=cfg.L_chunk,
        n_lat=cfg.latency_frames,
        batch_size=cfg.batch_size,
        num_workers=0,
    )

    def _save_ckpt(name: str, step: int) -> None:
        path = ckpt_dir / name
        torch.save({"step": step, "model": model.state_dict(), "cfg": asdict(cfg)}, path)
        print(f"[ckpt] saved {path}", flush=True)

    print("[val] building cached val set…", flush=True)
    val_t0 = time.monotonic()
    val_cache = build_val_cache(val_loader, cfg.val_n_batches, cfg, stats, DEVICE)
    print(
        f"[val] cached {len(val_cache)} batches "
        f"({sum(b[3].shape[0] for b in val_cache)} samples) in {time.monotonic() - val_t0:.1f}s",
        flush=True,
    )

    opt = AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
    sched = LambdaLR(opt, lr_schedule(cfg))
    model.train()
    L_ctx, n_lat = cfg.L_ctx, cfg.latency_frames

    print("[t+0.0s] building dataloader iter…", flush=True)
    it_t0 = time.monotonic()
    it = iter(train_loader)
    print(f"[t+{time.monotonic() - it_t0:.1f}s] iter built; fetching first batch…", flush=True)
    fetch_t0 = time.monotonic()
    raw = next(it)
    print(f"[t+{time.monotonic() - fetch_t0:.1f}s] first batch ready (B={cfg.batch_size})", flush=True)
    have_first = True
    step_t0 = time.monotonic()
    run_t0 = time.monotonic()
    for step in range(cfg.max_steps):
        if not have_first:
            try:
                raw = next(it)
            except StopIteration:
                it = iter(train_loader)
                raw = next(it)
        have_first = False
        data_t0 = time.monotonic()
        batch = _to_device(preprocess_inputs(raw, stats), DEVICE)
        actions_all = stack_ego_actions(batch)
        bridge = actions_all[:, L_ctx : L_ctx + n_lat, :] if n_lat > 0 else None
        a_target = actions_all[:, L_ctx + n_lat :, :]
        B = a_target.shape[0]
        t = torch.rand(B, device=DEVICE)
        z = torch.randn_like(a_target)
        t_b = t.view(B, 1, 1)
        a_t = (1 - t_b) * z + t_b * a_target
        v_target = a_target - z
        v_pred = model(batch, a_t, t, bridge=bridge)
        loss = F.mse_loss(v_pred, v_target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        step_dt = time.monotonic() - step_t0
        sps = B / step_dt
        wandb.log(
            {
                "train/loss": loss.item(),
                "train/lr": opt.param_groups[0]["lr"],
                "throughput/step_s": step_dt,
                "throughput/samples_per_s": sps,
                "throughput/data_overhead_s": data_t0 - step_t0,
            },
            step=step,
        )
        if step < 20 or step % 50 == 0:
            print(
                f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: loss {loss.item():.4f} "
                f"step_dt={step_dt * 1000:.0f}ms ({sps:.1f} samples/s)",
                flush=True,
            )
        step_t0 = time.monotonic()
        if cfg.val_every > 0 and step > 0 and step % cfg.val_every == 0:
            vl = val_loss(model, val_cache)
            wandb.log({"val/loss": vl}, step=step)
            print(f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: val_loss {vl:.4f}", flush=True)
        if cfg.eval_every > 0 and step > 0 and step % cfg.eval_every == 0:
            _save_ckpt(f"step_{step:06d}.pt", step)
            metrics = _vs_cpu_single_stage(model, stats, cfg, max_frames=cfg.eval_max_frames, replay_dir=replay_dir)
            wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)
            print(f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: closed_loop {metrics}", flush=True)
    vl_final = val_loss(model, val_cache)
    wandb.log({"val/loss": vl_final}, step=cfg.max_steps)
    print(f"[final] val_loss {vl_final:.4f}", flush=True)
    metrics_final = _vs_cpu_single_stage(model, stats, cfg, max_frames=cfg.eval_max_frames, replay_dir=replay_dir)
    wandb.log({f"eval/{k}": v for k, v in metrics_final.items()}, step=cfg.max_steps)
    print(f"[final] closed_loop {metrics_final}", flush=True)
    _save_ckpt("final.pt", cfg.max_steps)


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
    session_cfg = _default_session_cfg(replay_dir=replay_dir)
    stages = tuple(s for s in INCLUDED_STAGES if s is not melee.Stage.FOUNTAIN_OF_DREAMS)

    def src_factory(ego):
        return ToyModelSource(
            model=model,
            stats=stats,
            ego_prefix=ego,
            L_ctx=cfg.L_ctx,
            L_chunk=cfg.L_chunk,
            n_lat=cfg.latency_frames,
            n_flow_steps=cfg.n_flow_steps,
            device=DEVICE,
        )

    def coord_factory():
        coord = ToySelfPlayCoord(
            model=model,
            stats=stats,
            L_ctx=cfg.L_ctx,
            L_chunk=cfg.L_chunk,
            n_lat=cfg.latency_frames,
            n_flow_steps=cfg.n_flow_steps,
            device=DEVICE,
        )
        return coord.view("p1"), coord.view("p2")

    print("\n[eval] ============== vs-cpu ==============", flush=True)
    vs_cpu = sweep_stages_vs_cpu(src_factory, session_cfg=session_cfg, stages=stages, max_frames=15_000)
    for stage, s in vs_cpu:
        print(f"  {stage.name:18s} {s.as_dict() if s else 'CRASHED'}", flush=True)

    print("\n[eval] ============== self-play ==============", flush=True)
    sp = sweep_stages_self_play(coord_factory, session_cfg=session_cfg, stages=stages, max_frames=15_000)
    for stage, s in sp:
        print(f"  {stage.name:18s} {s.as_dict() if s else 'CRASHED'}", flush=True)


# %%
@dataclass
class Args:
    """Top-level CLI surface. Pass TrainConfig fields as kebab-case flags,
    e.g. ``--cfg.batch-size 128 --cfg.max-steps 100000``."""

    cfg: TrainConfig = field(default_factory=TrainConfig)
    eval: str | None = None  # ckpt path; if set, eval instead of train
    comment: str = ""


def main(args: Args) -> None:
    if args.eval is not None:
        eval_ckpt(args.eval)
        return
    cfg = args.cfg
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    auto_comment = f"lat{cfg.latency_frames}-{cfg.max_steps // 1000}k-b{cfg.batch_size}"
    train(cfg, stats, comment=args.comment or auto_comment)


if __name__ == "__main__":
    main(tyro.cli(Args))
