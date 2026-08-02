"""RLE action tokens: predict the next run-length token instead of the next frame. Forked from 016.

MOTIVATION
----------
016 (like 013) predicts the action ONE FRAME ahead, and the action stream is overwhelmingly
piecewise-constant: buttons hold ~93% of frames, the main stick is 34% exact-neutral, the c-stick
95% neutral. Most of the model's decisions are therefore "hold what you were holding", which is
exactly the regime where the measured failure lives (>90% ``pred_persistence``, a policy that mostly
repeats its last input). The offline tokenizer audit (``notebooks/tokenizer_audit.py``) showed that
byte-pair encoding over the JOINT per-frame symbol (the 4-tuple of group ids) compresses that stream
13.8x at vocab 4096 on held-out replays, and that the first 256 merges are 100% pure run-length
holds: BPE rediscovers RLE unprompted.

018 replaces the deployed per-frame head with a head over those RLE tokens. One decision now says
"do X for the next k frames"; the model spends its capacity on WHEN to change rather than on
re-affirming a hold 16 times. Everything else — trunk, spatial-feature flag, auxiliary multi-offset
frame heads, optimizer, eval machinery, decode hygiene where it still applies — is 016 verbatim, so
the studied axis is the output representation alone.

TOKENIZER (fitted offline by ``--fit-tokenizer``; a versioned JSON artifact)
---------------------------------------------------------------------------
* **Alphabet.** The ``alphabet_max`` most frequent observed per-frame 4-tuples (buttons 256 / main 65
  / c-stick 9 / triggers 25, from 013's unchanged quantizers), ordered by frequency. Token ids
  ``[0, n_alphabet)``; merges take the ids above.
* **Totality by NEAREST OBSERVED TUPLE.** A target frame whose tuple is not in the alphabet (unseen,
  or dropped by the cap — the audit measured a 0.5% held-out novel rate) is mapped to the nearest
  alphabet tuple under a per-group metric: button-combo HAMMING distance weighted by
  ``_BUTTON_DISTANCE_WEIGHT``, main/c-stick L2 between cluster centers, trigger L1 on the two
  shoulder values. The weight (10.0 per differing bit) exceeds the largest possible analog distance
  (2.83 + 2.83 + 2.0), so the substitute always agrees on the buttons first and then minimizes analog
  error; ties break toward the more frequent tuple (the alphabet is frequency-ordered and ``argmin``
  takes the lowest index). This is the ONE lossy step in the pipeline and it is measured, not
  assumed: ``val/substitution_rate`` and the fit's held-out rate report it. Deploy never needs it —
  the model can only emit vocabulary tokens.
* **Merge span cap.** A merge is accepted only if the merged token spans at most ``chunk_frames``
  (16 = ``VAL_L_CHUNK``) frames. Two reasons: open-loop commitment is bounded at 16 frames = 266ms
  (a token is executed to completion), and every target stays inside the frozen val window, so the
  bits/frame metric below is computable on exactly the val geometry every other experiment uses.
* Deterministic and seeded end to end: frequency ordering with index tie-breaks, ``np.unique``-based
  pair counting with the numerically-smallest-pair tie-break, and a fixed data seed recorded in the
  artifact. Loading validates the whole structure (version, id ranges, merge references, vocab
  arithmetic) and refuses an artifact fitted on a different ``data_root``.

MODEL
-----
016's trunk, unchanged, including ``cfg.spatial_features``. The ONLY architectural change is the
primary head: 016's offset-1 joint-355 head (four concatenated group vocabs) becomes a single
``Linear(d_model, vocab_size)`` predicting, at every valid context position ``t``, the FIRST token of
the greedy BPE encoding of frames ``t+1 .. t+16``. That target is a pure function of position ``t`` —
it does not depend on how earlier frames were segmented — so training stays fully parallel over
positions and there is no segmentation state to carry at inference.

The auxiliary multi-offset FRAME heads (offsets 5, 9, 13; ``aux_loss_weight``) are kept exactly as in
016. They are training-only trunk shapers, never deployed, and keeping them identical removes a
confound when comparing against the 016 arms.

LOSS: primary = plain-mean token NLL over valid positions; auxiliary = 016's per-offset, per-group
frame NLL at ``aux_loss_weight`` (with 016's ``transition_loss_weight`` reduction, default 1.0 = the
plain mean). Nothing else changed.

METRICS
-------
* ``val/loss`` = **bits per frame, chunk-comparable**. From position ``t``, take the token cover of
  frames ``t+1 .. t+16``: token 1 is scored at ``t``, token 2 at ``t + span(token 1)``, and so on,
  each from ITS OWN boundary position (teacher-forced, since the boundary positions are real context
  positions whose targets are the greedy first tokens). Summing those NLLs and dividing by 16 gives
  bits/frame for the same 16-frame action chunk, on the same discretizer grids and the same frozen
  ``VAL_L_CHUNK`` val windows that 016/013 score — so it is directly comparable to 016's factored
  per-frame sum, up to two documented biases: the last token may overshoot frame ``t+16`` (its bits
  are counted in full, so the number is an upper bound), and the tokenizer's nearest-tuple
  substitution makes the coded stream lossy at ``val/substitution_rate``.
* ``val/nll_{group}`` / ``val/brier_{group}`` / ``val/changeF1_{group}`` / ``val/pred_persistence_*``
  are computed on the FIRST-FUTURE-FRAME marginal and are EXACTLY comparable to 016's: every token
  determines the action of frame ``t+1``, so ``P(first frame = tuple) = Σ_token P(token)`` over the
  tokens sharing that first tuple, and one scatter-add per group turns the token distribution into
  the four per-group marginals 016's head emits directly. ``val/nll_frame1_marginals`` (their sum)
  is the exact analogue of 016's ``val/loss``.
* Token-level: realized tokens/frame under the cover, span histogram (mean/p50/p90/max and the
  share of 1-frame tokens), vocab utilization on val targets, and the substitution rate.
* 016's per-offset auxiliary NLLs, the copycat history ablation and the spatial ablation are kept
  (the ablations now measure KL/Δbits on the token distribution).

DEPLOY — RLE-native, variable horizon
-------------------------------------
``RLEPolicy`` is a ``BatchPolicy`` implemented in this file rather than a ``RecedingHorizon``
configuration: the harness takes any batch policy by injection, but ``RecedingHorizon``'s contract is
a FIXED execution horizon ``s``, and the whole point here is that the horizon is whatever the sampled
token says. Per slot it keeps a pending frame queue; slots with an empty queue are batched into one
trunk forward, one token is sampled from the temperature-scaled (optionally min-p filtered) token
softmax, and that token's frames are enqueued and executed to completion before the slot replans.
Button-support masking is unnecessary: the vocabulary contains only sequences observed in training.
The click => trigger fix is applied per decoded frame. An instant-restart boundary clears that slot's
queue and buffers. ``--cfg.exec-cadence k`` (``--eval-exec-cadence k`` at eval time) truncates every
token to its first ``k`` frames — execute ``min(k, span)``, then replan — for the fixed-cadence
ablation, and ``--bench-decode`` reports the amortized ms/frame against the 16.6ms real-time budget.

Run:
    uv run experiments/018_bpe_rle.py --fit-tokenizer --fit.out data/tokenizers/bpe4096.json
    uv run experiments/018_bpe_rle.py --cfg.tokenizer-path data/tokenizers/bpe4096.json
    uv run experiments/018_bpe_rle.py --eval <ckpt> --eval-temp 0.9
    uv run experiments/018_bpe_rle.py --bench-decode --eval <ckpt> --bench-slots 8
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
from collections.abc import Mapping
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
from hal.sim.inputs import ControllerInputs
from hal.sim.vec import BatchPolicy
from hal.sim.vec import Slot
from hal.training import scoring
from hal.training.canonical import flatten_canonical_frame
from hal.training.checkpoints import BackgroundUploader
from hal.training.checkpoints import load_for_resume
from hal.training.checkpoints import save_checkpoint
from hal.training.dataloader import VAL_L_CHUNK
from hal.training.dataloader import make_loader
from hal.training.dataloader import relabel_ego
from hal.training.ego_stats import load_consolidated_stats
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import NEUTRAL_ACTION
from hal.training.features import SPATIAL_COLUMNS
from hal.training.features import SPATIAL_FEATURES
from hal.training.features import SPATIAL_MASKS
from hal.training.features import Context
from hal.training.features import TrainBatch
from hal.training.features import action_vec_to_controller
from hal.training.features import preprocess
from hal.training.features import stack_actions
from hal.training.muon import SingleDeviceMuonWithAuxAdam
from hal.training.runs import make_run_name
from hal.training.runs import profile
from hal.training.runs import setup_run_dir

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_LN2 = math.log(2.0)
# Melee runs at 60 fps: one executed frame buys 1000/60 ms of decode budget.
_FRAME_MS = 1000.0 / 60.0

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

# The per-frame 4-tuple as one mixed-radix code: 256*65*9*25 distinct frames.
N_JOINT_CODES = math.prod(_GROUP_VOCABS)

# Nearest-tuple substitution metric. A differing button bit costs more than the largest possible
# analog disagreement (main 2.83 + c 2.83 + triggers 2.0 = 7.66), so the nearest observed tuple always
# matches the button combo as closely as possible first, then minimizes stick/trigger error.
_BUTTON_DISTANCE_WEIGHT = 10.0

_BUTTON_COUNTS_VERSION = 1
TOKENIZER_VERSION = 1

# Action-vector channels for the click=>trigger hygiene fix (digital L/R click => analog trigger = 1.0).
_TRIGGER_L_CH = ACTION_CHANNELS.index("trigger_l")
_TRIGGER_R_CH = ACTION_CHANNELS.index("trigger_r")
_BUTTON_L_CH = ACTION_CHANNELS.index("button_l")
_BUTTON_R_CH = ACTION_CHANNELS.index("button_r")


# %%
@dataclass
class TrainConfig:
    # Inherited 016 axis: append the derived spatial block (relative geometry, stage ledge / blastzone
    # distances, finite-difference velocities + validity masks) to every frame token.
    spatial_features: bool = True
    # GPT backbone
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 4
    # AUXILIARY multi-frame heads: one independent joint-A_VOCAB head per future-frame offset, exactly
    # 016's non-primary heads (016 ran offsets (1,5,9,13); offset 1 is now the token head). Never
    # deployed — a training-only trunk shaper, kept identical so 018 vs 016 is one axis.
    aux_head_offsets: tuple[int, ...] = (5, 9, 13)
    # PER-AUXILIARY-HEAD multiplier, 016 semantics unchanged.
    aux_loss_weight: float = 1.0
    # Per-sample ego-history input dropout (train only), 016 semantics unchanged.
    history_dropout_p: float = 0.0
    # Upweight transition-target positions in the AUXILIARY frame-head objective only (016 semantics).
    # The primary token loss is always a plain mean — a token IS the transition structure, so there is
    # no hold/transition split to reweight. 1.0 reduces exactly to the mean.
    transition_loss_weight: float = 1.0
    # Matchup conditioning (schema v5), 016 unchanged.
    char_vocab: int = 32
    char_dim: int = 12
    stage_vocab: int = 32
    stage_dim: int = 4
    # Closed-loop sampling temperature on the TOKEN softmax. Greedy argmax collapses the policy to a
    # do-nothing fixed point, so deployed play always samples.
    decode_temp: float = 1.0
    # min-p nucleus over the token distribution: keep tokens with p >= min_p * p_max, then renormalize.
    decode_min_p: float = 0.0
    # Force trigger_l/r to 1.0 on every decoded frame whose combo sets the digital L/R bit.
    decode_click_trigger_fix: bool = False
    # Deploy cadence. 0 = RLE-native: execute the sampled token to completion, then replan. k > 0
    # executes min(k, span) frames and replans (the fixed-cadence ablation).
    exec_cadence: int = 0
    # Context positions per sample at which the TOKEN head is trained (0 = dense over all valid
    # positions). The token logits are vocab_size wide, so dense at the default geometry would be
    # B * L_ctx * 4096 floats (>4 GB); a uniform draw from the pooled valid set is unbiased for the
    # same dense mean. The auxiliary frame heads stay dense, exactly as in 016.
    token_positions: int = 64
    # Reproducible training RNG and transformer context geometry.
    seed: int = 0
    L_ctx: int = 256
    # optimization
    batch_size: int = 1024
    grad_accum_steps: int = 1
    # Two LRs: Muon for the blocks' hidden matrices, AdamW for the input proj / heads / embeddings.
    muon_lr: float = 0.02
    adam_lr: float = 8.5e-4
    weight_decay: float = 0.01
    head_weight_decay: bool = True
    # Shared warmup/cosine schedule and training duration.
    warmup_steps: int = 500
    max_steps: int = 2**15
    amp_dtype: str = "bfloat16"  # "bfloat16" | "float32"
    allow_tf32: bool = True
    # eval cadence
    val_every: int = 1024
    val_n_batches: int = 16
    gradient_diagnostic_batch_size: int = 64
    diagnostic_rare_button_count: int = 100
    # Closed-loop evaluation cadence and per-boot frame budget. eval_every == 0 disables closed-loop
    # eval ENTIRELY (periodic and final) — the switch for smoke runs and boxes without an emulator.
    eval_every: int = 2048
    eval_max_frames: int = 7200
    eval_n_matchups: int = 16
    final_eval_n_matchups: int = 96
    eval_seed: int = 0
    eval_parallel_per_cpu: float = 1.0
    eval_overlap_training: bool = False
    eval_timeout_seconds: float = 900.0
    # checkpointing
    ckpt_every: int = 2048
    # data (v5 MDS carries the stage + p{1,2}_character + nana columns)
    data_root: str = "data/processed/ranked-anonymized-1/mds"
    # Fitted action tokenizer (JSON artifact from --fit-tokenizer). Required for a fresh run; resumed
    # and evaluated checkpoints carry their own embedded copy and ignore this path.
    tokenizer_path: str | None = None
    # Optional versioned JSON artifact of full-dataset button-combo counts (rarity diagnostics only).
    button_combo_counts_path: str | None = None
    # Streaming dataset cache and shuffle geometry.
    cache_limit_gb: int = 440
    shuffle_block_size: int = 2000
    windows_per_replay: int = 4
    val_split: str = "val"
    num_workers: int = 16
    prefetch_factor: int = 4


@dataclass
class FitConfig:
    """``--fit-tokenizer``: fit the RLE tokenizer on a sample of the train split.

    Reads the dataset named by ``TrainConfig.data_root`` through the ordinary training loader, so
    pointing ``--cfg.data-root`` at the production MDS is the only difference between the local smoke
    fit and the real one."""

    out: str = "data/tokenizers/018_bpe4096.json"
    vocab_size: int = 4096
    # Distinct per-frame tuples kept as alphabet tokens (most frequent first). Everything else is
    # substituted by its nearest kept tuple, so this caps the vocab on any corpus size.
    alphabet_max: int = 2048
    # Longest token, in frames. 16 = VAL_L_CHUNK: bounds open-loop commitment to 266ms and keeps every
    # training target inside the frozen val window.
    max_token_frames: int = VAL_L_CHUNK
    n_windows: int = 4096
    window_frames: int = 256
    windows_per_replay: int = 4
    batch_size: int = 64
    num_workers: int = 4
    # Trailing fraction of the sampled windows excluded from fitting; every reported compression /
    # substitution number is measured there, where unseen tuples actually occur.
    holdout_frac: float = 0.25


def _model_tag(cfg: TrainConfig, vocab_size: int) -> str:
    offs = ".".join(str(o) for o in cfg.aux_head_offsets)
    spatial = "-sp" if cfg.spatial_features else ""
    return f"rle-d{cfg.d_model}-L{cfg.n_layers}-h{cfg.n_heads}-Lc{cfg.L_ctx}-V{vocab_size}-aux{offs}{spatial}"


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


@jaxtyped(typechecker=beartype)
def group_codes(groups: Int[Tensor, "*batch n_groups"]) -> Int[Tensor, " *batch"]:
    """The 4-tuple of group ids as one mixed-radix code in ``[0, N_JOINT_CODES)``."""
    code = groups[..., 0]
    for g in range(1, N_GROUPS):
        code = code * _GROUP_VOCABS[g] + groups[..., g]
    return code


@jaxtyped(typechecker=beartype)
def code_groups(codes: Int[Tensor, " *batch"]) -> Int[Tensor, "*batch n_groups"]:
    """Inverse of ``group_codes``."""
    rest = codes
    out: list[Tensor] = []
    for g in reversed(range(N_GROUPS)):
        out.append(rest % _GROUP_VOCABS[g])
        rest = rest // _GROUP_VOCABS[g]
    return torch.stack(list(reversed(out)), dim=-1)


# %%
# --- the fitted tokenizer -----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ActionTokenizer:
    """A fitted RLE tokenizer: the versioned artifact, nothing derived.

    ``alphabet`` holds joint frame codes (``group_codes``) in token order — token id = index, most
    frequent first. ``merges`` are ``(left, right)`` token-id pairs in learned order — token id
    ``len(alphabet) + rank`` — so truncating the list is exactly a smaller-vocab tokenizer."""

    version: int
    data_root: str
    split: str
    seed: int
    max_token_frames: int
    vocab_size: int
    alphabet: tuple[int, ...]
    merges: tuple[tuple[int, int], ...]
    stats: dict[str, float]

    @property
    def merge_base(self) -> int:
        return len(self.alphabet)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "data_root": self.data_root,
            "split": self.split,
            "seed": self.seed,
            "max_token_frames": self.max_token_frames,
            "vocab_size": self.vocab_size,
            "alphabet": list(self.alphabet),
            "merges": [list(pair) for pair in self.merges],
            "stats": dict(self.stats),
        }


_TOKENIZER_KEYS = frozenset(
    {"version", "data_root", "split", "seed", "max_token_frames", "vocab_size", "alphabet", "merges", "stats"}
)


def tokenizer_from_dict(data: dict, *, source: str) -> ActionTokenizer:
    """Exhaustively validated load. Every structural invariant the encoder, the decode tables and the
    deployed policy rely on is checked here, so no other code path defends against a bad artifact."""
    if set(data) != set(_TOKENIZER_KEYS):
        raise ValueError(f"{source}: expected exactly keys {sorted(_TOKENIZER_KEYS)}, got {sorted(data)}")
    if data["version"] != TOKENIZER_VERSION:
        raise ValueError(f"{source}: tokenizer version {data['version']} != supported {TOKENIZER_VERSION}")
    for name in ("seed", "max_token_frames", "vocab_size"):
        value = data[name]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{source}: {name} must be an integer, got {value!r}")
    if data["max_token_frames"] < 1:
        raise ValueError(f"{source}: max_token_frames must be >= 1, got {data['max_token_frames']}")
    if not isinstance(data["data_root"], str) or not isinstance(data["split"], str):
        raise ValueError(f"{source}: data_root and split must be strings")
    alphabet = data["alphabet"]
    if not isinstance(alphabet, list) or any(not isinstance(c, int) or isinstance(c, bool) for c in alphabet):
        raise ValueError(f"{source}: alphabet must be a list of integer joint codes")
    if not alphabet:
        raise ValueError(f"{source}: alphabet is empty")
    if any(not 0 <= c < N_JOINT_CODES for c in alphabet):
        raise ValueError(f"{source}: alphabet holds a code outside [0, {N_JOINT_CODES})")
    if len(set(alphabet)) != len(alphabet):
        raise ValueError(f"{source}: alphabet has duplicate joint codes")
    merges = data["merges"]
    if not isinstance(merges, list):
        raise ValueError(f"{source}: merges must be a list of [left, right] pairs")
    merge_base = len(alphabet)
    for rank, pair in enumerate(merges):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"{source}: merge {rank} is not a [left, right] pair: {pair!r}")
        for side in pair:
            if not isinstance(side, int) or isinstance(side, bool) or not 0 <= side < merge_base + rank:
                raise ValueError(f"{source}: merge {rank} references token {side!r} outside [0, {merge_base + rank})")
    if data["vocab_size"] != merge_base + len(merges):
        raise ValueError(f"{source}: vocab_size {data['vocab_size']} != {merge_base} alphabet + {len(merges)} merges")
    stats = data["stats"]
    if not isinstance(stats, dict) or any(not isinstance(v, (int, float)) for v in stats.values()):
        raise ValueError(f"{source}: stats must be a dict of numbers")
    tok = ActionTokenizer(
        version=data["version"],
        data_root=data["data_root"],
        split=data["split"],
        seed=data["seed"],
        max_token_frames=data["max_token_frames"],
        vocab_size=data["vocab_size"],
        alphabet=tuple(alphabet),
        merges=tuple((int(a), int(b)) for a, b in merges),
        stats={str(k): float(v) for k, v in stats.items()},
    )
    spans = token_spans(tok)
    if int(spans.max()) > tok.max_token_frames:
        raise ValueError(f"{source}: a token spans {int(spans.max())} frames, over max_token_frames")
    return tok


def load_tokenizer(cfg: TrainConfig) -> ActionTokenizer:
    """Read + validate the configured artifact and check it describes the configured dataset."""
    if cfg.tokenizer_path is None:
        raise ValueError("a fresh run needs --cfg.tokenizer-path (fit one with --fit-tokenizer)")
    path = Path(cfg.tokenizer_path)
    tok = tokenizer_from_dict(json.loads(path.read_text()), source=str(path))
    if Path(tok.data_root).resolve() != Path(cfg.data_root).resolve():
        raise ValueError(
            f"{path}: tokenizer was fit on {tok.data_root}, but data_root is {cfg.data_root}; "
            "re-fit before training on a different dataset"
        )
    return tok


# %%
def token_spans(tok: ActionTokenizer) -> np.ndarray:
    """``[vocab_size]`` frames covered by each token (alphabet tokens span 1)."""
    spans = np.ones(tok.vocab_size, dtype=np.int64)
    for i, (left, right) in enumerate(tok.merges):
        spans[tok.merge_base + i] = spans[left] + spans[right]
    return spans


def token_frames(tok: ActionTokenizer) -> Tensor:
    """``[vocab_size, max_token_frames, n_groups]`` group ids each token expands to (rows past the
    token's span are zero-filled and never read; ``token_spans`` gives the true lengths)."""
    H = tok.max_token_frames
    frames = torch.zeros(tok.vocab_size, H, N_GROUPS, dtype=torch.long)
    alphabet = code_groups(torch.tensor(tok.alphabet, dtype=torch.long))  # [n_alphabet, n_groups]
    frames[: tok.merge_base, 0] = alphabet
    spans = token_spans(tok)
    for i, (left, right) in enumerate(tok.merges):
        token = tok.merge_base + i
        n_left = int(spans[left])
        frames[token, :n_left] = frames[left, :n_left]
        frames[token, n_left : n_left + int(spans[right])] = frames[right, : int(spans[right])]
    return frames


def build_merge_rank(tok: ActionTokenizer) -> Tensor:
    """``[vocab, vocab]`` int32 merge priority: entry ``(a, b)`` is the rank of the merge that turns the
    adjacent pair into ``merge_base + rank``, or ``-1`` when the pair never merges. A merge can only
    create pairs involving its own (later-numbered) token, so greedily merging the lowest-rank pair
    present is exactly BPE's in-order encoding."""
    rank = torch.full((tok.vocab_size, tok.vocab_size), -1, dtype=torch.int32)
    for i, (left, right) in enumerate(tok.merges):
        rank[left, right] = i
    return rank


def group_distance_tables() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Per-group distance matrices for nearest-observed-tuple substitution: button-combo Hamming
    (weighted), main/c-stick L2 between cluster centers, trigger L1 on the two shoulder values."""
    combos = torch.arange(scoring.N_BUTTON_COMBOS)
    bits = scoring.combo_to_buttons(combos)  # [256, 8] float {0,1}
    d_btn = _BUTTON_DISTANCE_WEIGHT * (bits[:, None, :] != bits[None, :, :]).sum(-1).float()
    main = scoring.STICK_CLUSTER_CENTERS_MAIN
    d_main = (main[:, None, :] - main[None, :, :]).pow(2).sum(-1).sqrt()
    c = scoring.STICK_CLUSTER_CENTERS_C
    d_c = (c[:, None, :] - c[None, :, :]).pow(2).sum(-1).sqrt()
    n_trig = scoring.TRIGGER_CENTERS.shape[0]
    trig_idx = torch.arange(n_trig * n_trig)
    trig_lr = torch.stack(
        [scoring.TRIGGER_CENTERS[trig_idx // n_trig], scoring.TRIGGER_CENTERS[trig_idx % n_trig]], dim=-1
    )
    d_trig = (trig_lr[:, None, :] - trig_lr[None, :, :]).abs().sum(-1)
    return d_btn, d_main, d_c, d_trig


@jaxtyped(typechecker=beartype)
def nearest_alphabet_token(
    codes: Int[Tensor, " n_code"],
    alphabet_groups: Int[Tensor, "n_alphabet n_groups"],
    distances: tuple[Tensor, Tensor, Tensor, Tensor],
    *,
    chunk: int = 4096,
) -> Int[Tensor, " n_code"]:
    """Nearest alphabet token for each joint code, under the summed per-group metric. Ties go to the
    lowest alphabet index, i.e. to the more frequent tuple (the alphabet is frequency-ordered)."""
    query = code_groups(codes)  # [n_code, n_groups]
    out = torch.empty(codes.shape[0], dtype=torch.long, device=codes.device)
    for start in range(0, codes.shape[0], chunk):
        q = query[start : start + chunk]
        dist = distances[0][q[:, _BUTTONS_G]][:, alphabet_groups[:, _BUTTONS_G]]
        for g in (_MAIN_G, _C_G, _TRIG_G):
            dist = dist + distances[g][q[:, g]][:, alphabet_groups[:, g]]
        out[start : start + chunk] = dist.argmin(-1)
    return out


# %%
@jaxtyped(typechecker=beartype)
def encode_windows(
    base: Int[Tensor, "n_window n_frame"], *, merge_rank: Int[Tensor, "vocab vocab"], merge_base: int
) -> tuple[Int[Tensor, "n_window n_frame"], Int[Tensor, " n_window"]]:
    """Greedy BPE over each window of per-frame alphabet tokens → the left-packed token sequence and
    its length. ``tokens[:, 0]`` is the training target: the first token of the encoding.

    Repeatedly merges each row's lowest-rank adjacent pair until no row has one. Runs on the training
    hot path, so every write goes through ``scatter_`` into a buffer with one TRASH column that
    absorbs the drops: boolean-mask indexing would force a device→host sync per iteration to size its
    output. Entries past a row's length are stale but never read (the pair mask uses the length)."""
    n, L = base.shape
    device = base.device
    buf = torch.cat([base, torch.zeros(n, 1, dtype=torch.long, device=device)], dim=1)  # trash column at L
    length = torch.full((n,), L, dtype=torch.long, device=device)
    big = torch.iinfo(torch.int32).max
    pair_pos = torch.arange(L - 1, device=device)
    for _ in range(L - 1):
        rank = merge_rank[buf[:, : L - 1], buf[:, 1:L]]
        live_pair = pair_pos[None, :] < (length - 1)[:, None]
        best, pos = rank.masked_fill(~live_pair | (rank < 0), big).min(dim=1)
        active = best < big
        if not bool(active.any()):
            break
        merged = merge_base + best.clamp(min=0).long()
        buf.scatter_(1, torch.where(active, pos, L)[:, None], merged[:, None])
        keep = torch.ones_like(buf, dtype=torch.bool)
        keep[:, L] = False
        keep.scatter_(1, torch.where(active, pos + 1, L)[:, None], False)
        dst = torch.where(keep, keep.long().cumsum(1) - 1, L).clamp(min=0, max=L)
        buf = torch.zeros_like(buf).scatter_(1, dst, buf)
        length = length - active.long()
    return buf[:, :L].contiguous(), length


# %%
# --- offline fit --------------------------------------------------------------------------------
def _merge_positions(stream: np.ndarray, a: int, b: int) -> np.ndarray:
    """Left-to-right non-overlapping occurrences of the pair ``(a, b)``. Distinct symbols can never
    overlap; only a self-pair (a run) needs the greedy scan."""
    hit = np.flatnonzero((stream[:-1] == a) & (stream[1:] == b))
    if a != b or hit.size < 2:
        return hit
    keep: list[int] = []
    prev = -2
    for pos in hit.tolist():
        if pos > prev + 1:
            keep.append(pos)
            prev = pos
    return np.asarray(keep, dtype=np.int64)


def _apply_merge(stream: np.ndarray, a: int, b: int, new_id: int) -> np.ndarray:
    pos = _merge_positions(stream, a, b)
    if pos.size == 0:
        return stream
    stream = stream.copy()
    stream[pos] = new_id
    drop = np.zeros(stream.size, dtype=bool)
    drop[pos + 1] = True
    return stream[~drop]


def bpe_train(stream: np.ndarray, n_base: int, n_merges: int, *, max_frames: int) -> list[tuple[int, int]]:
    """Learn up to ``n_merges`` merges over a flat token stream (``-1`` separates units a merge may not
    span), skipping any pair whose merged token would span more than ``max_frames`` frames.

    Deterministic: among admissible pairs the most frequent wins, ties going to the numerically
    smallest ``(left, right)`` code."""
    spans = np.ones(n_base + n_merges, dtype=np.int64)
    merges: list[tuple[int, int]] = []
    for new_id in range(n_base, n_base + n_merges):
        lo, hi = stream[:-1], stream[1:]
        ok = (lo >= 0) & (hi >= 0)
        if not ok.any():
            break
        code, count = np.unique((lo[ok] << 32) | hi[ok], return_counts=True)
        left, right = code >> 32, code & 0xFFFFFFFF
        admissible = (spans[left] + spans[right] <= max_frames) & (count >= 2)
        if not admissible.any():
            break
        pick = int(np.where(admissible, count, -1).argmax())
        a, b = int(left[pick]), int(right[pick])
        merges.append((a, b))
        spans[new_id] = spans[a] + spans[b]
        stream = _apply_merge(stream, a, b, new_id)
    return merges


def _fit_windows(cfg: TrainConfig, fit: FitConfig) -> Tensor:
    """``[n_windows, window_frames, A_DIM]`` raw ego action stream from the train split, read through
    the ordinary training loader (same windowing, same ego-port draw, same schema check)."""
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    loader = make_loader(
        data_root=cfg.data_root,
        split="train",
        remote=streams.remote_for_local(cfg.data_root),
        cache_limit=f"{cfg.cache_limit_gb}gb",
        shuffle_block_size=cfg.shuffle_block_size,
        stats=stats,
        L_ctx=fit.window_frames,
        L_chunk=fit.max_token_frames,
        batch_size=fit.batch_size,
        seed=cfg.seed,
        num_workers=fit.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        windows_per_replay=fit.windows_per_replay,
    )
    windows: list[Tensor] = []
    seen = 0
    for batch in loader:
        windows.append(stack_actions(batch.context.features))
        seen += windows[-1].shape[0]
        print(f"[fit] read {seen}/{fit.n_windows} windows", flush=True)
        if seen >= fit.n_windows:
            break
    if not windows:
        raise RuntimeError(f"train split under {cfg.data_root} yielded no windows")
    if seen < fit.n_windows:
        print(f"[fit] train split exhausted at {seen} windows (asked for {fit.n_windows})", flush=True)
    return torch.cat(windows)[: fit.n_windows]


def _resolve_fit_codes(
    codes: Tensor, alphabet: tuple[int, ...], distances: tuple[Tensor, Tensor, Tensor, Tensor]
) -> tuple[Tensor, float]:
    """Fit-time code → token map (alphabet hit, else nearest observed tuple) + the substitution rate."""
    lookup = torch.full((N_JOINT_CODES,), -1, dtype=torch.long)
    alpha = torch.tensor(alphabet, dtype=torch.long)
    lookup[alpha] = torch.arange(len(alphabet))
    flat = codes.reshape(-1)
    tokens = lookup[flat]
    missing = tokens < 0
    rate = float(missing.float().mean())
    if bool(missing.any()):
        unique_missing = torch.unique(flat[missing])
        substitute = nearest_alphabet_token(unique_missing, code_groups(alpha), distances)
        lookup[unique_missing] = substitute
        tokens = lookup[flat]
    return tokens.reshape(codes.shape), rate


@jaxtyped(typechecker=beartype)
def _cover_stats(
    base: Int[Tensor, "n_window n_frame"], *, merge_rank: Tensor, merge_base: int, spans: Tensor, H: int
) -> tuple[Tensor, Tensor]:
    """Walk the deployment/target cover across held-out windows: at each position encode the next ``H``
    frames, emit the first token, advance by its span. Returns the emitted TOKEN IDS (flat) and the
    per-window token count."""
    n, L = base.shape
    device = base.device
    pos = torch.zeros(n, dtype=torch.long, device=device)
    counts = torch.zeros(n, dtype=torch.long, device=device)
    emitted: list[Tensor] = []
    offsets = torch.arange(H, device=device)
    while True:
        live = pos + H <= L
        if not bool(live.any()):
            break
        idx = (pos[:, None] + offsets[None, :]).clamp(max=L - 1)
        first, _ = encode_windows(base.gather(1, idx), merge_rank=merge_rank, merge_base=merge_base)
        span = spans[first[:, 0]]
        emitted.append(first[live, 0])
        counts = counts + live.long()
        pos = pos + torch.where(live, span, torch.full_like(span, H))
    return torch.cat(emitted), counts


def fit_tokenizer(cfg: TrainConfig, fit: FitConfig, *, device: str = DEVICE) -> ActionTokenizer:
    """Fit alphabet + merges on a sample of the train split and measure compression on held-out windows.

    The alphabet is the ``alphabet_max`` most frequent per-frame tuples; every other frame is mapped to
    its nearest kept tuple, so the encoder is total and the vocabulary is bounded on any corpus. Merges
    are learned over the substituted stream with the ``max_token_frames`` span cap, and all reported
    numbers (tokens/frame, span histogram, substitution rate, utilization) come from the HELD-OUT
    windows, where unseen tuples occur at their real rate."""
    if fit.vocab_size < 2 or fit.alphabet_max <= 0 or fit.n_windows <= 0:
        raise ValueError("vocab_size >= 2, alphabet_max > 0 and n_windows > 0 are required")
    if not 0.0 < fit.holdout_frac < 1.0:
        raise ValueError(f"holdout_frac must be in (0, 1), got {fit.holdout_frac}")
    if fit.max_token_frames < 1 or fit.max_token_frames > fit.window_frames:
        raise ValueError(f"max_token_frames must be in [1, window_frames], got {fit.max_token_frames}")
    H = fit.max_token_frames
    raw = _fit_windows(cfg, fit)
    n_win, window_frames, _ = raw.shape
    n_fit_win = int(round(n_win * (1.0 - fit.holdout_frac)))
    if not 0 < n_fit_win < n_win:
        raise ValueError(f"holdout_frac={fit.holdout_frac} leaves {n_fit_win}/{n_win} fitting windows")
    groups = quantize_groups(
        scoring.STICK_CLUSTER_CENTERS_MAIN, scoring.STICK_CLUSTER_CENTERS_C, scoring.TRIGGER_CENTERS, raw
    )
    codes = group_codes(groups)  # [n_win, window_frames]
    fit_codes, held_codes = codes[:n_fit_win], codes[n_fit_win:]

    observed, counts = torch.unique(fit_codes.reshape(-1), return_counts=True)
    order = torch.argsort(counts, descending=True, stable=True)
    alphabet = tuple(int(c) for c in observed[order][: fit.alphabet_max].tolist())
    covered = int(counts[order][: fit.alphabet_max].sum())
    n_fit_frames = int(fit_codes.numel())
    print(
        f"[fit] {n_fit_win} fit / {n_win - n_fit_win} held-out windows, {n_fit_frames:,} fit frames, "
        f"{len(observed)} distinct tuples; alphabet {len(alphabet)} covers {covered / n_fit_frames:.4%}",
        flush=True,
    )
    distances = group_distance_tables()
    fit_tokens, fit_sub_rate = _resolve_fit_codes(fit_codes, alphabet, distances)

    n_merges = fit.vocab_size - len(alphabet)
    if n_merges <= 0:
        raise ValueError(f"vocab_size={fit.vocab_size} leaves no merges past a {len(alphabet)}-token alphabet")
    sep = torch.full((fit_tokens.shape[0], 1), -1, dtype=torch.long)
    stream = torch.cat([fit_tokens, sep], dim=1).reshape(-1).numpy()
    print(f"[fit] learning up to {n_merges} merges over {n_fit_frames:,} frames (span cap {H})…", flush=True)
    t0 = time.monotonic()
    merges = bpe_train(stream, len(alphabet), n_merges, max_frames=H)
    print(f"[fit] learned {len(merges)} merges in {time.monotonic() - t0:.1f}s", flush=True)

    tok = ActionTokenizer(
        version=TOKENIZER_VERSION,
        data_root=cfg.data_root,
        split="train",
        seed=cfg.seed,
        max_token_frames=H,
        vocab_size=len(alphabet) + len(merges),
        alphabet=alphabet,
        merges=tuple(merges),
        stats={},
    )
    held_tokens, held_sub_rate = _resolve_fit_codes(held_codes, alphabet, distances)
    spans = torch.from_numpy(token_spans(tok)).to(device)
    merge_rank = build_merge_rank(tok).to(device)
    emitted, per_window = _cover_stats(
        held_tokens.to(device), merge_rank=merge_rank, merge_base=tok.merge_base, spans=spans, H=H
    )
    emitted_spans = spans[emitted].float()
    covered_frames = float(emitted_spans.sum())
    q50, q90 = torch.quantile(emitted_spans, torch.tensor([0.5, 0.9], device=device)).tolist()
    used = int(torch.unique(emitted).numel())
    # Hold-vs-motif composition: a multi-frame token is a pure HOLD iff every frame in its span repeats
    # the first frame's tuple; otherwise it is a learned cross-channel MOTIF (e.g. press-then-release).
    # Answers how much of the vocabulary is RLE-like vs structure BPE found beyond run-lengths.
    frames_v = token_frames(tok).to(device)  # [vocab, H, n_groups]
    in_span = torch.arange(H, device=device)[None, :] < spans[:, None]
    is_hold = ((frames_v == frames_v[:, :1]).all(-1) | ~in_span).all(-1)  # [vocab]; span-1 trivially True
    emitted_multi = emitted_spans > 1
    vocab_multi = spans > 1
    frac_hold_emitted = float(is_hold[emitted][emitted_multi].float().mean()) if bool(emitted_multi.any()) else 0.0
    frac_hold_vocab = float(is_hold[vocab_multi].float().mean()) if bool(vocab_multi.any()) else 0.0
    stats = {
        "n_windows": float(n_win),
        "n_fit_windows": float(n_fit_win),
        "n_fit_frames": float(n_fit_frames),
        "n_held_frames": float(held_codes.numel()),
        "distinct_tuples": float(len(observed)),
        "alphabet_frame_coverage": covered / n_fit_frames,
        "substitution_rate_fit": fit_sub_rate,
        "substitution_rate": held_sub_rate,
        "tokens_per_frame": float(emitted.numel()) / covered_frames,
        "span_mean": float(emitted_spans.mean()),
        "span_p50": q50,
        "span_p90": q90,
        "span_max": float(emitted_spans.max()),
        "span_frac_len1": float((emitted_spans == 1).float().mean()),
        "tokens_per_window": float(per_window.float().mean()),
        "vocab_utilization": used / tok.vocab_size,
        "frac_hold_multi_emitted": frac_hold_emitted,
        "frac_hold_multi_vocab": frac_hold_vocab,
    }
    print(
        f"[fit] held-out {stats['tokens_per_frame']:.3f} tokens/frame "
        f"({N_GROUPS / stats['tokens_per_frame']:.1f}x vs 4 group ids/frame) | span mean "
        f"{stats['span_mean']:.2f} p50 {q50:.0f} p90 {q90:.0f} max {stats['span_max']:.0f} | "
        f"1-frame tokens {stats['span_frac_len1']:.1%} | substitution {held_sub_rate:.4%} | "
        f"vocab {tok.vocab_size} ({len(alphabet)} alphabet + {len(merges)} merges), "
        f"utilization {stats['vocab_utilization']:.1%} | multi-frame tokens: "
        f"{stats['frac_hold_multi_emitted']:.1%} pure-hold emitted, {stats['frac_hold_multi_vocab']:.1%} in vocab",
        flush=True,
    )
    return replace(tok, stats=stats)


def write_tokenizer(tok: ActionTokenizer, path: Path) -> None:
    """Atomically persist the artifact (and re-validate exactly what was written)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(tok.to_dict(), sort_keys=True)
    tokenizer_from_dict(json.loads(payload), source=str(path))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload)
    tmp.replace(path)
    print(f"[fit] wrote {path} ({len(payload) / 1e6:.2f} MB)", flush=True)


# %%
# --- GPT backbone (016 verbatim: rotary, RMSNorm, causal SDPA) --------------------------------
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


# %%
class GPT(nn.Module):
    """016's causal GPT over per-frame tokens, with the deployed head swapped for an RLE token head.

    ``hidden[i]`` feeds (a) ``token_head`` → the first token of the greedy encoding of frames
    ``i+1 .. i+max_token_frames``, which is what closed loop samples, and (b) one auxiliary joint
    ``A_VOCAB`` frame head per offset in ``cfg.aux_head_offsets`` (016's non-primary heads, unchanged
    and never deployed). The fitted tokenizer's lookup tables ride along as NON-persistent buffers
    rebuilt from the artifact that travels inside the checkpoint, so checkpoints stay weights-only and
    device placement follows ``.to()``."""

    main_centers: Tensor
    c_centers: Tensor
    trig_centers: Tensor
    button_combo_counts: Tensor
    tok_merge_rank: Tensor
    tok_frames: Tensor
    tok_spans: Tensor
    tok_first_groups: Tensor
    alphabet_groups: Tensor
    code_to_token: Tensor
    code_is_alphabet: Tensor

    def __init__(self, cfg: TrainConfig, tok: ActionTokenizer) -> None:
        super().__init__()
        if not cfg.decode_temp > 0:
            raise ValueError(f"decode_temp must be > 0, got {cfg.decode_temp}")
        if not 0.0 <= cfg.history_dropout_p <= 1.0:
            raise ValueError(f"history_dropout_p must be in [0, 1], got {cfg.history_dropout_p}")
        offs = tuple(cfg.aux_head_offsets)
        if any(o < 1 for o in offs) or len(set(offs)) != len(offs):
            raise ValueError(f"aux_head_offsets must be unique and >= 1, got {offs}")
        self.history_dropout_p = cfg.history_dropout_p
        self.aux_offsets = offs
        self.L_ctx = cfg.L_ctx
        self.spatial_features = cfg.spatial_features
        self.tokenizer = tok
        self.max_token_frames = tok.max_token_frames
        self.vocab_size = tok.vocab_size

        # Gamestate categoricals: one table per feature name, shared across the four players.
        self.cat_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocab, dim) for name, (vocab, dim) in CAT_FEATURES.items()}
        )
        self.char_emb = nn.Embedding(cfg.char_vocab, cfg.char_dim)
        self.stage_emb = nn.Embedding(cfg.stage_vocab, cfg.stage_dim)
        per_player = len(FLOAT_FEATURES) * 2 + sum(dim for _, dim in CAT_FEATURES.values())  # float+mask+cat
        d_in = len(_PLAYER_PREFIXES) * per_player + A_DIM + 2 * cfg.char_dim + cfg.stage_dim
        if cfg.spatial_features:
            d_in += len(SPATIAL_COLUMNS)

        self.ctx_proj = nn.Linear(d_in, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.token_head = nn.Linear(cfg.d_model, tok.vocab_size)
        self.aux_heads = nn.ModuleList([nn.Linear(cfg.d_model, A_VOCAB) for _ in offs])

        # Stick/trigger center grids (registered so they move with .to() and serialize).
        self.register_buffer("main_centers", scoring.STICK_CLUSTER_CENTERS_MAIN.clone())
        self.register_buffer("c_centers", scoring.STICK_CLUSTER_CENTERS_C.clone())
        self.register_buffer("trig_centers", scoring.TRIGGER_CENTERS.clone())
        # -1 means unavailable. A validated full-dataset artifact populates this at train start, and the
        # buffer then travels with checkpoints so later eval never depends on a mutable sidecar file.
        self.register_buffer("button_combo_counts", torch.full((scoring.N_BUTTON_COMBOS,), -1, dtype=torch.long))
        # Tokenizer tables. Derived from the embedded artifact, hence non-persistent.
        frames = token_frames(tok)
        alphabet = torch.tensor(tok.alphabet, dtype=torch.long)
        self.register_buffer("tok_merge_rank", build_merge_rank(tok), persistent=False)
        self.register_buffer("tok_frames", frames, persistent=False)
        self.register_buffer("tok_spans", torch.from_numpy(token_spans(tok)), persistent=False)
        self.register_buffer("tok_first_groups", frames[:, 0].clone(), persistent=False)
        self.register_buffer("alphabet_groups", code_groups(alphabet), persistent=False)
        # Memo table for frame code → token: alphabet hits are filled in now, substitutions are computed
        # once on first sight and cached (a pure function of the alphabet, so nothing to serialize).
        code_to_token = torch.full((N_JOINT_CODES,), -1, dtype=torch.int32)
        code_to_token[alphabet] = torch.arange(len(tok.alphabet), dtype=torch.int32)
        is_alphabet = torch.zeros(N_JOINT_CODES, dtype=torch.bool)
        is_alphabet[alphabet] = True
        self.register_buffer("code_to_token", code_to_token, persistent=False)
        self.register_buffer("code_is_alphabet", is_alphabet, persistent=False)
        for name, table in zip(("d_btn", "d_main", "d_c", "d_trig"), group_distance_tables(), strict=True):
            self.register_buffer(name, table, persistent=False)

    def distance_tables(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (self.d_btn, self.d_main, self.d_c, self.d_trig)

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

    def _spatial_features(self, features: dict[str, Tensor]) -> Tensor:
        """The derived spatial block in ``SPATIAL_COLUMNS`` order → ``[B, L, len(SPATIAL_COLUMNS)]``.
        ``preprocess`` emits it for every batch carrying the matchup ``stage`` column, so an absent
        key means the observation source predates matchup conditioning, not a transient gap."""
        missing = [name for name in SPATIAL_COLUMNS if name not in features]
        if missing:
            raise ValueError(
                f"spatial_features=True but the observation is missing {missing}; the source batch must "
                "carry the matchup 'stage' column for hal.training.features.preprocess to derive the block"
            )
        return torch.stack([features[name] for name in SPATIAL_COLUMNS], dim=-1)

    def _context_tokens(self, features: dict[str, Tensor]) -> Float[Tensor, "B L_ctx d_model"]:
        parts = [self._per_player_features(features, p) for p in _PLAYER_PREFIXES]
        if self.spatial_features:
            parts.append(self._spatial_features(features))
        # Ego controller history slice, assembled into a FRESH tensor (never mutating `features`, so the
        # targets built from stack_actions(features) stay intact). Per-sample history dropout (train only).
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
        """Backbone hidden (one rmsnorm'd vector per frame); callers apply the token / auxiliary heads."""
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


def resolve_codes(model: GPT, codes: Tensor) -> tuple[Tensor, float]:
    """Per-frame joint codes → alphabet token ids, substituting the nearest observed tuple for codes the
    alphabet does not contain, plus the substitution rate over the given frames. Substitutions are
    memoized in ``code_to_token`` (a pure function of the alphabet), so a code costs one nearest-neighbour
    search per process, not per step."""
    flat = codes.reshape(-1)
    tokens = model.code_to_token[flat].long()
    if bool((tokens < 0).any()):
        unseen = torch.unique(flat[tokens < 0])
        substitute = nearest_alphabet_token(unseen, model.alphabet_groups, model.distance_tables())
        model.code_to_token[unseen] = substitute.to(torch.int32)
        tokens = model.code_to_token[flat].long()
    rate = float((~model.code_is_alphabet[flat]).float().mean())
    return tokens.reshape(codes.shape), rate


@jaxtyped(typechecker=beartype)
def token_targets(model: GPT, base: Int[Tensor, "B L_full"], L_ctx: int) -> Int[Tensor, "B L_ctx"]:
    """The training target at every context position: the FIRST token of the greedy encoding of the
    ``max_token_frames`` frames that follow it. A pure function of the position — independent of how
    earlier frames were segmented — so every position trains in parallel and inference carries no
    segmentation state."""
    H = model.max_token_frames
    windows = base[:, 1:].unfold(1, H, 1)[:, :L_ctx]  # [B, L_ctx, H]
    tokens, _ = encode_windows(
        windows.reshape(-1, H).contiguous(), merge_rank=model.tok_merge_rank, merge_base=model.tokenizer.merge_base
    )
    return tokens[:, 0].reshape(base.shape[0], L_ctx)


@jaxtyped(typechecker=beartype)
def _valid_positions(ctx: Context, L_ctx: int) -> Bool[Tensor, "B L_ctx"]:
    """Context positions that are real (non-pad) frames. ``i >= ctx_pad`` also guarantees every target
    frame ``i + o`` (``o >= 1``) is real, so no further mask is needed."""
    pos = torch.arange(L_ctx, device=ctx.ctx_pad.device)
    return pos[None, :] >= ctx.ctx_pad[:, None]


@jaxtyped(typechecker=beartype)
def sample_positions(
    valid: Bool[Tensor, "B L_ctx"], n_per_sample: int, gen: torch.Generator | None = None
) -> Int[Tensor, " n_sel"]:
    """``B * n_per_sample`` flat indices into ``[B*L_ctx]``, drawn uniformly WITH REPLACEMENT from the
    pooled valid set — an unbiased estimator of the dense over-all-valid-positions mean, at a fraction
    of the memory a dense ``[B, L_ctx, vocab]`` logit tensor would need."""
    flat = valid.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    if flat.numel() == 0:
        raise ValueError("batch has no valid context positions (every sample is fully padded)")
    pick = torch.randint(flat.numel(), (valid.shape[0] * n_per_sample,), device=valid.device, generator=gen)
    return flat[pick]


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
    """One batch's loss terms: the primary token NLL at the sampled positions, plus 016's auxiliary
    per-offset/per-group frame NLL and its aligned transition flags over all valid positions."""

    token_nll: Tensor  # [n_sel] nats
    aux_nll: dict[tuple[int, str], Tensor]  # [n_valid] nats
    aux_trans: dict[tuple[int, str], Tensor]  # [n_valid] bool
    substitution_rate: float


def action_loss(model: GPT, batch: TrainBatch, cfg: TrainConfig, *, gen: torch.Generator | None = None) -> LossParts:
    """One backbone forward → the RLE token head at ``cfg.token_positions`` sampled positions per sample
    (dense when 0) + 016's auxiliary frame heads densely over valid positions.

    Positions are subsampled for the token head only because its logit tensor is ``vocab_size`` wide:
    dense at the default geometry would be ``B * L_ctx * 4096`` floats (>4 GB), while the estimator is
    unbiased for the same dense mean."""
    ctx = batch.context
    H = model.max_token_frames
    max_offset = max(model.aux_offsets, default=0)
    if batch.target.size(1) < max(H, max_offset):
        raise ValueError(
            f"target chunk has {batch.target.size(1)} frames, but the token horizon is {H} and the "
            f"farthest auxiliary head is {max_offset}"
        )
    h = model(ctx.features, ctx.ctx_pad)  # [B, L_ctx, d_model]
    a_full = torch.cat([stack_actions(ctx.features), batch.target], dim=1)
    q_full = _quantize(model, a_full)
    L_ctx = a_full.size(1) - batch.target.size(1)
    valid = _valid_positions(ctx, L_ctx)
    flat_valid = valid.reshape(-1)
    base, sub_rate = resolve_codes(model, group_codes(q_full[:, : L_ctx + H]))
    tgt_token = token_targets(model, base, L_ctx).reshape(-1)

    if cfg.token_positions > 0:
        sel = sample_positions(valid, cfg.token_positions, gen)
    else:
        sel = flat_valid.nonzero(as_tuple=False).squeeze(-1)
    h_sel = h.reshape(-1, h.shape[-1])[sel]
    token_logits = model.token_head(h_sel).float()
    token_nll = F.cross_entropy(token_logits, tgt_token[sel], reduction="none")

    aux_nll: dict[tuple[int, str], Tensor] = {}
    aux_trans: dict[tuple[int, str], Tensor] = {}
    if model.aux_offsets:
        bnd_full = scoring.transition_mask(q_full)  # pos t = (q[t+1] != q[t])
        for hi, o in enumerate(model.aux_offsets):
            logits = model.aux_heads[hi](h).float()
            tgt_idx = q_full[:, o : o + L_ctx]
            bnd_o = bnd_full[:, o - 1 : o - 1 + L_ctx]
            for name, c in group_nll(logits, tgt_idx, valid).items():
                aux_nll[(o, name)] = c
            for g, name in enumerate(_GROUP_NAMES):
                aux_trans[(o, name)] = bnd_o[..., g].reshape(-1)[flat_valid]
    return LossParts(token_nll=token_nll, aux_nll=aux_nll, aux_trans=aux_trans, substitution_rate=sub_rate)


def _weighted_mean(nll: Tensor, is_trans: Tensor, weight: float) -> Tensor:
    """Mean per-position NLL (nats) with transition positions upweighted by ``weight`` (016 semantics);
    ``weight == 1.0`` is exactly the plain mean."""
    if weight == 1.0:
        return nll.mean()
    w = torch.where(is_trans, weight, 1.0).to(nll.dtype)
    return (w * nll).sum() / w.sum()


def objective(parts: LossParts, aux_weight: float, transition_weight: float) -> Tensor:
    """Primary token NLL (plain mean over the sampled valid positions) + 016's auxiliary frame-head
    terms at ``aux_weight`` each, with the transition reweighting applied inside the auxiliary heads."""
    terms = [parts.token_nll.mean()]
    terms += [
        aux_weight * _weighted_mean(c, parts.aux_trans[key], transition_weight) for key, c in parts.aux_nll.items()
    ]
    return torch.stack(terms).sum()


# %%
@torch.no_grad()
def sample_tokens(
    model: GPT,
    h: Tensor,
    *,
    temp: float,
    min_p: float,
    argmax: bool = False,
    gen: torch.Generator | None = None,
) -> Int[Tensor, " n_slot"]:
    """One RLE token per row from the trunk hidden. No support masking is needed — the vocabulary
    contains only token sequences observed in training — so hygiene is temperature then min-p."""
    logits = model.token_head(h).float()
    if argmax:
        return logits.argmax(-1)
    probs = F.softmax(logits / temp, dim=-1)
    if min_p > 0:
        probs = probs * (probs >= min_p * probs.amax(dim=-1, keepdim=True))
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, 1, generator=gen).squeeze(-1)


@torch.no_grad()
def token_actions(model: GPT, tokens: Tensor, *, click_trigger_fix: bool) -> Float[Tensor, "n_slot n_frame d_action"]:
    """Sampled tokens → their frames as raw action vectors (rows past a token's span are stale and are
    sliced off by the caller using ``tok_spans``)."""
    a = _dequantize(model, model.tok_frames[tokens])
    if click_trigger_fix:
        a[..., _TRIGGER_L_CH] = torch.where(a[..., _BUTTON_L_CH] > 0.5, 1.0, a[..., _TRIGGER_L_CH])
        a[..., _TRIGGER_R_CH] = torch.where(a[..., _BUTTON_R_CH] > 0.5, 1.0, a[..., _TRIGGER_R_CH])
    return a


def _resolve_decode_args(temp: float, min_p: float, exec_cadence: int) -> None:
    """Validate the deploy knobs. 016's per-group temperatures and button-support floor are gone: a
    token spans several frames and all four groups, so there is no per-group axis to sharpen, and the
    vocabulary is already restricted to observed sequences."""
    if not math.isfinite(temp) or temp <= 0:
        raise ValueError(f"decode temperature must be > 0, got {temp}")
    if not math.isfinite(min_p) or not 0.0 <= min_p <= 1.0:
        raise ValueError(f"decode min_p must be in [0, 1], got {min_p}")
    if not isinstance(exec_cadence, int) or isinstance(exec_cadence, bool) or exec_cadence < 0:
        raise ValueError(f"exec_cadence must be a non-negative integer, got {exec_cadence!r}")


# %%
_PORT_TO_PREFIX: dict[int, str] = {1: "p1", 2: "p2"}


@dataclass
class _SlotState:
    """Per-slot rolling buffers + the frames still owed by the last sampled token."""

    flat_hist: list[dict] = field(default_factory=list)
    ego_hist: list[np.ndarray] = field(default_factory=list)
    queue: list[np.ndarray] = field(default_factory=list)
    last_id: int | None = None


def _slot_window(flat_hist: list[dict], ego_hist: list[np.ndarray], ego_prefix: str, L_ctx: int) -> dict:
    """``[1, L_ctx]`` raw-column batch for one slot's rolling buffers.

    Before the buffers fill (the first ``L_ctx`` closed-loop frames) the window is LEFT-PADDED with
    zeros; the pad is hidden from attention via ``ctx_pad``, so the policy acts from frame 0 while the
    buffer fills with real gameplay. The ego action history is front-padded by
    ``len(flat) - len(ego)`` neutrals so ``ego[i]`` stays aligned with the gamestate it produced — the
    real ``(post_i, pre_i)`` pairing, not padding."""
    pad = L_ctx - len(flat_hist)
    out: dict[str, np.ndarray] = {}
    for k in flat_hist[0]:
        dtype = np.int32 if isinstance(flat_hist[0][k], int) else np.float32
        vals = [h[k] for h in flat_hist]
        out[k] = np.array(([0] * pad) + vals if pad > 0 else vals, dtype=dtype)
    ego_aligned = [NEUTRAL_ACTION] * (len(flat_hist) - len(ego_hist)) + list(ego_hist)
    if pad > 0:
        ego_aligned = [NEUTRAL_ACTION] * pad + ego_aligned
    hist = np.stack(ego_aligned)
    for i, ch in enumerate(ACTION_CHANNELS):
        col = hist[:, i]
        out[f"{ego_prefix}_{ch}"] = (
            (col > 0.5).astype(np.int32) if ch.startswith("button_") else col.astype(np.float32)
        )
    out.pop("frame", None)
    return {k: v[None, ...] for k, v in relabel_ego(out, ego_prefix).items()}


class RLEPolicy:
    """Variable-horizon ``BatchPolicy``: sample one RLE token, execute its frames, then replan.

    Written here rather than configured out of ``hal.training.closed_loop.RecedingHorizon`` because
    that class's contract is a FIXED execution horizon ``s`` (replan clock, committed-prefix handoff,
    ``[n, s, A_DIM]`` chunks), and the horizon here is whatever the sampled token spans — 1 frame or
    16, differing across slots on the same frame. What it shares with ``RecedingHorizon`` is the
    invariant plumbing, and that is reused wholesale: ``flatten_canonical_frame`` for the observation,
    ``relabel_ego`` + ``preprocess`` for the feature codec, ``action_vec_to_controller`` for the wire.

    Instant-restart boundaries (the slot's canonical frame id drops below the last) clear that slot's
    buffers AND its pending queue, so no action planned for the previous match is ever executed in the
    next one. ``exec_cadence > 0`` truncates every token to its first ``k`` frames (the fixed-cadence
    ablation); 0 executes the token to completion. Construct fresh per eval wave."""

    def __init__(
        self,
        model: GPT,
        stats: dict[str, FeatureStats],
        cfg: TrainConfig,
        *,
        temp: float,
        min_p: float,
        click_trigger_fix: bool,
        exec_cadence: int,
        device: str = DEVICE,
        gen: torch.Generator | None = None,
    ) -> None:
        _resolve_decode_args(temp, min_p, exec_cadence)
        self.model = model
        self.stats = stats
        self.L_ctx = cfg.L_ctx
        self.temp = temp
        self.min_p = min_p
        self.click_trigger_fix = click_trigger_fix
        self.exec_cadence = exec_cadence
        self.device = device
        self.gen = gen
        self.slots: dict[Slot, _SlotState] = {}
        # Deployment counters: one forward per replan, however many frames that token buys.
        self.n_replans = 0
        self.n_frames = 0

    def __call__(self, frame_index: int, obs: Mapping[Slot, dict]) -> Mapping[Slot, ControllerInputs]:
        live = list(obs)
        for slot in live:
            st = self.slots.setdefault(slot, _SlotState())
            fid = obs[slot]["id"]
            if st.last_id is not None and fid < st.last_id:
                st.flat_hist.clear()
                st.ego_hist.clear()
                st.queue.clear()
            st.last_id = fid
            st.flat_hist.append(flatten_canonical_frame(obs[slot]))
            if len(st.flat_hist) > self.L_ctx:
                st.flat_hist.pop(0)
        due = [slot for slot in live if not self.slots[slot].queue]
        if due:
            self._replan(due)
        actions: dict[Slot, np.ndarray] = {}
        for slot in live:
            st = self.slots[slot]
            if not st.queue:
                raise RuntimeError(f"slot {slot} has no pending action after replanning")
            a = st.queue.pop(0)
            actions[slot] = a
            st.ego_hist.append(a.astype(np.float32))
            if len(st.ego_hist) > self.L_ctx:
                st.ego_hist.pop(0)
            self.n_frames += 1
        return {slot: action_vec_to_controller(a) for slot, a in actions.items()}

    @torch.no_grad()
    def _replan(self, due: list[Slot]) -> None:
        """One batched trunk forward + one sampled token per due slot; enqueue the token's frames."""
        per_slot = [
            _slot_window(
                self.slots[slot].flat_hist,
                self.slots[slot].ego_hist,
                _PORT_TO_PREFIX[slot.port],
                self.L_ctx,
            )
            for slot in due
        ]
        stacked = {k: np.concatenate([d[k] for d in per_slot], axis=0) for k in per_slot[0]}
        feats = {k: v.to(self.device) for k, v in preprocess(stacked, self.stats).items()}
        ctx_pad = torch.tensor(
            [max(0, self.L_ctx - len(self.slots[slot].flat_hist)) for slot in due],
            dtype=torch.long,
            device=self.device,
        )
        h = self.model(feats, ctx_pad)[:, -1]
        tokens = sample_tokens(self.model, h, temp=self.temp, min_p=self.min_p, gen=self.gen)
        spans = self.model.tok_spans[tokens].tolist()
        frames = token_actions(self.model, tokens, click_trigger_fix=self.click_trigger_fix).cpu().numpy()
        self.n_replans += len(due)
        for i, slot in enumerate(due):
            span = int(spans[i]) if self.exec_cadence == 0 else min(int(spans[i]), self.exec_cadence)
            self.slots[slot].queue = [frames[i, j] for j in range(span)]


@dataclass(frozen=True, slots=True)
class DecodeSettings:
    temp: float
    min_p: float
    click_trigger_fix: bool
    exec_cadence: int


def _decode_settings(
    cfg: TrainConfig,
    *,
    temp: float | None = None,
    min_p: float | None = None,
    click_trigger_fix: bool | None = None,
    exec_cadence: int | None = None,
) -> DecodeSettings:
    settings = DecodeSettings(
        temp=cfg.decode_temp if temp is None else temp,
        min_p=cfg.decode_min_p if min_p is None else min_p,
        click_trigger_fix=cfg.decode_click_trigger_fix if click_trigger_fix is None else click_trigger_fix,
        exec_cadence=cfg.exec_cadence if exec_cadence is None else exec_cadence,
    )
    _resolve_decode_args(settings.temp, settings.min_p, settings.exec_cadence)
    return settings


def make_policy(
    model: GPT,
    stats: dict[str, FeatureStats],
    cfg: TrainConfig,
    *,
    device: str = DEVICE,
    settings: DecodeSettings | None = None,
    decode_seed: int | None = None,
) -> RLEPolicy:
    """Fresh closed-loop policy for one eval wave (rolling state must not leak across waves)."""
    resolved = settings or _decode_settings(cfg)
    model_device = next(model.parameters()).device
    gen = None if decode_seed is None else torch.Generator(device=model_device).manual_seed(decode_seed)
    return RLEPolicy(
        model,
        stats,
        cfg,
        temp=resolved.temp,
        min_p=resolved.min_p,
        click_trigger_fix=resolved.click_trigger_fix,
        exec_cadence=resolved.exec_cadence,
        device=device,
        gen=gen,
    )


# %%
def validate_config(cfg: TrainConfig, *, tok: ActionTokenizer) -> None:
    """Fail before W&B, loader construction, or Dolphin startup on invalid experiment geometry."""
    positive_ints = {
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "L_ctx": cfg.L_ctx,
        "batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
        "max_steps": cfg.max_steps,
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
    }
    for name, value in positive_ints.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if cfg.d_model % cfg.n_heads != 0:
        raise ValueError(f"d_model={cfg.d_model} must be divisible by n_heads={cfg.n_heads}")
    head_dim = cfg.d_model // cfg.n_heads
    if head_dim % 2:
        raise ValueError(f"rotary attention head_dim=d_model/n_heads={head_dim} must be even")
    offsets = tuple(cfg.aux_head_offsets)
    if any(not isinstance(o, int) or isinstance(o, bool) or o < 1 for o in offsets):
        raise ValueError(f"aux_head_offsets must be positive integers, got {offsets}")
    if len(set(offsets)) != len(offsets):
        raise ValueError(f"aux_head_offsets must be unique, got {offsets}")
    if offsets and max(offsets) > VAL_L_CHUNK:
        raise ValueError(f"max(aux_head_offsets)={max(offsets)} exceeds frozen VAL_L_CHUNK={VAL_L_CHUNK}")
    if tok.max_token_frames > VAL_L_CHUNK:
        raise ValueError(f"tokenizer max_token_frames={tok.max_token_frames} exceeds VAL_L_CHUNK={VAL_L_CHUNK}")
    if not isinstance(cfg.token_positions, int) or isinstance(cfg.token_positions, bool) or cfg.token_positions < 0:
        raise ValueError(f"token_positions must be a non-negative integer, got {cfg.token_positions!r}")
    if cfg.token_positions > cfg.L_ctx:
        raise ValueError(f"token_positions={cfg.token_positions} exceeds L_ctx={cfg.L_ctx}")
    finite_nonnegative = {"aux_loss_weight": cfg.aux_loss_weight, "weight_decay": cfg.weight_decay}
    for name, value in finite_nonnegative.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    finite_positive = {
        "muon_lr": cfg.muon_lr,
        "adam_lr": cfg.adam_lr,
        "eval_parallel_per_cpu": cfg.eval_parallel_per_cpu,
        "eval_timeout_seconds": cfg.eval_timeout_seconds,
        "transition_loss_weight": cfg.transition_loss_weight,
    }
    for name, value in finite_positive.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    if not math.isfinite(cfg.history_dropout_p) or not 0.0 <= cfg.history_dropout_p <= 1.0:
        raise ValueError(f"history_dropout_p must be in [0, 1], got {cfg.history_dropout_p!r}")
    if not isinstance(cfg.spatial_features, bool):
        raise ValueError(f"spatial_features must be a bool, got {cfg.spatial_features!r}")
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
    if not cfg.data_root or not cfg.val_split:
        raise ValueError("data_root and val_split must be non-empty")
    if cfg.amp_dtype not in ("bfloat16", "float32"):
        raise ValueError(f"amp_dtype must be 'bfloat16' or 'float32', got {cfg.amp_dtype!r}")
    _resolve_decode_args(cfg.decode_temp, cfg.decode_min_p, cfg.exec_cadence)
    if cfg.decode_click_trigger_fix not in (True, False):
        raise ValueError(f"decode_click_trigger_fix must be a bool, got {cfg.decode_click_trigger_fix!r}")


def _load_model_state(model: GPT, state_dict: dict[str, Tensor]) -> None:
    """Load 018 state, tolerating only the count buffer when inspecting an older checkpoint. A
    checkpoint trained with a different ``spatial_features`` setting or a different tokenizer has a
    different ``ctx_proj`` / ``token_head`` shape and is rejected here."""
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing - {"button_combo_counts"} or unexpected:
        raise RuntimeError(f"checkpoint/model mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")


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
    else — input projection, token head, auxiliary heads, embeddings, biases — split by weight-decay
    eligibility. Exactly two LRs; the partition asserts full coverage so no parameter can silently
    escape an optimizer."""
    muon_params = [p for p in model.blocks.parameters() if p.ndim >= 2]
    muon_ids = {id(p) for p in muon_params}
    embed_ids = {id(p) for m in (model.cat_embeds, model.char_emb, model.stage_emb) for p in m.parameters()}
    head_modules = (model.token_head, model.aux_heads)
    head_ids = set() if cfg.head_weight_decay else {id(p) for m in head_modules for p in m.parameters()}

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for p in model.parameters():
        if id(p) in muon_ids:
            continue
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
@contextlib.contextmanager
def _evaluation_mode(model: nn.Module) -> Iterator[None]:
    """Temporarily enter eval mode and restore the exact prior mode on every exit."""
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def _masked_mean_bits(nats: Tensor, mask: Tensor) -> float:
    """Mean of per-position NLL (nats) over the masked subset, in bits; 0.0 when the subset is empty."""
    return (nats[mask].mean().item() / _LN2) if bool(mask.any()) else 0.0


def _bool_mean(parts: list[Tensor], *, invert: bool = False) -> float:
    values = torch.cat(parts)
    if values.numel() == 0:
        return 0.0
    return ((~values) if invert else values).float().mean().item()


@dataclass(frozen=True, slots=True)
class FirstFrameStats:
    """Everything the token distribution implies about the FIRST future frame, per context position.

    Every token names the action of frame ``t+1``, so ``P(frame t+1 group g = k)`` is the token mass
    on tokens whose first frame has that class — one scatter-add per group. These are therefore the
    exact analogues of 016's per-group head outputs and score against the same targets."""

    token_nll: Tensor  # [B, L_ctx] nats of the true first token
    group_nll: dict[str, Tensor]  # [B, L_ctx] nats of the true frame-1 class, per group
    group_brier: dict[str, Tensor]  # [B, L_ctx] multiclass Brier per group
    group_argmax: dict[str, Tensor]  # [B, L_ctx] argmax class per group
    btn_marginals: Tensor  # [B, L_ctx, 8] per-button press probability
    rare_mass: Tensor  # [B, L_ctx]
    unseen_mass: Tensor  # [B, L_ctx]
    trigger_full_l: Tensor  # [B, L_ctx] P(trigger_l == 1.0)
    trigger_full_r: Tensor  # [B, L_ctx]
    ablation_nll: dict[str, Tensor]  # ablated trunk → [B, L_ctx] token NLL
    ablation_kl: dict[str, Tensor]  # KL(full ‖ ablated) over the token distribution, bits


@torch.no_grad()
def first_frame_stats(
    model: GPT,
    h: Tensor,
    tgt_token: Tensor,
    q_first: Tensor,
    *,
    rare_mask: Tensor,
    unseen_mask: Tensor,
    ablations: dict[str, Tensor],
    chunk: int = 8,
) -> FirstFrameStats:
    """Token NLL + the implied first-future-frame marginals, computed in position chunks so the
    ``[B, L_ctx, vocab_size]`` logit tensor never has to exist all at once."""
    B, L, _ = h.shape
    device = h.device
    first_groups = model.tok_first_groups  # [vocab, n_groups]
    combo_bits = scoring.combo_to_buttons(torch.arange(scoring.N_BUTTON_COMBOS, device=device))
    n_trig = model.trig_centers.shape[0]
    token_nll = torch.empty(B, L, device=device)
    group_nll = {name: torch.empty(B, L, device=device) for name in _GROUP_NAMES}
    group_brier = {name: torch.empty(B, L, device=device) for name in _GROUP_NAMES}
    group_argmax = {name: torch.empty(B, L, dtype=torch.long, device=device) for name in _GROUP_NAMES}
    btn_marginals = torch.empty(B, L, _N_BUTTONS, device=device)
    rare_mass = torch.empty(B, L, device=device)
    unseen_mass = torch.empty(B, L, device=device)
    trigger_full_l = torch.empty(B, L, device=device)
    trigger_full_r = torch.empty(B, L, device=device)
    ablation_nll = {name: torch.empty(B, L, device=device) for name in ablations}
    ablation_kl = {name: torch.empty(B, L, device=device) for name in ablations}
    for start in range(0, L, chunk):
        sl = slice(start, min(start + chunk, L))
        logp = F.log_softmax(model.token_head(h[:, sl]).float(), dim=-1)  # [B, c, vocab]
        probs = logp.exp()
        token_nll[:, sl] = -logp.gather(2, tgt_token[:, sl, None]).squeeze(-1)
        for name, h_other in ablations.items():
            logq = F.log_softmax(model.token_head(h_other[:, sl]).float(), dim=-1)
            ablation_nll[name][:, sl] = -logq.gather(2, tgt_token[:, sl, None]).squeeze(-1)
            ablation_kl[name][:, sl] = (probs * (logp - logq)).sum(-1) / _LN2
        for g, name in enumerate(_GROUP_NAMES):
            marg = torch.zeros(probs.shape[0], probs.shape[1], _GROUP_VOCABS[g], device=device)
            marg.index_add_(2, first_groups[:, g], probs)
            true = q_first[:, sl, g]
            group_nll[name][:, sl] = -marg.gather(2, true[..., None]).squeeze(-1).clamp_min(1e-12).log()
            onehot = F.one_hot(true, _GROUP_VOCABS[g]).to(marg.dtype)
            group_brier[name][:, sl] = (marg - onehot).pow(2).sum(-1)
            group_argmax[name][:, sl] = marg.argmax(-1)
            if g == _BUTTONS_G:
                btn_marginals[:, sl] = marg @ combo_bits.to(marg.dtype)
                rare_mass[:, sl] = marg[..., rare_mask].sum(-1)
                unseen_mass[:, sl] = marg[..., unseen_mask].sum(-1)
            elif g == _TRIG_G:
                trig = marg.reshape(*marg.shape[:2], n_trig, n_trig)
                trigger_full_l[:, sl] = trig[:, :, -1, :].sum(-1)
                trigger_full_r[:, sl] = trig[:, :, :, -1].sum(-1)
    return FirstFrameStats(
        token_nll=token_nll,
        group_nll=group_nll,
        group_brier=group_brier,
        group_argmax=group_argmax,
        btn_marginals=btn_marginals,
        rare_mass=rare_mass,
        unseen_mass=unseen_mass,
        trigger_full_l=trigger_full_l,
        trigger_full_r=trigger_full_r,
        ablation_nll=ablation_nll,
        ablation_kl=ablation_kl,
    )


@jaxtyped(typechecker=beartype)
def cover_bits(
    token_nll: Float[Tensor, "B L_ctx"],
    tgt_token: Int[Tensor, "B L_ctx"],
    spans: Int[Tensor, " vocab"],
    valid: Bool[Tensor, "B L_ctx"],
    H: int,
) -> tuple[Float[Tensor, " n_start"], Int[Tensor, " n_start"]]:
    """Bits to code the next ``H`` frames from each start position, via the TOKEN COVER.

    Token 1 is scored at the start position, token 2 at ``start + span(token 1)``, and so on until the
    cover reaches ``H`` frames — each token from its own boundary position, which is itself a real
    context position whose target is the greedy first token, so the whole sum is teacher-forced. The
    last token may overshoot frame ``start + H``; its bits are counted in full, making this an upper
    bound on the exact ``H``-frame codelength. Start positions are restricted to ``t <= L_ctx-1-H`` so
    every boundary position stays inside the context. Also returns how many tokens each cover spent —
    the deployment-consistent token rate, which is NOT the per-position mean span (a cover lands on
    boundaries, i.e. on transitions, while an arbitrary context position usually sits mid-hold)."""
    B, L = token_nll.shape
    device = token_nll.device
    n_start = L - H
    if n_start <= 0:
        raise ValueError(f"L_ctx={L} must exceed the token horizon H={H} for the cover metric")
    start = torch.arange(n_start, device=device)[None, :].expand(B, n_start)
    cur = start.clone()
    covered = torch.zeros_like(start)
    n_tokens = torch.zeros_like(start)
    bits = torch.zeros(B, n_start, device=device)
    for _ in range(H):
        active = covered < H
        step_nll = token_nll.gather(1, cur)
        bits = bits + torch.where(active, step_nll, torch.zeros_like(step_nll))
        span = spans[tgt_token.gather(1, cur)]
        step = torch.where(active, span, torch.zeros_like(span))
        covered = covered + step
        n_tokens = n_tokens + active.long()
        cur = (cur + step).clamp(max=L - 1)
    keep = valid[:, :n_start]
    return (bits / (_LN2 * H))[keep], n_tokens[keep]


@torch.no_grad()
def val_metrics(model: GPT, val_cache: list[TrainBatch], cfg: TrainConfig) -> dict[str, float]:
    """Proper-scoring + sample-space metrics over the cached val batches.

    ``loss`` is the chunk-comparable bits/frame from the token cover; the per-group NLL / Brier /
    change / persistence family is computed on the first-future-frame marginal implied by the token
    distribution and is therefore exactly comparable to 016's same-named metrics; the auxiliary heads
    keep 016's ``nll_off{o}``; and the copycat / spatial ablations are re-expressed on the token
    distribution (Δ bits per token and KL, both at valid positions)."""
    with _evaluation_mode(model):
        return _val_metrics_eval(model, val_cache, cfg)


def _val_metrics_eval(model: GPT, val_cache: list[TrainBatch], cfg: TrainConfig) -> dict[str, float]:
    H = model.max_token_frames
    device = model.main_centers.device
    counts_available = bool((model.button_combo_counts >= 0).all())
    rare_mask = model.button_combo_counts < cfg.diagnostic_rare_button_count
    unseen_mask = model.button_combo_counts == 0
    cover: list[Tensor] = []
    cover_tokens: list[Tensor] = []
    token_nll_v: list[Tensor] = []
    span_v: list[Tensor] = []
    tokens_v: list[Tensor] = []
    sub_rates: list[float] = []
    aux_cat: dict[tuple[int, str], list[Tensor]] = {}
    group_nll_v: dict[str, list[Tensor]] = {}
    brier_v: dict[str, list[Tensor]] = {}
    trans_v: dict[str, list[Tensor]] = {}
    pred_change_v: dict[str, list[Tensor]] = {}
    pred_next_change_v: dict[str, list[Tensor]] = {}
    pred_persist_v: dict[str, list[Tensor]] = {}
    true_change_v: dict[str, list[Tensor]] = {}
    btn_probs: list[Tensor] = []
    btn_tgts: list[Tensor] = []
    multipress: list[Tensor] = []
    rare_mass: list[Tensor] = []
    unseen_mass: list[Tensor] = []
    click_invalid_l: list[Tensor] = []
    click_invalid_r: list[Tensor] = []
    ablation_dnll: dict[str, list[Tensor]] = {}
    ablation_kl: dict[str, list[Tensor]] = {}
    for batch in val_cache:
        ctx = batch.context
        h = model(ctx.features, ctx.ctx_pad)
        ablations: dict[str, Tensor] = {}
        # Copycat probe: a second backbone forward with the ego's own controller history zeroed.
        hist_ablated = dict(ctx.features)
        for ch in ACTION_CHANNELS:
            hist_ablated[f"ego_{ch}"] = torch.zeros_like(hist_ablated[f"ego_{ch}"])
        ablations["hist"] = model(hist_ablated, ctx.ctx_pad)
        if model.spatial_features:
            # The spatial block re-presented as fully INVALID — values zeroed, masks raised — i.e. the
            # representation of an unobserved frame, not merely a zeroed one.
            spatial_ablated = dict(ctx.features)
            for name in SPATIAL_FEATURES:
                spatial_ablated[name] = torch.zeros_like(spatial_ablated[name])
            for name in SPATIAL_MASKS:
                spatial_ablated[name] = torch.ones_like(spatial_ablated[name])
            ablations["spatial"] = model(spatial_ablated, ctx.ctx_pad)

        a_full = torch.cat([stack_actions(ctx.features), batch.target], dim=1)
        q_full = _quantize(model, a_full)
        L_ctx = a_full.size(1) - batch.target.size(1)
        valid = _valid_positions(ctx, L_ctx)
        flat_valid = valid.reshape(-1)
        base, sub_rate = resolve_codes(model, group_codes(q_full[:, : L_ctx + H]))
        sub_rates.append(sub_rate)
        tgt_token = token_targets(model, base, L_ctx)
        q_first = q_full[:, 1 : 1 + L_ctx]  # the action of frame t+1 per context position t
        stats = first_frame_stats(
            model, h, tgt_token, q_first, rare_mask=rare_mask, unseen_mask=unseen_mask, ablations=ablations
        )
        cover_b, cover_n = cover_bits(stats.token_nll, tgt_token, model.tok_spans, valid, H)
        cover.append(cover_b)
        cover_tokens.append(cover_n)
        token_nll_v.append(stats.token_nll.reshape(-1)[flat_valid])
        span_v.append(model.tok_spans[tgt_token].reshape(-1)[flat_valid])
        tokens_v.append(tgt_token.reshape(-1)[flat_valid])
        for name, values in stats.ablation_nll.items():
            ablation_dnll.setdefault(name, []).append((values - stats.token_nll).reshape(-1)[flat_valid])
        for name, values in stats.ablation_kl.items():
            ablation_kl.setdefault(name, []).append(values.reshape(-1)[flat_valid])

        cur_idx = _quantize(model, stack_actions(ctx.features))  # [B, L_ctx, n_groups] current frames
        true_change = scoring.transition_mask(torch.cat([cur_idx, q_first[:, -1:]], dim=1))
        adjacent_valid = valid[:, 1:] & valid[:, :-1]
        for g, name in enumerate(_GROUP_NAMES):
            group_nll_v.setdefault(name, []).append(stats.group_nll[name].reshape(-1)[flat_valid])
            brier_v.setdefault(name, []).append(stats.group_brier[name].reshape(-1)[flat_valid])
            pred_id = stats.group_argmax[name]
            tc = true_change[..., g]
            trans_v.setdefault(name, []).append(tc.reshape(-1)[flat_valid])
            pred_change = (pred_id != cur_idx[..., g]) & valid
            pred_change_v.setdefault(name, []).append(pred_change)
            pred_next_change_v.setdefault(name, []).append(pred_change.reshape(-1)[flat_valid])
            pred_persist_v.setdefault(name, []).append((pred_id[:, 1:] != pred_id[:, :-1])[adjacent_valid])
            true_change_v.setdefault(name, []).append(tc & valid)
        btn_probs.append(stats.btn_marginals.reshape(-1, _N_BUTTONS)[flat_valid])
        tgt_btn = _dequantize(model, q_first)[..., _N_CONT:].reshape(-1, _N_BUTTONS)[flat_valid]
        btn_tgts.append(tgt_btn)
        multipress.append((tgt_btn > 0.5).sum(-1) >= 2)
        if counts_available:
            rare_mass.append(stats.rare_mass.reshape(-1)[flat_valid])
            unseen_mass.append(stats.unseen_mass.reshape(-1)[flat_valid])
        marg_btn = stats.btn_marginals.reshape(-1, _N_BUTTONS)[flat_valid]
        click_invalid_l.append(
            marg_btn[:, _BUTTON_L_CH - _N_CONT] * (1.0 - stats.trigger_full_l.reshape(-1)[flat_valid])
        )
        click_invalid_r.append(
            marg_btn[:, _BUTTON_R_CH - _N_CONT] * (1.0 - stats.trigger_full_r.reshape(-1)[flat_valid])
        )
        if model.aux_offsets:
            for hi, o in enumerate(model.aux_offsets):
                logits = model.aux_heads[hi](h).float()
                tgt_idx = q_full[:, o : o + L_ctx]
                for name, c in group_nll(logits, tgt_idx, valid).items():
                    aux_cat.setdefault((o, name), []).append(c)

    token_nll = torch.cat(token_nll_v)
    spans = torch.cat(span_v).float()
    logloss, brier = scoring.bernoulli_scores_from_probs(torch.cat(btn_probs), torch.cat(btn_tgts))
    marginal_sum = sum(torch.cat(group_nll_v[name]).mean().item() for name in _GROUP_NAMES) / _LN2
    q50, q90 = torch.quantile(spans, torch.tensor([0.5, 0.9], device=device)).tolist()
    out = {
        "loss": torch.cat(cover).mean().item(),  # bits/frame over the 16-frame cover — the headline
        "nll_token": token_nll.mean().item() / _LN2,
        "nll_frame1_marginals": marginal_sum,  # 016's val/loss analogue (sum of frame-1 group marginals)
        # Deployment-consistent rate: tokens spent covering H frames from a cover boundary.
        "tokens_per_frame": torch.cat(cover_tokens).float().mean().item() / H,
        # Per-CONTEXT-POSITION span of the target token — larger, because an arbitrary position
        # usually sits mid-hold while a cover boundary sits on a transition.
        "span_mean": spans.mean().item(),
        "span_p50": q50,
        "span_p90": q90,
        "span_max": spans.max().item(),
        "span_frac_len1": (spans == 1).float().mean().item(),
        "vocab_utilization": torch.unique(torch.cat(tokens_v)).numel() / model.vocab_size,
        "substitution_rate": float(np.mean(sub_rates)),
        "btn_logloss": logloss.item(),
        "btn_brier": brier.item(),
        "btn_multipress": torch.cat(multipress).float().mean().item(),
        "btn_counts_available": float(counts_available),
        "click_trigger_invalid_l_mass": torch.cat(click_invalid_l).mean().item(),
        "click_trigger_invalid_r_mass": torch.cat(click_invalid_r).mean().item(),
    }
    out["click_trigger_invalid_mass"] = 0.5 * (
        out["click_trigger_invalid_l_mass"] + out["click_trigger_invalid_r_mass"]
    )
    if counts_available:
        out["btn_rare_mass"] = torch.cat(rare_mass).mean().item()
        out["btn_unseen_mass"] = torch.cat(unseen_mass).mean().item()
        out["btn_rare_count_threshold"] = float(cfg.diagnostic_rare_button_count)
    for name, parts in ablation_dnll.items():
        out[f"ablate_{name}_dnll_token"] = torch.cat(parts).mean().item() / _LN2  # positive ⇒ the input helps
        out[f"ablate_{name}_kl"] = torch.cat(ablation_kl[name]).mean().item()
    for name in _GROUP_NAMES:
        trans = torch.cat(trans_v[name])
        nll_g = torch.cat(group_nll_v[name])
        out[f"nll_{name}"] = nll_g.mean().item() / _LN2
        out[f"nll_{name}_trans"] = _masked_mean_bits(nll_g, trans)
        out[f"nll_{name}_hold"] = _masked_mean_bits(nll_g, ~trans)
        out[f"trans_rate_{name}"] = trans.float().mean().item()
        out[f"brier_{name}"] = torch.cat(brier_v[name]).mean().item()
        out[f"pred_change_rate_{name}"] = _bool_mean(pred_next_change_v[name])
        out[f"pred_persistence_{name}"] = _bool_mean(pred_persist_v[name], invert=True)
        out[f"changeF1_{name}"] = scoring.change_event_prf(
            torch.cat(pred_change_v[name]), torch.cat(true_change_v[name])
        )[2]
    for o in model.aux_offsets:
        out[f"nll_off{o}"] = sum(torch.cat(aux_cat[(o, name)]).mean().item() for name in _GROUP_NAMES) / _LN2
    return out


# %%
def _slice_batch(batch: TrainBatch, n: int) -> TrainBatch:
    """First ``n`` examples of a frozen batch, preserving Context structure."""
    return TrainBatch(
        context=Context(
            features={name: value[:n] for name, value in batch.context.features.items()},
            ctx_pad=batch.context.ctx_pad[:n],
        ),
        target=batch.target[:n],
    )


def _gradient_dot(a: tuple[Tensor, ...], b: tuple[Tensor, ...]) -> Tensor:
    return torch.stack([torch.dot(x.reshape(-1), y.reshape(-1)) for x, y in zip(a, b, strict=True)]).sum()


def _gradient_cosine(a: tuple[Tensor, ...], b: tuple[Tensor, ...], norm_a: Tensor, norm_b: Tensor) -> Tensor:
    denom = (norm_a * norm_b).clamp_min(torch.finfo(norm_a.dtype).tiny)
    return (_gradient_dot(a, b) / denom).clamp(-1.0, 1.0)


def gradient_diagnostics(model: GPT, batch: TrainBatch, cfg: TrainConfig) -> dict[str, float]:
    """016's shared-trunk gradient diagnostic, with the token head in the primary slot: does each
    auxiliary frame head ask the shared representation to move the same way as the deployed token
    head? Head parameters are excluded — the question is about the trunk. ``autograd.grad`` leaves
    ``parameter.grad`` untouched and eval mode disables history dropout, so this cannot perturb the
    optimizer or the RNG stream."""
    with _evaluation_mode(model):
        return _gradient_diagnostics_eval(model, batch, cfg)


def _gradient_diagnostics_eval(model: GPT, batch: TrainBatch, cfg: TrainConfig) -> dict[str, float]:
    diagnostic_batch = _slice_batch(batch, min(cfg.gradient_diagnostic_batch_size, batch.context.batch))
    parts = action_loss(model, diagnostic_batch, cfg)
    losses: dict[str, Tensor] = {"token": parts.token_nll.mean()}
    for o in model.aux_offsets:
        losses[f"aux{o}"] = torch.stack(
            [
                _weighted_mean(parts.aux_nll[(o, name)], parts.aux_trans[(o, name)], cfg.transition_loss_weight)
                for name in _GROUP_NAMES
            ]
        ).sum()
    trunk = tuple(p for name, p in model.named_parameters() if not name.startswith(("token_head.", "aux_heads.")))
    gradients: dict[str, tuple[Tensor, ...]] = {}
    for i, (name, loss) in enumerate(losses.items()):
        gradients[name] = tuple(g.detach() for g in torch.autograd.grad(loss, trunk, retain_graph=i + 1 < len(losses)))
    norms = {name: _gradient_dot(g, g).sqrt() for name, g in gradients.items()}
    out = {f"grad/{name}_norm": norm.item() for name, norm in norms.items()}
    for name in losses:
        if name != "token":
            out[f"grad/cos_token_{name}"] = _gradient_cosine(
                gradients["token"], gradients[name], norms["token"], norms[name]
            ).item()
    aux_names = [name for name in losses if name != "token"]
    if aux_names:
        weighted_aux = tuple(
            cfg.aux_loss_weight * sum((gradients[name][pi] for name in aux_names), start=torch.zeros_like(p))
            for pi, p in enumerate(trunk)
        )
        aux_norm = _gradient_dot(weighted_aux, weighted_aux).sqrt()
        out["grad/weighted_aux_norm"] = aux_norm.item()
        out["grad/weighted_aux_to_token_norm"] = (
            aux_norm / norms["token"].clamp_min(norms["token"].new_tensor(1e-30))
        ).item()
        out["grad/cos_token_weighted_aux"] = _gradient_cosine(
            gradients["token"], weighted_aux, norms["token"], aux_norm
        ).item()
    return out


# %%
def _slice_context(ctx: Context, n: int) -> Context:
    """First ``n`` rows of a Context (closed-loop-style batch for the decode benchmark)."""
    return Context(features={name: value[:n] for name, value in ctx.features.items()}, ctx_pad=ctx.ctx_pad[:n])


@torch.no_grad()
def bench_decode(model: GPT, ctx: Context, cfg: TrainConfig, *, n_iters: int) -> dict[str, float]:
    """Wall-clock per replan (one trunk forward + one sampled token + its frame expansion) and the
    AMORTIZED ms/frame that follows from the sampled token spans, against the 16.6 ms real-time budget.
    A replan buys ``span`` frames of play, so a policy that is over budget per replan can still be
    comfortably real-time."""
    settings = _decode_settings(cfg)
    with _evaluation_mode(model):

        def replan() -> Tensor:
            h = model(ctx.features, ctx.ctx_pad)[:, -1]
            tokens = sample_tokens(model, h, temp=settings.temp, min_p=settings.min_p)
            token_actions(model, tokens, click_trigger_fix=settings.click_trigger_fix)
            return model.tok_spans[tokens]

        for _ in range(2):
            spans = replan()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        span_total = 0.0
        for _ in range(n_iters):
            span_total += float(replan().float().mean())
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n_iters * 1000
    mean_span = span_total / n_iters
    cadence = mean_span if settings.exec_cadence == 0 else min(mean_span, float(settings.exec_cadence))
    out = {"ms_per_replan": ms, "mean_span": mean_span, "ms_per_frame": ms / cadence}
    print(
        f"[bench] slots={ctx.batch:3d}  {ms:7.2f} ms/replan  mean span {mean_span:5.2f} frames  "
        f"{out['ms_per_frame']:6.2f} ms/frame  budget {_FRAME_MS:.1f} ms/frame "
        f"({out['ms_per_frame'] / _FRAME_MS:.2f}x)",
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
    decode_temp: float
    decode_min_p: float
    decode_click_trigger_fix: bool
    exec_cadence: int


def _eval_protocol(
    cfg: TrainConfig,
    *,
    settings: DecodeSettings,
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
        decode_temp=settings.temp,
        decode_min_p=settings.min_p,
        decode_click_trigger_fix=settings.click_trigger_fix,
        exec_cadence=settings.exec_cadence,
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
    policy_factory: Callable[[], BatchPolicy],
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
    """In-training closed-loop eval vs lvl-9 CPU over prior-sampled char matchups. A fixed
    ``n_matchups`` controls statistical coverage; host CPU count controls only how many of those boots
    execute concurrently. Each policy wave gets an explicit, deterministic sampling seed."""
    settings = _decode_settings(cfg)
    protocol = _eval_protocol(
        cfg,
        settings=settings,
        default_n_matchups=cfg.eval_n_matchups,
        n_matchups=n_matchups,
        max_frames=max_frames,
        seed=eval_seed,
    )
    policy_index = itertools.count()

    def policy_factory() -> BatchPolicy:
        return make_policy(model, stats, cfg, settings=settings, decode_seed=protocol.seed + next(policy_index))

    with _evaluation_mode(model):
        return _run_eval_sweep(policy_factory, protocol=protocol, replay_dir=replay_dir, rows_path=rows_path)


# %%
# The fitted tokenizer travels INSIDE the checkpoint under this key, so an eval never depends on a
# sidecar JSON that could have been re-fitted, moved, or edited since training.
_TOKENIZER_CFG_KEY = "tokenizer"


def _ckpt_cfg(cfg: TrainConfig, tok: ActionTokenizer) -> dict:
    return {**asdict(cfg), _TOKENIZER_CFG_KEY: tok.to_dict()}


def _cfg_from_state(saved: dict) -> TrainConfig:
    """Rebuild a ``TrainConfig`` from a checkpoint's saved cfg dict, tolerating schema drift in
    *eval/host* knobs across code versions: keys no longer on ``TrainConfig`` are dropped and new
    fields take their defaults. Model-identity fields are unaffected — they're always present."""
    known = {f.name for f in fields(TrainConfig)}
    dropped = sorted(set(saved) - known - {_TOKENIZER_CFG_KEY})
    if dropped:
        print(f"[ckpt] dropping {len(dropped)} stale cfg key(s) not on current TrainConfig: {dropped}", flush=True)
    return TrainConfig(**{k: v for k, v in saved.items() if k in known})


def _split_saved_cfg(saved: dict, *, source: str) -> tuple[TrainConfig, ActionTokenizer]:
    if _TOKENIZER_CFG_KEY not in saved:
        raise ValueError(f"{source}: checkpoint carries no embedded tokenizer; it predates this experiment")
    return _cfg_from_state(saved), tokenizer_from_dict(saved[_TOKENIZER_CFG_KEY], source=f"{source}[tokenizer]")


# %%
def train(
    cfg: TrainConfig,
    stats: dict[str, FeatureStats],
    tok: ActionTokenizer,
    *,
    comment: str = "",
    resume_run: str | None = None,
    resume_state: dict | None = None,
) -> None:
    resumed_counts = resume_state is not None and "button_combo_counts" in resume_state["model"]
    button_combo_counts = None if resumed_counts else _load_button_combo_counts(cfg)
    validate_config(cfg, tok=tok)
    run_name = resume_run or make_run_name(
        Path(__file__).stem, _model_tag(cfg, tok.vocab_size), cfg.data_root, comment
    )
    uploader = BackgroundUploader(run_name)
    wandb.init(
        project="hal",
        name=run_name,
        id=resume_state["wandb_id"] if resume_state else None,
        resume="allow" if resume_state else None,
        tags=["gpt", "rle", f"d{cfg.d_model}", f"L{cfg.n_layers}", f"V{tok.vocab_size}"],
        config={
            **asdict(cfg),
            "tokenizer/vocab_size": tok.vocab_size,
            "tokenizer/n_alphabet": len(tok.alphabet),
            "tokenizer/n_merges": len(tok.merges),
            "tokenizer/max_token_frames": tok.max_token_frames,
            **{f"tokenizer/{k}": v for k, v in tok.stats.items()},
        },
    )
    # W&B's own step is a free-running monotonic timestamp; we plot everything against the training
    # step logged as data (``global_step``), so an async eval that finishes late can be logged at its
    # origin step without violating step monotonicity.
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
    model = GPT(cfg, tok).to(DEVICE)
    if button_combo_counts is not None:
        model.button_combo_counts.copy_(button_combo_counts.to(DEVICE))
    n_params = sum(p.numel() for p in model.parameters())
    if wandb.run is not None:
        wandb.run.summary["model/num_params"] = n_params
    print(
        f"[model] {_model_tag(cfg, tok.vocab_size)}  num_params={n_params / 1e6:.2f}M  "
        f"tokens/frame(fit) {tok.stats.get('tokens_per_frame', float('nan')):.3f}",
        flush=True,
    )
    # The target horizon must cover BOTH the token horizon and the farthest auxiliary frame head.
    L_chunk = max(tok.max_token_frames, max(cfg.aux_head_offsets, default=1))
    loader_kwargs = dict(
        data_root=cfg.data_root,
        remote=streams.remote_for_local(cfg.data_root),
        cache_limit=f"{cfg.cache_limit_gb}gb",
        shuffle_block_size=cfg.shuffle_block_size,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=L_chunk,
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
    # comparable across experiments regardless of the train-time L_chunk.
    val_loader = make_loader(split=cfg.val_split, num_workers=0, **{**loader_kwargs, "L_chunk": VAL_L_CHUNK})

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
        f"({sum(b.target.shape[0] for b in val_cache)} samples) in {time.monotonic() - val_t0:.1f}s",
        flush=True,
    )

    def _wandb_id() -> str | None:
        return wandb.run.id if wandb.run is not None else None

    def _eval_and_upload(step_tag: str, *, n_matchups: int) -> dict[str, float]:
        """Synchronous closed-loop eval on the live model + .slp upload (the final eval)."""
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
        out = {f"val/{k}": v for k, v in val_metrics(model, val_cache, cfg).items()}
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
            cfg=_ckpt_cfg(cfg, tok),
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
            # The worker and trainer otherwise inherit the same CUDA device. Waiting here makes the
            # subprocess asynchronous with respect to process setup/logging, but exclusive on the GPU.
            _drain_eval(wait=True)

    model.train()
    it = iter(train_loader)
    run_t0 = time.monotonic()
    for step in range(start_step, cfg.max_steps):
        with profile("step") as sw:
            opt.zero_grad()
            token_acc: list[Tensor] = []
            aux_acc: dict[tuple[int, str], list[Tensor]] = {}
            obj_acc: Tensor | None = None
            sub_acc: list[float] = []
            for _ in range(cfg.grad_accum_steps):
                try:
                    batch = next(it).to(DEVICE)
                except StopIteration:
                    it = iter(train_loader)
                    batch = next(it).to(DEVICE)
                with autocast:
                    parts = action_loss(model, batch, cfg)
                    obj = objective(parts, cfg.aux_loss_weight, cfg.transition_loss_weight)
                    loss = obj / cfg.grad_accum_steps
                loss.backward()
                obj_acc = obj.detach() if obj_acc is None else obj_acc + obj.detach()
                token_acc.append(parts.token_nll.detach())
                sub_acc.append(parts.substitution_rate)
                for k, v in parts.aux_nll.items():
                    aux_acc.setdefault(k, []).append(v.detach())
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))  # measure only
            opt.step()
            sched.step()
            if DEVICE == "cuda":
                torch.cuda.synchronize()
        assert obj_acc is not None  # grad_accum_steps >= 1
        token_nll_bits = torch.cat(token_acc).mean().item() / _LN2
        aux_bits = [
            sum(torch.cat(aux_acc[(o, name)]).mean().item() for name in _GROUP_NAMES) / _LN2
            for o in cfg.aux_head_offsets
        ]
        sps = cfg.batch_size * cfg.grad_accum_steps / sw.elapsed
        samples = (step + 1) * cfg.batch_size * cfg.grad_accum_steps
        log = {
            "global_step": step,
            "samples": samples,
            "tokens": samples * cfg.L_ctx,
            "train/loss": token_nll_bits,  # the deployed head: bits per RLE token
            "train/aux_loss": float(np.mean(aux_bits)) if aux_bits else 0.0,
            "train/objective": (obj_acc / cfg.grad_accum_steps).item() / _LN2,
            "train/substitution_rate": float(np.mean(sub_acc)),
            "lr/muon": next(g["lr"] for g in opt.param_groups if g["use_muon"]),
            "lr/adam": next(g["lr"] for g in opt.param_groups if not g["use_muon"]),
            "train/gnorm": grad_norm.item(),
            "throughput/step_s": sw.elapsed,
            "throughput/samples_per_s": sps,
        }
        if step < 20 or step % 50 == 0:
            print(
                f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: token_nll {token_nll_bits:.4f} "
                f"step_dt={sw.elapsed * 1000:.0f}ms ({sps:.1f} samples/s)",
                flush=True,
            )
        if cfg.ckpt_every > 0 and step > 0 and step % cfg.ckpt_every == 0:
            _save("latest.pt", step)
        if cfg.val_every > 0 and step > 0 and step % cfg.val_every == 0:
            vm = _val_log_dict()
            log.update(vm)
            print(
                f"[t+{time.monotonic() - run_t0:.0f}s] step {step}: bits/frame {vm['val/loss']:.3f} "
                f"tokens/frame {vm['val/tokens_per_frame']:.3f} btn_logloss {vm['val/btn_logloss']:.3f}",
                flush=True,
            )
        wandb.log(log)
        _drain_eval(wait=False)
        if cfg.eval_every > 0 and step > 0 and step % cfg.eval_every == 0:
            _launch_eval(step)

    _drain_eval(wait=True)  # finish the last async eval before the final pass
    vm_final = _val_log_dict()
    wandb.log({**vm_final, "global_step": cfg.max_steps})
    print(f"[final] bits/frame {vm_final['val/loss']:.3f}", flush=True)
    if cfg.eval_every > 0:
        _log_eval(cfg.max_steps, _eval_and_upload("final", n_matchups=cfg.final_eval_n_matchups))
    _save("final.pt", cfg.max_steps)
    uploader.close()


# %%
def _load_ckpt(ckpt_path: str) -> tuple[GPT, TrainConfig, dict[str, FeatureStats], dict]:
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg, tok = _split_saved_cfg(state["cfg"], source=ckpt_path)
    embedded_counts = "button_combo_counts" in state["model"]
    button_combo_counts = None if embedded_counts else _load_button_combo_counts(cfg)
    validate_config(cfg, tok=tok)
    model = GPT(cfg, tok).to(DEVICE)
    if button_combo_counts is not None:
        model.button_combo_counts.copy_(button_combo_counts.to(DEVICE))
    _load_model_state(model, state["model"])
    model.eval()
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    return model, cfg, stats, state


def eval_ckpt(
    ckpt_path: str,
    *,
    decode_temp: float | None = None,
    decode_min_p: float | None = None,
    decode_click_trigger_fix: bool | None = None,
    exec_cadence: int | None = None,
    eval_n_matchups: int | None = None,
    eval_max_frames: int | None = None,
    eval_seed: int | None = None,
    wandb_run_id: str | None = None,
    wandb_project: str = "hal",
    wandb_entity: str | None = None,
    wandb_label: str | None = None,
) -> None:
    """Load a checkpoint and run the prior-distribution vs-CPU sweep, printing the pooled metrics. Each
    override (temperature, min-p, click=>trigger fix, execution cadence) replaces the trained cfg for
    this eval only (test-time sweep); ``None`` keeps the trained value."""
    model, cfg, stats, state = _load_ckpt(ckpt_path)
    settings = _decode_settings(
        cfg,
        temp=decode_temp,
        min_p=decode_min_p,
        click_trigger_fix=decode_click_trigger_fix,
        exec_cadence=exec_cadence,
    )
    protocol = _eval_protocol(
        cfg,
        settings=settings,
        default_n_matchups=cfg.final_eval_n_matchups,
        n_matchups=eval_n_matchups,
        max_frames=eval_max_frames,
        seed=eval_seed,
    )
    print(
        f"[eval] loaded {ckpt_path}  step={state['step']}  device={DEVICE}  temp={settings.temp}  "
        f"min_p={settings.min_p}  click_trigger_fix={settings.click_trigger_fix}  "
        f"exec_cadence={settings.exec_cadence or 'token-native'}",
        flush=True,
    )
    replay_dir = Path(ckpt_path).resolve().parent / "eval_replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    policy_index = itertools.count()

    def policy_factory() -> BatchPolicy:
        return make_policy(model, stats, cfg, settings=settings, decode_seed=protocol.seed + next(policy_index))

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
    """Time one replan against the real-time budget on a real val Context, so the trunk sees the
    production feature set. Weights are irrelevant to the timing, so a fresh model on the configured
    tokenizer is fine when no checkpoint is given — but the SAMPLED SPANS are not: an untrained model
    samples near-uniformly and reports far shorter spans (hence a pessimistic ms/frame) than a trained
    one, whose spans concentrate on the long holds the tokenizer was fitted for."""
    if ckpt_path is None:
        tok = load_tokenizer(cfg)
        validate_config(cfg, tok=tok)
        stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
        model = GPT(cfg, tok).to(DEVICE)
    else:
        model, cfg, stats, _ = _load_ckpt(ckpt_path)
        tok = model.tokenizer
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
    print(f"[bench] device={DEVICE} {_model_tag(cfg, tok.vocab_size)} n_iters={n_iters}", flush=True)
    bench_decode(model, ctx, cfg, n_iters=n_iters)


# %%
@dataclass
class Args:
    """Top-level CLI surface. Pass TrainConfig fields as kebab-case flags, e.g. ``--cfg.d-model 512``."""

    cfg: TrainConfig = field(default_factory=TrainConfig)
    # tokenizer fitting (writes the versioned JSON artifact, then exits)
    fit_tokenizer: bool = False
    fit: FitConfig = field(default_factory=FitConfig)
    eval: str | None = None  # ckpt path; closed-loop eval instead of train
    eval_temp: float | None = None  # override the token-softmax temperature for --eval
    eval_min_p: float | None = None  # min-p nucleus over the token distribution
    eval_click_trigger_fix: bool | None = None  # force trigger_l/r to 1.0 on a digital L/R click
    eval_exec_cadence: int | None = None  # 0 = token-native; k>0 executes min(k, span) then replans
    eval_n_matchups: int | None = None  # manual --eval override; default is cfg.final_eval_n_matchups (96)
    eval_max_frames: int | None = None  # manual --eval override; default is checkpoint cfg.eval_max_frames
    eval_seed: int | None = None  # manual --eval sampling/bootstrap seed override
    wandb_run_id: str | None = None  # resume an existing run and log this manual eval to it
    wandb_project: str = "hal"
    wandb_entity: str | None = None
    wandb_label: str | None = None
    resume: str | None = None  # run_name to resume; pulls latest.pt (local, else R2)
    comment: str = ""
    # decode wall-clock benchmark (no Dolphin): ms per replan and amortized ms/frame.
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
    if args.fit_tokenizer:
        write_tokenizer(fit_tokenizer(args.cfg, args.fit), Path(args.fit.out))
        return
    if args.bench_decode:
        run_bench_decode(args.cfg, args.eval, args.bench_slots, args.bench_iters)
        return
    if args.eval is not None:
        eval_ckpt(
            args.eval,
            decode_temp=args.eval_temp,
            decode_min_p=args.eval_min_p,
            decode_click_trigger_fix=args.eval_click_trigger_fix,
            exec_cadence=args.eval_exec_cadence,
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
        # model-identity knobs MUST come from the checkpoint so a resume can't silently change them,
        # and the tokenizer always comes from the checkpoint, never from a sidecar path.
        saved_cfg, tok = _split_saved_cfg(state["cfg"], source=args.resume)
        d = TrainConfig()
        cfg = replace(saved_cfg, num_workers=d.num_workers, prefetch_factor=d.prefetch_factor)
        stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
        train(cfg, stats, tok, resume_run=args.resume, resume_state=state)
        return
    cfg = args.cfg
    tok = load_tokenizer(cfg)
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    auto_comment = f"rle{tok.vocab_size}-{cfg.max_steps // 1000}k-b{cfg.batch_size}"
    train(cfg, stats, tok, comment=args.comment or auto_comment)


if __name__ == "__main__":
    main(tyro.cli(Args))
