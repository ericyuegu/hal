"""Advantage-weighted regression on a sliding-window trunk, forked from experiment 020.

020 showed the AWR machinery runs but the policy played too passively, and its reward knobs put
87% of the weight mass on which segment of a match a window came from. This fork keeps 020's
objective and changes five things: the trunk is the SHARED one (``hal.training.trunk``) with
sliding-window attention, the sequence geometry is four times longer, the action head is 019's
within-frame chain, the reward knobs come from the reward explorer's tuned table
(``notebooks/awr_explorer.py``), and two new axes ride the same weighting — the demonstrator's
ranked tier and an IQL expectile critic.

The objective (unchanged from 020):

    r_t = -1 on an ego stock loss, +1 on an opponent stock loss   (+ damage / match-win shaping)
    G_t = sum_k gamma^k r_{t+k}                                    (reverse scan, FULL replay)
    A_t = G_{t+1} - V(s_t)          (V = a value head on the trunk; the return starts at the PREDICTED
                                     frame, so a reward that landed before the action could act is
                                     never part of the action's credit)
    w_t = clip(exp(A_t / beta), w_max),  rescaled to mean 1 over the batch
    loss = sum over heads/groups of  weighted_mean(w_t, per-frame NLL)  +  value MSE

``awr_enabled=False`` gives the plain mean NLL, so the AWR arm is one flag. That BC arm on this
trunk and this geometry is the control every other arm is measured against; do NOT compare 022
numbers with 020 or 016 runs, because the geometry moved.

What changed against 020

* Sliding-window attention. ``attn_window`` (128 frames) replaces the full-context mask. The trunk
  runs FlexAttention where it compiles and the dense mask where it does not; ``require_flex`` makes
  a cloud run fail instead of training about 4x slower on the fallback. A trained window also makes
  incremental decode exact, so ``eval_incremental_kv`` is on by default (and is rejected at full
  attention).
* A faster closed loop. Decode runs on fp16 weights, at ``high`` matmul precision, over a rolling KV
  cache fed one frame at a time — the profiler put 79% of a rollout frame in the policy and 59% in
  the trunk forward alone. See ``_load_ckpt`` and ``make_policy``.
* Geometry. ``L_ctx`` 256 -> 1024 with the token count per step held near 020's: ``batch_size`` is
  now the EFFECTIVE batch (128) and ``grad_accum_steps`` splits it into micro-batches of 64. Under
  a window the step cost is nearly flat in ``L_ctx``, and a longer window turns more of each
  whole-replay disk read into training tokens.
* Factored action heads (019's, which tested lean-positive against the independent joint head).
  Each offset head keeps one projection per group and predicts the four groups as a CHAIN in
  ``chain_order``; every non-terminal group feeds its class back into the running hidden state
  through a zero-initialized table:

      h_0 = h;   logits[g_i] = proj[g_i](h_i);   h_{i+1} = h_i + emb[g_i](id[g_i])

  Training teacher-forces the ancestors with the ground-truth ids of the SAME target frame, so the
  summed per-group NLL is the chain rule — the joint NLL of that frame, in 020's units. The AWR
  weight multiplies that per-position sum exactly as before. Decode runs the same chain on its own
  draws. The per-group NLLs alone are now conditionals and do NOT compare with a joint-head run.
* Reward knobs. Dense damage shaping, a moderate beta and a low weight cap (see ``TrainConfig``).
* Rank weighting. ``awr_rank_weights`` multiplies a window's weight by the EGO player's ranked tier
  (schema v7's ``p{1,2}_rank``) before the mean-1 rescale, so ``(1, 2, 4)`` moves master frames from
  31% to 57% of the gradient mass. It is a property of the weighting, not of the critic: with
  ``awr_enabled=False`` it gives rank-weighted BC, with no value head in the loop at all.
* An optional IQL critic. ``awr_critic="expectile"`` fits V by a TD expectile loss and takes the
  residual as the advantage, instead of regressing on the Monte-Carlo return (which carries the
  opponent's luck). Both critics log both sets of diagnostics, so the arms compare. A one-step
  residual is ``sqrt(1 - gamma^2)`` times the size of the Monte-Carlo advantage, so this arm needs
  its own ``awr_beta`` (see ``awr_expectile_tau``) or its weights come out uniform.
* Two fixes: ``final.pt`` is saved BEFORE the closed-loop eval, and validation runs two trunk
  forwards per batch instead of five.
* A leaner ``val/*`` block. A correlation study over ten comparable runs found the dropped metrics
  either uninformative or inverted against closed-loop play. What is left is a kill switch or a
  tracked hypothesis.

Checkpoints. The trunk moved into a submodule, so the state-dict keys gained a ``trunk.`` prefix,
and the heads moved to ``heads.{i}.proj.{group}`` / ``heads.{i}.emb.{group}``: 020 checkpoints do
not load here and are rejected with that message.

Run:
    uv run experiments/022_awr_rank.py
    uv run experiments/022_awr_rank.py --cfg.no-awr-enabled       # the BC control arm
    uv run experiments/022_awr_rank.py --cfg.chain-order buttons main_stick c_stick triggers
    # The rank arm. The tier column arrived with v7, so it needs BOTH v7 flags; validate_config says so.
    uv run experiments/022_awr_rank.py --cfg.awr-rank-weights 1.0 2.0 4.0 \
        --cfg.data-root data/processed/ranked-anonymized-1/mds-v7 --cfg.mds-schema-version 7
    # The IQL arm. Its advantage is ~17x smaller than the MC one, so beta scales with it.
    uv run experiments/022_awr_rank.py --cfg.awr-critic expectile --cfg.awr-expectile-tau 0.8 --cfg.awr-beta 0.05
    uv run experiments/022_awr_rank.py --audit-returns --cfg.data-root data/processed/dev/mds
    uv run experiments/022_awr_rank.py --eval <ckpt> --eval-temp 0.7
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
from typing import Literal

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
from scipy.signal import lfilter
from streaming import StreamingDataset
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import wandb
from hal import streams
from hal.data.feature_stats import FeatureStats
from hal.data.schema import Rank
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
from hal.training.trunk import Trunk
from hal.training.trunk import TrunkConfig
from hal.wire import mask_value

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_LN2 = math.log(2.0)

# Action-vector channel split (A_DIM=14): [0:6] sticks+triggers (continuous), [6:14] buttons {0,1}.
_N_CONT = 6
_N_BUTTONS = A_DIM - _N_CONT

# Per-frame input: all four players' gamestate concatenated in the feature dim.
_PLAYER_PREFIXES: tuple[str, ...] = ("ego", "ego_nana", "opp_nana", "opp")

# Output groups (fixed order; the canonical order of every per-group tensor and of the class-index
# columns quantize_groups emits) + their discrete vocab sizes from the scoring discretizers.
# cfg.chain_order permutes only the ORDER OF PREDICTION inside a frame.
_GROUP_NAMES: tuple[str, ...] = ("buttons", "main_stick", "c_stick", "triggers")
_GROUP_VOCABS: tuple[int, ...] = (
    scoring.N_BUTTON_COMBOS,  # 256
    scoring.STICK_CLUSTER_CENTERS_MAIN.shape[0],  # 65
    scoring.STICK_CLUSTER_CENTERS_C.shape[0],  # 9
    scoring.TRIGGER_CENTERS.shape[0] ** 2,  # 25 (joint L*5 + R)
)
N_GROUPS = len(_GROUP_NAMES)
_BUTTONS_G, _MAIN_G, _C_G, _TRIG_G = range(N_GROUPS)
_GROUP_INDEX: dict[str, int] = {name: g for g, name in enumerate(_GROUP_NAMES)}

_BUTTON_COUNTS_VERSION = 1

# Per-frame return and reward columns. They are named like every other per-player MDS column so that
# ``dataloader.relabel_ego`` renames them to ego/opp for free; ``EGO_RETURN_COLUMN`` and
# ``EGO_REWARD_COLUMN`` are what the collate reads. ``features._classify`` does not recognize either
# suffix, so they never reach the model as an input feature — the return is a TARGET for the value
# head and the reward is the TD critic's ``r``.
_RETURN_SUFFIX = "awr_return"
_REWARD_SUFFIX = "awr_reward"
_PORT_RETURN_COLUMNS: tuple[str, ...] = tuple(f"{port}_{_RETURN_SUFFIX}" for port in ("p1", "p2"))
EGO_RETURN_COLUMN = f"ego_{_RETURN_SUFFIX}"
EGO_REWARD_COLUMN = f"ego_{_REWARD_SUFFIX}"
# The ego's ranked tier, one uint8 per frame. v6 shards do not carry it; rank weighting is what
# makes it mandatory (see ``validate_config`` and ``_ego_rank_weights``).
EGO_RANK_COLUMN = "ego_rank"
_RANK_TIERS: tuple[Rank, ...] = (Rank.PLATINUM, Rank.DIAMOND, Rank.MASTER)
_RANK_WEIGHTS_OFF: tuple[float, float, float] = (1.0, 1.0, 1.0)
# The materialization that first carries p{1,2}_rank; rank weighting refuses anything older.
_RANK_MDS_VERSION = 7

# Action-vector channels for the click=>trigger hygiene fix (digital L/R click => analog trigger = 1.0).
_TRIGGER_L_CH = ACTION_CHANNELS.index("trigger_l")
_TRIGGER_R_CH = ACTION_CHANNELS.index("trigger_r")
_BUTTON_L_CH = ACTION_CHANNELS.index("button_l")
_BUTTON_R_CH = ACTION_CHANNELS.index("button_r")


# %%
@dataclass
class TrainConfig:
    # THE studied axis: weight each frame's imitation loss by how well that frame turned out.
    # False gives the plain mean NLL and no value loss — the BC control arm is this one flag.
    # Everything below tunes the weighting, never the architecture.
    awr_enabled: bool = True
    # Return discount. 0.99827 gives a 6.7 s half-life (~9.6 s horizon): one exchange, not one
    # match. 020 used 0.999 (11.5 s), which made a window's return mostly a statement about WHICH
    # SEGMENT of the match it came from — 87% of the weight variance sat between windows.
    awr_gamma: float = 0.99827
    # Weight temperature: w = exp(A / beta), in stock units. Small beta separates good frames
    # sharply and collapses the effective sample size; large beta flattens toward plain behavior
    # cloning. 0.8 (= 80% of a stock) is the audited dose under the knobs below: ESS ~0.70 and
    # almost no frame on the clip, so the weighting is a mild, deliberate tilt.
    awr_beta: float = 0.8
    # Ceiling on a single frame's weight, before the batch is rescaled to mean 1. Bounds the
    # variance one lucky trajectory can inject. Low here because the tuned reward is dense: the
    # clip is a guard, and the audit puts ~0% of frames on it.
    awr_weight_max: float = 5.0
    # Scalar on the value head's MSE inside the backprop objective. The policy loss never
    # backpropagates into the value head (the advantage is detached). The reverse path is NOT
    # closed by default: the MSE reaches the shared trunk through the value head's input unless
    # awr_value_detach_trunk is set.
    awr_value_loss_weight: float = 1.0
    # Feed the value head a detached trunk state. False lets the value MSE also train the trunk — a
    # second axis against the BC arm. True makes the AWR arm pure reweighting: V trains through
    # value_head alone and the trunk sees only the (weighted) imitation gradient.
    awr_value_detach_trunk: bool = False
    # Dense shaping term added to the sparse stock reward, per percent: the opponent's damage taken
    # minus the ego's, each clipped at >= 0 so a respawn's percent reset is not read as healing.
    # 0.01 makes a full 100% of damage worth one stock. This is the term that makes the reward
    # dense: with stock events alone the signal fires a few times a match.
    awr_damage_shaping: float = 0.01
    # Extra reward on the MATCH-DECIDING stock event (the drop that empties a player's stock count),
    # in stock units, on top of the ordinary +-1. 0.5 makes closing a game out worth 1.5 stocks.
    awr_win_reward: float = 0.5
    # Per-tier multiplier on the EGO player's frames, for (platinum, diamond, master). It scales the
    # raw weight BEFORE the mean-1 rescale, and every batch mixes all three tiers, so the ratio
    # survives the rescale exactly. (1, 1, 1) is off and needs no rank column at all; the rank arm
    # passes (1, 2, 4), which moves master frames from 31% to 57% of the gradient mass — a deliberate
    # data-mixture shift, not a tie-break. With any other setting an UNKNOWN tier raises, and the
    # dataset must be v7 (the version that carries p{1,2}_rank).
    awr_rank_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # Which critic produces the advantage. "mc" regresses V on the Monte-Carlo return (020's).
    # "expectile" is IQL's: V is fit by an expectile TD loss, so the target is a one-step bootstrap
    # rather than a whole sampled future, and tau > 0.5 leans toward the good outcomes instead of
    # averaging in the opponent's luck. The advantage is then the TD residual.
    awr_critic: Literal["mc", "expectile"] = "mc"
    # Expectile of the TD loss: |tau - 1[u < 0]| * u^2. 0.5 is plain MSE (a mean); 0.8 puts 4x the
    # gradient on residuals above the current V, which approximates a max over the actions taken from
    # that state. The mean shift tau puts into the weights is removed exactly by the mean-1 rescale.
    # The SCALE is a different matter and awr_beta DOES need a retune for this critic: the MC
    # advantage is the sum of the future one-step residuals, which are martingale differences, so
    # sigma(u) = sigma(A) * sqrt(1 - gamma^2) — a factor 17 at gamma=0.99827, set by gamma alone.
    # At awr_beta=0.8 the expectile weights come out near-uniform (ESS ~0.99) and the arm is BC with
    # a value head. Match 020's audited dose with awr_beta ~ 0.8 * sqrt(1 - gamma^2) ~ 0.05.
    awr_expectile_tau: float = 0.8
    # GPT backbone
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 4
    # Sliding-window attention: how many frames back a query may attend to, its own frame included.
    # 0 = full context. 8 layers x 128 frames reach ~1024 frames of receptive field through
    # multi-hop attention, so the long context is still used, and the step cost stays flat as L_ctx
    # grows. A trained window also makes incremental decode exact (see eval_incremental_kv).
    attn_window: int = 128
    # Fail if FlexAttention does not compile on the box instead of falling back to the dense mask.
    # Set it on a cloud run: the fallback trains about 4x slower and says so only in a log line.
    require_flex: bool = False
    # Multi-token (multi-frame) auxiliary output heads: one independent head per future-frame offset;
    # head o predicts the action o frames ahead. MUST contain 1 — per-frame closed-loop decode reads
    # only the offset-1 head. Spread offsets (1,5,9,13) keep the long-horizon auxiliary supervision
    # the 016-021 line trained with. Chunked execution (exec_horizon s > 1) needs the contiguous
    # prefix 1..s instead — but s=2 measured no throughput gain on top of the incremental decoder,
    # so the long-horizon aux signal wins.
    head_offsets: tuple[int, ...] = (1, 5, 9, 13)
    # Within-frame chain order (019's settled default). Every offset head predicts the four action
    # groups in this order, each non-terminal group conditioning the rest through a zero-initialized
    # table, so a fresh model is exactly the independent joint head 020 deployed. Must be a
    # permutation of _GROUP_NAMES. The order is a play knob, not the studied axis: the default puts
    # the cheap near-deterministic groups first and buttons last, so the button combo sees every
    # stick/trigger ancestor.
    chain_order: tuple[str, str, str, str] = ("c_stick", "triggers", "main_stick", "buttons")
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
    # point in closed loop, so deployed play always samples.
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
    # 2x 020's context under a 128-frame window. Longer L is near-free in step time but costs
    # batch diversity at a fixed token budget; L_ctx 512 keeps 128 distinct replays per step
    # (K=2), the same pool 020's weight normalization saw.
    L_ctx: int = 512
    # THE EFFECTIVE batch (020's field was the micro-batch). One optimizer step sees batch_size
    # samples, fed as grad_accum_steps micro-batches of batch_size / grad_accum_steps; the two must
    # divide. The default is ONE forward per step: no accumulation, so the AWR mean-1 weight
    # rescale sees the whole step's samples (accumulation biases the rank-weighted tier mixture,
    # measured -0.4% at accum 2). 256 x 512 = 131,072 tokens per step, matching 020's 512 x 256;
    # ~11.8 GiB peak in one forward, so it needs a 16 GiB card (the 3060 halves batch_size).
    batch_size: int = 256
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
    # torch.compile the model's forward for TRAINING. About 40% of the step's GPU time is unfused
    # elementwise work and autocast casts, against 26% in the matmuls, which is what a fused graph
    # takes back: measured 406 -> 231 ms per step (1.76x) at this geometry on a 3060, over a 30-step
    # bench whose loss trajectory tracks eager to 1e-6 relative. A compiled graph that then meets a
    # validation or gradient-diagnostic shape dies inside FlexAttention with a CUDA illegal memory
    # access, so every non-training forward runs eager — see ``_evaluation_mode``, which is the one
    # door they all go through.
    compile_trunk: bool = True
    # eval cadence
    val_every: int = 1024
    # 128 batches x 64 windows = 8,192 val replays (val draws one window per replay), the sample
    # size 020 validated on. A narrower val set widens the confidence interval on val NLL, and these
    # metrics are kill switches: a noisy one either fires late or fires for nothing.
    val_n_batches: int = 32
    # Exact per-head shared-trunk gradient Gram matrix on this many examples from the first frozen val
    # batch, computed only at validation cadence. Keeps the diagnostic observational and bounded-cost.
    # It retains one backward graph per head, so its cost follows examples x L_ctx: 16 windows of
    # 1024 frames is the 16,384 scored positions 020 measured at 64 x 256, and 64 of them OOMs a
    # 12 GiB card.
    gradient_diagnostic_batch_size: int = 16
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
    # Incremental (rolling-KV) closed-loop decode. With a trained attn_window the cache holds
    # exactly that window and RoPE is relative, so the decode is exact; at full attention the cache
    # drops history the full forward keeps, and validate_config rejects the combination (a
    # full-context arm must pass --cfg.no-eval-incremental-kv, so the two can never disagree in
    # silence). On because the one-token forward measured 2.0x the full recompute at L_ctx 256, and
    # the saving grows with the context.
    eval_incremental_kv: bool = True
    # Cast the model's float parameters to fp16 for closed-loop decode. The decode is launch-bound,
    # not precision-bound: fp16 weights + fp16 context measured 1.7x the fp32 forward, and the
    # sampled action is unchanged (the stick/trigger centers and every logit stay fp32). Autocast is
    # NOT the same thing and measured SLOWER — this casts the weights once, at load.
    eval_fp16: bool = True
    # If an eval is still running at the next boundary, the trainer waits up to this bound and
    # then kills the worker.
    eval_timeout_seconds: float = 2700.0
    # Final in-process mirrored h2h vs a reference run. The cloud box can self-destruct after
    # training, so the sweep runs inside train() and its records/replays upload before exit.
    # None disables the sweep.
    # The reference is the BC arm of THIS experiment: the geometry moved, so no older run is a
    # valid control.
    final_h2h_reference_run: str | None = None
    final_h2h_reference_experiment: str = "experiments/022_awr_rank.py"
    final_h2h_reference_label: str = "022-bc"
    final_h2h_self_label: str = "022-awr"
    final_h2h_n_configs: int = 64
    # checkpointing
    ckpt_every: int = 2048
    # data
    data_root: str = "data/processed/ranked-anonymized-1/mds-v7"
    # MDS materialization this run reads. The dataloader's per-row guard rejects any other version
    # — never silent.
    mds_schema_version: int = 7
    # Optional versioned JSON artifact containing full-dataset button-combo counts. Required when
    # decode_btn_support_min > 0; the 012 614-replay reference sample is not authoritative support.
    button_combo_counts_path: str | None = None
    # Streaming dataset cache and shuffle geometry.
    cache_limit_gb: int = 440
    shuffle_block_size: int = 2000
    # Each replay deserialized off disk yields this many non-overlapping windows, amortizing the
    # whole-replay read (the disk bottleneck) over K samples. Train only; val stays 1/replay so its
    # loss stays comparable across runs. Two windows of L_ctx + L_chunk = 1028 frames use 19.5% of
    # every replay read (simulated over the index frame counts, of which 7.23% of the context
    # positions are left padding that no head scores) — still about twice 020's 10% at L_ctx 256.
    # K stops at 2 because of the OTHER side of the trade: the AWR weights renormalize to mean 1
    # inside each micro-batch, so K windows per replay leave 64/K distinct replays to normalize
    # against. K=2 keeps 32; K=4 would leave 16, against 020's 128 — a between-window confound that
    # the AWR arms would carry and the BC arm would not.
    windows_per_replay: int = 2
    val_split: str = "val"
    num_workers: int = 16
    prefetch_factor: int = 4


def _model_tag(cfg: TrainConfig) -> str:
    """The architecture, for the run name. The ``chain`` token (019's) is what tells a factored-head
    run from a joint-head one, and spells the order it used, so ``_reward_tag`` — the WEIGHTING spec,
    which lands in the same run name — never repeats it."""
    offs = ".".join(str(o) for o in cfg.head_offsets)
    chain = "".join(name[0] for name in cfg.chain_order)  # c_stick,triggers,main_stick,buttons -> ctmb
    awr = f"-awr{cfg.awr_beta:g}" if cfg.awr_enabled else "-bc"
    return f"gpt-d{cfg.d_model}-L{cfg.n_layers}-h{cfg.n_heads}-Lc{cfg.L_ctx}-o{offs}-chain{chain}{awr}"


def _reward_tag(cfg: TrainConfig) -> str:
    """The full weighting spec, for the run name: ``g99827-b0.8-w5-d0.01-win0.5-rank1-2-4-iql0.8-swa128``.

    ``main`` appends it to the comment, so a hand-typed comment can never disagree with the flags
    the run actually used. Gamma drops its ``0.`` (the digits are the identity); every other number
    is printed as written. A knob appears only when it is on: a default run name carries no ``rank``
    and no ``iql`` token, so two arms can never share a name."""
    window = f"swa{cfg.attn_window}" if cfg.attn_window else "swafull"
    rank = (
        "-rank" + "-".join(f"{w:g}" for w in cfg.awr_rank_weights) if rank_weighting_on(cfg.awr_rank_weights) else ""
    )
    if not cfg.awr_enabled:
        # Rank weighting is a property of the objective's weighting, not of the critic, so a BC arm
        # can carry it — and then has to say so.
        return f"bc{rank}-{window}"
    gamma = f"{cfg.awr_gamma:g}".removeprefix("0.")
    critic = f"-iql{cfg.awr_expectile_tau:g}" if cfg.awr_critic == "expectile" else ""
    # The value MSE also trains the trunk unless this is set, which is a second axis against the BC
    # arm, so the run name has to say which of the two arms it is.
    detach = "-vdetach" if cfg.awr_value_detach_trunk else ""
    return (
        f"g{gamma}-b{cfg.awr_beta:g}-w{cfg.awr_weight_max:g}"
        f"-d{cfg.awr_damage_shaping:g}-win{cfg.awr_win_reward:g}{rank}{critic}{detach}-{window}"
    )


def _micro_batch(cfg: TrainConfig) -> int:
    """Samples per forward/backward. ``batch_size`` is the effective batch, split into
    ``grad_accum_steps`` equal micro-batches; ``validate_config`` pins that they divide."""
    return cfg.batch_size // cfg.grad_accum_steps


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


def match_point_events(stock: np.ndarray) -> np.ndarray:
    """1.0 on the frame a player's LAST stock is lost (the count drops to 0), else 0.0.

    A subset of ``stock_loss_events``: the same drop detection, kept only where the new count is
    zero. A ranked game that ends by quit-out never empties a stock count, so it has no event."""
    ids = np.asarray(stock).astype(np.int64)
    return stock_loss_events(stock) * (ids == 0).astype(np.float32)


def damage_taken(percent: np.ndarray) -> np.ndarray:
    """Per-frame INCREASE in a player's percent, clipped at >= 0.

    The drop back to 0 on a respawn (and the masked NaN of an unavailable frame) is a reset, not
    healing, so only rises count."""
    values = np.asarray(percent, dtype=np.float32)
    out = np.zeros(values.shape, dtype=np.float32)
    out[1:] = np.maximum(values[1:] - values[:-1], 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def frame_reward(
    sample: dict[str, np.ndarray], *, ego: str, opp: str, damage_shaping: float, win_reward: float
) -> np.ndarray:
    """Per-frame reward for the player at port ``ego`` over one whole replay.

    ``+1`` when the opponent loses a stock, ``-1`` when the ego does (both on the frame the drop
    becomes visible), plus ``win_reward`` extra on the match-deciding stock, plus ``damage_shaping``
    times the percent the opponent took minus the percent the ego took on that frame. The sparse
    stock term is the outcome the match is scored on; the shaping terms are off by default and
    exist to densify / re-rank the signal, not to redefine it."""
    reward = stock_loss_events(sample[f"{opp}_stock"]) - stock_loss_events(sample[f"{ego}_stock"])
    if win_reward:
        wins = match_point_events(sample[f"{opp}_stock"]) - match_point_events(sample[f"{ego}_stock"])
        reward = reward + win_reward * wins
    if damage_shaping:
        dealt = damage_taken(sample[f"{opp}_percent"]) - damage_taken(sample[f"{ego}_percent"])
        reward = reward + damage_shaping * dealt
    return reward


def discounted_returns(reward: np.ndarray, gamma: float) -> np.ndarray:
    """``G_t = sum_k gamma^k r_{t+k}`` to the END OF THE EPISODE, by one reverse scan.

    Run on the full replay before windowing: at gamma=0.999 the discount half-life (~693 frames)
    is longer than a whole train window, so a return summed inside the window would be truncated
    toward zero exactly where the credit lives.

    The scan is a one-pole IIR filter on the reversed reward — ``y[n] = x[n] + gamma*y[n-1]`` — so
    ``lfilter`` runs it in C. 020 ran the same recurrence as a Python ``accumulate`` over every
    frame of the replay, which profiling put at 39% of a dataloader worker's time (a replay is
    ~10.7k frames and each yields at most two windows). The filter carries the same double-precision
    accumulation in the same order, so the values are the ones the loop produced."""
    tail = lfilter([1.0], [1.0, -gamma], np.asarray(reward, dtype=np.float64)[::-1])
    return tail[::-1].astype(np.float32)


def replay_returns(
    sample: dict[str, np.ndarray], *, gamma: float, damage_shaping: float, win_reward: float
) -> dict[str, np.ndarray]:
    """Both ports' return AND reward columns for one replay row, keyed ``p{1,2}_awr_{return,reward}``.

    Named per port rather than per role because the sampler picks the ego port AFTER windowing:
    ``dataloader.relabel_ego`` then renames the right ones to ``ego_awr_*`` with no AWR-specific
    code, exactly as it does for every gamestate column.

    020 emitted the return alone. The TD critic needs the ``r`` the return was built from, and the
    scan already has it, so both leave here as ordinary per-frame columns. They travel through the
    same windowing and padding, which is what makes ``G_t = r_t + gamma * G_{t+1}`` hold position by
    position on the collated arrays — the identity the expectile residual is defined against."""
    out: dict[str, np.ndarray] = {}
    for port, other in (("p1", "p2"), ("p2", "p1")):
        reward = frame_reward(sample, ego=port, opp=other, damage_shaping=damage_shaping, win_reward=win_reward)
        out[f"{port}_{_REWARD_SUFFIX}"] = reward.astype(np.float32)
        out[f"{port}_{_RETURN_SUFFIX}"] = discounted_returns(reward, gamma)
    return out


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

    def __init__(self, replays: Iterable[dict], *, gamma: float, damage_shaping: float, win_reward: float) -> None:
        self._replays = replays
        self._gamma = gamma
        self._damage_shaping = damage_shaping
        self._win_reward = win_reward

    def __iter__(self) -> Iterator[dict]:
        for sample in self._replays:
            yield {
                **sample,
                **replay_returns(
                    sample, gamma=self._gamma, damage_shaping=self._damage_shaping, win_reward=self._win_reward
                ),
            }


@dataclass(frozen=True, slots=True)
class AWRBatch:
    """One ``TrainBatch`` plus the ego's per-frame reward, discounted return and ranked tier.

    ``TrainBatch`` is the shared, frozen train/eval contract, so the extra targets compose with it
    instead of extending it. ``returns[b, t]`` is ``G_{t+1}`` — the return from the frame the
    offset-1 head at context position ``t`` predicts. The value head trains to the same target, so
    ``V(s_t)`` estimates the expected outcome of the action about to be chosen and the advantage
    ``G_{t+1} - V(s_t)`` never includes a reward that landed before that action could act. (With
    sparse stock rewards the one-frame shift is cosmetic; with dense damage shaping it is real.)

    ``rewards`` is shifted by the same one frame, so ``returns[t] = rewards[t] + gamma *
    returns[t+1]`` holds exactly on these arrays — the identity the expectile TD residual rests on.
    ``rank``/``rank_weight`` are per SAMPLE, not per frame: a window comes from one replay, so its
    ego tier is one value (read from the window's LAST frame, which is never padding)."""

    batch: TrainBatch
    returns: Tensor  # [B, L_ctx] float32, G_{t+1}
    rewards: Tensor  # [B, L_ctx] float32, r_{t+1}
    rank: Tensor  # [B] uint8, the Rank of the ego player
    rank_weight: Tensor  # [B] float32, that tier's multiplier (all ones when rank weighting is off)

    def to(self, device: str | torch.device) -> AWRBatch:
        return AWRBatch(
            batch=self.batch.to(device),
            returns=self.returns.to(device, non_blocking=True),
            rewards=self.rewards.to(device, non_blocking=True),
            rank=self.rank.to(device, non_blocking=True),
            rank_weight=self.rank_weight.to(device, non_blocking=True),
        )

    def pin_memory(self) -> AWRBatch:
        return AWRBatch(
            batch=self.batch.pin_memory(),
            returns=self.returns.pin_memory(),
            rewards=self.rewards.pin_memory(),
            rank=self.rank.pin_memory(),
            rank_weight=self.rank_weight.pin_memory(),
        )

    def valid_returns(self, valid: Bool[Tensor, "B L_ctx"]) -> Float[Tensor, " n_valid"]:
        """``G_t`` flattened by the SAME validity mask the per-position NLLs use, so returns,
        value predictions, weights and NLLs are elementwise aligned."""
        return self.returns.reshape(-1)[valid.reshape(-1)]

    def valid_rank_weights(self, valid: Bool[Tensor, "B L_ctx"]) -> Float[Tensor, " n_valid"]:
        """The per-sample multiplier broadcast to that sample's scored positions."""
        return self.rank_weight[:, None].expand_as(valid).reshape(-1)[valid.reshape(-1)]

    def valid_ranks(self, valid: Bool[Tensor, "B L_ctx"]) -> Int[Tensor, " n_valid"]:
        """The per-sample tier broadcast the same way, so a per-tier statistic over positions is
        aligned with the weights it describes."""
        return self.rank[:, None].expand_as(valid).reshape(-1)[valid.reshape(-1)]


def rank_weighting_on(rank_weights: tuple[float, float, float]) -> bool:
    """Whether the tier multipliers do anything. Off means 022 never reads the rank column, so the
    experiment still trains on a v6 materialization that has none."""
    return tuple(rank_weights) != _RANK_WEIGHTS_OFF


def _ego_rank_weights(windows: list[dict], rank_weights: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """``(tier, multiplier)`` per window, read from the LAST frame of the ego rank column.

    The FIRST frame can be zero-filled left padding, which reads as ``Rank.UNKNOWN``; the last frame
    is always a real frame of the replay. With the multipliers off, a missing column (v6 data) and an
    UNKNOWN tier are both fine and weigh 1. With them on, either one is a silent 4x error in the data
    mixture, so both raise."""
    weighting = rank_weighting_on(rank_weights)
    if EGO_RANK_COLUMN not in windows[0]:
        if weighting:
            raise ValueError(
                f"awr_rank_weights={rank_weights} needs the {EGO_RANK_COLUMN!r} column, which this "
                "dataset does not carry; rank weighting requires a v7 materialization"
            )
        return np.zeros(len(windows), dtype=np.uint8), np.ones(len(windows), dtype=np.float32)
    rank = np.array([window[EGO_RANK_COLUMN][-1] for window in windows], dtype=np.uint8)
    if weighting and not rank.all():
        row = int(np.flatnonzero(rank == Rank.UNKNOWN)[0])
        frames = windows[row]["frame"]
        raise ValueError(
            f"window {row} of this batch (frames {int(frames[0])}..{int(frames[-1])}) has an UNKNOWN "
            f"ego rank, and awr_rank_weights={rank_weights} would weight it as if it were platinum"
        )
    # Index 0 is UNKNOWN, which only survives with the multipliers off, where every tier weighs 1.
    table = np.array((1.0, *rank_weights), dtype=np.float32)
    return rank, table[rank]


def collate_awr_batch(
    windows: list[dict],
    *,
    stats: dict[str, FeatureStats],
    L_ctx: int,
    rank_weights: tuple[float, float, float] = _RANK_WEIGHTS_OFF,
) -> AWRBatch:
    """Worker-side collate: hal's ``collate_train_batch`` builds the observation/action batch and the
    ego reward/return/rank columns are stacked beside it. The reward and return are sliced to the
    PREDICTED frames (``1 .. L_ctx``): position ``t`` carries ``r_{t+1}`` and ``G_{t+1}``, see
    :class:`AWRBatch`. The window always extends ``VAL_L_CHUNK`` target frames past the context, so
    the shifted slice never runs off the end. All three stay in the window dicts hal collates:
    ``features._classify`` does not recognize their names, so ``preprocess`` drops them and they can
    never reach the model as an input feature."""
    shifted = slice(1, L_ctx + 1)
    returns = np.stack([window[EGO_RETURN_COLUMN] for window in windows])[:, shifted]
    rewards = np.stack([window[EGO_REWARD_COLUMN] for window in windows])[:, shifted]
    rank, rank_weight = _ego_rank_weights(windows, rank_weights)
    batch = collate_train_batch(windows, stats=stats, L_ctx=L_ctx)
    return AWRBatch(
        batch=batch,
        returns=torch.from_numpy(np.ascontiguousarray(returns)),
        rewards=torch.from_numpy(np.ascontiguousarray(rewards)),
        rank=torch.from_numpy(rank),
        rank_weight=torch.from_numpy(rank_weight),
    )


def _attach_returns(loader: DataLoader, cfg: TrainConfig, stats: dict[str, FeatureStats]) -> DataLoader:
    """Re-point a loader hal built at the AWR sample stream and collate.

    hal keeps ownership of everything that is easy to get wrong — the StreamingDataset (shard
    prefetch depth, cache limit, shuffle geometry), the worker/pinning geometry and the window
    sampler itself; this replaces exactly two things: the replay stream (now return-labeled) and the
    collate (now emitting ``AWRBatch``). See ``ReturnLabeledReplays`` for the review flag."""
    sampler = loader.dataset
    if not isinstance(sampler, WindowDataset):
        raise TypeError(f"expected make_loader to yield a WindowDataset sampler, got {type(sampler).__name__}")
    sampler._mds = ReturnLabeledReplays(
        sampler._mds,
        gamma=cfg.awr_gamma,
        damage_shaping=cfg.awr_damage_shaping,
        win_reward=cfg.awr_win_reward,
    )
    loader.collate_fn = functools.partial(
        collate_awr_batch, stats=stats, L_ctx=cfg.L_ctx, rank_weights=cfg.awr_rank_weights
    )
    return loader


def awr_weights(
    advantage: Float[Tensor, " n_valid"] | None,
    *,
    beta: float,
    weight_max: float,
    rank_weight: Float[Tensor, " n_valid"] | None = None,
) -> tuple[Float[Tensor, " n_valid"], dict[str, float]]:
    """``w = clip(exp(A / beta), w_max) * rank``, rescaled so the batch mean is 1.

    The mean-1 rescale is cosmetic for the objective — ``_weighted_mean`` divides by ``sum(w)``,
    so the loss is batch-relative (self-normalized) with or without it. The self-normalization is
    what keeps the effective learning rate fixed while beta changes WHICH frames count, not how
    loudly; the rescale only pins the logged weight histogram to a fixed scale.
    The weights are pure data — the advantage must arrive detached, so no gradient can flow into the
    value head through the policy loss. Reported alongside: the effective sample size fraction
    ``(sum w)^2 / (N * sum w^2)`` (1.0 = uniform weighting, small = a few frames own the batch) and
    the fraction of positions sitting at the clip.

    ``advantage=None`` is the pure rank path: the raw weight is 1 everywhere and only the tier
    multiplier tilts it, so a rank-weighted BC arm needs no value head and no critic. The clip is on
    the advantage term ALONE, before the multiplier — a master frame may reach ``4 * w_max`` — and
    the clip fraction reports that term, not the product."""
    if advantage is None:
        if rank_weight is None:
            raise ValueError("awr_weights needs an advantage, a rank weight, or both; it was given neither")
        raw = torch.ones_like(rank_weight)
        clipped = torch.zeros_like(rank_weight)
    else:
        if advantage.requires_grad:
            raise ValueError("AWR weights must be built from a DETACHED advantage; the policy loss never trains V")
        raw = torch.exp(advantage / beta).clamp(max=weight_max)
        clipped = (raw >= weight_max).float()
    if rank_weight is not None:
        raw = raw * rank_weight
    weight = raw / raw.mean().clamp_min(torch.finfo(raw.dtype).tiny)
    n = weight.numel()
    if n == 0:
        return weight, {"ess": 0.0, "weight_max_frac": 0.0}
    ess = (weight.sum().pow(2) / (n * weight.pow(2).sum())).item()
    return weight, {"ess": ess, "weight_max_frac": clipped.mean().item()}


def rank_totals(rank: Int[Tensor, " n_valid"], weight: Float[Tensor, " n_valid"] | None) -> Float[Tensor, "2 n_rank"]:
    """Per-tier ``(position count, summed objective weight)`` — two reductions, no host copy.

    Totals rather than means so a step's micro-batches ADD: a tier that a micro-batch happened not
    to draw then contributes nothing, instead of a zero mean that would drag the step's number down.
    ``weight=None`` is the unweighted arm, where every position weighs 1."""
    idx = rank.long()
    counts = torch.bincount(idx, minlength=len(Rank)).float()
    per_position = counts.new_ones(rank.numel()) if weight is None else weight.to(counts.dtype)
    return torch.stack([counts, counts.new_zeros(len(Rank)).scatter_add_(0, idx, per_position)])


def rank_stats_from_totals(totals: Float[Tensor, "2 n_rank"]) -> dict[str, float]:
    """The logged tier block, from one host copy of ``rank_totals``.

    ``rank_weight_mean_*`` is the mean of the FINAL (rescaled) weight, so under rank-only weighting
    the three means sit exactly in the configured ratio, and under AWR + rank they drift from it by
    however much tier correlates with advantage — that drift is the signal, not a defect.
    ``rank_unknown_frac`` must be 0 on a v7 dataset; it is 1 on v6, which carries no tier at all."""
    count, total = totals.tolist()
    n = sum(count)
    out = {"rank_unknown_frac": count[Rank.UNKNOWN] / n if n else 0.0}
    for tier in _RANK_TIERS:
        name = tier.name.lower()
        out[f"rank_frac_{name}"] = count[tier] / n if n else 0.0
        out[f"rank_weight_mean_{name}"] = total[tier] / count[tier] if count[tier] else 0.0
    return out


def rank_stats(rank: Int[Tensor, " n_valid"], weight: Float[Tensor, " n_valid"] | None) -> dict[str, float]:
    """``rank_stats_from_totals`` over one batch, for the val pass and the tests."""
    return rank_stats_from_totals(rank_totals(rank, weight))


def expectile_td(
    value_grid: Float[Tensor, "B L_ctx"],
    rewards: Float[Tensor, "B L_ctx"],
    valid: Bool[Tensor, "B L_ctx"],
    *,
    gamma: float,
    tau: float,
) -> tuple[Float[Tensor, "B L_ctx"], Bool[Tensor, "B L_ctx"], Tensor]:
    """IQL's expectile TD residual ``u`` and its loss over one batch's value grid.

    ``u_t = r_{t+1} + gamma * V(s_{t+1}).detach() - V(s_t)``, in the collate's indexing where
    position ``t`` already carries the reward and return of the frame it predicts. The bootstrap is
    detached, so the loss pulls ``V(s_t)`` toward the target and never drags the target down to meet
    it. The loss is ``|tau - 1[u < 0]| * u^2``: at ``tau = 0.5`` exactly half the squared error, and
    above it a residual above the current V weighs ``tau / (1 - tau)`` times one below — an
    approximate max over the actions seen from that state, instead of the Monte-Carlo mean that
    averages in the opponent's luck.

    The last column has no successor state, so it is excluded from the loss and its residual is 0 —
    which is the neutral weight after ``exp(0)``, for the one position in ``L_ctx`` the objective
    still scores. ``valid`` is a prefix mask (``t >= ctx_pad``), so dropping its last column leaves
    exactly the positions where both ``t`` and ``t + 1`` are real."""
    target = rewards[:, :-1] + gamma * value_grid[:, 1:].detach()
    residual = F.pad(target - value_grid[:, :-1], (0, 1))
    td_valid = valid.clone()
    td_valid[:, -1] = False
    asymmetry = torch.where(residual < 0, 1.0 - tau, tau)
    return residual, td_valid, (asymmetry * residual.pow(2))[td_valid].mean()


@dataclass(frozen=True, slots=True)
class CriticParts:
    """What one batch's critic produced: the DETACHED advantage the weights are built from, the loss
    that fits V, and the diagnostics of BOTH critics so the arms stay comparable in W&B."""

    advantage: Float[Tensor, " n_valid"]
    loss: Tensor
    stats: dict[str, Tensor]


def critic_parts(
    value_grid: Float[Tensor, "B L_ctx"],
    batch: AWRBatch,
    valid: Bool[Tensor, "B L_ctx"],
    *,
    critic: str,
    gamma: float,
    tau: float,
) -> CriticParts:
    """Fit V and score the demonstrated frames, under the configured critic.

    ``mc`` regresses V on the sampled return and takes ``G_{t+1} - V(s_t)`` as the advantage (020's).
    ``expectile`` fits V by the expectile TD loss and takes the residual. Both report ``value_mse``,
    ``td_residual_mean`` and ``td_expectile_loss``: the numbers are elementwise cheap next to the
    backbone, and an arm's diagnostics are worth nothing if the other arm does not log them too."""
    flat_valid = valid.reshape(-1)
    value = value_grid.reshape(-1)[flat_valid]
    returns = batch.valid_returns(valid)
    value_mse = F.mse_loss(value, returns)
    residual, td_valid, td_loss = expectile_td(value_grid, batch.rewards, valid, gamma=gamma, tau=tau)
    stats = {
        "value_mse": value_mse.detach(),
        "td_residual_mean": residual[td_valid].mean().detach(),
        "td_expectile_loss": td_loss.detach(),
    }
    if critic == "mc":
        return CriticParts(advantage=(returns - value).detach(), loss=value_mse, stats=stats)
    if critic == "expectile":
        return CriticParts(advantage=residual.reshape(-1)[flat_valid].detach(), loss=td_loss, stats=stats)
    raise ValueError(f"unknown awr_critic {critic!r}")


def validate_chain_order(chain_order: tuple[str, ...]) -> tuple[str, ...]:
    """The prediction order must be a permutation of the four groups — every group predicted exactly
    once. Rejects a wrong length, a duplicate and an unknown name in one check."""
    chain = tuple(chain_order)
    if sorted(chain) != sorted(_GROUP_NAMES):
        raise ValueError(f"chain_order must be a permutation of {_GROUP_NAMES}, got {chain}")
    return chain


class FactoredHead(nn.Module):
    """One future-frame offset's action head, factorized WITHIN the frame (019's).

    Each group keeps its own projection (so the projections hold exactly the parameters of 020's one
    355-wide joint head). The groups are emitted in ``chain_order``; every non-terminal group feeds
    its chosen class back into the running hidden state through its own conditioning table:

        h_0 = h;   logits[g_i] = proj[g_i](h_i);   h_{i+1} = h_i + emb[g_i](id[g_i])

    The tables are zero-initialized, so at initialization ``logits_tf`` does not depend on the
    ancestors at all and this head IS the independent joint head. ``logits_tf`` teacher-forces the
    ancestors (training / validation); ``sample`` runs the same chain on its own draws (deploy)."""

    def __init__(self, d_model: int, chain_order: tuple[str, ...]) -> None:
        super().__init__()
        self.chain_order = validate_chain_order(chain_order)
        conditioning = set(self.chain_order[:-1])  # the terminal group conditions nothing
        # Built in _GROUP_NAMES order so the state-dict key order does not depend on chain_order.
        self.proj = nn.ModuleDict(
            {name: nn.Linear(d_model, _GROUP_VOCABS[_GROUP_INDEX[name]]) for name in _GROUP_NAMES}
        )
        self.emb = nn.ModuleDict(
            {
                name: nn.Embedding(_GROUP_VOCABS[_GROUP_INDEX[name]], d_model)
                for name in _GROUP_NAMES
                if name in conditioning
            }
        )
        for table in self.emb.values():
            nn.init.zeros_(table.weight)

    def logits_tf(self, h: Tensor, gt_idx: Tensor) -> dict[str, Tensor]:
        """Teacher-forced per-group logits ``{group: [..., vocab_g]}`` from hidden ``h`` ``[..., d_model]``
        and the GROUND-TRUTH class ids ``gt_idx`` ``[..., N_GROUPS]`` at the SAME target frame. Summing the
        four cross-entropies is then the chain rule, i.e. the joint NLL of that frame's action."""
        out: dict[str, Tensor] = {}
        x = h
        for i, name in enumerate(self.chain_order):
            out[name] = self.proj[name](x)
            if i + 1 < len(self.chain_order):
                x = x + self.emb[name](gt_idx[..., _GROUP_INDEX[name]])
        return out

    def sample(
        self,
        h: Tensor,
        *,
        group_temps: tuple[float, ...],
        btn_dead: Tensor | None,
        min_p: float,
        argmax: bool,
        gen: torch.Generator | None,
    ) -> Int[Tensor, "B n_groups"]:
        """Run the chain forward on its OWN draws: per group, support-mask (buttons only) → temperature
        → min-p → multinomial, then condition the remaining groups on the id just drawn. Returns class
        ids in ``_GROUP_NAMES`` order. ``argmax`` makes each step greedy — a greedy walk of the chain,
        NOT the joint mode — and deployed play never asks for it (greedy decode collapses the
        closed-loop policy to doing nothing)."""
        picks: dict[str, Tensor] = {}
        x = h
        for i, name in enumerate(self.chain_order):
            lg = self.proj[name](x).float()
            if btn_dead is not None and name == "buttons":
                lg = lg.masked_fill(btn_dead, float("-inf"))
            if argmax:
                pick = lg.argmax(-1)
            else:
                probs = F.softmax(lg / group_temps[_GROUP_INDEX[name]], dim=-1)
                if min_p > 0:
                    probs = probs * (probs >= min_p * probs.amax(dim=-1, keepdim=True))
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                pick = torch.multinomial(probs, 1, generator=gen).squeeze(-1)
            picks[name] = pick
            if i + 1 < len(self.chain_order):
                x = x + self.emb[name](pick)
        return torch.stack([picks[name] for name in _GROUP_NAMES], dim=-1)


class GPT(nn.Module):
    """Causal GPT over per-frame tokens with multi-token auxiliary heads. ``hidden[i]`` (causal) feeds
    one independent ``FactoredHead`` per offset in ``cfg.head_offsets``; head ``o`` predicts the action
    ``o`` frames ahead as a within-frame chain over the four groups (see ``FactoredHead``). Closed-loop
    decode uses only the offset-1 head (``primary_head_idx``); the rest are an auxiliary training signal.

    The block stack is ``hal.training.trunk.Trunk`` — the shared module, with sliding-window
    attention. Parameter creation order is 020's (embeddings, input projection, trunk, action heads,
    value head), so everything drawn BEFORE the heads matches a seeded 020 build; the heads
    themselves draw one projection per group instead of one joint matrix, so they (and the value
    head after them) take different draws from the same seed.

    ``value_head`` is the one architectural addition over a plain BC model: one scalar per frame off
    the same trunk hidden state, trained to the discounted return so the AWR advantage has a
    baseline (that MSE also trains the trunk unless ``cfg.awr_value_detach_trunk``). Decode never
    reads it."""

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
        self.chain_order = validate_chain_order(cfg.chain_order)

        # Gamestate categoricals: one table per feature name, shared across the four players.
        self.cat_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocab, dim) for name, (vocab, dim) in CAT_FEATURES.items()}
        )
        self.char_emb = nn.Embedding(cfg.char_vocab, cfg.char_dim)
        self.stage_emb = nn.Embedding(cfg.stage_vocab, cfg.stage_dim)
        per_player = len(FLOAT_FEATURES) * 2 + sum(dim for _, dim in CAT_FEATURES.values())  # float+mask+cat
        d_in = len(_PLAYER_PREFIXES) * per_player + A_DIM + 2 * cfg.char_dim + cfg.stage_dim  # 374

        self.ctx_proj = nn.Linear(d_in, cfg.d_model)
        self.trunk = Trunk(
            TrunkConfig(
                d_model=cfg.d_model,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                L_ctx=cfg.L_ctx,
                attn_window=cfg.attn_window,
                require_flex=cfg.require_flex,
            )
        )
        # One factorized head per future-frame offset (order matches self.head_offsets); every offset
        # runs the SAME chain, so the auxiliary heads train the conditioning the deployed head uses.
        self.heads = nn.ModuleList([FactoredHead(cfg.d_model, self.chain_order) for _ in offs])
        # V(s) for the AWR advantage. Last, so the trunk and the action heads keep their draws.
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
            # An absent sidecar reads as zeros, in the dtype of the block it is concatenated with
            # (fp16 under the eval cast, fp32 in training).
            parts.append(
                features[mk][..., None] if mk in features else torch.zeros(B, L, 1, device=device, dtype=ref.dtype)
            )
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

    def forward(self, features: dict[str, Tensor], ctx_pad: Int[Tensor, " B"]) -> Float[Tensor, "B L_ctx d_model"]:
        """Backbone hidden (one rmsnorm'd vector per frame); callers apply the per-offset heads."""
        return self.trunk(self._context_tokens(features), ctx_pad)

    @torch.no_grad()
    def forward_incremental(
        self,
        features: dict[str, Tensor],
        past: list[tuple[Tensor, Tensor] | None],
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """Encode one current token using per-layer rolling KV state. The trunk answers with the
        whole (one-token) sequence, so the last position is taken here."""
        h, new_past = self.trunk.forward_incremental(self._context_tokens(features), past)
        return h[:, -1], new_past


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


def group_nll(logits: dict[str, Tensor], tgt_idx: Tensor, valid: Tensor) -> dict[str, Tensor]:
    """Per-group categorical NLL (nats) over the VALID positions only. Returns ``{name: [n_valid]}``
    1D tensors (same ordering across groups) so callers reduce once for exact sample weighting.
    With teacher-forced ``logits`` these are CONDITIONALS whose sum is the joint NLL of the frame."""
    flat_valid = valid.reshape(-1)
    out: dict[str, Tensor] = {}
    for g, name in enumerate(_GROUP_NAMES):
        lg = logits[name].reshape(-1, logits[name].shape[-1])[flat_valid]
        out[name] = F.cross_entropy(lg, tgt_idx[..., g].reshape(-1)[flat_valid], reduction="none")
    return out


@dataclass(frozen=True, slots=True)
class LossParts:
    """One backbone forward's per-position pieces, all flattened by the SAME validity mask.

    ``nll`` and ``transition`` are keyed ``(offset, group_name)`` → ``[n_valid]`` (nats; bool);
    ``valid`` is the ``[B, L_ctx]`` mask that produced the flattening, so a caller can align any
    other per-position quantity (the AWR returns) with them.

    ``value_grid`` keeps ``V`` in its ``[B, L_ctx]`` shape, because the TD critic reads ``V(s_{t+1})``
    — the next COLUMN, which a flattened vector cannot name. ``value`` is that grid under the shared
    mask, derived rather than stored so the two can never disagree."""

    nll: dict[tuple[int, str], Tensor]
    transition: dict[tuple[int, str], Tensor]
    value_grid: Float[Tensor, "B L_ctx"]
    valid: Bool[Tensor, "B L_ctx"]

    @property
    def value(self) -> Float[Tensor, " n_valid"]:
        return self.value_grid.reshape(-1)[self.valid.reshape(-1)]


def action_loss(model: GPT, batch: TrainBatch, *, value_detach: bool = False) -> LossParts:
    """Dense multi-token NLL + aligned transition flags. Every valid context position predicts the action at
    each head offset; one shared backbone forward, one head each. ``a_full = [history | target]`` is quantized
    ONCE ([B, L_ctx+max_off, n_groups]) and its per-frame boundary mask computed once, both sliced per offset:
    for head ``o`` position ``i``'s target frame is ``i+o`` and it is a transition iff ``q[i+o] != q[i+o-1]``.
    That same quantized slice is ALSO the teacher-forcing input: the chain's ancestors are the ground-truth
    ids of the very frame being predicted, so summing the four group NLLs is the chain rule for the joint.
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
        tgt_idx = q_full[:, o : o + L_ctx]  # target frame i+o
        logits = {name: lg.float() for name, lg in model.heads[hi].logits_tf(h, tgt_idx).items()}
        bnd_o = bnd_full[:, o - 1 : o - 1 + L_ctx]  # transition at i iff q[i+o] != q[i+o-1]
        gnll = group_nll(logits, tgt_idx, valid)
        for g, name in enumerate(_GROUP_NAMES):
            nll[(o, name)] = gnll[name]
            trans[(o, name)] = bnd_o[..., g].reshape(-1)[flat_valid]
    value_in = h.detach() if value_detach else h
    value_grid = model.value_head(value_in).float().squeeze(-1)  # [B, L_ctx] V(s_i)
    return LossParts(nll=nll, transition=trans, value_grid=value_grid, valid=valid)


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


def _sample_action(
    model: GPT,
    head: FactoredHead,
    h: Tensor,
    *,
    group_temps: tuple[float, ...],
    btn_support_min: int,
    min_p: float,
    click_trigger_fix: bool,
    argmax: bool,
    gen: torch.Generator | None,
) -> Float[Tensor, "B d_action"]:
    """Sample one action vector by running ``head``'s chain over the hidden state ``[B, d_model]``: per
    group, support-mask -> temperature -> min-p -> multinomial (or ``argmax``), each draw conditioning the
    groups after it. Then dequantize + the click=>trigger fix. Shared by the offset-1 ``decode`` and the
    chunked ``chunk_from_hidden`` so the sampler never forks between the deploy paths."""
    dead = _btn_support_dead(model, btn_support_min, h.device) if btn_support_min >= 1 else None
    idx = head.sample(h, group_temps=group_temps, btn_dead=dead, min_p=min_p, argmax=argmax, gen=gen)
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
    offset-1 (primary) head only. The four groups are drawn as a CHAIN in ``chain_order`` — each group is
    sampled (per-group ``temp``-scaled softmax, optional min-p nucleus) and then conditions the groups
    after it. Decode-time hygiene applied in order support-mask -> per-group temperature -> min-p ->
    sample: ``btn_support_min`` >= 1 masks button combos with fewer than that many train frames to -inf;
    ``temps`` overrides ``temp`` per group (buttons, main_stick, c_stick, triggers); ``min_p`` > 0 keeps
    only classes with ``p >= min_p * p_max`` then renormalizes; ``click_trigger_fix`` forces trigger_l/r
    to 1.0 wherever the sampled combo sets the digital L/R bit. ``argmax`` walks the chain greedily,
    ignoring ``temps``/``min_p`` but respecting the mask. The value head takes no part: this is the plain
    BC sampler."""
    group_temps = _resolve_decode_args(temp, temps, btn_support_min, min_p, argmax)
    h = model(ctx.features, ctx.ctx_pad)[:, -1]  # [B, d_model]
    a = _sample_action(
        model,
        model.heads[model.primary_head_idx],
        h,
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
    Each offset runs its own within-frame chain; the chain is never carried ACROSS offsets (each head is a
    marginal over its own frame). Same sampling + hygiene as ``decode``; ``offsets == (1,)`` matches it."""
    group_temps = _resolve_decode_args(temp, temps, btn_support_min, min_p, argmax)
    h = model(ctx.features, ctx.ctx_pad)[:, -1]  # [B, d_model]
    return chunk_from_hidden(
        model,
        h,
        offsets,
        group_temps=group_temps,
        btn_support_min=btn_support_min,
        min_p=min_p,
        click_trigger_fix=click_trigger_fix,
        argmax=argmax,
        gen=gen,
    )


@torch.no_grad()
def chunk_from_hidden(
    model: GPT,
    h: Float[Tensor, "B d_model"],
    offsets: tuple[int, ...],
    *,
    group_temps: tuple[float, ...],
    btn_support_min: int = 0,
    min_p: float = 0.0,
    click_trigger_fix: bool = False,
    argmax: bool = False,
    gen: torch.Generator | None = None,
) -> Float[Tensor, "B s d_action"]:
    """The head + sampling half of a chunked decode, from a hidden state the caller already has.

    Split out because the incremental decoder produces that hidden state one frame at a time, in its
    own forward, and must not run the backbone a second time to sample from it."""
    actions: list[Tensor] = []
    for o in offsets:
        actions.append(
            _sample_action(
                model,
                model.heads[model.head_offsets.index(o)],
                h,
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
    if cfg.batch_size % cfg.grad_accum_steps:
        raise ValueError(
            f"batch_size={cfg.batch_size} is the EFFECTIVE batch and must divide into "
            f"grad_accum_steps={cfg.grad_accum_steps} equal micro-batches"
        )
    if not isinstance(cfg.attn_window, int) or isinstance(cfg.attn_window, bool) or cfg.attn_window < 0:
        raise ValueError(f"attn_window must be a non-negative integer (0 = full context), got {cfg.attn_window!r}")
    if cfg.eval_incremental_kv and cfg.attn_window == 0:
        raise ValueError(
            "eval_incremental_kv needs attn_window > 0: at full attention the rolling KV cache drops "
            "history the full forward keeps, so the decode is silently wrong past L_ctx frames. A "
            "full-context arm must say so: pass --cfg.no-eval-incremental-kv"
        )
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
    validate_chain_order(cfg.chain_order)
    if not isinstance(cfg.awr_enabled, bool):
        raise ValueError(f"awr_enabled must be a bool, got {cfg.awr_enabled!r}")
    if not math.isfinite(cfg.awr_gamma) or not 0.0 < cfg.awr_gamma <= 1.0:
        raise ValueError(f"awr_gamma must be in (0, 1], got {cfg.awr_gamma!r}")
    if not math.isfinite(cfg.awr_beta) or cfg.awr_beta <= 0:
        raise ValueError(f"awr_beta must be finite and > 0, got {cfg.awr_beta!r}")
    if not math.isfinite(cfg.awr_weight_max) or cfg.awr_weight_max <= 0:
        raise ValueError(f"awr_weight_max must be finite and > 0, got {cfg.awr_weight_max!r}")
    for name in ("awr_value_loss_weight", "awr_damage_shaping", "awr_win_reward"):
        value = getattr(cfg, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    if cfg.awr_critic not in ("mc", "expectile"):
        raise ValueError(f"awr_critic must be 'mc' or 'expectile', got {cfg.awr_critic!r}")
    if not math.isfinite(cfg.awr_expectile_tau) or not 0.0 < cfg.awr_expectile_tau < 1.0:
        raise ValueError(f"awr_expectile_tau must be in (0, 1), got {cfg.awr_expectile_tau!r}")
    weights = tuple(cfg.awr_rank_weights)
    if len(weights) != len(_RANK_TIERS):
        raise ValueError(f"awr_rank_weights needs one multiplier per tier {tuple(t.name for t in _RANK_TIERS)}")
    if any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError(f"awr_rank_weights must all be finite and > 0, got {weights}")
    if rank_weighting_on(weights) and cfg.mds_schema_version < _RANK_MDS_VERSION:
        # The tier column arrived with v7. Catch it here: the alternative is a run that dies on the
        # first collated batch, minutes into a cloud boot, with the loader's own error.
        raise ValueError(
            f"awr_rank_weights={weights} needs a v{_RANK_MDS_VERSION} materialization "
            f"(mds_schema_version={cfg.mds_schema_version} carries no rank column)"
        )
    if not isinstance(cfg.warmup_steps, int) or isinstance(cfg.warmup_steps, bool) or cfg.warmup_steps < 0:
        raise ValueError(f"warmup_steps must be a non-negative integer, got {cfg.warmup_steps!r}")
    if cfg.warmup_steps > cfg.max_steps:
        raise ValueError(f"warmup_steps={cfg.warmup_steps} exceeds max_steps={cfg.max_steps}")
    # final_eval_n_matchups = 0 skips the end-of-training closed-loop eval (an eval-incapable box).
    for name in ("val_every", "eval_every", "ckpt_every", "num_workers", "final_eval_n_matchups"):
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
    """Load 022 state, tolerating only the count buffer when inspecting an older checkpoint. A
    checkpoint without a ``value_head`` is rejected here as a missing key rather than silently
    loading with a randomly initialized baseline."""
    if any(key.startswith("blocks.") for key in state_dict):
        raise RuntimeError(
            "this checkpoint stores the transformer under 'blocks.*' — it is from 016/019/020/021, "
            "whose trunk is an inline copy. 022 holds the shared trunk under 'trunk.blocks.*', so "
            "the two are not interchangeable; evaluate that checkpoint with its own experiment file"
        )
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

    # Per-slot incremental state: the rolling KV cache and the hidden state of that slot's newest
    # frame. Slots are kept apart because instant-restart boundaries land on different frames.
    kv_cache: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    hidden: dict[int, torch.Tensor] = {}

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

    n_layers = len(model.trunk.blocks)

    @torch.no_grad()
    def encode_frame(ctx: Context) -> None:
        """Take this frame into every live slot's KV cache and keep the hidden state it produces.

        Called EVERY frame, so the cache never misses one while a chunk is being executed. Slots are
        batched by cache LENGTH, not by absolute frame: the cache is a rolling window and the trunk
        re-applies RoPE over its own positions, so two slots holding the same number of frames share
        one forward. Every slot saturates at ``attn_window``, so a wave settles into one forward per
        frame however far apart its matches drift."""
        if ctx.slot_ids is None or ctx.reset is None:
            raise ValueError("incremental closed-loop decode requires slot_ids and reset metadata")
        ids = [int(sid) for sid in ctx.slot_ids.tolist()]
        for sid, reset in zip(ids, ctx.reset.tolist(), strict=True):
            if reset:
                kv_cache.pop(sid, None)
                hidden.pop(sid, None)
        groups: dict[int, list[int]] = {}
        for row, sid in enumerate(ids):
            cached = kv_cache.get(sid)
            groups.setdefault(0 if cached is None else cached[0][0].size(2), []).append(row)
        for rows in groups.values():
            index = torch.tensor(rows, device=model_device, dtype=torch.long)
            features = {name: value.index_select(0, index) for name, value in ctx.features.items()}
            caches = [kv_cache.get(ids[row]) for row in rows]
            past: list[tuple[torch.Tensor, torch.Tensor] | None] = (
                [None] * n_layers
                if caches[0] is None
                else [
                    (
                        torch.cat([c[layer][0] for c in caches if c is not None], 0),
                        torch.cat([c[layer][1] for c in caches if c is not None], 0),
                    )
                    for layer in range(n_layers)
                ]
            )
            h, new = model.forward_incremental(features, past)
            for j, row in enumerate(rows):
                kv_cache[ids[row]] = [(k[j : j + 1], v[j : j + 1]) for k, v in new]
                hidden[ids[row]] = h[j]

    @torch.no_grad()
    def predict_incremental(ctx: Context, committed: np.ndarray | None) -> np.ndarray:
        """Sample a chunk from the hidden states ``encode_frame`` already computed — heads only, no
        second backbone forward. The heads 1..s all read the same state, so an execution horizon
        above 1 costs one extra head per frame it covers."""
        assert committed is None, "receding-horizon policy does not condition on a committed prefix"
        if ctx.slot_ids is None:
            raise ValueError("incremental closed-loop decode requires slot_ids metadata")
        h = torch.stack([hidden[int(sid)] for sid in ctx.slot_ids.tolist()])
        chunk = chunk_from_hidden(
            model,
            h,
            offsets,
            group_temps=settings.temps or (settings.temp,) * N_GROUPS,
            btn_support_min=settings.btn_support_min,
            min_p=settings.min_p,
            click_trigger_fix=settings.click_trigger_fix,
            gen=gen,
        )
        return chunk.cpu().numpy()

    incremental = cfg.eval_incremental_kv
    return RecedingHorizon(
        predict_chunk=predict_chunk,
        predict_incremental=predict_incremental if incremental else None,
        encode_frame=encode_frame if incremental else None,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=s,
        s=s,
        d=0,
        device=device,
        float_dtype=next(model.parameters()).dtype,
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
    parameter can silently escape an optimizer. The Muon sweep names the trunk's BLOCKS, not the
    trunk, so a parameter added beside the block stack later lands in AdamW and not silently here."""
    muon_params = [p for p in model.trunk.blocks.parameters() if p.ndim >= 2]
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
    this is exactly the plain mean; otherwise ``sum(w·nll)/sum(w)`` with ``w`` their product, so the
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
    position by the AWR weight ``sample_weight`` (None = the plain BC objective). The AWR weight applies to
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
        rewards=batch.rewards[:n],
        rank=batch.rank[:n],
        rank_weight=batch.rank_weight[:n],
    )


def _gradient_dot(a: tuple[Tensor, ...], b: tuple[Tensor, ...]) -> Tensor:
    return torch.stack([torch.dot(x.reshape(-1), y.reshape(-1)) for x, y in zip(a, b, strict=True)]).sum()


def _gradient_cosine(a: tuple[Tensor, ...], b: tuple[Tensor, ...], norm_a: Tensor, norm_b: Tensor) -> Tensor:
    denom = (norm_a * norm_b).clamp_min(torch.finfo(norm_a.dtype).tiny)
    return (_gradient_dot(a, b) / denom).clamp(-1.0, 1.0)


@contextlib.contextmanager
def _evaluation_mode(model: nn.Module) -> Iterator[None]:
    """Temporarily enter eval mode, on the UNCOMPILED forward, and restore both on every exit.

    Every non-training forward in this file goes through here, and ``compile_trunk`` installs the
    compiled forward as an INSTANCE attribute, so removing that attribute for the duration falls
    back to the class method. Validation and the gradient diagnostic each carry their own batch
    shape, and a compiled graph that meets one of them dies inside FlexAttention with a CUDA illegal
    memory access. Training keeps the compiled graph — and the same one, so its cache survives."""
    was_training = model.training
    compiled = model.__dict__.pop("forward", None)
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)
        if compiled is not None:
            model.forward = compiled


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


def _bool_mean(values: Tensor) -> float:
    return values.float().mean().item() if values.numel() else 0.0


def _group_kl_bits(logits_p: dict[str, Tensor], logits_q: dict[str, Tensor]) -> Tensor:
    """Summed-over-groups KL(p‖q) in bits per position. Both sides must be teacher-forced on the SAME
    ancestor ids, so this is the sum of the four CONDITIONAL KLs along that one ground-truth path — the
    joint KL restricted to it, not a marginal-by-marginal comparison."""
    first = logits_p[_GROUP_NAMES[0]]
    total = torch.zeros(first.shape[:-1], device=first.device)
    for name in _GROUP_NAMES:
        logp = F.log_softmax(logits_p[name], dim=-1)
        logq = F.log_softmax(logits_q[name], dim=-1)
        total = total + (logp.exp() * (logp - logq)).sum(-1)
    return total / _LN2


def _pooled_mean(parts: list[tuple[float, int]]) -> float:
    """The count-weighted mean of per-batch ``(mean, count)`` pairs — the mean over the pooled
    positions, up to summation order. One batch needs no combination at all, so its own number is
    returned untouched: multiplying by a count and dividing it out again would only round it."""
    if len(parts) == 1:
        return parts[0][0]
    count = sum(count for _, count in parts)
    return sum(value * count for value, count in parts) / count if count else 0.0


class _MeanAccumulator:
    """Validation metrics, collected one batch at a time.

    A 128-batch pass over 1024-frame windows scores 8.4M positions, so holding a per-position tensor
    for every metric until the end of the pass would cost gigabytes of VRAM. Each batch instead
    reports its own mean and the number of positions behind it, and nothing per-frame survives the
    loop iteration."""

    def __init__(self) -> None:
        self._parts: dict[str, list[tuple[float, int]]] = {}

    def add(self, key: str, value: float, count: int) -> None:
        """``value`` is this batch's mean over ``count`` positions. A count of 0 registers the key
        (the metric exists; its subset was empty in this batch) and moves the pooled mean nowhere."""
        self._parts.setdefault(key, []).append((value, count))

    def update(self, values: dict[str, float], count: int) -> None:
        for key, value in values.items():
            self.add(key, value, count)

    def means(self) -> dict[str, float]:
        return {key: _pooled_mean(parts) for key, parts in self._parts.items()}


def _hidden_pair(model: GPT, ctx: Context) -> tuple[Tensor, Tensor]:
    """The only two distinct trunk forwards a val batch needs: the state, and its history-ablated
    twin (the copycat probe, with the ego's own controller history zeroed). Both metric families
    read the pair inside one loop iteration, so a pass costs two forwards per batch, not five."""
    ablated_features = dict(ctx.features)
    for ch in ACTION_CHANNELS:
        ablated_features[f"ego_{ch}"] = torch.zeros_like(ablated_features[f"ego_{ch}"])
    return model(ctx.features, ctx.ctx_pad), model(ablated_features, ctx.ctx_pad)


@torch.no_grad()
def val_metrics(model: GPT, val_cache: list[AWRBatch], cfg: TrainConfig) -> tuple[dict[str, float], Tensor]:
    """Every validation number, in one pass over the frozen val batches, plus the pooled AWR weights.

    Dense multi-token proper scoring: per-offset NLL (``nll_off{o}``) tracks how predictability decays
    with horizon, and the offset-1 (deployed) head additionally drives button proper scoring,
    transition-vs-hold NLL splits, the button change-event F1 and change rate, and the copycat
    history-ablation probe. The AWR block adds how well the value head fits the return and what the
    weights it implies look like; the caller logs those weights as a histogram, because a collapsed
    distribution (a spike at the clip, everything else at zero) is the failure mode ``awr_beta``
    guards against and a mean alone hides it. Every NLL is UNWEIGHTED — AWR touches the backprop
    objective only — and the weights here are a diagnostic.

    Every per-group number reads a CHAIN CONDITIONAL, teacher-forced on the ground-truth ancestors of
    the same target frame; only ``loss`` and the ``nll_off{o}`` totals (the chain-rule joint) compare
    with a joint-head run.

    022 drops the metrics a ten-run correlation study found uninformative or inverted against
    closed-loop play: reconstruction accuracy, prediction persistence and flip-back rates, the
    dataset-constant transition and multipress rates, and the chance-level stick-side change F1 and
    Brier scores."""
    with _evaluation_mode(model):
        return _val_metrics_eval(model, val_cache, cfg)


def _val_metrics_eval(model: GPT, val_cache: list[AWRBatch], cfg: TrainConfig) -> tuple[dict[str, float], Tensor]:
    acc = _MeanAccumulator()
    # The two exceptions to scalar accumulation, both small. The tolerant change F1 matches events
    # with +-1 frame of slack, so it needs the frames themselves ([B, L_ctx] bools); the weight
    # histogram and the ESS are properties of the pooled weight vector, which the mean-1 rescale
    # inside ``awr_weights`` defines over the whole val set.
    btn_pred_change: list[Tensor] = []
    btn_true_change: list[Tensor] = []
    values: list[Tensor] = []
    returns: list[Tensor] = []
    advantages: list[Tensor] = []
    rank_weights: list[Tensor] = []
    ranks: list[Tensor] = []
    counts_available = bool((model.button_combo_counts >= 0).all())
    rare_mask = model.button_combo_counts < cfg.diagnostic_rare_button_count
    unseen_mask = model.button_combo_counts == 0
    combo_bits = scoring.combo_to_buttons(torch.arange(scoring.N_BUTTON_COMBOS, device=model.main_centers.device))
    device = next(model.parameters()).device
    for cached in val_cache:
        awr_batch = cached.to(device)  # the cache is in host memory; one batch is on the device
        batch = awr_batch.batch
        ctx = batch.context
        h, h_ablated = _hidden_pair(model, ctx)
        targets, valid = _multi_offset_targets(ctx, batch.target[:, : max(model.head_offsets)], model.head_offsets)
        flat_valid = valid.reshape(-1)
        n_valid = int(flat_valid.sum())
        cur_idx = _quantize(model, stack_actions(ctx.features))  # [B, L_ctx, n_groups] current frames
        comps: dict[tuple[int, str], Tensor] = {}
        ablated: dict[str, Tensor] = {}
        for hi, o in enumerate(model.head_offsets):
            tgt_idx = _quantize(model, targets[o])
            logits = {name: lg.float() for name, lg in model.heads[hi].logits_tf(h, tgt_idx).items()}
            comps.update({(o, name): c for name, c in group_nll(logits, tgt_idx, valid).items()})
            if o != 1:
                continue
            # The deployed head drives the button / transition / ablation stats.
            true_change = scoring.transition_mask(torch.cat([cur_idx, tgt_idx[:, -1:]], dim=1))  # [B,L_ctx,n_grp]
            # Same teacher-forced ancestors on both sides, so the ablation compares like with like.
            ablated_logits = {
                name: lg.float()
                for name, lg in model.heads[model.primary_head_idx].logits_tf(h_ablated, tgt_idx).items()
            }
            kl_bits = _group_kl_bits(logits, ablated_logits).reshape(-1)[flat_valid]  # KL(full ‖ ablated), bits
            acc.add("ablate_hist_kl", kl_bits.mean().item(), n_valid)
            ablated = group_nll(ablated_logits, tgt_idx, valid)
            for g, name in enumerate(_GROUP_NAMES):
                trans = true_change[..., g].reshape(-1)[flat_valid]
                acc.add(f"nll_{name}_trans", _masked_mean_bits(comps[(1, name)], trans), int(trans.sum()))
                acc.add(f"nll_{name}_hold", _masked_mean_bits(comps[(1, name)], ~trans), int((~trans).sum()))
            btn_logits = logits["buttons"]
            combo_probs = F.softmax(btn_logits.reshape(-1, scoring.N_BUTTON_COMBOS)[flat_valid], dim=-1)
            onehot = F.one_hot(tgt_idx[..., _BUTTONS_G].reshape(-1)[flat_valid], scoring.N_BUTTON_COMBOS).to(
                combo_probs.dtype
            )
            acc.add("brier_buttons", (combo_probs - onehot).pow(2).sum(-1).mean().item(), n_valid)
            pred_change = (btn_logits.argmax(-1) != cur_idx[..., _BUTTONS_G]) & valid
            btn_pred_change.append(pred_change)
            btn_true_change.append(true_change[..., _BUTTONS_G] & valid)
            acc.add("pred_change_rate_buttons", _bool_mean(pred_change.reshape(-1)[flat_valid]), n_valid)
            marginal_btn_probs = combo_probs @ combo_bits.to(combo_probs.dtype)
            tgt_btn = _dequantize(model, tgt_idx)[..., _N_CONT:].reshape(-1, _N_BUTTONS)[flat_valid]
            logloss, brier = scoring.bernoulli_scores_from_probs(marginal_btn_probs, tgt_btn)
            acc.add("btn_logloss", logloss.item(), n_valid)
            acc.add("btn_brier", brier.item(), n_valid)
            if counts_available:
                acc.add("btn_rare_mass", combo_probs[:, rare_mask].sum(-1).mean().item(), n_valid)
                acc.add("btn_unseen_mass", combo_probs[:, unseen_mask].sum(-1).mean().item(), n_valid)
            # Impossible-joint mass, now read as a CONDITIONAL: how much probability the buttons
            # conditional still puts on an L/R click when the ground-truth trigger of the SAME frame is
            # not full. A joint head could only multiply two independent marginals; here the buttons
            # conditional has already seen the true trigger id (default chain order), so a nonzero value
            # is a real contradiction. A chain order that puts buttons BEFORE triggers makes this weaker
            # (the buttons conditional no longer sees them) — read it beside cfg.chain_order.
            n_trig = model.trig_centers.shape[0]
            gt_trig = tgt_idx[..., _TRIG_G].reshape(-1)[flat_valid]
            l_not_full = (gt_trig // n_trig != n_trig - 1).to(marginal_btn_probs.dtype)
            r_not_full = (gt_trig % n_trig != n_trig - 1).to(marginal_btn_probs.dtype)
            invalid_l = marginal_btn_probs[:, _BUTTON_L_CH - _N_CONT] * l_not_full
            invalid_r = marginal_btn_probs[:, _BUTTON_R_CH - _N_CONT] * r_not_full
            acc.add("click_trigger_invalid_l_mass", invalid_l.mean().item(), n_valid)
            acc.add("click_trigger_invalid_r_mass", invalid_r.mean().item(), n_valid)
        primary = nll_breakdown({name: comps[(1, name)] for name in _GROUP_NAMES})
        acc.add("loss", primary["total"], n_valid)  # offset-1 total bits/frame (deployed policy)
        acc.update({f"nll_{name}": primary[name] for name in _GROUP_NAMES}, n_valid)
        acc.add(
            "cont_discrete_bits",
            (comps[(1, "main_stick")].mean() + comps[(1, "c_stick")].mean() + comps[(1, "triggers")].mean()).item()
            / _LN2,
            n_valid,
        )
        acc.update({f"nll_off{o}": _offset_total_bits(comps, o) for o in model.head_offsets}, n_valid)
        ablate_total = 0.0
        for name in _GROUP_NAMES:
            d = (ablated[name].mean() - comps[(1, name)].mean()).item() / _LN2  # positive ⇒ history helps
            acc.add(f"ablate_hist_dnll_{name}", d, n_valid)
            ablate_total += d
        acc.add("ablate_hist_dnll", ablate_total, n_valid)
        value_grid = model.value_head(h).float().squeeze(-1)
        critic = critic_parts(
            value_grid,
            awr_batch,
            valid,
            critic=cfg.awr_critic,
            gamma=cfg.awr_gamma,
            tau=cfg.awr_expectile_tau,
        )
        acc.add("td_residual_mean", critic.stats["td_residual_mean"].item(), n_valid)
        acc.add("td_expectile_loss", critic.stats["td_expectile_loss"].item(), n_valid)
        values.append(value_grid.reshape(-1)[flat_valid])
        returns.append(awr_batch.valid_returns(valid))
        # The advantage is a diagnostic here even for a BC arm: it says what the weighting WOULD do.
        advantages.append(critic.advantage)
        rank_weights.append(awr_batch.valid_rank_weights(valid))
        ranks.append(awr_batch.valid_ranks(valid))

    out = acc.means()
    out["btn_counts_available"] = float(counts_available)
    if counts_available:
        out["btn_rare_count_threshold"] = float(cfg.diagnostic_rare_button_count)
    out["click_trigger_invalid_mass"] = 0.5 * (
        out["click_trigger_invalid_l_mass"] + out["click_trigger_invalid_r_mass"]
    )
    out["changeF1_buttons"] = scoring.change_event_prf(torch.cat(btn_pred_change), torch.cat(btn_true_change))[2]
    value = torch.cat(values)
    target = torch.cat(returns)
    rank_on = rank_weighting_on(cfg.awr_rank_weights)
    weight, stats = awr_weights(
        torch.cat(advantages),
        beta=cfg.awr_beta,
        weight_max=cfg.awr_weight_max,
        rank_weight=torch.cat(rank_weights) if rank_on else None,
    )
    out.update(
        {
            "value_mse": F.mse_loss(value, target).item(),
            "value_mean": value.mean().item(),
            "return_mean": target.mean().item(),
            "return_std": target.std().item(),
            "awr_ess": stats["ess"],
            "awr_weight_max_frac": stats["weight_max_frac"],
            **rank_stats(torch.cat(ranks), weight),
        }
    )
    return out, weight


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
    """Loader arguments shared by the train and val splits. The loader yields MICRO-batches: one
    optimizer step consumes ``grad_accum_steps`` of them. ``schema_version`` declares which MDS
    materialization this run reads, so a stale (or newer) dataset raises on the first row."""
    return dict(
        data_root=cfg.data_root,
        remote=streams.remote_for_local(cfg.data_root),
        cache_limit=f"{cfg.cache_limit_gb}gb",
        shuffle_block_size=cfg.shuffle_block_size,
        stats=stats,
        L_ctx=cfg.L_ctx,
        L_chunk=max(cfg.head_offsets),  # target horizon must cover the farthest auxiliary head
        batch_size=_micro_batch(cfg),
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
            "AWR weights the BACKPROP objective only; train/loss and every val metric are unweighted, "
            "so the arms of this experiment compare directly with each other"
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
    if cfg.compile_trunk:
        # The BOUND method, not the module: reassigning ``model.trunk`` would rename every trunk key
        # in the state dict (``trunk._orig_mod.blocks…``) and break resume and eval.
        model.forward = torch.compile(model.forward, dynamic=False)
    n_params = sum(p.numel() for p in model.parameters())
    if wandb.run is not None:
        wandb.run.summary["model/num_params"] = n_params
    print(
        f"[model] {_model_tag(cfg)}  num_params={n_params / 1e6:.2f}M  "
        f"batch={cfg.batch_size} (micro {_micro_batch(cfg)} x accum {cfg.grad_accum_steps})",
        flush=True,
    )
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
    # The cache stays in host memory and each batch moves to the device inside the val loop: 128
    # batches of 64 x 1024 frames are 2.7 GiB of observations, which would sit on the GPU for the
    # whole run and buy nothing.
    val_cache = list(itertools.islice(val_loader, cfg.val_n_batches))
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
        """Flat ``val/*`` metric dict (one W&B section). Merged into the per-step log; no wandb.log
        here. One pass over the frozen val batches produces the NLL block and the AWR block
        together, at two trunk forwards per batch."""
        vm, weights = val_metrics(model, val_cache, cfg)
        out: dict[str, object] = {f"val/{k}": v for k, v in vm.items()}
        out["val/awr_weights"] = wandb.Histogram(weights.detach().cpu().numpy())
        out.update(gradient_diagnostics(model, val_cache[0].to(DEVICE), cfg))
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
    rank_on = rank_weighting_on(cfg.awr_rank_weights)
    run_t0 = time.monotonic()
    for step in range(start_step, cfg.max_steps):
        with profile("step") as sw:
            opt.zero_grad()
            comps_acc: dict[tuple[int, str], list[Tensor]] = {}
            obj_acc: Tensor | None = None
            awr_acc: dict[str, float] = {"value_loss": 0.0, "ess": 0.0, "weight_max_frac": 0.0}
            rank_acc: Tensor | None = None
            for _ in range(cfg.grad_accum_steps):
                try:
                    batch = next(it).to(DEVICE)
                except StopIteration:
                    it = iter(train_loader)
                    batch = next(it).to(DEVICE)
                with autocast:
                    parts = action_loss(model, batch.batch, value_detach=cfg.awr_value_detach_trunk)
                    # AWR: score the demonstrated frames by their outcome. The advantage is detached, so
                    # the policy loss never trains V; the value loss reaches the shared trunk unless
                    # awr_value_detach_trunk closes that path too. Rank weighting is a property of the
                    # weighting and not of the critic, so it also runs on its own (weighted BC): then
                    # there is no advantage, no value loss and no gradient into the value head at all.
                    rank_weight = batch.valid_rank_weights(parts.valid) if rank_on else None
                    weight: Tensor | None = None
                    value_loss = torch.zeros((), device=parts.value_grid.device)
                    if cfg.awr_enabled:
                        critic = critic_parts(
                            parts.value_grid,
                            batch,
                            parts.valid,
                            critic=cfg.awr_critic,
                            gamma=cfg.awr_gamma,
                            tau=cfg.awr_expectile_tau,
                        )
                        value_loss = critic.loss
                        weight, awr_stats = awr_weights(
                            critic.advantage,
                            beta=cfg.awr_beta,
                            weight_max=cfg.awr_weight_max,
                            rank_weight=rank_weight,
                        )
                    elif rank_weight is not None:
                        weight, awr_stats = awr_weights(
                            None, beta=cfg.awr_beta, weight_max=cfg.awr_weight_max, rank_weight=rank_weight
                        )
                    obj = objective(
                        parts.nll, parts.transition, cfg.aux_loss_weight, cfg.transition_loss_weight, weight
                    )
                    total = obj + cfg.awr_value_loss_weight * value_loss if cfg.awr_enabled else obj
                    loss = total / cfg.grad_accum_steps
                loss.backward()
                obj_acc = obj.detach() if obj_acc is None else obj_acc + obj.detach()
                if weight is None:
                    awr_acc["ess"] += 1.0 / cfg.grad_accum_steps  # unweighted: every position counts once
                else:
                    awr_acc["ess"] += awr_stats["ess"] / cfg.grad_accum_steps
                    awr_acc["weight_max_frac"] += awr_stats["weight_max_frac"] / cfg.grad_accum_steps
                if cfg.awr_enabled:
                    awr_acc["value_loss"] += value_loss.item() / cfg.grad_accum_steps
                totals = rank_totals(batch.valid_ranks(parts.valid), weight)
                rank_acc = totals if rank_acc is None else rank_acc + totals
                for k, v in parts.nll.items():
                    comps_acc.setdefault(k, []).append(v.detach())
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))  # measure only
            opt.step()
            sched.step()
            if DEVICE == "cuda":
                torch.cuda.synchronize()
        assert obj_acc is not None and rank_acc is not None  # grad_accum_steps >= 1
        objective_bits = (obj_acc / cfg.grad_accum_steps).item() / _LN2  # the actual backprop objective, bits
        comps_cat = {k: torch.cat(v) for k, v in comps_acc.items()}
        primary = nll_breakdown({name: comps_cat[(1, name)] for name in _GROUP_NAMES})
        aux_offsets = [o for o in cfg.head_offsets if o != 1]
        aux_loss = (
            sum(_offset_total_bits(comps_cat, o) for o in aux_offsets) / len(aux_offsets) if aux_offsets else 0.0
        )
        sps = cfg.batch_size / sw.elapsed  # batch_size is the effective batch: one step's samples
        samples = (step + 1) * cfg.batch_size
        log: dict[str, object] = {
            "global_step": step,
            "samples": samples,
            "tokens": samples * cfg.L_ctx,
            "train/loss": primary["total"],  # offset-1 head (deployed), UNWEIGHTED, so the arms compare
            **{f"train/nll_{name}": primary[name] for name in _GROUP_NAMES},
            "train/aux_loss": aux_loss,  # mean total bits/frame over the auxiliary (offset != 1) heads
            "train/objective": objective_bits,  # the weighted policy objective actually backpropagated, bits
            # The configured critic's loss in native units, not bits: V's MSE to the return under
            # "mc", the expectile TD loss under "expectile". Comparable only within a critic.
            "train/value_loss": awr_acc["value_loss"],
            "train/awr_ess": awr_acc["ess"],  # 1.0 = uniform weights; small = a few frames own the batch
            "train/awr_weight_max_frac": awr_acc["weight_max_frac"],
            # Tier mix and what each tier's frames are worth to the objective, pooled over the step's
            # micro-batches. Under rank-only weighting the three means are the configured ratio;
            # under AWR + rank they drift by however much tier correlates with advantage.
            **{f"train/{key}": value for key, value in rank_stats_from_totals(rank_acc).items()},
            "lr/muon": next(g["lr"] for g in opt.param_groups if g["use_muon"]),
            "lr/adam": next(g["lr"] for g in opt.param_groups if not g["use_muon"]),
            "train/gnorm": grad_norm.item(),
            "throughput/step_s": sw.elapsed,
            "throughput/samples_per_s": sps,
        }
        if step == start_step:
            # The trunk resolves flex-vs-dense at its first forward, so the answer exists only now.
            if wandb.run is not None:
                wandb.run.summary["model/attn_path"] = model.trunk.attn_path
            print(f"[model] attention path: {model.trunk.attn_path}, window={cfg.attn_window}", flush=True)
        if step < 20 or step % 50 == 0:
            # ESS rides the console line: a weight distribution collapsing onto a few frames is the
            # failure mode of this experiment, and it must be visible in train.log without W&B. It
            # is 1.0 on an unweighted arm and a real number on a rank-weighted BC arm, so it prints
            # either way; ``v_loss`` is the configured critic's loss, MSE or expectile.
            awr_note = f" ess {awr_acc['ess']:.3f}"
            if cfg.awr_enabled:
                awr_note += f" v_loss {awr_acc['value_loss']:.4f}"
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
    # Save BEFORE the closed-loop eval. A box that cannot boot Dolphin (the H100's glibc is older
    # than the build's floor) otherwise crashes at the finish line and the weights are lost.
    _save("final.pt", cfg.max_steps)
    if cfg.final_eval_n_matchups > 0:
        _log_eval(cfg.max_steps, _eval_and_upload("final", n_matchups=cfg.final_eval_n_matchups))
    else:
        print("[final] closed-loop eval skipped (final_eval_n_matchups=0)", flush=True)
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
    """The one door every eval path (``--eval``, the async eval worker, ``hal.scripts.h2h``) loads a
    checkpoint through, so the decode-speed settings live here and no entry point can forget them."""
    # train() sets this per step; eval never did, and the default ("highest") costs the closed-loop
    # forward 1.09x for precision the sampler throws away.
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = _cfg_from_state(state["cfg"])
    torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    embedded_counts = "button_combo_counts" in state["model"]
    button_combo_counts = None if embedded_counts else _load_button_combo_counts(cfg)
    validate_config(cfg, has_button_combo_counts=embedded_counts or button_combo_counts is not None)
    model = GPT(cfg).to(DEVICE)
    if button_combo_counts is not None:
        model.button_combo_counts.copy_(button_combo_counts.to(DEVICE))
    _load_model_state(model, state["model"])
    model.eval()
    if cfg.eval_fp16 and DEVICE == "cuda":
        _halve_for_decode(model)
    stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
    return model, cfg, stats, state


def _halve_for_decode(model: GPT) -> None:
    """Cast the model's float parameters to fp16, and put the quantization grids back to fp32.

    The grids are the decode's OUTPUT scale: a stick center sits on the 1/80 grid and every stored
    value must reproduce its exact byte through the pipe, which fp16's ~5e-4 of relative slack
    cannot promise. The trunk and the heads have no such contract — their logits are cast back to
    fp32 before the softmax."""
    grids = {name: getattr(model, name) for name in ("main_centers", "c_centers", "trig_centers")}
    model.half()
    for name, grid in grids.items():
        setattr(model, name, grid)  # the ORIGINAL tensor: a fp32->fp16->fp32 round trip loses 2e-4


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
        columns = replay_returns(
            sample, gamma=cfg.awr_gamma, damage_shaping=cfg.awr_damage_shaping, win_reward=cfg.awr_win_reward
        )
        per_replay.extend(columns[name] for name in _PORT_RETURN_COLUMNS)
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
        "win_reward": cfg.awr_win_reward,
        "mean": float(returns.mean()),
        "std": float(returns.std()),
        "within_replay_std": float(np.mean([g.std() for g in per_replay])),
        "near_zero_frac": float(np.mean(np.abs(returns) < 0.01)),
        "quantiles": {f"p{q}": float(np.percentile(returns, q)) for q in quantiles},
        "beta_table": rows,
    }
    print(
        f"[audit] {out['replays']} replays  {out['frames']} port-frames  gamma={cfg.awr_gamma}  "
        f"damage_shaping={cfg.awr_damage_shaping}  win_reward={cfg.awr_win_reward}",
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
    # The reward spec is appended, never typed: a hand-written comment cannot disagree with the flags.
    comment = args.comment or f"gpt-{cfg.max_steps // 1000}k-b{cfg.batch_size}"
    train(cfg, stats, comment=f"{comment}-{_reward_tag(cfg)}")


if __name__ == "__main__":
    main(tyro.cli(Args))
