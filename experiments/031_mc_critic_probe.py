"""Dense direct-Monte-Carlo critic probe for the frozen experiment-026 trunk.

This is deliberately an offline probe, not a deployable policy and not IQL.  For
every real context prefix it fits::

    V(s_t)                    -> G_{t+1}
    Q1(s_t, a_{t+1})          -> G_{t+1}
    Q4(s_t, a_{t+1:t+4})      -> G_{t+1}

The return is computed on the full replay before windows are selected.  A replay
whose terminal boundary is unknown is retained for BC but receives no MC critic
labels.  This prevents an incomplete tail from silently becoming a zero target.

The module contains the model and analysis contracts so a cheap three-seed probe
can be driven by a cluster launcher without coupling it to a particular launcher.
It is also importable on a data-free CPU checkout for contract tests.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from streaming import StreamingDataset
from torch import Tensor

from hal.data.policy_schema import decode_policy_replay
from hal.data.schema import check_schema_version
from hal.training import returns as returns_lib
from hal.training.dataloader import _make_window
from hal.training.dataloader import collate_train_batch
from hal.training.dataloader import collate_windows
from hal.training.ego_stats import load_consolidated_stats
from hal.training.features import A_DIM
from hal.training.features import BASE_ACTION_PROJECTION
from hal.training.features import V6_PLAYER_COLUMNS
from hal.training.features import Context
from hal.training.features import FeatureProjection
from hal.training.features import TrainBatch
from hal.training.features import stack_actions


def _load_026():
    path = Path(__file__).with_name("026_temporal_mtp.py")
    name = "hal_exp026_for_mc_probe"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load experiment dependency {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_exp026 = _load_026()
GROUP_NAMES: tuple[str, ...] = _exp026.GROUP_NAMES
N_GROUPS: int = _exp026.N_GROUPS
HORIZONS: tuple[int, int] = (1, 4)
MC_RETURN_SUFFIX = "mc_probe_return"
MC_VALID_SUFFIX = f"{MC_RETURN_SUFFIX}_valid"


class ControllerCodec(Protocol):
    embed_dim: int

    def quantize(self, actions: Tensor) -> Tensor: ...

    def embed_frame(self, indices: Tensor) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    action_d_model: int = 128
    action_layers: int = 2
    action_heads: int = 4
    action_ff_dim: int = 256
    head_hidden_dim: int = 256
    huber_delta: float = 1.0
    gamma: float = 0.99 ** (1.0 / 4.0)
    critic_seeds: tuple[int, ...] = (0, 1, 2)
    weight_betas: tuple[float, ...] = (0.2, 0.4, 0.8, 1.6, 3.2)
    weight_cap: float = 5.0

    def validate(self) -> None:
        if self.action_d_model <= 0 or self.action_layers != 2:
            raise ValueError("the probe requires a positive width and exactly two causal action layers")
        if self.action_d_model % self.action_heads:
            raise ValueError("action_d_model must be divisible by action_heads")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if self.huber_delta <= 0 or self.weight_cap <= 0:
            raise ValueError("Huber delta and weight cap must be positive")
        if len(self.critic_seeds) != 3 or len(set(self.critic_seeds)) != 3:
            raise ValueError("the gate is frozen to three distinct critic seeds")
        if any(beta <= 0 for beta in self.weight_betas):
            raise ValueError("weight betas must be positive")


def discounted_mc_return(reward: np.ndarray, gamma: float, *, terminated: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return ``G`` and its validity for one complete replay.

    A false ``terminated`` means the file ended without a known episode boundary.
    Partial returns are intentionally represented by NaN plus a false mask.
    The formula lives in :mod:`hal.training.returns`; this wrapper keeps the
    probe's tuple contract.
    """
    values = np.asarray(reward, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("reward must be one-dimensional")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    valid = np.full(values.shape, bool(terminated), dtype=np.bool_)
    if not terminated:
        return np.full(values.shape, np.nan, dtype=np.float32), valid
    return returns_lib.discounted_return(values, gamma), valid


infer_terminal_replay = returns_lib.infer_terminal_replay


def label_mc_replay(
    sample: dict,
    *,
    gamma: float,
    damage_shaping: float = 0.01,
    win_reward: float = 0.5,
    terminated: bool | None = None,
) -> dict:
    """Attach full-episode, per-port MC returns before random windowing."""
    return returns_lib.label_replay(
        sample,
        gamma=gamma,
        damage_shaping=damage_shaping,
        win_reward=win_reward,
        suffix=MC_RETURN_SUFFIX,
        terminated=terminated,
    )


@dataclass(frozen=True, slots=True)
class DenseAlignment:
    """Dense labels aligned to state positions, before flattening valid rows."""

    chunks: Tensor  # [B, L, 4, G] or [B, L, 4, A_DIM]
    target: Tensor  # [B, L], G_{t+1}
    q1_mask: Tensor  # [B, L]
    q4_mask: Tensor  # [B, L]

    @property
    def valid_prefix_counts(self) -> dict[int, int]:
        return {1: int(self.q1_mask.sum()), 4: int(self.q4_mask.sum())}


@dataclass(frozen=True, slots=True)
class ExtractionBatch:
    batch: TrainBatch
    returns: Tensor
    return_valid: Tensor
    state_indices: Tensor
    ego_ids: Tensor


def collate_extraction_batch(windows: list[dict], batch: TrainBatch) -> ExtractionBatch:
    """Keep raw MC label columns beside the normal preprocessed 026 batch."""
    stacked = collate_windows(windows)
    returns = torch.from_numpy(np.ascontiguousarray(stacked[f"ego_{MC_RETURN_SUFFIX}"]))
    valid = torch.from_numpy(np.ascontiguousarray(stacked[f"ego_{MC_VALID_SUFFIX}"])).bool()
    state_indices = torch.from_numpy(np.ascontiguousarray(stacked["mc_state_index"])).long()
    ego_ids = torch.from_numpy(np.asarray(stacked["mc_ego_id"], dtype=np.int64)).long()
    return ExtractionBatch(batch, returns, valid, state_indices, ego_ids)


def exhaustive_state_blocks(length: int, context_length: int) -> tuple[tuple[int, int], ...]:
    """Partition every state with four future actions into gap-free blocks."""
    if length < 0 or context_length <= 0:
        raise ValueError("length must be nonnegative and context_length positive")
    stop = max(length - 4, 0)
    return tuple((start, min(start + context_length, stop)) for start in range(0, stop, context_length))


def exhaustive_replay_windows(
    sample: dict,
    *,
    replay_id: str,
    context_length: int,
    projection: FeatureProjection | None,
) -> list[dict]:
    """Tile a replay so every ``(ego, s_t)`` is a target once and only once."""
    length = len(sample["frame"])
    windows: list[dict] = []
    for ego_id, ego_prefix in enumerate(("p1", "p2"), start=1):
        for state_start, state_stop in exhaustive_state_blocks(length, context_length):
            chunk_start = state_stop
            virtual_start = chunk_start - context_length
            pad = max(0, -virtual_start)
            window = _make_window(
                sample,
                ego_prefix=ego_prefix,
                start=virtual_start,
                pad=pad,
                length=context_length + 4,
                projection=projection,
            )
            window["ctx_pad"] = np.int64(min(pad, context_length))
            state_indices = np.full(context_length, -1, dtype=np.int64)
            selected_start = context_length - (state_stop - state_start)
            state_indices[selected_start:] = np.arange(state_start, state_stop, dtype=np.int64)
            window["mc_state_index"] = state_indices
            window["mc_ego_id"] = np.int64(ego_id)
            window["mc_replay_id"] = replay_id
            windows.append(window)
    return windows


def align_dense_prefixes(
    actions: Tensor,
    returns: Tensor,
    return_valid: Tensor,
    ctx_pad: Tensor,
    *,
    state_length: int,
) -> DenseAlignment:
    """Align every ``s_t`` with future actions and ``G_{t+1}``.

    ``actions`` and return columns cover at least ``state_length + 4`` frames.
    The extra tail is why the final context prefixes receive honest Q4 labels.
    """
    if actions.ndim not in (3, 4):
        raise ValueError("actions must be [B,T,A] or already-quantized [B,T,4]")
    batch, length = actions.shape[:2]
    if length < state_length + max(HORIZONS):
        raise ValueError("dense alignment requires four future action frames")
    if returns.shape != (batch, length) or return_valid.shape != (batch, length):
        raise ValueError("returns and validity must match the action time axes")
    if ctx_pad.shape != (batch,):
        raise ValueError("ctx_pad must be [B]")
    chunks = torch.stack([actions[:, offset : offset + state_length] for offset in range(1, 5)], dim=2)
    target = returns[:, 1 : state_length + 1]
    state_valid = torch.arange(state_length, device=actions.device)[None] >= ctx_pad[:, None]
    target_valid = return_valid[:, 1 : state_length + 1].bool()
    # The supplied extended window guarantees action availability.  Separate
    # masks are retained because the Q1/Q4 label counts are an audited contract.
    q1_mask = state_valid & target_valid
    q4_mask = state_valid & target_valid
    return DenseAlignment(chunks, target, q1_mask, q4_mask)


class Frozen026Trunk(nn.Module):
    """The exact 026 observation trunk, permanently detached from critic training."""

    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.policy = policy.requires_grad_(False).eval()

    @property
    def codec(self) -> ControllerCodec:
        return self.policy.codec

    def train(self, mode: bool = True):
        super().train(mode)
        self.policy.eval()
        return self

    @torch.no_grad()
    def forward(self, context: Context) -> Tensor:
        history = self.codec.quantize(stack_actions(context.features))
        return self.policy(context.features, context.ctx_pad, history).detach()


class CausalChunkCritic(nn.Module):
    """Position-aware action-sequence Q with Q1/Q4 causal readouts.

    ``condition_on_action=False`` is the exact-shape state-only control.  It runs
    the same parameters and token geometry but replaces every chunk with the
    structured neutral-controller token.
    """

    def __init__(
        self,
        codec: ControllerCodec,
        state_dim: int,
        cfg: ProbeConfig,
        *,
        condition_on_action: bool = True,
    ) -> None:
        super().__init__()
        cfg.validate()
        self.codec = codec
        self.condition_on_action = condition_on_action
        self.state_proj = nn.Linear(state_dim, cfg.action_d_model)
        self.action_proj = nn.Linear(N_GROUPS * codec.embed_dim, cfg.action_d_model)
        self.position = nn.Parameter(torch.empty(4, cfg.action_d_model))
        layer = nn.TransformerEncoderLayer(
            cfg.action_d_model,
            cfg.action_heads,
            cfg.action_ff_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.action_layers, norm=nn.LayerNorm(cfg.action_d_model))
        self.head = nn.Sequential(
            nn.Linear(cfg.action_d_model, cfg.head_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.head_hidden_dim, 1),
        )
        nn.init.normal_(self.position, std=0.02)
        neutral = torch.zeros(1, 4, A_DIM)
        self.register_buffer("neutral_indices", codec.quantize(neutral).squeeze(0))

    def forward(self, state: Tensor, chunks: Tensor) -> Tensor:
        if chunks.shape[-2:] != (4, N_GROUPS) or chunks.shape[:-2] != state.shape[:-1]:
            raise ValueError(f"expected state [...,D] and chunk [...,4,{N_GROUPS}]")
        flat_state = state.reshape(-1, state.shape[-1])
        flat_chunk = chunks.reshape(-1, 4, N_GROUPS)
        if not self.condition_on_action:
            flat_chunk = self.neutral_indices.expand(flat_chunk.shape[0], -1, -1)
        action = self.action_proj(self.codec.embed_frame(flat_chunk)) + self.position
        tokens = torch.cat((self.state_proj(flat_state)[:, None], action), dim=1)
        causal = torch.ones(5, 5, dtype=torch.bool, device=tokens.device).triu(1)
        encoded = self.encoder(tokens, mask=causal)
        values = self.head(encoded[:, (1, 4)]).squeeze(-1).float()
        return values.reshape(*state.shape[:-1], 2)


class ValueCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, state: Tensor) -> Tensor:
        return self.head(state).squeeze(-1).float()


class ProbeMember(nn.Module):
    """One independently seeded V, action Q, and exact-shape null-Q control."""

    def __init__(self, codec: ControllerCodec, state_dim: int, cfg: ProbeConfig, seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.value = ValueCritic(state_dim, cfg.head_hidden_dim)
            self.q = CausalChunkCritic(codec, state_dim, cfg, condition_on_action=True)
            # Pair the control at identical initialization; the only difference
            # at step zero is real versus null action tokens.
            self.control = copy.deepcopy(self.q)
            self.control.condition_on_action = False


class DenseMCProbe(nn.Module):
    """Frozen trunk plus the required three independent critic seeds."""

    def __init__(self, policy026: nn.Module, cfg: ProbeConfig | None = None) -> None:
        super().__init__()
        cfg = ProbeConfig() if cfg is None else cfg
        cfg.validate()
        self.trunk = Frozen026Trunk(policy026)
        state_dim = int(policy026.cfg.d_model)
        self.members = nn.ModuleList(
            [ProbeMember(self.trunk.codec, state_dim, cfg, seed) for seed in cfg.critic_seeds]
        )

    def trainable_parameters(self):
        return (parameter for parameter in self.members.parameters() if parameter.requires_grad)


def masked_huber(prediction: Tensor, target: Tensor, mask: Tensor, *, delta: float = 1.0) -> Tensor:
    """Huber over selected rows only, so invalid NaN targets never enter math."""
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have identical shapes")
    selected_prediction, selected_target = prediction[mask], target[mask]
    if selected_target.numel() == 0:
        raise ValueError("critic batch contains no valid MC labels")
    if not torch.isfinite(selected_target).all():
        raise ValueError("a valid MC label is not finite")
    return F.huber_loss(selected_prediction.float(), selected_target.float(), delta=delta)


def critic_losses(
    member: ProbeMember,
    state: Tensor,
    chunks: Tensor,
    labels: DenseAlignment,
    *,
    delta: float = 1.0,
) -> dict[str, Tensor]:
    q = member.q(state, chunks)
    control = member.control(state, chunks)
    return {
        "v": masked_huber(member.value(state), labels.target, labels.q1_mask, delta=delta),
        "q1": masked_huber(q[..., 0], labels.target, labels.q1_mask, delta=delta),
        "q4": masked_huber(q[..., 1], labels.target, labels.q4_mask, delta=delta),
        "control_q1": masked_huber(control[..., 0], labels.target, labels.q1_mask, delta=delta),
        "control_q4": masked_huber(control[..., 1], labels.target, labels.q4_mask, delta=delta),
    }


def ensemble_statistics(predictions: Tensor) -> tuple[Tensor, Tensor]:
    """Mean estimate and epistemic spread; never a clipped-double minimum."""
    if predictions.ndim < 1 or predictions.shape[0] < 2:
        raise ValueError("predictions must start with an ensemble axis of size >=2")
    return predictions.mean(0), predictions.std(0, unbiased=True)


def calibration_metrics(prediction: Tensor, target: Tensor, *, bins: int = 10) -> dict[str, float]:
    """Equal-count calibration summary for scalar return regression."""
    prediction, target = prediction.detach().float().flatten(), target.detach().float().flatten()
    finite = torch.isfinite(prediction) & torch.isfinite(target)
    prediction, target = prediction[finite], target[finite]
    if prediction.numel() < bins or bins < 1:
        raise ValueError("calibration needs at least one sample per bin")
    order = prediction.argsort()
    errors, counts = [], []
    for indices in torch.tensor_split(order, bins):
        if indices.numel():
            errors.append((prediction[indices].mean() - target[indices].mean()).square())
            counts.append(indices.numel())
    squared = torch.stack(errors)
    weights = prediction.new_tensor(counts) / prediction.numel()
    return {
        "calibration_rmse": float((squared * weights).sum().sqrt()),
        "mae": float((prediction - target).abs().mean()),
        "bias": float((prediction - target).mean()),
    }


def action_sensitivity(
    logged_prediction: Tensor,
    replacement_prediction: Tensor,
    shuffled_prediction: Tensor,
    target: Tensor,
) -> dict[str, float]:
    """Metrics used by the replacement and temporal-order gate."""

    def mae(value: Tensor) -> float:
        return float((value.float() - target.float()).abs().mean())

    logged = mae(logged_prediction)
    return {
        "logged_mae": logged,
        "replacement_mae": mae(replacement_prediction),
        "shuffled_mae": mae(shuffled_prediction),
        "replacement_degradation": mae(replacement_prediction) - logged,
        "shuffle_degradation": mae(shuffled_prediction) - logged,
        "replacement_prediction_delta": float((replacement_prediction - logged_prediction).abs().mean()),
        "shuffle_prediction_delta": float((shuffled_prediction - logged_prediction).abs().mean()),
    }


def support_stratified_derangement(chunks: Tensor) -> Tensor:
    """Replace chunks within coarse logged-action support, never by the same row."""
    if chunks.ndim != 3 or chunks.shape[1:] != (4, N_GROUPS) or chunks.shape[0] < 2:
        raise ValueError("derangement needs chunks [N,4,4] with N >= 2")
    # First-frame buttons are the most discrete/support-sensitive controller
    # factor.  Keep replacements within that support when at least two exist;
    # singleton strata fall back to the full logged set, still never off-dataset.
    buttons = chunks[:, 0, _exp026.BUTTONS_G].tolist()
    groups: dict[int, list[int]] = {}
    for index, button in enumerate(buttons):
        groups.setdefault(int(button), []).append(index)
    replacement = torch.empty(chunks.shape[0], dtype=torch.long, device=chunks.device)
    all_indices = list(range(chunks.shape[0]))
    for indices in groups.values():
        if len(indices) > 1:
            for position, index in enumerate(indices):
                replacement[index] = indices[(position + 1) % len(indices)]
        else:
            index = indices[0]
            replacement[index] = all_indices[(index + 1) % len(all_indices)]
    if torch.any(replacement == torch.arange(chunks.shape[0], device=chunks.device)):
        raise AssertionError("replacement mapping is not a derangement")
    return chunks[replacement]


def replay_blocked_lcb(
    logged_error: Tensor,
    ablated_error: Tensor,
    replay_ids: tuple[str, ...] | list[str],
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> float:
    """2.5% bootstrap bound for ablation degradation, resampling replays."""
    logged = logged_error.detach().double().flatten()
    ablated = ablated_error.detach().double().flatten()
    if logged.shape != ablated.shape or len(replay_ids) != logged.numel():
        raise ValueError("errors and replay ids must describe the same rows")
    unique = sorted(set(replay_ids))
    if len(unique) < 2:
        raise ValueError("replay-blocked uncertainty needs at least two replays")
    effects = torch.stack(
        [
            (
                ablated[torch.tensor([key == replay for key in replay_ids])]
                - logged[torch.tensor([key == replay for key in replay_ids])]
            ).mean()
            for replay in unique
        ]
    )
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(len(unique), (resamples, len(unique)), generator=generator)
    bootstrap = effects.cpu()[draws].mean(-1)
    return float(torch.quantile(bootstrap, 0.025))


def normalized_ess(weights: Tensor) -> float:
    values = weights.detach().double().flatten()
    if values.numel() == 0:
        return float("nan")
    return float(values.sum().square() / values.square().sum().clamp_min(1e-24) / values.numel())


def audit_advantage_weights(
    advantage: Tensor,
    replay_ids: list[str] | tuple[str, ...],
    *,
    betas: tuple[float, ...] = (0.2, 0.4, 0.8, 1.6, 3.2),
    raw_cap: float = 5.0,
) -> list[dict[str, float | bool]]:
    """Audit normalized AWR weights at both frame and replay statistical units."""
    values = advantage.detach().float().flatten()
    if len(replay_ids) != values.numel():
        raise ValueError("one replay id is required per advantage row")
    unique = sorted(set(replay_ids))
    rows: list[dict[str, float | bool]] = []
    for beta in betas:
        if beta <= 0:
            raise ValueError("beta must be positive")
        raw = torch.exp((values / beta).clamp(max=math.log(raw_cap)))
        clipped = values / beta >= math.log(raw_cap)
        normalized = raw / raw.mean().clamp_min(1e-12)
        replay_mass = torch.stack(
            [
                normalized[torch.tensor([key == replay for key in replay_ids], device=normalized.device)].mean()
                for replay in unique
            ]
        )
        frame_ess = normalized_ess(normalized)
        replay_ess = normalized_ess(replay_mass)
        clip_fraction = float(clipped.float().mean())
        rows.append(
            {
                "beta": beta,
                "frame_ess": frame_ess,
                "replay_ess": replay_ess,
                "clip_fraction": clip_fraction,
                "passes": frame_ess >= 0.2 and replay_ess >= 0.2 and clip_fraction <= 0.2,
            }
        )
    return rows


def action_conditioning_gate(
    seed_metrics: list[dict[str, float]],
    *,
    max_calibration_rmse: float,
    max_sampled_spread: float,
) -> bool:
    """Conservative all-seed promotion gate for action-conditioned Q."""
    if len(seed_metrics) != 3:
        raise ValueError("the action-conditioning gate requires exactly three seeds")
    required = {
        "q1_mae",
        "control_q1_mae",
        "q4_mae",
        "control_q4_mae",
        "q1_replacement_lcb",
        "q4_replacement_lcb",
        "shuffle_lcb",
        "sensitivity_effect_floor",
        "calibration_rmse",
        "policy_sampled_spread",
    }
    for metrics in seed_metrics:
        missing = required - metrics.keys()
        if missing:
            raise ValueError(f"gate metrics missing {sorted(missing)}")
        if not (
            metrics["q1_mae"] < metrics["control_q1_mae"]
            and metrics["q4_mae"] < metrics["control_q4_mae"]
            and metrics["q1_replacement_lcb"] > metrics["sensitivity_effect_floor"]
            and metrics["q4_replacement_lcb"] > metrics["sensitivity_effect_floor"]
            and metrics["shuffle_lcb"] > metrics["sensitivity_effect_floor"]
            and metrics["calibration_rmse"] <= max_calibration_rmse
            and metrics["policy_sampled_spread"] <= max_sampled_spread
        ):
            return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_026_model(path: Path, device: torch.device) -> tuple[nn.Module, object, dict]:
    """Load model/config without resolving the checkpoint's stale data path."""
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = _exp026.config_from_state(state["cfg"])
    _exp026.validate_config(cfg)
    model = _exp026.GPT(cfg).to(device)
    model.load_state_dict(state["model"])
    return model.eval(), cfg, state


def _load_feature_split(
    payload: dict,
    split: str,
    codec: ControllerCodec,
    device: torch.device,
    *,
    require_policy_chunks: bool = False,
) -> dict:
    if split not in payload or not isinstance(payload[split], dict):
        raise ValueError(f"feature artifact is missing {split!r}")
    source = payload[split]
    required = {"state", "chunks", "target", "q1_mask", "q4_mask", "replay_ids"}
    missing = required - source.keys()
    if missing:
        raise ValueError(f"{split} feature split is missing {sorted(missing)}")
    state = torch.as_tensor(source["state"], device=device).float()
    chunks = torch.as_tensor(source["chunks"], device=device)
    if chunks.shape[-1] == A_DIM:
        chunks = codec.quantize(chunks.float())
    chunks = chunks.long()
    target = torch.as_tensor(source["target"], device=device).float()
    q1_mask = torch.as_tensor(source["q1_mask"], device=device).bool()
    q4_mask = torch.as_tensor(source["q4_mask"], device=device).bool()
    replay_ids = tuple(str(value) for value in source["replay_ids"])
    count = state.shape[0]
    if chunks.shape != (count, 4, N_GROUPS) or target.shape != (count,):
        raise ValueError(f"{split} tensors must be state [N,D], chunks [N,4,4], target [N]")
    if q1_mask.shape != (count,) or q4_mask.shape != (count,) or len(replay_ids) != count:
        raise ValueError(f"{split} masks/replay_ids do not match N={count}")
    if not torch.isfinite(target[q1_mask | q4_mask]).all():
        raise ValueError(f"{split} contains a non-finite valid target")
    out = dict(
        state=state,
        chunks=chunks,
        target=target,
        q1_mask=q1_mask,
        q4_mask=q4_mask,
        replay_ids=replay_ids,
    )
    if "policy_chunks" in source:
        policy_chunks = torch.as_tensor(source["policy_chunks"], device=device)
        if policy_chunks.shape[-1] == A_DIM:
            policy_chunks = codec.quantize(policy_chunks.float())
        if policy_chunks.shape != (count, 4, N_GROUPS):
            raise ValueError(f"{split} policy_chunks must be [N,4,4]")
        out["policy_chunks"] = policy_chunks.long()
    elif require_policy_chunks:
        raise ValueError("validation split needs policy_chunks for the ensemble-agreement gate")
    return out


def _train_member(
    member: ProbeMember,
    data: dict,
    cfg: ProbeConfig,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> None:
    parameters = [parameter for parameter in member.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    generator = torch.Generator(device=data["state"].device).manual_seed(seed)
    valid_rows = data["q1_mask"] | data["q4_mask"]
    indices = valid_rows.nonzero(as_tuple=False).flatten()
    if indices.numel() == 0:
        raise ValueError("training split has no valid critic rows")
    member.train()
    for _ in range(steps):
        draw = indices[torch.randint(indices.numel(), (batch_size,), generator=generator, device=indices.device)]
        labels = DenseAlignment(
            data["chunks"][draw],
            data["target"][draw],
            data["q1_mask"][draw],
            data["q4_mask"][draw],
        )
        losses = critic_losses(
            member,
            data["state"][draw].detach(),
            data["chunks"][draw],
            labels,
            delta=cfg.huber_delta,
        )
        optimizer.zero_grad(set_to_none=True)
        sum(losses.values()).backward()
        optimizer.step()


@torch.no_grad()
def _evaluate_members(
    members: nn.ModuleList,
    data: dict,
    *,
    sensitivity_effect_floor: float,
    bootstrap_resamples: int,
) -> tuple[list[dict[str, float]], dict[str, Tensor]]:
    state, chunks, target = data["state"], data["chunks"], data["target"]
    mask = data["q4_mask"]
    if not mask.any():
        raise ValueError("validation split has no Q4 labels")
    logged, controls, replacements, shuffles, values, policy_predictions = [], [], [], [], [], []
    replacement = support_stratified_derangement(chunks)
    shuffled = chunks.flip(1)
    per_seed: list[dict[str, float]] = []
    for member in members:
        member.eval()
        q = member.q(state, chunks)
        control = member.control(state, chunks)
        replaced = member.q(state, replacement)
        shuffled_q = member.q(state, shuffled)
        value = member.value(state)
        policy_q = member.q(state, data["policy_chunks"])
        q1_mask = data["q1_mask"]
        q1_replays = tuple(key for key, keep in zip(data["replay_ids"], q1_mask.tolist(), strict=True) if keep)
        q4_replays = tuple(key for key, keep in zip(data["replay_ids"], mask.tolist(), strict=True) if keep)
        q1, c1, r1, y1 = (tensor[q1_mask] for tensor in (q[:, 0], control[:, 0], replaced[:, 0], target))
        q4, c4, r4, s4, v4, y = (
            tensor[mask] for tensor in (q[:, 1], control[:, 1], replaced[:, 1], shuffled_q[:, 1], value, target)
        )
        sensitivity = action_sensitivity(q4, r4, s4, y)
        calibration_q1 = calibration_metrics(q1, y1, bins=min(10, y1.numel()))
        calibration_q4 = calibration_metrics(q4, y, bins=min(10, y.numel()))
        per_seed.append(
            {
                "q1_mae": float((q1 - y1).abs().mean()),
                "control_q1_mae": float((c1 - y1).abs().mean()),
                "q4_mae": sensitivity["logged_mae"],
                "control_q4_mae": float((c4 - y).abs().mean()),
                "q1_replacement_degradation": float((r1 - y1).abs().mean() - (q1 - y1).abs().mean()),
                "q4_replacement_degradation": sensitivity["replacement_degradation"],
                "q1_replacement_lcb": replay_blocked_lcb(
                    (q1 - y1).abs(), (r1 - y1).abs(), q1_replays, resamples=bootstrap_resamples
                ),
                "q4_replacement_lcb": replay_blocked_lcb(
                    (q4 - y).abs(), (r4 - y).abs(), q4_replays, resamples=bootstrap_resamples
                ),
                "shuffle_lcb": replay_blocked_lcb(
                    (q4 - y).abs(), (s4 - y).abs(), q4_replays, resamples=bootstrap_resamples
                ),
                "sensitivity_effect_floor": sensitivity_effect_floor,
                **sensitivity,
                "calibration_rmse": max(calibration_q1["calibration_rmse"], calibration_q4["calibration_rmse"]),
                "q1_calibration_rmse": calibration_q1["calibration_rmse"],
                "q4_calibration_rmse": calibration_q4["calibration_rmse"],
            }
        )
        logged.append(q4)
        controls.append(c4)
        replacements.append(r4)
        shuffles.append(s4)
        values.append(v4)
        policy_predictions.append(policy_q)
    tensors = {
        "q": torch.stack(logged),
        "control": torch.stack(controls),
        "replacement": torch.stack(replacements),
        "shuffle": torch.stack(shuffles),
        "value": torch.stack(values),
        "target": target[mask],
        "policy_q": torch.stack(policy_predictions),
    }
    _, policy_spread = ensemble_statistics(tensors["policy_q"])
    for metrics in per_seed:
        metrics["policy_sampled_spread"] = float(policy_spread.mean())
    return per_seed, tensors


@torch.no_grad()
def build_feature_artifact(args: argparse.Namespace) -> dict:
    """Materialize dense frozen-026 states and MC labels from full replay rows."""
    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint)
    checkpoint_sha = _sha256(checkpoint)
    policy, policy_cfg, checkpoint_state = load_026_model(checkpoint, device)
    policy.eval()
    trunk = Frozen026Trunk(policy).to(device)
    data_root = args.data_root or policy_cfg.data_root
    stats_path = Path(data_root) / "stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"{stats_path} is required to reproduce 026 preprocessing; mount/materialize the policy MDS first"
        )
    stats = load_consolidated_stats(stats_path)
    if policy_cfg.observation_bundle == "base":
        projection = FeatureProjection(
            columns=BASE_ACTION_PROJECTION.columns | {f"ego_{MC_RETURN_SUFFIX}", f"ego_{MC_VALID_SUFFIX}"},
            derive_spatial=BASE_ACTION_PROJECTION.derive_spatial,
        )
        extra = None
    elif policy_cfg.observation_bundle == "v6_lean":
        projection, extra = None, V6_PLAYER_COLUMNS
    else:
        raise ValueError(f"unsupported 026 observation bundle {policy_cfg.observation_bundle!r}")
    result: dict[str, object] = {
        "schema_version": 1,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": int(checkpoint_state.get("step", -1)),
        "experiment_source_sha256": _sha256(Path(__file__)),
        "alignment": "state s_t, chunk a_{t+1:t+4}, target G_{t+1}",
        "return_config": {
            "gamma": ProbeConfig().gamma,
            "damage_shaping": args.damage_shaping,
            "win_reward": args.win_reward,
            "truncation_policy": "explicit mc_terminated else observed final stock-zero; otherwise masked",
        },
        "extraction_config": {
            "data_root": str(data_root),
            "remote": args.remote,
            "context_length": policy_cfg.L_ctx,
            "batch_size": args.extract_batch_size,
            "max_batches": args.max_extract_batches,
            "seed": args.extract_seed,
            "ego_orientations": ["p1", "p2"],
            "tiling": "deterministic exhaustive non-overlapping state blocks",
        },
        "split_policy": "existing MDS train/val directories; replay IDs checked disjoint after extraction",
    }
    for split in ("train", "val"):
        dataset = StreamingDataset(
            remote=f"{args.remote}/{split}" if args.remote else None,
            local=str(Path(data_root) / split),
            batch_size=1,
            shuffle=False,
            cache_limit=args.cache_limit if args.remote else None,
            predownload=512 if args.remote else None,
        )
        collected: dict[str, list] = {
            "state": [],
            "chunks": [],
            "target": [],
            "q1_mask": [],
            "q4_mask": [],
            "policy_chunks": [],
            "replay_ids": [],
        }
        expected_keys: set[tuple[str, int, int]] = set()
        emitted_keys: set[tuple[str, int, int]] = set()
        seen_replays = seen_batches = accepted_batches = valid_rows = 0
        generator = torch.Generator(device=device).manual_seed(args.extract_seed + (split == "val"))
        stop_early = False
        for compact in dataset:
            seen_replays += 1
            replay_id = str(compact["replay_id"])
            check_schema_version(
                {"schema_version": int(compact["source_schema_version"])},
                expected=policy_cfg.mds_schema_version,
            )
            sample = decode_policy_replay(compact)
            sample = label_mc_replay(
                sample,
                gamma=ProbeConfig().gamma,
                damage_shaping=args.damage_shaping,
                win_reward=args.win_reward,
            )
            windows = exhaustive_replay_windows(
                sample,
                replay_id=replay_id,
                context_length=policy_cfg.L_ctx,
                projection=projection,
            )
            for start in range(0, len(windows), args.extract_batch_size):
                if args.max_extract_batches and accepted_batches >= args.max_extract_batches:
                    stop_early = True
                    break
                source_windows = windows[start : start + args.extract_batch_size]
                seen_batches += 1
                normal = collate_train_batch(
                    source_windows,
                    stats=stats,
                    L_ctx=policy_cfg.L_ctx,
                    extra=extra,
                    projection=projection,
                )
                replay_ids = tuple(str(window["mc_replay_id"]) for window in source_windows)
                normal = TrainBatch(normal.context, normal.target, replay_ids)
                cpu_batch = collate_extraction_batch(source_windows, normal)
                batch = cpu_batch.batch.to(device)
                returns = cpu_batch.returns.to(device)
                return_valid = cpu_batch.return_valid.to(device)
                state_indices = cpu_batch.state_indices.to(device)
                full_actions = torch.cat((stack_actions(batch.context.features), batch.target[:, :4]), dim=1)
                aligned = align_dense_prefixes(
                    full_actions,
                    returns,
                    return_valid,
                    batch.context.ctx_pad,
                    state_length=policy_cfg.L_ctx,
                )
                keep = (aligned.q1_mask | aligned.q4_mask) & (state_indices >= 0)
                for row, window in enumerate(source_windows):
                    if bool(return_valid[row].any()):
                        ego_id = int(cpu_batch.ego_ids[row])
                        expected_keys.update(
                            (replay_id, ego_id, int(index))
                            for index in cpu_batch.state_indices[row].tolist()
                            if index >= 0
                        )
                count = int(keep.sum())
                if count == 0:
                    continue
                accepted_batches += 1
                hidden = trunk(batch.context)
                categorical_chunks = trunk.codec.quantize(aligned.chunks)
                history = trunk.codec.quantize(stack_actions(batch.context.features))
                selected_hidden = hidden[keep]
                selected_observed = history[keep]
                policy_chunks = policy.temporal.sample_indices(
                    selected_hidden[:, None],
                    selected_observed,
                    tuple(policy.head_offsets[:4]),
                    argmax=False,
                    gen=generator,
                )
                for name, tensor in (
                    ("state", selected_hidden),
                    ("chunks", categorical_chunks[keep]),
                    ("target", aligned.target[keep]),
                    ("q1_mask", aligned.q1_mask[keep]),
                    ("q4_mask", aligned.q4_mask[keep]),
                    ("policy_chunks", policy_chunks),
                ):
                    collected[name].append(tensor.cpu())
                for row, window in enumerate(source_windows):
                    row_keep = keep[row].cpu()
                    ego_id = int(cpu_batch.ego_ids[row])
                    selected_indices = cpu_batch.state_indices[row][row_keep].tolist()
                    keys = [(str(window["mc_replay_id"]), ego_id, int(index)) for index in selected_indices]
                    duplicate = emitted_keys.intersection(keys)
                    if duplicate:
                        raise RuntimeError(f"duplicate dense state keys, first={next(iter(duplicate))}")
                    emitted_keys.update(keys)
                    collected["replay_ids"].extend([str(window["mc_replay_id"])] * len(keys))
                valid_rows += count
            if stop_early:
                break
        if expected_keys != emitted_keys:
            missing = expected_keys - emitted_keys
            extra_keys = emitted_keys - expected_keys
            raise RuntimeError(f"dense-prefix audit failed: {len(missing)} missing, {len(extra_keys)} unexpected")
        if not valid_rows:
            raise RuntimeError(
                f"{split} produced no valid MC rows. Compact policy MDS has no end-block metadata; "
                "if final stock-zero is absent, rematerialize it with scalar mc_terminated rather than "
                "treating truncated tails as zero."
            )
        result[split] = {
            name: tuple(values) if name == "replay_ids" else torch.cat(values) for name, values in collected.items()
        }
        result[f"{split}_summary"] = {
            "scanned_batches": seen_batches,
            "accepted_batches": accepted_batches,
            "dense_valid_rows": valid_rows,
            "unique_replays": len(set(collected["replay_ids"])),
            "expected_dense_keys": len(expected_keys),
            "emitted_dense_keys": len(emitted_keys),
            "exhaustive": not bool(args.max_extract_batches),
            "scanned_replays": seen_replays,
            "replay_ids_sha256": hashlib.sha256("\n".join(sorted(set(collected["replay_ids"]))).encode()).hexdigest(),
        }
    overlap = set(result["train"]["replay_ids"]) & set(result["val"]["replay_ids"])
    if overlap:
        raise RuntimeError(f"source train/val splits leak {len(overlap)} replay id(s)")
    output = Path(args.features)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    print(json.dumps({key: value for key, value in result.items() if key.endswith("_summary")}, indent=2))
    return result


def run_probe(args: argparse.Namespace) -> dict:
    """Train/evaluate the probe from a checkpoint-keyed frozen-feature artifact.

    Artifact schema (``torch.save``): ``checkpoint_sha256`` and ``train``/``val``
    mappings containing state ``[N,d026]``, categorical or raw chunks ``[N,4,*]``,
    target, Q1/Q4 masks, and one replay id per row.  States must have been emitted
    by :class:`Frozen026Trunk`; the hash guard prevents accidental trunk drift.
    """
    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint)
    actual_sha = _sha256(checkpoint)
    artifact = torch.load(args.features, map_location="cpu", weights_only=False)
    if artifact.get("checkpoint_sha256") != actual_sha:
        raise ValueError("feature artifact checkpoint_sha256 does not match --checkpoint")
    policy, _policy_cfg, checkpoint_state = load_026_model(checkpoint, device)
    cfg = ProbeConfig()
    probe = DenseMCProbe(policy, cfg).to(device)
    train_data = _load_feature_split(artifact, "train", probe.trunk.codec, device)
    val_data = _load_feature_split(artifact, "val", probe.trunk.codec, device, require_policy_chunks=True)
    expected_dim = int(policy.cfg.d_model)
    if train_data["state"].shape[-1] != expected_dim or val_data["state"].shape[-1] != expected_dim:
        raise ValueError(f"frozen state width must equal checkpoint d_model={expected_dim}")
    overlap = set(train_data["replay_ids"]) & set(val_data["replay_ids"])
    if overlap:
        raise ValueError(f"train/val feature splits leak {len(overlap)} replay id(s)")
    for member, seed in zip(probe.members, cfg.critic_seeds, strict=True):
        _train_member(
            member,
            train_data,
            cfg,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=seed,
        )
    seed_metrics, tensors = _evaluate_members(
        probe.members,
        val_data,
        sensitivity_effect_floor=args.sensitivity_effect_floor,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    q_mean, q_spread = ensemble_statistics(tensors["q"])
    v_mean, _ = ensemble_statistics(tensors["value"])
    replay_ids = [key for key, keep in zip(val_data["replay_ids"], val_data["q4_mask"].tolist(), strict=True) if keep]
    audits = {
        "g_minus_v": audit_advantage_weights(tensors["target"] - v_mean, replay_ids),
        "q_minus_v": audit_advantage_weights(q_mean - v_mean, replay_ids),
    }
    gate = action_conditioning_gate(
        seed_metrics,
        max_calibration_rmse=args.max_calibration_rmse,
        max_sampled_spread=args.max_sampled_spread,
    )
    result = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_sha,
        "checkpoint_step": int(checkpoint_state.get("step", -1)),
        "artifact": str(Path(args.features)),
        "artifact_sha256": _sha256(Path(args.features)),
        "experiment_source_sha256": _sha256(Path(__file__)),
        "critic_seeds": list(cfg.critic_seeds),
        "probe_config": asdict(cfg),
        "training_config": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "bootstrap_resamples": args.bootstrap_resamples,
            "sensitivity_effect_floor": args.sensitivity_effect_floor,
            "max_calibration_rmse": args.max_calibration_rmse,
            "max_sampled_spread": args.max_sampled_spread,
        },
        "seed_metrics": seed_metrics,
        "ensemble_spread_mean": float(q_spread.mean()),
        "weight_audits": audits,
        "action_conditioning_gate": gate,
        "recommended_advantage": "mean(Q)-V" if gate else "G-V",
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    torch.save(
        {"members": probe.members.state_dict(), "config": cfg, "checkpoint_sha256": actual_sha}, output / "critics.pt"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="exact experiment-026 checkpoint")
    parser.add_argument("--features", required=True, help="checkpoint-keyed frozen dense-feature .pt artifact")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-features", action="store_true", help="create --features from replay MDS, then exit")
    mode.add_argument(
        "--build-and-run",
        action="store_true",
        help="create --features and immediately train/evaluate critics in this process",
    )
    parser.add_argument("--output", default="runs/031_mc_critic_probe")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-calibration-rmse", type=float, default=0.25)
    parser.add_argument("--max-sampled-spread", type=float, default=0.25)
    parser.add_argument("--sensitivity-effect-floor", type=float, default=0.005)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--data-root", help="policy MDS root; defaults to checkpoint config")
    parser.add_argument("--remote", help="optional remote MDS root")
    parser.add_argument("--cache-limit", default="128gb")
    parser.add_argument("--extract-batch-size", type=int, default=16)
    parser.add_argument("--extract-seed", type=int, default=0)
    parser.add_argument("--max-extract-batches", type=int, default=0, help="0 consumes the full split")
    parser.add_argument("--damage-shaping", type=float, default=0.01)
    parser.add_argument("--win-reward", type=float, default=0.5)
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if args.build_features:
        build_feature_artifact(args)
    elif args.build_and_run:
        build_feature_artifact(args)
        run_probe(args)
    else:
        run_probe(args)


if __name__ == "__main__":
    main()
