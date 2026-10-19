"""Replay fixed decision slots with one history memory per window."""

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, slots=True)
class ReplaySlots:
    """W windows, K slots, L history frames, D forced frames, Q opponent frames.

    Controller tensors use codec order (buttons, main stick, C-stick, triggers).
    F=16 uses frame-major decode order (C-stick, main stick, triggers, buttons).
    Conditions are normalized [log lambda, delay, replan]. Invalid slots and
    unexecuted factors never enter a loss reduction. All indices are int64,
    masks bool, and conditions float32. Validate once before compiled replay.
    """

    positions_WK: Tensor
    valid_WK: Tensor
    history_starts_WL: Tensor
    forced_queue_WKD4: Tensor
    delay_WK: Tensor
    replan_WK: Tensor
    condition_WK3: Tensor
    opponent_condition_WK3: Tensor
    opponent_queue_WKQ4: Tensor
    opponent_lengths_WK: Tensor
    sampled_actions_WK44: Tensor
    executed_WKF: Tensor


@dataclass(frozen=True, slots=True)
class ReplayCriticFeatures:
    """Detached critic inputs per [W, K] slot; d/t are trunk/decoder widths."""

    trunk_WKd: Tensor
    decoder_WKt: Tensor
    condition_WK3: Tensor
    opponent_condition_WK3: Tensor
    opponent_queue_WKQ4: Tensor
    opponent_lengths_WK: Tensor


@dataclass(frozen=True, slots=True)
class ReplayOutput:
    """FP32 factor logs [W,K,16], group probabilities [W,K,4,V_group].

    Both outputs use decode order. Log-probs are zero outside executed_WKF and
    valid_WK. Distributions retain the normalized conditional support, including
    unexecuted sampled frames; reductions must use factor_mask_WKF.
    """

    log_probs_WKF: Tensor
    distributions: tuple[Tensor, ...]
    factor_mask_WKF: Tensor
    features: ReplayCriticFeatures
