"""Fixed-shape joint PPO and direct KL on recorded window decisions."""

from dataclasses import dataclass

import torch
from torch import Tensor

from hal.models.window_replay import ReplayOutput


@dataclass(frozen=True, slots=True)
class ActorTargets:
    """Store behavior selected-action logs and reference full-head logs, all in nats.

    Heads use controller decode order. The denominator is the global selected
    decision count divided by the number of gradient-averaging ranks.
    The loss detaches every target.
    """

    behavior_log_probs_WKF: Tensor
    advantages_WK: Tensor
    selected_WK: Tensor
    kl_coefficient_WK: Tensor
    reference_log_distributions: tuple[Tensor, ...]
    advantage_mean: Tensor
    advantage_sigma: Tensor
    denominator: Tensor


@dataclass(frozen=True, slots=True)
class ActorLoss:
    loss: Tensor
    policy_loss: Tensor
    kl_loss: Tensor
    joint_ratio_WK: Tensor
    factor_kl_WKF: Tensor


def validate_actor_targets(output: ReplayOutput, targets: ActorTargets) -> None:
    """Check structure and active values; an empty rank provides no numerical coverage."""
    shape = tuple(output.log_probs_WKF.shape[:2])
    device = output.log_probs_WKF.device
    tensors = (
        (targets.behavior_log_probs_WKF, (*shape, 16), torch.float32),
        (targets.advantages_WK, shape, torch.float32),
        (targets.selected_WK, shape, torch.bool),
        (targets.kl_coefficient_WK, shape, torch.float32),
        (targets.advantage_mean, (), torch.float32),
        (targets.advantage_sigma, (), torch.float32),
        (targets.denominator, (), torch.float32),
    )
    for value, expected, dtype in tensors:
        if tuple(value.shape) != expected or value.dtype != dtype or value.device != device:
            raise ValueError("actor target shape, dtype or device mismatch")
    selected = targets.selected_WK
    if bool((selected & ~output.factor_mask_WKF.any(-1)).any()):
        raise ValueError("selected actor decisions need executed factors")
    if not bool(torch.isfinite(targets.advantage_mean)) or not bool(
        torch.isfinite(targets.advantage_sigma) & (targets.advantage_sigma > 0)
    ):
        raise ValueError("advantage statistics must be finite with positive sigma")
    if not bool(torch.isfinite(targets.denominator) & (targets.denominator > 0)):
        raise ValueError("actor denominator must be finite and positive")
    for value in (targets.advantages_WK, targets.kl_coefficient_WK):
        if not bool(torch.isfinite(value[selected]).all()):
            raise ValueError("selected actor targets must be finite")
    if bool((targets.kl_coefficient_WK[selected] < 0).any()):
        raise ValueError("KL coefficients must be nonnegative")
    executed = output.factor_mask_WKF & selected[..., None]
    for logs in (targets.behavior_log_probs_WKF[executed], output.log_probs_WKF[executed]):
        if not bool((torch.isfinite(logs) & (logs <= 0)).all()):
            raise ValueError("executed log-probs must be finite and nonpositive")
    if len(targets.reference_log_distributions) != 4:
        raise ValueError("reference needs four controller distributions")
    frame_mask = executed.reshape(*shape, 4, 4)
    for group, (policy_head_logs, reference_head_logs) in enumerate(
        zip(output.log_distributions, targets.reference_log_distributions, strict=True)
    ):
        if (
            reference_head_logs.shape != policy_head_logs.shape
            or reference_head_logs.dtype != torch.float32
            or reference_head_logs.device != device
        ):
            raise ValueError("reference distribution shape, dtype or device mismatch")
        policy_head_logs = policy_head_logs[frame_mask[..., group]]
        reference_head_logs = reference_head_logs[frame_mask[..., group]]
        for name, head_logs in (("policy", policy_head_logs), ("reference", reference_head_logs)):
            if bool((torch.isnan(head_logs) | torch.isposinf(head_logs)).any()) or not bool(
                torch.isclose(head_logs.logsumexp(-1), torch.zeros_like(head_logs[..., 0]), atol=1e-5).all()
            ):
                raise ValueError(f"{name} full-head log-probs must be normalized")
        if bool((torch.isfinite(policy_head_logs) != torch.isfinite(reference_head_logs)).any()):
            raise ValueError("policy and reference full-head log-probs must have identical support")


def actor_loss(output: ReplayOutput, targets: ActorTargets, *, clip_epsilon: float = 0.2) -> ActorLoss:
    """Calculate joint PPO and direct KL from stored log probabilities.

    Validate targets before capture. Detach all targets so gradients flow only
    through the current policy.
    """
    if not 0 < clip_epsilon < 1:
        raise ValueError("clip epsilon must be finite and between zero and one")
    selected = targets.selected_WK & output.factor_mask_WKF.any(-1)
    mask = output.factor_mask_WKF & selected[..., None]
    action_log_probs = torch.where(mask, output.log_probs_WKF, 0)
    behavior_action_log_probs = torch.where(mask, targets.behavior_log_probs_WKF.detach(), 0)
    joint_log_ratio = (action_log_probs - behavior_action_log_probs).sum(-1)
    joint_ratio = joint_log_ratio.exp()
    advantages = torch.where(selected, targets.advantages_WK.detach(), 0)
    normalized = (advantages - targets.advantage_mean.detach()) / targets.advantage_sigma.detach()
    surrogate = torch.minimum(
        joint_ratio * normalized, joint_ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * normalized
    )
    policy = torch.where(selected, -surrogate, 0).sum() / targets.denominator.detach()
    kls: list[Tensor] = []
    frame_mask = mask.reshape(*mask.shape[:2], 4, 4)
    for group, (policy_head_logs, reference_head_logs) in enumerate(
        zip(output.log_distributions, targets.reference_log_distributions, strict=True)
    ):
        active = frame_mask[..., group, None]
        legal = active & torch.isfinite(policy_head_logs)
        policy_head_logs = torch.where(legal, policy_head_logs, 0)
        reference_head_logs = torch.where(legal, reference_head_logs.detach(), 0)
        policy_head_probs = torch.where(legal, policy_head_logs.exp(), 0)
        kls.append((policy_head_probs * (policy_head_logs - reference_head_logs)).sum(-1))
    factor_kl = torch.stack(kls, -1).flatten(-2)
    kl_sum = torch.where(mask, factor_kl, 0).sum(-1)
    coefficient = torch.where(selected, targets.kl_coefficient_WK.detach(), 0)
    kl = (coefficient / targets.advantage_sigma.detach() * kl_sum).sum() / targets.denominator.detach()
    return ActorLoss(policy + kl, policy, kl, joint_ratio, torch.where(mask, factor_kl, 0))
