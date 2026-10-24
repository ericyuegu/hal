"""Fixed-shape joint PPO and direct KL on recorded window decisions."""

from dataclasses import dataclass

import torch
from torch import Tensor

from hal.models.window_replay import ReplayOutput


@dataclass(frozen=True, slots=True)
class ActorTargets:
    """Detach targets and divide local sums by the global decision count per gradient-averaging rank."""

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
    """Validate before capture; an empty local rank needs a positive global count."""
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
    for group, (p, q) in enumerate(zip(output.log_distributions, targets.reference_log_distributions, strict=True)):
        if q.shape != p.shape or q.dtype != torch.float32 or q.device != device:
            raise ValueError("reference distribution shape, dtype or device mismatch")
        p, q = p[frame_mask[..., group]], q[frame_mask[..., group]]
        if (
            bool((torch.isnan(p) | torch.isposinf(p)).any())
            or not bool(torch.isclose(p.logsumexp(-1), torch.zeros_like(p[..., 0]), atol=1e-5).all())
            or bool((torch.isnan(q) | torch.isposinf(q)).any())
            or not bool(torch.isclose(q.logsumexp(-1), torch.zeros_like(q[..., 0]), atol=1e-5).all())
            or bool((torch.isfinite(p) != torch.isfinite(q)).any())
        ):
            raise ValueError("reference distributions must be normalized on identical support")


def actor_loss(output: ReplayOutput, targets: ActorTargets, *, clip_epsilon: float = 0.2) -> ActorLoss:
    """Calculate joint PPO and direct KL from stored log probabilities.

    Validate targets before capture. Detach all targets so gradients flow only
    through the current policy.
    """
    if not 0 < clip_epsilon < 1:
        raise ValueError("clip epsilon must be finite and between zero and one")
    selected = targets.selected_WK & output.factor_mask_WKF.any(-1)
    mask = output.factor_mask_WKF & selected[..., None]
    new = torch.where(mask, output.log_probs_WKF, 0)
    old = torch.where(mask, targets.behavior_log_probs_WKF.detach(), 0)
    ratio = (new - old).sum(-1).exp()
    advantages = torch.where(selected, targets.advantages_WK.detach(), 0)
    normalized = (advantages - targets.advantage_mean.detach()) / targets.advantage_sigma.detach()
    surrogate = torch.minimum(ratio * normalized, ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * normalized)
    policy = torch.where(selected, -surrogate, 0).sum() / targets.denominator.detach()
    kls: list[Tensor] = []
    frame_mask = mask.reshape(*mask.shape[:2], 4, 4)
    for group, (log_probabilities, reference) in enumerate(
        zip(output.log_distributions, targets.reference_log_distributions, strict=True)
    ):
        active = frame_mask[..., group, None]
        legal = active & torch.isfinite(log_probabilities)
        log_p = torch.where(legal, log_probabilities, 0)
        log_q = torch.where(legal, reference.detach(), 0)
        p = torch.where(legal, log_p.exp(), 0)
        kls.append((p * (log_p - log_q)).sum(-1))
    factor_kl = torch.stack(kls, -1).flatten(-2)
    kl_sum = torch.where(mask, factor_kl, 0).sum(-1)
    coefficient = torch.where(selected, targets.kl_coefficient_WK.detach(), 0)
    kl = (coefficient / targets.advantage_sigma.detach() * kl_sum).sum() / targets.denominator.detach()
    return ActorLoss(policy + kl, policy, kl, ratio, torch.where(mask, factor_kl, 0))
