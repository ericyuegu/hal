"""Fixed-shape joint PPO, direct KL and a shared-model value loss on recorded window decisions."""

from dataclasses import dataclass

import torch
from torch import Tensor

from hal.models.self_play import VALUE_BINS
from hal.models.self_play import value_cross_entropy
from hal.models.window_replay import ReplayOutput


@dataclass(frozen=True, slots=True)
class ActorTargets:
    """Detach targets and divide local sums by the global decision count per gradient-averaging rank."""

    behavior_log_probs_WKF: Tensor
    advantages_WK: Tensor
    selected_WK: Tensor
    kl_coefficient_WK: Tensor
    reference_distributions: tuple[Tensor, ...]
    advantage_mean: Tensor
    advantage_sigma: Tensor
    denominator: Tensor
    # Critic targets for the value loss that trains the trunk.
    value_targets_WK: Tensor
    value_coefficient: Tensor


@dataclass(frozen=True, slots=True)
class ActorLoss:
    loss: Tensor
    policy_loss: Tensor
    kl_loss: Tensor
    value_loss: Tensor
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
        (targets.value_targets_WK, shape, torch.float32),
        (targets.value_coefficient, (), torch.float32),
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
    for value in (targets.advantages_WK, targets.kl_coefficient_WK, targets.value_targets_WK):
        if not bool(torch.isfinite(value[selected]).all()):
            raise ValueError("selected actor targets must be finite")
    if not bool(torch.isfinite(targets.value_coefficient) & (targets.value_coefficient >= 0)):
        raise ValueError("value coefficient must be finite and nonnegative")
    if bool((targets.kl_coefficient_WK[selected] < 0).any()):
        raise ValueError("KL coefficients must be nonnegative")
    executed = output.factor_mask_WKF & selected[..., None]
    for logs in (targets.behavior_log_probs_WKF[executed], output.log_probs_WKF[executed]):
        if not bool((torch.isfinite(logs) & (logs <= 0)).all()):
            raise ValueError("executed log-probs must be finite and nonpositive")
    if len(targets.reference_distributions) != 4:
        raise ValueError("reference needs four controller distributions")
    frame_mask = executed.reshape(*shape, 4, 4)
    for group, (p, q) in enumerate(zip(output.distributions, targets.reference_distributions, strict=True)):
        if q.shape != p.shape or q.dtype != torch.float32 or q.device != device:
            raise ValueError("reference distribution shape, dtype or device mismatch")
        p, q = p[frame_mask[..., group]], q[frame_mask[..., group]]
        if (
            not bool((torch.isfinite(p) & (p >= 0)).all())
            or not bool(torch.isclose(p.sum(-1), torch.ones_like(p[..., 0]), atol=1e-5).all())
            or not bool((torch.isfinite(q) & (q >= 0)).all())
            or not bool(torch.isclose(q.sum(-1), torch.ones_like(q[..., 0]), atol=1e-5).all())
            or bool(((p > 0) != (q > 0)).any())
        ):
            raise ValueError("reference distributions must be normalized on identical support")


def actor_loss(
    output: ReplayOutput, value_logits_WKB: Tensor | None, targets: ActorTargets, *, clip_epsilon: float = 0.2
) -> ActorLoss:
    """Calculate joint PPO, direct KL and value cross-entropy for the shared model.

    Validate targets before capture. Detach all targets so gradients flow only
    through the current policy. Callers compute value logits with fixed critic
    parameters and a detached decoder state, so the value term moves only the trunk.
    """
    if not 0 < clip_epsilon < 1:
        raise ValueError("clip epsilon must be finite and between zero and one")
    if value_logits_WKB is not None and value_logits_WKB.shape != (*targets.selected_WK.shape, VALUE_BINS):
        raise ValueError("value logits must be [W,K,value bins]")
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
    for group, (probabilities, reference) in enumerate(
        zip(output.distributions, targets.reference_distributions, strict=True)
    ):
        active = frame_mask[..., group, None]
        p = torch.where(active, probabilities, 0)
        q = torch.where(active, reference.detach(), 1)
        positive = p > 0
        log_p = torch.where(positive, p, 1).log()
        log_q = torch.where(positive, q, 1).log()
        kls.append((p * (log_p - log_q)).sum(-1))
    factor_kl = torch.stack(kls, -1).flatten(-2)
    kl_sum = torch.where(mask, factor_kl, 0).sum(-1)
    coefficient = torch.where(selected, targets.kl_coefficient_WK.detach(), 0)
    kl = (coefficient / targets.advantage_sigma.detach() * kl_sum).sum() / targets.denominator.detach()
    if value_logits_WKB is None:
        value = policy.new_zeros(())
    else:
        cross_entropy = value_cross_entropy(
            value_logits_WKB, torch.where(selected, targets.value_targets_WK.detach(), 0)
        )
        value = torch.where(selected, cross_entropy, 0).sum() / targets.denominator.detach()
    total = policy + kl + targets.value_coefficient.detach() * value
    return ActorLoss(total, policy, kl, value, ratio, torch.where(mask, factor_kl, 0))
