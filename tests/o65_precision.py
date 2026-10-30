"""Score common inputs with the recorded per-decision trainer and an FP32 reference."""

import dataclasses
from typing import Any

import torch
from o65_reference import load_control
from torch import Tensor

from hal.models.attention import KVMemory
from hal.models.replay_actor import ActorTargets
from hal.models.replay_actor import actor_loss
from hal.models.replay_actor import validate_actor_targets
from hal.models.self_play import SelfPlayTransformer
from hal.models.self_play import value_mean
from hal.models.window_replay import ReplayCriticFeatures
from hal.models.window_replay import ReplayOutput
from hal.models.window_replay import ReplaySlots
from hal.models.window_replay import replay_critic_logits

exp = load_control()


@dataclasses.dataclass(frozen=True, slots=True)
class ProxyBatch:
    features: dict[str, Tensor]
    observed: Tensor
    slots: ReplaySlots
    states: Any
    actions: Tensor
    advantages_WK: Tensor
    value_targets_WK: Tensor
    selected_WK: Tensor
    kl_coefficient_WK: Tensor


def old_forward(policy: Any, batch: ProxyBatch, *, grad: bool, fp32: bool = False) -> ReplayOutput:
    with (
        torch.set_grad_enabled(grad),
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=not fp32, cache_enabled=False),
    ):
        result = fp32_evaluate(policy, batch) if fp32 else policy._evaluate(batch.states, batch.actions)
    slots = batch.slots
    shape = tuple(slots.positions_WK.shape)

    def pad(value: Tensor) -> Tensor:
        target = value.new_zeros(*shape, *value.shape[1:])
        target[slots.valid_WK] = value
        return target

    features = ReplayCriticFeatures(
        *(pad(getattr(result.features, field.name)) for field in dataclasses.fields(result.features))
    )
    return ReplayOutput(
        pad(result.log_probs),
        tuple(pad(p) for p in result.distributions),
        slots.executed_WKF & slots.valid_WK[..., None],
        features,
    )


def fp32_evaluate(policy: Any, batch: ProxyBatch) -> Any:
    """Per-decision reference with FP32 dense attention; Flash requires BF16."""
    model = policy.model
    windows = batch.observed.shape[0]
    hidden = model.trunk.forward_dense(
        model.context_tokens(batch.features, batch.observed),
        torch.zeros(windows, dtype=torch.long, device="cuda"),
        force_dense=True,
    )
    key, value = model.temporal.history_attention.project_memory(hidden)
    rows = torch.arange(windows, device="cuda")[:, None].expand_as(batch.slots.valid_WK)[batch.slots.valid_WK]
    positions = batch.slots.positions_WK[batch.slots.valid_WK]
    outputs = []
    for start in range(0, len(rows), 128):
        window, position = rows[start : start + 128], positions[start : start + 128]
        states = batch.states[start : start + 128]
        outputs.append(
            policy._decode(
                states,
                hidden[window, position][:, None],
                batch.observed[window, position],
                torch.full((4, 4, len(window)), 0.5, device="cuda"),
                batch.actions[start : start + 128],
                history=KVMemory(
                    torch.stack((key[window], value[window])),
                    torch.arange(256, device="cuda")[None].expand(len(window), -1),
                    position,
                    256,
                ),
                count=4,
            )
        )
    return exp._concatenate_evaluations(outputs)


def values(model: SelfPlayTransformer, output: ReplayOutput) -> Tensor:
    return value_mean(replay_critic_logits(model, output.features.detached(), fixed_critic=False))


def targets(output: ReplayOutput, reference: ReplayOutput, behavior: Tensor, batch: ProxyBatch) -> ActorTargets:
    selected = batch.selected_WK & output.factor_mask_WKF.any(-1)
    # The control's GAE targets stay fixed across all numerical paths.
    advantages = batch.advantages_WK
    return ActorTargets(
        behavior,
        advantages,
        selected,
        batch.kl_coefficient_WK,
        reference.distributions,
        advantages[selected].mean(),
        advantages[selected].std(unbiased=False),
        selected.sum().float(),
        # Precision qualification covers the policy objective, so the value term is off.
        torch.zeros_like(advantages),
        advantages.new_zeros(()),
    )


def error(candidate: Tensor, reference: Tensor) -> dict[str, float]:
    delta = (candidate.detach().double() - reference.detach().double()).flatten()
    ref = reference.detach().double().flatten()
    return {
        "rmse": float(delta.square().mean().sqrt()),
        "max": float(delta.abs().max()),
        "relative_l2": float(delta.norm() / ref.norm().clamp_min(1e-30)),
    }


def flattened_gradients(loss: Tensor, parameters: tuple[torch.nn.Parameter, ...]) -> Tensor:
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    return torch.cat(
        [
            (torch.zeros_like(p) if g is None else g).detach().flatten()
            for p, g in zip(parameters, gradients, strict=True)
        ]
    )


def quantities(policy: Any, reference: Any, batch: ProxyBatch, behavior: Tensor, *, fp32: bool) -> dict[str, Tensor]:
    output = old_forward(policy, batch, grad=True, fp32=fp32)
    ref_output = old_forward(reference, batch, grad=False, fp32=fp32)
    target = targets(output, ref_output, behavior, batch)
    validate_actor_targets(output, target)
    logits = replay_critic_logits(policy.model, output.features, fixed_critic=True)
    loss = actor_loss(output, logits, target)
    actor_gradient = flattened_gradients(loss.loss, policy.model.actor_parameters())
    diagnostic_kl = actor_loss(output, logits, dataclasses.replace(target, selected_WK=output.factor_mask_WKF.any(-1)))
    mask = output.factor_mask_WKF
    return {
        "log_probs": output.log_probs_WKF[mask].detach(),
        "kl": diagnostic_kl.factor_kl_WKF[mask].detach(),
        "actor_gradients": actor_gradient,
    }
