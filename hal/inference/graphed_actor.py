"""Capture window replay, the actor objective and backward at fixed shapes."""

from collections.abc import Callable
from dataclasses import fields

import torch
from torch import Tensor
from torch import nn

from hal.inference.cuda_graph import CapturedCall
from hal.models.replay_actor import ActorLoss
from hal.models.replay_actor import ActorTargets
from hal.models.replay_actor import actor_loss
from hal.models.replay_actor import validate_actor_targets
from hal.models.window_replay import ReplayCriticFeatures
from hal.models.window_replay import ReplayOutput


def _target_tensors(targets: ActorTargets) -> tuple[Tensor, ...]:
    return (
        targets.behavior_log_probs_WKF,
        targets.advantages_WK,
        targets.selected_WK,
        targets.kl_coefficient_WK,
        *targets.reference_distributions,
        targets.advantage_mean,
        targets.advantage_sigma,
        targets.denominator,
    )


def _targets(tensors: tuple[Tensor, ...]) -> ActorTargets:
    return ActorTargets(
        tensors[0], tensors[1], tensors[2], tensors[3], tensors[4:8], tensors[8], tensors[9], tensors[10]
    )


class GraphedActorStep:
    """Own static inputs and gradient buffers; leave optimizer steps to the caller.

    forward receives a tuple of static tensor inputs and must run the trunk and
    replay_windows(..., grad=True). It must not close over differentiable trunk
    outputs computed before capture. Compile replay_windows in default mode with
    triton.cudagraphs=False and emulate_precision_casts=True, then call it from
    forward. Keep the trunk eager inside this capture: whole-trunk compilation
    failed the proxy precision gate on Torch 2.11. CUDA flex attention requires
    compilation, and BF16 replay needs cast preservation. Parameters and input storage keep
    their identity for this helper's lifetime. Call backward once per step;
    gradient clipping and Muon run afterwards. Outputs are cloned before return.
    Validate each new payload at its assembly boundary before backward.
    Release earlier eager autograd graphs before constructing this helper:
    Torch's retained leaf gradient nodes carry their original CUDA stream.
    """

    def __init__(
        self,
        forward: Callable[[tuple[Tensor, ...]], ReplayOutput],
        inputs: tuple[Tensor, ...],
        targets: ActorTargets,
        parameters: tuple[nn.Parameter, ...],
        *,
        clip_epsilon: float = 0.2,
    ) -> None:
        if not 0 < clip_epsilon < 1:
            raise ValueError("clip epsilon must be finite and between zero and one")
        if torch.is_inference_mode_enabled():
            raise ValueError("actor capture requires autograd outside inference_mode")
        if not inputs or not inputs[0].is_cuda or not parameters:
            raise ValueError("actor capture requires CUDA inputs and actor parameters")
        if len(targets.reference_distributions) != 4:
            raise ValueError("actor capture requires four reference distributions")
        device = inputs[0].device
        if any(not p.requires_grad or p.device != device for p in parameters):
            raise ValueError("actor parameters must require gradients on the input device")
        self.parameters = parameters
        self.input_count = len(inputs)
        static = tuple(t.detach().clone() for t in (*inputs, *_target_tensors(targets)))
        self.static_inputs = static
        with torch.no_grad():
            validate_actor_targets(forward(static[: self.input_count]), _targets(static[self.input_count :]))
        # Capture writes these buffers in place. An optimizer must not set them
        # to None; verify their identity before every replay.
        original_gradients = tuple(p.grad for p in parameters)
        self.gradients = tuple(torch.zeros_like(p) for p in parameters)
        for parameter, gradient in zip(parameters, self.gradients, strict=True):
            parameter.grad = gradient

        def backward() -> tuple[ActorLoss, ReplayOutput]:
            for gradient in self.gradients:
                gradient.zero_()
            with torch.enable_grad():
                output = forward(static[: self.input_count])
                result = actor_loss(output, _targets(static[self.input_count :]), clip_epsilon=clip_epsilon)
                result.loss.backward()
            detached = ReplayOutput(
                output.log_probs_WKF.detach(),
                tuple(p.detach() for p in output.distributions),
                output.factor_mask_WKF,
                output.features,
            )
            return ActorLoss(*(getattr(result, field.name).detach() for field in fields(result))), detached

        ready = False
        try:
            self.captured = CapturedCall(backward, static, device=device)
            ready = True
        finally:
            if not ready:
                for parameter, gradient in zip(parameters, original_gradients, strict=True):
                    parameter.grad = gradient

    def backward(self, inputs: tuple[Tensor, ...], targets: ActorTargets) -> ActorLoss:
        if len(inputs) != self.input_count:
            raise ValueError("actor input count differs from capture")
        values = (*inputs, *_target_tensors(targets))
        if any(
            source.shape != target.shape or source.dtype != target.dtype or source.device != target.device
            for source, target in zip(values, self.static_inputs, strict=True)
        ):
            raise ValueError("actor input shape, dtype or device differs from capture")
        if any(
            parameter.grad is not gradient for parameter, gradient in zip(self.parameters, self.gradients, strict=True)
        ):
            raise ValueError("captured actor gradient storage changed; do not set gradients to None")
        result, _ = self.captured(values)
        return ActorLoss(*(getattr(result, field.name).clone() for field in fields(result)))

    def evaluation(self) -> ReplayOutput:
        """Copy the most recent replay's outputs without another model pass."""
        output = self.captured.output[1]
        return ReplayOutput(
            output.log_probs_WKF.clone(),
            tuple(p.clone() for p in output.distributions),
            output.factor_mask_WKF.clone(),
            ReplayCriticFeatures(*(getattr(output.features, field.name).clone() for field in fields(output.features))),
        )
