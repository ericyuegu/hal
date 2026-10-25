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
        *targets.reference_log_distributions,
        targets.advantage_mean,
        targets.advantage_sigma,
        targets.denominator,
    )


def _targets(tensors: tuple[Tensor, ...]) -> ActorTargets:
    return ActorTargets(
        tensors[0], tensors[1], tensors[2], tensors[3], tensors[4:8], tensors[8], tensors[9], tensors[10]
    )


class GraphedActorStep:
    """Capture replay and backward with persistent inputs and gradient buffers.

    Compile the decoder with triton.cudagraphs=False and
    emulate_precision_casts=True. Release earlier eager autograd graphs before capture,
    because their leaf gradient nodes retain the original CUDA stream.
    Validate new inputs before backward, then clip and step the optimizer.
    """

    def __init__(
        self,
        forward: Callable[[tuple[Tensor, ...]], ReplayOutput],
        inputs: tuple[Tensor, ...],
        targets: ActorTargets,
        parameters: tuple[nn.Parameter, ...],
        *,
        clip_epsilon: float = 0.2,
        share_pool_with: torch.cuda.CUDAGraph | None = None,
        capture_stream: torch.cuda.Stream | None = None,
        gradients: tuple[Tensor, ...] | None = None,
    ) -> None:
        if not 0 < clip_epsilon < 1:
            raise ValueError("clip epsilon must be finite and between zero and one")
        if torch.is_inference_mode_enabled():
            raise ValueError("actor capture requires autograd outside inference_mode")
        if not inputs or not inputs[0].is_cuda or not parameters:
            raise ValueError("actor capture requires CUDA inputs and actor parameters")
        if len(targets.reference_log_distributions) != 4:
            raise ValueError("actor capture requires four reference distributions")
        device = inputs[0].device
        if any(not p.requires_grad or p.device != device for p in parameters):
            raise ValueError("actor parameters must require gradients on the input device")
        if gradients is not None and (
            len(gradients) != len(parameters)
            or any(
                gradient.shape != parameter.shape
                or gradient.dtype != parameter.dtype
                or gradient.device != parameter.device
                or gradient.requires_grad
                for gradient, parameter in zip(gradients, parameters, strict=True)
            )
        ):
            raise ValueError("shared actor gradients must match parameter shapes, dtypes and devices")
        self.parameters = parameters
        self.input_count = len(inputs)
        static = tuple(t.detach().clone() for t in (*inputs, *_target_tensors(targets)))
        self.static_inputs = static
        with torch.no_grad():
            validate_actor_targets(forward(static[: self.input_count]), _targets(static[self.input_count :]))
        # Graph replay writes into these gradient buffers at fixed addresses.
        original_gradients = tuple(p.grad for p in parameters)
        self.gradients = tuple(torch.zeros_like(p) for p in parameters) if gradients is None else gradients
        for parameter, gradient in zip(parameters, self.gradients, strict=True):
            parameter.grad = gradient

        def backward() -> tuple[ActorLoss, ReplayOutput]:
            for gradient in self.gradients:
                gradient.zero_()
            with torch.enable_grad():
                output = forward(static[: self.input_count])
                result = actor_loss(output, _targets(static[self.input_count :]), clip_epsilon=clip_epsilon)
                # Torch 2.11 backward can reuse saved forward buffers. Preserve
                # the reported evaluation before backward releases those values.
                detached = ReplayOutput(
                    output.log_probs_WKF.detach().clone(),
                    tuple(p.detach().clone() for p in output.log_distributions),
                    output.factor_mask_WKF.clone(),
                    ReplayCriticFeatures(
                        *(getattr(output.features, field.name).detach().clone() for field in fields(output.features))
                    ),
                )
                result.loss.backward()
            return ActorLoss(*(getattr(result, field.name).detach() for field in fields(result))), detached

        ready = False
        try:
            self.captured = CapturedCall(
                backward, static, device=device, share_pool_with=share_pool_with, capture_stream=capture_stream
            )
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
            tuple(p.clone() for p in output.log_distributions),
            output.factor_mask_WKF.clone(),
            ReplayCriticFeatures(*(getattr(output.features, field.name).clone() for field in fields(output.features))),
        )
