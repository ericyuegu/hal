"""Categorical sampling primitives for action sequence decoding."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def validate_sampling_temperature(temperature: float) -> float:
    if (
        not isinstance(temperature, int | float)
        or isinstance(temperature, bool)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError(f"sampling temperature must be finite and positive, got {temperature!r}")
    return float(temperature)


def sample_categorical(
    logits: Tensor,
    *,
    argmax: bool,
    uniform: Tensor | None = None,
    generator: torch.Generator | None = None,
    temperature: float = 1.0,
) -> Tensor:
    """Sample class indices, optionally from caller-provided uniforms."""
    temperature = validate_sampling_temperature(temperature)
    values = logits.float()
    if argmax:
        return values.argmax(dim=-1)
    probabilities = F.softmax(values / temperature, dim=-1)
    if uniform is None:
        return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
    if uniform.shape != probabilities.shape[:-1]:
        raise ValueError(f"uniform shape {tuple(uniform.shape)} != batch shape {tuple(probabilities.shape[:-1])}")
    uniform = uniform.to(device=probabilities.device, dtype=probabilities.dtype)
    return (probabilities.cumsum(-1) < uniform[..., None]).sum(-1).clamp_max(probabilities.shape[-1] - 1)


def sample_with_temperature(logits: Tensor, uniform: Tensor, temperature: Tensor) -> Tensor:
    """Use a runtime temperature tensor without specializing compiled decoding."""
    probabilities = F.softmax(logits.float() / temperature, dim=-1)
    return (probabilities.cumsum(-1) < uniform[..., None]).sum(-1).clamp_max(probabilities.shape[-1] - 1)
