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


def _sample_uniform(probabilities: Tensor, uniform: Tensor) -> Tensor:
    cumulative = probabilities.cumsum(-1)
    cumulative = cumulative / cumulative[..., -1:]
    upper = 1.0 - torch.finfo(probabilities.dtype).eps / 2
    uniform = uniform.to(device=probabilities.device, dtype=probabilities.dtype).clamp_max(upper)
    indices = torch.arange(probabilities.shape[-1], device=probabilities.device)
    positive = probabilities > 0
    last = torch.where(positive, indices, -1).amax(-1, keepdim=True)
    # Torch 2.11's parallel scan can round differently across a zero-mass
    # plateau. Counting CDF entries then selects masked classes near a boundary.
    # Take the first positive-mass crossing and give the last such class the
    # exact endpoint. Remove this constraint only when scan tests prove it safe.
    crossed = (cumulative > uniform[..., None]) | (indices == last)
    return torch.where(positive & crossed, indices, probabilities.shape[-1]).amin(-1)


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
    return _sample_uniform(probabilities, uniform)


def sample_with_temperature(logits: Tensor, uniform: Tensor, temperature: Tensor) -> Tensor:
    """Use a runtime temperature tensor without specializing compiled decoding."""
    probabilities = F.softmax(logits.float() / temperature, dim=-1)
    return _sample_uniform(probabilities, uniform)
