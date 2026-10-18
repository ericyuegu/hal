import pytest
import torch

from hal.models.sampling import sample_categorical
from hal.models.sampling import sample_with_temperature


def test_sample_categorical_accepts_fixed_uniforms() -> None:
    logits = torch.zeros(3, 4)
    uniform = torch.tensor([0.0, 0.3, 0.99])

    assert sample_categorical(logits, argmax=False, uniform=uniform).tolist() == [0, 1, 3]
    assert sample_categorical(logits, argmax=True).tolist() == [0, 0, 0]


def test_sample_categorical_applies_temperature_before_fixed_uniform_sampling() -> None:
    logits = torch.tensor([[0.0, torch.log(torch.tensor(9.0))]])
    uniform = torch.tensor([0.05])

    assert sample_categorical(logits, argmax=False, uniform=uniform, temperature=1.0).item() == 0
    assert sample_categorical(logits, argmax=False, uniform=uniform, temperature=0.1).item() == 1


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf"), True])
def test_sample_categorical_rejects_invalid_temperature(temperature: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        sample_categorical(torch.zeros(1, 2), argmax=False, temperature=temperature)


@pytest.mark.parametrize("uniform,expected", [(0.0, 1), (0.25, 2)])
def test_uniform_boundary_cannot_select_a_zero_probability_factor(uniform: float, expected: int) -> None:
    logits = torch.tensor([[-torch.inf, 0.0, torch.log(torch.tensor(3.0)), -torch.inf]])
    draw = torch.tensor([uniform])
    assert sample_categorical(logits, argmax=False, uniform=draw).item() == expected
    assert sample_with_temperature(logits, draw, torch.ones(1)).item() == expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_upper_uniform_cannot_select_masked_tail_after_cdf_rounding(dtype: torch.dtype, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    generator = torch.Generator().manual_seed(42)
    logits = torch.randn(1000, 256, generator=generator).to(device)
    logits[:, 128:] = -torch.inf
    upper = 1.0 - (2.0**-24 if dtype == torch.float32 else 1e-10)
    uniforms = torch.full((1000,), upper, dtype=dtype, device=device)
    actual = sample_categorical(logits, argmax=False, uniform=uniforms)
    assert actual.eq(127).all()
    actual = sample_with_temperature(logits, uniforms, torch.ones(1000, 1, device=device))
    assert actual.eq(127).all()
