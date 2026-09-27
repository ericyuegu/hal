import pytest
import torch

from hal.models.sampling import sample_categorical


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
