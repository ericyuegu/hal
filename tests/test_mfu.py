"""Tests for model FLOPs utilization telemetry."""

import pytest

from hal.training.mfu import bf16_dense_peak_flops
from hal.training.mfu import bf16_peak_source
from hal.training.mfu import model_flops_utilization


def test_l40s_dense_bf16_peak_is_recognized() -> None:
    peak = bf16_dense_peak_flops("NVIDIA L40S")

    assert peak == pytest.approx(362.05e12)
    assert bf16_peak_source("NVIDIA L40S") is not None
    assert bf16_dense_peak_flops("unknown accelerator") is None
    assert bf16_peak_source("unknown accelerator") is None


def test_b200_dense_bf16_peak_is_recognized() -> None:
    peak = bf16_dense_peak_flops("NVIDIA B200")

    assert peak == pytest.approx(2_250e12)
    assert bf16_peak_source("NVIDIA B200") is not None


def test_model_flops_utilization_uses_update_time() -> None:
    utilization = model_flops_utilization(
        flops_per_update=181_025_000_000_000,
        seconds_per_update=1.0,
        peak_flops=362.05e12,
    )

    assert utilization == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("flops_per_update", "seconds_per_update", "peak_flops"),
    ((0, 1.0, 1.0), (1, 0.0, 1.0), (1, 1.0, 0.0)),
)
def test_model_flops_utilization_rejects_nonpositive_inputs(
    flops_per_update: int,
    seconds_per_update: float,
    peak_flops: float,
) -> None:
    with pytest.raises(ValueError):
        model_flops_utilization(flops_per_update, seconds_per_update, peak_flops)
