"""Model FLOPs utilization helpers for training telemetry."""

from typing import Final

_L40S_BF16_DENSE_TFLOPS: Final[float] = 362.05
_NVIDIA_L40S_SPECS: Final[str] = "https://www.nvidia.com/en-us/data-center/l40s/"
_B200_BF16_DENSE_TFLOPS: Final[float] = 2_250.0
_NVIDIA_B200_SPECS: Final[str] = "https://www.nvidia.com/en-us/data-center/hgx/"


def bf16_dense_peak_flops(device_name: str) -> float | None:
    """Return the GPU's dense BF16 tensor-core peak, in FLOP/s."""
    normalized = device_name.upper()
    if "B200" in normalized:
        return _B200_BF16_DENSE_TFLOPS * 1e12
    if "L40S" in normalized:
        return _L40S_BF16_DENSE_TFLOPS * 1e12
    return None


def bf16_peak_source(device_name: str) -> str | None:
    """Return the source used for the GPU's dense BF16 peak."""
    normalized = device_name.upper()
    if "B200" in normalized:
        return _NVIDIA_B200_SPECS
    if "L40S" in normalized:
        return _NVIDIA_L40S_SPECS
    return None


def model_flops_utilization(
    flops_per_update: int,
    seconds_per_update: float,
    peak_flops: float,
) -> float:
    """Return achieved model FLOPs divided by hardware peak FLOPs."""
    if flops_per_update <= 0:
        raise ValueError("flops_per_update must be positive")
    if seconds_per_update <= 0:
        raise ValueError("seconds_per_update must be positive")
    if peak_flops <= 0:
        raise ValueError("peak_flops must be positive")
    return flops_per_update / (seconds_per_update * peak_flops)
