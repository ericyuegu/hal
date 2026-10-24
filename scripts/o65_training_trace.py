"""Compare untraced O65 training MFU with diagnostic kernel samples."""

import dataclasses
import json
import math
from collections import defaultdict
from pathlib import Path

import tyro

from hal.training.mfu import bf16_dense_peak_flops
from scripts.o65_trace import kernel_type


@dataclasses.dataclass(frozen=True, slots=True)
class Args:
    profile: tyro.conf.Positional[Path]
    top: int = 8


@dataclasses.dataclass(frozen=True, slots=True)
class PhaseFloor:
    name: str
    seconds: float
    flops: float
    peak: float

    @property
    def floor_seconds(self) -> float:
        return self.flops / self.peak

    @property
    def mfu(self) -> float:
        return self.floor_seconds / self.seconds


def phase_floors(profile: Path) -> tuple[PhaseFloor, ...]:
    resolved = json.loads((profile / "resolved.json").read_text())
    timed = json.loads((profile / "timed.json").read_text())
    if resolved["format"] != "hal.o65.gpu-profile-inputs.v1" or timed["format"] != "hal.o65.gpu-profile.v1":
        raise ValueError("unsupported O65 profile format")
    if timed["mode"] != "timed":
        raise ValueError("MFU requires the untraced update")
    if resolved["environment"]["world_size"] != "1":
        raise ValueError("training compute floors require a single-GPU profile")
    peak = bf16_dense_peak_flops(resolved["gpu"])
    if peak is None:
        raise ValueError(f"unknown BF16 peak for {resolved['gpu']}")
    metrics = timed["metrics"]
    rows = []
    for name in ("preparation", "actor", "critic"):
        seconds = float(metrics[f"systems/{name}_seconds"])
        flops = float(metrics[f"flops/{name}_total"])
        if not all(math.isfinite(value) and value > 0 for value in (seconds, flops)):
            raise ValueError(f"invalid timed work for {name}")
        rows.append(PhaseFloor(name, seconds, flops, peak))
    return tuple(rows)


def main(args: Args) -> None:
    rows = phase_floors(args.profile)
    print("| Phase | Wall s | Model TFLOP | BF16 compute floor s | Achieved TFLOP/s | MFU |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row.name} | {row.seconds:.3f} | {row.flops / 1e12:.1f} | {row.floor_seconds:.3f} "
            f"| {row.flops / row.seconds / 1e12:.1f} | {row.mfu:.1%} |"
        )
    print("\nCompute floors exclude memory, optimizer, nonlinearities and host work. The critic uses FP32.")
    print("FLOPs come from architecture-based timed metrics; estimates.json uses an older proxy model.")
    traced = json.loads((args.profile / "traced.json").read_text())
    if traced["format"] != "hal.o65.gpu-profile.v1" or traced["mode"] != "traced":
        raise ValueError("unsupported O65 trace report")
    # Bounded samples are diagnostic costs, not timings of the untraced update.
    for sample in traced["samples"]:
        if sample["site"].split("/")[0] not in ("reference", "preparation", "actor", "critic"):
            continue
        groups: dict[str, float] = defaultdict(float)
        kernels = sample["kernel_seconds_by_name"]
        for name, seconds in kernels.items():
            # CUDA template names can contain "cutlass" inside FlashAttention.
            group = "attention" if "flash" in name.lower() or "flex_attention" in name.lower() else kernel_type(name)
            groups[group] += seconds
        print(f"\n{sample['site']} sample {sample['index']}: {sample['kernel_count']} kernels")
        for name, seconds in sorted(groups.items(), key=lambda item: -item[1]):
            print(f"  {seconds * 1000:9.3f} ms {name}")
        for name, seconds in sorted(kernels.items(), key=lambda item: -item[1])[: args.top]:
            print(f"  {seconds * 1000:9.3f} ms {name[:130]}")


if __name__ == "__main__":
    main(tyro.cli(Args))
