"""Compare forced conditional distributions from matched 059 cache runs."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

GROUPS = ("c_stick", "main_stick", "triggers", "buttons")


def conditional_kl(control: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    if control.shape != candidate.shape:
        raise ValueError(f"conditional logit shape changed: {control.shape} != {candidate.shape}")
    log_p = F.log_softmax(control.float(), dim=-1)
    log_q = F.log_softmax(candidate.float(), dim=-1)
    p = log_p.exp()
    kl = torch.where(p > 0, p * (log_p - log_q), 0.0).sum(dim=-1)
    if not bool(torch.isfinite(kl).all()):
        raise ValueError("conditional KL contains nonfinite values")
    return kl


def summary(values: list[torch.Tensor]) -> dict[str, float | int]:
    joined = torch.cat([value.reshape(-1) for value in values])
    return {
        "count": joined.numel(),
        "mean_nats": joined.mean().item(),
        "p99_nats": torch.quantile(joined, 0.99).item(),
        "max_nats": joined.max().item(),
        "min_nats": joined.min().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    control = torch.load(args.control, map_location="cpu", weights_only=True)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=True)
    for name in ("bundle_sha256", "replay_sha256", "source_sha256", "base_frames", "sources"):
        if control[name] != candidate[name]:
            raise ValueError(f"capture identity {name} differs")
    if control["mode"] != "control" or candidate["mode"] != "candidate":
        raise ValueError("capture modes are not control/candidate")
    if set(control["logits"]) != set(candidate["logits"]):
        raise ValueError("captured source frames differ")
    by_phase: dict[str, list[torch.Tensor]] = {"before_eviction": [], "after_eviction": []}
    by_group: dict[str, list[torch.Tensor]] = {name: [] for name in GROUPS}
    for source in sorted(control["logits"]):
        old_groups = control["logits"][source]
        new_groups = candidate["logits"][source]
        if len(old_groups) != len(GROUPS) or len(new_groups) != len(GROUPS):
            raise ValueError("controller group count changed")
        for name, old, new in zip(GROUPS, old_groups, new_groups, strict=True):
            kl = conditional_kl(old, new)
            by_phase["before_eviction" if source < 256 else "after_eviction"].append(kl)
            by_group[name].append(kl)
    all_values = [item for phase in by_phase.values() for item in phase]
    result = {
        "bundle_sha256": control["bundle_sha256"],
        "replay_sha256": control["replay_sha256"],
        "source_sha256": control["source_sha256"],
        "control_gpu": control["gpu"],
        "candidate_gpu": candidate["gpu"],
        "control_torch": control["torch"],
        "candidate_torch": candidate["torch"],
        "sources": control["sources"],
        "overall": summary(all_values),
        "phase": {name: summary(values) for name, values in by_phase.items()},
        "group": {name: summary(values) for name, values in by_group.items()},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["overall"], indent=2))
    for name, measured in (("overall", result["overall"]), *result["phase"].items()):
        if measured["mean_nats"] > 5e-4 or measured["p99_nats"] > 5e-3:
            raise SystemExit(f"production BF16 conditional KL exceeds proposed limits in {name}")


if __name__ == "__main__":
    main()
