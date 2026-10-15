"""Plan or run the pre-registered early-checkpoint portable control."""

import argparse
import json
from pathlib import Path

import torch

from research.mechinterp_20260911 import historical_gameplay


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plan = {
        "checkpoint": str(args.checkpoint),
        "condition": "early_24576_portable_committed_prefix",
        "execution": {"compiled": False, "target": "local_cuda"},
        "output": str(args.output),
        "protocol": historical_gameplay.protocol_record(historical_gameplay.PROTOCOLS["short32"]),
        "replay_root": str(args.replay_root),
        "selection": "post_hoc_factorial_completion",
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    historical_gameplay.run_early_portable_control(
        args.checkpoint,
        args.output,
        args.replay_root,
        device=torch.device("cuda"),
    )


if __name__ == "__main__":
    main()
