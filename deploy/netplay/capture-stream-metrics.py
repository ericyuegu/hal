"""Capture one comparable netplay streaming measurement from live status files."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from datetime import UTC
from datetime import datetime
from pathlib import Path

from hal.netplay_service.health import read_runner_status
from hal.netplay_service.health import read_slot_status

_METRICS = ("game_fps", "frame_interval_p95_ms", "dolphin_step_p95_ms", "policy_round_trip_p95_ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--streaming", choices=("on", "off"), required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runner = read_runner_status(args.status_path)
    slots = []
    for slot in range(runner.slots):
        path = args.status_path.with_name(f"{args.status_path.name}.slot-{slot}.json")
        status = read_slot_status(path)
        missing = [name for name in _METRICS if getattr(status, name) is None]
        if missing:
            parser.error(f"slot {slot} has no active-game values for {', '.join(missing)}")
        slots.append(
            {
                "slot": slot,
                "role": "stream" if slot == 0 else "headless",
                **{name: getattr(status, name) for name in _METRICS},
            }
        )
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    payload = {
        "schema_version": 1,
        "label": args.label,
        "hardware": args.hardware,
        "streaming": args.streaming,
        "git_sha": git_sha,
        "recorded_at": datetime.now(UTC).isoformat(),
        "runner": asdict(runner),
        "slots": slots,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
