"""Capture the frozen 059 cached local scheduler's applied-action timeline."""

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import ControllerAction
from hal.eval.scheduling import ActionScheduler
from hal.eval.scheduling import FrameTiming
from hal.inference.api import PolicyInput
from hal.inference.api import action_plan
from hal.inference.transport import ActionTransport
from hal.wire import BUTTON_BITS


def planned_action(target: int) -> ControllerAction:
    return ControllerAction(
        (target % 5 - 2) / 2,
        (target % 7 - 3) / 3,
        (target % 3 - 1) / 2,
        0.0,
        (target % 4) / 3,
        0.0,
        BUTTON_BITS["a"] if target % 2 else BUTTON_BITS["b"],
    )


def main() -> None:
    source = Path(__file__).resolve()
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    checkout = Path.cwd()
    scheduler = ActionScheduler(FrameTiming(2, 0, 2, 4), 256, 1)
    transport = ActionTransport(2)
    applied = NEUTRAL_CONTROLLER_ACTION
    frames = []
    requests = []
    for frame in range(20):
        scheduler.observe(PolicyInput(7, frame, 1, {}, applied, transport.pending, reset=frame == 0))
        request = scheduler.request_plan() if frame <= 14 else None
        if request is not None:
            tail = (planned_action(frame + 3), planned_action(frame + 4))
            requests.append(
                {
                    "source_frame": frame,
                    "fixed_actions": [asdict(action) for action in request.fixed_actions],
                    "generated": [asdict(action) for action in tail],
                }
            )
            assert scheduler.accept_plan(action_plan(request, tail))
            scheduler.apply_ready_plan(frame)
        submitted = scheduler.action_to_submit(frame)
        applied = transport.submit(submitted)
        frames.append({"target_frame": frame + 1, "action": asdict(applied)})
    print(
        json.dumps(
            {
                "schema_version": 1,
                "source_git_sha": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
                ).strip(),
                "capture_source_sha256": source_sha256,
                "timing": {"input_delay": 2, "fixed_prefix": 2, "replan": 2, "horizon": 4},
                "requests": requests,
                "frames": frames,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
